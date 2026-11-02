"""GPU parity and collection capacity probes for process packing; zero training."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import traceback

import torch

from .workers import WorkerGroup, allocate_workers


def specs_from_header(bundle, num_envs):
    """Use the production allocator with exact outgoing-interval bin counts.

    Collector-only probes do not need an extra CPU copy/FK of every motion.
    Every child still loads and verifies all assigned data. Packed children
    additionally check actual local bank.num_bins against this allocation.
    """
    from types import SimpleNamespace
    from supermimic.motion import BIN_FRAMES
    bins = []
    for entry in bundle["clips"]:
        frames = entry["state_frames"]
        if type(frames) is not int or frames < 2:
            raise ValueError("reference clips need at least two integer states")
        bins.extend([int(entry["asset_id"])] * len(range(0, frames - 1, BIN_FRAMES)))
    return allocate_workers(bundle, SimpleNamespace(bin_asset=torch.tensor(bins, dtype=torch.long)), num_envs)


def compare_values(left, right, path="root"):
    """Exact replay contract check, including every nested transition field."""
    if type(left) is not type(right):
        raise AssertionError(path + ": value/container type changed")
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(path + ": tensor schema changed")
        if not torch.equal(left, right):
            delta = (left.double() - right.double()).abs()
            raise AssertionError(f"{path}: values differ; max_abs={delta.max().item()}")
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(path + ": keys changed")
        for key in left:
            compare_values(left[key], right[key], path + "." + key)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(path + ": length changed")
        for i, (a, b) in enumerate(zip(left, right)):
            compare_values(a, b, path + f"[{i}]")
    elif left != right:
        raise AssertionError(path + ": value changed")


def memory_snapshot(pids):
    import subprocess
    memory = {}
    for pid in pids:
        values = {}
        path = Path(f"/proc/{pid}/smaps_rollup")
        if path.exists():
            for line in path.read_text().splitlines():
                fields = line.split()
                if fields[0] in ("Rss:", "Pss:", "Pss_Anon:", "Private_Dirty:"):
                    values[fields[0][:-1]] = int(fields[1])
        memory[str(pid)] = values
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = [line.split(", ") for line in query.stdout.splitlines()]
    owned = [row for row in rows if row[0] in set(map(str, pids))]
    return {"host_kB_by_pid": memory, "owned_gpu_processes_pid_MiB_uuid": owned,
            "summed_Pss_kB": sum(row.get("Pss", 0) for row in memory.values()),
            "summed_gpu_MiB": sum(int(row[1]) for row in owned)}


def valid_gpu_memory_value(value):
    """Require finite, nonnegative MiB usable by the existing integer peak sum.

    nvidia-smi's nounits CSV reports integer MiB. A PID/UUID with N/A memory
    identifies a process but supplies no memory measurement.
    """
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0 and int(value) >= 0
    except (TypeError, ValueError, OverflowError):
        return False


def summarize_collector_gpu_samples(samples, collector_pids, collector_devices, parent_pid):
    """Require every collector and assigned GPU; the CPU parent needs no GPU."""
    from .resource_probe import summarize_gpu_samples
    owned = set(map(str, collector_pids))
    report = summarize_gpu_samples(samples, [parent_pid, *collector_pids], parent_pid)
    seen, measured, uuids, errors = set(), set(), set(), []
    for index, sample in enumerate(samples):
        for key, value in sample.items():
            if key.endswith("_error"):
                report["samples"][index][key] = value
                errors.append({"sample_index": index, "elapsed_seconds": sample["elapsed_seconds"],
                               "operation": key, "error": value})
        for row in sample.get("processes", []):
            if len(row) == 3 and row[0] in owned:
                seen.add(row[0])
                uuids.add(row[2])
                if valid_gpu_memory_value(row[1]):
                    measured.add(row[0])
    expected = min(len(owned), len(set(collector_devices)))
    missing = sorted(map(int, owned - seen))
    missing_memory = sorted(map(int, owned - measured))
    covered = bool(owned) and not missing and len(uuids) == expected
    complete = covered and not missing_memory and not errors and bool(report["owned_process_sampled_peak_MiB_by_gpu_uuid"])
    report.update(collector_pids=sorted(map(int, owned)), observed_collector_pids=sorted(map(int, seen)),
                  measured_collector_pids=sorted(map(int, measured)), missing_collector_memory_pids=missing_memory,
                  missing_collector_pids=missing, expected_collector_gpu_count=expected,
                  observed_collector_gpu_count=len(uuids), observed_collector_gpu_uuids=sorted(uuids),
                  sampling_errors=errors, measurement_complete=complete,
                  assigned_device_uuid_verified=covered, parent_gpu_required=False)
    return report


def run_collection(specs, config, *, steps, output, parity):
    from .host_memory import HostMemorySampler
    from .resource_probe import _GpuSampler
    output.mkdir(parents=True, exist_ok=False)
    config = {**config, "output": str(output / "workers")}
    report = {"assets_per_process": config["assets_per_process"], "asset_ids": [s.asset_id for s in specs],
        "environments": sum(s.num_envs for s in specs), "optimizer_steps": 0,
        "formal_training_started": False, "passed": False, "phase": "starting_collectors"}
    sampling_audit = None
    if config.get("sampling_audit"):
        from .evaluate import _read_bundle_header
        from .sampling_audit import ActualCollectionAudit
        from .config import source_identity
        from hoi.data.manifest import sha256_file
        report["source_before"] = source_identity()
        header = _read_bundle_header(config["bundle"])
        sampling_audit = ActualCollectionAudit(header, specs)
        report["bundle"] = {"path": str(Path(config["bundle"]).resolve()), "sha256": sha256_file(config["bundle"]),
                            "bundle_content_sha256": header["bundle_content_sha256"]}
        report["schema"] = "hoi_weighted_collection_probe_v1"
    write = lambda: (output / "progress.json").write_text(json.dumps(report, indent=2))
    write()
    workers, trace, error = None, [], None
    pids = [os.getpid()]
    host, gpu = HostMemorySampler(os.getpid()), _GpuSampler()
    started = time.monotonic()
    failure = None
    try:
        with host, gpu:
            try:
                workers = WorkerGroup(specs, config, device="cpu", timeout=900)
                pids += [p.pid for p in workers._processes]
                report.update(startup_seconds=time.monotonic() - started, processes=len(workers._processes),
                              memory_ready=memory_snapshot(pids), phase="collection")
                write()
                snapshot = workers.snapshot()
                if parity:
                    trace.append(snapshot)
                reset_frames = snapshot["f"].clone()
                action_rng = torch.Generator().manual_seed(4127)
                started, stepping_seconds = time.monotonic(), 0.
                for tick in range(steps):
                    action = .05 * torch.randn((workers.N, 29), generator=action_rng)
                    step_started = time.monotonic()
                    tr = workers.step(action)
                    if sampling_audit is not None:
                        sampling_audit.observe(tr)
                    stepping_seconds += time.monotonic() - step_started
                    snapshot = workers.snapshot()
                    if parity:
                        trace.append({"transition": tr, "snapshot": snapshot})
                    if tick in (7, 19, 31) and sampling_audit is None:
                        reset_snapshot = workers.reset(reset_frames)
                        if parity:
                            trace.append({"explicit_reset": reset_snapshot})
                    report.update(completed_ticks=tick + 1)
                    if sampling_audit is not None:
                        report["sampling"] = sampling_audit.report()
                    write()
                report.update(steps=steps, collection_seconds=time.monotonic() - started,
                              memory_final=memory_snapshot(pids), stepping_seconds=stepping_seconds)
                report["instrumented_wall_environment_steps_per_second"] = workers.N * steps / report["collection_seconds"]
                report["collector_environment_steps_per_second"] = workers.N * steps / stepping_seconds
                if parity:
                    torch.save(trace, output / "trace.pt")
                report["phase"] = "shutdown"
                if sampling_audit is not None:
                    report["sampling"] = sampling_audit.report(detail=True)
                    if not report["sampling"]["allocation_match"] or not report["sampling"]["completed_episodes"]:
                        raise RuntimeError("weighted collection must verify actual proportions across natural episode boundaries")
                write()
            except BaseException as exc:
                failure = exc
                report["error"] = traceback.format_exc()
            finally:
                try:
                    if workers is not None:
                        owned_processes = list(workers._processes)
                        workers.close()
                        report["shutdown_passed"] = True
                        report["collector_exit_codes"] = [p.exitcode for p in owned_processes]
                        if any(code != 0 for code in report["collector_exit_codes"]):
                            raise RuntimeError("collector exited unsuccessfully after close")
                        if config.get("audit_isolation"):
                            audits = [json.loads(path.read_text()) for path in sorted((output / "workers").glob("isolation_*.json"))]
                            report["isolation_audits"] = audits
                            if len(audits) != report["processes"] or not all(a["passed"] and a["checks"] for a in audits):
                                raise RuntimeError("native noninterference audit did not finish successfully")
                except BaseException as exc:
                    report["shutdown_error"] = traceback.format_exc()
                    if failure is None:
                        failure = exc
    except BaseException as exc:
        report["sampler_error"] = traceback.format_exc()
        if failure is None:
            failure = exc
    finally:
        if sampling_audit is not None:
            try:
                report["source_after"] = source_identity()
                report["source_unchanged"] = report["source_after"] == report["source_before"]
                if not report["source_unchanged"]:
                    raise RuntimeError("source files changed during weighted native collection")
            except Exception as exc:
                report["source_error"] = traceback.format_exc()
                if failure is None:
                    failure = exc
        report["host_memory"] = host.summary()
        report["gpu_memory"] = summarize_collector_gpu_samples(
            gpu.samples, pids[1:], config.get("collector_devices") or [config.get("device", "cuda:0")], os.getpid())
        report["resource_measurement_complete"] = (report["host_memory"]["measurement_complete"]
                                                   and report["gpu_memory"]["measurement_complete"])
        report["passed"] = failure is None and report["resource_measurement_complete"]
        report["phase"] = "completed" if report["passed"] else "failed"
        (output / "report.json").write_text(json.dumps(report, indent=2))
    if failure is not None:
        raise failure
    if not report["passed"]:
        raise RuntimeError("collector completed but resource sampling was incomplete; see report")
    return trace, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("parity", "capacity", "smoke"), required=True)
    parser.add_argument("--asset-ids", type=int, nargs="+", default=[0, 59, 209, 266])
    parser.add_argument("--parity-envs-per-asset", type=int, default=8)
    parser.add_argument("--num-envs", type=int, choices=(2048, 4096), default=4096)
    parser.add_argument("--assets-per-process", type=int, default=32)
    parser.add_argument("--collector-devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--worker-startup-concurrency", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--audit-isolation", action="store_true", help="Diagnostic-only exact checks that inactive native buffers/RNG stay unchanged")
    parser.add_argument("--sampling-audit", action="store_true", help="Count actual weighted transitions across natural resets; capacity mode only, no learner/optimizer")
    parser.add_argument("--robot-mjcf")
    args = parser.parse_args(argv)
    if args.audit_isolation and (args.mode == "capacity" or args.assets_per_process < 2):
        parser.error("isolation audit is only for a multi-asset smoke/parity panel")
    if args.sampling_audit and (args.mode != "capacity" or args.steps < 64):
        parser.error("weighted sampling audit requires complete capacity mode and at least 64 real control ticks")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"mode": args.mode, "optimizer_steps": 0, "formal_training_started": False, "passed": False}
    try:
        from supermimic.physics import PhysicsSettings
        from .config import bundle_identity, source_identity
        from .evaluate import _read_bundle_header
        print(json.dumps({"phase": "reading_bound_train_header", "mode": args.mode}), flush=True)
        bundle = _read_bundle_header(args.bundle)
        if bundle["partition"] != "train":
            raise ValueError("collector resource experiment requires the complete training bundle")
        report.update(source=source_identity(), data_identity=bundle_identity(bundle),
                      requested_envs=args.num_envs, requested_steps=args.steps,
                      scope="small diagnostic panel only" if args.mode in ("parity", "smoke") else "all training assets; collector only")
        specs = specs_from_header(bundle, args.num_envs)
        print(json.dumps({"phase": "allocation_ready", "groups": len(specs)}), flush=True)
        config = {"bundle": str(args.bundle), "physics": PhysicsSettings(hand_collision=True, self_collision=True).__dict__,
            "seed": 0, "device": "cuda:0", "gamma": .99, "launcher_args": [], "robot_mjcf": args.robot_mjcf,
            "collector_devices": args.collector_devices, "worker_startup_concurrency": args.worker_startup_concurrency,
            "assets_per_process": args.assets_per_process, "audit_isolation": args.audit_isolation,
            "sampling_audit": args.sampling_audit}
        if args.mode in ("parity", "smoke"):
            specs = [replace(s, num_envs=args.parity_envs_per_asset) for s in specs if s.asset_id in args.asset_ids]
            if {s.asset_id for s in specs} != set(args.asset_ids):
                raise ValueError("diagnostic panel asset missing from complete train bundle")
        if args.mode == "parity":
            first, baseline = run_collection(specs, {**config, "assets_per_process": 1}, steps=args.steps,
                                               output=args.output / "baseline", parity=True)
            report["baseline"] = baseline
            (args.output / "progress.json").write_text(json.dumps(report, indent=2))
            repeat, repeat_report = run_collection(specs, {**config, "assets_per_process": 1}, steps=args.steps,
                                                    output=args.output / "baseline_repeat", parity=True)
            report["baseline_repeat"] = repeat_report
            compare_values(first, repeat)
            del repeat
            report.update(baseline_repeat_exact=True)
            (args.output / "progress.json").write_text(json.dumps(report, indent=2))
            packed, packed_report = run_collection(specs, config, steps=args.steps,
                                                    output=args.output / "packed", parity=True)
            report["packed"] = packed_report
            compare_values(first, packed)
            report.update(baseline=baseline, baseline_repeat=repeat_report, packed=packed_report,
                          packed_exact=True, passed=True)
        else:
            _, measured = run_collection(specs, config, steps=args.steps, output=args.output / "collection", parity=False)
            report.update(collection=measured, passed=True, scope=("small panel integration/isolation only" if args.mode == "smoke" else "all training assets; collector only, no learner graph"))
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
