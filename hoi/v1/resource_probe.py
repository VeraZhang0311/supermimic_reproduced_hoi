"""Measure the full v1 resource profile without training.

Native collection and actor backward are real. No optimizer is stepped, no WM
policy gate is bypassed, and subset inputs are explicitly resource fixtures.
The explicit 2048-environment comparison changes only collector/replay width;
all training batches, replay depth and imagination horizon remain full size.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import torch


@dataclass(frozen=True)
class ProbeConfig:
    num_envs: int = 4096
    buffer_steps: int = 1000
    pi_batch: int = 4096
    horizon: int = 32
    expected_groups: int = 8
    collect_steps: int = 12

    def validate(self):
        if type(self.num_envs) is not int or self.num_envs not in (2048, 4096):
            raise ValueError("resource probe supports only 4096 environments or the explicit 2048-environment comparison")
        for key, expected in (("buffer_steps", 1000), ("pi_batch", 4096), ("horizon", 32)):
            if type(getattr(self, key)) is not int or getattr(self, key) != expected:
                raise ValueError(f"resource readiness probe preserves {key}={expected}; no silent reduction")
        if type(self.expected_groups) is not int or self.expected_groups < 1:
            raise ValueError("expected_groups must be a positive integer")
        if type(self.collect_steps) is not int or self.collect_steps < 10:
            raise ValueError("collect at least ten real ticks for episode history")
        return self


def deployment_comparison(probe, config):
    """Describe the permitted resource-only override without mutating training."""
    probe.validate()
    if config.num_envs != 4096:
        raise ValueError("resource comparison requires the unchanged 4096-environment production config")
    return {"resource_only": True,
            "production_num_envs": config.num_envs, "measured_num_envs": probe.num_envs,
            "explicit_overrides": {} if probe.num_envs == config.num_envs else {"num_envs": probe.num_envs},
            "production_readiness_compatible": probe.num_envs == config.num_envs,
            "data_selection_changed": False, "training_batches_changed": False,
            "replay_depth_changed": False, "imagination_horizon_changed": False}


def read_probe_config(path=None, *, seed=None):
    """Use the formal loader, preserving the existing full-size resource profile."""
    from .train import read_config
    config, changed = read_config(path)
    if changed:
        raise ValueError("resource probe preserves production numerical settings: " + str(sorted(changed)))
    if config.variant != "hoi_v1":
        raise ValueError("production resource probe measures the full hoi_v1 variant")
    if seed is not None:
        config = replace(config, seed=seed).validate()
    return config


def build_probe_models(config, mdp, bundle, device):
    """Mirror train.main's feature configuration before constructing the actor."""
    from .networks import Policy, WorldModel, Critic
    from .privileged import TEACHER_PROFILE, configure_teacher
    from .variants import geometry_input_enabled
    if config.actor_observation_profile == TEACHER_PROFILE:
        configure_teacher(mdp, bundle, device)
    return [model(geometry_input_enabled=geometry_input_enabled(config.variant),
                  **({"actor_observation_profile": config.actor_observation_profile} if model is Policy else {})).to(device)
            for model in (Policy, WorldModel, Critic)]


def initialize_probe_models(models, *, from_scratch=False, init_checkpoint=None, physics,
                            legacy_physics=None, allow_physics_change=False):
    """Fresh initialization never reads or migrates a checkpoint."""
    if bool(from_scratch) == (init_checkpoint is not None):
        raise ValueError("choose exactly one of --from-scratch or --init-checkpoint")
    if from_scratch:
        return {"mode": "explicit_from_scratch"}
    from .train import read_checkpoint
    from .checkpoint import migrate_legacy
    old, digest = read_checkpoint(init_checkpoint)
    initialization = migrate_legacy(old, *models, physics=physics, legacy_physics=legacy_physics,
                                    allow_physics_change=allow_physics_change)
    initialization.update(path=str(Path(init_checkpoint).resolve()), sha256=digest)
    return initialization


def initialize_cuda_memory_counters(device, *, cuda=None):
    """Initialize the selected allocator before its first peak-stat reset."""
    cuda = torch.cuda if cuda is None else cuda
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("CUDA memory counters require a CUDA device")
    cuda.set_device(selected)
    cuda.init()
    cuda.reset_peak_memory_stats(selected)
    return selected


def validate_real_start_mix(start, history, motion, *, expected_assets, batch_size, horizon):
    """Reject reference-only, mismatched, or single-asset imagination inputs."""
    assets = start["asset_id"]
    if assets.shape != (batch_size,) or history["f"].shape != (batch_size, 10):
        raise ValueError("real start/history shape differs from the requested full batch")
    if not torch.equal(motion.asset_ids[start["f"]], assets):
        raise ValueError("real start asset does not match its reference frame")
    if not torch.equal(history["asset_id"], assets[:, None].expand_as(history["asset_id"])):
        raise ValueError("real history crosses fixed asset groups")
    if not torch.equal(motion.asset_ids[history["f"]], history["asset_id"]):
        raise ValueError("real history reference frames do not match asset identity")
    if not bool((start["f"] + horizon <= motion.clip_end[motion.clip_id[start["f"]]]).all()):
        raise ValueError("real start lacks the full reference imagination horizon")
    observed = set(assets.detach().cpu().tolist())
    if observed != set(expected_assets):
        raise ValueError(f"real full batch must cover every static group: observed {sorted(observed)}")
    differences = history["f"][:, 1:] - history["f"][:, :-1]
    if not bool(((differences == 0) | (differences == 1)).all()):
        raise ValueError("real history is neither contiguous nor same-episode prefix padding")
    if bool(((differences == 0) & (differences == 1).cummax(-1).values).any()):
        raise ValueError("same-episode padding may occur only at the history prefix")
    if not torch.equal(history["f"][:, -1], start["f"]):
        raise ValueError("real history does not end at its start state")
    for key in ("s", "o", "a_prev", "xi", "xi_o"):
        if not torch.equal(history[key][:, -1], start[key]):
            raise ValueError(f"real history endpoint {key} differs from the actual replay start")
    return {"asset_counts": {str(a): int((assets == a).sum()) for a in sorted(observed)},
            "history_prefix_padding_edges": int((differences == 0).sum()),
            "history_edges": differences.numel(), "sampling": "actual replay starts with replacement",
            "reference_only_states": False, "all_static_groups_present": True}


def stratified_real_starts(replay, batch_size, motion, *, expected_assets, horizon, max_attempts=32):
    """Resource coverage only: retain real aligned rows, at least one per asset.

    Every candidate comes from the unchanged production Replay.sample_starts.
    The first candidate batch supplies the remaining rows. No reference-only
    states or synthetic histories are introduced, and the final batch stays B.
    """
    expected = list(expected_assets)
    if (type(batch_size) is not int or batch_size < 1 or type(max_attempts) is not int or max_attempts < 1
            or not expected or len(set(expected)) != len(expected) or batch_size < len(expected)):
        raise ValueError("resource start coverage requires unique assets, sufficient batch size and bounded positive attempts")
    expected_set = set(expected)
    remaining = set(expected)
    starts, histories = [], []
    baseline_start = baseline_history = None
    candidates = 0
    for attempt in range(max_attempts):
        start, history = replay.sample_starts(batch_size, 10, horizon=horizon, motion=motion)
        observed = set(start["asset_id"].detach().cpu().tolist())
        if not observed.issubset(expected_set):
            raise ValueError("resource replay candidate contains unexpected assets")
        validate_real_start_mix(start, history, motion, expected_assets=observed,
                                batch_size=batch_size, horizon=horizon)
        candidates += batch_size
        if baseline_start is None:
            baseline_start, baseline_history = start, history
        selected = []
        for asset in expected:
            if asset in remaining and asset in observed:
                selected.append((start["asset_id"] == asset).nonzero().flatten()[0])
                remaining.remove(asset)
        if selected:
            ids = torch.stack(selected)
            starts.append({key: value[ids] for key, value in start.items()})
            histories.append({key: value[ids] for key, value in history.items()})
        if not remaining:
            fill = batch_size - len(expected)
            combine = lambda parts, baseline: {key: torch.cat([piece[key] for piece in parts] + [baseline[key][:fill]])
                                               for key in baseline}
            selected_start = combine(starts, baseline_start)
            selected_history = combine(histories, baseline_history)
            validate_real_start_mix(selected_start, selected_history, motion, expected_assets=expected,
                                    batch_size=batch_size, horizon=horizon)
            return selected_start, selected_history, {
                "sampling": "resource_only_stratified_real_replay_starts_with_replacement",
                "production_sampling_changed": False, "minimum_rows_per_asset": 1,
                "final_batch_size": batch_size, "candidate_real_starts": candidates,
                "selection_attempts": attempt + 1, "max_selection_attempts": max_attempts}
    raise ValueError(f"insufficient real replay asset coverage after {max_attempts} bounded attempts; "
                     f"missing asset IDs: {sorted(remaining)}; no synthetic starts were used")


def full_active_wm_windows(replay, batch_size, window, *, max_attempts=32):
    """Select real, complete fit windows so resets cannot understate WM memory."""
    pieces, remaining, candidates = [], batch_size, 0
    for attempt in range(max_attempts):
        sampled = replay.sample_windows(max(2 * remaining, 32), window, split="fit")
        candidates += len(sampled["mask"])
        ids = sampled["mask"].bool().all(-1).nonzero().flatten()[:remaining]
        if len(ids):
            pieces.append({key: value[ids] for key, value in sampled.items()})
            remaining -= len(ids)
        if not remaining:
            result = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
            active = result["mask"].bool().sum(0)
            if active.tolist() != [batch_size] * window:
                raise RuntimeError("full WM capacity window unexpectedly contains a reset mask")
            return result, {"active_rows_per_step": active.tolist(), "candidate_real_windows": candidates,
                            "selection_attempts": attempt + 1, "sampling": "complete real fit windows with replacement"}
    raise ValueError("insufficient complete real WM windows; cannot claim the full-batch memory graph")


def critic_target_values(critic, arguments, *, horizon, batch, critic_minibatches):
    """Same complete, ordered target evaluation as Learner.critic_update."""
    target_batch = (horizon * batch + critic_minibatches - 1) // critic_minibatches
    return torch.cat([critic(*(feature[offset:offset + target_batch] for feature in arguments))
                      for offset in range(0, (horizon + 1) * batch, target_batch)]).reshape(horizon + 1, batch)


def summarize_gpu_samples(samples, process_ids, parent_pid):
    """Include every owned collector GPU, excluding unrelated devices/jobs."""
    owned = set(map(str, process_ids))
    assigned = {row[2] for sample in samples for row in sample.get("processes", [])
                if len(row) == 3 and row[0] in owned}
    filtered, peaks, external_peaks = [], {}, {}
    for sample in samples:
        processes = [row for row in sample.get("processes", []) if len(row) == 3 and row[2] in assigned]
        item = {"elapsed_seconds": sample["elapsed_seconds"],
                "assigned_gpus": [row for row in sample.get("gpus", []) if len(row) == 4 and row[0] in assigned],
                "assigned_gpu_processes": [row + [row[0] in owned] for row in processes]}
        totals, external = {}, {}
        for row in processes:
            try:
                target = totals if row[0] in owned else external
                target[row[2]] = target.get(row[2], 0) + int(row[1])
            except ValueError:
                continue
        for target, result in ((totals, peaks), (external, external_peaks)):
            for uuid, value in target.items():
                result[uuid] = max(result.get(uuid, 0), value)
        filtered.append(item)
    return {"assigned_device_uuids": sorted(assigned), "assigned_device_uuid_verified": bool(assigned) and any(
                row[0] == str(parent_pid) for sample in samples for row in sample.get("processes", []) if len(row) == 3),
            "samples": filtered, "owned_process_sampled_peak_MiB_by_gpu_uuid": peaks,
            "external_process_sampled_peak_MiB_by_gpu_uuid": external_peaks,
            "has_other_job_processes_on_assigned_device": bool(external_peaks)}


class _GpuSampler:
    """nvidia-smi includes memory owned by all separate simulator processes."""
    def __init__(self):
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            sample = {"elapsed_seconds": time.monotonic() - self.started}
            for key, query in (("gpus", "gpu=uuid,name,memory.total,memory.used"),
                               ("processes", "compute-apps=pid,used_memory,gpu_uuid")):
                try:
                    result = subprocess.run(["nvidia-smi", "--query-" + query, "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=5, check=True)
                    sample[key] = [line.strip().split(", ") for line in result.stdout.splitlines() if line.strip()]
                except (OSError, subprocess.SubprocessError) as exc:
                    sample[key + "_error"] = str(exc)
            self.samples.append(sample)
            self.stop.wait(1.)

    def __enter__(self):
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=11)


def finalize_resource_report(report, *, workers, sampler, host_memory, collector_devices, parent_pid):
    """Keep collection failures primary while completing the resource evidence.

    The caller writes the resulting report even if shutdown or a summarizer
    fails. A learner CUDA allocation and every collector must be observed.
    """
    errors = report.setdefault("finalization_errors", [])

    def record_error(operation, exc):
        errors.append({"operation": operation, "type": type(exc).__name__,
                       "message": str(exc), "traceback": traceback.format_exc()})

    try:
        if workers is not None:
            workers.close()
    except BaseException as exc:
        record_error("workers.close", exc)

    process_ids = report.get("process_ids", [parent_pid])
    collector_pids = [pid for pid in process_ids if str(pid) != str(parent_pid)]
    try:
        from .packed_probe import summarize_collector_gpu_samples, valid_gpu_memory_value
        report["nvidia_smi"] = summarize_collector_gpu_samples(
            samples=sampler.samples, collector_pids=collector_pids,
            collector_devices=collector_devices, parent_pid=parent_pid)
        learner_seen = any(len(row) == 3 and row[0] == str(parent_pid)
                           for sample in sampler.samples for row in sample.get("processes", []))
        learner_measured = any(len(row) == 3 and row[0] == str(parent_pid) and valid_gpu_memory_value(row[1])
                               for sample in sampler.samples for row in sample.get("processes", []))
        report["nvidia_smi"]["learner_gpu_required"] = True
        report["nvidia_smi"]["learner_gpu_observed"] = learner_seen
        report["nvidia_smi"]["learner_memory_observed"] = learner_measured
        report["nvidia_smi"]["measurement_complete"] = (
            report["nvidia_smi"]["measurement_complete"] and learner_measured)
    except BaseException as exc:
        record_error("summarize_gpu_samples", exc)
        report["nvidia_smi"] = {"measurement_complete": False, "learner_gpu_required": True,
                                "summary_unavailable": True}

    try:
        report["host_memory"] = host_memory.summary()
    except BaseException as exc:
        record_error("summarize_host_memory", exc)
        report["host_memory"] = {"measurement_complete": False, "summary_unavailable": True}

    report["resource_measurement_complete"] = (
        report["host_memory"].get("measurement_complete") is True
        and report["nvidia_smi"].get("measurement_complete") is True)
    report["passed"] = (report.get("passed") is True and not errors
                        and report["resource_measurement_complete"])
    report["memory_measurement"] = {"global_samples_include_separate_simulator_processes": True,
        "nvidia_smi_interval_seconds": 1., "sampled_peak_is_lower_bound": True,
        "parent_allocator_peaks_are_exact": True, "adam_two_moment_tensors_reserved_for_every_parameter": True,
        "torch_mem_get_info_includes_other_processes_on_assigned_gpu": True,
        "nvidia_smi_uuid_binding": "actual owned parent/worker PID compute-apps GPU UUIDs; unrelated GPUs filtered"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Same JSON training configuration as hoi.v1.train")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init-checkpoint", type=Path)
    initialization.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--allow-subset-resource-fixture", action="store_true")
    parser.add_argument("--expected-groups", type=int, default=8)
    parser.add_argument("--collect-steps", type=int, default=12)
    parser.add_argument("--probe-num-envs", type=int, choices=(2048, 4096), default=4096,
                        help="Resource-only collector/replay width; batches, horizon and complete asset coverage stay unchanged")
    parser.add_argument("--device", default="cuda:0")
    from .workers import add_collector_arguments, collector_device_list, startup_concurrency
    add_collector_arguments(parser)
    parser.add_argument("--robot-mjcf")
    parser.add_argument("--seed", type=int, default=None, help="Explicit override; otherwise use --config seed")
    parser.add_argument("--worker-cpu-threads", type=int, default=3)
    parser.add_argument("--legacy-hand-collision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy-self-collision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-initialization-physics-change", action="store_true")
    from supermimic.physics import add_physics_arguments, resolve_arguments
    add_physics_arguments(parser)
    args, launcher = parser.parse_known_args(argv)
    collector_devices = collector_device_list(args.device, args.collector_devices)
    startup_concurrency(args.worker_startup_concurrency, 1)
    probe = ProbeConfig(num_envs=args.probe_num_envs, expected_groups=args.expected_groups,
                        collect_steps=args.collect_steps).validate()
    if (args.seed is not None and args.seed < 0) or args.worker_cpu_threads < 1:
        parser.error("seed must be nonnegative and CPU threads positive")
    config = read_probe_config(args.config, seed=args.seed)
    comparison = deployment_comparison(probe, config)
    settings, _ = resolve_arguments(args)
    physics = asdict(settings)
    if not (settings.hand_collision and settings.self_collision):
        parser.error("v1 resource validation requires hand ON/self ON")
    from .config import bundle_identity, fresh_directory, source_identity, write_json
    from .evaluate import _read_bundle_header
    header = _read_bundle_header(args.bundle)
    if header["partition"] != "train":
        raise ValueError("resource collection uses the train partition only")
    subset = bool(header["scope"]["is_subset"])
    if subset and not args.allow_subset_resource_fixture:
        raise ValueError("subset bundle requires explicit --allow-subset-resource-fixture")
    output = fresh_directory(args.output)
    full_scope = ("full_training_bundle_resource_validation" if comparison["production_readiness_compatible"]
                  else "full_training_bundle_resource_comparison")
    report = {"scope": "subset_resource_fixture" if subset else full_scope,
              "probe": asdict(probe), "production_config": config.to_dict(),
              "deployment_comparison": comparison,
              "physics": physics, "source_identity": source_identity(),
              "data_identity": bundle_identity(header), "optimizer_steps": 0, "formal_training_started": False,
              "policy_gate_bypassed": False, "wm_gate_validated": False,
              "worker_cpu_threads": args.worker_cpu_threads, "collector_devices": collector_devices,
              "worker_startup_concurrency": args.worker_startup_concurrency, "assets_per_process": args.assets_per_process, "phases": [], "passed": False}
    write_json(output / "request.json", report)
    workers = None
    started = time.monotonic()
    sampler = _GpuSampler()
    from .host_memory import HostMemorySampler
    host_memory = HostMemorySampler(os.getpid())

    def phase(name, **values):
        torch.cuda.synchronize(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        record = {"name": name, "elapsed_seconds": time.monotonic() - started,
                  "parent_allocated_bytes": torch.cuda.memory_allocated(args.device),
                  "parent_reserved_bytes": torch.cuda.memory_reserved(args.device),
                  "parent_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                  "parent_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                  "all_process_gpu_used_bytes": total - free, "gpu_total_bytes": total, **values}
        report["phases"].append(record)
        write_json(output / f"phase_{len(report['phases']):02d}_{name}.json", record)

    try:
        if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("full native resource probe needs the prepared Linux CUDA runtime")
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.worker_cpu_threads)
        torch.set_num_threads(args.worker_cpu_threads)
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        import supermimic.networks as legacy_networks
        legacy_networks.AMP_DTYPE = torch.bfloat16 if config.amp == "bf16" else None
        from supermimic.kinematics import G1Kinematics
        from supermimic.robot_metadata import default_joint_pos_list
        from .prepare import load_bundle
        from .mdp import HOIMDP
        from .networks import frozen_parameters
        from .buffer import ReplayBuffer
        from .learner import Learner
        from .workers import allocate_workers, WorkerGroup, resource_preflight
        with sampler, host_memory:
            initialize_cuda_memory_counters(args.device)
            kin = G1Kinematics(args.device, mjcf_path=args.robot_mjcf)
            bank, shapes, bundle = load_bundle(args.bundle, kin, device=args.device)
            mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=args.device), shapes)
            models = build_probe_models(config, mdp, bundle, args.device)
            declared = {"hand_collision": args.legacy_hand_collision, "self_collision": args.legacy_self_collision}
            report["initialization"] = initialize_probe_models(models, from_scratch=args.from_scratch,
                init_checkpoint=args.init_checkpoint, physics=physics, legacy_physics=declared,
                allow_physics_change=args.allow_initialization_physics_change)
            phase("models_and_references")
            replay = ReplayBuffer(probe.buffer_steps, probe.num_envs, args.device)
            learner = Learner(*models, bank, mdp, shapes, replay, config)
            # Adam moments normally appear on the first optimizer step. Reserve
            # the same two full-size tensors while performing no such step.
            adam_reserve = [torch.zeros_like(p) for m in models for p in m.parameters() for _ in range(2)]
            phase("full_replay_and_adam_reserve", replay_bytes=sum(t.numel() * t.element_size() for t in replay.data.values()),
                  adam_moment_reserve_bytes=sum(t.numel() * t.element_size() for t in adam_reserve))
            specs = allocate_workers(bundle, bank, probe.num_envs)
            if len(specs) != probe.expected_groups:
                raise ValueError(f"expected {probe.expected_groups} static groups, found {len(specs)}")
            report["resources"] = resource_preflight(specs, device=args.device, collector_devices=args.collector_devices,
                                                       worker_startup_concurrency=args.worker_startup_concurrency, assets_per_process=args.assets_per_process)
            worker_config = {"bundle": str(args.bundle.resolve()), "output": str(output / "workers"),
                             "physics": physics, "robot_mjcf": args.robot_mjcf, "seed": config.seed,
                             "device": args.device, "gamma": config.gamma, "launcher_args": launcher,
                             "collector_devices": args.collector_devices, "worker_startup_concurrency": args.worker_startup_concurrency, "assets_per_process": args.assets_per_process}
            workers = WorkerGroup(specs, worker_config, device=args.device)
            report["native_workers"] = workers.native_reports
            report["process_ids"] = [os.getpid()] + [p.pid for p in workers._processes]
            phase("all_native_groups_ready")
            collecting = time.monotonic()
            for _ in range(probe.collect_steps):
                action = learner.action(workers.snapshot(), exploration=True, update_statistics=True)
                learner.observe(workers.step(action))
            torch.cuda.synchronize(args.device)
            elapsed = time.monotonic() - collecting
            phase("real_collection", ticks=probe.collect_steps, real_transitions=probe.collect_steps * probe.num_envs,
                  collection_seconds=elapsed, environment_steps_per_second=probe.collect_steps * probe.num_envs / elapsed)
            # Same production WM rollout/loss as Learner.wm_update, stopping
            # before Adam.step. One full-size batch establishes this phase's
            # peak; no ten-batch iteration throughput is inferred.
            from .state import unflatten_state, unflatten_object
            window, wm_selection = full_active_wm_windows(replay, config.wm_batch, config.wm_window)
            predicted_s, predicted_o, logits = learner.wm_rollout(window)
            losses = learner.wm.loss(predicted_s, predicted_o, unflatten_state(window["s1"]), unflatten_object(window["o1"]),
                                     logits, window["contact"], window["contact_valid"], window["mask"])
            losses["total"].backward()
            wm_norm = torch.nn.utils.clip_grad_norm_(learner.wm.parameters(), 1.)
            wm_gradients = [p.grad for p in learner.wm.parameters() if p.grad is not None]
            if not bool(torch.isfinite(wm_norm)) or not wm_gradients or not all(bool(torch.isfinite(g).all()) for g in wm_gradients):
                raise FloatingPointError("nonfinite or missing production WM gradients")
            phase("production_wm_backward", batch=config.wm_batch, window=config.wm_window,
                  finite_wm_gradients=True, wm_batches_measured=1, production_wm_batches=config.wm_batches,
                  optimizer_steps=0, real_window_selection=wm_selection, unclipped_gradient_norm=float(wm_norm))
            del window, predicted_s, predicted_o, logits, losses, wm_gradients
            learner.wm.zero_grad(set_to_none=True)
            learner.critic.zero_grad(set_to_none=True)
            start, history, start_selection = stratified_real_starts(replay, probe.pi_batch, bank,
                expected_assets=[s.asset_id for s in specs], horizon=probe.horizon)
            report["real_start_mix"] = validate_real_start_mix(start, history, bank,
                expected_assets=[s.asset_id for s in specs], batch_size=probe.pi_batch, horizon=probe.horizon)
            report["real_start_mix"].update(start_selection)
            for model in models:
                model.zero_grad(set_to_none=True)
            backward_started = time.monotonic()
            with frozen_parameters(learner.wm, learner.critic):
                trajectory = learner.imagine(start, history)
                if not bool(torch.isfinite(trajectory["objective"])):
                    raise FloatingPointError("nonfinite full-batch imagination objective")
                phase("full_imagination_forward", objective=float(trajectory["objective"].detach()))
                (-trajectory["objective"]).backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(learner.pi.parameters(), 1.)
                torch.nn.utils.clip_grad_value_(learner.pi.parameters(), .1)
                gradients = [p.grad for p in learner.pi.parameters() if p.grad is not None]
                finite = bool(torch.isfinite(actor_norm)) and bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
                if not finite:
                    raise FloatingPointError("nonfinite or missing full-batch actor gradients")
                if any(p.grad is not None for m in (learner.wm, learner.critic) for p in m.parameters()):
                    raise RuntimeError("frozen WM/critic received parameter gradients")
                phase("full_imagination_backward", finite_actor_gradients=finite,
                      actor_gradient_tensor_count=len(gradients), backward_and_forward_seconds=time.monotonic() - backward_started,
                      wm_and_critic_parameter_gradients_absent=True, unclipped_gradient_norm=float(actor_norm))
            # The critic uses all 32x4096 imagined transitions and trains its
            # PointNet in each minibatch. Retain the actor trajectory/gradients
            # and full target-value batch exactly as in Learner.critic_update.
            learner.wm.zero_grad(set_to_none=True)
            learner.critic.zero_grad(set_to_none=True)
            from .learner import _stack, _flatten, _detach
            from supermimic.mdp import normalize_xi
            with torch.no_grad():
                S = _flatten(_stack([_detach(s) for s in trajectory["states"]]))
                O = _flatten(_stack([_detach(o) for o in trajectory["objects"]]))
                f = torch.stack(trajectory["frames"]).flatten()
                expand = lambda x: x[None].expand((probe.horizon + 1,) + x.shape).flatten(0, 1)
                arguments = learner._critic_args(S, O, f, expand(normalize_xi(start["xi"])),
                                                expand(start["xi_o"]), expand(start["asset_id"]),
                                         expand(start["support_pose_xyzw"]), expand(start["support_present"]))
                learner.critic.update_normalizers(*arguments[:-1])
                values = critic_target_values(learner.critic, arguments, horizon=probe.horizon,
                                              batch=probe.pi_batch, critic_minibatches=config.critic_minibatches)
                target, targets = values[-1], []
                for tick in range(probe.horizon - 1, -1, -1):
                    target = trajectory["reward"][tick].detach() + config.gamma * (1 - trajectory["failure"][tick].detach()) * (
                        (1 - config.lam) * values[tick + 1] + config.lam * target)
                    targets.append(target)
                targets = torch.stack(targets[::-1]).flatten()
                inputs = tuple(x[:probe.horizon * probe.pi_batch].detach() for x in arguments)
            phase("production_full_critic_target_values", rows=(probe.horizon + 1) * probe.pi_batch,
                  full_pointcloud_batch=False, all_rows_preserved=True,
                  target_batch=(probe.horizon * probe.pi_batch + config.critic_minibatches - 1) // config.critic_minibatches,
                  actor_trajectory_retained=True)
            size = probe.horizon * probe.pi_batch
            ids = torch.randperm(size, device=args.device).tensor_split(config.critic_minibatches)[0]
            predicted_value = learner.critic(*(feature[ids] for feature in inputs))
            value_loss = (predicted_value - targets[ids]).square().mean()
            value_loss.backward()
            critic_norm = torch.nn.utils.clip_grad_norm_(learner.critic.parameters(), 1.)
            critic_gradients = [p.grad for p in learner.critic.parameters() if p.grad is not None]
            if not bool(torch.isfinite(critic_norm)) or not critic_gradients or not all(bool(torch.isfinite(g).all()) for g in critic_gradients):
                raise FloatingPointError("nonfinite or missing production critic gradients")
            phase("production_critic_backward", imagined_states=size, minibatch_size=len(ids),
                  critic_minibatches_measured=1, production_minibatches=config.critic_minibatches,
                  production_epochs=config.critic_epochs, finite_critic_gradients=True,
                  actor_trajectory_and_gradients_retained=True, optimizer_steps=0,
                  unclipped_gradient_norm=float(critic_norm))
            report["passed"] = True
            report["limitations"] = ["No optimizer step, convergence claim, or formal training",
                "Adam moments are fully reserved, but optimizer.step foreach scratch tensors and kernels were not executed",
                "One full-size WM batch and one full-size critic minibatch measured; this is not full-iteration throughput",
                "Real replay starts are resource-stratified for every asset, with replacement; production sampling is unchanged",
                "A subset resource fixture does not establish full-dataset readiness"]
    except BaseException as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        finalize_resource_report(report, workers=workers, sampler=sampler, host_memory=host_memory,
                                 collector_devices=collector_devices, parent_pid=os.getpid())
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    if not report["passed"]:
        raise RuntimeError("resource probe cleanup or memory measurement was incomplete; see report.json")
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json")}, allow_nan=False))


if __name__ == "__main__":
    main()
