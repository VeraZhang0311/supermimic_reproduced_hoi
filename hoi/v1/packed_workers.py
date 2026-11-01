"""Several unchanged fixed-asset solvers per isolated GPU process.

Only process ownership changes: each asset keeps its local motion bank, bins,
physics manager, RNG stream and environment quota. One active-context lease at
any time makes the kit-less IsaacLab singleton an explicit serial boundary.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time
import traceback

import torch

from .workers import (_cat, _cpu, _snapshot, collector_device_list, geometry_group,
                      isolate_worker_cuda, pack_worker_specs, resource_preflight, startup_concurrency)


def _add_failure_note(error, note):
    """Cleanup evidence must never replace the original failure, even pre-3.11."""
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            error.__notes__ = [*getattr(error, "__notes__", ()), note]
    except BaseException:
        pass


def _record_audit_failure(audit, path, original, error):
    audit["passed"] = False
    audit["error"] = original
    try:
        path.write_text(json.dumps(audit, indent=2))
    except BaseException as audit_error:
        _add_failure_note(error, "Writing isolation failure evidence also failed: " + repr(audit_error))


def _packed_main(connection, specs, config):
    slots = None
    audit = audit_path = None
    failure = None
    try:
        import argparse
        import gc
        import os
        assigned = config.get("device", "cuda:0")
        device, visible = isolate_worker_cuda(assigned)
        from supermimic.physics import PhysicsSettings
        PhysicsSettings(**config["physics"]).export()
        from isaaclab.app import add_launcher_args, launch_simulation
        from supermimic.kinematics import G1Kinematics
        from supermimic.robot import default_joint_pos_list
        from .prepare import load_bundle
        from .mdp import HOIMDP
        from .variants import reward_config
        from .native import HOITrackingSim, make_hoi_sim
        from .simulation_slot import SimulationSlots
        from .worker_rng import AssetRNG, asset_seed
        import warp as wp
        wp.init()
        wp.set_device(device)
        parser = argparse.ArgumentParser()
        add_launcher_args(parser)
        args = parser.parse_args(config.get("launcher_args", []))
        cfg = make_hoi_sim(device, args)
        records, reports, snapshots = [], [], []
        shared_shapes = None
        auditing = bool(config.get("audit_isolation", False))
        if auditing:
            if not 2 <= len(specs) <= 4:
                raise ValueError("native isolation audit requires two to four assets per process")
            from .native_isolation import capture_native_fingerprint, assert_native_unchanged
            audit_path = Path(config["output"]) / f"isolation_{specs[0].asset_id:04d}.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit = {"schema": "hoi_packed_native_noninterference_v1", "checks": [], "passed": False,
                     "physics_configuration_changed": False, "optimizer_steps": 0}
            with audit_path.open("x") as stream:
                json.dump(audit, stream, indent=2)

        def capture_inactive(active_id):
            if not auditing:
                return {}
            result = {}
            for other_spec, other_slot, other_rng, other_env, _ in records:
                if other_spec.asset_id != active_id:
                    with other_slot.activate():
                        result[other_spec.asset_id] = capture_native_fingerprint(other_env, other_rng)
            return result

        def verify_inactive(before, active_id, operation):
            if not auditing:
                return
            after = capture_inactive(active_id)
            for asset_id, previous in before.items():
                count = assert_native_unchanged(previous, after[asset_id],
                    label=f"{operation} asset {active_id} must not alter inactive asset {asset_id}")
                audit["checks"].append({"operation": operation, "active_asset": active_id,
                                       "inactive_asset": asset_id, "comparison": count})
            audit_path.write_text(json.dumps(audit, indent=2))

        with launch_simulation(cfg, args), torch.no_grad():
            slots = SimulationSlots(cfg)
            for spec in specs:
                inactive = capture_inactive(spec.asset_id)
                rng = AssetRNG(asset_seed(config.get("seed", 0), spec.asset_id), device)
                output = Path(config["output"]) / f"worker_{spec.asset_id:04d}"
                output.mkdir(parents=True, exist_ok=False)
                os.environ["SUPERMIMIC_ASSET_DIR"] = str(output / "robot_assets")
                with rng.activate():
                    # Construct per-asset kin/bank just like the baseline: preserve
                    # local bin count, normalization, EMA and sampled frame mapping.
                    kin = G1Kinematics(device, mjcf_path=config.get("robot_mjcf"))
                    bank, shapes, bundle = load_bundle(config["bundle"], kin, device=device,
                        require_train=config.get("require_train", True), asset_filter=[spec.asset_id], shared_shapes=shared_shapes)
                    shared_shapes = shapes
                    if bank.num_bins != spec.train_bins:
                        raise RuntimeError("packed worker motion bin count differs from allocation")
                    asset = next(a for a in bundle["assets"] if int(a["asset_id"]) == spec.asset_id)
                    if geometry_group(asset) != spec.geometry_group:
                        raise RuntimeError("packed worker geometry changed after allocation")
                    mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device), shapes,
                                cfg=reward_config(config.get("variant", "hoi_v1")))
                    with slots.create() as slot:
                        env = HOITrackingSim(slot.cfg, spec.num_envs, bank, mdp, asset, output,
                            gamma=config.get("gamma", .99), eval_mode=config.get("eval_mode", False),
                            reset_noise=config.get("reset_noise", True), robot_asset=bundle["native_robot_asset"])
                        del bundle
                        recipe, compiled = env.export_compiled_geometry(output / "compiled_geometry")
                        env.verify_compiled_geometry(compiled, compiled_hands=recipe["hands"])
                        del recipe, compiled
                        snapshot = _cpu(_snapshot(env))
                        report = {"physics": env.physics_report, "contact": env.contacts.manifest(),
                            "spec": asdict(spec), "seed": asset_seed(config.get("seed", 0), spec.asset_id),
                            "assigned_device": assigned, "local_device": device,
                            "cuda_visible_device_token": visible, "variant": config.get("variant", "hoi_v1"),
                            "process_model": "independent solver per asset; serial context lease",
                            "pid": os.getpid(), "slot_runtime": slots.runtime_report}
                report["slot_identity"] = slot.identity_report
                (output / "native.json").write_text(json.dumps(report, indent=2) + "\n")
                verify_inactive(inactive, spec.asset_id, "construct")
                records.append((spec, slot, rng, env, bank))
                reports.append(report)
                snapshots.append(snapshot)
                gc.collect()
                # A bounded timeout applies to each real asset initialization,
                # not the sum of all serial initializations in the pack.
                connection.send(("asset_ready", {"asset_id": spec.asset_id, "pid": os.getpid()}))
            connection.send(("ready", _cat(snapshots, "cpu"), reports))
            while True:
                message = connection.recv()
                if message[0] == "close":
                    break
                snapshots, transitions = [], []
                if message[0] == "step":
                    actions = message[1].split([s.num_envs for s in specs])
                    for (spec, slot, rng, env, bank), action in zip(records, actions):
                        inactive = capture_inactive(spec.asset_id)
                        with slot.activate(), rng.activate():
                            tr = env.step(action.to(device=device, dtype=torch.float32))
                            if hasattr(bank, "global_frame_index"):
                                tr["f"] = bank.global_frame_index[tr["f"]]
                            transitions.append(_cpu(tr))
                            snapshots.append(_cpu(_snapshot(env)))
                        verify_inactive(inactive, spec.asset_id, "step")
                    connection.send(("step", _cat(transitions, "cpu"), _cat(snapshots, "cpu")))
                elif message[0] == "reset":
                    pieces = message[1].split([s.num_envs for s in specs])
                    for (spec, slot, rng, env, bank), global_f in zip(records, pieces):
                        inactive = capture_inactive(spec.asset_id)
                        with slot.activate(), rng.activate():
                            global_f = global_f.to(device=device, dtype=torch.long)
                            mapping = bank.global_frame_index
                            frames = torch.searchsorted(mapping, global_f)
                            if bool((frames >= len(mapping)).any()) or not torch.equal(mapping[frames], global_f):
                                raise ValueError("reset frame does not belong to this packed asset")
                            env.reset(torch.arange(spec.num_envs, device=device), frames)
                            snapshots.append(_cpu(_snapshot(env)))
                        verify_inactive(inactive, spec.asset_id, "reset")
                    connection.send(("reset", _cat(snapshots, "cpu")))
                else:
                    raise ValueError(f"Unknown packed worker command {message[0]}")
            for _, slot, _, env, _ in records:
                with slot.activate():
                    env.close()
            slots.close()
            slots = None
            if auditing:
                audit["passed"] = True
                audit_path.write_text(json.dumps(audit, indent=2))
        # Sent only after native environments, slots, audit, and launcher
        # context have all successfully finished. Exit code is also checked.
        # Python/native object destruction and interpreter shutdown still run
        # after this ACK. If they stall, preserve thread stacks in the job log
        # before the parent's bounded exit grace expires.
        import faulthandler
        faulthandler.dump_traceback_later(10., repeat=True)
        connection.send(("closed",))
    except BaseException as exc:
        failure = exc
        original = traceback.format_exc()
        if audit is not None:
            _record_audit_failure(audit, audit_path, original, exc)
        try:
            connection.send(("error", original + "\n" + "\n".join(getattr(exc, "__notes__", ()))))
        except (OSError, EOFError):
            pass
        raise  # Nonzero child exit remains evidence even if its error pipe is unavailable.
    finally:
        if slots is not None:
            try:
                slots.close()
            except BaseException as cleanup_error:
                if failure is not None:
                    _add_failure_note(failure, "Native slot cleanup also failed: " + repr(cleanup_error))
                else:
                    raise
        try:
            connection.close()
        except OSError as cleanup_error:
            if failure is not None:
                _add_failure_note(failure, "Closing worker pipe also failed: " + repr(cleanup_error))
            else:
                raise


class CollectorShutdownError(RuntimeError):
    def __init__(self, details, surviving_pids):
        self.details, self.surviving_pids = details, surviving_pids
        super().__init__("packed collector shutdown failed: " + json.dumps(
            {"errors": details, "surviving_pids": surviving_pids}, sort_keys=True))


def shutdown_ack_budget(packs):
    """Whole-group grace for serial native teardown within the largest pack."""
    maximum = max((len(pack) for pack in packs), default=1)
    return min(180., max(30., 2 * maximum + 15.))


class PackedWorkerGroup:
    """Keep the baseline global env ordering across contiguous process packs."""
    def __init__(self, specs, config, device="cuda:0", timeout=300):
        self.specs, self.device, self.timeout = list(specs), device, timeout
        self.packs = pack_worker_specs(self.specs, config.get("assets_per_process", 1))
        self.N = sum(s.num_envs for s in specs)
        self._processes, self._pipes, self._snapshots, self.native_reports = [], [], [], []
        self._child_pipes = []
        self.startup_timeout = config.get("startup_timeout", 1800)
        resource_preflight(self.specs, device=device, max_processes=config.get("max_processes"),
            collector_devices=config.get("collector_devices"),
            worker_startup_concurrency=config.get("worker_startup_concurrency"),
            assets_per_process=config.get("assets_per_process", 1))
        devices = collector_device_list(config.get("device", device), config.get("collector_devices"))
        self.worker_devices = [devices[i % len(devices)] for i in range(len(self.packs))]
        context = mp.get_context("spawn")
        concurrency = startup_concurrency(config.get("worker_startup_concurrency"), len(self.packs))
        try:
            for first in range(0, len(self.packs), concurrency):
                end = min(first + concurrency, len(self.packs))
                for i in range(first, end):
                    parent, child = context.Pipe()
                    self._pipes.append(parent)
                    self._child_pipes.append(child)
                    process = context.Process(target=_packed_main,
                        args=(child, self.packs[i], {**config, "device": self.worker_devices[i]}),
                        name=f"hoi-assets-{self.packs[i][0].asset_id}-{self.packs[i][-1].asset_id}")
                    # Register ownership before start/child.close can fail or
                    # be interrupted, even though construction has not returned.
                    self._processes.append(process)
                    process.start()
                    child.close()
                    self._child_pipes[i] = None
                for i in range(first, end):
                    message = self._receive(i, timeout=self.startup_timeout, initializing=True)
                    if message[0] != "ready":
                        raise RuntimeError("packed worker did not publish initialized reset")
                    self._snapshots.append(message[1])
                    self.native_reports.extend(message[2])
        except BaseException as exc:
            try:
                self.close(force=True)
            except BaseException as cleanup_error:
                _add_failure_note(exc, "Collector startup cleanup also failed: " + str(cleanup_error))
                exc.collector_cleanup_errors = getattr(cleanup_error, "details", [str(cleanup_error)])
                exc.collector_live_processes = tuple(self._processes)
            raise

    def _receive(self, index, *, timeout=None, initializing=False):
        while True:
            if not self._pipes[index].poll(self.timeout if timeout is None else timeout):
                raise TimeoutError(f"packed worker {index} timed out; no state inferred")
            message = self._pipes[index].recv()
            if message[0] == "error":
                raise RuntimeError(f"packed worker {index} failed:\n{message[1]}")
            if initializing and message[0] == "asset_ready":
                continue
            return message

    def snapshot(self):
        return _cat(self._snapshots, self.device)

    def step(self, action):
        if action.shape != (self.N, 29):
            raise ValueError("central actions do not match packed environment batch")
        pieces = action.detach().cpu().split([sum(s.num_envs for s in pack) for pack in self.packs])
        for pipe, piece in zip(self._pipes, pieces):
            pipe.send(("step", piece))
        transitions, snapshots = [], []
        for i in range(len(self.packs)):
            message = self._receive(i)
            if message[0] != "step":
                raise RuntimeError("unexpected packed step response")
            transitions.append(message[1])
            snapshots.append(message[2])
        self._snapshots = snapshots
        return _cat(transitions, self.device)

    def reset(self, global_frames):
        frames = torch.as_tensor(global_frames, dtype=torch.long).detach().cpu()
        if frames.shape != (self.N,):
            raise ValueError("full collector reset requires one global frame per environment")
        pieces = frames.split([sum(s.num_envs for s in pack) for pack in self.packs])
        for pipe, piece in zip(self._pipes, pieces):
            pipe.send(("reset", piece))
        snapshots = []
        for i in range(len(self.packs)):
            message = self._receive(i)
            if message[0] != "reset":
                raise RuntimeError("unexpected packed reset response")
            snapshots.append(message[1])
        self._snapshots = snapshots
        return self.snapshot()

    def close(self, force=False):
        """Bounded whole-group cleanup with ACK, escalation and survivor evidence.

        Force cleanup is used before construction can return: it does not wait
        for assets that are still building, and never sends on a blocked pipe.
        Normal cleanup has one group-wide ACK deadline, then bounded join,
        terminate and kill phases. All peers are attempted after any failure.
        """
        errors = []
        acknowledged = set()
        shutdown_started = time.monotonic()
        processes, pipes = list(self._processes), list(self._pipes)
        # Tests may shorten these bounds; production bounds are for the whole
        # group, not multiplied by the number of packed processes.
        ack_timeout = getattr(self, "shutdown_ack_timeout", shutdown_ack_budget(getattr(self, "packs", ())))
        # Native close ACK does not mean the OS process has exited. Permit
        # bounded Python/CUDA teardown rather than killing a healthy child
        # after only two seconds. This is separate from the ACK deadline.
        join_timeout = getattr(self, "shutdown_join_timeout", shutdown_ack_budget(getattr(self, "packs", ())))
        terminate_timeout = getattr(self, "shutdown_terminate_timeout", 3.)
        kill_timeout = getattr(self, "shutdown_kill_timeout", 3.)

        def error(index, operation, exc):
            process = processes[index] if index < len(processes) else None
            errors.append({"worker": index, "pid": getattr(process, "pid", None),
                           "process_name": getattr(process, "name", None),
                           "operation": operation, "error": str(exc)})

        def alive(index):
            process = processes[index]
            if process.pid is None:
                return False
            try:
                return process.is_alive()
            except BaseException as exc:
                error(index, "is_alive", exc)
                return True  # Unknown liveness never permits dropping ownership.

        def joins(indices, timeout, operation):
            deadline = time.monotonic() + timeout
            for index in indices:
                try:
                    processes[index].join(timeout=max(0., deadline - time.monotonic()))
                except BaseException as exc:
                    error(index, operation, exc)

        started = [index for index, process in enumerate(processes) if process.pid is not None]
        if not force:
            deadline = time.monotonic() + ack_timeout
            receivers, ipc_errors = [], {}
            def request_close(index, pipe):
                try:
                    pipe.send(("close",))
                    while True:
                        remaining = max(0., deadline - time.monotonic())
                        if not pipe.poll(remaining):
                            raise TimeoutError("no closed ACK before the group deadline")
                        message = pipe.recv()
                        if message[0] == "error":
                            raise RuntimeError("remote shutdown error:\n" + str(message[1]))
                        if message == ("closed",):
                            acknowledged.add(index)
                            return
                        if message[0] not in ("step", "reset", "ready", "asset_ready"):
                            raise RuntimeError("unexpected shutdown response " + str(message[0]))
                        # An in-flight transition may precede the ACK. Drain
                        # it so the child cannot block on a full result pipe.
                        if time.monotonic() >= deadline:
                            raise TimeoutError("closed ACK deadline expired while draining transitions")
                except BaseException as exc:
                    ipc_errors[index] = exc
            for index, pipe in enumerate(pipes):
                try:
                    receiver = threading.Thread(target=request_close, args=(index, pipe), daemon=True)
                    receiver.start()
                    receivers.append((index, receiver))
                except BaseException as exc:
                    error(index, "closed_ack", exc)
            for index, receiver in receivers:
                try:
                    receiver.join(timeout=max(0., deadline - time.monotonic()))
                    if receiver.is_alive():
                        error(index, "closed_ack", "IPC timed out before the group deadline")
                    elif index in ipc_errors:
                        error(index, "closed_ack", ipc_errors[index])
                except BaseException as exc:
                    error(index, "closed_ack", exc)
            joins(started, join_timeout, "join")

        living = [index for index in started if alive(index)]
        if not force:
            for index in living:
                details = {"exit_grace_seconds": join_timeout,
                           "closed_ack_received": index in acknowledged,
                           "elapsed_seconds": time.monotonic() - shutdown_started}
                proc = Path("/proc") / str(processes[index].pid)
                for field in ("status", "wchan"):
                    try:
                        details[field] = (proc / field).read_text()
                    except OSError:
                        pass  # Linux-only diagnostic; cleanup remains portable.
                error(index, "process_exit_timeout", json.dumps(details))
        for index in living:
            try:
                processes[index].terminate()
            except BaseException as exc:
                error(index, "terminate", exc)
        joins(living, terminate_timeout, "join_after_terminate")
        living = [index for index in living if alive(index)]
        for index in living:
            try:
                processes[index].kill()
            except BaseException as exc:
                error(index, "kill", exc)
        joins(living, kill_timeout, "join_after_kill")
        surviving = [index for index in started if alive(index)]
        for index in surviving:
            error(index, "surviving_process", "still alive after terminate/kill; ownership retained")
        if not force:
            for index in started:
                if index not in surviving and processes[index].exitcode != 0:
                    error(index, "exitcode", processes[index].exitcode)

        for index, pipe in enumerate(pipes):
            try:
                pipe.close()
            except BaseException as exc:
                error(index, "close_parent_pipe", exc)
        for index, pipe in enumerate(getattr(self, "_child_pipes", ())):
            if pipe is not None:
                try:
                    pipe.close()
                except BaseException as exc:
                    error(index, "close_child_pipe", exc)
        # A caller can retry explicit force cleanup using retained live handles.
        self._processes = [processes[index] for index in surviving]
        self._pipes = [pipes[index] for index in surviving if index < len(pipes)]
        self._child_pipes = []
        print(json.dumps({"event": "packed_collector_shutdown", "force": force,
            "elapsed_seconds": time.monotonic() - shutdown_started,
            "ack_budget_seconds": ack_timeout, "exit_budget_seconds": join_timeout,
            "workers": [{"pid": p.pid, "exitcode": p.exitcode,
                         "closed_ack_received": i in acknowledged} for i, p in enumerate(processes)],
            "errors": errors, "surviving_pids": [processes[index].pid for index in surviving]}), flush=True)
        if errors:
            raise CollectorShutdownError(errors, [processes[index].pid for index in surviving])
