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
import traceback

import torch

from .workers import (_cat, _cpu, _snapshot, collector_device_list, geometry_group,
                      isolate_worker_cuda, pack_worker_specs, resource_preflight, startup_concurrency)


def _packed_main(connection, specs, config):
    slots = None
    audit = audit_path = None
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
    except BaseException:
        if audit is not None:
            audit["error"] = traceback.format_exc()
            audit_path.write_text(json.dumps(audit, indent=2))
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if slots is not None:
            try:
                slots.close()
            except BaseException:
                pass  # Original error was already delivered; process owns teardown.
        connection.close()


class PackedWorkerGroup:
    """Keep the baseline global env ordering across contiguous process packs."""
    def __init__(self, specs, config, device="cuda:0", timeout=300):
        self.specs, self.device, self.timeout = list(specs), device, timeout
        self.packs = pack_worker_specs(self.specs, config.get("assets_per_process", 1))
        self.N = sum(s.num_envs for s in specs)
        self._processes, self._pipes, self._snapshots, self.native_reports = [], [], [], []
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
                    process = context.Process(target=_packed_main,
                        args=(child, self.packs[i], {**config, "device": self.worker_devices[i]}),
                        name=f"hoi-assets-{self.packs[i][0].asset_id}-{self.packs[i][-1].asset_id}")
                    process.start()
                    child.close()
                    self._pipes.append(parent)
                    self._processes.append(process)
                for i in range(first, end):
                    message = self._receive(i, timeout=self.startup_timeout, initializing=True)
                    if message[0] != "ready":
                        raise RuntimeError("packed worker did not publish initialized reset")
                    self._snapshots.append(message[1])
                    self.native_reports.extend(message[2])
        except BaseException:
            self.close(force=True)
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
        # The baseline close implementation only owns pipes/processes.
        from .workers import WorkerGroup
        WorkerGroup.close(self, force=force)
