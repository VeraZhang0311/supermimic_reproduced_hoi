"""Fixed-geometry rollout processes with one central batched actor.

Each child owns exactly one SimulationContext. CPU tensor IPC is deliberate:
the first version favours explicit ownership and correctness; throughput and
GPU memory must be measured before selecting a full asset/scale experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import torch


@dataclass(frozen=True)
class WorkerSpec:
    asset_id: int
    num_envs: int
    train_bins: int
    geometry_group: str


def geometry_group(asset):
    """Exact scale, collision recipe and physical profile bind a static scene."""
    scale = asset["exact_scale"]
    if isinstance(scale, (int, float)):
        scale = [scale] * 3
    if len(scale) != 3 or any(not float(s) > 0 for s in scale):
        raise ValueError("exact_scale must contain three positive values")
    key = {"asset": asset["object_asset"], "exact_scale_hex": [float(s).hex() for s in scale],
           "collision_hash": asset["collision_hash"], "physical_profile": asset["physical_profile"]}
    return json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)


def allocate_workers(bundle, motion, num_envs=4096):
    """Largest-remainder allocation by compatible training bins, preserving ΣN."""
    if num_envs < 1:
        raise ValueError("total environment budget must be positive")
    if sorted(int(a["asset_id"]) for a in bundle["assets"]) != list(range(len(bundle["assets"]))):
        raise ValueError("Bundle asset IDs must be unique, contiguous and zero based")
    counts = torch.bincount(motion.bin_asset.detach().cpu(), minlength=len(bundle["assets"])).tolist()
    assets = [a for a in bundle["assets"] if counts[int(a["asset_id"])]]
    keys = [geometry_group(a) for a in assets]
    if len(set(keys)) != len(keys):
        raise ValueError("Bundle must deduplicate equivalent static geometry groups before allocating workers")
    if len(assets) > num_envs:
        raise ValueError(f"{len(assets)} fixed groups cannot each receive an environment from budget {num_envs}")
    weights = [counts[int(a["asset_id"])] for a in assets]
    total = sum(weights)
    residual = num_envs - len(assets)
    exact = [residual * count / total for count in weights]
    quota = [1 + int(x) for x in exact]
    order = sorted(range(len(assets)), key=lambda i: (-(exact[i] - int(exact[i])), int(assets[i]["asset_id"])))
    for i in order[:num_envs - sum(quota)]:
        quota[i] += 1
    specs = [WorkerSpec(int(a["asset_id"]), n, w, key) for a, n, w, key in zip(assets, quota, weights, keys)]
    assert sum(s.num_envs for s in specs) == num_envs
    return specs


def resource_preflight(specs, *, device, max_processes=None):
    if not specs or any(s.num_envs < 1 for s in specs):
        raise ValueError("each fixed geometry worker requires at least one environment")
    if max_processes is not None and len(specs) > max_processes:
        raise ValueError(f"{len(specs)} required fixed-geometry processes exceed explicit process capacity {max_processes}; "
                         "use a separately declared resource experiment; no groups were omitted")
    result = {"groups": len(specs), "num_envs": sum(s.num_envs for s in specs),
              "workers": [asdict(s) for s in specs], "process_model": "spawn; one SimulationContext per process",
              "transport": "CPU tensor IPC; central learner owns actor inference", "device": str(device),
              "gpu_memory_budget_validated": False,
              "native_throughput_validated": False}
    if str(device).startswith("cuda"):
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu_free_bytes=free, gpu_total_bytes=total)
    return result


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(v) for key, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _snapshot(env):
    from supermimic.mdp import flatten_state
    from .state import flatten_object
    S, O = env.state()
    f = env.f
    return {"s": flatten_state(S), "o": flatten_object(O), "a_prev": env.a_prev.clone(),
            "xi": env.xi.clone(), "xi_o": env.xi_o.clone(),
            "f": env.motion.global_frame_index[f] if hasattr(env.motion, "global_frame_index") else f.clone(),
            "ep_step": env.ep_step.clone(), "asset_id": env.asset_id.clone(),
            "contact_missing_counter": env.contact_missing_counter.clone()}


def _worker_main(connection, spec, config):
    """Every simulator import happens after spawn and physics profile setup."""
    env = None
    try:
        import argparse
        import os
        from supermimic.physics import PhysicsSettings
        PhysicsSettings(**config["physics"]).export()
        worker_seed = int(config.get("seed", 0)) + 100003 * spec.asset_id
        torch.manual_seed(worker_seed)
        output = Path(config["output"]) / f"worker_{spec.asset_id:04d}"
        output.mkdir(parents=True, exist_ok=False)
        os.environ["SUPERMIMIC_ASSET_DIR"] = str(output / "robot_assets")
        from isaaclab.app import add_launcher_args, launch_simulation
        from supermimic.kinematics import G1Kinematics
        from supermimic.robot import default_joint_pos_list
        from .prepare import load_bundle
        from .mdp import HOIMDP
        from .variants import reward_config
        from .native import HOITrackingSim, make_hoi_sim
        import warp as wp
        wp.init()
        parser = argparse.ArgumentParser()
        add_launcher_args(parser)
        args = parser.parse_args(config.get("launcher_args", []))
        device = config.get("device", "cuda:0")
        kin = G1Kinematics(device, mjcf_path=config.get("robot_mjcf"))
        bank, shapes, bundle = load_bundle(config["bundle"], kin, device=device, require_train=config.get("require_train", True),
                                          asset_filter=[spec.asset_id])
        asset = next(a for a in bundle["assets"] if int(a["asset_id"]) == spec.asset_id)
        if geometry_group(asset) != spec.geometry_group:
            raise RuntimeError("worker geometry binding changed after central allocation")
        mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device), shapes,
                     cfg=reward_config(config.get("variant", "hoi_v1")))
        cfg = make_hoi_sim(device, args)
        with launch_simulation(cfg, args), torch.no_grad():
            env = HOITrackingSim(cfg, spec.num_envs, bank, mdp, asset, output, gamma=config.get("gamma", .99),
                                 eval_mode=config.get("eval_mode", False), reset_noise=config.get("reset_noise", True),
                                 robot_asset=bundle["native_robot_asset"])
            recipe, compiled = env.export_compiled_geometry(output / "compiled_geometry")
            env.verify_compiled_geometry(compiled, compiled_hands=recipe["hands"])
            initial = {"physics": env.physics_report, "contact": env.contacts.manifest(), "spec": asdict(spec),
                       "seed": worker_seed, "startup_timeout_seconds": config.get("startup_timeout", 1800),
                       "variant": config.get("variant", "hoi_v1")}
            (output / "native.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
            connection.send(("ready", _cpu(_snapshot(env)), initial))
            while True:
                message = connection.recv()
                if message[0] == "close":
                    break
                if message[0] == "step":
                    action = message[1].to(device=device, dtype=torch.float32)
                    tr = env.step(action)
                    if hasattr(bank, "global_frame_index"):
                        tr["f"] = bank.global_frame_index[tr["f"]]
                    connection.send(("step", _cpu(tr), _cpu(_snapshot(env))))
                elif message[0] == "reset":
                    ids = message[1].to(device=device, dtype=torch.long)
                    frames = message[2]
                    if frames is not None:
                        global_f = frames.to(device=device, dtype=torch.long)
                        mapping = bank.global_frame_index
                        frames = torch.searchsorted(mapping, global_f)
                        if bool((frames >= len(mapping)).any()) or not torch.equal(mapping[frames], global_f):
                            raise ValueError("reset frame does not belong to this asset worker")
                    env.reset(ids, frames)
                    connection.send(("reset", _cpu(_snapshot(env))))
                else:
                    raise ValueError(f"Unknown rollout worker command {message[0]}")
            env.close()
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def _cat(values, device):
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values).to(device)
    if isinstance(values[0], dict):
        keys = set.intersection(*(set(v) for v in values))
        return {key: _cat([v[key] for v in values], device) for key in sorted(keys)}
    return values


class WorkerGroup:
    """Synchronous vector collector. Actor inference stays in the caller."""
    def __init__(self, specs, config, device="cuda:0", timeout=300):
        self.specs, self.device, self.timeout = list(specs), device, timeout
        self.startup_timeout = config.get("startup_timeout", 1800)
        self.N = sum(s.num_envs for s in specs)
        self._processes, self._pipes = [], []
        context = mp.get_context("spawn")
        self._snapshots, self.native_reports = [], []
        resource_preflight(self.specs, device=device, max_processes=config.get("max_processes"))
        try:
            for spec in self.specs:
                parent, child = context.Pipe()
                process = context.Process(target=_worker_main, args=(child, spec, config), name=f"hoi-asset-{spec.asset_id}")
                process.start()
                child.close()
                self._pipes.append(parent)
                self._processes.append(process)
            for i in range(len(self.specs)):
                message = self._receive(i, timeout=self.startup_timeout)
                if message[0] != "ready":
                    raise RuntimeError("worker did not publish its initialized joint reset")
                self._snapshots.append(message[1])
                self.native_reports.append(message[2])
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, i, timeout=None):
        pipe = self._pipes[i]
        if not pipe.poll(self.timeout if timeout is None else timeout):
            raise TimeoutError(f"HOI worker asset {self.specs[i].asset_id} timed out; no simulator state inferred")
        message = pipe.recv()
        if message[0] == "error":
            raise RuntimeError(f"HOI worker asset {self.specs[i].asset_id} failed:\n{message[1]}")
        return message

    def snapshot(self):
        return _cat(self._snapshots, self.device)

    def reset(self, global_frames):
        """Jointly reset every environment to explicitly supplied global frames."""
        frames = torch.as_tensor(global_frames, dtype=torch.long).detach().cpu()
        if frames.shape != (self.N,):
            raise ValueError("full collector reset requires one global frame per environment")
        pieces = frames.split([s.num_envs for s in self.specs])
        for pipe, spec, part in zip(self._pipes, self.specs, pieces):
            pipe.send(("reset", torch.arange(spec.num_envs), part))
        snapshots = []
        for i in range(len(self.specs)):
            message = self._receive(i)
            if message[0] != "reset":
                raise RuntimeError("unexpected worker reset response")
            snapshots.append(message[1])
        self._snapshots = snapshots
        return self.snapshot()

    def step(self, action):
        if action.shape != (self.N, 29):
            raise ValueError("central actions do not match the combined fixed-group batch")
        actions = action.detach().cpu().split([s.num_envs for s in self.specs])
        for pipe, a in zip(self._pipes, actions):
            pipe.send(("step", a))
        transitions, snapshots = [], []
        for i in range(len(self.specs)):
            message = self._receive(i)
            if message[0] != "step":
                raise RuntimeError("unexpected worker protocol response")
            transitions.append(message[1])
            snapshots.append(message[2])
        self._snapshots = snapshots
        return _cat(transitions, self.device)

    def close(self, force=False):
        for pipe in self._pipes:
            try:
                pipe.send(("close",))
            except (OSError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=2 if force else 15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in self._pipes:
            pipe.close()
        self._pipes, self._processes = [], []
