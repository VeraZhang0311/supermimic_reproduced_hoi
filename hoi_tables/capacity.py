"""Full-sized, zero-optimizer-update allocation and gradient validation.

The caller keeps the real native simulator and complete replay allocation alive.
Run this in a disposable validation process on the same allocation as training;
the formal training process must initialize fresh models/RMS/replay afterward.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
import hashlib
import math
import os
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist


class CapacityValidationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _hash_state(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((tuple(item.shape), str(item.dtype))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(type(item)).encode())
            for field in item: visit(field)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _models(learner):
    return {"pi": learner.pi, "wm": learner.wm, "critic": learner.critic}


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _configuration(learner, replay, env, global_config, require_production, expected_assets, workers_per_gpu):
    local = learner.config
    local.validate(); global_config.validate()
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if replay.N != local.num_envs or replay.T != local.buffer_steps or env.N != local.num_envs:
        raise ValueError("Capacity validation requires the actual configured native worlds and full replay storage")
    if replay.size < local.wm_window:
        raise ValueError("Real replay must be prefilled before capacity validation")
    if learner.buffer is not replay:
        raise ValueError("Capacity validation must use this learner's actual replay")
    if require_production:
        expected = {"num_envs": 4096, "buffer_steps": 1000, "wm_batch": 2048, "pi_batch": 4096,
                    "wm_batches": 10, "wm_window": 8, "horizon": 32, "critic_epochs": 5,
                    "critic_minibatches": 4, "wm_only_iters": 500, "experiment_kind": "formal"}
        if world != len(expected_assets) or torch.device(learner.device).type != "cuda":
            count = "two" if len(expected_assets)==2 else str(len(expected_assets))
            raise ValueError(f"Formal capacity validation requires {count} live CUDA/Gloo workers, one per exact profile asset")
        rank = dist.get_rank()
        if (len(set(expected_assets)) != len(expected_assets) or type(workers_per_gpu) is not int
                or workers_per_gpu < 1 or world % workers_per_gpu):
            raise ValueError("Invalid profile asset/GPU topology")
        if (env.asset.get("asset_id") != rank or env.asset.get("object_asset") != expected_assets[rank]):
            raise ValueError("Capacity worker does not own its exact profile asset")
        if dist.get_backend() != "gloo":
            raise ValueError("Formal tables capacity validation requires the configured Gloo synchronization")
        for name, value in expected.items():
            if getattr(global_config, name) != value:
                raise ValueError(f"Formal capacity configuration changed: {name}")
            wanted = value//world if name in ("num_envs", "wm_batch", "pi_batch") else value
            if getattr(local, name) != wanted:
                raise ValueError(f"Worker capacity configuration changed: {name}")
        if learner.gradient_sync is None:
            raise ValueError("Formal capacity validation requires the real distributed gradient synchronization")
        if "L40" not in torch.cuda.get_device_properties(learner.device).name:
            raise ValueError("Formal tables capacity must be validated on the authorized L40 class")
    return world


def _memory(device):
    device = torch.device(device)
    if device.type != "cuda":
        return {"available": False, "reason": "CPU contract fixture; no GPU capacity claim"}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    return {"available": True, "device": str(device), "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "driver_free_bytes": free, "driver_total_bytes": total,
            "driver_used_bytes": total-free,
            "includes_native_memory": "driver totals include live Warp/native allocations; PyTorch peaks exclude them"}


def _native(env, required):
    if getattr(env, "physics", None) is None:
        if required: raise ValueError("Formal capacity validation requires a real live native simulator")
        return {"available": False, "reason": "CPU contract fixture"}
    from hoi_residual.capacity import overflow_detail, capacity_readback
    status = overflow_detail(env.physics)
    if status["overflow_world_count"]:
        raise RuntimeError(f"Native overflow invalidates capacity evidence: {status['overflow_bit_counts']}")
    for state in (env.state(), env.object_state()):
        if not all(bool(torch.isfinite(value).all()) for value in state):
            raise FloatingPointError("Nonfinite native state invalidates capacity evidence")
    return {"available": True, "overflow": status, "allocation": capacity_readback(env.physics)}


def _full_real_window(replay, config):
    """Repeat only genuine complete fit windows to test worst-case active BxK.

    Normal training still samples short failed prefixes. A capacity probe must
    also exercise all B rows active for all K ticks, which random initial falls
    can otherwise hide. Replacement sampling does not invent transitions.
    """
    for _ in range(16):
        window = replay.sample_windows(config.wm_batch, config.wm_window, split="fit")
        complete = window["mask"].bool().all(1).nonzero().flatten()
        if len(complete):
            chosen = complete[torch.arange(config.wm_batch, device=complete.device) % len(complete)]
            return {key: value[chosen] if isinstance(value, torch.Tensor) and value.shape[0] == len(window["mask"]) else value
                    for key, value in window.items()}, len(complete)
    raise ValueError("Replay has no sampled complete real WM fit window; collect additional real transitions before preflight")


@contextmanager
def _zero_step_adam(learner, telemetry):
    """Allocate real Adam moments and conservative foreach scratch, block steps."""
    saved, scratch = {}, []
    try:
        for name, optimizer in learner.optimizers.items():
            if not isinstance(optimizer, torch.optim.Adam):
                raise TypeError("Capacity reservation currently covers the configured torch.optim.Adam only")
            saved[name] = {"state": {p: dict(state) for p, state in optimizer.state.items()},
                           "had_step": "step" in optimizer.__dict__, "step": optimizer.__dict__.get("step")}
            for group in optimizer.param_groups:
                if group.get("differentiable", False): raise ValueError("Differentiable Adam requires a separate capacity implementation")
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    if not state:
                        state["step"] = torch.zeros((), device=parameter.device if group.get("capturable") or group.get("fused") else "cpu")
                        state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        if group.get("amsgrad", False):
                            state["max_exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            telemetry["adam_state_bytes"][name] = sum(_bytes(v) for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))
            def blocked_step(*args, _name=name, _optimizer=optimizer, **kwargs):
                if args or kwargs:
                    raise RuntimeError("Capacity guard refuses optimizer closures or step arguments")
                gradients = [p.grad for group in _optimizer.param_groups for p in group["params"] if p.grad is not None]
                if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
                    raise FloatingPointError(f"Missing/nonfinite {_name} gradients during real capacity probe")
                telemetry["blocked_optimizer_calls"][_name] += 1
                telemetry["gradient_tensor_counts"][_name] = len(gradients)
                return None
            optimizer.step = blocked_step
        largest = max(sum(_bytes(p) for p in module.parameters()) for module in _models(learner).values())
        # Adam's CUDA foreach path can materialize a parameter-sized denominator
        # and auxiliary tensors. Reserve two largest-model byte equivalents
        # throughout the probe; no optimizer math or weight update is executed.
        scratch.append(torch.zeros(2*largest, dtype=torch.uint8, device=learner.device))
        telemetry["adam_foreach_scratch_reserved_bytes"] = 2*largest
        # Production prefill intentionally remains 64 real steps. Later replay
        # sampling scans up to T*N rows and creates index/mask/location tensors.
        # Reserve that future workspace explicitly instead of fabricating rows
        # or pretending this prefilled scan measured a full replay scan.
        replay_index_bytes = 128*learner.buffer.T*learner.buffer.N
        scratch.append(torch.zeros(replay_index_bytes, dtype=torch.uint8, device=learner.device))
        telemetry["replay_index_scratch_reserved_bytes"] = replay_index_bytes
        telemetry["replay_index_scratch_formula"] = "128 bytes * replay.capacity_steps * replay.num_envs"
        yield
    finally:
        for name, original in saved.items():
            optimizer = learner.optimizers[name]
            optimizer.state.clear()
            optimizer.state.update(original["state"])
            if original["had_step"]: optimizer.step = original["step"]
            elif "step" in optimizer.__dict__: del optimizer.__dict__["step"]
        scratch.clear()


def validate_capacity(learner, replay, env, global_config, *, require_production=True,
                      expected_assets=("largetable","smalltable"), workers_per_gpu=1):
    """Exercise full WM/BPTT/critic gradients with zero original Adam.step calls.

    Success reports are gathered across the live Gloo workers. On a local
    failure no new collective is entered: raise with the local report so the
    torchrun supervisor can terminate a peer blocked in an earlier collective.
    ``require_production=False`` exists only for CPU contract unit fixtures.
    """
    world = _configuration(learner, replay, env, global_config, require_production, expected_assets, workers_per_gpu)
    models = _models(learner)
    original_models = {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                       for name, module in models.items()}
    original_hash = {name: _hash_state(state) for name, state in original_models.items()}
    original_optimizers = {name: _hash_state(optimizer.state_dict()) for name, optimizer in learner.optimizers.items()}
    original_gradients = {name: [None if p.grad is None else p.grad.detach().cpu().clone() for p in module.parameters()]
                          for name, module in models.items()}
    requires_grad = [(p, p.requires_grad) for module in models.values() for p in module.parameters()]
    rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    device = torch.device(learner.device)
    if device.type == "cuda": rng["cuda"] = torch.cuda.get_rng_state(device)
    generators = [(obj.generator, obj.generator.get_state().clone()) for obj in (replay, getattr(env, "sampler", None))
                  if obj is not None and hasattr(obj, "generator")]
    scalar_state = {name: getattr(learner, name) for name in ("iteration",) if hasattr(learner, name)}
    contacts = {name: getattr(learner, name).clone() for name in ("contact_positive", "contact_negative") if hasattr(learner, name)}
    replay_before = {key: getattr(replay, key) for key in ("ptr", "size", "appended_steps")}
    replay_versions = {key: (id(value), value.data_ptr(), value._version) for key, value in replay.data.items()}
    asset = getattr(env, "asset", {})
    manifest = getattr(learner.motion, "manifest", {})
    report = {"schema": "hoi_tables_full_allocation_capacity_v1", "passed": False,
              "production_configuration": require_production, "rank": dist.get_rank() if world > 1 else 0,
              "world_size": world, "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "profile_assets":list(expected_assets),"workers_per_gpu":workers_per_gpu,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope": "real full-sized forward/backward and reserved Adam memory; no optimizer update or convergence claim",
              "global_config": global_config.to_dict(), "worker_config": learner.config.to_dict(),
              "input_identity": {"motion_manifest_sha256": getattr(learner.motion, "manifest_sha256", None),
                                 "reference_cache_sha256": manifest.get("cache", {}).get("sha256"),
                                 "selection": deepcopy(manifest.get("selection", {})),
                                 "object_asset": {key: deepcopy(asset[key]) for key in ("asset_id", "object_asset", "native_asset_sha256", "geometry_cache") if key in asset},
                                 "model_and_rms_sha256": original_hash,
                                 "reward_contract": deepcopy(getattr(learner.reward, "contract", {})),
                                 "reference_contact_cache": deepcopy(getattr(learner.reward, "contact_cache_metadata", None))},
              "replay_storage_bytes": sum(_bytes(tensor) for tensor in replay.data.values()),
              "replay_allocation": {"capacity_steps": replay.T, "num_envs": replay.N,
                                    "real_prefilled_steps": replay.size,
                                    "full_replay_scan_measured": replay.size == replay.T,
                                    "future_index_workspace": "explicit conservative reservation, not measured full-scan temporaries",
                                    "fabricated_transitions": False},
              "optimizer_updates": 0, "normalizer_updates": 0,
              "blocked_optimizer_calls": {name: 0 for name in learner.optimizers},
              "gradient_tensor_counts": {}, "adam_state_bytes": {}, "phases": []}
    error, started = None, time.monotonic()
    try:
        report["native_before"] = _native(env, require_production)
        report["memory_before"] = _memory(device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        with _zero_step_adam(learner, report):
            window, complete_count = _full_real_window(replay, learner.config)
            report["complete_real_wm_windows_in_sample"] = complete_count
            report["wm_validation_all_rows_all_ticks_active"] = bool(window["mask"].bool().all())
            for index in range(learner.config.wm_batches):
                metrics = learner.wm_update(window=window)
                if any(not math.isfinite(float(v)) for v in metrics.values()): raise FloatingPointError("Nonfinite full-sized WM capacity metrics")
                report["phases"].append({"phase": "wm", "batch": index+1, "metrics": metrics, "memory": _memory(device)})
            del window
            for optimizer in learner.optimizers.values(): optimizer.zero_grad(set_to_none=True)
            start = replay.sample_starts(learner.config.pi_batch, horizon=learner.config.horizon, motion=learner.motion)
            metrics = learner.policy_update(start=start)
            if any(not math.isfinite(float(v)) for v in metrics.values()): raise FloatingPointError("Nonfinite full-sized actor/critic capacity metrics")
            report["phases"].append({"phase": "actor_H32_and_all_critic_minibatches", "metrics": metrics, "memory": _memory(device)})
            del start
            expected_calls = {"wm": learner.config.wm_batches, "pi": 1,
                              "critic": learner.config.critic_epochs*learner.config.critic_minibatches}
            if report["blocked_optimizer_calls"] != expected_calls:
                raise RuntimeError("Capacity probe did not exercise every configured update/minibatch")
            report["memory_peak_with_adam_reservations"] = _memory(device)
        report["native_after"] = _native(env, require_production)
        if any(_hash_state(module.state_dict()) != original_hash[name] for name, module in models.items()):
            raise RuntimeError("Capacity probe attempted to mutate model parameters or RMS buffers")
        if any(_hash_state(opt.state_dict()) != original_optimizers[name] for name, opt in learner.optimizers.items()):
            raise RuntimeError("Capacity probe attempted to mutate original optimizer state")
        if (replay_before != {key: getattr(replay, key) for key in replay_before} or
                replay_versions != {key: (id(value), value.data_ptr(), value._version) for key, value in replay.data.items()}):
            raise RuntimeError("Capacity probe mutated replay tensors/pointers/counters")
        report["passed"] = True
    except Exception as exc:
        error = exc
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try: report["memory_on_error"] = _memory(device)
        except Exception as memory_error: report["memory_readback_error"] = str(memory_error)
    finally:
        for name, module in models.items():
            module.load_state_dict(original_models[name])
            for parameter, gradient in zip(module.parameters(), original_gradients[name]):
                parameter.grad = None if gradient is None else gradient.to(parameter)
        for parameter, flag in requires_grad: parameter.requires_grad_(flag)
        for name, value in scalar_state.items(): setattr(learner, name, value)
        for name, value in contacts.items(): getattr(learner, name).copy_(value)
        torch.set_rng_state(rng["torch"]); random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        if "cuda" in rng: torch.cuda.set_rng_state(rng["cuda"], device)
        for generator, state in generators: generator.set_state(state)
        report["elapsed_s"] = time.monotonic()-started
        report["models_and_rms_unchanged"] = all(_hash_state(module.state_dict()) == original_hash[name] for name, module in models.items())
        report["optimizer_states_unchanged"] = all(_hash_state(opt.state_dict()) == original_optimizers[name] for name, opt in learner.optimizers.items())
        report["replay_tensor_versions_unchanged"] = replay_versions == {key: (id(value), value.data_ptr(), value._version) for key, value in replay.data.items()}
        report["rng_restored"] = True
        report["requires_fresh_training_process"] = True
    if error is not None:
        raise CapacityValidationError(f"Tables capacity validation failed: {error}", report) from error
    peers = [None]*world
    if world > 1: dist.all_gather_object(peers, report)
    else: peers[0] = report
    if not all(peer["passed"] and peer["models_and_rms_unchanged"] and peer["optimizer_states_unchanged"] for peer in peers):
        report["passed"] = False
        raise CapacityValidationError("A distributed tables capacity peer did not pass", report)
    if require_production:
        devices = [(p["hostname"], p["memory_peak_with_adam_reservations"]["uuid"]) for p in peers]
        if all(uuid != "unavailable" for _, uuid in devices) and (
                len(set(devices)) != world//workers_per_gpu or any(n!=workers_per_gpu for n in Counter(devices).values())):
            report["passed"] = False
            raise CapacityValidationError("Formal capacity workers differ from the explicit physical GPU sharing topology", report)
    peaks = [p["memory_peak_with_adam_reservations"] for p in peers if p["memory_peak_with_adam_reservations"]["available"]]
    return {"schema": report["schema"], "passed": True, "world_size": world,
            "production_configuration": require_production, "optimizer_updates": 0,
            "max_peak_allocated_bytes": max((p["peak_allocated_bytes"] for p in peaks), default=0),
            "max_peak_reserved_bytes": max((p["peak_reserved_bytes"] for p in peaks), default=0),
            "workers": peers, "requires_fresh_training_process": True}
