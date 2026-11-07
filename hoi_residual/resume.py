"""Restore committed PPO updates into a fresh simulator and output directory."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import torch

from .data import sha256_file


LEGACY_BASELINE_CHECKPOINT = "439eca2b7b819914b7cbb318a124fbd94642ec1c566c564d170882143d07e531"
LEGACY_BASELINE_IMPLEMENTATION = "471e9be4e34aa5d94ec9b569999d30944229d6a1794ae36915831a836bc47e9a"


def _known_legacy(checkpoint, digest, current):
    """The only cross-version training migration is the measured 1509634 parent."""
    if digest != LEGACY_BASELINE_CHECKPOINT or checkpoint.get("iteration") != 600:
        return False
    old = checkpoint.get("residual_contract", {})
    old_hash = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if old_hash != LEGACY_BASELINE_IMPLEMENTATION or old.get("schema") != current.get("schema"):
        return False
    old_files, files = old["implementation_sha256"], current["implementation_sha256"]
    reviewed = {"contract.py", "env.py", "train.py"}
    return (set(old_files) <= set(files) <= set(old_files) | {"resume.py", "capacity.py"} and
            all(files[name] == sha for name, sha in old_files.items() if name not in reviewed))


def _validate_adam(saved, ppo, updates):
    """Validate every parameter's moments before handing them to PyTorch's slot-based loader."""
    current = ppo.optimizer.state_dict()["param_groups"]
    groups, state = saved.get("param_groups"), saved.get("state")
    if not isinstance(groups, list) or len(groups) != len(current) or not isinstance(state, dict):
        raise ValueError("resume Adam parameter groups or state missing")
    expected_steps = updates * ppo.config.epochs * ppo.config.minibatches
    seen = set()
    for source, target, live in zip(groups, current, ppo.optimizer.param_groups, strict=True):
        if source != target:
            raise ValueError("resume Adam parameter ordering or hyperparameters differ")
        for ident, parameter in zip(source["params"], live["params"], strict=True):
            slot = state.get(ident)
            if ident in seen or not isinstance(slot, dict):
                raise ValueError("resume Adam missing or duplicated parameter state")
            seen.add(ident)
            step = slot.get("step")
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != expected_steps:
                raise ValueError("resume Adam step differs from completed PPO updates")
            for name in ("exp_avg", "exp_avg_sq"):
                moment = slot.get(name)
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape or
                        moment.dtype != parameter.dtype or not torch.isfinite(moment).all() or
                        (name == "exp_avg_sq" and (moment < 0).any())):
                    raise ValueError(f"resume Adam invalid {name}")
    if set(state) != seen:
        raise ValueError("resume Adam contains unknown parameters")


def restore_training(path, expected_sha256, *, policy, ppo, identity, total_iterations,
                     num_envs, seed, capacity):
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError("resume checkpoint SHA256 mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise RuntimeError("resume checkpoint changed during read")
    if not isinstance(ck, dict) or ck.get("schema") != "hoi_residual_ppo_teacher_v2":
        raise ValueError("unsupported resume checkpoint schema")
    legacy = _known_legacy(ck, digest, identity["residual_contract"])
    if not legacy and ck.get("residual_contract") != identity["residual_contract"]:
        raise ValueError("resume implementation identity mismatch")
    iteration = ck.get("iteration")
    if type(iteration) is not int or not 0 < iteration < total_iterations:
        raise ValueError("--iterations must be later than the completed resume checkpoint")
    for key, expected in identity.items():
        if key == "residual_contract":
            continue
        if key == "gmt":
            # Storage may be relocated. Require identical checkpoint bytes,
            # source/target physics, analytic heads and all remaining metadata.
            actual = {k: v for k, v in ck.get("gmt", {}).items() if k != "checkpoint"}
            expected = {k: v for k, v in expected.items() if k != "checkpoint"}
        elif key == "execution":
            actual = {k: v for k, v in ck.get("execution", {}).items() if k != "device"}
            expected = {k: v for k, v in expected.items() if k != "device"}
        elif legacy and key == "experiment":
            actual = ck.get(key, {"reward_profile": "resmimic"})
        elif legacy and key == "reward_inputs":
            actual = ck.get(key, {})
        else:
            actual = ck.get(key)
        if actual != expected:
            raise ValueError(f"resume checkpoint {key} differs from requested training")
    if (ck.get("ppo_config") != asdict(ppo.config) or ck.get("architecture") != policy.architecture() or
            ck.get("ppo_updates", iteration if legacy else None) != iteration or
            ck.get("num_envs_per_rank", 512 if legacy else None) != num_envs or
            ck.get("seed", 0 if legacy else None) != seed):
        raise ValueError("resume PPO configuration, update count, seed, environment count or architecture mismatch")
    previous_capacity = ck.get("capacity", {})
    old_limits = {"epa_horizon": 24, **previous_capacity}
    new_limits = {"epa_horizon": 24, **capacity}
    if (old_limits.keys() != new_limits.keys() or
            any(type(old_limits[k]) is not int or old_limits[k] > new_limits[k] for k in new_limits)):
        raise ValueError("resume requires unchanged or increased solver buffer capacity")
    weights, expected = ck.get("residual"), policy.state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        raise ValueError("resume policy or normalizer keys mismatch")
    for key, value in weights.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != expected[key].dtype or
                value.shape != expected[key].shape or not torch.isfinite(value).all()):
            raise ValueError(f"invalid resume policy/normalizer tensor: {key}")
    if (weights["norm.var"] < 0).any() or weights["norm.count"].item() <= 0 or not (
            0 < weights["_normalizer_revision"].item() <= iteration):
        raise ValueError("invalid resume normalizer statistics")
    if not isinstance(ck.get("optimizer"), dict):
        raise ValueError("resume Adam state missing")
    _validate_adam(ck["optimizer"], ppo, iteration)
    policy.load_state_dict(weights, strict=True)
    ppo.optimizer.load_state_dict(ck["optimizer"])
    ppo.updates = iteration
    return {"checkpoint": str(Path(path).resolve()), "checkpoint_sha256": digest, "iteration": iteration,
            "ppo_updates": ppo.updates, "normalizer_revision": policy.normalizer_revision,
            "parent_capacity": previous_capacity, "capacity": capacity,
            "legacy_migration": legacy, "restored": ["actor", "critic", "shape_encoders", "RMS", "Adam", "PPO updates"],
            "fresh": ["simulator", "reset curriculum", "rollout histories", "per-rank RNG"],
            "normalizer_boundary": "parent RMS restored; unsaved pending rollout inputs replaced by fresh initial observations"}


def checkpoint_paths(root, iteration, final_iteration, save_every, recovery_save_every):
    archive = iteration % save_every == 0 or iteration == final_iteration
    paths = [root / f"checkpoint_{iteration:06d}.pt"] if archive else []
    if recovery_save_every and (archive or iteration % recovery_save_every == 0):
        paths.append(root / "recovery_latest.pt")
    return paths


def save_checkpoint(checkpoint, destinations):
    """Commit a clean PPO update, sharing the same bytes for archive and rolling copy."""
    first, *aliases = destinations
    with NamedTemporaryFile(dir=first.parent, prefix=".ppo-", suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, first)
        for alias in aliases:
            try:
                os.link(first, temporary)
            except OSError:
                # Some parallel filesystems disable hard links.
                with first.open("rb") as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(temporary, alias)
    finally:
        temporary.unlink(missing_ok=True)
