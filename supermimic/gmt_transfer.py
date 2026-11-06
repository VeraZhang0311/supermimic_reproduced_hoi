"""Strict full-state GMT fine-tuning across reference datasets, without train imports.

The new run gets its own iteration/wall/simulator counters. Parameters, RMS
buffers and all three Adam states are restored exactly. The caller continues
the LR schedule at ``it + result['learning_rate_iteration_offset']`` and saves
``result['transfer']`` in its new checkpoints.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .privileged import SCHEMA
from .robot_metadata import G1_BODY_NAMES, G1_JOINT_NAMES


MODEL_KEYS = ("wm", "pi", "critic")
OPTIMIZER_KEYS = ("opt_wm", "opt_pi", "opt_v")
REQUIRED_KEYS = (*MODEL_KEYS, *OPTIMIZER_KEYS, "iteration", "wall_time", "sim_steps",
                 "args", "gmt_contract", "saved_at")
CONTRACT_CHANGES = frozenset(("dataset", "motion_manifest_sha256"))
ARG_CHANGES = frozenset(("dataset", "motion_dir", "log_dir", "iterations", "wm_only_iters",
                         "transfer_from", "transfer_parent_iteration"))
MODEL_ARGS = ("goal_frames", "hist", "horizon", "amp")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _args(value):
    return dict(value) if isinstance(value, Mapping) else dict(vars(value))


def normalize_transfer_launcher_args(target_args):
    """Copy transfer CLI defaults into the launcher's explicit disabled state.

    Parent checkpoints capture arguments after launch normalization. Call this
    before strict transfer validation and update the actual launcher namespace
    with its result. Explicit camera/viewer/stream requests remain unchanged
    and must still match the parent. This is not applied to resume/from-scratch.
    """
    result = _args(target_args)
    if result.get("livestream") is None or result["livestream"] == -1:
        result["livestream"] = 0
    for key in ("enable_cameras", "kit_visualizer"):
        if result.get(key) is None:
            result[key] = False
    return result


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_contract_transfer(source, target):
    """Allow dataset/manifest differences only, including same-dataset subsets.

    This pure check also works with contracts from weight-only evaluation
    checkpoints. It neither loads models nor requires optimizer state.
    """
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("GMT transfer requires complete source and target contracts")
    for label, contract in (("source", source), ("target", target)):
        for key in ("version", "dim", "bodies", "actuator_state"):
            if _json(contract.get(key)) != _json(SCHEMA[key]):
                raise ValueError(f"GMT transfer {label} requires the current v3 actuator schema: {key}")
        if contract.get("body_names") != G1_BODY_NAMES or contract.get("joint_names") != G1_JOINT_NAMES:
            raise ValueError(f"GMT transfer {label} body/joint identity differs from hw3")
        if not isinstance(contract.get("physics"), dict) or not isinstance(contract.get("analytic_heads"), dict):
            raise ValueError(f"GMT transfer {label} is missing physics/analytic implementation identity")
        if contract.get("dataset") not in ("lafan1", "boneseed"):
            raise ValueError(f"GMT transfer {label} has an unsupported dataset")
        digest = contract.get("motion_manifest_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError(f"GMT transfer {label} has no valid motion manifest SHA256")
    for key in source.keys() | target.keys():
        if key not in CONTRACT_CHANGES and (key not in source or key not in target or _json(source[key]) != _json(target[key])):
            raise ValueError(f"GMT transfer forbids contract change: {key}")


def validate_transfer_checkpoint(checkpoint, target_contract, target_args, *, source_checkpoint,
                                 run_dir, additional_iterations=20_000, expected_parent_iteration=None):
    """Validate provenance/contract/arguments before touching model state.

    New CLI-only fields absent from the parent are permitted. Existing source
    arguments must match, except the explicit dataset/output/budget/warmup
    changes and transfer provenance flags. Existing resume semantics stay strict.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("GMT transfer requires a complete checkpoint dictionary")
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise ValueError("GMT transfer requires full models and optimizer checkpoint; missing: "+", ".join(missing))
    if type(additional_iterations) is not int or additional_iterations <= 0:
        raise ValueError("additional_iterations must be a positive new-update budget")
    iteration, wall, sim_steps = (checkpoint[key] for key in ("iteration", "wall_time", "sim_steps"))
    if type(iteration) is not int or iteration <= 0:
        raise ValueError("Parent iteration must be a positive integer")
    if expected_parent_iteration is not None and iteration != expected_parent_iteration:
        raise ValueError("Parent checkpoint does not match the explicitly selected iteration")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Parent wall_time must be finite and nonnegative")
    if type(sim_steps) is not int or sim_steps < 0 or not isinstance(checkpoint["saved_at"], str):
        raise ValueError("Parent checkpoint counters/save timestamp are invalid")
    validate_contract_transfer(checkpoint["gmt_contract"], target_contract)
    if checkpoint["gmt_contract"]["dataset"] == target_contract["dataset"]:
        raise ValueError("GMT fine-tuning requires a different target dataset")
    if checkpoint["gmt_contract"]["motion_manifest_sha256"] == target_contract["motion_manifest_sha256"]:
        raise ValueError("GMT fine-tuning requires a new target motion manifest")
    parent_args, target_args = _args(checkpoint["args"]), _args(target_args)
    for key in MODEL_ARGS:
        if key not in parent_args or key not in target_args:
            raise ValueError("Missing GMT architecture/runtime argument: "+key)
    for key, value in parent_args.items():
        if key not in ARG_CHANGES and (key not in target_args or _json(value) != _json(target_args[key])):
            raise ValueError("GMT transfer requires matching parent argument: "+key)
    if (parent_args.get("dataset") != checkpoint["gmt_contract"]["dataset"]
            or target_args.get("dataset") != target_contract["dataset"]):
        raise ValueError("Dataset arguments do not match their GMT contracts")
    if type(target_args.get("iterations")) is not int or target_args["iterations"] != additional_iterations:
        raise ValueError("New iterations must equal the additional-update budget")
    if type(target_args.get("wm_only_iters")) is not int or target_args["wm_only_iters"] != 0:
        raise ValueError("Transferred GMT must begin actor/critic updates with wm_only_iters=0")
    source_path, output = Path(source_checkpoint).resolve(), Path(run_dir).resolve()
    if output == source_path.parent:
        raise ValueError("Transfer output must differ from the source checkpoint directory")
    if "log_dir" not in parent_args or "log_dir" not in target_args:
        raise ValueError("Source and target log_dir are required to protect the original run")
    old_output = Path(parent_args["log_dir"]).resolve()
    if output == old_output or output.is_relative_to(old_output):
        raise ValueError("Transfer output must not reuse the original run directory")
    if Path(target_args["log_dir"]).resolve() != output:
        raise ValueError("Target log_dir does not match the new run directory")
    parent_contract = checkpoint["gmt_contract"]
    transfer = {
        "version": "gmt_cross_dataset_full_state_v1", "parent_checkpoint": str(source_path),
        "parent_iteration": iteration, "parent_wall_time": wall, "parent_sim_steps": sim_steps,
        "parent_saved_at": checkpoint["saved_at"], "parent_contract_sha256": hashlib.sha256(_json(parent_contract).encode()).hexdigest(),
        "source_dataset": parent_contract["dataset"], "source_motion_manifest_sha256": parent_contract["motion_manifest_sha256"],
        "target_dataset": target_contract["dataset"], "target_motion_manifest_sha256": target_contract["motion_manifest_sha256"],
        "target_contract_sha256": hashlib.sha256(_json(target_contract).encode()).hexdigest(),
        "additional_iterations": additional_iterations, "learning_rate_iteration_offset": iteration,
        "preserved": [*MODEL_KEYS, "all_model_normalizers", *OPTIMIZER_KEYS],
    }
    return {"iteration": 0, "wall_time": 0., "sim_steps": 0, "target_iterations": additional_iterations,
            "learning_rate_iteration_offset": iteration, "transfer": transfer}


def _validate_model_state(model, state, label):
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Incompatible model state keys: "+label)
    for name, value in state.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or value.layout != target.layout):
            raise ValueError(f"Incompatible model tensor dimensions/type: {label}.{name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite model state: {label}.{name}")


def _validate_adam_state(optimizer, state, label):
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("GMT full-state transfer requires Adam optimizers: "+label)
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("Incomplete optimizer state: "+label)
    groups, values = state["param_groups"], state["state"]
    if not isinstance(groups, list) or not isinstance(values, dict) or len(groups) != len(optimizer.param_groups):
        raise ValueError("Incompatible optimizer parameter groups: "+label)
    identifiers = []
    for saved, current in zip(groups, optimizer.param_groups, strict=True):
        if not isinstance(saved, dict) or not isinstance(saved.get("params"), list) or len(saved["params"]) != len(current["params"]):
            raise ValueError("Incompatible optimizer parameter count: "+label)
        if not set(current).issubset(saved):
            raise ValueError("Incomplete Adam parameter-group options: "+label)
        for identifier, parameter in zip(saved["params"], current["params"], strict=True):
            identifiers.append(identifier)
            entry = values.get(identifier)
            if not isinstance(entry, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(entry):
                raise ValueError("Missing populated Adam moments: "+label)
            for key, value in entry.items():
                if key == "step":
                    if isinstance(value, torch.Tensor):
                        valid = value.numel() == 1 and bool(torch.isfinite(value).all()) and float(value) >= 0
                    else:
                        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                    if not valid:
                        raise ValueError("Invalid Adam step counter: "+label)
                elif (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                      or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError("Incompatible Adam moment dimensions/type: "+label)
    if len(set(identifiers)) != len(identifiers) or set(values) != set(identifiers):
        raise ValueError("Optimizer state does not cover its exact parameter mapping: "+label)


def load_transfer_checkpoint(path, target_contract, target_args, *, wm, pi, critic,
                             opt_wm, opt_pi, opt_v, run_dir, additional_iterations=20_000,
                             expected_parent_iteration=None, map_location="cpu"):
    """Restore all models/RMS/Adam states; return fresh counters and provenance.

    Use an immutable snapshot of the selected full ``last.pt``, since the
    original training process may atomically replace its own last.pt later.
    Validation completes for all six destinations before any load occurs.
    """
    path = Path(path).resolve()
    before = _sha256(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if _sha256(path) != before:
        raise ValueError("Parent checkpoint changed while loading; snapshot it first")
    result = validate_transfer_checkpoint(checkpoint, target_contract, target_args,
        source_checkpoint=path, run_dir=run_dir, additional_iterations=additional_iterations,
        expected_parent_iteration=expected_parent_iteration)
    models, optimizers = (wm, pi, critic), (opt_wm, opt_pi, opt_v)
    for label, model in zip(MODEL_KEYS, models, strict=True):
        _validate_model_state(model, checkpoint[label], label)
    for label, optimizer, model in zip(OPTIMIZER_KEYS, optimizers, models, strict=True):
        if [id(p) for group in optimizer.param_groups for p in group["params"]] != [id(p) for p in model.parameters()]:
            raise ValueError("Optimizer parameter order/ownership does not match its model: "+label)
        _validate_adam_state(optimizer, checkpoint[label], label)
    for label, model in zip(MODEL_KEYS, models, strict=True):
        model.load_state_dict(checkpoint[label], strict=True)
    for label, optimizer in zip(OPTIMIZER_KEYS, optimizers, strict=True):
        optimizer.load_state_dict(checkpoint[label])
    result["transfer"]["parent_checkpoint_sha256"] = before
    return copy.deepcopy(result)
