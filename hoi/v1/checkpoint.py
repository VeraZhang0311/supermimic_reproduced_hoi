"""Strict HOI continuation and explicit, additive motion-only initialization.

Legacy initialization intentionally does not restore iteration or Adam tensors.
Same-schema continuation restores all three optimizers but still needs fresh
real replay prefill; simulator, replay and RNG states are not included.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from .networks import feature_schema
from .observations import LEGACY, MATCHING
from .actor_fusion import MLP_ACTOR, FUSION_ACTOR, actor_architecture_contract
from .randomization import LEGACY_DR, randomization_contract
from .robot_context import uses_torso_mass, torso_mass_contract

CHECKPOINT_SCHEMA = "supermimic_hoi_v1_checkpoint_1"
MODEL_KEYS = ("pi", "wm", "critic")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _physics(value):
    value = asdict(value) if is_dataclass(value) else dict(value)
    if any(type(value.get(k)) is not bool for k in ("hand_collision", "self_collision")):
        raise ValueError("checkpoint physics requires explicit boolean hand_collision and self_collision")
    # Keep native solver/asset/physical fields too, if supplied.
    _json(value)
    return value


def _data(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("a nonempty frozen data identity is required")
    _json(value)
    return deepcopy(value)


def _signature(module):
    result = {
        "geometry_input_enabled": module.geometry_input_enabled,
        "state_shapes": {key: list(value.shape) for key, value in module.state_dict().items()},
        "module_types": {key: type(value).__name__ for key, value in module.named_modules()},
        "normalizers": {key: {"eps": value.eps, "max_count": value.max_count}
                        for key, value in module.named_modules() if hasattr(value, "max_count") and hasattr(value, "eps")},
    }
    if getattr(module, "actor_architecture", MLP_ACTOR) == FUSION_ACTOR:
        result["actor_architecture_contract"] = actor_architecture_contract(module.actor_architecture, module.observation_profile)
    if getattr(module, "torso_mass_context", False):
        result["robot_mass_context"] = torso_mass_contract()
    return result


def _models(policy, wm, critic):
    return {"pi": policy, "wm": wm, "critic": critic}


def make_checkpoint(policy, wm, critic, optimizers, *, iteration, physics, data_identity,
                    initialization, extra=None, variant=None):
    from .variants import validate_variant, geometry_input_enabled
    variant = validate_variant(variant if variant is not None else (extra or {}).get("config", {}).get("variant", "hoi_v1"))
    if any(model.geometry_input_enabled != geometry_input_enabled(variant) for model in (policy, wm, critic)):
        raise ValueError("checkpoint variant and all three geometry-input flags must agree")
    if extra is not None and extra.get("config", {}).get("variant", variant) != variant:
        raise ValueError("checkpoint explicit variant differs from its saved configuration")
    if set(optimizers) != set(MODEL_KEYS):
        raise ValueError("all pi/wm/critic optimizer states are required")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint iteration must be a nonnegative integer")
    if not isinstance(initialization, dict) or not initialization:
        raise ValueError("record initialization checkpoint identity and migration report")
    _json(initialization)
    models = _models(policy, wm, critic)
    dr_profile = (extra or {}).get("config", {}).get("domain_randomization_profile", LEGACY_DR)
    mass_enabled = uses_torso_mass(dr_profile)
    if any(model.torso_mass_context != mass_enabled for model in (wm, critic)):
        raise ValueError("checkpoint WM/critic torso mass context differs from the DR profile")
    data = _data(data_identity)
    result = {
        "schema_version": CHECKPOINT_SCHEMA,
        "variant": variant,
        "feature_schema": feature_schema(policy.observation_profile, torso_mass_context=mass_enabled,
                                         actor_architecture=policy.actor_architecture),
        "architecture": {key: _signature(model) for key, model in models.items()},
        "physics": _physics(physics),
        "data_identity": data,
        "data_identity_sha256": hashlib.sha256(_json(data)).hexdigest(),
        "initialization": deepcopy(initialization),
        "iteration": iteration,
        "continuation": {"fresh_replay_prefill_required": True, "bitwise_resume": False,
                         "saved_simulator_replay_rng": False},
        "optimizers": {key: deepcopy(optimizers[key].state_dict()) for key in MODEL_KEYS},
        **{key: deepcopy(model.state_dict()) for key, model in models.items()},
    }
    if extra is not None:
        result["extra"] = deepcopy(extra)
    configured_profile = (extra or {}).get("config", {}).get("observation_profile", policy.observation_profile)
    if configured_profile != policy.observation_profile:
        raise ValueError("checkpoint actor profile differs from saved configuration")
    configured_architecture = (extra or {}).get("config", {}).get("actor_architecture", MLP_ACTOR)
    if configured_architecture != policy.actor_architecture:
        raise ValueError("checkpoint actor architecture differs from saved configuration")
    if policy.actor_architecture == FUSION_ACTOR:
        result["actor_architecture_contract"] = actor_architecture_contract(policy.actor_architecture, policy.observation_profile)
    if policy.observation_profile != LEGACY or dr_profile != LEGACY_DR:
        result["experiment_contract"] = {"observation_profile": policy.observation_profile,
                                         "domain_randomization": randomization_contract(dr_profile)}
    return result


def save_checkpoint(path, *args, **kwargs):
    """Exclusive create keeps scientific snapshots immutable."""
    checkpoint = make_checkpoint(*args, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return checkpoint


def _validate_state(model, state, name):
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError(f"{name}: checkpoint state keys differ from the exact model schema")
    for key, target in model.state_dict().items():
        if not isinstance(state[key], torch.Tensor) or state[key].shape != target.shape:
            raise ValueError(f"{name}.{key}: checkpoint tensor shape differs")
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"{name}.{key}: nonfinite checkpoint tensor")
        if (key.endswith(".var") or key.endswith(".count")) and (state[key] < 0).any():
            raise ValueError(f"{name}.{key}: invalid negative normalizer statistic")


def checkpoint_variant(checkpoint):
    """Only the original full objective may use a pre-variant v1 snapshot."""
    from .variants import validate_variant
    configured = checkpoint.get("extra", {}).get("config", {}).get("variant")
    variant = validate_variant(checkpoint.get("variant", configured if configured is not None else "hoi_v1"))
    if configured is not None and configured != variant:
        raise ValueError("checkpoint variant and saved training configuration differ")
    return variant


def checkpoint_training_progress(checkpoint):
    """Report actual current-run updates, never infer learning from iteration.

    A continuation enters a new train_loop and starts these counters at zero.
    Missing old counters remain unknown; an initialized motion policy is not
    described as untrained merely because this HOI run has no actor updates.
    """
    extra = checkpoint.get("extra", {})
    extra = extra if isinstance(extra, dict) else {}
    counts = {name: value if type(value := extra.get(name)) is int and value >= 0 else None
              for name in ("wm_updates_this_run", "policy_updates_this_run")}
    known = all(value is not None for value in counts.values())
    if not known:
        stage = "unknown"
    elif counts["policy_updates_this_run"] > 0:
        stage = "policy_updated_this_run"
    elif counts["wm_updates_this_run"] > 0:
        stage = "wm_only_this_run"
    else:
        stage = "no_optimizer_updates_this_run"
    return {**counts, "stage": stage, "counts_status": "known" if known else "unknown",
            "count_scope": "since current train_loop entry; counters reset on continuation; not cumulative across lineage",
            "stage_evidence": "explicit checkpoint extra counters only; never inferred from iteration or policy gate",
            "initialization_identity": "checkpoint.initialization separately records prior motion training or continuation"}


def _checkpoint_actor_features(checkpoint):
    """Recover exact architecture from the schema; missing legacy config means MLP."""
    for profile, architecture in ((LEGACY, MLP_ACTOR), (MATCHING, MLP_ACTOR), (MATCHING, FUSION_ACTOR)):
        for enabled in (False, True):
            if checkpoint.get("feature_schema") != feature_schema(
                    profile, torso_mass_context=enabled, actor_architecture=architecture):
                continue
            configured = checkpoint.get("extra", {}).get("config", {}).get("actor_architecture", MLP_ACTOR)
            if configured != architecture:
                raise ValueError("checkpoint actor architecture differs from saved configuration")
            contract = checkpoint.get("actor_architecture_contract")
            if architecture == FUSION_ACTOR:
                if contract != actor_architecture_contract(architecture, profile):
                    raise ValueError("checkpoint actor architecture contract differs or is missing")
            elif contract is not None:
                raise ValueError("checkpoint actor architecture contract differs from MLP schema")
            return profile, enabled, architecture
    raise ValueError("checkpoint feature schema/version differs; five-channel hand/foot-object contact requires an explicit migration for old three-channel HOI checkpoints")


def checkpoint_observation_profile(checkpoint):
    return _checkpoint_actor_features(checkpoint)[0]


def checkpoint_actor_architecture(checkpoint):
    return _checkpoint_actor_features(checkpoint)[2]


def checkpoint_torso_mass_context(checkpoint):
    return _checkpoint_actor_features(checkpoint)[1]


def checkpoint_randomization_profile(checkpoint):
    profile = checkpoint.get("extra", {}).get("config", {}).get("domain_randomization_profile", LEGACY_DR)
    contract = checkpoint.get("experiment_contract")
    observation = checkpoint_observation_profile(checkpoint)
    if checkpoint_torso_mass_context(checkpoint) != uses_torso_mass(profile):
        raise ValueError("checkpoint torso mass context is missing or differs from its DR profile")
    if observation != LEGACY or profile != LEGACY_DR:
        if contract != {"observation_profile": observation, "domain_randomization": randomization_contract(profile)}:
            raise ValueError("checkpoint experiment/DR contract differs or is missing")
    return profile


def validate_checkpoint_features(checkpoint, observation_profile=None, actor_architecture=None):
    """Reject incompatible continuation on CPU, before workers/output creation."""
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint feature schema/version differs; five-channel hand/foot-object contact requires an explicit migration for old three-channel HOI checkpoints")
    actual = checkpoint_observation_profile(checkpoint)
    if observation_profile is not None and actual != observation_profile:
        raise ValueError("checkpoint actor observation profile differs; initialize a new experiment instead of resuming v1")
    if actor_architecture is not None and checkpoint_actor_architecture(checkpoint) != actor_architecture:
        raise ValueError("checkpoint actor architecture differs; initialize a new experiment instead of resuming another architecture")
    checkpoint_randomization_profile(checkpoint)


def load_checkpoint(path, policy, wm, critic, optimizers=None, *, physics, data_identity, map_location="cpu", variant="hoi_v1",
                    domain_randomization_profile=None):
    from .variants import validate_variant, geometry_input_enabled
    checkpoint = torch.load(path, map_location=map_location, weights_only=False) if not isinstance(path, dict) else path
    validate_checkpoint_features(checkpoint, policy.observation_profile, policy.actor_architecture)
    if any(model.torso_mass_context != checkpoint_torso_mass_context(checkpoint) for model in (wm, critic)):
        raise ValueError("checkpoint WM/critic torso mass context differs from requested models")
    if domain_randomization_profile is not None and checkpoint_randomization_profile(checkpoint) != domain_randomization_profile:
        raise ValueError("checkpoint domain randomization profile differs")
    if checkpoint_variant(checkpoint) != validate_variant(variant):
        raise ValueError("checkpoint experiment variant differs from the requested continuation/evaluation")
    if type(checkpoint.get("iteration")) is not int or checkpoint["iteration"] < 0:
        raise ValueError("invalid checkpoint iteration")
    if not isinstance(checkpoint.get("initialization"), dict) or not checkpoint["initialization"]:
        raise ValueError("checkpoint initialization identity is missing")
    if checkpoint.get("continuation") != {"fresh_replay_prefill_required": True, "bitwise_resume": False,
                                         "saved_simulator_replay_rng": False}:
        raise ValueError("unsupported checkpoint replay/simulator/RNG restoration contract")
    if checkpoint.get("physics") != _physics(physics):
        raise ValueError("HOI continuation/evaluation physics differs from the saved checkpoint")
    data = _data(data_identity)
    if (checkpoint.get("data_identity") != data or
            checkpoint.get("data_identity_sha256") != hashlib.sha256(_json(data)).hexdigest()):
        raise ValueError("HOI checkpoint data/split/timebase/contact/asset identity differs")
    models = _models(policy, wm, critic)
    if any(model.geometry_input_enabled != geometry_input_enabled(variant) for model in models.values()):
        raise ValueError("checkpoint variant requires matching geometry-input flags in all three models")
    architecture = deepcopy(checkpoint.get("architecture"))
    # Before explicit variant metadata existed only full HOI was implemented.
    # Such local diagnostic snapshots may be read as hoi_v1, never an ablation.
    if (variant == "hoi_v1" and "variant" not in checkpoint and
            "variant" not in checkpoint.get("extra", {}).get("config", {}) and isinstance(architecture, dict)):
        for record in architecture.values():
            if isinstance(record, dict):
                record.setdefault("geometry_input_enabled", True)
    if architecture != {key: _signature(model) for key, model in models.items()}:
        raise ValueError("HOI checkpoint model architecture differs")
    for key, model in models.items():
        _validate_state(model, checkpoint.get(key), key)
    if optimizers is not None:
        if set(optimizers) != set(MODEL_KEYS) or set(checkpoint.get("optimizers", {})) != set(MODEL_KEYS):
            raise ValueError("HOI continuation requires all three optimizer states")
        # Validate optimizer tensor dimensions before mutating any live model.
        for key, optimizer in optimizers.items():
            saved = checkpoint["optimizers"][key]
            current = optimizer.state_dict()
            if len(saved["param_groups"]) != len(current["param_groups"]):
                raise ValueError(f"{key} optimizer parameter groups differ")
            for group, actual_group in zip(saved["param_groups"], optimizer.param_groups):
                if len(group["params"]) != len(actual_group["params"]):
                    raise ValueError(f"{key} optimizer parameter count differs")
                for pid, parameter in zip(group["params"], actual_group["params"]):
                    for name, value in saved["state"].get(pid, {}).items():
                        if isinstance(value, torch.Tensor):
                            if not torch.isfinite(value).all():
                                raise ValueError(f"{key} optimizer {name} is nonfinite")
                            if value.ndim and value.shape != parameter.shape:
                                raise ValueError(f"{key} optimizer {name} shape differs")
    for key, model in models.items():
        model.load_state_dict(checkpoint[key], strict=True)
    if optimizers is not None:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
    return checkpoint


def _copy_legacy(module, source, *, first, legacy_columns, last=None, legacy_rows=None, norm_prefixes=()):
    """Copy every legacy key; only declared first/last matrices may expand."""
    target = module.state_dict()
    expected = {k for k in target if any(k.startswith(p + ".") for p in norm_prefixes) or k.startswith("net.")}
    if first.startswith("dec"):
        expected = {k for k in target if k.startswith(("enc.", "dec.", "obs_norm.", "goal_norm."))}
    if set(source) != expected:
        raise ValueError(f"legacy keys do not match declared original architecture: missing={sorted(expected-set(source))}, extra={sorted(set(source)-expected)}")
    for key, value in source.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"invalid legacy tensor {key}")
        dest = target[key]
        if key == first + ".weight":
            if tuple(value.shape) != (dest.shape[0], legacy_columns):
                raise ValueError("legacy first-layer shape differs")
            dest.zero_()
            dest[:, :legacy_columns].copy_(value)
        elif last is not None and key in (last + ".weight", last + ".bias"):
            if value.shape[0] != legacy_rows or value.shape[1:] != dest.shape[1:]:
                raise ValueError("legacy output-layer shape differs")
            dest[:legacy_rows].copy_(value)
        else:
            if value.shape != dest.shape:
                raise ValueError(f"legacy tensor shape differs: {key}")
            dest.copy_(value)
    module.load_state_dict(target, strict=True)


def _last_linear_key(state, prefix="net"):
    keys = [key[:-7] for key in state if re.fullmatch(re.escape(prefix) + r"\.\d+\.weight", key) and state[key].ndim == 2]
    if not keys:
        raise ValueError("legacy model has no linear output layer")
    return max(keys, key=lambda key: int(key.split(".")[-1]))


@torch.no_grad()
def migrate_legacy(checkpoint, policy, wm, critic, *, physics, legacy_physics=None, allow_physics_change=False):
    if policy.observation_profile != LEGACY:
        raise ValueError("v1.1 actor cannot warm-start legacy observation weights; use explicit from-scratch or same-profile resume")
    if checkpoint.get("schema_version") is not None:
        raise ValueError("migrate_legacy only accepts motion-only checkpoints")
    saved = checkpoint.get("physics") or checkpoint.get("args", {})
    if not all(type(saved.get(key)) is bool for key in ("hand_collision", "self_collision")):
        if legacy_physics is None:
            raise ValueError("old checkpoint has unknown physics; explicitly declare both legacy collision settings")
        saved = _physics(legacy_physics)
        source_evidence = "explicit_legacy_declaration"
    else:
        saved = {key: saved[key] for key in ("hand_collision", "self_collision")}
        source_evidence = "saved_checkpoint"
        if legacy_physics is not None and saved != _physics(legacy_physics):
            raise ValueError("declared legacy physics contradicts the saved checkpoint")
    target_physics = _physics(physics)
    changed = any(saved[key] != target_physics[key] for key in ("hand_collision", "self_collision"))
    if changed and not allow_physics_change:
        raise ValueError("legacy initialization changes physics; explicitly authorize a separate physics-change experiment")
    if not all(key in checkpoint for key in ("pi", "wm")):
        raise ValueError("legacy checkpoint must include pi and wm")
    # Validate on copies first; a malformed checkpoint cannot partially mutate a run.
    p_copy, w_copy, c_copy = deepcopy(policy), deepcopy(wm), deepcopy(critic)
    _copy_legacy(p_copy, checkpoint["pi"], first="dec.0", legacy_columns=1022)
    _copy_legacy(w_copy, checkpoint["wm"], first="net.0", legacy_columns=133,
                 last=_last_linear_key(checkpoint["wm"]), legacy_rows=70, norm_prefixes=("in_norm", "out_norm"))
    has_critic = "critic" in checkpoint
    if has_critic:
        _copy_legacy(c_copy, checkpoint["critic"], first="net.0", legacy_columns=914, norm_prefixes=("norm",))
    for dest, source in ((policy, p_copy), (wm, w_copy), (critic, c_copy)):
        dest.load_state_dict(source.state_dict(), strict=True)
    return {"mode": "legacy_initialization_new_optimizers_and_replay", "iteration": 0,
            "source_iteration": checkpoint.get("iteration"), "critic": "copied" if has_critic else "fresh_initialization",
            "physics_source": saved, "physics_evidence": source_evidence, "physics_target": target_physics,
            "physics_changed": changed, "legacy_prefixes_zero_extended": {"actor": 1022, "wm": 133, "critic": 914 if has_critic else None},
            "bitwise_resume": False}
