"""Privileged dynamics context for WM/value only; never an actor observation."""
from __future__ import annotations

import torch

TORSO_MASS_CONTEXT_SCHEMA = "supermimic_torso_mass_context_1"
MASS_FIELDS = {"torso_mass": (1, torch.float32), "torso_mass_nominal": (1, torch.float32)}


def uses_torso_mass(profile):
    from .randomization import DR_PROFILES, MATCHING_DR
    if profile not in DR_PROFILES:
        raise ValueError(f"unknown domain randomization profile: {profile}")
    return profile == MATCHING_DR


def torso_mass_contract():
    return {"schema": TORSO_MASS_CONTEXT_SCHEMA, "raw_fields_kg": list(MASS_FIELDS),
            "source": "effective_simulator_torso_link_mass_readback_and_pre_DR_nominal",
            "encoding": "(torso_mass-torso_mass_nominal)/3kg", "dimension": 1,
            "normalization": "fixed_scale_no_running_statistics_no_clipping",
            "consumers": ["wm", "critic"], "actor_input": False,
            "imagination": "constant_from_real_start", "nominal_evaluation_context": 0.}


def normalize_torso_mass(actual, nominal):
    if (not isinstance(actual, torch.Tensor) or not isinstance(nominal, torch.Tensor) or
            actual.shape != nominal.shape or actual.ndim < 1 or actual.shape[-1] != 1):
        raise ValueError("torso mass and nominal mass must have matching (...,1) shapes")
    for value in (actual, nominal):
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("torso mass and nominal mass must be finite positive floating-point kg")
    return (actual - nominal) / 3.


def mass_context_from(values):
    for name in MASS_FIELDS:
        if name not in values:
            raise ValueError(f"torso mass context is missing {name}")
    return normalize_torso_mass(*(values[name] for name in MASS_FIELDS))


def append_mass_context(features, context, enabled):
    if not enabled:
        if context is not None:
            raise ValueError("torso mass context supplied to a legacy model")
        return features
    if (not isinstance(context, torch.Tensor) or context.shape != features.shape[:-1] + (1,) or
            not context.is_floating_point() or not bool(torch.isfinite(context).all())):
        raise ValueError("torso mass context must be finite with shape batch+(1,)")
    return torch.cat((features, context), -1)
