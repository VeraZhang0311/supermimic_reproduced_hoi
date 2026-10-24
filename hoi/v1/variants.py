"""Explicit independent experiments from the v1 design; the default is unchanged."""
from __future__ import annotations

VARIANTS = ("hoi_v1", "no_contact_objective", "no_geometry_input")


def validate_variant(variant):
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return variant


def geometry_input_enabled(variant):
    return validate_variant(variant) != "no_geometry_input"


def reward_config(variant):
    from .mdp import HOIRewardCfg
    validate_variant(variant)
    return HOIRewardCfg(w_contact=0.) if variant == "no_contact_objective" else HOIRewardCfg()


def max_step_reward(variant):
    """Variant objective bound; .15 remains the common full-v1 comparison scale."""
    from supermimic.motion import CONTROL_DT
    from .mdp import MAX_STEP_REWARD
    cfg = reward_config(variant)
    weight_sum = sum(getattr(cfg, name) for name in
        ("w_position", "w_orientation", "w_linear_velocity", "w_angular_velocity", "w_approach", "w_contact"))
    return MAX_STEP_REWARD + CONTROL_DT * (weight_sum - 2.5)
