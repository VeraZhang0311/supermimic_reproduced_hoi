"""HOI state blocks and frame conventions; all operations keep input gradients.

Internal quaternions use wxyz, exactly as the existing robot replay.  Object
linear velocity is measured at the object pose origin, not its COM.
"""
from __future__ import annotations

import torch

from supermimic.mdp import flatten_state, unflatten_state
from supermimic.rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat
from .contacts import CHANNELS

ROBOT_STATE_DIM = 71
OBJECT_STATE_DIM = 13
OBJECT_FEATURE_DIM = 15
OBJECT_INCREMENT_DIM = 12
OBJECT_CONTEXT_DIM = 12
OBJECT_OBS_DIM = 18
OBJECT_GOAL_DIM = 190
CONTACT_DIM = len(CHANNELS)


def flatten_object(O):
    p, R, v, w = O
    return torch.cat((p, mat_to_quat(R), v, w), -1)


def unflatten_object(x):
    if x.shape[-1] != OBJECT_STATE_DIM:
        raise ValueError(f"object replay must have 13 fields, got {x.shape[-1]}")
    return x[..., :3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]


def _rotate(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def object_features(S, O):
    """Relative pose, relative linear velocity, absolute omega in robot heading."""
    p, R, v, *_ = S
    op, oR, ov, ow = O
    Ht = yaw_mat(R).transpose(-1, -2)
    return torch.cat((_rotate(Ht, op - p), mat_to_6d(Ht @ oR),
                      _rotate(Ht, ov - v), _rotate(Ht, ow)), -1)


def object_targets(S, O, O1):
    """Object increments: heading-frame dp/dv/dw and object-local rotation."""
    p, R, v, w = O
    p1, R1, v1, w1 = O1
    Ht = yaw_mat(S[1]).transpose(-1, -2)
    return torch.cat((_rotate(Ht, p1 - p), so3_log(R.transpose(-1, -2) @ R1),
                      _rotate(Ht, v1 - v), _rotate(Ht, w1 - w)), -1)


def integrate_object(S, O, increment):
    if increment.shape[-1] != OBJECT_INCREMENT_DIM:
        raise ValueError("object increments must have 12 fields")
    p, R, v, w = O
    dp, dr, dv, dw = increment.split(3, -1)
    H = yaw_mat(S[1])
    return p + _rotate(H, dp), R @ so3_exp(dr), v + _rotate(H, dv), w + _rotate(H, dw)


def object_state_error(S, O_hat, O):
    """Prediction error using the ground-truth robot heading at the scored tick."""
    return object_targets(S, O, O_hat)


def object_tracking_errors(S, O, reference):
    """15D critic errors: position3, relative orientation6, v3, omega3."""
    p, R, v, w = O
    rp, rR, rv, rw = reference
    Ht = yaw_mat(S[1]).transpose(-1, -2)
    return torch.cat((_rotate(Ht, rp - p), mat_to_6d(R.transpose(-1, -2) @ rR),
                      _rotate(Ht, rv - v), _rotate(Ht, rw - w)), -1)


# Explicit aliases make block ownership visible at API call sites.
flatten_robot = flatten_state
unflatten_robot = unflatten_state
