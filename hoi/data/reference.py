"""Pure-Torch ULTRA [N,630] decoding for HOI v1, independent of simulators.

This adapter does not invoke the legacy MotionLib resampler (which consumes
CSV and imports native kinematics). Its velocity formulas deliberately match
MotionLib: central pose differences with one-sided boundaries, world SO(3)
angular differences. FK remains a later native integration step.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from supermimic.rotations import quat_slerp, quat_to_mat, so3_log

from .timebase import CanonicalTimebase

SCHEMA_VERSION = "hoi_v1_reference_v1"
QUATERNION_NORM_TOLERANCE = 1e-3
_QUATERNION_SLICES = ((3, 7), (74, 78), (201, 357))


def validate_motion(frames: torch.Tensor) -> None:
    """Reject checkpoints/rollouts and malformed source tensors explicitly.

    Validation does not normalize or repair the input. Float16/bfloat16 source
    values are accepted, then promoted for interpolation/rotation arithmetic.
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError("Expected a plain floating Tensor[N,630], not a checkpoint or rollout container")
    if frames.layout != torch.strided or frames.is_quantized or not frames.is_floating_point():
        raise TypeError("Expected a dense, real floating Tensor[N,630]")
    if frames.ndim != 2 or frames.shape[1] != 630:
        raise ValueError(f"Expected Tensor[N,630], got shape {tuple(frames.shape)}")
    if frames.shape[0] < 1:
        raise ValueError("Source motion must contain at least one state frame")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("Source motion contains non-finite values")
    for start, stop in _QUATERNION_SLICES:
        quats = frames[:, start:stop].reshape(-1, 4).to(torch.float64)
        deviation = (torch.linalg.vector_norm(quats, dim=-1) - 1).abs()
        if bool((deviation > QUATERNION_NORM_TOLERANCE).any()):
            raise ValueError(
                f"Non-unit or zero xyzw quaternion in columns {start}:{stop}; "
                f"maximum norm deviation {float(deviation.max()):.6g} "
                f"exceeds {QUATERNION_NORM_TOLERANCE}"
            )
    contacts = frames[:, 591:630]
    if not bool(((contacts == 0) | (contacts == 1)).all()):
        raise ValueError("Source contact flags in columns 591:630 must be binary 0/1")


def _joint_order(source_joint_names: Sequence[str], target_joint_names: Sequence[str]) -> list[int]:
    for label, names in (("source_joint_names", source_joint_names), ("target_joint_names", target_joint_names)):
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise TypeError(f"{label} must be a sequence of 29 unique joint names")
        if len(names) != 29 or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{label} must contain exactly 29 non-empty joint names")
        if len(set(names)) != 29:
            raise ValueError(f"{label} contains duplicate joint names")
    if set(source_joint_names) != set(target_joint_names):
        missing = sorted(set(target_joint_names) - set(source_joint_names))
        extra = sorted(set(source_joint_names) - set(target_joint_names))
        raise ValueError(f"Joint-name sets differ: missing source joints {missing}; extra source joints {extra}")
    lookup = {name: index for index, name in enumerate(source_joint_names)}
    return [lookup[name] for name in target_joint_names]


def _xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., 3:], quaternion[..., :3]), dim=-1)


def _wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)


def _gradient(value: torch.Tensor, dt: float) -> torch.Tensor:
    result = torch.zeros_like(value)
    result[1:-1] = (value[2:] - value[:-2]) / (2 * dt)
    result[0] = (value[1] - value[0]) / dt
    result[-1] = (value[-1] - value[-2]) / dt
    return result


def _world_angular_velocity(rotation: torch.Tensor, dt: float) -> torch.Tensor:
    result = rotation.new_zeros((len(rotation), 3))
    result[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * dt)
    result[0] = so3_log(rotation[1] @ rotation[0].T) / dt
    result[-1] = so3_log(rotation[-1] @ rotation[-2].T) / dt
    return result


def adapt_motion(
    frames: torch.Tensor,
    source_joint_names: Sequence[str],
    target_joint_names: Sequence[str],
    timebase: CanonicalTimebase | None = None,
) -> dict:
    """Decode and resample one motion without modifying its source tensor.

    ``source_sampled`` preserves original column/joint order and source velocity
    fields. ``robot`` joint fields are reordered by name into target order.
    Source and pose-derived velocities have distinct keys; no FPS multiplier,
    COM conversion, FK, heading alignment or simulator stepping is performed.
    All returned state fields use the same T + 1 grid; metadata records T.
    """
    validate_motion(frames)
    order = _joint_order(source_joint_names, target_joint_names)
    timebase = timebase if timebase is not None else CanonicalTimebase()
    if type(timebase) is not CanonicalTimebase:
        raise TypeError("HOI v1 requires the versioned CanonicalTimebase")
    source_frames = len(frames)
    control_steps = timebase.control_steps(source_frames)
    if control_steps < 1:
        raise ValueError(
            f"Motion with {source_frames} source state frames produces fewer than two "
            "control-grid states; pose-derived velocities cannot be constructed"
        )
    grid = timebase.sample_grid(source_frames, device=frames.device)
    values = frames if frames.dtype == torch.float64 else frames.to(torch.float32)
    first, second = values[grid["i0"]], values[grid["i1"]]
    alpha = grid["alpha"].to(values.dtype)
    sampled = first * (1 - alpha[:, None]) + second * alpha[:, None]
    for start, stop in _QUATERNION_SLICES:
        qa = _xyzw_to_wxyz(first[:, start:stop].reshape(len(first), -1, 4))
        qb = _xyzw_to_wxyz(second[:, start:stop].reshape(len(first), -1, 4))
        # Reject invalid source norms above; normalize accepted roundoff before
        # SLERP so dot/angle calculations retain their unit-quaternion meaning.
        qa = torch.nn.functional.normalize(qa, dim=-1)
        qb = torch.nn.functional.normalize(qb, dim=-1)
        sampled[:, start:stop] = _wxyz_to_xyzw(quat_slerp(qa, qb, alpha[:, None])).flatten(1)
    sampled[:, 591:630] = values[grid["nearest"], 591:630]

    root_pos, root_quat = sampled[:, :3], sampled[:, 3:7]
    object_pos, object_quat = sampled[:, 71:74], sampled[:, 74:78]
    joint_order = torch.tensor(order, device=frames.device, dtype=torch.int64)
    joint_pos = sampled[:, 13:42].index_select(1, joint_order)
    root_rot = quat_to_mat(_xyzw_to_wxyz(root_quat))
    object_rot = quat_to_mat(_xyzw_to_wxyz(object_quat))
    source_duration = (source_frames - 1) / timebase.source_hz
    control_duration = control_steps / timebase.control_hz
    # The integer remainder cannot become a tiny negative tail at an endpoint.
    tail_remainder = (5 * (source_frames - 1) - 6 * control_steps) / 300
    metadata = {
        **timebase.metadata(),
        "schema_version": SCHEMA_VERSION,
        "source_schema": "ultra_tensor_N_630",
        "source_frames": source_frames,
        "control_steps": control_steps,
        "state_frames": len(sampled),
        "canonical_duration_s": source_duration,
        "last_control_time_s": control_duration,
        "tail_remainder_s": tail_remainder,
        "source_joint_names": list(source_joint_names),
        "target_joint_names": list(target_joint_names),
        "joint_order": order,
        "joint_order_direction": "target_index_to_source_index",
        "source_dtype": str(frames.dtype).removeprefix("torch."),
        "reference_dtype": str(sampled.dtype).removeprefix("torch."),
        "quaternion_order": "xyzw",
        "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
        "world_frame": "source_world_z_up_no_recenter",
        "robot_reference_velocity_reference_point": "root_link_origin",
        "object_reference_velocity_reference_point": "recorded_object_pose_origin",
        "object_pose_origin_native_alignment_status": "unverified",
        "source_body_order": "original_39_body_order_unmodified",
        "source_contact_semantics": "source_body_flags_not_hand_object_pair_truth",
        "body_reference_mode": "source_fields_only_not_robot_fk",
        "robot_fk_status": "not_computed_by_data_adapter",
    }
    return {
        **grid,
        "source_sampled": sampled,
        "robot": {
            "root_pos": root_pos,
            "root_quat_xyzw": root_quat,
            "root_rot": root_rot,
            "joint_pos": joint_pos,
            "source_v": sampled[:, 7:10],
            "source_omega": sampled[:, 10:13],
            "source_joint_qd": sampled[:, 42:71].index_select(1, joint_order),
            "derived_ref_v": _gradient(root_pos, timebase.control_dt_s),
            "derived_ref_omega": _world_angular_velocity(root_rot, timebase.control_dt_s),
            "derived_ref_joint_qd": _gradient(joint_pos, timebase.control_dt_s),
        },
        "object": {
            "pos": object_pos,
            "quat_xyzw": object_quat,
            "rot": object_rot,
            "source_v": sampled[:, 78:81],
            "source_omega": sampled[:, 81:84],
            "derived_ref_v": _gradient(object_pos, timebase.control_dt_s),
            "derived_ref_omega": _world_angular_velocity(object_rot, timebase.control_dt_s),
        },
        "body": {
            "pos": sampled[:, 84:201].reshape(len(sampled), 39, 3),
            "quat_xyzw": sampled[:, 201:357].reshape(len(sampled), 39, 4),
            "source_v": sampled[:, 357:474].reshape(len(sampled), 39, 3),
            "source_omega": sampled[:, 474:591].reshape(len(sampled), 39, 3),
            "contact_flags": sampled[:, 591:630],
        },
        "metadata": metadata,
    }
