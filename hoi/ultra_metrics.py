"""Shared whole-clip diagnostics for native ULTRA and the IsaacLab port.

The object-pose gate is an engineering diagnostic, not ULTRA's published success
criterion or a contact/drop judge. All rotations use xyzw quaternions.
"""

import math

import torch


CRITERIA = {
    "version": "ultra-migration-panel-v1",
    "body_frame_mean_limit_m": 0.5,
    "object_position_limit_m": 0.1,
    "object_orientation_limit_deg": 30.0,
    "object_violation_duration_s": 0.3,
    "pelvis_low_diagnostic_m": 0.3,
}


def first_sustained_violation(mask, dt, duration):
    """Return the run's starting sample after at least duration of bad samples."""
    needed = math.ceil(duration / dt - 1e-9)
    count = 0
    for index, bad in enumerate(mask.tolist()):
        count = count + 1 if bad else 0
        if count >= needed:
            return index - needed + 1
    return None


def clip_metrics(packed, control_dt, exclude_initial=0):
    if control_dt <= 0:
        raise ValueError("control_dt must be positive")
    required = ("body_pos", "object_state", "reference", "q", "action")
    values = {
        key: packed[key][exclude_initial:].detach().cpu().float() for key in required
    }
    length = len(values["reference"])
    if length < 1 or any(len(value) != length for value in values.values()):
        raise ValueError("Missing or inconsistent scored frames")
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite scored state/action/reference")
    body, obj, ref = values["body_pos"], values["object_state"], values["reference"]
    body_distances = (body - ref[:, 84:201].reshape(-1, 39, 3)).norm(dim=-1)
    body_error = body_distances.mean(-1)
    object_error = (obj[:, :3] - ref[:, 71:74]).norm(dim=-1)
    a, b = obj[:, 3:7], ref[:, 74:78]
    if (a.norm(dim=-1) < 1e-6).any() or (b.norm(dim=-1) < 1e-6).any():
        raise ValueError("Invalid object quaternion")
    dot = ((a / a.norm(dim=-1, keepdim=True)) * (b / b.norm(dim=-1, keepdim=True))).sum(
        -1
    )
    angle = 2 * torch.acos(dot.abs().clamp(0, 1)) * 180 / math.pi
    object_bad = (object_error > CRITERIA["object_position_limit_m"]) | (
        angle > CRITERIA["object_orientation_limit_deg"]
    )
    object_failure = first_sustained_violation(
        object_bad, control_dt, CRITERIA["object_violation_duration_s"]
    )
    body_bad = torch.where(body_error > CRITERIA["body_frame_mean_limit_m"])[0]

    def stats(value):
        return {
            "mean": value.mean().item(),
            "p95": torch.quantile(value, 0.95).item(),
            "max": value.max().item(),
            "final": value[-1].item(),
        }

    if "time" in packed:
        reference_time = packed["time"][exclude_initial:].cpu().float()
    else:
        reference_time = packed["progress"][exclude_initial:].cpu().float() / 60.0
    phc_pass = len(body_bad) == 0
    object_pass = object_failure is None
    return {
        "criteria_version": CRITERIA["version"],
        "scored_frames": length,
        "scored_duration_s": length * control_dt,
        "reference_start_s": reference_time[0].item(),
        "reference_end_s": reference_time[-1].item(),
        "body_position_m": stats(body_error),
        "hand_position_m": stats(body_distances[:, [28, 38]].mean(-1)),
        "object_position_m": stats(object_error),
        "object_orientation_deg": stats(angle),
        "joint_rmse_rad": (values["q"] - ref[:, 13:42]).square().mean().sqrt().item(),
        "action_at_clip_fraction": (values["action"].abs() >= 1 - 1e-6)
        .float()
        .mean()
        .item(),
        "phc_pass": phc_pass,
        "object_pose_pass": object_pass,
        "body_and_object_pose_pass": phc_pass and object_pass,
        "first_phc_failure_reference_s": reference_time[body_bad[0]].item()
        if len(body_bad)
        else None,
        "first_object_failure_reference_s": reference_time[object_failure].item()
        if object_failure is not None
        else None,
        "minimum_pelvis_height_m": body[:, 0, 2].min().item(),
        "pelvis_below_0_3m": bool(
            (body[:, 0, 2] < CRITERIA["pelvis_low_diagnostic_m"]).any()
        ),
    }
