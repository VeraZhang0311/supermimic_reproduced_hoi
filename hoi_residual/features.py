"""v1.1 torso object inputs with the unchanged privileged GMT prefix.

History stores each pose in its own collection-time torso heading frame.
Seven future targets and errors use the CURRENT torso frame. FPS128 enters
the trainable policy encoder separately, never the raw RMS block.
"""
import torch
from supermimic.privileged import PRIVILEGED_DIM, tan_norm
from supermimic.rotations import yaw_mat

HISTORY = 10
FUTURE_STEPS = (1, 2, 5, 7, 12, 18, 25)
FUTURE_TIMES = tuple(step * .02 for step in FUTURE_STEPS)
BLOCKS = {"robot_privileged": PRIVILEGED_DIM, "object_history_torso_10x9": 90,
          "object_reference_torso_7x9": 63, "object_tracking_errors_7x15": 105,
          "object_oracle_velocity": 6, "object_physics": 12}
ORACLE_DIM = sum(BLOCKS.values())  # 3647; +29 base actions, then +64 learned shape.
SCHEMA = "hoi_oracle_hw3_torso_fps128_v2"


def observation_contract():
    return {"schema": SCHEMA, "raw_dim": ORACLE_DIM, "blocks": BLOCKS,
            "history": [10, 9], "object_anchor": "current_torso_origin_and_heading",
            "history_frame": "own_collection_time_torso_heading_per_tick",
            "future_steps": list(FUTURE_STEPS), "future_times_s": list(FUTURE_TIMES),
            "reference": "direct_position_and_rotation_in_current_actual_torso_frame",
            "tracking_errors": "per_future_slot_position3_rotation6_linear_velocity3_angular_velocity3",
            "rotation6": "X_axis_then_Z_axis", "error_rotation": "HtorsoT_Rref_RobjectT_Htorso",
            "oracle_velocity": "world_object_linear_and_angular_velocity_rotated_by_current_torso_heading",
            "startup": "repeat_first", "reset": "repeat_reset_observation_only_for_reset_worlds",
            "endpoint": "repeat_last", "validity_masks": False,
            "shape": "separate_FPS128_object_link_local_metres_to_trainable_64D_encoder",
            "asset_identity_input": False, "bbox_input": False}


def torso_pose(kin, state):
    """Frozen hw3 named link FK, including all three waist joints."""
    index = kin.body_names.index("torso_link")
    positions, rotations, _, _ = kin.forward(state)
    return positions[..., index, :], rotations[..., index, :, :]


def object_observation(torso, obj):
    position, rotation = torso
    H = yaw_mat(rotation).transpose(-1, -2)
    return torch.cat(((H @ (obj[0] - position).unsqueeze(-1)).squeeze(-1),
                      tan_norm(H @ obj[1])), -1)


class ObjectHistory:
    """Per-world pose history; value final states before reset erases history."""
    def __init__(self, current):
        self._validate(current)
        self.values = current[:, None].expand(-1, HISTORY, -1).clone()

    @staticmethod
    def _validate(current):
        if current.ndim != 2 or current.shape[-1] != 9 or not torch.isfinite(current).all():
            raise ValueError("Object history requires finite [world,9] observations")

    def advance(self, current):
        self._validate(current)
        if current.shape != self.values[:, -1].shape:
            raise ValueError("Object history world count changed")
        self.values = torch.cat((self.values[:, 1:], current[:, None]), dim=1)

    def reset(self, ids, current):
        self._validate(current)
        if current.shape != self.values[:, -1].shape:
            raise ValueError("Object history reset requires the full world axis")
        self.values[ids] = current[ids, None].expand(-1, HISTORY, -1)


def future_frames(motion, frames):
    steps = torch.tensor(FUTURE_STEPS, device=frames.device)
    ends = motion.clip_end[motion.clip_id[frames]]
    return torch.minimum(frames[..., None] + steps, ends[..., None])


def oracle_features(robot_privileged, torso, obj, motion, frames, object_context, object_history):
    batch = len(frames)
    if (robot_privileged.shape != (batch, PRIVILEGED_DIM) or object_history.shape != (batch, 10, 9)
            or object_context.shape != (batch, 12)):
        raise ValueError("HOI oracle/history/context shape contract violated")
    position, rotation = torso
    H = yaw_mat(rotation).transpose(-1, -2)
    Hfuture = H[:, None]
    p, R, v, w = obj
    pr, Rr, vr, wr = motion.object_ref(future_frames(motion, frames))
    rotate = lambda x: (Hfuture @ x.unsqueeze(-1)).squeeze(-1)
    goals = torch.cat((rotate(pr - position[:, None]), tan_norm(Hfuture @ Rr)), -1)
    errors = torch.cat((rotate(pr - p[:, None]),
                        tan_norm(Hfuture @ Rr @ R[:, None].transpose(-1, -2)
                                 @ Hfuture.transpose(-1, -2)),
                        rotate(vr - v[:, None]), rotate(wr - w[:, None])), -1)
    velocity = torch.cat(((H @ v.unsqueeze(-1)).squeeze(-1),
                          (H @ w.unsqueeze(-1)).squeeze(-1)), -1)
    result = torch.cat((robot_privileged, object_history.flatten(-2), goals.flatten(-2),
                        errors.flatten(-2), velocity, object_context), -1)
    if result.shape != (batch, ORACLE_DIM) or not torch.isfinite(result).all():
        raise ValueError("HOI oracle shape or finite-value contract violated")
    return result
