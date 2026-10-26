"""Actor-only observation contracts. Legacy WM, critic and rewards stay intact.

The v1.1 profile matches V5's visible fields with explicit user exceptions:
ten object-history frames, repeated startup/end frames, no validity masks,
the existing 50 Hz reference/FK pipeline and 0.1 reference joint-speed scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.rotations import yaw_mat

LEGACY = "v1"
MATCHING = "v1_1_observation_matching"
PROFILES = (LEGACY, MATCHING)
FUTURE_STEPS = (1, 2, 5, 7, 12, 18, 25)
FUTURE_TIMES = tuple(step * .02 for step in FUTURE_STEPS)


@dataclass(frozen=True)
class ObservationSpec:
    robot_dim: int
    human_goal_dim: int
    object_dim: int
    object_goal_dim: int
    history: int = 10


def observation_spec(profile):
    if profile == LEGACY:
        return ObservationSpec(99, 730, 18, 190)
    if profile == MATCHING:
        return ObservationSpec(93, 448, 9, 63)
    raise ValueError(f"unknown actor observation profile: {profile}")


def rotation_xz(matrix):
    """V5 tan/normal convention: concatenated X and Z matrix columns."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 2]), -1)


class ActorObservations:
    def __init__(self, mdp, profile=LEGACY):
        self.mdp, self.motion, self.kin, self.profile = mdp, mdp.motion, mdp.kin, profile
        self.spec = observation_spec(profile)
        if profile == MATCHING:
            if not hasattr(self.kin, "body_pose"):
                raise ValueError("v1.1 requires differentiable named-body FK")
            names = [self.kin.body_names[i] for i in self.kin.tracked_idx]
            self.torso_index = names.index("torso_link")
            # Same existing sensor-noise magnitudes, reordered to V5's 93D layout.
            self.noise = torch.cat((torch.full((3,), .2), torch.full((29,), .01),
                                    torch.full((29,), .5), torch.zeros(29),
                                    torch.full((3,), .05))).to(mdp.default_q)

    def torso_pose(self, state):
        return self.kin.body_pose("torso_link", state[0], state[1], state[4])

    def robot(self, state, previous, xi, frame):
        if self.profile == LEGACY:
            return self.mdp.obs(state, previous, xi, frame)
        _, rotation, _, omega, joints, speeds = state
        gyro = (rotation.transpose(-1, -2) @ omega.unsqueeze(-1)).squeeze(-1)
        # Preserve the existing calibrated sensor/PD-zero convention.
        relative_joints = joints - self.mdp.default_q - xi[..., 1:30]
        gravity = -rotation[..., 2, :]
        return torch.cat((gyro, relative_joints, speeds, previous, gravity), -1)

    def add_noise(self, observation):
        if self.profile == LEGACY:
            return self.mdp.add_noise(observation)
        return observation + (2 * torch.rand_like(observation) - 1) * self.noise

    def object(self, state, obj, frame):
        if self.profile == LEGACY:
            return self.mdp.object_obs(state, obj, frame)
        position, rotation = self.torso_pose(state)
        inverse_heading = yaw_mat(rotation).transpose(-1, -2)
        delta = (inverse_heading @ (obj[0] - position).unsqueeze(-1)).squeeze(-1)
        return torch.cat((delta, rotation_xz(inverse_heading @ obj[1])), -1)

    def future_frames(self, frame):
        steps = torch.tensor(FUTURE_STEPS, device=frame.device)
        end = self.motion.clip_end[self.motion.clip_id[frame]]
        return torch.minimum(frame[..., None] + steps, end[..., None])

    def human_goal(self, state, frame):
        if self.profile == LEGACY:
            return self.motion.goal(frame, 10)
        future = self.future_frames(frame)
        _, current_rotation = self.torso_pose(state)
        reference_rotation = self.motion.body_R[future, self.torso_index]
        relative = current_rotation.transpose(-1, -2).unsqueeze(-3) @ reference_rotation
        # Source and scaling intentionally remain the existing SuperMimic ones.
        return torch.cat((rotation_xz(relative), self.motion.qd[future] * .1,
                          self.motion.q[future]), -1).flatten(-2)

    def object_goal(self, state, obj, frame):
        if self.profile == LEGACY:
            return self.motion.object_goal(state, obj, frame, 10)
        future = self.future_frames(frame)
        position, rotation = self.torso_pose(state)
        inverse_heading = yaw_mat(rotation).transpose(-1, -2).unsqueeze(-3)
        delta = (inverse_heading @ (self.motion.object_p[future] - position.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        return torch.cat((delta, rotation_xz(inverse_heading @ self.motion.object_R[future])), -1).flatten(-2)


def observation_contract(profile):
    spec = observation_spec(profile)
    result = {"profile": profile, "robot_history": [10, spec.robot_dim],
              "human_goal_dim": spec.human_goal_dim, "object_history": [10, spec.object_dim],
              "object_goal_dim": spec.object_goal_dim}
    if profile == MATCHING:
        result.update(future_steps=list(FUTURE_STEPS), future_times_s=list(FUTURE_TIMES),
                      object_anchor="current_torso_heading", human_rotation="current_full_torso_relative",
                      proprio_frame="pelvis", rotation6="X_axis_then_Z_axis",
                      startup="repeat_first", endpoint="repeat_last", validity_masks=False,
                      reference_velocity="existing_50hz_pose_difference_times_0.1",
                      shape="unchanged_FPS128_object_local_metres",
                      normalization="clipped_RMS_robot_human;independent_unclipped_RMS_object_history9_goal63",
                      object_normalization={"type": "RunningMeanStd", "history_dim": 9, "future_dim": 63,
                          "history_statistics": "shared_across_10_ticks_update_current_real_tick_once",
                          "future_statistics": "per_feature_per_future_slot", "epsilon": .01,
                          "denominator": "sqrt(variance+epsilon_squared)", "max_count": 2e8, "clip": None,
                          "statistics_update": "real_collection_only", "frozen_in": ["imagination", "evaluation"]},
                      joint_zero_convention="existing_q_minus_default_minus_PD_offset")
    return result
