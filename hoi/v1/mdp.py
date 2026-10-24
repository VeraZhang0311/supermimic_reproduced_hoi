"""Additive HOI heads; the original human MDP is called without changing it."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.mdp import MDP, RewardCfg
from supermimic.motion import CONTROL_DT
from supermimic.rotations import mat_to_6d, rot_angle_sq, yaw_mat

MAX_STEP_REWARD = .15
HUMAN_MAX_STEP_REWARD = .1
OBJECT_OBS_DIM = 18
OBJECT_GOAL_DIM = 190


@dataclass
class HOIRewardCfg:
    position_std: float = .10
    orientation_std: float = .30
    linear_velocity_std: float = .50
    angular_velocity_std: float = 1.0
    approach_std: float = .05
    approach_temperature: float = .02
    w_position: float = 1.0
    w_orientation: float = .5
    w_linear_velocity: float = .25
    w_angular_velocity: float = .25
    w_approach: float = .25
    w_contact: float = .25
    object_term_min_m: float = 1.0
    object_term_bbox_multiplier: float = 2.0
    object_soft_temperature: float = .10


def _masked_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def object_invariant_features(S, O):
    from .state import object_features
    return object_features(S, O)


class HOIMDP:
    max_step_reward = MAX_STEP_REWARD

    def __init__(self, kin, motion, default_q, shape_bank, cfg=None, human_cfg=None):
        self.kin, self.motion, self.shapes = kin, motion, shape_bank
        self.cfg = cfg if cfg is not None else HOIRewardCfg()
        weight_sum = sum(getattr(self.cfg, name) for name in ("w_position", "w_orientation", "w_linear_velocity",
                                                              "w_angular_velocity", "w_approach", "w_contact"))
        # Anchor the unchanged main configuration at the exact historical v1
        # constant, while deriving ablation sampler ceilings from active heads.
        self.max_step_reward = MAX_STEP_REWARD + CONTROL_DT * (weight_sum - 2.5)
        self.human = MDP(kin, motion, default_q, human_cfg if human_cfg is not None else RewardCfg())
        self.default_q = default_q

    def obs(self, S, a_prev, xi, f):
        return self.human.obs(S, a_prev, xi, f)

    def add_noise(self, observation):
        return self.human.add_noise(observation)

    def object_obs(self, S, O, f):
        ht = yaw_mat(S[1]).transpose(-1, -2)
        ref = self.motion.ref_object(f)
        return torch.cat(((ht @ (O[0] - S[0]).unsqueeze(-1)).squeeze(-1), mat_to_6d(ht @ O[1]),
                          (ht @ (ref[0] - O[0]).unsqueeze(-1)).squeeze(-1),
                          mat_to_6d(O[1].transpose(-1, -2) @ ref[1])), -1)

    def object_errors(self, S, O, f):
        ht = yaw_mat(S[1]).transpose(-1, -2)
        ref = self.motion.ref_object(f)
        return torch.cat(((ht @ (ref[0] - O[0]).unsqueeze(-1)).squeeze(-1),
                          mat_to_6d(O[1].transpose(-1, -2) @ ref[1]),
                          (ht @ (ref[2] - O[2]).unsqueeze(-1)).squeeze(-1),
                          (ht @ (ref[3] - O[3]).unsqueeze(-1)).squeeze(-1)), -1)

    def legacy_critic_features(self, S, f, xi):
        """184 raw old features; the caller appends the original 730D goal."""
        return self.human.critic_features(S, f, xi)

    def critic_features(self, S, O, f, xi):
        """Non-shape raw blocks; xiO is supplied separately by the learner.

        Prefix184 + human goal730 + object goal190 + invariant15 + error15 +
        expected/mask4 = 1138. Append physical context12 then shape64 = 1214.
        """
        expected, valid = self.motion.expected(f)
        return torch.cat((self.legacy_critic_features(S, f, xi), self.motion.goal(f, 10),
                          self.motion.object_goal(S, O, f, 10), object_invariant_features(S, O),
                          self.object_errors(S, O, f), expected, valid), -1)

    def reward(self, S, O, f, a, a_prev, actual_contact, contact_valid=None, terms=False):
        c = self.cfg
        human_reward, human_terms = self.human.reward(S, f, a, a_prev, terms=True)
        reference = self.motion.ref_object(f)
        ep = (O[0] - reference[0]).square().sum(-1)
        er = rot_angle_sq(O[1], reference[1])
        ev = (O[2] - reference[2]).square().sum(-1)
        ew = (O[3] - reference[3]).square().sum(-1)
        position = torch.exp(-ep / c.position_std ** 2)
        orientation = torch.exp(-er / c.orientation_std ** 2)
        linear_velocity = torch.exp(-ev / c.linear_velocity_std ** 2)
        angular_velocity = torch.exp(-ew / c.angular_velocity_std ** 2)
        expected, valid = self.motion.expected(f)
        # Pair probabilities are supplied by the contact head in imagination,
        # and interval labels by the recorder for real transitions.
        actual = actual_contact[..., :2]
        actual_mask = torch.ones_like(actual) if contact_valid is None else contact_valid[..., :2].to(actual)
        contact = _masked_mean(expected * actual + (1 - expected) * (1 - actual), valid * actual_mask)
        distance = self.shapes.hand_distances(self.kin, S, O, self.motion.asset_ids[f])
        weight = torch.softmax(-distance.square() / c.approach_temperature ** 2, -1)
        distance2 = (weight * distance.square()).sum(-1)
        hand_approach = torch.exp(-distance2 / c.approach_std ** 2)
        approach = _masked_mean(hand_approach, expected * valid)
        extra = (c.w_position * position + c.w_orientation * orientation + c.w_linear_velocity * linear_velocity +
                 c.w_angular_velocity * angular_velocity + c.w_approach * approach + c.w_contact * contact) * CONTROL_DT
        result = human_reward + extra
        if not terms:
            return result
        return result, {**human_terms, "human_reward": human_reward, "hoi_reward": result,
                        "object_position": position * CONTROL_DT, "object_orientation": orientation * CONTROL_DT,
                        "object_linear_velocity": linear_velocity * CONTROL_DT,
                        "object_angular_velocity": angular_velocity * CONTROL_DT,
                        "hand_approach": approach * CONTROL_DT, "contact_match": contact * CONTROL_DT,
                        "e_object_position": ep, "e_object_orientation": er,
                        "e_object_linear_velocity": ev, "e_object_angular_velocity": ew,
                        "expected_positive": (expected * valid).sum(-1), "contact_valid_pairs": (valid * actual_mask).sum(-1),
                        "missing_contact": (expected * valid * actual_mask * (1 - actual)).sum(-1),
                        "extra_contact": ((1 - expected) * valid * actual_mask * actual).sum(-1),
                        "palm_penetration_m": (-distance).clamp_min(0).amax(dim=(-1, -2))}

    def object_failure_threshold(self, f):
        return (self.shapes.bbox_diagonal(self.motion.asset_ids[f]) * self.cfg.object_term_bbox_multiplier).clamp_min(self.cfg.object_term_min_m)

    def termination(self, S, O, f):
        distance = torch.linalg.vector_norm(O[0] - self.motion.object_p[f], dim=-1)
        return self.human.termination(S, f) | (distance > self.object_failure_threshold(f))

    def soft_termination(self, S, O, f, tz=.02, tt=.04):
        robot_fail = self.human.soft_termination(S, f, tz, tt)
        distance = torch.linalg.vector_norm(O[0] - self.motion.object_p[f], dim=-1)
        object_fail = torch.sigmoid((distance - self.object_failure_threshold(f)) / self.cfg.object_soft_temperature)
        return 1 - (1 - robot_fail) * (1 - object_fail)
