"""Analytic GMT heads on the prepared physical G1, without simulation/training."""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from supermimic.g1_assets import asset_cwd
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, MAX_STEP_REWARD, OBS_DIM
from supermimic.robot_metadata import MJCF_PATH
from supermimic.rotations import so3_exp


class Reference:
    def __init__(self, kin, state):
        self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd = state
        self.bodies = kin.forward(*state)
        self.body_p = self.bodies[0]

    def reference_bodies(self, frames):
        return tuple(x[frames] for x in self.bodies)


@unittest.skipUnless(Path(MJCF_PATH).is_file(), "prepare real hw3 assets first")
class GMTHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(3)
        asset = Path(MJCF_PATH)
        # MuJoCo's Windows native loader requires an ASCII relative filename.
        with asset_cwd(asset.parent), patch("supermimic.kinematics.MJCF_PATH", asset.name):
            cls.kin = G1Kinematics("cpu")
        cls.default_q = ((cls.kin.q_lo + cls.kin.q_hi) / 2).unsqueeze(0)

    def reference_state(self):
        return (torch.tensor([[.2, -.3, .9]]), so3_exp(torch.tensor([[0., 1., 0.]])),
                torch.tensor([[.2, -.1, .03]]), torch.tensor([[.1, -.2, .3]]),
                self.default_q.clone(), torch.linspace(-.2, .2, 29).unsqueeze(0))

    def make_mdp(self, state):
        return MDP(self.kin, Reference(self.kin, state), self.default_q[0])

    def test_tilted_pose_reward_and_critic_are_common_yaw_and_xy_invariant(self):
        reference = self.reference_state()
        state = list(reference)
        state[0] = state[0] + torch.tensor([[.07, -.04, .02]])
        state[1] = so3_exp(torch.tensor([[.7, 0., 0.]])) @ state[1]
        state[2] = state[2] + .03
        state[3] = state[3] - .05
        state[4] = state[4] + torch.linspace(-.08, .09, 29)
        mdp = self.make_mdp(reference)
        f, xi, a = torch.tensor([0]), torch.zeros(1, 33), torch.zeros(1, 29)
        reward, terms = mdp.reward(state, f, a, a, terms=True)
        critic = mdp.critic_features(state, f, xi)
        G = so3_exp(torch.tensor([[0., 0., .9]]))
        shift = torch.tensor([[1.3, -2.1, 0.]])
        def transform(S):
            p, R, v, w, q, qd = S
            rotate = lambda x: (G @ x.unsqueeze(-1)).squeeze(-1)
            return rotate(p)+shift, G@R, rotate(v), rotate(w), q, qd
        transformed = self.make_mdp(transform(reference))
        reward1, terms1 = transformed.reward(transform(state), f, a, a, terms=True)
        torch.testing.assert_close(reward, reward1, atol=2e-7, rtol=2e-5)
        for key in ("e_body_pos", "e_body_ori", "e_root_pos"):
            torch.testing.assert_close(terms[key], terms1[key], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(critic, transformed.critic_features(transform(state), f, xi),
                                   atol=2e-6, rtol=2e-5)

    def test_exact_reference_reaches_max_tracking_reward_without_penalties(self):
        state = self.reference_state()
        mdp = self.make_mdp(state)
        a = torch.zeros(1, 29)
        reward, terms = mdp.reward(state, torch.tensor([0]), a, a, terms=True)
        # acos is clamped just inside [-1,1], producing a tiny positive angle
        # even for identical orientations. Preserve that established convention.
        torch.testing.assert_close(terms["track"], torch.tensor([MAX_STEP_REWARD]), atol=2e-6, rtol=0)
        torch.testing.assert_close(reward, terms["track"])
        self.assertEqual(float(terms["action_rate"]), 0.)
        self.assertEqual(float(terms["joint_limit"]), 0.)

    def test_height_hard_and_soft_termination_share_the_same_boundary(self):
        state = list(self.reference_state())
        mdp = self.make_mdp(state)
        state[0] = torch.zeros(1, 3, dtype=torch.float64)
        state[1] = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        mdp.motion.root_p, mdp.motion.root_R = state[:2]
        actual = list(state)
        actual[0] = torch.zeros(3, 3, dtype=torch.float64)
        actual[0][:, 2] = torch.tensor([.249, .25, .251], dtype=torch.float64)
        actual[1] = state[1].expand(3, -1, -1)
        f = torch.zeros(3, dtype=torch.long)
        self.assertEqual(mdp.termination(actual, f).tolist(), [False, False, True])
        soft = mdp.soft_termination(actual, f)
        self.assertTrue(float(soft[0]) < .5 < float(soft[2]))
        self.assertAlmostEqual(float(soft[1]), .5, delta=3e-6)

    def test_tilt_hard_and_soft_termination_share_the_same_boundary(self):
        state = list(self.reference_state())
        mdp = self.make_mdp(state)
        state[0] = torch.zeros(1, 3, dtype=torch.float64)
        state[1] = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        mdp.motion.root_p, mdp.motion.root_R = state[:2]
        actual = list(state)
        actual[0] = state[0].expand(3, -1)
        e = torch.tensor([.799, .8, .801], dtype=torch.float64)
        c, s = 1-e, torch.sqrt(1-(1-e).square())
        actual[1] = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)
        actual[1][:, 1, 1], actual[1][:, 2, 2] = c, c
        actual[1][:, 1, 2], actual[1][:, 2, 1] = -s, s
        f = torch.zeros(3, dtype=torch.long)
        self.assertEqual(mdp.termination(actual, f).tolist(), [False, False, True])
        soft = mdp.soft_termination(actual, f)
        self.assertTrue(float(soft[0]) < .5 < float(soft[2]))
        self.assertAlmostEqual(float(soft[1]), .5, delta=3e-6)

    def test_joint_offset_changes_proprioception_while_fk_and_reward_use_absolute_q(self):
        state = self.reference_state()
        mdp = self.make_mdp(state)
        f, a, xi = torch.tensor([0]), torch.zeros(1, 29), torch.zeros(1, 33)
        base = mdp.obs(state, a, xi, f)
        xi[:, 1:30] = torch.linspace(-.04, .05, 29)
        offset = mdp.obs(state, a, xi, f)
        self.assertEqual(offset.shape, (1, OBS_DIM))
        torch.testing.assert_close(offset[:, :29], base[:, :29]-xi[:, 1:30])
        torch.testing.assert_close(offset[:, 29:], base[:, 29:])
        # The reference was built from absolute physical joint angles. Feeding
        # encoder-offset angles into FK would spoil this exact physical match.
        physical_bodies = self.kin.forward(*state)[0]
        wrong_state = list(state)
        wrong_state[4] = state[4]-xi[:, 1:30]
        self.assertGreater(float((self.kin.forward(*wrong_state)[0]-physical_bodies).abs().max()), .001)
        _, exact = mdp.reward(state, f, a, a, terms=True)
        _, wrong = mdp.reward(wrong_state, f, a, a, terms=True)
        self.assertLess(float(exact["e_body_pos"]), 1e-10)
        self.assertGreater(float(wrong["e_body_pos"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
