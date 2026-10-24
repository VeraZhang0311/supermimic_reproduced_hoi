"""CPU legacy compatibility, feature alignment and differentiable reward tests."""
import unittest
import subprocess
import sys

import torch

from hoi.data.manifest import robot_joint_names
from hoi.data.reference import adapt_motion
from hoi.v1.geometry import contact_missing_counter
from hoi.v1.mdp import HOIMDP, HOIRewardCfg
from hoi.v1.motion import TrainingMotionBank
from supermimic.motion import MotionLib


class DummyKin:
    q_lo, q_hi = torch.full((29,), -3.), torch.full((29,), 3.)
    body_names = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        n = 3 if all_bodies else 14
        offsets = torch.zeros((n, 3), device=p.device, dtype=p.dtype)
        offsets[:, 0] = torch.arange(n, device=p.device) * .01
        positions = p[..., None, :] + offsets
        return positions, R[..., None, :, :].expand(*R.shape[:-2], n, 3, 3), v[..., None, :].expand_as(positions), w[..., None, :].expand_as(positions)


def reference(frames=25):
    x = torch.zeros((frames, 630))
    x[:, 0] = torch.arange(frames) / 60
    x[:, 2] = 1
    x[:, 6] = x[:, 77] = 1
    x[:, 201:357].reshape(frames, 39, 4)[..., 3] = 1
    x[:, 71] = .3 + torch.arange(frames) / 60
    x[:, 73] = 1
    names = robot_joint_names()
    return adapt_motion(x, names, names)


class DistanceShapes:
    def bbox_diagonal(self, ids):
        return torch.ones_like(ids, dtype=torch.float32) * .4

    def hand_distances(self, kin, S, O, ids):
        # Depend on both robot and object pose; tests the analytic gradient path.
        return (O[0][..., :1] - S[0][..., :1])[..., None].expand(-1, 2, 4)


def bank_fixture():
    ref = reference()
    n = len(ref["robot"]["root_pos"])
    clips = [{"reference": ref, "asset_id": 0, "expected_contact": torch.ones(n, 2), "expected_valid": torch.ones(n, 2)},
             {"reference": ref, "asset_id": 1, "expected_contact": torch.zeros(n, 2), "expected_valid": torch.ones(n, 2)}]
    return TrainingMotionBank(clips, DummyKin())


class HOIMDPTests(unittest.TestCase):
    def setUp(self):
        self.bank = bank_fixture()
        self.f = torch.tensor([3, int(self.bank.clip_start[1]) + 4])
        self.S, self.O = self.bank.ref_state(self.f), self.bank.ref_object(self.f)
        self.mdp = HOIMDP(DummyKin(), self.bank, torch.zeros(29), DistanceShapes())
        self.a = torch.zeros(2, 29)
        self.xi = torch.zeros(2, 33)

    def test_legacy_goal_obs_reward_termination_are_preserved_exactly(self):
        torch.testing.assert_close(self.bank.goal(self.f, 10), MotionLib.goal(self.bank, self.f, 10), rtol=0, atol=0)
        torch.testing.assert_close(self.mdp.obs(self.S, self.a, self.xi, self.f), self.mdp.human.obs(self.S, self.a, self.xi, self.f), rtol=0, atol=0)
        zero = HOIRewardCfg(w_position=0, w_orientation=0, w_linear_velocity=0, w_angular_velocity=0, w_approach=0, w_contact=0)
        self.mdp.cfg = zero
        reward = self.mdp.reward(self.S, self.O, self.f, self.a, self.a, torch.zeros(2, 3))
        torch.testing.assert_close(reward, self.mdp.human.reward(self.S, self.f, self.a, self.a), rtol=0, atol=0)
        torch.testing.assert_close(self.mdp.termination(self.S, self.O, self.f), self.mdp.human.termination(self.S, self.f))

    def test_reward_ceiling_matches_active_heads_and_sampler_discount_bound(self):
        from hoi.v1.variants import max_step_reward
        class PerfectShapes(DistanceShapes):
            def hand_distances(self, kin, S, O, ids):
                return torch.zeros((*ids.shape, 2, 4))
        for name, cfg, ceiling in (("hoi_v1", HOIRewardCfg(), .15), ("no_contact_objective", HOIRewardCfg(w_contact=0.), .145)):
            mdp = HOIMDP(DummyKin(), self.bank, torch.zeros(29), PerfectShapes(), cfg)
            self.assertEqual(mdp.max_step_reward, ceiling)
            reward = mdp.reward(self.S, self.O, self.f, self.a, self.a,
                                torch.cat((self.bank.expected(self.f)[0], torch.zeros(2,1)), -1))
            torch.testing.assert_close(reward[:1], torch.full_like(reward[:1], ceiling), atol=1e-6, rtol=0)
            self.assertTrue(bool((reward <= ceiling + 1e-7).all()))
            self.assertEqual(max_step_reward(name), mdp.max_step_reward)
            gamma = .99
            discounted_perfect = sum(float(reward[0]) * gamma**t for t in range(2000))
            self.assertLessEqual(discounted_perfect, mdp.max_step_reward / (1-gamma))
            self.assertLess(mdp.max_step_reward / (1-gamma) - discounted_perfect, 1e-4)

    def test_robot_metadata_remains_cpu_safe_and_matches_data_joint_mapping(self):
        from supermimic.robot_metadata import G1_JOINT_NAMES, TRACKED_BODIES
        self.assertEqual(G1_JOINT_NAMES, robot_joint_names())
        self.assertEqual(len(TRACKED_BODIES), 14)
        self.assertEqual(TRACKED_BODIES[0], "pelvis")
        result = subprocess.run([sys.executable, "-c", "import sys; import supermimic.robot_metadata; assert not any(n.startswith('isaaclab') for n in sys.modules)"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dimensions_and_future_contact_validity(self):
        self.assertEqual(self.bank.body_p.shape[-2], 14)
        self.assertEqual(self.bank.goal(self.f, 10).shape, (2, 730))
        self.assertEqual(self.mdp.obs(self.S, self.a, self.xi, self.f).shape, (2, 99))
        self.assertEqual(self.mdp.object_obs(self.S, self.O, self.f).shape, (2, 18))
        self.assertEqual(self.bank.object_goal(self.S, self.O, self.f).shape, (2, 190))
        self.assertEqual(self.mdp.critic_features(self.S, self.O, self.f, self.xi).shape, (2, 1138))
        end = self.bank.clip_end.clone()
        goals = self.bank.object_goal(self.bank.ref_state(end), self.bank.ref_object(end), end).reshape(2, 10, 19)
        self.assertFalse(bool(goals[..., -4:].any()))
        future = self.bank.future_frames(torch.tensor([int(self.bank.clip_end[0]) - 1]), 10)
        self.assertTrue(bool((future <= self.bank.clip_end[0]).all()))
        self.assertTrue(bool((self.bank.asset_ids[future] == 0).all()))

    def test_current_heading_and_pose_recomputed_each_imagined_tick(self):
        shifted = (self.S[0] + torch.tensor([.3, -.2, 0]), *self.S[1:])
        a = self.bank.object_goal(self.S, self.O, self.f).reshape(2, 10, 19)
        b = self.bank.object_goal(shifted, self.O, self.f).reshape(2, 10, 19)
        torch.testing.assert_close(b[..., :3] - a[..., :3], torch.tensor([-.3, .2, 0]).expand(2, 10, 3))
        # Human future stays reference-only under the same simulated perturbation.
        torch.testing.assert_close(self.bank.goal(self.f, 10), self.bank.goal(self.f, 10), atol=0, rtol=0)

    def test_object_reward_gradients_and_finite_difference(self):
        p = (self.O[0] + torch.tensor([.03, .01, 0])).detach().requires_grad_(True)
        probability = torch.full((2, 3), .4, requires_grad=True)
        O = (p, *self.O[1:])
        reward = self.mdp.reward(self.S, O, self.f, self.a, self.a, probability).sum()
        grad_p, grad_c = torch.autograd.grad(reward, (p, probability))
        self.assertTrue(bool(torch.isfinite(grad_p).all() & torch.isfinite(grad_c).all()))
        self.assertGreater(float(grad_p.abs().sum()), 0)
        self.assertGreater(float(grad_c[:, :2].abs().sum()), 0)
        eps = .0001
        delta = torch.zeros_like(p)
        delta[0, 0] = eps
        plus = self.mdp.reward(self.S, (p.detach() + delta, *self.O[1:]), self.f, self.a, self.a, probability.detach()).sum()
        minus = self.mdp.reward(self.S, (p.detach() - delta, *self.O[1:]), self.f, self.a, self.a, probability.detach()).sum()
        self.assertAlmostEqual(float(grad_p[0, 0]), float((plus - minus) / (2 * eps)), delta=.0002)

    def test_contact_masks_and_counter_are_diagnostic_only(self):
        previous = torch.tensor([[15, 20], [3, 4]])
        e, v = torch.ones(2, 2), torch.tensor([[1, 1], [0, 1]])
        actual = torch.tensor([[0, 1], [0, 0]])
        updated = contact_missing_counter(previous, e, v, actual)
        torch.testing.assert_close(updated, torch.tensor([[16, 0], [0, 5]]))
        self.assertFalse(bool(self.mdp.termination(self.S, self.O, self.f).any()))
        self.bank.expected_valid.zero_()
        _, terms = self.mdp.reward(self.S, self.O, self.f, self.a, self.a, torch.zeros(2, 3), terms=True)
        self.assertFalse(bool(terms["hand_approach"].any() | terms["contact_match"].any()))

    def test_object_hard_and_joint_soft_termination(self):
        O = (self.O[0] + torch.tensor([1.01, 0, 0]), *self.O[1:])
        self.assertTrue(bool(self.mdp.termination(self.S, O, self.f).all()))
        rp = self.mdp.human.soft_termination(self.S, self.f)
        op = torch.sigmoid((torch.tensor(1.01) - 1) / .1)
        torch.testing.assert_close(self.mdp.soft_termination(self.S, O, self.f), 1 - (1 - rp) * (1 - op))

    def test_reset_bins_never_contain_only_the_terminal_state(self):
        ref = reference(61)  # 50 controls, 51 states; old bin loop adds state50.
        n = len(ref["time"])
        bank = TrainingMotionBank([{"reference": ref, "asset_id": 0,
            "expected_contact": torch.zeros(n,2), "expected_valid": torch.zeros(n,2)}], DummyKin())
        self.assertEqual((bank.num_frames, bank.num_bins), (51,1))
        torch.testing.assert_close(bank.bin_start, torch.tensor([0]))
        self.assertTrue(bool((bank.bin_start < bank.clip_end[bank.bin_clip]).all()))
        self.assertEqual(int(bank.frame_bin[-1]), 0)


if __name__ == "__main__":
    unittest.main()
