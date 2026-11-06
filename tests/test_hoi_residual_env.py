"""CPU tests of physical-unit rewards and oracle coordinates; no native simulation."""
import math
from types import SimpleNamespace
import unittest

import torch

from hoi_residual.features import BLOCKS, ORACLE_DIM, oracle_features
from hoi_residual.rewards import object_distance, reward
from supermimic.privileged import PRIVILEGED_DIM


def z_rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class OracleCoordinateTests(unittest.TestCase):
    def fixture(self, batch=2):
        p = torch.tensor([[.2, -.1, .8]]).expand(batch, -1)
        R = torch.eye(3).expand(batch, -1, -1)
        v = torch.tensor([[.1, .2, .3]]).expand(batch, -1)
        w = torch.tensor([[0., 0., .4]]).expand(batch, -1)
        q = torch.zeros(batch, 29)
        S = (p, R, v, w, q, q)
        O = (p + torch.tensor([.3, -.2, .1]), R, v * 2, w * 2)
        motion = SimpleNamespace(clip_end=torch.tensor([20]), clip_id=torch.zeros(21, dtype=torch.long), object_ref=lambda f: O)
        return S, O, motion

    def test_schema_keeps_robot_prefix_exact(self):
        S, O, motion = self.fixture()
        robot = torch.randn(2, PRIVILEGED_DIM)
        result = oracle_features(robot, S, O, motion, torch.tensor([0, 10]), torch.zeros(2, 12),
                                 torch.tensor([0, 7]), torch.zeros(6))
        self.assertEqual(sum(BLOCKS.values()), ORACLE_DIM)
        self.assertEqual(result.shape, (2, 3442))
        self.assertTrue(torch.equal(result[:, :PRIVILEGED_DIM], robot))
        self.assertTrue(torch.equal(result[:, -14:-6].sum(-1), torch.ones(2)))

    def test_shared_translation_and_yaw_do_not_change_oracle(self):
        S, O, motion = self.fixture()
        args = (torch.zeros(2, PRIVILEGED_DIM), torch.tensor([0, 10]), torch.zeros(2, 12), torch.tensor([0, 7]), torch.zeros(6))
        result = oracle_features(args[0], S, O, motion, *args[1:])
        Q, shift = z_rotation(.7), torch.tensor([1.1, -2.3, 0.])
        transform = lambda state: (state[0] @ Q.T + shift, Q @ state[1], state[2] @ Q.T, state[3] @ Q.T, *state[4:])
        S2, O2 = transform(S), transform(O)
        motion2 = SimpleNamespace(clip_end=motion.clip_end, clip_id=motion.clip_id, object_ref=lambda f: O2)
        shifted = oracle_features(args[0], S2, O2, motion2, *args[1:])
        self.assertTrue(torch.allclose(result, shifted, atol=3e-7))

    def test_nonfinite_or_bad_width_rejected(self):
        S, O, motion = self.fixture()
        for context in (torch.zeros(2, 11), torch.full((2, 12), float("nan"))):
            with self.assertRaises(ValueError):
                oracle_features(torch.zeros(2, PRIVILEGED_DIM), S, O, motion, torch.tensor([0, 10]),
                                context, torch.tensor([0, 7]), torch.zeros(6))

    def test_object_error_is_mean_distance_in_meters(self):
        p, R = torch.zeros(2, 3), torch.eye(3).expand(2, -1, -1)
        points = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
        shifted = p + torch.tensor([.3, .4, 0.])
        self.assertTrue(torch.allclose(object_distance((p, R), (shifted, R), points), torch.full((2,), .5)))
        rotate = z_rotation(math.pi / 2).expand(2, -1, -1)
        self.assertTrue(torch.allclose(object_distance((p, R), (p, rotate), points), torch.full((2,), math.sqrt(2))))

    def test_reward_dt_and_surface_error_metric_units(self):
        S, O, motion = self.fixture()
        motion.ref_state = lambda f: S
        motion.reference_bodies = lambda f: (S[0][:, None], None, None, None)
        kin = SimpleNamespace(forward=lambda *s: (s[0][:, None], None, None, None))
        mdp = SimpleNamespace(kin=kin, motion=motion, soft_hi=torch.full((29,), 10.), soft_lo=torch.full((29,), -10.))
        zero = torch.zeros(2, 29)
        result, terms = reward(mdp, S, O, torch.tensor([0, 10]), zero, zero, torch.zeros(4, 3), zero)
        self.assertTrue(torch.allclose(result, torch.full((2,), 19.2 * .02)))
        self.assertTrue(torch.equal(terms["object_surface_error_m"], torch.zeros(2)))
        self.assertTrue(torch.allclose(terms["object_surface"], torch.full((2,), 2 * .02)))


if __name__ == "__main__":
    unittest.main()
