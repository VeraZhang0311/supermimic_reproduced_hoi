"""CPU tests of physical-unit rewards and oracle coordinates; no native simulation."""
import math
from types import SimpleNamespace
import unittest

import torch

from hoi_residual.features import (BLOCKS, FUTURE_STEPS, FUTURE_TIMES, ORACLE_DIM,
                                   ObjectHistory, future_frames, object_observation,
                                   observation_contract, oracle_features, torso_pose)
from hoi_residual.rewards import object_distance, reward
from supermimic.privileged import PRIVILEGED_DIM
from supermimic.rotations import yaw_mat


def z_rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def x_rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]])


def xz(matrix):
    # Independent statement of the v1.1 X-axis, then Z-axis convention.
    return torch.cat((matrix[..., :, 0], matrix[..., :, 2]), -1)


class NamedTorsoFixture:
    """A synthetic named-link FK; torso differs from root in yaw and tilt."""
    body_names = ["pelvis", "unrelated_link", "torso_link"]

    def forward(self, state):
        p, R = state[:2]
        torso_position = p + (R @ p.new_tensor([.1, -.05, .4])[:, None]).squeeze(-1)
        torso_rotation = R @ z_rotation(.5) @ x_rotation(.3)
        positions = torch.stack((p, p + .2, torso_position), 1)
        rotations = torch.stack((R, R @ x_rotation(-.4), torso_rotation), 1)
        return positions, rotations, torch.zeros_like(positions), torch.zeros_like(positions)


def make_motion(arrays):
    motion = SimpleNamespace(clip_end=torch.tensor([30, 61]),
                             clip_id=torch.cat((torch.zeros(31), torch.ones(31))).long(),
                             arrays=arrays, reference_calls=[])
    def reference(frames):
        motion.reference_calls.append(frames.clone())
        return tuple(value[frames] for value in motion.arrays)
    motion.object_ref = reference
    return motion


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

    def observation_fixture(self):
        S, _, _ = self.fixture()
        p = S[0].clone()
        p[1] += torch.tensor([.5, .2, 0.])
        S = (p, torch.stack((z_rotation(.2), z_rotation(-.1))), *S[2:])
        t = torch.arange(62, dtype=torch.float32)
        refs = (torch.stack((.01 * t, .02 * t + (t >= 31) * 2., .9 + .1 * torch.sin(t * .05)), -1),
                torch.stack([z_rotation(float(v) * .07) @ x_rotation(.2) for v in t]),
                torch.stack((.1 + t * .01, .2 - t * .02, .03 + t * .003), -1),
                torch.stack((t * 0 + .2, t * 0 + .1, t * .005), -1))
        frames = torch.tensor([2, 33])
        p, R, v, w = (array[frames].clone() for array in refs)
        O = (p + torch.tensor([.08, -.04, .02]), R @ x_rotation(-.1), v + .3, w - .2)
        torso = torso_pose(NamedTorsoFixture(), S)
        history = ObjectHistory(object_observation(torso, O))
        return S, torso, O, make_motion(refs), frames, history

    def test_named_torso_heading_matches_v11_and_differs_from_root_or_full_torso(self):
        S, torso, O, _, _, _ = self.observation_fixture()
        positions, rotations, _, _ = NamedTorsoFixture().forward(S)
        torch.testing.assert_close(torso[0], positions[:, 2], atol=0, rtol=0)
        torch.testing.assert_close(torso[1], rotations[:, 2], atol=0, rtol=0)
        H = yaw_mat(torso[1]).transpose(-1, -2)
        expected = torch.cat(((H @ (O[0] - torso[0])[..., None]).squeeze(-1), xz(H @ O[1])), -1)
        observed = object_observation(torso, O)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        root_H = yaw_mat(S[1]).transpose(-1, -2)
        root_anchored = (root_H @ (O[0] - S[0])[..., None]).squeeze(-1)
        full_torso = (torso[1].transpose(-1, -2) @ (O[0] - torso[0])[..., None]).squeeze(-1)
        self.assertFalse(torch.allclose(observed[:, :3], root_anchored))
        self.assertFalse(torch.allclose(observed[:, :3], full_torso))
        # Pose observations contain neither velocity nor reference errors.
        changed = (O[0], O[1], O[2] + 100, O[3] - 100)
        torch.testing.assert_close(object_observation(torso, changed), observed, atol=0, rtol=0)

    def test_schema_keeps_robot_prefix_exact(self):
        _, torso, O, motion, frames, history = self.observation_fixture()
        robot = torch.randn(2, PRIVILEGED_DIM)
        context = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        result = oracle_features(robot, torso, O, motion, frames, context, history.values)
        self.assertEqual(sum(BLOCKS.values()), ORACLE_DIM)
        self.assertEqual(result.shape, (2, 3647))
        self.assertTrue(torch.equal(result[:, :PRIVILEGED_DIM], robot))
        self.assertTrue(torch.equal(result[:, 3371:3461].reshape(2, 10, 9), history.values))
        self.assertTrue(torch.equal(result[:, -12:], context))
        self.assertEqual(list(BLOCKS.values()), [3371, 90, 63, 105, 6, 12])
        self.assertFalse(observation_contract()["asset_identity_input"])
        self.assertFalse(observation_contract()["bbox_input"])

    def test_seven_future_slots_clamp_each_clip_without_crossing_boundary(self):
        _, _, _, motion, _, _ = self.observation_fixture()
        self.assertEqual(FUTURE_STEPS, (1, 2, 5, 7, 12, 18, 25))
        torch.testing.assert_close(torch.tensor(FUTURE_TIMES), torch.tensor([.02, .04, .10, .14, .24, .36, .50]))
        expected = torch.tensor([[1, 2, 5, 7, 12, 18, 25], [32, 33, 36, 38, 43, 49, 56]])
        torch.testing.assert_close(future_frames(motion, torch.tensor([0, 31])), expected)
        ends = future_frames(motion, torch.tensor([29, 60]))
        torch.testing.assert_close(ends, torch.tensor([[30] * 7, [61] * 7]))
        torch.testing.assert_close(motion.clip_id[ends], torch.tensor([[0] * 7, [1] * 7]))

    def test_direct_goals_and_tracking_errors_share_exact_targets_in_current_torso(self):
        _, torso, O, motion, frames, history = self.observation_fixture()
        result = oracle_features(torch.zeros(2, PRIVILEGED_DIM), torso, O, motion, frames,
                                 torch.zeros(2, 12), history.values)
        requested = torch.tensor([[3, 4, 7, 9, 14, 20, 27], [34, 35, 38, 40, 45, 51, 58]])
        self.assertEqual(len(motion.reference_calls), 1)
        torch.testing.assert_close(motion.reference_calls[0], requested)
        goals, errors = result[:, 3461:3524].reshape(2, 7, 9), result[:, 3524:3629].reshape(2, 7, 15)
        for world in range(2):
            H = yaw_mat(torso[1][world]).T
            for slot, frame in enumerate(requested[world]):
                pr, Rr, vr, wr = (array[frame] for array in motion.arrays)
                expected_goal = torch.cat((H @ (pr - torso[0][world]), xz(H @ Rr)))
                expected_error = torch.cat((H @ (pr - O[0][world]),
                                            xz(H @ Rr @ O[1][world].T @ H.T),
                                            H @ (vr - O[2][world]), H @ (wr - O[3][world])))
                torch.testing.assert_close(goals[world, slot], expected_goal)
                torch.testing.assert_close(errors[world, slot], expected_error)
            torch.testing.assert_close(result[world, 3629:3635],
                                       torch.cat((H @ O[2][world], H @ O[3][world])))
        # At the end all seven direct goals AND error slots repeat the same
        # final state; no unseen first frame from the following clip leaks in.
        ended = oracle_features(torch.zeros(2, PRIVILEGED_DIM), torso, O, motion,
                                torch.tensor([29, 60]), torch.zeros(2, 12), history.values)
        for value in (ended[:, 3461:3524].reshape(2, 7, 9), ended[:, 3524:3629].reshape(2, 7, 15)):
            torch.testing.assert_close(value, value[:, :1].expand_as(value), atol=0, rtol=0)

    def test_history_preserves_collection_frames_and_isolates_partial_reset(self):
        _, torso, O, _, _, _ = self.observation_fixture()
        initial = object_observation(torso, O)
        history = ObjectHistory(initial)
        self.assertTrue(torch.equal(history.values, initial[:, None].expand(-1, 10, -1)))
        initial.zero_()
        self.assertFalse(torch.equal(history.values[:, -1], initial))
        recorded = []
        for tick in range(12):
            moving = (torso[0] + torch.tensor([tick * .03, 0., 0.]), z_rotation(tick * .04) @ torso[1])
            observed = object_observation(moving, O)
            recorded.append(observed.clone())
            history.advance(observed)
        self.assertTrue(torch.equal(history.values, torch.stack(recorded[-10:], 1)))
        self.assertFalse(torch.allclose(history.values[:, 0], history.values[:, -1]))
        before = history.values.clone()
        reset_observation = recorded[-1] + 3.
        history.reset(torch.tensor([1]), reset_observation)
        self.assertTrue(torch.equal(history.values[0], before[0]))
        self.assertTrue(torch.equal(history.values[1], reset_observation[1, None].expand(10, -1)))

    def test_shared_translation_and_yaw_do_not_change_oracle(self):
        S, torso, O, motion, frames, history = self.observation_fixture()
        robot, context = torch.randn(2, PRIVILEGED_DIM), torch.randn(2, 12)
        result = oracle_features(robot, torso, O, motion, frames, context, history.values)
        Q, shift = z_rotation(.7), torch.tensor([1.1, -2.3, 0.])
        transform = lambda state: (state[0] @ Q.T + shift, Q @ state[1], state[2] @ Q.T, state[3] @ Q.T, *state[4:])
        S2, O2 = transform(S), transform(O)
        torso2, motion2 = torso_pose(NamedTorsoFixture(), S2), make_motion(transform(motion.arrays))
        history2 = ObjectHistory(object_observation(torso2, O2))
        shifted = oracle_features(robot, torso2, O2, motion2, frames, context, history2.values)
        torch.testing.assert_close(result, shifted, atol=2e-6, rtol=1e-6)

    def test_nonfinite_or_bad_width_rejected(self):
        _, torso, O, motion, frames, history = self.observation_fixture()
        for context in (torch.zeros(2, 11), torch.full((2, 12), float("nan"))):
            with self.assertRaises(ValueError):
                oracle_features(torch.zeros(2, PRIVILEGED_DIM), torso, O, motion, frames, context, history.values)
        for invalid in (history.values[:, :9], torch.full((2, 10, 9), float("nan"))):
            with self.assertRaises(ValueError):
                oracle_features(torch.zeros(2, PRIVILEGED_DIM), torso, O, motion, frames,
                                torch.zeros(2, 12), invalid)
        invalid = (torch.full_like(O[0], float("inf")), *O[1:])
        with self.assertRaises(ValueError):
            oracle_features(torch.zeros(2, PRIVILEGED_DIM), torso, invalid, motion, frames,
                            torch.zeros(2, 12), history.values)
        with self.assertRaises(ValueError):
            ObjectHistory(torch.full((2, 9), float("nan")))
        with self.assertRaises(ValueError):
            history.advance(torch.zeros(3, 9))
        with self.assertRaises(ValueError):
            history.reset(torch.tensor([0]), torch.zeros(1, 9))

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
