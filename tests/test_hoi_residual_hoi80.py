import copy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hoi_residual.data import ASSETS
from hoi_residual.evaluate_hoi80 import _verified_protocol, aggregate, collect, score_lane
from supermimic.mdp import clamp_action


class PanelTests(unittest.TestCase):
    def two_lanes(self):
        class Env:
            device, N = "cpu", 2
            def __init__(self):
                self.f = torch.tensor([0, 3])
                self.a_prev, self.xi = torch.zeros(2, 29), torch.zeros(2, 33)
                self.tick = 0
            def state(self):
                pos = torch.zeros(2, 3)
                pos[:, 0] = self.tick * .01
                return (pos, torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 3), torch.zeros(2, 3),
                        torch.zeros(2, 29), torch.zeros(2, 29))
            def object_state(self):
                return (torch.tensor([[0., 0., 0.], [2., 0., 0.]]), torch.eye(3).repeat(2, 1, 1),
                        torch.zeros(2, 3), torch.zeros(2, 3))
            def step(self, action):
                self.tick += 1
                self.f = self.f + 1
                self.a_prev = action.clone()
                # Both lanes fail immediately, even though the second lasts longer.
                return dict(previous_q=torch.zeros(2, 29), previous_qd=torch.zeros(2, 29),
                            reward=torch.ones(2) * self.tick, terminated=torch.ones(2, dtype=torch.bool),
                            terms={"object_surface_error_m": torch.ones(2) * self.tick * .01,
                                   "object_surface": torch.zeros(2), "action_rate": torch.zeros(2)})
            def object_context(self):
                return torch.zeros(2, 12)
            def actuator_state(self):
                return torch.zeros(2, 88)

        env = Env()
        starts, ends = env.f.clone(), torch.tensor([2, 7])
        calls, inputs = [], []
        def goal(frames, history):
            calls.append(frames.clone())
            return torch.zeros(2, 730)
        def object_ref(frames):
            self.assertTrue(bool((frames[0] <= 2).all() and (frames[1] >= 3).all()))
            return tuple(v[:1].expand(*frames.shape, *v.shape[1:]) for v in env.object_state())
        motion = SimpleNamespace(clip_start=starts, clip_end=ends, object_ref=object_ref, goal=goal,
                                 clip_id=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]), cases=[
                                     dict(scored_control_steps=2, id="short", object_asset=ASSETS[0], source_group="a",
                                          canonical_index=0, motion_sha256="m", reference_sha256="r")])
        def act(oracle, base, points, *, deterministic):
            self.assertTrue(deterministic)
            self.assertEqual(tuple(points.shape), (128, 3))
            inputs.append(oracle[:, 3371:3461].reshape(2, 10, 9))
            return torch.full_like(base, .35)
        kin = SimpleNamespace(body_names=["torso_link"], forward=lambda s: (s[0][:, None], s[1][:, None], None, None))
        result = collect(env, motion, SimpleNamespace(obs=lambda *args: torch.zeros(2, 99)),
                         lambda *args: torch.zeros(2, 3371), kin, lambda *args: torch.full((2, 29), .25),
                         SimpleNamespace(act=act), torch.zeros(128, 3), SimpleNamespace(sample_contacts=lambda: None))
        return env, motion, calls, inputs, result

    def test_failed_unequal_lanes_continue_with_frozen_endpoints_and_torso_histories(self):
        env, motion, calls, inputs, result = self.two_lanes()
        self.assertEqual(env.tick, 4)
        self.assertEqual(torch.stack(calls).tolist(), [[0, 3], [1, 4], [1, 5], [1, 6]])
        self.assertTrue(torch.equal(env.f, motion.clip_end))
        self.assertEqual(tuple(result["robot_state_wxyz"].shape), (5, 2, 71))
        torch.testing.assert_close(result["action"], clamp_action(result["base_action"] + result["residual_action"]))
        torch.testing.assert_close(inputs[0], inputs[0][:, -1:].expand_as(inputs[0]))
        torch.testing.assert_close(inputs[-1][:, -1, 0], torch.tensor([-.03, 1.97]))

    @unittest.skipUnless(find_spec("hoi_zero_shot"), "canonical03 helper package required for the legacy scorer")
    def test_short_lane_rollout_scores_interval_1_to_T_with_exact_canonical_helpers(self):
        _verified_protocol()
        _, motion, _, _, rollout = self.two_lanes()
        ref = np.zeros((3, 630), dtype=np.float32)
        ref[:, 77] = 1.
        ref[0, 71] = 50.  # Initial state is deliberately far from reference and never scored.
        class Markers:
            body_names, joint_names = tuple(range(39)), tuple(range(29))
            def __call__(self, state):
                return state[0][:, None].expand(-1, 39, -1)
        with TemporaryDirectory() as directory:
            mdp = SimpleNamespace(reward=lambda state, frames, actions, previous: torch.ones_like(frames).float(),
                                  termination=lambda state, frames: frames == 1)
            case = score_lane(Path(directory), motion, 0, ref, rollout, Markers(), mdp)
            self.assertTrue(case["human_pass"] and case["object_pass"] and case["joint_pass"])
            self.assertEqual(case["scored_frames"], 2)
            self.assertEqual(case["resets_within_clip"], 0)
            self.assertEqual(case["mean_hoi_reward_per_step"], 1.5)
            self.assertAlmostEqual(case["mean_object_surface_error_m"], .015)
            self.assertEqual(case["first_author_robot_failure_step"], 1)
            with np.load(Path(directory) / case["rollout"], allow_pickle=False) as saved:
                self.assertEqual(saved["robot_state_wxyz"].shape, (3, 71))
                self.assertEqual(saved["action"].shape, (2, 29))
                self.assertEqual(saved["residual_termination"].tolist(), [True, True])
                np.testing.assert_array_equal(saved["time"], [0., .02, .04])

    def test_aggregate_rejects_missing_duration_reordered_id_or_mixed_checkpoint(self):
        reports, panel = [], []
        for asset_idx, asset in enumerate(ASSETS):
            cases = []
            for j in range(10):
                index = asset_idx * 10 + j
                case = dict(id=f"c{index}", object_asset=asset, canonical_index=index,
                            scored_frames=index + 1, control_steps=index + 1, criteria={},
                            human_pass=True, object_pass=(j == 0), joint_pass=(j == 0),
                            mean_hoi_reward_per_step=index, mean_object_surface_error_m=index,
                            mean_object_surface_reward_per_step=0., mean_action_rate_reward_per_step=0.,
                            author_robot_normalized_return=1.)
                cases.append(case)
                panel.append(dict(id=case["id"], object_asset=asset, scored_control_steps=index + 1))
            reports.append(dict(asset=asset, schema="fixture", checkpoint_sha256="f", iteration=400,
                                gmt_checkpoint_sha256="g", gmt_iteration=4000, canonical_manifest_sha256="c",
                                ultra_manifest_sha256="u", asset_catalog_sha256="a", shape_bank_sha256="s",
                                protocol_files_sha256={}, cases=cases, summary={}))
        canonical = {"cases": panel}
        summary = aggregate(reports, canonical)["summary"]
        self.assertEqual(summary["joint_pass_count"], 8)
        self.assertEqual(summary["equal_clip_means"]["mean_hoi_reward_per_step"], 39.5)
        for key, value in (("id", "duplicate"), ("scored_frames", 0), ("canonical_index", 3)):
            invalid = copy.deepcopy(reports)
            invalid[0]["cases"][0][key] = value
            with self.assertRaisesRegex(ValueError, "panel or scored duration"):
                aggregate(invalid, canonical)
        invalid = copy.deepcopy(reports)
        invalid[-1]["checkpoint_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            aggregate(invalid, canonical)


if __name__ == "__main__":
    unittest.main()
