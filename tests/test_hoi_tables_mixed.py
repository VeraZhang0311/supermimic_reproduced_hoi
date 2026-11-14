import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch

from hoi_tables.batched_fk import BatchedFK, privileged_call
from hoi_tables.mixed import PROPORTIONS
from hoi_tables.replay import ReplayBuffer, FIELDS
from hoi_tables.table_actions import episode_metrics
from hoi_tables.telemetry import _curated_metric
from supermimic.privileged import PrivilegedFeatures, RobotKinematics38
from supermimic.rotations import so3_exp

from tests.test_hoi_tables_learning import Motion, fixture, window


class MixedTests(unittest.TestCase):
    def test_action_metrics_use_completed_episodes_and_skip_empty_rates(self):
        totals = [(3, 1, 1.5), (0, 0, 0.), (2, 2, -.1), (0, 0, 0.), (1, 0, .8)]
        metrics = _curated_metric({"iteration": 8050, "stage": "actor_critic", **episode_metrics(totals)})
        self.assertEqual(metrics["tables_by_action/carry/episode_count"], 3)
        self.assertAlmostEqual(metrics["tables_by_action/carry/failure_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["tables_by_action/carry/normalized_return"], .5)
        self.assertEqual(metrics["tables_by_action/grasp_table_leg/episode_count"], 0)
        self.assertNotIn("tables_by_action/grasp_table_leg/failure_rate", metrics)

    def test_fixed_lanes_produce_exact_wm_and_actor_quotas(self):
        lanes = tuple(int(32 * weight) for weight in PROPORTIONS)
        replay = ReplayBuffer(8, sum(lanes), asset_env_counts=lanes)
        row = {name: torch.zeros((32, *shape), dtype=dtype) for name, (shape, dtype) in FIELDS.items()}
        row["asset_id"] = torch.repeat_interleave(torch.arange(10), torch.tensor(lanes))
        row["episode_id"] = torch.arange(32)
        for tick in range(4):
            row["f"].fill_(tick)
            row["next_f"].fill_(tick + 1)
            replay.append(row)
        motion = SimpleNamespace(clip_end=torch.tensor([64]), clip_id=torch.zeros(64, dtype=torch.long))
        for batch in (replay.sample_starts(64, horizon=32, motion=motion), replay.sample_windows(64, 3)):
            ids = batch["asset_id"].reshape(64, -1)[:, 0]
            self.assertEqual(ids.bincount().tolist(), [count * 2 for count in lanes])
        row["asset_id"] = row["asset_id"].roll(1)
        with self.assertRaisesRegex(ValueError, "fixed native lanes"):
            replay.append(row)

    def test_grouped_full_update_matches_single_batch(self):
        baseline, start = fixture(B=4, H=3)
        optimized = deepcopy(baseline)
        points = start["points"][[0, 1]].clone()
        start["asset_id"] = torch.tensor([0, 1, 1, 0])
        start["points"] = points[start["asset_id"]]
        optimized.asset_points = points
        optimized.wm_microbatch = optimized.actor_microbatch = 1
        optimized.reward.by_asset = dict.fromkeys(range(2), optimized.reward)
        samples = window(baseline, start)
        torch.manual_seed(7)
        expected = baseline.update(2, window=samples, start=start)
        torch.manual_seed(7)
        actual = optimized.update(2, window=samples, start=start)
        self.assertEqual(set(actual["asset_wm_total"]), {0, 1})
        self.assertAlmostEqual(sum(actual["asset_wm_total"].values()) / 2, actual["wm_total"], places=5)
        for key in ("wm_total", "pi_loss", "v_loss"):
            self.assertAlmostEqual(expected[key], actual[key], places=4)
        for first, second in ((baseline.pi, optimized.pi), (baseline.wm, optimized.wm), (baseline.critic, optimized.critic)):
            for left, right in zip(first.parameters(), second.parameters(), strict=True):
                torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)

    def test_asset_h32_uses_only_complete_windows(self):
        learner, start = fixture(B=4, H=3)
        start["asset_id"] = torch.tensor([0, 0, 1, 1])
        learner.asset_points = start["points"][:2]
        samples = window(learner, start, K=32)
        samples["mask"][2:, -1] = 0
        report = learner.wm_validation(samples)
        self.assertEqual(report["horizons"]["32"]["valid_samples"], 1)
        self.assertEqual(set(report["asset_h32_object_rmse_m"]), {0})
        self.assertAlmostEqual(report["asset_h32_object_rmse_m"][0],
                               report["horizons"]["32"]["object_position_rmse_m"], places=5)

    def test_privileged_batched_fk_forward_and_gradient(self):
        kin = RobotKinematics38()
        batched = BatchedFK(kin, privileged=True)
        generator = torch.Generator().manual_seed(27)
        values = [torch.randn(12, dim, generator=generator).requires_grad_() for dim in (3, 3, 3, 3, 29, 29)]
        state = (values[0], so3_exp(values[1]), *values[2:])
        expected, actual = kin.forward(state), batched.forward(state)
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
        probes = [torch.randn(tensor.shape, generator=generator) for tensor in expected]

        def grad(outputs, retain):
            return torch.autograd.grad(sum((x * probe).sum() for x, probe in zip(outputs, probes)), values, retain_graph=retain)

        for left, right in zip(grad(expected, True), grad(actual, False), strict=True):
            torch.testing.assert_close(left, right, atol=4e-5, rtol=4e-5)

        motion = Motion()
        motion.privileged_reference = lambda frames: batched.forward(motion.ref_state(frames))
        features = PrivilegedFeatures(kin, motion)
        frames = torch.arange(len(values[0]))
        action, context, actuator = torch.zeros(12, 29), torch.zeros(12, 33), torch.zeros(12, 88)
        args = state, frames, action, context, action, action, actuator
        target = features(*args)
        optimized = privileged_call(features, *args, bodies=batched.forward(state))
        torch.testing.assert_close(optimized, target, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
