import copy
from types import SimpleNamespace
import unittest

import torch

from hoi_tables.sampling import AdaptiveRSI, RSIConfig


def motion():
    # Include terminal-only bins deliberately: the sampler must remove them.
    frame_bin = torch.cat((torch.arange(101)//50, 3+torch.arange(101)//50))
    return SimpleNamespace(partition="train", allowed_for_training=True, num_bins=6,
                           num_frames=202, bin_start=torch.tensor([0, 50, 100, 101, 151, 201]),
                           bin_clip=torch.tensor([0, 0, 0, 1, 1, 1]),
                           clip_start=torch.tensor([0, 101]), clip_end=torch.tensor([100, 201]),
                           clip_asset_ids=torch.tensor([0, 1]), frame_bin=frame_bin)


class TestAdaptiveRSI(unittest.TestCase):
    def test_asset_balance_outgoing_intervals_and_warmup(self):
        sampler = AdaptiveRSI(motion(), seed=3)
        sampler.update(torch.tensor([0]*10), torch.zeros(10), torch.zeros(10), elapsed_steps=torch.full((10,), 50))
        bins, probability = sampler.probabilities(iteration=500)
        self.assertTrue(torch.allclose(probability, torch.full((4,), .25)))
        self.assertEqual(set(bins.tolist()), {0, 1, 3, 4})
        frames, sampled = sampler.sample_frames(2001, iteration=500)
        self.assertLessEqual(abs((sampled < 3).sum().item()-(sampled >= 3).sum().item()), 1)
        self.assertTrue(bool((frames < sampler.motion.clip_end[sampler.motion.bin_clip[sampled]]).all()))
        self.assertFalse(bool(torch.isin(sampled, torch.tensor([2, 5])).any()))

    def test_fixed_asset_failure_priority_and_mastery_pool(self):
        sampler = AdaptiveRSI(motion(), asset_index=0, seed=4)
        sampler.update(torch.tensor([0]*10), torch.ones(10), torch.ones(10), elapsed_steps=torch.full((10,), 50))
        sampler.update(torch.tensor([1]*10), torch.zeros(10), torch.zeros(10), elapsed_steps=torch.full((10,), 50))
        bins, p = sampler.probabilities(iteration=700)
        # Uniform .5 contributes .25 each, hard .3 prefers bin1 and the
        # independently reserved .2 consolidation mass prefers mastered bin0.
        self.assertEqual(bins.tolist(), [0, 1])
        self.assertAlmostEqual(p[0].item(), .45, places=6)
        self.assertAlmostEqual(p[1].item(), .55, places=6)
        frames, chosen = sampler.sample_frames(200)
        self.assertTrue(bool((frames < 100).all()))
        self.assertTrue(bool((chosen < 2).all()))

    def test_no_mastery_without_duration_or_good_tracking(self):
        sampler = AdaptiveRSI(motion(), asset_index=0)
        sampler.update(torch.tensor([0]*20), torch.ones(20), torch.ones(20))
        sampler.update(torch.tensor([1]*20), torch.ones(20), torch.zeros(20), elapsed_steps=torch.full((20,), 50))
        _, probability = sampler.probabilities(iteration=700)
        self.assertTrue(torch.allclose(probability, torch.full((2,), .5)))

    def test_failure_credit_stays_in_clip_and_reaches_preceding_bin(self):
        sampler = AdaptiveRSI(motion(), asset_index=1)
        sampler.update(torch.tensor([3]), torch.tensor([False]), torch.tensor([0.]),
                       failure_frames=torch.tensor([180]), elapsed_steps=torch.tensor([79]))
        self.assertGreater(sampler.failure_observations[3].item(), 0)
        self.assertGreater(sampler.failure_observations[4].item(), 0)
        self.assertEqual(sampler.failure_observations[:3].sum().item(), 0)
        self.assertEqual(sampler.mastery_visits[4].item(), 0)

    def test_rng_statistics_resume_and_identity_rejection(self):
        sampler = AdaptiveRSI(motion(), seed=13)
        sampler.update(torch.tensor([0, 3]), torch.tensor([False, True]), torch.tensor([0., 1.]), elapsed_steps=torch.tensor([50, 50]))
        sampler.sample_frames(12, iteration=700)
        state = copy.deepcopy(sampler.state_dict())
        restored = AdaptiveRSI(motion(), seed=0)
        restored.load_state_dict(state)
        for left, right in zip(sampler.sample_frames(50), restored.sample_frames(50)):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(sampler.visits, restored.visits))
        different = motion()
        different.bin_start[1] = 49
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            AdaptiveRSI(different).load_state_dict(state)

    def test_protected_eval_and_incompatible_bins_rejected(self):
        evaluation = motion()
        evaluation.partition = "protected_eval"
        with self.assertRaisesRegex(ValueError, "training partition"):
            AdaptiveRSI(evaluation)
        sampler = AdaptiveRSI(motion(), asset_index=0)
        for bad in (2, 3, -1, 6):
            with self.assertRaises(ValueError):
                sampler.update([bad], [False], [0.])

    def test_bad_episode_identity_cannot_poison_curriculum(self):
        sampler = AdaptiveRSI(motion())
        for kwargs in ({"elapsed_steps": [-1]}, {"failure_frames": [180]}):
            with self.assertRaises(ValueError):
                sampler.update([0], [False], [0.], **kwargs)
            self.assertEqual(sampler.visits.sum().item(), 0)
            self.assertEqual(sampler.failure_observations.sum().item(), 0)

    def test_no_failure_sentinel_only_allowed_for_success(self):
        sampler = AdaptiveRSI(motion())
        sampler.update([0, 3], [True, False], [1., 0.], failure_frames=[-1, 180], elapsed_steps=[50, 50])
        self.assertEqual(sampler.visits.sum().item(), 2)
        with self.assertRaisesRegex(ValueError, "Actual failures"):
            sampler.update([0], [False], [0.], failure_frames=[-1])

    def test_one_failing_bin_priority_is_capped(self):
        reference = SimpleNamespace(partition="train", allowed_for_training=True,
                                    num_bins=100, num_frames=5001,
                                    bin_start=torch.arange(100)*50,
                                    bin_clip=torch.zeros(100, dtype=torch.long),
                                    clip_start=torch.tensor([0]), clip_end=torch.tensor([5000]),
                                    clip_asset_ids=torch.tensor([0]),
                                    frame_bin=(torch.arange(5001)//50).clamp_max(99))
        sampler = AdaptiveRSI(reference)
        sampler.update([0], [False], [0.])
        _, probability = sampler.probabilities(iteration=700)
        # 50% uniform + empty mastery fallback20%uniform contributes.007;
        # hard pool cannot place more than10/100 mass on the failing bin.
        self.assertLessEqual(probability[0].item(), .037001)
        self.assertGreater(probability[0].item(), probability[1].item())


if __name__ == "__main__":
    unittest.main()
