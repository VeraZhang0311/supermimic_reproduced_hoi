"""Rare real-replay splits must not be mistaken for missing coverage."""
import unittest
from unittest.mock import patch

import torch

from hoi.v1.buffer import ReplayBuffer


class WindowSamplingTests(unittest.TestCase):
    @staticmethod
    def continuous_buffer(length=1000, environments=2):
        replay = ReplayBuffer(length, environments)
        replay.size = length
        replay.data["f"][:] = torch.arange(length)[:, None]
        replay.data["ep_step"][:] = torch.arange(length)[:, None]
        return replay

    def test_production_holdout_batch_fills_with_rare_32_step_windows(self):
        replay = self.continuous_buffer()
        # 19 of 969 possible start ages qualify; replacement is intentional.
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            window = replay.sample_windows(2048, 32, split="holdout")
            self.assertEqual(tuple(window["f"].shape), (2048, 32))
            self.assertTrue(bool(window["mask"].eq(1).all()))
            self.assertTrue(bool(((window["f"] // 50).remainder(20) == 0).all()))
            self.assertEqual(set(window["f"][:, 0].tolist()), set(range(19)))

    def test_metadata_candidates_match_actual_windows_across_ring_and_resets(self):
        replay = self.continuous_buffer(17, 3)
        replay.ptr = 6
        rows = replay._row(torch.arange(replay.size))
        replay.data["f"][rows] = torch.arange(990, 1007)[:, None]
        replay.data["ep_step"][rows] = torch.arange(17, dtype=torch.int32)[:, None]
        replay.data["done"][rows[3], 0] = True
        replay.data["asset_id"][rows[5:], 1] = 1
        replay.data["ep_step"][rows[8:], 2] -= 8
        replay.data["f"][rows[10:], 0] += 1000
        for size in (1, 4, 8):
            count = replay.size - size + 1
            ages = torch.arange(count).repeat_interleave(replay.N)
            envs = torch.arange(replay.N).repeat(count)
            window = replay._windows_at(ages, envs, size)
            held = (window["f"] // 50).remainder(20) == 0
            for split in ("fit", "holdout"):
                keep = ((held if split == "holdout" else ~held) | ~window["mask"].bool()).all(-1)
                torch.testing.assert_close(replay._eligible_window_starts(size, split), keep.nonzero().flatten())

    def test_terminal_prefix_can_be_sampled_without_crossing_split_boundary(self):
        replay = self.continuous_buffer(8, 1)
        replay.data["f"][:, 0] = torch.arange(49, 57)
        replay.data["done"][0, 0] = True
        torch.manual_seed(4)
        window = replay.sample_windows(2048, 8, split="holdout")
        self.assertTrue(bool(window["f"][:, 0].eq(49).all()))
        self.assertTrue(bool(window["mask"][:, 0].eq(1).all()))
        self.assertTrue(bool(window["mask"][:, 1:].eq(0).all()))

    def test_absent_split_still_fails_after_bounded_metadata_check(self):
        replay = self.continuous_buffer(40, 2)
        with patch.object(replay, "_windows_at", wraps=replay._windows_at) as windows:
            with self.assertRaisesRegex(ValueError, "no eligible starts"):
                replay.sample_windows(2048, 8, split="fit")
            self.assertEqual(windows.call_count, 32)


if __name__ == "__main__":
    unittest.main()
