"""Check the variable-duration boundary that differs from a serial rollout."""
import unittest

import torch

from hoi_tables.hoi80_batch import AssetRollout, _completed_lane_resets


class BatchTests(unittest.TestCase):
    def test_only_finished_lanes_recycle(self):
        self.assertEqual(dict(_completed_lane_resets([2, 3, 5])), {2: [0], 3: [1], 4: [0]})

    def test_finished_lane_padding_never_enters_metrics(self):
        frames = {"reference_frame": torch.tensor([[10, 20], [11, 21], [12, 22], [11, 23]])}
        controls = {"reward": torch.tensor([[.1, .1], [.1, .1], [100., .1]]),
                    "failed": torch.tensor([[False, False], [False, False], [True, False]]),
                    "robot_failed": torch.tensor([[False, False], [False, False], [True, False]]),
                    "object_failed": torch.zeros(3, 2, dtype=torch.bool),
                    "quality": torch.ones(3, 2, dtype=torch.bool),
                    "clipped": torch.zeros(3, 2, dtype=torch.bool)}
        rollout = AssetRollout(frames, controls, [2, 3], [], {})
        entry = {"clip_id": "a", "source_group": "a", "object_asset": "box"}
        short, saved_frames, saved_controls = rollout.clip(0, entry)
        self.assertEqual(saved_frames["reference_frame"].tolist(), [10, 11, 12])
        self.assertEqual(len(saved_controls["reward"]), 2)
        self.assertFalse(short["failed"])
        self.assertAlmostEqual(short["return"], .2)


if __name__ == "__main__":
    unittest.main()
