"""Capacity checks must preserve production sizes and use real aligned inputs."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from hoi.v1.resource_probe import ProbeConfig, validate_real_start_mix, full_active_wm_windows, summarize_gpu_samples, initialize_cuda_memory_counters


def fixture():
    asset_ids = torch.cat((torch.zeros(100), torch.ones(100))).long()
    motion = SimpleNamespace(asset_ids=asset_ids, clip_id=asset_ids, clip_end=torch.tensor([99, 199]))
    start = {"asset_id": torch.tensor([0, 1]), "f": torch.tensor([40, 140])}
    history = {"asset_id": start["asset_id"][:, None].expand(2, 10).clone(),
               "f": start["f"][:, None] - torch.arange(9, -1, -1)}
    for key, size in (("s", 71), ("o", 13), ("a_prev", 29), ("xi", 33), ("xi_o", 12)):
        start[key] = torch.randn(2, size)
        history[key] = start[key][:, None].expand(2, 10, size).clone()
    return start, history, motion


class ResourceProbeTest(unittest.TestCase):
    def test_uninitialized_allocator_cannot_reset_peak_counters(self):
        class Allocator:
            selected = None
            initialized = False
            reset = False
            def set_device(self, selected):
                self.selected = selected
            def init(self):
                if self.selected is None:
                    raise RuntimeError("CUDA device must be selected before initialization")
                self.initialized = True
            def reset_peak_memory_stats(self, selected):
                if not self.initialized or selected != self.selected:
                    raise RuntimeError("Invalid device argument")
                self.reset = True
        allocator = Allocator()
        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            allocator.reset_peak_memory_stats(torch.device("cuda:0"))
        self.assertEqual(initialize_cuda_memory_counters("cuda:0", cuda=allocator), torch.device("cuda:0"))
        self.assertTrue(allocator.reset)
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            initialize_cuda_memory_counters("cpu", cuda=allocator)

    def test_config_preserves_production_capacity(self):
        self.assertEqual(ProbeConfig().validate().pi_batch, 4096)
        for key, value in (("num_envs", 8), ("buffer_steps", 96), ("pi_batch", 8), ("horizon", 16),
                           ("collect_steps", 9), ("expected_groups", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(ProbeConfig(), **{key: value}).validate()

    def test_real_multi_asset_history_and_prefix_padding(self):
        start, history, motion = fixture()
        history["f"][0, :3] = 33
        report = validate_real_start_mix(start, history, motion, expected_assets=[0, 1], batch_size=2, horizon=32)
        self.assertEqual(report["asset_counts"], {"0": 1, "1": 1})
        self.assertEqual(report["history_prefix_padding_edges"], 2)
        self.assertFalse(report["reference_only_states"])

    def test_mismatched_asset_and_single_asset_cannot_pass(self):
        start, history, motion = fixture()
        with self.assertRaisesRegex(ValueError, "every static group"):
            validate_real_start_mix(start, history, motion, expected_assets=[0, 1, 2], batch_size=2, horizon=32)
        history["asset_id"][0, 4] = 1
        with self.assertRaisesRegex(ValueError, "crosses fixed"):
            validate_real_start_mix(start, history, motion, expected_assets=[0, 1], batch_size=2, horizon=32)

    def test_nonprefix_padding_and_wrong_physical_endpoint_rejected(self):
        start, history, motion = fixture()
        history["f"][0] = torch.tensor([32, 33, 34, 34, 35, 36, 37, 38, 39, 40])
        with self.assertRaisesRegex(ValueError, "only at the history prefix"):
            validate_real_start_mix(start, history, motion, expected_assets=[0, 1], batch_size=2, horizon=32)
        start, history, motion = fixture()
        history["o"][0, -1, 0] += 1
        with self.assertRaisesRegex(ValueError, "endpoint o"):
            validate_real_start_mix(start, history, motion, expected_assets=[0, 1], batch_size=2, horizon=32)

    def test_full_horizon_reference_range_required(self):
        start, history, motion = fixture()
        with self.assertRaisesRegex(ValueError, "full reference"):
            validate_real_start_mix(start, history, motion, expected_assets=[0, 1], batch_size=2, horizon=64)

    def test_wm_memory_graph_requires_all_real_rows_active(self):
        class Replay:
            def sample_windows(self, batch, window, split):
                assert split == "fit"
                mask = torch.ones(batch, window)
                mask[::2, -1] = 0
                return {"mask": mask, "f": torch.arange(batch)[:, None].expand(batch, window)}
        windows, selection = full_active_wm_windows(Replay(), 4, 8)
        self.assertEqual(windows["mask"].shape, (4, 8))
        self.assertEqual(selection["active_rows_per_step"], [4] * 8)
        self.assertTrue(bool((windows["f"][:, 0] % 2 == 1).all()))

        class AlwaysReset:
            def sample_windows(self, batch, window, split):
                return {"mask": torch.zeros(batch, window)}
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            full_active_wm_windows(AlwaysReset(), 4, 8, max_attempts=2)

    def test_gpu_receipt_excludes_other_devices_and_separates_other_jobs(self):
        samples = [{"elapsed_seconds": 1., "gpus": [["GPU-owned", "L40", "44000", "2200"],
                                                     ["GPU-other", "L40", "44000", "43000"]],
                    "processes": [["10", "1000", "GPU-owned"], ["11", "500", "GPU-owned"],
                                  ["12", "100", "GPU-owned"], ["99", "42000", "GPU-other"]]}]
        report = summarize_gpu_samples(samples, [10, 11], 10)
        self.assertEqual(report["assigned_device_uuids"], ["GPU-owned"])
        self.assertEqual(report["owned_process_sampled_peak_MiB_by_gpu_uuid"], {"GPU-owned": 1500})
        self.assertEqual(report["external_process_sampled_peak_MiB_by_gpu_uuid"], {"GPU-owned": 100})
        self.assertEqual(len(report["samples"][0]["assigned_gpus"]), 1)
        self.assertTrue(report["has_other_job_processes_on_assigned_device"])


if __name__ == "__main__":
    unittest.main()
