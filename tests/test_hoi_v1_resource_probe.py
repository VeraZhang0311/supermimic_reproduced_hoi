"""Capacity checks must preserve production sizes and use real aligned inputs."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import unittest

import torch

from hoi.v1.resource_probe import stratified_real_starts, read_probe_config, build_probe_models, initialize_probe_models, ProbeConfig, validate_real_start_mix, full_active_wm_windows, summarize_gpu_samples, initialize_cuda_memory_counters


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
    def test_formal_teacher_config_keeps_capacity_and_config_seed(self):
        from hoi.v1.config import TrainConfig
        from hoi.v1.privileged import TEACHER_PROFILE
        path = Path(__file__).resolve().parents[1] / "configs/hoi_v1_privileged_teacher.json"
        config = read_probe_config(path)
        self.assertEqual(config, replace(TrainConfig(), actor_observation_profile=TEACHER_PROFILE))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.json"
            path.write_text(json.dumps({"actor_observation_profile": TEACHER_PROFILE, "seed": 71}))
            self.assertEqual(read_probe_config(path).seed, 71)
            self.assertEqual(read_probe_config(path, seed=9).seed, 9)
            path.write_text(json.dumps({"experiment_kind": "smoke", "num_envs": 8}))
            with self.assertRaisesRegex(ValueError, "preserves production"):
                read_probe_config(path)
            path.write_text(json.dumps({"unknown_field": True}))
            with self.assertRaisesRegex(ValueError, "unknown training"):
                read_probe_config(path)

    def test_model_construction_binds_teacher_features_and_full_actor(self):
        from hoi.v1.privileged import TEACHER_PROFILE, PRIVILEGED_DIM
        config = replace(read_probe_config(), actor_observation_profile=TEACHER_PROFILE)
        mdp, bundle = SimpleNamespace(), {"reference_kinematics": "bound fixture"}
        with patch("hoi.v1.privileged.configure_teacher") as configure:
            pi, wm, critic = build_probe_models(config, mdp, bundle, "cpu")
        configure.assert_called_once_with(mdp, bundle, "cpu")
        self.assertTrue(pi.is_privileged_teacher)
        self.assertEqual(pi.dec[0].in_features, 1298 + 26 + PRIVILEGED_DIM)
        self.assertEqual(pi.privileged_norm.mean.shape[-1], PRIVILEGED_DIM)
        self.assertEqual(wm.contact_head[-1].out_features, 16)
        self.assertEqual(critic.net[0].in_features, 1548)
        with patch("hoi.v1.privileged.configure_teacher") as configure:
            baseline = build_probe_models(read_probe_config(), mdp, bundle, "cpu")[0]
        configure.assert_not_called()
        self.assertFalse(baseline.is_privileged_teacher)
        self.assertEqual(baseline.dec[0].in_features, 1298 + 26)

    def test_fresh_initialization_never_reads_checkpoint_or_mutates_weights(self):
        models = [torch.nn.Linear(3, 2)]
        original = {key: value.clone() for key, value in models[0].state_dict().items()}
        with patch("hoi.v1.train.read_checkpoint", side_effect=AssertionError("must not read")), \
             patch("hoi.v1.checkpoint.migrate_legacy", side_effect=AssertionError("must not migrate")):
            result = initialize_probe_models(models, from_scratch=True, physics={})
        self.assertEqual(result, {"mode": "explicit_from_scratch"})
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, original[key])
        for fresh, checkpoint in ((False, None), (True, Path("checkpoint.pt"))):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                initialize_probe_models(models, from_scratch=fresh, init_checkpoint=checkpoint, physics={})

    def test_legacy_initialization_preserves_explicit_physics_and_source_identity(self):
        models = [object(), object(), object()]
        path = Path("legacy.pt")
        old, physics, legacy = {"old": True}, {"hand_collision": True}, {"hand_collision": False}
        with patch("hoi.v1.train.read_checkpoint", return_value=(old, "sha")) as read, \
             patch("hoi.v1.checkpoint.migrate_legacy", return_value={"mode": "migration"}) as migrate:
            result = initialize_probe_models(models, init_checkpoint=path, physics=physics,
                                             legacy_physics=legacy, allow_physics_change=True)
        read.assert_called_once_with(path)
        migrate.assert_called_once_with(old, *models, physics=physics, legacy_physics=legacy, allow_physics_change=True)
        self.assertEqual(result, {"mode": "migration", "path": str(path.resolve()), "sha256": "sha"})

    def test_resource_stratification_retains_rare_asset_real_row_and_history(self):
        base_start, base_history, motion = fixture()
        candidates = []
        class Replay:
            calls = 0
            def sample_starts(self, batch, hist, *, horizon, motion):
                assert (batch, hist, horizon) == (4, 10, 32)
                ids = torch.tensor([0, 0, 0, 0] if self.calls == 0 else [0, 0, 0, 1])
                start = {key: value[ids].clone() for key, value in base_start.items()}
                history = {key: value[ids].clone() for key, value in base_history.items()}
                tokens = torch.arange(self.calls * 4, (self.calls + 1) * 4).float()
                start["row_token"] = tokens
                history["row_token"] = tokens[:, None].expand(4, 10).clone()
                # Every field retains its candidate row, including the scene payload.
                start["support_pose_xyzw"] = tokens[:, None, None].expand(4, 2, 7).clone()
                history["support_pose_xyzw"] = start["support_pose_xyzw"][:, None].expand(4, 10, 2, 7).clone()
                candidates.append((start, history))
                self.calls += 1
                return start, history
        replay = Replay()
        start, history, report = stratified_real_starts(replay, 4, motion, expected_assets=[0, 1], horizon=32)
        self.assertEqual(start["asset_id"].shape, (4,))
        self.assertEqual(history["f"].shape, (4, 10))
        self.assertEqual(set(start["asset_id"].tolist()), {0, 1})
        self.assertEqual(start["row_token"][start["asset_id"] == 1].tolist(), [7.])
        for row, token in enumerate(start["row_token"].int().tolist()):
            candidate_start, candidate_history = candidates[token // 4]
            for key in start:
                torch.testing.assert_close(start[key][row], candidate_start[key][token % 4])
                torch.testing.assert_close(history[key][row], candidate_history[key][token % 4])
        self.assertEqual(report["selection_attempts"], 2)
        self.assertEqual(report["candidate_real_starts"], 8)
        self.assertEqual(report["final_batch_size"], 4)
        self.assertFalse(report["production_sampling_changed"])
        self.assertIn("resource_only_stratified", report["sampling"])

    def test_resource_stratification_missing_asset_is_bounded_and_explicit(self):
        start, history, motion = fixture()
        class Replay:
            calls = 0
            def sample_starts(self, batch, hist, **kwargs):
                self.calls += 1
                ids = torch.zeros(batch, dtype=torch.long)
                return ({key: value[ids] for key, value in start.items()},
                        {key: value[ids] for key, value in history.items()})
        replay = Replay()
        with self.assertRaisesRegex(ValueError, "missing asset IDs:.*1"):
            stratified_real_starts(replay, 4, motion, expected_assets=[0, 1], horizon=32, max_attempts=3)
        self.assertEqual(replay.calls, 3)
        for kwargs in ({"batch_size": 1, "expected_assets": [0, 1]},
                       {"batch_size": 4, "expected_assets": [0, 0]},
                       {"batch_size": 4, "expected_assets": []},
                       {"batch_size": 4, "expected_assets": [0, 1], "max_attempts": 0}):
            with self.assertRaises(ValueError):
                stratified_real_starts(replay, motion=motion, horizon=32, **kwargs)
        self.assertEqual(replay.calls, 3)

    def test_resource_stratification_uses_unchanged_real_replay_entry(self):
        from tests.test_hoi_v1_learner import make_learner, transition
        torch.manual_seed(42)
        learner = make_learner(horizon=3)
        second = int((learner.motion.asset_ids == 1).nonzero().flatten()[0])
        for step in range(24):
            learner.buffer.add(transition(learner, [step, second + step], step))
        start, history, report = stratified_real_starts(learner.buffer, 32, learner.motion,
            expected_assets=[0, 1], horizon=3)
        self.assertEqual(start["f"].shape, (32,))
        self.assertEqual(history["f"].shape, (32, 10))
        self.assertEqual(set(start["asset_id"].tolist()), {0, 1})
        torch.testing.assert_close(start["s"], history["s"][:, -1])
        self.assertFalse(report["production_sampling_changed"])

    def test_resource_stratification_rejects_misaligned_real_candidate(self):
        start, history, motion = fixture()
        history["o"][1, -1, 0] += 1
        replay = SimpleNamespace(sample_starts=lambda *args, **kwargs: (start, history))
        with self.assertRaisesRegex(ValueError, "endpoint o"):
            stratified_real_starts(replay, 2, motion, expected_assets=[0, 1], horizon=32)

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
