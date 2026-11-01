"""Capacity checks must preserve production sizes and use real aligned inputs."""
from dataclasses import replace
from contextlib import ExitStack
from types import SimpleNamespace
from pathlib import Path
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import unittest

import torch

from hoi.v1.resource_probe import stratified_real_starts, read_probe_config, build_probe_models, initialize_probe_models, ProbeConfig, deployment_comparison, validate_real_start_mix, full_active_wm_windows, summarize_gpu_samples, initialize_cuda_memory_counters, finalize_resource_report


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

    def test_2048_comparison_changes_only_collector_and_replay_width(self):
        config = read_probe_config()
        original = config.to_dict()
        for count in (2048, 4096):
            probe = ProbeConfig(num_envs=count, expected_groups=273).validate()
            report = deployment_comparison(probe, config)
            self.assertEqual((probe.buffer_steps, probe.pi_batch, probe.horizon), (1000, 4096, 32))
            self.assertEqual(report["explicit_overrides"], {"num_envs": 2048} if count == 2048 else {})
            self.assertEqual(report["production_readiness_compatible"], count == 4096)
            self.assertEqual(report["measured_num_envs"], count)
            self.assertEqual(report["production_num_envs"], 4096)
            for key in ("data_selection_changed", "training_batches_changed", "replay_depth_changed",
                        "imagination_horizon_changed"):
                self.assertFalse(report[key])
            self.assertEqual(config.to_dict(), original)
        for count in (1024, 2048.0, True):
            with self.assertRaises(ValueError):
                ProbeConfig(num_envs=count).validate()
        with self.assertRaisesRegex(ValueError, "unchanged 4096"):
            deployment_comparison(ProbeConfig(num_envs=2048), replace(config, num_envs=2048))

    def test_both_environment_counts_keep_every_one_of_273_assets(self):
        from hoi.v1.workers import allocate_workers
        assets = [{"asset_id": index, "object_asset": str(index), "exact_scale": 1.,
                   "collision_hash": str(index), "physical_profile": {}} for index in range(273)]
        motion = SimpleNamespace(bin_asset=torch.arange(273).repeat_interleave(torch.arange(1, 274)))
        original = json.dumps(assets, sort_keys=True)
        for count in (2048, 4096):
            probe = ProbeConfig(num_envs=count, expected_groups=273).validate()
            specs = allocate_workers({"assets": assets}, motion, probe.num_envs)
            self.assertEqual([spec.asset_id for spec in specs], list(range(273)))
            self.assertEqual(sum(spec.num_envs for spec in specs), count)
            self.assertTrue(all(spec.num_envs > 0 for spec in specs))
            self.assertEqual([spec.train_bins for spec in specs], list(range(1, 274)))
            self.assertEqual(json.dumps(assets, sort_keys=True), original)

    def test_2048_collectors_still_measure_full_4096_imagination_batch(self):
        probe = ProbeConfig(num_envs=2048, expected_groups=273).validate()
        asset_ids = torch.arange(273).repeat_interleave(100)
        motion = SimpleNamespace(asset_ids=asset_ids, clip_id=asset_ids,
                                 clip_end=torch.arange(273) * 100 + 99)
        class Replay:
            # Fixture models 2048 environment streams; replacement sampling
            # supplies the unchanged 4096 actor rows, including every asset.
            N = probe.num_envs
            def sample_starts(self, batch, hist, *, horizon, motion):
                assert (self.N, batch, hist, horizon) == (2048, 4096, 10, 32)
                ids = torch.arange(batch) % 273
                start = {"asset_id": ids, "f": ids * 100 + 40}
                history = {"asset_id": ids[:, None].expand(batch, 10).clone(),
                           "f": start["f"][:, None] - torch.arange(9, -1, -1)}
                for key in ("s", "o", "a_prev", "xi", "xi_o"):
                    start[key] = torch.arange(batch).float()[:, None]
                    history[key] = start[key][:, None].expand(batch, 10, 1).clone()
                return start, history
        start, history, report = stratified_real_starts(Replay(), probe.pi_batch, motion,
            expected_assets=list(range(273)), horizon=probe.horizon)
        self.assertEqual(start["f"].shape, (4096,))
        self.assertEqual(history["f"].shape, (4096, 10))
        self.assertEqual(set(start["asset_id"].tolist()), set(range(273)))
        self.assertEqual(report["final_batch_size"], 4096)
        self.assertFalse(report["production_sampling_changed"])

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

    def test_full_resource_report_requires_learner_and_collector_gpu_samples(self):
        for rows, host_complete, expected in (
            ([["10", "1000", "GPU-A"], ["11", "500", "GPU-A"]], True, True),
            ([["11", "500", "GPU-A"]], True, False),
            ([["10", "1000", "GPU-A"]], True, False),
            ([["10", "1000", "GPU-A"], ["11", "500", "GPU-A"]], False, False),
        ):
            with self.subTest(rows=rows, host_complete=host_complete):
                report = {"passed": True, "process_ids": [10, 11]}
                host = MagicMock()
                host.summary.return_value = {"measurement_complete": host_complete}
                finalize_resource_report(report, workers=None,
                    sampler=SimpleNamespace(samples=[{"elapsed_seconds": 1., "processes": rows}]),
                    host_memory=host, collector_devices=["cuda:0"], parent_pid=10)
                self.assertEqual(report["passed"], expected)
                self.assertEqual(report["resource_measurement_complete"], expected)
                self.assertEqual(report["nvidia_smi"]["collector_pids"], [11])
                self.assertTrue(report["nvidia_smi"]["learner_gpu_required"])

    def test_finalize_resource_report_retains_all_cleanup_failures(self):
        worker, host = MagicMock(), MagicMock()
        worker.close.side_effect = OSError("close failure")
        host.summary.side_effect = ValueError("host summary failure")
        report = {"passed": True, "process_ids": [10, 11], "error": {"message": "original"}}
        with patch("hoi.v1.packed_probe.summarize_collector_gpu_samples", side_effect=ValueError("gpu summary failure")) as summarize:
            finalize_resource_report(report, workers=worker, sampler=SimpleNamespace(samples=[]),
                                     host_memory=host, collector_devices=["cuda:0"], parent_pid=10)
        summarize.assert_called_once_with(samples=[], collector_pids=[11], collector_devices=["cuda:0"], parent_pid=10)
        self.assertEqual(report["error"]["message"], "original")
        self.assertFalse(report["passed"])
        self.assertEqual([entry["operation"] for entry in report["finalization_errors"]],
                         ["workers.close", "summarize_gpu_samples", "summarize_host_memory"])

    def test_learner_requires_its_own_valid_memory_measurement(self):
        for value, expected in (("N/A", False), ("nan", False), ("inf", False),
                                ("-1", False), ("0", True), ("1024", True)):
            with self.subTest(memory=value):
                report = {"passed": True, "process_ids": [10, 11]}
                host = MagicMock(); host.summary.return_value = {"measurement_complete": True}
                finalize_resource_report(report, workers=None,
                    sampler=SimpleNamespace(samples=[{"elapsed_seconds": 1., "processes": [
                        ["10", value, "GPU-A"], ["11", "500", "GPU-B"]]}]),
                    host_memory=host, collector_devices=["cuda:1"], parent_pid=10)
                self.assertTrue(report["nvidia_smi"]["learner_gpu_observed"])
                self.assertEqual(report["nvidia_smi"]["learner_memory_observed"], expected)
                self.assertEqual(report["passed"], expected)

    def test_learner_later_numeric_sample_recovers_unavailable_memory(self):
        report = {"passed": True, "process_ids": [10, 11]}
        host = MagicMock(); host.summary.return_value = {"measurement_complete": True}
        samples = [{"elapsed_seconds": 1., "processes": [["10", "N/A", "GPU-A"], ["11", "500", "GPU-B"]]},
                   {"elapsed_seconds": 2., "processes": [["10", "1024", "GPU-A"], ["11", "600", "GPU-B"]]}]
        finalize_resource_report(report, workers=None, sampler=SimpleNamespace(samples=samples),
                                 host_memory=host, collector_devices=["cuda:1"], parent_pid=10)
        self.assertTrue(report["passed"])
        self.assertEqual(report["nvidia_smi"]["owned_process_sampled_peak_MiB_by_gpu_uuid"],
                         {"GPU-A": 1024, "GPU-B": 600})

    def test_main_finally_writes_report_and_preserves_collection_error_despite_cleanup_errors(self):
        """Exercise main's actual finally after a constructed worker fails."""
        from hoi.v1.resource_probe import main
        from hoi.v1.packed_probe import summarize_collector_gpu_samples
        worker, learner, host, gpu = (MagicMock() for _ in range(4))
        child_pid, parent_pid = 999999, os.getpid()
        worker._processes = [SimpleNamespace(pid=child_pid)]
        worker.native_reports = []
        worker.close.side_effect = OSError("secondary close failure")
        original = RuntimeError("primary collection failure")
        learner.action.side_effect = original
        host.summary.return_value = {"measurement_complete": True}
        gpu.samples = [{"elapsed_seconds": 1., "processes": [
            [str(parent_pid), "1000", "GPU-fixture"], [str(child_pid), "500", "GPU-fixture"]]}]
        replay = SimpleNamespace(data={"fixture": torch.zeros(1)})
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            return real_tensor(*args, **{**kwargs, "device": "cpu"})

        for summary_fails in (False, True):
            with self.subTest(summary_fails=summary_fails), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory) / "probe"
                replacements = {
                    "hoi.v1.evaluate._read_bundle_header": {"partition": "train", "scope": {"is_subset": False}},
                    "hoi.v1.config.bundle_identity": {}, "hoi.v1.config.source_identity": {},
                    "hoi.v1.resource_probe._GpuSampler": gpu,
                    "hoi.v1.host_memory.HostMemorySampler": host,
                    "hoi.v1.resource_probe.initialize_cuda_memory_counters": None,
                    "supermimic.kinematics.G1Kinematics": MagicMock(),
                    "hoi.v1.prepare.load_bundle": (MagicMock(), MagicMock(), {}),
                    "hoi.v1.mdp.HOIMDP": MagicMock(),
                    "hoi.v1.resource_probe.build_probe_models": [],
                    "hoi.v1.buffer.ReplayBuffer": replay,
                    "hoi.v1.learner.Learner": learner,
                    "hoi.v1.workers.allocate_workers": [SimpleNamespace(asset_id=0)],
                    "hoi.v1.workers.resource_preflight": {},
                    "hoi.v1.workers.WorkerGroup": worker,
                    "torch.cuda.is_available": True, "torch.cuda.mem_get_info": (1000, 2000),
                    "torch.cuda.synchronize": None, "torch.cuda.memory_allocated": 0,
                    "torch.cuda.memory_reserved": 0, "torch.cuda.max_memory_allocated": 0,
                    "torch.cuda.max_memory_reserved": 0, "torch.manual_seed": None,
                    "torch.set_num_threads": None,
                }
                for target, value in replacements.items():
                    stack.enter_context(patch(target, return_value=value))
                stack.enter_context(patch("torch.tensor", side_effect=cpu_tensor))
                stack.enter_context(patch("supermimic.networks.AMP_DTYPE"))
                summarize = stack.enter_context(patch("hoi.v1.packed_probe.summarize_collector_gpu_samples",
                    side_effect=ValueError("secondary gpu summary failure") if summary_fails else summarize_collector_gpu_samples))
                with self.assertRaises(RuntimeError) as caught:
                    main(["--bundle", "fixture.json", "--output", str(output), "--from-scratch",
                          "--expected-groups", "1", "--device", "cuda:0"])
                self.assertIs(caught.exception, original)
                summarize.assert_called_once_with(samples=gpu.samples, collector_pids=[child_pid],
                                                  collector_devices=["cuda:0"], parent_pid=parent_pid)
                saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
                self.assertFalse(saved["passed"])
                self.assertEqual(saved["error"]["message"], "primary collection failure")
                operations = [entry["operation"] for entry in saved["finalization_errors"]]
                self.assertIn("workers.close", operations)
                self.assertEqual("summarize_gpu_samples" in operations, summary_fails)


if __name__ == "__main__":
    unittest.main()
