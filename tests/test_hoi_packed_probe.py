"""Collector probes must preserve production allocation and report failures."""
from contextlib import ExitStack
import json
from itertools import count
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hoi.v1.motion import TrainingMotionBank
from hoi.v1.packed_probe import compare_values, run_collection, specs_from_header, summarize_collector_gpu_samples
from hoi.v1.workers import allocate_workers, WorkerSpec


def asset(index):
    return {"asset_id": index, "object_asset": f"asset-{index}", "exact_scale": 1.,
            "collision_hash": f"hash-{index}", "physical_profile": {"mass": 1.}}


def clip(n, asset_id):
    zeros = torch.zeros(n, 3)
    identity = torch.eye(3).repeat(n, 1, 1)
    return {"asset_id": asset_id, "expected_contact": torch.zeros(n, 16),
            "expected_valid": torch.ones(n, 16), "reference": {
                "robot": {"root_pos": zeros, "root_rot": identity,
                          "derived_ref_v": zeros, "derived_ref_omega": zeros,
                          "joint_pos": torch.zeros(n, 29), "derived_ref_joint_qd": torch.zeros(n, 29)},
                "object": {"pos": zeros, "rot": identity,
                           "derived_ref_v": zeros, "derived_ref_omega": zeros}}}


class FixtureKinematics:
    def forward(self, root_p, root_r, root_v, root_w, q, qd):
        return (root_p[:, None].expand(-1, 14, -1).clone(),
                root_r[:, None].expand(-1, 14, -1, -1).clone(),
                root_v[:, None].expand(-1, 14, -1).clone(),
                root_w[:, None].expand(-1, 14, -1).clone())


class FinalSampleHost:
    def __init__(self, *_):
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finished = True

    def summary(self):
        return {"final_sample_included": self.finished, "measurement_complete": True}


class FixtureGpu:
    def __init__(self):
        self.samples = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.samples.append({"elapsed_seconds": 1., "processes": [["999999", "100", "GPU-fixture"]]})


class PackedProbeTests(unittest.TestCase):
    def test_header_bins_match_actual_bank_at_outgoing_interval_boundaries(self):
        lengths = [2, 50, 51, 52, 100, 101]
        clips = [clip(n, i) for i, n in enumerate(lengths)]
        # Extra clip on asset zero tests aggregation, and asset six is inactive.
        clips.append(clip(52, 0))
        bundle = {"assets": [asset(i) for i in range(7)], "clips": [
            {"asset_id": c["asset_id"], "state_frames": len(c["reference"]["robot"]["root_pos"])}
            for c in clips]}
        bank = TrainingMotionBank(clips, FixtureKinematics(), partition="train")
        self.assertEqual(bank.num_bins, 11)
        for budget in (6, 17, 2048, 4096):
            with self.subTest(budget=budget):
                actual = specs_from_header(bundle, budget)
                self.assertEqual(actual, allocate_workers(bundle, bank, budget))
                self.assertEqual([s.train_bins for s in actual], [3, 1, 1, 2, 2, 2])
                self.assertEqual(sum(s.num_envs for s in actual), budget)
                self.assertEqual([s.asset_id for s in actual], list(range(6)))

    def test_header_rejects_invalid_state_count_and_insufficient_envs(self):
        bundle = {"assets": [asset(0)], "clips": [{"asset_id": 0, "state_frames": 2}]}
        for n in (0, 1, -1, 2.0, True, "52"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                specs_from_header({**bundle, "clips": [{"asset_id": 0, "state_frames": n}]}, 8)
        with self.assertRaises(ValueError):
            specs_from_header(bundle, 0)
        bundle = {"assets": [asset(0), asset(1)], "clips": [
            {"asset_id": 0, "state_frames": 2}, {"asset_id": 1, "state_frames": 2}]}
        with self.assertRaisesRegex(ValueError, "cannot each receive"):
            specs_from_header(bundle, 1)

    def test_compare_rejects_tensor_dtype_shape_value_and_nested_key_changes(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        left = [{"transition": {"s": tensor}, "reset": (torch.tensor([2]),)}]
        compare_values(left, [{"transition": {"s": tensor.clone()}, "reset": (torch.tensor([2]),)}])
        for changed in (tensor.double(), tensor.reshape(3, 2), tensor + 1):
            with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, r"root\[0\].transition.s"):
                compare_values(left, [{"transition": {"s": changed}, "reset": (torch.tensor([2]),)}])
        with self.assertRaisesRegex(AssertionError, "keys changed"):
            compare_values(left, [{"transition": {"s": tensor, "new": 1}, "reset": (torch.tensor([2]),)}])
        with self.assertRaisesRegex(AssertionError, "length changed"):
            compare_values(left, [])
        with self.assertRaises(AssertionError):
            compare_values(torch.tensor([float("nan")]), torch.tensor([float("nan")]))

    def test_compare_rejects_container_schema_changes(self):
        for left, right in (({}, []), ([], {}), ([1], (1,))):
            with self.subTest(left=left, right=right), self.assertRaises(AssertionError):
                compare_values(left, right)

    def run_fixture(self, *, constructor_error=None, step_error=None, close_error=None, parity=True,
                    pids=(999999,), devices=("cuda:0",), gpu_samples=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / "probe"
        worker = MagicMock()
        worker.N = 2
        worker._processes = [SimpleNamespace(pid=pid) for pid in pids]
        worker.snapshot.return_value = {"f": torch.tensor([0, 0])}
        worker.step.return_value = {"s": torch.ones(2, 4)}
        worker.step.side_effect = step_error
        worker.close.side_effect = close_error
        with ExitStack() as stack:
            stack.enter_context(patch("hoi.v1.packed_probe.time.monotonic", side_effect=count(1)))
            stack.enter_context(patch("hoi.v1.packed_probe.WorkerGroup", return_value=worker,
                                      side_effect=constructor_error))
            stack.enter_context(patch("hoi.v1.packed_probe.memory_snapshot", return_value={"mocked": True}))
            stack.enter_context(patch("hoi.v1.host_memory.HostMemorySampler", FinalSampleHost))
            if gpu_samples is None:
                stack.enter_context(patch("hoi.v1.resource_probe._GpuSampler", FixtureGpu))
            else:
                sampler = FixtureGpu()
                sampler.samples = gpu_samples
                stack.enter_context(patch.object(FixtureGpu, "__exit__", return_value=None))
                stack.enter_context(patch("hoi.v1.resource_probe._GpuSampler", return_value=sampler))
            try:
                result = run_collection([WorkerSpec(0, 2, 1, "fixture")],
                                        {"assets_per_process": 1, "collector_devices": list(devices)},
                                        steps=2, output=output, parity=parity)
                error = None
            except BaseException as exc:
                result, error = None, exc
        report = json.loads((output / "report.json").read_text())
        return result, error, report, worker, output

    def test_success_has_artifact_complete_ticks_zero_training_and_cleanup(self):
        result, error, report, worker, output = self.run_fixture()
        self.assertIsNone(error)
        self.assertTrue(report["passed"])
        self.assertEqual(report["phase"], "completed")
        self.assertEqual(report["completed_ticks"], 2)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["formal_training_started"])
        self.assertEqual(len(result[0]), 3)
        self.assertTrue((output / "trace.pt").is_file())
        self.assertGreater(report["stepping_seconds"], 0)
        worker.close.assert_called_once()

    def test_constructor_or_collection_failure_never_reports_passed(self):
        for errors in ({"constructor_error": RuntimeError("startup failed")},
                       {"step_error": RuntimeError("step failed")}):
            with self.subTest(errors=errors):
                _, error, report, worker, _ = self.run_fixture(**errors)
                self.assertIsInstance(error, RuntimeError)
                self.assertFalse(report["passed"])
                self.assertEqual(report["phase"], "failed")
                self.assertIn(str(error), report["error"])
                if "constructor_error" in errors:
                    worker.close.assert_not_called()
                else:
                    worker.close.assert_called_once()

    def test_physics_success_but_close_failure_reports_false(self):
        _, error, report, worker, _ = self.run_fixture(close_error=ValueError("close failed"))
        self.assertIsInstance(error, ValueError)
        self.assertEqual(report["completed_ticks"], 2)
        self.assertFalse(report["passed"])
        self.assertIn("close failed", report["shutdown_error"])
        worker.close.assert_called_once()

    def test_original_collection_exception_survives_secondary_close_failure(self):
        _, error, report, _, _ = self.run_fixture(step_error=RuntimeError("first failure"),
                                                close_error=ValueError("close failed"))
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("first failure", report["error"])
        self.assertIn("close failed", report["shutdown_error"])
        self.assertFalse(report["passed"])

    def test_report_includes_sampler_exit_final_samples(self):
        _, error, report, _, _ = self.run_fixture(parity=False)
        self.assertIsNone(error)
        self.assertTrue(report["host_memory"]["final_sample_included"])
        self.assertEqual(len(report["gpu_memory"]["samples"]), 1)

    def test_successful_collection_with_incomplete_resource_sample_is_failed(self):
        with patch.object(FinalSampleHost, "summary", return_value={"measurement_complete": False}):
            _, error, report, _, _ = self.run_fixture()
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(report["passed"])
        self.assertFalse(report["resource_measurement_complete"])
        self.assertEqual(report["completed_ticks"], 2)
        with patch.object(FixtureGpu, "__exit__", return_value=None):
            _, error, report, _, _ = self.run_fixture()
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(report["passed"])
        self.assertFalse(report["resource_measurement_complete"])

    def test_gpu_coverage_accepts_cpu_parent_and_fewer_collectors_than_devices(self):
        samples = [{"elapsed_seconds": 1., "processes": [["21", "100", "GPU-A"],
                    ["22", "120", "GPU-B"], ["90", "999", "GPU-unrelated"]]}]
        report = summarize_collector_gpu_samples(samples, [21, 22], ["cuda:0", "cuda:1", "cuda:1", "cuda:2"], 20)
        self.assertTrue(report["measurement_complete"])
        self.assertTrue(report["assigned_device_uuid_verified"])
        self.assertFalse(report["parent_gpu_required"])
        self.assertEqual(report["expected_collector_gpu_count"], 2)
        self.assertEqual(report["observed_collector_gpu_count"], 2)
        self.assertEqual(report["missing_collector_pids"], [])
        self.assertNotIn("GPU-unrelated", report["observed_collector_gpu_uuids"])

    def test_missing_collector_or_gpu_and_intermittent_errors_never_pass(self):
        good = {"elapsed_seconds": 1., "processes": [["21", "100", "GPU-A"], ["22", "120", "GPU-B"]]}
        cases = [
            ([{"elapsed_seconds": 1., "processes": [["21", "100", "GPU-A"]]}], [22], 1),
            ([{"elapsed_seconds": 1., "processes": [["21", "100", "GPU-A"], ["22", "120", "GPU-A"]]}], [], 1),
            ([good, {"elapsed_seconds": 2., "processes_error": "timeout", "gpus_error": "query failed"}], [], 2),
        ]
        for samples, missing, gpu_count in cases:
            with self.subTest(samples=samples):
                _, error, report, _, _ = self.run_fixture(pids=(21, 22), devices=("cuda:0", "cuda:1"), gpu_samples=samples)
                self.assertIsInstance(error, RuntimeError)
                self.assertFalse(report["passed"])
                self.assertFalse(report["resource_measurement_complete"])
                gpu = report["gpu_memory"]
                self.assertEqual(gpu["missing_collector_pids"], missing)
                self.assertEqual(gpu["expected_collector_gpu_count"], 2)
                self.assertEqual(gpu["observed_collector_gpu_count"], gpu_count)
                if len(samples) == 2:
                    self.assertEqual(gpu["samples"][1]["processes_error"], "timeout")
                    self.assertEqual(gpu["samples"][1]["gpus_error"], "query failed")
                    self.assertEqual(len(gpu["sampling_errors"]), 2)

    def test_every_collector_needs_numeric_memory_not_only_pid_and_uuid(self):
        for invalid in ("N/A", "nan", "inf", "-inf", "-1", "-0.5", "invalid"):
            with self.subTest(invalid=invalid):
                samples = [{"elapsed_seconds": 1., "processes": [
                    ["20", "1000", "GPU-A"], ["21", "100", "GPU-A"], ["22", invalid, "GPU-B"]]}]
                report = summarize_collector_gpu_samples(samples, [21, 22], ["cuda:0", "cuda:1"], 20)
                self.assertTrue(report["assigned_device_uuid_verified"])
                self.assertEqual(report["missing_collector_pids"], [])
                self.assertEqual(report["measured_collector_pids"], [21])
                self.assertEqual(report["missing_collector_memory_pids"], [22])
                self.assertFalse(report["measurement_complete"])

    def test_later_valid_zero_memory_counts_and_preserves_normal_peak_sums(self):
        samples = [{"elapsed_seconds": 1., "processes": [
            ["21", "100", "GPU-A"], ["22", "N/A", "GPU-B"]]},
            {"elapsed_seconds": 2., "processes": [
                ["21", "120", "GPU-A"], ["22", "0", "GPU-B"]]}]
        report = summarize_collector_gpu_samples(samples, [21, 22], ["cuda:0", "cuda:1"], 20)
        self.assertTrue(report["measurement_complete"])
        self.assertEqual(report["measured_collector_pids"], [21, 22])
        self.assertEqual(report["missing_collector_memory_pids"], [])
        self.assertEqual(report["owned_process_sampled_peak_MiB_by_gpu_uuid"], {"GPU-A": 120, "GPU-B": 0})

    def test_collection_with_only_unavailable_collector_memory_fails_closed(self):
        _, error, report, _, _ = self.run_fixture(gpu_samples=[{
            "elapsed_seconds": 1., "processes": [["999999", "N/A", "GPU-fixture"]]}])
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(report["passed"])
        self.assertEqual(report["gpu_memory"]["missing_collector_memory_pids"], [999999])


if __name__ == "__main__":
    unittest.main()
