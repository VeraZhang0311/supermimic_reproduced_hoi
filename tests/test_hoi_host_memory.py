import errno
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from hoi.v1.host_memory import HostMemorySampler, parse_smaps_rollup, sample_process_tree


def rollup(rss=100, pss=60, anon=40):
    return (f"Rss: {rss} kB\nPss: {pss} kB\nPss_Anon: {anon} kB\n"
            "Anonymous: 50 kB\nPrivate_Clean: 2 kB\nPrivate_Dirty: 30 kB\n"
            "Private_Hugetlb: 4 kB\nShared_Clean: 14 kB\n")


class HostMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proc = Path(self.temp.name)

    def process(self, pid, children=(), text=None):
        task = self.proc / str(pid) / "task" / str(pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / "children").write_text(" ".join(map(str, children)))
        (self.proc / str(pid) / "smaps_rollup").write_text(rollup() if text is None else text)

    def test_parse_units_and_distinct_anonymous_private_metrics(self):
        result = parse_smaps_rollup(rollup())
        self.assertEqual(result["rss_bytes"], 100 * 1024)
        self.assertEqual(result["pss_bytes"], 60 * 1024)
        self.assertEqual(result["pss_anon_bytes"], 40 * 1024)
        self.assertEqual(result["private_bytes"], 36 * 1024)
        self.assertIsNone(result["private_anon_bytes"])
        sparse = parse_smaps_rollup("Rss: 2 kB\nPss: 1 kB\n")
        self.assertIsNone(sparse["pss_anon_bytes"])
        self.assertIsNone(sparse["private_bytes"])

    def test_invalid_or_missing_required_counters_rejected(self):
        for text in ("Rss: 1 kB\n", "Rss: -1 kB\nPss: 0 kB\n",
                     "Rss: 1 MB\nPss: 1 kB\n", "Rss: x kB\nPss: 1 kB\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_smaps_rollup(text)

    def test_recursive_tree_unique_and_never_reads_unrelated_process(self):
        self.process(10, (11, 12, 11))
        self.process(11, (13,))
        self.process(12, (13,))
        self.process(13, (10,))
        self.process(99, (), "must not be read")
        original = Path.read_text
        reads = []

        def read(path, *args, **kwargs):
            reads.append(path.relative_to(self.proc).parts[0])
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            result = sample_process_tree(10, proc_root=self.proc)
        self.assertEqual(set(reads), {"10", "11", "12", "13"})
        self.assertEqual(result["measured_pids"], [10, 11, 12, 13])
        self.assertEqual(result["totals"]["pss_bytes"], 240 * 1024)
        self.assertEqual(result["totals"]["rss_bytes"], 400 * 1024)
        self.assertTrue(result["complete"])

    def test_child_exits_during_discovery_or_memory_read(self):
        self.process(10, (11, 12))
        self.process(12)
        (self.proc / "12" / "smaps_rollup").unlink()
        result = sample_process_tree(10, proc_root=self.proc)
        self.assertEqual(result["exited_pids"], [11, 12])
        self.assertEqual(result["measured_pids"], [10])
        self.assertTrue(result["complete"])
        self.assertEqual(result["errors"], [])

    def test_permission_failure_not_hidden_or_treated_as_exit(self):
        self.process(10, (11,))
        self.process(11)
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path == self.proc / "11" / "smaps_rollup":
                raise PermissionError(errno.EACCES, "denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            result = sample_process_tree(10, proc_root=self.proc)
        self.assertFalse(result["complete"])
        self.assertEqual(result["exited_pids"], [])
        self.assertEqual(result["errors"][0]["type"], "PermissionError")
        self.assertEqual(result["errors"][0]["pid"], 11)
        self.assertEqual(result["errors"][0]["operation"], "read_smaps_rollup")

    def test_children_permission_failure_marks_tree_incomplete(self):
        self.process(10)
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "children":
                raise PermissionError(errno.EACCES, "denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            result = sample_process_tree(10, proc_root=self.proc)
        self.assertFalse(result["complete"])
        self.assertEqual(result["measured_pids"], [10])
        self.assertEqual(result["errors"][0]["operation"], "read_children")

    def test_optional_counter_unknown_in_any_process_keeps_total_unknown(self):
        self.process(10, (11,))
        self.process(11, text="Rss: 5 kB\nPss: 3 kB\n")
        result = sample_process_tree(10, proc_root=self.proc)
        self.assertEqual(result["totals"]["pss_bytes"], 63 * 1024)
        self.assertIsNone(result["totals"]["pss_anon_bytes"])

    def test_peaks_preserve_timestamp_and_measured_pid_set(self):
        self.process(10)
        sampler = HostMemorySampler(10, proc_root=self.proc)
        first = sampler.sample()
        self.process(10, (11,))
        self.process(11)
        second = sampler.sample()
        self.process(10)
        sampler.sample()
        report = sampler.summary()
        peak = report["peaks"]["pss_bytes"]
        self.assertEqual(peak["bytes"], 120 * 1024)
        self.assertEqual(peak["pids"], [10, 11])
        self.assertEqual(peak["timestamp_utc"], second["timestamp_utc"])
        self.assertGreaterEqual(second["timestamp_utc"], first["timestamp_utc"])
        self.assertTrue(report["measurement_complete"])
        self.assertTrue(report["sampled_peaks_are_lower_bounds"])
        json.dumps(report, allow_nan=False)

    def test_context_background_samples_and_stops_on_body_exception(self):
        self.process(10)
        sampler = HostMemorySampler(10, interval_seconds=0.01, proc_root=self.proc)
        sampled = threading.Event()
        original = sampler.sample

        def sample():
            result = original()
            if threading.current_thread().name == "hoi-host-memory":
                sampled.set()
            return result

        with patch.object(sampler, "sample", sample):
            with self.assertRaisesRegex(RuntimeError, "workload failure"):
                with sampler:
                    self.assertTrue(sampled.wait(2.0))
                    raise RuntimeError("workload failure")
        self.assertFalse(sampler._thread.is_alive())
        self.assertGreaterEqual(sampler.summary()["sample_count"], 3)
        with self.assertRaisesRegex(RuntimeError, "twice"):
            sampler.__enter__()

    def test_final_sample_error_does_not_mask_workload_failure(self):
        self.process(10)
        sampler = HostMemorySampler(10, interval_seconds=10, proc_root=self.proc)
        original = sampler.sample
        calls = 0

        def sample():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("sample failure")
            return original()

        with patch.object(sampler, "sample", sample):
            with self.assertRaisesRegex(RuntimeError, "workload failure"):
                with sampler:
                    raise RuntimeError("workload failure")
        report = sampler.summary()
        self.assertFalse(report["measurement_complete"])
        self.assertEqual(report["sampler_errors"][0]["operation"], "final_sample")

    def test_invalid_sampler_arguments(self):
        for root in (0, -1, True, 1.5):
            with self.subTest(root=root), self.assertRaises(ValueError):
                HostMemorySampler(root)
        for interval in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                HostMemorySampler(10, interval_seconds=interval)


if __name__ == "__main__":
    unittest.main()
