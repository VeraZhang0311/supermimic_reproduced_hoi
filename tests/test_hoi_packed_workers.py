"""CPU packing/IPC contracts; fake children never initialize a simulator."""
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch

from hoi.v1.packed_workers import (PackedWorkerGroup, CollectorShutdownError, _add_failure_note,
                                   _record_audit_failure, shutdown_ack_budget)
from hoi.v1.workers import WorkerGroup, WorkerSpec, _cat, pack_worker_specs, resource_preflight


def specs():
    return [WorkerSpec(a, n, n * 3, str(a)) for a, n in zip((2, 9, 14, 27, 59), (2, 1, 4, 3, 1))]


def delayed_exit_child(pipe):
    """Real CPU child: native-close ACK precedes interpreter exit."""
    pipe.send(("ready",))
    assert pipe.recv() == ("close",)
    pipe.send(("closed",))
    time.sleep(3.)
    pipe.close()


def snapshot(pack, frames=None):
    assets = torch.cat([torch.full((s.num_envs,), s.asset_id) for s in pack])
    return {"asset_id": assets, "f": assets * 100 if frames is None else frames.clone()}


class FakePipe:
    def __init__(self, messages=()):
        self.messages, self.sent, self.polls, self.closed = list(messages), [], [], False

    def send(self, message):
        self.sent.append(message)
        if message == ("close",):
            self.messages.append(("closed",))

    def poll(self, timeout):
        self.polls.append(timeout)
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.parents, self.jobs, self.events = [], [], []

    def Pipe(self):
        parent = FakePipe()
        self.parents.append(parent)
        return parent, FakePipe()

    def Process(self, *, target, args, name):
        index = len(self.jobs)
        self.jobs.append((target, args, name))
        return FakeProcess(index, self.events)


class FakeProcess:
    def __init__(self, index, events):
        self.index, self.events = index, events
        self.pid = self.exitcode = None
        self.alive = False
        self.ignore_terminate = self.ignore_kill = self.ignore_join = False
        self.join_error = None
    def start(self):
        self.pid, self.alive = 1000 + self.index, True
        self.events.append(("start", self.index))
    def join(self, **kwargs):
        self.events.append(("join", self.index))
        if self.join_error:
            raise self.join_error
        if self.alive and not self.ignore_join:
            self.alive, self.exitcode = False, 0
    def is_alive(self): return self.alive
    def terminate(self):
        self.events.append(("terminate", self.index))
        if not self.ignore_terminate:
            self.alive, self.exitcode = False, -15
    def kill(self):
        self.events.append(("kill", self.index))
        if not self.ignore_kill:
            self.alive, self.exitcode = False, -9


def fake_receive(context, *, optional_all=False):
    def receive(group, index, timeout=None, initializing=False):
        pack = group.packs[index] if isinstance(group, PackedWorkerGroup) else (group.specs[index],)
        parent = context.parents[index]
        if not parent.sent:
            context.events.append(("ready", index))
            reports = [{"spec": asdict(s), "asset_id": s.asset_id} for s in pack]
            return "ready", snapshot(pack), reports if isinstance(group, PackedWorkerGroup) else reports[0]
        message = parent.sent[-1]
        if message[0] == "reset":
            frames = message[1] if isinstance(group, PackedWorkerGroup) else message[2]
            return "reset", snapshot(pack, frames)
        if message[0] != "step":
            raise AssertionError(message[0])
        pieces = message[1].split([s.num_envs for s in pack])
        transitions = []
        for spec, action in zip(pack, pieces):
            tr = {"a": action.clone(), **snapshot((spec,)), "terms": {"reward": action[:, 0].clone()}}
            if optional_all or spec.asset_id != 14:
                tr.update(ep_len=torch.tensor([float(spec.asset_id)]), ep_ret_norm=torch.tensor([.5]))
            transitions.append(tr)
        return "step", _cat(transitions, "cpu"), snapshot(pack)
    return receive


class PackedWorkerTests(unittest.TestCase):
    def test_packing_preserves_every_spec_quota_and_row_order(self):
        original = specs()
        for size, lengths in ((1, [1] * 5), (2, [2, 2, 1]), (8, [5])):
            packs = pack_worker_specs(original, size)
            self.assertEqual([len(p) for p in packs], lengths)
            flattened = [s for p in packs for s in p]
            self.assertEqual(flattened, original)
            self.assertTrue(all(a is b for a, b in zip(flattened, original)))
            self.assertEqual(sum(s.num_envs for s in flattened), 11)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                pack_worker_specs(original, value)
        for invalid in ([], original + original[:1]):
            with self.assertRaises(ValueError):
                pack_worker_specs(invalid, 2)

    def test_resource_assignment_moves_whole_packs_and_process_capacity_counts_packs(self):
        report = resource_preflight(specs(), device="cpu", collector_devices=["cuda:1", "cuda:3"],
                                    assets_per_process=2, worker_startup_concurrency=9, max_processes=3)
        self.assertEqual((report["groups"], report["processes"], report["num_envs"]), (5, 3, 11))
        self.assertEqual(report["workers"], [asdict(s) for s in specs()])
        self.assertEqual([r["device"] for r in report["worker_device_assignment"]],
                         ["cuda:1", "cuda:1", "cuda:3", "cuda:3", "cuda:1"])
        self.assertEqual(report["worker_startup_concurrency"], 3)
        with self.assertRaisesRegex(ValueError, "process capacity"):
            resource_preflight(specs(), device="cpu", assets_per_process=2, max_processes=2)

    def test_factory_packs_processes_but_expands_reports_and_preserves_step_reset_rows(self):
        context = FakeContext()
        config = {"assets_per_process": 2, "device": "cuda:0", "collector_devices": ["cuda:1", "cuda:3"],
                  "worker_startup_concurrency": 2}
        with patch("hoi.v1.packed_workers.mp.get_context", return_value=context), \
                patch.object(PackedWorkerGroup, "_receive", fake_receive(context)):
            group = WorkerGroup(specs(), config, device="cpu")
            self.assertIsInstance(group, PackedWorkerGroup)
            self.assertEqual(context.events, [("start", 0), ("start", 1), ("ready", 0), ("ready", 1),
                                              ("start", 2), ("ready", 2)])
            self.assertEqual([job[1][1] for job in context.jobs], pack_worker_specs(specs(), 2))
            self.assertEqual([job[1][2]["device"] for job in context.jobs], ["cuda:1", "cuda:3", "cuda:1"])
            self.assertEqual(config["device"], "cuda:0")
            self.assertEqual([r["asset_id"] for r in group.native_reports], [s.asset_id for s in specs()])
            expected_rows = [2, 2, 9, 14, 14, 14, 14, 27, 27, 27, 59]
            self.assertEqual(group.snapshot()["asset_id"].tolist(), expected_rows)
            actions = torch.arange(11 * 29, dtype=torch.float32).reshape(11, 29).requires_grad_()
            tr = group.step(actions)
            self.assertTrue(torch.equal(tr["a"], actions))
            self.assertFalse(tr["a"].requires_grad)
            self.assertEqual(tr["asset_id"].tolist(), expected_rows)
            self.assertTrue(torch.equal(tr["terms"]["reward"], actions[:, 0]))
            self.assertEqual([len(p.sent[-1][1]) for p in context.parents], [3, 7, 1])
            # Match baseline intersection behavior when some assets have no episode endings.
            self.assertNotIn("ep_len", tr)
            frames = torch.arange(11, dtype=torch.long) + 1000
            after = group.reset(frames)
            self.assertTrue(torch.equal(after["f"], frames))
            self.assertEqual(after["asset_id"].tolist(), expected_rows)
            self.assertTrue(torch.equal(group.snapshot()["f"], frames))
            for bad in (torch.zeros(10, 29), torch.zeros(11, 28)):
                with self.assertRaises(ValueError):
                    group.step(bad)
            with self.assertRaises(ValueError):
                group.reset(torch.zeros(10))
            group.close()
            self.assertTrue(all(p.closed for p in context.parents))

    def test_optional_episode_rows_match_baseline_for_any_pack_size(self):
        outputs = []
        for size in (1, 2, 3, 5):
            context = FakeContext()
            cls = WorkerGroup if size == 1 else PackedWorkerGroup
            with patch("hoi.v1.workers.mp.get_context", return_value=context), \
                    patch.object(cls, "_receive", fake_receive(context, optional_all=True)):
                group = WorkerGroup(specs(), {"assets_per_process": size}, device="cpu")
                if size == 1:
                    self.assertIs(type(group), WorkerGroup)
                    self.assertEqual(len(context.jobs), 5)
                outputs.append(group.step(torch.zeros(11, 29)))
                group.close()
        for output in outputs:
            self.assertEqual(output["ep_len"].tolist(), [2., 9., 14., 27., 59.])
            self.assertTrue(torch.equal(output["ep_ret_norm"], outputs[0]["ep_ret_norm"]))
            self.assertTrue(torch.equal(output["asset_id"], outputs[0]["asset_id"]))

    def test_omitted_pack_option_uses_existing_single_asset_child(self):
        context = FakeContext()
        with patch("hoi.v1.workers.mp.get_context", return_value=context), \
                patch.object(WorkerGroup, "_receive", fake_receive(context)):
            group = WorkerGroup(specs(), {}, device="cpu")
            self.assertIs(type(group), WorkerGroup)
            self.assertEqual([job[1][1] for job in context.jobs], specs())
            self.assertTrue(all(job[0].__name__ == "_worker_main" for job in context.jobs))
            self.assertEqual(len(group.native_reports), 5)
            group.close()

    def test_receive_progress_timeout_and_child_errors_do_not_infer_state(self):
        group = object.__new__(PackedWorkerGroup)
        group.timeout = 7
        pipe = FakePipe([("asset_ready", {"asset_id": 2}), ("asset_ready", {"asset_id": 9}),
                         ("ready", {"f": torch.tensor([1])}, [])])
        group._pipes = [pipe]
        self.assertEqual(group._receive(0, timeout=17, initializing=True)[0], "ready")
        self.assertEqual(pipe.polls, [17, 17, 17])
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            group._receive(0)
        self.assertEqual(pipe.polls[-1], 7)
        pipe.messages.append(("error", "known child failure"))
        with self.assertRaisesRegex(RuntimeError, "known child failure"):
            group._receive(0)

    def test_startup_failure_closes_started_pack_processes(self):
        context = FakeContext()
        def fail(group, index, **kwargs):
            raise TimeoutError("fixture startup timeout")
        with patch("hoi.v1.packed_workers.mp.get_context", return_value=context), \
                patch.object(PackedWorkerGroup, "_receive", fail):
            with self.assertRaisesRegex(TimeoutError, "fixture startup timeout"):
                WorkerGroup(specs(), {"assets_per_process": 2, "worker_startup_concurrency": 2}, device="cpu")
        self.assertEqual(len(context.jobs), 2)
        self.assertTrue(all(p.closed and not p.sent for p in context.parents))
        self.assertIn(("join", 0), context.events)
        self.assertIn(("join", 1), context.events)

    def shutdown_fixture(self, count=2):
        group = object.__new__(PackedWorkerGroup)
        events = []
        processes = [FakeProcess(i, events) for i in range(count)]
        for process in processes:
            process.start()
        group._processes = processes.copy()
        group._pipes = [FakePipe() for _ in processes]
        group._child_pipes = []
        group.shutdown_ack_timeout = .03
        group.shutdown_join_timeout = group.shutdown_terminate_timeout = group.shutdown_kill_timeout = .01
        return group, processes, events

    def test_normal_shutdown_drains_pending_step_then_requires_ack_and_zero_exit(self):
        group, processes, events = self.shutdown_fixture()
        group._pipes[0].messages.append(("step", {"f": torch.tensor([1])}, {}))
        pipes = group._pipes.copy()
        group.close()
        self.assertTrue(all(pipe.closed and not pipe.messages for pipe in pipes))
        self.assertTrue(all(process.exitcode == 0 for process in processes))
        self.assertFalse(any(event[0] in ("kill", "terminate") for event in events))
        self.assertEqual(group._processes, [])

    def test_real_child_may_exit_more_than_two_seconds_after_closed_ack(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=delayed_exit_child, args=(child,))
        group = object.__new__(PackedWorkerGroup)
        group._processes, group._pipes, group._child_pipes = [process], [parent], []
        group.packs = [(None,) * 4]
        try:
            process.start()
            child.close()
            self.assertTrue(parent.poll(30.))
            self.assertEqual(parent.recv(), ("ready",))
            group.close()
            self.assertEqual(process.exitcode, 0)
            self.assertFalse(process.is_alive())
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.)
            parent.close()
            child.close()

    def test_remote_shutdown_error_is_reported_and_other_processes_are_joined(self):
        group, processes, events = self.shutdown_fixture()
        group._pipes[0].messages.append(("error", "ORIGINAL native slot shutdown error"))
        pipes = group._pipes.copy()
        with self.assertRaisesRegex(CollectorShutdownError, "ORIGINAL native slot shutdown error") as caught:
            group.close()
        self.assertEqual(caught.exception.details[0]["worker"], 0)
        self.assertIn(("join", 1), events)
        self.assertTrue(all(pipe.closed for pipe in pipes))
        self.assertEqual(group._processes, [])

    def test_missing_ack_and_late_nonzero_exit_cannot_pass(self):
        group, processes, _ = self.shutdown_fixture()
        group._pipes[0].send = lambda message: None
        processes[1].alive, processes[1].exitcode = False, 1
        with self.assertRaises(CollectorShutdownError) as caught:
            group.close()
        self.assertEqual({error["operation"] for error in caught.exception.details}, {"closed_ack", "exitcode"})
        self.assertTrue(all(not process.alive for process in processes))

    def test_force_escalates_to_kill_and_retains_any_survivor_handles(self):
        group, processes, events = self.shutdown_fixture()
        for process in processes:
            process.ignore_join = process.ignore_terminate = True
        processes[1].ignore_kill = True
        with self.assertRaises(CollectorShutdownError) as caught:
            group.close(force=True)
        self.assertFalse(processes[0].alive)
        self.assertTrue(processes[1].alive)
        self.assertEqual(group._processes, [processes[1]])
        self.assertEqual(caught.exception.surviving_pids, [1001])
        self.assertIn(("kill", 0), events)
        self.assertIn(("kill", 1), events)
        processes[1].ignore_kill = False
        group.close(force=True)
        self.assertEqual(group._processes, [])

    def test_startup_original_error_survives_cleanup_failure_and_every_peer_is_attempted(self):
        context = FakeContext()
        created = []
        original_factory = context.Process
        def process_factory(**kwargs):
            process = original_factory(**kwargs)
            if not created:
                process.join_error = OSError("secondary cleanup join error")
            created.append(process)
            return process
        context.Process = process_factory
        with patch("hoi.v1.packed_workers.mp.get_context", return_value=context), \
             patch.object(PackedWorkerGroup, "_receive", side_effect=TimeoutError("ORIGINAL startup timeout")):
            with self.assertRaisesRegex(TimeoutError, "ORIGINAL startup timeout") as caught:
                WorkerGroup(specs(), {"assets_per_process": 2, "worker_startup_concurrency": 2}, device="cpu")
        self.assertTrue(any("secondary cleanup join error" in note for note in caught.exception.__notes__))
        self.assertTrue(caught.exception.collector_cleanup_errors)
        self.assertIn(("terminate", 1), context.events)
        self.assertIn(("join", 1), context.events)
        self.assertTrue(all(not process.alive for process in created))
        self.assertTrue(all(pipe.closed for pipe in context.parents))

    def test_started_process_is_registered_before_child_pipe_close_can_fail(self):
        context = FakeContext()
        original_pipe = context.Pipe
        def pipe_factory():
            parent, child = original_pipe()
            def fail(): raise OSError("child pipe close failure")
            child.close = fail
            return parent, child
        context.Pipe = pipe_factory
        with patch("hoi.v1.packed_workers.mp.get_context", return_value=context):
            with self.assertRaisesRegex(OSError, "child pipe close failure") as caught:
                WorkerGroup(specs(), {"assets_per_process": 2}, device="cpu")
        self.assertIn(("terminate", 0), context.events)
        self.assertIn(("join", 0), context.events)
        self.assertTrue(context.parents[0].closed)
        self.assertIn("cleanup also failed", caught.exception.__notes__[0])

    def test_blocked_pipe_io_is_bounded_and_peers_are_still_cleaned(self):
        group, processes, events = self.shutdown_fixture()
        release = threading.Event()
        group._pipes[0].send = lambda message: release.wait(10.)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(CollectorShutdownError, "IPC timed out"):
                group.close()
        finally:
            release.set()
        self.assertLess(time.monotonic() - started, 2.)
        self.assertTrue(all(not process.alive for process in processes))
        self.assertIn(("join", 1), events)

    def test_failure_note_fallback_never_replaces_error(self):
        class OldError(Exception):
            add_note = None
        original = OldError("root cause")
        _add_failure_note(original, "cleanup evidence")
        self.assertEqual(str(original), "root cause")
        self.assertEqual(original.__notes__, ["cleanup evidence"])

    def test_shutdown_grace_scales_with_largest_pack_and_is_globally_bounded(self):
        self.assertEqual(shutdown_ack_budget([(1,)]), 30.)
        self.assertEqual(shutdown_ack_budget([tuple(range(32))] * 9), 79.)
        self.assertEqual(shutdown_ack_budget([tuple(range(273))]), 180.)

    def test_late_launcher_or_ack_error_invalidates_previously_passed_audit(self):
        audit = {"passed": True, "checks": [{"unchanged": True}]}
        original = RuntimeError("late launcher cleanup failure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            _record_audit_failure(audit, path, "ORIGINAL traceback", original)
            report = json.loads(path.read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"], "ORIGINAL traceback")
        self.assertEqual(report["checks"], [{"unchanged": True}])


if __name__ == "__main__":
    unittest.main()
