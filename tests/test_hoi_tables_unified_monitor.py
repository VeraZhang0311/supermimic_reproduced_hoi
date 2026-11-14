"""Single-writer stream merge tests, without SDK, scheduler, GPU or network."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from hoi_tables import unified_monitor as monitor
from hoi_tables.experiments import TABLE_ASSETS, ULTRA_ASSETS
from hoi_tables.telemetry import exclusive_writer


class Run:
    def __init__(self, sdk, kwargs):
        self.sdk = sdk
        self.id, self.entity, self.project = kwargs["id"], kwargs["entity"], kwargs["project"]
        self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"
        self.summary = sdk.summary
        self.step = max((r["_step"] for r in sdk.history), default=-1)+1

    def define_metric(self, name, **kwargs): self.sdk.axes.append((name, kwargs))

    def log(self, values, *, commit):
        # No explicit scientific iteration may be passed as SDK step.
        assert commit
        self.sdk.history.append({**values, "_step": self.step})
        self.sdk.new_logs.append({**values, "_step": self.step})
        self.step += 1
        if self.sdk.fail_after_commit:
            self.sdk.fail_after_commit = False
            raise RuntimeError("connection interrupted after server commit")

    def finish(self, *, exit_code): self.sdk.finishes.append(exit_code)


class SDK:
    def __init__(self):
        self.history, self.new_logs, self.axes, self.calls, self.finishes = [], [], [], [], []
        self.summary, self.fail_after_commit = {}, False

    def Settings(self, **kwargs): return kwargs

    def init(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["resume"] == "must"
        return Run(self, kwargs)

    def Api(self, **kwargs):
        def run(path):
            def scan_history(min_step=0, page_size=1000):
                return iter([dict(row) for row in self.history if row["_step"] >= min_step])
            return SimpleNamespace(scan_history=scan_history)
        return SimpleNamespace(run=run)


class Controller:
    def __init__(self, root, *, failed=False, finished=False):
        self.failed = failed
        self.state = {"evaluations": {}, "finished": finished, "missing_milestones": [], "failed_milestones": []}
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def add(self, iteration=2000, assets=2):
        path = self.root/f"eval-{iteration}.json"
        payload = {"eval/iteration": iteration, "eval/success_rate": .4, "eval/mean_normalized_return": .5,
                   "eval/clip_count": 22 if assets == 2 else 80}
        path.write_text(json.dumps(payload))
        request = {"iteration": iteration, "checkpoint_sha256": {f"rank_{i:02d}": f"{i:064x}" for i in range(assets)}}
        self.state["evaluations"][str(iteration)] = {"status": "complete", "result": str(path),
                    "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "request": request}

    def tick(self):
        if self.failed: raise RuntimeError("evaluation job failed")
        return self.state

    def _request(self, entry): return entry["request"]

    def validate_result(self, path, request):
        result = json.loads(Path(path).read_text())
        assert result["eval/iteration"] == request["iteration"]
        return result

    def _parent_exitcode(self): return None


def row(iteration, value=1.):
    return {"iteration": iteration, "stage": "wm_only" if iteration <= 500 else "actor_critic",
            "wm_total": value, "sim_rew": value/10, "real_steps": iteration*8, "resets": 1}


class UnifiedMonitorTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/".cache/unified_monitor_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.training, self.control = self.base/"formal_03", self.base/"unified_01"
        self.training.mkdir()
        metadata = {"global_config": {"num_envs": 4096, "wm_only_iters": 500}}
        raw = json.dumps(metadata).encode()
        (self.training/"run_metadata.json").write_bytes(raw)
        self.registration = {"id": "44ad3b72", "entity": "tianzizh-university-of-pennsylvania", "project": "hoiretarget-tables",
                             "name": "tables_v1-20261107-170411", "metadata_sha256": hashlib.sha256(raw).hexdigest()}
        (self.training/"wandb_registration.json").write_text(json.dumps(self.registration))
        self.sdk = SDK()

    def write(self, rank, record):
        path = self.training/f"rank_{rank:02d}/metrics.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a") as stream: stream.write(json.dumps(record)+"\n")

    def run_once(self, **kwargs):
        return monitor.monitor(self.control, self.training, assets=("largetable", "smalltable"),
                               wandb_module=self.sdk, stop_requested=lambda: True, **kwargs)

    def test_legacy_handoff_uses_server_training_iterations_and_uploads_only_missing_events(self):
        for iteration in range(1, 6):
            self.write(0, row(iteration)); self.write(1, row(iteration, 3.))
            if iteration <= 3:
                self.sdk.history.append({"_step": iteration, "iteration": iteration, "stage": "wm_only"})
        originals = {p: p.read_bytes() for p in self.training.rglob("*") if p.is_file()}
        self.assertEqual(self.run_once(), 0)
        self.assertEqual([r["iteration"] for r in self.sdk.new_logs], [4, 5])
        self.assertEqual([r["_step"] for r in self.sdk.new_logs], [4, 5])
        self.assertEqual(self.sdk.new_logs[0]["wm_total"], 2.)
        self.assertEqual(len(self.sdk.calls), 1)
        self.assertTrue(self.sdk.calls[0]["settings"]["x_disable_meta"])
        for path, raw in originals.items(): self.assertEqual(path.read_bytes(), raw)

    def test_mixed_jsonl_publishes_asset_and_action_metrics_to_the_same_run(self):
        record = row(8050)
        record["assets"] = {asset: {"sim_rew": .15, "real_steps": 256} for asset in ULTRA_ASSETS + TABLE_ASSETS}
        record["tables_by_action/carry/failure_rate"] = .25
        (self.training / "metrics.jsonl").write_text(json.dumps(record) + "\n")
        self.assertEqual(monitor.monitor(self.control, self.training, assets=("mixed",),
                                         wandb_module=self.sdk, stop_requested=lambda: True), 0)
        event, = self.sdk.new_logs
        self.assertEqual(self.sdk.calls[0]["id"], self.registration["id"])
        self.assertEqual(event["assets/largetable/sim_rew"], .15)
        self.assertEqual(event["tables_by_action/carry/failure_rate"], .25)

    def test_late_evaluation_keeps_checkpoint_axis_and_does_not_overwrite_training_stage(self):
        self.write(0, row(2200)); self.write(1, row(2200))
        self.sdk.history.append({"_step": 2300, "iteration": 2200, "stage": "actor_critic"})
        self.sdk.summary.update(stage="actor_critic", status="running")
        controller = Controller(self.base/"eval_control"); controller.add(2000)
        self.assertEqual(self.run_once(supervisor=controller), 0)
        self.assertEqual(len(self.sdk.new_logs), 1)
        event = self.sdk.new_logs[0]
        self.assertEqual(event["_step"], 2301)
        self.assertEqual(event["eval/iteration"], 2000)
        self.assertNotIn("iteration", event)
        self.assertEqual(self.sdk.summary["stage"], "actor_critic")
        self.assertEqual(self.sdk.summary["status"], "running")
        self.assertIn(("eval/*", {"step_metric": "eval/iteration"}), self.sdk.axes)
        self.assertIn(("wm_total", {"step_metric": "iteration"}), self.sdk.axes)
        self.assertFalse(any(name == "*" for name, _ in self.sdk.axes))

    def test_restart_after_server_commit_before_local_delivery_deduplicates_by_event_id(self):
        for rank in range(2): self.write(rank, row(1))
        self.sdk.fail_after_commit = True
        self.assertEqual(self.run_once(), 1)
        self.assertEqual(len(self.sdk.history), 1)
        self.assertEqual(self.run_once(), 0)
        self.assertEqual(len(self.sdk.history), 1)
        ack = json.loads((self.control/"server_ack.json").read_text())
        self.assertIn("train:000001", ack["events"])

    def test_pending_local_outbox_without_server_ack_is_replayed(self):
        payload = monitor.aggregate_training([row(1), row(1)], ("largetable", "smalltable"))
        monitor.EventOutbox(self.control).enqueue("train:000001", "train", payload)
        for rank in range(2): self.write(rank, row(1))
        self.assertEqual(self.run_once(), 0)
        self.assertEqual(len(self.sdk.new_logs), 1)

    def test_eval_failure_stops_eval_scheduler_but_training_telemetry_continues(self):
        controller = Controller(self.base/"eval_control", failed=True)
        for rank in range(2): self.write(rank, row(1))
        self.assertEqual(self.run_once(supervisor=controller), 0)
        self.assertEqual(len(self.sdk.new_logs), 1)
        self.assertEqual(self.sdk.new_logs[0][monitor.EVENT_KIND], "train")
        self.assertEqual(self.sdk.summary["eval/status"], "failed")
        self.assertEqual(self.sdk.summary["status"], "running")

    def test_formal_writer_lock_prevents_second_sdk_process(self):
        with exclusive_writer(self.training):
            self.assertEqual(self.run_once(), 1)
        self.assertEqual(self.sdk.calls, [])

    def test_eval_scheduler_lock_prevents_overlap_with_legacy_eval_monitor(self):
        evaluation = self.base/"eval_control"
        with exclusive_writer(evaluation):
            self.assertEqual(self.run_once(eval_control=evaluation), 1)
        self.assertEqual(self.sdk.calls, [])

    def test_generic_eight_asset_project_and_supervisor_factory(self):
        registration = dict(self.registration, id="ultra-new01", project="hoiretarget-boxes", name="ultra_v1-20261107")
        (self.training/"wandb_registration.json").write_text(json.dumps(registration))
        assets = tuple(f"box_{i}" for i in range(8))
        for rank in range(8): self.write(rank, row(1, float(rank+1)))
        controller = Controller(self.base/"eval_control"); controller.add(2000, assets=8)
        calls = []
        def factory(**kwargs): calls.append(kwargs); return controller
        self.assertEqual(monitor.monitor(self.control, self.training, assets=assets, wandb_module=self.sdk,
                    supervisor_factory=factory, supervisor_kwargs={"profile": "ultra"}, stop_requested=lambda: True), 0)
        self.assertEqual(calls, [{"profile": "ultra"}])
        self.assertEqual(self.sdk.calls[0]["project"], "hoiretarget-boxes")
        self.assertEqual(self.sdk.calls[0]["id"], "ultra-new01")
        self.assertEqual(self.sdk.new_logs[0]["wm_total"], 4.5)
        self.assertEqual(self.sdk.new_logs[1]["eval/clip_count"], 80)

    def test_mutated_complete_eval_is_rejected_without_losing_training_stream(self):
        controller = Controller(self.base/"eval_control"); controller.add()
        Path(controller.state["evaluations"]["2000"]["result"]).write_text('{"eval/iteration":2000,"eval/success_rate":1}')
        for rank in range(2): self.write(rank, row(1))
        self.assertEqual(self.run_once(supervisor=controller), 0)
        self.assertEqual(len(self.sdk.new_logs), 1)
        self.assertEqual(self.sdk.summary["eval/status"], "failed")

    def test_handoff_rejects_changed_metadata_and_durable_payload_conflicts(self):
        (self.training/"run_metadata.json").write_text('{"modified":true}')
        self.assertEqual(self.run_once(), 1)
        self.assertEqual(self.sdk.calls, [])
        outbox = monitor.EventOutbox(self.base/"outbox")
        outbox.enqueue("train:000001", "train", {"iteration": 1})
        with self.assertRaisesRegex(ValueError, "changed"):
            outbox.enqueue("train:000001", "train", {"iteration": 1, "wm_total": 2.})

    def test_two_stream_events_are_not_reuploaded_after_successful_handoff(self):
        for rank in range(2): self.write(rank, row(1))
        controller = Controller(self.base/"eval_control"); controller.add()
        self.assertEqual(self.run_once(supervisor=controller), 0)
        self.assertEqual(len(self.sdk.new_logs), 2)
        self.assertEqual(self.run_once(supervisor=controller), 0)
        self.assertEqual(len(self.sdk.new_logs), 2)
        self.assertEqual(len(monitor.EventOutbox(self.control).events), 2)


if __name__ == "__main__":
    unittest.main()
