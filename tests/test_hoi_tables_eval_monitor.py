"""Scheduler/W&B replay tests; no real scheduler, SDK, GPU or network calls."""
import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile

from hoi_tables import eval_monitor as monitor
from tests.test_hoi_tables_telemetry import FakeWandb


class EvalWandb(FakeWandb):
    def __init__(self):
        super().__init__()
        self.definitions = []

    def init(self, **kwargs):
        run = super().init(**kwargs)
        run.define_metric = lambda name, **values: self.definitions.append((name, values))
        return run


class FakeSlurm:
    def __init__(self):
        self.jobs, self.calls = {}, []
        self.lose_reply = False

    def submit(self, request_path, source_root, **kwargs):
        request = json.loads(Path(request_path).read_text())
        job = str(100+len(self.calls))
        self.calls.append((str(request_path), kwargs))
        self.jobs[job] = {"name": request["job_name"], "state": "RUNNING", "exitcode": None,
                          "request": request, "request_path": str(request_path)}
        if self.lose_reply:
            self.lose_reply = False
            raise subprocess.TimeoutExpired(["sbatch"], 45)
        return job

    def find(self, name, created):
        return [job for job, record in self.jobs.items() if record["name"] == name]

    def status(self, job):
        return {k: self.jobs[str(job)][k] for k in ("state", "exitcode")}


class EvalMonitorTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/".cache/eval_monitor_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.campaign = Path(self.temporary.name)
        self.training = self.campaign/"formal_03"
        self.training.mkdir()
        self.control = self.campaign/"protected_eval_2000"
        self.source = self.campaign/"source_eval01"
        (self.source/"hoi_tables").mkdir(parents=True)
        (self.source/"hoi_tables/evaluate.py").write_text('"""immutable test evaluator"""\n')
        self.training_source = self.campaign/"source_train03"
        self.training_source.mkdir()
        (self.campaign/"input").mkdir()
        clips = [{"clip_id": f"{asset}_{i}", "object_asset": asset, "control_steps": i+2,
                  "state_frames": i+3, "partition": "protected_eval"} for asset in monitor.ASSETS for i in range(11)]
        self.manifest = self.campaign/"input/manifest.json"
        self.manifest.write_text(json.dumps({"clips": clips, "cache": {"sha256": "b"*64}}))
        (self.training/"run_metadata.json").write_text(json.dumps({"source_sha256": "a"*64,
                    "manifest_sha256": monitor.sha256(self.manifest), "global_config": {"wm_only_iters": 500}}))
        (self.training/"wandb_registration.json").write_text(json.dumps({"id": "44ad3b72", "entity": monitor.ENTITY, "project": monitor.PROJECT}))
        self.slurm = FakeSlurm()
        self.sdk = EvalWandb()
        self.args = dict(parent_run_id="44ad3b72", eval_id="eval-new01", name="tables_v1_eval-20261107-180000", wandb_module=self.sdk)

    def supervisor(self, **kwargs):
        return monitor.Supervisor(self.control, self.campaign, self.training, self.source, self.training_source,
                                  "44ad3b72", slurm=self.slurm, **kwargs)

    def checkpoints(self, iteration, ranks=(0, 1)):
        for rank in ranks:
            path = self.training/f"rank_{rank:02d}/checkpoint_{iteration:06d}.pt"
            path.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", f"rank{rank}:iteration{iteration}")

    def result(self, request):
        expected = monitor.expected_clips(self.manifest)
        clips = [{"clip_id": name, "asset": clip["asset"], "control_steps": clip["control_steps"],
                  "reference_control_steps": clip["control_steps"], "state_frames": clip["control_steps"]+1,
                  "start_frame": 10, "reference_start_frame": 10,
                  "final_frame": 10+clip["control_steps"], "reference_end_frame": 10+clip["control_steps"],
                  "full_reference_duration": True, "no_failure_reset": True,
                  "reset_calls_during_rollout": 0, "failed": True} for name, clip in expected.items()]
        metrics = {key: .25 for key in monitor.METRICS}
        return {"passed": True, "smoke": False, "scheduled_evaluation": True,
                "iteration": request["iteration"], "parent_run_id": request["parent_run_id"],
                "manifest_sha256": request["manifest_sha256"],
                "partition": "protected_eval", "checkpoint_sha256": request["checkpoint_sha256"],
                "clip_count": 22, "control_steps": sum(c["control_steps"] for c in clips), "clips": clips,
                "no_normalizer_update": True, "no_optimizer_update": True,
                "full_reference_duration": True, "no_failure_reset": True, "metrics": metrics,
                "per_asset": {asset: {"clip_count": 11, "control_steps": sum(c["control_steps"] for c in clips if c["asset"] == asset),
                                      "metrics": metrics.copy()} for asset in monitor.ASSETS}}

    def finish(self, job, *, failed=False):
        record = self.slurm.jobs[job]
        record.update(state="FAILED" if failed else "COMPLETED", exitcode=1 if failed else 0)
        if not failed:
            path = Path(record["request"]["output"])/"result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.result(record["request"])))

    def test_atomic_pair_required_and_backlog_runs_one_gpu_job_at_a_time(self):
        self.checkpoints(2000, ranks=(0,))
        supervisor = self.supervisor()
        self.assertEqual(supervisor.tick()["evaluations"], {})
        self.checkpoints(2000, ranks=(1,)); self.checkpoints(4000)
        supervisor.tick(); supervisor.tick()
        self.assertEqual(len(self.slurm.calls), 1)
        self.finish("100")
        state = supervisor.tick()
        self.assertEqual(state["evaluations"]["2000"]["status"], "complete")
        self.assertEqual(len(self.slurm.calls), 2)
        self.assertEqual(sum(r["state"] == "RUNNING" for r in self.slurm.jobs.values()), 1)

    def test_lost_submit_reply_and_monitor_restart_do_not_duplicate_job(self):
        self.checkpoints(2000)
        self.slurm.lose_reply = True
        supervisor = self.supervisor(); supervisor.tick()
        self.assertEqual(supervisor.state["evaluations"]["2000"]["status"], "submission_pending")
        restarted = self.supervisor(); restarted.tick()
        self.assertEqual(len(self.slurm.calls), 1)
        self.assertEqual(restarted.state["evaluations"]["2000"]["job_id"], "100")

    def test_full_coverage_flags_checkpoint_and_source_bytes_must_match(self):
        self.checkpoints(2000)
        supervisor = self.supervisor(); supervisor.tick()
        request = self.slurm.jobs["100"]["request"]
        result = self.result(request)
        path = self.control/"result-test.json"
        path.write_text(json.dumps(result))
        metrics = monitor.validate_result(path, request)
        self.assertEqual(metrics["eval/clip_count"], 22)
        self.assertEqual(metrics["eval/iteration"], 2000)
        self.assertEqual(metrics["eval/success_rate"], .25)  # Policy failure is not an evaluator protocol failure.
        for change in ("missing", "cropped", "reset", "normalizer", "parent", "hash", "manifest"):
            invalid = copy.deepcopy(result)
            if change == "missing": invalid["clips"].pop()
            if change == "cropped": invalid["clips"][0]["control_steps"] -= 1
            if change == "reset": invalid["clips"][0]["reset_calls_during_rollout"] = 1
            if change == "normalizer": invalid["no_normalizer_update"] = False
            if change == "parent": invalid["parent_run_id"] = "wrong-run"
            if change == "hash": invalid["checkpoint_sha256"]["rank_00"] = "0"*64
            if change == "manifest": invalid["manifest_sha256"] = "0"*64
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError, msg=change): monitor.validate_result(path, request)
        path.write_text(json.dumps(result))
        (self.source/"hoi_tables/evaluate.py").write_text('"""modified evaluator"""\n')
        with self.assertRaisesRegex(ValueError, "source bytes"):
            monitor.validate_result(path, request)

    def test_empty_checkpoint_map_and_wrong_parent_registration_are_rejected(self):
        self.checkpoints(2000)
        supervisor = self.supervisor(); supervisor.tick()
        request = copy.deepcopy(self.slurm.jobs["100"]["request"])
        request["checkpoints"] = {}
        with self.assertRaisesRegex(ValueError, "both immutable"):
            monitor.validate_request(request)
        registration = self.training/"wandb_registration.json"
        registration.write_text(json.dumps({"id": "other-run", "entity": monitor.ENTITY, "project": monitor.PROJECT}))
        with self.assertRaisesRegex(ValueError, "parent run"):
            self.supervisor()

    def test_worker_failure_stops_before_dispatching_more_bad_jobs(self):
        self.checkpoints(2000); self.checkpoints(4000)
        supervisor = self.supervisor(); supervisor.tick(); self.finish("100", failed=True)
        with self.assertRaisesRegex(RuntimeError, "worker or result"):
            supervisor.tick()
        self.assertEqual(supervisor.state["evaluations"]["2000"]["status"], "failed")
        self.assertEqual(len(self.slurm.calls), 1)
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            self.supervisor().tick()

    def test_training_terminal_without_exit_sentinel_drains_existing_then_marks_missing(self):
        self.slurm.jobs["1512413"] = {"state": "NODE_FAIL", "exitcode": 0}
        self.checkpoints(2000)
        supervisor = self.supervisor(train_job="1512413"); supervisor.tick()
        self.assertFalse(supervisor.state.get("finished", False))
        self.finish("100")
        state = supervisor.tick()
        self.assertTrue(state["finished"])
        self.assertEqual(state["parent_training_exitcode"], 1)
        self.assertEqual(state["missing_milestones"], list(range(4000, 20001, 2000)))

    def test_eval_run_is_independent_has_custom_axis_and_never_logs_fake_zero(self):
        metadata_before = (self.training/"run_metadata.json").read_bytes()
        registration = monitor.register(self.control, self.training, **self.args)
        self.assertEqual(registration["parent_run_id"], "44ad3b72")
        run = self.sdk.runs["eval-new01"]
        self.assertEqual(run.logs, [])
        self.assertEqual(self.sdk.init_calls[0]["resume"], "never")
        self.assertIn(("eval/*", {"step_metric": "eval/iteration"}), self.sdk.definitions)
        self.assertTrue(self.sdk.init_calls[0]["config"]["eval_only"])
        self.assertNotIn(str(self.training), json.dumps(self.sdk.init_calls[0]["config"]))
        self.assertEqual(metadata_before, (self.training/"run_metadata.json").read_bytes())
        with self.assertRaises(ValueError):
            monitor.register(self.control/"bad", self.training, **dict(self.args, eval_id="44ad3b72"))

    def test_modified_completed_result_is_not_uploaded_on_monitor_resume(self):
        monitor.register(self.control, self.training, **self.args)
        self.checkpoints(2000)
        supervisor = self.supervisor(); supervisor.tick(); self.finish("100")
        (self.training/"training_exitcode").write_text("1")
        supervisor.tick()
        entry = supervisor.state["evaluations"]["2000"]
        result_path = Path(entry["result"])
        result = json.loads(result_path.read_text()); result["metrics"]["success_rate"] = .9
        result_path.write_text(json.dumps(result))
        self.assertEqual(monitor.monitor(self.control, self.campaign, self.training, self.source, self.training_source,
                                        slurm=self.slurm, sleep=lambda _: None, **self.args), 1)
        self.assertEqual(self.sdk.runs["eval-new01"].logs, [])

    def test_all_ten_milestones_publish_once_and_finish_without_touching_parent(self):
        monitor.register(self.control, self.training, **self.args)
        for iteration in monitor.MILESTONES: self.checkpoints(iteration)
        parent_files = {str(p): p.read_bytes() for p in self.training.rglob("*") if p.is_file()}
        def advance(_):
            running = [job for job, record in self.slurm.jobs.items() if record["state"] == "RUNNING"]
            self.assertEqual(len(running), 1)
            self.finish(running[0])
            if self.slurm.jobs[running[0]]["request"]["iteration"] == 20000:
                (self.training/"training_exitcode").write_text("0")
        result = monitor.monitor(self.control, self.campaign, self.training, self.source, self.training_source,
                                 slurm=self.slurm, sleep=advance, **self.args)
        self.assertEqual(result, 0)
        run = self.sdk.runs["eval-new01"]
        self.assertEqual([step for step, _ in run.logs], list(monitor.MILESTONES))
        self.assertEqual(len(self.slurm.calls), 10)
        self.assertEqual(self.sdk.init_calls[1]["resume"], "must")
        for path, data in parent_files.items(): self.assertEqual(Path(path).read_bytes(), data)
        # Re-open the same ledger/run: all complete results are deduplicated.
        self.assertEqual(monitor.monitor(self.control, self.campaign, self.training, self.source, self.training_source,
                                        slurm=self.slurm, sleep=lambda _: None, **self.args), 0)
        self.assertEqual(len(run.logs), 10)
        self.assertEqual(len(self.slurm.calls), 10)


if __name__ == "__main__":
    unittest.main()
