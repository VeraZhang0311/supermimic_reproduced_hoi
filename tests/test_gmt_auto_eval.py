"""Offline supervisor protocol tests: no scheduler, SDK or GPU connection."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import gmt_auto_eval as auto
from tools import wandb_gmt_monitor as monitor


class FakeConfig(dict):
    def __init__(self):
        super().__init__()
        self.updates = []
    def update(self, value, *, allow_val_change=False):
        self.updates.append((dict(value), allow_val_change))
        super().update(value)


class FakeRun:
    def __init__(self, sdk):
        self.sdk, self.summary, self.metrics, self.finishes = sdk, sdk.server_summary, [], []
        options = sdk.calls[-1]
        self.id, self.project, self.entity = (options[key] for key in ("id", "project", "entity"))
        self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"
        self.config = FakeConfig()
        if "config" in options:
            self.config.update(options["config"], allow_val_change=True)
    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))
    def log(self, record):  # Passing a global step is deliberately a TypeError.
        self.sdk.logs.append(dict(record))
        self.summary.update(record)
    def finish(self, *, exit_code):
        self.finishes.append(exit_code)


class FakeSDK:
    def __init__(self, iteration=0):
        self.server_summary = {"iteration": iteration}
        self.logs, self.calls, self.runs = [], [], []
        self.api_calls = 0
    def Api(self):
        self.api_calls += 1
        return SimpleNamespace(run=lambda path: SimpleNamespace(summary=self.server_summary,
            scan_history=lambda keys: [row for row in self.logs if all(key in row for key in keys)]))
    def Settings(self, **kwargs):
        return kwargs
    def init(self, **kwargs):
        self.calls.append(kwargs)
        run = FakeRun(self)
        self.runs.append(run)
        return run


class FakeScheduler:
    def __init__(self):
        self.jobs = {"1506052": {"state": "RUNNING", "exit_code": None}}
        self.submissions, self.names = [], {}
        self.fail_reply = False
    def submit(self, script, **kwargs):
        job = str(800 + len(self.submissions))
        self.submissions.append((job, kwargs))
        self.names[job] = kwargs["name"]
        self.jobs[job] = {"state": "PENDING", "exit_code": None}
        if self.fail_reply:
            self.fail_reply = False
            raise TimeoutError("reply lost after acceptance")
        return job
    def status(self, job):
        return self.jobs[str(job)]
    def find_jobs(self, name, since):
        return [job for job in self.names if self.names[job] == name]


class AutoEvalTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.args = SimpleNamespace(run_dir=self.run, control_dir=self.root/"control", source_root=self.root/"source",
            train_job_id="1506052", wandb_entity="team", wandb_project="supermimic_gmt", wandb_id="c0107041",
            total_iterations=4000, interval=2000, eval_script=self.root/"eval.sbatch.sh", handoff_file=self.root/"handoff.json")
        self.scheduler, self.sdk = FakeScheduler(), FakeSDK()
        self.now = [1000.]
    def tearDown(self):
        self.tmp.cleanup()
    def make(self):
        return auto.Supervisor(self.args, scheduler=self.scheduler, monitor_module=monitor, clock=lambda: self.now[0])
    def handoff(self):
        self.args.handoff_file.write_text(json.dumps({"old_monitor_stopped": True, "old_core_stopped": True,
                                                     "run_id": "c0107041"}))
    def checkpoint(self, iteration):
        path = self.run / f"ckpt_{iteration:06d}.pt"
        path.write_bytes(f"immutable checkpoint {iteration}".encode())
        return path
    def result(self, iteration, *, job_state="COMPLETED"):
        ledger = json.loads((self.args.control_dir/"ledger.json").read_text())
        entry = ledger["checkpoints"][str(iteration)]
        attempt = entry["attempts"][-1]
        receipt = {"passed": True, "checkpoint_iteration": iteration, "checkpoint_sha256": entry["sha256"],
            "checkpoint_path": entry["checkpoint"], "norm_return": .55, "norm_return_framew": .54,
            "n_failed": 1, "n_total": 40, "failed": ["walk1_subject1"], "train_wall_h": .2,
            "saved_at": "2026-11-05 22:00:00", "eval_at": "2026-11-05 22:02:00"}
        auto.atomic_json(attempt["output"], receipt)
        self.scheduler.jobs[attempt["job_id"]] = {"state": job_state, "exit_code": 0 if job_state=="COMPLETED" else 1}
        return Path(attempt["output"]), receipt

    def test_ready_handoff_prevents_sdk_and_job_submission(self):
        self.checkpoint(2000)
        supervisor = self.make()
        self.assertTrue((self.args.control_dir/"supervisor_ready.json").exists())
        self.assertFalse(supervisor.start_writer(self.sdk))
        self.assertEqual(self.sdk.calls, [])
        self.assertEqual(self.scheduler.submissions, [])
        self.args.handoff_file.write_text(json.dumps({"old_monitor_stopped": True, "old_core_stopped": False,
                                                     "run_id": "c0107041"}))
        self.assertFalse(supervisor.start_writer(self.sdk))
        self.handoff()
        self.assertTrue(supervisor.start_writer(self.sdk))
        self.assertEqual(self.sdk.calls[0]["resume"], "must")
        self.assertTrue(self.sdk.calls[0]["settings"]["x_disable_stats"])
        self.assertFalse(self.sdk.calls[0]["save_code"])

    def test_late_evaluation_independent_axis_and_csv_and_restart_dedup(self):
        self.sdk.server_summary["iteration"] = 2500
        header = ",".join(monitor.CSV_FIELDS) + "\n"
        def row(it):
            return ",".join(str(it if key=="iteration" else .1) for key in monitor.CSV_FIELDS)+"\n"
        (self.run/"train_log.csv").write_text(header+row(2000)+row(2500)+row(2550))
        self.checkpoint(2000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        self.assertEqual([record["iteration"] for record in self.sdk.logs], [2550])
        self.assertEqual(len(self.scheduler.submissions), 1)
        self.result(2000)
        supervisor.tick()
        evaluation = self.sdk.logs[-1]
        self.assertEqual(evaluation["eval/iteration"], 2000)
        self.assertEqual(evaluation["eval/n_total"], 40)
        self.assertNotIn("iteration", evaluation)
        self.assertIn(("wm_loss", {"step_metric": "iteration"}), self.sdk.runs[0].metrics)
        self.assertIn(("eval/*", {"step_metric": "eval/iteration"}), self.sdk.runs[0].metrics)
        with (self.run/"eval.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(list(rows[0]), list(auto.EVAL_COLUMNS))
        self.assertEqual(rows[0]["iteration"], "2000")
        count = len(self.sdk.logs)
        restarted = self.make()
        restarted.start_writer(self.sdk)
        restarted.tick()
        self.assertEqual(len(self.scheduler.submissions), 1)
        self.assertEqual(len(self.sdk.logs), count)

    def test_claim_before_submission_adopts_lost_reply_without_duplicate(self):
        self.checkpoint(2000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        self.scheduler.fail_reply = True
        supervisor.tick()
        self.assertIsNone(supervisor.state["checkpoints"]["2000"]["attempts"][0]["job_id"])
        restarted = self.make()
        restarted.start_writer(self.sdk)
        restarted.tick()
        attempt = restarted.state["checkpoints"]["2000"]["attempts"][0]
        self.assertEqual(attempt["job_id"], "800")
        self.assertEqual(len(self.scheduler.submissions), 1)
        self.assertEqual(len(restarted.state["checkpoints"]["2000"]["attempts"]), 1)

    def test_unknown_claim_never_retries_submission(self):
        self.checkpoint(2000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        self.scheduler.fail_reply = True
        supervisor.tick()
        self.scheduler.names.clear()  # No accounting/live proof: do not guess.
        for _ in range(3):
            self.now[0] += 600
            supervisor.tick()
        self.assertEqual(len(self.scheduler.submissions), 1)
        self.assertEqual(self.sdk.server_summary["eval/submission_uncertain_count"], 1)

    def test_three_failed_attempts_continue_other_ckpts_without_failing_trainer_run(self):
        self.checkpoint(2000)
        self.checkpoint(4000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        for _ in range(3):
            attempt = supervisor.state["checkpoints"]["2000"]["attempts"][-1]
            auto.atomic_json(Path(attempt["output"]+".error.json"), {"passed": False})
            self.scheduler.jobs[attempt["job_id"]] = {"state": "FAILED", "exit_code": 1}
            supervisor.tick()
        self.assertTrue(supervisor.state["checkpoints"]["2000"]["exhausted"])
        self.assertEqual(len(supervisor.state["checkpoints"]["2000"]["attempts"]), 3)
        self.assertEqual(len(self.scheduler.submissions), 4)  # Three attempts + next checkpoint.
        self.result(4000)
        (self.run/"training_exitcode").write_text("0\n")
        report = supervisor.tick()
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_iterations"], [2000])
        self.assertEqual(self.sdk.runs[-1].finishes, [0])
        self.assertEqual(self.sdk.server_summary["eval/status"], "incomplete")

    def test_full_drain_finishes_only_after_final_result(self):
        self.checkpoint(2000)
        self.checkpoint(4000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        (self.run/"training_exitcode").write_text("0")
        self.assertIsNone(supervisor.tick())
        self.result(2000)
        self.assertIsNone(supervisor.tick())
        self.result(4000)
        report = supervisor.tick()
        self.assertTrue(report["passed"])
        self.assertEqual(report["successful_iterations"], [2000, 4000])
        self.assertEqual(self.sdk.runs[-1].finishes, [0])

    def test_receipt_cannot_false_pass_hash_count_or_nonfinite(self):
        path = self.checkpoint(2000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        output, receipt = self.result(2000)
        for key, value in (("checkpoint_sha256", "0"*64), ("n_total", 39), ("norm_return", float("nan")),
                           ("n_failed", True), ("checkpoint_iteration", 4000)):
            bad = {**receipt, key: value}
            output.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                auto.validate_result(output, path, 2000, receipt["checkpoint_sha256"])
        auto.atomic_json(output, receipt)
        path.write_bytes(b"changed immutable checkpoint")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            auto.validate_result(output, path, 2000, receipt["checkpoint_sha256"])

    def test_result_waits_for_successful_job_exit(self):
        self.checkpoint(2000)
        supervisor = self.make()
        self.handoff()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        self.result(2000, job_state="RUNNING")
        supervisor.tick()
        self.assertEqual(self.sdk.logs, [])
        self.assertEqual(len(self.scheduler.submissions), 1)

    def test_control_binding_prevents_wrong_run_reuse(self):
        self.make()
        self.args.wandb_id = "different"
        with self.assertRaisesRegex(ValueError, "different"):
            self.make()

    def new_run_mode(self, *, clips=512):
        self.args.create_run = True
        self.args.run_name = "boneseed-parent26k-new20k"
        self.args.handoff_file = None
        self.args.expected_clips = clips
        self.args.eval_manifest = self.root / "fixed_eval_manifest.json"
        self.args.eval_manifest.write_text(json.dumps({"count": clips, "ids": ["fixed fixture identity"]}))

    def test_create_run_skips_remote_query_publishes_ready_and_late_curated_config(self):
        self.new_run_mode()
        supervisor = self.make()
        with patch.object(self.sdk, "Api", side_effect=AssertionError("New run must not be queried")):
            self.assertTrue(supervisor.start_writer(self.sdk))
        options = self.sdk.calls[0]
        self.assertEqual(options["resume"], "never")
        self.assertEqual(options["name"], self.args.run_name)
        self.assertEqual(options["config"], {"monitor_protocol": "gmt_numeric_csv_v1", "eval/clip_count": 512,
                                             "eval/manifest_sha256": auto.sha256(self.args.eval_manifest)})
        ready = json.loads((self.run / "wandb_ready.json").read_text())
        self.assertEqual(ready["id"], self.args.wandb_id)
        self.assertEqual(ready["project"], self.args.wandb_project)
        self.assertEqual(ready["url"], self.sdk.runs[0].url)
        self.assertTrue((self.run / "wandb_create_claim.json").exists())
        self.assertEqual(supervisor.state["binding"]["expected_clips"], 512)
        self.assertEqual(supervisor.state["binding"]["eval_manifest_sha256"], auto.sha256(self.args.eval_manifest))
        (self.run / "args.json").write_text(json.dumps({"dataset": "boneseed", "num_envs": 4096,
            "iterations": 20000, "wm_only_iters": 0, "amp": "bf16", "transfer_parent_iteration": 26000,
            "log_dir": "/private/path",
            "api_key": "never-upload-this"}))
        (self.run / "gmt_contract.json").write_text(json.dumps({"version": "robot_privileged_gmt_hw3_actuator_v3",
            "dim": 3371, "bodies": 38, "private_payload": "never-upload-this"}))
        supervisor.drain_training()
        config = self.sdk.runs[0].config
        self.assertEqual(config["dataset"], "boneseed")
        self.assertEqual(config["wm_only_iters"], 0)
        self.assertEqual(config["transfer_parent_iteration"], 26000)
        self.assertEqual(config["eval/clip_count"], 512)
        self.assertEqual(config["eval/manifest_sha256"], auto.sha256(self.args.eval_manifest))
        self.assertEqual(self.sdk.runs[0].summary["eval/clip_count"], 512)
        self.assertNotIn("never-upload-this", json.dumps(config))
        self.assertNotIn("/private/path", json.dumps(config))
        self.assertTrue(supervisor.state["writer_config_published"])
        count = len(config.updates)
        supervisor.drain_training()
        self.assertEqual(len(config.updates), count)
        self.assertEqual(self.scheduler.submissions, [])

    def test_create_run_never_recreates_existing_id_or_allows_second_control_writer(self):
        self.new_run_mode()
        supervisor = self.make()
        supervisor.start_writer(self.sdk)
        self.assertTrue(supervisor.start_writer(self.sdk))
        self.assertEqual(len(self.sdk.calls), 1)
        restarted = self.make()
        with self.assertRaises(FileExistsError):
            restarted.start_writer(self.sdk)
        self.args.control_dir = self.root / "second_control"
        second = self.make()
        with self.assertRaises(FileExistsError):
            second.start_writer(self.sdk)
        self.assertEqual(len(self.sdk.calls), 1)
        self.assertEqual(self.sdk.api_calls, 0)

    def test_create_run_lost_init_reply_is_not_retried_or_assumed_safe(self):
        self.new_run_mode()
        supervisor = self.make()
        with patch.object(self.sdk, "init", side_effect=TimeoutError("uncertain remote initialization")):
            with self.assertRaises(TimeoutError):
                supervisor.start_writer(self.sdk)
        self.assertFalse((self.run / "wandb_ready.json").exists())
        self.assertEqual(json.loads((self.run / "wandb_error.json").read_text())["error_type"], "TimeoutError")
        restarted = self.make()
        with self.assertRaises(FileExistsError):
            restarted.start_writer(self.sdk)
        self.assertEqual(self.sdk.calls, [])
        self.assertEqual(self.sdk.api_calls, 0)

    def test_fixed_512_receipt_requires_manifest_sha_count_and_failed_range(self):
        self.new_run_mode()
        self.checkpoint(2000)
        supervisor = self.make()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        output, receipt = self.result(2000)
        receipt.update(n_total=512, eval_manifest_sha256=auto.sha256(self.args.eval_manifest))
        auto.atomic_json(output, receipt)
        entry = supervisor.state["checkpoints"]["2000"]
        accepted = auto.validate_result(output, entry["checkpoint"], 2000, entry["sha256"],
            expected_clips=512, expected_eval_manifest_sha256=receipt["eval_manifest_sha256"])
        self.assertEqual(accepted["n_total"], 512)
        self.assertEqual(accepted["eval_manifest_sha256"], receipt["eval_manifest_sha256"])
        for key, value in (("eval_manifest_sha256", "0"*64), ("eval_manifest_sha256", None),
                           ("n_total", 40), ("n_failed", 513)):
            auto.atomic_json(output, {**receipt, key: value})
            with self.assertRaises(ValueError):
                auto.validate_result(output, entry["checkpoint"], 2000, entry["sha256"],
                    expected_clips=512, expected_eval_manifest_sha256=receipt["eval_manifest_sha256"])
        auto.atomic_json(output, receipt)
        supervisor.tick()
        self.assertEqual(supervisor.state["checkpoints"]["2000"]["result"]["n_total"], 512)
        self.assertEqual(self.sdk.logs[-1]["eval/n_total"], 512)
        self.assertEqual(set(self.scheduler.submissions[0][1]), {"name", "source_root", "run_dir", "checkpoint", "output"})

    def test_manifest_binding_cannot_change_or_be_omitted_for_512(self):
        self.new_run_mode()
        supervisor = self.make()
        self.args.eval_manifest.write_text('{"count":512,"changed":true}')
        with self.assertRaisesRegex(ValueError, "different"):
            self.make()
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            supervisor.submit_next()
        self.args.control_dir = self.root / "missing_manifest_control"
        self.args.eval_manifest = None
        with self.assertRaisesRegex(ValueError, "require an evaluation manifest"):
            self.make()

    def test_new_writer_drains_512_evaluations_before_finishing_after_training_exit(self):
        self.new_run_mode()
        self.checkpoint(2000)
        self.checkpoint(4000)
        supervisor = self.make()
        supervisor.start_writer(self.sdk)
        supervisor.tick()
        (self.run / "training_exitcode").write_text("0\n")
        self.assertIsNone(supervisor.tick())
        self.assertEqual(self.sdk.runs[0].finishes, [])
        for iteration in (2000, 4000):
            output, receipt = self.result(iteration)
            receipt.update(n_total=512, eval_manifest_sha256=auto.sha256(self.args.eval_manifest))
            auto.atomic_json(output, receipt)
            result = supervisor.tick()
            if iteration == 2000:
                self.assertIsNone(result)
                self.assertEqual(self.sdk.runs[0].finishes, [])
            else:
                self.assertTrue(result["passed"])
        self.assertEqual(self.sdk.runs[0].finishes, [0])
        self.assertEqual(self.sdk.api_calls, 0)

    def test_legacy_full40_ledger_remains_compatible(self):
        self.make()
        path = self.args.control_dir / "ledger.json"
        state = json.loads(path.read_text())
        for key in ("expected_clips", "eval_manifest_sha256", "create_run"):
            state["binding"].pop(key)
        auto.atomic_json(path, state)
        supervisor = self.make()
        self.assertFalse(supervisor.create_run)
        self.assertEqual(supervisor.expected_clips, 40)
        self.assertIsNone(supervisor.eval_manifest_sha256)

    def test_completed_job_missing_from_squeue_uses_sacct(self):
        def command(args):
            if args[0] == "squeue":
                raise subprocess.CalledProcessError(1, args, stderr="Invalid job id specified")
            return "800|COMPLETED|0:0"
        scheduler = auto.Slurm()
        with patch.object(scheduler, "_run", side_effect=command):
            self.assertEqual(scheduler.status("800"), {"state": "COMPLETED", "exit_code": 0})

    @unittest.skipUnless(os.name == "posix", "Linux fcntl runtime only")
    def test_fcntl_lock_prevents_two_supervisors(self):
        with auto.exclusive_lock(self.args.control_dir):
            with self.assertRaises(BlockingIOError):
                with auto.exclusive_lock(self.args.control_dir):
                    self.fail("Second supervisor acquired held lock")


if __name__ == "__main__":
    unittest.main()
