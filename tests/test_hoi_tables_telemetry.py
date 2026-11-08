"""Offline W&B sidecar contracts; no SDK, credentials, login or network calls."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hoi_tables import telemetry


class FakeRun:
    def __init__(self, identity):
        self.id, self.entity, self.project = identity["id"], identity["entity"], identity["project"]
        self.url = f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"
        self.summary, self.logs, self.finishes = {}, [], []
        self.step = 0

    def log(self, values, *, step, commit):
        assert commit
        assert step >= self.step
        self.logs.append((step, dict(values)))
        self.step = step+1

    def finish(self, *, exit_code):
        self.finishes.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.runs, self.init_calls, self.projects, self.settings = {}, [], [], []

    def Api(self, **kwargs):
        def execute(document, values):
            self.projects.append(values)
            return {"upsertModel": {"project": {"name": values["name"], "access": values["access"]}}}
        return SimpleNamespace(CREATE_PROJECT="sdk-shipped-project-mutation", client=SimpleNamespace(execute=execute))

    def Settings(self, **kwargs):
        self.settings.append(kwargs)
        return kwargs

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if kwargs["resume"] == "never":
            if kwargs["id"] in self.runs:
                raise RuntimeError("old run identity must not be reused")
            self.runs[kwargs["id"]] = FakeRun(kwargs)
        elif kwargs["resume"] == "must":
            if kwargs["id"] not in self.runs:
                raise RuntimeError("registered run missing")
        else:
            raise AssertionError("Telemetry must not silently create a resume target")
        return self.runs[kwargs["id"]]


def metadata():
    return {"global_config": {"num_envs": 4096, "wm_only_iters": 500, "horizon": 32,
                              "actor_hidden": [2048, 1024, 512, 256], "amp": "bf16",
                              "checkpoint_path": "/secret/checkpoint.pt", "api_key": "secret-key"},
            "schema": {"version": "hoi_tables_hw3_privileged_teacher_v1", "wm_input_dim": 313,
                       "wm_output_dim": 82, "contact": ["left_hand_object", "right_hand_object", "object_ground", "left_foot_object", "right_foot_object"]},
            "source_sha256": "a"*64, "manifest_sha256": "b"*64,
            "branch": "hoi-retarget/tables-v1.0", "from_scratch": True,
            "train_clips": 204, "protected_eval": 22, "gpus": 2, "gpu_type": "L40", "total_env": 4096,
            "absolute_data_path": "/mnt/private/data", "credential": "secret-value"}


def row(i, value=1., stage=None):
    return {"iteration": i, "stage": stage or ("wm_only" if i <= 500 else "actor_critic"),
            "wm_total": value, "sim_rew": value/10, "real_steps": i*8, "resets": 2,
            "reward_body_position": value/20, "wm_val_h8_valid_samples": 4,
            "wm_val_contact_recall_left_hand_object": .75, "raw_rollout": [1, 2, 3], "api_key": "secret-key"}


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value)+"\n")


class TestTablesTelemetry(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1]/".cache/telemetry_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/"run_metadata.json").write_text(json.dumps(metadata()))
        self.sdk = FakeWandb()
        self.kwargs = {"run_id": "tablesfresh01", "run_name": "tables_v1-20261107-120000", "wandb_module": self.sdk}

    def register(self):
        return telemetry.register(self.root, **self.kwargs)

    def test_private_registration_has_no_fake_training_step_and_is_idempotent(self):
        registration = self.register()
        self.assertEqual(self.sdk.projects[0]["access"], "PRIVATE")
        self.assertTrue(registration["private_project"])
        self.assertEqual(self.sdk.init_calls[0]["resume"], "never")
        run = self.sdk.runs[self.kwargs["run_id"]]
        self.assertEqual(run.logs, [])
        self.assertEqual(run.summary["status"], "training_not_started")
        self.assertEqual(run.finishes, [0])
        self.assertEqual(self.register(), registration)
        self.assertEqual(len(self.sdk.init_calls), 1)
        self.assertTrue(self.sdk.settings[0]["disable_git"])
        self.assertTrue(self.sdk.settings[0]["x_disable_meta"])

    def test_authorized_config_and_metrics_exclude_files_credentials_and_rollouts(self):
        config = telemetry.curated_config(metadata())
        encoded = json.dumps(config)
        for forbidden in ("secret", "api_key", "checkpoint_path", "absolute_data_path"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(config["global_config"]["num_envs"], 4096)
        selected = telemetry._curated_metric(row(1))
        self.assertNotIn("raw_rollout", selected)
        self.assertNotIn("api_key", selected)
        self.assertIn("reward_body_position", selected)
        self.assertIn("wm_val_contact_recall_left_hand_object", selected)
        with self.assertRaises(ValueError):
            telemetry._curated_metric(dict(row(1), wm_total=float("nan")))

    def test_one_global_commit_per_common_iteration_including_wm_warmup(self):
        self.register()
        for i in (1, 2, 501):
            append(self.root/"rank_00/metrics.jsonl", row(i, 2.))
            append(self.root/"rank_01/metrics.jsonl", row(i, 4.))
        (self.root/"training_exitcode").write_text("0\n")
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 0)
        run = self.sdk.runs[self.kwargs["run_id"]]
        self.assertEqual([step for step, _ in run.logs], [1, 2, 501])
        self.assertEqual([r["stage"] for _, r in run.logs], ["wm_only", "wm_only", "actor_critic"])
        self.assertEqual(run.logs[0][1]["wm_total"], 3.)
        self.assertEqual(run.logs[0][1]["assets/largetable/wm_total"], 2.)
        self.assertEqual(run.logs[0][1]["assets/smalltable/wm_total"], 4.)
        self.assertEqual(run.logs[0][1]["real_steps"], 16)
        self.assertEqual(run.logs[0][1]["wm_val_h8_valid_samples"], 8)
        self.assertEqual(self.sdk.init_calls[1]["resume"], "must")
        self.assertEqual(run.finishes, [0, 0])

    def test_partial_lines_and_worker_lag_wait_without_half_iteration_upload(self):
        self.register()
        left, right = self.root/"rank_00/metrics.jsonl", self.root/"rank_01/metrics.jsonl"
        append(left, row(1)); append(right, row(1))
        second = json.dumps(row(2))
        with left.open("a") as stream: stream.write(second[:15])
        calls = []
        def sleep(_):
            calls.append(1)
            with left.open("a") as stream: stream.write(second[15:]+"\n")
            append(right, row(2))
            (self.root/"training_exitcode").write_text("0")
        self.assertEqual(telemetry.monitor(self.root, sleep=sleep, **self.kwargs), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual([s for s, _ in self.sdk.runs[self.kwargs["run_id"]].logs], [1, 2])

    def test_monitor_restart_uses_server_step_and_deduplicates_local_cursor(self):
        self.register()
        for rank in range(2):
            for i in (1, 2): append(self.root/f"rank_{rank:02d}/metrics.jsonl", row(i))
        (self.root/"training_exitcode").write_text("0")
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 0)
        # Emulate a crash after SDK commit but before durable local cursor.
        state = json.loads((self.root/"wandb_state.json").read_text())
        state["last_iteration"] = 1
        (self.root/"wandb_state.json").write_text(json.dumps(state))
        for rank in range(2): append(self.root/f"rank_{rank:02d}/metrics.jsonl", row(3))
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 0)
        self.assertEqual([s for s, _ in self.sdk.runs[self.kwargs["run_id"]].logs], [1, 2, 3])

    def test_failed_training_without_any_metrics_finishes_failed_run(self):
        self.register()
        (self.root/"training_exitcode").write_text("7")
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 7)
        run = self.sdk.runs[self.kwargs["run_id"]]
        self.assertEqual(run.logs, [])
        self.assertEqual(run.finishes[-1], 7)
        self.assertEqual(run.summary["status"], "failed")

    def test_privacy_failure_prevents_run_creation_and_successful_partial_log_is_error(self):
        with patch.object(self.sdk, "Api", return_value=SimpleNamespace(
                CREATE_PROJECT="mutation", client=SimpleNamespace(execute=lambda *args: {
                    "upsertModel": {"project": {"name": telemetry.PROJECT, "access": "PUBLIC"}}}))):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                self.register()
        self.assertEqual(self.sdk.init_calls, [])
        self.register()
        for rank in range(2): append(self.root/f"rank_{rank:02d}/metrics.jsonl", row(1))
        with (self.root/"rank_00/metrics.jsonl").open("a") as stream:
            stream.write('{"iteration":2')
        (self.root/"training_exitcode").write_text("0")
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 1)

    def test_identity_reuse_wrong_project_and_changed_metadata_rejected(self):
        self.register()
        bad = dict(self.kwargs, project="old-residual-project")
        self.assertEqual(telemetry.monitor(self.root, **bad), 1)
        data = metadata(); data["global_config"]["num_envs"] = 8
        (self.root/"run_metadata.json").write_text(json.dumps(data))
        self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 1)
        self.assertEqual(len(self.sdk.init_calls), 1)

    def test_errors_are_sanitized_and_live_writer_lock_prevents_duplicates(self):
        self.register()
        with telemetry.exclusive_writer(self.root):
            self.assertEqual(telemetry.monitor(self.root, **self.kwargs), 1)
        self.assertEqual(len(self.sdk.init_calls), 1)
        error = json.loads((self.root/"wandb_error.json").read_text())
        self.assertEqual(set(error), {"phase", "error_type"})

    def test_idle_timeout_terminates_orphaned_monitor(self):
        self.register()
        clock = iter([0., 0., 2.])
        self.assertEqual(telemetry.monitor(self.root, idle_timeout=1., clock=lambda: next(clock), sleep=lambda _: None, **self.kwargs), 1)
        self.assertEqual(json.loads((self.root/"wandb_error.json").read_text())["error_type"], "TimeoutError")

    def test_combined_reward_standard_deviation_includes_between_asset_gap(self):
        left, right = row(1), row(1)
        left.update(real_reward_mean=0., real_reward_std=1.)
        right.update(real_reward_mean=4., real_reward_std=1.)
        result = telemetry.aggregate([left, right])
        self.assertEqual(result["real_reward_mean"], 2.)
        self.assertAlmostEqual(result["real_reward_std"], 5**.5)


if __name__ == "__main__":
    unittest.main()
