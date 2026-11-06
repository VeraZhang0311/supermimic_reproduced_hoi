"""Offline sidecar protocol tests; no W&B SDK, login or network required."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import wandb_gmt_monitor as monitor


def csv_row(iteration):
    return ",".join(str(iteration if i == 0 else i*.01) for i in range(len(monitor.CSV_FIELDS)))+"\n"


class Config(dict):
    def update(self, other, *, allow_val_change=False):
        assert allow_val_change
        super().update(other)


class FakeRun:
    id, project, entity = "test01", "supermimic_gmt", "team"
    url = "https://wandb.ai/team/supermimic_gmt/runs/test01"
    def __init__(self):
        self.summary, self.config, self.logs, self.finishes = {}, Config(), [], []
    def log(self, data, *, step):
        self.logs.append((step, dict(data)))
    def finish(self, *, exit_code):
        self.finishes.append(exit_code)


class FakeWandb:
    def __init__(self):
        self.run, self.init_calls, self.settings = FakeRun(), [], None
    def Settings(self, **kwargs):
        self.settings = kwargs
        return SimpleNamespace(**kwargs)
    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.run.config.update(kwargs["config"], allow_val_change=True)
        return self.run


class MonitorTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temporary.name)
        self.csv = self.root / "train_log.csv"
        self.header = ",".join(monitor.CSV_FIELDS)+"\n"
    def tearDown(self):
        self.temporary.cleanup()
    def write_config(self):
        (self.root / "args.json").write_text(json.dumps({"wm_only_iters": 500, "dataset": "lafan1",
                                                        "log_dir": "/private/run", "api_key": "secret-value"}))
        (self.root / "gmt_contract.json").write_text(json.dumps({"version": "robot_privileged_gmt_hw3_actuator_v3",
            "dim": 3371, "bodies": 38, "actuator_state": {"dim": 88, "secret": "secret-value"}}))
    def run_monitor(self, sdk, **kwargs):
        return monitor.monitor(self.root, entity="team", project="supermimic_gmt", run_id="test01",
                               run_name="gmt-test", wandb_module=sdk, **kwargs)

    def test_reader_keeps_partial_header_and_rows_and_deduplicates(self):
        reader = monitor.CSVReader(self.csv)
        self.assertEqual(reader.poll(), [])
        self.csv.write_text(self.header[:20])
        self.assertEqual(reader.poll(), [])
        with self.csv.open("a") as stream:
            stream.write(self.header[20:]+csv_row(50)+csv_row(100)[:12])
        self.assertEqual([row["iteration"] for row in reader.poll()], [50])
        self.assertEqual(reader.poll(), [])
        with self.csv.open("a") as stream:
            stream.write(csv_row(100)[12:]+csv_row(100)+csv_row(50)+csv_row(150))
        self.assertEqual([row["iteration"] for row in reader.poll()], [100, 150])
        self.assertEqual(reader.last_iteration, 150)

    def test_reader_rejects_extra_columns_nonfinite_and_bad_iterations(self):
        self.csv.write_text(self.header.rstrip()+",unapproved\n")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            monitor.CSVReader(self.csv).poll()
        for row in (csv_row(50).replace("0.01", "nan", 1), csv_row(1.5), "50,0.1\n"):
            self.csv.write_text(self.header+row)
            with self.assertRaises(ValueError):
                monitor.CSVReader(self.csv).poll()

    def test_reader_rejects_truncation(self):
        self.csv.write_text(self.header+csv_row(50))
        reader = monitor.CSVReader(self.csv)
        reader.poll()
        self.csv.write_text("")
        with self.assertRaisesRegex(ValueError, "truncated"):
            reader.poll()

    def test_curated_config_never_copies_paths_secrets_or_nested_payloads(self):
        result = monitor.curated_config({"num_envs": 4096, "dataset": "lafan1", "amp": "bf16", "no_critic": False,
            "lr": {"secret": "do-not-send"}, "seed": float("nan"), "api_key": "do-not-send",
            "motion_dir": "/private/do-not-send", "resume": "/weights/do-not-send"},
            {"dim": 3371, "bodies": 38, "version": "robot_privileged_gmt_hw3_actuator_v3",
             "source": "do-not-send", "analytic_heads": {"control_dt": .02, "private": "do-not-send"},
             "physics": {"profile": "fengqiao_hw3_mujoco_v1", "joints": ["do-not-send"]},
             "actuator_state": {"dim": 88, "private": "do-not-send"}, "urdf_sha256": "a"*64})
        self.assertEqual(result["num_envs"], 4096)
        self.assertEqual(result["actuator_state_dim"], 88)
        self.assertEqual(result["physics_profile"], "fengqiao_hw3_mujoco_v1")
        self.assertNotIn("do-not-send", json.dumps(result))
        self.assertNotIn("lr", result)
        self.assertNotIn("seed", result)

    def test_monitor_late_config_partial_rows_duplicate_stages_and_failure_exit(self):
        sdk, ticks = FakeWandb(), []
        def sleep(_):
            ticks.append(1)
            if len(ticks) == 1:
                ready = json.loads((self.root / "wandb_ready.json").read_text())
                self.assertEqual(ready, {"url": sdk.run.url, "id": "test01", "project": "supermimic_gmt", "entity": "team"})
                self.assertEqual(sdk.run.summary["stage"], "initializing")
                self.write_config()
                self.csv.write_text(self.header+csv_row(50)+csv_row(500)[:12])
            else:
                with self.csv.open("a") as stream:
                    stream.write(csv_row(500)[12:]+csv_row(50)+csv_row(550)+"600,partial")
                (self.root / "training_exitcode").write_text("7\n")
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(self.run_monitor(sdk, sleep=sleep), 7)
            self.assertEqual(os.environ["WANDB_DISABLE_CODE"], "true")
        self.assertEqual([step for step, _ in sdk.run.logs], [50, 500, 550])
        self.assertEqual([data["stage"] for _, data in sdk.run.logs], ["wm_warmup", "wm_warmup", "actor_critic"])
        self.assertEqual(sdk.run.finishes, [7])
        self.assertEqual(sdk.run.summary["training_exitcode"], 7)
        self.assertEqual(sdk.run.config["privileged_dim"], 3371)
        self.assertNotIn("secret-value", json.dumps(sdk.run.config))
        self.assertEqual(sdk.settings, {"disable_git": True, "console": "off", "x_disable_stats": True})
        options = sdk.init_calls[0]
        self.assertEqual(options["mode"], "online")
        self.assertEqual(options["resume"], "never")
        self.assertIs(options["save_code"], False)
        self.assertFalse((self.root / "wandb_error.json").exists())

    def test_final_drain_reads_append_after_initial_poll(self):
        sdk = FakeWandb()
        self.csv.write_text(self.header+csv_row(50))
        def exitcode(_):
            with self.csv.open("a") as stream:
                stream.write(csv_row(100))
            return 0
        with patch.object(monitor, "read_exitcode", exitcode):
            self.assertEqual(self.run_monitor(sdk), 0)
        self.assertEqual([step for step, _ in sdk.run.logs], [50, 100])
        self.assertEqual(sdk.run.finishes, [0])

    def test_error_handshake_excludes_exception_text_and_environment(self):
        sdk = FakeWandb()
        def fail(**_):
            raise RuntimeError("api_key=secret-value /private/path")
        sdk.init = fail
        self.assertEqual(self.run_monitor(sdk), 1)
        error = (self.root / "wandb_error.json").read_text()
        self.assertNotIn("secret-value", error)
        self.assertNotIn("api_key", error)
        self.assertNotIn("/private", error)
        self.assertEqual(json.loads(error)["phase"], "initializing")
        self.assertFalse((self.root / "wandb_ready.json").exists())

    def test_existing_handshake_is_never_replaced_or_reinitialized(self):
        sdk = FakeWandb()
        ready = self.root / "wandb_ready.json"
        ready.write_text('{"id":"existing"}')
        self.assertEqual(self.run_monitor(sdk), 1)
        self.assertEqual(ready.read_text(), '{"id":"existing"}')
        self.assertEqual(sdk.init_calls, [])


if __name__ == "__main__":
    unittest.main()
