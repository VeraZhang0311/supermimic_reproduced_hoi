"""No real network calls: whitelist extraction, SDK privacy and dry-run proof."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hoi.v1.track import build_payload, destination, live_training, main, send_online, _privacy_settings, _legacy_base_url


SECRET = "DO_NOT_UPLOAD_PRIVATE_CREDENTIAL_OR_SOURCE_PATH"


def inputs():
    plan = {"config": {"num_envs": 4096, "lr": .0002, "seed": 0, "amp": "bf16", "experiment_kind": "formal", "api_key": SECRET},
            "physics": {"hand_collision": True, "self_collision": True, "robot_usd": SECRET},
            "data_identity": {"bundle_content_sha256": "a" * 64, "manifest": {"path": SECRET, "content_sha256": "b" * 64},
                              "assets": [{"source": SECRET, "raw_motion": [[1, 2, 3]]}]},
            "source_identity": {"content_sha256": "c" * 64, "files": {SECRET: "d" * 64}},
            "bundle": SECRET, "launcher_args": [SECRET], "credentials": SECRET}
    records = [{"iteration": 1, "sim_steps": 4096, "lr": .0002, "elapsed_seconds": 2.,
                "wm": [{"total": 2., "robot": 1., "object": 1., "contact": .1, "source": SECRET}],
                "actor": {"pi_loss": -.2, "v_loss": .1, "pi_gradnorm": .03, "token": SECRET},
                "real": [{"human_reward_normalized": .9, "hoi_reward_normalized": .7, "terminations": 0, "raw_state": SECRET}],
                "fit_contact_positive": [2, 4, 6], "fit_contact_negative": [20, 40, 60],
                "policy_gate": {"passed": False, "blockers": [SECRET]}, "sourcecode": SECRET}]
    evaluation = {"partition": "eval80", "checkpoint_sha256": "e" * 64, "checkpoint": SECRET,
                  "aggregate": {"count": 1, "human_normalized_return": .9, "hoi_normalized_return": .7, "password": SECRET},
                  "cases": [{"id": SECRET, "object_asset": SECRET, "rollout": SECRET, "raw": SECRET,
                             "human_normalized_return": .9, "hoi_normalized_return": .7, "phc_pass": True,
                             "object_position_m": {"mean": .04, "max": .1, "path": SECRET},
                             "contact": {"expected_contact_coverage": .8, "missing_contact_intervals": [2, 0], "source": SECRET}}]}
    return plan, records, evaluation


class TrackingEnvironment(unittest.TestCase):
    def setUp(self):
        # Each test declares tracking intent; developer shell settings cannot
        # accidentally select a real destination, overlay or ambient metadata.
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith("WANDB_") and key != "HOI_WANDB_SDK_PATH"}
        clean["WANDB_API_KEY"] = SECRET
        self.environment = patch.dict(os.environ, clean, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)


class TrackingTest(TrackingEnvironment):
    def test_wm_quality_diagnostic_is_distinct_from_actual_policy_schedule(self):
        plan, rows, evaluation = inputs()
        plan["policy_update_schedule"] = {"wm_quality_gate_enabled": False, "private": SECRET}
        rows[0]["policy_gate"]["passed"] = True
        rows[0]["wm_quality"] = {"passed": False, "blockers": [SECRET], "controls_policy_updates": False}
        payload = build_payload(plan, rows, evaluation)
        self.assertFalse(payload["config"]["training/wm_quality_gate_enabled"])
        metrics = payload["events"][0]["metrics"]
        self.assertTrue(metrics["gate/policy_passed"])
        self.assertFalse(metrics["diagnostics/wm_quality_passed"])
        self.assertFalse(payload["summary"]["last/diagnostics/wm_quality_passed"])
        self.assertNotIn(SECRET, json.dumps(payload, allow_nan=False))

    def test_variants_and_curated_new_diagnostics_exclude_event_times_and_annotations(self):
        plan,rows,evaluation=inputs()
        plan["config"]["variant"]="no_contact_objective"
        evaluation["variant"]="no_contact_objective"
        case=evaluation["cases"][0]
        case["objective_normalized_return"]=.8
        case["source_flag_comparison"]={"valid_intervals":[3,4],"mismatch_fraction_valid":[.2,.3],"body_names":SECRET}
        case["contact"]["actual_transition_events"]={"channels":{"object_ground_support":{
            "onset_count":2,"onset_endpoint_times_s":[.2,.6],"raw":SECRET}}}
        case["simulator_imagined_reward_gap"]={"horizons":{"8":{"valid_intervals":8,"absolute_reward_gap_mean":.02,"actions":SECRET}}}
        payload=build_payload(plan,rows,evaluation)
        self.assertEqual(payload["config"]["training/variant"],"no_contact_objective")
        self.assertEqual(payload["config"]["evaluation/variant"],"no_contact_objective")
        self.assertEqual(payload["summary"]["evaluation/case_00000/contact/actual_events/object_ground/onset_count"],2)
        self.assertEqual(payload["summary"]["evaluation/case_00000/paired_reward_gap/8/absolute_reward_gap_mean"],.02)
        self.assertNotIn(SECRET,json.dumps(payload))
        self.assertNotIn("onset_endpoint_times_s",json.dumps(payload))

    def test_unknown_strings_paths_raw_data_and_credentials_are_dropped(self):
        payload = build_payload(*inputs())
        self.assertNotIn("training/wm_quality_gate_enabled", payload["config"])
        self.assertNotIn("diagnostics/wm_quality_passed", payload["events"][0]["metrics"])
        outgoing = json.dumps(payload, allow_nan=False)
        self.assertNotIn(SECRET, outgoing)
        self.assertNotIn("sourcecode", outgoing)
        self.assertNotIn("raw_motion", outgoing)
        self.assertNotIn("robot_usd", outgoing)
        self.assertNotIn("rollout/", outgoing)
        self.assertEqual(payload["config"]["source/content_sha256"], "c" * 64)
        self.assertNotIn("files", payload["config"])
        self.assertEqual(payload["events"][0]["metrics"]["wm/total"], 2.)
        self.assertEqual(payload["summary"]["evaluation/case_00000/object_position_m/mean"], .04)
        self.assertTrue(all(value is False for value in payload["privacy"].values()))

    def test_nonfinite_values_and_strings_in_known_numeric_fields_are_not_uploaded(self):
        plan, records, evaluation = inputs()
        plan["config"].update(lr=float("nan"), num_envs=SECRET, amp=SECRET, experiment_kind=SECRET)
        records[0]["actor"].update(pi_loss=float("inf"), v_loss=SECRET)
        evaluation["aggregate"]["human_normalized_return"] = float("-inf")
        evaluation["cases"][0]["object_position_m"]["p95"] = float("nan")
        payload = build_payload(plan, records, evaluation)
        self.assertNotIn(SECRET, json.dumps(payload, allow_nan=False))
        self.assertNotIn("training/lr", payload["config"])
        self.assertNotIn("training/num_envs", payload["config"])
        self.assertNotIn("actor/pi_loss", payload["events"][0]["metrics"])
        self.assertNotIn("evaluation/human_normalized_return", payload["summary"])

    def test_every_batch_gets_a_new_UUID_and_ordered_iterations(self):
        first, second = build_payload(*inputs()), build_payload(*inputs())
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(len(first["run_id"]), 32)
        self.assertEqual(first["resume"], "never")
        plan, rows, evaluation = inputs()
        with self.assertRaisesRegex(ValueError, "increasing"):
            build_payload(plan, rows + rows, evaluation)

    def test_dry_run_does_not_import_wandb_or_call_network(self):
        plan, records, evaluation = inputs()
        import builtins
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "wandb" or name.startswith("wandb."):
                self.fail("dry-run must not import the SDK")
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in (("plan.json", plan), ("evaluation.json", evaluation)):
                (root / name).write_text(json.dumps(value))
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            output = root / "tracking"
            with patch("builtins.__import__", side_effect=guarded), patch("socket.socket", side_effect=AssertionError("network forbidden")):
                payload = main(["--run-dir", str(root), "--evaluation", str(root / "evaluation.json"), "--output", str(output), "--dry-run"])
            self.assertEqual(json.loads((output / "receipt.json").read_text())["network_calls"], 0)
            self.assertNotIn(SECRET, (output / "payload.json").read_text())
            self.assertEqual(payload["resume"], "never")
            with self.assertRaises(FileExistsError):
                main(["--run-dir", str(root), "--output", str(output)])

    def test_online_wrapper_uses_only_curated_values_and_disables_capture(self):
        class Settings:
            _disable_stats: bool
            _disable_meta: bool
            _save_requirements: bool
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        class Run:
            url = "https://example.invalid/private-run"
            def __init__(self):
                self.summary, self.logs, self.finished = {}, [], False
            def log(self, metrics, step):
                self.logs.append((step, metrics))
            def finish(self):
                self.finished = True
        run, calls = Run(), []
        def init(**kwargs):
            calls.append(kwargs)
            return run
        fake = SimpleNamespace(Settings=Settings, init=init)
        payload = build_payload(*inputs())
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": fake}), patch.dict(os.environ, {"WANDB_API_KEY": SECRET}):
            previous = os.environ.get("WANDB_MODE")
            receipt = send_online(payload, temp)  # fake SDK only; never a real network call
            self.assertEqual(os.environ.get("WANDB_MODE"), previous)
        self.assertEqual(calls[0]["resume"], "never")
        self.assertEqual(calls[0]["entity"], "tianzizh-university-of-pennsylvania")
        self.assertEqual(calls[0]["project"], "SuperMimic_Upstream_Reproduction")
        settings = calls[0]["settings"].kwargs
        self.assertTrue(settings["disable_code"])
        self.assertTrue(settings["disable_git"])
        self.assertTrue(settings["_disable_stats"])
        self.assertTrue(settings["_disable_meta"])
        self.assertFalse(settings["_save_requirements"])
        self.assertFalse(settings["save_code"])
        self.assertTrue(settings["disable_job_creation"])
        self.assertEqual(settings["console"], "off")
        self.assertEqual(settings["host"], "")
        self.assertEqual(settings["program"], "hoi_v1_metrics")
        self.assertEqual(settings["program_relpath"], "hoi_v1_metrics")
        self.assertEqual(settings["ignore_globs"], ("*", "**/*"))
        self.assertNotIn(SECRET, json.dumps(calls[0]["config"]))
        self.assertNotIn(SECRET, json.dumps(run.summary))
        self.assertTrue(run.finished)
        self.assertFalse(receipt["artifact_uploads"])
        self.assertEqual(settings["base_url"], _legacy_base_url())

    def test_direct_SDK_boundary_rejects_mutated_payload_before_import(self):
        payload = build_payload(*inputs())
        payload["config"]["private_source"] = SECRET
        with patch.dict("sys.modules", {"wandb": None}):
            with self.assertRaisesRegex(ValueError, "whitelist"):
                send_online(payload, ".")

    def test_unknown_SDK_cannot_silently_enable_metadata_capture(self):
        class UnknownSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        with self.assertRaisesRegex(RuntimeError, "cannot prove"):
            _privacy_settings(SimpleNamespace(Settings=UnknownSettings), "https://example.invalid")

    def test_explicit_project_override_and_destination_drift_rejected(self):
        self.assertEqual(destination(), ("tianzizh-university-of-pennsylvania", "SuperMimic_Upstream_Reproduction"))
        with patch.dict(os.environ, {"WANDB_PROJECT": "SuperMimic_HOI_v1_20261023"}):
            payload = build_payload(*inputs())
            self.assertEqual(payload["project"], "SuperMimic_HOI_v1_20261023")
            with patch.dict(os.environ, {"WANDB_PROJECT": "different_project"}), patch.dict("sys.modules", {"wandb": None}):
                with self.assertRaisesRegex(ValueError, "destination"):
                    send_online(payload, ".")
        for value in ("org/project", "project,key=secret", "project\n", "https://private.invalid/project"):
            with self.subTest(value=value), patch.dict(os.environ, {"WANDB_PROJECT": value}):
                with self.assertRaisesRegex(ValueError, "entity/project"):
                    build_payload()


def fake_sdk(*, fail_init=False):
    """No network, artifacts, source capture or implicit file methods exist."""
    class Settings:
        x_disable_stats: bool
        x_disable_meta: bool
        x_save_requirements: bool
        x_disable_machine_info: bool
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class Run:
        url = "https://example.invalid/new-private-run"
        def __init__(self):
            self.summary, self.logs, self.finish_calls = {}, [], 0
        def log(self, metrics, step):
            self.logs.append((step, deepcopy(metrics)))
        def finish(self):
            self.finish_calls += 1
    run, calls, environments = Run(), [], []
    def init(**kwargs):
        calls.append(kwargs)
        environments.append(dict(os.environ))
        if fail_init:
            raise RuntimeError("synthetic SDK init error")
        return run
    return SimpleNamespace(Settings=Settings, init=init), run, calls, environments


class LiveTrackingTest(TrackingEnvironment):
    def test_live_uses_one_fresh_run_finite_metrics_and_checkpoint_summary_only(self):
        sdk, run, calls, environments = fake_sdk()
        plan, rows, _ = inputs()
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}), \
                patch.dict(os.environ, {"WANDB_SYNC": "1", "WANDB_PROJECT": "SuperMimic_HOI_v1_20261023",
                                        "WANDB_NAME": SECRET, "WANDB_NOTES": SECRET, "WANDB_TAGS": SECRET,
                                        "WANDB_PROGRAM": SECRET, "WANDB_RESUME": "must-not-resume",
                                        "WANDB_ERROR_REPORTING": "true"}):
            output = Path(temp) / "tracking"
            with patch("socket.socket", side_effect=AssertionError("no real network in mock test")), live_training(plan, output) as tracker:
                tracker.log_record(rows[0])
                tracker.checkpoint(500, 5000, 0)
                second = deepcopy(rows[0])
                second["iteration"] = 50
                second["actor"]["pi_loss"] = float("inf")
                tracker.log_record(second)
                with self.assertRaisesRegex(ValueError, "increase"):
                    tracker.log_record(second)
            self.assertEqual(os.environ["WANDB_NAME"], SECRET)
            self.assertEqual(os.environ["WANDB_ERROR_REPORTING"], "true")
            receipt = json.loads((output / "live_finish.json").read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["last_metric_step"], 50)
            self.assertEqual(receipt["run_id"], calls[0]["id"])
            self.assertNotIn(SECRET, "\n".join(file.read_text() for file in output.glob("*.json")))
            with self.assertRaises(FileExistsError):
                with live_training(plan, output):
                    self.fail("exclusive output must reject a duplicate producer")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["resume"], calls[0]["project"], calls[0]["job_type"]),
                         ("never", "SuperMimic_HOI_v1_20261023", "hoi_v1_training"))
        self.assertEqual([step for step, _ in run.logs], [1, 50])
        self.assertNotIn("actor/pi_loss", run.logs[1][1])
        self.assertNotIn(SECRET, json.dumps([run.logs, run.summary, calls[0]["config"]], allow_nan=False))
        self.assertEqual(run.finish_calls, 1)
        self.assertEqual(run.summary["checkpoint/iteration"], 500)
        self.assertEqual(run.summary["checkpoint/wm_updates_this_run"], 5000)
        self.assertEqual(run.summary["checkpoint/policy_updates_this_run"], 0)
        self.assertTrue(run.summary["run/completed"])
        self.assertEqual(environments[0]["WANDB_ERROR_REPORTING"], "false")
        self.assertNotIn("WANDB_NAME", environments[0])
        self.assertNotIn("WANDB_NOTES", environments[0])
        self.assertNotIn("WANDB_TAGS", environments[0])
        self.assertNotIn("WANDB_PROGRAM", environments[0])
        self.assertNotIn("WANDB_RESUME", environments[0])
        settings = calls[0]["settings"].kwargs
        self.assertEqual(settings["base_url"], _legacy_base_url())
        self.assertTrue(settings["x_disable_meta"])
        self.assertTrue(settings["x_disable_stats"])
        self.assertFalse(settings["x_save_requirements"])
        self.assertTrue(settings["x_disable_machine_info"])
        self.assertEqual(settings["host"], "")
        self.assertEqual(settings["program"], "hoi_v1_metrics")
        self.assertEqual(settings["program_relpath"], "hoi_v1_metrics")
        self.assertEqual(settings["console"], "off")

    def test_disabled_live_never_imports_SDK_or_creates_tracking_output(self):
        import builtins
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if "wandb" in name or "tracking_bootstrap" in name:
                self.fail("disabled tracking must not import SDK/bootstrap")
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"WANDB_SYNC": "0"}), \
                patch("builtins.__import__", side_effect=guarded), patch("socket.socket", side_effect=AssertionError("network forbidden")):
            output = Path(temp) / "tracking"
            with live_training(inputs()[0], output) as tracker:
                self.assertIsNone(tracker)
            self.assertFalse(output.exists())

    def test_exception_finishes_same_run_and_restores_environment_without_retry(self):
        sdk, run, calls, _ = fake_sdk()
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}), \
                patch.dict(os.environ, {"WANDB_SYNC": "1", "WANDB_MODE": "disabled", "WANDB_NOTES": SECRET}):
            output = Path(temp) / "tracking"
            with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                with live_training(inputs()[0], output) as tracker:
                    tracker.log_record(inputs()[1][0])
                    raise RuntimeError("training interrupted")
            self.assertEqual(os.environ["WANDB_MODE"], "disabled")
            self.assertEqual(os.environ["WANDB_NOTES"], SECRET)
            self.assertEqual(json.loads((output / "live_finish.json").read_text())["status"], "failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(run.finish_calls, 1)
        self.assertFalse(run.summary["run/completed"])

    def test_failed_init_restores_capture_environment_and_does_not_retry(self):
        sdk, run, calls, _ = fake_sdk(fail_init=True)
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}), \
                patch.dict(os.environ, {"WANDB_SYNC": "1", "WANDB_NOTES": SECRET}):
            with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                with live_training(inputs()[0], Path(temp) / "tracking"):
                    self.fail("failed init must not proceed to training")
            self.assertEqual(os.environ["WANDB_NOTES"], SECRET)
            receipt = json.loads((Path(temp) / "tracking/live_failure.json").read_text())
            self.assertEqual((receipt["status"], receipt["failure_phase"]), ("tracking_failed", "initialization"))
            self.assertNotIn(SECRET, json.dumps(receipt))
        self.assertEqual(len(calls), 1)
        self.assertEqual(run.finish_calls, 0)

    def test_metric_or_shutdown_error_keeps_sanitized_failure_and_finish_receipts(self):
        for failing_phase in ("metric_sync", "summary_sync", "shutdown"):
            with self.subTest(failing_phase=failing_phase):
                sdk, run, calls, _ = fake_sdk()
                def fail(*args, **kwargs):
                    raise RuntimeError(SECRET)
                if failing_phase == "summary_sync":
                    class FailingSummary(dict):
                        def __setitem__(self, *args):
                            raise RuntimeError(SECRET)
                    run.summary = FailingSummary()
                else:
                    setattr(run, "log" if failing_phase == "metric_sync" else "finish", fail)
                with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}), \
                        patch.dict(os.environ, {"WANDB_SYNC": "1", "WANDB_ERROR_REPORTING": "true"}):
                    output = Path(temp) / "tracking"
                    with self.assertRaises(RuntimeError) as raised:
                        with live_training(inputs()[0], output) as tracker:
                            tracker.log_record(inputs()[1][0])
                    failure = json.loads((output / "live_failure.json").read_text())
                    finish = json.loads((output / "live_finish.json").read_text())
                    self.assertEqual(failure["failure_phase"], failing_phase)
                    self.assertEqual(finish["status"], "tracking_failed")
                    self.assertNotIn(SECRET, str(raised.exception))
                    self.assertNotIn(SECRET, "\n".join(file.read_text() for file in output.glob("*.json")))
                    self.assertEqual(os.environ["WANDB_ERROR_REPORTING"], "true")
                self.assertEqual(len(calls), 1)

    def test_batch_failure_receipt_has_no_arbitrary_SDK_error_text(self):
        sdk, _, calls, _ = fake_sdk()
        sdk.init = lambda **kwargs: (_ for _ in ()).throw(RuntimeError(SECRET))
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}):
            with self.assertRaises(RuntimeError) as raised:
                send_online(build_payload(*inputs()), temp)
            receipt = json.loads((Path(temp) / "online_failure.json").read_text())
        self.assertEqual((receipt["status"], receipt["failure_phase"]), ("tracking_failed", "batch_sync"))
        self.assertNotIn(SECRET, json.dumps(receipt))
        self.assertNotIn(SECRET, str(raised.exception))

    def test_shutdown_failure_does_not_replace_the_original_learner_exception(self):
        sdk, run, _, _ = fake_sdk()
        def fail():
            raise RuntimeError(SECRET)
        run.finish = fail
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", {"wandb": sdk}), patch.dict(os.environ, {"WANDB_SYNC": "1"}):
            output = Path(temp) / "tracking"
            with self.assertRaisesRegex(ArithmeticError, "synthetic learner failure"):
                with live_training(inputs()[0], output):
                    raise ArithmeticError("synthetic learner failure")
            receipt = json.loads((output / "live_finish.json").read_text())
            self.assertEqual((receipt["status"], receipt["failure_phase"]), ("tracking_failed", "shutdown"))
            self.assertNotIn(SECRET, json.dumps(receipt))


if __name__ == "__main__":
    unittest.main()
