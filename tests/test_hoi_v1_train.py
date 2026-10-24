"""CPU orchestration with synthetic workers and the real learner/checkpoint.

Scripted gate reports test control flow only. They are explicitly not native
physics evidence and do not authorize or execute a physical training run.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.v1.checkpoint import load_checkpoint
from hoi.v1.state import flatten_state, flatten_object, unflatten_state, unflatten_object
from supermimic.mdp import normalize_xi
from hoi.v1.train import main, read_config, train_loop
from tests.test_hoi_v1_learner import make_learner, transition


PHYSICS = {"hand_collision": True, "self_collision": True}
DATA_ID = {"scope": "synthetic_CPU_orchestration_only", "frozen_identity": "test"}


def scripted_wm_report(passed):
    metric = {"valid_samples": 20, "object_normalized_error": .1 if passed else .8,
              "zero_increment_normalized_error": .5, "constant_velocity_normalized_error": .4}
    return {"finite": True, "scope": "training_reference_bins_reserved_from_wm_fit_not_eval80_or_test40", "real_executed_actions": True,
            "horizon_reports": {str(k): {**metric, "phases": {p: dict(metric) for p in ("no_contact", "contact", "onset", "release")}}
                                for k in (1,8,32)},
            "contact_diagnostics": {"positive_counts": [20]*3, "negative_counts": [20]*3, "precision": [.8]*3,
                                    "recall": [.8]*3, "brier_per_channel": [.1]*3, "prevalence_baseline_brier": [.25]*3}}


class SyntheticWorkers:
    def __init__(self, learner):
        self.learner, self.steps, self.snapshot_calls = learner, 0, 0
        first = transition(learner, [0,70], 0)
        self.S, self.O = unflatten_state(first["s"]), unflatten_object(first["o"])
        self.previous = torch.zeros(2,29)

    def snapshot(self):
        self.snapshot_calls += 1
        value = transition(self.learner, [self.steps,70+self.steps], self.steps)
        value.update(s=flatten_state(self.S), o=flatten_object(self.O), a_prev=self.previous.clone())
        return value

    def step(self, action):
        learner = self.learner
        frame = torch.tensor([self.steps,70+self.steps])
        value = transition(learner, frame, self.steps, a=action, a_prev=self.previous)
        S, O = self.S, self.O
        next_s = (S[0] + S[2] * .02, S[1], S[2] + action[:, :3] * .001,
                  S[3], S[4] + action * .01, S[5] + action * .01)
        next_o = (O[0] + O[2] * .02 + action[:, :3] * .002, O[1], O[2] + action[:, :3] * .01, O[3])
        actual = torch.full((2,3), float((self.steps // 4) % 2))
        reward, terms = learner.mdp.reward(next_s, next_o, frame + 1, action, self.previous, actual, terms=True)
        value.update(s=flatten_state(S), s1=flatten_state(next_s), o=flatten_object(O), o1=flatten_object(next_o),
                     contact=actual, impulse=actual * .04, r=reward.detach(), r_human=terms["human_reward"].detach(),
                     r_hoi=(reward - terms["human_reward"]).detach())
        self.S, self.O = tuple(x.detach() for x in next_s), tuple(x.detach() for x in next_o)
        self.previous = action.detach().clone()
        self.steps += 1
        return value


class TrainingFlowTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        torch.manual_seed(101)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def learner(self, iterations=2):
        learner = make_learner()
        learner.config = replace(learner.config, iterations=iterations, warmup_steps=8, wm_only_iters=0,
                                 ckpt_every=1, log_every=1, gate_every=1)
        return learner

    def run_loop(self, learner, workers, name, passed=True, native=True):
        output = self.root / name
        output.mkdir()
        with patch.object(learner, "wm_validation", side_effect=lambda: scripted_wm_report(passed)), redirect_stdout(io.StringIO()):
            result = train_loop(learner, workers, output, physics=PHYSICS, data_identity=DATA_ID,
                                initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": native})
        return output, result

    def test_native_gate_false_prevents_any_collection_or_normalization(self):
        learner = self.learner()
        workers = SyntheticWorkers(learner)
        count = learner.pi.obs_norm.count.clone()
        with self.assertRaisesRegex(ValueError, "native evidence"):
            self.run_loop(learner, workers, "closed_native", native=False)
        self.assertEqual((workers.steps, workers.snapshot_calls, learner.buffer.size), (0,0,0))
        torch.testing.assert_close(count, learner.pi.obs_norm.count)

    def test_prefill_wm_warmup_still_saves_complete_checkpoint(self):
        learner = self.learner()
        learner.config = replace(learner.config, wm_only_iters=500)
        workers = SyntheticWorkers(learner)
        before_policy = {name: value.detach().clone() for name, value in learner.pi.named_parameters()}
        output, result = self.run_loop(learner, workers, "wm_only", passed=False)
        self.assertEqual(result["policy_updates"], 0)
        self.assertEqual(result["wm_updates"], 2)
        self.assertEqual(workers.steps, 8 + 2 + 3)
        for name, parameter in learner.pi.named_parameters():
            torch.testing.assert_close(parameter, before_policy[name], rtol=0, atol=0)
        saved = torch.load(result["checkpoint"], weights_only=False, map_location="cpu")
        self.assertEqual(set(saved["optimizers"]), {"pi","wm","critic"})
        self.assertEqual(saved["iteration"], 2)
        self.assertFalse(saved["extra"]["wm_gate"]["passed"])
        self.assertFalse(saved["extra"]["policy_gate"]["passed"])
        self.assertFalse(saved["extra"]["policy_update_schedule"]["wm_quality_gate_enabled"])
        self.assertEqual(saved["extra"]["wm_updates_this_run"], 2)
        self.assertEqual(saved["extra"]["policy_updates_this_run"], 0)
        self.assertTrue(json.loads((output / "prefill.json").read_text())["fresh_replay"])

    def test_resume_restores_and_updates_despite_failed_wm_quality_then_freshly_prefills(self):
        learner = self.learner()
        workers = SyntheticWorkers(learner)
        _, first = self.run_loop(learner, workers, "first")
        self.assertEqual(first["policy_updates"], 2)
        saved = torch.load(first["checkpoint"], weights_only=False, map_location="cpu")
        resumed = self.learner(iterations=4)
        load_checkpoint(saved, resumed.pi, resumed.wm, resumed.critic, resumed.optimizers,
                        physics=PHYSICS, data_identity=DATA_ID)
        resumed.iteration, resumed.sim_steps = saved["iteration"], saved["extra"]["sim_steps"]
        for name in ("contact_positive", "contact_negative"):
            getattr(resumed, name).copy_(saved["extra"][name])
        self.assertEqual(resumed.buffer.size, 0)
        torch.testing.assert_close(resumed.pi.obs_norm.count, saved["pi"]["obs_norm.count"], atol=0, rtol=0)
        self.assertTrue(resumed.optimizers["wm"].state)
        count = float(resumed.pi.obs_norm.count)
        output, result = self.run_loop(resumed, SyntheticWorkers(resumed), "continuation", passed=False)
        self.assertEqual(result["iteration"], 4)
        self.assertEqual(result["policy_updates"], 2)
        self.assertEqual(resumed.buffer.size, 13)
        continued = torch.load(result["checkpoint"], weights_only=False, map_location="cpu")
        self.assertEqual(continued["extra"]["policy_updates_this_run"], 2)
        self.assertEqual(continued["extra"]["wm_updates_this_run"], 2)
        self.assertFalse(continued["extra"]["wm_gate"]["passed"])
        self.assertTrue(continued["extra"]["policy_gate"]["passed"])
        self.assertGreater(float(resumed.pi.obs_norm.count), count)
        self.assertEqual(result["sim_steps"], first["sim_steps"] + 26)
        prefill = json.loads((output / "prefill.json").read_text())
        self.assertEqual(prefill["vector_steps"], 8)
        self.assertTrue(prefill["fresh_replay"])
        self.assertFalse(prefill["bitwise_continuation"])

    def test_failed_quality_does_not_delay_first_post_warmup_policy_update(self):
        learner = self.learner(iterations=4)
        learner.config = replace(learner.config, wm_only_iters=2)
        before = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in (("pi", learner.pi), ("critic", learner.critic))}
        output, result = self.run_loop(learner, SyntheticWorkers(learner), "quality_diagnostic", passed=False)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["policy_updates"] for row in rows], [0, 0, 1, 2])
        self.assertEqual([row["actor"] is not None for row in rows], [False, False, True, True])
        self.assertTrue(all(not row["wm_quality"]["passed"] for row in rows))
        self.assertTrue(all(not row["wm_quality"]["controls_policy_updates"] for row in rows))
        self.assertEqual((result["wm_updates"], result["policy_updates"]), (4, 2))
        for name, model in (("pi", learner.pi), ("critic", learner.critic)):
            self.assertTrue(any(not torch.equal(value, before[name][key]) for key, value in model.named_parameters()))

    def test_quality_regression_does_not_pause_actor_or_critic(self):
        learner = self.learner(iterations=3)
        output = self.root / "quality_regression"
        output.mkdir()
        reports = [scripted_wm_report(value) for value in (True, False, False)]
        with patch.object(learner, "wm_validation", side_effect=reports), redirect_stdout(io.StringIO()):
            result = train_loop(learner, SyntheticWorkers(learner), output, physics=PHYSICS, data_identity=DATA_ID,
                                initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": True})
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["policy_updates"] for row in rows], [1, 2, 3])
        self.assertEqual([row["wm_quality"]["passed"] for row in rows], [True, False, False])
        self.assertTrue(all(row["policy_gate"]["passed"] and row["actor"] is not None for row in rows))
        self.assertEqual(result["policy_updates"], 3)

    def test_sparse_holdout_is_recorded_without_blocking_post_warmup_updates(self):
        learner = self.learner()
        output = self.root / "sparse_holdout"
        output.mkdir()
        with patch.object(learner, "wm_validation", side_effect=ValueError("insufficient holdout replay coverage")), redirect_stdout(io.StringIO()):
            result = train_loop(learner, SyntheticWorkers(learner), output, physics=PHYSICS, data_identity=DATA_ID,
                                initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": True})
        self.assertEqual((result["wm_updates"], result["policy_updates"]), (2, 2))
        diagnostic = json.loads((output / "wm_gate_000002.json").read_text())
        self.assertEqual(diagnostic["validation"]["status"], "insufficient_real_holdout_coverage")
        self.assertFalse(diagnostic["gate"]["controls_policy_updates"])

    def test_nonfinite_holdout_is_a_hard_error_before_policy_update(self):
        learner = self.learner()
        output = self.root / "nonfinite_holdout"
        output.mkdir()
        report = scripted_wm_report(True)
        report["finite"] = False
        with patch.object(learner, "wm_validation", return_value=report), patch.object(learner, "policy_update") as policy, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite WM holdout"):
                train_loop(learner, SyntheticWorkers(learner), output, physics=PHYSICS, data_identity=DATA_ID,
                           initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": True})
        policy.assert_not_called()
        self.assertTrue((output / "wm_gate_000001.json").is_file())

    def test_config_preserves_formal_defaults_and_marks_smoke_deviations(self):
        config, deviations = read_config()
        self.assertEqual((config.num_envs, config.wm_window, config.horizon, config.wm_batch, config.pi_batch), (4096,8,32,2048,4096))
        self.assertEqual(deviations, {})
        path = self.root / "config.json"
        path.write_text(json.dumps({"num_envs": 2}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "smoke"):
            read_config(path)
        path.write_text(json.dumps({"num_envs": 2, "experiment_kind": "smoke"}), encoding="utf-8")
        smoke, deviations = read_config(path)
        self.assertEqual(smoke.num_envs, 2)
        self.assertEqual(deviations["num_envs"]["default"], 4096)
        path.write_text(json.dumps({"unknown_option": 1}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown"):
            read_config(path)

    def test_observe_reserves_heldout_rows_from_wm_normalizers_and_class_counts(self):
        learner = self.learner()
        value = transition(learner, [0,70], 0)
        # A conspicuously different heldout row would corrupt means/counts if
        # observe mixed validation transitions into model fitting statistics.
        value["s"][0, 13:42] = 100
        value["a"][0] = 100
        value["xi_o"][0] = 100
        value["contact"][0] = 1
        value["contact"][1] = 0
        expected = deepcopy(learner.wm)
        ids = torch.tensor([1])
        expected.update_normalizers(unflatten_state(value["s"][ids]), unflatten_object(value["o"][ids]), value["a"][ids],
                                    normalize_xi(value["xi"][ids]), value["xi_o"][ids], learner.shapes.points(value["asset_id"][ids]),
                                    unflatten_state(value["s1"][ids]), unflatten_object(value["o1"][ids]))
        learner.observe(value)
        for name in ("in_norm", "out_norm", "object_in_norm", "object_out_norm"):
            for key, tensor in getattr(expected, name).state_dict().items():
                torch.testing.assert_close(getattr(learner.wm, name).state_dict()[key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(learner.contact_positive, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(learner.contact_negative, torch.ones(3, dtype=torch.float64))
        # The raw reserved transition remains available to WM diagnostics.
        self.assertEqual(float(learner.buffer.data["a"][0,0,0]), 100)

    def test_subset_formal_is_rejected_before_worker_or_output_creation(self):
        header = {"partition": "train", "scope": {"is_subset": True}, "assets": [], "clips": [], "bundle_content_sha256": "a"*64}
        output = self.root / "formal_subset"
        with patch("hoi.v1.evaluate._read_bundle_header", return_value=header), patch.dict(os.environ):
            with self.assertRaisesRegex(ValueError, "subset"):
                main(["--bundle", str(self.root / "input.json"), "--output", str(output), "--preflight"])
        self.assertFalse(output.exists())

    def test_wm_does_not_consume_masked_reset_actions_or_other_assets(self):
        learner = self.learner()
        rows = [transition(learner, [70 + step, 80 + step], step) for step in range(8)]
        window = {key: torch.stack([row[key] for row in rows], 1) for key in rows[0]}
        window["mask"] = torch.ones(2, 8)
        window["mask"][0, 2:] = 0
        window["mask"][1, 4:] = 0
        inactive = ~window["mask"].bool()
        window["a"][inactive] = float("nan")
        window["xi_o"][inactive] = float("nan")
        window["asset_id"][inactive] = 99999
        states, objects, logits = learner.wm_rollout(window)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in states + objects + (logits,)))
        torch.testing.assert_close(objects[0][0, 2:], objects[0][0, 1:2].expand(6, -1))
        torch.testing.assert_close(objects[0][1, 4:], objects[0][1, 3:4].expand(4, -1))
        torch.testing.assert_close(logits[inactive], torch.zeros_like(logits[inactive]))

    def test_periodic_eval_dispatches_only_after_immutable_checkpoint_is_saved(self):
        learner = self.learner(iterations=4)
        learner.config = replace(learner.config, eval_every=2, ckpt_every=3, wm_only_iters=500)
        output = self.root / "periodic"
        output.mkdir()
        dispatched = []
        def submit(path, iteration):
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["iteration"], iteration)
            self.assertTrue(path.is_file())
            dispatched.append(iteration)
            return {"iteration": iteration, "status": "synthetic_dispatch_no_job_submitted"}
        with patch.object(learner, "wm_validation", return_value=scripted_wm_report(False)), redirect_stdout(io.StringIO()):
            result = train_loop(learner, SyntheticWorkers(learner), output, physics=PHYSICS, data_identity=DATA_ID,
                                initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": True},
                                evaluation_callback=submit)
        self.assertEqual(dispatched, [2, 4])
        self.assertEqual([row["iteration"] for row in result["evaluation_dispatches"]], [2, 4])
        self.assertEqual(result["policy_updates"], 0)
        self.assertEqual(sorted(path.name for path in output.glob("checkpoint*.pt")),
                         ["checkpoint_000002.pt", "checkpoint_000003.pt", "checkpoint_000004.pt"])

    def test_live_hooks_match_first_step_log_cadence_and_post_save_counts(self):
        learner = self.learner(iterations=4)
        learner.config = replace(learner.config, log_every=3, ckpt_every=2, wm_only_iters=500)
        output = self.root / "live_hooks"
        output.mkdir()
        records, checkpoints = [], []
        class Tracker:
            def log_record(inner, record):
                records.append(deepcopy(record))
            def checkpoint(inner, iteration, wm_updates, policy_updates):
                path = output / f"checkpoint_{iteration:06d}.pt"
                saved = torch.load(path, weights_only=False, map_location="cpu")
                self.assertEqual(saved["iteration"], iteration)
                self.assertEqual(saved["extra"]["wm_updates_this_run"], wm_updates)
                self.assertEqual(saved["extra"]["policy_updates_this_run"], policy_updates)
                checkpoints.append((iteration, wm_updates, policy_updates))
        with patch.object(learner, "wm_validation", return_value=scripted_wm_report(False)), redirect_stdout(io.StringIO()):
            result = train_loop(learner, SyntheticWorkers(learner), output, physics=PHYSICS, data_identity=DATA_ID,
                                initialization={"mode": "synthetic_CPU_test"}, native_gate={"passed": True}, tracker=Tracker())
        local_rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["iteration"] for row in records], [1, 3, 4])
        self.assertEqual(json.loads(json.dumps(records, allow_nan=False)), local_rows)
        self.assertEqual(checkpoints, [(2, 2, 0), (4, 4, 0)])
        self.assertEqual((result["wm_updates"], result["policy_updates"]), (4, 0))
        self.assertTrue(all(torch.isfinite(torch.tensor(loss["total"])) for row in records for loss in row["wm"]))

    def test_continuation_observes_its_first_iteration_even_between_regular_logs(self):
        learner = self.learner(iterations=5)
        learner.iteration = 3
        learner.config = replace(learner.config, log_every=50, ckpt_every=500, wm_only_iters=500)
        output, result = self.run_loop(learner, SyntheticWorkers(learner), "first_continuation_log", passed=False)
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["iteration"] for row in records], [4, 5])
        self.assertEqual(records[0]["wm_updates"], 1)
        self.assertEqual(result["wm_updates"], 2)
        self.assertEqual([path.name for path in output.glob("checkpoint*.pt")], ["checkpoint_000005.pt"])


if __name__ == "__main__":
    unittest.main()
