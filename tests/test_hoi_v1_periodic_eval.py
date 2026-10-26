"""CPU dispatcher proofs; all scheduler calls are mocked, never submitted."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi.data.manifest import sha256_file
from hoi.v1.checkpoint import CHECKPOINT_SCHEMA
from hoi.v1.config import bundle_identity, json_identity
from hoi.v1.networks import FEATURE_SCHEMA
from hoi.v1.periodic_eval import schedule_evaluation, validate_canonical80, verify_dispatch
from hoi.v1.prepare import BUNDLE_SCHEMA, _digest_json


class PeriodicEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"files": {"hoi/v1/synthetic_fixture.py": "a" * 64}}
        self.source["content_sha256"] = json_identity(self.source["files"])
        self.addCleanup(patch.stopall)
        patch("hoi.v1.periodic_eval.source_identity", return_value=deepcopy(self.source)).start()
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith("WANDB_") and key != "HOI_WANDB_SDK_PATH"}
        clean.update(WANDB_SYNC="0", WANDB_API_KEY="CREDENTIAL_SENTINEL")
        patch.dict(os.environ, clean, clear=True).start()
        self.panel = [{"motion": f"synthetic_{i:03d}.pt", "motion_sha256": f"{i:064x}",
                       "source_group": f"group_{i}", "source_frames": 100 + i} for i in range(80)]
        self.manifest = {"profile": "canonical-v4", "panels": {"eval80": self.panel}}
        patch("hoi.v1.periodic_eval.read_manifest", return_value=self.manifest).start()
        patch("hoi.v1.evaluate.read_manifest", return_value=self.manifest).start()
        manifest, recipes = self.root / "manifest.json", self.root / "recipes.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        recipes.write_text("{}", encoding="utf-8")
        common = {"schema_version": BUNDLE_SCHEMA, "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
                  "recipes": {"path": str(recipes), "sha256": sha256_file(recipes)},
                  "timebase": {"source_hz": 60, "control_hz": 50}, "robot_fk": {"mjcf_sha256": "b" * 64},
                  "native_robot_asset": {"sha256": "c" * 64}, "reference_kinematics": {"urdf_sha256": "d" * 64},
                  "assets": [{"asset_id": 0, "object_asset": "synthetic", "exact_scale": [1., 1., 1.],
                              "collision_hash": "e" * 64, "native_asset_sha256": "f" * 64,
                              "physical_profile": {"mass": 1.}, "geometry_cache": {"sha256": "1" * 64}}],
                  "hands": {"geometry_cache": {"sha256": "2" * 64}}}
        self.train = {**deepcopy(common), "partition": "train", "scope": {"is_subset": False}, "clips": []}
        self.evaluation = {**deepcopy(common), "partition": "eval80",
                           "scope": {"available_motions": 80, "selected_motions": 80, "is_subset": False},
                           "clips": [{"source_motion": c["motion"], "source_motion_sha256": c["motion_sha256"],
                                      "source_group": c["source_group"], "source_frames": c["source_frames"]} for c in self.panel]}
        self.train_path, self.eval_path = self.root / "train.json", self.root / "eval.json"
        self.save_bundle(self.train_path, self.train)
        self.save_bundle(self.eval_path, self.evaluation)
        self.checkpoint = {"schema_version": CHECKPOINT_SCHEMA, "feature_schema": deepcopy(FEATURE_SCHEMA), "iteration": 5000,
                           "physics": {"hand_collision": True, "self_collision": True},
                           "data_identity": bundle_identity(self.train), "extra": {"source_identity": deepcopy(self.source)}}
        self.checkpoint["data_identity_sha256"] = json_identity(self.checkpoint["data_identity"])
        # This fixture exercises dispatch metadata, not network loading; the real
        # evaluator's strict model/state compatibility has its own CPU tests.
        self.ck = self.root / "checkpoint_005000.pt"
        torch.save(self.checkpoint, self.ck)
        self.launcher = self.root / "eval.sh"
        self.launcher.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        runtime, isaac = self.root / "runtime", self.root / "isaac"
        runtime.mkdir()
        isaac.mkdir()
        self.environment = {"SUPERMIMIC_ROOT": str(Path(__file__).resolve().parents[1]),
                            "SUPERMIMIC_RUNTIME": str(runtime), "ISAACLAB_PATH": str(isaac)}
        self.output = self.root / "evaluations"

    def save_bundle(self, path, header):
        header.pop("bundle_content_sha256", None)
        header["bundle_content_sha256"] = _digest_json(header)
        path.write_text(json.dumps(header), encoding="utf-8")

    def call(self, **kw):
        return schedule_evaluation(self.ck, 5000, self.train_path, self.eval_path,
                                   self.output, self.launcher, self.environment, **kw)

    def request(self, receipt):
        return json.loads(Path(receipt["dispatch"]).read_text(encoding="utf-8"))

    def test_checkpoint_reader_accepts_v11_mass_schema_without_dispatch(self):
        from hoi.v1.networks import feature_schema
        from hoi.v1.observations import MATCHING
        from hoi.v1.randomization import MATCHING_DR, randomization_contract
        from hoi.v1.periodic_eval import _checkpoint
        self.checkpoint["feature_schema"] = feature_schema(MATCHING, torso_mass_context=True)
        self.checkpoint["extra"]["config"] = {"domain_randomization_profile": MATCHING_DR}
        self.checkpoint["experiment_contract"] = {"observation_profile": MATCHING,
                                                "domain_randomization": randomization_contract(MATCHING_DR)}
        torch.save(self.checkpoint, self.ck)
        with patch("hoi.v1.periodic_eval.source_identity", return_value=self.source):
            digest, _, _, _ = _checkpoint(self.ck, 5000, self.train)
        self.assertEqual(digest, sha256_file(self.ck))

    def test_dry_run_reviewable_exact_hashes_no_scheduler_or_credentials(self):
        with patch("hoi.v1.periodic_eval.subprocess.run", side_effect=AssertionError("no scheduler in dry run")):
            result = self.call(dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["scheduler_calls"], 0)
        request = self.request(result)
        self.assertEqual(request["checkpoint"]["sha256"], sha256_file(self.ck))
        self.assertEqual(request["source_content_sha256"], self.source["content_sha256"])
        self.assertEqual(request["tracking"]["resume"], "never")
        self.assertEqual(request["tracking"]["run_id_policy"], "fresh_uuid_at_sync")
        self.assertNotIn("run_id", request["tracking"])
        self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(request))
        self.assertEqual(set(request["environment"]), set(self.environment))
        self.assertFalse((Path(result["dispatch"]).parent / "submit.claim").exists())
        self.assertFalse(Path(result["output"]).exists())
        self.assertTrue(verify_dispatch(result["dispatch"])["verified"])

    def test_submit_independent_once_then_retry_returns_original_job(self):
        dry = self.call(dry_run=True)
        completed = subprocess.CompletedProcess([], 0, "123456;syntheticcluster\n", "")
        with patch("hoi.v1.periodic_eval.shutil.which", return_value="/opt/slurm/bin/sbatch"), \
                patch("hoi.v1.periodic_eval.subprocess.run", return_value=completed) as run:
            result = self.call()
            retry = self.call()
        self.assertEqual(result, retry)
        self.assertEqual(result["job_id"], "123456")
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["evaluation_job_key"], dry["evaluation_job_key"])
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "/opt/slurm/bin/sbatch")
        self.assertIn("--parsable", argv)
        exports = next(arg for arg in argv if arg.startswith("--export="))
        self.assertNotIn("ALL", exports)
        self.assertNotIn("WANDB_API_KEY", exports)
        self.assertIn("WANDB_SYNC=0", exports)
        self.assertIn("EVALUATION_DISPATCH=", exports)
        self.assertIn("CHECKPOINT=", exports)
        self.assertEqual(run.call_args.kwargs["timeout"], 30)
        self.assertNotIn("shell", run.call_args.kwargs)
        client_env = run.call_args.kwargs["env"]
        self.assertNotIn("WANDB_API_KEY", client_env)
        self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(client_env))
        self.assertTrue(set(client_env) <= {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "SLURM_CONF", "SLURM_CONF_SERVER"})

    def test_explicit_private_destination_endpoint_overlay_propagate_without_credentials(self):
        overlay = self.root / "independent SDK overlay"
        overlay.mkdir()
        declared = {"WANDB_SYNC": "1", "WANDB_ENTITY": "tianzizh-university-of-pennsylvania",
                    "WANDB_PROJECT": "SuperMimic_HOI_v1_20261023", "WANDB_BASE_URL": "https://api.forge.coreweave.com",
                    "HOI_WANDB_SDK_PATH": str(overlay), "WANDB_NOTES": "CREDENTIAL_SENTINEL",
                    "SLURM_JWT": "CREDENTIAL_SENTINEL", "OTHER_SECRET": "CREDENTIAL_SENTINEL"}
        with patch.dict(os.environ, declared):
            with patch("hoi.v1.periodic_eval.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "123456\n", "")) as run:
                result = self.call()
            request = self.request(result)
            self.assertEqual(request["tracking"]["project"], declared["WANDB_PROJECT"])
            self.assertEqual(request["tracking"]["base_url"], declared["WANDB_BASE_URL"])
            self.assertEqual(request["tracking"]["sdk_overlay"], str(overlay.resolve()))
            self.assertTrue(verify_dispatch(result["dispatch"])["verified"])
            exports = next(arg for arg in run.call_args.args[0] if arg.startswith("--export="))
            for key in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_BASE_URL", "HOI_WANDB_SDK_PATH", "WANDB_SYNC"):
                self.assertIn(f"{key}={declared[key]}", exports)
            self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(request) + exports)
            self.assertNotIn("WANDB_NOTES", exports)
            self.assertNotIn("WANDB_API_KEY", exports)
            self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(run.call_args.kwargs["env"]))
            self.assertNotIn("SLURM_JWT", run.call_args.kwargs["env"])
            self.assertNotIn("OTHER_SECRET", run.call_args.kwargs["env"])
            for field, value in (("WANDB_PROJECT", "different_private_project"),
                                 ("WANDB_BASE_URL", "https://api.wandb.ai"), ("WANDB_SYNC", "0")):
                with self.subTest(field=field), patch.dict(os.environ, {field: value}):
                    with self.assertRaisesRegex(ValueError, "destination differs"):
                        verify_dispatch(result["dispatch"])
            other_overlay = self.root / "other overlay"
            other_overlay.mkdir()
            with patch.dict(os.environ, {"HOI_WANDB_SDK_PATH": str(other_overlay)}):
                with self.assertRaisesRegex(ValueError, "destination differs"):
                    verify_dispatch(result["dispatch"])

    def test_destination_and_overlay_validation_precede_any_dispatch_output(self):
        for declaration in ({"WANDB_PROJECT": "unsafe,OTHER_KEY=value"},
                            {"WANDB_BASE_URL": "https://user:password@api.forge.coreweave.com"},
                            {"HOI_WANDB_SDK_PATH": str(self.root / "missing overlay")}):
            with self.subTest(declaration=declaration), patch.dict(os.environ, declaration):
                with self.assertRaises(ValueError):
                    self.call(dry_run=True)
                self.assertFalse(self.output.exists())

    def test_unknown_submission_outcome_never_creates_second_producer(self):
        for suffix, outcome in (("timeout", subprocess.TimeoutExpired("sbatch", 30)),
                                ("badreply", subprocess.CompletedProcess([], 0, "not-a-job", "")),
                                ("unavailable", FileNotFoundError())):
            self.output = self.root / suffix
            with patch("hoi.v1.periodic_eval.subprocess.run", side_effect=outcome if isinstance(outcome, Exception) else None,
                       return_value=None if isinstance(outcome, Exception) else outcome) as run:
                result = self.call()
                retry = self.call()
            self.assertEqual(result["status"], "submission_outcome_unknown")
            self.assertEqual(retry, result)
            run.assert_called_once()
            self.assertTrue((Path(result["dispatch"]).parent / "submit.claim").is_file())
            self.assertNotIn("stderr", result)

    def test_interrupted_claim_prevents_submission_and_request_change_rejected(self):
        dry = self.call(dry_run=True)
        (Path(dry["dispatch"]).parent / "submit.claim").write_text("exclusive", encoding="utf-8")
        with patch("hoi.v1.periodic_eval.subprocess.run", side_effect=AssertionError("claim blocks scheduler")):
            self.assertEqual(self.call()["status"], "submission_outcome_unknown")
        self.launcher.write_text("changed launcher", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different immutable"):
            self.call(dry_run=True)

    def test_exact_case_identity_duration_group_and_scope_checked_before_output(self):
        for change in (lambda h: h["clips"][0].update(source_motion="wrong.pt"),
                       lambda h: h["clips"][0].update(source_frames=99),
                       lambda h: h["clips"].__setitem__(0, deepcopy(h["clips"][1])),
                       lambda h: h["scope"].update(is_subset=True)):
            header = deepcopy(self.evaluation)
            change(header)
            self.save_bundle(self.eval_path, header)
            with self.assertRaises(ValueError):
                self.call(dry_run=True)
            self.assertFalse(self.output.exists())

    def test_checkpoint_physics_iteration_source_and_training_contract_rejected(self):
        for change in (lambda c: c["physics"].update(hand_collision=False),
                       lambda c: c.update(iteration=10000),
                       lambda c: c.update(feature_schema={}),
                       lambda c: c["data_identity"].update(partition="validation"),
                       lambda c: c["extra"]["source_identity"].update(content_sha256="0" * 64)):
            ck = deepcopy(self.checkpoint)
            change(ck)
            torch.save(ck, self.ck)
            with self.assertRaises(ValueError):
                self.call(dry_run=True)
            self.assertFalse(self.output.exists())

    def test_queued_job_revalidates_bytes_and_source_before_native_evaluation(self):
        dry = self.call(dry_run=True)
        self.launcher.write_text("changed queued source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "launcher bytes changed"):
            verify_dispatch(dry["dispatch"])
        self.launcher.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        with patch("hoi.v1.periodic_eval.source_identity", return_value={"files": {}, "content_sha256": json_identity({})}):
            with self.assertRaisesRegex(ValueError, "executable source differs"):
                verify_dispatch(dry["dispatch"])

    def test_environment_whitelist_sync_and_due_iteration_enforced(self):
        self.environment["WANDB_API_KEY"] = "CREDENTIAL_SENTINEL"
        with self.assertRaisesRegex(ValueError, "exactly the three"):
            self.call(dry_run=True)
        self.environment.pop("WANDB_API_KEY")
        with patch.dict(os.environ, {"WANDB_SYNC": "1"}):
            self.assertTrue(self.request(self.call(dry_run=True))["tracking"]["sync_enabled"])
        with patch.dict(os.environ, {"WANDB_SYNC": "credential-like-bad-value"}):
            with self.assertRaisesRegex(ValueError, "WANDB_SYNC"):
                self.call(dry_run=True)
        with self.assertRaisesRegex(ValueError, "5000"):
            schedule_evaluation(self.ck, 500, self.train_path, self.eval_path, self.output,
                                self.launcher, self.environment, dry_run=True)

    def test_training_preflight_verifies_full80_before_output_and_preserves_user_schedule(self):
        from hoi.v1.train import main, read_config
        config, deviations = read_config(Path(__file__).resolve().parents[1] / "configs/hoi_v1_first.json")
        self.assertEqual((config.iterations, config.ckpt_every, config.eval_every), (20000, 500, 5000))
        self.assertEqual(deviations, {})
        output = self.root / "main_preflight"
        args = ["--bundle", str(self.train_path), "--eval-bundle", str(self.eval_path),
                "--output", str(output), "--preflight"]
        self.evaluation["clips"][0]["source_frames"] -= 1
        self.save_bundle(self.eval_path, self.evaluation)
        with self.assertRaisesRegex(ValueError, "full source durations"):
            main(args)
        self.assertFalse(output.exists())
        self.evaluation["clips"][0]["source_frames"] += 1
        self.save_bundle(self.eval_path, self.evaluation)
        with redirect_stdout(io.StringIO()):
            plan = main(args)
        self.assertFalse(plan["training_started"])
        self.assertEqual(plan["periodic_evaluation"]["every_iterations"], 5000)
        self.assertEqual(plan["periodic_evaluation"]["checkpoint_every"], 500)
        self.assertEqual(plan["periodic_evaluation"]["case_count"], 80)
        self.assertFalse((output / "workers").exists())

    def test_source_identity_ignores_python_bytecode_and_runtime_caches(self):
        from hoi.v1.config import source_identity
        source_root = self.root / "source_fixture"
        for folder in ("hoi/v1", "supermimic", "configs", "cluster"):
            (source_root / folder).mkdir(parents=True)
        source_file = source_root / "hoi/v1/config.py"
        source_file.write_text("x = 1\n", encoding="utf-8")
        with patch("hoi.v1.config.__file__", str(source_file)):
            before = source_identity()
            (source_root / "hoi/v1/__pycache__").mkdir()
            (source_root / "hoi/v1/__pycache__/config.cpython-311.pyc").write_bytes(b"synthetic bytecode")
            (source_root / ".cache/hoi-v1-eval/123").mkdir(parents=True)
            (source_root / ".cache/hoi-v1-eval/123/warp.bin").write_bytes(b"compiled runtime cache")
            self.assertEqual(source_identity(), before)
            source_file.write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(source_identity()["content_sha256"], before["content_sha256"])


if __name__ == "__main__":
    unittest.main()
