"""Local launch gates; never submits jobs or creates online tracking runs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hoi.v1.actor_fusion import FUSION_ACTOR, actor_architecture_contract
from hoi.v1.observations import MATCHING
from hoi.v1.train import read_config
from tools import hoi_v1_2_launch_train as launch


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return launch.sha256_file(path)


class LaunchGatesTests(unittest.TestCase):
    def fixture(self, root):
        architecture = actor_architecture_contract(FUSION_ACTOR, MATCHING)
        source = {"content_sha256": "a" * 64, "files": {"hoi/v1/actor_fusion.py": "b" * 64}}
        config, _ = read_config(Path(__file__).resolve().parents[1] / launch.CONFIG)
        stage = {"root": str(root), "branch": launch.BRANCH, "production_config": config.to_dict(),
                 "source_identity": source, "actor_architecture_contract": architecture,
                 "training_output": str(root / "results/new_training"),
                 "train_bundle": "/read-only/train.json", "eval_bundle": "/read-only/eval80.json",
                 "robot_mjcf": "/read-only/robot.xml"}
        stage_sha = dump(root / "data/v12_staging.json", stage)
        native_path = root / "results/native.json"
        native_sha = dump(native_path, {"passed": True})
        resource_path = root / "results/resource.json"
        resource = {"passed": True, "source_identity": source, "config": config.to_dict(),
                    "actor_architecture_contract": architecture, "actual_optimizer_steps": 43,
                    "actual_optimizer_steps_by_model": {"wm": 1, "pi": 2, "critic": 40},
                    "probe_models_discarded": True, "fusion_live_second_actor_backward": True}
        resource_sha = dump(resource_path, resource)
        complete = {"passed": True, "source_identity": source, "stage_sha256": stage_sha,
                    "native_evidence": str(native_path), "native_evidence_sha256": native_sha,
                    "resource_report": str(resource_path), "resource_report_sha256": resource_sha}
        complete_sha = dump(root / "results/v12_validation_complete.json", complete)
        receipt = {"schema": "hoi_v1_2_validated_commit_receipt_1", "commit": "c" * 40,
                   "remote_commit": "c" * 40, "branch": launch.BRANCH, "pushed": True,
                   "source_identity": source, "stage_sha256": stage_sha,
                   "validation_complete_sha256": complete_sha}
        commit_sha = dump(root / "data/v12_commit_receipt.json", receipt)
        return stage, complete, receipt, {"EXPECTED_STAGE_SHA256": stage_sha,
                                         "EXPECTED_COMMIT_RECEIPT_SHA256": commit_sha}

    def test_complete_gpu_and_commit_receipts_bind_the_same_validated_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage, complete, receipt, environment = self.fixture(root)
            with patch.dict(os.environ, environment):
                actual_complete, actual_receipt = launch.verify_training_authorization(root, stage)
            self.assertEqual(actual_complete, complete)
            self.assertEqual(actual_receipt, receipt)

    def test_stage_rejects_a_different_runtime_or_isaaclab_before_source_execution(self):
        for variable in ("SUPERMIMIC_RUNTIME", "ISAACLAB_PATH"):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                stage = {"root": str(root), "branch": launch.BRANCH,
                         "runtime": str(root / "validated_runtime"), "isaaclab": str(root / "validated_isaaclab")}
                digest = dump(root / "data/v12_staging.json", stage)
                environment = {"SUPERMIMIC_ROOT": str(root), "EXPECTED_STAGE_SHA256": digest,
                               "SUPERMIMIC_RUNTIME": stage["runtime"], "ISAACLAB_PATH": stage["isaaclab"]}
                environment[variable] = str(root / "different_runtime")
                with patch.dict(os.environ, environment), self.assertRaisesRegex(ValueError, "runtime/IsaacLab"):
                    launch.verify_stage()

    def test_mutated_evidence_and_an_unpushed_or_other_branch_commit_are_rejected(self):
        for mutation in ("evidence", "unpushed", "branch", "remote_commit", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                stage, complete, receipt, environment = self.fixture(root)
                if mutation == "evidence":
                    dump(Path(complete["resource_report"]), {"passed": False})
                else:
                    if mutation == "unpushed": receipt["pushed"] = False
                    elif mutation == "branch": receipt["branch"] = "hoi/v1.1-observation-matching"
                    elif mutation == "remote_commit": receipt["remote_commit"] = "d" * 40
                    else: receipt["source_identity"] = {}
                    environment["EXPECTED_COMMIT_RECEIPT_SHA256"] = dump(root / "data/v12_commit_receipt.json", receipt)
                with patch.dict(os.environ, environment), self.assertRaises(ValueError):
                    launch.verify_training_authorization(root, stage)

    def test_noop_probe_or_fusion_without_second_backward_cannot_authorize_training(self):
        for mutation in ("steps", "counts", "discarded", "fusion"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                stage, complete, receipt, environment = self.fixture(root)
                resource = launch.read_json(complete["resource_report"])
                if mutation == "steps": resource["actual_optimizer_steps"] = 0
                elif mutation == "counts": resource["actual_optimizer_steps_by_model"]["pi"] = 1
                elif mutation == "discarded": resource["probe_models_discarded"] = False
                else: resource["fusion_live_second_actor_backward"] = False
                complete["resource_report_sha256"] = dump(Path(complete["resource_report"]), resource)
                receipt["validation_complete_sha256"] = dump(root / "results/v12_validation_complete.json", complete)
                environment["EXPECTED_COMMIT_RECEIPT_SHA256"] = dump(root / "data/v12_commit_receipt.json", receipt)
                with patch.dict(os.environ, environment), self.assertRaisesRegex(ValueError, "GPU/Adam/fusion"):
                    launch.verify_training_authorization(root, stage)

    def test_training_uses_from_scratch_and_the_explicit_two_hour_eval_launcher(self):
        root = Path("D:/new_experiment")
        stage = {"train_bundle": "/data/train", "eval_bundle": "/data/eval", "robot_mjcf": "/data/robot"}
        with patch.dict(os.environ, {"SUPERMIMIC_RUNTIME": "/read-only/runtime", "ISAACLAB_PATH": "/read-only/isaaclab"}):
            arguments = launch.training_arguments(root, stage, {"native_evidence": "/new/native.json"})
        self.assertIn("--from-scratch", arguments)
        self.assertNotIn("--resume", arguments)
        self.assertNotIn("--init-checkpoint", arguments)
        self.assertEqual(arguments[arguments.index("--evaluation-launcher") + 1],
                         str(root / "cluster/hoi_v1_2_evaluate.sbatch.sh"))
        script = (Path(__file__).resolve().parents[1] / "cluster/hoi_v1_2_evaluate.sbatch.sh").read_text()
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3", script)
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            self.assertIn("export " + key + "=expandable_segments:True", script)

    def test_duplicate_training_claim_is_exclusive_and_survives_failed_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            claim = Path(folder) / "claim.json"
            launch.write_exclusive(claim, {"job": "first"})
            with self.assertRaises(FileExistsError):
                launch.write_exclusive(claim, {"job": "second"})
            self.assertEqual(launch.read_json(claim), {"job": "first"})

    def test_training_rejects_inherited_wandb_id_before_claim_or_process_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage, complete, receipt, environment = self.fixture(root)
            environment.update({"WANDB_SYNC": "1", "WANDB_ENTITY": launch.ENTITY,
                                "WANDB_PROJECT": launch.PROJECT, "WANDB_RUN_ID": "old-run"})
            with patch.dict(os.environ, environment), patch.object(launch, "verify_stage", return_value=(root, stage)), patch.object(
                    launch, "verify_training_authorization", return_value=(complete, receipt)), patch.object(
                    launch.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "inherit"):
                launch.main()
            run.assert_not_called()
            self.assertFalse((root / "results/v12_training_claim.json").exists())


if __name__ == "__main__":
    unittest.main()
