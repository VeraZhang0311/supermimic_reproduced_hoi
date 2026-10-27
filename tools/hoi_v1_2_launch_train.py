"""Launch only the validated, committed v1.2 experiment into a fresh output.

The staged checkout and shared runtime remain immutable. Validation models are
disposable; the training process always creates new parameters and optimizers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BRANCH = "hoi/v1.2-perception-fusion"
CONFIG = "configs/hoi_v1_2_perception_fusion.json"
ENTITY = "tianzizh-university-of-pennsylvania"
PROJECT = "SuperMimic_HOI_v1_20261023"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_stage():
    root = Path(os.environ["SUPERMIMIC_ROOT"]).resolve()
    stage_path = root / "data/v12_staging.json"
    stage_sha = sha256_file(stage_path)
    if stage_sha != os.environ["EXPECTED_STAGE_SHA256"]:
        raise ValueError("staged v1.2 manifest hash differs")
    stage = read_json(stage_path)
    if str(root) != stage["root"] or stage.get("branch") != BRANCH:
        raise ValueError("staged v1.2 checkout/branch differs")
    for variable, key in (("SUPERMIMIC_RUNTIME", "runtime"), ("ISAACLAB_PATH", "isaaclab")):
        if Path(os.environ[variable]).resolve() != Path(stage[key]).resolve():
            raise ValueError("staged read-only runtime/IsaacLab differs")
    for name, digest in stage["files"].items():
        path = root / name
        if (Path(name).is_absolute() or not path.resolve().is_relative_to(root) or
                path.is_symlink() or sha256_file(path) != digest):
            raise ValueError("staged source/helper identity differs: " + name)
    from hoi.v1.config import source_identity
    from hoi.v1.train import read_config
    from hoi.v1.observations import observation_contract
    from hoi.v1.actor_fusion import FUSION_ACTOR, actor_architecture_contract
    config, changes = read_config(root / CONFIG)
    if (changes or config.to_dict() != stage["production_config"] or
            source_identity() != stage["source_identity"]):
        raise ValueError("staged production configuration/source differs")
    if (config.iterations, config.num_envs, config.pi_batch, config.horizon, config.actor_architecture) != (
            20000, 4096, 4096, 32, FUSION_ACTOR):
        raise ValueError("v1.2 requires unchanged 20k/4096/H32 production settings")
    if observation_contract(config.observation_profile) != stage["actor_observation_contract"]:
        raise ValueError("v1.2 observation contract differs")
    if actor_architecture_contract(config.actor_architecture, config.observation_profile) != stage["actor_architecture_contract"]:
        raise ValueError("v1.2 fusion architecture contract differs")
    output = Path(stage["training_output"])
    if not output.is_absolute() or not output.resolve().is_relative_to(root / "results"):
        raise ValueError("v1.2 training output must be explicitly inside the new checkout results")
    return root, stage


def verify_training_authorization(root, stage):
    complete_path = root / "results/v12_validation_complete.json"
    complete = read_json(complete_path)
    stage_sha = sha256_file(root / "data/v12_staging.json")
    if (complete.get("passed") is not True or complete.get("source_identity") != stage["source_identity"] or
            complete.get("stage_sha256") != stage_sha):
        raise ValueError("matching complete GPU validation is required before training")
    for key in ("native_evidence", "resource_report"):
        path = Path(complete[key])
        if not path.resolve().is_relative_to(root / "results") or sha256_file(path) != complete[key + "_sha256"]:
            raise ValueError("v1.2 validation evidence identity differs")
    resource = read_json(complete["resource_report"])
    if (resource.get("passed") is not True or resource.get("source_identity") != stage["source_identity"] or
            resource.get("config") != stage["production_config"] or
            resource.get("actor_architecture_contract") != stage["actor_architecture_contract"] or
            resource.get("actual_optimizer_steps") != 43 or
            resource.get("actual_optimizer_steps_by_model") != {"wm": 1, "pi": 2, "critic": 40} or
            resource.get("probe_models_discarded") is not True or
            resource.get("fusion_live_second_actor_backward") is not True):
        raise ValueError("full production GPU/Adam/fusion validation is incomplete")
    commit_path = root / "data/v12_commit_receipt.json"
    if sha256_file(commit_path) != os.environ["EXPECTED_COMMIT_RECEIPT_SHA256"]:
        raise ValueError("validated Git commit receipt hash differs")
    receipt = read_json(commit_path)
    commit = receipt.get("commit", "")
    if (receipt.get("schema") != "hoi_v1_2_validated_commit_receipt_1" or
            not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) or
            receipt.get("branch") != BRANCH or receipt.get("pushed") is not True or
            receipt.get("remote_commit") != commit or receipt.get("source_identity") != stage["source_identity"] or
            receipt.get("stage_sha256") != stage_sha or
            receipt.get("validation_complete_sha256") != sha256_file(complete_path)):
        raise ValueError("v1.2 must be committed and pushed after matching validation")
    return complete, receipt


def training_arguments(root, stage, complete):
    return [sys.executable, "-m", "hoi.v1.train", "--config", str(root / CONFIG),
            "--bundle", stage["train_bundle"], "--native-evidence", complete["native_evidence"],
            "--eval-bundle", stage["eval_bundle"], "--from-scratch", "--hand-collision", "--self-collision",
            "--robot-mjcf", stage["robot_mjcf"], "--evaluation-runtime", os.environ["SUPERMIMIC_RUNTIME"],
            "--evaluation-isaaclab", os.environ["ISAACLAB_PATH"],
            "--evaluation-launcher", str(root / "cluster/hoi_v1_2_evaluate.sbatch.sh")]


def main():
    root, stage = verify_stage()
    complete, receipt = verify_training_authorization(root, stage)
    if (os.environ.get("WANDB_SYNC") != "1" or os.environ.get("WANDB_ENTITY") != ENTITY or
            os.environ.get("WANDB_PROJECT") != PROJECT):
        raise ValueError("v1.2 requires the explicitly selected new private W&B training run")
    if any(os.environ.get(name) for name in ("WANDB_RUN_ID", "WANDB_RESUME", "INIT_CHECKPOINT", "RESUME_CHECKPOINT")):
        raise ValueError("v1.2 from-scratch training cannot inherit a prior run/checkpoint")
    output = Path(stage["training_output"])
    if Path(os.environ["RUN_DIR"]).resolve() != output.resolve() or output.exists():
        raise ValueError("training requires its exact new, nonexistent RUN_DIR")
    claim = {"schema": "hoi_v1_2_training_claim_1", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "pid": os.getpid(), "output": str(output), "commit": receipt["commit"],
             "source_identity": stage["source_identity"], "stage_sha256": os.environ["EXPECTED_STAGE_SHA256"],
             "from_scratch": True, "iterations": 20000, "tracking_resume": "never"}
    # The exclusive claim intentionally survives failures/unknown outcomes.
    # A retry must be inspected rather than launching a second producer.
    write_exclusive(root / "results/v12_training_claim.json", claim)
    args = training_arguments(root, stage, complete)
    preflight = root / "results/v12_formal_preflight"
    if not preflight.exists():
        subprocess.run(args + ["--output", str(preflight), "--preflight"], cwd=root, check=True)
    plan = read_json(preflight / "plan.json")
    if (plan["source_identity"] != stage["source_identity"] or plan["config"] != stage["production_config"] or
            plan["actor_observation_contract"] != stage["actor_observation_contract"] or
            plan["actor_architecture_contract"] != stage["actor_architecture_contract"] or
            not plan["native_gate"]["passed"] or not plan["native_dr_gate"]["passed"] or plan["training_started"] or
            plan["smoke_deviations"] or not plan["initialization_request"]["from_scratch"] or
            plan["policy_update_schedule"]["wm_quality_gate_enabled"] or
            plan["periodic_evaluation"]["every_iterations"] != 5000):
        raise ValueError("v1.2 formal preflight differs from validated experiment")
    result = subprocess.run(args + ["--output", str(output)], cwd=root)
    write_exclusive(root / "results/v12_training_exit.json", {"schema": "hoi_v1_2_training_exit_1",
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "commit": receipt["commit"],
                    "output": str(output), "returncode": result.returncode})
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
