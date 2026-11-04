"""One training attempt per Slurm allocation; the scheduler owns requeue.

Run under an exclusive campaign flock. No sbatch/requeue calls on training errors.
Each allocation gets fresh replay, output and W&B history, preserving old attempts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_finite(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_finite(child)


def validate(checkpoint, seed):
    from hoi.v1.checkpoint import validate_checkpoint_features
    validate_checkpoint_features(checkpoint)
    for key in ("schema_version", "actor_observation_profile", "variant", "feature_schema", "architecture",
                "physics", "data_identity", "data_identity_sha256", "continuation"):
        if checkpoint[key] != seed[key]:
            raise ValueError("checkpoint identity mismatch: " + key)
    iteration = checkpoint["iteration"]
    if type(iteration) is not int or not seed["iteration"] <= iteration <= seed["extra"]["config"]["iterations"]:
        raise ValueError("checkpoint iteration outside campaign")
    if checkpoint["extra"]["config"] != seed["extra"]["config"]:
        raise ValueError("checkpoint training configuration differs")
    for key in ("pi", "wm", "critic"):
        if checkpoint[key].keys() != seed[key].keys():
            raise ValueError("checkpoint model keys differ")
        for name, value in checkpoint[key].items():
            if value.shape != seed[key][name].shape:
                raise ValueError("checkpoint model shape differs")
    if set(checkpoint["optimizers"]) != {"pi", "wm", "critic"}:
        raise ValueError("incomplete checkpoint optimizers")
    for name, optimizer in checkpoint["optimizers"].items():
        expected = seed["optimizers"][name]
        if optimizer.keys() != expected.keys() or optimizer["state"].keys() != expected["state"].keys():
            raise ValueError("optimizer state structure differs")
        if len(optimizer["param_groups"]) != len(expected["param_groups"]):
            raise ValueError("optimizer parameter groups differ")
        for actual_group, expected_group in zip(optimizer["param_groups"], expected["param_groups"]):
            if actual_group["params"] != expected_group["params"]:
                raise ValueError("optimizer parameter order differs")
        for parameter, state in optimizer["state"].items():
            expected_state = expected["state"][parameter]
            if state.keys() != expected_state.keys():
                raise ValueError("optimizer moments missing")
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.shape != expected_state[key].shape:
                    raise ValueError("optimizer moment shape differs")
    for key in ("sim_steps", "contact_positive", "contact_negative", "sampling_bin_counts"):
        if key not in checkpoint["extra"]:
            raise ValueError("incomplete continuation state: " + key)
    check_finite(checkpoint)


def select_checkpoint(campaign, seed_path, seed_sha):
    seed_path = Path(seed_path).resolve()
    if digest(seed_path) != seed_sha:
        raise ValueError("initial checkpoint hash changed")
    seed = torch.load(seed_path, map_location="cpu", weights_only=False)
    validate(seed, seed)
    candidates = []
    for path in Path(campaign).glob("attempts/*/run/checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_([0-9]+)\.pt", path.name)
        if match and (path.parent.parent / "attempt.json").is_file():
            candidates.append((int(match[1]), path))
    rejected = []
    for iteration, path in sorted(candidates, key=lambda item: (item[0], str(item[1])), reverse=True):
        if iteration < seed["iteration"]:
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            validate(checkpoint, seed)
            if checkpoint["iteration"] != iteration:
                raise ValueError("checkpoint filename/iteration mismatch")
        except Exception as exc:
            rejected.append({"path": str(path), "reason": type(exc).__name__ + ": " + str(exc)[:300]})
            continue
        return path.resolve(), checkpoint, rejected
    return seed_path, seed, rejected


def prepare_attempt(campaign, seed_path, seed_sha, job_id, restart_count, run_name):
    if not re.fullmatch(r"[0-9]+", job_id) or not re.fullmatch(r"[0-9]+", restart_count):
        raise ValueError("Slurm job/restart identifiers must be numeric")
    path, checkpoint, rejected = select_checkpoint(campaign, seed_path, seed_sha)
    attempt = Path(campaign) / "attempts" / f"job_{job_id}_r{int(restart_count):04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = {"job_id": job_id, "restart_count": int(restart_count), "checkpoint": str(path),
               "checkpoint_sha256": digest(path), "iteration": checkpoint["iteration"],
               "output": str(attempt / "run"), "run_name": run_name + f"-r{int(restart_count):04d}",
               "rejected_checkpoints": rejected, "fresh_replay": True, "bitwise_resume": False,
               "training_complete": checkpoint["iteration"] == checkpoint["extra"]["config"]["iterations"]}
    (attempt / "attempt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def recover_due_evaluation(receipt, environment):
    """Close the save->evaluation window if preemption occurred between them.

    Use the original checkpoint's evaluation directory so an existing receipt
    is reused. Ambiguous submissions still stop for reconciliation, never duplicate.
    """
    iteration = receipt["iteration"]
    if iteration not in (5000, 10000, 15000, 20000):
        return
    from hoi.v1.periodic_eval import schedule_evaluation
    source = Path(environment["SUPERMIMIC_ROOT"])
    dataset = Path(environment["COMPACT_DATASET"])
    checkpoint = Path(receipt["checkpoint"])
    result = schedule_evaluation(checkpoint, iteration, dataset / "bundle_train/bundle.json",
                                 dataset / "bundle_protected_eval/bundle.json", checkpoint.parent / "evaluations",
                                 source / "cluster/hoi_compact_teacher_evaluate.sbatch.sh",
                                 {k: environment[k] for k in ("SUPERMIMIC_ROOT", "SUPERMIMIC_RUNTIME", "ISAACLAB_PATH")})
    if result["status"] != "submitted":
        raise RuntimeError("evaluation submission needs reconciliation: " + result["status"])


def main():
    torch.set_num_threads(3)
    environment = dict(os.environ)
    source = Path(environment["SUPERMIMIC_ROOT"])
    snapshot = json.loads((source / "source_snapshot.json").read_text())
    for name, sha in snapshot["files"].items():
        if digest(source / name) != sha:
            raise ValueError("frozen source changed: " + name)
    receipt = prepare_attempt(environment["CAMPAIGN_ROOT"], environment["INITIAL_CHECKPOINT"],
                              environment["INITIAL_CHECKPOINT_SHA256"], environment["SLURM_JOB_ID"],
                              environment.get("SLURM_RESTART_COUNT", "0"), environment["HOI_WANDB_RUN_NAME"])
    print(json.dumps({"automatic_continuation": receipt}), flush=True)
    environment.update(RESUME_CHECKPOINT=receipt["checkpoint"], RESUME_CHECKPOINT_SHA256=receipt["checkpoint_sha256"],
                       RESUME_ITERATION=str(receipt["iteration"]), RUN_DIR=receipt["output"],
                       HOI_WANDB_RUN_NAME=receipt["run_name"])
    recover_due_evaluation(receipt, environment)
    if receipt["training_complete"]:
        return
    os.execve("/bin/bash", ["bash", str(source / "cluster/hoi_compact_teacher_three_gpu.sbatch.sh")], environment)


if __name__ == "__main__":
    main()
