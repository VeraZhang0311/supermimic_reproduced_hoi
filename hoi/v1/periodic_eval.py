"""Dispatch one independent, frozen canonical HOI80 job per checkpoint.

The exclusive claim is deliberately never removed: an interrupted or ambiguous
sbatch reply must be reconciled against Slurm before any new producer is made.
Dry runs only save the reviewable request and never invoke a scheduler.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import torch

from hoi.data.manifest import read_manifest, sha256_file
from .checkpoint import CHECKPOINT_SCHEMA, checkpoint_variant
from .config import bundle_identity, json_identity, source_identity, write_json
from .evaluate import _read_bundle_header, validate_evaluation_identity
from .networks import FEATURE_SCHEMA
from .track import destination, _base_url

DISPATCH_SCHEMA = "supermimic_hoi_v1_periodic_eval_1"
ENVIRONMENT_KEYS = ("SUPERMIMIC_ROOT", "SUPERMIMIC_RUNTIME", "ISAACLAB_PATH")
EVALUATION_ITERATIONS = (5000, 10000, 15000, 20000)
SCHEDULER_CLIENT_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "SLURM_CONF", "SLURM_CONF_SERVER")


def validate_canonical80(training_header, evaluation_header):
    """Check exact held-out identities and duration, not just an 80-row count."""
    validate_evaluation_identity(training_header, evaluation_header)
    scope = evaluation_header.get("scope", {})
    if (evaluation_header.get("partition") != "eval80" or scope.get("is_subset") is not False or
            scope.get("available_motions") != 80 or scope.get("selected_motions") != 80 or
            len(evaluation_header.get("clips", ())) != 80):
        raise ValueError("periodic evaluation requires all 80 canonical eval80 cases")
    manifest = read_manifest(evaluation_header["manifest"]["path"])
    panel = manifest.get("panels", {}).get("eval80", ())
    if manifest.get("profile") != "canonical-v4" or len(panel) != 80:
        raise ValueError("periodic evaluation requires the canonical-v4 eval80 manifest")
    expected = {(case["motion"], case["motion_sha256"], case["source_group"], case["source_frames"])
                for case in panel}
    actual = {(case["source_motion"], case["source_motion_sha256"], case["source_group"], case["source_frames"])
              for case in evaluation_header["clips"]}
    if (len(expected) != 80 or len(actual) != 80 or actual != expected or
            len({case["source_group"] for case in evaluation_header["clips"]}) != 80):
        raise ValueError("eval80 cases or full source durations differ from the canonical panel")
    return True


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _environment(environment):
    if not isinstance(environment, dict) or set(environment) != set(ENVIRONMENT_KEYS):
        raise ValueError("scheduler environment requires exactly the three explicit runtime/source paths")
    result = {}
    for key in ENVIRONMENT_KEYS:
        value = environment[key]
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f"{key} must be a nonempty explicit path")
        value = str(Path(value).resolve())
        if any(character in value for character in (",", "\x00", "\n", "\r")):
            raise ValueError(f"{key} cannot be encoded in Slurm's explicit export list")
        result[key] = value
        if not Path(value).is_dir():
            raise ValueError(f"{key} must identify an existing prepared directory")
    if Path(result["SUPERMIMIC_ROOT"]) != Path(__file__).resolve().parents[2]:
        raise ValueError("dispatch must execute from the same frozen SUPERMIMIC_ROOT checkout")
    sync = os.environ.get("WANDB_SYNC", "0")
    if sync not in ("0", "1"):
        raise ValueError("WANDB_SYNC must be 0 or 1")
    return result, sync


def _tracking_environment():
    entity, project = destination()
    values = {"WANDB_ENTITY": entity, "WANDB_PROJECT": project, "WANDB_BASE_URL": _base_url()}
    overlay = os.environ.get("HOI_WANDB_SDK_PATH")
    if overlay:
        overlay = Path(overlay).expanduser().resolve()
        if not overlay.is_dir():
            raise ValueError("HOI_WANDB_SDK_PATH must be an existing independent SDK overlay directory")
        values["HOI_WANDB_SDK_PATH"] = str(overlay)
    if any(any(character in value for character in (",", "\x00", "\n", "\r")) for value in values.values()):
        raise ValueError("W&B destination cannot be encoded in the explicit Slurm export list")
    return values


def _checkpoint(path, iteration, training_header):
    # Hash and decode the same immutable byte snapshot, rather than racing two reads.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or
            checkpoint.get("feature_schema") != FEATURE_SCHEMA or checkpoint.get("iteration") != iteration):
        raise ValueError("evaluation checkpoint schema, features or iteration differ")
    physics = checkpoint.get("physics", {})
    if physics.get("hand_collision") is not True or physics.get("self_collision") is not True:
        raise ValueError("v1 periodic evaluation requires checkpoint hand ON and self ON")
    identity = bundle_identity(training_header)
    if checkpoint.get("data_identity") != identity or checkpoint.get("data_identity_sha256") != json_identity(identity):
        raise ValueError("checkpoint does not bind the exact original training bundle identity")
    source = checkpoint.get("extra", {}).get("source_identity", {})
    current = source_identity()
    if (not isinstance(source.get("files"), dict) or source.get("content_sha256") != json_identity(source["files"]) or
            source != current):
        raise ValueError("checkpoint executable source differs from the frozen evaluation checkout")
    return digest, physics, source["content_sha256"], checkpoint_variant(checkpoint)


def verify_dispatch(path):
    """Recheck the immutable request on the allocated node, before native setup."""
    request = _read_json(path)
    core = {key: value for key, value in request.items() if key not in ("request_sha256", "evaluation_job_key")}
    if request.get("schema_version") != DISPATCH_SCHEMA or json_identity(core) != request.get("request_sha256"):
        raise ValueError("evaluation dispatch request schema or hash changed")
    tracking = _tracking_environment()
    sync = os.environ.get("WANDB_SYNC", "0")
    if sync not in ("0", "1"):
        raise ValueError("WANDB_SYNC must be 0 or 1")
    if (request["tracking"]["entity"] != tracking["WANDB_ENTITY"] or request["tracking"]["project"] != tracking["WANDB_PROJECT"] or
            request["tracking"]["base_url"] != tracking["WANDB_BASE_URL"] or
            request["tracking"].get("sdk_overlay") != tracking.get("HOI_WANDB_SDK_PATH") or
            request["tracking"]["sync_enabled"] != (sync == "1")):
        raise ValueError("evaluation W&B destination differs from the frozen dispatch")
    training_header = _read_bundle_header(request["training_bundle"]["path"])
    evaluation_header = _read_bundle_header(request["evaluation_bundle"]["path"])
    validate_canonical80(training_header, evaluation_header)
    for name in ("checkpoint", "training_bundle", "evaluation_bundle", "launcher"):
        if sha256_file(request[name]["path"]) != request[name]["sha256"]:
            raise ValueError(f"frozen evaluation {name} bytes changed after dispatch")
    checkpoint_sha, physics, source_sha, variant = _checkpoint(Path(request["checkpoint"]["path"]), request["iteration"], training_header)
    if (checkpoint_sha != request["checkpoint"]["sha256"] or physics != request["physics"] or
            source_sha != request["source_content_sha256"] or variant != request["variant"] or
            json_identity(bundle_identity(training_header)) != request["training_data_identity_sha256"] or
            training_header["bundle_content_sha256"] != request["training_bundle"]["content_sha256"] or
            evaluation_header["bundle_content_sha256"] != request["evaluation_bundle"]["content_sha256"]):
        raise ValueError("frozen checkpoint/source/bundle contracts changed after dispatch")
    if request["protocol"] != {"partition": "eval80", "cases": 80, "full_duration": True,
                               "optimizer_updates": 0, "normalization_updates": 0,
                               "no_failure_reset": True, "independent_gpu_job": True}:
        raise ValueError("evaluation dispatch full-duration frozen protocol differs")
    return {"verified": True, "evaluation_job_key": request["evaluation_job_key"],
            "checkpoint_sha256": checkpoint_sha, "source_content_sha256": source_sha}


def _result(directory, request, status, **extra):
    return {"schema_version": DISPATCH_SCHEMA, "status": status,
            "evaluation_job_key": request["evaluation_job_key"], "iteration": request["iteration"],
            "dispatch": str(directory / "dispatch.json"), "output": request["output"], **extra}


def schedule_evaluation(checkpoint_path, iteration, training_bundle, eval_bundle,
                        output_root, launcher_path, environment, dry_run=False):
    """Validate and asynchronously submit a separate GPU evaluation at 5k steps.

    A dry run writes the exact request but no claim. Retrying the same request
    after submission returns its existing job id. Any claim without a confirmed
    reply returns ``submission_outcome_unknown`` and never resubmits.
    """
    if type(iteration) is not int or iteration not in EVALUATION_ITERATIONS:
        raise ValueError("periodic evaluation iterations are 5000, 10000, 15000 and 20000")
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    paths = [Path(value).resolve() for value in (checkpoint_path, training_bundle, eval_bundle, launcher_path)]
    checkpoint_path, training_bundle, eval_bundle, launcher_path = paths
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing frozen evaluation input: {path}")
        if any(character in str(path) for character in (",", "\x00", "\n", "\r")):
            raise ValueError("evaluation paths cannot be encoded in the explicit Slurm export list")
    runtime, sync = _environment(environment)
    tracking = _tracking_environment()
    training_header, evaluation_header = _read_bundle_header(training_bundle), _read_bundle_header(eval_bundle)
    validate_canonical80(training_header, evaluation_header)
    checkpoint_sha, physics, source_sha, variant = _checkpoint(checkpoint_path, iteration, training_header)
    directory = Path(output_root).resolve() / f"iteration_{iteration:06d}"
    if any(character in str(directory) for character in (",", "\x00", "\n", "\r")):
        raise ValueError("evaluation output cannot be encoded in the Slurm export list")
    core = {"schema_version": DISPATCH_SCHEMA, "iteration": iteration,
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "training_bundle": {"path": str(training_bundle), "sha256": sha256_file(training_bundle),
                                "content_sha256": training_header["bundle_content_sha256"]},
            "evaluation_bundle": {"path": str(eval_bundle), "sha256": sha256_file(eval_bundle),
                                  "content_sha256": evaluation_header["bundle_content_sha256"]},
            "training_data_identity_sha256": json_identity(bundle_identity(training_header)),
            "source_content_sha256": source_sha, "physics": physics, "variant": variant,
            "launcher": {"path": str(launcher_path), "sha256": sha256_file(launcher_path)},
            "environment": runtime, "output": str(directory / "results"),
            "protocol": {"partition": "eval80", "cases": 80, "full_duration": True,
                         "optimizer_updates": 0, "normalization_updates": 0,
                         "no_failure_reset": True, "independent_gpu_job": True},
            "tracking": {"entity": tracking["WANDB_ENTITY"], "project": tracking["WANDB_PROJECT"],
                         "base_url": tracking["WANDB_BASE_URL"], "visibility": "existing_private_project",
                         "sdk_overlay": tracking.get("HOI_WANDB_SDK_PATH"),
                         "resume": "never", "run_id_policy": "fresh_uuid_at_sync", "sync_enabled": sync == "1"}}
    request_sha = json_identity(core)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "dispatch.json"
    request = {**core, "request_sha256": request_sha, "evaluation_job_key": uuid.uuid4().hex}
    try:
        write_json(request_path, request)
    except FileExistsError:
        request = _read_json(request_path)
        saved_core = {key: value for key, value in request.items() if key not in ("request_sha256", "evaluation_job_key")}
        if request.get("request_sha256") != request_sha or json_identity(saved_core) != request_sha:
            raise ValueError("this iteration already binds a different immutable evaluation request")
    receipt_path, claim_path = directory / "submission.json", directory / "submit.claim"
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if receipt.get("evaluation_job_key") != request["evaluation_job_key"]:
            raise ValueError("evaluation submission receipt does not match its request")
        return receipt
    if claim_path.exists():
        return _result(directory, request, "submission_outcome_unknown", resubmitted=False,
                       reason="exclusive claim exists; reconcile Slurm before any new submission")
    if dry_run:
        return _result(directory, request, "dry_run", scheduler_calls=0)
    if Path(core["output"]).exists():
        raise ValueError("evaluation results already exist; preserve them instead of submitting")
    try:
        write_json(claim_path, {"evaluation_job_key": request["evaluation_job_key"], "request_sha256": request_sha})
    except FileExistsError:
        return _result(directory, request, "submission_outcome_unknown", resubmitted=False,
                       reason="another dispatcher acquired the exclusive submission claim")
    exports = {**runtime, **tracking, "CHECKPOINT": str(checkpoint_path), "TRAINING_BUNDLE": str(training_bundle),
               "EVAL_BUNDLE": str(eval_bundle), "OUTPUT": core["output"],
               "EVALUATION_JOB_KEY": request["evaluation_job_key"], "EVALUATION_DISPATCH": str(request_path), "WANDB_SYNC": sync}
    sbatch = shutil.which("sbatch")
    if sbatch is None and Path("/opt/slurm/bin/sbatch").is_file():
        sbatch = "/opt/slurm/bin/sbatch"
    command = [sbatch or "sbatch", "--parsable", f"--job-name=sm-hoi-v1-eval-{iteration:06d}",
               f"--output={directory / 'slurm-%j.out'}", f"--error={directory / 'slurm-%j.err'}",
               "--export=" + ",".join(f"{key}={value}" for key, value in exports.items()), str(launcher_path)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True,
                                   cwd=runtime["SUPERMIMIC_ROOT"], timeout=30,
                                   env={key: os.environ[key] for key in SCHEDULER_CLIENT_ENVIRONMENT if key in os.environ})
        match = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?", completed.stdout.strip())
        if completed.returncode != 0 or match is None:
            receipt = _result(directory, request, "submission_outcome_unknown", scheduler_calls=1,
                              returncode=completed.returncode, resubmitted=False,
                              reason="sbatch did not return a confirmed job id; exclusive claim retained")
        else:
            receipt = _result(directory, request, "submitted", scheduler_calls=1,
                              job_id=match.group(1), cluster=match.group(2), resubmitted=False)
    except (OSError, subprocess.TimeoutExpired):
        # Do not include arbitrary subprocess output/environment in the receipt.
        receipt = _result(directory, request, "submission_outcome_unknown", scheduler_calls=1, resubmitted=False,
                          reason="sbatch interrupted or unavailable; exclusive claim retained")
    write_json(receipt_path, receipt)
    return receipt


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Verify a frozen dispatch; never submits jobs")
    parser.add_argument("--verify", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_dispatch(args.verify), allow_nan=False))


if __name__ == "__main__":
    main()
