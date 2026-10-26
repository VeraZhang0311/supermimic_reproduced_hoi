"""Stdlib-only handoff checks: one allocation, immutable source, fresh training."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def check(mode, environment=None):
    env = os.environ if environment is None else environment
    if mode not in ("before", "after-validation"):
        raise ValueError("unknown combined-job handoff phase")
    root, operations = Path(env["SUPERMIMIC_ROOT"]).resolve(), Path(env["OPERATIONS_ROOT"]).resolve()
    stage_path, manifest_path = root / "data/v11_staging.json", operations / "operations.json"
    if _sha(stage_path) != env["EXPECTED_STAGE_SHA256"] or _sha(manifest_path) != env["EXPECTED_OPERATIONS_SHA256"]:
        raise ValueError("combined-job source/operations manifest hash differs")
    stage, manifest = _read(stage_path), _read(manifest_path)
    if stage["root"] != str(root) or manifest["root"] != str(operations) or manifest["stage_sha256"] != env["EXPECTED_STAGE_SHA256"]:
        raise ValueError("combined-job source and operations binding differs")
    for base, files in ((root, stage["files"]), (operations, manifest["files"])):
        for name, digest in files.items():
            if _sha(base / name) != digest:
                raise ValueError("combined-job immutable file differs: " + name)
    allocation = {key: env.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS")}
    if any(not allocation[key] for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES")):
        raise ValueError("combined pipeline needs one real GPU Slurm allocation")
    config = stage["production_config"]
    if (config["iterations"], config["num_envs"], config["pi_batch"], config["horizon"], config["wm_only_iters"]) != (20000, 4096, 4096, 32, 500):
        raise ValueError("combined pipeline changed the validated production configuration")
    started = operations / "run/pipeline_started.json"
    record = {"utc": datetime.now(timezone.utc).isoformat(), "allocation": allocation,
              "source_content_sha256": stage["source_identity"]["content_sha256"]}
    if mode == "before":
        for name in ("v11_native_dr", "v11_resource", "v11_validation_complete.json", "hoi_v1_1_object_rms_scratch_20k"):
            if (root / "results" / name).exists():
                raise ValueError("preserve previous outputs instead of reusing a combined pipeline: " + name)
        _record(started, record)
    else:
        if _read(started)["allocation"] != allocation:
            raise ValueError("validation and training must use the same Slurm allocation and GPU")
        complete = _read(root / "results/v11_validation_complete.json")
        if (complete.get("passed") is not True or complete.get("actual_optimizer_steps") != 0 or
                complete.get("source_identity") != stage["source_identity"] or complete.get("job_id") != allocation["SLURM_JOB_ID"]):
            raise ValueError("current allocation has no successful matching native/capacity validation")
        for name in ("native_evidence", "resource_report"):
            path = Path(complete[name]).resolve()
            path.relative_to(root / "results")
            if _sha(path) != complete[name + "_sha256"]:
                raise ValueError("validation artifact changed before training")
        resource = _read(Path(complete["resource_report"]))
        if (resource.get("passed") is not True or resource.get("actual_optimizer_steps") != 0 or
                resource.get("all_model_parameters_unchanged") is not True or resource.get("config") != config or
                resource.get("source_identity") != stage["source_identity"] or
                resource.get("noop_optimizer_step_calls") != {"pi": 1, "wm": 1, "critic": 20}):
            raise ValueError("production capacity/backward verification failed")
        _record(operations / "run/validation_passed.json", {**record, "probe_optimizer_steps": 0,
                "fresh_training_process_next": True, "second_training_submission": False})
    print(json.dumps({"phase": mode, "passed": True, "allocation": allocation}), flush=True)


if __name__ == "__main__":
    check(sys.argv[1])
