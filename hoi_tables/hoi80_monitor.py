"""Canonical HOI80 scheduler adapter for the unified owning-run writer.

No W&B SDK, trainer, simulator or GPU module is imported here. Eight immutable
rank checkpoints and all 80 original cases are bound before a one-GPU worker
evaluates assets sequentially. TablesReward survival and canonical PHC/object
success remain separate metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re

from . import eval_monitor as base
from .experiments import ULTRA_ASSETS as ASSETS
from .unified_monitor import owning_run

PROTOCOL = "hoi_ultra_full_canonical80_periodic_eval_v1"
SCHEMA = "hoi_ultra_hoi80_evaluation_v1"
CANONICAL_SHA256 = "5c2832ba04c1e4061d9a1a1864f820f7ad949e1dd2ab982efff1dbf9511049a2"
ASSET_COUNTS = (5, 15, 6, 14, 4, 16, 11, 9)
CONTROL_STEPS = 23718
MAX_CLIP_STEPS = 629
HOI80_METRICS = ("human_pass_rate", "object_pass_rate", "joint_pass_rate")
PROTOCOL_FILES = {
    "data.py": "21b834f57da4e388a3c108ca8b34e108a698d034eb5077784421b3aa04138fc1",
    "metrics.py": "087742c554f3cf8de5edd83966c9e9479d6614eb1424983c11c5fc35d7700c17",
    "native.py": "450e066fb6dfc7cdb85c8b3b0ca9290556868bf9f19596f88c4977bc1bb53f2c",
    "ultra_kinematics.py": "c5547f78fedbab91e78ea9f9eeb1dc68b232e65f279a2c079a916b0539890710",
    "ultra_math.py": "274d172996fe40f3a5745cf6ea09d617448e2c33e8d15ad8f92ea0bd3b604f85",
}


def canonical_cases(canonical):
    if base.sha256(canonical) != CANONICAL_SHA256:
        raise ValueError("Canonical HOI80 manifest SHA256 differs from the frozen panel")
    raw = json.loads(Path(canonical).read_text())
    cases = raw.get("cases", [])
    if len(cases) != 80:
        raise ValueError("Canonical HOI80 requires the complete original 80 cases")
    result = {}
    for index, case in enumerate(cases):
        stem = PurePosixPath(case["motion"].replace("\\", "/")).stem
        identity = "ultra/"+stem
        controls = case["scored_control_steps"]
        frames = case["resampled_frames"]
        if (identity in result or case["object_asset"] not in ASSETS or type(controls) is not int
                or controls < 1 or frames != controls+1):
            raise ValueError("Canonical HOI80 case identity or T+1 duration changed")
        result[identity] = {"asset": case["object_asset"], "control_steps": controls,
                            "state_frames": frames, "canonical_index": index, "case_id": case["id"],
                            "motion_sha256": case["motion_sha256"]}
    counts = tuple(sum(c["asset"] == asset for c in result.values()) for asset in ASSETS)
    if counts != ASSET_COUNTS or sum(c["control_steps"] for c in result.values()) != CONTROL_STEPS or max(c["control_steps"] for c in result.values()) != MAX_CLIP_STEPS:
        raise ValueError("Canonical HOI80 heterogeneous asset coverage or duration changed")
    return result


def expected_clips(manifest):
    return canonical_cases(Path(manifest).parent/"hoi80_manifest.json")


def checkpoint_pair(training_root, iteration):
    if type(iteration) is not int or iteration not in base.MILESTONES:
        raise ValueError("HOI80 scheduling requires an exact 2000-iteration milestone")
    return {f"rank_{rank:02d}": str(Path(training_root)/f"rank_{rank:02d}/checkpoint_{iteration:06d}.pt") for rank in range(8)}


def _experiment(path):
    value = json.loads(Path(path).read_text())
    evaluation = value.get("evaluation", {})
    if (value.get("schema") != "hoi_direct_teacher_experiment_v1" or value.get("profile") != "ultra"
            or evaluation.get("kind") != "canonical_hoi80" or evaluation.get("every") != 2000
            or evaluation.get("owning_training_run") is not True):
        raise ValueError("Ultra experiment must schedule full canonical HOI80 on its owning training run")
    helper = evaluation.get("helper_root")
    if not isinstance(helper, str) or not helper or not Path(helper).is_dir():
        raise ValueError("Ultra experiment requires the existing frozen canonical helper source")
    helper = Path(helper).resolve()
    for name, digest in PROTOCOL_FILES.items():
        if base.sha256(helper/"hoi_zero_shot"/name) != digest:
            raise ValueError("Canonical HOI80 helper source bytes changed")
    return value, helper


def _input_paths(campaign):
    inputs = Path(campaign)/"input"
    return {"manifest": inputs/"manifest.json", "asset_bundle": inputs/"assets.json",
            "shape_bank": inputs/"shape_bank.json", "geometry_bundle": inputs/"geometry_bundle.json",
            "canonical": inputs/"hoi80_manifest.json", "reference_contacts": inputs/"reference_contacts.pt"}


def validate_request(request):
    if request.get("protocol") != PROTOCOL or request.get("iteration") not in base.MILESTONES:
        raise ValueError("Invalid direct-teacher HOI80 evaluation request")
    ranks = {f"rank_{rank:02d}" for rank in range(8)}
    if set(request.get("checkpoints", {})) != ranks or set(request.get("checkpoint_sha256", {})) != ranks:
        raise ValueError("Canonical HOI80 requires all eight same-iteration rank checkpoints")
    canonical_expected = checkpoint_pair(request["training_root"], request["iteration"])
    for rank, path in request["checkpoints"].items():
        if (Path(path).resolve() != Path(canonical_expected[rank]).resolve()
                or not base.complete_checkpoint(path) or base.sha256(path) != request["checkpoint_sha256"][rank]):
            raise ValueError("HOI80 checkpoint path or bytes changed after dispatch")
    inputs = request.get("input_sha256", {})
    if set(inputs) != {"manifest", "asset_bundle", "shape_bank", "geometry_bundle", "canonical", "reference_contacts"}:
        raise ValueError("HOI80 request lacks a complete immutable input identity")
    for key, digest in inputs.items():
        if base.sha256(request[key]) != digest:
            raise ValueError("HOI80 bound input changed: "+key)
    if request.get("manifest_sha256") != inputs["manifest"] or request.get("canonical_sha256") != CANONICAL_SHA256:
        raise ValueError("HOI80 training or canonical manifest binding changed")
    if base.source_tree_hash(request["source_root"]) != request["eval_source_sha256"]:
        raise ValueError("Frozen HOI80 evaluation source bytes changed")
    if base.sha256(Path(request["training_source"])/"SOURCE_MANIFEST.json") != request["training_source_manifest_sha256"]:
        raise ValueError("Frozen Ultra training source manifest changed")
    if base.sha256(request["experiment_config"]) != request["experiment_sha256"]:
        raise ValueError("Frozen Ultra experiment configuration changed")
    _, helper = _experiment(request["experiment_config"])
    if helper != Path(request["helper_root"]).resolve():
        raise ValueError("HOI80 helper source binding changed")
    owning_run(request["training_root"], request["parent_run_id"])
    return canonical_cases(request["canonical"])


def _rates(summary):
    values = {}
    for key in (*HOI80_METRICS, "tables_reward_survival_rate"):
        value = summary.get(key)
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError("HOI80 success measures must be finite rates in [0,1]")
        values[key] = value
    return values


def _check_rates(summary, clips):
    rates = _rates(summary)
    for key in HOI80_METRICS:
        field = key.removesuffix("_rate")
        if any(type(c.get(field)) is not bool for c in clips):
            raise ValueError("HOI80 per-clip canonical outcomes must be Boolean")
        if abs(rates[key]-sum(c[field] for c in clips)/len(clips)) > 1e-12:
            raise ValueError("HOI80 summary canonical rate differs from its complete case outcomes")
    if any(type(c.get("failed")) is not bool or c["joint_pass"] != (c["human_pass"] and c["object_pass"]) for c in clips):
        raise ValueError("HOI80 per-clip joint/survival outcomes are inconsistent")
    survival = sum(not c["failed"] for c in clips)/len(clips)
    if abs(rates["tables_reward_survival_rate"]-survival) > 1e-12 or abs(summary["success_rate"]-survival) > 1e-12:
        raise ValueError("HOI80 canonical success was confused with TablesReward survival")
    return rates


def validate_result(path, request):
    expected = validate_request(request)
    report = json.loads(Path(path).read_text())
    if (report.get("schema") != SCHEMA or report.get("passed") is not True
            or report.get("smoke") is not False or report.get("scheduled_evaluation") is not True):
        raise ValueError("Only a complete scheduled direct-teacher HOI80 result may be uploaded")
    for key in ("iteration", "parent_run_id", "checkpoint_sha256", "manifest_sha256", "canonical_sha256", "eval_source_sha256"):
        if report.get(key) != request[key]:
            raise ValueError("HOI80 result identity differs from its request: "+key)
    if report.get("partition") not in ("canonical_hoi80", "protected_eval"):
        raise ValueError("HOI80 result has the wrong evaluation partition")
    for flag in ("no_normalizer_update", "no_optimizer_update", "full_reference_duration", "no_failure_reset"):
        if report.get(flag) is not True:
            raise ValueError("HOI80 evaluation violated its frozen/full-duration contract")
    if report.get("input_sha256") != request["input_sha256"] or report.get("protocol_files_sha256") != PROTOCOL_FILES:
        raise ValueError("HOI80 result input or canonical scoring implementation identity differs")
    models = report.get("model_state_sha256", {})
    if (set(models) != {"policy", "wm", "critic"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in models.values())
            or report.get("policy_state_sha256_before") != models.get("policy")
            or report.get("policy_state_sha256_after") != models.get("policy")):
        raise ValueError("HOI80 frozen model/RMS identity proof is incomplete")
    if any(report.get(k) != 0 for k in ("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates")):
        raise ValueError("HOI80 evaluation performed training updates")
    clips = report.get("clips", [])
    if (report.get("clip_count") != 80 or len(clips) != 80
            or Counter(c.get("clip_id") for c in clips) != Counter(expected.keys())):
        raise ValueError("Canonical HOI80 result is missing, duplicating or substituting cases")
    for clip in clips:
        ref = expected[clip["clip_id"]]
        if (clip.get("canonical_case_id"), clip.get("canonical_index"), clip.get("motion_sha256")) != (
                ref["case_id"], ref["canonical_index"], ref["motion_sha256"]):
            raise ValueError("HOI80 scored case differs from the original canonical source motion")
        if (clip.get("asset"), clip.get("control_steps"), clip.get("reference_control_steps"), clip.get("state_frames")) != (
                ref["asset"], ref["control_steps"], ref["control_steps"], ref["state_frames"]):
            raise ValueError("HOI80 case changed asset or shortened its reference duration")
        if (clip.get("start_frame") != clip.get("reference_start_frame")
                or clip.get("final_frame") != clip.get("reference_end_frame")
                or clip.get("final_frame")-clip.get("start_frame") != ref["control_steps"]
                or clip.get("full_reference_duration") is not True or clip.get("no_failure_reset") is not True
                or clip.get("reset_calls_during_rollout") != 0):
            raise ValueError("HOI80 case was reset or failed to reach the exact reference end")
        if (clip.get("scored_frames") != ref["control_steps"] or clip.get("first_scored_state_index") != 1
                or clip.get("final_scored_state_index") != ref["control_steps"]):
            raise ValueError("HOI80 must score all post-action states and exclude the initial state")
    if report.get("control_steps") != CONTROL_STEPS:
        raise ValueError("HOI80 must score all 23718 control intervals")
    metrics = {"eval/iteration": request["iteration"], "eval/clip_count": 80, "eval/control_steps": CONTROL_STEPS}
    metrics.update({"eval/"+key: value for key, value in base._numeric_metrics(report.get("metrics")).items()})
    rates = _check_rates(report["metrics"], clips)
    metrics.update({"eval/hoi80/"+key: rates[key] for key in HOI80_METRICS})
    metrics["eval/tables_reward_survival_rate"] = rates["tables_reward_survival_rate"]
    if set(report.get("per_asset", {})) != set(ASSETS):
        raise ValueError("HOI80 per-asset summaries are incomplete")
    for index, asset in enumerate(ASSETS):
        summary = report["per_asset"][asset]
        steps = sum(c["control_steps"] for c in expected.values() if c["asset"] == asset)
        if summary.get("clip_count") != ASSET_COUNTS[index] or summary.get("control_steps") != steps:
            raise ValueError("HOI80 per-asset heterogeneous coverage changed")
        metrics.update({f"eval/assets/{asset}/"+key: value for key, value in base._numeric_metrics(summary.get("metrics")).items()})
        rates = _check_rates(summary["metrics"], [c for c in clips if c["asset"] == asset])
        metrics.update({f"eval/hoi80/assets/{asset}/"+key: rates[key] for key in HOI80_METRICS})
        metrics[f"eval/assets/{asset}/tables_reward_survival_rate"] = rates["tables_reward_survival_rate"]
    return metrics


class Slurm(base.Slurm):
    def submit(self, request_path, source_root, *, partition="batch", gres="gpu:a10:1", cpus=4, memory="12G", walltime="02:00:00"):
        request = json.loads(Path(request_path).read_text())
        try:
            validate_request(request)
        except Exception as error:
            # This is a definite pre-submit identity failure, not an ambiguous
            # scheduler acknowledgement eligible for lost-reply reconciliation.
            raise RuntimeError("Invalid immutable HOI80 submission intent") from error
        reply = self._run(["sbatch", "--parsable", "--no-requeue", f"--job-name={request['job_name']}",
            f"--partition={partition}", f"--gres={gres}", f"--cpus-per-task={cpus}", f"--mem={memory}",
            f"--time={walltime}", f"--output={Path(request_path).parent}/slurm-%j.log",
            str(Path(source_root)/"cluster/hoi_ultra_eval.sbatch.sh"), str(request_path), str(source_root)]).split(";", 1)[0]
        if not reply.isdecimal(): raise ValueError("HOI80 sbatch reply has no numeric job ID")
        return reply


class Supervisor(base.Supervisor):
    protocol = PROTOCOL
    expected_clips = staticmethod(expected_clips)
    checkpoint_pair = staticmethod(checkpoint_pair)
    validate_result = staticmethod(validate_result)
    validate_parent = staticmethod(owning_run)

    def __init__(self, root, campaign, training_root, source_root, training_source, parent_run_id,
                 *, slurm=None, train_job=None, resources=None):
        campaign = Path(campaign)
        paths = _input_paths(campaign)
        experiment_path = campaign/"experiment.json"
        _, helper = _experiment(experiment_path)
        canonical_cases(paths["canonical"])
        registration = owning_run(training_root, parent_run_id)
        metadata = json.loads((Path(training_root)/"run_metadata.json").read_text())
        for key in ("source_sha256", "manifest_sha256", "asset_catalog_sha256", "shape_bank_sha256", "reference_contacts_sha256"):
            if not isinstance(metadata.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", metadata[key]):
                raise ValueError("Ultra training metadata lacks a complete immutable source/data identity")
        hashes = {key: base.sha256(path) for key, path in paths.items()}
        for key, path_key in (("manifest_sha256", "manifest"), ("asset_catalog_sha256", "asset_bundle"),
                              ("shape_bank_sha256", "shape_bank"), ("reference_contacts_sha256", "reference_contacts")):
            if metadata[key] != hashes[path_key]:
                raise ValueError("Ultra evaluation inputs differ from their owning training metadata")
        from .evaluate_hoi80 import validate_source
        source = validate_source(training_source, source_root)
        self._extras = {**{key: str(path.resolve()) for key, path in paths.items()},
                       "canonical_sha256": CANONICAL_SHA256, "input_sha256": hashes,
                       "helper_root": str(helper), "helper_source_sha256": dict(PROTOCOL_FILES),
                       "experiment_config": str(experiment_path.resolve()), "experiment_sha256": base.sha256(experiment_path),
                       "training_source_manifest_sha256": source["training_source_manifest_sha256"],
                       "owning_project": registration["project"]}
        super().__init__(root, campaign, training_root, source_root, training_source, parent_run_id,
                         slurm=slurm or Slurm(), train_job=train_job, resources=resources)
        binding = {key: self._extras[key] for key in ("canonical_sha256", "input_sha256", "experiment_sha256", "training_source_manifest_sha256")}
        if self.state_path.exists() and any(self.state.get(key) != value for key, value in binding.items()):
            raise ValueError("Existing HOI80 ledger belongs to different canonical/source inputs")
        self.identity.update(binding)
        self.state.update(binding)

    def request_extra(self):
        return dict(self._extras)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--validate-request", type=Path, required=True)
    args = parser.parse_args()
    validate_request(json.loads(args.validate_request.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
