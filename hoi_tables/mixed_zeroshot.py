"""Run and publish protected tables evaluations between scheduled milestones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import eval_monitor as base
from .experiments import TABLE_ASSETS
from .mixed_monitor import Supervisor as ScheduledSupervisor, _coverage, table_cases
from .table_actions import action_key
from .telemetry import atomic_json

ITERATION = 8000
PROTOCOL = "hoi_ultra_8000_tables20_zeroshot_v1"
SUPPLEMENTAL = "hoi_mixed_tables20_supplemental_v1"
READ_ONLY = ("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _binding(campaign, training_root, iteration):
    config = _read(campaign / "experiment.json")
    if iteration != ITERATION:
        if (type(iteration) is not int or not ITERATION < iteration <= config["training"]["iterations"]
                or iteration % config["training"]["checkpoint_every"] or iteration % 2000 == 0):
            raise ValueError("Supplemental tables20 requires a saved, non-milestone mixed checkpoint")
        path = training_root / f"checkpoint_{iteration:06d}.pt"
        if not base.complete_checkpoint(path):
            raise ValueError("Supplemental mixed checkpoint is incomplete")
        return {"checkpoints": {"mixed": str(path)}, "checkpoint_sha256": {"mixed": base.sha256(path)},
                "table_manifest_sha256": base.sha256(campaign / "tables/manifest.json")}
    request_path = Path(config["ultra_initialization"])
    request = _read(request_path)
    result = _read(request_path.parent / "evaluation/result.json")
    identity = _read(training_root / "identity.json")["ultra_initialization"]
    hashes = request["checkpoint_sha256"]
    if (request["iteration"] != ITERATION or result["iteration"] != ITERATION
            or result["passed"] is not True or result["clip_count"] != 80
            or set(request["checkpoints"]) != set(hashes) or set(hashes) != {f"rank_{i:02d}" for i in range(8)}
            or result["checkpoint_sha256"] != hashes
            or identity != {"iteration": ITERATION, "checkpoint_sha256": hashes,
                            "model_state_sha256": result["model_state_sha256"]}):
        raise ValueError("Zero-shot source differs from the verified mixed training initialization")
    return {"checkpoints": request["checkpoints"], "checkpoint_sha256": hashes,
            "model_state_sha256": result["model_state_sha256"],
            "ultra_request_sha256": base.sha256(request_path),
            "table_manifest_sha256": base.sha256(campaign / "tables/manifest.json")}


def prepare(campaign, training_root, directory, *, iteration=ITERATION):
    campaign, training_root, directory = (Path(p).resolve() for p in (campaign, training_root, directory))
    parent_run_id = _read(training_root / "wandb_registration.json")["id"]
    base.validate_parent(training_root, parent_run_id)
    request = {"protocol": PROTOCOL if iteration == ITERATION else SUPPLEMENTAL,
               "iteration": iteration, "parent_run_id": parent_run_id,
               "campaign": str(campaign), "training_root": str(training_root),
               "output": str(directory / "evaluation"), **_binding(campaign, training_root, iteration)}
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "request.json", request)


def _validate(request):
    campaign, training_root = map(Path, (request["campaign"], request["training_root"]))
    base.validate_parent(training_root, request["parent_run_id"])
    iteration = request["iteration"]
    if (type(iteration) is not int or request["protocol"] != (PROTOCOL if iteration == ITERATION else SUPPLEMENTAL)
            or Path(request["output"]).resolve().is_relative_to(training_root.resolve())
            or any(request[key] != value for key, value in _binding(campaign, training_root, iteration).items())):
        raise ValueError("Invalid standalone tables20 evaluation request")
    return table_cases(campaign / "tables/manifest.json")


def evaluate_asset(request, asset_id, *, record_trajectories=False):
    from .mixed import evaluation_checkpoint, load_source, read_experiment
    from .mixed_eval import table_asset
    from .mixed_train import _models
    from .native_hands import load_native_hands

    _validate(request)
    experiment = read_experiment(request["campaign"])
    if request["protocol"] == SUPPLEMENTAL:
        state = evaluation_checkpoint(experiment, request["checkpoints"]["mixed"], request["iteration"],
                                      request["checkpoint_sha256"]["mixed"], supplemental=True)
        if state["model_state_sha256"] != request.get("model_state_sha256", state["model_state_sha256"]):
            raise ValueError("Supplemental evaluator loaded different model tensors")
        table_asset(request, asset_id, state=state, record_trajectories=record_trajectories)
        return
    native = load_native_hands(experiment.ultra["paths"]["native_hands"])
    models = _models(experiment.config, "cpu")
    _, lineage = load_source(experiment, models, native)
    if any(lineage[key] != request[key] for key in ("iteration", "checkpoint_sha256", "model_state_sha256")):
        raise ValueError("Loaded Ultra 8000 does not match zero-shot request")
    state = {"policy_state": models["policy"].state_dict(), "identity": {"native_hands": native.identity},
             "model_state_sha256": lineage["model_state_sha256"]}
    table_asset(request, asset_id, state=state, record_trajectories=record_trajectories)


def aggregate(request):
    from .evaluate import summarize
    from .table_trajectories import verify_trajectory

    expected = _validate(request)
    output = Path(request["output"])
    identity = {key: request[key] for key in ("iteration", "parent_run_id", "checkpoint_sha256")}
    identity["model_state_sha256"] = request.get("model_state_sha256")
    if identity["model_state_sha256"] is None:
        identity["model_state_sha256"] = _read(output / "tables20" / TABLE_ASSETS[0] / "report.json")["model_state_sha256"]
    clips = []
    for asset in TABLE_ASSETS:
        report = _read(output / "tables20" / asset / "report.json")
        policy_hash = identity["model_state_sha256"]["policy"]
        if (report["asset"] != asset or report["passed"] is not True
                or any(report[key] != value for key, value in identity.items())
                or any(report[key] != policy_hash for key in ("policy_state_sha256_before", "policy_state_sha256_after"))
                or any(report[key] != 0 for key in READ_ONLY)
                or any(clip["asset"] != asset for clip in report["clips"])):
            raise ValueError("Table worker failed frozen-policy validation")
        clips.extend(report["clips"])
    _coverage(clips, expected, table=True)
    if any("trajectory_sha256" in clip for clip in clips):
        for clip in clips:
            verify_trajectory(output / "tables20", clip, verify_payload=True)
    atomic_json(output / "result.json", {**identity, "protocol": request["protocol"], "passed": True, "clips": clips,
                "metrics": summarize(clips), **dict.fromkeys(READ_ONLY, 0)})


def validate_result(path, request):
    from .evaluate import summarize

    expected = _validate(request)
    result = _read(path)
    identity = ("protocol", "iteration", "parent_run_id", "checkpoint_sha256")
    if "model_state_sha256" in request:
        identity += ("model_state_sha256",)
    if (result["passed"] is not True or any(result[key] != request[key] for key in identity)
            or any(result[key] != 0 for key in READ_ONLY)):
        raise ValueError("Standalone tables20 result identity mismatch")
    clips = result["clips"]
    _coverage(clips, expected, table=True)
    if result["metrics"] != summarize(clips):
        raise ValueError("Standalone tables20 aggregate differs from its clip records")
    metrics = {"eval/iteration": request["iteration"], "eval/tables20/zero_shot": int(request["protocol"] == PROTOCOL)}

    def add_group(prefix, rows):
        values = base._numeric_metrics(summarize(rows))
        values.update(clip_count=len(rows), control_steps=sum(row["control_steps"] for row in rows),
                      failure_rate=sum(row["failed"] for row in rows) / len(rows))
        metrics.update({f"{prefix}/{key}": value for key, value in values.items()})

    add_group("eval/tables20", clips)
    for asset in TABLE_ASSETS:
        add_group(f"eval/tables20/per_asset/{asset}", [row for row in clips if row["asset"] == asset])
    for action in sorted({row["action"] for row in clips}):
        add_group(f"eval/tables20/per_action/{action_key(action)}", [row for row in clips if row["action"] == action])
    return metrics


class Supervisor(ScheduledSupervisor):
    """Include validated standalone results in the owning writer."""

    @staticmethod
    def validate_result(path, request):
        if request["protocol"] in (PROTOCOL, SUPPLEMENTAL):
            return validate_result(path, request)
        return ScheduledSupervisor.validate_result(path, request)

    def tick(self):
        state = super().tick()
        intents = [self.root / "zeroshot_008000.json", *sorted(self.root.glob("tables20_[0-9]*.json"))]
        for intent in intents:
            iteration = int(intent.stem.rsplit("_", 1)[1])
            key = str(iteration)
            if key in state["evaluations"] or not intent.is_file():
                continue
            entry = _read(intent)
            if self.slurm.status(entry["job_id"])["state"] == "COMPLETED":
                request = self._request(entry)
                if (request["parent_run_id"] != self.parent_run_id or request["training_root"] != str(self.training_root)
                        or request["iteration"] != iteration):
                    raise ValueError("Standalone intent does not match the training run and iteration")
                result = Path(request["output"]) / "result.json"
                metrics = validate_result(result, request)
                state["evaluations"][key] = {**entry, "status": "complete", "metrics": metrics,
                                              "result": str(result), "result_sha256": base.sha256(result)}
                self.save()
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--table-asset", type=int, choices=range(len(TABLE_ASSETS)))
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--record-trajectories", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    request = _read(args.request)
    if args.aggregate:
        aggregate(request)
    else:
        evaluate_asset(request, args.table_asset, record_trajectories=args.record_trajectories)


if __name__ == "__main__":
    main()
