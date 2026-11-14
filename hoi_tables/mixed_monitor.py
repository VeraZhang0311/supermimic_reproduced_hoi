"""Schedule one checkpoint-bound HOI80 and protected tables20 evaluation."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from . import eval_monitor as base
from .hoi80_monitor import canonical_cases, _check_rates
from .experiments import TABLE_ASSETS
from .telemetry import atomic_json
from .table_actions import action_key
from .trajectories import validate_trajectory

PROTOCOL = "hoi_ultra_tables_same_run_eval_v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inputs(campaign):
    config = _read(campaign / "experiment.json")
    ultra = _read(config["ultra_experiment"])
    paths = {name: ultra["paths"][name] for name in ("manifest", "asset_bundle", "geometry_bundle", "shape_bank", "reference_contacts")}
    paths["canonical"] = ultra["evaluation"]["canonical"]
    return config, ultra, paths


def table_cases(path):
    manifest = _read(path)
    rows = {c["clip_id"]: c for c in manifest["clips"] if c["partition"] == "protected_eval"}
    if len(rows) != 20 or Counter(c["object_asset"] for c in rows.values()) != dict.fromkeys(TABLE_ASSETS, 10):
        raise ValueError("Mixed protected tables require exactly 10 clips per asset")
    return rows


def validate_request(request):
    campaign = Path(request["campaign"])
    config, ultra, paths = _inputs(campaign)
    iteration = request["iteration"]
    if (request["protocol"] != PROTOCOL or type(iteration) is not int
            or not 8000 < iteration <= config["training"]["iterations"] or iteration % 2000
            or request["paths"] != paths or request["helper_root"] != ultra["evaluation"]["helper_root"]
            or request["experiment_sha256"] != base.sha256(campaign / "experiment.json")
            or request["table_manifest_sha256"] != base.sha256(campaign / "tables/manifest.json")
            or request["ultra_inputs"] != {key: base.sha256(path) for key, path in paths.items()}):
        raise ValueError("Mixed evaluation request input or cadence changed")
    checkpoint = Path(request["training_root"]) / f"checkpoint_{iteration:06d}.pt"
    if request["checkpoints"] != {"mixed": str(checkpoint)} or request["checkpoint_sha256"] != {"mixed": base.sha256(checkpoint)}:
        raise ValueError("Mixed evaluation checkpoint binding changed")
    if base.source_tree_hash(request["source_root"]) != request["eval_source_sha256"]:
        raise ValueError("Mixed evaluator source changed")
    output = Path(request["output"]).resolve()
    if any(output.is_relative_to(Path(request[key]).resolve()) for key in ("training_root", "source_root")):
        raise ValueError("Mixed eval output must be separate from training and source directories")
    base.validate_parent(request["training_root"], request["parent_run_id"])
    return table_cases(campaign / "tables/manifest.json"), canonical_cases(paths["canonical"])


def _coverage(clips, expected, *, table):
    if Counter(c["clip_id"] for c in clips) != Counter(expected.keys()):
        raise ValueError("Mixed eval has missing, duplicate or additional cases")
    for clip in clips:
        source = expected[clip["clip_id"]]
        if (clip["asset"] != source["object_asset" if table else "asset"]
                or clip["state_frames"] != source["state_frames"] or clip["control_steps"] != source["control_steps"]
                or clip["final_frame"] - clip["start_frame"] != source["control_steps"]
                or not clip["full_reference_duration"] or not clip["no_failure_reset"] or clip["reset_calls_during_rollout"]):
            raise ValueError("Mixed eval case does not match its full frozen reference")
        if table and (clip["action"], clip["source_group"]) != (source["action"], source["source_group"]):
            raise ValueError("Mixed protected table case action or source changed")
        if not table and (clip["canonical_case_id"], clip["canonical_index"], clip["motion_sha256"]) != (
                source["case_id"], source["canonical_index"], source["motion_sha256"]):
            raise ValueError("Mixed HOI80 case canonical identity changed")


def validate_result(path, request):
    from .table_trajectories import verify_trajectory

    table_expected, ultra_expected = validate_request(request)
    result = _read(path)
    if (result["protocol"] != PROTOCOL or result["passed"] is not True or result["iteration"] != request["iteration"]
            or result["parent_run_id"] != request["parent_run_id"] or result["checkpoint_sha256"] != request["checkpoint_sha256"]):
        raise ValueError("Mixed evaluation aggregate identity mismatch")
    metrics = {"eval/iteration": request["iteration"]}
    for name, expected in (("hoi80", ultra_expected), ("tables20", table_expected)):
        report_path = Path(path).parent / name / "result.json"
        if base.sha256(report_path) != result["reports"][name]:
            raise ValueError("Mixed suite result changed after aggregation")
        report = _read(report_path)
        for key in ("iteration", "parent_run_id", "checkpoint_sha256", "model_state_sha256"):
            if report[key] != result[key]:
                raise ValueError(f"Mixed suite checkpoint identity differs: {name}/{key}")
        if report["passed"] is not True or any(report[k] != 0 for k in ("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates")):
            raise ValueError("Mixed evaluation must be complete and read-only")
        clips = report["clips"]
        _coverage(clips, expected, table=name == "tables20")
        if set(report["per_asset"]) != {clip["asset"] for clip in clips}:
            raise ValueError("Mixed evaluation per-asset report coverage differs")
        if name == "tables20" and set(report["per_action"]) != {clip["action"] for clip in clips}:
            raise ValueError("Mixed tables per-action report coverage differs")
        if name == "hoi80":
            rates = _check_rates(report["metrics"], clips)
            metrics.update({f"eval/hoi80/{key}": value for key, value in rates.items()})
            for clip in clips:
                validate_trajectory(report_path.parent, clip, report)
        if name == "tables20" and request.get("record_table_trajectories"):
            for clip in clips:
                verify_trajectory(report_path.parent, clip)
        metrics[f"eval/{name}/clip_count"] = len(clips)
        metrics[f"eval/{name}/control_steps"] = sum(c["control_steps"] for c in clips)
        for key, value in base._numeric_metrics(report["metrics"]).items():
            metrics[f"eval/{name}/{key}"] = value
        for group in ("per_asset", "per_action"):
            for label, summary in report.get(group, {}).items():
                namespace = label.replace(" ", "_")
                for key, value in base._numeric_metrics(summary["metrics"]).items():
                    metrics[f"eval/{name}/{group}/{namespace}/{key}"] = value
                if name == "hoi80" and group == "per_asset":
                    rates = _check_rates(summary["metrics"], [c for c in clips if c["asset"] == label])
                    metrics.update({f"eval/hoi80/per_asset/{namespace}/{key}": value for key, value in rates.items()})
                if name == "tables20" and group == "per_action":
                    rows = [clip for clip in clips if clip["action"] == label]
                    if summary["clip_count"] != len(rows):
                        raise ValueError("Protected table action clip count differs")
                    prefix = f"eval/tables20/per_action/{action_key(label)}"
                    metrics[f"{prefix}/clip_count"] = len(rows)
                    metrics[f"{prefix}/failure_rate"] = sum(clip["failed"] for clip in rows) / len(rows)
    return metrics


class Slurm(base.Slurm):
    def submit(self, request_path, source_root, **resources):
        request = _read(request_path)
        reply = self._run(["sbatch", "--parsable", f"--job-name={request['job_name']}",
                           f"--output={Path(request_path).parent}/slurm-%j.log",
                           str(Path(source_root) / "cluster/hoi_mixed_eval.sbatch.sh"), str(request_path), str(source_root)])
        job = reply.split(";", 1)[0]
        if not job.isdecimal():
            raise ValueError("Mixed evaluation scheduler did not return a numeric job ID")
        return job


class Supervisor(base.Supervisor):
    protocol = PROTOCOL
    validate_result = staticmethod(validate_result)

    def __init__(self, root, campaign, training_root, source_root, training_source, parent_run_id, *, train_job=None):
        self.root, self.campaign, self.training_root = map(lambda path: Path(path).resolve(), (root, campaign, training_root))
        self.source_root, self.training_source = Path(source_root).resolve(), Path(training_source).resolve()
        self.parent_run_id, self.train_job, self.slurm, self.resources = parent_run_id, train_job, Slurm(), {}
        self.validate_parent(self.training_root, parent_run_id)
        config, ultra, self.paths = _inputs(self.campaign)
        self.last_iteration = config["training"]["iterations"]
        self.milestones = range(10000, self.last_iteration + 1, 2000)
        self.helper_root = ultra["evaluation"]["helper_root"]
        self.identity = {"protocol": PROTOCOL, "parent_run_id": parent_run_id, "campaign": str(self.campaign),
                         "training_root": str(self.training_root), "source_root": str(self.source_root),
                         "training_source": str(self.training_source), "experiment_sha256": base.sha256(self.campaign / "experiment.json"),
                         "table_manifest_sha256": base.sha256(self.campaign / "tables/manifest.json"),
                         "eval_source_sha256": base.source_tree_hash(self.source_root)}
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "eval_supervisor.json"
        self.state = _read(self.state_path) if self.state_path.exists() else {**self.identity, "evaluations": {}}
        source_fields = ("source_root", "eval_source_sha256")
        if any(self.state.get(key) != value for key, value in self.identity.items() if key not in source_fields):
            raise ValueError("Mixed eval ledger belongs to a different experiment or run")
        if any(self.state.get(key) != self.identity[key] for key in source_fields):
            if any(entry["status"] != "complete" for entry in self.state["evaluations"].values()):
                raise ValueError("Cannot replace the evaluator source while evaluations are outstanding")
            self.state.update({key: self.identity[key] for key in source_fields})
            self.save()

    def _discover(self):
        for iteration in self.milestones:
            key = str(iteration)
            checkpoint = self.training_root / f"checkpoint_{iteration:06d}.pt"
            if key in self.state["evaluations"] or not base.complete_checkpoint(checkpoint):
                continue
            directory = self.root / f"iter_{iteration:06d}"
            directory.mkdir(exist_ok=True)
            path = directory / "request.json"
            request = {**self.identity, "iteration": iteration, "created_utc": base.utc(), "paths": self.paths,
                       "record_table_trajectories": True,
                       "job_name": f"hoi-mix-eval-{self.parent_run_id}-{iteration}", "helper_root": self.helper_root,
                       "checkpoints": {"mixed": str(checkpoint)}, "checkpoint_sha256": {"mixed": base.sha256(checkpoint)},
                       "output": str(directory / "evaluation"),
                       "ultra_inputs": {key: base.sha256(value) for key, value in self.paths.items()}}
            if path.exists():
                request = _read(path)
                validate_request(request)
            else:
                atomic_json(path, request)
            self.state["evaluations"][key] = {"status": "ready", "request": str(path), "request_sha256": base.sha256(path)}
            self.save()
