"""SDK-free sequential protected-22 checkpoint evaluation scheduler.

The active trainer and its frozen source are read-only.
Each immutable two-rank checkpoint pair gets one durable submission intent.
An ambiguous sbatch reply is reconciled by unique job name, never resubmitted.
Only a complete, manifest-bound 22-clip result is accepted. The unified monitor
publishes it to its owning training run with eval/iteration as the plotting axis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import zipfile

from .telemetry import ASSETS, ENTITY, PROJECT, atomic_json

PROTOCOL = "hoi_tables_full22_periodic_eval_v1"
INTERVAL = 2000
MILESTONES = tuple(range(INTERVAL, 20001, INTERVAL))
METRICS = ("success_rate", "mean_normalized_return", "object_position_error_m",
           "tracked_body_position_error_m", "frame_weighted_normalized_return",
           "object_orientation_error_rad", "mean_failure_free_seconds", "tracking_quality_fraction")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def complete_checkpoint(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("/data.pkl") for name in archive.namelist())
    except (FileNotFoundError, zipfile.BadZipFile):
        return False


def source_tree_hash(root):
    """Bind executable evaluation source bytes, excluding assets and caches."""
    root = Path(root)
    files = sorted(p for folder in ("hoi_tables", "hoi_residual", "supermimic", "cluster")
                   for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh"))
    if not files:
        raise ValueError("Evaluation source snapshot is empty")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def checkpoint_pair(training_root, iteration):
    if type(iteration) is not int or iteration not in MILESTONES:
        raise ValueError("Scheduled evaluation requires a 2000-step milestone")
    return {f"rank_{rank:02d}": str(Path(training_root)/f"rank_{rank:02d}/checkpoint_{iteration:06d}.pt") for rank in range(2)}


def expected_clips(manifest_path):
    value = json.loads(Path(manifest_path).read_text())
    clips = {entry["clip_id"]: {"asset": entry["object_asset"], "control_steps": int(entry["control_steps"])}
             for entry in value["clips"] if entry["partition"] == "protected_eval"}
    if len(clips) != 22 or {asset: sum(c["asset"] == asset for c in clips.values()) for asset in ASSETS} != dict.fromkeys(ASSETS, 11):
        raise ValueError("Periodic evaluation must bind all 22 protected clips, eleven per table")
    if any(c["control_steps"] < 1 for c in clips.values()):
        raise ValueError("Protected evaluation clip has no control intervals")
    return clips


def validate_request(request):
    if request.get("protocol") != PROTOCOL or request.get("iteration") not in MILESTONES:
        raise ValueError("Invalid scheduled evaluation request")
    if (set(request.get("checkpoint_sha256", {})) != {"rank_00", "rank_01"}
            or set(request.get("checkpoints", {})) != {"rank_00", "rank_01"}):
        raise ValueError("Evaluation requires both immutable rank checkpoints")
    for rank, path in request["checkpoints"].items():
        if not complete_checkpoint(path) or sha256(path) != request["checkpoint_sha256"][rank]:
            raise ValueError("Evaluation checkpoint bytes changed after dispatch")
    if sha256(request["manifest"]) != request["manifest_sha256"]:
        raise ValueError("Evaluation manifest changed after dispatch")
    if source_tree_hash(request["source_root"]) != request["eval_source_sha256"]:
        raise ValueError("Frozen evaluation source bytes changed after dispatch")
    return expected_clips(request["manifest"])


def validate_parent(training_root, parent_run_id):
    registration = json.loads((Path(training_root)/"wandb_registration.json").read_text())
    if (registration.get("id"), registration.get("entity"), registration.get("project")) != (parent_run_id, ENTITY, PROJECT):
        raise ValueError("Evaluation parent run does not match the active training registration")


def _numeric_metrics(values):
    if not isinstance(values, dict) or not set(METRICS) <= values.keys():
        raise ValueError("Full evaluation summary metrics are missing")
    result = {}
    for key in METRICS:
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Evaluation metrics must be finite numeric scalars")
        if key in ("success_rate", "tracking_quality_fraction") and not 0 <= value <= 1:
            raise ValueError("Invalid evaluation success/quality rate")
        result[key] = value
    return result


def validate_result(path, request):
    expected = validate_request(request)
    result = json.loads(Path(path).read_text())
    if result.get("passed") is not True or result.get("smoke") is not False or result.get("scheduled_evaluation") is not True:
        raise ValueError("Only completed scheduled evaluations may be published")
    if (result.get("iteration"), result.get("parent_run_id"), result.get("partition"), result.get("checkpoint_sha256")) != (
            request["iteration"], request["parent_run_id"], "protected_eval", request["checkpoint_sha256"]):
        raise ValueError("Evaluation result identity does not match the dispatched checkpoint pair")
    if result.get("manifest_sha256") != request["manifest_sha256"]:
        raise ValueError("Evaluation result manifest identity differs from the protected reference")
    if "eval_source_sha256" in result and result["eval_source_sha256"] != request["eval_source_sha256"]:
        raise ValueError("Evaluation result source identity differs from the frozen evaluator")
    for flag in ("no_normalizer_update", "no_optimizer_update", "full_reference_duration", "no_failure_reset"):
        if result.get(flag) is not True:
            raise ValueError("Evaluation failed its frozen/full-duration execution contract")
    clips = result.get("clips", [])
    if result.get("clip_count") != 22 or len(clips) != 22 or {c.get("clip_id") for c in clips} != set(expected):
        raise ValueError("Evaluation must include every protected clip exactly once")
    controls = 0
    for clip in clips:
        reference = expected[clip["clip_id"]]
        length = reference["control_steps"]
        if (clip.get("asset"), clip.get("control_steps"), clip.get("reference_control_steps"), clip.get("state_frames")) != (
                reference["asset"], length, length, length+1):
            raise ValueError("Evaluation clip was cropped or changed reference identity")
        if (clip.get("start_frame") != clip.get("reference_start_frame")
                or clip.get("final_frame") != clip.get("reference_end_frame")
                or clip.get("final_frame")-clip.get("start_frame") != length
                or clip.get("full_reference_duration") is not True
                or clip.get("no_failure_reset") is not True
                or clip.get("reset_calls_during_rollout") != 0):
            raise ValueError("Evaluation did not execute the complete reference without failure resets")
        controls += length
    if result.get("control_steps") != controls:
        raise ValueError("Evaluation total control duration differs from the frozen protected set")
    metrics = {"eval/iteration": request["iteration"], "eval/clip_count": 22, "eval/control_steps": controls}
    metrics.update({"eval/"+k: v for k, v in _numeric_metrics(result.get("metrics")).items()})
    per_asset = result.get("per_asset", {})
    if set(per_asset) != set(ASSETS):
        raise ValueError("Evaluation asset summaries are incomplete")
    for asset in ASSETS:
        summary = per_asset[asset]
        duration = sum(c["control_steps"] for c in expected.values() if c["asset"] == asset)
        if summary.get("clip_count") != 11 or summary.get("control_steps") != duration:
            raise ValueError("Per-asset protected evaluation coverage is incomplete")
        metrics.update({f"eval/assets/{asset}/"+k: v for k, v in _numeric_metrics(summary.get("metrics")).items()})
    return metrics


class Slurm:
    def _run(self, args):
        return subprocess.run(args, capture_output=True, text=True, timeout=45, check=True).stdout.strip()

    def submit(self, request_path, source_root, *, partition="batch", gres="gpu:a10:1", cpus=4, memory="12G", walltime="02:00:00"):
        request = json.loads(Path(request_path).read_text())
        command = ["sbatch", "--parsable", "--no-requeue", f"--job-name={request['job_name']}",
                   f"--partition={partition}", f"--gres={gres}", f"--cpus-per-task={cpus}", f"--mem={memory}",
                   f"--time={walltime}", f"--output={Path(request_path).parent}/slurm-%j.log",
                   str(Path(source_root)/"cluster/hoi_tables_eval.sbatch.sh"), str(request_path), str(source_root)]
        reply = self._run(command).split(";", 1)[0]
        if not reply.isdecimal():
            raise ValueError("Scheduler submit did not return a numeric job ID")
        return reply

    def find(self, name, created):
        since = (datetime.fromisoformat(created)-timedelta(days=1)).strftime("%Y-%m-%d")
        rows = self._run(["sacct", "-X", "--noheader", "--parsable2", "--name", name,
                          "--starttime", since, "--format=JobIDRaw,JobName%128"])
        result = {line.split("|")[0] for line in rows.splitlines() if len(line.split("|")) >= 2 and line.split("|")[1] == name}
        queued = self._run(["squeue", "--noheader", "--name", name, "--format=%i|%j"])
        result.update(line.split("|")[0] for line in queued.splitlines() if len(line.split("|")) == 2 and line.split("|")[1] == name)
        return sorted(result)

    def status(self, job):
        try:
            queued = self._run(["squeue", "--noheader", "--jobs", str(job), "--format=%T"])
        except subprocess.CalledProcessError as error:
            if "invalid job id" not in (error.stderr or "").lower(): raise
            queued = ""
        if queued:
            return {"state": queued.splitlines()[0].strip().split("+")[0], "exitcode": None}
        records = self._run(["sacct", "-X", "--noheader", "--parsable2", "--jobs", str(job), "--format=State,ExitCode"])
        if not records:
            return {"state": "UNKNOWN", "exitcode": None}
        state, code = records.splitlines()[0].split("|")[:2]
        return {"state": state.split()[0].split("+")[0], "exitcode": int(code.split(":")[0])}


class Supervisor:
    """Durable scheduler ledger, independent from SDK/network telemetry."""
    protocol = PROTOCOL
    expected_clips = staticmethod(expected_clips)
    checkpoint_pair = staticmethod(checkpoint_pair)
    validate_result = staticmethod(validate_result)
    validate_parent = staticmethod(validate_parent)

    def request_extra(self):
        """Profile-specific immutable request bindings; empty for tables."""
        return {}

    def __init__(self, root, campaign, training_root, source_root, training_source, parent_run_id,
                 *, slurm=None, train_job=None, resources=None):
        self.root, self.campaign, self.training_root = Path(root), Path(campaign), Path(training_root)
        self.source_root, self.training_source = Path(source_root), Path(training_source)
        self.parent_run_id, self.train_job = parent_run_id, train_job
        self.validate_parent(self.training_root, parent_run_id)
        self.slurm, self.resources = slurm or Slurm(), resources or {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.campaign/"input/manifest.json"
        self.expected_clips(self.manifest)
        self.state_path = self.root/"eval_supervisor.json"
        self.identity = {"protocol": self.protocol, "parent_run_id": parent_run_id,
                         "training_root": str(self.training_root.resolve()), "source_root": str(self.source_root.resolve()),
                         "training_source": str(self.training_source.resolve()), "manifest_sha256": sha256(self.manifest),
                         "eval_source_sha256": source_tree_hash(self.source_root)}
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {**self.identity, "evaluations": {}}
        if any(self.state.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Existing evaluation ledger belongs to another source/data/training campaign")

    def save(self):
        atomic_json(self.state_path, self.state)

    def _discover(self):
        for iteration in MILESTONES:
            key = str(iteration)
            if key in self.state["evaluations"]: continue
            pair = self.checkpoint_pair(self.training_root, iteration)
            if not all(complete_checkpoint(path) for path in pair.values()): continue
            directory = self.root/f"iter_{iteration:06d}"
            directory.mkdir(exist_ok=True)
            request = {**self.identity, "iteration": iteration, "created_utc": utc(),
                       "job_name": f"tables-eval-{self.parent_run_id}-{iteration}",
                       "checkpoints": pair, "checkpoint_sha256": {k: sha256(v) for k, v in pair.items()},
                       "manifest": str(self.manifest.resolve()),
                       "asset_bundle": str((self.campaign/"input/assets.json").resolve()),
                       "shape_bank": str((self.campaign/"input/shape_bank.json").resolve()),
                       "reference_contacts": str((self.campaign/"formal_inputs01/reference_contacts.pt").resolve()),
                       "output": str((directory/"evaluation").resolve()), **self.request_extra()}
            request_path = directory/"request.json"
            if request_path.exists():
                existing = json.loads(request_path.read_text())
                for field in ("protocol", "iteration", "parent_run_id", "checkpoint_sha256", "manifest_sha256"):
                    if existing.get(field) != request[field]:
                        raise ValueError("Existing evaluation intent has different immutable inputs")
                request = existing
            else:
                atomic_json(request_path, request)
            self.state["evaluations"][key] = {"status": "ready", "request": str(request_path),
                                                "request_sha256": sha256(request_path)}
            self.save()

    def _request(self, entry):
        if sha256(entry["request"]) != entry["request_sha256"]:
            raise ValueError("Evaluation submission intent changed")
        return json.loads(Path(entry["request"]).read_text())

    def _parent_exitcode(self):
        try:
            value = (self.training_root/"training_exitcode").read_text().strip()
            if value: return int(value)
        except FileNotFoundError:
            pass
        if self.train_job is not None:
            status = self.slurm.status(self.train_job)
            if status["state"] in TERMINAL:
                return (status["exitcode"] or 0) if status["state"] == "COMPLETED" else (status["exitcode"] or 1)
        return None

    def tick(self):
        self._discover()
        entries = self.state["evaluations"]
        if any(entry["status"] == "failed" for entry in entries.values()):
            raise RuntimeError("A protected evaluation failed; refusing to dispatch further GPU jobs")
        active = [e for e in entries.values() if e["status"] in ("submission_pending", "submitted")]
        if len(active) > 1:
            raise RuntimeError("More than one protected evaluation producer is active")
        if active:
            entry = active[0]
            request = self._request(entry)
            if entry["status"] == "submission_pending":
                jobs = self.slurm.find(request["job_name"], request["created_utc"])
                if len(jobs) > 1:
                    raise RuntimeError("Duplicate evaluation scheduler submissions detected")
                if jobs:
                    entry.update(status="submitted", job_id=jobs[0])
                    self.save()
                elif time.time()-entry["submission_started"] > 300:
                    raise RuntimeError("Lost scheduler submit reply remains unresolved; refusing duplicate submission")
            if entry["status"] == "submitted":
                status = self.slurm.status(entry["job_id"])
                entry["scheduler_state"] = status["state"]
                if status["state"] in TERMINAL:
                    try:
                        if status["state"] != "COMPLETED" or status["exitcode"] not in (0, None):
                            raise RuntimeError("GPU evaluation job did not complete successfully")
                        result_path = Path(request["output"])/"result.json"
                        metrics = self.validate_result(result_path, request)
                        entry.update(status="complete", metrics=metrics, result=str(result_path), result_sha256=sha256(result_path))
                    except Exception as error:
                        entry.update(status="failed", error_type=type(error).__name__, error=str(error),
                                     result=str(Path(request["output"])/"result.json"), scheduler_exitcode=status["exitcode"])
                    self.save()
                    if entry["status"] == "failed":
                        raise RuntimeError("Protected evaluation worker or result identity failed; review durable ledger")
        active = any(e["status"] in ("submission_pending", "submitted") for e in entries.values())
        if not active:
            ready = sorted((int(i), e) for i, e in entries.items() if e["status"] == "ready")
            if ready:
                _, entry = ready[0]
                entry.update(status="submission_pending", submission_started=time.time())
                self.save()  # Durable claim precedes the scheduler side effect.
                try:
                    job = self.slurm.submit(entry["request"], self.source_root, **self.resources)
                    entry.update(status="submitted", job_id=str(job))
                    self.save()
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                    # Lost acceptance replies are reconciled on the next tick.
                    entry["submit_reply_lost"] = True
                    self.save()
                active = True
        parent_exitcode = self._parent_exitcode()
        pending = active or any(e["status"] == "ready" for e in entries.values())
        if parent_exitcode is not None and not pending:
            missing = [i for i in MILESTONES if str(i) not in entries]
            self.state.update(finished=True, parent_training_exitcode=parent_exitcode, missing_milestones=missing,
                              failed_milestones=[int(i) for i, e in entries.items() if e["status"] == "failed"], finished_utc=utc())
            self.save()
        return self.state



DISABLED_WRITER = (
    "Separate evaluation W&B writers are disabled; use python -m "
    "hoi_tables.unified_monitor to merge evaluation into its owning training run"
)


def register(*args, **kwargs):
    """Reject the historical separate-run API without importing an SDK."""
    raise RuntimeError(DISABLED_WRITER)


def monitor(*args, **kwargs):
    """Reject the historical separate-run API without importing an SDK."""
    raise RuntimeError(DISABLED_WRITER)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__, epilog=DISABLED_WRITER)
    parser.add_argument("--validate-request", type=Path)
    args, unknown = parser.parse_known_args(argv)
    if unknown or args.validate_request is None:
        parser.error(DISABLED_WRITER)
    validate_request(json.loads(args.validate_request.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
