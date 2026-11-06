"""CPU supervisor for immutable GMT checkpoints and one W&B writer.

No training, GPU runtime, optimizer or simulator is imported here. Handoff is
explicit: the old SDK monitor and its core must be stopped before the supplied
JSON file confirms both facts. Evaluation runs in separate Slurm allocations.
Alternatively --create-run owns a new run from the training batch itself.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
EVAL_COLUMNS = ("iteration", "train_wall_h", "saved_at", "eval_at", "norm_return",
                "norm_return_framew", "n_failed", "failed")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(control):
    """Linux advisory lock held for the supervisor's entire SDK lifetime."""
    import fcntl
    control = Path(control)
    control.mkdir(parents=True, exist_ok=True)
    with (control / "supervisor.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def handoff_confirmed(path, run_id):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (isinstance(value, dict) and value.get("old_monitor_stopped") is True
            and value.get("old_core_stopped") is True and value.get("run_id") == run_id)


def load_monitor(source_root):
    path = Path(source_root) / "tools/wandb_gmt_monitor.py"
    spec = importlib.util.spec_from_file_location("_gmt_frozen_csv_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_result(path, checkpoint, iteration, expected_sha, *, expected_clips=40,
                    expected_eval_manifest_sha256=None):
    """Accept only a complete, manifest-bound receipt for these immutable bytes."""
    if type(expected_clips) is not int or expected_clips <= 0:
        raise ValueError("Expected clip count must be positive")
    if expected_clips != 40 and expected_eval_manifest_sha256 is None:
        raise ValueError("Custom evaluation counts require a bound manifest SHA256")
    if expected_eval_manifest_sha256 is not None and (not isinstance(expected_eval_manifest_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_eval_manifest_sha256) is None):
        raise ValueError("Expected evaluation manifest SHA256 is invalid")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise ValueError("Evaluation receipt did not pass")
    if value.get("checkpoint_iteration") != iteration or type(value.get("checkpoint_iteration")) is not int:
        raise ValueError("Evaluation iteration mismatch")
    if value.get("checkpoint_sha256") != expected_sha or sha256(checkpoint) != expected_sha:
        raise ValueError("Evaluation checkpoint hash mismatch")
    if "checkpoint_path" in value and Path(value["checkpoint_path"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Evaluation checkpoint path mismatch")
    for key in ("norm_return", "norm_return_framew", "train_wall_h"):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError(f"Evaluation {key} must be finite")
    if value["train_wall_h"] < 0:
        raise ValueError("Negative training wall time")
    if type(value.get("n_total")) is not int or value["n_total"] != expected_clips:
        raise ValueError(f"Evaluation must cover all {expected_clips} clips")
    if expected_eval_manifest_sha256 is not None and value.get("eval_manifest_sha256") != expected_eval_manifest_sha256:
        raise ValueError("Evaluation manifest SHA256 mismatch")
    if type(value.get("n_failed")) is not int or not 0 <= value["n_failed"] <= expected_clips:
        raise ValueError("Invalid failed clip count")
    failed = value.get("failed")
    if (not isinstance(failed, list) or not all(isinstance(name, str) and name for name in failed)
            or len(failed) != value["n_failed"] or len(set(failed)) != len(failed)):
        raise ValueError("Failed clip identities do not match the count")
    if any(not isinstance(value.get(key), str) or not value[key] for key in ("saved_at", "eval_at")):
        raise ValueError("Evaluation receipt timestamps missing")
    # Preserve only the audited metadata/metrics; no raw motion or arbitrary fields.
    result = {key: value[key] for key in ("checkpoint_iteration", "checkpoint_sha256", "norm_return",
              "norm_return_framew", "n_failed", "n_total", "failed", "train_wall_h", "saved_at", "eval_at")}
    if expected_eval_manifest_sha256 is not None:
        result["eval_manifest_sha256"] = expected_eval_manifest_sha256
    return result


def rebuild_eval_csv(run_dir, checkpoints):
    results = sorted((entry["result"] for entry in checkpoints.values() if entry.get("result")),
                     key=lambda row: row["checkpoint_iteration"])
    path = Path(run_dir) / "eval.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        accepted = {row["checkpoint_iteration"] for row in results}
        if any(int(row["iteration"]) not in accepted for row in previous):
            raise ValueError("Existing evaluation CSV has rows outside this receipt ledger; preserving it")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EVAL_COLUMNS)
    for row in results:
        writer.writerow((row["checkpoint_iteration"], f"{row['train_wall_h']:.4f}", row["saved_at"],
                         row["eval_at"], f"{row['norm_return']:.4f}", f"{row['norm_return_framew']:.4f}",
                         row["n_failed"], "|".join(row["failed"])))
    temporary = path.with_name(f".eval.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Slurm:
    def _run(self, command):
        return subprocess.run(command, text=True, timeout=45, capture_output=True, check=True).stdout.strip()

    def submit(self, script, *, name, source_root, run_dir, checkpoint, output):
        reply = self._run(["sbatch", "--parsable", f"--job-name={name}", str(script),
                           "--source-root", str(source_root), "--run-dir", str(run_dir),
                           "--checkpoint", str(checkpoint), "--output", str(output)])
        job = reply.split(";", 1)[0]
        if not re.fullmatch(r"\d+", job):
            raise ValueError("sbatch did not return a numeric job ID")
        return job

    def status(self, job):
        try:
            queued = self._run(["squeue", "--noheader", "--jobs", str(job), "--format=%T"])
        except subprocess.CalledProcessError as error:
            if "invalid job id" not in (error.stderr or "").lower():
                raise
            queued = ""
        if queued:
            return {"state": queued.splitlines()[0].strip(), "exit_code": None}
        rows = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--jobs", str(job),
                          "--format=JobIDRaw,State,ExitCode"])
        for row in rows.splitlines():
            fields = row.split("|")
            if len(fields) >= 3 and fields[0] == str(job):
                state = fields[1].split()[0].rstrip("+")
                code, signal = (int(part) for part in fields[2].split(":"))
                return {"state": state, "exit_code": code or (128 + signal if signal else 0)}
        return {"state": "UNKNOWN", "exit_code": None}

    def find_jobs(self, name, since):
        # Job names are unique to run+checkpoint+attempt. Search live AND
        # accounting records before adopting a submission with a lost reply.
        live = self._run(["squeue", "--noheader", "--name", name, "--format=%i"])
        start = dt.datetime.fromtimestamp(since - 60).strftime("%Y-%m-%dT%H:%M:%S")
        past = self._run(["sacct", "--noheader", "--parsable2", "--allocations", "--name", name,
                          "--starttime", start, "--format=JobIDRaw,JobName%128"])
        ids = {line.strip() for line in live.splitlines() if re.fullmatch(r"\d+", line.strip())}
        for line in past.splitlines():
            fields = line.split("|")
            if len(fields) >= 2 and fields[1] == name and re.fullmatch(r"\d+", fields[0]):
                ids.add(fields[0])
        return sorted(ids)


class Supervisor:
    """Testable polling state machine. Constructor neither imports nor starts W&B."""
    def __init__(self, args, scheduler=None, monitor_module=None, clock=time.time):
        self.args, self.scheduler, self.clock = args, scheduler or Slurm(), clock
        self.control, self.run_dir = Path(args.control_dir), Path(args.run_dir)
        self.create_run = bool(getattr(args, "create_run", False))
        if self.create_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.expected_clips = getattr(args, "expected_clips", 40)
        self.eval_manifest = getattr(args, "eval_manifest", None)
        if type(self.expected_clips) is not int or self.expected_clips <= 0:
            raise ValueError("Expected clip count must be positive")
        if self.expected_clips != 40 and self.eval_manifest is None:
            raise ValueError("Custom evaluation counts require an evaluation manifest")
        self.eval_manifest_sha256 = sha256(self.eval_manifest) if self.eval_manifest is not None else None
        self.control.mkdir(parents=True, exist_ok=True)
        self.state_path = self.control / "ledger.json"
        binding = {"run_dir": str(self.run_dir.resolve()), "source_root": str(Path(args.source_root).resolve()),
                   "train_job_id": str(args.train_job_id), "wandb_entity": args.wandb_entity,
                   "wandb_project": args.wandb_project, "wandb_id": args.wandb_id,
                   "total_iterations": args.total_iterations, "interval": args.interval,
                   "expected_clips": self.expected_clips, "eval_manifest_sha256": self.eval_manifest_sha256,
                   "create_run": self.create_run}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            previous = dict(self.state.get("binding", {}))
            # A legacy full-40 takeover ledger has the same effective defaults.
            for key, default in (("expected_clips", 40), ("eval_manifest_sha256", None), ("create_run", False)):
                previous.setdefault(key, default)
            if previous != binding:
                raise ValueError("Control directory belongs to a different training/evaluation run")
            self.state["binding"] = binding
        else:
            unrelated = [p for p in self.control.iterdir() if p.name != "supervisor.lock"]
            if unrelated:
                raise FileExistsError("New control directory is not empty")
            self.state = {"version": 1, "binding": binding, "created_at": utc(), "train_cursor": 0,
                          "eval_logged": [], "checkpoints": {}}
            self.save()
        (self.control / "attempts").mkdir(exist_ok=True)
        self.monitor = monitor_module or load_monitor(args.source_root)
        self.reader = self.monitor.CSVReader(self.run_dir / "train_log.csv")
        self.run, self.last_heartbeat = None, 0.
        ready = self.control / "supervisor_ready.json"
        if not ready.exists():
            self.monitor.write_exclusive_json(ready, {"pid": os.getpid(), "utc": utc(), "binding": binding,
                "status": "waiting_for_new_run_initialization" if self.create_run else "waiting_for_explicit_old_monitor_and_core_handoff",
                "wandb_started": False})

    def save(self):
        atomic_json(self.state_path, self.state)

    def start_writer(self, sdk=None):
        if not self.create_run and not handoff_confirmed(self.args.handoff_file, self.args.wandb_id):
            return False
        if self.run is not None:
            return True
        if sdk is None:
            import wandb as sdk
        server_cursor, config = 0, None
        if self.create_run:
            if self.state.get("writer_started_at") or (self.run_dir / "wandb_ready.json").exists():
                raise FileExistsError("New W&B run was already initialized; require explicit handoff to resume it")
            # Persistent exclusive claim protects against a second control dir
            # or a lost init reply. Never guess whether the server created it.
            self.monitor.write_exclusive_json(self.run_dir / "wandb_create_claim.json", {
                "id": self.args.wandb_id, "entity": self.args.wandb_entity, "project": self.args.wandb_project,
                "control_dir": str(self.control.resolve()), "pid": os.getpid(), "utc": utc()})
            config = self.monitor.read_config(self.run_dir)
        else:
            remote = sdk.Api().run(f"{self.args.wandb_entity}/{self.args.wandb_project}/{self.args.wandb_id}")
            summary = dict(remote.summary)
            server_cursor = max(int(summary.get("iteration", 0)), int(summary.get("gmt/train_cursor", 0)))
            self.state["train_cursor"] = max(self.state["train_cursor"], server_cursor)
            logged = set(self.state["eval_logged"])
            logged.update(int(value) for value in summary.get("eval/logged_iterations", []))
            # Reconcile an interrupted log->ledger update from durable server history.
            for row in remote.scan_history(keys=["eval/iteration"]):
                value = row.get("eval/iteration")
                if type(value) in (int, float) and math.isfinite(value) and int(value) == value:
                    logged.add(int(value))
            self.state["eval_logged"] = sorted(logged)
        os.environ["WANDB_DISABLE_CODE"] = "true"
        settings = sdk.Settings(disable_git=True, console="off", x_disable_stats=True)
        options = dict(entity=self.args.wandb_entity, project=self.args.wandb_project,
                       id=self.args.wandb_id, resume="never" if self.create_run else "must", mode="online", save_code=False,
                       settings=settings, dir=str(self.control))
        if self.create_run:
            creation_config = dict(config or {"monitor_protocol": "gmt_numeric_csv_v1"})
            creation_config["eval/clip_count"] = self.expected_clips
            if self.eval_manifest_sha256 is not None:
                creation_config["eval/manifest_sha256"] = self.eval_manifest_sha256
            options.update(name=getattr(self.args, "run_name", None) or self.run_dir.name,
                           config=creation_config)
        try:
            self.run = sdk.init(**options)
        except Exception as error:
            if self.create_run:
                self.monitor.write_exclusive_json(self.run_dir / "wandb_error.json", {
                    "error_type": type(error).__name__, "phase": "initializing_new_run",
                    "message": "New W&B initialization failed; creation claim retained to prevent duplicate writers"})
            raise
        if self.run is None:
            raise RuntimeError("W&B resume returned no run")
        self.run.define_metric("iteration")
        for key in self.monitor.CSV_FIELDS:
            if key != "iteration":
                self.run.define_metric(key, step_metric="iteration")
        self.run.define_metric("eval/iteration")
        self.run.define_metric("eval/*", step_metric="eval/iteration")
        self.state["writer_started_at"] = utc()
        if self.create_run:
            self.state["writer_config_published"] = config is not None
            self.run.summary["stage"] = "initializing"
            self.run.summary["eval/clip_count"] = self.expected_clips
            if self.eval_manifest_sha256 is not None:
                self.run.summary["eval/manifest_sha256"] = self.eval_manifest_sha256
        self.save()
        atomic_json(self.control / "writer_ready.json", {"pid": os.getpid(), "utc": utc(),
                     "run_id": self.args.wandb_id, "resume": options["resume"], "server_train_cursor": server_cursor})
        if self.create_run:
            if self.run.id != self.args.wandb_id:
                raise ValueError("Initialized W&B run identity differs from the requested new run")
            self.monitor.write_exclusive_json(self.run_dir / "wandb_ready.json", {
                "url": self.run.url, "id": self.run.id, "project": self.run.project, "entity": self.run.entity})
        return True

    def discover(self):
        for path in sorted(self.run_dir.glob("ckpt_*.pt")):
            match = re.fullmatch(r"ckpt_(\d{6})\.pt", path.name)
            if not match:
                continue
            iteration = int(match.group(1))
            if iteration <= 0 or iteration > self.args.total_iterations or iteration % self.args.interval:
                continue
            key = str(iteration)
            if key not in self.state["checkpoints"]:
                self.state["checkpoints"][key] = {"checkpoint": str(path.resolve()), "sha256": sha256(path),
                                                        "attempts": [], "result": None, "exhausted": False}
                self.save()

    def drain_training(self):
        warmup = 500
        config = self.monitor.read_config(self.run_dir)
        if config is not None:
            warmup = config.get("wm_only_iters", 500)
            if self.create_run and not self.state.get("writer_config_published", False):
                self.run.config.update(config, allow_val_change=True)
                self.state["writer_config_published"] = True
                self.save()
        for record in self.reader.poll():
            iteration = record["iteration"]
            if iteration <= self.state["train_cursor"]:
                continue
            stage = "wm_warmup" if iteration <= warmup else "actor_critic"
            # Deliberately no global step: a delayed eval2000 may arrive after
            # train2500. Both domains use their independently defined x axes.
            self.run.log({**record, "stage": stage})
            self.run.summary["gmt/train_cursor"] = iteration
            self.run.summary["stage"] = stage
            self.state["train_cursor"] = iteration
            self.save()

    def accept_results_and_poll(self):
        active = []
        changed = False
        for key, entry in sorted(self.state["checkpoints"].items(), key=lambda item: int(item[0])):
            if not entry["attempts"]:
                continue
            attempt = entry["attempts"][-1]
            if attempt.get("terminal"):
                continue
            if not attempt.get("job_id"):
                found = self.scheduler.find_jobs(attempt["job_name"], attempt["submitted_at"])
                if len(found) > 1:
                    raise RuntimeError("Multiple jobs match a claimed evaluation attempt; refusing duplicate scheduling")
                if found:
                    attempt["job_id"] = found[0]
                    self.save()
                else:
                    # Cannot distinguish a lost sbatch response from an unissued
                    # request. Never guess and submit a second GPU allocation.
                    attempt["state"] = "SUBMISSION_UNCERTAIN"
                    active.append(attempt)
                    continue
            status = self.scheduler.status(attempt["job_id"])
            attempt["state"] = status["state"]
            output = Path(attempt["output"])
            if output.exists() and not entry.get("result") and status["state"] in TERMINAL:
                try:
                    if status["state"] != "COMPLETED" or status["exit_code"] != 0:
                        raise ValueError("Evaluation job did not complete with exit code zero")
                    result = validate_result(output, entry["checkpoint"], int(key), entry["sha256"],
                        expected_clips=self.expected_clips, expected_eval_manifest_sha256=self.eval_manifest_sha256)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    attempt["receipt_error"] = type(error).__name__ + ": " + str(error)
                else:
                    entry["result"], entry["receipt_path"] = result, str(output)
                    entry["receipt_sha256"] = sha256(output)
                    changed = True
                    self.save()
            if status["state"] not in TERMINAL:
                active.append(attempt)
                continue
            error_path = output.with_suffix(output.suffix + ".error.json")
            # Shared filesystems can expose the completed job before its result.
            if not entry.get("result") and not error_path.exists() and not attempt.get("receipt_error"):
                if "terminal_seen_at" not in attempt:
                    attempt["terminal_seen_at"] = self.clock()
                    self.save()
                if self.clock() - attempt["terminal_seen_at"] < 120:
                    active.append(attempt)
                    continue
            attempt["terminal"] = True
            if not entry.get("result"):
                attempt["failure"] = {"utc": utc(), "job_id": attempt["job_id"], "state": status["state"],
                                      "reason": attempt.get("receipt_error", "No successful validated evaluation receipt"),
                                      "worker_error_receipt": str(error_path) if error_path.exists() else None}
                failure = self.control / "attempts" / f"ckpt_{int(key):06d}_attempt{attempt['number']:02d}.failure.json"
                if not failure.exists():
                    self.monitor.write_exclusive_json(failure, attempt["failure"])
                entry["exhausted"] = len(entry["attempts"]) >= 3
            self.save()
        if len(active) > 1:
            raise RuntimeError("Ledger contains multiple active evaluation jobs")
        if changed or any(entry.get("result") for entry in self.state["checkpoints"].values()):
            rebuild_eval_csv(self.run_dir, self.state["checkpoints"])
        return active

    def publish_results(self):
        logged = set(self.state["eval_logged"])
        for key, entry in sorted(self.state["checkpoints"].items(), key=lambda item: int(item[0])):
            if not entry.get("result") or int(key) in logged:
                continue
            result = entry["result"]
            values = {"eval/iteration": int(key), **{f"eval/{name}": result[name] for name in
                       ("norm_return", "norm_return_framew", "n_failed", "n_total", "train_wall_h")}}
            values["eval/completion_fraction"] = 1 - result["n_failed"] / result["n_total"]
            self.run.log(values)
            logged.add(int(key))
            self.run.summary["eval/logged_iterations"] = sorted(logged)
            self.run.summary["eval/latest_iteration"] = int(key)
            self.run.summary["eval/latest_norm_return"] = result["norm_return"]
            self.state["eval_logged"] = sorted(logged)
            self.save()

    def submit_next(self):
        if self.eval_manifest is not None and sha256(self.eval_manifest) != self.eval_manifest_sha256:
            raise ValueError("Bound evaluation manifest changed; refusing to submit evaluation")
        for key, entry in sorted(self.state["checkpoints"].items(), key=lambda item: int(item[0])):
            if entry.get("result") or entry["exhausted"]:
                continue
            number = len(entry["attempts"]) + 1
            name = f"gmt-eval-{self.args.wandb_id}-{int(key):06d}-a{number}"
            output = self.control / "attempts" / f"ckpt_{int(key):06d}_attempt{number:02d}.json"
            if output.exists() or output.with_suffix(output.suffix + ".error.json").exists():
                raise FileExistsError("Evaluation attempt output already exists outside its ledger")
            attempt = {"number": number, "job_name": name, "output": str(output.resolve()),
                       "submitted_at": self.clock(), "job_id": None, "state": "SUBMITTING", "terminal": False}
            entry["attempts"].append(attempt)
            self.save()  # Durable claim BEFORE sbatch; adopt an orphan on restart.
            try:
                attempt["job_id"] = self.scheduler.submit(self.args.eval_script, name=name,
                    source_root=self.args.source_root, run_dir=self.run_dir, checkpoint=entry["checkpoint"], output=output)
                attempt["state"] = "SUBMITTED"
            except Exception as error:
                # sbatch can accept a job before its client loses the reply.
                # Keep the durable claim and reconcile by unique name next poll.
                attempt["state"] = "SUBMISSION_UNCERTAIN"
                attempt["submission_error_type"] = type(error).__name__
                self.monitor.write_exclusive_json(output.with_suffix(".submission_uncertain.json"),
                    {"job_name": name, "utc": utc(), "error_type": type(error).__name__,
                     "message": "Submission reply uncertain; querying Slurm by unique name; no automatic resubmission"})
            self.save()
            return True
        return False

    def training_outcome(self):
        code = self.monitor.read_exitcode(self.run_dir)
        if code is not None:
            return code
        status = self.scheduler.status(str(self.args.train_job_id))
        if status["state"] in TERMINAL:
            return status["exit_code"] if status["exit_code"] is not None else (0 if status["state"] == "COMPLETED" else 1)
        return None

    def tick(self):
        if self.run is None:
            raise RuntimeError("Writer handoff has not completed")
        self.drain_training()
        self.discover()
        active = self.accept_results_and_poll()
        self.publish_results()
        if not active and self.submit_next():
            active = [True]
        errors = sum(bool(entry["exhausted"]) for entry in self.state["checkpoints"].values())
        if self.clock() - self.last_heartbeat >= 60:
            uncertain = sum(attempt.get("state") == "SUBMISSION_UNCERTAIN" for entry in self.state["checkpoints"].values()
                            for attempt in entry["attempts"] if not attempt.get("terminal"))
            self.run.summary.update({"eval/supervisor_heartbeat_utc": utc(), "eval/supervisor_pid": os.getpid(),
                                     "eval/error_count": errors, "eval/active_job_count": len(active),
                                     "eval/submission_uncertain_count": uncertain})
            self.last_heartbeat = self.clock()
        outcome = self.training_outcome()
        if outcome is None or active:
            return None
        expected = set(range(self.args.interval, self.args.total_iterations + 1, self.args.interval))
        succeeded = {int(key) for key, entry in self.state["checkpoints"].items() if entry.get("result")}
        passed = outcome == 0 and expected.issubset(succeeded)
        report = {"passed": passed, "training_exitcode": outcome, "expected_iterations": sorted(expected),
                  "successful_iterations": sorted(succeeded), "missing_iterations": sorted(expected - succeeded),
                  "exhausted_evaluations": errors, "utc": utc(), "training_source_modified": False,
                  "training_started_or_resumed": False, "wandb_id": self.args.wandb_id}
        self.run.summary.update({"eval/status": "complete" if passed else "incomplete",
                                 "eval/all_expected_complete": passed, "eval/error_count": errors,
                                 "training_exitcode": outcome})
        # Evaluation failure must never mark an otherwise successful trainer failed.
        self.run.finish(exit_code=outcome)
        atomic_json(self.control / "supervisor_complete.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "control-dir", "source-root", "eval-script"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--handoff-file", type=Path)
    parser.add_argument("--create-run", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--expected-clips", type=int, default=40)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--train-job-id", required=True)
    for name in ("wandb-entity", "wandb-project", "wandb-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--total-iterations", type=int, default=30000)
    parser.add_argument("--interval", type=int, default=2000)
    parser.add_argument("--poll", type=float, default=30.)
    args = parser.parse_args()
    if args.interval <= 0 or args.total_iterations <= 0 or args.total_iterations % args.interval or not math.isfinite(args.poll) or args.poll <= 0:
        parser.error("Require positive polling/interval and a total divisible by the checkpoint interval")
    if not args.create_run and args.handoff_file is None:
        parser.error("--handoff-file is required unless --create-run is selected")
    if args.expected_clips <= 0 or (args.expected_clips != 40 and args.eval_manifest is None):
        parser.error("Require positive --expected-clips and an --eval-manifest for a custom clip count")
    with exclusive_lock(args.control_dir):
        supervisor = Supervisor(args)
        complete = args.control_dir / "supervisor_complete.json"
        if complete.exists():
            return 0 if json.loads(complete.read_text())["passed"] else 1
        while not supervisor.start_writer():
            time.sleep(min(args.poll, 60.))
        while True:
            result = supervisor.tick()
            if result is not None:
                print(json.dumps(result, indent=2), flush=True)
                return 0 if result["passed"] else 1
            time.sleep(min(args.poll, 60.))


if __name__ == "__main__":
    raise SystemExit(main())
