"""One owning training-run writer for training and delayed evaluation metrics.

SDK _step is a monotonic transport counter, never a checkpoint iteration.
Training uses the explicit ``iteration`` axis and evaluation ``eval/iteration``.
Durable event IDs/payload hashes and server history acknowledgement reconcile
restarts, including a handoff from the historical iteration-stepped sidecar.
Native training logs, checkpoints and frozen runtime sources are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import time

from .telemetry import (JSONLReader, METRIC_FIELDS, TERM_NAMES, _curated_metric,
                        _sdk, atomic_json, exclusive_writer)

PROTOCOL = "hoi_privileged_unified_training_eval_logging_v1"
EVENT_ID = "telemetry/event_id"
PAYLOAD_SHA = "telemetry/payload_sha256"
EVENT_KIND = "telemetry/kind"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_event_id(iteration):
    return f"train:{iteration:06d}"


def evaluation_event_id(iteration, checkpoint_hashes):
    return f"eval:{iteration:06d}:{payload_hash(checkpoint_hashes)[:24]}"


def aggregate_training(records, assets):
    """Equal native-worker contribution; counts sum, loss/reward means average."""
    rows = [_curated_metric(row) for row in records]
    if len(rows) != len(assets) or len({r["iteration"] for r in rows}) != 1 or len({r["stage"] for r in rows}) != 1:
        raise ValueError("Training asset workers disagree on iteration or stage")
    result = {"iteration": rows[0]["iteration"], "stage": rows[0]["stage"]}
    for asset, row in zip(assets, rows):
        result.update({f"assets/{asset}/{key}": value for key, value in row.items() if key not in ("iteration", "stage")})
    common = set.intersection(*(set(row) for row in rows))-{"iteration", "stage"}
    for key in common:
        values = [row[key] for row in rows]
        if key in ("real_steps", "resets", "real_steps_collected", "real_transitions") or key.endswith("_valid_samples"):
            result[key] = sum(values)
        elif key.endswith("_min"):
            result[key] = min(values)
        elif key.endswith("_max"):
            result[key] = max(values)
        else:
            result[key] = sum(values)/len(values)
    if {"real_reward_mean", "real_reward_std"} <= common:
        mean = result["real_reward_mean"]
        result["real_reward_std"] = math.sqrt(max(0., sum(r["real_reward_std"]**2+r["real_reward_mean"]**2 for r in rows)/len(rows)-mean**2))
    return result


def curate_mixed_training(record):
    from .experiments import TABLE_ASSETS, ULTRA_ASSETS

    payload = _curated_metric(record)
    assets = record.get("assets", {})
    if set(assets) != set(ULTRA_ASSETS + TABLE_ASSETS):
        raise ValueError("Mixed telemetry is missing native assets")
    for asset, row in assets.items():
        values = _curated_metric({**row, "iteration": payload["iteration"], "stage": payload["stage"]})
        payload.update({f"assets/{asset}/{key}": value for key, value in values.items()
                        if key not in ("iteration", "stage")})
    return payload


def owning_run(training_root, expected_run_id=None):
    root = Path(training_root)
    record = json.loads((root/"wandb_registration.json").read_text())
    if expected_run_id is not None and record.get("id") != expected_run_id:
        raise ValueError("Unified writer target differs from the owning training registration")
    for key in ("id", "entity", "project"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", record[key]):
            raise ValueError("Training W&B registration has an invalid identity")
    if record.get("metadata_sha256") != file_hash(root/"run_metadata.json"):
        raise ValueError("Training registration metadata changed before the logging handoff")
    return record


class EventOutbox:
    """Append-only complete event payloads; local 'sent' is not a server ack."""
    def __init__(self, root):
        self.root, self.events = Path(root), {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/"events.jsonl"
        if self.path.exists():
            data = self.path.read_bytes()
            boundary = data.rfind(b"\n")+1
            for line in data[:boundary].splitlines():
                if not line.strip(): continue
                record = json.loads(line)
                if payload_hash(record["payload"]) != record["payload_sha256"]:
                    raise ValueError("Durable telemetry event payload was modified")
                previous = self.events.get(record["event_id"])
                if previous and previous["payload_sha256"] != record["payload_sha256"]:
                    raise ValueError("Conflicting durable telemetry event identity")
                self.events[record["event_id"]] = record
            if boundary != len(data):
                # Recover only our own interrupted append, never native logs.
                with self.path.open("r+b") as stream:
                    stream.truncate(boundary); stream.flush(); os.fsync(stream.fileno())

    @staticmethod
    def _append(path, value):
        raw = (canonical(value)+"\n").encode()
        with Path(path).open("ab", buffering=0) as stream:
            while raw:
                written = stream.write(raw)
                if not written: raise OSError("Incomplete durable telemetry append")
                raw = raw[written:]
            os.fsync(stream.fileno())

    def enqueue(self, event_id, kind, payload):
        record = {"event_id": event_id, "kind": kind, "payload_sha256": payload_hash(payload), "payload": payload}
        previous = self.events.get(event_id)
        if previous:
            if previous["payload_sha256"] != record["payload_sha256"]:
                raise ValueError("A native event changed after durable enqueue")
            return previous
        self._append(self.path, record)
        self.events[event_id] = record
        return record

    def sent(self, event_id, internal_step):
        self._append(self.root/"deliveries.jsonl", {"event_id": event_id, "internal_step": internal_step})


def server_acknowledgements(module, identity, *, min_step=0):
    """Fresh server reads acknowledge semantic events, not internal-step axes."""
    run = module.Api(timeout=45).run(f"{identity['entity']}/{identity['project']}/{identity['id']}")
    events, last_step = {}, min_step-1
    for row in run.scan_history(min_step=min_step, page_size=1000):
        step = row.get("_step")
        if type(step) in (int, float) and math.isfinite(step): last_step = max(last_step, int(step))
        event_id = row.get(EVENT_ID)
        if event_id is None:
            # Legacy sidecars logged only training and used _step=iteration.
            # This compatibility path never interprets an evaluation as train.
            iteration = row.get("iteration")
            if type(iteration) in (int, float) and math.isfinite(iteration) and iteration >= 1 and int(iteration) == iteration:
                events[training_event_id(int(iteration))] = None
            continue
        digest = row.get(PAYLOAD_SHA)
        if not isinstance(event_id, str) or not re.fullmatch(r"(?:train:\d{6}|eval:\d{6}:[a-f0-9]{24})", event_id):
            raise ValueError("Server telemetry event identity is invalid")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Server telemetry payload acknowledgement is invalid")
        if event_id in events and events[event_id] not in (None, digest):
            raise ValueError("Server contains conflicting telemetry payloads")
        events[event_id] = digest
    return events, last_step


def define_axes(run):
    run.define_metric("iteration")
    run.define_metric("eval/iteration")
    for key in sorted(METRIC_FIELDS | {"reward_"+term for term in TERM_NAMES}):
        run.define_metric(key, step_metric="iteration")
    for pattern in ("assets/*", "wm_val_*", "rewards/*", "real_reward_terms/*"):
        run.define_metric(pattern, step_metric="iteration")
    run.define_metric("eval/*", step_metric="eval/iteration")
    run.define_metric("tables_by_action/*", step_metric="iteration")


def _exitcode(training_root, train_job, supervisor):
    try:
        text = (Path(training_root)/"training_exitcode").read_text().strip()
        if text: return int(text)
    except FileNotFoundError:
        pass
    if supervisor is not None and hasattr(supervisor, "_parent_exitcode"):
        return supervisor._parent_exitcode()
    if train_job is not None:
        from .eval_monitor import Slurm, TERMINAL
        status = Slurm().status(train_job)
        if status["state"] in TERMINAL:
            return (status["exitcode"] or 0) if status["state"] == "COMPLETED" else (status["exitcode"] or 1)
    return None


def monitor(root, training_root, *, assets, supervisor=None, supervisor_factory=None,
            supervisor_kwargs=None, eval_control=None, expected_run_id=None, train_job=None,
            wandb_module=None, poll_interval=5., ack_interval=60., sleep=time.sleep,
            clock=time.monotonic, stop_requested=lambda: False):
    """Merge both streams, keeping evaluation faults out of training telemetry."""
    root, training_root, run, phase = Path(root), Path(training_root), None, "claim"
    root.mkdir(parents=True, exist_ok=True)
    try:
        if not assets or len(set(assets)) != len(assets) or any(not re.fullmatch(r"[A-Za-z0-9_]+", a) for a in assets):
            raise ValueError("Unified logging needs the profile's exact ordered asset names")
        if min(poll_interval, ack_interval) <= 0 or not all(math.isfinite(v) for v in (poll_interval, ack_interval)):
            raise ValueError("Unified telemetry polling intervals must be positive")
        with ExitStack() as stack:
            # These are exactly the old producer locks; an independent lock in
            # a new directory would not fence the previous SDK writer.
            stack.enter_context(exclusive_writer(training_root))
            if eval_control is not None and Path(eval_control).resolve() != training_root.resolve():
                stack.enter_context(exclusive_writer(eval_control))
            identity = owning_run(training_root, expected_run_id)
            module = _sdk(wandb_module)
            phase = "resuming_owning_training_run"
            settings = module.Settings(disable_git=True, console="off", x_disable_stats=True, x_disable_meta=True)
            run = module.init(entity=identity["entity"], project=identity["project"], id=identity["id"], name=identity["name"],
                              mode="online", resume="must", save_code=False, settings=settings, dir=str(root),
                              config={"unified_telemetry_protocol": PROTOCOL, "training_metric_axis": "iteration", "evaluation_metric_axis": "eval/iteration"})
            if run is None or run.id != identity["id"]: raise RuntimeError("SDK resumed a different training run")
            define_axes(run)
            phase = "server_ack_bootstrap"
            acknowledged, last_server_step = server_acknowledgements(module, identity)
            outbox = EventOutbox(root)
            enqueued = set(acknowledged)
            for event_id, digest in acknowledged.items():
                if digest is not None and event_id in outbox.events and outbox.events[event_id]["payload_sha256"] != digest:
                    raise ValueError("Durable payload differs from the server acknowledgement")
            atomic_json(root/"server_ack.json", {"id": identity["id"], "last_internal_step": last_server_step, "events": acknowledged})
            # Always rebuild from native logs after restart. A local byte cursor
            # can be ahead of network persistence and is not an upload ack.
            mixed = assets == ("mixed",)
            readers = ([JSONLReader(training_root / "metrics.jsonl", curate=curate_mixed_training)] if mixed else
                       [JSONLReader(training_root/f"rank_{i:02d}/metrics.jsonl") for i in range(len(assets))])
            eval_failed, eval_state = False, None
            if supervisor is None and supervisor_factory is not None:
                try:
                    supervisor = supervisor_factory(**(supervisor_kwargs or {}))
                except Exception as error:
                    eval_failed = True
                    atomic_json(root/"eval_error.json", {"phase": "supervisor_initialization", "error_type": type(error).__name__})
                    run.summary["eval/status"] = "failed"
            run.summary["telemetry/status"] = "monitoring"
            atomic_json(root/"unified_ready.json", {"id": identity["id"], "entity": identity["entity"], "project": identity["project"],
                                                     "url": run.url, "protocol": PROTOCOL, "bootstrap_events": len(acknowledged),
                                                     "last_internal_step": last_server_step, "assets": list(assets)})
            last_ack = clock()
            last_train_iteration = max((int(key.split(":")[1]) for key in acknowledged if key.startswith("train:")), default=0)
            last_eval_iteration = max((int(key.split(":")[1]) for key in acknowledged if key.startswith("eval:")), default=0)

            def emit(event_id, kind, payload):
                if event_id in enqueued:
                    if event_id in acknowledged and acknowledged[event_id] is not None and acknowledged[event_id] != payload_hash(payload):
                        raise ValueError("A logged event's scalar payload changed")
                    return
                record = outbox.enqueue(event_id, kind, payload)
                # Deliberately omit `step=`. SDK's monotonically advancing
                # transport counter is independent of both scientific axes.
                run.log({**payload, EVENT_ID: event_id, PAYLOAD_SHA: record["payload_sha256"], EVENT_KIND: kind}, commit=True)
                enqueued.add(event_id)
                outbox.sent(event_id, int(run.step)-1)

            while True:
                phase = "training_metrics"
                for reader in readers: reader.poll()
                while all(r.pending for r in readers):
                    heads = [min(r.pending) for r in readers]
                    if len(set(heads)) != 1: raise ValueError("Training rank metric histories diverged")
                    iteration = heads[0]
                    if mixed:
                        payload = readers[0].pending[iteration]
                    else:
                        payload = aggregate_training([reader.pending[iteration] for reader in readers], assets)
                    emit(training_event_id(iteration), "train", payload)
                    for reader in readers: del reader.pending[iteration]
                    if iteration >= last_train_iteration:
                        last_train_iteration = iteration
                        run.summary["iteration"] = last_train_iteration
                        run.summary["stage"] = payload["stage"]
                        run.summary["status"] = "running"
                phase = "evaluation_scheduler"
                if supervisor is not None and not eval_failed:
                    try:
                        eval_state = supervisor.tick()
                        for iteration, entry in sorted((int(i), entry) for i, entry in eval_state["evaluations"].items()):
                            if entry["status"] != "complete": continue
                            request = supervisor._request(entry)
                            event_id = evaluation_event_id(iteration, request["checkpoint_sha256"])
                            if event_id in enqueued: continue
                            if file_hash(entry["result"]) != entry["result_sha256"]:
                                raise ValueError("Completed evaluation result changed before unified publication")
                            payload = supervisor.validate_result(entry["result"], request)
                            if payload.get("eval/iteration") != iteration or "iteration" in payload:
                                raise ValueError("Evaluation payload used the wrong scientific axis")
                            emit(event_id, "eval", payload)
                            last_eval_iteration = max(last_eval_iteration, iteration)
                            run.summary["eval/iteration"] = last_eval_iteration
                        run.summary["eval/status"] = "finished" if eval_state.get("finished") else "monitoring"
                        if eval_state.get("finished"):
                            run.summary["eval/missing_milestones"] = eval_state.get("missing_milestones", [])
                            run.summary["eval/failed_milestones"] = eval_state.get("failed_milestones", [])
                    except Exception as error:
                        # A bad evaluation never silences the training stream.
                        eval_failed = True
                        atomic_json(root/"eval_error.json", {"phase": "evaluation_scheduler_or_publication", "error_type": type(error).__name__})
                        run.summary["eval/status"] = "failed"
                phase = "server_acknowledgement"
                if clock()-last_ack >= ack_interval:
                    try:
                        new_ack, step = server_acknowledgements(module, identity, min_step=last_server_step+1)
                        for event_id, digest in new_ack.items():
                            if event_id in outbox.events and digest is not None and outbox.events[event_id]["payload_sha256"] != digest:
                                raise ValueError("Server acknowledged a different scalar payload")
                        acknowledged.update(new_ack); last_server_step = max(last_server_step, step)
                        atomic_json(root/"server_ack.json", {"id": identity["id"], "last_internal_step": last_server_step, "events": acknowledged})
                    except ValueError:
                        raise  # Conflicting server identity is not a transient RPC failure.
                    except Exception as error:
                        atomic_json(root/"server_ack_error.json", {"error_type": type(error).__name__})
                    last_ack = clock()
                try:
                    parent_exit = _exitcode(training_root, train_job, supervisor)
                except Exception as error:
                    parent_exit = None
                    atomic_json(root/"parent_status_error.json", {"error_type": type(error).__name__})
                if parent_exit is not None:
                    run.summary["status"] = "finished" if parent_exit == 0 else "failed"
                    run.summary["training_exitcode"] = parent_exit
                atomic_json(root/"unified_state.json", {"id": identity["id"], "last_train_iteration": last_train_iteration,
                            "last_eval_iteration": last_eval_iteration, "last_server_internal_step": last_server_step,
                            "eval_failed": eval_failed, "readers": [r.state_dict() for r in readers]})
                eval_finished = supervisor is None or eval_failed or (eval_state is not None and eval_state.get("finished"))
                stopping = bool(stop_requested())
                if stopping or (parent_exit is not None and eval_finished):
                    if parent_exit == 0 and any(r.pending or (r.path.exists() and r.path.stat().st_size != r.offset) for r in readers):
                        raise ValueError("Successful trainer left incomplete per-rank metric rows")
                    run.summary["telemetry/status"] = "handoff_complete" if stopping else "finished"
                    run.finish(exit_code=0 if stopping else parent_exit)
                    final_ack, step = server_acknowledgements(module, identity, min_step=last_server_step+1)
                    acknowledged.update(final_ack); last_server_step = max(last_server_step, step)
                    atomic_json(root/"server_ack.json", {"id": identity["id"], "last_internal_step": last_server_step, "events": acknowledged})
                    atomic_json(root/"unified_done.json", {"id": identity["id"], "handoff": stopping,
                                "training_exitcode": parent_exit, "eval_failed": eval_failed,
                                "last_train_iteration": last_train_iteration, "last_eval_iteration": last_eval_iteration})
                    return 0 if stopping else parent_exit
                sleep(poll_interval)
    except Exception as error:
        atomic_json(root/"unified_error.json", {"phase": phase, "error_type": type(error).__name__})
        if run is not None:
            try:
                run.summary["telemetry/status"] = "failed"
                run.finish(exit_code=1)
            except Exception: pass
        return 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--assets", required=True, help="Exact ordered comma-separated asset names for all native ranks")
    parser.add_argument("--run-id", help="Optional assertion against the owning training registration")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--eval-control", type=Path)
    parser.add_argument("--eval-source", type=Path)
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--train-job")
    parser.add_argument("--supervisor-module", default="hoi_tables.eval_monitor")
    parser.add_argument("--poll-interval", type=float, default=5.)
    args = parser.parse_args()
    stopped = {"value": False}
    def stop(signum, frame): stopped["value"] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    factory, kwargs = None, None
    if args.eval_control is not None:
        if not re.fullmatch(r"hoi_tables\.[a-zA-Z0-9_]+", args.supervisor_module): parser.error("Invalid supervisor module")
        if any(v is None for v in (args.campaign, args.eval_source, args.training_source)):
            parser.error("Evaluation scheduling requires --campaign, --eval-source and --training-source")
        factory = importlib.import_module(args.supervisor_module).Supervisor
        identity = owning_run(args.training_root, args.run_id)
        kwargs = {"root": args.eval_control, "campaign": args.campaign, "training_root": args.training_root,
                  "source_root": args.eval_source, "training_source": args.training_source,
                  "parent_run_id": identity["id"], "train_job": args.train_job}
    return monitor(args.root, args.training_root, assets=tuple(args.assets.split(",")),
                   supervisor_factory=factory, supervisor_kwargs=kwargs, eval_control=args.eval_control,
                   expected_run_id=args.run_id, train_job=args.train_job, poll_interval=args.poll_interval,
                   stop_requested=lambda: stopped["value"])


if __name__ == "__main__":
    raise SystemExit(main())
