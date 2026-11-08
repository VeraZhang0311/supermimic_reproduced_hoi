"""Standalone scalar-only W&B telemetry for the two-asset tables teacher.

This module imports neither torch nor the simulator. Use the existing telemetry
SDK environment. First register a fresh private run with --register-only, then
resume that exact run inside the training allocation. The wrapper publishes an
integer ``training_exitcode`` only after both trainers have closed their logs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

PROJECT = "hoiretarget-tables"
ENTITY = "tianzizh-university-of-pennsylvania"
PROTOCOL = "hoi_tables_scalar_sidecar_v1"
ASSETS = ("largetable", "smalltable")
STAGES = {"wm_only", "actor_critic"}
GLOBAL_FIELDS = {"num_envs", "iterations", "buffer_steps", "utd", "horizon", "hist", "wm_window",
                 "wm_batches", "wm_batch", "pi_batch", "lr", "gamma", "lam", "action_noise",
                 "critic_epochs", "critic_minibatches", "warmup_steps", "wm_only_iters", "lr_decay_iters",
                 "ckpt_every", "seed", "amp", "experiment_kind", "wm_hidden", "actor_hidden", "critic_hidden", "contact_hidden"}
SCHEMA_FIELDS = {"version", "wm_input_dim", "wm_output_dim", "wm", "wm_output", "actor_critic_oracle_dim",
                 "shape_dim", "actor_critic_blocks", "base_action_input", "residual_action", "contact",
                 "contact_schema", "actuator_prediction", "v11_source"}
HASH_FIELDS = {"source_sha256", "manifest_sha256", "asset_catalog_sha256", "shape_bank_sha256", "reference_contacts_sha256"}
COUNT_FIELDS = {"train_clips", "protected_eval", "evaluation_clips", "train_source_groups", "asset_count", "gpus", "total_env"}
TERM_NAMES = {"body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity",
              "root_position", "root_orientation", "action_rate", "joint_limits", "object_position",
              "object_orientation", "object_linear_velocity", "object_angular_velocity", "hand_approach", "contact_match"}
METRIC_FIELDS = {"lr", "real_steps", "resets", "wm_total", "wm_robot", "wm_object", "wm_contact", "wm_wm_gradnorm",
                 "wm_gradnorm", "pi_loss", "v_loss", "pi_gradnorm", "imag_return_per_step", "imag_survival",
                 "real_reward_mean", "real_reward_std", "real_reward_min", "real_reward_max",
                 "real_reward_normalized", "real_episode_length_mean", "real_term_frac", "real_trunc_frac",
                 "real_quality_fraction", "real_object_position_error_m", "real_object_orientation_error_rad",
                 "real_body_position_error_m", "real_steps_collected", "real_transitions", "real_seconds",
                 "wall_seconds", "wall_hours", "iterations_per_second", "sim_rew", "term_frac", "reset_frac", "ep_len", "ep_ret_norm", "real_failure_fraction", "real_truncation_fraction",
                 "real_termination_fraction", "real_episode_length", "real_episode_return_normalized"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe_schema(value):
    if value is None or type(value) in (bool, int) or _finite(value):
        return value
    if (isinstance(value, str) and len(value) <= 160 and not value.startswith("/")
            and not re.match(r"^[A-Za-z]:", value) and not any(c in value for c in ("\\", "://"))):
        return value
    if isinstance(value, (tuple, list)):
        return [_safe_schema(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_schema(v) for k, v in value.items()
                if isinstance(k, str) and re.fullmatch(r"[a-zA-Z0-9_]+", k)}
    raise ValueError("Unsafe non-scalar configuration value")


def curated_config(metadata):
    """Only explicitly authorized training configuration, schema and hashes."""
    result = {"monitor_protocol": PROTOCOL, "upload_scope": "scalar_metrics_and_curated_config_only"}
    config = metadata.get("global_config", {})
    result["global_config"] = {k: _safe_schema(config[k]) for k in GLOBAL_FIELDS if k in config}
    schema = metadata.get("schema", {})
    result["schema"] = {k: _safe_schema(schema[k]) for k in SCHEMA_FIELDS if k in schema}
    for key in HASH_FIELDS:
        if key in metadata:
            value = metadata[key]
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError("Invalid artifact identity hash")
            result[key] = value
    for key in COUNT_FIELDS:
        if key in metadata and type(metadata[key]) is int and metadata[key] >= 0:
            result[key] = metadata[key]
    if metadata.get("branch") in ("hoi-retarget/tables-v1.0", "hoi-retarget/tables-boxes-v1.0"):
        result["branch"] = metadata["branch"]
    experiment = metadata.get("experiment")
    if isinstance(experiment, dict) and experiment.get("profile") in ("tables", "ultra"):
        result["experiment"] = {"profile": experiment["profile"], "assets": _safe_schema(experiment["assets"]),
                                "evaluation_kind": experiment["evaluation"]["kind"],
                                "evaluation_every": experiment["evaluation"]["every"],
                                "evaluation_on_training_run": True,
                                "evaluation_role": "canonical_development80" if experiment["profile"] == "ultra" else "protected_tables22"}
        if experiment.get("actor_backward_schedule") in ("parallel", "serialized_per_gpu"):
            result["experiment"]["actor_backward_schedule"] = experiment["actor_backward_schedule"]
    if metadata.get("from_scratch") is True:
        result["from_scratch"] = True
    if metadata.get("gpu_type") == "L40":
        result["gpu_type"] = "L40"
    return result


def _identity(root, *, entity, project, run_id, run_name=None):
    if project != PROJECT or entity != ENTITY:
        raise ValueError("Tables telemetry must use the authorized project and entity")
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", run_id):
        raise ValueError("Invalid independent tables run ID")
    path = Path(root)/"run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    profile = metadata.get("experiment", {}).get("profile", "tables")
    prefix = "ultra_boxes_v1-" if profile == "ultra" else "tables_v1-"
    name = run_name or prefix+datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(re.escape(prefix)+r"[A-Za-z0-9_-]+", name):
        raise ValueError("Training run names must begin with "+prefix)
    return {"id": run_id, "entity": entity, "project": project, "name": name,
            "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, curated_config(metadata)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_writer(root):
    """Kernel-released lock avoids stale PID claims after process interruption."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/"wandb_monitor.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _sdk(module):
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    if module is None:
        import wandb as module
    return module


def _ensure_private_project(module, entity, project):
    # The already installed 0.23.1 SDK exposes access on CREATE_PROJECT even
    # though create_project() does not accept it. Reuse its authenticated API
    # transport and shipped GraphQL document, without handling API-key values.
    api = module.Api(timeout=45)
    response = api.client.execute(api.CREATE_PROJECT, {"entityName": entity, "name": project, "access": "PRIVATE"})
    payload = response["upsertModel"]
    actual = payload.get("project") or payload.get("model")
    if not isinstance(actual, dict) or actual.get("access") != "PRIVATE" or actual.get("name") != project:
        raise RuntimeError("Private W&B project access was not verified")
    return True


def _init(module, root, identity, config, *, resume):
    settings = module.Settings(disable_git=True, console="off", x_disable_stats=True, x_disable_meta=True)
    run = module.init(entity=identity["entity"], project=identity["project"], id=identity["id"], name=identity["name"],
                      mode="online", resume=resume, save_code=False, settings=settings,
                      dir=str(root), config=config)
    if run is None or run.id != identity["id"]:
        raise RuntimeError("W&B returned the wrong tables run identity")
    return run


def _validate_registration(registration, identity):
    for key in ("id", "entity", "project", "metadata_sha256"):
        if registration.get(key) != identity[key]:
            raise ValueError("Existing tables run registration does not match this campaign")
    if registration.get("private_project") is not True:
        raise ValueError("Tables project privacy has not been verified")


def register(root, *, entity=ENTITY, project=PROJECT, run_id, run_name=None, wandb_module=None):
    """Create a fresh run without logging a fake training iteration zero."""
    root, run, phase = Path(root), None, "registration"
    try:
        with exclusive_writer(root):
            identity, config = _identity(root, entity=entity, project=project, run_id=run_id, run_name=run_name)
            path = root/"wandb_registration.json"
            if path.exists():
                previous = json.loads(path.read_text())
                _validate_registration(previous, identity)
                return previous
            module = _sdk(wandb_module)
            phase = "private_project"
            _ensure_private_project(module, entity, project)
            phase = "initializing"
            run = _init(module, root, identity, config, resume="never")
            run.summary["status"] = "training_not_started"
            run.summary["stage"] = "registered"
            phase = "register_finish"
            run.finish(exit_code=0)
            registration = {**identity, "url": run.url, "private_project": True,
                            "protocol": PROTOCOL, "status": "training_not_started"}
            atomic_json(path, registration)
            return registration
    except Exception as error:
        atomic_json(root/"wandb_error.json", {"phase": phase, "error_type": type(error).__name__})
        if run is not None:
            try: run.finish(exit_code=1)
            except Exception: pass
        raise RuntimeError("Tables W&B registration failed; inspect sanitized wandb_error.json") from None


def _curated_metric(record):
    iteration = record.get("iteration")
    if type(iteration) is not int or iteration < 1 or record.get("stage") not in STAGES:
        raise ValueError("Invalid tables training metric identity/stage")
    result = {"iteration": iteration, "stage": record["stage"]}
    for key, value in record.items():
        diagnostic = bool(re.fullmatch(r"wm_val_h(?:1|8|32)_(?:robot_mse|object_mse|object_position_rmse_m|valid_samples)", key))
        diagnostic |= bool(re.fullmatch(r"wm_val_(?:contact_recall|contact_precision|contact_brier)_(?:left_hand_object|right_hand_object|object_ground|left_foot_object|right_foot_object)", key))
        allowed = (key in METRIC_FIELDS or diagnostic
                   or any(key == prefix+term for prefix in ("rewards/", "real_reward_terms/", "reward_") for term in TERM_NAMES))
        if allowed:
            if not _finite(value):
                raise ValueError("Authorized training metrics must be finite numeric scalars")
            result[key] = value
    return result


class JSONLReader:
    def __init__(self, path, state=None):
        self.path = Path(path)
        state = state or {}
        self.offset, self.last_read = int(state.get("offset", 0)), int(state.get("last_read", 0))
        self.pending = {int(k): v for k, v in state.get("pending", {}).items()}

    def poll(self):
        try:
            with self.path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size < self.offset:
                    raise ValueError("Training metrics file was truncated")
                stream.seek(self.offset)
                data = stream.read()
        except FileNotFoundError:
            return 0
        boundary = data.rfind(b"\n")+1
        if not boundary:
            return 0
        complete = data[:boundary]
        rows = []
        for line in complete.splitlines():
            if not line.strip(): continue
            row = _curated_metric(json.loads(line))
            if row["iteration"] <= self.last_read:
                continue
            rows.append(row)
            self.last_read = row["iteration"]
        self.offset += boundary
        self.pending.update({r["iteration"]: r for r in rows})
        return len(rows)

    def state_dict(self):
        return {"offset": self.offset, "last_read": self.last_read, "pending": self.pending}


def aggregate(records):
    records = [_curated_metric(record) for record in records]
    if len(records) != 2 or len({r["iteration"] for r in records}) != 1 or len({r["stage"] for r in records}) != 1:
        raise ValueError("Both asset workers must agree on iteration and training stage")
    result = {"iteration": records[0]["iteration"], "stage": records[0]["stage"]}
    for asset, record in zip(ASSETS, records):
        result.update({f"assets/{asset}/{key}": value for key, value in record.items() if key not in ("iteration", "stage")})
    common = set.intersection(*(set(row) for row in records))-{"iteration", "stage"}
    for key in common:
        values = [r[key] for r in records]
        if key in ("real_steps", "resets", "real_steps_collected", "real_transitions") or key.endswith("_valid_samples"):
            result[key] = sum(values)
        elif key.endswith("_min"):
            result[key] = min(values)
        elif key.endswith("_max"):
            result[key] = max(values)
        else:
            result[key] = sum(values)/2
    if {"real_reward_mean", "real_reward_std"} <= common:
        mean = result["real_reward_mean"]
        result["real_reward_std"] = math.sqrt(max(0., sum(r["real_reward_std"]**2+r["real_reward_mean"]**2 for r in records)/2-mean**2))
    return result


def read_exitcode(root):
    try: text = (Path(root)/"training_exitcode").read_text().strip()
    except FileNotFoundError: return None
    if not text: return None
    if not re.fullmatch(r"-?\d+", text): raise ValueError("Malformed training completion sentinel")
    return int(text)


def monitor(root, *, entity=ENTITY, project=PROJECT, run_id, run_name=None, poll_interval=5.,
            idle_timeout=3600., wandb_module=None, sleep=time.sleep, clock=time.monotonic):
    """Resume only the registered campaign and exit with its training status."""
    root, run, phase = Path(root), None, "claim"
    try:
        if not _finite(poll_interval) or not _finite(idle_timeout) or min(poll_interval, idle_timeout) <= 0:
            raise ValueError("Monitor polling and idle timeout must be positive")
        with exclusive_writer(root):
            identity, config = _identity(root, entity=entity, project=project, run_id=run_id, run_name=run_name)
            registration = json.loads((root/"wandb_registration.json").read_text())
            _validate_registration(registration, identity)
            identity["name"] = registration["name"]
            state_path = root/"wandb_state.json"
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            if previous and previous.get("id") != run_id:
                raise ValueError("Telemetry cursor belongs to another run")
            module = _sdk(wandb_module)
            phase = "resuming"
            run = _init(module, root, identity, config, resume="must")
            # SDK resume reads the server's committed next step. If local state
            # was saved before network persistence, replay from the server's
            # acknowledgement rather than skipping metrics based on local IO.
            last = max(0, int(getattr(run, "step", 0))-1)
            saved = previous.get("readers", [{}, {}]) if previous.get("last_iteration") == last else [{}, {}]
            readers = [JSONLReader(root/f"rank_{i:02d}/metrics.jsonl", saved[i]) for i in range(2)]
            run.summary["status"] = "initializing"
            atomic_json(root/"wandb_ready.json", {**registration, "status": "monitoring", "last_iteration": last})
            last_activity = clock()
            while True:
                phase = "reading_metrics"
                for reader in readers:
                    if reader.poll(): last_activity = clock()
                    reader.pending = {k: v for k, v in reader.pending.items() if k > last}
                while all(r.pending for r in readers):
                    heads = [min(r.pending) for r in readers]
                    if len(set(heads)) != 1:
                        raise ValueError("Asset worker iteration histories diverged")
                    iteration = heads[0]
                    record = aggregate([r.pending[iteration] for r in readers])
                    phase = "logging_metrics"
                    run.log(record, step=iteration, commit=True)
                    last = iteration
                    for reader in readers: del reader.pending[iteration]
                    run.summary["iteration"] = last
                    run.summary["stage"] = record["stage"]
                    run.summary["status"] = "running"
                    atomic_json(state_path, {"protocol": PROTOCOL, "id": run_id, "last_iteration": last,
                                            "readers": [r.state_dict() for r in readers]})
                phase = "completion"
                exitcode = read_exitcode(root)
                if exitcode is not None:
                    # Trainer streams are closed before publishing this file.
                    # Complete rows already drained above; partial/unpaired
                    # final data can occur on a failed distributed allocation.
                    incomplete = any(r.pending or (r.path.exists() and r.path.stat().st_size != r.offset) for r in readers)
                    if exitcode == 0 and incomplete:
                        raise ValueError("Successful training has unmatched per-asset metrics")
                    run.summary["training_exitcode"] = exitcode
                    run.summary["status"] = "finished" if exitcode == 0 else "failed"
                    run.finish(exit_code=exitcode)
                    atomic_json(root/"wandb_done.json", {"id": run_id, "last_iteration": last,
                                                         "training_exitcode": exitcode, "unpaired_rows": sum(len(r.pending) for r in readers)})
                    return exitcode
                if clock()-last_activity > idle_timeout:
                    raise TimeoutError("No training metrics or completion within the configured idle budget")
                sleep(poll_interval)
    except Exception as error:
        atomic_json(root/"wandb_error.json", {"phase": phase, "error_type": type(error).__name__})
        if run is not None:
            try: run.finish(exit_code=1)
            except Exception: pass
        return 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--id", dest="run_id", required=True)
    parser.add_argument("--name", dest="run_name")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=5.)
    parser.add_argument("--idle-timeout", type=float, default=3600.)
    args = parser.parse_args()
    values = vars(args)
    registration = values.pop("register_only")
    if registration:
        values.pop("poll_interval"); values.pop("idle_timeout")
        try:
            print(json.dumps(register(**values), allow_nan=False))
            return 0
        except RuntimeError:
            print(json.dumps({"error": "registration_failed", "details": "wandb_error.json"}))
            return 1
    return monitor(**values)


if __name__ == "__main__":
    raise SystemExit(main())
