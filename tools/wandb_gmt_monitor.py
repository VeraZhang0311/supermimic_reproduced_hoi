"""Standalone numeric-CSV W&B sidecar; never imports the training runtime.

No SDK import occurs until monitor() starts. The only application data sent
are the explicitly listed CSV metrics and a curated, path-free config summary.
The wrapper starts training after wandb_ready.json, then writes its integer
exit status to training_exitcode after the trainer has closed its CSV.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import time


# These are the 16 actual columns in the accepted train.py, in its own order.
CSV_FIELDS = (
    "iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss",
    "v_loss", "imag_rew", "pi_gradnorm", "imag_survival", "sim_rew",
    "ep_len", "ep_ret_norm", "term_frac", "it_per_s", "lr",
)
NUMERIC_ARGS = (
    "num_envs", "iterations", "buffer_steps", "utd", "horizon", "goal_frames",
    "hist", "wm_window", "wm_batches", "wm_batch", "pi_batch", "lr", "gamma",
    "lam", "action_noise", "critic_epochs", "critic_minibatches", "warmup_steps",
    "wm_only_iters", "lr_decay_iters", "ckpt_every", "log_every", "seed", "transfer_parent_iteration",
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def curated_config(args, contract):
    """Allowlist individual scalar values; never copy arbitrary nested objects."""
    result = {"monitor_protocol": "gmt_numeric_csv_v1"}
    for key in NUMERIC_ARGS:
        if _number(args.get(key)):
            result[key] = args[key]
    if type(args.get("no_critic")) is bool:
        result["no_critic"] = args["no_critic"]
    for key, allowed in (("dataset", ("lafan1", "boneseed")), ("amp", ("none", "bf16"))):
        if args.get(key) in allowed:
            result[key] = args[key]
    if contract.get("version") == "robot_privileged_gmt_hw3_actuator_v3":
        result["observation_schema"] = contract["version"]
    for key in ("dim", "bodies"):
        if _number(contract.get(key)):
            result["privileged_"+key] = contract[key]
    if type(contract.get("self_collision")) is bool:
        result["self_collision"] = contract["self_collision"]
    heads = contract.get("analytic_heads", {})
    physics = contract.get("physics", {})
    actuator = contract.get("actuator_state", {})
    for source, key, output in ((heads, "control_dt", "control_dt"),
                                (heads, "observation_dim", "proprio_dim"),
                                (physics, "physics_dt", "physics_dt"),
                                (physics, "control_decimation", "control_decimation"),
                                (actuator, "dim", "actuator_state_dim")):
        if isinstance(source, dict) and _number(source.get(key)):
            result[output] = source[key]
    if isinstance(physics, dict) and physics.get("profile") == "fengqiao_hw3_mujoco_v1":
        result["physics_profile"] = physics["profile"]
    for key in ("urdf_sha256", "motion_manifest_sha256"):
        value = contract.get(key)
        if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
            result[key] = value
    return result


def read_config(run_dir):
    """Trainer writes JSON after sidecar handshake; tolerate a write in progress."""
    try:
        args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
        contract = json.loads((run_dir / "gmt_contract.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(args, dict) or not isinstance(contract, dict):
        raise ValueError("Training config must contain JSON objects")
    return curated_config(args, contract)


class CSVReader:
    """Incrementally consume complete newline-terminated, monotonic CSV rows."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset, self.pending, self.header = 0, b"", False
        self.last_iteration = -1

    def poll(self):
        try:
            with self.path.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size < self.offset:
                    raise ValueError("Training CSV was truncated")
                stream.seek(self.offset)
                data = stream.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        rows = []
        for line in lines:
            if not line.strip():
                continue
            columns = next(csv.reader(io.StringIO(line.decode("utf-8"))))
            if not self.header:
                if tuple(columns) != CSV_FIELDS:
                    raise ValueError("Training CSV does not match the metric allowlist")
                self.header = True
                continue
            if len(columns) != len(CSV_FIELDS):
                raise ValueError("Malformed complete training CSV row")
            values = [float(value) for value in columns]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite training metric")
            iteration = values[0]
            if not iteration.is_integer() or iteration < 0:
                raise ValueError("Training iteration must be a nonnegative integer")
            iteration = int(iteration)
            if iteration <= self.last_iteration:
                continue
            self.last_iteration = iteration
            record = dict(zip(CSV_FIELDS, values))
            record["iteration"] = iteration
            rows.append(record)
        return rows


def write_exclusive_json(path, value):
    """Publish a complete handshake atomically, without replacing an old run."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            # Windows rename is atomic and refuses an existing destination;
            # Windows sandbox profiles may prohibit creating hard links.
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_exitcode(run_dir):
    try:
        value = (run_dir / "training_exitcode").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    if not value:  # The wrapper may just have opened the sentinel.
        return None
    if not re.fullmatch(r"-?\d+", value):
        raise ValueError("Training exit sentinel must contain an integer")
    return int(value)


def monitor(run_dir, *, entity, project, run_id, run_name, poll_interval=5.,
            wandb_module=None, sleep=time.sleep):
    """Block until training_exitcode, or publish a sanitized local error."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run, phase = None, "initializing"
    reader = CSVReader(run_dir / "train_log.csv")
    try:
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("Polling interval must be positive")
        if (run_dir / "wandb_ready.json").exists() or (run_dir / "wandb_error.json").exists():
            raise FileExistsError("Monitor handshake already exists")
        os.environ["WANDB_DISABLE_CODE"] = "true"
        if wandb_module is None:
            import wandb as wandb_module
        settings = wandb_module.Settings(disable_git=True, console="off", x_disable_stats=True)
        config = read_config(run_dir)
        run = wandb_module.init(entity=entity, project=project, id=run_id, name=run_name,
                                mode="online", resume="never", save_code=False, settings=settings,
                                dir=str(run_dir), config=config or {"monitor_protocol": "gmt_numeric_csv_v1"})
        if run is None:
            raise RuntimeError("W&B online initialization did not return a run")
        run.summary["stage"] = "initializing"
        write_exclusive_json(run_dir / "wandb_ready.json", {
            "url": run.url, "id": run.id, "project": run.project, "entity": run.entity,
        })
        while True:
            phase = "configuration"
            if config is None:
                config = read_config(run_dir)
                if config is not None:
                    run.config.update(config, allow_val_change=True)
            warmup = config.get("wm_only_iters", 500) if config is not None else 500
            phase = "logging"
            def drain():
                for record in reader.poll():
                    stage = "wm_warmup" if record["iteration"] <= warmup else "actor_critic"
                    run.log({**record, "stage": stage}, step=record["iteration"])
                    run.summary["stage"] = stage
            drain()
            phase = "exit_sentinel"
            exitcode = read_exitcode(run_dir)
            if exitcode is not None:
                # Sentinel is written only after the trainer exits/closes CSV.
                phase = "final_drain"
                drain()
                run.summary["training_exitcode"] = exitcode
                phase = "finishing"
                run.finish(exit_code=exitcode)
                return exitcode
            sleep(poll_interval)
    except Exception as error:
        # Exception text/tracebacks can contain credentials, environment values,
        # or local paths. The local handshake deliberately records none of them.
        try:
            write_exclusive_json(run_dir / "wandb_error.json", {
                "error_type": type(error).__name__, "phase": phase,
                "last_iteration": reader.last_iteration,
                "message": "W&B monitor failed; no exception text or environment was recorded",
            })
        except OSError:
            pass
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-interval", type=float, default=5.)
    args = parser.parse_args()
    code = monitor(args.run_dir, entity=args.entity, project=args.project,
                   run_id=args.run_id, run_name=args.run_name, poll_interval=args.poll_interval)
    return code if 0 <= code <= 255 else 1


if __name__ == "__main__":
    raise SystemExit(main())
