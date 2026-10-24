"""Whitelist-only batch/live tracking; never resume a W&B run.

Input documents are local. Only declared configuration scalars, identity
digests and finite summary metrics enter the outgoing payload. Source paths,
case names, free text, artifacts, weights and raw trajectories do not.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
import uuid

from .config import TrainConfig, fresh_directory, write_json
from .variants import VARIANTS

ENTITY = "tianzizh-university-of-pennsylvania"
PROJECT = "SuperMimic_Upstream_Reproduction"
CHANNELS = ("left_hand_object", "right_hand_object", "object_ground")
CASE_SCALARS = (
    "control_steps", "state_frames", "full_reference_completed", "first_supermimic_failure_step", "first_hoi_failure_step",
    "supermimic_norm_return", "human_normalized_return", "hoi_normalized_return", "hoi_reference_mask_normalized_return",
    "hoi_reference_mask_reward_upper_bound", "human_full_duration_reward", "hoi_full_duration_reward", "hoi_extra_full_duration_reward",
    "objective_normalized_return", "objective_max_step_reward",
    "phc_pass", "object_pose_pass", "body_and_object_pose_pass", "pose_human_expected_contact_joint_pass",
    "first_phc_failure_reference_s", "first_object_failure_reference_s", "minimum_pelvis_height_m", "pelvis_below_0_3m",
    "joint_rmse_rad", "scored_frames", "scored_duration_s", "action_abs_ge_one_fraction", "action_abs_ge_twenty_fraction",
)
ERROR_GROUPS = ("body_position_m", "hand_position_m", "object_position_m", "object_orientation_deg",
                "object_linear_velocity_mps", "object_angular_velocity_radps")


def destination():
    """Explicit environment override, with the historical batch default kept."""
    values = (os.environ.get("WANDB_ENTITY", ENTITY), os.environ.get("WANDB_PROJECT", PROJECT))
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) for value in values):
        raise ValueError("W&B entity/project must be explicit names without URL, path or export-list syntax")
    return values


def _validate_destination(payload):
    if (payload.get("entity"), payload.get("project")) != destination() or payload.get("resume") != "never":
        raise ValueError("tracking destination differs from the explicit environment/default or resume policy")
    if not re.fullmatch(r"[0-9a-f]{32}", payload.get("run_id", "")):
        raise ValueError("tracking run ID must be a newly generated UUID")


def _number(value):
    try:
        return type(value) in (int, float, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return value if isinstance(value, list) else []


def _put(target, key, value):
    if _number(value):
        target[key] = value


def _digest(value):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _hash(target, key, value):
    value = _digest(value)
    if value is not None:
        target[key] = value


def _nested(value, *keys):
    for key in keys:
        value = value.get(key, {}) if isinstance(value, dict) else {}
    return value


def _configuration(plan, evaluation):
    out = {"method": "HOI_v1_full_reference", "tracking_schema": "hoi_v1_curated_tracking_1"}
    raw = _mapping(plan.get("config"))
    for key, default in TrainConfig().to_dict().items():
        value = raw.get(key)
        if type(default) is bool:
            if type(value) is bool:
                out["training/" + key] = value
        elif type(default) is int:
            if type(value) is int and value >= 0 and _number(value):
                out["training/" + key] = value
        elif type(default) is float:
            if type(value) in (int, float) and _number(value):
                out["training/" + key] = value
        elif key == "amp" and value in ("none", "bf16"):
            out["training/amp"] = value
        elif key == "experiment_kind" and value in ("formal", "smoke"):
            out["training/experiment_kind"] = value
        elif key == "variant" and value in VARIANTS:
            out["training/variant"] = value
    quality_gate = _nested(plan, "policy_update_schedule", "wm_quality_gate_enabled")
    if type(quality_gate) is bool:
        out["training/wm_quality_gate_enabled"] = quality_gate
    physics = _mapping(plan.get("physics", evaluation.get("physics", {})))
    for key in ("hand_collision", "self_collision"):
        if type(physics.get(key)) is bool:
            out["physics/" + key] = physics[key]
    identity = _mapping(plan.get("data_identity", evaluation.get("training_data_identity", {})))
    hashes = {"dataset/bundle_content_sha256": identity.get("bundle_content_sha256"),
              "dataset/manifest_content_sha256": _nested(identity, "manifest", "content_sha256"),
              "dataset/recipes_sha256": _nested(identity, "recipes", "sha256"),
              "dataset/hands_geometry_sha256": identity.get("hands_geometry_sha256"),
              "robot/mjcf_sha256": _nested(identity, "robot_fk", "mjcf_sha256"),
              "robot/reference_urdf_sha256": _nested(identity, "reference_kinematics", "urdf", "sha256"),
              "robot/native_asset_sha256": _nested(identity, "native_robot_asset", "sha256"),
              "source/content_sha256": _nested(plan, "source_identity", "content_sha256"),
              "evaluation/bundle_content_sha256": evaluation.get("evaluation_bundle_content_sha256"),
              "evaluation/checkpoint_sha256": evaluation.get("checkpoint_sha256")}
    for key, value in hashes.items():
        _hash(out, key, value)
    partition = evaluation.get("partition")
    if partition in ("train", "eval80", "test40", "validation", "regression", "test"):
        out["evaluation/partition"] = partition
    if evaluation.get("variant") in VARIANTS:
        out["evaluation/variant"] = evaluation["variant"]
    for key in ("selected_motions", "available_motions", "is_subset"):
        _put(out, "dataset/" + key, _nested(plan, "scope", key))
    return out


def _mean(values):
    return sum(values) / len(values) if values else None


def _training_event(record):
    if type(record.get("iteration")) is not int or record["iteration"] < 0:
        raise ValueError("training metric rows require a nonnegative integer iteration")
    metrics = {}
    for key in ("iteration", "sim_steps", "lr", "elapsed_seconds", "wm_updates", "policy_updates"):
        _put(metrics, "train/" + key, record.get(key))
    for key in ("total", "robot", "object", "contact"):
        values = [row[key] for row in _rows(record.get("wm")) if isinstance(row, dict) and _number(row.get(key))]
        _put(metrics, "wm/" + key, _mean(values))
    actor = _mapping(record.get("actor"))
    for key in ("pi_loss", "v_loss", "pi_gradnorm", "imag_human_and_hoi_return_per_step", "imag_survival"):
        _put(metrics, "actor/" + key, actor.get(key))
    for key in ("human_reward_normalized", "hoi_reward_normalized", "objective_reward_normalized", "terminations"):
        values = [row[key] for row in _rows(record.get("real")) if isinstance(row, dict) and _number(row.get(key))]
        _put(metrics, "real/" + key, _mean(values))
    for source, prefix in (("fit_contact_positive", "contact/fit_positive"), ("fit_contact_negative", "contact/fit_negative")):
        values = record.get(source)
        if isinstance(values, list) and len(values) == 3:
            for index, channel in enumerate(CHANNELS):
                _put(metrics, prefix + "/" + channel, values[index])
    if type(_nested(record, "policy_gate", "passed")) is bool:
        metrics["gate/policy_passed"] = record["policy_gate"]["passed"]
    if type(_nested(record, "wm_quality", "passed")) is bool:
        metrics["diagnostics/wm_quality_passed"] = record["wm_quality"]["passed"]
    return {"step": record["iteration"], "metrics": metrics}


def _evaluation_summary(report):
    metrics = {}
    for key in ("count", "phc_pass", "object_pose_pass", "body_and_object_pose_pass", "supermimic_n_failed",
                "supermimic_norm_return", "human_normalized_return", "hoi_normalized_return", "hoi_reference_mask_normalized_return", "objective_normalized_return"):
        _put(metrics, "evaluation/" + key, _nested(report, "aggregate", key))
    for key in ("checkpoint_iteration", "optimizer_updates", "normalization_updates", "no_failure_reset", "physical_objects", "elapsed_seconds"):
        _put(metrics, "evaluation/" + key, report.get(key))
    for index, case in enumerate(_rows(report.get("cases"))):
        if not isinstance(case, dict):
            continue
        prefix = f"evaluation/case_{index:05d}/"
        for key in CASE_SCALARS:
            _put(metrics, prefix + key, case.get(key))
        for key in ERROR_GROUPS:
            for statistic in ("mean", "p95", "max", "final"):
                _put(metrics, prefix + key + "/" + statistic, _nested(case, key, statistic))
        contact = _mapping(case.get("contact"))
        for key in ("expected_positive_intervals", "expected_contact_coverage", "expected_valid_fraction", "all_known_expected_contact_matched"):
            _put(metrics, prefix + "contact/" + key, contact.get(key))
        for key in ("missing_contact_intervals", "extra_contact_intervals", "max_missing_streak_steps", "max_missing_streak_seconds", "missing_streak_ge16"):
            values = contact.get(key)
            if isinstance(values, list) and len(values) == 2:
                for hand, value in zip(("left", "right"), values):
                    _put(metrics, prefix + "contact/" + key + "/" + hand, value)
        source_flags = _mapping(case.get("source_flag_comparison"))
        for key in ("valid_intervals", "mismatch_intervals", "mismatch_fraction_valid"):
            values = source_flags.get(key)
            if isinstance(values, list) and len(values) == 2:
                for hand, value in zip(("left", "right"), values):
                    _put(metrics, prefix + "source_flag_comparison/" + key + "/" + hand, value)
        for kind, channels in (("actual", CHANNELS), ("expected", CHANNELS[:2])):
            for channel in channels:
                source_channel = "object_ground_support" if channel == "object_ground" else channel
                for key in ("onset_count", "release_count", "valid_adjacent_intervals"):
                    _put(metrics, prefix + "contact/" + kind + "_events/" + channel + "/" + key,
                         _nested(contact, kind + "_transition_events", "channels", source_channel, key))
        for horizon in ("1", "8", "32"):
            for key in ("sequence_count", "valid_intervals", "signed_reward_gap_mean", "absolute_reward_gap_mean", "reward_gap_rmse"):
                _put(metrics, prefix + "paired_reward_gap/" + horizon + "/" + key,
                     _nested(case, "simulator_imagined_reward_gap", "horizons", horizon, key))
    return metrics


def _validate_payload(payload):
    """Recheck outgoing data at the SDK boundary, including direct API calls."""
    if set(payload) != {"schema_version", "run_id", "entity", "project", "resume", "config", "events", "summary", "privacy"}:
        raise ValueError("unknown outgoing tracking field")
    if payload["schema_version"] != "hoi_v1_curated_tracking_1":
        raise ValueError("unknown tracking payload version")
    _validate_destination(payload)
    config = payload["config"]
    enum_fields = {"method": ("HOI_v1_full_reference",), "tracking_schema": ("hoi_v1_curated_tracking_1",),
                   "training/amp": ("none", "bf16"), "training/experiment_kind": ("formal", "smoke"),
                   "training/variant": VARIANTS, "evaluation/variant": VARIANTS,
                   "evaluation/partition": ("train", "eval80", "test40", "validation", "regression", "test")}
    numeric_fields = {"training/" + key for key, value in TrainConfig().to_dict().items() if type(value) in (int, float, bool)}
    numeric_fields.add("training/wm_quality_gate_enabled")
    numeric_fields.update({"physics/hand_collision", "physics/self_collision", "dataset/selected_motions", "dataset/available_motions", "dataset/is_subset"})
    hash_fields = {"dataset/bundle_content_sha256", "dataset/manifest_content_sha256", "dataset/recipes_sha256", "dataset/hands_geometry_sha256",
                   "robot/mjcf_sha256", "robot/reference_urdf_sha256", "robot/native_asset_sha256", "source/content_sha256",
                   "evaluation/bundle_content_sha256", "evaluation/checkpoint_sha256"}
    for key, value in config.items():
        if key in enum_fields and value in enum_fields[key]:
            continue
        if key in numeric_fields and _number(value):
            continue
        if key in hash_fields and _digest(value):
            continue
        raise ValueError("outgoing configuration violates the whitelist")
    train_keys = {"train/" + key for key in ("iteration", "sim_steps", "lr", "elapsed_seconds", "wm_updates", "policy_updates")}
    train_keys.update("wm/" + key for key in ("total", "robot", "object", "contact"))
    train_keys.update("actor/" + key for key in ("pi_loss", "v_loss", "pi_gradnorm", "imag_human_and_hoi_return_per_step", "imag_survival"))
    train_keys.update("real/" + key for key in ("human_reward_normalized", "hoi_reward_normalized", "objective_reward_normalized", "terminations"))
    train_keys.update("contact/" + sign + "/" + channel for sign in ("fit_positive", "fit_negative") for channel in CHANNELS)
    train_keys.add("gate/policy_passed")
    train_keys.add("diagnostics/wm_quality_passed")
    previous = -1
    for event in payload["events"]:
        if set(event) != {"step", "metrics"} or type(event["step"]) is not int or event["step"] <= previous:
            raise ValueError("outgoing training events are malformed or unordered")
        if any(key not in train_keys or not _number(value) for key, value in event["metrics"].items()):
            raise ValueError("outgoing training metrics violate the whitelist")
        previous = event["step"]
    aggregate = {"count", "phc_pass", "object_pose_pass", "body_and_object_pose_pass", "supermimic_n_failed", "supermimic_norm_return",
                 "human_normalized_return", "hoi_normalized_return", "hoi_reference_mask_normalized_return", "objective_normalized_return", "checkpoint_iteration",
                 "optimizer_updates", "normalization_updates", "no_failure_reset", "physical_objects", "elapsed_seconds"}
    case_keys = set(CASE_SCALARS)
    case_keys.update(group + "/" + statistic for group in ERROR_GROUPS for statistic in ("mean", "p95", "max", "final"))
    case_keys.update("contact/" + key for key in ("expected_positive_intervals", "expected_contact_coverage", "expected_valid_fraction", "all_known_expected_contact_matched"))
    case_keys.update("contact/" + key + "/" + side for key in ("missing_contact_intervals", "extra_contact_intervals", "max_missing_streak_steps", "max_missing_streak_seconds", "missing_streak_ge16") for side in ("left", "right"))
    case_keys.update("source_flag_comparison/" + key + "/" + side for key in ("valid_intervals", "mismatch_intervals", "mismatch_fraction_valid") for side in ("left", "right"))
    case_keys.update("contact/" + kind + "_events/" + channel + "/" + key
                     for kind, channels in (("actual", CHANNELS), ("expected", CHANNELS[:2])) for channel in channels
                     for key in ("onset_count", "release_count", "valid_adjacent_intervals"))
    case_keys.update("paired_reward_gap/" + horizon + "/" + key for horizon in ("1", "8", "32")
                     for key in ("sequence_count", "valid_intervals", "signed_reward_gap_mean", "absolute_reward_gap_mean", "reward_gap_rmse"))
    for key, value in payload["summary"].items():
        match = re.fullmatch(r"evaluation/case_\d{5,}/(.+)", key)
        allowed = (key in {"evaluation/" + name for name in aggregate} or
                   key in {"checkpoint/iteration", "checkpoint/wm_updates_this_run", "checkpoint/policy_updates_this_run", "run/completed"} or
                   (key.startswith("last/") and key[5:] in train_keys) or (match and match.group(1) in case_keys))
        if not allowed or not _number(value):
            raise ValueError("outgoing evaluation summaries violate the whitelist")
    if any(value is not False for value in payload["privacy"].values()):
        raise ValueError("tracking capture/upload policy must remain disabled")


def build_payload(plan=None, records=(), evaluation=None):
    plan, evaluation = plan or {}, evaluation or {}
    events = [_training_event(row) for row in records]
    if any(events[index]["step"] <= events[index - 1]["step"] for index in range(1, len(events))):
        raise ValueError("batch training records must have strictly increasing iterations")
    summary = _evaluation_summary(evaluation)
    if events:
        summary.update({"last/" + key: value for key, value in events[-1]["metrics"].items()})
    entity, project = destination()
    payload = {"schema_version": "hoi_v1_curated_tracking_1", "run_id": uuid.uuid4().hex,
               "entity": entity, "project": project, "resume": "never", "config": _configuration(plan, evaluation),
               "events": events, "summary": summary,
               "privacy": {"source_uploads": False, "raw_data_uploads": False, "weight_uploads": False,
                           "rollout_uploads": False, "artifact_uploads": False, "console_capture": False,
                           "system_metadata_capture": False, "code_capture": False, "git_capture": False}}
    json.dumps(payload, allow_nan=False)
    _validate_payload(payload)
    return payload


def _legacy_base_url():
    # Reuse the existing tracking endpoint without importing its executable CLI.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "track.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == "WANDB_BASE_URL":
                    return node.value.value
    raise ValueError("existing HOI tracking endpoint declaration is unavailable")


def _base_url():
    value = os.environ.get("WANDB_BASE_URL") or _legacy_base_url()
    url = urlsplit(value)
    if url.scheme not in ("https", "http") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("configured W&B endpoint must not contain credentials/query fragments")
    return value


def _privacy_settings(wandb, base_url):
    fields = set(getattr(wandb.Settings, "model_fields", {}) or getattr(wandb.Settings, "__fields__", {}))
    for cls in wandb.Settings.__mro__:
        fields.update(getattr(cls, "__annotations__", {}))
    # A non-None empty host prevents the SDK from inferring/serializing the
    # machine hostname even when its other metadata collection is disabled.
    values = {"base_url": base_url, "host": "", "program": "hoi_v1_metrics", "program_relpath": "hoi_v1_metrics",
              "console": "off", "disable_code": True, "disable_git": True,
              "save_code": False, "ignore_globs": ("*", "**/*"), "silent": True, "disable_job_creation": True}
    for alternatives, value in ((("x_disable_stats", "_disable_stats"), True),
                                (("x_disable_meta", "_disable_meta"), True),
                                (("x_save_requirements", "_save_requirements"), False)):
        name = next((field for field in alternatives if field in fields), None)
        if name is None:
            raise RuntimeError("installed W&B cannot prove system metadata/stats/requirements capture is disabled")
        values[name] = value
    # This additional control exists in the isolated current SDK. Legacy SDKs
    # still use the required stats/meta flags and explicit empty host above.
    for name in ("x_disable_machine_info", "_disable_machine_info"):
        if name in fields:
            values[name] = True
            break
    return wandb.Settings(**values)


@contextmanager
def _privacy_environment(base_url):
    values = {"WANDB_BASE_URL": base_url, "WANDB_MODE": "online", "WANDB_DISABLE_CODE": "true", "WANDB_DISABLE_GIT": "true",
              "WANDB_CONSOLE": "off", "WANDB_SILENT": "true", "WANDB_IGNORE_GLOBS": "*,**/*", "WANDB_DISABLE_JOB_CREATION": "true",
              "WANDB_ERROR_REPORTING": "false"}
    metadata = ("WANDB_NAME", "WANDB_NOTES", "WANDB_TAGS", "WANDB_RUN_GROUP", "WANDB_JOB_TYPE", "WANDB_RUN_ID",
                "WANDB_RESUME", "WANDB_SWEEP_ID", "WANDB_CONFIG_PATHS", "WANDB_CODE_DIR", "WANDB_PROGRAM",
                "WANDB_PROGRAM_RELPATH", "WANDB_DOCKER", "WANDB_LAUNCH_CONFIG_PATH", "WANDB_LAUNCH")
    before = {key: os.environ.get(key) for key in (*values, *metadata)}
    for key in metadata:
        os.environ.pop(key, None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def send_online(payload, output):
    """Only called after explicit --online; this function never resumes a run."""
    _validate_payload(payload)
    base_url = _base_url()
    from .tracking_bootstrap import tracking_sdk
    try:
        with _privacy_environment(base_url), tracking_sdk(base_url) as wandb:
            settings = _privacy_settings(wandb, base_url)
            run = wandb.init(entity=payload["entity"], project=payload["project"], id=payload["run_id"], name="HOI-v1-batch-" + payload["run_id"][:8],
                             job_type="hoi_v1_batch", resume="never", mode="online", save_code=False,
                             dir=str(Path(output) / "wandb_runtime"), settings=settings, config=payload["config"])
            try:
                for event in payload["events"]:
                    run.log(event["metrics"], step=event["step"])
                for key, value in payload["summary"].items():
                    run.summary[key] = value
                url = run.url
            finally:
                run.finish()
    except BaseException as exc:
        # SDK exceptions can contain arbitrary server text or credentials.
        # Serialize only this enum and expose a fixed error to the caller.
        write_json(Path(output) / "online_failure.json", {"status": "tracking_failed", "failure_phase": "batch_sync",
                   "run_id": payload["run_id"], "entity": payload["entity"], "project": payload["project"], "resume": "never"})
        if isinstance(exc, Exception):
            raise RuntimeError("W&B batch synchronization failed; inspect online_failure.json") from None
        raise
    return {"run_id": payload["run_id"], "url": url, "entity": payload["entity"], "project": payload["project"], "resume": "never", "source_uploads": False,
            "raw_data_uploads": False, "weights_uploads": False, "rollout_uploads": False, "artifact_uploads": False}


class LiveTrainingTracker:
    """One fresh training run, with only the existing curated metric boundary."""
    def __init__(self, plan, output):
        self.payload = build_payload(plan)
        self.output = fresh_directory(output)
        write_json(self.output / "payload.json", self.payload)
        self.last_step, self.finished, self.run = -1, False, None
        self.run_url, self.failure_phase = None, None
        self.base_url = _base_url()
        self._session = ExitStack()
        try:
            self._session.enter_context(_privacy_environment(self.base_url))
            from .tracking_bootstrap import tracking_sdk
            wandb = self._session.enter_context(tracking_sdk(self.base_url))
            self.run = wandb.init(entity=self.payload["entity"], project=self.payload["project"],
                                  id=self.payload["run_id"], name="HOI-v1-train-" + self.payload["run_id"][:8],
                                  job_type="hoi_v1_training", resume="never", mode="online", save_code=False,
                                  dir=str(self.output / "wandb_runtime"), settings=_privacy_settings(wandb, self.base_url),
                                  config=self.payload["config"])
            self.run_url = self.run.url
            write_json(self.output / "live_start.json", self.receipt("online_started"))
        except BaseException as exc:
            try:
                if self.run is not None:
                    try:
                        self.run.finish()
                    except BaseException:
                        pass  # Preserve the original initialization exception.
            finally:
                try:
                    self._record_failure("initialization")
                finally:
                    self._session.close()
            if isinstance(exc, Exception):
                raise RuntimeError("W&B initialization failed; inspect live_failure.json") from None
            raise

    def receipt(self, status):
        return {"status": status, "run_id": self.payload["run_id"], "entity": self.payload["entity"],
                "project": self.payload["project"], "base_url": self.base_url, "resume": "never",
                "url": self.run_url,
                "source_uploads": False, "raw_data_uploads": False, "weight_uploads": False,
                "rollout_uploads": False, "artifact_uploads": False}

    def _record_failure(self, phase):
        if phase not in ("initialization", "metric_sync", "summary_sync", "shutdown"):
            raise ValueError("unknown tracking failure phase")
        if self.failure_phase is None:
            self.failure_phase = phase
            write_json(self.output / "live_failure.json", {**self.receipt("tracking_failed"), "failure_phase": phase})

    def _summary(self, values):
        try:
            _validate_payload({**self.payload, "events": [], "summary": values})
            for key, value in values.items():
                self.run.summary[key] = value
        except BaseException as exc:
            self._record_failure("summary_sync")
            if isinstance(exc, Exception):
                raise RuntimeError("W&B summary synchronization failed; inspect live_failure.json") from None
            raise

    def log_record(self, record):
        if self.finished:
            raise RuntimeError("live training tracker is already finished")
        event = _training_event(record)
        if event["step"] <= self.last_step:
            raise ValueError("live training steps must increase within the same fresh run")
        _validate_payload({**self.payload, "events": [event], "summary": {}})
        try:
            self.run.log(event["metrics"], step=event["step"])
        except BaseException as exc:
            self._record_failure("metric_sync")
            if isinstance(exc, Exception):
                raise RuntimeError("W&B metric synchronization failed; inspect live_failure.json") from None
            raise
        self.last_step = event["step"]
        self._summary({"last/" + key: value for key, value in event["metrics"].items()})

    def checkpoint(self, iteration, wm_updates, policy_updates):
        if self.finished or any(type(value) is not int or value < 0 for value in (iteration, wm_updates, policy_updates)):
            raise ValueError("checkpoint tracking requires a live run and explicit nonnegative counts")
        self._summary({"checkpoint/iteration": iteration, "checkpoint/wm_updates_this_run": wm_updates,
                       "checkpoint/policy_updates_this_run": policy_updates})

    def finish(self, *, completed):
        if self.finished:
            return
        if type(completed) is not bool:
            raise ValueError("live completion status must be boolean")
        self.finished = True
        try:
            try:
                self._summary({"run/completed": completed})
            finally:
                try:
                    self.run.finish()
                except BaseException as exc:
                    self._record_failure("shutdown")
                    if isinstance(exc, Exception):
                        raise RuntimeError("W&B shutdown failed; inspect live_failure.json") from None
                    raise
        finally:
            try:
                status = "tracking_failed" if self.failure_phase else "completed" if completed else "failed"
                write_json(self.output / "live_finish.json", {**self.receipt(status), "last_metric_step": self.last_step,
                                                              "failure_phase": self.failure_phase})
            finally:
                self._session.close()


@contextmanager
def live_training(plan, output):
    """WANDB_SYNC=1 alone opts into SDK/network; disabled runs import no SDK."""
    enabled = os.environ.get("WANDB_SYNC", "0")
    if enabled not in ("0", "1"):
        raise ValueError("WANDB_SYNC must be 0 or 1")
    if enabled == "0":
        yield None
        return
    tracker = LiveTrainingTracker(plan, output)
    try:
        yield tracker
    except BaseException:
        # A reporting shutdown error must not replace a learner/native error.
        # finish() still records its own fixed failure phase before raising.
        try:
            tracker.finish(completed=False)
        except BaseException:
            pass
        raise
    else:
        tracker.finish(completed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="completed training directory; reads plan.json and metrics.jsonl")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default: prepare inspectable local payload only")
    mode.add_argument("--online", action="store_true", help="explicitly connect and send only the prepared whitelist payload")
    args = parser.parse_args(argv)
    plan_path = args.plan or (args.run_dir / "plan.json" if args.run_dir else None)
    metrics_path = args.metrics or (args.run_dir / "metrics.jsonl" if args.run_dir else None)
    if not any((plan_path, metrics_path, args.evaluation)):
        parser.error("supply a run directory or explicit plan/metrics/evaluation input")
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path else {}
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path else []
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else {}
    payload = build_payload(plan, records, evaluation)
    output = fresh_directory(args.output)
    write_json(output / "payload.json", payload)
    receipt = send_online(payload, output) if args.online else {"status": "dry_run_only", "network_calls": 0, "run_id": payload["run_id"]}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": "online_sent" if args.online else "dry_run_only", "run_id": payload["run_id"],
                      "training_events": len(payload["events"]), "summary_values": len(payload["summary"])}))
    return payload


if __name__ == "__main__":
    main()
