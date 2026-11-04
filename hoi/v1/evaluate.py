"""Frozen HOI-trained policy evaluation, retaining complete reference durations.

This is separate from the historical ``hoi.run`` zero-shot protocol.  Evaluation
never constructs an optimizer/replay and never updates any normalizer.  The
39-body diagnostics use the bound ULTRA virtual URDF, not the 14 reward bodies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from hoi.data.manifest import read_manifest, sha256_file
from hoi.data.timebase import CanonicalTimebase
from hoi.ultra_metrics import clip_metrics
from .checkpoint import load_checkpoint, checkpoint_variant, checkpoint_training_progress
from .config import fresh_directory, write_json, source_identity
from .contacts import CHANNELS, LABEL_SCHEMA
from .networks import Policy, WorldModel, Critic
from .state import unflatten_state, unflatten_object


ROLLOUT_SCHEMA = "supermimic_hoi_v1_full_duration_rollout_2"


def _evaluation_provenance(checkpoint):
    """Keep training and evaluation source identities distinct across snapshots."""
    training = checkpoint.get("extra", {}).get("source_identity")
    known = isinstance(training, dict) and bool(training)
    return {"evaluation_source_identity": source_identity(),
            "checkpoint_training_source_identity": deepcopy(training) if known else {"status": "unknown", "reason": "checkpoint did not record training source identity"},
            "checkpoint_training_source_identity_status": "recorded" if known else "unknown",
            "checkpoint_training_progress": checkpoint_training_progress(checkpoint)}


def _load_evaluation_snapshot(path):
    from .train import read_checkpoint
    checkpoint, digest = read_checkpoint(path)
    return checkpoint, digest, _evaluation_provenance(checkpoint)


def _verify_evaluation_source(expected):
    if source_identity() != expected:
        raise RuntimeError("evaluation executable source changed during frozen evaluation")


def _cpu(value):
    return value.detach().cpu().clone()


def _stats(value):
    value = value.detach().cpu().float()
    return {"mean": float(value.mean()), "p95": float(torch.quantile(value, .95)),
            "max": float(value.max()), "final": float(value[-1])}


def _virtual_state(S, fk39, joint_order):
    """Use link-origin root velocity; virtual body velocities are URDF-COM."""
    from supermimic.rotations import mat_to_quat
    p, R, v, w, q, qd = S
    wxyz = mat_to_quat(R)
    root = torch.cat((p, wxyz[..., [1, 2, 3, 0]], v, w), -1)
    result = fk39.forward(root, q[..., joint_order], qd[..., joint_order], root_velocity_is_com=False)
    if result["body_pos"].shape[-2:] != (39, 3) or result["body_rot"].shape[-2:] != (39, 4):
        raise ValueError("PHC evaluation requires actual 39-body virtual FK and xyzw rotations")
    return result


def _state_record(snapshot, motion, fk39, joint_order):
    S, O = unflatten_state(snapshot["s"]), unflatten_object(snapshot["o"])
    body = _virtual_state(S, fk39, joint_order)
    f = snapshot["f"]
    values = {"s": snapshot["s"], "o": snapshot["o"], "q": S[4], "qd": S[5],
              "reference_object_position": motion.object_p[f], "reference_object_velocity": motion.object_v[f],
              "reference_object_omega": motion.object_w[f], **body}
    from .state import flatten_state, flatten_object
    support_keys=("support_pose_xyzw","reference_support_pose_xyzw","support_present")
    if any(key in snapshot for key in support_keys):
        if not all(key in snapshot for key in support_keys):
            raise ValueError("native support display states must be recorded together")
        if snapshot[support_keys[0]].shape != (len(f),2,7) or snapshot[support_keys[1]].shape != (len(f),2,7) or snapshot[support_keys[2]].shape != (len(f),2):
            raise ValueError("native support display states lost fixed slot/frame alignment")
        values.update({key:snapshot[key] for key in support_keys})
    elif any(getattr(motion,"scene_supports",[])):
        raise ValueError("evaluation cannot omit declared native scene support states")
    if motion.source_reference is not None:
        values["reference_source630"] = motion.source_reference[f]
    ref_body = _virtual_state(motion.ref_state(f), fk39, joint_order)
    values["reference_body_pos"] = ref_body["body_pos"]
    values["reference_body_rot"] = ref_body["body_rot"]
    values["reference_robot"] = flatten_state(motion.ref_state(f))
    values["reference_object"] = flatten_object(motion.ref_object(f))
    for key, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite evaluation state/reference: {key}")
    return {key: _cpu(value) for key, value in values.items()}


def _contact_metrics(control, expected, valid):
    from .diagnostics import contact_transition_events, validate_actual_contact_fields
    validate_actual_contact_fields(control, len(expected))
    expected_transitions = contact_transition_events(expected, valid, tuple("object_ground_support" if name == "object_ground" else name for name in CHANNELS))
    actual_transitions = contact_transition_events(control["contact"], control["contact_valid"],
                                                   tuple("object_ground_support" if name == "object_ground" else name for name in CHANNELS))
    actual = control["contact"] > .5
    pair_valid = control["contact_valid"].bool()
    expected, valid = expected.bool(), valid.bool() & pair_valid
    missing = expected & ~actual & valid
    extra = ~expected & actual & valid
    streak = torch.zeros(len(CHANNELS), dtype=torch.long)
    maximum = streak.clone()
    for step in missing:
        streak = torch.where(step, streak + 1, 0)
        maximum = torch.maximum(maximum, streak)
    positive = expected & valid
    expected_total = int(positive.sum())
    covered = int((positive & actual).sum())
    has_pair_truth = bool(valid.any())
    return {"channels": list(CHANNELS), "label_schema": LABEL_SCHEMA,
            "actual_positive_intervals": ((control["contact"] > .5) & control["contact_valid"].bool()).sum(0).tolist(),
            "actual_valid_intervals": control["contact_valid"].bool().sum(0).tolist(),
            "expected_positive_intervals": expected_total,
            "expected_contact_coverage": covered / expected_total if expected_total else None,
            "expected_valid_fraction": float(valid.float().mean()),
            "actual_pair_valid_fraction": control["contact_valid"].float().mean(0).tolist(),
            "actual_contact_fraction": (control["contact"].float() * control["contact_valid"]).sum(0).div(
                control["contact_valid"].sum(0).clamp_min(1)).tolist(),
            "maximum_interval_normal_impulse_Ns": control["contact_impulse_Ns"].amax(0).tolist(),
            "missing_contact_intervals": missing.sum(0).tolist(), "extra_contact_intervals": extra.sum(0).tolist(),
            "max_missing_streak_steps": maximum.tolist(), "max_missing_streak_seconds": (maximum * .02).tolist(),
            "missing_streak_ge16": (maximum >= 16).tolist(),
            "all_known_expected_contact_matched": bool(not missing.any() and not extra.any()) if has_pair_truth else None,
            "actual_transition_events": actual_transitions, "expected_transition_events": expected_transitions,
            "definition": "16 actual region-pair interval channels; grouped tri-state reference targets; unknown excluded; no functional-success assertion"}


def _canonical_tracking_metrics(states, control, joint_order, body_names):
    """Score audited G1 reference arrays directly, without manufacturing ULTRA source fields."""
    from hoi.ultra_metrics import CRITERIA, first_sustained_violation
    from supermimic.rotations import rot_angle_sq
    ref_s = unflatten_state(states["reference_robot"])[0:6]
    ref_o = unflatten_object(states["reference_object"])
    actual_o = unflatten_object(states["o"])
    body = states["body_pos"][1:]
    ref_body = states["reference_body_pos"][1:]
    if body.shape != ref_body.shape or body.shape[1:] != (39, 3):
        raise ValueError("canonical tracking metrics require aligned 39-body reference FK")
    distances = (body - ref_body).norm(dim=-1)
    body_error = distances.mean(-1)
    object_error = (actual_o[0][1:] - ref_o[0][1:]).norm(dim=-1)
    angle = rot_angle_sq(actual_o[1][1:], ref_o[1][1:]).clamp_min(0).sqrt() * (180 / torch.pi)
    object_bad = (object_error > CRITERIA["object_position_limit_m"]) | (angle > CRITERIA["object_orientation_limit_deg"])
    failure = first_sustained_violation(object_bad, .02, CRITERIA["object_violation_duration_s"])
    body_bad = torch.where(body_error > CRITERIA["body_frame_mean_limit_m"])[0]
    names = list(body_names or [])
    hand_ids = [names.index(name) for name in ("left_rubber_hand", "right_rubber_hand") if name in names]
    T = len(control["action"])
    result = {"criteria_version": CRITERIA["version"], "reference_basis": "audited_G1_reference_same_bound_URDF_FK",
              "scored_frames": T, "scored_duration_s": T * .02, "reference_start_s": .02, "reference_end_s": T * .02,
              "body_position_m": _stats(body_error), "object_position_m": _stats(object_error),
              "object_orientation_deg": _stats(angle),
              "joint_rmse_rad": float((states["q"][1:] - ref_s[4][1:]).square().mean().sqrt()),
              "action_at_clip_fraction": float((control["action"].abs() >= 1 - 1e-6).float().mean()),
              "phc_pass": not len(body_bad), "object_pose_pass": failure is None,
              "body_and_object_pose_pass": not len(body_bad) and failure is None,
              "first_phc_failure_reference_s": (int(body_bad[0]) + 1) * .02 if len(body_bad) else None,
              "first_object_failure_reference_s": (failure + 1) * .02 if failure is not None else None,
              "minimum_pelvis_height_m": float(states["s"][1:, 2].min()),
              "pelvis_below_0_3m": bool((states["s"][1:, 2] < CRITERIA["pelvis_low_diagnostic_m"]).any())}
    result["hand_position_m"] = _stats(distances[:, hand_ids].mean(-1)) if len(hand_ids) == 2 else None
    return result


def score_rollout(states, control, motion, clip_id, joint_order, *, max_human_reward=.1, source_body_names=None, variant="hoi_v1"):
    """Recomputable CPU scoring, with controls T and states/references T+1."""
    T = len(control["action"])
    if T < 1 or any(len(value) != T + 1 for value in states.values()):
        raise ValueError("full rollout requires T controls and T+1 states/reference fields")
    if any(len(value) != T for value in control.values()):
        raise ValueError("evaluation control fields have inconsistent durations")
    source = states.get("reference_source630")
    if source is not None and source.shape != (T + 1, 630):
        raise ValueError("full source reference630 is required for independent PHC scoring")
    obj = states["o"]
    object_xyzw = torch.cat((obj[:, :3], obj[:, [4, 5, 6, 3]], obj[:, 7:]), -1)
    packed = {"body_pos": states["body_pos"], "object_state": object_xyzw, "reference": source,
              "q": states["q"][:, joint_order], "action": torch.cat((torch.zeros_like(control["action"][:1]), control["action"])),
              "time": torch.arange(T + 1, dtype=torch.float64) * .02}
    metrics = clip_metrics(packed, .02, exclude_initial=1) if source is not None else _canonical_tracking_metrics(states, control, joint_order, source_body_names)
    # The legacy metric's |a|>=1 statistic is not saturation of G1 PD offsets.
    metrics["action_abs_ge_one_fraction"] = metrics.pop("action_at_clip_fraction")
    metrics["action_abs_ge_twenty_fraction"] = float((control["action"].abs() >= 20).float().mean())
    human_term = control["human_termination"].bool()
    terminated = torch.where(human_term)[0]
    first_human = int(terminated[0]) + 1 if len(terminated) else -1
    author_steps = first_human if first_human >= 0 else T
    human_return = control["human_reward"][:author_steps].sum()
    start = int(motion.clip_start[clip_id])
    frames = torch.arange(start + 1, start + T + 1, device=motion.root_p.device)
    expected, valid = (value.detach().cpu() for value in motion.expected(frames))
    contacts = _contact_metrics(control, expected, valid)
    from .diagnostics import source_flag_comparison
    source_comparison = (source_flag_comparison(source[1:], expected[:, :2], valid[:, :2], source_body_names)
                         if source is not None else {"available": False, "source_is_pair_truth": False,
                         "affects_reward_or_gate": False, "reason": "source has no original ULTRA630 contact flags"})
    task_failure = torch.where(control["hoi_termination"].bool())[0]
    contact_pass = contacts["all_known_expected_contact_matched"]
    full_human = control["human_reward"].sum()
    full_hoi = control["reward"].sum()
    approach_available = ((expected[:, :2] > .5) & (valid[:, :2] > .5)).any(-1)
    # Signed contact reward has a positive ceiling only when a positive target exists.
    contact_available = ((expected > .5) & (valid > .5) & control["contact_valid"].bool()).any(-1)
    # Human .10 + object pose/velocity .04 are always available. Each of
    # approach and pair-match contributes at most .005 when its mask is valid.
    from .variants import reward_config, max_step_reward
    cfg, objective_max = reward_config(variant), max_step_reward(variant)
    always_available = .1 + .02 * sum(getattr(cfg, key) for key in ("w_position", "w_orientation", "w_linear_velocity", "w_angular_velocity"))
    upper_bound = (always_available + .02 * cfg.w_approach * approach_available.float() +
                   .02 * cfg.w_contact * contact_available.float()).sum()
    metrics.update({"control_steps": T, "state_frames": T + 1, "full_reference_completed": T == int(motion.clip_len[clip_id]) - 1,
                    "first_supermimic_failure_step": first_human, "supermimic_norm_return": float(human_return / (T * max_human_reward)),
                    "human_normalized_return": float(human_return / (T * max_human_reward)),
                    "hoi_normalized_return": float(full_hoi / (T * .15)),
                    "objective_normalized_return": float(full_hoi / (T * objective_max)), "objective_max_step_reward": objective_max,
                    "hoi_reference_mask_reward_upper_bound": float(upper_bound),
                    "hoi_reference_mask_normalized_return": float(full_hoi / upper_bound),
                    "first_hoi_failure_step": int(task_failure[0]) + 1 if len(task_failure) else -1,
                    "human_full_duration_reward": float(full_human),
                    "hoi_full_duration_reward": float(full_hoi), "hoi_extra_full_duration_reward": float(full_hoi - full_human), "contact": contacts,
                    "source_flag_comparison": source_comparison,
                    "object_linear_velocity_mps": _stats((states["o"][1:, 7:10] - states["reference_object_velocity"][1:]).norm(dim=-1)),
                    "object_angular_velocity_radps": _stats((states["o"][1:, 10:13] - states["reference_object_omega"][1:]).norm(dim=-1)),
                    "pose_human_expected_contact_joint_pass": (metrics["body_and_object_pose_pass"] and first_human < 0 and contact_pass)
                    if contact_pass is not None else None})
    return metrics


@torch.no_grad()
def evaluate_group(policy, motion, mdp, shapes, workers, clip_ids, *, fk39, joint_names, output, metadata=None,
                   verify_tracked_fk=True, wm=None, diagnostic_batch_size=128, variant="hoi_v1"):
    """Evaluate one fixed-asset worker group; failure never truncates a case.

    Worker snapshots must use global bank frames and retain the post-action
    state. Any reset or early frame jump is rejected, rather than scored.
    """
    group_metadata = deepcopy(metadata or {})
    group_metadata.setdefault("evaluation_source_content_sha256", source_identity()["content_sha256"])
    group_metadata.setdefault("checkpoint_training_source_content_sha256", None)
    group_metadata.setdefault("checkpoint_training_source_identity_status", "unknown")
    group_metadata.setdefault("checkpoint_training_progress", checkpoint_training_progress({}))
    if (group_metadata.get("contact_label_schema", LABEL_SCHEMA) != LABEL_SCHEMA or
            tuple(group_metadata.get("contact_channels", CHANNELS)) != CHANNELS):
        raise ValueError("evaluation metadata differs from the current 16-channel contact contract")
    group_metadata["contact_channels"] = list(CHANNELS)
    group_metadata["contact_label_schema"] = LABEL_SCHEMA
    if "contact_schema" in group_metadata:
        _validate_contact_contract(group_metadata["contact_schema"])
    if len(fk39.body_names) != 39 or len(set(fk39.body_names)) != 39:
        raise ValueError("PHC requires the bound original 39-body name order")
    if len(joint_names) != 29 or len(set(joint_names)) != 29 or len(fk39.dof_names) != 29 or set(joint_names) != set(fk39.dof_names):
        raise ValueError("runtime/ULTRA joint names do not match")
    device = next(policy.parameters()).device
    ids = torch.tensor(clip_ids, device=device, dtype=torch.long)
    if len(ids) != workers.N or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("one unique evaluation clip is required per worker environment")
    starts, ends = motion.clip_start[ids], motion.clip_end[ids]
    lengths = ends - starts
    if bool((lengths < 1).any()):
        raise ValueError("evaluation clips require at least one full control interval")
    assets = motion.asset_ids[starts]
    if len(assets.unique()) != 1:
        raise ValueError("evaluate_group only accepts one fixed geometry group")
    order = [joint_names.index(name) for name in fk39.dof_names]
    snapshot = workers.reset(starts)
    if not torch.equal(snapshot["f"], starts) or not torch.equal(snapshot["asset_id"], assets):
        raise ValueError("evaluation reset must reach the exact clip start and asset")
    if not bool((snapshot["ep_step"] == 0).all()):
        raise ValueError("evaluation reset did not clear episode history")
    before = {key: _cpu(value) for key, value in policy.state_dict().items()}
    training = policy.training
    policy.eval()
    output = fresh_directory(output)
    records, controls, robot_hist, object_hist = [], [], None, None
    previous_teacher_state = None
    first = _state_record(snapshot, motion, fk39, order)
    if verify_tracked_fk:
        from supermimic.robot_metadata import TRACKED_BODIES
        common = [fk39.body_names.index(name) for name in TRACKED_BODIES]
        tracked = mdp.kin.forward(*unflatten_state(snapshot["s"]))[0]
        if not torch.allclose(first["body_pos"][:, common], tracked.detach().cpu(), atol=2e-4, rtol=0):
            raise ValueError("ULTRA39 and original 14-body FK differ beyond the kinematic gate")
    records.append(first)
    if "support_pose_xyzw" in first:
        from .scene_supports import SUPPORT_IDS, SCENE_SUPPORT_SCHEMA, omni_support_recipe, digest
        group_metadata["scene_support_display"]={"schema":SCENE_SUPPORT_SCHEMA,"support_ids":list(SUPPORT_IDS),
            "pose_source":"saved_native_world_transforms_relative_to_environment_origin",
            "reference_pose_source":"saved_sampled_reset_frame_held_for_episode",
            "quaternion":"xyzw","geometry_recipe":omni_support_recipe(),"geometry_recipe_sha256":digest(omni_support_recipe())}
    try:
        for step in range(int(lengths.max())):
            f = torch.minimum(starts + step, ends)
            if not torch.equal(snapshot["f"], f):
                raise RuntimeError("worker reset/jumped frames during full-duration evaluation")
            S, O = unflatten_state(snapshot["s"]), unflatten_object(snapshot["o"])
            robot = mdp.obs(S, snapshot["a_prev"], snapshot["xi"], f)
            obj = mdp.object_obs(S, O, f)
            hg, og = motion.goal(f, 10), motion.object_goal(S, O, f, 10)
            robot_hist = robot[:, None].expand(-1, 10, -1).clone() if robot_hist is None else torch.cat((robot_hist[:, 1:], robot[:, None]), 1)
            object_hist = obj[:, None].expand(-1, 10, -1).clone() if object_hist is None else torch.cat((object_hist[:, 1:], obj[:, None]), 1)
            from .scene_context import scene_from_record
            actor_extra = {"scene_context":scene_from_record(S,snapshot)}
            if getattr(policy, "is_privileged_teacher", False):
                previous = S if previous_teacher_state is None else previous_teacher_state
                actor_extra["privileged"] = mdp.privileged_features(
                    S, O, f, snapshot["a_prev"], snapshot["xi"], snapshot["xi_o"], previous)
                previous_teacher_state = tuple(value.clone() for value in S)
            action = policy(robot_hist, hg, object_hist, og, shapes.points(assets), **actor_extra)
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError("frozen evaluation policy produced nonfinite actions")
            transition = workers.step(action)
            from .diagnostics import validate_actual_contact_fields
            validate_actual_contact_fields({"contact": transition["contact"], "contact_valid": transition["contact_valid"],
                                            "contact_impulse_Ns": transition["impulse"]}, workers.N)
            after = workers.snapshot()
            f1 = torch.minimum(f + 1, ends)
            if not torch.equal(after["f"], f1) or not torch.equal(transition["f"], f):
                raise RuntimeError("post-action frame alignment failed; possible failure reset")
            if not torch.equal(transition["a"], action):
                raise RuntimeError("worker executed an action different from the saved policy action")
            if not torch.equal(after["s"], transition["s1"]) or not torch.equal(after["o"], transition["o1"]):
                raise RuntimeError("worker snapshot differs from terminal state; evaluation reset is forbidden")
            S1, O1 = unflatten_state(after["s"]), unflatten_object(after["o"])
            human = mdp.human.reward(S1, f1, action, snapshot["a_prev"])
            human_termination = mdp.human.termination(S1, f1)
            expected, expected_valid = motion.expected(f1)
            values = {"action": action, "a_prev": snapshot["a_prev"], "robot_observation": robot, "object_observation": obj,
                      "human_goal": hg, "object_goal": og, "reward": transition["r"], "human_reward": human,
                      "human_termination": human_termination, "hoi_termination": transition["term"],
                      "actual_contact": transition["contact"], "contact": transition["contact"],
                      "contact_valid": transition["contact_valid"], "contact_impulse_Ns": transition["impulse"],
                      "expected_contact": expected, "expected_valid": expected_valid,
                      "contact_missing_counter": transition["contact_missing_counter"],
                      "xi": snapshot["xi"], "xi_o": snapshot["xi_o"],
                      "support_pose_xyzw":snapshot["support_pose_xyzw"],"support_present":snapshot["support_present"]}
            if "contact_force" in transition:
                values["contact_force_N"] = transition["contact_force"]
            if "failure_reason" in transition:
                values["failure_reason"] = transition["failure_reason"]
            if any(not bool(torch.isfinite(value).all()) for value in values.values()):
                raise FloatingPointError("nonfinite evaluation control/contact field")
            controls.append({key: _cpu(value) for key, value in values.items()})
            records.append(_state_record(after, motion, fk39, order))
            snapshot = after
        if any(not torch.equal(before[key], _cpu(value)) for key, value in policy.state_dict().items()):
            raise RuntimeError("frozen evaluation changed policy weights or normalization statistics")
    finally:
        policy.train(training)
    case_reports = []
    for index, clip in enumerate(clip_ids):
        T = int(lengths[index])
        states = {key: torch.stack([r[key][index] for r in records[:T + 1]]) for key in records[0]}
        control = {key: torch.stack([r[key][index] for r in controls[:T]]) for key in controls[0]}
        metrics = score_rollout(states, control, motion, clip, order, source_body_names=list(fk39.body_names), variant=variant)
        if wm is not None:
            from .diagnostics import executed_action_reward_gap
            metrics["simulator_imagined_reward_gap"] = executed_action_reward_gap(
                wm, mdp, states, control, torch.arange(int(starts[index]), int(ends[index]) + 1, device=device),
                shapes.points(assets[index]), batch_size=diagnostic_batch_size)
        else:
            metrics["simulator_imagined_reward_gap"] = {"available": False, "reason": "frozen checkpoint WM not supplied"}
        name = f"case_{clip:05d}"
        extra = deepcopy(group_metadata)
        clip_metadata = extra.get("clips", [{}] * len(motion.clip_len))[clip] if "clips" in extra else {}
        # Curated sources already have an audited 50 Hz grid; never reinterpret native frame counts as 60 Hz.
        source_frames = clip_metadata.get("source_frames") if motion.source_reference is not None else None
        clock = CanonicalTimebase().sample_grid(source_frames) if source_frames is not None else {"time": torch.arange(T + 1, dtype=torch.float64) * .02}
        if len(clock["time"]) != T + 1:
            raise ValueError("source60/control50 canonical frame count differs from recorded evaluation")
        payload = {"schema_version": ROLLOUT_SCHEMA, "states": states, "control": control, **clock,
                   "metadata": {"clip_id": clip, "name": motion.names[clip], "partition": motion.partition, "variant": variant,
                                "asset_id": int(assets[index]), "control_dt": .02, "robot_object_quaternion": "wxyz",
                                "virtual_body_quaternion": "xyzw", "body_names": list(fk39.body_names),
                                "joint_names": list(joint_names), "virtual_dof_names": list(fk39.dof_names),
                                "body_velocity_reference_point": "URDF_COM; source velocity semantics remain unverified",
                                "timebase": CanonicalTimebase().metadata() if source_frames is not None else {"control_hz": 50, "source": "audited_curated_target_grid"}, "no_failure_reset": True,
                                "source_frames": source_frames,
                                "canonical_duration_s": (source_frames - 1) / 60 if source_frames is not None else None,
                                "last_control_time_s": T / 50,
                                "tail_remainder_s": (5 * (source_frames - 1) - 6 * T) / 300 if source_frames is not None else None,
                                "optimizer_updates": 0, "normalization_updates": 0, "robot_observation_noise": False,
                                "action_timing": "action/obs/goal[t] produce state[t+1]; T controls and T+1 states/reference",
                                "failure_reason_bits": {"human_height": 1, "human_tilt": 2, "object_distance": 4,
                                                        "clip_end": 8, "max_episode_steps": 16},
                                "provenance": motion.provenance[clip], **extra}, "metrics": metrics}
        path = output / f"{name}.pt"
        with path.open("xb") as stream:
            torch.save(payload, stream)
        case_reports.append({"id": motion.names[clip], "clip_id": clip, "asset_id": int(assets[index]),
                             "rollout": str(path.resolve()), "rollout_sha256": sha256_file(path), **metrics})
    write_json(output / "evaluation.json", {"cases": case_reports, "optimizer_updates": 0, "normalization_updates": 0,
                                           "contact_channels": list(CHANNELS), "contact_label_schema": LABEL_SCHEMA,
                                           "no_failure_reset": True, "physical_objects": True})
    return case_reports


def _load_fk39(bundle, joint_names, device):
    from hoi.ultra_kinematics import UltraKinematics
    virtual = bundle.get("reference_kinematics")
    if not virtual or len(virtual["body_names"]) != 39 or set(virtual["dof_names"]) != set(joint_names):
        raise ValueError("evaluation requires the original bound ULTRA39 virtual URDF and 29 DOF names")
    if sha256_file(virtual["urdf"]["path"]) != virtual["urdf"]["sha256"]:
        raise ValueError("evaluation virtual URDF bytes changed")
    return UltraKinematics(virtual["urdf"]["path"], virtual["body_names"], virtual["dof_names"], device)


def _read_bundle_header(path):
    from .prepare import BUNDLE_SCHEMA, _digest_json
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    from hoi.data.curated_bundle import CURATED_BUNDLE_SCHEMA, read_curated_bundle_header
    if bundle.get("schema_version") == CURATED_BUNDLE_SCHEMA:
        return read_curated_bundle_header(path)
    digest = bundle.pop("bundle_content_sha256", None)
    if bundle.get("schema_version") != BUNDLE_SCHEMA or digest != _digest_json(bundle):
        raise ValueError("training/evaluation bundle schema/content hash changed")
    bundle["bundle_content_sha256"] = digest
    if sha256_file(bundle["manifest"]["path"]) != bundle["manifest"]["sha256"]:
        raise ValueError("bound dataset manifest bytes changed")
    if sha256_file(bundle["recipes"]["path"]) != bundle["recipes"]["sha256"]:
        raise ValueError("bound collider/physical/hand recipe bytes changed")
    read_manifest(bundle["manifest"]["path"])
    return bundle


def validate_evaluation_identity(training_bundle, evaluation_bundle):
    """Allow distinct partitions only under identical data/geometry contracts."""
    if training_bundle["partition"] != "train":
        raise ValueError("checkpoint identity must bind a train-partition bundle")
    for key in ("manifest", "timebase", "recipes", "robot_fk", "reference_kinematics"):
        if not training_bundle.get(key) or training_bundle.get(key) != evaluation_bundle.get(key):
            raise ValueError(f"training/evaluation {key} contract differs")
    if training_bundle.get("native_robot_asset") != evaluation_bundle.get("native_robot_asset"):
        raise ValueError("training/evaluation verified native robot/hand overlay differs")
    def asset_contracts(bundle):
        fields = ("asset_id", "object_asset", "exact_scale", "collision_hash", "native_asset_sha256", "physical_profile")
        return [{**{key: asset[key] for key in fields}, "geometry_cache_sha256": asset["geometry_cache"]["sha256"]}
                for asset in bundle["assets"]]
    if asset_contracts(training_bundle) != asset_contracts(evaluation_bundle):
        raise ValueError("training/evaluation asset, scale, collider or physical profile differs")
    if training_bundle["hands"]["geometry_cache"]["sha256"] != evaluation_bundle["hands"]["geometry_cache"]["sha256"]:
        raise ValueError("training/evaluation hand geometry differs")
    return True


def aggregate_evaluation_cases(cases):
    """The same per-clip aggregation for the full partition and named subsets."""
    from hoi.run import aggregate
    if not cases:
        raise ValueError("cannot aggregate an empty evaluation")
    values = aggregate(cases)
    for key in ("human_normalized_return", "hoi_normalized_return", "hoi_reference_mask_normalized_return", "objective_normalized_return"):
        values[key] = sum(case[key] for case in cases) / len(cases)
    return values


def grail_sitting_evaluation(bundle, training_bundle, cases):
    """Derive the held-out sitting result from complete, identity-checked cases.

    Case clip_id is the motion-bank index, whereas bundle clip_id is the frozen
    source identifier. Membership comes exclusively from the validated bundle.
    """
    if not bundle.get("compact_selection") or bundle.get("partition") != "protected_eval":
        return None
    compact = bundle["compact_selection"]
    if (training_bundle.get("partition") != "train" or
            training_bundle.get("compact_selection", {}).get("split_content_sha256") != compact["split_content_sha256"]):
        raise ValueError("independent sitting evaluation requires the paired compact train split")
    clips = bundle["clips"]
    indexed = {}
    for case in cases:
        index = case.get("clip_id")
        if type(index) is not int or not 0 <= index < len(clips) or index in indexed:
            raise ValueError("evaluation has duplicate or unknown clip indices")
        clip = clips[index]
        if (type(case.get("asset_id")) is not int or case["asset_id"] != clip["asset_id"]
                or case.get("id") != clip["name"]):
            raise ValueError("evaluation case identity differs from its frozen clip/asset")
        indexed[index] = case
    if set(indexed) != set(range(len(clips))):
        raise ValueError("independent sitting metrics require every protected evaluation clip exactly once")
    selected = [i for i, clip in enumerate(clips) if clip["dataset"] == "grail_sitting"]
    if not selected:
        raise ValueError("compact protected evaluation has no independent GRAIL sitting clips")
    train_clips = training_bundle["clips"]
    groups = {clips[i]["source_group"] for i in selected}
    if groups & {c["source_group"] for c in train_clips}:
        raise ValueError("GRAIL evaluation source group overlaps training")
    assets = {a["asset_id"]: a for a in bundle["assets"]}
    train_assets = {a["asset_id"]: a for a in training_bundle["assets"]}
    chairs = sorted({clips[i]["asset_id"] for i in selected})
    eval_native = {assets[i]["native_equivalence_sha256"] for i in chairs}
    train_native = {train_assets[c["asset_id"]]["native_equivalence_sha256"] for c in train_clips}
    if eval_native & train_native:
        raise ValueError("GRAIL evaluation native chair overlaps training")
    per_chair = []
    for asset_id in chairs:
        members = [i for i in selected if clips[i]["asset_id"] == asset_id]
        per_chair.append({"asset_id": asset_id, "original_asset_id": assets[asset_id]["original_asset_id"],
                          "source_group_count": len({clips[i]["source_group"] for i in members}),
                          "aggregate": aggregate_evaluation_cases([indexed[i] for i in members])})
    return {"schema": "hoi_grail_sitting_eval_v1", "dataset": "grail_sitting",
            "mode": "held_out_chair_asset_and_source_group", "source_group_disjoint": True,
            "native_asset_disjoint": True, "human_template_independence": "not_established_by_published_metadata",
            "augmentation_count": 0, "chair_count": len(chairs), "source_group_count": len(groups),
            "coverage_complete": True, "manifest_clip_ids": [clips[i]["clip_id"] for i in selected],
            "evaluation_bundle_content_sha256": bundle["bundle_content_sha256"],
            "aggregate": aggregate_evaluation_cases([indexed[i] for i in selected]), "per_chair": per_chair}


def _validate_contact_contract(contact):
    from .networks import FEATURE_SCHEMA
    label = FEATURE_SCHEMA["actual_contact_label"]
    if (label["schema"] != LABEL_SCHEMA or tuple(FEATURE_SCHEMA["contact"]) != CHANNELS or
            contact.get("schema") != LABEL_SCHEMA or contact.get("threshold_N") != label["threshold_N"] or
            contact.get("num_substeps") != label["num_substeps"] or contact.get("substep_dt") != .005 or
            contact.get("interval") != "(t,t+1]" or
            tuple(contact.get("channels", ())) != CHANNELS):
        raise ValueError("evaluation actual-pair contact labels differ from the checkpoint feature contract")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="prepared protected_eval/eval80/test40 partition bundle")
    parser.add_argument("--training-bundle", type=Path, required=True, help="original checkpoint train bundle identity")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-mjcf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canonical-hoi80-bundle", type=Path,
                        help="historical fixed HOI80 registry; select matching protected clips, keeping current data/physics")
    from supermimic.physics import add_physics_arguments, resolve_arguments
    add_physics_arguments(parser)
    args, launcher = parser.parse_known_args(argv)
    checkpoint, checkpoint_sha, provenance = _load_evaluation_snapshot(args.checkpoint)
    variant = checkpoint_variant(checkpoint)
    from .variants import geometry_input_enabled, reward_config, max_step_reward
    settings, _ = resolve_arguments(args, checkpoint.get("physics"), require_legacy_explicit=True)
    training_bundle, header = _read_bundle_header(args.training_bundle), _read_bundle_header(args.bundle)
    validate_evaluation_identity(training_bundle, header)
    selection = None
    selected_ids = None
    if args.canonical_hoi80_bundle:
        from .canonical_hoi80 import select_canonical_hoi80
        selected_ids, selection = select_canonical_hoi80(training_bundle, header, args.canonical_hoi80_bundle)
        selected_ids = set(selected_ids)
    from .config import bundle_identity
    identity = bundle_identity(training_bundle)
    from supermimic.kinematics import G1Kinematics
    from .prepare import load_bundle
    from .mdp import HOIMDP
    from supermimic.robot_metadata import default_joint_pos_list
    from .workers import WorkerGroup, WorkerSpec, geometry_group
    import supermimic.networks as legacy_networks
    torch.manual_seed(args.seed)
    kin = G1Kinematics(args.device, mjcf_path=args.robot_mjcf)
    motion, shapes, bundle = load_bundle(args.bundle, kin, device=args.device, require_train=False, retain_source_reference=True)
    if "reference_joint_names" in bundle:
        joint_names = bundle["reference_joint_names"]
    else:
        manifest = read_manifest(bundle["manifest"]["path"])
        joint_names = manifest["bindings"]["target_joint_names"]
    fk39 = _load_fk39(bundle, joint_names, args.device)
    mdp = HOIMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), shapes, cfg=reward_config(variant))
    from .privileged import TEACHER_PROFILE, configure_teacher
    actor_profile = checkpoint.get("actor_observation_profile", "v1")
    if actor_profile == TEACHER_PROFILE:
        configure_teacher(mdp, bundle, args.device)
    policy, wm, critic = [model(geometry_input_enabled=geometry_input_enabled(variant),
                                **({"actor_observation_profile": actor_profile} if model is Policy else {})).to(args.device)
                          for model in (Policy, WorldModel, Critic)]
    load_checkpoint(checkpoint, policy, wm, critic, physics=asdict(settings), data_identity=identity, variant=variant)
    legacy_networks.AMP_DTYPE = None
    output = fresh_directory(args.output)
    started = time.monotonic()
    results, native = [], []
    for asset in bundle["assets"]:
        clip_ids = torch.where(motion.asset_ids[motion.clip_start] == int(asset["asset_id"]))[0].tolist()
        if selected_ids is not None:
            clip_ids = [clip for clip in clip_ids if clip in selected_ids]
        if not clip_ids:
            continue
        spec = WorkerSpec(int(asset["asset_id"]), len(clip_ids), len(clip_ids), geometry_group(asset))
        config = {"physics": asdict(settings), "output": str(output / f"native_{spec.asset_id:04d}"),
                  "bundle": str(args.bundle.resolve()), "device": args.device, "require_train": False,
                  "eval_mode": True, "reset_noise": False, "launcher_args": launcher,
                  "robot_mjcf": args.robot_mjcf, "seed": args.seed, "variant": variant}
        with torch.no_grad():
            workers = WorkerGroup([spec], config, device=args.device)
            try:
                _validate_contact_contract(workers.native_reports[0]["contact"])
                results.extend(evaluate_group(policy, motion, mdp, shapes, workers, clip_ids, fk39=fk39, wm=wm, variant=variant,
                                             joint_names=joint_names, output=output / f"asset_{spec.asset_id:04d}",
                                             metadata={"checkpoint_sha256": checkpoint_sha, "checkpoint_iteration": checkpoint["iteration"],
                                                       "evaluation_source_content_sha256": provenance["evaluation_source_identity"]["content_sha256"],
                                                       "checkpoint_training_source_content_sha256": provenance["checkpoint_training_source_identity"].get("content_sha256"),
                                                       "checkpoint_training_source_identity_status": provenance["checkpoint_training_source_identity_status"],
                                                       "checkpoint_training_progress": provenance["checkpoint_training_progress"],
                                                       "bundle_content_sha256": bundle["bundle_content_sha256"], "clips": bundle["clips"],
                                                       "physics": asdict(settings), "contact_schema": workers.native_reports[0]["contact"]}))
                native.extend(workers.native_reports)
            finally:
                workers.close()
    if sha256_file(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("checkpoint bytes changed during frozen evaluation")
    aggregate_values = aggregate_evaluation_cases(results)
    report = {"protocol": "HOI v1 full-reference oracle-pose policy evaluation; full-duration physical evaluation",
              "contact_channels": list(CHANNELS), "contact_label_schema": LABEL_SCHEMA,
              "partition": bundle["partition"], "scope": bundle["scope"], "checkpoint_sha256": checkpoint_sha,
              "variant": variant, "objective_max_step_reward": max_step_reward(variant), "common_hoi_comparison_step_reward": .15,
              "checkpoint_iteration": checkpoint["iteration"], "physics": asdict(settings), "cases": results,
              "aggregate": aggregate_values, "native": native, "optimizer_updates": 0, "normalization_updates": 0,
              "no_failure_reset": True, "physical_objects": True, "elapsed_seconds": time.monotonic() - started,
              "training_data_identity": identity, "evaluation_bundle_content_sha256": bundle["bundle_content_sha256"], **provenance}
    if selection is not None:
        if {case["clip_id"] for case in results} != selected_ids or len(results) != 80:
            raise RuntimeError("HOI80 evaluation did not cover exactly the selected cases")
        report["scope"] = selection
        report["selection"] = selection
    sitting = grail_sitting_evaluation(bundle, training_bundle, results) if selection is None else None
    if sitting is not None:
        report["grail_sitting_eval"] = sitting
    _verify_evaluation_source(provenance["evaluation_source_identity"])
    write_json(output / "evaluation.json", report)
    print(json.dumps({"evaluation": str(output / "evaluation.json"), "aggregate": report["aggregate"]}, allow_nan=False))


if __name__ == "__main__":
    main()
