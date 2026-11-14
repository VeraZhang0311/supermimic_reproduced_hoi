"""Evaluate the clips of one HOI80 asset together in independent native worlds."""
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch

from .evaluate import CONTROL_DT, MAX_STEP_REWARD, validate_nominal_snapshot
from .trajectories import BODY_COUNT, CONTROL_SHAPES, DT, _write_payload
from hoi_residual.features import object_observation, oracle_features, torso_pose
from supermimic.mdp import clamp_action, flatten_state
from supermimic.rotations import mat_to_quat, so3_log


def _native_frame(env, motion, kin, indices, state, obj):
    frame = env.f.clone()
    reference = motion.ref_state(frame)
    reference_object = motion.object_ref(frame)
    reference_pos, reference_rot, _, _ = kin.forward(reference)
    pose = env.robot.data.body_link_pose_w.torch.detach().clone()
    if pose.shape != (env.N, BODY_COUNT, 7):
        raise ValueError("HOI80 requires a native pose for every link in each world")

    def object_state(values):
        return torch.cat((values[0], mat_to_quat(values[1]), values[2], values[3]), dim=-1)

    return {
        "actual_pos": pose[..., :3] - env.scene.env_origins[:, None, :],
        "actual_quat": pose[..., [6, 3, 4, 5]],
        "reference_pos": reference_pos[:, indices],
        "reference_quat": mat_to_quat(reference_rot[:, indices]),
        "actual_object_pos": obj[0], "actual_object_quat": mat_to_quat(obj[1]),
        "reference_object_pos": reference_object[0], "reference_object_quat": mat_to_quat(reference_object[1]),
        "robot_state": flatten_state(state), "reference_robot_state": flatten_state(reference),
        "object_state": object_state(obj), "reference_object_state": object_state(reference_object),
        "reference_frame": frame,
    }


def _tracking_errors(mdp, motion, frame, actual, obj):
    reference, obj_ref = motion.ref_state(frame), motion.object_ref(frame)
    body_error = (mdp.kin.forward(*actual)[0] - motion.reference_bodies(frame)[0]).norm(dim=-1)
    return {
        "object_position_error_m": (obj[0] - obj_ref[0]).norm(dim=-1),
        "object_orientation_error_rad": so3_log(obj[1] @ obj_ref[1].transpose(-1, -2)).norm(dim=-1),
        "root_position_error_m": (actual[0] - reference[0]).norm(dim=-1),
        "root_orientation_error_rad": so3_log(actual[1] @ reference[1].transpose(-1, -2)).norm(dim=-1),
        "tracked_body_position_error_m": body_error.mean(dim=-1),
        "tracked_body_max_position_error_m": body_error.amax(dim=-1),
    }


def _stack(rows):
    return {name: torch.stack([row[name] for row in rows]) for name in rows[0]}


def _completed_lane_resets(lengths):
    """Recycle finished worlds on valid reference frames while other clips finish."""
    resets = defaultdict(list)
    for lane, length in enumerate(lengths):
        for tick in range(length, max(lengths), length):
            resets[tick].append(lane)
    return resets


@dataclass
class AssetRollout:
    frames: dict
    controls: dict
    lengths: list
    names: list
    initialization: dict

    def clip(self, lane, entry):
        steps = self.lengths[lane]
        frames = {name: value[:steps+1, lane] for name, value in self.frames.items()}
        controls = {name: value[:steps, lane].cpu().numpy() for name, value in self.controls.items()}
        start, end = int(frames["reference_frame"][0]), int(frames["reference_frame"][-1])
        if end - start != steps:
            raise RuntimeError("HOI80 clip did not reach its exact final reference frame")
        failed_at = np.flatnonzero(controls["failed"])
        first_failure = int(failed_at[0]) + 1 if len(failed_at) else None
        free_steps = steps if first_failure is None else first_failure - 1
        total = float(controls["reward"].sum(dtype=np.float64))
        errors = {name: {"mean": float(value.mean(dtype=np.float64)), "max": float(value.max())}
                  for name, value in controls.items() if name.endswith(("_error_m", "_error_rad"))}
        record = {
            "clip_id": entry["clip_id"], "source_group": entry["source_group"], "asset": entry["object_asset"],
            "partition": "protected_eval", "state_frames": steps+1, "control_steps": steps,
            "reference_control_steps": steps, "reference_duration_s": steps * CONTROL_DT,
            "start_frame": start, "final_frame": end, "reference_start_frame": start, "reference_end_frame": end,
            "full_reference_duration": True, "no_failure_reset": True, "reset_calls_during_rollout": 0,
            "failed": first_failure is not None, "first_failure_step": first_failure,
            "failure_free_steps": free_steps, "failure_free_seconds": free_steps * CONTROL_DT,
            "terminated_steps": len(failed_at), "robot_failure_steps": int(controls["robot_failed"].sum()),
            "object_failure_steps": int(controls["object_failed"].sum()), "return": total,
            "normalized_return": total / (steps * MAX_STEP_REWARD),
            "tracking_quality_fraction": float(controls["quality"].mean()),
            "tracking_quality_all_steps": bool(controls["quality"].all()),
            "action_clipped_fraction": float(controls["clipped"].mean()),
            "reward_terms_mean": {key.removeprefix("term:"): float(value.mean(dtype=np.float64))
                                  for key, value in controls.items() if key.startswith("term:")},
            **errors,
        }
        return record, frames, controls

    def save(self, output, frames, controls, record, identity, reference):
        arrays = {name: value.cpu().numpy() for name, value in frames.items()}
        arrays.update({name: controls[name] for name in CONTROL_SHAPES})
        arrays.update(time=np.arange(record["state_frames"], dtype=np.float64) * DT,
                      names=np.array(self.names), canonical_reference=np.asarray(reference))
        return _write_payload(output, arrays, record, identity, self.names)


@torch.no_grad()
def rollout_asset(env, motion, mdp, privileged, policy, kin, shape, asset, references):
    if motion.partition != "protected_eval" or motion.for_training or env.N != len(motion.entries):
        raise ValueError("Batch evaluation needs one world per protected clip")
    names = list(env.robot.body_names)
    if len(names) != BODY_COUNT or set(names) != set(kin.body_names):
        raise ValueError("HOI80 requires the original native 38-link order")
    indices = [kin.body_names.index(name) for name in names]
    starts = motion.clip_start
    lengths = (motion.clip_end - starts).tolist()
    if len(lengths) != env.N or any(length < 1 for length in lengths):
        raise ValueError("HOI80 clips require a complete control interval in each world")
    initial_rows = None
    if references is not None:
        from hoi_zero_shot.native import initialize_canonical_object
        initial_rows = np.stack([reference[0] for reference in references])

    def reset(lanes):
        ids = torch.as_tensor(lanes, device=env.device, dtype=torch.long)
        env.reset(ids, starts[ids])
        if initial_rows is None:
            env.refresh()
            return {}
        receipt = initialize_canonical_object(env, initial_rows[lanes], env_ids=ids)
        env.history.reset(ids, object_observation(torso_pose(kin, env.state()), env.object_state()))
        return receipt

    initialization = reset(list(range(env.N)))
    validate_nominal_snapshot(env, env.snapshot(), asset)
    episodes = env.episode_id.clone()
    lengths_tensor = motion.clip_end - starts
    resets = _completed_lane_resets(lengths)
    shape_embedding = policy.shape_encoder(shape).expand(env.N, -1)
    frames = [_native_frame(env, motion, kin, indices, env.state(), env.object_state())]
    controls = []
    consistent = torch.ones((), device=env.device, dtype=torch.bool)
    for tick in range(max(lengths)):
        if tick in resets:
            reset(resets[tick])
        snapshot = env.snapshot()
        state, obj = env.state(), env.object_state()
        robot = privileged(state, snapshot["f"], snapshot["a_prev"], snapshot["xi"],
                           snapshot["q_prev"], snapshot["qd_prev"], snapshot["actuator"])
        oracle = oracle_features(robot, torso_pose(kin, state), obj, motion, snapshot["f"],
                                 snapshot["xi_o"], snapshot["object_history"])
        raw = policy(oracle, shape, shape_embedding=shape_embedding)
        action = clamp_action(raw)
        result = env.step(action)
        state, obj = env.state(), env.object_state()
        robot_failed = mdp.termination(state, env.f)
        errors = _tracking_errors(mdp, motion, env.f, state, obj)
        object_failed = errors["object_position_error_m"] > env.reward_model.object_failure_threshold(env.f)
        consistent &= ((env.episode_id == episodes) | (tick >= lengths_tensor)).all()
        consistent &= (env.f == starts + (tick % lengths_tensor) + 1).all()
        consistent &= (result["terminated"] == (robot_failed | object_failed)).all()
        quality = ((errors["tracked_body_max_position_error_m"] < .5) & (errors["object_position_error_m"] < .1)
                   & (errors["object_orientation_error_rad"] < math.pi / 6))
        controls.append({"action": action, "reward": result["reward"], "failed": result["terminated"],
                         "contact": result["contact"], "contact_valid": result["contact_valid"],
                         "robot_failed": robot_failed, "object_failed": object_failed, "quality": quality,
                         "clipped": (action != raw).any(dim=-1), **errors,
                         **{"term:" + name: value for name, value in result["terms"].items()}})
        frames.append(_native_frame(env, motion, kin, indices, state, obj))
    if not bool(consistent):
        raise RuntimeError("HOI80 batch changed a clip's frame sequence, episode, or failure rule")
    validate_nominal_snapshot(env, env.snapshot(), asset)
    controls = _stack(controls)
    if any(not bool(torch.isfinite(value).all()) for value in controls.values()):
        raise FloatingPointError("HOI80 batch produced nonfinite control or tracking data")
    return AssetRollout(_stack(frames), controls, lengths, names, initialization)
