"""Terminal reset/readback audit; never advances physics or resumes collection."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import torch
from .scene_supports import SUPPORT_IDS, validate_native_scene_supports

SCHEMA = "hoi_native_terminal_scene_reset_audit_v1"
PHASE = "terminal_auxiliary_after_smoke_state_discarded"
SELECTION_RULE = "every_asset_clip_first_middle_last_and_presence_change_neighbors_v1"
TOLERANCE = 2e-5
BINDING_KEYS = ("support_id", "binding_sha256", "source_support_sha256", "target_timebase_sha256")


def reset_sample_plan(motion, asset_id):
    """Validate every bound clip, then select named samples without resampling time."""
    validate_native_scene_supports(motion)
    selected = [i for i, start in enumerate(motion.clip_start) if int(motion.asset_ids[start]) == int(asset_id)]
    if not selected:
        raise ValueError("terminal scene audit requires at least one asset clip")
    if len(set(motion.names[i] for i in selected)) != len(selected):
        raise ValueError("terminal scene audit requires unique clip names")
    records = getattr(motion, "bound_scene_supports", [[] for _ in motion.names])
    clips, samples = [], []
    for i in selected:
        n, start = int(motion.clip_len[i]), int(motion.clip_start[i])
        if n < 2:
            raise ValueError("terminal scene audit requires two or more clip states")
        present = torch.zeros(n, len(SUPPORT_IDS), dtype=torch.bool)
        for record in records[i]:
            position = torch.as_tensor(record["position"], device="cpu")
            present[:, SUPPORT_IDS.index(record["support_id"])] = position[:, 2] > -5.
        changes = (torch.nonzero((present[1:] != present[:-1]).any(-1), as_tuple=False).flatten() + 1).tolist()
        local = sorted({0, n // 2, n - 1, *changes, *(i - 1 for i in changes)})
        clips.append({"name": motion.names[i], "clip_index": i, "state_frames": n,
            "selected_local_frames": local, "presence_change_frames": changes,
            "bindings": [{key: record[key] for key in BINDING_KEYS} for record in records[i]]})
        samples.extend({"clip_index": i, "clip_name": motion.names[i], "local_frame": f, "frame": start + f} for f in local)
    return clips, samples


def _expected(motion, samples, declared, device):
    # Build independently from bound clip records, not support_bank.poses or
    # support_bank.sample: the audit can therefore detect incorrect indexing.
    pose = torch.zeros(len(samples), len(SUPPORT_IDS), 7, device=device)
    pose[..., 6] = 1.
    present = torch.zeros(len(samples), len(SUPPORT_IDS), device=device, dtype=torch.bool)
    records = getattr(motion, "bound_scene_supports", [[] for _ in motion.names])
    for slot, sid in enumerate(SUPPORT_IDS):
        if sid not in declared:
            continue
        pose[:, slot, 2] = -10. if sid == "table1" else -5.
        for row, sample in enumerate(samples):
            matching = [r for r in records[sample["clip_index"]] if r["support_id"] == sid]
            if not matching:
                continue
            record, f = matching[0], sample["local_frame"]
            pose[row, slot, :3] = torch.as_tensor(record["position"][f], device=device, dtype=pose.dtype)
            q = torch.as_tensor(record["quaternion_wxyz"][f], device=device, dtype=pose.dtype)
            pose[row, slot, 3:] = q[[1, 2, 3, 0]]
            present[row, slot] = pose[row, slot, 2] > -5.
    return pose, present


@contextmanager
def _guard_method(obj, name, counter, key):
    marker = object()
    previous = obj.__dict__.get(name, marker)
    def forbidden(*args, **kwargs):
        counter[key] += 1
        raise RuntimeError("terminal scene reset audit prohibits " + name)
    setattr(obj, name, forbidden)
    try:
        yield
    finally:
        if previous is marker:
            delattr(obj, name)
        else:
            setattr(obj, name, previous)


def audit_scene_resets(env, output):
    """After all smoke results are fixed: reset, read back, then discard state.

    env.refresh may propagate written kinematics; this is not a renderer, a
    contact test or a rollout. Simulator and underlying solver integration are
    explicitly blocked, and their existing step counters must stay unchanged.
    """
    if env.env.reset_noise is not False or env.__dict__.get("cooking_only", False):
        raise ValueError("terminal scene reset audit requires reset_noise=False and standard native environment")
    if env.__dict__.get("active_pair_probe") is not None or "_post_advance_capture" in env.__dict__:
        raise ValueError("terminal scene reset audit cannot run before probes/visual capture finish")
    path = Path(output)
    if path.exists():
        raise FileExistsError("terminal scene audit requires a fresh artifact")
    sim, recorder = env.env.sim, env.contacts
    if recorder.manager._solver is not recorder.adapter:
        raise RuntimeError("terminal scene audit requires the installed recorder solver adapter")
    before = int(sim.get_physics_step_count())
    substeps = int(recorder.calls)
    clips, requests = reset_sample_plan(env.motion, int(env.asset["asset_id"]))
    declared = list(env.supports)
    required = {b["support_id"] for c in clips for b in c["bindings"]}
    if set(declared) != required:
        raise ValueError("terminal reset audit scene does not load the complete asset support bank")
    counters = {"attempted_advance_calls": 0, "attempted_sim_step_calls": 0, "attempted_solver_step_calls": 0}
    result = {"schema": SCHEMA, "enabled": True, "passed": False, "phase": PHASE,
        "selection_rule": SELECTION_RULE, "asset_id": int(env.asset["asset_id"]), "num_envs": env.N,
        "clip_names": [c["name"] for c in clips], "clips": clips, "declared_support_ids": declared,
        "requested_sample_count": len(requests), "completed_sample_count": 0, "reset_batches": 0,
        "reset_noise": False, "state_discarded": True, "physics_steps_before": before,
        "recorder_substeps_before": substeps, "maximum_errors": {"position_m": 0., "quaternion": 0.,
            "linear_velocity_m_s": 0., "angular_velocity_rad_s": 0.}, "presence_mismatches": 0,
        "scope": "all asset clips; first/middle/last and presence-boundary neighbors only; terminal root-state readback, not all-frame physical tracking or geom-cache refresh",
        "rng_scope": "terminal resets may resample PD offsets; no subsequent physics/network/collection; state discarded",
        "formal_training_started": False, "samples": []}
    error = None
    try:
        with ExitStack() as guards:
            guards.enter_context(_guard_method(env, "_advance", counters, "attempted_advance_calls"))
            guards.enter_context(_guard_method(sim, "step", counters, "attempted_sim_step_calls"))
            guards.enter_context(_guard_method(recorder.adapter, "step", counters, "attempted_solver_step_calls"))
            for start in range(0, len(requests), env.N):
                chunk = requests[start:start + env.N]
                ids = torch.arange(len(chunk), device=env.device)
                frames = torch.tensor([r["frame"] for r in chunk], device=env.device, dtype=torch.long)
                expected, expected_present = _expected(env.motion, chunk, declared, env.device)
                env.reset(ids, frames)
                observed = env.support_state()
                actual = observed["support_pose_xyzw"][ids]
                held = observed["reference_support_pose_xyzw"][ids]
                present = observed["support_present"][ids]
                velocity = torch.zeros(len(chunk), len(SUPPORT_IDS), 6, device=env.device)
                for sid, support in env.supports.items():
                    velocity[:, SUPPORT_IDS.index(sid)] = support.data.root_link_vel_w.torch[ids]
                if not bool(torch.isfinite(actual).all() and torch.isfinite(held).all() and torch.isfinite(velocity).all()):
                    raise RuntimeError("terminal scene reset audit found nonfinite readback")
                errors = {"position_m": float(torch.maximum((actual[..., :3] - expected[..., :3]).abs().max(),
                            (held[..., :3] - expected[..., :3]).abs().max())),
                    "quaternion": max(float(torch.minimum((x[..., 3:] - expected[..., 3:]).abs().amax(-1),
                            (x[..., 3:] + expected[..., 3:]).abs().amax(-1)).max()) for x in (actual, held)),
                    "linear_velocity_m_s": float(velocity[..., :3].abs().max()),
                    "angular_velocity_rad_s": float(velocity[..., 3:].abs().max())}
                mismatch = int((present.bool() != expected_present).sum())
                for key, value in errors.items(): result["maximum_errors"][key] = max(result["maximum_errors"][key], value)
                result["presence_mismatches"] += mismatch
                values = [x.detach().cpu().tolist() for x in (expected, expected_present, actual, held, present, velocity)]
                for i, request in enumerate(chunk):
                    result["samples"].append({**request, "environment_id": i, **dict(zip(
                        ("expected_pose_xyzw", "expected_present", "actual_pose_xyzw", "held_pose_xyzw", "actual_present", "actual_velocity"),
                        [v[i] for v in values]))})
                result["reset_batches"] += 1
                if max(errors.values()) > TOLERANCE or mismatch:
                    raise RuntimeError(f"terminal scene reset readback differs from bound reference: {errors}, mask={mismatch}")
                result["completed_sample_count"] += len(chunk)
    except BaseException as exc:
        error = exc
    result.update(counters)
    result.update(physics_steps_after=int(sim.get_physics_step_count()), recorder_substeps_after=int(recorder.calls))
    result["additional_physics_steps"] = result["physics_steps_after"] - before
    if result["additional_physics_steps"] != 0 or result["recorder_substeps_after"] != substeps or any(counters.values()):
        error = error or RuntimeError("terminal reset audit changed/attempted physics integration")
    result["passed"] = error is None and result["completed_sample_count"] == len(requests)
    if error is not None: result["error"] = {"type": type(error).__name__, "message": str(error)}
    with path.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2)
    if error is not None: raise error
    header = {k: v for k, v in result.items() if k != "samples"}
    return {**header, "artifact": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
