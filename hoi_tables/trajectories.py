"""Versioned native-pose recordings for offline display; no policy or physics replay.

Import remains stdlib-only so the CPU owning-run monitor can verify manifests.
Actual robot links come directly from the native articulation. Reference hw3
poses are computed while recording, never reconstructed by the video renderer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SCHEMA = "hoi_native_pose_trajectory_v1"
DT = .02
FPS = 50
BODY_COUNT = 38
STATE_SHAPES = {
    "actual_pos": (38, 3), "actual_quat": (38, 4),
    "reference_pos": (38, 3), "reference_quat": (38, 4),
    "actual_object_pos": (3,), "actual_object_quat": (4,),
    "reference_object_pos": (3,), "reference_object_quat": (4,),
    "robot_state": (71,), "reference_robot_state": (71,),
    "object_state": (13,), "reference_object_state": (13,), "reference_frame": (),
}
CONTROL_SHAPES = {"action": (29,), "reward": (), "failed": (),
                  "contact": (5,), "contact_valid": (5,)}
IDENTITY_FIELDS = ("iteration", "parent_run_id", "checkpoint_sha256", "model_state_sha256")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_check(arrays, meta):
    import numpy as np
    from supermimic.robot_metadata import G1_BODY_NAMES
    n, steps = meta["state_frames"], meta["control_steps"]
    if type(n) is not int or type(steps) is not int or steps < 1 or n != steps + 1:
        raise ValueError("Trajectory requires every T+1 state and T actions")
    required = {*STATE_SHAPES, *CONTROL_SHAPES, "time", "names", "canonical_reference"}
    if set(arrays) != required:
        raise ValueError("Trajectory payload field schema differs")
    for fields, length in ((STATE_SHAPES, n), (CONTROL_SHAPES, steps)):
        for name, tail in fields.items():
            value = arrays[name]
            if value.shape != (length, *tail) or not np.isfinite(value).all():
                raise ValueError("Trajectory shape/nonfinite field: " + name)
    names = list(map(str, arrays["names"]))
    if len(names) != BODY_COUNT or set(names) != set(G1_BODY_NAMES) or names != meta["body_names"]:
        raise ValueError("Trajectory must bind all38 native body names in recorded order")
    if arrays["time"].shape != (n,) or not np.allclose(arrays["time"], np.arange(n) * DT, rtol=0, atol=1e-10):
        raise ValueError("Trajectory clock differs from the complete50Hz grid")
    expected = np.arange(meta["reference_start_frame"], meta["reference_end_frame"] + 1)
    if not np.array_equal(arrays["reference_frame"], expected) or len(expected) != n:
        raise ValueError("Trajectory reference frames were reset, skipped or truncated")
    for name in ("actual_quat", "reference_quat", "actual_object_quat", "reference_object_quat"):
        if not np.allclose(np.linalg.norm(arrays[name], axis=-1), 1., rtol=0, atol=1e-3):
            raise ValueError("Trajectory has a nonunit wxyz quaternion: " + name)
    for name in ("robot_state", "reference_robot_state", "object_state", "reference_object_state"):
        if not np.allclose(np.linalg.norm(arrays[name][:, 3:7], axis=-1), 1., rtol=0, atol=1e-3):
            raise ValueError("Trajectory root state has a nonunit wxyz quaternion: " + name)
    for prefix, state_name in (("actual", "robot_state"), ("reference", "reference_robot_state")):
        if not np.allclose(arrays[prefix + "_pos"][:, names.index("pelvis")], arrays[state_name][:, :3], rtol=0, atol=2e-5):
            raise ValueError("Trajectory native/reference link origin differs from its root state")
    # Preserve all pinned source columns, including any source unknown flags.
    ref = arrays["canonical_reference"]
    if ref.ndim != 2 or ref.shape[0] != n or ref.shape[1] != meta["canonical_reference_width"]:
        raise ValueError("Trajectory canonical reference lost source frames/columns")


def _write_payload(worker_output, arrays, clip, identity, body_names):
    """Publish the manifest last, after a complete, validated compressed payload."""
    import numpy as np
    root = Path(worker_output).resolve()
    index = clip["canonical_index"]
    if type(index) is not int or index not in range(80):
        raise ValueError("Trajectory requires the original canonical case index")
    meta = {"schema": SCHEMA, "clip_id": clip["clip_id"], "asset": clip["asset"],
            "canonical_index": index, "canonical_case_id": clip["canonical_case_id"],
            "state_frames": clip["state_frames"], "control_steps": clip["control_steps"],
            "reference_start_frame": clip["reference_start_frame"],
            "reference_end_frame": clip["reference_end_frame"],
            "dt": DT, "fps": FPS, "body_count": BODY_COUNT, "body_names": list(body_names),
            "quaternion_order": "wxyz", "coordinates": "common world minus environment origin; no alignment",
            "robot_state_layout": "p3,quat_wxyz4,world_linear_velocity3,world_angular_velocity3,joint_pos29,joint_vel29",
            "object_state_layout": "p3,quat_wxyz4,world_linear_velocity3,world_angular_velocity3",
            "actual_pose_source": "robot.data.body_link_pose_w; native link frames, xyzw reordered to wxyz",
            "reference_pose_source": "frozen hw3 teacher kinematics at recording time",
            "actual_pose_fk_reconstruction": False, "physics_replay_required": False,
            "canonical_reference_width": int(arrays["canonical_reference"].shape[-1]),
            "canonical_reference_layout": "unmodified pinned PHC39 source columns with original xyzw quaternions; separate from native38 poses",
            "identity": identity, "metrics": clip}
    _payload_check(arrays, meta)
    directory = root / "trajectories"
    directory.mkdir(exist_ok=True)
    path = directory / f"case_{index:05d}.npz"
    manifest = path.with_suffix(".json")
    if path.exists() or manifest.exists():
        raise FileExistsError("Preserve existing trajectory artifacts; use a fresh evaluation output")
    temporary = path.with_suffix(".npz.part")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    pointer = {"schema": SCHEMA, "path": path.relative_to(root.parent).as_posix(),
               "sha256": sha256(path), "metadata_path": manifest.relative_to(root.parent).as_posix(),
               "state_frames": meta["state_frames"], "control_steps": meta["control_steps"],
               "body_count": BODY_COUNT, "dt": DT, "fps": FPS}
    meta["payload"] = {"path": pointer["path"], "sha256": pointer["sha256"]}
    pending = manifest.with_suffix(".json.part")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(manifest)
    pointer["metadata_sha256"] = sha256(manifest)
    return pointer


class NativeTrajectory:
    """Capture only reads simulator state; existing rollout still owns all steps."""
    def __init__(self, motion, teacher_kin):
        self.motion, self.kin = motion, teacher_kin
        self.frames, self.controls, self.body_names = [], [], None

    def capture(self, env, state, obj):
        import torch
        from supermimic.mdp import flatten_state
        from supermimic.rotations import mat_to_quat
        if env.N != 1 or len(self.frames) != len(self.controls):
            raise ValueError("Native recording requires one environment and one state per interval boundary")
        with torch.no_grad():
            names = list(env.robot.body_names)
            if len(names) != BODY_COUNT or set(names) != set(self.kin.body_names):
                raise ValueError("Native recording requires the exact38-link hw3 skeleton")
            if self.body_names is not None and names != self.body_names:
                raise ValueError("Native body order changed during recording")
            self.body_names = names
            indices = [self.kin.body_names.index(name) for name in names]
            pose = env.robot.data.body_link_pose_w.torch.detach().clone()
            if pose.shape != (1, BODY_COUNT, 7):
                raise ValueError("Native articulation pose shape differs")
            positions = pose[0, :, :3] - env.scene.env_origins[0]
            if not torch.allclose(positions[names.index("pelvis")], state[0][0], atol=2e-5, rtol=0):
                raise ValueError("Native body frames and actual root state disagree")
            reference = self.motion.ref_state(env.f)
            ref_object = self.motion.object_ref(env.f)
            rp, rR, _, _ = self.kin.forward(reference)
            pack_object = lambda value: torch.cat((value[0], mat_to_quat(value[1]), value[2], value[3]), -1)[0]
            values = {"actual_pos": positions, "actual_quat": pose[0, :, [6, 3, 4, 5]],
                      "reference_pos": rp[0, indices], "reference_quat": mat_to_quat(rR[0, indices]),
                      "actual_object_pos": obj[0][0], "actual_object_quat": mat_to_quat(obj[1])[0],
                      "reference_object_pos": ref_object[0][0],
                      "reference_object_quat": mat_to_quat(ref_object[1])[0],
                      "robot_state": flatten_state(state)[0], "reference_robot_state": flatten_state(reference)[0],
                      "object_state": pack_object(obj), "reference_object_state": pack_object(ref_object),
                      "reference_frame": env.f[0]}
            self.frames.append({name: value.detach().cpu().numpy().copy() for name, value in values.items()})

    def control(self, action, result):
        if len(self.frames) != len(self.controls) + 1:
            raise ValueError("Trajectory action has no unique preceding initial state")
        values = {"action": action[0], "reward": result["reward"][0], "failed": result["terminated"][0],
                  "contact": result["contact"][0], "contact_valid": result["contact_valid"][0]}
        self.controls.append({name: value.detach().cpu().numpy().copy() for name, value in values.items()})

    def save(self, worker_output, clip, identity, canonical_reference):
        import numpy as np
        if len(self.frames) != clip["state_frames"] or len(self.controls) != clip["control_steps"]:
            raise ValueError("Trajectory did not cover the complete scored clip")
        arrays = {name: np.stack([frame[name] for frame in self.frames]) for name in STATE_SHAPES}
        arrays.update({name: np.stack([step[name] for step in self.controls]) for name in CONTROL_SHAPES})
        arrays.update(time=np.arange(len(self.frames), dtype=np.float64) * DT,
                      names=np.array(self.body_names), canonical_reference=np.asarray(canonical_reference))
        return _write_payload(worker_output, arrays, clip, identity, self.body_names)


def validate_trajectory(evaluation_root, clip, identity, *, verify_payload=False):
    """Bind one artifact to its exact checkpoint/clip; never silently skip missing data."""
    root = Path(evaluation_root).resolve()
    pointer = clip.get("trajectory", {})
    if not isinstance(pointer, dict) or pointer.get("schema") != SCHEMA:
        raise ValueError("HOI80 clip lacks its required native trajectory")
    paths = {}
    for name in ("path", "metadata_path"):
        relative = pointer.get(name)
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Trajectory artifact paths must be relative to the evaluation")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Trajectory file is missing or escapes its evaluation")
        paths[name] = target
    for name, field in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
        if not re.fullmatch(r"[a-f0-9]{64}", str(pointer.get(field, ""))) or sha256(paths[name]) != pointer[field]:
            raise ValueError("Trajectory artifact checksum differs")
    meta = json.loads(paths["metadata_path"].read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("payload") != {"path": pointer["path"], "sha256": pointer["sha256"]}:
        raise ValueError("Trajectory metadata/payload binding differs")
    for name in ("clip_id", "asset", "canonical_case_id", "canonical_index", "state_frames", "control_steps",
                 "reference_start_frame", "reference_end_frame"):
        if meta.get(name) != clip[name]:
            raise ValueError("Trajectory clip identity or full duration differs: " + name)
    if any(meta.get("identity", {}).get(name) != identity[name] for name in IDENTITY_FIELDS):
        raise ValueError("Trajectory checkpoint/run identity differs")
    for name, expected in (("state_frames", clip["state_frames"]), ("control_steps", clip["control_steps"]),
                           ("body_count", BODY_COUNT), ("dt", DT), ("fps", FPS)):
        if pointer.get(name) != expected or meta.get(name) != expected:
            raise ValueError("Trajectory coverage/clock differs: " + name)
    if (meta.get("quaternion_order") != "wxyz" or len(meta.get("body_names", [])) != BODY_COUNT
            or len(set(meta["body_names"])) != BODY_COUNT or meta.get("actual_pose_fk_reconstruction") is not False):
        raise ValueError("Trajectory native rendering frame contract differs")
    if verify_payload:
        import numpy as np
        with np.load(paths["path"], allow_pickle=False) as loaded:
            _payload_check(dict(loaded), meta)
    return pointer
