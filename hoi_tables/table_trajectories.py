"""Save and verify native 50 Hz trajectories from the scored tables20 rollout."""
from pathlib import Path

from .trajectories import CONTROL_SHAPES, DT, STATE_SHAPES, _payload_check, sha256


def trajectory_path(root, clip):
    return Path(root) / clip["asset"] / (Path(clip["clip_id"]).name + ".npz")


def save_trajectory(asset_output, clip, frames, controls, body_names):
    import numpy as np

    arrays = {name: frames[name].cpu().numpy() for name in STATE_SHAPES}
    arrays.update({name: controls[name] for name in CONTROL_SHAPES})
    arrays.update(names=np.array(body_names), time=np.arange(clip["state_frames"], dtype=np.float64) * DT)
    path = trajectory_path(Path(asset_output).parent, clip)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return sha256(path)


def verify_trajectory(root, clip, *, verify_payload=False):
    path = trajectory_path(root, clip)
    if sha256(path) != clip["trajectory_sha256"]:
        raise ValueError("Table trajectory checksum differs")
    if verify_payload:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            arrays = dict(payload)
        # The shared native-pose validator also accepts an empty source
        # reference matrix for clips that have no HOI80 canonical columns.
        arrays["canonical_reference"] = np.empty((clip["state_frames"], 0))
        _payload_check(arrays, {**clip, "body_names": list(map(str, arrays["names"])), "canonical_reference_width": 0})
        failure_steps = np.flatnonzero(arrays["failed"])
        first_failure = int(failure_steps[0]) + 1 if len(failure_steps) else None
        if (first_failure != clip["first_failure_step"] or bool(len(failure_steps)) != clip["failed"]
                or float(arrays["reward"].sum(dtype=np.float64)) != clip["return"]):
            raise ValueError("Table trajectory and scored clip outcomes differ")
    return path
