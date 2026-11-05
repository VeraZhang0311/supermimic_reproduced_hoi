"""Native BONES-SEED G1 CSV -> bounded-memory float32 50Hz reference library.

Preparation reads one CSV at a time; training stores only generalized states.
Body FK and rotation matrices are computed for the indexed minibatch, never
materialized for the entire dataset. No retargeting, heading alignment or crop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .motion import BIN_FRAMES, CONTROL_DT, MotionLib, _gradient
from .privileged import euler_xyz_quat
from .robot_metadata import G1_JOINT_NAMES
from .rotations import quat_slerp, quat_to_mat, so3_log

FORMAT = "bones_seed_g1_float32_50hz_v1"
DEFAULT_LIBRARY = str(Path(__file__).resolve().parents[1] / "data/boneseed_g1")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path, src_fps=120.):
    """Named cm / extrinsic xyz Euler degrees / joint degrees, not LAFAN CSV."""
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError("source fps must be finite and positive")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
        expected = ["Frame", *["root_translate"+a for a in "XYZ"],
                    *["root_rotate"+a for a in "XYZ"], *[n+"_dof" for n in G1_JOINT_NAMES]]
        if len(header) != len(set(header)) or set(header) != set(expected):
            raise ValueError(f"{path}: expected native BONES G1 named columns")
        raw = np.loadtxt(stream, delimiter=",", dtype=np.float64, ndmin=2)
    if raw.shape[1] != len(header) or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"{path}: expected >=2 finite frames with consistent columns")
    idx = {n: i for i, n in enumerate(header)}
    frame = raw[:, idx["Frame"]]
    if not np.allclose(frame, np.round(frame), atol=1e-8, rtol=0) or not np.allclose(np.diff(frame), 1, atol=1e-8, rtol=0):
        raise ValueError(f"{path}: nonconsecutive frame IDs")
    p = torch.tensor(raw[:, [idx["root_translate"+a] for a in "XYZ"]] / 100., dtype=torch.float32)
    quat = euler_xyz_quat(torch.tensor(np.deg2rad(raw[:, [idx["root_rotate"+a] for a in "XYZ"]]), dtype=torch.float32))
    q = torch.tensor(np.deg2rad(raw[:, [idx[n+"_dof"] for n in G1_JOINT_NAMES]]), dtype=torch.float32)
    count = int(np.floor((len(raw)-1) / src_fps / CONTROL_DT + 1e-8)) + 1
    if count < 2:
        raise ValueError(f"{path}: less than one complete 50Hz transition")
    t = torch.arange(count, dtype=torch.float32) * CONTROL_DT * src_fps
    i = t.floor().long().clamp(max=len(raw)-2)
    a = (t-i).clamp(0, 1)
    p = p[i]*(1-a[:, None]) + p[i+1]*a[:, None]
    q = q[i]*(1-a[:, None]) + q[i+1]*a[:, None]
    quat = quat_slerp(quat[i], quat[i+1], a)
    R = quat_to_mat(quat)
    w = torch.zeros_like(p)
    w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2*CONTROL_DT)
    w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
    w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
    state = torch.cat((p, quat, _gradient(p, CONTROL_DT), w, q, _gradient(q, CONTROL_DT)), -1)
    if not torch.isfinite(state).all():
        raise ValueError(f"{path}: nonfinite converted state")
    return state


def prepare(source, output, src_fps=120., selection=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    files = sorted(source.rglob("*.csv"))
    if not files:
        raise ValueError("no BONES CSV files found (pass the extracted g1/csv directory)")
    selection_hash = None
    if selection is not None:
        selection_hash = sha256(selection)
        selected = json.loads(Path(selection).read_text(encoding="utf-8"))
        if not isinstance(selected, list) or not selected or any(not isinstance(x, str) for x in selected) or len(set(selected)) != len(selected):
            raise ValueError("selection must be a nonempty JSON list of unique source-relative CSV paths")
        by_name = {f.relative_to(source).as_posix(): f for f in files}
        if set(selected) - set(by_name):
            raise ValueError("selection names missing from source directory")
        files = [by_name[n] for n in sorted(selected)]
    # Refuse an existing destination; incomplete preparation never has a manifest.
    output.mkdir(parents=True, exist_ok=False)
    clips, digest, count = [], hashlib.sha256(), 0
    with (output / "states.bin").open("xb") as stream:
        for index, path in enumerate(files):
            before = sha256(path)
            state = load_csv(path, src_fps)
            if sha256(path) != before:
                raise ValueError(f"source changed during preparation: {path}")
            payload = state.numpy().astype("<f4", copy=False).tobytes()
            stream.write(payload)
            digest.update(payload)
            clips.append({"name": path.relative_to(source).as_posix(), "frames": len(state), "source_sha256": before})
            count += len(state)
            if (index+1) % 1000 == 0:
                print(f"prepared {index+1}/{len(files)} clips, {count} frames", flush=True)
    manifest = {"format": FORMAT, "dataset": "bones-studio/seed", "control_dt": CONTROL_DT,
                "source_fps": src_fps, "source_position_unit": "cm", "source_angles": "extrinsic_xyz_degrees",
                "joint_names": G1_JOINT_NAMES, "state_layout": "p3,quat_wxyz4,v3,w3,q29,qd29",
                "dtype": "<f4", "frames": count, "clips": clips, "states_sha256": digest.hexdigest(),
                "selection": "all_supplied_csvs" if selection is None else "explicit_path_list",
                "selection_sha256": selection_hash, "exact_paper_selection": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


class Indexed:
    """Tensor-like indexed derived reference arrays (keeps existing MDP formulas)."""
    def __init__(self, compute, count):
        self.compute, self.count = compute, count

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.compute(index[0])[(..., *index[1:])]
        return self.compute(index)

    def cpu(self):
        return self.compute(slice(None)).cpu()


class BoneseedMotionLib(MotionLib):
    def __init__(self, path, kin, device):
        path = Path(path)
        manifest_path = path / "manifest.json"
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (m["format"] != FORMAT or m["control_dt"] != CONTROL_DT or m["joint_names"] != G1_JOINT_NAMES
                or m["dtype"] != "<f4" or m["state_layout"] != "p3,quat_wxyz4,v3,w3,q29,qd29"):
            raise ValueError("incompatible BONES reference manifest")
        lengths = [c["frames"] for c in m["clips"]]
        self.names = [c["name"] for c in m["clips"]]
        if not lengths or any(type(n) is not int or n < 2 for n in lengths) or len(set(self.names)) != len(lengths):
            raise ValueError("invalid BONES clip lengths/identities")
        self.num_frames = sum(lengths)
        states = path / "states.bin"
        if (self.num_frames != m["frames"] or states.stat().st_size != self.num_frames*71*4
                or sha256(states) != m["states_sha256"]):
            raise ValueError("BONES reference state size/hash mismatch")
        self.manifest_sha256 = sha256(manifest_path)
        self.device = device
        self.dataset = "boneseed"
        self.kin = kin
        self._mapped = np.memmap(states, dtype="<f4", mode="c", shape=(self.num_frames, 71))
        self.states = torch.from_numpy(self._mapped).to(device)
        self.root_p, self.root_v, self.root_w = self.states[:, :3], self.states[:, 7:10], self.states[:, 10:13]
        self.q, self.qd = self.states[:, 13:42], self.states[:, 42:71]
        self.root_R = Indexed(lambda f: quat_to_mat(self.states[f, 3:7]), self.num_frames)
        self.root_gz = Indexed(lambda f: self.root_R[f][..., 2, 2], self.num_frames)
        self.root_v_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_v[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        self.root_w_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_w[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        for name, j in (("body_p", 0), ("body_R", 1), ("body_v", 2), ("body_w", 3)):
            setattr(self, name, Indexed(lambda f, j=j: kin.forward(*self.ref_state(f))[j], self.num_frames))
        self.clip_len = torch.tensor(lengths, device=device, dtype=torch.long)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lengths), device=device), self.clip_len)
        # Vectorized indexing avoids a Python loop over every 1s bin.
        # Sample transition starts, never an isolated last frame (e.g. 51 frames).
        nb = (self.clip_len - 1 + BIN_FRAMES - 1) // BIN_FRAMES
        bin_base = nb.cumsum(0) - nb
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lengths), device=device), nb)
        offset = torch.arange(int(nb.sum()), device=device) - bin_base[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + BIN_FRAMES*offset
        self.num_bins = len(self.bin_start)
        frame = torch.arange(self.num_frames, device=device)
        local_bin = torch.minimum((frame - self.clip_start[self.clip_id]) // BIN_FRAMES, nb[self.clip_id]-1)
        self.frame_bin = bin_base[self.clip_id] + local_bin

    def reference_bodies(self, f):
        return self.kin.forward(*self.ref_state(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="extracted native g1/csv tree")
    parser.add_argument("--output", required=True, help="new library directory")
    parser.add_argument("--source-fps", type=float, default=120.)
    parser.add_argument("--selection", help="optional frozen JSON list of CSV paths relative to source")
    args = parser.parse_args()
    torch.set_num_threads(3)
    m = prepare(args.source, args.output, args.source_fps, args.selection)
    print(f"{len(m['clips'])} clips, {m['frames']} frames; exact_paper_selection=False")


if __name__ == "__main__":
    main()
