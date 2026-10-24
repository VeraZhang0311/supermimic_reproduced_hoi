"""HOI references on the canonical control grid, preserving the legacy human goal.

No CSV round trip or second resampling occurs.  The adapter already computed the
pose-derived velocities; the original 14 tracked-body FK and goal are reused.
"""
from __future__ import annotations

import torch

from supermimic.motion import BIN_FRAMES, MotionLib
from supermimic.rotations import mat_to_6d, yaw_mat


class TrainingMotionBank(MotionLib):
    def __init__(self, clips, kin, device="cpu", partition="train", retain_source_reference=None):
        if partition not in ("train", "eval80", "test40", "validation", "regression", "test"):
            raise ValueError("unknown reference partition")
        if not clips:
            raise ValueError("a reference bank requires at least one clip")
        self.partition, self.device = partition, str(device)
        self.names = [str(c.get("name", c.get("source_group", i))) for i, c in enumerate(clips)]
        self.provenance = [c.get("provenance", c.get("reference", c).get("provenance", {})) for c in clips]
        data = [c.get("reference", c) for c in clips]
        lengths = [len(d["robot"]["root_pos"]) for d in data]
        if min(lengths) < 2:
            raise ValueError("all clips require at least two control-grid states")
        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_start = torch.cat((self.clip_len.new_zeros(1), self.clip_len.cumsum(0)[:-1]))
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(clips), device=device), self.clip_len)
        self.num_frames = sum(lengths)
        cat = lambda values: torch.cat(values).to(device=device, dtype=torch.float32)
        keep_source = partition != "train" if retain_source_reference is None else retain_source_reference
        self.source_reference = cat([d["source_sampled"] for d in data]) if keep_source else None
        for attr, field in (("root_p", "root_pos"), ("root_R", "root_rot"),
                            ("root_v", "derived_ref_v"), ("root_w", "derived_ref_omega"),
                            ("q", "joint_pos"), ("qd", "derived_ref_joint_qd")):
            setattr(self, attr, cat([d["robot"][field] for d in data]))
        for attr, field in (("object_p", "pos"), ("object_R", "rot"),
                            ("object_v", "derived_ref_v"), ("object_w", "derived_ref_omega")):
            setattr(self, attr, cat([d["object"][field] for d in data]))
        self.asset_ids = torch.cat([torch.full((n,), int(c["asset_id"]), dtype=torch.long, device=device)
                                    for c, n in zip(clips, lengths)])
        self.asset_id = self.asset_ids
        expected, valid = [], []
        for c, n in zip(clips, lengths):
            e, m = torch.as_tensor(c["expected_contact"]), torch.as_tensor(c["expected_valid"])
            if e.shape != (n, 2) or m.shape != (n, 2):
                raise ValueError("expected schedule must be T+1 by two hand-object pairs")
            if not bool((((e == 0) | (e == 1)) & ((m == 0) | (m == 1))).all()):
                raise ValueError("expected contact and masks must be binary")
            expected.append(e * m)
            valid.append(m)
        self.expected_contact, self.expected_valid = cat(expected), cat(valid)
        with torch.no_grad():
            blocks = [kin.forward(*(x[start:start + 65536] for x in self.ref_arrays()))
                      for start in range(0, self.num_frames, 65536)]
            self.body_p, self.body_R, self.body_v, self.body_w = [torch.cat([b[i] for b in blocks]) for i in range(4)]
        if self.body_p.shape[-2] != 14:
            raise ValueError("legacy human reward requires the original 14-body FK")
        rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]
        starts, ids = [], []
        for clip, n in enumerate(lengths):
            # A reset bin must contain an outgoing control interval. The old
            # motion-only library also bins a terminal-only n%50==1 state;
            # this HOI bank fixes that boundary without changing the old entry.
            for offset in range(0, n - 1, BIN_FRAMES):
                starts.append(int(self.clip_start[clip]) + offset)
                ids.append(clip)
        self.bin_start = torch.tensor(starts, dtype=torch.long, device=device)
        self.bin_clip = torch.tensor(ids, dtype=torch.long, device=device)
        self.num_bins = len(starts)
        self.frame_bin = torch.empty(self.num_frames, dtype=torch.long, device=device)
        for b, (start, clip) in enumerate(zip(starts, ids)):
            self.frame_bin[start:min(start + BIN_FRAMES, int(self.clip_end[clip]) + 1)] = b
        for clip in range(len(clips)):
            last_bin = torch.nonzero(self.bin_clip == clip, as_tuple=False)[-1, 0]
            self.frame_bin[self.clip_end[clip]] = last_bin
        self.bin_asset = self.asset_ids[self.bin_start]
        if not all(bool(torch.isfinite(x).all()) for x in self.ref_arrays() + self.ref_object_arrays()):
            raise ValueError("non-finite reference")

    def ref_arrays(self):
        return self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd

    def ref_object_arrays(self):
        return self.object_p, self.object_R, self.object_v, self.object_w

    def ref_object(self, f):
        return tuple(x[f] for x in self.ref_object_arrays())

    def object_goal(self, S, O, f, K=10):
        """19D/future tick in the *current* actual/imagined robot heading.

        Beyond-end poses clamp exactly like the human goal.  Contact validity
        becomes zero outside the clip rather than inventing a held command.
        """
        fut = self.future_frames(f, K)
        ht = yaw_mat(S[1]).transpose(-1, -2).unsqueeze(-3)
        dp = (ht @ (self.object_p[fut] - S[0].unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(ht @ self.object_R[fut])
        v = (ht @ self.object_v[fut].unsqueeze(-1)).squeeze(-1)
        w = (ht @ self.object_w[fut].unsqueeze(-1)).squeeze(-1)
        raw_fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        in_clip = raw_fut <= self.clip_end[self.clip_id[f]].unsqueeze(-1)
        valid = self.expected_valid[fut] * in_clip.unsqueeze(-1)
        expected = self.expected_contact[fut] * valid
        return torch.cat((dp, rot, v, w, expected, valid), -1).flatten(-2)

    def expected(self, f):
        return self.expected_contact[f], self.expected_valid[f]

    def compatible_bins(self, asset_id):
        return torch.nonzero(self.bin_asset == asset_id, as_tuple=False).flatten()
