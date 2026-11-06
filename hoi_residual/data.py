"""Frozen ULTRA source-group splits and a motion library for the hw3 GMT.

The original 630-column files remain read-only. Robot and object pose share the
canonical 60 Hz interpretation and are sampled on the same integer 50 Hz grid.
The published files' generating clock is not known; 60 Hz is a protocol choice.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import torch

from supermimic.robot_metadata import G1_JOINT_NAMES
from supermimic.rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT, BIN_FRAMES = 0.02, 50

SCHEMA = "ultra_hw3_residual_source_groups_v1"
CANONICAL_SPLITS_SHA256 = "7d66cd15b27b41988ac3ef761fb82f15204b5ab741a696f95102801dac8b62e2"
CANONICAL_MEMBERSHIP_SHA256 = "4df03e6b8d3b12a39b4361eb0c740b13b49af836007476872c910098cc26db43"
CANONICAL_BINDINGS_SHA256 = "e71c9160a817effd023145b697a4bf97ca396b81bb748178726f894da3a88073"
CANONICAL_COUNTS = {"train": (2651, 381), "validation": (517, 80),
                    "regression": (655, 100), "test": (243, 40)}
ASSETS = tuple(sorted(f"{obj}_{scale}_{scale}_{scale}" for obj in
                      ("largebox", "smallbox", "plasticbox", "suitcase") for scale in ("080", "100")))
SOURCE_ROOT = "/mnt/projects/ll/tianzizh/ProtoMotions/data/omomo/OMOMO_retarget_aug_v4"
TIMEBASE = {"mode": "canonical_60hz", "source_fps_assumed": 60,
            "source_timebase_status": "protocol_assumed_not_generation_verified",
            "control_dt": CONTROL_DT, "state_grid": "tick/50; source_index=6*tick/5",
            "velocity": "pose_derived_at_50hz_within_each_clip",
            "source_contact_semantics": "39_source_body_flags_not_pair_contact_truth"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source_group(path):
    return "_".join(PurePosixPath(path.replace("\\", "/")).stem.split("_")[:3])


def _validate_partitions(partitions, canonical=True):
    if set(partitions) != set(CANONICAL_COUNTS):
        raise ValueError("frozen split must contain train, validation, regression and test")
    seen_paths, seen_hashes, groups = set(), set(), {}
    summary = {}
    for partition, entries in partitions.items():
        if not entries:
            raise ValueError("empty frozen partition")
        local_groups = set()
        for entry in entries:
            path, digest = entry["motion"], entry["motion_sha256"]
            group = entry["source_group"]
            if group != _source_group(path):
                raise ValueError("source_group differs from filename identity")
            if ".." in PurePosixPath(path.replace("\\", "/")).parts:
                raise ValueError("parent traversal in source path")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid source SHA256")
            if path in seen_paths or digest in seen_hashes:
                raise ValueError("duplicate source path or hash in frozen partitions")
            if group in groups and groups[group] != partition:
                raise ValueError("source-group leakage between partitions")
            if entry["object_asset"] not in ASSETS or entry["object"] != entry["object_asset"].split("_")[0]:
                raise ValueError("ULTRA scope is the four supported objects and eight exact assets")
            seen_paths.add(path)
            seen_hashes.add(digest)
            groups[group] = partition
            local_groups.add(group)
        if canonical and (len(entries), len(local_groups)) != CANONICAL_COUNTS[partition]:
            raise ValueError("canonical frozen counts changed")
        summary[partition] = {"motions": len(entries), "source_groups": len(local_groups),
                              "assets": dict(sorted(Counter(e["object_asset"] for e in entries).items()))}
    if canonical and any(set(s["assets"]) != set(ASSETS) for s in summary.values()):
        raise ValueError("canonical split must cover every asset")
    if canonical:
        fields = ("motion", "motion_sha256", "source_group", "object", "object_asset")
        membership = {name: sorted(tuple(entry[key] for key in fields) for entry in entries)
                      for name, entries in partitions.items()}
        digest = hashlib.sha256(json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != CANONICAL_MEMBERSHIP_SHA256:
            raise ValueError("canonical source-group membership changed")
    return summary


def build_manifest(splits_bytes, bindings_bytes, *, source_root=None, profile="canonical-v4"):
    """Create metadata only; never open protected motion tensors here."""
    if profile not in ("canonical-v4", "fixture"):
        raise ValueError("unknown split profile")
    canonical = profile == "canonical-v4"
    split_sha = hashlib.sha256(splits_bytes).hexdigest()
    if canonical and split_sha != CANONICAL_SPLITS_SHA256:
        raise ValueError("canonical split SHA256 mismatch")
    if canonical and hashlib.sha256(bindings_bytes).hexdigest() != CANONICAL_BINDINGS_SHA256:
        raise ValueError("canonical source joint bindings SHA256 mismatch")
    split = json.loads(splits_bytes.decode("utf-8-sig"))
    bindings = json.loads(bindings_bytes.decode("utf-8-sig"))
    names = bindings.get("source_joint_names")
    if not isinstance(names, list) or len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("source joint-name binding must exactly cover the 29 hw3 joints")
    evidence = bindings.get("joint_order_evidence", {})
    if not evidence.get("fixture_sha256") or not evidence.get("field"):
        raise ValueError("source joint order requires external metadata evidence")
    partitions = split["partitions"]
    summary = _validate_partitions(partitions, canonical)
    entries = []
    for partition, cases in partitions.items():
        for original in cases:
            entry = dict(original, partition=partition,
                         role="train" if partition == "train" else "protected_eval")
            path = entry["motion"].replace("\\", "/")
            if source_root is not None:
                if not path.startswith(SOURCE_ROOT + "/"):
                    raise ValueError("source-root remap only accepts the exact canonical source prefix")
                entry["resolved_motion"] = str(Path(source_root) / path[len(SOURCE_ROOT) + 1:])
            else:
                entry["resolved_motion"] = entry["motion"]
            entries.append(entry)
    manifest = {"schema": SCHEMA, "profile": profile, "split_sha256": split_sha,
                "bindings_sha256": hashlib.sha256(bindings_bytes).hexdigest(),
                "source_joint_names": names, "target_joint_names": list(G1_JOINT_NAMES),
                "joint_order_evidence": evidence, "timebase": TIMEBASE,
                "object_assets": list(ASSETS), "partitions": summary, "clips": entries,
                "protection": {"optimizer_partition": "train", "source_groups_disjoint": True,
                               "protected_eval_partitions": ["validation", "regression", "test"],
                               "validation_regression": "historically_used_for_development_not_untouched",
                               "test": "held_out_from_this_training_historical_access_not_asserted",
                               "hoi80": "development_evaluation_not_an_independent_test"}}
    manifest["content_sha256"] = content_hash(manifest)
    return manifest


def prepare_manifest(frozen_inputs, output, *, source_root=None):
    """Read named members without extracting an archive or altering source files."""
    with tarfile.open(frozen_inputs, "r:gz") as archive:
        split = archive.extractfile("splits.json").read()
        bindings = archive.extractfile("bindings.json").read()
    manifest = build_manifest(split, bindings, source_root=source_root)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA or m.get("content_sha256") != content_hash(m):
        raise ValueError("ULTRA manifest schema or content hash mismatch")
    if m.get("object_assets") != list(ASSETS) or m.get("timebase") != TIMEBASE:
        raise ValueError("ULTRA asset/timebase contract changed")
    if m.get("target_joint_names") != list(G1_JOINT_NAMES):
        raise ValueError("target robot joint order differs")
    names = m.get("source_joint_names", [])
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("invalid source joint binding")
    if m["profile"] not in ("canonical-v4", "fixture"):
        raise ValueError("unknown manifest profile")
    canonical = m["profile"] == "canonical-v4"
    if canonical and m.get("split_sha256") != CANONICAL_SPLITS_SHA256:
        raise ValueError("canonical split identity changed")
    if canonical and (m.get("bindings_sha256") != CANONICAL_BINDINGS_SHA256 or names != list(G1_JOINT_NAMES)):
        raise ValueError("canonical source joint-name binding changed")
    partitions = {name: [] for name in CANONICAL_COUNTS}
    for entry in m["clips"]:
        p = entry["partition"]
        if p not in partitions or entry["role"] != ("train" if p == "train" else "protected_eval"):
            raise ValueError("partition role changed")
        partitions[p].append(entry)
    if _validate_partitions(partitions, canonical) != m["partitions"]:
        raise ValueError("partition summary differs")
    return m


def select_clips(manifest, *, partition="train", for_training=True, asset=None, one_per_asset=False):
    if for_training and partition != "train":
        raise ValueError("Protected eval cannot be used for PPO training or normalizer updates")
    if partition not in (*CANONICAL_COUNTS, "protected_eval"):
        raise ValueError("unknown ULTRA partition")
    if isinstance(asset, int):
        if not 0 <= asset < len(ASSETS):
            raise ValueError("invalid exact asset index")
        asset = ASSETS[asset]
    if asset is not None and asset not in ASSETS:
        raise ValueError("invalid exact asset identifier")
    entries = [e for e in manifest["clips"] if
               (e["role"] == "protected_eval" if partition == "protected_eval" else e["partition"] == partition)
               and (asset is None or e["object_asset"] == asset)]
    if one_per_asset:
        first = {}
        for entry in entries:
            first.setdefault(entry["object_asset"], entry)
        entries = [first[key] for key in sorted(first)]
    if not entries:
        raise ValueError("no clips for the requested partition/asset")
    return entries


def _gradient(x):
    out = torch.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / (2 * CONTROL_DT)
    out[0], out[-1] = (x[1] - x[0]) / CONTROL_DT, (x[-1] - x[-2]) / CONTROL_DT
    return out


def _angular_velocity(rotation):
    out = rotation.new_zeros((len(rotation), 3))
    out[1:-1] = so3_log(rotation[2:] @ rotation[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
    out[0] = so3_log(rotation[1] @ rotation[0].T) / CONTROL_DT
    out[-1] = so3_log(rotation[-1] @ rotation[-2].T) / CONTROL_DT
    return out


def adapt_motion(frames, source_joint_names):
    """Return robot71, object13 and original39 contact flags on one T+1 grid."""
    if (not isinstance(frames, torch.Tensor) or frames.layout != torch.strided or not frames.is_floating_point()
            or frames.ndim != 2 or frames.shape[1] != 630 or len(frames) < 3):
        raise ValueError("ULTRA requires a dense floating Tensor[N,630] with a complete 50 Hz transition")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("nonfinite ULTRA source data")
    if len(source_joint_names) != 29 or len(set(source_joint_names)) != 29 or set(source_joint_names) != set(G1_JOINT_NAMES):
        raise ValueError("source joint names mismatch")
    for start, end in ((3, 7), (74, 78), (201, 357)):
        quats = frames[:, start:end].reshape(-1, 4).double()
        if bool(((quats.norm(dim=-1) - 1).abs() > 1e-3).any()):
            raise ValueError("invalid source xyzw quaternion")
    flags = frames[:, 591:630]
    if not bool(((flags == 0) | (flags == 1)).all()):
        raise ValueError("source body contact flags must be binary")
    frames = frames.float()
    steps = 5 * (len(frames) - 1) // 6
    tick = torch.arange(steps + 1, device=frames.device)
    numerator = tick * 6
    i0, rem = numerator // 5, numerator % 5
    i1 = torch.minimum(i0 + 1, torch.tensor(len(frames) - 1, device=frames.device))
    alpha = rem.float() / 5
    interpolate = lambda x: x[i0] * (1 - alpha[:, None]) + x[i1] * alpha[:, None]
    def rotation(start):
        q = frames[:, start:start + 4]
        q = torch.nn.functional.normalize(torch.cat((q[:, 3:4], q[:, :3]), -1), dim=-1)
        result = quat_slerp(q[i0], q[i1], alpha)
        return result, quat_to_mat(result)
    p, op = interpolate(frames[:, :3]), interpolate(frames[:, 71:74])
    quat, R = rotation(3)
    oq, oR = rotation(74)
    order = [source_joint_names.index(n) for n in G1_JOINT_NAMES]
    q = interpolate(frames[:, 13:42])[:, order]
    robot = torch.cat((p, quat, _gradient(p), _angular_velocity(R), q, _gradient(q)), -1)
    obj = torch.cat((op, oq, _gradient(op), _angular_velocity(oR)), -1)
    nearest = torch.minimum((numerator + 2) // 5, torch.tensor(len(frames) - 1, device=frames.device))
    return robot, obj, frames[nearest, 591:630].clone()


class Indexed:
    """Lazy reference fields, with the same indexing interface as BoneseedMotionLib."""
    def __init__(self, compute, count):
        self.compute, self.count = compute, count

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.compute(index[0])[(..., *index[1:])]
        return self.compute(index)

    def cpu(self):
        return self.compute(slice(None)).cpu()


class UltraMotionLib:
    """Same GMT kinematics/goals; object and contact references are additional."""
    def __init__(self, manifest_path, kin, device, *, partition="train", for_training=True,
                 asset=None, asset_filter=None, one_per_asset=False):
        if asset_filter is not None:
            if asset is not None:
                raise ValueError("choose asset or asset_filter, not both")
            asset = asset_filter
        self.manifest = read_manifest(manifest_path)
        self.entries = select_clips(self.manifest, partition=partition, for_training=for_training,
                                    asset=asset, one_per_asset=one_per_asset)
        self.device, self.kin, self.partition = device, kin, partition
        self.for_training, self.dataset = for_training, "ultra"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.object_assets = list(ASSETS)
        self.names = [PurePosixPath(e["motion"]).stem for e in self.entries]
        states, objects, contacts, lengths = [], [], [], []
        for entry in self.entries:
            raw = Path(entry["resolved_motion"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["motion_sha256"]:
                raise ValueError(f"source motion SHA256 mismatch: {entry['motion']}")
            frames = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            robot, obj, flags = adapt_motion(frames, self.manifest["source_joint_names"])
            states.append(robot)
            objects.append(obj)
            contacts.append(flags)
            lengths.append(len(robot))
        self.states = torch.cat(states).to(device)
        self.objects = torch.cat(objects).to(device)
        self.contacts = torch.cat(contacts).to(device)
        self.num_frames = len(self.states)
        self.root_p, self.root_v, self.root_w = self.states[:, :3], self.states[:, 7:10], self.states[:, 10:13]
        self.q, self.qd = self.states[:, 13:42], self.states[:, 42:71]
        self.root_R = Indexed(lambda f: quat_to_mat(self.states[f, 3:7]), self.num_frames)
        self.root_gz = Indexed(lambda f: self.root_R[f][..., 2, 2], self.num_frames)
        self.root_v_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_v[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        self.root_w_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_w[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        for name, index in (("body_p", 0), ("body_R", 1), ("body_v", 2), ("body_w", 3)):
            setattr(self, name, Indexed(lambda f, index=index: kin.forward(*self.ref_state(f))[index], self.num_frames))
        self.clip_len = torch.tensor(lengths, device=device, dtype=torch.long)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lengths), device=device), self.clip_len)
        self.clip_asset_ids = torch.tensor([ASSETS.index(e["object_asset"]) for e in self.entries], device=device)
        self.asset_ids = self.clip_asset_ids[self.clip_id]
        nb = (self.clip_len - 1 + BIN_FRAMES - 1) // BIN_FRAMES
        bases = nb.cumsum(0) - nb
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lengths), device=device), nb)
        offset = torch.arange(int(nb.sum()), device=device) - bases[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + BIN_FRAMES * offset
        self.num_bins = len(self.bin_start)
        frame = torch.arange(self.num_frames, device=device)
        local_bin = torch.minimum((frame - self.clip_start[self.clip_id]) // BIN_FRAMES, nb[self.clip_id] - 1)
        self.frame_bin = bases[self.clip_id] + local_bin
        self.selection_contract = {"partition": partition, "for_training": for_training,
                                   "one_per_asset": one_per_asset, "asset": asset,
                                   "clips": len(self.entries), "source_groups": len({e["source_group"] for e in self.entries}),
                                   "selected_motion_sha256": [e["motion_sha256"] for e in self.entries]}

    def reference_bodies(self, f):
        return self.kin.forward(*self.ref_state(f))

    def ref_state(self, f):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def future_frames(self, f, K):
        end = self.clip_end[self.clip_id[f]]
        future = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, f, K):
        # Exact motion-only GMT goal layout; no object slots enter the base GMT.
        future = self.future_frames(f, K)
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[future] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[future])
        features = torch.cat((dp, rot, self.root_v_local[future], self.root_w_local[future],
                              self.q[future], self.qd[future] * 0.1), -1)
        return features.reshape(f.shape[0], -1)

    def object_ref(self, f):
        x = self.objects[f]
        return x[..., :3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]

    def contact_ref(self, f):
        """Original 39 source-body binary flags, not pair-contact measurements."""
        return self.contacts[f]

    def compatible_bins(self, asset_id):
        if isinstance(asset_id, str):
            asset_id = ASSETS.index(asset_id)
        return (self.clip_asset_ids[self.bin_clip] == asset_id).nonzero().flatten()
