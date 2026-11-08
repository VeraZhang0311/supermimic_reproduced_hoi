"""Versioned 30 -> 50 Hz table references on the current hw3 robot.

The immutable source Parquet is decoded once during preparation. Runtime reads
only a hashed, already-resampled tensor cache. Source contact annotations remain
provenance; they are never reported as native pair impulse measurements.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path

import torch

from hoi_residual.data import (Indexed, UltraMotionLib, _angular_velocity,
                               _gradient, content_hash, sha256_file)
from supermimic.robot_metadata import G1_JOINT_NAMES
from supermimic.rotations import quat_slerp, quat_to_mat

ASSETS = ("largetable", "smalltable")
CONTROL_DT, BIN_FRAMES = .02, 50
SCHEMA = "hoi_retarget_tables_hw3_reference_v1"
CACHE_SCHEMA = "hoi_retarget_tables_50hz_cache_v1"
TIMEBASE = {"source_hz": 30, "control_hz": 50, "control_dt": CONTROL_DT,
            "source_quaternion_order": "wxyz", "source_position_units": "m",
            "state_grid": "tick/50; source_index=3*tick/5",
            "state_control_alignment": "T+1_states_T_controls",
            "velocity": "pose_derived_at_50hz_within_each_clip",
            "trajectory_scaling": "source_already_applied_no_additional_scale",
            "mesh_scaling": "source_object_scale_once_during_original_geometry_preparation",
            "source_contact_semantics": "source_annotations_not_native_pair_impulses"}


def read_selection(path):
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    names = selection.get("included", [])
    if selection.get("schema") != "hoi_tables_user_selection_v1" or not names or len(set(names)) != len(names):
        raise ValueError("invalid or duplicated tables selection")
    if any(n in selection.get("excluded", {}) for n in names):
        raise ValueError("excluded clips re-entered the whitelist")
    if any(len(n.split("_")) != 3 or n.split("_")[1] not in ASSETS for n in names):
        raise ValueError("tables whitelist contains an unsupported source identity")
    return selection


def _quaternion(value, n, field):
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.shape != (n, 4) or not torch.isfinite(tensor).all() or ((tensor.norm(dim=-1)-1).abs() > 1e-3).any():
        raise ValueError(f"invalid wxyz quaternion: {field}")
    return torch.nn.functional.normalize(tensor, dim=-1).float()


def adapt_row(row):
    """Resample exact declared 30 Hz poses once; no trajectory rescaling."""
    n = int(row["n_frames"])
    if float(row["fps"]) != 30. or n < 2:
        raise ValueError("table reference requires >=2 source frames at declared 30 Hz")
    names = row["dof_names"]
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("source joint names do not exactly cover hw3 joints")
    arrays = {}
    for name, width in (("root_pos",3),("object_pos",3),("dof_pos",29)):
        value = torch.as_tensor(row[name], dtype=torch.float32)
        if value.shape != (n,width) or not torch.isfinite(value).all():
            raise ValueError(f"invalid source pose array: {name}")
        arrays[name] = value
    rq, oq = _quaternion(row["root_rot"],n,"root_rot"), _quaternion(row["object_rot"],n,"object_rot")
    # Integer rational grid avoids floating-point endpoint drift.
    steps = 5*(n-1)//3
    tick = torch.arange(steps+1)
    numerator = tick*3
    lower, fraction = numerator//5, (numerator%5).float()/5
    upper = (lower+1).clamp_max(n-1)
    lerp = lambda x: x[lower]*(1-fraction[:,None])+x[upper]*fraction[:,None]
    p, op = lerp(arrays["root_pos"]), lerp(arrays["object_pos"])
    q = lerp(arrays["dof_pos"])[:,[names.index(name) for name in G1_JOINT_NAMES]]
    rq = quat_slerp(rq[lower],rq[upper],fraction)
    oq = quat_slerp(oq[lower],oq[upper],fraction)
    states = torch.cat((p,rq,_gradient(p),_angular_velocity(quat_to_mat(rq)),q,_gradient(q)),-1)
    objects = torch.cat((op,oq,_gradient(op),_angular_velocity(quat_to_mat(oq))),-1)
    if len(states) < 2 or not torch.isfinite(states).all() or not torch.isfinite(objects).all():
        raise ValueError("invalid prepared table state")
    nearest = ((numerator+2)//5).clamp_max(n-1)
    flags = torch.as_tensor(row.get("per_link_contact_flags",[[False]*4]*n))
    if flags.shape != (n,4) or not ((flags == 0)|(flags == 1)).all():
        raise ValueError("invalid source contact flags")
    return {"states": states.contiguous(), "objects": objects.contiguous(),
            "source_contacts": flags[nearest].bool().contiguous()}


def trajectory_hash(row):
    """Pose-only source identity detects duplicate captures across partitions."""
    digest = hashlib.sha256()
    for field in ("root_pos","root_rot","dof_pos","object_pos","object_rot"):
        value = torch.as_tensor(row[field],dtype=torch.float64).contiguous()
        digest.update(field.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assign_partitions(rows, *, fraction=.1, seed=0):
    """Deterministic per-object heldout quota, stratified across subjects.

    Source sequence_id is indivisible. Exact duplicate pose captures are also
    indivisible; their IDs are assigned together rather than split by row order.
    """
    if not 0 < fraction < .5:
        raise ValueError("protected fraction must be between zero and one half")
    groups, fingerprint_group = {}, {}
    for row in rows:
        name = row["subject"]
        group = row.get("sequence_id", "omomo/"+name)
        if group != "omomo/"+name:
            raise ValueError("unexpected OMOMO source group identity")
        fingerprint = trajectory_hash(row)
        group = fingerprint_group.setdefault((row["object"],fingerprint),group)
        groups.setdefault(group,[]).append(row)
    strata = defaultdict(lambda: defaultdict(list))
    for group, members in groups.items():
        obj, subject = members[0]["object"], members[0]["subject"].split("_")[0]
        if any(r["object"] != obj for r in members):
            raise ValueError("one source group spans different table classes")
        strata[obj][subject].append(group)
    protected = set()
    rank = lambda group: hashlib.sha256(f"tables-v1:{seed}:{group}".encode()).hexdigest()
    for obj, subjects in sorted(strata.items()):
        target = max(1,round(sum(map(len,subjects.values()))*fraction))
        quota = {s:min(len(gs)-1,math.floor(len(gs)*fraction)) for s,gs in subjects.items()}
        order = sorted(subjects,key=lambda s:(-(len(subjects[s])*fraction-math.floor(len(subjects[s])*fraction)),rank(s)))
        for subject in order:
            if sum(quota.values()) >= target: break
            if quota[subject] < len(subjects[subject])-1: quota[subject] += 1
        for subject, gs in subjects.items():
            protected.update(sorted(gs,key=rank)[:quota[subject]])
    result = {}
    for group, members in groups.items():
        for row in members:
            result[row["subject"]] = "protected_eval" if group in protected else "train"
    if set(result.values()) != {"train","protected_eval"}:
        raise ValueError("selection cannot support disjoint train and protected eval")
    return result


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("content_sha256") != content_hash(manifest):
        raise ValueError("tables manifest identity mismatch")
    if manifest.get("object_assets") != list(ASSETS) or manifest.get("timebase") != TIMEBASE:
        raise ValueError("tables asset/timebase contract changed")
    if manifest.get("target_joint_names") != list(G1_JOINT_NAMES):
        raise ValueError("tables target joint order changed")
    groups, fingerprints, identities = {}, {}, set()
    for entry in manifest["clips"]:
        role = entry["partition"]
        if role not in ("train","protected_eval") or entry["role"] != role:
            raise ValueError("invalid tables partition role")
        if entry["object_asset"] not in ASSETS or entry["clip_id"] in identities:
            raise ValueError("invalid or duplicated table clip")
        identities.add(entry["clip_id"])
        for mapping,key in ((groups,entry["source_group"]),(fingerprints,entry["source_trajectory_sha256"])):
            if key in mapping and mapping[key] != role:
                raise ValueError("source group/duplicate trajectory leakage")
            mapping[key] = role
    if len(identities) != manifest["selection"]["included_count"]:
        raise ValueError("prepared clip count differs from whitelist")
    return manifest


class TableMotionLib(UltraMotionLib):
    """The existing hw3 motion/reference API backed by prepared table tensors."""
    def __init__(self, manifest_path, kin, device, *, partition="train", for_training=True,
                 asset=None, asset_filter=None, one_per_asset=False):
        if asset_filter is not None:
            if asset is not None: raise ValueError("choose asset or asset_filter")
            asset = asset_filter
        if for_training and partition != "train":
            raise ValueError("protected eval cannot update optimizer, normalizers or RSI")
        if partition not in ("train","protected_eval"):
            raise ValueError("unknown tables partition")
        if isinstance(asset,int):
            if not 0 <= asset < len(ASSETS): raise ValueError("invalid table asset index")
            asset = ASSETS[asset]
        if asset is not None and asset not in ASSETS: raise ValueError("invalid table asset")
        self.manifest = read_manifest(manifest_path)
        self.entries = [e for e in self.manifest["clips"] if e["partition"] == partition and (asset is None or e["object_asset"] == asset)]
        if one_per_asset:
            first = {}
            for entry in self.entries: first.setdefault(entry["object_asset"],entry)
            self.entries = [first[k] for k in sorted(first)]
        if not self.entries: raise ValueError("empty tables partition/asset selection")
        self.device,self.kin,self.partition = device,kin,partition
        self.for_training,self.allowed_for_training,self.dataset = for_training,for_training and partition == "train","hoi_retarget_tables"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.object_assets = list(ASSETS)
        self.names = [e["subject"] for e in self.entries]
        cache_path = Path(self.manifest["cache"]["path"])
        if not cache_path.is_absolute(): cache_path = Path(manifest_path).parent/cache_path
        raw = cache_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest["cache"]["sha256"]:
            raise ValueError("table tensor cache SHA256 mismatch")
        cache = torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
        if cache.get("schema") != CACHE_SCHEMA or cache.get("timebase") != TIMEBASE:
            raise ValueError("table cache schema/timebase mismatch")
        values = [cache["clips"][e["clip_id"]] for e in self.entries]
        for value,entry in zip(values,self.entries):
            if value["states"].shape != (entry["state_frames"],71) or value["objects"].shape != (entry["state_frames"],13):
                raise ValueError("table cached state shape changed")
            if not torch.isfinite(value["states"]).all() or not torch.isfinite(value["objects"]).all():
                raise ValueError("nonfinite cached table reference")
        lengths = [len(v["states"]) for v in values]
        self.states = torch.cat([v["states"] for v in values]).to(device)
        self.objects = torch.cat([v["objects"] for v in values]).to(device)
        self.contacts = torch.cat([v["source_contacts"] for v in values]).to(device)
        self.num_frames = len(self.states)
        self.object_p,self.object_v,self.object_w = self.objects[:,:3],self.objects[:,7:10],self.objects[:,10:13]
        self.object_R = Indexed(lambda f:quat_to_mat(self.objects[f,3:7]),self.num_frames)
        self.root_p,self.root_v,self.root_w = self.states[:,:3],self.states[:,7:10],self.states[:,10:13]
        self.q,self.qd = self.states[:,13:42],self.states[:,42:71]
        self.root_R = Indexed(lambda f:quat_to_mat(self.states[f,3:7]),self.num_frames)
        self.root_gz = Indexed(lambda f:self.root_R[f][...,2,2],self.num_frames)
        self.root_v_local = Indexed(lambda f:(self.root_R[f].transpose(-1,-2)@self.root_v[f].unsqueeze(-1)).squeeze(-1),self.num_frames)
        self.root_w_local = Indexed(lambda f:(self.root_R[f].transpose(-1,-2)@self.root_w[f].unsqueeze(-1)).squeeze(-1),self.num_frames)
        for name,index in (("body_p",0),("body_R",1),("body_v",2),("body_w",3)):
            setattr(self,name,Indexed(lambda f,index=index:kin.forward(*self.ref_state(f))[index],self.num_frames))
        self.clip_len = torch.tensor(lengths,device=device,dtype=torch.long)
        self.clip_start = self.clip_len.cumsum(0)-self.clip_len
        self.clip_end = self.clip_start+self.clip_len-1
        self.clip_id = torch.repeat_interleave(torch.arange(len(lengths),device=device),self.clip_len)
        self.clip_asset_ids = torch.tensor([ASSETS.index(e["object_asset"]) for e in self.entries],device=device)
        self.asset_ids = self.clip_asset_ids[self.clip_id]
        nb = (self.clip_len-1+BIN_FRAMES-1)//BIN_FRAMES
        bases = nb.cumsum(0)-nb
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lengths),device=device),nb)
        offset = torch.arange(int(nb.sum()),device=device)-bases[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip]+BIN_FRAMES*offset
        self.num_bins = len(self.bin_start)
        frame = torch.arange(self.num_frames,device=device)
        local_bin = torch.minimum((frame-self.clip_start[self.clip_id])//BIN_FRAMES,nb[self.clip_id]-1)
        self.frame_bin = bases[self.clip_id]+local_bin
        self.selection_contract = {"partition":partition,"for_training":for_training,"one_per_asset":one_per_asset,
                                   "asset":asset,"clips":len(self.entries),"source_groups":len({e['source_group'] for e in self.entries}),
                                   "selected_motion_sha256":[e["source_trajectory_sha256"] for e in self.entries]}

    def compatible_bins(self, asset_id):
        if isinstance(asset_id,str): asset_id = ASSETS.index(asset_id)
        return (self.clip_asset_ids[self.bin_clip] == asset_id).nonzero().flatten()

    def ref_object(self, frames):
        """v1.1 name for the same object reference used by current hw3 code."""
        return self.object_ref(frames)


HOIMotionLib = TableMotionLib
