"""Hash-bound scene supports, separate from learned object/ground contact partners.

OmniContact pins each kinematic table to the sampled reference frame at reset,
with zero velocity for the episode. This reproduces that timing; it does not
invent moving platforms from per-frame source positions.
"""
from __future__ import annotations

import hashlib
import json
import re
import torch

SCENE_SUPPORT_SCHEMA = "hoi_scene_support_reset_frame_v1"
SUPPORT_IDS = ("table1", "table2")
SUPPORT_ROLES = (20, 21)
SOURCE_REVISION = "19cde38364a5b48e47348b270c895eabe2ba3ad9"
ENV_CFG_SHA256 = "d737b55175c0a539d0862698a527dbf00b09b95e3e111b175b45249591b69c6c"
COMMANDS_SHA256 = "f1e356a97dc7580eb5c81b1c119b4c8a15c1a1bfdeafdeb08613582f0b270d04"
POSE_MODE = "sampled_reference_frame_held_for_episode"


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def omni_support_recipe():
    """Explicit source values, including the inherited scene material.

    Source env_cfg.py lines 67-114, 46-51, 513; commands.py lines 1002-1020,
    1141. Contact offset is retained as source evidence, not silently translated
    to a different MuJoCo solver margin; native effective parameters are audited.
    """
    return {"repository": "Ingrid789/OmniContact", "revision": SOURCE_REVISION,
            "env_cfg_sha256": ENV_CFG_SHA256, "commands_sha256": COMMANDS_SHA256,
            "size_m": [.3, .25, .01], "kinematic": True, "disable_gravity": True,
            "mass_kg": 10000., "linear_damping": 1000., "angular_damping": 1000.,
            "source_contact_offset_m": .002, "source_rest_offset_m": 0.,
            "static_friction": 1., "dynamic_friction": 1., "restitution": 0.,
            "material_source": "env_cfg scene terrain material assigned to simulation default",
            "pose_mode": POSE_MODE}


def bind_scene_supports(raw_supports, source_timestamps_s, target_timestamps_s):
    """Resolve the known audited Omni cuboids onto the existing target grid.

    Unknown/incomplete source support descriptions are errors. Raw source
    metadata remains in the curated manifest and is bound by its own digest.
    """
    if not isinstance(raw_supports, (list, tuple)):
        raise ValueError("scene-support metadata must be restored before binding")
    if not raw_supports:
        return []
    source = torch.as_tensor(source_timestamps_s, dtype=torch.float64)
    target = torch.as_tensor(target_timestamps_s, dtype=torch.float64)
    if source.ndim != 1 or target.ndim != 1 or len(source) < 2 or len(target) < 2 or not bool(torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise ValueError("support timebase requires finite source/target timestamps")
    if bool((source.diff() <= 0).any() or (target.diff() <= 0).any()) or target[0] < source[0] - 1e-10 or target[-1] > source[-1] + 1e-10:
        raise ValueError("support timebase must be increasing without extrapolation")
    if not torch.allclose(target.diff(), torch.full_like(target.diff(), .02), atol=1e-9, rtol=0):
        raise ValueError("support target timebase must match the unchanged 50 Hz states")
    right = torch.searchsorted(source, target).clamp(1, len(source)-1)
    left = right-1
    weight = ((target-source[left])/(source[right]-source[left]))[:,None]
    result, seen = [], set()
    for raw in raw_supports:
        sid = raw.get("partner_id")
        recipe = raw.get("recipe", {})
        if sid not in SUPPORT_IDS or sid in seen or raw.get("kind") != "cuboid" or raw.get("kinematic") is not True or raw.get("source_present") is not True:
            raise ValueError("support requires one explicitly bound Omni table1/table2 cuboid")
        if recipe.get("repository") != "Ingrid789/OmniContact" or recipe.get("revision") != SOURCE_REVISION or recipe.get("sha256") != ENV_CFG_SHA256 or recipe.get("path") != "source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py":
            raise ValueError("support source recipe identity is not audited")
        if raw.get("source_field") != sid + "_pos_w":
            raise ValueError("support source position field identity mismatch")
        size = torch.as_tensor(raw.get("size_m"), dtype=torch.float64)
        if size.shape != (3,) or not torch.equal(size, torch.tensor([.3,.25,.01], dtype=torch.float64)):
            raise ValueError("support size differs from audited author cuboid")
        position = torch.as_tensor(raw.get("position"), dtype=torch.float64)
        quaternion = torch.as_tensor(raw.get("quaternion"), dtype=torch.float64)
        if position.shape != (len(source),3) or quaternion.shape != (len(source),4) or not bool(torch.isfinite(position).all() and torch.isfinite(quaternion).all()):
            raise ValueError("support source trajectory shape/finite check failed")
        identity = torch.zeros_like(quaternion); identity[:,0] = 1
        if not torch.allclose(quaternion, identity, atol=1e-12, rtol=0):
            raise ValueError("audited Omni support reset requires identity wxyz orientation")
        parked = bool((position[:,2] < -5).all())
        if raw.get("parked_below_scene") is not parked:
            raise ValueError("support parked source flag mismatch")
        uses = bool((position.abs() > 1e-6).any()) and not parked
        if sid == "table1" and raw.get("official_table1_uses") is not uses:
            raise ValueError("table1 author activation flag mismatch")
        p = position[left] + weight*(position[right]-position[left])
        if sid == "table1" and not uses:
            p = torch.zeros_like(p); p[:,2] = -10.
        q = torch.zeros(len(target),4,dtype=torch.float64); q[:,0] = 1
        bound = {"schema": SCENE_SUPPORT_SCHEMA, "support_id": sid, "pose_mode": POSE_MODE,
                 "source_support_sha256": digest(raw), "geometry_recipe": omni_support_recipe(),
                 "geometry_recipe_sha256": digest(omni_support_recipe()),
                 "source_timebase_sha256": digest(source), "target_timebase_sha256": digest(target),
                 "position": p, "quaternion_wxyz": q, "source_state_count": len(source),
                 "target_state_count": len(target), "source_table1_uses": uses if sid == "table1" else None}
        bound["binding_sha256"] = digest(bound)
        result.append(bound); seen.add(sid)
    return sorted(result, key=lambda x: SUPPORT_IDS.index(x["support_id"]))


def validate_bound_support(record, raw=None, state_count=None):
    if not isinstance(record, dict) or record.get("schema") != SCENE_SUPPORT_SCHEMA or record.get("support_id") not in SUPPORT_IDS or record.get("pose_mode") != POSE_MODE:
        raise ValueError("invalid native support schema/identity/timing")
    if record.get("binding_sha256") != digest({k:v for k,v in record.items() if k != "binding_sha256"}):
        raise ValueError("native support binding hash changed")
    if record.get("geometry_recipe") != omni_support_recipe() or record.get("geometry_recipe_sha256") != digest(omni_support_recipe()):
        raise ValueError("native support physical recipe changed")
    if raw is not None and record.get("source_support_sha256") != digest(raw):
        raise ValueError("native support differs from retained source metadata")
    n = record.get("target_state_count")
    p, q = torch.as_tensor(record["position"]), torch.as_tensor(record["quaternion_wxyz"])
    if type(n) is not int or n < 2 or (state_count is not None and n != state_count) or p.shape != (n,3) or q.shape != (n,4) or not bool(torch.isfinite(p).all() and torch.isfinite(q).all()):
        raise ValueError("native support target trajectory mismatch")
    identity = torch.zeros_like(q); identity[:,0] = 1
    if not torch.allclose(q, identity, atol=1e-12, rtol=0):
        raise ValueError("native support orientation changed")
    return record


def validate_native_scene_supports(motion):
    raw = getattr(motion, "scene_supports", None)
    if raw is None:
        if any(getattr(motion, "bound_scene_supports", [])):
            raise ValueError("bound support requires retained raw scene-support provenance")
        return
    if not isinstance(raw, (list,tuple)) or len(raw) != len(motion.names):
        raise ValueError("scene-support metadata must align with the native motion clips")
    if any(not isinstance(value, (list,tuple)) for value in raw):
        raise ValueError("scene-support metadata must be restored before native simulation")
    bound = getattr(motion, "bound_scene_supports", [[] for _ in raw])
    if not isinstance(bound, (list,tuple)) or len(bound) != len(raw):
        raise ValueError("bound scene-support metadata must align with clips")
    for i, (sources, records) in enumerate(zip(raw,bound)):
        if len(sources) != len(records):
            raise ValueError("Native scene cannot omit declared scene supports; explicit audited support assets and trajectories are required")
        if len({r.get("support_id") for r in records}) != len(records):
            raise ValueError("duplicate native support identity")
        by_id = {s.get("partner_id"):s for s in sources}
        for record in records:
            if record.get("support_id") not in by_id:
                raise ValueError("native support source identity mismatch")
            validate_bound_support(record, by_id[record["support_id"]], int(motion.clip_len[i]))


class NativeSceneSupportBank:
    """Immutable geometry slots and reference reset poses for one object worker."""
    def __init__(self, motion, asset_id, device):
        validate_native_scene_supports(motion)
        self.records = getattr(motion, "bound_scene_supports", [[] for _ in motion.names])
        selected = [i for i,start in enumerate(motion.clip_start) if int(motion.asset_ids[start]) == int(asset_id)]
        self.support_ids = tuple(sid for sid in SUPPORT_IDS if any(any(r['support_id']==sid for r in self.records[i]) for i in selected))
        self.poses = {}
        self.present = {}
        for sid in self.support_ids:
            pose = torch.zeros(len(motion.asset_ids),7,dtype=torch.float32,device=device)
            pose[:,2] = -10. if sid == "table1" else -5.
            pose[:,6] = 1.  # native xyzw
            present = torch.zeros(len(motion.asset_ids),dtype=torch.bool,device=device)
            for i in selected:
                records = [r for r in self.records[i] if r['support_id']==sid]
                if not records:
                    continue
                record = records[0]; start=int(motion.clip_start[i]); n=int(motion.clip_len[i])
                pose[start:start+n,:3] = torch.as_tensor(record['position'],device=device,dtype=pose.dtype)
                q = torch.as_tensor(record['quaternion_wxyz'],device=device,dtype=pose.dtype)
                pose[start:start+n,3:] = q[:,[1,2,3,0]]
                # Official missing/disabled tables park at -5 m or -10 m.
                present[start:start+n] = pose[start:start+n,2] > -5.
            self.poses[sid] = pose
            self.present[sid] = present
        self.manifest = {"schema": SCENE_SUPPORT_SCHEMA, "pose_mode": POSE_MODE,
                         "declared_support_ids": list(self.support_ids), "geometry_recipe": omni_support_recipe(),
                         "geometry_recipe_sha256": digest(omni_support_recipe()),
                         "clips": [{"name": motion.names[i], "bindings": [{k:r[k] for k in ('support_id','binding_sha256','source_support_sha256','target_timebase_sha256')} for r in self.records[i]]} for i in selected]}

    def sample(self, support_id, frames, env_origins):
        pose = self.poses[support_id][frames].clone()
        pose[:,:3] += env_origins
        return pose


def compiled_support_role(geom_name, body_name):
    """Named support colliders cannot become target Object or infinite ground."""
    for sid, role in zip(SUPPORT_IDS, SUPPORT_ROLES):
        if re.search(r"(?:^|[/_])Support_" + sid + r"(?:$|[/_])", geom_name + " " + body_name):
            return role
    return None
