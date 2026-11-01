"""Entity-pair, interval contact labels; no body net-force approximation.

The native adapter targets Newton 1.8/MuJoCo-Warp 3.15.  The public
``mujoco_warp.contact_force`` decodes pyramidal/elliptic constraint forces.
It must run after *each* solver substep: an endpoint contact read cannot
represent the preceding control interval. Native imports are lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType

import torch

from .scene_supports import SUPPORT_IDS, SUPPORT_ROLES, compiled_support_role

CHANNELS = (
    "left_hand_object", "right_hand_object", "left_forearm_object", "right_forearm_object",
    "left_upper_arm_object", "right_upper_arm_object", "torso_object", "pelvis_object",
    "left_thigh_object", "right_thigh_object", "left_shin_object", "right_shin_object",
    "left_foot_object", "right_foot_object", "head_object", "object_ground",
)
BODY_GROUP_LINKS = (
    ("left_rubber_hand", "left_wrist_yaw_link", "left_wrist_pitch_link"),
    ("right_rubber_hand", "right_wrist_yaw_link", "right_wrist_pitch_link"),
    ("left_elbow_link", "left_wrist_roll_link"), ("right_elbow_link", "right_wrist_roll_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("torso_link", "logo_link"), ("pelvis_contour_link",),
    ("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link"),
    ("right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link"),
    ("left_knee_link",), ("right_knee_link",),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"), ("head_link",),
)
REGION_LINKS = MappingProxyType(dict(zip(CHANNELS[:-1], BODY_GROUP_LINKS)))
CHANNEL_INDEX = MappingProxyType({name: index for index, name in enumerate(CHANNELS)})
LABEL_SCHEMA = "hoi_v1_region_pair_normal_impulse_v3"
GROUP_SCHEMA = "g1_contact_regions_15_v1"
GROUND, LEFT, RIGHT, OBJECT, OTHER = 0, 1, 2, 3, 4
LEFT_FOOT, RIGHT_FOOT = 5, 6
BODY_ROLES = (LEFT, RIGHT, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, LEFT_FOOT, RIGHT_FOOT, 17)
CHANNEL_ROLES = (*BODY_ROLES, GROUND)
LINK_ROLES = MappingProxyType({link: role for links, role in zip(BODY_GROUP_LINKS, BODY_ROLES) for link in links})
FOOT_LINKS = {LEFT_FOOT: "left_ankle_roll_link", RIGHT_FOOT: "right_ankle_roll_link"}
# These fixed visual/collision links may be folded into the listed physical body.
# Collider identity is retained instead of calling a welded head a torso contact.
_FIXED_PARENTS = {
    "left_rubber_hand": "left_wrist_yaw_link", "right_rubber_hand": "right_wrist_yaw_link",
    "head_link": "torso_link", "logo_link": "torso_link", "pelvis_contour_link": "pelvis",
}


def pair_channel(role_a: torch.Tensor, role_b: torch.Tensor) -> torch.Tensor:
    """Map only region-object or object-ground pairs; ground/self never substitute."""
    result = torch.full_like(role_a, -1)
    for channel, role in enumerate(CHANNEL_ROLES):
        matched = ((role_a == role) & (role_b == OBJECT)) | ((role_b == role) & (role_a == OBJECT))
        result = torch.where(matched, channel, result)
    return result


def normal_impulse_to_label(impulse, dt=0.02, threshold=1.0):
    if dt <= 0 or threshold < 0:
        raise ValueError("dt must be positive; threshold must be nonnegative")
    if not bool(torch.isfinite(impulse).all()) or bool((impulse < 0).any()):
        raise ValueError("normal impulse must be finite and nonnegative")
    force = impulse / dt
    return force, force > threshold


def aggregate_pairs(geom_role, geoms, env, normal_force, *, num_envs, substep_dt):
    """CPU-testable equivalent of the native GPU reduction, units N s."""
    if substep_dt <= 0:
        raise ValueError("substep dt must be positive")
    if geoms.ndim != 2 or geoms.shape[-1] != 2:
        raise ValueError("pair geoms must have shape [contacts,2]")
    good_geom = (geoms >= 0).all(-1) & (geoms < len(geom_role)).all(-1)
    safe = geoms.clamp(0, max(0, len(geom_role) - 1))
    channels = pair_channel(geom_role[safe[:, 0]], geom_role[safe[:, 1]])
    valid = good_geom & (channels >= 0) & (env >= 0) & (env < num_envs)
    out = torch.zeros(num_envs * len(CHANNELS), device=normal_force.device, dtype=normal_force.dtype)
    index = env.clamp(0, num_envs - 1) * len(CHANNELS) + channels.clamp_min(0)
    values = torch.where(valid, normal_force.clamp_min(0) * substep_dt, 0.0)
    out.scatter_add_(0, index, values)
    return out.reshape(num_envs, len(CHANNELS))


def aggregate_ground_pairs(geom_role, geoms, env, normal_force, *, roles, num_envs, substep_dt, partner=GROUND):
    """Separate negative-control impulses; these never enter object channels."""
    safe = geoms.clamp(0, len(geom_role) - 1)
    a, b = geom_role[safe[:, 0]], geom_role[safe[:, 1]]
    valid = (geoms >= 0).all(-1) & (geoms < len(geom_role)).all(-1) & (env >= 0) & (env < num_envs)
    result = torch.zeros(num_envs, len(roles), device=normal_force.device, dtype=normal_force.dtype)
    for column, role in enumerate(roles):
        matched = ((a == partner) & (b == role)) | ((b == partner) & (a == role))
        values = torch.where(valid & matched, normal_force.clamp_min(0) * substep_dt, 0.)
        result[:, column].scatter_add_(0, env.clamp(0, num_envs - 1), values)
    return result


def canonical_geom_link(geom_name, body_name):
    """Resolve canonical identity with an explicit allowlist for welded children.

    A misleading geom label cannot relabel a movable knee as an ankle. Fixed
    head/pelvis/hand collider names, however, survive their rigid-body merging.
    """
    def suffix(name, link):
        return bool(re.search(r"(?:^|[/_])" + re.escape(link) + r"$", name))
    bodies = [link for link in (*LINK_ROLES, "pelvis") if suffix(body_name, link)]
    body = bodies[0] if len(bodies) == 1 else None
    if body is None:
        return None
    candidates = [link for link in LINK_ROLES if re.search(
        r"(?:^|[/_])" + re.escape(link) + r"(?=$|[/_])", geom_name)]
    candidates = [link for link in candidates if link == body or _FIXED_PARENTS.get(link) == body]
    if candidates:
        # A full authored path may contain both the welded parent and child.
        # The deepest explicitly allowed source link names the collider.
        return max(candidates, key=lambda link: geom_name.rfind(link))
    return body if body in LINK_ROLES else None


def compiled_geom_links(model):
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    return [canonical_geom_link(name, bodies[int(model.geom_bodyid[i])]) for i, name in enumerate(names)]


def compiled_geom_roles(model, *, require_all_regions=False):
    """Bind active canonical colliders; recorder requires every region per world."""
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    links = compiled_geom_links(model)
    roles, envs = [], []
    for i, name in enumerate(names):
        body = bodies[int(model.geom_bodyid[i])]
        active = bool(int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))
        if not active:
            role = OTHER
        elif compiled_support_role(name, body) is not None:
            role = compiled_support_role(name, body)
        elif links[i] is not None:
            role = LINK_ROLES[links[i]]
        elif re.search(r"(?:^|[/_])Object$", body):
            role = OBJECT
        elif int(model.geom_bodyid[i]) == 0 and "ground" in name.lower():
            role = GROUND
        else:
            role = OTHER
        match = re.search(r"env[_/](\d+)", name + " " + body)
        envs.append(int(match.group(1)) if match else -1)
        roles.append(role)
    required = (*BODY_ROLES, OBJECT, GROUND) if require_all_regions else (LEFT, RIGHT, LEFT_FOOT, RIGHT_FOOT, OBJECT, GROUND)
    if any(role not in roles for role in required):
        raise RuntimeError(f"Missing native pair entities: {dict(zip(names, roles))}")
    return roles, envs, names


class SolverContactAdapter:
    """Per-instance solver adapter; preserves step inputs/outputs exactly.

    Newton's public post-step hook fires once after all four substeps. The
    pinned backend therefore exposes this explicit solver adapter rather than
    mislabelling its endpoint sensor as an interval measurement. The manager
    keeps ownership of integration, actuation, collision detection and buffers.
    Native validation uses the supported ``use_cuda_graph=False`` option.
    """
    def __init__(self, solver, recorder):
        self._wrapped = solver
        self.recorder = recorder

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def step(self, state_in, state_out, control, contacts, dt, **kwargs):
        result = self._wrapped.step(state_in, state_out, control, contacts, dt, **kwargs)
        self.recorder.after_substep(dt)
        return result


@dataclass
class PairContactRecorder:
    manager: object
    num_envs: int
    control_dt: float = 0.02
    num_substeps: int = 4
    threshold: float = 1.0

    def __post_init__(self):
        import mujoco_warp
        import warp as wp
        self.wp, self.mjw = wp, mujoco_warp
        self.solver = self.manager._solver
        if not all(hasattr(self.solver, name) for name in ("mj_model", "mjw_model", "mjw_data")):
            raise RuntimeError("HOI pair recorder requires native MuJoCo-Warp solver")
        if self.manager._cfg.use_cuda_graph:
            raise RuntimeError("Interval recorder currently requires Newton use_cuda_graph=False")
        self.roles, self.geom_envs, self.names = compiled_geom_roles(self.solver.mj_model, require_all_regions=True)
        self.links = compiled_geom_links(self.solver.mj_model)
        data = self.solver.mjw_data
        self.device = wp.to_torch(data.contact.geom).device
        self.geom_role = torch.tensor(self.roles, device=self.device, dtype=torch.long)
        self.geom_env = torch.tensor(self.geom_envs, device=self.device, dtype=torch.long)
        self.capacity = int(data.naconmax)
        self.id_tensor = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        self.ids = wp.from_torch(self.id_tensor, dtype=wp.int32)
        self.forces = wp.zeros(self.capacity, dtype=wp.spatial_vector, device=data.contact.geom.device)
        self.indices = torch.arange(self.capacity, device=self.device)
        self.impulse = torch.zeros(self.num_envs, len(CHANNELS), device=self.device)
        self.body_ground_impulse = torch.zeros(self.num_envs, len(BODY_ROLES), device=self.device)
        self.support_impulse = torch.zeros(self.num_envs, len(SUPPORT_IDS), len(BODY_ROLES) + 1, device=self.device)
        self.hand_ground_impulse = torch.zeros(self.num_envs, 2, device=self.device)
        self.foot_ground_impulse = torch.zeros(self.num_envs, 2, device=self.device)
        self.calls = 0
        nworld = int(getattr(self.solver.mjw_model, "nworld", data.efc.force.shape[0]))
        if nworld == self.num_envs:
            self.topology = "one_mujoco_world_per_environment"
        elif nworld == 1 and all(e >= 0 for e, r in zip(self.geom_envs, self.roles) if r in (*BODY_ROLES, OBJECT)):
            self.topology = "one_world_with_named_environments"
        else:
            raise RuntimeError(f"Cannot bind MuJoCo world IDs to {self.num_envs} environments (nworld={nworld})")
        if self.topology == "one_world_with_named_environments":
            for env_id in range(self.num_envs):
                present = {role for role, env in zip(self.roles, self.geom_envs) if env == env_id}
                if not set((*BODY_ROLES, OBJECT)).issubset(present):
                    raise RuntimeError(f"Missing full-body pair entities in environment {env_id}")
        self.adapter = SolverContactAdapter(self.solver, self)
        self.manager._solver = self.adapter

    def begin_interval(self):
        self.impulse.zero_()
        self.body_ground_impulse.zero_()
        self.support_impulse.zero_()
        self.hand_ground_impulse.zero_()
        self.foot_ground_impulse.zero_()
        self.calls = 0

    def after_substep(self, dt):
        if abs(float(dt) * self.num_substeps - self.control_dt) > 1e-10:
            raise RuntimeError("Native solver timestep changed from the declared control contract")
        data = self.solver.mjw_data
        self.mjw.contact_force(self.solver.mjw_model, data, self.ids, False, self.forces)
        geom = self.wp.to_torch(data.contact.geom).long()
        nactive = self.indices < self.wp.to_torch(data.nacon)[0]
        world = self.wp.to_torch(data.contact.worldid).long()
        force = self.wp.to_torch(self.forces)[:, 0]
        if self.topology == "one_world_with_named_environments":
            safe = geom.clamp(0, len(self.roles) - 1)
            ea, eb = self.geom_env[safe[:, 0]], self.geom_env[safe[:, 1]]
            compatible = (ea == eb) | (ea < 0) | (eb < 0)
            if bool((nactive & ~compatible & (force.abs() > 1e-6)).any()):
                raise RuntimeError("Native contact crosses environment boundaries; contaminated dynamics cannot enter replay")
            world = torch.maximum(ea, eb)
            nactive &= compatible
        good_geom = (geom >= 0).all(-1) & (geom < len(self.roles)).all(-1)
        if bool((nactive & (~good_geom | (world < 0) | (world >= self.num_envs))).any()):
            raise RuntimeError("Invalid active native contact identity; labels cannot enter replay")
        if not bool(torch.isfinite(force[nactive]).all()):
            raise RuntimeError("Nonfinite active native contact force; labels cannot enter replay")
        env = torch.where(nactive, world, -1)
        self.impulse.add_(aggregate_pairs(self.geom_role, geom, env, force,
                                        num_envs=self.num_envs, substep_dt=float(dt)))
        for target, roles in ((self.body_ground_impulse, BODY_ROLES),
                              (self.hand_ground_impulse, (LEFT, RIGHT)),
                              (self.foot_ground_impulse, (LEFT_FOOT, RIGHT_FOOT))):
            target.add_(aggregate_ground_pairs(self.geom_role, geom, env, force,
                        roles=roles, num_envs=self.num_envs, substep_dt=float(dt)))
        for index, partner in enumerate(SUPPORT_ROLES):
            self.support_impulse[:, index].add_(aggregate_ground_pairs(self.geom_role, geom, env, force,
                roles=(*BODY_ROLES, OBJECT), partner=partner, num_envs=self.num_envs, substep_dt=float(dt)))
        self.calls += 1

    def end_interval(self):
        if self.calls != self.num_substeps:
            raise RuntimeError(f"Expected {self.num_substeps} substeps, recorded {self.calls}")
        force, label = normal_impulse_to_label(self.impulse, self.control_dt, self.threshold)
        return {"impulse": self.impulse.clone(), "contact_force": force,
                "contact": label.float(), "contact_valid": torch.ones_like(label, dtype=torch.bool),
                "body_ground_impulse": self.body_ground_impulse.clone(),
                "support_impulse": self.support_impulse.clone(),
                "hand_ground_impulse": self.hand_ground_impulse.clone(),
                "foot_ground_impulse": self.foot_ground_impulse.clone()}

    def manifest(self):
        return {"schema": LABEL_SCHEMA, "channels": CHANNELS, "units": "normal impulse N*s; interval mean normal force N",
                "interval": "(t,t+1]", "aggregation": "sum over contact points and all physics substeps",
                "normal_force_api": "mujoco_warp.contact_force(to_world_frame=False)[0]",
                "substep_dt": self.control_dt / self.num_substeps, "num_substeps": self.num_substeps,
                "threshold_N": self.threshold, "topology": self.topology,
                "geom_names": self.names, "geom_roles": self.roles, "geom_canonical_links": self.links,
                "support_ids": list(SUPPORT_IDS), "support_roles": list(SUPPORT_ROLES),
                "support_diagnostic_channels": [*CHANNELS[:-1], "object_support"],
                "group_schema": GROUP_SCHEMA, "region_links": {k: list(v) for k, v in REGION_LINKS.items()},
                "region_collision_geom_counts": {name: self.roles.count(role) for name, role in zip(CHANNELS[:-1], BODY_ROLES)},
                "foot_body_links": {"left": FOOT_LINKS[LEFT_FOOT], "right": FOOT_LINKS[RIGHT_FOOT]},
                "foot_collision_geom_counts": {side: sum(link == side + "_ankle_roll_link" and role == foot_role for link, role in zip(self.links, self.roles))
                    for side, foot_role in (("left", LEFT_FOOT), ("right", RIGHT_FOOT))},
                "use_cuda_graph": False}

    def close(self):
        if self.manager._solver is self.adapter:
            self.manager._solver = self.solver
