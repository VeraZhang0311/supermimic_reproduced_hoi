"""Five v1.1 pair labels from actual hw3 solver interval normal impulses."""
from __future__ import annotations

import numpy as np
import torch

from hoi_residual.v11_contacts import _geom_roles
from supermimic.g1_runtime import _entity_mapping
from supermimic.g1_physics import CONTROL_DT, PHYSICS_DT, CONTROL_DECIMATION

CHANNELS = ("left_hand_object", "right_hand_object", "object_ground",
            "left_foot_object", "right_foot_object")
LABEL_SCHEMA = "hoi_v1_pair_normal_impulse_v2"


def geometry_roles(model, newton_model, robot_geoms, object_count):
    roles = _geom_roles(model, newton_model, robot_geoms, object_count)
    feet, _ = _entity_mapping(model, {"left_ankle_roll_link", "right_ankle_roll_link"}, "body", newton_model)
    for role, name in enumerate(("left_ankle_roll_link", "right_ankle_roll_link"), start=3):
        ids = np.flatnonzero(model.geom_bodyid == feet[name])
        if not len(ids) or (roles[ids] != -1).any():
            raise RuntimeError("Foot/object identity binding is ambiguous")
        roles[ids] = role
    import mujoco
    ground = np.flatnonzero((model.geom_bodyid == 0) & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)))
    if not len(ground):
        raise RuntimeError("Five-channel recorder requires an actual ground plane")
    roles[ground] = 5
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    if not active[roles != -1].all():
        raise RuntimeError("Inactive native collider in five-channel pair binding")
    return roles


def _kernel(wp):
    @wp.kernel
    def accumulate(active_count: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                   world: wp.array(dtype=wp.int32), role: wp.array(dtype=wp.int32),
                   wrench: wp.array(dtype=wp.spatial_vector), dt: float,
                   impulse: wp.array(dtype=wp.float32, ndim=2)):
        i = wp.tid()
        if i < active_count[0]:
            pair = geom[i]
            if pair[0] >= 0 and pair[1] >= 0:
                a, b = role[pair[0]], role[pair[1]]
                other = int(-1)
                if a == 2:
                    other = b
                elif b == 2:
                    other = a
                channel = int(-1)
                if other == 0:
                    channel = 0
                elif other == 1:
                    channel = 1
                elif other == 5:
                    channel = 2
                elif other == 3:
                    channel = 3
                elif other == 4:
                    channel = 4
                if channel >= 0:
                    wp.atomic_add(impulse, world[i], channel, wp.max(wrench[i][0], 0.0) * dt)
    return accumulate


class PairContacts:
    def __init__(self, env):
        import mujoco_warp
        import warp as wp
        self.wp, self.physics = wp, env.physics
        self.solver, self.data = env.physics.solver, env.physics.solver.mjw_data
        if self.data.efc.force.shape[0] != env.N:
            raise RuntimeError("Five-channel contact binding requires one solver world per environment")
        roles = geometry_roles(self.solver.mj_model, self.physics.manager.get_model(),
                               self.physics.geom_ids, len(env.asset["collision_meshes"]))
        self.roles = wp.array(roles, dtype=wp.int32, device=self.data.contact.geom.device)
        capacity = int(self.data.naconmax)
        self.indices = wp.array(np.arange(capacity, dtype=np.int32), device=self.roles.device)
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.roles.device)
        self.impulse = wp.zeros((env.N, 5), dtype=wp.float32, device=self.roles.device)
        self.accumulate, self.contact_force = _kernel(wp), mujoco_warp.contact_force
        self.evidence = {"channels": list(CHANNELS), "label_schema": LABEL_SCHEMA,
                         "geom_role_counts": {str(k): int((roles == k).sum()) for k in range(6)},
                         "threshold_N": 1., "substeps": CONTROL_DECIMATION}

    def _after_substep(self, dt):
        if abs(dt - PHYSICS_DT) > 1e-10:
            raise RuntimeError("Unexpected five-channel contact timebase")
        self.contact_force(self.solver.mjw_model, self.data, self.indices, False, self.forces)
        self.wp.launch(self.accumulate, dim=len(self.indices),
                       inputs=[self.data.nacon, self.data.contact.geom, self.data.contact.worldid,
                               self.roles, self.forces, dt, self.impulse], device=self.roles.device)
        self.substeps += 1

    def step(self, target):
        self.impulse.zero_()
        self.substeps = 0
        self.physics.step_target(target, after_substep=self._after_substep)
        if self.substeps != CONTROL_DECIMATION:
            raise RuntimeError("Incomplete five-channel contact interval")
        self.last_force = self.wp.to_torch(self.impulse).clone() / CONTROL_DT
        if not bool(torch.isfinite(self.last_force).all()):
            raise FloatingPointError("Nonfinite native pair-contact force")
        labels = (self.last_force > 1.).float()
        return labels, torch.ones_like(labels)
