"""Read native left/right hand–object contacts over a complete control interval.

v1.1 counts a hand as touching when its summed normal impulse divided by the
20 ms control interval exceeds 1 N. Its native matcher selects rubber-hand
geoms only. The hw3 robot splits each rubber hand into palm, fingers and thumb.
"""
from __future__ import annotations

import numpy as np
import torch

from supermimic.g1_physics import CONTROL_DECIMATION, CONTROL_DT, PHYSICS_DT
from supermimic.g1_runtime import _entity_mapping


def _geom_roles(model, newton_model, robot_geoms, expected_object_geoms, *, hand_names=None):
    """Map compiled geometry to -1 (other), 0/1 (left/right hand) or 2 (object)."""
    roles = np.full(model.ngeom, -1, dtype=np.int32)
    for channel, side in enumerate(("left", "right")):
        if hand_names is not None:
            ids = [robot_geoms[name] for name in hand_names[side]]
            if not ids or len(set(ids)) != len(ids) or (roles[ids] != -1).any():
                raise RuntimeError(f"Ambiguous native {side} hand collider binding")
            roles[ids] = channel
            continue
        for part in ("palm", "fingers", "thumb"):
            suffix = f"_{side}_rubber_hand_{part}_hull"
            matches = [index for name, index in robot_geoms.items() if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one compiled {side} rubber {part} collider, got {matches}")
            roles[matches[0]] = channel
    bodies, _ = _entity_mapping(model, {"Object"}, "body", newton_model)
    object_ids = np.flatnonzero(model.geom_bodyid == bodies["Object"])
    if len(object_ids) != expected_object_geoms or not len(object_ids) or (roles[object_ids] != -1).any():
        raise RuntimeError("Unexpected native hand/object collider binding")
    roles[object_ids] = 2
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    if not active[roles != -1].all():
        raise RuntimeError("Inactive native collider in v1.1 hand/object binding")
    return roles


def _impulse_kernel(wp):
    @wp.kernel
    def accumulate(
        active_count: wp.array(dtype=wp.int32),
        geom: wp.array(dtype=wp.vec2i),
        world: wp.array(dtype=wp.int32),
        role: wp.array(dtype=wp.int32),
        wrench: wp.array(dtype=wp.spatial_vector),
        dt: float,
        impulse: wp.array(dtype=wp.float32, ndim=2),
    ):
        index = wp.tid()
        if index >= active_count[0]:
            return
        pair = geom[index]
        if pair[0] < 0 or pair[1] < 0:
            return
        a, b = role[pair[0]], role[pair[1]]
        if a == 2 and (b == 0 or b == 1):
            wp.atomic_add(impulse, world[index], b, wp.max(wrench[index][0], 0.0) * dt)
        elif b == 2 and (a == 0 or a == 1):
            wp.atomic_add(impulse, world[index], a, wp.max(wrench[index][0], 0.0) * dt)

    return accumulate


class HandObjectContacts:
    """Read-only MJWarp recorder using the existing 200 Hz motor callback."""

    def __init__(self, env):
        import mujoco_warp
        import warp as wp

        self.physics = env.physics
        self.solver = env.physics.solver
        self.wp = wp
        self.contact_force = mujoco_warp.contact_force
        self.data = self.solver.mjw_data
        if self.data.efc.force.shape[0] != env.N:
            raise RuntimeError("v1.1 contact binding requires one solver world per environment")
        roles = _geom_roles(self.solver.mj_model, self.physics.manager.get_model(),
                            self.physics.geom_ids, len(env.asset["collision_meshes"]))
        self.device = self.data.contact.geom.device
        self.roles = wp.array(roles, dtype=wp.int32, device=self.device)
        capacity = int(self.data.naconmax)
        self.indices = wp.array(np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self.impulse = wp.zeros((env.N, 2), dtype=wp.float32, device=self.device)
        self.accumulate = _impulse_kernel(wp)

    def _after_substep(self, dt):
        if abs(dt - PHYSICS_DT) > 1e-10:
            raise RuntimeError("v1.1 contact interval received an unexpected solver timestep")
        data = self.data
        self.contact_force(self.solver.mjw_model, data, self.indices, False, self.forces)
        self.wp.launch(self.accumulate, dim=self.indices.shape[0],
                       inputs=[data.nacon, data.contact.geom, data.contact.worldid, self.roles,
                               self.forces, dt, self.impulse], device=self.device)
        self.substeps += 1

    def step(self, target):
        self.impulse.zero_()
        self.substeps = 0
        self.physics.step_target(target, after_substep=self._after_substep)
        if self.substeps != CONTROL_DECIMATION:
            raise RuntimeError(f"Incomplete v1.1 contact interval: {self.substeps} substeps")
        force = self.wp.to_torch(self.impulse) / CONTROL_DT
        if not torch.isfinite(force).all():
            raise FloatingPointError("Nonfinite hand/object contact force")
        contact = force > 1.0
        return contact.float(), torch.ones_like(contact)
