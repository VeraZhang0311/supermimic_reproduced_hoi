"""Entity-pair, interval contact labels; no body net-force approximation.

The native adapter targets Newton 1.8/MuJoCo-Warp 3.15.  The public
``mujoco_warp.contact_force`` decodes pyramidal/elliptic constraint forces.
It must run after *each* solver substep: an endpoint contact read cannot
represent the preceding control interval. Native imports are lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch

CHANNELS = ("left_hand_object", "right_hand_object", "object_ground")
LABEL_SCHEMA = "hoi_v1_pair_normal_impulse_v1"
GROUND, LEFT, RIGHT, OBJECT, OTHER = 0, 1, 2, 3, 4


def pair_channel(role_a: torch.Tensor, role_b: torch.Tensor) -> torch.Tensor:
    """Return -1 for every unrelated pair, including a hand touching ground."""
    lo, hi = torch.minimum(role_a, role_b), torch.maximum(role_a, role_b)
    return torch.where((lo == LEFT) & (hi == OBJECT), 0,
                       torch.where((lo == RIGHT) & (hi == OBJECT), 1,
                                   torch.where((lo == GROUND) & (hi == OBJECT), 2, -1)))


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
    out = torch.zeros(num_envs * 3, device=normal_force.device, dtype=normal_force.dtype)
    index = env.clamp(0, num_envs - 1) * 3 + channels.clamp_min(0)
    values = torch.where(valid, normal_force.clamp_min(0) * substep_dt, 0.0)
    out.scatter_add_(0, index, values)
    return out.reshape(num_envs, 3)


def compiled_geom_roles(model):
    """Bind actual rubber-hand colliders and the literal scene ground only."""
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    roles, envs = [], []
    for i, name in enumerate(names):
        body = bodies[int(model.geom_bodyid[i])]
        active = bool(int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))
        if active and "left_rubber_hand" in name:
            role = LEFT
        elif active and "right_rubber_hand" in name:
            role = RIGHT
        elif active and ("_Object" in body or "/Object" in body):
            role = OBJECT
        elif active and int(model.geom_bodyid[i]) == 0 and "ground" in name.lower():
            role = GROUND
        else:
            role = OTHER
        match = re.search(r"env[_/](\d+)", name + " " + body)
        envs.append(int(match.group(1)) if match else -1)
        roles.append(role)
    if any(role not in roles for role in (LEFT, RIGHT, OBJECT, GROUND)):
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
        self.roles, self.geom_envs, self.names = compiled_geom_roles(self.solver.mj_model)
        data = self.solver.mjw_data
        self.device = wp.to_torch(data.contact.geom).device
        self.geom_role = torch.tensor(self.roles, device=self.device, dtype=torch.long)
        self.geom_env = torch.tensor(self.geom_envs, device=self.device, dtype=torch.long)
        self.capacity = int(data.naconmax)
        self.id_tensor = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        self.ids = wp.from_torch(self.id_tensor, dtype=wp.int32)
        self.forces = wp.zeros(self.capacity, dtype=wp.spatial_vector, device=data.contact.geom.device)
        self.indices = torch.arange(self.capacity, device=self.device)
        self.impulse = torch.zeros(self.num_envs, 3, device=self.device)
        self.hand_ground_impulse = torch.zeros(self.num_envs, 2, device=self.device)
        self.calls = 0
        nworld = int(getattr(self.solver.mjw_model, "nworld", data.efc.force.shape[0]))
        if nworld == self.num_envs:
            self.topology = "one_mujoco_world_per_environment"
        elif nworld == 1 and all(e >= 0 for e, r in zip(self.geom_envs, self.roles) if r in (LEFT, RIGHT, OBJECT)):
            self.topology = "one_world_with_named_environments"
        else:
            raise RuntimeError(f"Cannot bind MuJoCo world IDs to {self.num_envs} environments (nworld={nworld})")
        self.adapter = SolverContactAdapter(self.solver, self)
        self.manager._solver = self.adapter

    def begin_interval(self):
        self.impulse.zero_()
        self.hand_ground_impulse.zero_()
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
        env = torch.where(nactive, world, -1)
        self.impulse.add_(aggregate_pairs(self.geom_role, geom, env, force,
                                        num_envs=self.num_envs, substep_dt=float(dt)))
        safe = geom.clamp(0, len(self.roles) - 1)
        ra, rb = self.geom_role[safe[:, 0]], self.geom_role[safe[:, 1]]
        lo, hi = torch.minimum(ra, rb), torch.maximum(ra, rb)
        is_hand_ground = (lo == GROUND) & ((hi == LEFT) | (hi == RIGHT)) & (env >= 0) & (env < self.num_envs)
        is_hand_ground &= (geom >= 0).all(-1)
        index = env.clamp(0, self.num_envs - 1) * 2 + (hi - LEFT).clamp(0, 1)
        value = torch.where(is_hand_ground, force.clamp_min(0) * float(dt), 0.)
        self.hand_ground_impulse.view(-1).scatter_add_(0, index, value)
        self.calls += 1

    def end_interval(self):
        if self.calls != self.num_substeps:
            raise RuntimeError(f"Expected {self.num_substeps} substeps, recorded {self.calls}")
        force, label = normal_impulse_to_label(self.impulse, self.control_dt, self.threshold)
        return {"impulse": self.impulse.clone(), "contact_force": force,
                "contact": label.float(), "contact_valid": torch.ones_like(label, dtype=torch.bool),
                "hand_ground_impulse": self.hand_ground_impulse.clone()}

    def manifest(self):
        return {"schema": LABEL_SCHEMA, "channels": CHANNELS, "units": "normal impulse N*s; interval mean normal force N",
                "interval": "(t,t+1]", "aggregation": "sum over contact points and all physics substeps",
                "normal_force_api": "mujoco_warp.contact_force(to_world_frame=False)[0]",
                "substep_dt": self.control_dt / self.num_substeps, "num_substeps": self.num_substeps,
                "threshold_N": self.threshold, "topology": self.topology,
                "geom_names": self.names, "geom_roles": self.roles, "use_cuda_graph": False}

    def close(self):
        if self.manager._solver is self.adapter:
            self.manager._solver = self.solver
