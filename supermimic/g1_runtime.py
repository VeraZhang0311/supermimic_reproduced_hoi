"""Isaac Lab/Newton adapter for the shared hw3 profile.

Use one instance for the robot in GMT or residual HOI. Policy targets are
combined before step_target; neither caller may apply a second PD/friction.
The adapter owns native MuJoCo contact/friction arrays. Pending Newton
property notifications are followed by a pre-actuator copy so they cannot
silently reset the soft-hand material. Other scene properties remain owned
by Isaac Lab. Requires the pinned runtime's batched MuJoCo model layout.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .g1_assets import collider_role, verify_assets, load_model
from .g1_physics import (ASSET_DIR, CONTROL_DECIMATION, G1Motor, PHYSICS_DT,
                         friction_threshold, motor_of, physics_contract)
from .robot_metadata import G1_JOINT_NAMES


def _basename(name):
    return name.rsplit("/", 1)[-1]


def _entity_mapping(model, canonical_names, kind, newton_model=None):
    """Resolve identities from exact source labels and Newton's export rule.

    SolverMuJoCo exports a shape as ``f"{shape_label}_{shape_index}"`` and
    body/joint labels with '/' replaced by '_'. The source label remains a
    full USD path; repeated wrapper/mesh components are allowed, but lookup
    is never based on MuJoCo entity order or a guessed numeric suffix.
    """
    canonical_names = set(canonical_names)
    count = {"geom": model.ngeom, "body": model.nbody, "joint": model.njnt}[kind]
    accessor = getattr(model, kind)
    native = {accessor(i).name: i for i in range(count) if accessor(i).name}
    result, proof = {}, {}

    def bind(canonical, exported, rule, label=None, index=None):
        if exported not in native:
            return
        entity = native[exported]
        if canonical in result and result[canonical] != entity:
            raise RuntimeError(f"Ambiguous hw3 {kind} identity {canonical}: {proof[canonical]['mujoco_name']} and {exported}")
        result[canonical] = entity
        proof[canonical] = {"mujoco_name": exported, "rule": rule,
                            "newton_label": label, "newton_index": index}

    for name in native:
        leaf = _basename(name)
        if leaf in canonical_names:
            bind(leaf, name, "canonical USD/MJCF leaf")
    if newton_model is not None:
        attribute = {"geom": "shape_label", "body": "body_label", "joint": "joint_label"}[kind]
        for index, label in enumerate(getattr(newton_model, attribute)):
            parts = str(label).split("/")
            # A massless child body can have a canonical massive parent in
            # its path. It must never be mistaken for that parent.
            components = ([parts[-1]] if parts[-1] in canonical_names else []) if kind == "body" else [part for part in parts if part in canonical_names]
            if not components:
                continue
            if kind != "body" and len(set(components)) != 1:
                raise RuntimeError(f"Ambiguous canonical hw3 {kind} components in Newton label {label}")
            canonical = components[-1]
            exported = f"{label}_{index}" if kind == "geom" else str(label).replace("/", "_")
            bind(canonical, exported, "Newton SolverMuJoCo exact label export", str(label), index)
    missing = canonical_names - result.keys()
    if missing:
        raise RuntimeError(f"hw3 {kind} identity mismatch; unresolved {sorted(missing)}; "
                           f"actual native names sample={list(native)[:8]}")
    return result, proof


def _copy_kernels(wp):
    """Generic row copies work for scalar, vec2/3/5 and integer fields.

    Kernels, rather than a full-array restore, preserve materials belonging
    to HOI objects and the scene when Newton refreshes them at a boundary.
    """
    @wp.kernel
    def batched(source: wp.array(dtype=Any, ndim=2),
                destination: wp.array(dtype=Any, ndim=2),
                indices: wp.array(dtype=wp.int32)):
        world, item = wp.tid()
        index = indices[item]
        destination[world, index] = source[world, index]

    @wp.kernel
    def shared(source: wp.array(dtype=Any, ndim=1),
               destination: wp.array(dtype=Any, ndim=1),
               indices: wp.array(dtype=wp.int32)):
        item = wp.tid()
        index = indices[item]
        destination[index] = source[index]

    return batched, shared


def _inertia_tensor(model, body):
    import mujoco
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body]) @ rotation.T


def _audit_inertials(model, expected, bodies):
    """Float-precision checks plus two explicitly named degenerate markers."""
    report = {"mass_atol_kg": 1e-8, "mass_rtol": 2e-7, "com_atol_m": 2e-8, "com_rtol": 2e-7,
              "tensor_atol_kg_m2": 1e-9, "tensor_rtol": 2e-6, "bodies": {}}
    for name in [expected.body(i).name for i in range(1, expected.nbody) if expected.body_mass[i] > 0]:
        i, j = bodies[name], expected.body(name).id
        actual, source = _inertia_tensor(model, i), _inertia_tensor(expected, j)
        np.testing.assert_allclose(model.body_mass[i], expected.body_mass[j], atol=1e-8, rtol=2e-7,
                                   err_msg=f"hw3 body mass: {name}")
        np.testing.assert_allclose(model.body_ipos[i], expected.body_ipos[j], atol=2e-8, rtol=2e-7,
                                   err_msg=f"hw3 body COM: {name}")
        status, correction = "source preserved to float precision", None
        zero_marker = (name in {"LL_FOOT", "LR_FOOT"} and
                       abs(float(expected.body_mass[j]) - 1e-9) < 1e-15 and
                       int(expected.body_jntnum[j]) == 0 and np.max(np.abs(source)) < 1e-14)
        if zero_marker and np.max(np.abs(actual - source)) > 1e-14:
            # Installed Newton geometry/inertia.py uses eigenvalue floor
            # 1e-10 plus an absolute adjustment of 1e-6 for a zero tensor.
            # This exception is only for the two fixed FOOT markers, not
            # the physical ankle_roll bodies carrying their foot colliders.
            target = np.eye(3) * (1e-10 + 1e-6)
            np.testing.assert_allclose(actual, target, atol=5e-13, rtol=2e-6,
                                       err_msg=f"hw3 named fixed-marker regularization: {name}")
            status = "named Newton zero-inertia marker regularization; differs from source"
            correction = {"source": "Newton geometry/inertia.py validate_and_correct_inertia",
                          "eigenvalue_floor": 1e-10, "absolute_adjustment": 1e-6,
                          "expected_tensor": target.tolist(), "target_atol": 5e-13, "target_rtol": 2e-6}
        else:
            np.testing.assert_allclose(actual, source, atol=1e-9, rtol=2e-6,
                                       err_msg=f"hw3 full body inertia tensor: {name}")
        report["bodies"][name] = {"status": status,
                                   "source_mass_kg": float(expected.body_mass[j]), "native_mass_kg": float(model.body_mass[i]),
                                   "mass_abs_error_kg": abs(float(model.body_mass[i]-expected.body_mass[j])),
                                   "source_com_m": expected.body_ipos[j].tolist(), "native_com_m": model.body_ipos[i].tolist(),
                                   "com_max_abs_error_m": float(np.max(np.abs(model.body_ipos[i]-expected.body_ipos[j]))),
                                   "source_tensor_kg_m2": source.tolist(), "native_tensor_kg_m2": actual.tolist(),
                                   "tensor_max_abs_error_kg_m2": float(np.max(np.abs(actual-source))),
                                   "regularization": correction}
    return report


def audit_model(model, asset_dir=ASSET_DIR, newton_model=None):
    """Check the compiled runtime geometry/inertias against prepared MJCF.

    World objects/ground are allowed. Compare each named robot collider,
    including welded local transforms through its world pose at q=0.
    """
    import mujoco
    from .collision import audit_self_collision
    receipt = verify_assets(asset_dir)
    expected = load_model(asset_dir / "g1_hw3.xml")
    names, geom_proof = _entity_mapping(model, receipt["collision_roles"], "geom", newton_model)
    joints, joint_proof = _entity_mapping(model, G1_JOINT_NAMES, "joint", newton_model)
    required_bodies = {expected.body(i).name for i in range(1, expected.nbody) if expected.body_mass[i] > 0}
    required_bodies.add("pelvis")
    bodies, body_proof = _entity_mapping(model, required_bodies, "body", newton_model)
    d, e = mujoco.MjData(model), mujoco.MjData(expected)
    # Imported startup positions are not the mechanical zero pose. Compare
    # install transforms at the same named-joint coordinates explicitly.
    for name in G1_JOINT_NAMES:
        d.qpos[model.jnt_qposadr[joints[name]]] = 0.
        e.qpos[expected.joint(name).qposadr[0]] = 0.
    mujoco.mj_forward(model, d); mujoco.mj_forward(expected, e)
    # The spawned root can have an initial world translation/orientation.
    actual_pelvis = bodies["pelvis"]
    ep = expected.body("pelvis").id
    R = d.xmat[actual_pelvis].reshape(3, 3); Re = e.xmat[ep].reshape(3, 3)
    for name, i in names.items():
        j = expected.geom(name).id
        if int(model.geom_type[i]) != int(expected.geom_type[j]):
            raise RuntimeError(f"hw3 geom type changed: {name}")
        kind = int(model.geom_type[i])
        # MuJoCo size slots have type-specific semantics. Newton preserves
        # an unused third capsule/cylinder slot; it is not another radius.
        # Mesh dimensions come from vertices, never geom_size. Their full
        # installed hull support is checked below instead.
        size_dimensions = {int(mujoco.mjtGeom.mjGEOM_SPHERE): 1,
                           int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2,
                           int(mujoco.mjtGeom.mjGEOM_CYLINDER): 2,
                           int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 3,
                           int(mujoco.mjtGeom.mjGEOM_BOX): 3,
                           int(mujoco.mjtGeom.mjGEOM_MESH): 0}
        if kind not in size_dimensions:
            raise RuntimeError(f"Unsupported hw3 robot collider type for exact audit: {name}, type={kind}")
        dimension = size_dimensions[kind]
        if dimension:
            np.testing.assert_allclose(model.geom_size[i, :dimension], expected.geom_size[j, :dimension], atol=2e-6, rtol=1e-5)
        if kind != int(mujoco.mjtGeom.mjGEOM_MESH):
            np.testing.assert_allclose(R.T @ (d.geom_xpos[i]-d.xpos[actual_pelvis]),
                                       Re.T @ (e.geom_xpos[j]-e.xpos[ep]), atol=3e-5, rtol=1e-5,
                                       err_msg=f"hw3 installed primitive center: {name}")
        if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis = R.T @ d.geom_xmat[i].reshape(3, 3)[:, 2]
            expected_axis = Re.T @ e.geom_xmat[j].reshape(3, 3)[:, 2]
            # Rotation about the symmetry axis and reversal of that axis do
            # not change these primitives. The installed axis itself must.
            np.testing.assert_allclose(np.outer(axis, axis), np.outer(expected_axis, expected_axis), atol=3e-5, rtol=1e-5)
        # A mesh importer may re-center/reorient vertices. Compare its world
        # hull point cloud instead of interpreting file vertices as colliders.
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            def points(m, data, g):
                mid = m.geom_dataid[g]; start = m.mesh_vertadr[mid]; count = m.mesh_vertnum[mid]
                p = m.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                return p
            a = (points(model, d, i)-d.xpos[actual_pelvis]) @ R
            b = (points(expected, e, j)-e.xpos[ep]) @ Re
            # Hull vertices may be pruned by the importer; both support
            # functions must agree in sampled directions (no visual-only test).
            rng = np.random.default_rng(0)
            axes = np.concatenate((np.eye(3), -np.eye(3), rng.normal(size=(96, 3))))
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            np.testing.assert_allclose((a @ axes.T).max(0), (b @ axes.T).max(0), atol=3e-5, rtol=1e-5,
                                       err_msg=f"hw3 installed mesh hull support: {name}")
        if not model.geom_contype[i] or not model.geom_conaffinity[i]:
            raise RuntimeError(f"hw3 collider disabled: {name}")
    dofs = [int(model.jnt_dofadr[joints[n]]) for n in G1_JOINT_NAMES]
    np.testing.assert_allclose(model.dof_armature[dofs], [motor_of(n)[2] for n in G1_JOINT_NAMES], rtol=1e-5)
    np.testing.assert_allclose(model.dof_damping[dofs], 0, atol=1e-7)
    for name in G1_JOINT_NAMES:
        j, k = joints[name], expected.joint(name).id
        np.testing.assert_allclose(model.jnt_range[j], expected.jnt_range[k], atol=1e-6)
        if float(model.jnt_stiffness[j]) != 0.:
            raise RuntimeError(f"hw3 found a passive joint spring: {name}")
    inertial_report = _audit_inertials(model, expected, bodies)
    return {"profile": receipt["profile"], "collider_count": len(names),
            "geometry_support_tolerance_m": 3e-5, "self_collision": audit_self_collision(model),
            "inertial_audit": inertial_report,
            "identity_mapping": {"geoms": geom_proof, "bodies": body_proof, "joints": joint_proof}}, names, dofs, expected


class G1PhysicsRuntime:
    def __init__(self, sim, robot, scene, randomize=True, asset_dir=ASSET_DIR):
        import warp as wp
        self.wp, self.sim, self.robot, self.scene = wp, sim, robot, scene
        self.manager = sim.backend_manager
        self.solver = self.manager._solver
        self.N = robot.num_instances
        self.device = sim.cfg.device
        self.randomize = randomize
        if list(robot.joint_names) != G1_JOINT_NAMES:
            raise RuntimeError("Reorder G1 commands by name; implicit joint-order conversion is forbidden")
        if abs(self.manager.get_physics_dt() - PHYSICS_DT) > 1e-9:
            raise RuntimeError("hw3 must recompute explicit torque every 5 ms")
        self.report, self.geom_ids, self.dofs, self.template = audit_model(self.solver.mj_model, asset_dir, self.manager.get_model())
        self.contract = physics_contract(asset_dir)
        self._initialize_solver_timestep()
        self.graph_scheduling = self._initialize_graph_scheduling()
        self.report["graph_scheduling"] = self.graph_scheduling
        self.report["solver_options"] = self.audit_solver_options()
        self.motor = G1Motor(self.N, self.device, randomize=randomize)
        self._buffers, self._arrays, self._destination_owners = {}, {}, {}
        self._copy_batched, self._copy_shared = _copy_kernels(wp)
        self._initialize_floor_materials()
        self._bind_native_fields()
        self.report["external_contact_priority"] = self.audit_external_contact_priority()
        from .g1_contactmix import install_hand_contact_multiply
        import mujoco
        hand_ids = [i for name, i in self.geom_ids.items() if collider_role(name) != "body"]
        world_planes = np.flatnonzero((self.solver.mj_model.geom_bodyid == 0) &
                                     (self.solver.mj_model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)))
        self.contactmix = install_hand_contact_multiply(self.solver.mjw_model, hand_ids,
                                                       ground_geom_ids=world_planes)
        self.report["hand_contact_mix"] = self.contactmix.contract
        self.manager.register_pre_actuator_callback(self._restore_native_fields)

    def _initialize_solver_timestep(self):
        # SolverMuJoCo.step sets mj_model.opt.timestep=dt on its CPU path
        # and mjw_model.opt.timestep.fill_(dt) on its Warp path. Before the
        # first step both can retain the inactive MuJoCo .002 default.
        interval = float(self.manager.get_solver_dt())
        if abs(interval - PHYSICS_DT) > 1e-9:
            raise RuntimeError(f"hw3 actual solver interval must be 5 ms, got {interval}")
        self.solver.mj_model.opt.timestep = interval
        self.solver.mjw_model.opt.timestep.fill_(interval)

    def _initialize_graph_scheduling(self):
        """Select MJWarp's supported scheduler before its first CUDA capture.

        MJWarp put_model defaults graph_conditional=True and only warns on
        older drivers. Warp's eager loop can run there, but graph capture
        rejects conditional nodes. This documented option changes scheduling
        only; solver iterations, contact settings and integration stay intact.
        Warp 1.17's capability query takes no device argument. CPU execution
        needs no CUDA capability query and keeps its existing option.
        """
        device = self.wp.get_device(self.device)
        opt = self.solver.mjw_model.opt
        original = bool(opt.graph_conditional)
        supported = bool(self.wp.is_conditional_graph_supported()) if device.is_cuda else None
        if device.is_cuda and not supported:
            opt.graph_conditional = False
        return {"device": str(device), "is_cuda": bool(device.is_cuda),
                "conditional_graph_supported": supported,
                "original_graph_conditional": original,
                "selected_graph_conditional": bool(opt.graph_conditional),
                "fallback_applied": bool(device.is_cuda and not supported and original),
                "scope": "MuJoCo Warp graph scheduling only; physical solver options unchanged"}

    def audit_external_contact_priority(self):
        """The fixed hand material requires lower-priority scene colliders."""
        model = self.solver.mj_model
        priority = self.wp.to_torch(self.solver.mjw_model.geom_priority)
        owned = list(self.geom_ids.values())
        owned_set = set(owned)
        explicit = set()
        for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True):
            if int(a) in owned_set:
                explicit.add(int(b))
            if int(b) in owned_set:
                explicit.add(int(a))
        checked = []
        for i in range(model.ngeom):
            if i in owned_set:
                continue
            active = (i in explicit or bool(np.any(
                (model.geom_contype[owned] & model.geom_conaffinity[i]) |
                (model.geom_conaffinity[owned] & model.geom_contype[i]))))
            if not active:
                continue
            value = int(priority[i])
            if value >= 1:
                raise RuntimeError(f"hw3 external collider {model.geom(i).name} has priority {value}; "
                                   "set object/ground priority=0 to preserve the soft-hand normal material")
            checked.append({"name": model.geom(i).name, "priority": value})
        return {"requirement": "external priority below hand priority=1; use object/ground priority=0",
                "checked": checked}

    def _initialize_floor_materials(self):
        """Keep the Newton source of truth consistent across later resyncs."""
        from newton import GeoType, ModelFlags
        model = self.manager.get_model()
        shape_body = self.wp.to_torch(model.shape_body)
        shape_type = self.wp.to_torch(model.shape_type)
        mu = self.wp.to_torch(model.shape_material_mu)
        world_planes = (shape_body < 0) & (shape_type == int(GeoType.PLANE))
        mu[world_planes] = .3 if self.randomize else 1.
        self.manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def audit_solver_options(self):
        """Check CPU and actual Warp solver options against the shared profile."""
        import mujoco
        from .g1_physics import CONTACT_CONE, CONTACT_IMPRATIO
        expected_cone = int(getattr(mujoco.mjtCone, "mjCONE_" + CONTACT_CONE.upper()))
        report = {}
        for name, model in (("mujoco_template", self.solver.mj_model), ("mujoco_warp", self.solver.mjw_model)):
            opt = model.opt
            if hasattr(opt, "impratio"):
                ratios = np.asarray(opt.impratio).reshape(-1)
            else:
                # MuJoCo Warp 3.15+ stores the transformed value used by its
                # contact solver, not an opt.impratio Python float.
                inverse_sqrt = self.wp.to_torch(opt.impratio_invsqrt).detach().cpu().numpy()
                ratios = 1. / np.square(inverse_sqrt)
            if int(opt.cone) != expected_cone or not np.allclose(ratios, CONTACT_IMPRATIO, atol=2e-6, rtol=1e-6):
                raise RuntimeError(f"hw3 solver options differ in {name}: cone={opt.cone}, impratio={ratios.tolist()}")
            if name == "mujoco_template":
                timesteps = np.asarray(opt.timestep).reshape(-1)
            else:
                timesteps = self.wp.to_torch(opt.timestep).detach().cpu().numpy()
            if not np.allclose(timesteps, PHYSICS_DT, atol=1e-9, rtol=1e-6):
                raise RuntimeError(f"hw3 solver timestep differs in {name}: {timesteps.tolist()}")
            report[name] = {"cone": CONTACT_CONE, "impratio": ratios.tolist(), "timestep": timesteps.tolist()}
            if name == "mujoco_warp":
                actual = bool(opt.graph_conditional)
                selected = getattr(self, "graph_scheduling", None)
                if selected is not None and actual != selected["selected_graph_conditional"]:
                    raise RuntimeError("MuJoCo Warp graph scheduler changed after capability selection")
                report[name]["graph_conditional"] = actual
        return report

    def _bind_native_fields(self):
        # The Newton importer must not retain an active position/velocity servo.
        m = self.solver.mj_model
        robot_joints = {int(m.dof_jntid[d]) for d in self.dofs}
        import mujoco
        joint_transmissions = {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) in joint_transmissions and int(m.actuator_trnid[a, 0]) in robot_joints:
                if np.any(m.actuator_biasprm[a, :3] != 0):
                    raise RuntimeError("hw3 found a second servo behind the explicit PD controller")
        fields = ("geom_friction", "geom_solref", "geom_solimp", "geom_priority", "geom_condim", "geom_margin", "geom_gap", "dof_frictionloss")
        # These are topology fields in MuJoCo Warp, NOT world-batched arrays.
        self._shared_fields = {"geom_priority", "geom_condim"}
        for name in fields:
            original = getattr(self.solver.mjw_model, name)
            view = self.wp.to_torch(original)
            base = getattr(m, name)
            if name in self._shared_fields:
                if tuple(view.shape) != tuple(base.shape):
                    raise RuntimeError(f"Unsupported shared hw3 field layout: {name} {tuple(view.shape)}")
                values = view.clone().contiguous()
            else:
                if tuple(view.shape[1:]) != tuple(base.shape) or view.shape[0] not in (1, self.N):
                    raise RuntimeError(f"Unsupported batched hw3 field layout: {name} {tuple(view.shape)}")
                values = view.clone().contiguous()
                if values.shape[0] == 1 and self.N > 1:
                    values = values.repeat(self.N, *([1]*base.ndim))
            self._buffers[name] = values
            # Stable source and destination allocations are retained for graph
            # capture. Every step restores them after pending Newton resyncs.
            self._destination_owners[name] = values.clone()
            array = self.wp.from_torch(self._destination_owners[name], dtype=original.dtype)
            setattr(self.solver.mjw_model, name, array)
            self._arrays[name] = self.wp.from_torch(values, dtype=original.dtype)
        for name, i in self.geom_ids.items():
            j = self.template.geom(name).id
            for field in fields[:-1]:
                value = getattr(self.template, field)[j]
                value = torch.as_tensor(value.copy() if hasattr(value, "copy") else value,
                                        device=self.device, dtype=self._buffers[field].dtype)
                if field in self._shared_fields:
                    self._buffers[field][i] = value
                else:
                    self._buffers[field][:, i] = value
        self._geom_indices = self.wp.array(list(self.geom_ids.values()), dtype=self.wp.int32, device=self.device)
        self._dof_indices = self.wp.array(self.dofs, dtype=self.wp.int32, device=self.device)
        # Native max-pair mixing should not replace randomized body mu<1 by
        # an implicit default floor mu=1. Match the existing GMT convention.
        floor_mu = .3 if self.randomize else 1.
        floor_ids = np.flatnonzero((m.geom_bodyid == 0) & (m.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)))
        friction = self.wp.to_torch(self.solver.mjw_model.geom_friction)
        for i in floor_ids:
            friction[:, int(i), 0] = floor_mu
            friction[:, int(i), 1:] = 0.
        self.set_surface_friction(torch.ones(self.N, device=self.device))
        self.update_friction(torch.zeros(self.N, 29, device=self.device))
        self._restore_native_fields()

    def _restore_native_fields(self):
        for name, source in self._arrays.items():
            dest = getattr(self.solver.mjw_model, name)
            if dest.shape != source.shape:
                raise RuntimeError(f"Newton resync changed hw3 field allocation: {name}")
            indices = self._dof_indices if name == "dof_frictionloss" else self._geom_indices
            if name in self._shared_fields:
                self.wp.launch(self._copy_shared, dim=len(self.geom_ids), inputs=[source, dest, indices], device=self.device)
            else:
                self.wp.launch(self._copy_batched, dim=(self.N, len(self.dofs) if name == "dof_frictionloss" else len(self.geom_ids)),
                               inputs=[source, dest, indices], device=self.device)

    def readback(self):
        """Read actual solver arrays after a boundary, for native validation."""
        report = {}
        for name, expected in self._buffers.items():
            actual = self.wp.to_torch(getattr(self.solver.mjw_model, name))
            ids = self.dofs if name == "dof_frictionloss" else list(self.geom_ids.values())
            if name in self._shared_fields:
                delta = (actual[ids] - expected[ids]).abs()
            else:
                delta = (actual[:, ids] - expected[:, ids]).abs()
            maximum = float(delta.max()) if delta.numel() else 0.
            if maximum > 2e-6 or not bool(torch.isfinite(delta).all()):
                raise RuntimeError(f"hw3 native readback changed: {name}, max error {maximum}")
            report[name] = {"shape": list(actual.shape), "max_abs_error": maximum}
        return report

    def set_surface_friction(self, dynamic_mu):
        if dynamic_mu.shape != (self.N,) or not bool(torch.isfinite(dynamic_mu).all()) or bool((dynamic_mu < 0).any()):
            raise ValueError("Invalid per-environment surface friction")
        for name, i in self.geom_ids.items():
            hand = collider_role(name) != "body"
            self._buffers["geom_friction"][:, i, 0] = 1. if hand else dynamic_mu
            self._buffers["geom_friction"][:, i, 1:] = 0

    def update_friction(self, qd):
        self._buffers["dof_frictionloss"][:, self.dofs] = friction_threshold(qd, self.motor.friction_scale)

    def reset(self, ids, pose):
        self.motor.reset(ids, pose)
        self.robot.set_joint_effort_target_index(target=torch.zeros(len(ids), 29, device=self.device), env_ids=ids)

    def actuator_state(self):
        """Exact controller state at a 50 Hz boundary, including reset state."""
        from .actuator_state import export_motor_state
        return export_motor_state(self.motor)

    def step_target(self, target, after_substep=None):
        """Advance one 50 Hz interval; optional HOI impulse accumulator at 200 Hz."""
        self.contactmix.ensure_installed()
        for _ in range(CONTROL_DECIMATION):
            q, qd = self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch
            effort = self.motor.compute(target, q, qd)
            # compute returns a fresh tensor; keep the exact submitted command
            # for native delivery checks without adding a per-step copy.
            self.last_motor_effort = effort
            self.update_friction(qd)
            self.robot.set_joint_effort_target_index(target=effort)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            if not all(bool(torch.isfinite(x).all()) for x in (self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch)):
                raise FloatingPointError("nonfinite hw3 substep state")
            overflow = self.wp.to_torch(self.solver.mjw_data.overflow)
            if bool((overflow != 0).any()):
                raise RuntimeError("native contact/constraint buffer overflow")
            if after_substep is not None:
                after_substep(PHYSICS_DT)
