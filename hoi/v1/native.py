"""Independent HOI native scene and joint robot/object rollout contract.

Importing this module does not initialize IsaacLab. Instantiate HOITrackingSim
inside the launcher context. Geometry is fixed for an environment's lifetime;
reset sampling is restricted to its exact asset/scale/profile group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from supermimic.mdp import flatten_state
from supermimic.motion import CONTROL_DT
from supermimic.rotations import mat_to_quat, quat_to_mat
from .contacts import GROUND, LEFT, OBJECT, RIGHT, PairContactRecorder
from .state import flatten_object
from .probes import palm_patch_points, _write_palm_qa

LEGACY_CAPACITY = {"nconmax": 128, "njmax": 500}
HOI_CAPACITY = {"nconmax": 512, "njmax": 2048}


def configure_hoi_capacity(config):
    """Increase only HOI allocation limits; retain the legacy source config."""
    from copy import deepcopy
    result = deepcopy(config)
    result.physics.use_cuda_graph = False
    for name, value in HOI_CAPACITY.items():
        setattr(result.physics.solver_cfg, name, value)
    return result


def compiled_convex_mesh(model, mesh_id):
    """Read the actual collision hull facets from MuJoCo's native graph.

    ``mesh_face`` is a visual surface and is not the collision authority.
    Graph layout is documented at the linked primary API reference. Global
    vertex IDs are relative to this full mesh, before unused vertices are cut.
    """
    import hashlib
    import numpy as np
    address = int(model.mesh_graphadr[mesh_id])
    if address < 0:
        raise RuntimeError("Native mesh has no collision convex graph; refusing a visual-mesh approximation")
    graph = np.asarray(model.mesh_graph)
    nv, nf = map(int, graph[address:address + 2])
    vertex_offset, vertex_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    facet_offset = address + 2 + 3 * nv + 3 * nf
    global_faces = graph[facet_offset:facet_offset + 3 * nf].reshape(nf, 3).astype(np.int64)
    referenced = np.unique(global_faces)
    if nv < 4 or nf < 4 or len(referenced) != nv or int(referenced.min()) < 0 or int(referenced.max()) >= vertex_count:
        raise RuntimeError("Malformed native convex-hull graph")
    original_vertices = np.asarray(model.mesh_vert[vertex_offset:vertex_offset + vertex_count])
    vertices = original_vertices[referenced].astype(np.float64)
    faces = np.searchsorted(referenced, global_faces)
    record = graph[address:facet_offset + 3 * nf]
    evidence = {"source": "mjModel.mesh_graph native collision hull facets", "graph_address": address,
                "reference": "https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#convex-hulls",
                "raw_vertex_count": vertex_count, "raw_face_count": face_count, "hull_vertex_count": nv, "hull_face_count": nf,
                "source_vertex_ids": referenced.tolist(), "graph_record_sha256": hashlib.sha256(record.tobytes()).hexdigest(),
                "full_mesh_vertex_buffer_sha256": hashlib.sha256(original_vertices.tobytes()).hexdigest()}
    return vertices, faces, evidence


def link_to_com_velocity(R, v_link, omega_world, local_com):
    offset = (R @ local_com.unsqueeze(-1)).squeeze(-1)
    return v_link + torch.linalg.cross(omega_world, offset, dim=-1)


def com_to_link_velocity(R, v_com, omega_world, local_com):
    offset = (R @ local_com.unsqueeze(-1)).squeeze(-1)
    return v_com - torch.linalg.cross(omega_world, offset, dim=-1)


def shallow_support_contact(palm_world, outward_normal_world, object_offsets_world, penetration=.002):
    """Place a convex-union support vertex against a measured palm surface."""
    normal = outward_normal_world
    if penetration <= 0 or not torch.allclose(normal.norm(dim=-1), torch.ones_like(normal[..., 0]), atol=2e-6, rtol=0):
        raise ValueError("shallow contact needs a positive penetration and unit outward normals")
    index = (object_offsets_world * normal[:, None]).sum(-1).argmin(-1)
    support = object_offsets_world[torch.arange(len(index), device=index.device), index]
    return palm_world - support - penetration * normal, index


def physical_context(mass, com, inertia, friction, ground_friction):
    """12D context from effective native values; inertia is in object axes."""
    entries = torch.stack((inertia[..., 0, 0], inertia[..., 1, 1], inertia[..., 2, 2],
                           inertia[..., 0, 1], inertia[..., 0, 2], inertia[..., 1, 2]), -1)
    return torch.cat((mass.unsqueeze(-1), com, entries, friction.unsqueeze(-1), ground_friction.unsqueeze(-1)), -1)


def validate_physical_profile(profile):
    mass = float(profile["mass"])
    com = torch.as_tensor(profile["com"], dtype=torch.float64)
    inertia = torch.as_tensor(profile["inertia"], dtype=torch.float64).reshape(3, 3)
    friction, ground = float(profile["friction"]), float(profile["ground_friction"])
    if not torch.isfinite(torch.cat((com, inertia.flatten(), torch.tensor([mass, friction, ground])))).all():
        raise ValueError("object physical profile contains nonfinite values")
    if mass <= 0 or friction < 0 or ground < 0 or com.shape != (3,):
        raise ValueError("invalid mass/COM/friction")
    if not torch.allclose(inertia, inertia.T, rtol=0, atol=1e-10) or (torch.linalg.eigvalsh(inertia) <= 0).any():
        raise ValueError("object inertia must be symmetric positive definite")
    eig = torch.linalg.eigvalsh(inertia)
    if eig[-1] > eig[:2].sum() + 1e-8:
        raise ValueError("object inertia violates the rigid-body triangle inequality")
    return mass, com, inertia, friction, ground








def prepare_object_usd(asset, output):
    """Flatten an independent copy; source USD is read only and already scaled."""
    import numpy as np
    from pxr import Gf, Usd, UsdPhysics
    from scipy.spatial.transform import Rotation
    from supermimic.physics import sha
    source = Path(asset["native_asset"])
    if "native_asset_sha256" not in asset or sha(source) != asset["native_asset_sha256"]:
        raise ValueError("native object asset requires a matching SHA256 binding")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "object.usda"
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Unable to open object USD {source}")
    stage.Flatten().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError("v1 requires exactly one object rigid body")
    mass, com, inertia, _, _ = validate_physical_profile(asset["physical_profile"])
    mass_api = UsdPhysics.MassAPI.Apply(bodies[0])
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    values, axes = np.linalg.eigh(inertia.numpy())
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    rigid = UsdPhysics.RigidBodyAPI(bodies[0])
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders or any(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders):
        raise ValueError("bound collision recipe must contain enabled native colliders")
    stage.GetRootLayer().Save()
    return path


def make_hoi_sim(device, args):
    from supermimic.env import make_sim
    return configure_hoi_capacity(make_sim(device, args))


class HOITrackingSim:
    def __init__(self, sim_cfg, num_envs, motion, mdp, asset, output, *, gamma=.99,
                 eval_mode=False, reset_noise=True, max_episode_steps=500, robot_asset=None):
        import isaaclab.sim as sim_utils
        import warp as wp
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.utils import configclass
        from supermimic.env import TrackingSceneCfg, TrackingSim
        asset_id = int(asset["asset_id"])
        compatible = motion.compatible_bins(asset_id)
        if not len(compatible):
            raise ValueError("worker asset has no compatible motion bins")
        usd = prepare_object_usd(asset, Path(output) / "object_asset")

        @configclass
        class HOISceneCfg(TrackingSceneCfg):
            object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd), rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0., angular_damping=0.)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(4., 4., 2.)))

        def scene_factory(**kwargs):
            config = HOISceneCfg(**kwargs)
            if robot_asset is not None:
                from supermimic.physics import load_hand_asset, sha
                if sha(robot_asset["path"]) != robot_asset["sha256"]:
                    raise ValueError("Bundle's fixed robot hand overlay bytes changed")
                load_hand_asset(robot_asset["path"])
                config.robot.spawn.usd_path = str(robot_asset["path"])
            return config

        class AssetTrackingSim(TrackingSim):
            def sample_frames(this, n):
                m = this.motion
                bins = compatible
                soft = torch.softmax(-25 * this.bin_G[bins] / this.G_max, 0)
                this.bin_w_avg[bins] = .999 * this.bin_w_avg[bins] + .001 * soft
                avg = this.bin_w_avg[bins]
                probs = .5 / len(bins) + .4 * soft + .1 * avg / avg.sum()
                sampled = bins[torch.multinomial(probs, n, replacement=True)]
                start = m.bin_start[sampled]
                end = torch.minimum(start + 50, m.clip_end[m.bin_clip[sampled]])
                # Last-frame bins have no outgoing transition; use clip-1.
                start = torch.minimum(start, m.clip_end[m.bin_clip[sampled]] - 1)
                frame = start + (torch.rand(n, device=this.device) * (end - start).float()).long()
                return frame, sampled

        self.env = AssetTrackingSim(sim_cfg, num_envs, motion, mdp.human, gamma=gamma, eval_mode=eval_mode,
                                   max_episode_steps=max_episode_steps, reset_noise=reset_noise, scene_factory=scene_factory)
        self.device, self.N, self.motion, self.mdp = self.env.device, num_envs, motion, mdp
        self.asset = asset
        self.output = Path(output)
        self.object_usd = usd
        self.asset_id = torch.full((num_envs,), asset_id, device=self.device, dtype=torch.long)
        self.object = self.env.scene["object"]
        self.env.G_max = mdp.max_step_reward / (1 - gamma)
        self.contact_missing_counter = torch.zeros(num_envs, 2, device=self.device, dtype=torch.int32)
        self.physics_report = self._apply_and_read_physics()
        # MuJoCo-Warp exposes this supported scheduling option independently of
        # Newton's outer graph setting. No package/runtime files are patched.
        solver = self.env.sim.backend_manager._solver
        solver.mjw_model.opt.graph_conditional = bool(wp.is_conditional_graph_supported())
        self.contacts = PairContactRecorder(self.env.sim.backend_manager, num_envs)
        self.reset(torch.arange(num_envs, device=self.device), self.env.f.clone())
        self.physics_report["robot_fk_comparison"] = self._verify_native_fk()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _apply_and_read_physics(self):
        import warp as wp
        from newton import ModelFlags
        mass, com, inertia, friction, ground = validate_physical_profile(self.asset["physical_profile"])
        obj, env = self.object, self.env
        m = torch.full_like(obj.data.body_mass.torch, mass)
        c = com.to(device=self.device, dtype=torch.float32).reshape(1, 1, 3).expand(self.N, 1, 3).contiguous()
        I = inertia.to(device=self.device, dtype=torch.float32).reshape(1, 1, 9).expand(self.N, 1, 9).contiguous()
        obj.set_masses_index(masses=m)
        obj.set_coms_index(coms=c)
        obj.set_inertias_index(inertias=I)
        manager = env.sim.backend_manager
        manager.add_model_change(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)
        model = manager.get_model()
        material = wp.to_torch(obj._root_view.get_attribute("shape_material_mu", model))
        material[:] = friction
        all_mu = wp.to_torch(model.shape_material_mu)
        shape_body = wp.to_torch(model.shape_body)
        all_mu[shape_body < 0] = ground
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        read_m = obj.data.body_mass.torch[:, 0]
        read_c = obj.data.body_com_pose_b.torch[:, 0, :3]
        read_I = obj.data.body_inertia.torch.reshape(self.N, 1, 3, 3)[:, 0]
        for name, actual, expected in (("mass", read_m, m[:, 0]), ("COM", read_c, c[:, 0]), ("inertia", read_I, I[:, 0].reshape(-1, 3, 3))):
            if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError(f"Object native {name} readback differs from physical profile")
        read_mu = material.reshape(self.N, -1)
        if not torch.allclose(read_mu, torch.full_like(read_mu, friction), atol=1e-7, rtol=0):
            raise RuntimeError("Object friction readback differs")
        read_ground = all_mu[shape_body < 0]
        if not read_ground.numel() or not torch.allclose(read_ground, torch.full_like(read_ground, ground), atol=1e-7, rtol=0):
            raise RuntimeError("Scene ground friction readback differs")
        mu = read_mu[:, 0].clone()
        self.xi_o = physical_context(read_m, read_c, read_I, mu, torch.full_like(mu, float(read_ground[0])))
        gravity = wp.to_torch(model.gravity)
        if not torch.isfinite(gravity).all() or not bool((gravity[..., 2] < 0).all()):
            raise RuntimeError("Native scene gravity must act downwards")
        import mujoco
        mj = manager._solver.mj_model
        objects = [i for i in range(mj.nbody) if "Object" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
        if not objects or any(float(mj.body_mass[i]) <= 0 or int(mj.body_jntnum[i]) != 1 or
                              int(mj.jnt_type[int(mj.body_jntadr[i])]) != int(mujoco.mjtJoint.mjJNT_FREE) for i in objects):
            raise RuntimeError("Object must compile as a positive-mass dynamic free rigid body")
        from supermimic.physics import sha
        fixed_native = {"cone": int(mj.opt.cone), "integrator": int(mj.opt.integrator),
                        "solver_iterations": int(mj.opt.iterations), "line_search_iterations": int(mj.opt.ls_iterations),
                        "solver_tolerance": float(mj.opt.tolerance), "geom_margin_m": mj.geom_margin.tolist(),
                        "geom_gap_m": mj.geom_gap.tolist(), "geom_contact_dimensions": mj.geom_condim.tolist(),
                        "geom_solref": mj.geom_solref.tolist(), "geom_solimp": mj.geom_solimp.tolist(),
                        "body_exclusion_signatures": mj.exclude_signature.tolist(),
                        "constraint_capacity_per_world": int(manager._solver.mjw_data.njmax),
                        "total_contact_capacity": int(manager._solver.mjw_data.naconmax)}
        return {"mass": read_m.detach().cpu().tolist(), "com": read_c.detach().cpu().tolist(),
                "inertia": read_I.detach().cpu().tolist(), "object_friction": mu.detach().cpu().tolist(),
                "ground_friction": ground, "gravity": gravity.detach().cpu().tolist(),
                "object_pose_origin": "link", "velocity_reference_point": "link_origin", "frame": "world",
                "control_dt": CONTROL_DT, "substeps": 4, "integration_dt": CONTROL_DT / 4,
                "use_cuda_graph": False, "linear_damping": 0., "angular_damping": 0.,
                "physics": env.physics_settings, "collision_hash": self.asset["collision_hash"],
                "fixed_native_parameters": fixed_native,
                "capacity_configuration": {"legacy_motion_only": LEGACY_CAPACITY, "hoi_v1": HOI_CAPACITY,
                    "scope": "HOI-only allocation; PD, integration, solver settings and physical properties preserved",
                    "reason": "Observed NEFC overflow at njmax500 during actual plasticbox080 reference-reset zero-action rollout",
                    "overflow_policy": "abort without replay insertion or successful native gate"},
                "flattened_object_usd": str(self.object_usd), "flattened_object_usd_sha256": sha(self.object_usd),
                "compiled_collision_geometry_verified_against_bundle": False}

    def state(self):
        S = self.env.state()
        data = self.object.data
        pose, vel = data.root_link_pose_w.torch.clone(), data.root_link_vel_w.torch.clone()
        R = quat_to_mat(torch.cat((pose[:, 6:7], pose[:, 3:6]), -1))
        O = (pose[:, :3] - self.env.scene.env_origins, R, vel[:, :3], vel[:, 3:6])
        return S, O

    def _verify_native_fk(self):
        """Compare reward/palm FK with effective native link frames."""
        kin = self.mdp.human.kin
        S, _ = self.state()
        p, R, _, _ = kin.forward(*S, all_bodies=True)
        native = self.env.robot.data.body_link_pose_w.torch
        names = list(self.env.robot.body_names)
        shared = [name for name in kin.body_names if name in names]
        if not all(side + "_wrist_yaw_link" in shared for side in ("left", "right")):
            raise RuntimeError("Actual wrist link frames unavailable for geometry alignment")
        expected_ids = [kin.body_names.index(n) for n in shared]
        native_ids = [names.index(n) for n in shared]
        q = native[:, native_ids, 3:7]
        native_R = quat_to_mat(torch.cat((q[..., 3:4], q[..., :3]), -1))
        ep = torch.linalg.vector_norm(p[:, expected_ids] - (native[:, native_ids, :3] - self.scene.env_origins[:, None]), dim=-1)
        er = (R[:, expected_ids] - native_R).abs().amax(dim=(-1, -2))
        result = {"shared_body_names": shared, "max_link_position_error_m": float(ep.max()),
                  "max_rotation_matrix_error": float(er.max()), "position_tolerance_m": 5e-5, "rotation_tolerance": 5e-5}
        if result["max_link_position_error_m"] > 5e-5 or result["max_rotation_matrix_error"] > 5e-5:
            raise RuntimeError(f"Native robot and analytic reward/palm FK differ: {result}")
        result["passed"] = True
        return result

    def reset(self, env_ids, frames=None):
        if not len(env_ids):
            return
        if frames is not None and not bool((self.motion.asset_ids[frames] == self.asset_id[env_ids]).all()):
            raise ValueError("Cannot reset a geometry group to an incompatible motion")
        self.env.reset(env_ids, frames)
        f = self.env.f[env_ids]
        p, R, v, w = self.motion.ref_object(f)
        q = mat_to_quat(R)
        pose = torch.cat((p + self.env.scene.env_origins[env_ids], q[:, 1:4], q[:, :1]), -1).contiguous()
        self.object.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        if hasattr(self.object, "write_root_link_velocity_to_sim_index"):
            self.object.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v, w), -1).contiguous(), env_ids=env_ids)
        else:
            v_com = link_to_com_velocity(R, v, w, self.xi_o[env_ids, 1:4])
            self.object.write_root_com_velocity_to_sim_index(root_velocity=torch.cat((v_com, w), -1).contiguous(), env_ids=env_ids)
        self.env.scene.reset(env_ids)
        self.env.refresh()
        self.contact_missing_counter[env_ids] = 0
        _, actual = self.state()
        if not torch.allclose(actual[0][env_ids], p, atol=2e-5, rtol=0):
            raise RuntimeError("Object and robot reset did not reach the same reference phase")
        if not torch.allclose(actual[2][env_ids], v, atol=2e-5, rtol=1e-5):
            raise RuntimeError("Object native reset link-origin velocity disagrees with reference")
        if not torch.allclose(actual[1][env_ids], R, atol=2e-5, rtol=0) or not torch.allclose(actual[3][env_ids], w, atol=2e-5, rtol=1e-5):
            raise RuntimeError("Object native reset orientation/angular velocity disagrees with reference")

    def _overflow_gate(self):
        import mujoco_warp
        import warp as wp
        value = wp.to_torch(self.contacts.solver.mjw_data.overflow)
        # All recorded capacity overflow bits invalidate dynamics/contact labels.
        capacity = 0
        for name in ("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD", "NVMAX", "EPA_HORIZON"):
            capacity |= int(getattr(mujoco_warp.OverflowType, name, 0))
        if bool(((value & capacity) != 0).any()):
            failure = {"error": "MuJoCo-Warp collision/constraint capacity overflow", "bits": value.cpu().tolist(),
                       "object_asset": self.asset["object_asset"], "pair_probe": getattr(self, "active_pair_probe", None),
                       "reference_frames": self.env.f.detach().cpu().tolist(),
                       "constraint_capacity_per_world": int(self.contacts.solver.mjw_data.njmax),
                       "total_contact_capacity": int(self.contacts.solver.mjw_data.naconmax),
                       "action": "abort; invalid dynamics/contact labels must not enter replay"}
            with (self.output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            raise RuntimeError(f"MuJoCo-Warp collision/constraint capacity overflow: {value.cpu().tolist()}")

    def _advance(self, action):
        e = self.env
        target = e.default_q + e.xi[:, 1:30] + e.alpha * action
        e.robot.set_joint_position_target_index(target=target)
        e.scene.write_data_to_sim()
        self.contacts.begin_interval()
        e.sim.step(render=False)
        e.scene.update(CONTROL_DT)
        self._overflow_gate()
        S, O = self.state()
        names = ("robot_p", "robot_R", "robot_v", "robot_w", "robot_q", "robot_qd",
                 "object_p", "object_R", "object_v", "object_w")
        bad = [name for name, value in zip(names, S + O) if not bool(torch.isfinite(value).all())]
        if bad:
            failure = {"error": "nonfinite native physical state", "fields": bad, "asset_id": int(self.asset["asset_id"]),
                       "action": "abort without reset, replay insertion, or successful native gate"}
            with (self.output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2)
            torch.save({"robot": tuple(v.detach().cpu() for v in S), "object": tuple(v.detach().cpu() for v in O),
                        "action": action.detach().cpu(), "frame": self.env.f.detach().cpu()}, self.output / "failure_state.pt")
            raise RuntimeError(f"Native physics produced nonfinite state: {bad}")
        return self.contacts.end_interval()

    def _role_point(self, role):
        import warp as wp
        data = wp.to_torch(self.contacts.solver.mjw_data.geom_xpos)
        ids = torch.where(self.contacts.geom_role == role)[0]
        if self.contacts.topology == "one_mujoco_world_per_environment":
            return data[:, ids[0]].clone()
        points = []
        for env_id in range(self.N):
            selected = ids[self.contacts.geom_env[ids] == env_id]
            if not len(selected):
                raise RuntimeError("Probe geom cannot be mapped to a native environment")
            points.append(data[0, selected[0]])
        return torch.stack(points).clone()

    def calibrate_pair_probes(self, ticks=8):
        """Shallow surface-contact probes in a disposable validation environment.

        The source dataset and legacy scene are untouched. Probe transitions
        never enter replay. All channels integrate real solver forces across
        four substeps; hand/ground is independently measured as a negative
        identity control. Test placements and outcomes are saved verbatim.
        """
        ids = torch.arange(self.N, device=self.device)
        frames = self.f.clone()
        actions = torch.zeros(self.N, 29, device=self.device)
        report = {}
        geometries = self.mdp.shapes.geometries[int(self.asset["asset_id"])]
        vertices = torch.cat([m["vertices"] for m in geometries["collision_meshes"]]).to(self.device, dtype=torch.float32)
        for name, role in (("left_hand_object", LEFT), ("right_hand_object", RIGHT), ("object_ground", GROUND), ("hand_ground_negative", GROUND)):
            self.active_pair_probe = name
            self.reset(ids, frames)
            # Populate solver kinematics and save a measured baseline, then
            # write only this disposable probe's object/root initial state.
            self._advance(actions)
            S, O = self.state()
            q = mat_to_quat(O[1])
            position = O[0] + self.scene.env_origins
            probe_velocity = torch.zeros(self.N, 6, device=self.device)
            definition = {"placement": "object_ground_support_surface", "prescribed_penetration_m": .002}
            if name in ("left_hand_object", "right_hand_object"):
                side = "left" if role == LEFT else "right"
                hand = self.mdp.shapes.hands[side]
                points = torch.as_tensor(hand["palm_points"], device=self.device, dtype=torch.float32)
                point_id = int((points - points.mean(0)).square().sum(-1).argmin())
                local = points[point_id] + torch.as_tensor(hand["mount"], device=self.device)
                kin = self.mdp.human.kin
                P, R, V, W = kin.forward(*S, all_bodies=True)
                wrist = kin.body_names.index(side + "_wrist_yaw_link")
                offset = (R[:, wrist] @ local[:, None]).squeeze(-1)
                palm = P[:, wrist] + offset + self.scene.env_origins
                local_normal = position.new_tensor([0., -1. if side == "left" else 1., 0.])
                normal = (R[:, wrist] @ local_normal[:, None]).squeeze(-1)
                world_offsets = (O[1] @ vertices.T).transpose(-1, -2)
                # The object support surface lies 2mm inside the measured
                # actual palm plane. Center embedding would create thousands
                # of irrelevant overlaps and does not calibrate a hand touch.
                position, support_id = shallow_support_contact(palm, normal, world_offsets)
                probe_velocity[:, :3] = V[:, wrist] + torch.linalg.cross(W[:, wrist], offset)
                definition = {"placement": "actual_palm_patch_vs_object_support_surface",
                              "palm_point_index": point_id, "object_support_vertex_indices": support_id.cpu().tolist(),
                              "palm_point_world": palm.cpu().tolist(), "outward_normal_world": normal.cpu().tolist(),
                              "prescribed_penetration_m": .002,
                              "object_linear_velocity": "matched to measured palm-point velocity"}
            elif name == "object_ground":
                world_offsets = (O[1] @ vertices.T).transpose(-1, -2)
                position[:, 2] = -world_offsets[..., 2].amin(-1) - .002
            else:
                definition = {"placement": "hand_ground_negative_control; object_displaced_clear"}
                left = self._role_point(LEFT)
                root_q = mat_to_quat(S[1])
                root_p = S[0] + self.scene.env_origins
                root_p[:, 2] -= left[:, 2] + .015
                root_pose = torch.cat((root_p, root_q[:, 1:4], root_q[:, :1]), -1).contiguous()
                self.robot.write_root_link_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
                position[:] = position.new_tensor([8., 8., 5.]) + self.scene.env_origins
            pose = torch.cat((position, q[:, 1:4], q[:, :1]), -1).contiguous()
            self.object.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
            self.object.write_root_com_velocity_to_sim_index(root_velocity=probe_velocity, env_ids=ids)
            self.env.refresh()
            max_force, hand_ground = torch.zeros(3, device=self.device), torch.zeros(2, device=self.device)
            impulse_trace, force_trace, hand_ground_trace = [], [], []
            try:
                for _ in range(ticks):
                    pair = self._advance(actions)
                    max_force = torch.maximum(max_force, pair["contact_force"].amax(0))
                    hand_ground = torch.maximum(hand_ground, (pair["hand_ground_impulse"] / CONTROL_DT).amax(0))
                    impulse_trace.append(pair["impulse"].cpu().tolist())
                    force_trace.append(pair["contact_force"].cpu().tolist())
                    hand_ground_trace.append(pair["hand_ground_impulse"].cpu().tolist())
            except BaseException as exc:
                failure = {"probe": name, "definition": definition, "object_initial_pose_xyzw": pose.cpu().tolist(),
                           "object_initial_com_velocity_world": probe_velocity.cpu().tolist(),
                           "completed_interval_impulses_Ns": impulse_trace, "completed_interval_forces_N": force_trace,
                           "error_type": type(exc).__name__, "error": str(exc), "previous_complete_probes": report,
                           "scientific_status": "failed native calibration; no successful pair labels inferred"}
                with (self.output / "pair-probes-failure.json").open("x", encoding="utf-8") as stream:
                    json.dump(failure, stream, indent=2)
                raise
            report[name] = {"object_initial_pose_xyzw": pose.cpu().tolist(), "ticks": ticks,
                            "definition": definition, "object_initial_com_velocity_world": probe_velocity.cpu().tolist(),
                            "maximum_interval_normal_force_N": max_force.cpu().tolist(),
                            "maximum_hand_ground_interval_force_N": hand_ground.cpu().tolist(),
                            "interval_normal_impulse_Ns": impulse_trace, "interval_mean_normal_force_N": force_trace,
                            "interval_hand_ground_impulse_Ns": hand_ground_trace,
                            "control_dt_s": CONTROL_DT, "substeps_per_interval": 4}
            (self.output / "pair-probes-progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        left_ok = report["left_hand_object"]["maximum_interval_normal_force_N"][0] > self.contacts.threshold
        right_ok = report["right_hand_object"]["maximum_interval_normal_force_N"][1] > self.contacts.threshold
        ground_ok = report["object_ground"]["maximum_interval_normal_force_N"][2] > self.contacts.threshold
        negative = report["hand_ground_negative"]
        negative_ok = max(negative["maximum_hand_ground_interval_force_N"]) > self.contacts.threshold and max(negative["maximum_interval_normal_force_N"][:2]) == 0
        report["passed"] = left_ok and right_ok and ground_ok and negative_ok
        report["checks"] = {"left_hand_object": left_ok, "right_hand_object": right_ok,
                            "object_ground": ground_ok, "hand_ground_not_mislabelled": negative_ok}
        self.active_pair_probe = None
        self.reset(ids, frames)
        return report

    def step(self, action):
        e = self.env
        if action.shape != (self.N, 29) or not bool(torch.isfinite(action).all()):
            raise ValueError("Expected finite [N,29] G1 PD-offset actions")
        S, O = self.state()
        f, a_prev, ep_step = e.f.clone(), e.a_prev.clone(), e.ep_step.clone()
        pair = self._advance(action)
        S1, O1 = self.state()
        # Frames only clamp for full-duration batched evaluation; training resets
        # at clip ends. A short clip cannot read into its neighbour's reference.
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        f1 = torch.minimum(f + 1, clip_end)
        reward, terms = self.mdp.reward(S1, O1, f1, action, a_prev, pair["contact"], pair["contact_valid"], terms=True)
        term = self.mdp.termination(S1, O1, f1)
        timeout = (f + 1 >= clip_end) | (ep_step + 1 >= e.max_episode_steps)
        done = term | timeout
        # Diagnostic-only reason bits; no additional termination is introduced.
        human = self.mdp.human
        height_fail = (self.motion.root_p[f1, 2] - S1[0][:, 2]).abs() > human.cfg.term_height
        tilt_fail = (self.motion.root_R[f1, 2, 2] - S1[1][:, 2, 2]).abs() > human.cfg.term_tilt
        object_fail = torch.linalg.vector_norm(O1[0] - self.motion.object_p[f1], dim=-1) > self.mdp.object_failure_threshold(f1)
        failure_reason = (height_fail.int() | (tilt_fail.int() << 1) | (object_fail.int() << 2) |
                          ((f + 1 >= clip_end).int() << 3) | ((ep_step + 1 >= e.max_episode_steps).int() << 4))
        expected, valid = self.motion.expected(f1)
        missing = (expected > .5) & (valid > .5) & pair["contact_valid"][:, :2] & (pair["contact"][:, :2] < .5)
        self.contact_missing_counter = torch.where(missing, self.contact_missing_counter + 1, 0)
        tr = {"s": flatten_state(S), "s1": flatten_state(S1), "o": flatten_object(O), "o1": flatten_object(O1),
              "a": action.clone(), "a_prev": a_prev, "xi": e.xi.clone(), "xi_o": self.xi_o.clone(), "f": f,
              "ep_step": ep_step, "done": done, "term": term, "asset_id": self.asset_id.clone(),
              "failure_reason": failure_reason,
              "r": reward, "r_human": terms["human_reward"], "r_hoi": reward - terms["human_reward"],
              "contact_missing_counter": self.contact_missing_counter.clone(), "terms": terms, **pair}
        e.ep_disc_ret += (e.gamma ** ep_step.float()) * reward
        e.ep_ret += reward
        e.f, e.ep_step, e.a_prev = f1, ep_step + 1, action.clone()
        if e.eval_mode:
            e.failed_eval |= term & ~e.done_eval
            e.done_eval |= done
            return tr
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            length = e.ep_step[ids].float()
            boot = torch.where(term[ids], 0., (e.gamma ** length) * (e.ep_ret[ids] / length) / (1 - e.gamma))
            G = e.ep_disc_ret[ids] + boot
            bins = e.start_bin[ids]
            e.bin_G.index_put_((bins,), .8 * e.bin_G[bins] + .2 * G)
            tr["ep_len"] = length
            tr["ep_ret_norm"] = e.ep_ret[ids] / (length * self.mdp.max_step_reward)
            # s1/o1/contacts above were cloned before this joint reset.
            self.reset(ids)
        return tr

    def close(self):
        self.contacts.close()

    def export_compiled_geometry(self, output):
        """Export exact compiled convex meshes in object/hand link frames.

        This creates a new explicit-hull USD; subsequent preparation does not
        need to infer PhysX cooking or recook an unbound source mesh.
        """
        import hashlib
        import mujoco
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation
        from supermimic.physics import sha
        from hoi.hand_collision import HAND_SHA, ORIGIN, RECIPE
        from .geometry import collision_distance
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        model = self.contacts.solver.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        object_body = next(i for i, n in enumerate(names) if "Object" in n and int(model.body_jntnum[i]) == 1)

        def extract(role, body, offset=None):
            meshes = []
            for g, r in enumerate(self.contacts.roles):
                if r != role:
                    continue
                # Export first environment only when the CPU model encodes all.
                ge = self.contacts.geom_envs[g]
                if self.contacts.topology == "one_world_with_named_environments" and ge != 0:
                    continue
                if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise RuntimeError("Compiled collision export currently requires mesh colliders")
                mesh_id = int(model.geom_dataid[g])
                vertices, faces, evidence = compiled_convex_mesh(model, mesh_id)
                world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                if offset is not None:
                    local -= np.asarray(offset)
                meshes.append({"vertices": torch.tensor(local), "faces": torch.tensor(faces, dtype=torch.long),
                               "source_native_geom": self.contacts.names[g], "native_convex_evidence": evidence})
            if not meshes:
                raise RuntimeError("No actual compiled collision meshes were exported")
            return meshes

        object_meshes = extract(OBJECT, object_body)
        hands = {}
        repo = Path(__file__).resolve().parents[2]
        for side, role in (("left", LEFT), ("right", RIGHT)):
            body = next(i for i, n in enumerate(names) if side + "_wrist_yaw_link" in n)
            meshes = extract(role, body, ORIGIN[side])
            stl = repo / "data/g1_description/meshes" / f"{side}_rubber_hand.STL"
            if sha(stl) != HAND_SHA[side]:
                raise RuntimeError("Native hand source STL identity changed")
            from .prepare import read_triangle_mesh
            source_mesh, _ = read_triangle_mesh({"path": str(stl), "sha256": HAND_SHA[side]}, stl.parent)
            points, patch_definition = palm_patch_points(meshes, [source_mesh], side)
            if float(collision_distance(points, meshes).abs().max()) > 1e-5:
                raise RuntimeError("Ray-cast palm patch does not lie on the actual collider union surface")
            _write_palm_qa(output / f"{side}_palm_patch.png", source_mesh["vertices"], points)
            from .geometry import tensor_sha256
            from . import probes
            evidence = {"schema": "hoi_v1_palm_patch_evidence_v1", "source_stl_sha256": HAND_SHA[side],
                        "probes_source": {"path": str(Path(probes.__file__)), "sha256": sha(probes.__file__)},
                        "definition": patch_definition, "points_sha256": tensor_sha256(points),
                        "collision_meshes_sha256": hashlib.sha256(json.dumps(
                            [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist()} for m in meshes],
                            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "visual_qa": {"path": str(output / f"{side}_palm_patch.png"), "sha256": sha(output / f"{side}_palm_patch.png")}}
            evidence["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            hands[side] = {"mount": ORIGIN[side], "source_stl": {"path": str(stl), "sha256": HAND_SHA[side]},
                           "collision_meshes": [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist(), "frame": "hand_local"} for m in meshes],
                           "palm_points": points.tolist(), "recipe": RECIPE,
                           "native_convex_evidence": [m["native_convex_evidence"] for m in meshes],
                           "probe_definition": patch_definition, "palm_selection_verified": True,
                           "palm_selection_evidence": evidence,
                           "visual_qa": {"path": str(output / f"{side}_palm_patch.png"), "projections": ["XY", "XZ", "YZ"],
                                         "source": "blue", "palm_patch": "red"}}
        self.physics_report["native_convex_hull_export"] = {
            "source": "mjModel.mesh_graph", "all_geometries_graph_backed": True,
            "object_hulls": len(object_meshes), "hand_hulls": {side: len(hands[side]["collision_meshes"]) for side in hands},
            "object_evidence_sha256": hashlib.sha256(json.dumps([m["native_convex_evidence"] for m in object_meshes], sort_keys=True).encode()).hexdigest(),
            "hand_evidence_sha256": {side: hashlib.sha256(json.dumps(hands[side]["native_convex_evidence"], sort_keys=True).encode()).hexdigest() for side in hands}}
        target = output / "object_explicit_hulls.usda"
        stage = Usd.Stage.CreateNew(str(target))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
        stage.SetDefaultPrim(root)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        p = self.asset["physical_profile"]
        mass.CreateMassAttr(float(p["mass"]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, p["com"])))
        values, axes = np.linalg.eigh(np.asarray(p["inertia"]))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = Rotation.from_matrix(axes).as_quat()
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        for i, mesh in enumerate(object_meshes):
            prim = UsdGeom.Mesh.Define(stage, f"/Object/hull_{i:03d}")
            prim.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in mesh["vertices"]])
            prim.CreateFaceVertexCountsAttr([3] * len(mesh["faces"]))
            prim.CreateFaceVertexIndicesAttr(mesh["faces"].flatten().tolist())
            prim.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("convexHull")
        stage.GetRootLayer().Save()
        record = {"object_asset": self.asset["object_asset"], "exact_scale": self.asset["exact_scale"],
                  "mesh_to_object": np.eye(4).tolist(), "native_asset": str(target), "native_asset_sha256": sha(target),
                  "physical_profile": p, "cooking_recipe": "compiled_Newton_CoACD_pointsets_explicit_convexHull_import_geometric_equivalence_requires_native_check",
                  "collision_meshes": [{"vertices": m["vertices"].tolist(), "faces": m["faces"].tolist(), "frame": "object_local"} for m in object_meshes],
                  "compiled_geometry_frame": "object_root_link_origin", "compiled_mesh_names": [m["source_native_geom"] for m in object_meshes],
                  "native_convex_evidence": [m["native_convex_evidence"] for m in object_meshes],
                  "source_native_asset": self.asset["native_asset"], "source_native_asset_sha256": self.asset["native_asset_sha256"]}
        recipe = {"schema": "hoi_v1_compiled_geometry_recipe_v1", "assets": [record], "hands": hands,
                  "native_physics_readback": self.physics_report,
                  "hand_runtime_report": self.env.physics_settings["hand_asset"],
                  "native_robot_asset": {"path": self.env.physics_settings["robot_usd"], "sha256": self.env.physics_settings["robot_usd_sha256"]},
                  "limitations": "Actual Newton CoACD meshes; does not claim original PhysX cooker equivalence"}
        if getattr(self, "reference_kinematics", None) is not None:
            recipe["reference_kinematics"] = self.reference_kinematics
        (output / "assets.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        return recipe, object_meshes

    def verify_compiled_geometry(self, meshes, *, compiled_hands=None, tolerance=1e-5):
        """Bidirectional union-surface and bbox check against bound recipe."""
        from .geometry import collision_distance, deterministic_surface_points
        expected = self.mdp.shapes.geometries[int(self.asset["asset_id"])]["collision_meshes"]
        errors = []
        for source, target in ((meshes, expected), (expected, meshes)):
            sample = deterministic_surface_points(source, 2048, 0).double()
            # Exclude internal faces of overlapping convex pieces: they are
            # not part of the collider union's external boundary.
            boundary = collision_distance(sample, source).abs() < 2e-6
            if not bool(boundary.any()):
                raise RuntimeError("Unable to compare collision-union external surfaces")
            errors.append(float(collision_distance(sample[boundary], target).abs().max()))
        actual_vertices = torch.cat([m["vertices"].double() for m in meshes])
        expected_vertices = torch.cat([m["vertices"].double() for m in expected])
        box_error = max(float((actual_vertices.amin(0) - expected_vertices.amin(0)).abs().max()),
                        float((actual_vertices.amax(0) - expected_vertices.amax(0)).abs().max()))
        passed = max(*errors, box_error) <= tolerance
        result = {"passed": passed, "max_bidirectional_surface_errors_m": errors, "bbox_max_abs_error_m": box_error,
                  "tolerance_m": tolerance, "surface_samples_per_direction": 2048,
                  "test": "compiled convex union vs SHA-bound geometry cache in object link frame"}
        if not passed:
            raise RuntimeError(f"Native collision recipe changed: {result}")
        self.physics_report["compiled_collision_geometry_verified_against_bundle"] = True
        self.physics_report["compiled_geometry_comparison"] = result
        if compiled_hands is None:
            raise RuntimeError("Native geometry gate must compare actual hand colliders and palm probes too")
        hand_comparisons = {}
        for side in ("left", "right"):
            hand = self.mdp.shapes.hands[side]
            actual_hand = compiled_hands[side]
            observed = [{"vertices": torch.tensor(m["vertices"], dtype=torch.float64), "faces": torch.tensor(m["faces"], dtype=torch.long)}
                        for m in actual_hand["collision_meshes"]]
            errors = []
            for a, b in ((observed, hand["collision_meshes"]), (hand["collision_meshes"], observed)):
                sample = deterministic_surface_points(a, 1024, 0).double()
                boundary = collision_distance(sample, a).abs() < 2e-6
                if not bool(boundary.any()):
                    raise RuntimeError("Cannot establish actual hand collider exterior")
                errors.append(float(collision_distance(sample[boundary], b).abs().max()))
            actual_points = torch.tensor(actual_hand["palm_points"], dtype=torch.float32)
            expected_points = torch.as_tensor(hand["palm_points"], dtype=torch.float32)
            point_error = float((actual_points - expected_points).abs().max()) if actual_points.shape == expected_points.shape else float("inf")
            if max(*errors, point_error) > tolerance:
                raise RuntimeError(f"{side} hand collider/palm recipe changed: surfaces {errors}, point error {point_error}")
            hand_comparisons[side] = {"passed": True, "bidirectional_surface_error_m": errors, "palm_points_error_m": point_error,
                                      "palm_evidence_sha256": hand.get("palm_selection_evidence", {}).get("evidence_sha256")}
        self.physics_report["compiled_hand_geometry_comparison"] = hand_comparisons
        self.physics_report["independent_native_geometry_queries"] = self._native_distance_audit()
        return result

    def _native_distance_audit(self, tolerance=5e-5):
        """Compare cached surfaces with independent native convex GJK distances.

        CPU mjData is private: no GPU state, physics step or dataset pose is
        modified. AABB lower bounds only exclude pairs which cannot improve the
        measured native distance; every remaining cache pair uses exact mesh
        distance, rather than mesh-ray queries against a visual surface.
        """
        import mujoco
        import numpy as np
        from .geometry import exact_mesh_gap
        model = self.contacts.solver.mj_model
        data = mujoco.MjData(model)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        body = next(i for i, n in enumerate(names) if "Object" in n and int(model.body_jntnum[i]) == 1)
        address = int(model.jnt_qposadr[int(model.body_jntadr[body])])
        object_geoms = [g for g, role in enumerate(self.contacts.roles) if role == OBJECT and int(model.geom_bodyid[g]) == body]
        objects = self.mdp.shapes.geometries[int(self.asset["asset_id"])]["collision_meshes"]
        radius = max(float(torch.as_tensor(m["vertices"]).norm(dim=-1).max()) for m in objects)
        from hoi.hand_collision import ORIGIN
        from supermimic.rotations import so3_exp
        records = []
        for side, role in (("left", LEFT), ("right", RIGHT)):
            wrist = next(i for i, name in enumerate(names) if side + "_wrist_yaw_link" in name)
            hand_geoms = [g for g, r in enumerate(self.contacts.roles) if r == role and
                          (self.contacts.topology == "one_mujoco_world_per_environment" or self.contacts.geom_envs[g] == 0)]
            hands = self.mdp.shapes.hands[side]["collision_meshes"]
            mount = torch.tensor(ORIGIN[side], dtype=torch.float64)
            hand_radius = max(float((torch.as_tensor(m["vertices"]).double() + mount).norm(dim=-1).max()) for m in hands)
            for index, direction in enumerate((np.array([1., .3, .2]), np.array([-.2, .5, 1.]))):
                direction /= np.linalg.norm(direction)
                mujoco.mj_forward(model, data)
                data.qpos[address:address + 3] = data.xpos[wrist] + direction * (radius + hand_radius + .15)
                q = mat_to_quat(so3_exp(torch.tensor([.17, -.23, .11]) * index)).numpy()
                data.qpos[address + 3:address + 7] = q
                mujoco.mj_forward(model, data)
                native = min(float(mujoco.mj_geomDistance(model, data, h, o, 100., np.empty(6)))
                             for h in hand_geoms for o in object_geoms)
                if native <= 0 or native >= 100:
                    raise RuntimeError("Independent geometry query did not produce a separated convex distance")
                def world(mesh, rigid_body, offset=None):
                    v = torch.as_tensor(mesh["vertices"]).double()
                    if offset is not None:
                        v = v + offset
                    rotation = torch.tensor(data.xmat[rigid_body].reshape(3, 3))
                    return {"vertices": v @ rotation.T + torch.tensor(data.xpos[rigid_body]), "faces": mesh["faces"]}
                hand_world = [world(m, wrist, mount) for m in hands]
                object_world = [world(m, body) for m in objects]
                best, visited = float("inf"), 0
                for hand in hand_world:
                    for obj in object_world:
                        h, o = hand["vertices"], obj["vertices"]
                        lower = torch.maximum(torch.maximum(h.amin(0) - o.amax(0), o.amin(0) - h.amax(0)), h.new_zeros(3)).norm()
                        if float(lower) > min(best, native + tolerance):
                            continue
                        gap, penetration = exact_mesh_gap([hand], [obj])
                        if float(penetration) > tolerance:
                            raise RuntimeError("Cached collider overlaps a separated native geometry probe")
                        best, visited = min(best, float(gap)), visited + 1
                error = abs(best - native)
                records.append({"hand": side, "pose": index, "native_distance_m": native, "cached_surface_distance_m": best,
                                "abs_error_m": error, "cache_pairs_queried": visited, "cache_pairs_total": len(hands) * len(objects)})
                if not np.isfinite(error) or error > tolerance:
                    raise RuntimeError(f"Independent native convex distance differs from cached geometry: {records[-1]}")
        return {"passed": True, "api": "mujoco.mj_geomDistance convex collision distance", "tolerance_m": tolerance,
                "overlap": "excluded; actual pair force probes independently verify overlap identity", "samples": records}


def _load_legacy_fixture(path, asset_id, kin, device):
    """Read one immutable canonical case solely for the native physics probe."""
    import hashlib
    from types import SimpleNamespace
    from supermimic.mdp import MDP
    from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list
    from supermimic.physics import sha
    from hoi.data.reference import adapt_motion
    from .motion import TrainingMotionBank
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    key = list(manifest["assets"])[asset_id]
    source_asset = manifest["assets"][key]
    source_case = next(c for c in manifest["cases"] if c["object_asset"] == key)
    if sha(source_case["motion"]) != source_case["motion_sha256"]:
        raise ValueError("legacy fixture source motion changed")
    frames = torch.load(source_case["motion"], weights_only=True, map_location="cpu")
    reference = adapt_motion(frames, manifest["robot_joint_names"], G1_JOINT_NAMES)
    n = len(reference["time"])
    unknown = torch.zeros(n, 2)
    # Expected-contact masks are unknown here. This path never computes HOI
    # reward, updates a normalizer, creates replay or enters a learner.
    bank = TrainingMotionBank([{"name": source_case["id"], "asset_id": 0, "reference": reference,
                                "expected_contact": unknown, "expected_valid": unknown}], kin, device=device, partition="validation")
    profile = source_asset["metadata"]["object_properties"][0]
    scale = [int(x) / 100 for x in key.split("_")[-3:]]
    asset = {"asset_id": 0, "object_asset": key, "exact_scale": scale,
             "native_asset": source_asset["usd"], "native_asset_sha256": source_asset["usd_sha256"],
             "collision_hash": hashlib.sha256((source_asset["usd_sha256"] + ":Newton_CoACD_native_probe").encode()).hexdigest(),
             "physical_profile": {**profile, "friction": .6, "ground_friction": .3}}
    mdp = SimpleNamespace(human=MDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device)), max_step_reward=.15)
    source = {"fixture_manifest": str(path / "manifest.json"), "fixture_manifest_sha256": sha(path / "manifest.json"),
              "case": source_case, "scope": "single_asset_physics_probe; no HOI reward or training",
              "reference_kinematics": {"urdf": {"path": str(path / "metadata/ultra_g1.urdf"), "sha256": sha(path / "metadata/ultra_g1.urdf")},
                                       "body_names": source_asset["metadata"]["body_names"], "dof_names": source_asset["metadata"]["dof_names"]}}
    return bank, mdp, asset, source


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation
    from supermimic.physics import add_physics_arguments, resolve_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--legacy-prepared-fixture", type=Path)
    parser.add_argument("--robot-asset-evidence", type=Path, help="Fixture-only recipe binding an already verified exact hand overlay")
    parser.add_argument("--asset-id", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    add_physics_arguments(parser)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    settings, _ = resolve_arguments(args)
    if not (settings.hand_collision and settings.self_collision):
        parser.error("v1 native gate requires the specified hand ON/self ON profile")
    args.output.mkdir(parents=True, exist_ok=False)
    import hashlib
    repo = Path(__file__).resolve().parents[2]
    from .config import source_identity
    source_code = source_identity()["files"]
    (args.output / "source.json").write_text(json.dumps(source_code, indent=2) + "\n", encoding="utf-8")
    import os
    os.environ["SUPERMIMIC_ASSET_DIR"] = str(args.output / "robot_assets")
    from supermimic.kinematics import G1Kinematics
    from supermimic.robot import default_joint_pos_list
    import warp as wp
    wp.init()
    torch.manual_seed(args.seed)
    device = "cuda:0"
    kin = G1Kinematics(device)
    if args.legacy_prepared_fixture:
        bank, mdp, asset, source = _load_legacy_fixture(args.legacy_prepared_fixture, args.asset_id, kin, device)
        robot_asset = None
        if args.robot_asset_evidence:
            recipe = json.loads(args.robot_asset_evidence.read_text(encoding="utf-8"))
            robot_asset = recipe.get("native_robot_asset")
            if robot_asset is None and recipe.get("hand_runtime_report"):
                saved = recipe["hand_runtime_report"]
                robot_asset = {"path": saved["usd"], "sha256": saved["usd_sha256"]}
            if robot_asset is None:
                raise ValueError("Robot evidence recipe does not bind an exact verified hand overlay")
            source["robot_asset_evidence"] = {"path": str(args.robot_asset_evidence), "sha256": hashlib.sha256(args.robot_asset_evidence.read_bytes()).hexdigest()}
    else:
        if args.robot_asset_evidence:
            parser.error("--robot-asset-evidence applies only to the independent legacy physics fixture")
        from hoi.v1.prepare import load_bundle
        from hoi.v1.mdp import HOIMDP
        bank, shapes, bundle = load_bundle(args.bundle, kin, device=device, require_train=True, asset_filter=[args.asset_id])
        asset = next(a for a in bundle["assets"] if int(a["asset_id"]) == args.asset_id)
        mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device), shapes)
        robot_asset = bundle.get("native_robot_asset")
        if robot_asset is None:
            raise ValueError("HOI bundle must bind an exact verified native robot hand overlay")
        source = {"bundle": str(args.bundle), "scope": "single_asset_native_bundle_gate",
                  "bundle_content_sha256": bundle["bundle_content_sha256"]}
    cfg = make_hoi_sim(device, args)
    with launch_simulation(cfg, args), torch.no_grad():
        env = HOITrackingSim(cfg, args.num_envs, bank, mdp, asset, args.output, eval_mode=False, reset_noise=False, robot_asset=robot_asset)
        env.reference_kinematics = source.get("reference_kinematics")
        from hoi.hand_collision import compiled_audit
        from supermimic.physics import compiled_self_collision_audit
        hand_audit = compiled_audit(env.contacts.solver.mj_model, True)
        self_audit = compiled_self_collision_audit(env.contacts.solver.mj_model, True)
        export, meshes = env.export_compiled_geometry(args.output / "compiled_geometry")
        if args.legacy_prepared_fixture:
            from types import SimpleNamespace
            mdp.shapes = SimpleNamespace(geometries=[{"collision_meshes": meshes}], hands=export["hands"])
            env.physics_report["independent_native_geometry_queries"] = env._native_distance_audit()
        else:
            env.verify_compiled_geometry(meshes, compiled_hands=export["hands"])
        probes = env.calibrate_pair_probes()
        (args.output / "pair-probes.json").write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
        totals = torch.zeros(3, device=device)
        maximum_force = torch.zeros(3, device=device)
        for _ in range(args.steps):
            action = torch.zeros(args.num_envs, 29, device=device)
            tr = env._advance(action) if args.legacy_prepared_fixture else env.step(action)
            totals += tr["contact"].sum(0)
            maximum_force = torch.maximum(maximum_force, tr["contact_force"].amax(0))
        palm_verified = False
        if not args.legacy_prepared_fixture:
            palm_verified = all(bool(mdp.shapes.hands[s].get("palm_selection_verified", False)) for s in ("left", "right"))
        report = {"physics": env.physics_report, "contact": env.contacts.manifest(), "steps": args.steps, "source": source,
                  "source_code_sha256": source_code,
                  "asset_id": int(asset["asset_id"]), "object_asset": asset["object_asset"],
                  "seed": args.seed,
                  "bundle_content_sha256": source.get("bundle_content_sha256"), "palm_selection_verified": palm_verified,
                  "palm_selection_evidence": None if args.legacy_prepared_fixture else {s: mdp.shapes.hands[s].get("palm_selection_evidence") for s in ("left", "right")},
                  "compiled_hand_audit": hand_audit, "compiled_self_collision_audit": self_audit,
                  "compiled_geometry_export": str(args.output / "compiled_geometry/assets.json"),
                  "num_envs": args.num_envs, "contact_positives": totals.cpu().tolist(),
                  "maximum_interval_force_N": maximum_force.cpu().tolist(),
                  "zero_action_smoke_completed": True, "formal_training_started": False,
                  "native_pair_calibration_complete": probes["passed"], "pair_probes": probes}
        (args.output / "native.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        env.close()


if __name__ == "__main__":
    main()
