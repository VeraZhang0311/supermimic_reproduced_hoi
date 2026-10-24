"""CPU contracts only; these tests do not claim native physics validation."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
import hashlib
import json

import torch

from hoi.v1.contacts import (CHANNELS, LABEL_SCHEMA, GROUND, LEFT, RIGHT, OBJECT, OTHER, LEFT_FOOT, RIGHT_FOOT,
                             PairContactRecorder, SolverContactAdapter, compiled_geom_roles,
                             aggregate_pairs, aggregate_ground_pairs, normal_impulse_to_label)
from hoi.v1.native import (HOITrackingSim, compiled_convex_mesh, compiled_foot_probe_surface,
                           compiled_foot_world_minimum, compiled_foot_geometry_audit,
                           lift_foot_probe_above_ground,
                           com_to_link_velocity, link_to_com_velocity, palm_patch_points, physical_context,
                           validate_physical_profile, shallow_support_contact, configure_hoi_capacity)
from hoi.v1.workers import allocate_workers, geometry_group, resource_preflight
from supermimic.rotations import so3_exp


def asset(i, scale=1.):
    return {"asset_id": i, "object_asset": f"box{i}", "exact_scale": [scale] * 3,
            "collision_hash": "c" * 64, "physical_profile": {
                "mass": 2., "com": [.1, -.05, .2], "inertia": [[.3, 0., 0.], [0., .4, 0.], [0., 0., .5]],
                "friction": .6, "ground_friction": .3}}


def mesh_foot_fixture(environments=1, *, named=False):
    import mujoco
    vertices = [[x, y, z] for x in (-.05, .05) for y in (-.04, .04) for z in (-.02, .02)]
    values = " ".join(str(value) for vertex in vertices for value in vertex)
    bodies = []
    for environment in range(environments):
        for side in ("left", "right"):
            prefix = f"env_{environment}_" if named else ""
            bodies.append(f'''<body name="{prefix}{side}_ankle_roll_link">
                <geom name="{prefix}{side}_heel" type="mesh" mesh="foot" pos="-.04 0 -.03"/>
                <geom name="{prefix}{side}_toe" type="mesh" mesh="foot" pos=".06 0 -.03"/>
                <geom name="{prefix}{side}_visual" type="sphere" size=".2" contype="0" conaffinity="0"/>
                </body>''')
    return mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="foot" vertex="{values}"/></asset>
        <worldbody>{''.join(bodies)}</worldbody></mujoco>''')


def mesh_foot_manifest(folder):
    from hoi.v1.foot_collision import FOOT_PROFILE, FOOT_SCHEMA, FOOT_SHA, ULTRA_URDF_SHA256
    from hoi.hand_collision import RECIPE
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    pieces = [{"vertices": [[x + offset, y, z - .03] for x in (-.05, .05) for y in (-.04, .04) for z in (-.02, .02)],
               "faces": faces} for offset in (-.04, .06)]
    cooked = {}
    for side in ("left", "right"):
        path = Path(folder) / (side + "-hulls.json")
        path.write_text(json.dumps({"recipe": RECIPE, "stl_sha256": FOOT_SHA[side], "pieces": pieces}), encoding="utf-8")
        cooked[side] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema": FOOT_SCHEMA, "profile": FOOT_PROFILE,
            "overlay": {"sha256": "a" * 64}, "cook_sha256": "b" * 64,
            "original_urdf": {"sha256": ULTRA_URDF_SHA256},
            "cook": {"algorithm": "CoACD", "recipe": dict(RECIPE), "equivalent_to_physx_vhacd": False,
                     "cooker_and_usd_imports_isolated": True,
                     "original_ultra_vhacd": {"max_convex_hulls": 5, "max_num_vertices_per_ch": 16, "resolution": 60000}},
            "source_meshes": {side: {"sha256": FOOT_SHA[side], "collision_origin": [0., 0., 0.],
                "collision_rpy": [0., 0., 0.], "mesh_scale": [1., 1., 1.], "link": side + "_ankle_roll_link"}
                for side in ("left", "right")},
            "sides": {side: {"hull_count": 2, "cooked_geometry": cooked[side]} for side in ("left", "right")}}


class NativeContractTests(unittest.TestCase):
    def test_hoi_capacity_copy_preserves_legacy_and_all_physical_settings(self):
        solver = SimpleNamespace(nconmax=128, njmax=500, cone="pyramidal", integrator="implicitfast",
                                 iterations=100, ls_iterations=50, tolerance=1e-6, use_mujoco_contacts=True)
        config = SimpleNamespace(dt=.02, device="cuda:0", physics=SimpleNamespace(
            solver_cfg=solver, use_cuda_graph=True, num_substeps=4, gravity=[0., 0., -9.81]))
        actual = configure_hoi_capacity(config)
        self.assertIsNot(actual.physics.solver_cfg, solver)
        self.assertEqual((solver.nconmax, solver.njmax, config.physics.use_cuda_graph), (128, 500, True))
        self.assertEqual((actual.physics.solver_cfg.nconmax, actual.physics.solver_cfg.njmax), (512, 2048))
        self.assertEqual({k: v for k, v in vars(actual.physics.solver_cfg).items() if k not in ("nconmax", "njmax")},
                         {k: v for k, v in vars(solver).items() if k not in ("nconmax", "njmax")})
        self.assertEqual((actual.dt, actual.physics.num_substeps, actual.physics.gravity), (.02, 4, [0., 0., -9.81]))
        self.assertFalse(actual.physics.use_cuda_graph)

    def test_shallow_contact_places_only_support_surface_inside_palm_plane(self):
        vertices = torch.tensor([[x, y, z] for x in [-.2, .2] for y in [-.1, .1] for z in [-.3, .3]])
        rotations = so3_exp(torch.tensor([[.3, -.2, .1], [-.6, .1, .2]]))
        offsets = (rotations @ vertices.T).transpose(-1, -2)
        normals = torch.tensor([[1., 0., 0.], [0., -.8, .6]])
        palm = torch.tensor([[1., 2., 3.], [-2., 1., .5]])
        position, support = shallow_support_contact(palm, normals, offsets)
        distances = ((position[:, None] + offsets - palm[:, None]) * normals[:, None]).sum(-1)
        torch.testing.assert_close(distances.amin(-1), torch.full((2,), -.002), atol=2e-7, rtol=0)
        target = position + offsets[torch.arange(2), support]
        torch.testing.assert_close(target, palm - .002 * normals)
        with self.assertRaisesRegex(ValueError, "unit outward"):
            shallow_support_contact(palm, normals * 2, offsets)

    def test_foot_support_tie_break_uses_lower_box_edge_without_changing_hand_default(self):
        # Put upper corners first to reproduce the historical arbitrary argmin.
        vertices = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.1, .1) for z in (.3, -.3)], dtype=torch.float64)
        rotations = so3_exp(torch.tensor([[0., 0., .4], [0., 0., -.7]], dtype=torch.float64))
        offsets = (rotations @ vertices.T).transpose(-1, -2)
        normals = rotations[:, :, 0]
        surface = torch.tensor([[.1, .2, .03], [.3, -.2, .02]], dtype=torch.float64)
        hand_origin, hand_index = shallow_support_contact(surface, normals, offsets)
        old_index = (offsets * normals[:, None]).sum(-1).argmin(-1)
        self.assertTrue(torch.equal(hand_index, old_index))
        expected_hand = surface - offsets[torch.arange(2), old_index] - .002 * normals
        self.assertTrue(torch.equal(hand_origin, expected_hand))
        foot_origin, foot_index = shallow_support_contact(surface, normals, offsets, lowest_support=True)
        selected = offsets[torch.arange(2), foot_index]
        self.assertTrue(bool((selected[:, 2] == -.3).all()))
        minimum_z = (foot_origin[:, None, 2] + offsets[..., 2]).amin(-1)
        torch.testing.assert_close(minimum_z, surface[:, 2], atol=1e-12, rtol=0)
        overlap = ((foot_origin[:, None] + offsets - surface[:, None]) * normals[:, None]).sum(-1).amin(-1)
        torch.testing.assert_close(overlap, torch.full((2,), -.002, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertTrue(bool(((hand_origin[:, None, 2] + offsets[..., 2]).amin(-1) < -.5).all()))

    def test_tilted_foot_pair_lift_preserves_contact_and_relative_state(self):
        vertices = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.1, .1) for z in (-.3, .3)], dtype=torch.float64)
        rotation = so3_exp(torch.tensor([[.3, -.5, .2], [-.6, .2, -.4]], dtype=torch.float64))
        offsets = (rotation @ vertices.T).transpose(-1, -2)
        normals = torch.tensor([[.8, 0., -.6], [.6, 0., -.8]], dtype=torch.float64)
        surface = torch.tensor([[.1, .2, .03], [-.3, -.2, .05]], dtype=torch.float64)
        root = torch.tensor([[0., 0., .8], [1., -2., .9]], dtype=torch.float64)
        origin, _ = shallow_support_contact(surface, normals, offsets, lowest_support=True)
        new_root, new_origin, new_surface, lift, initial, final = lift_foot_probe_above_ground(root, origin, surface, offsets)
        self.assertTrue(bool((initial < 0).all()))
        self.assertTrue(bool((lift > 0).all()))
        torch.testing.assert_close(final, torch.full((2,), .002, dtype=torch.float64), atol=1e-12, rtol=0)
        torch.testing.assert_close(new_origin - new_root, origin - root, atol=1e-12, rtol=0)
        torch.testing.assert_close(new_surface - new_root, surface - root, atol=1e-12, rtol=0)
        before = ((origin[:, None] + offsets - surface[:, None]) * normals[:, None]).sum(-1)
        after = ((new_origin[:, None] + offsets - new_surface[:, None]) * normals[:, None]).sum(-1)
        torch.testing.assert_close(before, after, atol=1e-12, rtol=0)
        torch.testing.assert_close(after.amin(-1), torch.full((2,), -.002, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_independent_mujoco_convex_distance_audit(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("optional CPU MuJoCo unavailable; no native evidence inferred")
        from hoi.v1.contacts import compiled_geom_roles
        vertices = torch.tensor([[x, y, z] for x in [-.1, .1] for y in [-.1, .1] for z in [-.1, .1]], dtype=torch.float64)
        faces = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                              [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        v = " ".join(str(float(x)) for x in vertices.flatten())
        f = " ".join(str(int(x)) for x in faces.flatten())
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset><mesh name="cube" vertex="{v}" face="{f}"/></asset><worldbody>
            <geom name="ground" type="plane" size="10 10 .1"/>
            <body name="left_wrist_yaw_link" pos="0 .5 1"><geom name="left_rubber_hand" type="mesh" mesh="cube" pos=".0415 .003 0"/></body>
            <body name="right_wrist_yaw_link" pos="0 -.5 1"><geom name="right_rubber_hand" type="mesh" mesh="cube" pos=".0415 -.003 0"/></body>
            <body name="left_ankle_roll_link" pos="0 .1 .1"><geom name="left_sole" type="sphere" size=".005"/></body>
            <body name="right_ankle_roll_link" pos="0 -.1 .1"><geom name="right_sole" type="sphere" size=".005"/></body>
            <body name="env_0_Object" pos="2 0 1"><freejoint/><geom name="object_mesh" type="mesh" mesh="cube"/></body>
            </worldbody></mujoco>''')
        hull_vertices, hull_faces, hull_proof = compiled_convex_mesh(model, 0)
        self.assertEqual((len(hull_vertices), len(hull_faces)), (8, 12))
        self.assertEqual(hull_proof["hull_vertex_count"], 8)
        roles, envs, _ = compiled_geom_roles(model)
        env = HOITrackingSim.__new__(HOITrackingSim)
        env.contacts = SimpleNamespace(solver=SimpleNamespace(mj_model=model), roles=roles, geom_envs=envs,
                                       topology="one_mujoco_world_per_environment")
        mesh = {"vertices": vertices, "faces": faces}
        env.asset = {"asset_id": 0}
        env.mdp = SimpleNamespace(shapes=SimpleNamespace(geometries=[{"collision_meshes": [mesh]}],
                                                         hands={s: {"collision_meshes": [mesh]} for s in ("left", "right")}))
        proof = env._native_distance_audit()
        self.assertTrue(proof["passed"])
        self.assertEqual(len(proof["samples"]), 4)
        self.assertTrue(all(row["abs_error_m"] < 5e-5 for row in proof["samples"]))

    def test_native_hull_graph_is_collision_authority(self):
        import numpy as np
        facets = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        graph = np.r_[4, 4, np.zeros(4, dtype=int), np.arange(4), np.zeros(16, dtype=int), facets.flatten()]
        model = SimpleNamespace(mesh_graphadr=[0], mesh_graph=graph, mesh_vertadr=[0], mesh_vertnum=[5], mesh_facenum=[6],
                                mesh_vert=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, .2, .2]]))
        vertices, faces, proof = compiled_convex_mesh(model, 0)
        self.assertEqual(vertices.shape, (4, 3))
        self.assertTrue(np.array_equal(faces, facets))
        self.assertEqual(proof["source_vertex_ids"], [0, 1, 2, 3])
        self.assertEqual((proof["raw_vertex_count"], proof["hull_vertex_count"]), (5, 4))
        model.mesh_graphadr = [-1]
        with self.assertRaises(RuntimeError):
            compiled_convex_mesh(model, 0)

    def test_com_link_conversion_in_rotated_world(self):
        R = so3_exp(torch.tensor([[.2, -.3, .4]]))
        v, w, com = torch.tensor([[1., 2., 3.]]), torch.tensor([[.5, -.3, 1.2]]), torch.tensor([[.1, .2, -.1]])
        vcom = link_to_com_velocity(R, v, w, com)
        torch.testing.assert_close(vcom - v, torch.linalg.cross(w, (R @ com.unsqueeze(-1)).squeeze(-1)))
        torch.testing.assert_close(com_to_link_velocity(R, vcom, w, com), v)

    def test_pair_identity_rejects_hand_ground_and_self_contact(self):
        roles = torch.tensor([GROUND, LEFT, RIGHT, OBJECT, OTHER])
        # Both directions count; hand-ground/self/other never do.
        pairs = torch.tensor([[1, 3], [3, 1], [2, 3], [3, 0], [1, 0], [1, 2], [4, 3], [-1, 3]])
        impulse = aggregate_pairs(roles, pairs, torch.zeros(8, dtype=torch.long), torch.ones(8) * 10,
                                  num_envs=1, substep_dt=.005)
        torch.testing.assert_close(impulse, torch.tensor([[.1, .05, .05, 0., 0.]]))

    def test_interval_sums_all_four_substeps(self):
        roles = torch.tensor([LEFT, RIGHT, OBJECT, GROUND])
        sums = torch.zeros(2, len(CHANNELS))
        for n in [2., 4., 0., 8.]:
            sums += aggregate_pairs(roles, torch.tensor([[0, 2], [2, 3], [1, 2]]),
                                    torch.tensor([0, 1, 1]), torch.tensor([n, 10., 1.]), num_envs=2, substep_dt=.005)
        force, labels = normal_impulse_to_label(sums)
        torch.testing.assert_close(force, torch.tensor([[3.5, 0., 0., 0., 0.], [0., 1., 10., 0., 0.]]))
        self.assertEqual(labels.tolist(), [[True, False, False, False, False], [False, False, True, False, False]])

    def test_foot_roles_use_actual_active_ankle_roll_body_not_visual_or_shin_names(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="ground" type="plane" size="10 10 .1"/>
          <body name="left_wrist_yaw_link"><geom name="left_rubber_hand" size=".01"/></body>
          <body name="right_wrist_yaw_link"><geom name="right_rubber_hand" size=".01"/></body>
          <body name="_World_envs_env_0_Robot_left_ankle_roll_link">
            <geom name="actual_left_toe" pos=".12 .03 -.03" size=".005"/>
            <geom name="left_visual" size=".1" contype="0" conaffinity="0"/>
          </body>
          <body name="/World/envs/env_1/Robot/right_ankle_roll_link"><geom name="actual_right_toe" size=".005"/></body>
          <body name="left_knee_link"><geom name="misleading_left_ankle_roll_link" size=".02"/></body>
          <body name="left_ankle_pitch_link"><geom name="misleading_foot" size=".02"/></body>
          <body name="foo_left_ankle_roll_link_old"><geom name="suffix_old" size=".02"/></body>
          <body name="env_0_Object"><geom name="object" size=".02"/></body>
        </worldbody></mujoco>''')
        roles, envs, names = compiled_geom_roles(model)
        by_name = dict(zip(names, roles))
        self.assertEqual(by_name["actual_left_toe"], LEFT_FOOT)
        self.assertEqual(by_name["actual_right_toe"], RIGHT_FOOT)
        self.assertEqual(envs[names.index("actual_left_toe")], 0)
        self.assertEqual(envs[names.index("actual_right_toe")], 1)
        for name in ("left_visual", "misleading_left_ankle_roll_link", "misleading_foot", "suffix_old"):
            self.assertEqual(by_name[name], OTHER, name)
        model.geom_contype[names.index("actual_right_toe")] = 0
        model.geom_conaffinity[names.index("actual_right_toe")] = 0
        with self.assertRaisesRegex(RuntimeError, "Missing native pair entities"):
            compiled_geom_roles(model)

    def test_foot_object_pair_directions_and_ground_negative_controls(self):
        self.assertEqual(CHANNELS, ("left_hand_object", "right_hand_object", "object_ground", "left_foot_object", "right_foot_object"))
        self.assertEqual(LABEL_SCHEMA, "hoi_v1_pair_normal_impulse_v2")
        roles = torch.tensor([LEFT_FOOT, RIGHT_FOOT, OBJECT, GROUND, OTHER])
        pairs = torch.tensor([[0, 2], [2, 0], [1, 2], [2, 1], [0, 3], [3, 1], [4, 2], [0, 1], [9, 2]])
        force = torch.tensor([4., 6., 8., 2., 100., 200., 300., 400., 500.])
        ids = torch.zeros(len(pairs), dtype=torch.long)
        impulse = aggregate_pairs(roles, pairs, ids, force, num_envs=1, substep_dt=.005)
        torch.testing.assert_close(impulse, torch.tensor([[0., 0., 0., .05, .05]]))
        ground = aggregate_ground_pairs(roles, pairs, ids, force, roles=(LEFT_FOOT, RIGHT_FOOT), num_envs=1, substep_dt=.005)
        torch.testing.assert_close(ground, torch.tensor([[.5, 1.]]))

    def test_four_substep_foot_impulses_keep_strict_greater_than_one_newton(self):
        roles = torch.tensor([LEFT_FOOT, RIGHT_FOOT, OBJECT])
        sums = torch.zeros(2, len(CHANNELS))
        for left in (2., 4., 0., 8.):
            sums += aggregate_pairs(roles, torch.tensor([[0, 2], [2, 1]]), torch.tensor([0, 1]),
                                    torch.tensor([left, 1.]), num_envs=2, substep_dt=.005)
        forces, labels = normal_impulse_to_label(sums)
        torch.testing.assert_close(forces[:, 3:], torch.tensor([[3.5, 0.], [0., 1.]]))
        self.assertEqual(labels[:, 3:].tolist(), [[True, False], [False, False]])

    def test_native_recorder_isolates_named_environments_and_rejects_cross_env_dynamics(self):
        # The same native after_substep reduction runs against a thin CPU array
        # view; only the backend force API is replaced, not its pair logic.
        recorder = PairContactRecorder.__new__(PairContactRecorder)
        recorder.num_envs, recorder.control_dt, recorder.num_substeps = 2, .02, 4
        recorder.topology = "one_world_with_named_environments"
        recorder.roles = [LEFT_FOOT, OBJECT, RIGHT_FOOT, OBJECT, GROUND]
        recorder.geom_role = torch.tensor(recorder.roles)
        recorder.geom_env = torch.tensor([0, 0, 1, 1, -1])
        recorder.indices = torch.arange(3)
        data = SimpleNamespace(contact=SimpleNamespace(geom=torch.tensor([[0, 1], [2, 3], [4, 2]]),
                                                     worldid=torch.zeros(3, dtype=torch.long)), nacon=torch.tensor([3]))
        recorder.solver = SimpleNamespace(mjw_model=object(), mjw_data=data)
        recorder.wp = SimpleNamespace(to_torch=lambda value: value)
        recorder.mjw = SimpleNamespace(contact_force=lambda *args: None)
        recorder.ids = None
        recorder.forces = torch.zeros(3, 6)
        recorder.forces[:, 0] = torch.tensor([8., 4., 3.])
        recorder.impulse = torch.zeros(2, len(CHANNELS))
        recorder.hand_ground_impulse = torch.zeros(2, 2)
        recorder.foot_ground_impulse = torch.zeros(2, 2)
        recorder.calls = 0
        recorder.after_substep(.005)
        torch.testing.assert_close(recorder.impulse[:, 3:], torch.tensor([[.04, 0.], [0., .02]]))
        torch.testing.assert_close(recorder.foot_ground_impulse, torch.tensor([[0., 0.], [0., .015]]))
        data.contact.geom[0] = torch.tensor([0, 3])
        with self.assertRaisesRegex(RuntimeError, "crosses environment boundaries"):
            recorder.after_substep(.005)

    def test_compiled_foot_surface_uses_real_sphere_support_and_rejects_proxy(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="left_ankle_roll_link">
          <geom name="heel" pos="-.05 .025 -.03" size=".005"/>
          <geom name="toe" pos=".12 .03 -.03" size=".005"/>
        </body></worldbody></mujoco>''')
        point, normal, evidence = compiled_foot_probe_surface(model, [0, 1])
        torch.testing.assert_close(point, torch.tensor([.125, .03, -.03], dtype=torch.float64))
        torch.testing.assert_close(normal, torch.tensor([1., 0., 0.], dtype=torch.float64))
        self.assertEqual(evidence["geom_name"], "toe")
        self.assertEqual(evidence["body_name"], "left_ankle_roll_link")
        self.assertEqual(evidence["radius_m"], .005)
        vertices = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.1, .1) for z in (-.3, .3)], dtype=torch.float64)
        position, support = shallow_support_contact(point[None], normal[None], vertices[None])
        actual = position[0] + vertices[support[0]]
        torch.testing.assert_close(actual, point - .002 * normal)
        # This placement really intersects the compiled sphere by exactly 2mm.
        center = torch.tensor(model.geom_pos[evidence["geom_id"]])
        self.assertAlmostEqual(float((actual - center).norm()), .005 - .002)
        model.geom_type[1] = mujoco.mjtGeom.mjGEOM_BOX
        with self.assertRaisesRegex(RuntimeError, "actual compiled ankle-roll collision spheres"):
            compiled_foot_probe_surface(model, [0, 1])

    def test_compiled_mesh_toe_surface_is_facet_interior_and_global_support(self):
        model = mesh_foot_fixture()
        point, normal, evidence = compiled_foot_probe_surface(model, [0, 1])
        self.assertEqual(evidence["primitive"], "actual_compiled_convex_mesh")
        self.assertEqual(evidence["selection"], "interior_of_exposed_global_support_facet")
        self.assertEqual(evidence["all_foot_hull_count"], 2)
        self.assertGreaterEqual(len(evidence["facet_vertex_ids"]), 4)
        torch.testing.assert_close(point, torch.tensor([.11, 0., -.03], dtype=torch.float64), atol=1e-7, rtol=0)
        torch.testing.assert_close(normal, torch.tensor([1., 0., 0.], dtype=torch.float64), atol=1e-7, rtol=0)
        object_vertices = torch.tensor([[x, y, z] for x in (-.2, .2) for y in (-.1, .1) for z in (-.3, .3)], dtype=torch.float64)
        origin, _ = shallow_support_contact(point[None], normal[None], object_vertices[None])
        distances = (origin[0] + object_vertices - point) @ normal
        self.assertAlmostEqual(float(distances.min()), -.002, places=10)
        self.assertLessEqual(evidence["maximum_other_vertex_plane_excess_m"], 1e-7)
        self.assertTrue(bool(torch.isfinite(point).all()))
        model.mesh_graphadr[0] = -1
        with self.assertRaisesRegex(RuntimeError, "collision convex graph"):
            compiled_foot_probe_surface(model, [0, 1])

    def test_rotated_mesh_ground_uses_all_hull_world_vertices(self):
        model = mesh_foot_fixture()
        rotation = so3_exp(torch.tensor([[.7, .5, -.2], [-.4, -.8, .1]], dtype=torch.float64))
        position = torch.tensor([[1., 2., .4], [-2., 1., .7]], dtype=torch.float64)
        minimum, evidence = compiled_foot_world_minimum(model, [0, 1], position, rotation)
        # Independent raw compiled MuJoCo geom transforms (not visual STL).
        vertices = []
        from supermimic.rotations import quat_to_mat
        for geom in (0, 1):
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            raw = torch.tensor(model.mesh_vert[start:start + count], dtype=torch.float64)
            vertices.append(raw @ quat_to_mat(torch.tensor(model.geom_quat[geom])).T + torch.tensor(model.geom_pos[geom]))
        vertices = torch.cat(vertices)
        expected = ((rotation @ vertices.T).transpose(-1, -2) + position[:, None])[..., 2].amin(-1)
        torch.testing.assert_close(minimum, expected, atol=1e-12, rtol=0)
        shifted = position.clone()
        shifted[:, 2] -= minimum + .002
        actual, _ = compiled_foot_world_minimum(model, [0, 1], shifted, rotation)
        torch.testing.assert_close(actual, torch.full((2,), -.002, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertEqual(evidence["hull_count"], 2)

    def test_compiled_foot_audit_covers_every_environment_and_rejects_spheres(self):
        import mujoco
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        manifest = mesh_foot_manifest(temporary.name)
        model = mesh_foot_fixture(2, named=True)
        kwargs = dict(num_envs=2, topology="one_world_with_named_environments", manifest=manifest,
                      manifest_sha256="c" * 64, robot_asset_sha256="a" * 64,
                      reference_model=mesh_foot_fixture(), reference_mjcf_sha256="d" * 64)
        report = compiled_foot_geometry_audit(model, **kwargs)
        self.assertTrue(report["passed"])
        for side in ("left", "right"):
            self.assertEqual(report["sides"][side]["model_collision_geom_count"], 4)
            self.assertEqual(report["sides"][side]["model_environment_count"], 2)
        # A sphere mixed into one environment is not hidden by env0-only audit.
        model.geom_type[7] = mujoco.mjtGeom.mjGEOM_SPHERE
        with self.assertRaisesRegex(RuntimeError, "all actual compiled.*meshes"):
            compiled_foot_geometry_audit(model, **kwargs)
        model = mesh_foot_fixture()
        single = compiled_foot_geometry_audit(model, num_envs=4096, topology="one_mujoco_world_per_environment",
            manifest=manifest, manifest_sha256="c" * 64, robot_asset_sha256="a" * 64,
            reference_model=mesh_foot_fixture(), reference_mjcf_sha256="d" * 64)
        self.assertEqual(single["sides"]["left"]["model_environment_count"], 1)
        self.assertEqual(single["sides"]["left"]["simulated_environment_count"], 4096)
        with self.assertRaisesRegex(RuntimeError, "every environment"):
            compiled_foot_geometry_audit(mesh_foot_fixture(1, named=True), **kwargs)
        changed = mesh_foot_fixture(2, named=True)
        changed.body_mass[3] *= 1.2
        with self.assertRaisesRegex(RuntimeError, "mass/COM/inertia differs"):
            compiled_foot_geometry_audit(changed, **kwargs)
        for error in ("offset", "scale"):
            changed = mesh_foot_fixture(2, named=True)
            if error == "offset":
                changed.geom_pos[7, 0] += .003
            else:
                changed.mesh_vert[:] *= 1.1
            with self.assertRaisesRegex(RuntimeError, "differ from bound ankle-local cooked geometry"):
                compiled_foot_geometry_audit(changed, **kwargs)
        # Reordering importer hulls is allowed, not confused with a bad mount.
        reordered = mesh_foot_fixture(2, named=True)
        saved = reordered.geom_pos[6].copy()
        reordered.geom_pos[6] = reordered.geom_pos[7]
        reordered.geom_pos[7] = saved
        matched = compiled_foot_geometry_audit(reordered, **kwargs)
        self.assertTrue(matched["sides"]["left"]["native_vs_cooked_geometry"]["passed"])

    def test_substep_adapter_preserves_solver_call(self):
        calls = []
        solver = SimpleNamespace(step=lambda *a, **kw: calls.append((a, kw)) or 17, native="preserved")
        recorder = SimpleNamespace(after_substep=lambda dt: calls.append(dt))
        adapter = SolverContactAdapter(solver, recorder)
        self.assertEqual(adapter.step("in", "out", "control", "contacts", .005, observables="x"), 17)
        self.assertEqual(calls, [(("in", "out", "control", "contacts", .005), {"observables": "x"}), .005])
        self.assertEqual(adapter.native, "preserved")

    def test_readback_context_order(self):
        context = physical_context(torch.tensor([2.]), torch.tensor([[.1, .2, .3]]),
                                   torch.tensor([[[1., .1, .2], [.1, 2., .3], [.2, .3, 3.]]]),
                                   torch.tensor([.6]), torch.tensor([.3]))
        torch.testing.assert_close(context, torch.tensor([[2., .1, .2, .3, 1., 2., 3., .1, .2, .3, .6, .3]]))

    def test_invalid_inertia_rejected(self):
        p = asset(0)["physical_profile"]
        validate_physical_profile(p)
        p["inertia"][0][0] = -1.
        with self.assertRaises(ValueError):
            validate_physical_profile(p)

    def test_palm_rays_select_inner_collider_surface(self):
        vertices = torch.tensor([[x, y, z] for x in [0., .08] for y in [-.02, .02] for z in [-.03, .03]])
        faces = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                              [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        mesh = {"vertices": vertices, "faces": faces}
        left, definition = palm_patch_points([mesh], [mesh], "left")
        right, _ = palm_patch_points([mesh], [mesh], "right")
        torch.testing.assert_close(left[:, 1], torch.full((16,), -.02))
        torch.testing.assert_close(right[:, 1], torch.full((16,), .02))
        self.assertTrue(definition["palm_selection_verified"])
        self.assertEqual(len(definition["source_triangle_indices"]), 16)

    def test_quotas_preserve_total_and_proportions(self):
        bundle = {"assets": [asset(0), asset(1), asset(2)]}
        motion = SimpleNamespace(bin_asset=torch.tensor([0] + [1] * 3 + [2] * 6))
        specs = allocate_workers(bundle, motion, 4096)
        self.assertEqual(sum(s.num_envs for s in specs), 4096)
        self.assertEqual([s.train_bins for s in specs], [1, 3, 6])
        self.assertGreater(specs[2].num_envs, specs[1].num_envs)
        report = resource_preflight(specs, device="cpu")
        self.assertFalse(report["gpu_memory_budget_validated"])
        with self.assertRaises(ValueError):
            resource_preflight(specs, device="cpu", max_processes=2)

    def test_exact_scale_and_profile_separate_groups(self):
        a, b = asset(0), asset(0, 1.00000001)
        self.assertNotEqual(geometry_group(a), geometry_group(b))
        b = asset(0)
        b["physical_profile"]["mass"] = 2.1
        self.assertNotEqual(geometry_group(a), geometry_group(b))
        with self.assertRaises(ValueError):
            allocate_workers({"assets": [a, b]}, SimpleNamespace(bin_asset=torch.tensor([0, 1])), 4096)


if __name__ == "__main__":
    unittest.main()
