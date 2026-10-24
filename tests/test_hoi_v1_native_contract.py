"""CPU contracts only; these tests do not claim native physics validation."""
import unittest
from types import SimpleNamespace

import torch

from hoi.v1.contacts import (GROUND, LEFT, RIGHT, OBJECT, OTHER, SolverContactAdapter,
                             aggregate_pairs, normal_impulse_to_label)
from hoi.v1.native import HOITrackingSim, compiled_convex_mesh, com_to_link_velocity, link_to_com_velocity, palm_patch_points, physical_context, validate_physical_profile, shallow_support_contact, configure_hoi_capacity
from hoi.v1.workers import allocate_workers, geometry_group, resource_preflight
from supermimic.rotations import so3_exp


def asset(i, scale=1.):
    return {"asset_id": i, "object_asset": f"box{i}", "exact_scale": [scale] * 3,
            "collision_hash": "c" * 64, "physical_profile": {
                "mass": 2., "com": [.1, -.05, .2], "inertia": [[.3, 0., 0.], [0., .4, 0.], [0., 0., .5]],
                "friction": .6, "ground_friction": .3}}


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
        torch.testing.assert_close(impulse, torch.tensor([[.1, .05, .05]]))

    def test_interval_sums_all_four_substeps(self):
        roles = torch.tensor([LEFT, RIGHT, OBJECT, GROUND])
        sums = torch.zeros(2, 3)
        for n in [2., 4., 0., 8.]:
            sums += aggregate_pairs(roles, torch.tensor([[0, 2], [2, 3], [1, 2]]),
                                    torch.tensor([0, 1, 1]), torch.tensor([n, 10., 1.]), num_envs=2, substep_dt=.005)
        force, labels = normal_impulse_to_label(sums)
        torch.testing.assert_close(force, torch.tensor([[3.5, 0., 0.], [0., 1., 10.]]))
        self.assertEqual(labels.tolist(), [[True, False, False], [False, False, True]])

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
