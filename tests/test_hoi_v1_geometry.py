"""Metric geometry, exact collider queries, deterministic FPS, and SDF gradients."""
import unittest
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from hoi.v1.geometry import (ShapeAssetBank, build_geometry_cache, collision_distance, deterministic_surface_points,
    exact_mesh_gap, expected_schedule_from_distances, farthest_point_sample, palm_world_points,
    point_mesh_signed_distance, query_grid, tensor_sha256, validate_mesh,
    _point_mesh_signed_distance, _prepare_exact_meshes, _rigid_mesh_vertices, _exact_mesh_gap)
from hoi.data.manifest import build_manifest, robot_joint_names, write_manifest
from hoi.v1.prepare import (load_bundle, prepare_bundle, _validate_palm_evidence, _reference_contacts,
                           _reference_contact_quality, _aggregate_reference_contact_quality)
from hoi.hand_collision import HAND_SHA, ORIGIN
from hoi.v1.merge_recipes import merge_recipes


def cube(half=.5):
    vertices = torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=torch.float64) * half
    faces = torch.tensor([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return {"vertices": vertices, "faces": faces}


class BundleKinematics:
    """Top-level picklable CPU fixture for process-parallel preparation tests."""
    body_names = ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        n = 3 if all_bodies else 14
        return (p[..., None, :].expand(*p.shape[:-1], n, 3), R[..., None, :, :].expand(*R.shape[:-2], n, 3, 3),
                v[..., None, :].expand(*v.shape[:-1], n, 3), w[..., None, :].expand(*w.shape[:-1], n, 3))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.mesh = cube()
        cls.cache = build_geometry_cache([cls.mesh], grid_resolution=20, candidate_count=512, max_grid_error=.05)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_signed_distance_and_watertight_validation(self):
        v, f = self.mesh["vertices"], self.mesh["faces"]
        p = torch.tensor([[0,0,0],[.6,0,0],[.5,.1,0],[1,1,0]], dtype=torch.float64)
        result = point_mesh_signed_distance(p, v, f)
        torch.testing.assert_close(result, torch.tensor([-.5,.1,0,2**-.5], dtype=torch.float64), atol=1e-10, rtol=0)
        with self.assertRaisesRegex(ValueError, "watertight"):
            validate_mesh(v, f[:-1])
        reverse = f.clone()
        reverse[0] = reverse[0].flip(0)
        with self.assertRaisesRegex(ValueError, "winding"):
            validate_mesh(v, reverse)

    def test_signed_aabb_pruning_matches_bruteforce_inside_overlap_and_transforms(self):
        angle = torch.tensor(.53, dtype=torch.float64)
        c, s = angle.cos(), angle.sin()
        rot = torch.stack((c,-s,c*0,s,c,c*0,c*0,c*0,c*0+1)).reshape(3,3)
        translated = {**self.mesh, "vertices": (rot @ self.mesh["vertices"].T).T + torch.tensor([.6,.1,.2])}
        nested = cube(.2)
        distant = {**self.mesh, "vertices": self.mesh["vertices"] + torch.tensor([4.,2.,-1.])}
        generator = torch.Generator().manual_seed(19)
        points = torch.cat((torch.rand((512,3), generator=generator, dtype=torch.float64)*6-2,
                            torch.tensor([[0,0,0],[.2,.2,.2],[.5,.5,.5],[.5,0,0],[-.5,0,0],[.6,.1,.2],[4.,2.,-1.]], dtype=torch.float64)))
        for meshes in ([self.mesh, nested, translated, distant], [distant, translated, nested, self.mesh]):
            brute = torch.stack([point_mesh_signed_distance(points, m["vertices"], m["faces"]) for m in meshes]).min(0).values
            actual = collision_distance(points, meshes)
            torch.testing.assert_close(actual, brute, atol=1e-12, rtol=0)
            torch.testing.assert_close(actual.sign(), brute.sign(), atol=0, rtol=0)
        near = torch.tensor([[.51,0,0],[.52,.1,0]], dtype=torch.float64)
        with patch("hoi.v1.geometry._point_mesh_signed_distance", wraps=_point_mesh_signed_distance) as query:
            collision_distance(near, [self.mesh, distant])
            self.assertGreater(query.call_count, 0)
            self.assertLess(sum(len(call.args[0]) for call in query.call_args_list), len(near)*2)

    def test_surface_fps_is_reproducible_and_keeps_metric_origin(self):
        offset = torch.tensor([1.2, -.3, .8], dtype=torch.float64)
        mesh = {**self.mesh, "vertices": self.mesh["vertices"] + offset}
        first = deterministic_surface_points([mesh], 512, 3)
        second = deterministic_surface_points([mesh], 512, 3)
        self.assertEqual(tensor_sha256(first), tensor_sha256(second))
        points = farthest_point_sample(first)
        self.assertEqual(points.shape, (128, 3))
        self.assertGreater(float(points[:, 0].min()), .69)
        self.assertLess(float(point_mesh_signed_distance(points, mesh["vertices"], mesh["faces"]).abs().max()), 1e-6)
        self.assertEqual(tensor_sha256(points), tensor_sha256(farthest_point_sample(second)))

    def test_grid_gradient_and_exterior_gradient_match_finite_difference(self):
        g = self.cache
        p = torch.tensor([[.53,.12,.1],[.8,.12,.1]], dtype=torch.float64, requires_grad=True)
        value = query_grid(g["sdf_grid"], g["grid_min"], g["grid_max"], p)
        gradient, = torch.autograd.grad(value.sum(), p)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(gradient[:, 0], torch.ones(2, dtype=torch.float64), atol=1e-5, rtol=0)
        eps = 1e-5
        delta = torch.tensor([[eps,0,0],[eps,0,0]], dtype=torch.float64)
        fd = (query_grid(g["sdf_grid"], g["grid_min"], g["grid_max"], p.detach() + delta) - query_grid(g["sdf_grid"], g["grid_min"], g["grid_max"], p.detach() - delta)) / (2*eps)
        torch.testing.assert_close(gradient[:, 0], fd, atol=1e-6, rtol=0)

    def test_exact_surface_gap_and_penetration(self):
        separated = {**self.mesh, "vertices": self.mesh["vertices"] + torch.tensor([1.03,0,0])}
        gap, penetration = exact_mesh_gap([self.mesh], [separated])
        self.assertAlmostEqual(float(gap), .03, places=6)
        self.assertEqual(float(penetration), 0)
        intersect = {**self.mesh, "vertices": self.mesh["vertices"] + torch.tensor([.9,.1,.1])}
        gap, penetration = exact_mesh_gap([self.mesh], [intersect])
        self.assertEqual(float(gap), 0)
        self.assertGreater(float(penetration), .02)

    def test_pair_aabb_pruning_keeps_exact_gap_and_all_overlap_penetration(self):
        def shifted(mesh, offset):
            return {**mesh, "vertices": mesh["vertices"] + torch.tensor(offset)}
        angle = torch.tensor(.71, dtype=torch.float64)
        c, s = angle.cos(), angle.sin()
        rotation = torch.stack((c,-s,c*0,s,c,c*0,c*0,c*0,c*0+1)).reshape(3,3)
        rotated = {**self.mesh, "vertices": self.mesh["vertices"] @ rotation.T + torch.tensor([1.01,.2,.1])}
        first = [self.mesh, shifted(cube(.2), [4.,1.,.3]), shifted(self.mesh, [-5.,2.,1.])]
        cases = [
            [shifted(self.mesh, [1.03,0,0]), shifted(self.mesh, [9.,2.,1.])],
            [shifted(self.mesh, [1.,0,0]), shifted(cube(.1), [4.,1.,.3])],
            [shifted(self.mesh, [.9,.1,.1]), shifted(cube(.15), [4.,1.,.3])],
            [rotated, shifted(self.mesh, [8.,1.,.3])],
            [cube(.1), shifted(cube(.15), [4.02,1.,.3])]]
        for second in cases:
            brute = [exact_mesh_gap([a], [b]) for a in first for b in second]
            expected = min(float(g) for g, _ in brute), max(float(p) for _, p in brute)
            for a, b in ((first, second), (first[::-1], second[::-1]), (second, first)):
                actual = exact_mesh_gap(a, b)
                self.assertAlmostEqual(float(actual[0]), expected[0], places=12)
                self.assertAlmostEqual(float(actual[1]), expected[1], places=12)
        with patch("hoi.v1.geometry._point_mesh_signed_distance", wraps=_point_mesh_signed_distance) as query:
            exact_mesh_gap(first, cases[0])
            self.assertGreater(query.call_count, 0)
            self.assertLess(query.call_count, 2 * len(first) * len(cases[0]))
        # An initial touching pair gives zero gap; the nested pair must still
        # contribute its larger penetration audit despite having no crossing.
        actual = exact_mesh_gap(first, cases[1])
        self.assertEqual(float(actual[0]), 0.)
        self.assertGreater(float(actual[1]), .09)

    def test_edge_face_crossing_without_interior_vertices_and_touch_are_unknown(self):
        first = {**self.mesh, "vertices": self.mesh["vertices"] * torch.tensor([4.,.1,.1])}
        second = {**self.mesh, "vertices": self.mesh["vertices"] * torch.tensor([.1,4.,.1])}
        zero, eye = torch.zeros((1,3)), torch.eye(3)[None]
        reference = {"robot": {"root_pos": zero, "root_rot": eye, "derived_ref_v": zero,
                              "derived_ref_omega": zero, "joint_pos": torch.zeros((1,29)),
                              "derived_ref_joint_qd": torch.zeros((1,29))},
                     "object": {"pos": zero, "rot": eye}}
        for z, must_intersect in ((0., True), (.1, True), (.2, False)):
            moved = {**first, "vertices": first["vertices"] + torch.tensor([0,0,z])}
            gap, depth, crossing = exact_mesh_gap([moved], [second], return_intersection=True)
            self.assertEqual(crossing, must_intersect)
            self.assertEqual(float(depth), 0.)
            # All vertices lie outside the other thin rod, even at the true
            # crossing. Vertex-depth-only masking therefore misses this case.
            d0 = point_mesh_signed_distance(moved["vertices"], second["vertices"], second["faces"])
            d1 = point_mesh_signed_distance(second["vertices"], moved["vertices"], moved["faces"])
            self.assertGreater(float(d0.min()), 0.)
            self.assertGreater(float(d1.min()), 0.)
            hands = {side: {"mount": ORIGIN[side], "collision_meshes": [
                {**moved, "vertices": moved["vertices"] - torch.tensor(ORIGIN[side], dtype=torch.float64)}]}
                for side in ("left", "right")}
            labels = _reference_contacts(reference, BundleKinematics(), hands, {"collision_meshes": [second]})
            self.assertTrue(bool((labels["surface_intersection"] == must_intersect).all()))
            self.assertTrue(bool((labels["geometry_valid"] == (not must_intersect)).all()))
            self.assertTrue(bool((labels["expected_valid"] == (not must_intersect)).all()))
            torch.testing.assert_close(labels["surface_gap_m"], torch.full((1,2), float(gap)))
            self.assertIn("unresolved_boundary_touch", labels["metadata"]["penetration_policy"])
            self.assertEqual(len(exact_mesh_gap([moved], [second])), 2)

    def test_prevalidated_rigid_path_preserves_all_outputs_and_strict_public_validation(self):
        from supermimic.rotations import so3_exp
        one = _prepare_exact_meshes([self.mesh])[0]
        objects = [cube(.2), {**self.mesh, "vertices": self.mesh["vertices"] + torch.tensor([4.,2.,1.])}]
        prepared_objects = _prepare_exact_meshes(objects)
        for tick in range(8):
            rotation = so3_exp(torch.tensor([.17,-.11,.23], dtype=torch.float64) * tick)
            local = self.mesh["vertices"] @ rotation.T + torch.tensor([tick*.2,.1,-.03])
            explicit = {**self.mesh, "vertices": local}
            generic = exact_mesh_gap([explicit], objects, return_intersection=True)
            cached = _exact_mesh_gap([_rigid_mesh_vertices(one, local)], prepared_objects, return_intersection=True)
            torch.testing.assert_close(generic[0], cached[0], atol=0, rtol=0)
            torch.testing.assert_close(generic[1], cached[1], atol=0, rtol=0)
            self.assertEqual(generic[2], cached[2])
        with self.assertRaisesRegex(ValueError, "watertight"):
            exact_mesh_gap([{**self.mesh, "faces": self.mesh["faces"][:-1]}], objects)
        with self.assertRaisesRegex(ValueError, "unchanged topology"):
            _rigid_mesh_vertices(one, torch.zeros((7,3)))
        invalid = self.mesh["vertices"].clone()
        invalid[0,0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite vertices"):
            _rigid_mesh_vertices(one, invalid)

    def test_reference_contact_batch_validates_static_topology_once(self):
        frames = 12
        zeros, eye = torch.zeros((frames,3)), torch.eye(3)[None].expand(frames,-1,-1)
        reference = {"robot": {"root_pos": zeros, "root_rot": eye, "derived_ref_v": zeros,
                              "derived_ref_omega": zeros, "joint_pos": torch.zeros((frames,29)),
                              "derived_ref_joint_qd": torch.zeros((frames,29))},
                     "object": {"pos": zeros + torch.tensor([0,0,2.]), "rot": eye}}
        hands = {side: {"mount": ORIGIN[side], "collision_meshes": [cube(.01)]} for side in ("left","right")}
        with patch("hoi.v1.geometry.validate_mesh", wraps=validate_mesh) as validation:
            labels = _reference_contacts(reference, BundleKinematics(), hands, {"collision_meshes": [self.mesh]})
            self.assertEqual(validation.call_count, 3)
        self.assertEqual(labels["surface_gap_m"].shape, (frames,2))
        self.assertTrue(bool(labels["geometry_valid"].all()))

    def test_contact_quality_uses_interval_endpoints_and_reports_unknown_without_pass_claim(self):
        contacts = {"expected_contact": torch.tensor([[0,0],[1,0],[1,1],[0,1],[0,0],[0,0]]),
                    "expected_valid": torch.tensor([[0,0],[1,1],[1,1],[1,1],[0,1],[0,0]]),
                    "surface_gap_m": torch.tensor([[.01,.01],[.01,.05],[.01,.01],[.05,.01],[.03,.05],[.01,.01]]),
                    "vertex_penetration_m": torch.tensor([[.03,.03],[0,0],[0,0],[0,0],[0,0],[0,.03]]),
                    "surface_intersection": torch.tensor([[1,1],[0,0],[0,0],[0,0],[0,0],[1,0]], dtype=torch.bool),
                    "metadata": {"positive_gap_m":.02,"negative_gap_m":.04,"severe_vertex_penetration_m":.02}}
        quality = _reference_contact_quality(contacts)
        self.assertEqual(quality, {"interval_endpoints":5,"valid":[3,4],"positive":[2,2],"negative":[1,2],
                         "unknown_total":[2,1],"gray_unknown":[1,0],"surface_intersection":[1,0],
                         "severe_vertex_penetration":[0,1],"adjacent_valid_onset":[0,1],"adjacent_valid_release":[1,1]})
        clips = [{"asset_id":0,"reference_contact_quality":quality}, {"asset_id":1,"reference_contact_quality":quality}]
        aggregate = _aggregate_reference_contact_quality(clips,[{"asset_id":i,"object_asset":f"box{i}"} for i in range(2)])
        self.assertEqual(aggregate["whole_partition"]["valid"],[6,8])
        self.assertEqual(aggregate["by_asset"][1]["surface_intersection"],[1,0])
        self.assertNotIn("passed",aggregate)

    def test_expected_masks_include_ambiguous_and_severe_unknown(self):
        gap = torch.tensor([[.019,.025],[.04,.01]])
        e, m = expected_schedule_from_distances(gap, torch.tensor([[0,0],[0,.03]]))
        torch.testing.assert_close(e, torch.tensor([[1.,0],[0,0]]))
        torch.testing.assert_close(m, torch.tensor([[1.,0],[1,0]]))

    def test_shape_bank_context_and_mixed_asset_query_preserve_gradient(self):
        profile = {"mass": 2., "com": [.1,0,0], "inertia": [[1,0,0],[0,2,0],[0,0,3]], "friction": .5, "ground_friction": .3}
        bank = ShapeAssetBank([{"physical_profile": profile}, {"physical_profile": profile}], [self.cache, self.cache], {})
        ids = torch.tensor([1,0])
        p = torch.tensor([[[.53,0,0]], [[.54,0,0]]], requires_grad=True)
        d = bank.query(ids, p)
        gradient, = torch.autograd.grad(d.sum(), p)
        self.assertEqual(d.shape, (2,1))
        self.assertTrue(bool((gradient[..., 0] > .9).all()))
        self.assertEqual(bank.points(ids).shape, (2,128,3))
        context = bank.physical_context(ids)
        self.assertEqual(context.shape, (2,12))
        torch.testing.assert_close(context[0], torch.tensor([2.,.1,0,0,1,2,3,0,0,0,.5,.3]))


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sources"
        self.source.mkdir()
        partitions = {key: [] for key in ("train", "validation", "regression", "test")}
        for index, partition in enumerate(("train", "train", "validation", "regression", "test")):
            frames = torch.zeros((3, 630))
            frames[:, 6] = frames[:, 77] = 1
            frames[:, 201:357].reshape(3, 39, 4)[..., 3] = 1
            frames[:, 71] = 1
            frames[:, 0] = index * .01
            path = self.source / f"sub{index:02d}_box_{index:03d}_100_100_100.pt"
            torch.save(frames, path)
            partitions[partition].append({"motion": str(path), "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_group": f"sub{index:02d}_box_{index:03d}", "object": "box", "object_asset": f"box{index % 2}"})
        def dump(name, value):
            path = self.root / name
            path.write_text(json.dumps(value), encoding="utf-8")
            return path
        split = dump("split.json", {"partitions": partitions})
        evaluation = dump("evaluation.json", {"cases": [{**partitions["validation"][0], "id": "val", "source_frames": 3}]})
        test = dump("test.json", {"cases": [{**partitions["test"][0], "id": "test", "source_frames": 3}]})
        bindings = dump("bindings.json", {"source_joint_names": robot_joint_names(), "objects": {}})
        self.manifest = self.root / "manifest.json"
        write_manifest(build_manifest(split, evaluation, test, bindings_path=bindings), self.manifest)
        native = self.root / "bound_only.usda"
        native.write_text("synthetic test USD binding; never interpreted by native runtime", encoding="utf-8")
        native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
        hand = cube(.01)
        serial_hand = {"vertices": hand["vertices"].tolist(), "faces": hand["faces"].tolist()}
        asset = cube(.1)
        serial_asset = {"vertices": asset["vertices"].tolist(), "faces": asset["faces"].tolist()}
        profile = {"mass": 1, "com": [0,0,0], "inertia": [[1,0,0],[0,1,0],[0,0,1]], "friction": .5}
        recipe = {"hands": {side: {"collision_meshes": [serial_hand], "palm_points": [[.01,0,0]]} for side in ("left", "right")},
                  "assets": [{"object_asset": f"box{index}", "exact_scale": [1,1,1], "mesh_to_object": torch.eye(4).tolist(),
                              "native_asset": str(native), "native_asset_sha256": native_sha,
                              "physical_profile": profile, "collision_meshes": [serial_asset],
                              "distance_recipe": {"grid_resolution": 12, "candidate_count": 128, "max_grid_error": .05}}
                             for index in range(2)]}
        self.recipes = dump("recipes.json", recipe)
        self.kin = BundleKinematics()

    def test_full_bundle_roundtrip_asset_filter_and_hash_guards(self):
        output = self.root / "new_bundle"
        original = {path: path.read_bytes() for path in self.source.iterdir()}
        bundle = prepare_bundle(self.manifest, self.recipes, output, self.kin)
        self.assertEqual(bundle["scope"]["selected_motions"], 2)
        self.assertEqual(bundle["validation"]["native_physics"], "not_run")
        bank, shapes, loaded = load_bundle(output / "bundle.json", self.kin)
        self.assertEqual(bank.num_frames, 4)
        self.assertEqual(shapes.points(torch.tensor([1])).shape, (1,128,3))
        filtered, _, _ = load_bundle(output / "bundle.json", self.kin, asset_filter=[1])
        torch.testing.assert_close(filtered.global_frame_index, torch.tensor([2,3]))
        self.assertTrue(bool((filtered.asset_ids == 1).all()))
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaises(FileExistsError):
            prepare_bundle(self.manifest, self.recipes, output, self.kin)
        contact = Path(bundle["clips"][0]["contact_cache"]["path"])
        contact.write_bytes(contact.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "cache SHA256"):
            load_bundle(output / "bundle.json", self.kin)

    def test_changed_source_is_rejected_before_reference_is_trusted(self):
        output = self.root / "source_guard_bundle"
        bundle = prepare_bundle(self.manifest, self.recipes, output, self.kin)
        source = Path(bundle["clips"][0]["resolved_source_motion"])
        source.write_bytes(source.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "source motion bytes changed"):
            load_bundle(output / "bundle.json", self.kin)

    def test_resource_fixture_is_explicit_subset_bound_to_original_manifest(self):
        original_manifest = self.manifest.read_bytes()
        output = self.root / "resource_fixture"
        bundle = prepare_bundle(self.manifest, self.recipes, output, self.kin, one_per_asset_resource_fixture=True)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)
        self.assertTrue(bundle["scope"]["is_subset"])
        self.assertEqual(bundle["scope"]["purpose"], "one_per_asset_resource_fixture")
        self.assertEqual([c["asset_id"] for c in bundle["clips"]], [0,1])
        self.assertEqual(bundle["manifest"]["sha256"], hashlib.sha256(original_manifest).hexdigest())
        bank, _, _ = load_bundle(output / "bundle.json", self.kin)
        self.assertEqual(bank.num_frames, 4)
        for kwargs in ({"limit": 1}, {"partition": "eval80"}):
            with self.assertRaisesRegex(ValueError, "resource fixture requires train"):
                prepare_bundle(self.manifest, self.recipes, self.root / "invalid_resource_fixture", self.kin,
                               one_per_asset_resource_fixture=True, **kwargs)

    def test_changed_collision_recipe_is_rejected_on_load(self):
        output = self.root / "recipe_guard_bundle"
        prepare_bundle(self.manifest, self.recipes, output, self.kin)
        self.recipes.write_bytes(self.recipes.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "collision recipe bytes changed"):
            load_bundle(output / "bundle.json", self.kin)

    def test_validation_bundle_cannot_supply_training(self):
        output = self.root / "validation_bundle"
        prepare_bundle(self.manifest, self.recipes, output, self.kin, partition="eval80")
        with self.assertRaisesRegex(ValueError, "only train partition"):
            load_bundle(output / "bundle.json", self.kin)
        bank, _, _ = load_bundle(output / "bundle.json", self.kin, require_train=False)
        self.assertEqual(bank.partition, "eval80")

    def test_parallel_preparation_keeps_frozen_order_and_calibrated_palm_evidence(self):
        recipe = json.loads(self.recipes.read_text(encoding="utf-8"))
        for side in ("left", "right"):
            recipe["hands"][side]["palm_selection_verified"] = True
            recipe["hands"][side]["palm_selection_evidence"] = {"scope": "synthetic_cpu_fixture_only", "selected_face_indices": [6]}
        self.recipes.write_text(json.dumps(recipe), encoding="utf-8")
        output = self.root / "parallel_bundle"
        bundle = prepare_bundle(self.manifest, self.recipes, output, self.kin, jobs=2)
        self.assertEqual([c["asset_id"] for c in bundle["clips"]], [0,1])
        self.assertEqual(bundle["preparation"]["cpu_processes"], 2)
        bank, shapes, _ = load_bundle(output / "bundle.json", self.kin)
        self.assertEqual(bank.num_frames, 4)
        self.assertTrue(shapes.hands["left"]["palm_selection_verified"])
        self.assertEqual(shapes.hands["right"]["palm_selection_evidence"]["selected_face_indices"], [6])

    def test_identity_only_stage1_object_binding_uses_canonical_asset_identity(self):
        bindings_path = self.root / "bindings.json"
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        bindings["objects"] = {f"box{i}": {"asset_id": f"box{i}", "status": "identity_only_scale_and_mesh_transform_unverified"} for i in range(2)}
        bindings_path.write_text(json.dumps(bindings), encoding="utf-8")
        manifest_path = self.root / "identity_only_manifest.json"
        write_manifest(build_manifest(self.root / "split.json", self.root / "evaluation.json", self.root / "test.json",
                                      bindings_path=bindings_path), manifest_path)
        output = self.root / "identity_bound_bundle"
        bundle = prepare_bundle(manifest_path, self.recipes, output, self.kin)
        self.assertEqual([entry["asset_id"] for entry in bundle["clips"]], [0,1])
        self.assertEqual(bundle["validation"]["native_physics"], "not_run")

    def test_palm_evidence_chain_rejects_changed_probe_and_qa_artifacts(self):
        # This isolates immutable evidence validation. Reproducing the physical
        # STL ray mapping is a separate canonical preparation/native check.
        mesh = cube(.01)
        points = torch.tensor([[.01,0,0]] * 16)
        code, qa = self.root / "probe_definition.py", self.root / "qa.png"
        code.write_bytes(b"synthetic fixture definition")
        qa.write_bytes(b"synthetic fixture QA artifact")
        encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        record = lambda path: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        evidence = {"schema": "hoi_v1_palm_patch_evidence_v1", "source_stl_sha256": HAND_SHA["left"],
            "points_sha256": tensor_sha256(points),
            "collision_meshes_sha256": hashlib.sha256(encode([{"vertices": mesh["vertices"].tolist(), "faces": mesh["faces"].tolist()}])).hexdigest(),
            "probes_source": record(code), "visual_qa": record(qa),
            "definition": {"version": "central_inner_palm_raycast_v1", "palm_selection_verified": True,
                           "approach_axis": "-y", "source_triangle_indices": [6]*16, "compiled_triangle_indices": [6]*16}}
        evidence["evidence_sha256"] = hashlib.sha256(encode(evidence)).hexdigest()
        source = {"sha256": HAND_SHA["left"], "path": "source-byte-validation-is-tested-separately"}
        _validate_palm_evidence("left", source, [mesh], points, evidence, self.root)
        with self.assertRaisesRegex(ValueError, "points hash"):
            _validate_palm_evidence("left", source, [mesh], points + .001, evidence, self.root)
        qa.write_bytes(b"different QA artifact")
        with self.assertRaisesRegex(ValueError, "visual_qa bytes changed"):
            _validate_palm_evidence("left", source, [mesh], points, evidence, self.root)

    def test_merge_uses_only_master_hands_and_exact_train_asset_coverage(self):
        original = json.loads(self.recipes.read_text(encoding="utf-8"))
        master = copy.deepcopy(original)
        master["assets"] = original["assets"][1:]
        master["native_robot_asset"] = {"path": original["assets"][0]["native_asset"], "sha256": original["assets"][0]["native_asset_sha256"]}
        for side in ("left", "right"):
            master["hands"][side]["palm_selection_verified"] = True
            master["hands"][side]["palm_selection_evidence"] = {"scope": "synthetic_cpu_fixture_only"}
        other = copy.deepcopy(original)
        other["assets"] = original["assets"][:1]
        other["hands"]["left"]["palm_points"] = [[999,999,999]]
        master_path, other_path = self.root / "master.json", self.root / "other.json"
        master_path.write_text(json.dumps(master), encoding="utf-8")
        other_path.write_text(json.dumps(other), encoding="utf-8")
        output = self.root / "merged.json"
        result = merge_recipes(self.manifest, master_path, [other_path], output)
        self.assertEqual([a["object_asset"] for a in result["assets"]], ["box0","box1"])
        self.assertEqual(result["hands"], master["hands"])
        self.assertEqual(result["native_robot_asset"], master["native_robot_asset"])
        with self.assertRaises(FileExistsError):
            merge_recipes(self.manifest, master_path, [other_path], output)
        with self.assertRaisesRegex(ValueError, "missing"):
            merge_recipes(self.manifest, master_path, [], self.root / "incomplete.json")


if __name__ == "__main__":
    unittest.main()
