"""ULTRA profile boundaries: exact objects/timebase and frozen contact labels."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi_residual.assets import _digest_json
from hoi_residual.data import ASSETS, TIMEBASE, build_manifest, sha256_file
from hoi_residual.shape import CONTRACT as SHAPE_CONTRACT, point_sha256
from hoi_tables import ultra
from supermimic.robot_metadata import G1_JOINT_NAMES


def hand_evidence():
    return {side: {"colliders": [{"path": f"/current/{side}_{i}.stl", "sha256": f"{side}-{i}",
                                   "origin_xyz": [0., 0., 0.], "origin_rpy": [0., 0., 0.]} for i in range(3)],
                    "probe_count": 128, "probe_sha256": hashlib.sha256(torch.zeros(128, 3).numpy().tobytes()).hexdigest(),
                    "probe_frame": "current_wrist_link_metres",
                    "calibrated_to_current_projection_max_m": .00001} for side in ("left", "right")}


def hand_geometry():
    mesh = {"vertices": torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64),
            "faces": torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])}
    return {side: {"palm_points": torch.zeros(128, 3), "collision_meshes": [deepcopy(mesh) for _ in range(3)]}
            for side in ("left", "right")}


def fixture(root):
    vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    mesh = {"vertices": vertices, "faces": faces, "frame": "object_local"}
    assets, shape_entries = [], []
    points = torch.arange(384, dtype=torch.float32).reshape(128, 3) / 1000
    for index, name in enumerate(reversed(ASSETS)):
        native = root / (name + ".usda"); native.write_bytes(name.encode())
        scale = .8 if "_080" in name else 1.
        asset = {"asset_id": index, "object_asset": name, "exact_scale": [scale] * 3,
                 "collision_hash": name + "-collision", "collision_meshes": [deepcopy(mesh)],
                 "native_asset": str(native), "native_asset_sha256": sha256_file(native),
                 "physical_profile": {"mass": 1., "com": [0., 0., 0.], "inertia": torch.eye(3).tolist(),
                                      "friction": .6, "ground_friction": .3}}
        cache = {"collision_meshes": [deepcopy(mesh)], "sdf_grid": torch.zeros(2, 2, 2),
                 "grid_min": torch.full((3,), -1.), "grid_max": torch.ones(3),
                 "bbox_diagonal": torch.tensor(3.464), "points": points.clone(),
                 "metadata": {"collision_hash": asset["collision_hash"], "exact_scale": asset["exact_scale"],
                              "frame": "object_link_local_metres", "point_count": 128, "point_hash": point_sha256(points),
                              "grid_resolution": 2, "grid_axis_order": "zyx", "sdf_sign": "negative_inside_collider_union"}}
        path = root / (name + ".pt"); torch.save(cache, path)
        asset["geometry_cache"] = {"path": path.name, "sha256": sha256_file(path)}
        assets.append(asset)
        shape_entries.append({"object_asset": name, "collision_hash": asset["collision_hash"],
                              "exact_scale": asset["exact_scale"], "point_sha256": point_sha256(points),
                              "points": points.tolist(), "source_cache": deepcopy(asset["geometry_cache"])})
    hands = root / "hands.pt"; torch.save({"not_old_labels": True}, hands)
    bundle = {"schema_version": "supermimic_hoi_v1_bundle_v1", "assets": deepcopy(assets),
              "hands": {"geometry_cache": {"path": hands.name, "sha256": sha256_file(hands)}},
              "clips": [{"old_contact_labels_must_never_be_loaded": "missing/old.pt"}]}
    bundle["bundle_content_sha256"] = _digest_json(bundle)
    bundle_path = root / "geometry.json"; bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    catalog = {"schema_version": "hoi_residual_ultra_object_catalog_v1",
               "assets": [{k: v for k, v in asset.items() if k != "geometry_cache"} for asset in assets]}
    catalog["catalog_content_sha256"] = _digest_json(catalog)
    catalog_path = root / "assets.json"; catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    shape = {"schema": "hoi_residual_fps128_shape_bank_v1", "contract": deepcopy(SHAPE_CONTRACT),
             "catalog_sha256": sha256_file(catalog_path), "source_bundle": str(bundle_path),
             "source_bundle_sha256": sha256_file(bundle_path),
             "assets": sorted(shape_entries, key=lambda e: e["object_asset"])}
    shape["content_sha256"] = _digest_json(shape)
    shape_path = root / "shape.json"; shape_path.write_text(json.dumps(shape), encoding="utf-8")
    partitions = {}
    for part_index, partition in enumerate(("train", "validation", "regression", "test")):
        partitions[partition] = []
        for asset_index, name in enumerate(ASSETS):
            frames = torch.zeros(4, 630)
            frames[:, 6], frames[:, 77] = 1., 1.
            frames[:, 201:357].reshape(4, 39, 4)[:, :, 3] = 1.
            frames[:, 0] = torch.arange(4) * .1 + part_index + asset_index / 100
            frames[:, 71] = frames[:, 0] + 1.
            kind, scale, *_ = name.split("_")
            group = f"sub{part_index+1}_{kind}_{asset_index:03d}"
            path = root / (group + f"_{scale}_{scale}_{scale}.pt"); torch.save(frames, path)
            partitions[partition].append({"motion": str(path), "motion_sha256": sha256_file(path),
                                          "source_group": group, "object": kind, "object_asset": name})
    bindings = {"source_joint_names": list(G1_JOINT_NAMES), "joint_order_evidence": {"fixture_sha256": "fixture", "field": "names"}}
    manifest = build_manifest(json.dumps({"partitions": partitions}).encode(), json.dumps(bindings).encode(), profile="fixture")
    manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    canonical = root / "canonical.json"; canonical.write_text("fixture development80", encoding="utf-8")
    entries = [deepcopy(e) for e in manifest["clips"] if e["partition"] == "validation"]
    cases = [{"state_frames": 3, "control_steps": 2, "object_asset": e["object_asset"], "case_id": f"case{i}"}
             for i, e in enumerate(entries)]
    binding = {"full_manifest": manifest, "entries": entries, "cases": cases,
               "canonical_manifest_sha256": sha256_file(canonical), "ultra_manifest_sha256": sha256_file(manifest_path)}
    return manifest_path, catalog_path, bundle_path, shape_path, canonical, binding


class ReferenceOnlyReward:
    def __init__(self, mdp, assets, hands):
        self.hand_evidence = hand_evidence()
        self.hands = hand_geometry()
        self._reference_contacts = {}
        self._frozen_contacts = None
        self.contract = {"unchanged_formula": True}

    def _expected(self, frames):
        e = torch.stack(((frames % 2 == 0).float(), (frames % 3 == 1).float()), -1)
        v = torch.ones_like(e); v[frames % 3 == 2, 0] = 0.
        e[v == 0] = 0.
        return e, v


class UltraTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.catalog, self.geometry, self.shape, self.canonical, self.binding = fixture(self.root)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(ultra, "current_hands", return_value=(hand_geometry(), hand_evidence())))

    def motion(self, **kwargs):
        return ultra.UltraTeacherMotion(self.manifest, None, "cpu", canonical=self.canonical, **kwargs)

    def test_sorted_asset_rebinding_retains_exact_scaled_hulls_and_physics(self):
        assets, hands = ultra.load_ultra_assets(self.catalog, self.geometry)
        self.assertEqual([a["object_asset"] for a in assets], list(ASSETS))
        self.assertEqual([a["asset_id"] for a in assets], list(range(8)))
        self.assertEqual(assets[0]["exact_scale"], [.8] * 3)
        self.assertEqual(assets[0]["physical_profile"]["friction"], .6)
        self.assertTrue(all(a["teacher_binding"]["additional_runtime_scale"] == 1 for a in assets))
        self.assertEqual(Path(hands["path"]), self.root / "hands.pt")
        points, metadata = ultra.load_ultra_shape(self.shape, self.catalog)
        self.assertEqual(tuple(points.shape), (8, 128, 3))
        self.assertEqual(metadata["catalog_sha256"], sha256_file(self.catalog))

    def test_geometry_and_hand_drift_are_rejected(self):
        bundle = json.loads(self.geometry.read_text()); bundle["assets"][0]["physical_profile"]["mass"] = 2.
        bundle.pop("bundle_content_sha256"); bundle["bundle_content_sha256"] = _digest_json(bundle)
        self.geometry.write_text(json.dumps(bundle))
        with self.assertRaisesRegex(ValueError, "geometry/profile"):
            ultra.load_ultra_assets(self.catalog, self.geometry)
        evidence = hand_evidence(); evidence["left"]["calibrated_to_current_projection_max_m"] = .008
        ultra._check_hands(evidence, hand_geometry())  # OLD calibration displacement is allowed.
        geometry = hand_geometry(); geometry["left"]["palm_points"][:] = 2.
        evidence["left"]["probe_sha256"] = hashlib.sha256(geometry["left"]["palm_points"].numpy().tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "current hull union"):
            ultra._check_hands(evidence, geometry)

    def test_shape_points_must_match_actual_bound_fps_cache(self):
        shape = json.loads(self.shape.read_text()); entry = shape["assets"][0]
        changed = torch.tensor(entry["points"]); changed[0, 0] += .01
        entry["points"], entry["point_sha256"] = changed.tolist(), point_sha256(changed)
        shape.pop("content_sha256"); shape["content_sha256"] = _digest_json(shape)
        self.shape.write_text(json.dumps(shape))
        with self.assertRaisesRegex(ValueError, "exact geometry-cache FPS128"):
            ultra.load_ultra_shape(self.shape, self.catalog)

    def test_ultra_retains_60hz_grid_and_does_not_mutate_manifest(self):
        motion = self.motion(asset=0)
        self.assertEqual(motion.clip_len.tolist(), [3])
        self.assertTrue(torch.allclose(motion.root_p[:, 0], torch.tensor([0., .12, .24])))
        self.assertEqual(motion.manifest["timebase"], TIMEBASE)
        self.assertEqual(motion.entries[0]["state_frames"], 3)
        self.assertTrue(motion.entries[0]["clip_id"].startswith("ultra/"))
        self.assertNotIn("clip_id", motion.manifest["clips"][0])
        self.assertTrue(motion.allowed_for_training)
        with self.assertRaisesRegex(ValueError, "Protected eval"):
            self.motion(partition="validation", for_training=True)

    def _cache_setup(self):
        self.stack.enter_context(patch.object(ultra, "canonical_binding", return_value=deepcopy(self.binding)))
        self.stack.enter_context(patch.object(ultra, "TablesReward", ReferenceOnlyReward))
        self.stack.enter_context(patch.object(ultra, "URDF_PATH", str(self.manifest)))
        self.stack.enter_context(patch.object(ultra, "MJCF_PATH", str(self.canonical)))

    def _write_cache(self):
        self._cache_setup()
        assets, hands = ultra.load_ultra_assets(self.catalog, self.geometry)
        train = self.motion(for_training=False)
        protected = self.motion(partition="validation", for_training=False)
        ultra._augment(protected, self.manifest, self.canonical, "protected_eval")
        records = {}
        for motion in (train, protected):
            reward = ReferenceOnlyReward(None, assets, hands)
            for index in range(len(motion.entries)):
                record = ultra._record(motion, reward, index, 2)
                records[record["clip_id"]] = record
        identity = ultra.contact_identity(train, assets, hands)
        value = {"schema": ultra.CONTACT_SCHEMA, "complete": True, "clips": records,
                 "metadata": {"identity": identity, "clip_count": 16, "state_frames": 48,
                              "partitions": {"train": 8, "protected_eval": 8}, "reference_only": True,
                              "optimizer_updates": 0, "normalizer_updates": 0, "rsi_updates": 0}}
        value["content_sha256"] = _digest_json(ultra._content(value))
        path = self.root / "contacts.pt"; torch.save(value, path)
        return path, assets, hands, value

    def test_frozen_labels_remap_by_source_id_across_asset_filter_and_order(self):
        path, assets, hands, value = self._write_cache()
        motion = self.motion(asset=2)
        e, v, metadata = ultra.load_contact_cache(path, motion, assets, hands)
        source = value["clips"][motion.entries[0]["clip_id"]]
        self.assertTrue(torch.equal(e, source["expected"]))
        self.assertTrue(torch.equal(v, source["valid"]))
        self.assertEqual(metadata["clip_count"], 16)
        reward = ultra.make_ultra_reward(SimpleNamespace(motion=motion), assets, hands, path)
        self.assertTrue(torch.equal(reward._frozen_contacts[0], e))
        self.assertTrue(reward.contract["unchanged_formula"])

    def test_cache_rejects_partition_tensor_and_incomplete_inventory_drift(self):
        path, assets, hands, value = self._write_cache()
        motion = self.motion(asset=0)
        def save(changed):
            changed["content_sha256"] = _digest_json(ultra._content(changed)); torch.save(changed, path)
        changed = deepcopy(value); next(iter(changed["clips"].values()))["partition"] = "protected_eval"; save(changed)
        with self.assertRaisesRegex(ValueError, "partition identity"):
            ultra.load_contact_cache(path, motion, assets, hands)
        changed = deepcopy(value); next(iter(changed["clips"].values()))["expected"][0, 0] = .5; save(changed)
        with self.assertRaisesRegex(ValueError, "binary"):
            ultra.load_contact_cache(path, motion, assets, hands)
        changed = deepcopy(value); changed["clips"].pop(next(iter(changed["clips"]))); save(changed)
        with self.assertRaisesRegex(ValueError, "all train2651"):
            ultra.load_contact_cache(path, motion, assets, hands)

    def test_new_cache_required_old_table_cache_or_robot_identity_rejected(self):
        path, assets, hands, value = self._write_cache()
        motion = self.motion(asset=0)
        value["schema"] = "hoi_tables_exact_reference_contact_cache_v1"
        value["content_sha256"] = _digest_json(ultra._content(value)); torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "schema"):
            ultra.load_contact_cache(path, motion, assets, hands)
        value["schema"] = ultra.CONTACT_SCHEMA
        value["metadata"]["identity"]["target_robot"]["urdf_sha256"] = "old robot"
        value["content_sha256"] = _digest_json(ultra._content(value)); torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "current robot"):
            ultra.load_contact_cache(path, motion, assets, hands)

    def test_preparation_refuses_existing_file_before_any_computation(self):
        path = self.root / "exists.pt"; path.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            ultra.prepare_contact_cache(self.manifest, self.catalog, self.geometry, self.shape, path)
        self.assertEqual(path.read_bytes(), b"keep")

    def test_canonical_factory_retains_legacy_scoring_fields_and_readonly_partition(self):
        def backend(manifest, canonical, kin, device, asset=None):
            motion = ultra.UltraMotionLib(manifest, kin, device, partition="validation", for_training=False, asset=asset)
            motion.cases = [self.binding["cases"][ASSETS.index(e["object_asset"])] for e in motion.entries]
            for case in motion.cases:
                case.update(reference_path="reference.npz", reference_sha256="reference")
            motion.legacy_body_names = ["legacy39"]
            motion.legacy_joint_names = list(G1_JOINT_NAMES)
            motion.legacy_urdf_path, motion.legacy_urdf_sha256 = "legacy.urdf", "urdf"
            return motion
        module = SimpleNamespace(Canonical80Motion=backend, _reference_arrays=lambda *a: ("reference-array", None))
        with patch.object(ultra, "canonical_binding", return_value=deepcopy(self.binding)), \
                patch.object(ultra, "_protocol_module", return_value=module):
            motion = ultra.HOI80TeacherMotion(self.manifest, None, "cpu", asset=3, canonical=self.canonical)
        self.assertEqual(motion.partition, "protected_eval")
        self.assertFalse(motion.for_training or motion.allowed_for_training)
        self.assertEqual(motion.entries[0]["source_partition"], "validation")
        self.assertEqual(motion.entries[0]["state_frames"], 3)
        self.assertEqual(len(motion.all_canonical_clip_ids), 8)
        self.assertEqual(motion.legacy_body_names, ["legacy39"])
        with patch.object(ultra, "_protocol_module", return_value=module):
            self.assertEqual(motion.load_reference(0), "reference-array")

    def test_preparation_roundtrip_uses_current_exact_algorithm_and_full_inventory(self):
        self._cache_setup()
        def protected_factory(manifest, kin, device, *, asset, canonical):
            motion = self.motion(partition="validation", for_training=False, asset=asset)
            ultra._augment(motion, self.manifest, self.canonical, "protected_eval")
            return motion
        output, progress = self.root / "new_contacts.pt", []
        with patch("supermimic.kinematics.G1Kinematics", return_value=None), \
                patch.object(ultra, "HOI80TeacherMotion", side_effect=protected_factory):
            report = ultra.prepare_contact_cache(self.manifest, self.catalog, self.geometry, self.shape, output,
                                                  canonical=self.canonical, chunk_size=2, progress=progress.append)
        self.assertEqual(report["clip_count"], 16)
        self.assertEqual(report["state_frames"], 48)
        self.assertEqual(report["partitions"], {"train": 8, "protected_eval": 8})
        self.assertEqual(progress[-1]["completed_clips"], 16)
        assets, hands = ultra.load_ultra_assets(self.catalog, self.geometry)
        expected, valid, metadata = ultra.load_contact_cache(output, self.motion(asset=0), assets, hands)
        self.assertEqual(tuple(expected.shape), (3, 2))
        self.assertTrue(torch.equal(expected[~valid], torch.zeros_like(expected[~valid])))
        self.assertFalse(metadata["identity"]["contract"]["old_robot_contact_labels_reused"])
        self.assertEqual(metadata["optimizer_updates"], 0)

    def test_cache_rejects_declared_training_updates(self):
        path, assets, hands, value = self._write_cache()
        value["metadata"]["normalizer_updates"] = 1
        value["content_sha256"] = _digest_json(ultra._content(value)); torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "zero training updates"):
            ultra.load_contact_cache(path, self.motion(asset=0), assets, hands)


if __name__ == "__main__":
    unittest.main()
