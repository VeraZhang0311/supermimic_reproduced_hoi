"""Check FPS cache provenance, metric geometry identity and serialization."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hoi_residual.assets import CATALOG_SCHEMA, EXPECTED_ASSETS, _digest_json, sha256_file
from hoi_residual.shape import CONTRACT, prepare_shape_bank, load_shape_bank, point_sha256


def signed_json(path, value, key):
    value = deepcopy(value)
    value.pop(key, None)
    value[key] = _digest_json(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(directory):
    # Synthetic tetrahedra and reproducible metric point tensors. Tests concern
    # preservation/binding of precomputed FPS data, not native surface physics.
    mesh = {"vertices": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], "frame": "object_local"}
    source_assets, points = [], []
    for i, name in enumerate(sorted(EXPECTED_ASSETS)):
        point = torch.arange(384, dtype=torch.float32).reshape(128, 3) * .001 + i
        points.append(point)
        metadata = {**{k: CONTRACT[k] for k in ("frame", "point_count", "sampling", "candidate_count", "seed")},
                    "point_hash": point_sha256(point), "collision_hash": str(i) * 64,
                    "exact_scale": [.8, .8, .8] if "080" in name else [1., 1., 1.]}
        cache = directory / f"geometry_{i}.pt"
        torch.save({"points": point, "collision_meshes": [deepcopy(mesh)], "metadata": metadata}, cache)
        source_assets.append({"asset_id": i, "object_asset": name, "collision_hash": metadata["collision_hash"],
                              "exact_scale": metadata["exact_scale"], "collision_meshes": [deepcopy(mesh)],
                              "physical_profile": {"mass": 1., "com": [0., 0., 0.], "inertia": torch.eye(3).tolist(),
                                                   "friction": .6, "ground_friction": .3},
                              "native_asset": "not_loaded.usda", "native_asset_sha256": "0" * 64,
                              "geometry_cache": {"path": cache.name, "sha256": sha256_file(cache)}})
    source, catalog = directory / "source.json", directory / "catalog.json"
    signed_json(source, {"schema_version": "supermimic_hoi_v1_bundle_v1", "assets": list(reversed(source_assets))},
                "bundle_content_sha256")
    targets = deepcopy(source_assets)
    for asset in targets:
        asset.pop("geometry_cache")
    signed_json(catalog, {"schema_version": CATALOG_SCHEMA, "assets": targets}, "catalog_content_sha256")
    return source, catalog, torch.stack(points)


class ShapeBankTests(unittest.TestCase):
    def test_exact_metric_points_survive_json_and_canonical_asset_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, catalog, expected = fixture(directory)
            bank = directory / "shape.json"
            metadata = prepare_shape_bank(source, catalog, bank)
            points, loaded = load_shape_bank(bank, catalog)
            self.assertTrue(torch.equal(points, expected))
            self.assertEqual(metadata, loaded)
            self.assertEqual(loaded["assets"], sorted(EXPECTED_ASSETS))
            self.assertEqual(loaded["bank_sha256"], sha256_file(bank))
            self.assertFalse(loaded["contract"]["scale_normalization"])
            with self.assertRaises(FileExistsError):
                prepare_shape_bank(source, catalog, bank)

    def test_changed_points_rejected_even_after_resigning_bank_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, catalog, _ = fixture(directory)
            bank = directory / "shape.json"
            prepare_shape_bank(source, catalog, bank)
            value = json.loads(bank.read_text(encoding="utf-8"))
            value["assets"][0]["points"][0][0] += .01
            signed_json(bank, value, "content_sha256")
            with self.assertRaisesRegex(ValueError, "point tensor hash"):
                load_shape_bank(bank, catalog)

    def test_asset_order_and_catalog_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, catalog, _ = fixture(directory)
            bank = directory / "shape.json"
            prepare_shape_bank(source, catalog, bank)
            value = json.loads(bank.read_text(encoding="utf-8"))
            value["assets"].reverse()
            signed_json(bank, value, "content_sha256")
            with self.assertRaisesRegex(ValueError, "canonical asset order"):
                load_shape_bank(bank, catalog)
            value["assets"].reverse()
            signed_json(bank, value, "content_sha256")
            content = json.loads(catalog.read_text(encoding="utf-8"))
            content["assets"][0]["physical_profile"]["mass"] = 1.1
            signed_json(catalog, content, "catalog_content_sha256")
            with self.assertRaisesRegex(ValueError, "catalog identity"):
                load_shape_bank(bank, catalog)

    def test_source_cache_frame_and_collision_geometry_are_checked(self):
        for mutation, message in (("frame", "metadata mismatch: frame"),
                                  ("mesh", "collision vertices"),
                                  ("hash", "cache SHA256")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                source, catalog, _ = fixture(directory)
                value = json.loads(source.read_text(encoding="utf-8"))
                binding = value["assets"][0]["geometry_cache"]
                path = directory / binding["path"]
                cache = torch.load(path, weights_only=True)
                if mutation == "frame":
                    cache["metadata"]["frame"] = "world"
                else:
                    cache["collision_meshes"][0]["vertices"][0][0] = .5
                torch.save(cache, path)
                if mutation != "hash":
                    binding["sha256"] = sha256_file(path)
                    signed_json(source, value, "bundle_content_sha256")
                with self.assertRaisesRegex(ValueError, message):
                    prepare_shape_bank(source, catalog, directory / "shape.json")


if __name__ == "__main__":
    unittest.main()
