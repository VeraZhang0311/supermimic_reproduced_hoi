"""Fast historical intent reuse must preserve targets and their source identity."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hoi_residual.assets import _digest_json
from hoi_residual.data import ASSETS, build_manifest, sha256_file
from hoi_residual.v11_reward import LABEL_SCHEMA
from hoi_tables import ultra, v11_reference_contacts as reused
from supermimic.robot_metadata import G1_JOINT_NAMES


def write_bundle(path, value):
    value = deepcopy(value); value.pop("bundle_content_sha256", None)
    value["bundle_content_sha256"] = _digest_json(value)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReusedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        partitions = {}
        self.assets = []
        for aid, name in enumerate(ASSETS):
            scale = .8 if "_080" in name else 1.
            self.assets.append({"asset_id": aid, "object_asset": name, "exact_scale": [scale] * 3,
                                "collision_hash": name + "-collision", "geometry_cache": {"sha256": name + "-current-sdf"}})
        for pindex, partition in enumerate(("train", "validation", "regression", "test")):
            partitions[partition] = []
            for aid, name in enumerate(ASSETS):
                group = f"sub{pindex+1}_{name.split('_')[0]}_{aid:03d}"
                partitions[partition].append({"motion": f"/immutable/{group}_100.pt", "source_group": group,
                    "motion_sha256": f"{pindex*8+aid+1:064x}", "object": name.split('_')[0], "object_asset": name})
        bindings = {"source_joint_names": list(G1_JOINT_NAMES),
                    "joint_order_evidence": {"fixture_sha256": "fixture", "field": "names"}}
        self.full = build_manifest(json.dumps({"partitions": partitions}).encode(), json.dumps(bindings).encode(), profile="fixture")
        self.manifest = self.root / "manifest.json"; self.manifest.write_text(json.dumps(self.full))
        self.canonical = self.root / "canonical.json"; self.canonical.write_text("fixture-development80")
        protected = [e for e in self.full["clips"] if e["partition"] == "validation"]
        self.binding = {"entries": protected, "cases": [{"state_frames": 3} for _ in protected],
                        "canonical_manifest_sha256": sha256_file(self.canonical)}
        self.inventory = ultra._inventory(self.full, self.binding)
        self.train, self.eval = self.root / "train.json", self.root / "eval.json"
        self.source_pairs = {}
        for path, partition in ((self.train, "train"), (self.eval, "validation")):
            clips = []
            for aid, entry in enumerate(e for e in self.full["clips"] if e["partition"] == partition):
                e = torch.tensor([[aid % 2, 1], [1, 0], [0, 1]], dtype=torch.int8)
                v = torch.tensor([[1, 1], [0, 1], [1, 1]], dtype=torch.float32)
                cache = self.root / f"{partition}_{aid}.pt"
                torch.save({"expected_contact": e, "expected_valid": v,
                            "metadata": {"label_schema": LABEL_SCHEMA}}, cache)
                self.source_pairs[entry["motion_sha256"]] = (e.clone(), v.clone())
                clips.append({"asset_id": aid, "source_motion_sha256": entry["motion_sha256"],
                              "source_group": entry["source_group"], "source_frames": 4, "state_frames": 3,
                              "contact_cache": {"path": cache.name, "sha256": sha256_file(cache)}})
            bundle = {"schema_version": "supermimic_hoi_v1_bundle_v1", "partition": "train" if partition == "train" else "eval80",
                      "assets": deepcopy(self.assets), "clips": clips,
                      "timebase": {"source_fps_assumed": 60., "control_dt_s": .02, "timebase_mode": "canonical_60hz"},
                      "reference_kinematics": {"urdf": {"path": "/old/legacy39.urdf", "sha256": "old39"}},
                      "robot_fk": {"origin": "old_v11_generation_robot"},
                      "hands": {"geometry_cache": {"path": "/old/hands.pt", "sha256": "old-calibration"}}}
            write_bundle(path, bundle)
        self.output = self.root / "reused.pt"
        self.hands = {"path": "/current/calibration.pt", "sha256": "current-calibration"}
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(reused, "_inventory", return_value=(deepcopy(self.full), deepcopy(self.binding), deepcopy(self.inventory))))

    def prepare(self, evals=None):
        return reused.prepare_reused_cache(self.manifest, self.train, evals or [self.eval], self.canonical, self.output)

    def motion(self, partition="train", order=(2, 0)):
        entries = [deepcopy(e) for e in self.inventory.values() if e["partition"] == partition]
        entries = [entries[i] for i in order]
        for e in entries: e["state_frames"] = 3
        return SimpleNamespace(entries=entries, num_frames=3*len(entries), device="cpu",
            source_manifest_path=self.manifest, canonical_manifest_path=self.canonical,
            ultra_manifest_sha256=sha256_file(self.manifest))

    def rewrite_source(self, path, mutate):
        bundle = json.loads(path.read_text()); mutate(bundle); write_bundle(path, bundle)

    def rewrite_pack(self, mutate):
        value = torch.load(self.output, weights_only=True)
        mutate(value); value["content_sha256"] = _digest_json(reused._content(value)); torch.save(value, self.output)

    def test_pack_preserves_old_values_dtypes_and_unknown_positive_without_geometry_work(self):
        with patch.object(ultra, "current_hands", side_effect=AssertionError("must not project hands")), \
                patch.object(ultra, "_record", side_effect=AssertionError("must not recompute labels")), \
                patch("supermimic.kinematics.G1Kinematics", side_effect=AssertionError("must not construct FK")):
            report = self.prepare()
        self.assertEqual(report["clip_count"], 16)
        self.assertEqual(report["state_frames"], 48)
        self.assertEqual(report["partitions"], {"train": 8, "protected_eval": 8})
        self.assertEqual(report["label_origin"], "v11_cached_reference_intent")
        self.assertFalse(report["contract"]["current_hw3_geometry_recomputed"])
        self.assertNotIn("target_robot", report["identity"])
        value = torch.load(self.output, weights_only=True)
        for record in value["clips"].values():
            expected, valid = self.source_pairs[record["source_trajectory_sha256"]]
            self.assertEqual(record["expected"].dtype, expected.dtype)
            self.assertEqual(record["valid"].dtype, valid.dtype)
            self.assertTrue(torch.equal(record["expected"], expected))
            self.assertTrue(torch.equal(record["valid"], valid))
            self.assertEqual(record["expected"][1, 0], 1)
            self.assertEqual(record["valid"][1, 0], 0)
        self.assertEqual(report["source_bundles"][0]["source_generation_robot_fk"]["robot_fk"]["origin"], "old_v11_generation_robot")

    def test_runtime_remaps_clip_order_and_has_same_full_metadata_for_train_and_hoi80(self):
        self.prepare()
        e, v, metadata = reused.load_reused_cache(self.output, self.motion(), self.assets, self.hands)
        expected = torch.cat([self.source_pairs[x["source_trajectory_sha256"]][0] for x in self.motion().entries]).float()
        self.assertTrue(torch.equal(e, expected)); self.assertEqual(v.dtype, torch.bool)
        self.assertEqual(metadata["file_sha256"], sha256_file(self.output))
        _, _, eval_metadata = reused.load_reused_cache(self.output, self.motion("protected_eval", (6,)), self.assets, self.hands)
        metadata.pop("selected_clips"); eval_metadata.pop("selected_clips")
        self.assertEqual(metadata, eval_metadata)
        self.assertFalse(metadata["current_reward_geometry_binding"]["used_to_regenerate_reference_labels"])

    def test_minimal_ultra_route_installs_targets_without_changing_reward_geometry(self):
        self.prepare(); motion = self.motion()
        current = SimpleNamespace(contract={"hand_geometry": "current-palm-evidence"}, hands={}, hand_evidence={})
        with patch.object(ultra, "TablesReward", return_value=current), patch.object(ultra, "_check_hands"):
            reward = ultra.make_ultra_reward(SimpleNamespace(motion=motion), self.assets, self.hands, self.output)
        self.assertIs(reward, current)
        self.assertEqual(reward.contract["hand_geometry"], "current-palm-evidence")
        self.assertEqual(reward.contract["frozen_reference_contacts"]["label_origin"], reused.LABEL_ORIGIN)
        self.assertEqual(reward.contract["frozen_reference_contacts"]["file_sha256"], sha256_file(self.output))
        self.assertEqual(tuple(reward._frozen_contacts[0].shape), (6, 2))

    def test_multiple_eval_sources_allow_identical_values_and_reject_conflicts(self):
        duplicate = self.root / "duplicate.json"; duplicate.write_bytes(self.eval.read_bytes())
        self.prepare([self.eval, duplicate])
        value = torch.load(self.output, weights_only=True)
        self.assertTrue(all(len(r["source_bindings"]) == (1 if r["partition"] == "train" else 2) for r in value["clips"].values()))
        self.output.unlink()
        data = json.loads(duplicate.read_text()); clip = data["clips"][0]
        changed = self.root / "conflicting.pt"
        torch.save({"expected_contact": torch.zeros(3, 2, dtype=torch.int8), "expected_valid": torch.ones(3, 2),
                    "metadata": {"label_schema": LABEL_SCHEMA}}, changed)
        clip["contact_cache"] = {"path": changed.name, "sha256": sha256_file(changed)}; write_bundle(duplicate, data)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.prepare([self.eval, duplicate])

    def test_incomplete_hoi80_cannot_fall_back_to_current_geometry(self):
        self.rewrite_source(self.eval, lambda b: b["clips"].pop())
        with self.assertRaisesRegex(ValueError, "coverage incomplete"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_group_asset_and_source_grid_are_rejected(self):
        original = self.eval.read_bytes()
        for mutate, text in ((lambda b: b["clips"][0].update(source_group="wrong"), "SHA/group"),
                             (lambda b: b["clips"][0].update(asset_id=7), "exact object asset"),
                             (lambda b: b["clips"][0].update(state_frames=4), "Tplus1")):
            self.eval.write_bytes(original); self.rewrite_source(self.eval, mutate)
            with self.assertRaisesRegex(ValueError, text): self.prepare()

    def test_source_cache_hash_and_binary_label_schema_are_verified(self):
        bundle = json.loads(self.eval.read_text()); clip = bundle["clips"][0]
        cache = self.root / clip["contact_cache"]["path"]
        cache.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"): self.prepare()
        torch.save({"expected_contact": torch.full((3, 2), .5), "expected_valid": torch.ones(3, 2),
                    "metadata": {"label_schema": LABEL_SCHEMA}}, cache)
        clip["contact_cache"]["sha256"] = sha256_file(cache); write_bundle(self.eval, bundle)
        with self.assertRaisesRegex(ValueError, "finite binary"): self.prepare()

    def test_load_rejects_tensor_source_and_complete_inventory_drift(self):
        self.prepare(); original = self.output.read_bytes()
        mutations = [(lambda x: x["clips"].pop(next(iter(x["clips"]))), "complete train2651"),
                     (lambda x: next(iter(x["clips"].values()))["expected"].zero_(), "tensor values"),
                     (lambda x: next(iter(x["clips"].values()))["source_bindings"][0].update(bundle_sha256="unknown"), "provenance")]
        for mutate, text in mutations:
            self.output.write_bytes(original); self.rewrite_pack(mutate)
            with self.assertRaisesRegex(ValueError, text):
                reused.load_reused_cache(self.output, self.motion(), self.assets, self.hands)

    def test_no_overwrite_and_wrong_timebase_fail_before_publishing(self):
        self.output.write_bytes(b"keep")
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertEqual(self.output.read_bytes(), b"keep"); self.output.unlink()
        self.rewrite_source(self.train, lambda b: b["timebase"].update(source_fps_assumed=30))
        with self.assertRaisesRegex(ValueError, "canonical60"): self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
