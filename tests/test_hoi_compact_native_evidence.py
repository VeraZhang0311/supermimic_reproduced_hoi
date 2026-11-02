"""No rewritten native hashes, weakened parent gates, or fake collection shares."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hoi.data.compact_native_evidence import (SCHEMA, _binding,
    build_compact_native_evidence, validate_compact_native_evidence)
from tests.test_hoi_v1_gates import native_report
from hoi.v1.config import json_identity


class CompactNativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rtrain, reval = native_report(1), native_report(0)
        reval["bundle_content_sha256"] = "b" * 64
        assets = [{"asset_id": i, "native_equivalence_sha256": str(i) * 64,
                   "physical_profile": copy.deepcopy(rtrain["physics"]["bound_physical_profile"]),
                   "native_asset": "unchanged_{}.usda".format(i), "collision_hash": "collision_" + str(i),
                   "geometry_cache": {"sha256": str(i) * 64}}
                  for i in (0, 1)]
        self.parents = {
            "train": {"partition": "train", "assets": assets, "clips": [{"asset_id": 1}], "bundle_content_sha256": "a"*64},
            "protected_eval": {"partition": "protected_eval", "assets": assets, "clips": [{"asset_id": 0}], "bundle_content_sha256": "b"*64}}
        remapped = [{**copy.deepcopy(assets[old]), "asset_id": new, "original_asset_id": old}
                    for new, old in enumerate((1, 0))]
        self.target = {"partition": "train", "bundle_content_sha256": "c"*64, "assets": remapped,
                       "clips": [{"asset_id": 0}], "compact_selection": {"asset_id_map": [
                           {"asset_id": a["asset_id"], "original_asset_id": a["original_asset_id"],
                            "native_equivalence_sha256": a["native_equivalence_sha256"]} for a in remapped]}}
        self.target_path = self.write("bundle.json", self.target)
        self.parent_train_path = self.write("native_train.json", {"schema_version": "hoi_v1_native_evidence_manifest_v1", "reports": [rtrain]})
        self.parent_eval_path = self.write("native_eval.json", {"schema_version": "hoi_v1_native_evidence_manifest_v1", "reports": [reval]})
        self.collection = {"schema": "hoi_weighted_collection_probe_v1", "passed": True,
            "optimizer_steps": 0, "formal_training_started": False,
            "bundle": {**_binding(self.target_path)[0], "bundle_content_sha256": "c"*64},
            "asset_ids": [0], "environments": 8, "steps": 64,
            "sampling": {"schema": "hoi_actual_transition_sampling_v1", "allocation_match": True,
                "total_transitions": 512, "vector_steps": 64, "num_envs": 8, "completed_episodes": 3,
                "asset_counts": {"0": 512}, "category_counts": {"synthetic_class": 512},
                "stratum_counts": {"synthetic_stratum": 512}},
            "shutdown_passed": True, "collector_exit_codes": [0]}
        source_files = {"hoi/v1/native.py": "d" * 64, "hoi/v1/task_sampling.py": "e" * 64}
        self.collection["source_before"] = {"files": source_files, "content_sha256": json_identity(source_files)}
        self.collection["source_after"] = copy.deepcopy(self.collection["source_before"])
        self.collection["source_unchanged"] = True
        self.collection_path = self.write("collection.json", self.collection)
        self.wrapper = build_compact_native_evidence(self.target_path, self.parent_train_path,
            self.parent_eval_path, self.collection_path)
        # The fixture isolates lineage validation, tested by the dataset module.
        # Parent native acceptance is real and exercised with its full contract fixture.
        self.inputs = patch("hoi.data.compact_native_evidence._validated_inputs",
            side_effect=lambda _: (copy.deepcopy(self.target), copy.deepcopy(self.parents)))
        self.inputs.start(); self.addCleanup(self.inputs.stop)
        self.specs = patch("hoi.v1.packed_probe.specs_from_header", return_value=[SimpleNamespace(asset_id=0,num_envs=8)])
        self.spec_mock = self.specs.start(); self.addCleanup(self.specs.stop)
        self.plan = patch("hoi.v1.task_sampling.allocation_report", return_value={
            "category_env_counts": {"synthetic_class": 8}, "stratum_env_counts": {"synthetic_stratum": 8}})
        self.plan.start(); self.addCleanup(self.plan.stop)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def validate(self, **kwargs):
        return validate_compact_native_evidence(self.wrapper, **kwargs)

    def rebind_collection(self):
        self.write("collection.json", self.collection)
        self.wrapper["collection"] = _binding(self.collection_path)[0]

    def test_original_reports_unchanged_and_eval_only_asset_inherited(self):
        before = (self.parent_train_path.read_bytes(), self.parent_eval_path.read_bytes())
        result = self.validate(expected_bundle_sha256="c"*64, required_asset_ids=[0],
            expected_asset_profiles={a["asset_id"]: a["physical_profile"] for a in self.target["assets"]})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["asset_ids"], [0])
        self.assertEqual([(r["asset_id"],r["original_asset_id"],r["parent_partition"]) for r in result["inherited_registry"]],
                         [(0,1,"train"),(1,0,"protected_eval")])
        self.assertEqual(before, (self.parent_train_path.read_bytes(), self.parent_eval_path.read_bytes()))
        self.assertFalse(result["original_reports_modified"])
        self.assertEqual(result["actual_collection"]["source_content_sha256"],
                         self.collection["source_before"]["content_sha256"])

    def test_collection_source_identity_must_be_complete_consistent_and_unchanged(self):
        original = copy.deepcopy(self.collection)
        changes = (
            {"source_before": None},
            {"source_before": {"files": {}, "content_sha256": json_identity({})}},
            {"source_before": {"files": {"hoi/v1/native.py": "invalid"}, "content_sha256": "a" * 64}},
            {"source_before": {**original["source_before"], "content_sha256": "a" * 64}},
            {"source_after": {"files": {}, "content_sha256": json_identity({})}},
            {"source_unchanged": False},
            {"source_error": "failed to read final source"},
        )
        for change in changes:
            with self.subTest(change=change):
                self.collection = {**copy.deepcopy(original), **change}
                self.rebind_collection()
                result = self.validate()
                self.assertFalse(result["passed"])
                self.assertIn("source", result["blockers"][0])

    def test_bound_parent_file_tampering_rejected(self):
        self.parent_train_path.write_text("{}", encoding="utf-8")
        result = self.validate()
        self.assertFalse(result["passed"])
        self.assertIn("bytes changed", result["blockers"][0])

    def test_rebound_but_invalid_parent_probe_still_rejected(self):
        old = json.loads(self.parent_train_path.read_text())
        old["reports"][0]["pair_probes"]["passed"] = False
        self.write("native_train.json", old)
        self.wrapper["parent_evidence"]["train"] = _binding(self.parent_train_path)[0]
        result = self.validate()
        self.assertFalse(result["passed"])
        self.assertIn("original train native evidence failed", result["blockers"][0])

    def test_original_bundle_hash_not_replaced_by_compact_hash(self):
        old = json.loads(self.parent_train_path.read_text())
        old["reports"][0]["bundle_content_sha256"] = "c"*64
        self.write("native_train.json", old)
        self.wrapper["parent_evidence"]["train"] = _binding(self.parent_train_path)[0]
        self.assertFalse(self.validate()["passed"])

    def test_wrong_target_or_requested_coverage_rejected(self):
        self.assertFalse(self.validate(expected_bundle_sha256="d"*64)["passed"])
        self.assertFalse(self.validate(required_asset_ids=[0,1])["passed"])

    def test_changed_geometry_rejected_even_with_same_native_identity_hash(self):
        self.target["assets"][0]["collision_hash"] = "changed"
        self.write("bundle.json", self.target)
        self.wrapper["bundle"] = _binding(self.target_path)[0]
        result = self.validate()
        self.assertFalse(result["passed"])
        self.assertIn("geometry/profile/cache identity", result["blockers"][0])

    def test_missing_parent_eval_evidence_rejected(self):
        del self.wrapper["parent_evidence"]["protected_eval"]
        self.assertFalse(self.validate()["passed"])

    def test_no_native_collection_or_optimizer_activity_rejected(self):
        for field, value in (("passed",False),("optimizer_steps",1),("formal_training_started",True),("steps",63)):
            with self.subTest(field=field):
                original = self.collection[field]
                self.collection[field] = value; self.rebind_collection()
                self.assertFalse(self.validate()["passed"])
                self.collection[field] = original
        self.rebind_collection()

    def test_wrong_collection_bundle_rejected(self):
        self.collection["bundle"]["bundle_content_sha256"] = "a"*64
        self.rebind_collection()
        self.assertFalse(self.validate()["passed"])

    def test_self_reported_match_cannot_hide_wrong_counts(self):
        self.collection["sampling"]["category_counts"] = {"wrong_class":512}
        self.rebind_collection()
        result = self.validate()
        self.assertFalse(result["passed"])
        self.assertIn("independently recomputed", result["blockers"][0])

    def test_asset_counts_must_match_production_quota(self):
        self.spec_mock.return_value = [SimpleNamespace(asset_id=0,num_envs=7)]
        self.assertFalse(self.validate()["passed"])

    def test_missing_natural_reset_or_bad_shutdown_rejected(self):
        self.collection["sampling"]["completed_episodes"] = 0
        self.rebind_collection(); self.assertFalse(self.validate()["passed"])
        self.collection["sampling"]["completed_episodes"] = 3
        for changes in ({"shutdown_passed":False},{"collector_exit_codes":[-15]},{"collector_exit_codes":[]}):
            with self.subTest(changes=changes):
                changed = {k:self.collection[k] for k in changes}
                self.collection.update(changes); self.rebind_collection()
                self.assertFalse(self.validate()["passed"])
                self.collection.update(changed)
        self.rebind_collection()


if __name__ == "__main__":
    unittest.main()
