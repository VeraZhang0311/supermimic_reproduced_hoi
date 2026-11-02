"""Independent sitting summaries must use complete frozen membership, not names."""
import copy
import unittest

from hoi.v1.evaluate import aggregate_evaluation_cases, grail_sitting_evaluation
from hoi.v1.track import _evaluation_summary, _validate_payload, build_payload


class GrailSittingEvaluationTests(unittest.TestCase):
    def setUp(self):
        assets = [{"asset_id": i, "original_asset_id": 100 + i,
                   "native_equivalence_sha256": str(i) * 64} for i in range(4)]
        self.bundle = {"partition": "protected_eval", "bundle_content_sha256": "e" * 64,
            "compact_selection": {"split_content_sha256": "a" * 64}, "assets": assets,
            "clips": [{"clip_id": "source-" + str(i), "name": "clip-" + str(i),
                       "dataset": dataset, "asset_id": asset, "source_group": group}
                      for i, (dataset, asset, group) in enumerate([
                          ("grail_sitting", 1, "chair-a"), ("omnicontact", 0, "omni"),
                          ("grail_sitting", 2, "chair-b"), ("grail_sitting", 1, "chair-a")])]}
        self.training = {"partition": "train", "compact_selection": {"split_content_sha256": "a" * 64},
                         "assets": assets, "clips": [{"asset_id": 3, "source_group": "train-chair"}]}
        self.cases = []
        for i, clip in enumerate(self.bundle["clips"]):
            value = (0.2, 0.99, 0.4, 0.6)[i]
            self.cases.append({"clip_id": i, "id": clip["name"], "asset_id": clip["asset_id"],
                "phc_pass": i != 2, "object_pose_pass": True, "body_and_object_pose_pass": i != 2,
                "first_supermimic_failure_step": 3 if i == 2 else -1, "supermimic_norm_return": value,
                "human_normalized_return": value, "hoi_normalized_return": value / 2,
                "hoi_reference_mask_normalized_return": value * 2, "objective_normalized_return": value / 3})

    def result(self, cases=None):
        return grail_sitting_evaluation(self.bundle, self.training, self.cases if cases is None else cases)

    def test_complete_dynamic_membership_reuses_same_aggregation_and_excludes_other_datasets(self):
        result = self.result(list(reversed(self.cases)))
        expected = aggregate_evaluation_cases([self.cases[i] for i in (0, 2, 3)])
        self.assertEqual(result["aggregate"], expected)
        self.assertAlmostEqual(result["aggregate"]["human_normalized_return"], .4)
        self.assertEqual(result["manifest_clip_ids"], ["source-0", "source-2", "source-3"])
        self.assertEqual(result["chair_count"], 2)
        self.assertEqual(result["source_group_count"], 2)
        self.assertEqual(result["augmentation_count"], 0)
        self.assertTrue(result["source_group_disjoint"])
        self.assertTrue(result["native_asset_disjoint"])
        self.assertEqual(result["human_template_independence"], "not_established_by_published_metadata")
        self.assertEqual(result["per_chair"][0]["aggregate"], aggregate_evaluation_cases([self.cases[0], self.cases[3]]))
        self.assertEqual(result["per_chair"][1]["aggregate"], aggregate_evaluation_cases([self.cases[2]]))

    def test_missing_duplicate_unknown_or_bool_case_indices_rejected(self):
        for cases in (self.cases[:-1], self.cases + [self.cases[0]],
                      [*self.cases[:-1], {**self.cases[-1], "clip_id": 90}],
                      [{**self.cases[0], "clip_id": False}, *self.cases[1:]]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.result(cases)

    def test_wrong_asset_or_name_rejected_even_for_non_grail_case(self):
        for index in (0, 1):
            for field, value in (("asset_id", 77), ("id", "other-clip"), ("asset_id", True)):
                cases = copy.deepcopy(self.cases)
                cases[index][field] = value
                with self.subTest(index=index, field=field, value=value), self.assertRaisesRegex(ValueError, "identity"):
                    self.result(cases)

    def test_source_group_and_native_chair_leaks_are_rejected(self):
        self.training["clips"][0]["source_group"] = "chair-b"
        with self.assertRaisesRegex(ValueError, "source group overlaps"):
            self.result()
        self.training["clips"][0]["source_group"] = "train-chair"
        self.training["assets"] = copy.deepcopy(self.training["assets"])
        self.training["assets"][3]["native_equivalence_sha256"] = self.bundle["assets"][1]["native_equivalence_sha256"]
        with self.assertRaisesRegex(ValueError, "native chair overlaps"):
            self.result()

    def test_paired_split_required_and_legacy_reports_unchanged(self):
        self.training["compact_selection"]["split_content_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "paired compact"):
            self.result()
        self.bundle.pop("compact_selection")
        self.assertIsNone(self.result())

    def test_manifest_membership_not_case_self_report_controls_sitting_membership(self):
        for case in self.cases:
            case["dataset"] = "grail_sitting" if case["clip_id"] == 1 else "ultra"
        self.assertEqual(self.result()["aggregate"]["count"], 3)

    def test_tracking_only_uploads_declared_sitting_aggregate_scalars(self):
        sitting = self.result()
        sitting["raw_trajectory"] = "private-marker"
        sitting["aggregate"]["unknown_scalar"] = 42
        sitting["per_chair"][0]["raw"] = "private-marker"
        summary = _evaluation_summary({"grail_sitting_eval": sitting})
        self.assertEqual(summary["evaluation/grail_sitting/count"], 3)
        self.assertEqual(summary["evaluation/grail_sitting/chair_0001/count"], 2)
        self.assertEqual(summary["evaluation/grail_sitting/augmentation_count"], 0)
        self.assertNotIn("private-marker", str(summary))
        self.assertNotIn("source-0", str(summary))
        self.assertFalse(any("unknown_scalar" in k or "manifest_clip_ids" in k for k in summary))
        payload = build_payload(evaluation={"grail_sitting_eval": sitting})
        self.assertEqual(payload["summary"], summary)
        for key in ("evaluation/grail_sitting/raw_trajectory", "evaluation/grail_sitting/chair_0001/unknown_scalar"):
            payload["summary"][key] = 1
            with self.assertRaisesRegex(ValueError, "whitelist"):
                _validate_payload(payload)
            del payload["summary"][key]


if __name__ == "__main__":
    unittest.main()
