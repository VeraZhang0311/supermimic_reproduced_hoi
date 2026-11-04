from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from hoi.v1.canonical_hoi80 import map_canonical_cases, select_canonical_hoi80


def fixture():
    old, clips = [], []
    for obj in ("largebox", "smallbox", "plasticbox", "suitcase"):
        for number in range(20):
            name = f"{obj}_{number:03d}"
            old.append(dict(name=name, source_motion_sha256=name, source_frames=121, state_frames=101))
            clips.append(dict(dataset="ultra", clip_id=name, source_group=name, asset_id=number % 2,
                              source_frames=121, state_frames=101,
                              source_provenance=dict(protected_eval80=True, motion_sha256=name,
                                                     id=name, object=obj)))
    return ({"partition": "train", "clips": []},
            {"partition": "protected_eval", "clips": clips}, {"clips": old})


class CanonicalHOI80Tests(unittest.TestCase):
    def test_exact_membership_with_unrelated_protected_clips(self):
        train, evaluation, canonical = fixture()
        extra = deepcopy(evaluation["clips"][0]); extra["clip_id"] = "not_selected"
        evaluation["clips"].insert(0, extra)
        before = deepcopy(evaluation)
        ids, rows = map_canonical_cases(train, evaluation, canonical)
        self.assertEqual(ids, list(range(1, 81)))
        self.assertEqual(sum(r["state_frames"] - 1 for r in rows), 8000)
        self.assertEqual(evaluation, before)

    def test_wrong_bytes_duration_protection_or_missing_case_rejected(self):
        for change in ("bytes", "duration", "protected", "missing", "duplicate", "category"):
            with self.subTest(change=change):
                train, evaluation, canonical = fixture()
                c = evaluation["clips"][0]
                if change == "bytes": c["source_provenance"]["motion_sha256"] = "changed"
                if change == "duration": c["state_frames"] -= 1
                if change == "protected": c["source_provenance"]["protected_eval80"] = False
                if change == "missing": evaluation["clips"].pop()
                if change == "duplicate": evaluation["clips"].append(deepcopy(c))
                if change == "category": c["source_provenance"]["object"] = "other"
                with self.assertRaises(ValueError): map_canonical_cases(train, evaluation, canonical)

    def test_source_group_leakage_rejected(self):
        train, evaluation, canonical = fixture()
        c = deepcopy(evaluation["clips"][0]); c["clip_id"] = "another_augmentation"
        train["clips"].append(c)
        with self.assertRaisesRegex(ValueError, "leaked"):
            map_canonical_cases(train, evaluation, canonical)

    def test_arbitrary_eighty_cannot_claim_canonical_registry(self):
        train, evaluation, canonical = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps(canonical), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                select_canonical_hoi80(train, evaluation, path)


if __name__ == "__main__":
    unittest.main()
