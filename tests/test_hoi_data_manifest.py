"""Manifest identity, split isolation and portable path-remapping boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hoi.data import manifest as manifest_module
from hoi.data.manifest import (
    build_manifest,
    content_sha256,
    membership_sha256,
    read_manifest,
    resolve_motion_path,
    write_manifest,
)


def motion(subject, group, variant="100_100_100", *, object_name="smallbox"):
    filename = f"{subject}_{object_name}_{group}_{variant}.pt"
    return {
        "motion": f"/mnt/source/v4/{filename}",
        "motion_sha256": hashlib.sha256(filename.encode()).hexdigest(),
        "source_group": f"{subject}_{object_name}_{group}",
        "object": object_name,
        "object_asset": f"{object_name}_100_100_100",
    }


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = {
            "seed": 20261013,
            "scope": "synthetic fixture; no claim of canonical-v4 identity",
            "group_rule": "first three motion filename fields; all variants stay together",
            "partitions": {
                "train": [motion("sub01", "000"), motion("sub02", "001")],
                "validation": [motion("sub03", "002"), motion("sub04", "003")],
                "regression": [motion("sub05", "004")],
                "test": [motion("sub06", "005")],
            },
        }
        self.eval = {"cases": [self.case("validation", 0, "smallbox_080")]}
        self.test = {"cases": [self.case("test", 0, "smallbox_test_000")]}
        self.split_path = self.root / "split.json"
        self.eval_path = self.root / "eval.json"
        self.test_path = self.root / "test.json"

    def case(self, partition, index, case_id):
        return {
            **copy.deepcopy(self.split["partitions"][partition][index]),
            "id": case_id,
            "source_frames": 61,
        }

    def write_inputs(self):
        for path, payload in (
            (self.split_path, self.split),
            (self.eval_path, self.eval),
            (self.test_path, self.test),
        ):
            path.write_text(json.dumps(payload), encoding="utf-8")

    def build(self, **kwargs):
        self.write_inputs()
        return build_manifest(self.split_path, self.eval_path, self.test_path, **kwargs)

    def selection(self, cases):
        path = self.root / "selection.json"
        path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
        return path

    def test_custom_manifest_preserves_source_identity_and_records_missing_selection(self):
        manifest = self.build()
        self.assertEqual(manifest["schema_version"], "supermimic_hoi_data_manifest_v1")
        self.assertEqual(manifest["profile"], "custom")
        self.assertEqual(set(manifest["partitions"]), set(self.split["partitions"]))
        for partition, cases in self.split["partitions"].items():
            for source, entry in zip(cases, manifest["partitions"][partition]):
                for key, value in source.items():
                    self.assertEqual(entry[key], value)
        self.assertEqual(manifest["panels"]["eval80"][0]["id"], "smallbox_080")
        self.assertEqual(manifest["validation"]["train_selection"], "not_provided")
        self.assertFalse(manifest["validation"]["adapter_ready"])

    def test_train_selection_must_match_every_train_identity(self):
        cases = [dict(case, partition="train") for case in self.split["partitions"]["train"]]
        manifest = self.build(train_selection_path=self.selection(cases))
        self.assertNotEqual(manifest["validation"]["train_selection"], "not_provided")
        for bad in (
            cases[:1],
            [dict(cases[0], partition="validation"), cases[1]],
            [cases[0], dict(self.split["partitions"]["regression"][0], partition="train")],
            [cases[0], dict(cases[1], motion_sha256="a" * 64)],
            [cases[0], cases[0]],
        ):
            with self.subTest(selection=bad):
                with self.assertRaises(ValueError):
                    self.build(train_selection_path=self.selection(bad))

    def test_augmented_variants_of_one_source_group_cannot_cross_partitions(self):
        self.split["partitions"]["validation"].append(motion("sub01", "000", "080_080_080"))
        with self.assertRaises(ValueError):
            self.build()

    def test_recorded_source_group_must_match_filename(self):
        self.split["partitions"]["train"][0]["source_group"] = "sub99_smallbox_999"
        with self.assertRaises(ValueError):
            self.build()

    def test_duplicate_path_hash_and_basename_are_rejected(self):
        original = copy.deepcopy(self.split)
        first = self.split["partitions"]["train"][0]
        alternatives = [
            copy.deepcopy(first),
            dict(motion("sub07", "006"), motion_sha256=first["motion_sha256"]),
            dict(first, motion=first["motion"].replace("/mnt/source/v4/", "/other/archive/"),
                 motion_sha256="e" * 64),
        ]
        for duplicate in alternatives:
            with self.subTest(duplicate=duplicate):
                self.split = copy.deepcopy(original)
                self.split["partitions"]["train"].append(duplicate)
                with self.assertRaises(ValueError):
                    self.build()

    def test_unknown_partition_is_rejected(self):
        self.split["partitions"]["extra_train"] = [motion("sub07", "006")]
        with self.assertRaises(ValueError):
            self.build()

    def test_malformed_motion_hashes_are_rejected(self):
        for digest in ("a" * 63, "g" * 64, "", None):
            with self.subTest(digest=digest):
                self.split["partitions"]["train"][0]["motion_sha256"] = digest
                with self.assertRaises(ValueError):
                    self.build()

    def test_eval_panel_rejects_wrong_partition_and_changed_identity_fields(self):
        correct = self.case("validation", 0, "smallbox_080")
        variants = [
            self.case("train", 0, "smallbox_080"),
            dict(correct, motion_sha256="b" * 64),
            dict(correct, source_group="sub99_smallbox_999"),
            dict(correct, object_asset="smallbox_080_080_080"),
            dict(correct, object="largebox"),
        ]
        for case in variants:
            with self.subTest(case=case):
                self.eval["cases"] = [case]
                with self.assertRaises(ValueError):
                    self.build()

    def test_test_panel_cannot_select_validation_motion(self):
        self.test["cases"] = [self.case("validation", 0, "smallbox_test_000")]
        with self.assertRaises(ValueError):
            self.build()

    def test_panel_ids_must_be_unique(self):
        self.eval["cases"].append(self.case("validation", 1, "smallbox_080"))
        with self.assertRaises(ValueError):
            self.build()

    def test_panel_source_frame_counts_must_be_positive_integers(self):
        for frames in (0, -1, True, "61", 61.5):
            with self.subTest(frames=frames):
                self.eval["cases"][0]["source_frames"] = frames
                with self.assertRaises(ValueError):
                    self.build()

    def test_custom_fixture_cannot_be_labeled_canonical_v4(self):
        with self.assertRaises(ValueError):
            self.build(profile="canonical-v4")

    def test_manifest_roundtrip_and_output_refuses_overwrite(self):
        manifest = self.build()
        output = self.root / "new-manifest.json"
        write_manifest(manifest, output)
        self.assertEqual(read_manifest(output), manifest)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_manifest(manifest, output)
        self.assertEqual(output.read_bytes(), before)

    def test_manifest_self_digest_detects_entry_tampering(self):
        output = self.root / "manifest.json"
        write_manifest(self.build(), output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["partitions"]["train"][0]["motion_sha256"] = "f" * 64
        output.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_manifest(output)

    def test_read_manifest_rejects_unknown_schema(self):
        output = self.root / "manifest.json"
        write_manifest(self.build(), output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["schema_version"] = "unrecognized"
        output.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            read_manifest(output)

    def test_reader_rechecks_custom_semantics_after_self_digest_is_recomputed(self):
        original = self.build()
        changes = (
            ("counts", "train", "motions", 999, "counts disagree"),
            ("timebase", "control_dt_s", None, .017, "unsupported timebase"),
            ("validation", "adapter_ready", None, True, "readiness"),
            ("validation", "training_ready", None, True, "readiness"),
            ("validation", "train_selection", None, "verified_exact_train_identities", "selection status"),
        )
        for index, (section, key, child, replacement, reason) in enumerate(changes):
            with self.subTest(section=section, key=key):
                forged = copy.deepcopy(original)
                if child is None:
                    forged[section][key] = replacement
                else:
                    forged[section][key][child] = replacement
                forged["manifest_content_sha256"] = content_sha256(forged)
                output = self.root / f"forged-custom-{index}.json"
                write_manifest(forged, output)
                with self.assertRaisesRegex(ValueError, reason):
                    read_manifest(output)

    def test_selection_explicit_boundary_flags_must_be_boolean_true(self):
        cases = [dict(case, partition="train") for case in self.split["partitions"]["train"]]
        selection = self.selection(cases)
        safe = {"cases": cases, "training_only": True, "validation_source_groups_disjoint": True}
        selection.write_text(json.dumps(safe), encoding="utf-8")
        self.build(train_selection_path=selection)
        for flag in ("training_only", "validation_source_groups_disjoint"):
            for rejected in (False, 0, "true"):
                with self.subTest(flag=flag, value=rejected):
                    selection.write_text(json.dumps({**safe, flag: rejected}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"{flag} must be true"):
                        self.build(train_selection_path=selection)

    def test_canonical_reader_enforces_fixed_input_and_membership_pins_beyond_self_hashes(self):
        # Keep real reader/panel/identity logic, including canonical 80/40 sizes.
        # Only the protocol constants use this self-contained synthetic corpus.
        self.split["partitions"]["validation"] = [motion(f"subV{i:03d}", f"{i:03d}") for i in range(80)]
        self.split["partitions"]["test"] = [motion(f"subT{i:03d}", f"{i:03d}") for i in range(40)]
        self.eval = {"cases": [self.case("validation", i, f"eval_{i:03d}") for i in range(80)]}
        self.test = {"cases": [self.case("test", i, f"test_{i:03d}") for i in range(40)]}
        original = self.build()
        expected_hashes = {role: identity["sha256"] for role, identity in original["inputs"].items()}
        expected_counts = {part: (count["motions"], count["source_groups"])
                           for part, count in original["counts"].items()}
        original["profile"] = "canonical-v4"
        original["dataset_name"] = "OMOMO_retarget_aug_v4"
        original["manifest_content_sha256"] = content_sha256(original)
        with patch.object(manifest_module, "CANONICAL_HASHES", expected_hashes), \
             patch.object(manifest_module, "CANONICAL_COUNTS", expected_counts), \
             patch.object(manifest_module, "CANONICAL_MEMBERSHIP_SHA256", original["membership_sha256"]):
            baseline = self.root / "canonical-fixture.json"
            write_manifest(original, baseline)
            self.assertEqual(read_manifest(baseline), original)

            changed_input = copy.deepcopy(original)
            changed_input["inputs"]["splits"]["sha256"] = "f" * 64
            changed_input["manifest_content_sha256"] = content_sha256(changed_input)
            input_path = self.root / "canonical-forged-input.json"
            write_manifest(changed_input, input_path)
            with self.assertRaisesRegex(ValueError, "input SHA256 pin mismatch"):
                read_manifest(input_path)

            changed_membership = copy.deepcopy(original)
            changed_membership["partitions"]["train"][0]["motion_sha256"] = "e" * 64
            changed_membership["membership_sha256"] = membership_sha256(
                changed_membership["partitions"], changed_membership["panels"])
            changed_membership["manifest_content_sha256"] = content_sha256(changed_membership)
            membership_path = self.root / "canonical-forged-membership.json"
            write_manifest(changed_membership, membership_path)
            with self.assertRaisesRegex(ValueError, "membership pin mismatch"):
                read_manifest(membership_path)


class PathMappingTests(unittest.TestCase):
    def test_mapping_matches_path_components_and_preserves_remainder(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            mappings = [("/mnt/source/v4", local)]
            self.assertEqual(Path(resolve_motion_path("/mnt/source/v4/clip.pt", mappings)), local / "clip.pt")
            self.assertEqual(Path(resolve_motion_path("/mnt/source/v4/nested/clip.pt", mappings)), local / "nested/clip.pt")
            if os.name == "nt":
                # An unmatched foreign source must fail, rather than becoming
                # a drive-relative Windows path or matching the v4 prefix.
                with self.assertRaises(ValueError):
                    resolve_motion_path("/mnt/source/v40/clip.pt", mappings)
                untouched = resolve_motion_path(r"C:\dataset40\clip.pt", [(r"C:\dataset4", local)])
                self.assertEqual(Path(untouched), Path(r"C:\dataset40\clip.pt"))
            else:
                untouched = str(resolve_motion_path("/mnt/source/v40/clip.pt", mappings))
                self.assertEqual(untouched, "/mnt/source/v40/clip.pt")

    def test_longest_explicit_prefix_wins_and_conflicting_maps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary)
            mappings = [("/mnt/source", local / "general"),
                        ("/mnt/source/v4", local / "specific")]
            resolved = resolve_motion_path("/mnt/source/v4/clip.pt", mappings)
            self.assertEqual(Path(resolved), local / "specific" / "clip.pt")
            with self.assertRaises(ValueError):
                resolve_motion_path("/mnt/source/v4/clip.pt", [
                    ("/mnt/source/v4", local / "one"),
                    ("/mnt/source/v4", local / "other"),
                ])

    def test_mapping_does_not_turn_source_identity_into_a_relative_path(self):
        with self.assertRaises(ValueError):
            resolve_motion_path("relative/clip.pt")

    def test_parent_traversal_is_rejected_before_remapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            for source in (
                "/mnt/source/v4/../private.pt",
                "/mnt/source/v4/nested/../../private.pt",
                r"C:\dataset\..\private.pt",
            ):
                with self.subTest(source=source):
                    with self.assertRaises(ValueError):
                        resolve_motion_path(source, [("/mnt/source/v4", Path(temporary))])


if __name__ == "__main__":
    unittest.main()
