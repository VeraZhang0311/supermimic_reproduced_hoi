"""CPU end-to-end preparation gates using real serialized source tensors."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from hoi.data.manifest import read_manifest, robot_joint_names
from hoi.data.prepare import main, validate_dataset


def tensor_motion(states, offset):
    frames = torch.zeros(states, 630, dtype=torch.float32)
    times = torch.arange(states, dtype=torch.float32) / 60
    frames[:, 0] = offset + 2 * times
    frames[:, 71] = offset - 3 * times
    frames[:, 6] = 1
    frames[:, 77] = 1
    frames[:, 201:357].reshape(states, 39, 4)[:, :, 3] = 1
    frames[:, 7:13] = 99
    frames[:, 78:84] = 77
    frames[:, 591:630] = (torch.arange(states) % 2)[:, None]
    return frames


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = self.root / "immutable-sources"
        self.sources.mkdir()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.prefix = "/mnt/synthetic/v4"
        self.local_paths = {}
        self.partitions = {name: [] for name in ("train", "validation", "regression", "test")}
        for index, partition in enumerate(("train", "train", "validation", "regression", "test")):
            filename = f"sub{index:02d}_smallbox_{index:03d}_100_100_100.pt"
            source = self.sources / filename
            torch.save(tensor_motion(13, index), source)
            entry = {
                "motion": f"{self.prefix}/{filename}",
                "motion_sha256": self.digest(source),
                "source_group": f"sub{index:02d}_smallbox_{index:03d}",
                "object": "smallbox",
                "object_asset": "smallbox_100_100_100",
            }
            self.partitions[partition].append(entry)
            self.local_paths[entry["source_group"]] = source
        self.split_path = self.inputs / "split.json"
        self.eval_path = self.inputs / "eval.json"
        self.test_path = self.inputs / "test.json"
        self.bindings_path = self.inputs / "bindings.json"
        self.selection_path = self.inputs / "selection.json"
        self.eval_frames = self.test_frames = 13

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def source(self, partition="train", index=0):
        return self.local_paths[self.partitions[partition][index]["source_group"]]

    def rewrite_source(self, payload, partition="train", index=0):
        path = self.source(partition, index)
        torch.save(payload, path)
        self.partitions[partition][index]["motion_sha256"] = self.digest(path)

    def json_input(self, path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def write_inputs(self):
        self.json_input(self.split_path, {"partitions": self.partitions})
        self.json_input(self.eval_path, {"cases": [{
            **self.partitions["validation"][0], "id": "smallbox_080", "source_frames": self.eval_frames,
        }]})
        self.json_input(self.test_path, {"cases": [{
            **self.partitions["test"][0], "id": "smallbox_test_000", "source_frames": self.test_frames,
        }]})
        self.json_input(self.selection_path, {"cases": [
            dict(entry, partition="train") for entry in self.partitions["train"]
        ]})
        self.json_input(self.bindings_path, {
            "source_joint_names": robot_joint_names(),
            "joint_order_evidence": {"status": "synthetic_test_only"},
            "objects": {},
        })

    def cli(self, arguments):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main([str(value) for value in arguments])

    def build(self, *, mapping=True, mapping_root=None):
        self.write_inputs()
        output = self.root / "manifest-output"
        arguments = [
            "manifest", "--splits", self.split_path, "--eval-cases", self.eval_path,
            "--test-cases", self.test_path, "--train-selection", self.selection_path,
            "--bindings", self.bindings_path, "--profile", "custom", "--output", output,
        ]
        if mapping:
            arguments.extend(["--path-map", f"{self.prefix}={mapping_root or self.sources}"])
        self.assertEqual(self.cli(arguments), 0)
        return output / "manifest.json"

    def report(self, output):
        return json.loads((output / "validation.json").read_text(encoding="utf-8"))

    def test_cli_cached_subset_has_601_to_501_states_without_mutating_source_bytes(self):
        self.rewrite_source(tensor_motion(601, 0))
        before = {path: path.read_bytes() for path in self.local_paths.values()}
        manifest_path = self.build()
        manifest = read_manifest(manifest_path)
        self.assertTrue(manifest["validation"]["adapter_ready"])
        self.assertFalse(manifest["validation"]["training_ready"])
        output = self.root / "validation-output"
        result = self.cli([
            "validate", "--manifest", manifest_path, "--limit", 1, "--cache", "--output", output,
        ])
        self.assertEqual(result, 0)
        report = self.report(output)
        self.assertTrue(report["success"])
        self.assertEqual(report["scope"], {
            "partition": "train", "available_motions": 2, "selected_motions": 1, "is_subset": True,
        })
        self.assertFalse(report["training_started"])
        self.assertEqual(report["native_physics"], "not_run")
        digest = self.partitions["train"][0]["motion_sha256"]
        cache_path = output / f"reference_{digest}.pt"
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.assertEqual(cached["metadata"]["control_steps"], 500)
        self.assertEqual(cached["metadata"]["state_frames"], 501)
        self.assertEqual(cached["metadata"]["canonical_duration_s"], 10)
        self.assertEqual(cached["time"].shape, (501,))
        self.assertEqual(cached["time"].dtype, torch.float64)
        self.assertEqual(float(cached["time"][-1]), 10)
        self.assertEqual(cached["robot"]["joint_pos"].shape, (501, 29))
        self.assertEqual(cached["object"]["pos"].shape, (501, 3))
        self.assertEqual(cached["body"]["pos"].shape, (501, 39, 3))
        self.assertEqual(cached["body"]["contact_flags"].shape, (501, 39))
        torch.testing.assert_close(cached["robot"]["root_pos"][:, 0].double(),
                                   2 * cached["time"], atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(cached["object"]["pos"][:, 0].double(),
                                   -3 * cached["time"], atol=3e-6, rtol=1e-6)
        self.assertEqual(cached["provenance"]["source_motion_sha256"], digest)
        self.assertEqual(report["motion_audits"][0]["cache"]["sha256"], self.digest(cache_path))
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_default_validation_reads_every_train_motion_and_no_held_out_motion(self):
        manifest_path = self.build()
        # Invalid held-out data must remain outside this default train audit.
        self.source("regression").write_bytes(b"unselected source is never decoded")
        output = self.root / "validation-output"
        report = validate_dataset(manifest_path, output)
        self.assertTrue(report["success"])
        self.assertEqual(report["validated_motions"], 2)
        self.assertFalse(report["scope"]["is_subset"])
        self.assertEqual({row["source_group"] for row in report["motion_audits"]},
                         {entry["source_group"] for entry in self.partitions["train"]})
        self.assertEqual(self.report(output), report)
        self.assertFalse(list(output.glob("*.pt")))

    def test_changed_source_sha_fails_with_a_persisted_report(self):
        manifest_path = self.build()
        self.source().write_bytes(b"source was changed after identity was frozen")
        output = self.root / "validation-output"
        self.assertEqual(self.cli(["validate", "--manifest", manifest_path, "--output", output]), 1)
        report = self.report(output)
        self.assertFalse(report["success"])
        self.assertEqual(report["validated_motions"], 0)
        self.assertIn("SHA256 mismatch", report["errors"][0]["error"])
        self.assertEqual(len(report["errors"]), 1)

    def test_checkpoint_container_with_matching_sha_is_rejected_as_wrong_schema(self):
        self.rewrite_source({"model": {"weight": torch.ones(2)}, "iteration": 100})
        manifest_path = self.build()
        output = self.root / "validation-output"
        report = validate_dataset(manifest_path, output, cache=True)
        self.assertFalse(report["success"])
        self.assertIn("plain ULTRA motion Tensor", report["errors"][0]["error"])
        self.assertFalse(list(output.glob("*.pt")))
        self.assertEqual(self.report(output), report)

    def test_weights_only_rejection_is_reported_without_unrestricted_pickle_fallback(self):
        # This is a real torch archive whose harmless object requires a pickle
        # global that the restricted loader rejects. No decoder is mocked.
        self.rewrite_source({"unsupported_checkpoint_object": object()})
        manifest_path = self.build()
        output = self.root / "validation-output"
        self.assertEqual(self.cli(["validate", "--manifest", manifest_path, "--output", output]), 1)
        report = self.report(output)
        self.assertFalse(report["success"])
        self.assertEqual(report["validated_motions"], 0)
        self.assertTrue(report["errors"])

    def test_declared_frame_count_is_checked_against_real_tensor(self):
        self.partitions["train"][0]["source_frames"] = 14
        manifest_path = self.build()
        output = self.root / "validation-output"
        report = validate_dataset(manifest_path, output)
        self.assertFalse(report["success"])
        self.assertIn("frame count", report["errors"][0]["error"])
        self.assertEqual(report["validated_motions"], 0)

    def test_panel_frame_count_is_checked_even_when_split_entry_has_no_count(self):
        self.eval_frames = 14
        manifest_path = self.build()
        output = self.root / "validation-output"
        report = validate_dataset(manifest_path, output, partition="eval80")
        self.assertFalse(report["success"])
        self.assertIn("frame count", report["errors"][0]["error"])

    def test_missing_file_is_a_failure_and_is_not_silently_skipped(self):
        manifest_path = self.build()
        self.source().unlink()
        output = self.root / "validation-output"
        report = validate_dataset(manifest_path, output)
        self.assertFalse(report["success"])
        self.assertEqual(report["validated_motions"], 0)
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["motion"], self.partitions["train"][0]["motion"])
        self.assertEqual(self.report(output), report)

    def test_cli_path_map_override_can_relocate_source_bytes(self):
        manifest_path = self.build(mapping_root=self.root / "source-on-another-host")
        output = self.root / "validation-output"
        self.assertEqual(self.cli([
            "validate", "--manifest", manifest_path, "--path-map", f"{self.prefix}={self.sources}",
            "--output", output,
        ]), 0)
        report = self.report(output)
        self.assertEqual(report["validated_motions"], 2)
        for audit in report["motion_audits"]:
            self.assertEqual(Path(audit["resolved_motion_path"]).parent, self.sources)

    def test_missing_foreign_path_mapping_is_rejected(self):
        manifest_path = self.build(mapping=False)
        output = self.root / "validation-output"
        code = self.cli(["validate", "--manifest", manifest_path, "--output", output])
        if os.name == "nt":
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
        else:
            # A POSIX host treats /mnt/... as native; absent source bytes still
            # fail the audit and must never be substituted by basename.
            self.assertEqual(code, 1)
            self.assertFalse(self.report(output)["success"])

    def test_existing_output_is_refused_and_previous_contents_survive(self):
        manifest_path = self.build()
        output = self.root / "already-used-output"
        output.mkdir()
        marker = output / "previous-result.txt"
        marker.write_text("preserve this result", encoding="utf-8")
        self.assertEqual(self.cli(["validate", "--manifest", manifest_path, "--output", output]), 2)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve this result")
        self.assertFalse((output / "validation.json").exists())

    def test_outputs_must_be_separate_from_a_remapped_source_root(self):
        manifest_path = self.build()
        output = self.sources / "forbidden-derived-output"
        with self.assertRaises(ValueError):
            validate_dataset(manifest_path, output, cache=True)
        self.assertFalse(output.exists())

    def test_outputs_must_be_separate_from_native_source_root_without_path_maps(self):
        for entries in self.partitions.values():
            for entry in entries:
                entry["motion"] = str(self.local_paths[entry["source_group"]])
        manifest_path = self.build(mapping=False)
        output = self.sources / "forbidden-derived-output"
        with self.assertRaises(ValueError):
            validate_dataset(manifest_path, output, cache=True)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
