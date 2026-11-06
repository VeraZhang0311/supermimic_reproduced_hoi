"""Lossless packed projection, source integrity, and fixed small eval subset."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.prepare_bones_filtered import (FORMAT, G1_JOINT_NAMES, STATE_LAYOUT,
                                         prepare, prepare_eval, sha256)


class PrepareFilteredTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="bones_projection_", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "old_packed"
        self.source.mkdir()
        self.output = self.root / "new_full"
        # Unequal, deliberately unsorted clip names and arbitrary world velocities
        # make resampling, name sorting, differentiation, and local rotation fail.
        self.lengths = [3, 6, 2, 5]
        frames = sum(self.lengths)
        self.raw = (np.arange(frames * 253).reshape(frames, 253) / 1000).astype("<f4")
        self.raw[:, 61:65] = [0., 0., 0., -1.]
        # Body copies deliberately differ from the authoritative explicit root.
        self.raw[:, 71:] += 100
        entries, offset = [], 0
        for name, length in zip(["zeta", "beta", "alpha", "gamma"], self.lengths):
            entries.append({"name": name, "offset": offset, "frames": length,
                            "fps": 50.0, "duration": (length - 1) / 50.,
                            "npz_sha256": "a" * 64, "source_sha256": "b" * 64})
            offset += length
        self.index = {"schema_version": 1, "state_file": "states.npy", "dtype": "float32",
                      "quaternion_order": "wxyz", "joint_names": G1_JOINT_NAMES,
                      "body_names": ["pelvis", *[f"body{i}" for i in range(13)]],
                      "root_body": "pelvis", "state_dim": 253, "default_pose": None,
                      "total_frames": frames, "clips": entries}
        self.save_source()

    def save_source(self):
        np.save(self.source / "states.npy", self.raw)
        self.index["state_file_bytes"] = (self.source / "states.npy").stat().st_size
        self.index["state_file_sha256"] = sha256(self.source / "states.npy")
        self.save_index()

    def save_index(self):
        (self.source / "index.json").write_text(json.dumps(self.index), encoding="utf-8")

    @staticmethod
    def read_states(directory, frames):
        return np.fromfile(directory / "states.bin", dtype="<f4").reshape(frames, 71)

    def test_exact_projection_and_complete_clip_lineage_with_bounded_chunks(self):
        before = {p.name: sha256(p) for p in self.source.iterdir()}
        from tools.prepare_bones_filtered import _project
        sizes = []

        def observe(block):
            sizes.append(len(block))
            return _project(block)

        with patch("tools.prepare_bones_filtered._project", side_effect=observe):
            result = prepare(self.source, self.output, chunk_frames=2)
        self.assertEqual(sizes, [2] * 8)
        expected = np.concatenate((self.raw[:, 58:71], self.raw[:, :58]), axis=1)
        actual = self.read_states(self.output, len(self.raw))
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual([x["name"] for x in result["clips"]], [x["name"] for x in self.index["clips"]])
        self.assertEqual([x["frames"] for x in result["clips"]], self.lengths)
        for old, new in zip(self.index["clips"], result["clips"]):
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertEqual(new["source_offset"], old["offset"])
        self.assertEqual(result["format"], FORMAT)
        self.assertEqual(result["state_layout"], STATE_LAYOUT)
        self.assertEqual(result["joint_names"], G1_JOINT_NAMES)
        self.assertEqual(result["dtype"], "<f4")
        self.assertEqual(result["control_dt"], .02)
        self.assertEqual(result["frames"], len(self.raw))
        self.assertEqual(result["states_sha256"], sha256(self.output / "states.bin"))
        self.assertEqual(result["lineage"]["source_index_sha256"], before["index.json"])
        self.assertEqual(result["lineage"]["source_state_file_sha256"], before["states.npy"])
        self.assertTrue(result["lineage"]["source_file_hash_verified"])
        self.assertEqual({p.name: sha256(p) for p in self.source.iterdir()}, before)
        self.assertFalse(list(self.output.glob("*.partial")))

    def test_fixed_eval_copies_complete_clips_and_is_explicit_training_subset(self):
        eval_dir = self.root / "eval"
        full = prepare(self.source, self.output, chunk_frames=3, eval_output=eval_dir, eval_clips=3, eval_seed=0)
        full_hash = sha256(self.output / "manifest.json")
        evaluation = json.loads((eval_dir / "manifest.json").read_text())
        selection = json.loads((eval_dir / "selection.json").read_text())
        order = sorted(range(4), key=lambda i: self.index["clips"][i]["name"])
        sampled = random.Random(0).sample(range(4), 3)
        chosen = [order[i] for i in sampled]
        self.assertEqual(selection["sampled_sorted_indices"], sampled)
        self.assertEqual(selection["names"], [self.index["clips"][i]["name"] for i in chosen])
        self.assertEqual(evaluation["selection"], "eval_training_subset")
        self.assertFalse(evaluation["held_out"])
        self.assertFalse(selection["training_excludes_selection"])
        self.assertEqual(evaluation["parent_manifest_sha256"], full_hash)
        self.assertEqual(selection["parent_manifest_sha256"], full_hash)
        self.assertEqual(evaluation["selection_sha256"], sha256(eval_dir / "selection.json"))
        self.assertEqual(len(full["clips"]), 4)  # Training keeps every eval clip.
        self.assertEqual(evaluation["states_sha256"], sha256(eval_dir / "states.bin"))
        expected = []
        for i in chosen:
            entry = self.index["clips"][i]
            block = self.raw[entry["offset"]:entry["offset"] + entry["frames"]]
            expected.append(np.concatenate((block[:, 58:71], block[:, :58]), axis=1))
        self.assertEqual(self.read_states(eval_dir, evaluation["frames"]).tobytes(), np.concatenate(expected).tobytes())
        for entry, i in zip(evaluation["clips"], chosen):
            self.assertEqual(entry["frames"], self.lengths[i])
            self.assertEqual(entry["parent_offset"], self.index["clips"][i]["offset"])
        second = self.root / "eval_again"
        prepare_eval(self.output, second, clips=3, seed=0, chunk_frames=1)
        self.assertEqual(sha256(second / "states.bin"), evaluation["states_sha256"])

    def test_existing_destinations_and_source_descendants_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.output)
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / "child")
        with self.assertRaises(ValueError):
            prepare(self.source, self.root / "full2", eval_output=self.root / "full2" / "eval", eval_clips=2)
        self.assertFalse((self.root / "full2").exists())
        with self.assertRaises(ValueError):
            prepare(self.source, self.root / "full3", eval_output=self.root / "eval3", eval_clips=5)
        self.assertFalse((self.root / "full3").exists())

    def test_metadata_disagreement_never_creates_manifest(self):
        original = copy.deepcopy(self.index)
        mutations = [lambda x: x.update(quaternion_order="xyzw"),
                     lambda x: x.update(joint_names=list(reversed(G1_JOINT_NAMES))),
                     lambda x: x["clips"][1].update(offset=999),
                     lambda x: x["clips"][1].update(fps=120),
                     lambda x: x["clips"][1].update(duration=100),
                     lambda x: x["clips"][1].update(name="zeta"),
                     lambda x: x.update(total_frames=999),
                     lambda x: x.update(state_file_bytes=4)]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                self.index = copy.deepcopy(original)
                mutate(self.index)
                self.save_index()
                output = self.root / f"bad{i}"
                with self.assertRaises(ValueError):
                    prepare(self.source, output, chunk_frames=2)
                self.assertFalse((output / "manifest.json").exists())

    def test_corrupt_source_hash_and_nonfinite_state_cannot_publish_manifest(self):
        self.index["state_file_sha256"] = "0" * 64
        self.save_index()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare(self.source, self.output, chunk_frames=2)
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertTrue((self.output / "states.bin.partial").exists())
        self.raw[5, 0] = np.nan
        self.save_source()
        bad = self.root / "nonfinite"
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prepare(self.source, bad, chunk_frames=2)
        self.assertFalse((bad / "manifest.json").exists())

    def test_source_mutation_and_invalid_quaternion_are_rejected(self):
        from tools.prepare_bones_filtered import _project
        changed = False

        def mutate(block):
            nonlocal changed
            if not changed:
                changed = True
                self.index["extra"] = "modified during prepare"
                self.save_index()
            return _project(block)

        with patch("tools.prepare_bones_filtered._project", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                prepare(self.source, self.output, chunk_frames=2)
        self.assertFalse((self.output / "manifest.json").exists())
        self.raw[0, 61:65] = 0
        self.save_source()
        with self.assertRaisesRegex(ValueError, "quaternion"):
            prepare(self.source, self.root / "bad_quat")

    def test_eval_rejects_damaged_full_library(self):
        prepare(self.source, self.output)
        with (self.output / "states.bin").open("r+b") as stream:
            stream.write(b"XXXX")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_eval(self.output, self.root / "bad_eval", clips=2)
        self.assertFalse((self.root / "bad_eval").exists())


if __name__ == "__main__":
    unittest.main()
