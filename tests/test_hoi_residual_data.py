import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hoi_residual.data import (ASSETS, UltraMotionLib, adapt_motion, build_manifest,
                               content_hash, read_manifest, select_clips)
from supermimic.robot_metadata import G1_JOINT_NAMES


def source_motion(n=61, offset=0):
    raw = torch.zeros(n, 630)
    raw[:, 6] = 1
    raw[:, 77] = 1
    raw[:, 201:357].reshape(n, 39, 4)[:, :, 3] = 1
    time = torch.arange(n) / 60
    raw[:, 0] = time + offset
    raw[:, 2] = 0.8
    raw[:, 71] = 2 * time + offset
    raw[:, 73] = 0.4
    raw[:, 13:42] = time[:, None] * torch.arange(29)[None, :] / 100
    raw[:, 591] = (torch.arange(n) % 2).float()
    return raw


def fixture_manifest(root):
    partitions = {}
    for index, partition in enumerate(("train", "validation", "regression", "test")):
        path = root / f"sub{index}_largebox_000_fixture.pt"
        torch.save(source_motion(offset=index / 10), path)
        partitions[partition] = [{"motion": str(path), "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "source_group": f"sub{index}_largebox_000", "object": "largebox",
                                  "object_asset": "largebox_100_100_100"}]
    bindings = {"source_joint_names": list(G1_JOINT_NAMES),
                "joint_order_evidence": {"fixture_sha256": "0" * 64, "field": "metadata.dof_names"}}
    manifest = build_manifest(json.dumps({"partitions": partitions}).encode(), json.dumps(bindings).encode(), profile="fixture")
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


class FakeKin:
    def forward(self, p, R, v, w, q, qd):
        return p[:, None], R[:, None], v[:, None], w[:, None]


class UltraResidualDataTests(unittest.TestCase):
    def test_shared_60_to_50_grid_velocities_and_contacts(self):
        raw = source_motion(601)
        robot, obj, contacts = adapt_motion(raw, G1_JOINT_NAMES)
        self.assertEqual(robot.shape, (501, 71))
        self.assertEqual(obj.shape, (501, 13))
        self.assertEqual(contacts.shape, (501, 39))
        self.assertTrue(torch.allclose(robot[:, 0], torch.arange(501) / 50))
        self.assertTrue(torch.allclose(obj[:, 0], 2 * robot[:, 0]))
        self.assertTrue(torch.allclose(robot[:, 7], torch.ones(501), atol=4e-5))
        self.assertTrue(torch.allclose(obj[:, 7], torch.full((501,), 2.), atol=8e-5))
        indices = (torch.arange(501) * 6 + 2) // 5
        self.assertTrue(torch.equal(contacts[:, 0], raw[indices, 591]))

    def test_joint_mapping_uses_external_names(self):
        names = list(reversed(G1_JOINT_NAMES))
        robot, _, _ = adapt_motion(source_motion(), names)
        expected = source_motion()[-1, 13:42].flip(0)
        self.assertTrue(torch.allclose(robot[-1, 13:42], expected))

    def test_bad_quaternion_contact_and_nonfinite_fail(self):
        for column, value in ((6, 0), (591, 0.5), (10, float("nan"))):
            with self.subTest(column=column):
                raw = source_motion()
                raw[2, column] = value
                with self.assertRaises(ValueError):
                    adapt_motion(raw, G1_JOINT_NAMES)

    def test_protected_never_selected_for_training(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest = fixture_manifest(Path(td))
            for partition in ("validation", "regression", "test", "protected_eval"):
                with self.assertRaisesRegex(ValueError, "Protected eval"):
                    select_clips(manifest, partition=partition)
            self.assertEqual(len(select_clips(manifest, partition="protected_eval", for_training=False)), 3)

    def test_group_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            _, m = fixture_manifest(Path(td))
            entries = {p: [dict(e) for e in m["clips"] if e["partition"] == p] for p in m["partitions"]}
            entries["validation"][0]["source_group"] = entries["train"][0]["source_group"]
            entries["validation"][0]["motion"] = entries["train"][0]["motion"].replace("fixture.pt", "another.pt")
            bindings = {"source_joint_names": G1_JOINT_NAMES, "joint_order_evidence": m["joint_order_evidence"]}
            with self.assertRaisesRegex(ValueError, "leakage"):
                build_manifest(json.dumps({"partitions": entries}).encode(), json.dumps(bindings).encode(), profile="fixture")

    def test_manifest_role_and_content_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path, m = fixture_manifest(Path(td))
            m["clips"][1]["role"] = "train"
            m["content_sha256"] = content_hash(m)
            path.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role"):
                read_manifest(path)

    def test_motion_library_contract_and_bins(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = fixture_manifest(Path(td))
            lib = UltraMotionLib(path, FakeKin(), "cpu", one_per_asset=True)
            self.assertEqual(lib.num_frames, 51)
            self.assertEqual(lib.num_bins, 1)  # no isolated bin for the final state
            self.assertEqual(lib.goal(torch.tensor([0, 49]), 10).shape, (2, 730))
            self.assertTrue(bool((lib.future_frames(torch.tensor([49]), 10) <= 50).all()))
            asset = ASSETS.index("largebox_100_100_100")
            self.assertEqual(lib.compatible_bins(asset).tolist(), [0])
            self.assertEqual(lib.object_ref(torch.tensor([0]))[0].shape, (1, 3))
            self.assertEqual(lib.contact_ref(torch.tensor([0])).shape, (1, 39))

    def test_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path, m = fixture_manifest(Path(td))
            source = Path(m["clips"][0]["motion"])
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                UltraMotionLib(path, FakeKin(), "cpu")

    def test_training_does_not_read_protected_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path, m = fixture_manifest(Path(td))
            for entry in m["clips"]:
                if entry["role"] == "protected_eval":
                    Path(entry["motion"]).unlink()
            self.assertEqual(len(UltraMotionLib(path, FakeKin(), "cpu").entries), 1)
            with self.assertRaisesRegex(ValueError, "Protected eval"):
                UltraMotionLib(path, FakeKin(), "cpu", partition="test")

    def test_gmt_goal_is_identical_to_shared_implementation(self):
        from supermimic.motion import MotionLib
        with tempfile.TemporaryDirectory() as td:
            path, _ = fixture_manifest(Path(td))
            lib = UltraMotionLib(path, FakeKin(), "cpu")
            frames = torch.tensor([0, 1, 40, 49, 50])
            self.assertTrue(torch.equal(lib.goal(frames, 10), MotionLib.goal(lib, frames, 10)))


if __name__ == "__main__":
    unittest.main()
