"""Frozen reference-intent labels must survive worker filtering and reject drift."""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hoi_residual.assets import _digest_json
from hoi_tables import prepare_contacts as contacts
from hoi_tables.data import ASSETS, CACHE_SCHEMA, SCHEMA, TIMEBASE, TableMotionLib, content_hash, sha256_file
from supermimic.robot_metadata import G1_JOINT_NAMES

REAL_CONTACT_IDENTITY = contacts.contact_identity
REAL_WORKER_INITIALIZER = contacts._initialize_worker


def fixture(root):
    cache = {"schema": CACHE_SCHEMA, "timebase": TIMEBASE, "clips": {}}
    entries = []
    for index, (name, role, length) in enumerate((
            ("sub1_largetable_001", "train", 3), ("sub2_smalltable_001", "train", 4),
            ("sub3_largetable_001", "protected_eval", 4), ("sub4_smalltable_001", "protected_eval", 5))):
        clip_id = "omomo/unitree_g1/" + name
        state, obj = torch.zeros(length, 71), torch.zeros(length, 13)
        state[:, 3], obj[:, 3] = 1., 1.
        state[:, 0] = torch.arange(length) * .02 + index
        cache["clips"][clip_id] = {"states": state, "objects": obj, "source_contacts": torch.zeros(length, 4, dtype=torch.bool)}
        entries.append({"clip_id": clip_id, "subject": name, "source_group": "omomo/" + name,
                        "object_asset": name.split("_")[1], "partition": role, "role": role,
                        "state_frames": length, "source_trajectory_sha256": f"{index:064x}"})
    cache_path = root/"reference_50hz.pt"; torch.save(cache, cache_path)
    manifest = {"schema": SCHEMA, "object_assets": list(ASSETS), "timebase": TIMEBASE,
                "target_joint_names": list(G1_JOINT_NAMES), "clips": entries,
                "cache": {"path": cache_path.name, "sha256": sha256_file(cache_path)},
                "selection": {"included_count": 4, "sha256": "selection", "workbook_sha256": "workbook", "excluded": {}}}
    manifest["content_sha256"] = content_hash(manifest)
    path = root/"manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
    catalog = root/"assets.json"
    catalog.write_text(json.dumps({"hands": {"geometry_cache": {"path": "hands.pt", "sha256": "hands"}}}), encoding="utf-8")
    return path, catalog


class ReferenceOnlyReward:
    """Cheap stand-in for exact geometry; no policy, optimizer or simulator."""
    def __init__(self, mdp, assets, hands, *, native_hands=None):
        self.hand_evidence, self._reference_contacts = {}, {}

    def _expected(self, frames):
        expected = torch.stack(((frames % 2 == 0).float(), (frames % 3 == 1).float()), -1)
        valid = torch.ones_like(expected)
        valid[frames % 3 == 0, 1] = 0.
        return expected, valid


def _parallel_test_identity(motion, assets, binding, **kwargs):
    return {"manifest_sha256": motion.manifest_sha256, "target_robot": "current-URDF",
            "selection": "user-whitelist", "hand_geometry": "current-hw3-probes",
            "implementation": "exact-v11",
            "object_geometry": {a["object_asset"]: {"cache": a["object_asset"] + "-geometry"} for a in assets}}


def _parallel_test_initializer(*args):
    """Spawn-safe cheap dependency fixture; exercises the real pool/task code."""
    assets = [{"asset_id": i, "object_asset": name} for i, name in enumerate(ASSETS)]
    with patch.object(contacts, "_new_kinematics", return_value=object()), \
            patch.object(contacts, "load_assets", return_value=assets), \
            patch.object(contacts, "TablesReward", ReferenceOnlyReward), \
            patch.object(contacts, "contact_identity", side_effect=_parallel_test_identity):
        REAL_WORKER_INITIALIZER(*args)


class FrozenContactCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.catalog = fixture(self.root)
        self.output = self.root/"contact_reference.pt"
        self.assets = [{"asset_id": i, "object_asset": name} for i, name in enumerate(ASSETS)]
        self.binding = {"path": "hands.pt", "sha256": "hands"}
        self.robot = "current-URDF"
        self.geometry = {name: {"cache": name + "-geometry"} for name in ASSETS}
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(contacts, "load_assets", return_value=self.assets))
        stack.enter_context(patch.object(contacts, "TablesReward", ReferenceOnlyReward))
        stack.enter_context(patch.object(contacts, "contact_identity", side_effect=self.identity))

    def identity(self, motion, assets, binding, **kwargs):
        return {"manifest_sha256": motion.manifest_sha256, "target_robot": self.robot,
                "selection": "user-whitelist", "hand_geometry": "current-hw3-probes",
                "implementation": "exact-v11", "object_geometry": {a["object_asset"]: self.geometry[a["object_asset"]] for a in assets}}

    def prepare(self):
        return contacts.prepare_contact_cache(self.manifest, self.catalog, self.output,
                                              chunk_size=2, kin=object())

    def motion(self, **kwargs):
        return TableMotionLib(self.manifest, None, "cpu", for_training=False, **kwargs)

    def mutate(self, change, *, recalculate_content=False):
        value = torch.load(self.output, map_location="cpu", weights_only=True)
        change(value)
        if recalculate_content:
            value["content_sha256"] = _digest_json(contacts._content_identity(value))
        torch.save(value, self.output)

    def test_all_partitions_reference_only_and_progress(self):
        events = []
        before = sha256_file(self.manifest)
        report = contacts.prepare_contact_cache(self.manifest, self.catalog, self.output, chunk_size=2,
                                                kin=object(), progress=events.append)
        self.assertEqual(report["clip_count"], 4)
        self.assertEqual(report["state_frames"], 16)
        self.assertEqual(report["partitions"], {"train": 2, "protected_eval": 2})
        self.assertTrue(report["reference_only"])
        self.assertEqual((report["optimizer_updates"], report["normalizer_updates"], report["rsi_updates"]), (0, 0, 0))
        self.assertEqual(len(events), 4)
        self.assertEqual(events[-1]["completed_frames"], 16)
        self.assertEqual(sha256_file(self.manifest), before)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_clip_ids_remap_filtered_local_frame_indices(self):
        self.prepare()
        full = self.motion()
        full_e, full_v, _ = contacts.load_contact_cache(self.output, full, self.assets, self.binding)
        self.assertEqual(full_e.dtype, torch.float32)
        self.assertEqual(full_v.dtype, torch.bool)
        # Smalltable starts at frame 3 in the full train library, but at local
        # frame 0 in this filtered worker. It must retain its cached frame-3 label.
        filtered = self.motion(asset="smalltable", one_per_asset=True)
        e, v, metadata = contacts.load_contact_cache(self.output, filtered, self.assets[1:], self.binding)
        torch.testing.assert_close(e, full_e[3:])
        torch.testing.assert_close(v, full_v[3:])
        self.assertEqual(float(e[0, 0]), 0.)
        self.assertEqual(metadata["selected_clips"], [filtered.entries[0]["clip_id"]])
        eval_motion = self.motion(partition="protected_eval", asset="smalltable")
        e, v, _ = contacts.load_contact_cache(self.output, eval_motion, self.assets, self.binding)
        self.assertEqual(e.shape, (5, 2))
        self.assertFalse(eval_motion.allowed_for_training)
        # Worker-local entry order is also irrelevant to the frozen global map.
        full.entries = list(reversed(full.entries))
        reordered, _, _ = contacts.load_contact_cache(self.output, full, self.assets, self.binding)
        torch.testing.assert_close(reordered, torch.cat((full_e[3:], full_e[:3])))

    def test_current_robot_or_object_geometry_drift_rejected(self):
        self.prepare()
        motion = self.motion()
        self.robot = "different-URDF"
        with self.assertRaisesRegex(ValueError, "robot/hand/manifest"):
            contacts.load_contact_cache(self.output, motion, self.assets, self.binding)
        self.robot = "current-URDF"
        self.geometry["smalltable"] = {"cache": "changed"}
        with self.assertRaisesRegex(ValueError, "object geometry"):
            contacts.load_contact_cache(self.output, motion, self.assets, self.binding)

    def test_label_bytes_drift_rejected_even_with_same_metadata(self):
        self.prepare()
        def change(value):
            first = next(iter(value["clips"].values()))
            first["expected"][0, 0] = 1. - first["expected"][0, 0]
        self.mutate(change)
        with self.assertRaisesRegex(ValueError, "tensor bytes"):
            contacts.load_contact_cache(self.output, self.motion(), self.assets, self.binding)

    def test_missing_protected_clip_and_false_complete_flag_rejected(self):
        self.prepare()
        def remove_eval(value):
            key = next(k for k, r in value["clips"].items() if r["partition"] == "protected_eval")
            del value["clips"][key]
        self.mutate(remove_eval, recalculate_content=True)
        with self.assertRaisesRegex(ValueError, "omits or adds"):
            contacts.load_contact_cache(self.output, self.motion(asset=0), self.assets, self.binding)
        self.mutate(lambda value: value.update(complete=False), recalculate_content=True)
        with self.assertRaisesRegex(ValueError, "completeness"):
            contacts.load_contact_cache(self.output, self.motion(), self.assets, self.binding)

    def test_source_partition_and_serialized_dtype_changes_rejected(self):
        self.prepare()
        def change_partition(value):
            next(iter(value["clips"].values()))["partition"] = "protected_eval"
        self.mutate(change_partition, recalculate_content=True)
        with self.assertRaisesRegex(ValueError, "partition identity"):
            contacts.load_contact_cache(self.output, self.motion(), self.assets, self.binding)
        value = torch.load(self.output, map_location="cpu", weights_only=True)
        next(iter(value["clips"].values()))["partition"] = "train"
        next(iter(value["clips"].values()))["valid"] = next(iter(value["clips"].values()))["valid"].float()
        value["content_sha256"] = _digest_json(contacts._content_identity(value)); torch.save(value, self.output)
        with self.assertRaisesRegex(ValueError, "dtypes"):
            contacts.load_contact_cache(self.output, self.motion(), self.assets, self.binding)

    def test_hand_identity_is_independent_of_checkout_path(self):
        def evidence(prefix):
            return {side: {"colliders": [{"path": prefix + "/" + side + ".stl", "sha256": side,
                                          "origin_xyz": [0., 0., 0.], "origin_rpy": [0., 0., 0.]}],
                           "probe_count": 128, "probe_sha256": side + "-probes", "probe_frame": "current_wrist_link_metres"}
                    for side in ("left", "right")}
        self.assertEqual(contacts._hands_identity(evidence("/prepare_source/data")),
                         contacts._hands_identity(evidence("/formal_source/data")))

    def test_real_identity_binds_current_urdf_mjcf_and_selection_hashes(self):
        urdf, mjcf, hands, native = [self.root/name for name in ("robot.urdf", "robot.xml", "hands.pt", "object.usda")]
        for path in (urdf, mjcf, hands, native): path.write_bytes(path.name.encode())
        assets = [{"asset_id": index, "object_asset": name, "collision_hash": name,
                   "geometry_cache": {"sha256": name + "-sdf"}, "native_asset": str(native),
                   "native_asset_sha256": sha256_file(native), "exact_scale": [1., 1., 1.]}
                  for index, name in enumerate(ASSETS)]
        evidence = {side: {"colliders": [{"path": "/irrelevant/" + side + ".stl", "sha256": side,
                                         "origin_xyz": [0., 0., 0.], "origin_rpy": [0., 0., 0.]}],
                           "probe_count": 128, "probe_sha256": side + "-probes", "probe_frame": "current_wrist_link_metres"}
                    for side in ("left", "right")}
        # The public helper is patched in setUp for cache serialization tests;
        # call its retained implementation here to inspect real file bindings.
        with patch.object(contacts, "URDF_PATH", str(urdf)), patch.object(contacts, "MJCF_PATH", str(mjcf)), \
                patch.object(contacts, "current_hands", return_value=({}, evidence)), \
                patch.object(contacts, "load_geometry_cache"), patch.object(contacts, "_code_identity", return_value={"algorithm": "frozen"}):
            identity = REAL_CONTACT_IDENTITY(self.motion(), assets, {"path": str(hands), "sha256": sha256_file(hands)})
            self.assertEqual(identity["target_robot"]["urdf_sha256"], sha256_file(urdf))
            self.assertEqual(identity["target_robot"]["mjcf_sha256"], sha256_file(mjcf))
            self.assertEqual(identity["selection"]["workbook_sha256"], "workbook")
            self.assertEqual(set(identity["object_geometry"]), set(ASSETS))
            urdf.write_bytes(b"changed current hand mount")
            changed = REAL_CONTACT_IDENTITY(self.motion(), assets, {"path": str(hands), "sha256": sha256_file(hands)})
            self.assertNotEqual(identity["target_robot"], changed["target_robot"])
            with self.assertRaisesRegex(ValueError, "selected worker asset"):
                REAL_CONTACT_IDENTITY(self.motion(), assets[:1], {"path": str(hands), "sha256": sha256_file(hands)})

    def test_spawn_workers_match_serial_clip_tensors_and_hashes(self):
        self.prepare()
        parallel_path = self.root/"contact_parallel.pt"
        events = []
        with patch.object(contacts, "_new_kinematics", return_value=object()), \
                patch.object(contacts, "_initialize_worker", _parallel_test_initializer):
            report = contacts.prepare_contact_cache(self.manifest, self.catalog, parallel_path,
                                                    chunk_size=2, workers=2, worker_threads=1,
                                                    progress=events.append)
        serial = torch.load(self.output, map_location="cpu", weights_only=True)
        parallel = torch.load(parallel_path, map_location="cpu", weights_only=True)
        self.assertEqual(list(parallel["clips"]), sorted(serial["clips"]))
        self.assertEqual(parallel["metadata"]["identity"], serial["metadata"]["identity"])
        for name, record in serial["clips"].items():
            other = parallel["clips"][name]
            torch.testing.assert_close(record["expected"], other["expected"], rtol=0, atol=0)
            self.assertTrue(torch.equal(record["valid"], other["valid"]))
            self.assertEqual(record["expected_sha256"], other["expected_sha256"])
            self.assertEqual(record["valid_sha256"], other["valid_sha256"])
            self.assertEqual(contacts._clip_identity(record), contacts._clip_identity(other))
        self.assertEqual(report["execution"]["effective_workers"], {"train": 2, "protected_eval": 2})
        self.assertEqual(report["execution"]["process_start_method"], "spawn")
        self.assertEqual(report["state_frames"], 16)
        self.assertEqual(len(events), 4)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["normalizer_updates"], 0)
        self.assertEqual(report["rsi_updates"], 0)

    def test_invalid_worker_budget_and_initializer_failure_are_fatal(self):
        for workers, threads in ((0, 1), (True, 1), (2, 0)):
            with self.assertRaisesRegex(ValueError, "positive"):
                contacts.prepare_contact_cache(self.manifest, self.catalog, self.output,
                                                workers=workers, worker_threads=threads, kin=object())
        with patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "4"}):
            self.assertEqual(contacts._worker_count(8, 1, 2), 2)
            with self.assertRaisesRegex(ValueError, "allocated CPU"):
                contacts._worker_count(8, 1, 20)
        # An initializer failure becomes a single task exception instead of an
        # endless Pool initializer-respawn loop, and never publishes an output.
        original_context = contacts._WORKER_CONTEXT
        try:
            with patch.object(contacts, "_new_kinematics", side_effect=ValueError("broken frozen geometry")):
                REAL_WORKER_INITIALIZER(str(self.manifest), str(self.catalog), "train", 2, 1, {}, None)
            with self.assertRaisesRegex(RuntimeError, "broken frozen geometry"):
                contacts._worker_clip("omomo/unitree_g1/sub1_largetable_001")
        finally:
            contacts._WORKER_CONTEXT = original_context
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
