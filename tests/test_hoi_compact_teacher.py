from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hoi.data import compact_teacher as compact
from hoi.data.curated import canonical_bytes, digest, sha, read_split, write_split
from hoi.data.curated_bundle import prepare_curated_bundle, read_curated_bundle_header
from hoi.v1.prepare import load_bundle
from tests import test_hoi_curated_data as fixtures


def clip(dataset, name, part="train", *, aid=0, task="unspecified", group=None, obj=None):
    return {"dataset": dataset, "clip_id": name, "partition": part, "asset_id": aid,
            "source_group": group or part + ":" + name, "source_provenance": {"object": obj},
            "task": task, "mandatory_protected": part == "protected_eval"}


def parents():
    result = {}
    manifest = {"path": "immutable", "sha256": "a" * 64, "content_sha256": "b" * 64}
    for part, chairs in (("train", compact.TRAIN_CHAIRS), ("protected_eval", compact.EVAL_CHAIRS)):
        clips = [clip("ultra", part + "_box", part, aid=3, obj="largebox"),
                 clip("hoi_retarget", part + "_box_duplicate", part, aid=4, obj="largebox"),
                 clip("hoi_retarget", part + "_lamp", part, aid=5, obj="floorlamp"),
                 clip("omnicontact", part + "_carry", part, aid=6, task="box/case1_carry"),
                 clip("omnicontact", part + "_push", part, aid=7, task="box/case2_push"),
                 clip("omnicontact", part + "_slide", part, aid=8, task="box/case3_slide")]
        clips += [clip("omnicontact", part + "_" + str(i), part, aid=9, task=t) for i, t in enumerate(compact.SOCCER_TASKS)]
        clips += [clip("grail_sitting", "sitting__" + name + "__000", part, aid=aid, task="sitting")
                  for aid, name in chairs.items()]
        clips += [clip("grail_sitting", part + "_ornate_throne", part, aid=15, task="sitting")]
        result[part] = {"partition": part, "manifest": manifest, "scope": {"is_subset": False}, "clips": clips,
                        "assets": [{"asset_id": i, "native_equivalence_sha256": str(i), "physics": {"mass": i + 1}}
                                   for i in range(201)]}
    return result


class CompactSelectionTests(unittest.TestCase):
    def test_exact_selection_preserves_all_soccer_and_rejects_duplicate_box_categories(self):
        source = parents(); before = deepcopy(source)
        selected, excluded = compact.select_parent_clips(source)
        self.assertEqual(source, before)
        for part in selected:
            self.assertEqual(len(selected[part]), 13)
            self.assertEqual({c["task"] for c in selected[part] if c["dataset"] == "omnicontact"},
                             {"box/case1_carry", *compact.SOCCER_TASKS})
            self.assertFalse(any(c["dataset"] == "hoi_retarget" and c["source_provenance"]["object"] in compact.ULTRA_OBJECTS
                                 for c in selected[part]))
        self.assertEqual(len(excluded), 8)
        self.assertEqual((selected, excluded), compact.select_parent_clips(source))

    def test_chair_identity_and_complete_membership_are_required(self):
        source = parents()
        chair = next(c for c in source["train"]["clips"] if c["dataset"] == "grail_sitting")
        chair["clip_id"] = "sitting__different_chair__000"
        with self.assertRaisesRegex(ValueError, "ID/name/task"):
            compact.select_parent_clips(source)
        source = parents()
        source["protected_eval"]["clips"] = [c for c in source["protected_eval"]["clips"] if c["asset_id"] != 35]
        with self.assertRaisesRegex(ValueError, "each approved GRAIL"):
            compact.select_parent_clips(source)

    def test_unknown_labels_and_wrong_partition_fail_closed(self):
        for row in (clip("omnicontact", "x", task="box/new_task"),
                    clip("hoi_retarget", "x", obj="unknown"),
                    clip("ultra", "x", obj="unknown"), clip("new_dataset", "x")):
            with self.assertRaises(ValueError):
                compact.selection_reason(row, "train")
        with self.assertRaisesRegex(ValueError, "original train/eval"):
            compact.selection_reason(clip("ultra", "x", "protected_eval", obj="largebox"), "train")

    def test_full_parent_protection_and_leakage_checked_before_exclusion(self):
        source = parents()
        source["train"]["clips"][0]["mandatory_protected"] = True
        with self.assertRaisesRegex(ValueError, "protected source"):
            compact.select_parent_clips(source)
        source = parents()
        # Even a subsequently excluded record cannot hide parent leakage.
        source["protected_eval"]["clips"][1]["source_group"] = source["train"]["clips"][1]["source_group"]
        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            compact.select_parent_clips(source)

    def test_registry_remaps_ids_without_changing_any_physical_metadata(self):
        source = parents(); selected, _ = compact.select_parent_clips(source)
        assets, mapping = compact._registry(source, selected)
        self.assertEqual([a["asset_id"] for a in assets], list(range(len(assets))))
        old = {a["asset_id"]: a for a in source["train"]["assets"]}
        self.assertLess(len(assets), len(old))
        for a in assets:
            self.assertEqual({k: v for k, v in a.items() if k not in ("asset_id", "original_asset_id")},
                             {k: v for k, v in old[a["original_asset_id"]].items() if k != "asset_id"})
        self.assertEqual({r["original_asset_id"] for r in mapping}, {c["asset_id"] for cs in selected.values() for c in cs})
        source["protected_eval"]["assets"][3]["physics"]["mass"] += 1
        with self.assertRaisesRegex(ValueError, "physical asset registry"):
            compact._registry(source, selected)


class CompactDerivativeTests(unittest.TestCase):
    """Real tiny parent bundle/cache files exercise ancestry and no-cache-rewrite."""
    def setUp(self):
        fixtures.CuratedBundleTests.setUp(self)
        # The geometry fixture has two ULTRA rows. Add its actual object category
        # to source evidence before freezing the new test parent split.
        metadata_path = self.root / "metadata.json"
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        for row in records:
            root = Path(row["output"])
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            metadata["provenance"]["object"] = "largebox"
            (root / "metadata.json").write_bytes(canonical_bytes(metadata))
            success = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
            success["metadata_sha256"] = sha(root / "metadata.json")
            (root / "SUCCESS.json").write_bytes(canonical_bytes(success))
            row["provenance"] = metadata["provenance"]
            row["metadata_sha256"] = success["metadata_sha256"]
        metadata_path.write_bytes(canonical_bytes(records))
        self.split = self.root / "parent_split" / "split.json"
        write_split(self.selection, metadata_path, self.split.parent)
        self.parent_paths = {}
        for part in ("train", "protected_eval"):
            out = self.root / ("parent_" + part)
            prepare_curated_bundle(self.split, self.recipe, out, self.kin, partition=part)
            self.parent_paths[part] = out / "bundle.json"
        # Production parents share the same prepared geometry registry. Make
        # the tiny fixture share those exact cache bindings as well.
        training = read_curated_bundle_header(self.parent_paths["train"])
        evaluation = read_curated_bundle_header(self.parent_paths["protected_eval"])
        evaluation["assets"] = deepcopy(training["assets"])
        evaluation["bundle_content_sha256"] = digest({k: v for k, v in evaluation.items() if k != "bundle_content_sha256"})
        self.parent_paths["protected_eval"].write_bytes(canonical_bytes(evaluation))
        # Isolate dataset-identity integration from the independently tested
        # sixteen-category policy; this two-clip fixture has no sitting corpus.
        for target in ("TRAIN_CHAIRS", "EVAL_CHAIRS"):
            patcher = patch.object(compact, target, {})
            patcher.start(); self.addCleanup(patcher.stop)
        sampler = patch("hoi.v1.task_sampling.default_sampling_contract", return_value=None)
        sampler.start(); self.addCleanup(sampler.stop)
        self.output = self.root / "derived"
        self.result = compact.materialize_compact_dataset(self.parent_paths["train"], self.parent_paths["protected_eval"], self.output)

    def test_new_complete_split_and_parent_caches_load_without_rewriting(self):
        split, rows = read_split(self.output / "split/split.json")
        self.assertEqual(split["schema"], compact.SPLIT_SCHEMA)
        self.assertEqual([len(rows[p]) for p in ("train", "protected_eval")], [1, 1])
        for part in ("train", "protected_eval"):
            parent = read_curated_bundle_header(self.parent_paths[part])
            path = self.output / ("bundle_" + part) / "bundle.json"
            derived = read_curated_bundle_header(path)
            self.assertFalse(derived["scope"]["is_subset"])
            self.assertEqual(derived["clips"][0]["reference_cache"], parent["clips"][0]["reference_cache"])
            self.assertNotEqual(derived["manifest"]["content_sha256"], parent["manifest"]["content_sha256"])
            bank, _, _ = load_bundle(path, self.kin, require_train=part == "train")
            self.assertEqual(bank.num_frames, 3)
            self.assertEqual(compact.validated_compact_parents(derived)[part], parent)

    def test_changed_derived_physics_rejected_even_with_resealed_bundle(self):
        path = self.output / "bundle_train/bundle.json"
        bundle = json.loads(path.read_text(encoding="utf-8"))
        bundle["assets"][0]["physical_profile"]["mass"] += .01
        bundle["bundle_content_sha256"] = digest({k: v for k, v in bundle.items() if k != "bundle_content_sha256"})
        path.write_bytes(canonical_bytes(bundle))
        with self.assertRaisesRegex(ValueError, "exact authenticated derivative"):
            read_curated_bundle_header(path)

    def test_changed_original_bundle_bytes_rejected(self):
        path = self.parent_paths["train"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "parent bundle bytes changed"):
            read_curated_bundle_header(self.output / "bundle_train/bundle.json")

    def test_partition_membership_cannot_be_resealed_into_a_different_selection(self):
        root = self.output / "split"
        split = json.loads((root / "split.json").read_text(encoding="utf-8"))
        filename = root / "train.jsonl"
        rows = [json.loads(s) for s in filename.read_text(encoding="utf-8").splitlines()]
        rows[0]["source_group"] = "changed"
        filename.write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
        split["partitions"]["train"]["sha256"] = sha(filename)
        split["content_sha256"] = digest({k: v for k, v in split.items() if k != "content_sha256"})
        (root / "split.json").write_bytes(canonical_bytes(split))
        with self.assertRaisesRegex(ValueError, "exact approved parent selection"):
            read_split(root / "split.json")

    def test_parent_cache_tamper_is_detected_at_load(self):
        path = self.output / "bundle_train/bundle.json"
        bundle = read_curated_bundle_header(path)
        cache = Path(bundle["clips"][0]["reference_cache"]["path"])
        cache.write_bytes(cache.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "cache SHA256"):
            load_bundle(path, self.kin)

    def test_existing_output_is_never_overwritten(self):
        before = (self.output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            compact.materialize_compact_dataset(self.parent_paths["train"], self.parent_paths["protected_eval"], self.output)
        self.assertEqual(before, (self.output / "report.json").read_bytes())

    def test_materialization_fully_validates_each_parent_once_without_global_cache(self):
        from hoi.data import curated_bundle
        before = {part: path.read_bytes() for part, path in self.parent_paths.items()}
        with patch.object(curated_bundle, "read_curated_bundle_header", wraps=read_curated_bundle_header) as reader:
            result = compact.materialize_compact_dataset(self.parent_paths["train"], self.parent_paths["protected_eval"], self.root / "second")
        self.assertEqual(reader.call_count, 2)
        self.assertEqual({Path(call.args[0]) for call in reader.call_args_list}, set(self.parent_paths.values()))
        self.assertEqual(before, {part: path.read_bytes() for part, path in self.parent_paths.items()})
        # Snapshot reuse is confined to construction. Each public read retains
        # full validation even in the same process that materialized the files.
        with patch.object(curated_bundle, "read_curated_bundle_header", wraps=read_curated_bundle_header) as reader:
            compact.read_compact_bundle_header(result["bundles"]["train"]["path"])
            compact.read_compact_bundle_header(result["bundles"]["train"]["path"])
        self.assertEqual(reader.call_count, 4)


if __name__ == "__main__":
    unittest.main()
