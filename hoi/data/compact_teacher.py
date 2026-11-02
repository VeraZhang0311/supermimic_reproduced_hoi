"""Audited, non-destructive selection of the compact privileged-teacher corpus.

The derivative owns its complete train/eval partitions, but reuses frozen
reference/native caches byte-for-byte. Its ancestry is checked before an old
reference manifest identity can be accepted. No augmentation or repartitioning
occurs here; omitted protected examples never become training examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .curated import canonical_bytes, digest, load_json, sha

SPLIT_SCHEMA = "hoi_compact_teacher_split_v1"
BUNDLE_SCHEMA = "hoi_compact_teacher_bundle_v1"
ULTRA_OBJECTS = ("largebox", "smallbox", "plasticbox", "suitcase")
HOI_OBJECTS = ("clothesstand", "floorlamp", "largetable", "monitor", "smalltable",
               "trashcan", "tripod", "whitechair", "woodchair")
SOCCER_TASKS = ("soccer/case1_kick_forward", "soccer/case2_kick_left",
                "soccer/case3_kick_right", "soccer/case4_carry", "soccer/case5_pickball_kick")
TRAIN_CHAIRS = {
    2: "maple_shaker_peg_back_dining_chair_simple",
    122: "emeco_navy_chair_brushed_aluminum",
    169: "simple_pine_kitchen_chair_with_turned_legs",
    172: "laminated_plywood_bent_dining_chair_stackable",
    200: "monobloc_white_plastic_stackable_patio_chair",
}
EVAL_CHAIRS = {
    35: "scandinavian_birch_plywood_stacking_chair_minimal",
    48: "swedish_gustavian_pale_gray_painted_dining_chair",
    87: "oak_t_back_vintage_farmhouse_dining_chair_stained",
    124: "pine_farmhouse_trestle_dining_chair_distressed_cream",
    199: "country_mission_style_chair_quartersawn_oak",
}


def selection_policy():
    return {"schema": "hoi_compact_teacher_selection_v1", "ultra": "all_current_curated",
            "hoi_retarget_exclude_objects": list(ULTRA_OBJECTS),
            "omnicontact_tasks": ["box/case1_carry", *SOCCER_TASKS],
            "grail_train_chairs": {str(k): v for k, v in TRAIN_CHAIRS.items()},
            "grail_protected_eval_chairs": {str(k): v for k, v in EVAL_CHAIRS.items()},
            "chair_selection_basis": "published_metadata_common_chair_names; not a new visual acceptance claim",
            "omni_table_selection_basis": "all_carry_retained; geometric_near_table_audit_includes_7_boundary_clips",
            "partition_rule": "preserve_parent_membership_and_source_groups; no augmentation",
            "protected_source_rule": "excluded_eval_remains_protected_in_parent; never_reassigned_to_train"}


def clip_key(row):
    return row["dataset"], row["clip_id"]


def selection_reason(clip, partition):
    """Return (included, reason) without guessing missing source task labels."""
    if partition not in ("train", "protected_eval") or clip.get("partition") != partition:
        raise ValueError("compact selection requires the original train/eval membership")
    dataset = clip["dataset"]
    if dataset == "ultra":
        if clip["source_provenance"].get("object") not in ULTRA_OBJECTS:
            raise ValueError("unexpected ULTRA object in frozen curated input")
        return True, "retain_all_ultra_curated"
    if dataset == "hoi_retarget":
        name = clip["source_provenance"].get("object")
        if name in ULTRA_OBJECTS:
            return False, "exclude_hoi_object_category_covered_by_ultra"
        if name not in HOI_OBJECTS:
            raise ValueError("unknown HOI-Retarget object cannot silently enter compact selection")
        return True, "retain_hoi_other_object"
    if dataset == "omnicontact":
        task = clip["task"]
        if task == "box/case1_carry":
            return True, "retain_complete_table_interaction_carry"
        if task in SOCCER_TASKS:
            return True, "retain_all_soccer_tasks"
        if task in ("box/case2_push", "box/case3_slide"):
            return False, "exclude_omni_box_without_table_interaction"
        raise ValueError("unknown OmniContact task cannot silently enter compact selection")
    if dataset == "grail_sitting":
        chairs = TRAIN_CHAIRS if partition == "train" else EVAL_CHAIRS
        old_id = clip["asset_id"]
        if old_id not in chairs:
            return False, "exclude_grail_outside_frozen_common_chairs"
        if clip.get("task") != "sitting" or not clip["clip_id"].startswith("sitting__" + chairs[old_id] + "__"):
            raise ValueError("GRAIL asset ID/name/task binding differs from approved chair selection")
        return True, "retain_grail_train_chair" if partition == "train" else "retain_independent_grail_eval_chair"
    raise ValueError("unknown dataset cannot silently enter compact selection")


def _sealed_json(path, field="content_sha256"):
    value = load_json(path)
    if value.get(field) != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError("compact ancestor/manifest content hash mismatch")
    return value


def _binding(path, *, bundle=False):
    path = Path(path).resolve()
    result = {"path": str(path), "sha256": sha(path)}
    if bundle:
        result["bundle_content_sha256"] = _sealed_json(path, "bundle_content_sha256")["bundle_content_sha256"]
    return result


def _bound_parent(spec, *, full=False):
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("compact parent bundle bytes changed")
    if full:
        # The original header reader already parses and validates its content
        # digest. Do not first parse/canonicalize the same 100+ MB file here.
        from .curated_bundle import read_curated_bundle_header
        parent = read_curated_bundle_header(spec["path"])
    else:
        parent = _sealed_json(spec["path"], "bundle_content_sha256")
    if parent.get("compact_selection") is not None:
        raise ValueError("compact selection must derive directly from the frozen curated parent")
    if parent["bundle_content_sha256"] != spec["bundle_content_sha256"]:
        raise ValueError("compact parent bundle content identity changed")
    return parent


def select_parent_clips(parents):
    """Deterministic selection and leakage checks independent of cache loading."""
    if set(parents) != {"train", "protected_eval"}:
        raise ValueError("compact teacher requires both frozen partitions")
    selected, excluded, groups, keys = {}, [], {}, set()
    for partition in ("train", "protected_eval"):
        parent = parents[partition]
        if parent.get("partition") != partition or parent["scope"].get("is_subset"):
            raise ValueError("compact parents must be complete original train/eval bundles")
        if parent["manifest"] != parents["train"]["manifest"]:
            raise ValueError("compact parents must bind the same frozen split")
        selected[partition] = []
        for clip in parent["clips"]:
            key, group = clip_key(clip), clip["source_group"]
            if key in keys or (group in groups and groups[group] != partition):
                raise ValueError("parent duplicate clip or source-group leakage")
            if clip.get("mandatory_protected") and partition != "protected_eval":
                raise ValueError("parent protected source was placed in training")
            keys.add(key); groups[group] = partition
            keep, reason = selection_reason(clip, partition)
            if keep:
                selected[partition].append(clip)
            else:
                excluded.append({"partition": partition, "dataset": key[0], "clip_id": key[1],
                                 "source_group": group, "original_asset_id": clip["asset_id"], "reason": reason})
        expected = set(TRAIN_CHAIRS if partition == "train" else EVAL_CHAIRS)
        actual = {c["asset_id"] for c in selected[partition] if c["dataset"] == "grail_sitting"}
        if actual != expected:
            raise ValueError("each approved GRAIL chair must have its complete original partition clips")
        if not selected[partition]:
            raise ValueError("compact partition cannot be empty")
    return selected, excluded


def _registry(parents, selected, *, copy_assets=True):
    originals = sorted({c["asset_id"] for cs in selected.values() for c in cs})
    maps = {part: {a["asset_id"]: a for a in b["assets"]} for part, b in parents.items()}
    assets, mapping = [], []
    for new, old in enumerate(originals):
        a = maps["train"].get(old)
        if a is None or a != maps["protected_eval"].get(old):
            raise ValueError("parent partitions disagree on exact physical asset registry")
        # Reports need only the count and ID map; avoid copying large collision
        # meshes merely to discard them immediately after counting.
        assets.append({**deepcopy(a), "asset_id": new, "original_asset_id": old} if copy_assets else a)
        mapping.append({"asset_id": new, "original_asset_id": old,
                        "native_equivalence_sha256": a["native_equivalence_sha256"]})
    return assets, mapping


def selection_report(parents, selected, excluded):
    assets, mapping = _registry(parents, selected, copy_assets=False)
    report = {"schema": "hoi_compact_teacher_selection_report_v1", "selection_policy": selection_policy(),
              "asset_id_map": mapping, "registry_assets": len(assets), "partitions": {},
              "excluded_by_reason": dict(sorted(Counter(r["reason"] for r in excluded).items())),
              "augmentation": False, "training_started": False,
              "scope": "data selection/identity only; no new physics or tracking-success claim"}
    for part, cs in selected.items():
        grail = [c for c in cs if c["dataset"] == "grail_sitting"]
        report["partitions"][part] = {"clips": len(cs), "used_assets": len({c["asset_id"] for c in cs}),
                                      "source_groups": len({c["source_group"] for c in cs}),
                                      "by_dataset": dict(sorted(Counter(c["dataset"] for c in cs).items())),
                                      "grail_original_asset_ids": sorted({c["asset_id"] for c in grail}),
                                      "grail_clips": len(grail), "grail_source_groups": len({c["source_group"] for c in grail}),
                                      "excluded_clips": len(parents[part]["clips"]) - len(cs)}
    report["grail_evaluation"] = {"mode": "held_out_chair_asset_and_source_group",
                                   "source_group_disjoint": True, "native_asset_disjoint": True,
                                   "human_template_independence": "not_established_by_published_metadata"}
    return report


def _partition_path(root, name):
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("compact partition path must stay inside its manifest directory")
    result = (root / candidate).resolve()
    if not result.is_relative_to(root):
        raise ValueError("compact partition path escapes its manifest directory")
    return result


def _read_compact_context(path, *, full_parents=False, _materialization_snapshot=None):
    value = _sealed_json(path)
    if value.get("schema") != SPLIT_SCHEMA or value.get("selection_policy") != selection_policy():
        raise ValueError("compact split schema/approved selection policy differs")
    if set(value["parent_bundles"]) != {"train", "protected_eval"}:
        raise ValueError("compact split lost its paired parent identities")
    if _materialization_snapshot is None:
        parents = {part: _bound_parent(spec, full=full_parents) for part, spec in value["parent_bundles"].items()}
    else:
        # Private, call-local reuse while constructing both derivative files.
        # Public readers never receive a snapshot and revalidate every ancestor.
        parents = _materialization_snapshot["parents"]
        if value["parent_bundles"] != _materialization_snapshot["parent_specs"]:
            raise ValueError("materialization parent bindings changed")
        for part, spec in value["parent_bundles"].items():
            if (sha(spec["path"]) != spec["sha256"] or
                    parents[part]["bundle_content_sha256"] != spec["bundle_content_sha256"]):
                raise ValueError("compact parent bundle bytes changed during materialization")
    selected, excluded = select_parent_clips(parents)
    original = value["parent_manifest"]
    if original != parents["train"]["manifest"] or sha(original["path"]) != original["sha256"]:
        raise ValueError("compact original split identity changed")
    from .curated import read_split
    if _materialization_snapshot is None:
        parent_split, parent_rows = read_split(original["path"])
    else:
        parent_split, parent_rows = _materialization_snapshot["parent_split_rows"]
    if parent_split["content_sha256"] != original["content_sha256"]:
        raise ValueError("compact parent split content changed")
    if parent_split.get("schema") == SPLIT_SCHEMA:
        raise ValueError("compact parent split cannot itself be a derivative")
    expected = {}
    for part in selected:
        lookup = {clip_key(r): r for r in parent_rows[part]}
        if len(lookup) != len(parent_rows[part]) or set(lookup) != {clip_key(c) for c in parents[part]["clips"]}:
            raise ValueError("parent bundle must cover every original partition row")
        expected[part] = []
        for clip in selected[part]:
            row = lookup[clip_key(clip)]
            for field, field_value in row.items():
                # Preparation resolves this formerly optional metadata field
                # to the explicitly hash-bound native robot recipe.
                if field == "robot_geometry_sha256" and field_value is None:
                    if clip.get(field) != parents[part].get(field):
                        raise ValueError("selected parent clip robot recipe binding differs")
                    continue
                if clip.get(field) != field_value:
                    raise ValueError("selected parent clip differs from frozen split row: " + field)
            expected[part].append(row)
    if value.get("report") != selection_report(parents, selected, excluded):
        raise ValueError("compact selection report/asset mapping changed")
    root = Path(path).resolve().parent
    if set(value["partitions"]) != set(expected):
        raise ValueError("compact split must declare both complete selected partitions")
    rows = {}
    for part, spec in value["partitions"].items():
        filename = _partition_path(root, spec["path"])
        if sha(filename) != spec["sha256"]:
            raise ValueError("compact partition bytes changed")
        rows[part] = [json.loads(line) for line in filename.read_text(encoding="utf-8").splitlines()]
        if spec["count"] != len(rows[part]) or rows[part] != expected[part]:
            raise ValueError("compact partition must be the exact approved parent selection in original order")
    return value, rows, parents, selected


def read_compact_split(path):
    value, rows, _, _ = _read_compact_context(path)
    return value, rows


def _derive_bundle(partition, parents, selected, split, split_path):
    from hoi.v1.task_sampling import default_sampling_contract
    parent = parents[partition]
    assets, mapping = _registry(parents, selected)
    remap = {r["original_asset_id"]: r["asset_id"] for r in mapping}
    # Only the explicitly replaced collections below are modified. Deep-copying
    # every parent clip/mesh here before immediately replacing them duplicates
    # hundreds of MB and provides no additional isolation.
    result = dict(parent)
    result["assets"] = assets
    result["clips"] = [{**deepcopy(c), "asset_id": remap[c["asset_id"]], "original_asset_id": c["asset_id"]}
                       for c in selected[partition]]
    result["object_geometry_bindings"] = [{**deepcopy(b), "asset_id": remap[b["asset_id"]]}
                                          for b in parent["object_geometry_bindings"] if b["asset_id"] in remap]
    result["manifest"] = {"path": str(Path(split_path).resolve()), "sha256": sha(split_path),
                          "content_sha256": split["content_sha256"]}
    result["scope"] = {"available_motions": len(result["clips"]), "selected_motions": len(result["clips"]),
                       "is_subset": False, "purpose": "compact_privileged_teacher",
                       "parent_available_motions": len(parent["clips"]), "selection_is_parent_subset": True}
    result["compact_selection"] = {"schema": BUNDLE_SCHEMA, "split_content_sha256": split["content_sha256"],
                                   "parent_bundle": split["parent_bundles"][partition],
                                   "reference_manifest_content_sha256": parent["manifest"]["content_sha256"],
                                   "asset_id_map": mapping}
    if partition == "train":
        result["task_sampling"] = default_sampling_contract()
    else:
        result.pop("task_sampling", None)
    result.pop("bundle_content_sha256", None)
    result["bundle_content_sha256"] = digest(result)
    return result


def read_compact_bundle_header(path):
    bundle = _sealed_json(path, "bundle_content_sha256")
    if bundle.get("compact_selection", {}).get("schema") != BUNDLE_SCHEMA:
        raise ValueError("compact bundle lineage schema mismatch")
    spec = bundle["manifest"]
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("compact bundle split bytes changed")
    split, _, parents, selected = _read_compact_context(spec["path"], full_parents=True)
    if spec["content_sha256"] != split["content_sha256"] or bundle["partition"] not in selected:
        raise ValueError("compact bundle split/partition identity mismatch")
    expected = _derive_bundle(bundle["partition"], parents, selected, split, spec["path"])
    if bundle != expected:
        raise ValueError("compact bundle differs from the exact authenticated derivative")
    return bundle


def validated_compact_parents(bundle):
    """For native evidence inheritance: return fully verified original headers."""
    if bundle.get("compact_selection", {}).get("schema") != BUNDLE_SCHEMA:
        raise ValueError("native inheritance requires a compact bundle")
    spec = bundle["manifest"]
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("compact evidence split changed")
    split, _, parents, selected = _read_compact_context(spec["path"], full_parents=True)
    if bundle != _derive_bundle(bundle["partition"], parents, selected, split, spec["path"]):
        raise ValueError("native inheritance requires the exact authenticated derivative")
    return parents


def plan_compact_dataset(train_bundle, eval_bundle):
    specs = {part: _binding(path, bundle=True) for part, path in
             (("train", train_bundle), ("protected_eval", eval_bundle))}
    parents = {part: _bound_parent(spec) for part, spec in specs.items()}
    selected, excluded = select_parent_clips(parents)
    return {"parent_bundles": specs, "report": selection_report(parents, selected, excluded),
            "excluded": excluded, "validation": "metadata_only; full source/cache identity checked during materialization/load"}


def materialize_compact_dataset(train_bundle, eval_bundle, output):
    """Create a new immutable derivative; never rewrite original caches/assets."""
    from .curated import read_split
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("compact output must be new: " + str(output))
    specs = {part: _binding(path, bundle=True) for part, path in
             (("train", train_bundle), ("protected_eval", eval_bundle))}
    parents = {part: _bound_parent(spec, full=True) for part, spec in specs.items()}
    selected, excluded = select_parent_clips(parents)
    original_manifest = parents["train"]["manifest"]
    parent_split, parent_rows = read_split(original_manifest["path"])
    snapshot = {"parents": parents, "parent_specs": specs,
                "parent_split_rows": (parent_split, parent_rows)}
    # Build all metadata in memory before writing a fresh directory.
    partitions = {}
    for part, clips in selected.items():
        lookup = {clip_key(r): r for r in parent_rows[part]}
        partitions[part] = [lookup[clip_key(c)] for c in clips]
    report = selection_report(parents, selected, excluded)
    output.mkdir(parents=True, exist_ok=False)
    split_root = output / "split"
    split_root.mkdir()
    files = {}
    for part, rows in partitions.items():
        filename = split_root / (part + ".jsonl")
        filename.write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
        files[part] = {"path": filename.name, "sha256": sha(filename), "count": len(rows)}
    split = {"schema": SPLIT_SCHEMA, "selection_id": "compact_privileged_teacher_v1",
             "selection_policy": selection_policy(), "parent_bundles": specs,
             "parent_manifest": original_manifest, "partitions": files, "report": report}
    split["content_sha256"] = digest(split)
    split_path = split_root / "split.json"
    split_path.write_bytes(canonical_bytes(split) + b"\n")
    _read_compact_context(split_path, _materialization_snapshot=snapshot)
    bundle_paths = {}
    for part in selected:
        directory = output / ("bundle_" + part)
        directory.mkdir()
        bundle = _derive_bundle(part, parents, selected, split, split_path)
        path = directory / "bundle.json"
        path.write_bytes(canonical_bytes(bundle) + b"\n")
        # Ancestors were fully validated once above. Check exactly the bytes we
        # just wrote against that authenticated in-memory derivation; a later
        # public load still performs full parent/split/dependency validation.
        if _sealed_json(path, "bundle_content_sha256") != bundle:
            raise ValueError("written compact bundle differs from authenticated derivation")
        bundle_paths[part] = _binding(path, bundle=True)
    (output / "excluded.jsonl").write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in excluded))
    result = {"schema": "hoi_compact_teacher_materialization_v1", "bundles": bundle_paths,
              "split": _binding(split_path), "report": report,
              "native_acceptance": "requires_explicit_parent_evidence_inheritance_and_fresh_collection_check",
              "source_caches_rewritten": False, "physics_steps": 0, "training_started": False}
    (output / "report.json").write_bytes(canonical_bytes(result) + b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "materialize"))
    parser.add_argument("--train-bundle", type=Path, required=True)
    parser.add_argument("--eval-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        value = plan_compact_dataset(args.train_bundle, args.eval_bundle)
        with args.output.open("xb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
    else:
        value = materialize_compact_dataset(args.train_bundle, args.eval_bundle, args.output)
    print(json.dumps(value["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
