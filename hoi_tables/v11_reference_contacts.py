"""Pack authorized v1.1 reference intent without geometric recomputation.

These labels are historical task references. Their generating robot/FK/hand
geometry is recorded separately from the current physical reward geometry.
No FK, SDF query, contact classification, simulator or learning is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch

from hoi_residual.assets import _digest_json
from hoi_residual.data import ASSETS, read_manifest, sha256_file
from hoi_residual.v11_reward import LABEL_SCHEMA


SCHEMA = "hoi_ultra_v11_cached_reference_intent_v1"
LABEL_ORIGIN = "v11_cached_reference_intent"
LABEL_SOURCE = {"branch": "hoi/v1.1-observation-matching",
                "commit": "66d918488ec085377bb0d97c4c029ded9d729d86"}
CONTRACT = {"label_origin": LABEL_ORIGIN, "label_schema": LABEL_SCHEMA,
            "current_hw3_geometry_recomputed": False, "source_values_preserved": True,
            "source_body_contact_flags_used": False,
            "state_alignment": "Tplus1_reference_states; interval_reward_indexes_next_state",
            "meaning": "historical v1.1 task reference intent; not current hw3 geometric truth"}


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode() + tensor.numpy().tobytes()).hexdigest()


def _binary_pair(expected, valid, frames):
    for value in (expected, valid):
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != (frames, 2)
                or not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all()):
            raise ValueError("v1.1 expected/valid must be finite binary Tplus1 x2 tensors")
    # Preserve both tensors, including expected values under an unknown mask.
    # The unchanged reward masks them; repacking must not silently rewrite them.
    return expected.detach().cpu().contiguous().clone(), valid.detach().cpu().contiguous().clone()


def _resolve(binding, parent):
    result = deepcopy(binding)
    path = Path(result["path"])
    result["path"] = str((path if path.is_absolute() else Path(parent) / path).resolve())
    return result


def _bundle(path, *, train=False):
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    declared = value.pop("bundle_content_sha256", None)
    if value.get("schema_version") != "supermimic_hoi_v1_bundle_v1" or declared != _digest_json(value):
        raise ValueError("v1.1 source bundle schema/content hash mismatch")
    if train and value.get("partition") != "train":
        raise ValueError("v1.1 training intent requires the historical train bundle")
    asset_ids = [a["asset_id"] for a in value["assets"]]
    names = [a["object_asset"] for a in value["assets"]]
    if (len(set(asset_ids)) != len(asset_ids) or len(set(names)) != len(names)
            or not set(names).issubset(ASSETS) or train and set(names) != set(ASSETS)):
        raise ValueError("v1.1 source exact object asset IDs are invalid")
    for asset in value["assets"]:
        scale = .8 if asset["object_asset"].endswith("_080_080_080") else 1.
        if asset.get("exact_scale") != [scale] * 3:
            raise ValueError("v1.1 source exact object scale differs from asset name")
    timebase = value.get("timebase", {})
    if (timebase.get("source_fps_assumed") != 60 or timebase.get("control_dt_s") != .02
            or timebase.get("timebase_mode") != "canonical_60hz"):
        raise ValueError("v1.1 source bundle must retain canonical60/control50 timing")
    identity = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": declared, "partition": value.get("partition"),
                "clip_count": len(value["clips"]), "timebase": deepcopy(timebase),
                "source_generation_robot_fk": {key: deepcopy(value.get(key)) for key in
                    ("reference_kinematics", "robot_fk", "native_robot_asset", "robot_collision_extension", "hands")},
                "object_geometry": {a["object_asset"]: {key: deepcopy(a.get(key)) for key in
                    ("exact_scale", "collision_hash", "geometry_cache")} for a in value["assets"]}}
    return path, value, identity


def _clip_identity(entry):
    return {key: entry[key] for key in ("clip_id", "source_group", "object_asset", "partition",
                                       "source_partition", "source_trajectory_sha256", "state_frames")}


def _implementation():
    return {"hoi_tables/v11_reference_contacts.py":
            hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def _inventory(manifest, canonical, helper_root=None):
    # The existing selector hashes canonical inputs and validates membership;
    # it constructs neither a motion library nor kinematics.
    from .ultra import canonical_binding, _inventory as inventory
    full = read_manifest(manifest)
    binding = canonical_binding(manifest, canonical, helper_root=helper_root)
    return full, binding, inventory(full, binding)


def _read_record(path, bundle, bundle_identity, clip, target):
    asset_by_id = {a["asset_id"]: a for a in bundle["assets"]}
    asset = asset_by_id.get(clip.get("asset_id"))
    if (asset is None or asset["object_asset"] != target["object_asset"]
            or clip["source_motion_sha256"] != target["source_trajectory_sha256"]
            or clip.get("source_group") != target["source_group"]
            or clip.get("object_asset", asset["object_asset"]) != target["object_asset"]):
        raise ValueError("v1.1 source motion SHA/group/exact object asset mismatch")
    frames = clip.get("state_frames")
    if type(frames) is not int or frames < 2 or "state_frames" in target and frames != target["state_frames"]:
        raise ValueError("v1.1 source Tplus1 state-frame count mismatch")
    source_frames = clip.get("source_frames")
    if source_frames is not None and (type(source_frames) is not int or source_frames < 3
                                     or frames != 5 * (source_frames - 1) // 6 + 1):
        raise ValueError("v1.1 source/control Tplus1 frame grid mismatch")
    cache = _resolve(clip["contact_cache"], path.parent)
    raw = Path(cache["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cache["sha256"]:
        raise ValueError("v1.1 source contact cache SHA256 mismatch")
    pair = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if pair.get("metadata", {}).get("label_schema") != LABEL_SCHEMA:
        raise ValueError("v1.1 cached reference label schema mismatch")
    expected, valid = _binary_pair(pair["expected_contact"], pair["expected_valid"], frames)
    identity = deepcopy(target); identity["state_frames"] = frames
    return {**_clip_identity(identity), "expected": expected, "valid": valid,
            "expected_sha256": _tensor_hash(expected), "valid_sha256": _tensor_hash(valid),
            "source_bindings": [{"bundle_sha256": bundle_identity["sha256"],
                                 "bundle_partition": bundle_identity["partition"], "contact_cache": cache,
                                 "source_frames": source_frames, "source_state_frames": frames,
                                 "source_expected_dtype": str(expected.dtype), "source_valid_dtype": str(valid.dtype),
                                 "source_expected_sha256": _tensor_hash(expected), "source_valid_sha256": _tensor_hash(valid)}]}


def _content(value):
    return {"schema": value["schema"], "complete": value["complete"], "metadata": value["metadata"],
            "clips": {name: {key: field for key, field in record.items() if key not in ("expected", "valid")}
                      for name, record in value["clips"].items()}}


def prepare_reused_cache(manifest, train_bundle, eval_bundles, canonical, output, helper_root=None):
    """Repack source values into a NEW full train2651 + development80 cache."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if isinstance(eval_bundles, (str, Path)):
        eval_bundles = [eval_bundles]
    if not eval_bundles:
        raise ValueError("v1.1 reuse requires evaluation bundles covering all canonical80")
    full, binding, inventory = _inventory(manifest, canonical, helper_root)
    by_sha = {e["source_trajectory_sha256"]: e for e in inventory.values()}
    if len(by_sha) != len(inventory):
        raise ValueError("v1.1 reuse inventory contains duplicate motion SHA")
    records, sources = {}, []
    paths = [(train_bundle, True)] + [(p, False) for p in eval_bundles]
    for source_path, training in paths:
        path, bundle, source_identity = _bundle(source_path, train=training)
        sources.append(source_identity)
        seen = set()
        for clip in bundle["clips"]:
            digest = clip.get("source_motion_sha256")
            if digest in seen:
                raise ValueError("Duplicate v1.1 source motion SHA within one bundle")
            seen.add(digest)
            target = by_sha.get(digest)
            if target is None or (target["partition"] == "train") != training:
                continue
            record = _read_record(path, bundle, source_identity, clip, target)
            name = record["clip_id"]
            if name in records:
                prior = records[name]
                if (_clip_identity(prior) != _clip_identity(record)
                        or prior["expected_sha256"] != record["expected_sha256"]
                        or prior["valid_sha256"] != record["valid_sha256"]):
                    raise ValueError("Conflicting v1.1 cached reference intent across evaluation bundles")
                prior["source_bindings"].extend(record["source_bindings"])
            else:
                records[name] = record
        expected_train = {e["source_trajectory_sha256"] for e in inventory.values() if e["partition"] == "train"}
        if training and seen != expected_train:
            raise ValueError("v1.1 train bundle must cover exactly the frozen train motion SHA set")
        if sha256_file(path) != source_identity["sha256"]:
            raise RuntimeError("v1.1 source bundle changed during repacking")
    if set(records) != set(inventory):
        missing = sorted(set(inventory) - set(records))
        raise ValueError(f"v1.1 reference cache coverage incomplete: missing {len(missing)} train/canonical80 clips")
    object_sources = {}
    for source in sources:
        for name, geometry in source["object_geometry"].items():
            identity = {k: geometry[k] for k in ("exact_scale", "collision_hash")}
            if name in object_sources and object_sources[name] != identity:
                raise ValueError("v1.1 source object geometry differs between train/eval bundles")
            object_sources[name] = identity
    metadata = {"label_origin": LABEL_ORIGIN, "label_source": deepcopy(LABEL_SOURCE), "contract": deepcopy(CONTRACT),
                "identity": {"ultra_manifest_sha256": sha256_file(manifest),
                             "canonical_manifest_sha256": binding["canonical_manifest_sha256"],
                             "manifest_timebase": deepcopy(full["timebase"]),
                             "repacking_implementation_sha256": _implementation()},
                "source_bundles": sources, "source_object_geometry": object_sources,
                "clip_count": len(records), "state_frames": sum(e["state_frames"] for e in records.values()),
                "partitions": {name: sum(e["partition"] == name for e in records.values()) for name in ("train", "protected_eval")},
                "reference_only": True, "optimizer_updates": 0, "normalizer_updates": 0, "rsi_updates": 0}
    value = {"schema": SCHEMA, "complete": True, "metadata": metadata,
             "clips": {name: records[name] for name in sorted(records)}}
    value["content_sha256"] = _digest_json(_content(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(value, stream)
    return {**metadata, "file_sha256": sha256_file(output), "content_sha256": value["content_sha256"]}


def load_reused_cache(path, motion, assets, hands_binding, *, _raw=None, _value=None):
    """Validate and map a self-contained source-bound pack to local clip order.

    Old source files were byte-verified during packing. Runtime validates the
    complete packed payload/provenance, avoiding thousands of source-file opens.
    Current reward geometry is recorded separately and never called the label
    generator. The owning checkpoint binds the packed file SHA for evaluation.
    """
    raw = Path(path).read_bytes() if _raw is None else _raw
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True) if _value is None else _value
    if (value.get("schema") != SCHEMA or value.get("complete") is not True
            or value.get("content_sha256") != _digest_json(_content(value))):
        raise ValueError("v1.1 reused cache schema/content/completeness mismatch")
    metadata = value["metadata"]
    if (metadata.get("label_origin") != LABEL_ORIGIN or metadata.get("label_source") != LABEL_SOURCE
            or metadata.get("contract") != CONTRACT or metadata.get("reference_only") is not True
            or any(metadata.get(k) != 0 for k in ("optimizer_updates", "normalizer_updates", "rsi_updates"))):
        raise ValueError("v1.1 reused cache origin/reference-only contract mismatch")
    full, binding, inventory = _inventory(motion.source_manifest_path, motion.canonical_manifest_path,
                                         getattr(motion, "hoi80_helper_root", None))
    identity = {"ultra_manifest_sha256": motion.ultra_manifest_sha256,
                "canonical_manifest_sha256": binding["canonical_manifest_sha256"],
                "manifest_timebase": deepcopy(full["timebase"]), "repacking_implementation_sha256": _implementation()}
    if metadata["identity"] != identity:
        raise ValueError("v1.1 reused cache manifest/canonical/repacker identity mismatch")
    if set(value["clips"]) != set(inventory) or metadata["clip_count"] != len(inventory):
        raise ValueError("v1.1 reused cache must cover complete train2651/canonical80 inventory")
    bundle_hashes = {b["sha256"] for b in metadata["source_bundles"]}
    if not bundle_hashes or not any(b["partition"] == "train" for b in metadata["source_bundles"]):
        raise ValueError("v1.1 reused cache lacks source bundle provenance")
    for name, record in value["clips"].items():
        target = inventory[name]
        if (any(record[k] != target[k] for k in _clip_identity(record) if k != "state_frames")
                or type(record["state_frames"]) is not int or record["state_frames"] < 2
                or "state_frames" in target and record["state_frames"] != target["state_frames"]):
            raise ValueError("v1.1 reused cache clip/source/asset/Tplus1 identity mismatch")
        e, v = _binary_pair(record["expected"], record["valid"], record["state_frames"])
        if _tensor_hash(e) != record["expected_sha256"] or _tensor_hash(v) != record["valid_sha256"]:
            raise ValueError("v1.1 reused cache tensor values changed")
        if not record["source_bindings"] or any(s["bundle_sha256"] not in bundle_hashes
                or s["source_state_frames"] != record["state_frames"]
                or s["source_expected_sha256"] != record["expected_sha256"]
                or s["source_valid_sha256"] != record["valid_sha256"] for s in record["source_bindings"]):
            raise ValueError("v1.1 reused cache source cache provenance mismatch")
    if (metadata["state_frames"] != sum(e["state_frames"] for e in value["clips"].values())
            or metadata["partitions"] != {p: sum(e["partition"] == p for e in value["clips"].values())
                                         for p in ("train", "protected_eval")}):
        raise ValueError("v1.1 reused cache aggregate counts mismatch")
    if len(assets) != 8 or [a["object_asset"] for a in assets] != list(ASSETS):
        raise ValueError("v1.1 runtime binding requires all eight sorted exact assets")
    current_objects = {}
    for index, asset in enumerate(assets):
        source = metadata["source_object_geometry"][asset["object_asset"]]
        if asset["asset_id"] != index or any(asset[k] != source[k] for k in ("exact_scale", "collision_hash")):
            raise ValueError("v1.1 reference exact object identity differs from runtime asset")
        current_objects[asset["object_asset"]] = {"asset_id": index, "exact_scale": asset["exact_scale"],
                                                "collision_hash": asset["collision_hash"],
                                                "geometry_cache_sha256": asset["geometry_cache"]["sha256"]}
    selected = [value["clips"][e["clip_id"]] for e in motion.entries]
    if any(_clip_identity(r) != _clip_identity(e) for r, e in zip(selected, motion.entries, strict=True)):
        raise ValueError("v1.1 reused cache worker clip lengths/order/source mismatch")
    expected = torch.cat([e["expected"] for e in selected]).float().to(motion.device)
    valid = torch.cat([e["valid"] for e in selected]).bool().to(motion.device)
    if len(expected) != motion.num_frames:
        raise ValueError("v1.1 reused cache worker-local frame count mismatch")
    result = deepcopy(metadata)
    result.update(file_sha256=hashlib.sha256(raw).hexdigest(), content_sha256=value["content_sha256"],
                  selected_clips=[e["clip_id"] for e in motion.entries],
                  current_reward_geometry_binding={"used_to_regenerate_reference_labels": False,
                      "objects": current_objects, "hand_calibration_cache_sha256": hands_binding["sha256"],
                      "current_projected_palms": "unchanged TablesReward.hand_geometry contract"})
    return expected, valid, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "train-bundle", "canonical", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--eval-bundle", action="append", required=True)
    parser.add_argument("--helper-root")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    report = prepare_reused_cache(args.manifest, args.train_bundle, args.eval_bundle, args.canonical,
                                  args.output, helper_root=args.helper_root)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
