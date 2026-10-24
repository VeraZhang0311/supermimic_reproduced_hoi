"""Merge native object exports under one immutable, calibrated hand overlay.

Only the master contributes robot/hands. Other probes contribute their object
records; independently cooked hands can never silently replace the master.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from hoi.data.manifest import read_manifest, sha256_file


def merge_recipes(manifest_path, master_path, object_paths, output):
    manifest = read_manifest(manifest_path)
    required = list(dict.fromkeys(entry["object_asset"] for entry in manifest["partitions"]["train"]))
    paths = [Path(master_path).resolve()] + [Path(path).resolve() for path in object_paths]
    records, provenance = {}, []
    master = None
    for path in paths:
        recipe = json.loads(path.read_text(encoding="utf-8"))
        if master is None:
            master = recipe
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
        for asset in recipe["assets"]:
            name = asset["object_asset"]
            if name in records and records[name] != asset:
                raise ValueError(f"ambiguous repeated compiled export for {name}; choose one immutable object export")
            native = Path(asset["native_asset"])
            if not native.is_absolute() or sha256_file(native) != asset["native_asset_sha256"]:
                raise ValueError("native object exports require absolute, byte-verified USD paths")
            records[name] = deepcopy(asset)
    missing, extra = set(required) - records.keys(), records.keys() - set(required)
    if missing or extra:
        raise ValueError(f"object exports must match the frozen train assets exactly: missing={sorted(missing)}, extra={sorted(extra)}")
    for side in ("left", "right"):
        hand = master["hands"][side]
        if hand.get("palm_selection_verified") is not True or not hand.get("palm_selection_evidence"):
            raise ValueError("the master must contain calibrated, traceable palm probes")
    native_robot = master.get("native_robot_asset")
    if native_robot is None:
        receipt = master["hand_runtime_report"]
        native_robot = {"path": receipt["usd"], "sha256": receipt["usd_sha256"]}
    if not Path(native_robot["path"]).is_absolute() or sha256_file(native_robot["path"]) != native_robot["sha256"]:
        raise ValueError("master native robot/hand overlay is not byte-verified")
    result = {"schema": "hoi_v1_master_hand_multi_object_recipe_v1", "assets": [records[name] for name in required],
              "hands": deepcopy(master["hands"]), "native_robot_asset": deepcopy(native_robot),
              "merge_sources": {"master": provenance[0], "object_exports": provenance,
                                "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": sha256_file(manifest_path)},
                                "asset_order": "first occurrence in the frozen train manifest",
                                "hand_rule": "only master robot/hands; nonmaster hand exports ignored"}}
    if master.get("reference_kinematics") is not None:
        result["reference_kinematics"] = deepcopy(master["reference_kinematics"])
    if master.get("hand_runtime_report") is not None:
        result["hand_runtime_report"] = deepcopy(master["hand_runtime_report"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--master", required=True, help="native export defining the one frozen robot/hand overlay")
    parser.add_argument("--objects", nargs="+", required=True, help="remaining native object-export assets.json files")
    parser.add_argument("--output", required=True, help="new merged recipe file")
    args = parser.parse_args(argv)
    result = merge_recipes(args.manifest, args.master, args.objects, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "assets": [a["object_asset"] for a in result["assets"]],
                      "native_robot_asset": result["native_robot_asset"], "training_started": False}))


if __name__ == "__main__":
    main()
