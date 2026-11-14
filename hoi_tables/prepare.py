"""Prepare the user-selected tables as a frozen 50 Hz hw3 reference bundle."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from supermimic.robot_metadata import G1_JOINT_NAMES
from .assets import prepare_assets
from .data import (ASSETS,CACHE_SCHEMA,SCHEMA,TIMEBASE,adapt_row,assign_partitions,
                   content_hash,read_manifest,read_selection,sha256_file,trajectory_hash)

DEFAULT_SOURCE_ROOT = "/mnt/projects/ll/tianzizh/hoi_corpus/sources/hoi_retarget/b5ac8125e497e6735d6314e63a0e1eedf6229336"
DEFAULT_ASSET_BUNDLE = "/mnt/projects/ll/tianzizh/hoi_compact_teacher_20261101_01/dataset_04/bundle_train/bundle.json"


def prepare(selection_path, source_root, output, *, source_asset_bundle=DEFAULT_ASSET_BUNDLE,
            references_only=False):
    import pyarrow.parquet as pq

    selection_path,source_root,output = map(Path,(selection_path,source_root,output))
    raw_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    roles, actions = None, {}
    if raw_selection.get("schema") == "hoi_ultra_tables_selection_v1":
        train, protected = raw_selection["train"], raw_selection["protected_eval"]
        if set(train) & set(protected) or len(train) != 203 or len(protected) != 20:
            raise ValueError("Mixed tables selection must bind disjoint train203 and protected20")
        actions = train | protected
        roles = {name: "train" if name in train else "protected_eval" for name in actions}
        selection = {"included": sorted(actions), "excluded": {}, "split": raw_selection["split"],
                     "source_revision": raw_selection["source"]["revision"],
                     "workbook_sha256": raw_selection["workbook"]["sha256"]}
    else:
        selection = read_selection(selection_path)
    source = source_root/"data/omomo/unitree_g1/train-00000-of-00001.parquet"
    source_sha = sha256_file(source)
    if roles is not None and source_sha != raw_selection["source"]["sha256"]:
        raise ValueError("Mixed tables source Parquet changed")
    fields = ["clip_id","subject","sequence_id","dataset","robot","schema_version","object","fps","n_frames",
              "dof_names","root_pos","root_rot","dof_pos","object_pos","object_rot","object_model_path",
              "object_scale","object_aug_scale","scale_object_mesh","scale_object_trajectory",
              "qc_pass","qc_flags","contact_link_names","per_link_contact_flags"]
    rows = pq.read_table(source,columns=fields,filters=[("subject","in",selection["included"])]).to_pylist()
    lookup = {r["subject"]:r for r in rows}
    if len(rows) != len(lookup) or set(lookup) != set(selection["included"]):
        raise ValueError("source does not uniquely match the exact user whitelist")
    rows = [lookup[name] for name in selection["included"]]
    for row in rows:
        name = row["subject"]
        if row["dataset"] != "omomo" or row["robot"] != "unitree_g1" or row["schema_version"] != 2 or row["clip_id"] != "omomo/unitree_g1/"+name:
            raise ValueError("unsupported table source schema/identity")
        if row["object"] != name.split("_")[1] or row["object_model_path"] != f"assets/objects/{row['object']}.urdf":
            raise ValueError("table object identity differs from the whitelist")
        if abs(row["object_scale"]-.83) > 1e-6 or row["object_aug_scale"] != 1. or not row["scale_object_mesh"] or not row["scale_object_trajectory"]:
            raise ValueError("unverified table scale convention")
        if not row["qc_pass"] or row["qc_flags"]:
            raise ValueError("QC-flagged table unexpectedly remains after user exclusions")
    if roles is None:
        roles = assign_partitions(rows, fraction=selection["split"]["protected_fraction"], seed=selection["split"]["seed"])
    else:
        for asset in ASSETS:
            actual = Counter(actions[name] for name in protected if name.split("_")[1] == asset)
            if actual != Counter(selection["split"]["protected_quotas"][asset]) or sum(actual.values()) != 10:
                raise ValueError("Mixed tables protected action quotas differ")
    output.mkdir(parents=True,exist_ok=False)
    cache = {"schema":CACHE_SCHEMA,"timebase":TIMEBASE,"clips":{}}
    clips = []
    for row in rows:
        value = adapt_row(row)
        cache["clips"][row["clip_id"]] = value
        role = roles[row["subject"]]
        clips.append({"clip_id":row["clip_id"],"subject":row["subject"],"source_group":row["sequence_id"],
                      "object":row["object"],"object_asset":row["object"],"asset_id":ASSETS.index(row["object"]),
                      "partition":role,"role":role,"source_frames":row["n_frames"],
                      "source_trajectory_sha256":trajectory_hash(row),"state_frames":len(value["states"]),
                      "control_steps":len(value["states"])-1,"source_contact_links":row["contact_link_names"],
                      "source_scale":row["object_scale"],"source_qc_flags":row["qc_flags"]})
        if actions:
            clips[-1]["action"] = actions[row["subject"]]
    cache_path = output/"reference_50hz.pt"
    torch.save(cache,cache_path)
    if sha256_file(source) != source_sha: raise RuntimeError("source parquet changed during preparation")
    manifest = {"schema":SCHEMA,"profile":"ultra_tables_extension_v1" if actions else "user_tables_v1",
                "source_revision":selection["source_revision"],
                "source":{"path":str(source.resolve()),"sha256":source_sha},"timebase":TIMEBASE,
                "cache":{"path":"reference_50hz.pt","sha256":sha256_file(cache_path)},
                "selection":{"path":str(selection_path.resolve()),"sha256":sha256_file(selection_path),
                             "workbook_sha256":selection["workbook_sha256"],"included_count":len(clips),"excluded":selection["excluded"]},
                "target_joint_names":list(G1_JOINT_NAMES),"object_assets":list(ASSETS),"clips":clips,
                "split":selection["split"],"partitions":{role:dict(Counter(e["object_asset"] for e in clips if e["partition"] == role)) for role in ("train","protected_eval")},
                "protection":{"optimizer_partition":"train","source_groups_disjoint":True,
                              "duplicate_trajectories_disjoint":True,"protected_eval_updates":False}}
    manifest["content_sha256"] = content_hash(manifest)
    manifest_path = output/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    read_manifest(manifest_path)
    report = {"schema":"hoi_tables_preparation_report_v1","passed":True,"clips":len(clips),
              "partitions":manifest["partitions"],"manifest_sha256":sha256_file(manifest_path),
              "cache_sha256":manifest["cache"]["sha256"],"timebase":TIMEBASE,"native_scene_validated":False}
    if not references_only:
        catalog = prepare_assets(source_asset_bundle, output)
        report.update(catalog_content_sha256=catalog["catalog_content_sha256"],
                      physical_profiles={a["object_asset"]:a["physical_profile"] for a in catalog["assets"]})
    (output/"preparation_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",default="configs/hoi_tables_selection.json")
    parser.add_argument("--source-root",default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-asset-bundle",default=DEFAULT_ASSET_BUNDLE)
    parser.add_argument("--output",required=True)
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.source_root, args.output, source_asset_bundle=args.source_asset_bundle,
                             references_only=args.references_only), indent=2))


if __name__ == "__main__":
    main()
