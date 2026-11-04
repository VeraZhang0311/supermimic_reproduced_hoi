"""Select the historical canonical 80 from an intact curated evaluation bank."""
from collections import Counter
import json
from pathlib import Path

from hoi.data.manifest import sha256_file
from .prepare import _digest_json

CANONICAL_CONTENT_SHA256 = "adc9875cbe823546efc151b1ad9d07589a71808636675687ee549c0f790e0715"


def map_canonical_cases(training, evaluation, canonical):
    """Match original motion bytes and full duration, never just a filename."""
    if training["partition"] != "train" or evaluation["partition"] != "protected_eval":
        raise ValueError("HOI80 requires train and protected_eval partitions")
    old = canonical["clips"]
    if len(old) != 80 or len({c["name"] for c in old}) != 80:
        raise ValueError("canonical HOI80 must contain 80 unique motions")
    lookup = {(c["dataset"], c["clip_id"]): (i, c) for i, c in enumerate(evaluation["clips"])}
    if len(lookup) != len(evaluation["clips"]):
        raise ValueError("duplicate evaluation clips")
    train_keys = {(c["dataset"], c["clip_id"]) for c in training["clips"]}
    train_groups = {c["source_group"] for c in training["clips"]}
    selected, rows = [], []
    for source in old:
        key = ("ultra", source["name"])
        if key not in lookup:
            raise ValueError("canonical motion missing from protected evaluation")
        i, clip = lookup[key]
        if key in train_keys or clip["source_group"] in train_groups:
            raise ValueError("canonical HOI80 motion/source group leaked into train")
        provenance = clip["source_provenance"]
        if (not provenance.get("protected_eval80") or
                provenance.get("motion_sha256") != source["source_motion_sha256"] or
                any(clip[k] != source[k] for k in ("source_frames", "state_frames"))):
            raise ValueError("canonical motion identity/duration differs")
        selected.append(i)
        rows.append({"bank_clip_id": i, "clip_id": clip["clip_id"],
                     "canonical_case_id": provenance["id"], "object": provenance["object"],
                     "asset_id": clip["asset_id"], "source_motion_sha256": provenance["motion_sha256"],
                     "state_frames": clip["state_frames"]})
    if Counter(r["object"] for r in rows) != {k: 20 for k in ("largebox", "smallbox", "plasticbox", "suitcase")}:
        raise ValueError("HOI80 requires 20 cases per canonical object class")
    if len({r["canonical_case_id"] for r in rows}) != 80:
        raise ValueError("duplicate canonical case IDs")
    return selected, rows


def select_canonical_hoi80(training, evaluation, path):
    canonical = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = canonical.pop("bundle_content_sha256", None)
    if digest != CANONICAL_CONTENT_SHA256 or _digest_json(canonical) != digest:
        raise ValueError("canonical HOI80 registry identity changed")
    indices, rows = map_canonical_cases(training, evaluation, canonical)
    return indices, {"name": "canonical_hoi80", "registry_path": str(Path(path).resolve()),
                     "registry_sha256": sha256_file(path), "registry_content_sha256": digest,
                     "parent_bundle_content_sha256": evaluation["bundle_content_sha256"],
                     "selected_motions": 80, "available_motions": len(evaluation["clips"]),
                     "is_subset": True, "full_duration": True, "cases": rows}
