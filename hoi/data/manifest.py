"""Audit source identities and split membership without loading motion tensors.

The manifest describes a data adapter, not verified native physics or training
readiness. Missing source-joint metadata and training selection remain explicit.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .timebase import CanonicalTimebase


SCHEMA_VERSION = "supermimic_hoi_data_manifest_v1"
PARTITIONS = ("train", "validation", "regression", "test")
IDENTITY_FIELDS = ("motion", "motion_sha256", "source_group", "object", "object_asset")
CANONICAL_HASHES = {
    "splits": "7d66cd15b27b41988ac3ef761fb82f15204b5ab741a696f95102801dac8b62e2",
    "eval80": "459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002",
    "test40": "b7e61757f230930fe18672f619149d7201ccdb6ab7a6ac06de0ae1634cf7674d",
    "train_selection": "8c141eb60650b9df79b92acfeeea785f796c7f18f52bf179467562c7cc315b23",
}
CANONICAL_COUNTS = {
    "train": (2651, 381), "validation": (517, 80),
    "regression": (655, 100), "test": (243, 40),
}
CANONICAL_MEMBERSHIP_SHA256 = "8e3fedaab59f3f75c9890175b6b77eb0e76d71358902898d7d2dde61dfa890c8"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def content_sha256(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _load_input(path, role, profile):
    path = Path(path)
    # Hash the exact bytes being decoded, rather than hashing a second read.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if profile == "canonical-v4" and role in CANONICAL_HASHES:
        if digest != CANONICAL_HASHES[role]:
            raise ValueError(f"{role}: frozen input SHA256 mismatch: {path}")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{role}: expected a JSON object")
    return value, {"path": str(path.resolve()), "sha256": digest}


def _source_path(path):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("motion path must be a nonempty absolute path")
    normal = path.replace("\\", "/")
    if ".." in PurePosixPath(normal).parts:
        raise ValueError("parent traversal is not permitted in source paths")
    if not (PurePosixPath(normal).is_absolute() or PureWindowsPath(path).is_absolute()):
        raise ValueError(f"source motion path must be absolute: {path}")
    return normal


def source_group_from_path(path):
    name = PurePosixPath(_source_path(path)).name
    if not name.endswith(".pt") or len(name[:-3].split("_")) < 3:
        raise ValueError(f"not an ULTRA motion filename: {name}")
    return "_".join(name[:-3].split("_")[:3])


def resolve_motion_path(source_path, path_maps=()):
    """Resolve an explicit source-prefix remap without changing source identity.

    Matching is by whole path components, choosing the longest prefix. Remaps
    never search by basename, so missing or ambiguous data cannot be substituted.
    """
    normal = _source_path(source_path)
    candidates = []
    for prefix, local_root in path_maps:
        prefix = _source_path(str(prefix)).rstrip("/")
        if normal == prefix or normal.startswith(prefix + "/"):
            candidates.append((len(prefix), prefix, Path(local_root).resolve()))
    if candidates:
        longest = max(item[0] for item in candidates)
        matches = [item for item in candidates if item[0] == longest]
        if len({str(item[2]) for item in matches}) != 1:
            raise ValueError("ambiguous path remap for motion")
        _, prefix, root = matches[0]
        suffix = normal[len(prefix):].lstrip("/")
        target = root.joinpath(*PurePosixPath(suffix).parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError("resolved motion escapes remap root")
        return target
    path = Path(source_path)
    if not path.is_absolute():
        raise ValueError(f"source path is not native on this platform; supply --path-map: {source_path}")
    return path.resolve()


def robot_joint_names():
    """Read the existing robot order without importing IsaacLab or its config."""
    path = Path(__file__).resolve().parents[2] / "supermimic" / "robot.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "G1_JOINT_NAMES" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("G1_JOINT_NAMES was not found in robot.py")


def _entry(value):
    if not isinstance(value, dict):
        raise ValueError("motion entry must be an object")
    result = deepcopy(value)
    for key in IDENTITY_FIELDS:
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"motion entry requires string {key}")
    _source_path(result["motion"])
    if not re.fullmatch(r"[0-9a-f]{64}", result["motion_sha256"]):
        raise ValueError("motion_sha256 must be 64 lowercase hexadecimal characters")
    if result["source_group"] != source_group_from_path(result["motion"]):
        raise ValueError("source_group does not match the original motion filename")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result["object_asset"]):
        raise ValueError("object_asset must be an asset identifier, not a path")
    if "source_frames" in result and (
        type(result["source_frames"]) is not int or result["source_frames"] < 2
    ):
        raise ValueError("source_frames must be an integer >= 2")
    return result


def _partitions(value, profile):
    if not isinstance(value, dict) or set(value) != set(PARTITIONS):
        raise ValueError(f"partitions must contain exactly {PARTITIONS}")
    result, counts = {}, {}
    seen_paths, seen_hashes, seen_names, group_partition = set(), set(), set(), {}
    for partition in PARTITIONS:
        if not isinstance(value[partition], list) or not value[partition]:
            raise ValueError(f"{partition}: expected a nonempty list of motions")
        entries = [_entry(item) for item in value[partition]]
        groups = set()
        for entry in entries:
            path = _source_path(entry["motion"])
            name = PurePosixPath(path).name
            digest, group = entry["motion_sha256"], entry["source_group"]
            if path in seen_paths or digest in seen_hashes or name in seen_names:
                raise ValueError("duplicate motion path, SHA256 or basename across the split manifest")
            if group in group_partition and group_partition[group] != partition:
                raise ValueError(f"source-group leakage: {group}")
            seen_paths.add(path)
            seen_hashes.add(digest)
            seen_names.add(name)
            group_partition[group] = partition
            groups.add(group)
        counts[partition] = {"motions": len(entries), "source_groups": len(groups)}
        if profile == "canonical-v4" and (len(entries), len(groups)) != CANONICAL_COUNTS[partition]:
            raise ValueError(f"{partition}: canonical-v4 counts differ")
        result[partition] = entries
    return result, counts


def _identity(entry):
    return tuple(entry[key] for key in IDENTITY_FIELDS)


def membership_sha256(partitions, panels):
    payload = {
        "partitions": {name: sorted(list(_identity(entry)) for entry in entries)
                       for name, entries in partitions.items()},
        "panels": {name: sorted([case["id"], *_identity(case), case["source_frames"]] for case in cases)
                   for name, cases in panels.items()},
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _panel(value, entries, name, expected_count=None):
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list) or not value["cases"]:
        raise ValueError(f"{name}: expected nonempty cases list")
    cases, identifiers, groups = [], set(), set()
    allowed = {_identity(entry) for entry in entries}
    for raw in value["cases"]:
        case = _entry(raw)
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"{name}: missing or duplicate case id")
        if case["source_group"] in groups:
            raise ValueError(f"{name}: expected one canonical case per source group")
        if _identity(case) not in allowed:
            raise ValueError(f"{name}: case identity does not match its held-out partition")
        if "source_frames" not in case:
            raise ValueError(f"{name}: case must declare source_frames")
        identifiers.add(identifier)
        groups.add(case["source_group"])
        cases.append(case)
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"{name}: canonical case count differs")
    return cases


def _selection(value, train):
    for flag in ("training_only", "validation_source_groups_disjoint"):
        if flag in value and value[flag] is not True:
            raise ValueError(f"train_selection: {flag} must be true")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("train_selection: expected nonempty cases list")
    identities = []
    for case in cases:
        if not isinstance(case, dict) or case.get("partition") != "train":
            raise ValueError("train_selection: every case must have partition=train")
        identities.append(_identity(_entry(case)))
    if len(set(identities)) != len(identities):
        raise ValueError("train_selection: duplicate identities")
    if set(identities) != {_identity(entry) for entry in train}:
        raise ValueError("train_selection does not exactly match the frozen train partition")


def _bindings(value):
    target = robot_joint_names()
    if value is None:
        return {"source_joint_names": None, "target_joint_names": target,
                "joint_order_evidence": {"status": "not_provided"}, "objects": {}}
    source = value.get("source_joint_names")
    if (not isinstance(source, list) or len(source) != 29 or
        any(not isinstance(name, str) for name in source) or
        len(set(source)) != 29 or set(source) != set(target)):
        raise ValueError("bindings.source_joint_names must name each of the existing 29 G1 joints once")
    if "target_joint_names" in value and value["target_joint_names"] != target:
        raise ValueError("target_joint_names must preserve the existing SuperMimic robot order")
    evidence, objects = value.get("joint_order_evidence", {}), value.get("objects", {})
    if not isinstance(evidence, dict) or not isinstance(objects, dict):
        raise ValueError("joint_order_evidence and objects must be JSON objects")
    return {**deepcopy(value), "source_joint_names": source, "target_joint_names": target,
            "joint_order_evidence": evidence or {"status": "declared_not_native_verified"},
            "objects": objects}


def build_manifest(split_path, eval_path, test_path, *, train_selection_path=None,
                   bindings_path=None, path_maps=(), profile="custom", run_eval_cases_path=None):
    """Build a self-contained audit, even when source tensor files are unavailable.

    canonical-v4 pins the known local split/panel bytes and, when provided, the
    cluster training selection. A missing selection or joint mapping is explicit;
    this API never guesses source joint order from a runtime robot configuration.
    """
    if profile not in ("canonical-v4", "custom"):
        raise ValueError("profile must be canonical-v4 or custom")
    splits, split_identity = _load_input(split_path, "splits", profile)
    evaluation, eval_identity = _load_input(eval_path, "eval80", profile)
    test, test_identity = _load_input(test_path, "test40", profile)
    partitions, counts = _partitions(splits.get("partitions"), profile)
    for name, field in (("group_counts", "source_groups"), ("variant_counts", "motions")):
        if name in splits and splits[name] != {part: count[field] for part, count in counts.items()}:
            raise ValueError(f"declared {name} disagrees with split entries")
    panels = {
        "eval80": _panel(evaluation, partitions["validation"], "eval80", 80 if profile == "canonical-v4" else None),
        "test40": _panel(test, partitions["test"], "test40", 40 if profile == "canonical-v4" else None),
    }
    inputs = {"splits": split_identity, "eval80": eval_identity, "test40": test_identity}
    if train_selection_path is not None:
        selection, inputs["train_selection"] = _load_input(train_selection_path, "train_selection", profile)
        _selection(selection, partitions["train"])
    if run_eval_cases_path is not None:
        run_eval, inputs["run_eval80"] = _load_input(run_eval_cases_path, "run_eval80", profile)
        cases = _panel(run_eval, partitions["validation"], "run_eval80")
        compare = lambda items: {case["id"]: (_identity(case), case["source_frames"]) for case in items}
        if compare(cases) != compare(panels["eval80"]):
            raise ValueError("run_eval80 does not match canonical eval80 case identities and frame counts")
    binding_input = None
    if bindings_path is not None:
        binding_input, inputs["bindings"] = _load_input(bindings_path, "bindings", profile)
    bindings = _bindings(binding_input)
    mappings = []
    for prefix, root in path_maps:
        mappings.append([_source_path(str(prefix)).rstrip("/"), str(Path(root).resolve())])
    # Check maps now, including conflicting roots, without probing the data files.
    for mapping in mappings:
        resolve_motion_path(mapping[0] + "/probe.pt", mappings)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "dataset_name": "OMOMO_retarget_aug_v4" if profile == "canonical-v4" else "custom_ULTRA_format",
        "timebase": CanonicalTimebase().metadata(),
        "inputs": inputs,
        "split_rule": "first_three_filename_fields_all_variants_together",
        "source_scope": splits.get("scope", "unspecified; pretraining coverage not established"),
        "counts": counts,
        "partitions": partitions,
        "panels": panels,
        "membership_sha256": membership_sha256(partitions, panels),
        "bindings": bindings,
        "path_maps": mappings,
        "validation": {
            "split_membership": "verified_json_identities_and_disjoint_source_groups",
            "train_selection": "verified_exact_train_identities" if train_selection_path else "not_provided",
            "run_eval80": "verified_exact_canonical_identities" if run_eval_cases_path else "not_provided",
            "motion_bytes": "not_read",
            "adapter_ready": bindings["source_joint_names"] is not None,
            "training_ready": False,
            "native_physics": "not_checked",
            "source_timebase": "protocol_assumed_not_generation_verified",
            "source_velocity_reference_point": "unverified",
            "asset_scale_transform": "declared_bindings_only_not_native_verified",
        },
    }
    manifest["manifest_content_sha256"] = content_sha256(manifest)
    return manifest


def write_manifest(manifest, output_file):
    """Exclusive create: never overwrite a previous manifest or source input."""
    output_file = Path(output_file)
    if manifest.get("manifest_content_sha256") != content_sha256(manifest):
        raise ValueError("manifest content SHA256 mismatch before write")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_manifest(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a SuperMimic HOI v1 data manifest")
    if value.get("manifest_content_sha256") != content_sha256(value):
        raise ValueError("manifest content SHA256 mismatch")
    profile = value.get("profile")
    if profile not in ("custom", "canonical-v4"):
        raise ValueError("unknown manifest profile")
    if profile == "canonical-v4":
        for role, expected in CANONICAL_HASHES.items():
            identity = value.get("inputs", {}).get(role)
            if role == "train_selection" and identity is None:
                continue
            if not isinstance(identity, dict) or identity.get("sha256") != expected:
                raise ValueError(f"{role}: canonical input SHA256 pin mismatch")
    partitions, counts = _partitions(value.get("partitions"), profile)
    if counts != value.get("counts"):
        raise ValueError("manifest counts disagree with entries")
    for name, partition, size in (("eval80", "validation", 80), ("test40", "test", 40)):
        _panel({"cases": value["panels"][name]}, partitions[partition], name,
               size if profile == "canonical-v4" else None)
    membership = membership_sha256(partitions, value["panels"])
    if value.get("membership_sha256") != membership:
        raise ValueError("manifest membership SHA256 mismatch")
    if profile == "canonical-v4" and membership != CANONICAL_MEMBERSHIP_SHA256:
        raise ValueError("canonical-v4 motion/panel membership pin mismatch")
    _bindings(value.get("bindings") if value.get("bindings", {}).get("source_joint_names") is not None else None)
    if value.get("timebase") != CanonicalTimebase().metadata():
        raise ValueError("unsupported timebase or velocity convention")
    validation = value.get("validation", {})
    ready = value["bindings"].get("source_joint_names") is not None
    if validation.get("adapter_ready") is not ready or validation.get("training_ready") is not False:
        raise ValueError("manifest readiness does not match this data-only stage")
    selection_status = "verified_exact_train_identities" if "train_selection" in value["inputs"] else "not_provided"
    if validation.get("train_selection") != selection_status:
        raise ValueError("manifest selection status disagrees with input provenance")
    return value
