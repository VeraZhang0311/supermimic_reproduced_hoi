"""Authenticate unchanged native evidence across a compact dataset derivation.

Original reports keep their original asset IDs and bundle hashes.  This wrapper
validates those reports under their original contracts, proves the compact
asset/clip lineage, and separately requires a real zero-optimizer collection
audit of the compact train bundle.  It never relabels historical probes as new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hoi_compact_native_evidence_v1"
COLLECTION_SCHEMA = "hoi_weighted_collection_probe_v1"
SAMPLING_SCHEMA = "hoi_actual_transition_sampling_v1"


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _binding(path):
    value, sha = _read(path)
    return {"path": str(Path(path).resolve()), "sha256": sha}, value


def _bound(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("compact evidence requires an exact path/SHA binding")
    value, actual = _read(record["path"])
    if actual != record["sha256"]:
        raise ValueError("compact evidence bound file bytes changed: " + record["path"])
    return value


def _count_map(value, name):
    if (not isinstance(value, dict) or not value or
            any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in value.items())):
        raise ValueError(name + " must contain nonnegative integer counts")
    return value


def _validate_collection(report, bundle, required):
    if not isinstance(report, dict):
        raise ValueError("compact collection report must be an object")
    if (report.get("schema") != COLLECTION_SCHEMA or report.get("passed") is not True or
            report.get("optimizer_steps") != 0 or type(report.get("optimizer_steps")) is not int or
            report.get("formal_training_started") is not False):
        raise ValueError("compact collection must pass without starting training or an optimizer")
    from hoi.v1.config import json_identity
    source = report.get("source_before")
    if (not isinstance(source, dict) or set(source) != {"files", "content_sha256"}
            or not isinstance(source["files"], dict) or not source["files"]
            or any(not isinstance(k, str) or not k or not isinstance(v, str)
                   or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for k, v in source["files"].items())
            or source["content_sha256"] != json_identity(source["files"])):
        raise ValueError("compact collection source identity is missing or inconsistent")
    if (report.get("source_after") != source or report.get("source_unchanged") is not True
            or report.get("source_error")):
        raise ValueError("compact collection source changed during the native probe")
    bound_bundle = report.get("bundle", {})
    if bound_bundle.get("bundle_content_sha256") != bundle["bundle_content_sha256"]:
        raise ValueError("compact collection belongs to another bundle")
    file_bundle = _bound({k: bound_bundle[k] for k in ("path", "sha256")})
    if file_bundle != bundle:
        raise ValueError("compact collection bundle bytes differ from validated target")
    if report.get("asset_ids") != required:
        raise ValueError("compact collection must cover every required train asset exactly")
    n, steps = report.get("environments"), report.get("steps")
    if type(n) is not int or n < 1 or type(steps) is not int or steps < 64:
        raise ValueError("compact collection requires at least 64 real control ticks")
    sampling = report.get("sampling", {})
    if (sampling.get("schema") != SAMPLING_SCHEMA or sampling.get("allocation_match") is not True or
            type(sampling.get("total_transitions")) is not int or sampling["total_transitions"] != n * steps):
        raise ValueError("compact collection lacks verified actual transition proportions")
    for field in ("category_counts", "stratum_counts", "asset_counts"):
        if sum(_count_map(sampling.get(field), field).values()) != n * steps:
            raise ValueError("compact collection count totals differ: " + field)
    counts = sampling["asset_counts"]
    if set(counts) != {str(i) for i in required} or any(v <= 0 for v in counts.values()):
        raise ValueError("compact collection did not measure every required asset")
    # The normal production allocator supplies independent integer quotas.
    from hoi.v1.packed_probe import specs_from_header
    from hoi.v1.task_sampling import allocation_report
    specs = specs_from_header(bundle, n)
    expected = {str(s.asset_id): s.num_envs * steps for s in specs}
    if counts != expected:
        raise ValueError("actual compact asset transitions differ from production allocation")
    plan = allocation_report(bundle, specs)
    for observed, quota in (("category_counts", "category_env_counts"), ("stratum_counts", "stratum_env_counts")):
        if sampling[observed] != {str(k): v * steps for k, v in plan[quota].items()}:
            raise ValueError("actual compact " + observed + " differ from independently recomputed allocation")
    if (sampling.get("vector_steps") != steps or sampling.get("num_envs") != n or
            type(sampling.get("completed_episodes")) is not int or sampling["completed_episodes"] < 1):
        raise ValueError("compact collection must exercise at least one natural episode reset")
    exits = report.get("collector_exit_codes")
    if (report.get("shutdown_passed") is not True or report.get("shutdown_error") or report.get("error") or
            not isinstance(exits, list) or not exits or any(type(x) is not int or x != 0 for x in exits)):
        raise ValueError("compact native collectors did not close successfully")
    return {"environments": n, "steps": steps, "total_transitions": n * steps,
            "completed_episodes": sampling["completed_episodes"],
            "source_content_sha256": source["content_sha256"]}


def _validated_inputs(bundle_path):
    from hoi.data.compact_teacher import validated_compact_parents
    # validated_compact_parents reconstructs and compares the exact derivative,
    # including the content seal, while fully validating both original headers.
    # Avoid doing those large parent geometry validations twice in one gate.
    bundle, _ = _read(bundle_path)
    return bundle, validated_compact_parents(bundle)


def _validate_payload(payload, *, expected_physics, required_asset_ids,
                      expected_bundle_sha256, expected_asset_profiles):
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("unknown compact native evidence schema")
    if payload.get("original_reports_modified") is not False:
        raise ValueError("compact inheritance must preserve original native reports")
    target = _bound(payload["bundle"])
    # This is the authenticated derivation path, not a permissive hash alias.
    bundle, parents = _validated_inputs(payload["bundle"]["path"])
    if bundle != target or bundle.get("partition") != "train":
        raise ValueError("compact native evidence must bind the validated train bundle")
    if expected_bundle_sha256 is not None and bundle["bundle_content_sha256"] != expected_bundle_sha256:
        raise ValueError("compact native evidence belongs to another target bundle")
    required = sorted({int(c["asset_id"]) for c in bundle["clips"]})
    if required_asset_ids is not None and sorted(set(required_asset_ids)) != required:
        raise ValueError("requested compact coverage differs from the complete training partition")
    asset_registry = {int(a["asset_id"]): a for a in bundle["assets"]}
    if expected_asset_profiles is not None:
        actual = {i: a["physical_profile"] for i, a in asset_registry.items()}
        if actual != expected_asset_profiles:
            raise ValueError("compact physical profile registry differs from expected target")
    if set(parents) != {"train", "protected_eval"}:
        raise ValueError("compact native inheritance requires both authenticated parent partitions")
    bindings = payload.get("parent_evidence", {})
    if set(bindings) != set(parents):
        raise ValueError("compact native inheritance requires original evidence for both partitions")
    parent_gates, parent_reports = {}, {}
    from hoi.v1.gates import validate_native_evidence
    for partition, parent in parents.items():
        evidence = _bound(bindings[partition])
        if not isinstance(evidence, dict) or evidence.get("schema_version") == SCHEMA:
            raise ValueError("inheritance requires original native reports, not another wrapper")
        reports = evidence.get("reports", [evidence])
        if not isinstance(reports, list):
            raise ValueError("invalid original native report list")
        parent_required = sorted({c["asset_id"] for c in parent["clips"]})
        gate = validate_native_evidence(evidence, expected_physics=expected_physics,
            required_asset_ids=parent_required, expected_bundle_sha256=parent["bundle_content_sha256"],
            expected_asset_profiles={int(a["asset_id"]): a["physical_profile"] for a in parent["assets"]})
        if not gate["passed"]:
            raise ValueError("original " + partition + " native evidence failed: " + "; ".join(gate["blockers"]))
        parent_gates[partition] = gate
        parent_reports[partition] = {int(r.get("asset_id", r.get("spec", {}).get("asset_id"))): r for r in reports}
    mapping = bundle.get("compact_selection", {}).get("asset_id_map")
    if not isinstance(mapping, list) or len(mapping) != len(asset_registry):
        raise ValueError("compact native evidence lacks a complete asset identity map")
    inherited = []
    for item in mapping:
        new_id, old_id = item["asset_id"], item["original_asset_id"]
        asset = asset_registry[new_id]
        if asset.get("original_asset_id") != old_id or asset["native_equivalence_sha256"] != item["native_equivalence_sha256"]:
            raise ValueError("compact asset identity map differs from its registry")
        partition = "train" if old_id in parent_reports["train"] else "protected_eval"
        if old_id not in parent_reports[partition]:
            raise ValueError("compact asset lacks original accepted native evidence")
        parent_asset = next((a for a in parents[partition]["assets"] if a["asset_id"] == old_id), None)
        if parent_asset is None:
            raise ValueError("compact asset lacks its original native registry entry")
        # Compare every preserved native asset field, not just physical scalars.
        actual = {k: v for k, v in asset.items() if k not in ("asset_id", "original_asset_id")}
        original = {k: v for k, v in parent_asset.items() if k != "asset_id"}
        if actual != original:
            raise ValueError("compact native geometry/profile/cache identity differs from parent")
        report = parent_reports[partition][old_id]
        report_sha = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        inherited.append({"asset_id": new_id, "original_asset_id": old_id, "parent_partition": partition,
                          "native_equivalence_sha256": asset["native_equivalence_sha256"],
                          "original_report_content_sha256": report_sha})
    if len({r["asset_id"] for r in inherited}) != len(asset_registry):
        raise ValueError("duplicate compact native asset mapping")
    collection = _bound(payload["collection"])
    measured = _validate_collection(collection, bundle, required)
    return {"asset_ids": required, "inherited_registry": inherited, "parent_gates": parent_gates,
            "actual_collection": measured, "original_reports_modified": False,
            "scope": "unchanged native geometry/physics inherited from authenticated parents; compact sampling integration measured anew"}


def validate_compact_native_evidence(report_or_path, *, expected_physics=None, required_asset_ids=None,
                                     expected_bundle_sha256=None, expected_asset_profiles=None):
    """Return a fail-closed native gate result compatible with gates.py."""
    if isinstance(report_or_path, (str, Path)):
        try:
            payload, sha = _read(report_or_path)
        except (OSError, ValueError) as exc:
            return {"schema_version": "hoi_v1_native_gate_v2", "passed": False, "blockers": [str(exc)],
                    "evidence_sha256": None, "asset_ids": [], "status": "blocked"}
    else:
        payload = report_or_path
        sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result = {"schema_version": "hoi_v1_native_gate_v2", "passed": False, "blockers": [],
              "evidence_sha256": sha, "asset_ids": [], "status": "blocked"}
    try:
        result.update(_validate_payload(payload, expected_physics=expected_physics,
            required_asset_ids=required_asset_ids, expected_bundle_sha256=expected_bundle_sha256,
            expected_asset_profiles=expected_asset_profiles))
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, StopIteration) as exc:
        result["blockers"].append(str(exc))
    else:
        result.update(passed=True, status="passed")
    return result


def build_compact_native_evidence(bundle_path, train_evidence, protected_eval_evidence, collection_path):
    """Create a binding wrapper without copying or rewriting original reports."""
    bundle, _ = _binding(bundle_path)
    train, _ = _binding(train_evidence)
    evaluation, _ = _binding(protected_eval_evidence)
    collection, _ = _binding(collection_path)
    return {"schema_version": SCHEMA, "bundle": bundle,
            "parent_evidence": {"train": train, "protected_eval": evaluation},
            "collection": collection, "original_reports_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--parent-train-evidence", type=Path, required=True)
    parser.add_argument("--parent-eval-evidence", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_compact_native_evidence(args.bundle, args.parent_train_evidence,
        args.parent_eval_evidence, args.collection)
    gate = validate_compact_native_evidence(payload)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("native_evidence.json", payload), ("gate.json", gate)):
        (args.output / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": gate["passed"], "blockers": gate["blockers"], "output": str(args.output)}))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
