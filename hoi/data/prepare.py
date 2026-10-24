"""CPU-only HOI v1 manifest construction and canonical 60->50 reference audit.

Examples and source-binding evidence requirements are in hoi/data/README.md.
No command in this module loads a policy, imports IsaacLab, or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import sys

from .manifest import (PARTITIONS, build_manifest, read_manifest, resolve_motion_path,
                       sha256_file, write_manifest)
from .timebase import CanonicalTimebase


def _path_map(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("path maps must be SOURCE_PREFIX=LOCAL_ROOT")
    prefix, root = value.split("=", 1)
    if not prefix or not root:
        raise argparse.ArgumentTypeError("both path-map sides must be nonempty")
    return prefix, root


def _fresh_output(path, *, protected_files=(), source_roots=()):
    output = Path(path).resolve()
    for protected in protected_files:
        protected = Path(protected).resolve()
        if protected == output or protected.is_relative_to(output):
            raise ValueError("output directory must not contain an input/source file")
    for root in source_roots:
        root = Path(root).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("output directory must be separate from remapped source data")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _selected_entries(manifest, partition):
    if partition in PARTITIONS:
        return manifest["partitions"][partition]
    return manifest["panels"][partition]


def _assert_adapted(adapted, source_frames):
    """Validate the serialized state/control boundary, not just one frame count."""
    import torch

    clock = CanonicalTimebase()
    steps, states = clock.control_steps(source_frames), clock.state_frames(source_frames)
    meta = adapted["metadata"]
    if (meta["source_frames"], meta["control_steps"], meta["state_frames"]) != (source_frames, steps, states):
        raise ValueError("adapted frame metadata disagrees with the integer timebase")
    times = adapted["time"]
    if times.dtype != torch.float64 or times.shape != (states,):
        raise ValueError("state times must be float64 with shape T+1")
    expected_times = torch.arange(states, dtype=torch.float64) / 50
    torch.testing.assert_close(times, expected_times, rtol=0, atol=1e-12)
    torch.testing.assert_close(adapted["source_index"], expected_times * 60, rtol=0, atol=1e-10)
    expected_shapes = {
        ("robot", "root_pos"): (states, 3), ("robot", "root_quat_xyzw"): (states, 4),
        ("robot", "joint_pos"): (states, 29), ("robot", "derived_ref_v"): (states, 3),
        ("robot", "derived_ref_omega"): (states, 3),
        ("robot", "derived_ref_joint_qd"): (states, 29),
        ("object", "pos"): (states, 3), ("object", "quat_xyzw"): (states, 4),
        ("object", "derived_ref_v"): (states, 3), ("object", "derived_ref_omega"): (states, 3),
        ("body", "pos"): (states, 39, 3), ("body", "quat_xyzw"): (states, 39, 4),
        ("body", "contact_flags"): (states, 39),
    }
    for (block, field), shape in expected_shapes.items():
        if tuple(adapted[block][field].shape) != shape:
            raise ValueError(f"wrong adapted shape: {block}.{field}")

    def check_tensors(value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("adapted cache contains non-finite values")
        elif isinstance(value, dict):
            for item in value.values():
                check_tensors(item)

    check_tensors(adapted)
    if not 0 <= meta["tail_remainder_s"] < clock.control_dt_s:
        raise ValueError("canonical tail must be less than one complete control interval")


def _velocity_diagnostics(adapted):
    import torch

    result = {"interpretation": "raw_numeric_difference_only_source_reference_point_unverified"}
    for block in ("robot", "object"):
        result[block] = {}
        for name in ("v", "omega"):
            difference = adapted[block][f"source_{name}"] - adapted[block][f"derived_ref_{name}"]
            result[block][name] = {
                "rms_component_difference": float(torch.sqrt(difference.double().square().mean())),
                "max_abs_component_difference": float(difference.abs().max()),
            }
    return result


def validate_dataset(manifest_path, output, *, partition="train", limit=None,
                     path_maps=(), cache=False):
    """Hash/decode/adapt selected source tensors and write a new audit directory.

    A limited audit is explicitly a subset. Missing files fail the audit rather
    than being skipped. Native physics, contact pairs and source FPS are not
    validated by this CPU stage.
    """
    import torch

    from .reference import adapt_motion

    manifest = read_manifest(manifest_path)
    if partition not in (*PARTITIONS, "eval80", "test40"):
        raise ValueError("unknown partition or panel")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    bindings = manifest["bindings"]
    if bindings.get("source_joint_names") is None:
        raise ValueError("source joint order is not bound; supply --bindings when building the manifest")
    entries = _selected_entries(manifest, partition)
    selected = entries[:limit] if limit is not None else entries
    mappings = list(path_maps) if path_maps else manifest["path_maps"]
    # Resolve before creating output: never risk emitting a cache into source data.
    resolved = [resolve_motion_path(entry["motion"], mappings) for entry in selected]
    input_files = [manifest_path, *resolved]
    output = _fresh_output(output, protected_files=input_files,
                           source_roots=[root for _, root in mappings] + [path.parent for path in resolved])
    report = {
        "schema_version": "supermimic_hoi_data_validation_v1",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "profile": manifest["profile"],
        "timebase": manifest["timebase"],
        "scope": {"partition": partition, "available_motions": len(entries),
                  "selected_motions": len(selected), "is_subset": len(selected) != len(entries)},
        "effective_path_maps": [[str(a), str(b)] for a, b in mappings],
        "joint_order_evidence": bindings["joint_order_evidence"],
        "native_physics": "not_run",
        "training_started": False,
        "success": False,
        "motion_audits": [],
        "errors": [],
    }
    expected_frames = {}
    for cases in manifest["panels"].values():
        expected_frames.update({case["motion_sha256"]: case["source_frames"] for case in cases})
    for entry, path in zip(selected, resolved):
        try:
            # Decode exactly the bytes that were hashed; never use unrestricted pickle.
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != entry["motion_sha256"]:
                raise ValueError("source motion SHA256 mismatch")
            frames = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if not isinstance(frames, torch.Tensor):
                raise ValueError("source file is not a plain ULTRA motion Tensor; checkpoint/rollout rejected")
            expected = entry.get("source_frames", expected_frames.get(digest))
            if expected is not None and len(frames) != expected:
                raise ValueError("source frame count disagrees with the canonical case manifest")
            adapted = adapt_motion(frames, bindings["source_joint_names"], bindings["target_joint_names"])
            _assert_adapted(adapted, len(frames))
            audit = {
                **{key: entry[key] for key in ("motion", "motion_sha256", "source_group", "object", "object_asset")},
                "resolved_motion_path": str(path),
                "source_bytes": len(raw),
                "metadata": adapted["metadata"],
                "velocity_diagnostics": _velocity_diagnostics(adapted),
                "status": "source_bytes_schema_and_control_grid_verified",
            }
            if cache:
                cache_path = output / f"reference_{digest}.pt"
                adapted["provenance"] = {
                    "source_motion": entry["motion"], "source_motion_sha256": digest,
                    "manifest_content_sha256": manifest["manifest_content_sha256"],
                    "manifest_file_sha256": report["manifest_file_sha256"],
                    "object_asset": entry["object_asset"], "source_group": entry["source_group"],
                }
                with cache_path.open("xb") as stream:
                    torch.save(adapted, stream)
                audit["cache"] = {"path": str(cache_path), "sha256": sha256_file(cache_path)}
            report["motion_audits"].append(audit)
        except (OSError, ValueError, TypeError, RuntimeError, AssertionError, pickle.UnpicklingError, EOFError) as error:
            report["errors"].append({"motion": entry["motion"], "error": str(error)})
            # Stop on the first bad source: do not silently select easier examples.
            break
    report["success"] = not report["errors"] and len(report["motion_audits"]) == len(selected)
    report["validated_motions"] = len(report["motion_audits"])
    _write_json(output / "validation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subcommands.add_parser("manifest", help="audit frozen JSON identities without loading Torch")
    manifest_parser.add_argument("--splits", type=Path, required=True)
    manifest_parser.add_argument("--eval-cases", type=Path, required=True)
    manifest_parser.add_argument("--test-cases", type=Path, required=True)
    manifest_parser.add_argument("--train-selection", type=Path)
    manifest_parser.add_argument("--run-eval-cases", type=Path)
    manifest_parser.add_argument("--bindings", type=Path)
    manifest_parser.add_argument("--profile", choices=("canonical-v4", "custom"), default="canonical-v4")
    manifest_parser.add_argument("--path-map", type=_path_map, action="append", default=[])
    manifest_parser.add_argument("--output", type=Path, required=True, help="new directory; never reused")
    validate_parser = subcommands.add_parser("validate", help="verify source SHA/schema/time sampling on CPU")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--partition", choices=(*PARTITIONS, "eval80", "test40"), default="train")
    validate_parser.add_argument("--limit", type=int, help="explicit deterministic prefix subset")
    validate_parser.add_argument("--path-map", type=_path_map, action="append", default=[])
    validate_parser.add_argument("--cache", action="store_true", help="save independent derived reference tensors")
    validate_parser.add_argument("--output", type=Path, required=True, help="new directory; never reused")
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = build_manifest(args.splits, args.eval_cases, args.test_cases,
                                      train_selection_path=args.train_selection, bindings_path=args.bindings,
                                      path_maps=args.path_map, profile=args.profile,
                                      run_eval_cases_path=args.run_eval_cases)
            inputs = [value["path"] for value in manifest["inputs"].values()]
            native_roots = [Path(entry["motion"]).parent for entries in manifest["partitions"].values()
                            for entry in entries if Path(entry["motion"]).is_absolute()]
            output = _fresh_output(args.output, protected_files=inputs,
                                   source_roots=[root for _, root in manifest["path_maps"]] + native_roots)
            write_manifest(manifest, output / "manifest.json")
            print(json.dumps({"manifest": str(output / "manifest.json"), "counts": manifest["counts"],
                              "validation": manifest["validation"]}, ensure_ascii=False))
            return 0
        report = validate_dataset(args.manifest, args.output, partition=args.partition,
                                  limit=args.limit, path_maps=args.path_map, cache=args.cache)
        print(json.dumps({"report": str(args.output.resolve() / "validation.json"),
                          "success": report["success"], "validated_motions": report["validated_motions"],
                          "scope": report["scope"], "errors": report["errors"]}, ensure_ascii=False))
        return 0 if report["success"] else 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        print(f"HOI data preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
