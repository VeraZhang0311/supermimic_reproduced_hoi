"""Freeze exact v1.1 reference hand-object intent for every selected table clip.

This is CPU reference preprocessing, with no simulator, optimizer or RMS update.
Labels retain the surface-intersection unknown semantics of TablesReward._expected.
They are stored by immutable clip_id, independently of a worker's asset filter,
partition or local concatenated frame indices.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from hoi_residual.assets import _digest_json
from supermimic.robot_metadata import URDF_PATH, MJCF_PATH
from .assets import load_assets, load_geometry_cache
from .data import TableMotionLib, sha256_file
from .reward import TablesReward, current_hands

SCHEMA = "hoi_tables_exact_reference_contact_cache_v1"
LABEL_SCHEMA = "reference_hand_object_geometric_intent_v2_surface_intersection_unknown"
_WORKER_CONTEXT = None
CONTACT_CONTRACT = {
    "label_schema": LABEL_SCHEMA,
    "channels": ["left_hand_object", "right_hand_object"],
    "reference_source_branch": "hoi/v1.1-observation-matching",
    "reference_source_commit": "66d9184",
    "target_branch": "hoi-retarget/tables-v1.0",
    "target_base_commit": "78cd5b2c3530f9aa47b0c944436e59d5dc054093",
    "positive_gap_m": .02,
    "negative_gap_m": .04,
    "severe_penetration_m": .02,
    "intersection": "edge-triangle-crossing-or-touching-depth-is-unknown",
    "method": "TablesReward._expected_exact_collision_triangles",
    "state_alignment": "one_pair_target_per_Tplus1_reference_state; interval_reward_indexes_next_state",
    "source_contact_flags_used": False,
    "training_updates": False,
}


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = str((tuple(value.shape), str(value.dtype))).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _code_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("hoi_tables/prepare_contacts.py", "hoi_tables/reward.py", "hoi_tables/geometry.py",
             "hoi_tables/data.py", "supermimic/kinematics.py", "supermimic/rotations.py",
             "supermimic/robot_metadata.py")
    return {name: hashlib.sha256((root/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in names}


def _hands_identity(evidence):
    # Source checkout paths can differ between preprocessing and training. Byte
    # hashes, transforms and projected probe bytes define the actual geometry.
    result = {}
    for side in ("left", "right"):
        record = evidence[side]
        result[side] = {
            "colliders": [{"filename": Path(c["path"]).name, "sha256": c["sha256"],
                           "origin_xyz": c["origin_xyz"], "origin_rpy": c["origin_rpy"]}
                          for c in record["colliders"]],
            "probe_count": record["probe_count"],
            "probe_sha256": record["probe_sha256"],
            "probe_frame": record["probe_frame"],
        }
    return result


def contact_identity(motion, assets, hands_binding, *, hand_evidence=None):
    """Current geometry/implementation identity, independent of worker filters."""
    if hand_evidence is None:
        _, hand_evidence = current_hands(hands_binding)
    binding_path = Path(hands_binding["path"])
    if sha256_file(binding_path) != hands_binding["sha256"]:
        raise ValueError("reference-contact hand cache bytes changed")
    geometries = {}
    for asset in assets:
        load_geometry_cache(asset)
        if sha256_file(asset["native_asset"]) != asset["native_asset_sha256"]:
            raise ValueError("reference-contact native object geometry bytes changed")
        geometries[asset["object_asset"]] = {
            "asset_id": int(asset["asset_id"]),
            "collision_hash": asset["collision_hash"],
            "geometry_cache_sha256": asset["geometry_cache"]["sha256"],
            "native_asset_sha256": asset["native_asset_sha256"],
            "exact_scale": asset["exact_scale"],
        }
    if not {entry["object_asset"] for entry in motion.entries}.issubset(geometries):
        raise ValueError("reference-contact identity lacks a selected worker asset")
    selection = motion.manifest["selection"]
    return {"contract": deepcopy(CONTACT_CONTRACT),
            "manifest_sha256": motion.manifest_sha256,
            "manifest_content_sha256": motion.manifest["content_sha256"],
            "reference_cache_sha256": motion.manifest["cache"]["sha256"],
            "selection": {k: selection[k] for k in ("sha256", "workbook_sha256", "included_count", "excluded")},
            "target_robot": {"urdf_sha256": sha256_file(URDF_PATH),
                             "mjcf_sha256": sha256_file(MJCF_PATH)},
            "hands_cache_sha256": hands_binding["sha256"],
            "hand_geometry": _hands_identity(hand_evidence),
            "object_geometry": geometries,
            "implementation_sha256": _code_identity()}


def _labels(expected, valid, count):
    if not isinstance(expected, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise ValueError("reference-contact labels must be tensors")
    if expected.shape != (count, 2) or valid.shape != (count, 2):
        raise ValueError("reference-contact state/channel count differs")
    if (not torch.isfinite(expected).all() or not torch.isfinite(valid).all() or
            not ((expected == 0) | (expected == 1)).all() or
            not ((valid == 0) | (valid == 1)).all()):
        raise ValueError("reference-contact expected/valid must be finite binary values")
    expected = expected.detach().cpu().float().contiguous()
    valid = valid.detach().cpu().bool().contiguous()
    if (expected[~valid] != 0).any():
        raise ValueError("unknown reference-contact targets must not be positive")
    return expected, valid


def _clip_identity(entry):
    return {key: entry[key] for key in ("clip_id", "source_group", "object_asset", "partition",
                                       "source_trajectory_sha256", "state_frames")}


def _content_identity(value):
    return {"schema": value["schema"], "complete": value["complete"],
            "metadata": value["metadata"],
            "clips": {name: {key: field for key, field in entry.items()
                             if key not in ("expected", "valid")}
                      for name, entry in value["clips"].items()}}


def _new_kinematics():
    from supermimic.kinematics import G1Kinematics
    return G1Kinematics("cpu")


def _read_assets_and_hands(asset_bundle):
    asset_bundle = Path(asset_bundle)
    assets = load_assets(asset_bundle)
    catalog = json.loads(asset_bundle.read_text(encoding="utf-8"))
    binding = catalog["hands"]["geometry_cache"]
    if not Path(binding["path"]).is_absolute():
        binding = dict(binding, path=str(asset_bundle.parent/binding["path"]))
    return assets, binding


def _compute_clip_record(motion, reward, index, chunk_size):
    """The same exact geometry query and masks in serial and spawned workers."""
    entry = motion.entries[index]
    start, end = int(motion.clip_start[index]), int(motion.clip_end[index]) + 1
    expected_parts, valid_parts = [], []
    for first in range(start, end, chunk_size):
        frame = torch.arange(first, min(first+chunk_size, end))
        with torch.no_grad():
            expected, valid = reward._expected(frame)
        expected_parts.append(expected.cpu()); valid_parts.append(valid.cpu())
        # Bound temporary frame dictionaries to one chunk, rather than the
        # whole partition, while keeping the exact reference geometry algorithm.
        reward._reference_contacts.clear()
    expected, valid = _labels(torch.cat(expected_parts), torch.cat(valid_parts), end-start)
    return {**_clip_identity(entry), "expected": expected, "valid": valid,
            "expected_sha256": _tensor_hash(expected), "valid_sha256": _tensor_hash(valid)}


def _initialize_worker(manifest_path, asset_bundle, partition, chunk_size, threads, identity):
    """Read frozen inputs and construct one CPU FK/reward per spawned process."""
    global _WORKER_CONTEXT
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(threads)
        torch.set_num_threads(threads)
        kin = _new_kinematics()
        assets, binding = _read_assets_and_hands(asset_bundle)
        motion = TableMotionLib(manifest_path, kin, "cpu", partition=partition, for_training=False)
        reward = TablesReward(SimpleNamespace(kin=kin, motion=motion), assets, binding)
        actual_identity = contact_identity(motion, assets, binding, hand_evidence=reward.hand_evidence)
        if actual_identity != identity:
            raise RuntimeError("reference-contact worker input identity differs from parent")
        _WORKER_CONTEXT = {"motion": motion, "reward": reward, "chunk_size": chunk_size,
                           "indices": {e["clip_id"]: i for i, e in enumerate(motion.entries)}}
    except Exception as error:
        # Pool normally respawns an initializer that raises indefinitely.
        # Carry the failure to a task so the parent fails once and terminates
        # its pool; an incomplete output is never published.
        _WORKER_CONTEXT = {"error": f"{type(error).__name__}: {error}"}


def _worker_clip(clip_id):
    context = _WORKER_CONTEXT
    if context is None or "error" in context:
        message = "uninitialized worker" if context is None else context["error"]
        raise RuntimeError("reference-contact worker initialization failed: " + message)
    if clip_id not in context["indices"]:
        raise ValueError("contact task is outside the worker's frozen partition")
    return _compute_clip_record(context["motion"], context["reward"],
                                context["indices"][clip_id], context["chunk_size"])


def _worker_count(requested, threads, clip_count):
    if type(requested) is not int or requested < 1 or type(threads) is not int or threads < 1:
        raise ValueError("contact workers and worker threads must be positive integers")
    effective = min(requested, clip_count)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    available = int(allocation) if allocation else os.cpu_count()
    if available is not None and effective*threads > available:
        raise ValueError("contact workers*threads exceeds the allocated CPU count")
    return effective


def prepare_contact_cache(manifest_path, asset_bundle, output, *, chunk_size=32, kin=None, progress=None,
                          workers=1, worker_threads=1):
    """Compute all train and protected-eval reference labels, then save once.

    ``kin`` and ``progress`` permit a test harness or a caller-owned progress UI;
    the CLI constructs the current compiled G1 kinematics on CPU. An existing
    output is never overwritten. Partial selections are not valid cache files.
    With ``workers>1``, spawn a bounded pool per partition. Each worker loads
    frozen inputs/FK/reward once and handles whole clips; completion order
    cannot change final sorted clip identities or tensor hashes. This performs
    reference-only geometry queries for held-out clips, with zero learning.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("contact preprocessing chunk size must be positive")
    if type(workers) is not int or workers < 1 or type(worker_threads) is not int or worker_threads < 1:
        raise ValueError("contact workers and worker threads must be positive integers")
    if workers > 1 and kin is not None:
        raise ValueError("custom kinematics is supported only by serial contact preprocessing")
    manifest_path, asset_bundle, output = map(Path, (manifest_path, asset_bundle, output))
    if output.exists():
        raise FileExistsError(output)
    if kin is None:
        kin = _new_kinematics()
    assets, hands_binding = _read_assets_and_hands(asset_bundle)
    clips, identity, completed_frames = {}, None, 0
    effective_workers = {}
    started = time.monotonic()
    for partition in ("train", "protected_eval"):
        motion = TableMotionLib(manifest_path, kin, "cpu", partition=partition, for_training=False)
        mdp = SimpleNamespace(kin=kin, motion=motion)
        reward = TablesReward(mdp, assets, hands_binding)
        candidate_identity = contact_identity(motion, assets, hands_binding, hand_evidence=reward.hand_evidence)
        if identity is not None and candidate_identity != identity:
            raise RuntimeError("reference-contact inputs changed between partitions")
        identity = candidate_identity
        entries = {entry["clip_id"]: entry for entry in motion.entries}
        if len(entries) != len(motion.entries):
            raise ValueError("duplicate reference-contact clip in partition")
        effective = _worker_count(workers, worker_threads, len(entries))
        effective_workers[partition] = effective

        def accept(record):
            nonlocal completed_frames
            clip_id = record["clip_id"]
            if clip_id in clips:
                raise ValueError("one contact clip appears in multiple partitions")
            if clip_id not in entries or _clip_identity(record) != _clip_identity(entries[clip_id]):
                raise ValueError("worker contact clip/source/partition identity mismatch")
            expected, valid = _labels(record["expected"], record["valid"], entries[clip_id]["state_frames"])
            if _tensor_hash(expected) != record["expected_sha256"] or _tensor_hash(valid) != record["valid_sha256"]:
                raise ValueError("worker contact tensor hash mismatch")
            clips[clip_id] = record
            completed_frames += entries[clip_id]["state_frames"]
            if progress is not None:
                progress({"clip_id": clip_id, "partition": partition, "completed_clips": len(clips),
                          "completed_frames": completed_frames, "elapsed_s": time.monotonic()-started})

        if effective == 1:
            for index in range(len(motion.entries)):
                accept(_compute_clip_record(motion, reward, index, chunk_size))
        else:
            # Long clips begin first to reduce stragglers. The task queue holds
            # clip IDs only; motion/geometry are read once by each initializer.
            tasks = sorted(entries, key=lambda name: (-entries[name]["state_frames"], name))
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=effective, initializer=_initialize_worker,
                              initargs=(str(manifest_path.resolve()), str(asset_bundle.resolve()), partition,
                                        chunk_size, worker_threads, identity)) as pool:
                for record in pool.imap_unordered(_worker_clip, tasks, chunksize=1):
                    accept(record)
        # Re-read real current hand geometry and file hashes after exact work.
        if contact_identity(motion, assets, hands_binding) != identity:
            raise RuntimeError("reference-contact inputs changed during preprocessing")
    manifest = motion.manifest
    if set(clips) != {entry["clip_id"] for entry in manifest["clips"]} or len(clips) != manifest["selection"]["included_count"]:
        raise ValueError("frozen reference-contact cache does not cover the full whitelist")
    if sha256_file(manifest_path) != identity["manifest_sha256"]:
        raise RuntimeError("reference manifest changed during contact preprocessing")
    clips = {name: clips[name] for name in sorted(clips)}
    metadata = {"identity": identity, "clip_count": len(clips), "state_frames": completed_frames,
                "partitions": {name: sum(e["partition"] == name for e in clips.values())
                               for name in ("train", "protected_eval")},
                "elapsed_s": time.monotonic()-started, "reference_only": True,
                "execution": {"requested_workers": workers, "effective_workers": effective_workers,
                              "worker_threads": worker_threads, "chunk_size": chunk_size,
                              "process_start_method": "spawn" if max(effective_workers.values()) > 1 else "serial"},
                "optimizer_updates": 0, "normalizer_updates": 0, "rsi_updates": 0}
    value = {"schema": SCHEMA, "complete": True, "metadata": metadata, "clips": clips}
    value["content_sha256"] = _digest_json(_content_identity(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(value, stream)
    return {**metadata, "content_sha256": value["content_sha256"], "file_sha256": sha256_file(output)}


def load_contact_cache(path, motion, assets, hands_binding):
    """Return expected/valid tensors in the current worker's local frame order.

    Every stored clip and both frozen partitions must remain present, including
    when this worker loads one asset or a smoke-only single-clip selection.
    """
    raw = Path(path).read_bytes()
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (value.get("schema") != SCHEMA or value.get("complete") is not True or
            value.get("content_sha256") != _digest_json(_content_identity(value))):
        raise ValueError("reference-contact cache schema/content/completeness mismatch")
    identity = value["metadata"]["identity"]
    current = deepcopy(contact_identity(motion, assets, hands_binding))
    # A caller may pass just the selected asset; the frozen file still contains
    # the complete two-asset geometry identity and full-whitelist labels.
    expected_geometry = identity.get("object_geometry", {})
    actual_geometry = current.pop("object_geometry")
    if any(expected_geometry.get(name) != binding for name, binding in actual_geometry.items()):
        raise ValueError("reference-contact object geometry identity mismatch")
    expected_identity = {key: field for key, field in identity.items() if key != "object_geometry"}
    if current != expected_identity:
        raise ValueError("reference-contact robot/hand/manifest/selection/implementation identity mismatch")
    original = {entry["clip_id"]: entry for entry in motion.manifest["clips"]}
    if set(value["clips"]) != set(original) or len(original) != value["metadata"]["clip_count"]:
        raise ValueError("reference-contact cache omits or adds whitelist clips")
    if (sum(e["state_frames"] for e in original.values()) != value["metadata"]["state_frames"] or
            value["metadata"]["partitions"] != {name: sum(e["partition"] == name for e in original.values())
                                                for name in ("train", "protected_eval")}):
        raise ValueError("reference-contact cache aggregate counts differ")
    for name, record in value["clips"].items():
        entry = original[name]
        if _clip_identity(record) != _clip_identity(entry):
            raise ValueError("reference-contact clip/source/partition identity mismatch")
        if record["expected"].dtype != torch.float32 or record["valid"].dtype != torch.bool:
            raise ValueError("reference-contact stored tensor dtypes differ")
        expected, valid = _labels(record["expected"], record["valid"], entry["state_frames"])
        if _tensor_hash(expected) != record["expected_sha256"] or _tensor_hash(valid) != record["valid_sha256"]:
            raise ValueError("reference-contact label tensor bytes changed")
    expected = torch.cat([value["clips"][entry["clip_id"]]["expected"] for entry in motion.entries])
    valid = torch.cat([value["clips"][entry["clip_id"]]["valid"] for entry in motion.entries])
    if len(expected) != motion.num_frames:
        raise ValueError("reference-contact cache does not match worker-local reference length")
    metadata = deepcopy(value["metadata"])
    metadata.update(content_sha256=value["content_sha256"], file_sha256=hashlib.sha256(raw).hexdigest(),
                    selected_clips=[entry["clip_id"] for entry in motion.entries])
    return expected.to(motion.device), valid.to(motion.device), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-bundle", required=True)
    parser.add_argument("--output", required=True, help="new .pt file; refuses overwrite")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="CPU processes per partition, e.g. 8 or 16")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(args.threads)
    report = prepare_contact_cache(args.manifest, args.asset_bundle, args.output,
                                   chunk_size=args.chunk_size,
                                   workers=args.workers, worker_threads=args.threads,
                                   progress=lambda value: print(json.dumps(value), flush=True))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
