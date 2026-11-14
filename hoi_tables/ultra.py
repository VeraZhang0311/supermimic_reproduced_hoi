"""ULTRA inputs for the direct privileged teacher, without changing its objective.

ULTRA retains its canonical, explicitly assumed 60 Hz source clock.  This module
does not manufacture a tables/30 Hz manifest or inherit the historical robot.
Existing object hulls, SDFs and FPS points are bound by bytes. Reference intent
can use the separately identified, user-authorized v1.1 cached targets; the
historical exact-current-geometry cache reader remains available.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace

import torch

from hoi_residual.assets import _digest_json, load_assets as _load_assets
from hoi_residual.data import ASSETS, UltraMotionLib, read_manifest, sha256_file
from hoi_residual.shape import load_shape_bank as _load_shape_bank
from hoi_residual.shape import point_sha256
from supermimic.robot_metadata import URDF_PATH, MJCF_PATH
from .assets import load_geometry_cache
from .prepare_contacts import _hands_identity, _labels, _tensor_hash
from .reward import TablesReward, current_hands


DEFAULT_CANONICAL = "/mnt/projects/ll/tianzizh/supermimic_hoi_zeroshot/results/hoi80_inputs/manifest.json"
DEFAULT_HELPER_ROOT = "/mnt/projects/ll/tianzizh/gmt26k_hoi80_zeroshot_20261106_03/source"
CANONICAL_SHA256 = "5c2832ba04c1e4061d9a1a1864f820f7ad949e1dd2ab982efff1dbf9511049a2"
CONTACT_SCHEMA = "hoi_direct_ultra_exact_reference_contacts_v1"
PROTOCOL_HASHES = {
    "data.py": "21b834f57da4e388a3c108ca8b34e108a698d034eb5077784421b3aa04138fc1",
    "metrics.py": "087742c554f3cf8de5edd83966c9e9479d6614eb1424983c11c5fc35d7700c17",
    "native.py": "450e066fb6dfc7cdb85c8b3b0ca9290556868bf9f19596f88c4977bc1bb53f2c",
    "ultra_kinematics.py": "c5547f78fedbab91e78ea9f9eeb1dc68b232e65f279a2c079a916b0539890710",
    "ultra_math.py": "274d172996fe40f3a5745cf6ea09d617448e2c33e8d15ad8f92ea0bd3b604f85",
}
CONTACT_CONTRACT = {
    "source_branch": "hoi/v1.1-observation-matching", "source_commit": "66d9184",
    "method": "unchanged_TablesReward._expected_exact_current_collision_triangles",
    "label_schema": "reference_hand_object_geometric_intent_v2_surface_intersection_unknown",
    "positive_gap_m": .02, "negative_gap_m": .04, "severe_penetration_m": .02,
    "intersection": "edge-triangle-crossing-or-touching-depth-is-unknown",
    "state_alignment": "Tplus1_reference_states; interval_reward_indexes_next_state",
    "old_robot_contact_labels_reused": False, "source_body_flags_used": False,
    "optimizer_updates": 0, "normalizer_updates": 0, "rsi_updates": 0,
}


def _binding(binding, parent):
    result = deepcopy(binding)
    path = Path(result["path"])
    if not path.is_absolute():
        path = Path(parent) / path
    result["path"] = str(path.resolve())
    if sha256_file(path) != result["sha256"]:
        raise ValueError("ULTRA bound cache bytes changed")
    return result


def _geometry_bundle(path):
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    declared = value.pop("bundle_content_sha256", None)
    if value.get("schema_version") != "supermimic_hoi_v1_bundle_v1" or declared != _digest_json(value):
        raise ValueError("ULTRA geometry bundle content/schema mismatch")
    assets = value["assets"]
    if len(assets) != 8 or {a["object_asset"] for a in assets} != set(ASSETS):
        raise ValueError("ULTRA geometry bundle must bind the exact eight assets")
    return value, hashlib.sha256(raw).hexdigest()


def _check_hands(evidence, hands=None, *, hull_count=3):
    if set(evidence) != {"left", "right"}:
        raise ValueError("Current hw3 hand evidence must cover left and right")
    for value in evidence.values():
        distance = value["calibrated_to_current_projection_max_m"]
        # This is the displacement of OLD calibrated probes before the exact
        # same projection used by tables. It is not a current surface error.
        if not 0 <= distance < float("inf") or len(value["colliders"]) != hull_count:
            raise ValueError("Current hw3 hand hull/projection evidence is invalid")
    if hands is not None:
        import numpy as np
        import trimesh
        if set(hands) != {"left", "right"}:
            raise ValueError("Current projected palm geometry is incomplete")
        for side, hand in hands.items():
            points = hand["palm_points"].detach().cpu().float().contiguous()
            record = evidence[side]
            if (points.shape != (record["probe_count"], 3) or not torch.isfinite(points).all()
                    or hashlib.sha256(points.numpy().tobytes()).hexdigest() != record["probe_sha256"]
                    or len(hand["collision_meshes"]) != hull_count):
                raise ValueError("Current projected palm probe bytes/frame differ")
            distances = []
            for mesh in hand["collision_meshes"]:
                hull = trimesh.Trimesh(torch.as_tensor(mesh["vertices"]).numpy(),
                                       torch.as_tensor(mesh["faces"]).numpy(), process=False)
                distances.append(trimesh.proximity.closest_point_naive(hull, points.numpy())[1])
            if float(np.min(np.stack(distances), axis=0).max()) > 1e-6:
                raise ValueError("Projected palm probes must lie on the current hull union within 1um")


def load_ultra_assets(asset_bundle, geometry_bundle):
    """Return canonical sorted assets plus verified current-hand source binding.

    Only object geometry/nominal properties are inherited. Old asset IDs,
    robot physics/FK and native acceptance are not reused. Label provenance
    is selected separately by the reference-contact cache reader.
    """
    asset_bundle, geometry_bundle = Path(asset_bundle), Path(geometry_bundle)
    assets = _load_assets(asset_bundle)
    source, geometry_sha = _geometry_bundle(geometry_bundle)
    by_name = {a["object_asset"]: a for a in source["assets"]}
    catalog_sha = sha256_file(asset_bundle)
    for index, asset in enumerate(assets):
        original = by_name[asset["object_asset"]]
        if any(asset[key] != original[key] for key in
               ("exact_scale", "collision_hash", "collision_meshes", "physical_profile")):
            raise ValueError("ULTRA object geometry/profile differs between catalog and SDF bundle")
        asset["asset_id"] = index
        asset["geometry_cache"] = _binding(original["geometry_cache"], geometry_bundle.parent)
        load_geometry_cache(asset)
        asset["teacher_binding"] = {"catalog_sha256": catalog_sha, "geometry_bundle_sha256": geometry_sha,
                                    "additional_runtime_scale": 1., "historical_robot_physics_reused": False}
    hands = _binding(source["hands"]["geometry_cache"], geometry_bundle.parent)
    current, evidence = current_hands(hands)
    _check_hands(evidence, current)
    if sha256_file(geometry_bundle) != geometry_sha or sha256_file(asset_bundle) != catalog_sha:
        raise RuntimeError("ULTRA object inputs changed while binding")
    return assets, hands


def load_ultra_shape(shape_bank, asset_bundle):
    """Verify raw FPS128 bytes against their actual SDF/FPS cache and hulls."""
    points, metadata = _load_shape_bank(shape_bank, asset_bundle)
    bank = json.loads(Path(shape_bank).read_text(encoding="utf-8"))
    assets = _load_assets(asset_bundle)
    for index, (entry, asset) in enumerate(zip(bank["assets"], assets, strict=True)):
        asset["geometry_cache"] = _binding(entry["source_cache"], Path(shape_bank).parent)
        cache = load_geometry_cache(asset)
        if not torch.equal(points[index], cache["points"]):
            raise ValueError("ULTRA raw shape bank differs from exact geometry-cache FPS128 points")
    return points, metadata


def _protocol_module(helper_root=None):
    if helper_root is not None or Path(DEFAULT_HELPER_ROOT).is_dir():
        selected = Path(helper_root or DEFAULT_HELPER_ROOT).resolve()
        if not (selected / "hoi_zero_shot/__init__.py").is_file():
            raise ValueError("Frozen canonical HOI80 helper root is missing")
        if str(selected) not in sys.path:
            sys.path.insert(0, str(selected))
    try:
        import hoi_zero_shot
        from hoi_zero_shot import data
    except ImportError as error:
        raise ImportError("Add the frozen gmt26k_hoi80_zeroshot_20261106_03/source directory to PYTHONPATH") from error
    root = Path(hoi_zero_shot.__file__).resolve().parent
    if helper_root is not None and root.parent != Path(helper_root).resolve():
        raise ValueError("A different canonical HOI80 helper package was already imported")
    if any(sha256_file(root / name) != digest for name, digest in PROTOCOL_HASHES.items()):
        raise ValueError("Canonical HOI80 protocol implementation hash mismatch")
    return data


def canonical_binding(manifest, canonical=None, *, helper_root=None):
    """Bind the canonical development80, including all reference/URDF hashes."""
    canonical = canonical or DEFAULT_CANONICAL
    if sha256_file(canonical) != CANONICAL_SHA256:
        raise ValueError("Canonical HOI80 manifest hash mismatch")
    binding = _protocol_module(helper_root).bind_canonical80(manifest, canonical)
    binding["entries"] = [_entry(e, c["state_frames"], "protected_eval")
                          for e, c in zip(binding["entries"], binding["cases"], strict=True)]
    return binding


def _entry(entry, length, partition):
    result = deepcopy(entry)
    result.update(clip_id="ultra/" + PurePosixPath(entry["motion"].replace("\\", "/")).stem,
                  source_trajectory_sha256=entry["motion_sha256"], state_frames=int(length),
                  source_partition=entry.get("source_partition", entry["partition"]), partition=partition,
                  role="train" if partition == "train" else "protected_eval")
    return result


def _augment(motion, manifest, canonical, partition):
    motion.entries = [_entry(e, n, partition) for e, n in zip(motion.entries, motion.clip_len.tolist(), strict=True)]
    if len({e["clip_id"] for e in motion.entries}) != len(motion.entries):
        raise ValueError("ULTRA motion IDs collide after canonical filename binding")
    motion.partition = partition
    motion.allowed_for_training = motion.for_training and partition == "train"
    motion.source_manifest_path = str(Path(manifest).resolve())
    motion.canonical_manifest_path = str(Path(canonical or DEFAULT_CANONICAL).resolve())
    motion.ultra_manifest_sha256 = sha256_file(manifest)
    motion.object_p, motion.object_v, motion.object_w = motion.objects[:, :3], motion.objects[:, 7:10], motion.objects[:, 10:13]
    from hoi_residual.data import Indexed
    from supermimic.rotations import quat_to_mat
    motion.object_R = Indexed(lambda f: quat_to_mat(motion.objects[f, 3:7]), motion.num_frames)
    motion.selection_contract.update(allowed_for_training=motion.allowed_for_training,
                                     source_timebase_status="protocol_assumed_not_generation_verified",
                                     source_fps_assumed=60, profile="ultra_direct_teacher_v1")


class UltraTeacherMotion(UltraMotionLib):
    def __init__(self, manifest_path, kin, device, *, partition="train", for_training=True,
                 asset=None, one_per_asset=False, canonical=None, helper_root=None):
        super().__init__(manifest_path, kin, device, partition=partition, for_training=for_training,
                         asset=asset, one_per_asset=one_per_asset)
        _augment(self, manifest_path, canonical, partition)
        self.hoi80_helper_root = helper_root

    def ref_object(self, frames):
        return self.object_ref(frames)


class HOI80TeacherMotion(UltraTeacherMotion):
    def __init__(self, manifest_path, kin, device, *, asset=None, canonical=None, helper_root=None):
        # Retain the verified legacy selector without changing its code or the
        # existing ULTRA adapter. Its FK is the supplied current hardware FK.
        binding = canonical_binding(manifest_path, canonical, helper_root=helper_root)
        original = _protocol_module(helper_root).Canonical80Motion(manifest_path, canonical or DEFAULT_CANONICAL,
                                                        kin, device, asset=asset)
        self.__dict__.update(original.__dict__)
        _augment(self, manifest_path, canonical, "protected_eval")
        self.for_training, self.allowed_for_training = False, False
        self.hoi80_helper_root = helper_root
        self.all_canonical_cases = deepcopy(binding["cases"])
        self.all_canonical_clip_ids = [_entry(e, c["state_frames"], "protected_eval")["clip_id"]
                                       for e, c in zip(binding["entries"], binding["cases"], strict=True)]
        self.selection_contract.update(partition="protected_eval", for_training=False,
                                       allowed_for_training=False,
                                       evaluation_role="canonical_development80_not_independent_test")

    def load_reference(self, clip):
        case = self.cases[clip]
        return _protocol_module(self.hoi80_helper_root)._reference_arrays(case["reference_path"], case["reference_sha256"],
                                                    case["state_frames"])[0]


def _code_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("hoi_tables/ultra.py", "hoi_tables/reward.py", "hoi_tables/geometry.py",
             "hoi_tables/assets.py", "hoi_tables/prepare_contacts.py", "hoi_residual/data.py",
             "hoi_residual/assets.py", "hoi_residual/shape.py", "hoi_residual/v11_reward.py",
             "supermimic/kinematics.py", "supermimic/rotations.py", "supermimic/robot_metadata.py")
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in names}


def _inventory(manifest, binding):
    result = {}
    for source in manifest["clips"]:
        if source["partition"] == "train":
            entry = _entry(source, 0, "train")
            entry.pop("state_frames")
            result[entry["clip_id"]] = entry
    for source, case in zip(binding["entries"], binding["cases"], strict=True):
        entry = _entry(source, case["state_frames"], "protected_eval")
        if entry["clip_id"] in result:
            raise ValueError("ULTRA training and HOI80 source groups overlap")
        result[entry["clip_id"]] = entry
    if manifest["profile"] == "canonical-v4" and (len(result) != 2731 or len(binding["cases"]) != 80):
        raise ValueError("ULTRA contact inventory must cover train2651 plus development80")
    train_groups = {e["source_group"] for e in result.values() if e["partition"] == "train"}
    if train_groups & {e["source_group"] for e in result.values() if e["partition"] != "train"}:
        raise ValueError("ULTRA training and HOI80 source groups overlap")
    return result


def contact_identity(motion, assets, hands_binding, *, hand_evidence=None, binding=None):
    binding = binding or canonical_binding(motion.source_manifest_path, motion.canonical_manifest_path,
                                           helper_root=getattr(motion, "hoi80_helper_root", None))
    if hand_evidence is None:
        current, hand_evidence = current_hands(hands_binding)
        _check_hands(hand_evidence, current)
    else:
        _check_hands(hand_evidence)
    if sha256_file(hands_binding["path"]) != hands_binding["sha256"]:
        raise ValueError("ULTRA hand cache bytes changed")
    if len(assets) != 8 or [a["object_asset"] for a in assets] != list(ASSETS):
        raise ValueError("ULTRA contact identity requires all eight sorted assets")
    geometries = {}
    for asset in assets:
        load_geometry_cache(asset)
        if sha256_file(asset["native_asset"]) != asset["native_asset_sha256"]:
            raise ValueError("ULTRA native asset changed")
        geometries[asset["object_asset"]] = {"asset_id": asset["asset_id"], "collision_hash": asset["collision_hash"],
                    "exact_scale": asset["exact_scale"], "geometry_cache_sha256": asset["geometry_cache"]["sha256"],
                    "native_asset_sha256": asset["native_asset_sha256"]}
    catalog_bindings = {(_digest_json(a["teacher_binding"])) for a in assets}
    if len(catalog_bindings) != 1:
        raise ValueError("ULTRA catalog/geometry bundle bindings differ across assets")
    return {"contract": deepcopy(CONTACT_CONTRACT), "ultra_manifest_sha256": motion.ultra_manifest_sha256,
            "canonical_manifest_sha256": binding["canonical_manifest_sha256"],
            "canonical_protocol_sha256": deepcopy(PROTOCOL_HASHES),
            "object_input_binding": deepcopy(assets[0]["teacher_binding"]),
            "target_robot": {"urdf_sha256": sha256_file(URDF_PATH), "mjcf_sha256": sha256_file(MJCF_PATH)},
            "hands_cache_sha256": hands_binding["sha256"], "hand_geometry": _hands_identity(hand_evidence),
            "object_geometry": geometries, "implementation_sha256": _code_identity()}


def _clip_identity(entry):
    return {key: entry[key] for key in ("clip_id", "source_group", "object_asset", "partition",
                                       "source_partition", "source_trajectory_sha256", "state_frames")}


def _content(value):
    return {"schema": value["schema"], "complete": value["complete"], "metadata": value["metadata"],
            "clips": {name: {k: v for k, v in record.items() if k not in ("expected", "valid")}
                      for name, record in value["clips"].items()}}


def _record(motion, reward, index, chunk_size):
    entry = motion.entries[index]
    start, end = int(motion.clip_start[index]), int(motion.clip_end[index]) + 1
    ep, vp = [], []
    for first in range(start, end, chunk_size):
        with torch.no_grad():
            e, v = reward._expected(torch.arange(first, min(first + chunk_size, end), device=motion.device))
        ep.append(e.cpu()); vp.append(v.cpu())
        reward._reference_contacts.clear()
    expected, valid = _labels(torch.cat(ep), torch.cat(vp), end - start)
    return {**_clip_identity(entry), "expected": expected, "valid": valid,
            "expected_sha256": _tensor_hash(expected), "valid_sha256": _tensor_hash(valid)}


def load_contact_cache(path, motion, assets, hands_binding, *, native_hands=None):
    raw = Path(path).read_bytes()
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    from .v11_reference_contacts import SCHEMA as REUSED_SCHEMA, load_reused_cache
    if value.get("schema") == REUSED_SCHEMA:
        return load_reused_cache(path, motion, assets, hands_binding, _raw=raw, _value=value)
    if native_hands is not None:
        raise ValueError("Native Ultra hands require the original Ultra v1.1 reference labels")
    if (value.get("schema") != CONTACT_SCHEMA or value.get("complete") is not True
            or value.get("content_sha256") != _digest_json(_content(value))):
        raise ValueError("ULTRA reference contact cache schema/completeness/content mismatch")
    if (value["metadata"].get("reference_only") is not True
            or any(value["metadata"].get(k) != 0 for k in ("optimizer_updates", "normalizer_updates", "rsi_updates"))):
        raise ValueError("ULTRA contact cache must be reference-only with zero training updates")
    binding = canonical_binding(motion.source_manifest_path, motion.canonical_manifest_path,
                                helper_root=getattr(motion, "hoi80_helper_root", None))
    if value["metadata"]["identity"] != contact_identity(motion, assets, hands_binding, binding=binding):
        raise ValueError("ULTRA contact current robot/hand/geometry/manifest/implementation identity mismatch")
    inventory = _inventory(motion.manifest, binding)
    if set(value["clips"]) != set(inventory) or value["metadata"]["clip_count"] != len(inventory):
        raise ValueError("ULTRA contact cache must cover all train2651 and canonical80 clips")
    for name, record in value["clips"].items():
        target = inventory[name]
        if any(record[k] != target[k] for k in _clip_identity(record) if k != "state_frames"):
            raise ValueError("ULTRA contact clip/source/partition identity mismatch")
        if (type(record["state_frames"]) is not int or record["state_frames"] < 2
                or "state_frames" in target and record["state_frames"] != target["state_frames"]):
            raise ValueError("ULTRA contact state-frame count mismatch")
        if record["expected"].dtype != torch.float32 or record["valid"].dtype != torch.bool:
            raise ValueError("ULTRA contact stored label dtypes differ")
        e, v = _labels(record["expected"], record["valid"], record["state_frames"])
        if _tensor_hash(e) != record["expected_sha256"] or _tensor_hash(v) != record["valid_sha256"]:
            raise ValueError("ULTRA contact label tensor bytes changed")
    records = [value["clips"][e["clip_id"]] for e in motion.entries]
    if any(_clip_identity(record) != _clip_identity(entry)
           for record, entry in zip(records, motion.entries, strict=True)):
        raise ValueError("ULTRA contact cache selected worker lengths/identity differ")
    frames = sum(record["state_frames"] for record in value["clips"].values())
    partitions = {p: sum(c["partition"] == p for c in value["clips"].values()) for p in ("train", "protected_eval")}
    if frames != value["metadata"]["state_frames"] or partitions != value["metadata"]["partitions"]:
        raise ValueError("ULTRA contact aggregate frame/partition counts differ")
    e, v = torch.cat([x["expected"] for x in records]), torch.cat([x["valid"] for x in records])
    if len(e) != motion.num_frames:
        raise ValueError("ULTRA contact cache worker-local frame count differs")
    metadata = deepcopy(value["metadata"])
    metadata.update(file_sha256=hashlib.sha256(raw).hexdigest(), content_sha256=value["content_sha256"],
                    selected_clips=[entry["clip_id"] for entry in motion.entries])
    return e.to(motion.device), v.to(motion.device), metadata


def make_ultra_reward(mdp, assets, hands_binding, reference_contacts=None, *, native_hands=None):
    reward = TablesReward(mdp, assets, hands_binding, native_hands=native_hands)
    _check_hands(reward.hand_evidence, reward.hands, hull_count=3 if native_hands is None else 5)
    if reference_contacts is not None:
        expected, valid, metadata = load_contact_cache(reference_contacts, mdp.motion, assets, hands_binding,
                                                      native_hands=native_hands)
        reward._frozen_contacts = expected, valid
        reward.contract["frozen_reference_contacts"] = metadata
    return reward


def _prepare_piece(spec, partition, asset, chunk_size, threads):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    from supermimic.kinematics import G1Kinematics
    kin = G1Kinematics("cpu")
    assets, hands = load_ultra_assets(spec["asset_bundle"], spec["geometry_bundle"])
    factory = UltraTeacherMotion if partition == "train" else HOI80TeacherMotion
    kwargs = {"partition": "train", "for_training": False} if partition == "train" else {}
    motion = factory(spec["manifest"], kin, "cpu", asset=asset, canonical=spec["canonical"], **kwargs)
    reward = make_ultra_reward(SimpleNamespace(kin=kin, motion=motion), assets, hands)
    return {"identity": contact_identity(motion, assets, hands, hand_evidence=reward.hand_evidence),
            "clips": [_record(motion, reward, i, chunk_size) for i in range(len(motion.entries))]}


def prepare_contact_cache(manifest, asset_bundle, geometry_bundle, shape_bank, output, *,
                          canonical=None, chunk_size=32, workers=1, worker_threads=1, progress=None):
    """Compute a NEW 2731-clip CPU cache; no old robot contact labels are read.

    The CLI is intended for a scheduler CPU allocation. It never starts a job,
    simulator, optimizer, normalizer or training run. Output refuses overwrite.
    """
    if any(type(v) is not int or v < 1 for v in (chunk_size, workers, worker_threads)):
        raise ValueError("ULTRA contact CPU workers/threads/chunk-size must be positive integers")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    spec = {"manifest": str(Path(manifest).resolve()), "asset_bundle": str(Path(asset_bundle).resolve()),
            "geometry_bundle": str(Path(geometry_bundle).resolve()),
            "canonical": str(Path(canonical or DEFAULT_CANONICAL).resolve())}
    full = read_manifest(manifest)
    binding = canonical_binding(manifest, spec["canonical"])
    inventory = _inventory(full, binding)
    load_ultra_shape(shape_bank, asset_bundle)
    assets, hands = load_ultra_assets(asset_bundle, geometry_bundle)
    pseudo = SimpleNamespace(source_manifest_path=spec["manifest"], canonical_manifest_path=spec["canonical"],
                             ultra_manifest_sha256=sha256_file(manifest))
    identity = contact_identity(pseudo, assets, hands, binding=binding)
    effective = min(workers, 16)
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if effective * worker_threads > allocated:
        raise ValueError("ULTRA contact workers*threads exceeds allocated CPUs")
    clips = {}

    def accept(piece):
        if piece["identity"] != identity:
            raise RuntimeError("ULTRA contact worker input identity changed")
        for record in piece["clips"]:
            name = record["clip_id"]
            if name in clips or name not in inventory:
                raise ValueError("Duplicate/unselected ULTRA contact worker clip")
            target = inventory[name]
            if any(record[k] != target[k] for k in _clip_identity(record) if k != "state_frames"):
                raise ValueError("ULTRA contact worker source/partition drift")
            clips[name] = record
        if progress is not None:
            progress({"completed_clips": len(clips), "state_frames": sum(x["state_frames"] for x in clips.values())})

    tasks = [(partition, asset) for partition in ("train", "protected_eval") for asset in range(8)]
    if effective == 1:
        for partition, asset in tasks:
            accept(_prepare_piece(spec, partition, asset, chunk_size, worker_threads))
    else:
        with ProcessPoolExecutor(max_workers=effective, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_prepare_piece, spec, partition, asset, chunk_size, worker_threads)
                       for partition, asset in tasks]
            for future in futures:
                accept(future.result())
    if set(clips) != set(inventory):
        raise ValueError("Incomplete ULTRA train2651/canonical80 contact cache")
    if contact_identity(pseudo, assets, hands) != identity:
        raise RuntimeError("ULTRA contact inputs changed during preprocessing")
    metadata = {"identity": identity, "clip_count": len(clips),
                "state_frames": sum(x["state_frames"] for x in clips.values()),
                "partitions": {p: sum(c["partition"] == p for c in clips.values()) for p in ("train", "protected_eval")},
                "reference_only": True, "optimizer_updates": 0, "normalizer_updates": 0, "rsi_updates": 0,
                "execution": {"workers": effective, "worker_threads": worker_threads, "chunk_size": chunk_size},
                "shape_bank_sha256": sha256_file(shape_bank)}
    value = {"schema": CONTACT_SCHEMA, "complete": True, "metadata": metadata,
             "clips": {name: clips[name] for name in sorted(clips)}}
    value["content_sha256"] = _digest_json(_content(value))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(value, stream)
    return {**metadata, "file_sha256": sha256_file(output), "content_sha256": value["content_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-contacts", "preflight"))
    for name in ("manifest", "asset-bundle", "geometry-bundle", "shape-bank"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--canonical", default=DEFAULT_CANONICAL)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare-contacts":
        if not args.output:
            parser.error("prepare-contacts requires --output NEW_FILE")
        report = prepare_contact_cache(args.manifest, args.asset_bundle, args.geometry_bundle,
                                       args.shape_bank, args.output, canonical=args.canonical,
                                       chunk_size=args.chunk_size, workers=args.workers, worker_threads=args.threads,
                                       progress=lambda value: print(json.dumps(value), flush=True))
    else:
        binding = canonical_binding(args.manifest, args.canonical)
        assets, hands = load_ultra_assets(args.asset_bundle, args.geometry_bundle)
        points, shape = load_ultra_shape(args.shape_bank, args.asset_bundle)
        report = {"passed": True, "asset_count": len(assets), "shape": list(points.shape),
                  "shape_identity": shape, "hands_sha256": hands["sha256"],
                  "train_clips": read_manifest(args.manifest)["partitions"]["train"]["motions"],
                  "canonical_clips": len(binding["cases"]), "canonical_role": "development_evaluation_not_independent_test",
                  "canonical_control_steps": sum(c["control_steps"] for c in binding["cases"]),
                  "contact_cache_computed": False, "optimizer_updates": 0, "normalizer_updates": 0}
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
