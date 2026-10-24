"""Build immutable, source-bound HOI training bundles on CPU.

This command prepares geometry and references only. Native probes remain an
explicit separate gate; a successful bundle is not evidence of simulator or
training success. All outputs are exclusive-created in a fresh directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import logging
import math
import multiprocessing
from pathlib import Path
import struct
import time

import torch

from hoi.data.manifest import read_manifest, resolve_motion_path, sha256_file
from hoi.data.prepare import _assert_adapted, _fresh_output
from hoi.data.reference import adapt_motion
from hoi.hand_collision import HAND_SHA, ORIGIN

from .geometry import (ShapeAssetBank, build_geometry_cache, collision_distance, exact_mesh_gap,
                       expected_schedule_from_distances, tensor_sha256, validate_mesh,
                       _prepare_exact_meshes, _rigid_mesh_vertices, _exact_mesh_gap)
from .motion import TrainingMotionBank

BUNDLE_SCHEMA = "supermimic_hoi_v1_bundle_v1"
LOGGER = logging.getLogger(__name__)


def native_robot_asset_binding(path, expected_sha256):
    """Bind a hand overlay and optional explicitly versioned foot extension."""
    path = Path(path).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("bound native robot/hand overlay bytes differ")
    hand = path.parent / "hand-asset.json"
    if not hand.is_file():
        raise ValueError("native rubber-hand overlay requires its original hand-asset.json receipt")
    result = {"path": str(path), "sha256": expected_sha256,
              "hand_asset_manifest": {"path": str(hand.resolve()), "sha256": sha256_file(hand)}}
    foot = path.parent / "foot-asset.json"
    if foot.is_file():
        from .foot_collision import validate_overlay, FOOT_PROFILE
        validate_overlay(path)
        result["foot_asset_manifest"] = {"path": str(foot.resolve()), "sha256": sha256_file(foot)}
        result["foot_collision_profile"] = FOOT_PROFILE
    return result


def bind_robot_collision_overlay(bundle_path, output, robot_asset, expected_sha256):
    """Rebind only robot collision identity; preserve immutable data/cache bytes.

    Object/hand geometry and reference FK are unchanged by replacing foot
    colliders. Their original recipe and all cache paths/hashes remain bound.
    The independent foot manifest supplies the new robot collision recipe.
    """
    from .evaluate import _read_bundle_header
    source = Path(bundle_path).resolve()
    bundle = deepcopy(_read_bundle_header(source))
    binding = native_robot_asset_binding(robot_asset, expected_sha256)
    if "foot_asset_manifest" not in binding:
        raise ValueError("robot collision rebind requires a verified foot overlay")
    from .foot_collision import validate_overlay
    feet = validate_overlay(binding["path"])
    base = feet["base_robot_asset"]
    previous = bundle.get("native_robot_asset", {})
    if (Path(previous.get("path", "/missing")).resolve() != Path(base["path"]).resolve() or
            previous.get("sha256") != base["sha256"]):
        raise ValueError("foot overlay must inherit the exact bundle robot/hand asset")
    original_hash = bundle.pop("bundle_content_sha256")
    bundle["native_robot_asset"] = binding
    bundle["robot_collision_extension"] = {
        "schema": "hoi_v1_robot_collision_rebind_v1", "profile": binding["foot_collision_profile"],
        "source_bundle": {"path": str(source), "sha256": sha256_file(source), "content_sha256": original_hash},
        "unchanged": "reference/source/timebase/FK/object/hand geometry and caches; only native foot collision changes"}
    bundle["validation"] = {**bundle["validation"], "native_physics": "not_run", "pair_instrumentation": "not_run"}
    bundle["bundle_content_sha256"] = _digest_json(bundle)
    output = _fresh_output(output)
    with (output / "bundle.json").open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return bundle


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _save(path, value):
    with Path(path).open("xb") as stream:
        torch.save(value, stream)
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _bound_load(record):
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"cache SHA256 mismatch: {record['path']}")
    return torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")


def _resolve(path, root):
    value = Path(path)
    return value if value.is_absolute() else root / value


def read_triangle_mesh(record, root):
    """Read an explicit collider mesh; never invoke an implicit convex cooker."""
    if "vertices" in record and "faces" in record:
        vertices, faces = torch.tensor(record["vertices"], dtype=torch.float64), torch.tensor(record["faces"])
        identity = {"inline_geometry_sha256": _digest_json({"vertices": record["vertices"], "faces": record["faces"]})}
    else:
        path = _resolve(record["path"], root)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") != digest:
            raise ValueError(f"collider mesh bytes differ: {path}")
        identity = {"path": str(path.resolve()), "sha256": digest}
        if path.suffix.lower() == ".pt":
            mesh = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            vertices, faces = mesh["vertices"], mesh["faces"]
        elif path.suffix.lower() == ".obj":
            vertices, faces = [], []
            for line in raw.decode("utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    indices = [int(x.split("/")[0]) for x in parts[1:]]
                    indices = [x - 1 if x > 0 else len(vertices) + x for x in indices]
                    for j in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[j], indices[j + 1]])
            vertices, faces = torch.tensor(vertices, dtype=torch.float64), torch.tensor(faces)
        elif path.suffix.lower() == ".stl":
            triangles = []
            binary_count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else -1
            if len(raw) == 84 + binary_count * 50:
                for offset in range(84, len(raw), 50):
                    values = struct.unpack_from("<12fH", raw, offset)
                    triangles.append([values[3:6], values[6:9], values[9:12]])
            else:
                values = [[float(v) for v in line.split()[1:]] for line in raw.decode("ascii").splitlines() if line.strip().startswith("vertex ")]
                if len(values) % 3:
                    raise ValueError("malformed ASCII STL")
                triangles = [values[i:i + 3] for i in range(0, len(values), 3)]
            values = torch.tensor(triangles, dtype=torch.float64).reshape(-1, 3)
            vertices, inverse = values.unique(dim=0, return_inverse=True)
            faces = inverse.reshape(-1, 3)
        else:
            raise ValueError("collider must be explicit triangle OBJ/STL or weights-only PT; native USD is bound separately")
    vertices, faces = validate_mesh(vertices, faces)
    return {"vertices": vertices, "faces": faces}, identity


def _profile(profile):
    required = {"mass", "com", "inertia", "friction"}
    if not required <= profile.keys():
        raise ValueError(f"physical profile missing {sorted(required - profile.keys())}")
    mass = float(profile["mass"])
    com, inertia = torch.as_tensor(profile["com"], dtype=torch.float64), torch.as_tensor(profile["inertia"], dtype=torch.float64)
    if inertia.shape == (3,):
        inertia = torch.diag(inertia)
    if not math.isfinite(mass) or mass <= 0 or com.shape != (3,) or inertia.shape != (3, 3) or not bool(torch.isfinite(inertia).all() & torch.isfinite(com).all()):
        raise ValueError("invalid mass/COM/inertia")
    if not torch.allclose(inertia, inertia.T, atol=1e-10, rtol=0) or not bool((torch.linalg.eigvalsh(inertia) > 0).all()):
        raise ValueError("physical inertia must be symmetric positive definite in object frame")
    eigenvalues = torch.linalg.eigvalsh(inertia)
    if float(eigenvalues[-1]) > float(eigenvalues[:2].sum()) + 1e-10:
        raise ValueError("physical inertia violates the rigid-body triangle inequality")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in (profile["friction"], profile.get("ground_friction", .3))):
        raise ValueError("friction must be nonnegative")
    return {**profile, "mass": mass, "com": com.tolist(), "inertia": inertia.tolist(), "ground_friction": profile.get("ground_friction", .3)}


def _validate_palm_evidence(side, source, meshes, points, evidence, root, *, custom=False, recompute=False):
    # Custom synthetic CPU fixtures cannot establish canonical native readiness.
    if custom and evidence.get("schema") != "hoi_v1_palm_patch_evidence_v1":
        return
    if evidence.get("schema") != "hoi_v1_palm_patch_evidence_v1":
        raise ValueError("canonical calibrated palm probes require versioned ray/source/QA evidence")
    if points.shape != (16, 3):
        raise ValueError("canonical palm patch must contain all sixteen ray-surface samples")
    canonical_json = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    if hashlib.sha256(canonical_json(payload)).hexdigest() != evidence.get("evidence_sha256"):
        raise ValueError("palm selection evidence content hash differs")
    if source is None or evidence.get("source_stl_sha256") != source["sha256"] or source["sha256"] != HAND_SHA[side]:
        raise ValueError("palm evidence is not bound to the original source STL")
    if tensor_sha256(points) != evidence.get("points_sha256"):
        raise ValueError("palm evidence points hash differs")
    serial_meshes = [{"vertices": mesh["vertices"].tolist(), "faces": mesh["faces"].tolist()} for mesh in meshes]
    if hashlib.sha256(canonical_json(serial_meshes)).hexdigest() != evidence.get("collision_meshes_sha256"):
        raise ValueError("palm evidence belongs to different compiled hand colliders")
    for name in ("probes_source", "visual_qa"):
        record = evidence[name]
        if sha256_file(_resolve(record["path"], root)) != record["sha256"]:
            raise ValueError(f"palm evidence {name} bytes changed")
    definition = evidence["definition"]
    if definition.get("version") != "central_inner_palm_raycast_v1" or definition.get("palm_selection_verified") is not True:
        raise ValueError("unsupported or unverified palm patch definition")
    if definition.get("approach_axis") != ("-y" if side == "left" else "+y"):
        raise ValueError("palm patch approach axis differs from the mirrored inner palm")
    if len(definition.get("source_triangle_indices", [])) != 16 or len(definition.get("compiled_triangle_indices", [])) != 16:
        raise ValueError("palm evidence must retain all sixteen source/collider triangle identities")
    if recompute:
        from .probes import palm_patch_points
        source_mesh, _ = read_triangle_mesh(source, root)
        expected, actual_definition = palm_patch_points(meshes, [source_mesh], side)
        matches = actual_definition.keys() == definition.keys()
        for key, value in actual_definition.items():
            recorded = definition.get(key)
            if key == "source_palm_points":
                matches &= torch.allclose(torch.tensor(value), torch.tensor(recorded), atol=1e-7, rtol=0)
            elif isinstance(value, float):
                matches &= isinstance(recorded, (int, float)) and abs(value - recorded) <= 1e-7
            else:
                matches &= value == recorded
        if not torch.allclose(points, expected, atol=1e-7, rtol=0) or not matches:
            raise ValueError("palm ray/source-face mapping does not reproduce the calibration evidence")


def _hands(recipe, root, custom):
    result = {}
    for side in ("left", "right"):
        hand = recipe[side]
        source = hand.get("source_stl")
        if source is not None:
            source_path = _resolve(source["path"], root)
            if sha256_file(source_path) != source["sha256"]:
                raise ValueError("hand source STL SHA256 differs")
            source = {**source, "path": str(source_path.resolve())}
        if not custom:
            if source is None or source.get("sha256") != HAND_SHA[side]:
                raise ValueError("canonical hands require the byte-identical original rubber-hand STL")
        mount = hand.get("mount", ORIGIN[side])
        if list(mount) != ORIGIN[side]:
            raise ValueError("hand mount must preserve the verified fixed wrist transform")
        meshes, identities = [], []
        for mesh in hand["collision_meshes"]:
            value, identity = read_triangle_mesh(mesh, root)
            meshes.append(value)
            identities.append(identity)
        points = torch.as_tensor(hand["palm_points"], dtype=torch.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or not bool(torch.isfinite(points).all()):
            raise ValueError("explicit actual palm-surface points are required")
        error = collision_distance(points, meshes).abs()
        if float(error.max()) > .001:
            raise ValueError("palm probes are not on the cooked hand surface within 1mm")
        selection_verified = hand.get("palm_selection_verified", False)
        selection_evidence = hand.get("palm_selection_evidence", {})
        if type(selection_verified) is not bool or not isinstance(selection_evidence, dict):
            raise ValueError("palm selection verification must be boolean with an evidence record")
        if selection_verified and not selection_evidence:
            raise ValueError("verified palm selection requires explicit calibration evidence")
        if selection_verified:
            _validate_palm_evidence(side, source, meshes, points, selection_evidence, root, custom=custom, recompute=True)
        result[side] = {"mount": list(mount), "palm_points": points, "collision_meshes": meshes,
                        "source_stl": source, "collision_identities": identities,
                        "palm_points_sha256": tensor_sha256(points), "probe_surface_error_max_m": float(error.max()),
                        "palm_selection_verified": selection_verified, "palm_selection_evidence": selection_evidence,
                        "native_convex_evidence": hand.get("native_convex_evidence"),
                        "probe_definition": hand.get("probe_definition"), "visual_qa": hand.get("visual_qa"),
                        "recipe": hand.get("recipe", "explicit_supplied_collision_hulls")}
    if len(result["left"]["palm_points"]) != len(result["right"]["palm_points"]):
        raise ValueError("left/right palm probes require equal counts")
    return result


def _reference_contacts(reference, kin, hands, geometry):
    robot, obj = reference["robot"], reference["object"]
    S = (robot["root_pos"], robot["root_rot"], robot["derived_ref_v"], robot["derived_ref_omega"], robot["joint_pos"], robot["derived_ref_joint_qd"])
    with torch.no_grad():
        body_p, body_R, _, _ = kin.forward(*S, all_bodies=True)
    gaps = torch.empty((len(body_p), 2), dtype=torch.float32)
    penetration = torch.empty_like(gaps)
    intersections = torch.empty_like(gaps, dtype=torch.bool)
    objects = _prepare_exact_meshes(geometry["collision_meshes"])
    for side_id, side in enumerate(("left", "right")):
        index = kin.body_names.index(side + "_wrist_yaw_link")
        mount = torch.tensor(hands[side]["mount"], dtype=body_p.dtype)
        hand_queries = _prepare_exact_meshes(hands[side]["collision_meshes"])
        for frame in range(len(body_p)):
            meshes = []
            transform = obj["rot"][frame].T.double()
            for hand_mesh in hand_queries:
                world = body_p[frame, index].double() + (body_R[frame, index].double() @ (hand_mesh.vertices + mount.double()).T).T
                local = (transform @ (world - obj["pos"][frame].double()).T).T
                meshes.append(_rigid_mesh_vertices(hand_mesh, local))
            gap, depth, crossing = _exact_mesh_gap(meshes, objects, return_intersection=True)
            gaps[frame, side_id], penetration[frame, side_id] = gap, depth
            intersections[frame, side_id] = crossing
    # Crossing surfaces with any measured penetration are conservatively unknown:
    # vertex depth is not a guaranteed maximum-translation depth. This excludes
    # small overlaps as well as severe ones, rather than asserting trusted intent.
    # Thin colliders can cross with every vertex outside the opposite collider.
    # Keep the edge-face flag independently of the sampled depth. The exact
    # triangle predicate also includes some boundary touches; these remain
    # explicitly unknown rather than being declared nonpenetrating by epsilon.
    trustworthy = ~intersections & ~((gaps <= 1e-7) & (penetration > 1e-7))
    expected, valid = expected_schedule_from_distances(gaps, penetration, trustworthy)
    return {"expected_contact": expected, "expected_valid": valid, "surface_gap_m": gaps,
            "vertex_penetration_m": penetration, "surface_intersection": intersections, "geometry_valid": trustworthy,
            "metadata": {"label_schema": "reference_hand_object_geometric_intent_v2_surface_intersection_unknown", "positive_gap_m": .02,
                         "negative_gap_m": .04, "severe_vertex_penetration_m": .02,
                         "surface_query": "exact_triangle_vertex_face_edge_edge_and_intersections",
                         "penetration_policy": "all_edge_face_intersections_unknown_including_unresolved_boundary_touch_regardless_of_vertex_depth",
                         "source_39_flags": "preserved_only_never_pair_truth"}}


def _asset_binding_key(entry, bindings):
    declared = bindings["objects"].get(entry["motion"], bindings["objects"].get(entry["object_asset"], entry["object_asset"]))
    return (declared.get("binding_key") or declared.get("object_asset") or entry["object_asset"]) if isinstance(declared, dict) else declared


_CONTACT_QUALITY_COUNTS = ("valid", "positive", "negative", "unknown_total", "gray_unknown",
                           "surface_intersection", "severe_vertex_penetration", "adjacent_valid_onset", "adjacent_valid_release")


def _reference_contact_quality(contacts):
    """Counts for T interval endpoints; geometric intent is not actual pair truth."""
    expected, valid = contacts["expected_contact"][1:] > .5, contacts["expected_valid"][1:].bool()
    gaps, depth = contacts["surface_gap_m"][1:], contacts["vertex_penetration_m"][1:]
    meta = contacts["metadata"]
    connected = valid[1:] & valid[:-1]
    values = {"valid": valid, "positive": expected & valid, "negative": ~expected & valid,
              "unknown_total": ~valid,
              "gray_unknown": ~valid & (gaps > meta["positive_gap_m"]) & (gaps < meta["negative_gap_m"]),
              "surface_intersection": contacts["surface_intersection"][1:],
              "severe_vertex_penetration": depth >= meta["severe_vertex_penetration_m"],
              "adjacent_valid_onset": connected & expected[1:] & ~expected[:-1],
              "adjacent_valid_release": connected & ~expected[1:] & expected[:-1]}
    return {"interval_endpoints": len(expected), **{key: value.sum(0).tolist() for key, value in values.items()}}


def _aggregate_reference_contact_quality(clips, assets):
    def total(selected):
        return {"motions": len(selected), "interval_endpoints": sum(c["reference_contact_quality"]["interval_endpoints"] for c in selected),
                **{key: [sum(c["reference_contact_quality"][key][side] for c in selected) for side in range(2)]
                   for key in _CONTACT_QUALITY_COUNTS}}
    return {"schema_version": "hoi_v1_reference_contact_quality_v1", "hand_order": ["left", "right"],
            "definition": "contacts[1:] are T geometric-intent interval endpoints; onset/release require adjacent valid endpoints; unknown reason counts may overlap; no actual pair truth or subjective pass threshold",
            "whole_partition": total(clips),
            "by_asset": [{"asset_id": asset["asset_id"], "object_asset": asset["object_asset"],
                           **total([c for c in clips if c["asset_id"] == asset["asset_id"]])} for asset in assets]}


def _prepare_clip(index, entry, path, *, bindings, manifest_sha, lookup, kin, hands, geometries, output):
    started = time.perf_counter()
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["motion_sha256"]:
        raise ValueError("source motion SHA256 mismatch")
    frames = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    reference = adapt_motion(frames, bindings["source_joint_names"], bindings["target_joint_names"])
    _assert_adapted(reference, len(frames))
    key = _asset_binding_key(entry, bindings)
    if key not in lookup:
        raise ValueError(f"missing exact asset/scale binding for {entry['motion']}")
    asset_id = lookup[key]
    reference["provenance"] = {"source_motion": entry["motion"], "source_motion_sha256": digest,
                               "manifest_content_sha256": manifest_sha, "source_group": entry["source_group"]}
    adapted = time.perf_counter()
    contacts = _reference_contacts(reference, kin, hands, geometries[asset_id])
    queried = time.perf_counter()
    record = {"name": Path(entry["motion"]).stem, "source_group": entry["source_group"], "source_motion": entry["motion"],
            "resolved_source_motion": str(Path(path).resolve()), "source_motion_sha256": digest, "asset_id": asset_id,
            "state_frames": reference["metadata"]["state_frames"], "source_frames": len(frames),
            "reference_contact_quality": _reference_contact_quality(contacts),
            "reference_cache": _save(Path(output) / f"reference_{index:05d}.pt", reference),
            "contact_cache": _save(Path(output) / f"contact_{index:05d}.pt", contacts)}
    record["timing_seconds"] = {"source_read_verify_adapt": adapted - started, "exact_contact_geometry": queried - adapted,
                                "cache_write": time.perf_counter() - queried}
    LOGGER.info("clip_complete index=%d asset=%d states=%d timing_seconds=%s", index, asset_id,
                record["state_frames"], json.dumps(record["timing_seconds"], sort_keys=True))
    return record


_PREPARATION_WORKER = None


def _initialize_preparation_worker(context):
    global _PREPARATION_WORKER
    if context.pop("_log_progress", False):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    torch.set_num_threads(1)
    _PREPARATION_WORKER = context


def _prepare_worker_task(task):
    return _prepare_clip(*task, **_PREPARATION_WORKER)


def prepare_bundle(manifest_path, recipes_path, output, kin, *, partition="train", limit=None, path_maps=(), jobs=1,
                   one_per_asset_resource_fixture=False):
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer CPU process count")
    manifest = read_manifest(manifest_path)
    recipe_path = Path(recipes_path).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    root = recipe_path.parent
    custom = manifest["profile"] == "custom"
    if not custom and manifest["validation"]["train_selection"] != "verified_exact_train_identities":
        raise ValueError("canonical preparation requires verified frozen training selection")
    entries = manifest["partitions"].get(partition, manifest["panels"].get(partition))
    if entries is None:
        raise ValueError("unknown partition")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = entries if limit is None else entries[:limit]
    if one_per_asset_resource_fixture:
        if limit is not None or partition != "train":
            raise ValueError("one-per-asset resource fixture requires train partition without --limit")
        required = {asset.get("binding_key", asset["object_asset"]) for asset in recipe["assets"]}
        selected, seen = [], set()
        for entry in entries:
            key = _asset_binding_key(entry, manifest["bindings"])
            if key in required and key not in seen:
                selected.append(entry)
                seen.add(key)
        if seen != required:
            raise ValueError(f"resource fixture lacks train motions for asset bindings: {sorted(required - seen)}")
    mappings = tuple(path_maps) + tuple(manifest["path_maps"])
    paths = [resolve_motion_path(e["motion"], mappings) for e in selected]
    output = _fresh_output(output, protected_files=(manifest_path, recipes_path), source_roots=[p.parent for p in paths])
    bindings = manifest["bindings"]
    if bindings["source_joint_names"] is None:
        raise ValueError("source joint-name evidence is required")
    hands = _hands(recipe["hands"], root, custom)
    hand_cache = _save(output / "hand_geometry.pt", hands)
    assets, geometries = [], []
    for asset_id, entry in enumerate(recipe["assets"]):
        geometry_started = time.perf_counter()
        LOGGER.info("geometry_start asset=%d name=%s components=%d", asset_id, entry["object_asset"], len(entry["collision_meshes"]))
        profile = _profile(entry["physical_profile"])
        scale = torch.as_tensor(entry["exact_scale"], dtype=torch.float64)
        if scale.numel() == 1:
            scale = scale.repeat(3)
        transform = torch.as_tensor(entry["mesh_to_object"], dtype=torch.float64)
        if scale.shape != (3,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()) or transform.shape != (4, 4):
            raise ValueError("exact scale3 and mesh_to_object4x4 are required")
        if not bool(torch.isfinite(transform).all()) or not torch.allclose(transform[3], transform.new_tensor([0, 0, 0, 1])):
            raise ValueError("invalid mesh_to_object homogeneous transform")
        rotation = transform[:3, :3]
        if not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=rotation.dtype), atol=1e-7, rtol=0) or float(torch.det(rotation)) < 0:
            raise ValueError("mesh_to_object rotation must be proper; scale belongs in exact_scale")
        native = _resolve(entry["native_asset"], root)
        native_sha = sha256_file(native)
        if native_sha != entry["native_asset_sha256"]:
            raise ValueError("native asset SHA256 mismatch")
        meshes, mesh_ids = [], []
        for mesh_record in entry["collision_meshes"]:
            mesh, identity = read_triangle_mesh(mesh_record, root)
            frame = mesh_record.get("frame", "mesh")
            if frame == "mesh":
                mesh["vertices"] = (rotation @ (mesh["vertices"] * scale).T).T + transform[:3, 3]
            elif frame != "object_local":
                raise ValueError("collision mesh frame must be mesh or object_local")
            meshes.append(mesh)
            mesh_ids.append({**identity, "frame": frame})
        collision_hash = _digest_json({"mesh_identities": mesh_ids, "exact_scale": scale.tolist(), "mesh_to_object": transform.tolist(),
                                       "cooking_recipe": entry.get("cooking_recipe", "explicit_meshes_no_recook")})
        if "collision_hash" in entry and entry["collision_hash"] != collision_hash:
            raise ValueError("declared collision recipe hash differs")
        geometry = build_geometry_cache(meshes, **entry.get("distance_recipe", {}), metadata={"collision_hash": collision_hash,
                                               "mesh_to_object": transform.tolist(), "exact_scale": scale.tolist(), "mesh_identities": mesh_ids})
        record = {**entry, "asset_id": asset_id, "native_asset": str(native.resolve()), "native_asset_sha256": native_sha,
                  "physical_profile": profile, "collision_hash": collision_hash,
                  "geometry_cache": _save(output / f"geometry_{asset_id:04d}.pt", geometry)}
        assets.append(record)
        geometries.append(geometry)
        record["geometry_preparation_seconds"] = time.perf_counter() - geometry_started
        LOGGER.info("geometry_complete asset=%d elapsed_seconds=%.6f", asset_id, record["geometry_preparation_seconds"])
    lookup = {}
    for asset in assets:
        key = asset.get("binding_key", asset["object_asset"])
        if key in lookup:
            raise ValueError("duplicate asset binding_key; each exact scale requires an unambiguous key")
        lookup[key] = asset["asset_id"]
    context = {"bindings": bindings, "manifest_sha": manifest["manifest_content_sha256"], "lookup": lookup,
               "kin": kin, "hands": hands, "geometries": [{"collision_meshes": g["collision_meshes"]} for g in geometries], "output": str(output)}
    tasks = [(index, entry, str(path)) for index, (entry, path) in enumerate(zip(selected, paths))]
    if jobs == 1:
        clips = [_prepare_clip(*task, **context) for task in tasks]
    else:
        # map retains the frozen manifest order regardless of completion order.
        # Children only read source assets and exclusively create their own files.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_preparation_worker,
                                 initargs=({**context, "_log_progress": LOGGER.isEnabledFor(logging.INFO)},)) as executor:
            clips = list(executor.map(_prepare_worker_task, tasks, chunksize=1))
    bundle = {"schema_version": BUNDLE_SCHEMA, "partition": partition, "timebase": manifest["timebase"],
              "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": sha256_file(manifest_path), "content_sha256": manifest["manifest_content_sha256"]},
              "recipes": {"path": str(recipe_path), "sha256": sha256_file(recipe_path)}, "assets": assets,
              "hands": {"geometry_cache": hand_cache}, "clips": clips,
              "reference_contact_quality": _aggregate_reference_contact_quality(clips, assets),
              "scope": {"available_motions": len(entries), "selected_motions": len(selected),
                        "is_subset": one_per_asset_resource_fixture or len(selected) != len(entries),
                        "purpose": "one_per_asset_resource_fixture" if one_per_asset_resource_fixture else "reference_bundle",
                        "selection": "first_motion_per_exact_asset_in_frozen_manifest_order" if one_per_asset_resource_fixture else "frozen_manifest_prefix"},
              "preparation": {"cpu_processes": jobs, "worker_torch_threads": 1 if jobs > 1 else torch.get_num_threads(),
                              "clip_order": "frozen_manifest_order_independent_of_completion_order"},
              "validation": {"cpu_data_geometry": "verified", "native_physics": "not_run", "pair_instrumentation": "not_run",
                             "canonical_hand_identity": not custom, "training_started": False}}
    if hasattr(kin, "mjcf_path"):
        bundle["robot_fk"] = {"mjcf_path": str(Path(kin.mjcf_path).resolve()), "mjcf_sha256": sha256_file(kin.mjcf_path),
                              "tracked_bodies": [kin.body_names[index] for index in kin.tracked_idx]}
    robot_asset = recipe.get("native_robot_asset")
    if robot_asset is None and recipe.get("hand_runtime_report"):
        report = recipe["hand_runtime_report"]
        robot_asset = {"path": report["usd"], "sha256": report["usd_sha256"]}
    if robot_asset is not None:
        native_robot = _resolve(robot_asset["path"], root)
        bundle["native_robot_asset"] = native_robot_asset_binding(native_robot, robot_asset["sha256"])
    elif not custom:
        raise ValueError("canonical bundle must bind the exact cooked native robot/hand overlay; workers must not recook hands")
    if "reference_kinematics" in recipe:
        virtual = recipe["reference_kinematics"]
        body_names, dof_names = virtual["body_names"], virtual["dof_names"]
        if len(body_names) != 39 or len(set(body_names)) != 39 or set(dof_names) != set(bindings["source_joint_names"]) or len(dof_names) != 29:
            raise ValueError("reference kinematics must bind the original 39-body and 29-DOF name orders")
        urdf = _resolve(virtual["urdf"]["path"], root)
        if sha256_file(urdf) != virtual["urdf"]["sha256"]:
            raise ValueError("reference virtual-URDF SHA256 differs")
        bundle["reference_kinematics"] = {**virtual, "urdf": {"path": str(urdf.resolve()), "sha256": virtual["urdf"]["sha256"]}}
    bundle["bundle_content_sha256"] = _digest_json(bundle)
    with (output / "bundle.json").open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return bundle


def load_bundle(path, kin, device="cpu", require_train=True, asset_filter=None, verify_source=True,
                retain_source_reference=None):
    path = Path(path)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    digest = bundle.pop("bundle_content_sha256", None)
    if bundle.get("schema_version") != BUNDLE_SCHEMA or digest != _digest_json(bundle):
        raise ValueError("bundle schema/content hash mismatch")
    bundle["bundle_content_sha256"] = digest
    if require_train and bundle["partition"] != "train":
        raise ValueError("only train partition may supply replay/normalizer/learner data")
    if "robot_fk" in bundle:
        robot = bundle["robot_fk"]
        if not hasattr(kin, "mjcf_path") or sha256_file(kin.mjcf_path) != robot["mjcf_sha256"]:
            raise ValueError("reference and runtime robot FK MJCF identity differ")
    if "native_robot_asset" in bundle:
        native_robot = bundle["native_robot_asset"]
        if sha256_file(native_robot["path"]) != native_robot["sha256"]:
            raise ValueError("bound native robot/hand overlay bytes changed")
        record = native_robot["hand_asset_manifest"]
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("bound native hand-asset receipt bytes changed")
        if "foot_asset_manifest" in native_robot:
            record = native_robot["foot_asset_manifest"]
            if sha256_file(record["path"]) != record["sha256"]:
                raise ValueError("bound native foot-asset receipt bytes changed")
            if native_robot_asset_binding(native_robot["path"], native_robot["sha256"]) != native_robot:
                raise ValueError("bound native foot overlay identity differs")
        elif (Path(native_robot["path"]).parent / "foot-asset.json").is_file():
            raise ValueError("native foot extension requires its explicit manifest binding")
    if "reference_kinematics" in bundle:
        virtual = bundle["reference_kinematics"]
        if sha256_file(virtual["urdf"]["path"]) != virtual["urdf"]["sha256"]:
            raise ValueError("bound virtual reference URDF bytes changed")
    if sha256_file(bundle["manifest"]["path"]) != bundle["manifest"]["sha256"]:
        raise ValueError("source manifest bytes changed")
    if sha256_file(bundle["recipes"]["path"]) != bundle["recipes"]["sha256"]:
        raise ValueError("source collision recipe bytes changed")
    manifest = read_manifest(bundle["manifest"]["path"])
    if manifest["manifest_content_sha256"] != bundle["manifest"]["content_sha256"]:
        raise ValueError("source manifest identity changed")
    partition_entries = manifest["partitions"].get(bundle["partition"], manifest["panels"].get(bundle["partition"]))
    if partition_entries is None:
        raise ValueError("bundle partition is not in the bound manifest")
    identities = {(entry["motion"], entry["motion_sha256"]): entry for entry in partition_entries}
    seen = set()
    for entry in bundle["clips"]:
        identity = (entry["source_motion"], entry["source_motion_sha256"])
        if identity not in identities or identity in seen or entry["source_group"] != identities[identity]["source_group"]:
            raise ValueError("bundle clip is duplicated or does not belong to its declared partition")
        seen.add(identity)
    selected_assets = None if asset_filter is None else set(int(x) for x in asset_filter)
    clips, global_frames, offset = [], [], 0
    for entry in bundle["clips"]:
        n = entry["state_frames"]
        if selected_assets is not None and entry["asset_id"] not in selected_assets:
            offset += n
            continue
        if verify_source and sha256_file(entry["resolved_source_motion"]) != entry["source_motion_sha256"]:
            raise ValueError("source motion bytes changed since bundle preparation")
        reference, contacts = _bound_load(entry["reference_cache"]), _bound_load(entry["contact_cache"])
        if len(reference["robot"]["root_pos"]) != n:
            raise ValueError("bundle state_frames differs from reference cache")
        if reference["provenance"]["source_motion_sha256"] != entry["source_motion_sha256"]:
            raise ValueError("reference cache belongs to another source")
        if reference["provenance"]["manifest_content_sha256"] != bundle["manifest"]["content_sha256"]:
            raise ValueError("reference cache belongs to another manifest")
        _assert_adapted(reference, entry["source_frames"])
        clips.append({**entry, "reference": reference, "expected_contact": contacts["expected_contact"], "expected_valid": contacts["expected_valid"]})
        global_frames.append(torch.arange(offset, offset + n, dtype=torch.long, device=device))
        offset += n
    bank = TrainingMotionBank(clips, kin, device, bundle["partition"], retain_source_reference=retain_source_reference)
    bank.global_frame_index = torch.cat(global_frames)
    bank.asset_id = bank.asset_ids
    geometries = []
    for asset in bundle["assets"]:
        if sha256_file(asset["native_asset"]) != asset["native_asset_sha256"]:
            raise ValueError("bound native asset bytes changed")
        # Geometry remains CPU until points/individual SDF queries are requested;
        # worker filters therefore do not duplicate all grids on each GPU.
        geometry = _bound_load(asset["geometry_cache"])
        if geometry["metadata"]["collision_hash"] != asset["collision_hash"] or tensor_sha256(geometry["points"]) != geometry["metadata"]["point_hash"]:
            raise ValueError("geometry cache metadata/hash differs from its asset binding")
        geometries.append(geometry)
    hands = _bound_load(bundle["hands"]["geometry_cache"])
    for side in ("left", "right"):
        hand = hands[side]
        if hand["mount"] != ORIGIN[side] or tensor_sha256(hand["palm_points"]) != hand["palm_points_sha256"]:
            raise ValueError("hand mounting transform/probe hash changed")
        if hand.get("source_stl") is not None:
            source = hand["source_stl"]
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("bound hand source STL bytes changed")
        if hand.get("palm_selection_verified"):
            _validate_palm_evidence(side, hand.get("source_stl"), hand["collision_meshes"], hand["palm_points"],
                                    hand["palm_selection_evidence"], Path(bundle["recipes"]["path"]).parent,
                                    custom=not bundle["validation"]["canonical_hand_identity"])
    shapes = ShapeAssetBank(bundle["assets"], geometries, hands, device)
    return bank, shapes, bundle


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True, help="explicit collider/physical/hand recipe JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", default="train", choices=("train", "eval80", "test40", "validation", "regression", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--one-per-asset-resource-fixture", action="store_true",
                        help="explicit nonformal train subset: first real motion per exact asset, bound to the unchanged canonical manifest")
    parser.add_argument("--robot-mjcf", help="explicit robot FK asset; original default is preserved")
    parser.add_argument("--jobs", type=int, default=1, help="explicit CPU processes for clip adaptation/contact labels; each child uses one Torch thread")
    args = parser.parse_args(argv)
    from supermimic.kinematics import G1Kinematics
    bundle = prepare_bundle(args.manifest, args.assets, args.output, G1Kinematics("cpu", mjcf_path=args.robot_mjcf), partition=args.partition,
                            limit=args.limit, jobs=args.jobs, one_per_asset_resource_fixture=args.one_per_asset_resource_fixture)
    print(json.dumps({"bundle": str(Path(args.output).resolve() / "bundle.json"), "scope": bundle["scope"], "validation": bundle["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
