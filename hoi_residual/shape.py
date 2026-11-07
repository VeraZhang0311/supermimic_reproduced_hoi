"""Reuse the bound v1.1 FPS128 policy geometry without changing any collider.

The old geometry cache is a read-only source of points, not of robot code,
physics, rewards or trajectories. Export only its exact float32 FPS128 points
after checking the cache, mesh and coordinate contracts. Training consumes the
small self-contained JSON bank and never imports the old HOI implementation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .assets import load_assets, sha256_file


SCHEMA = "hoi_residual_fps128_shape_bank_v1"
CONTRACT = {
    "frame": "object_link_local_metres",
    "point_count": 128,
    "sampling": "area_weighted_surface_candidates_then_deterministic_FPS",
    "candidate_count": 4096,
    "seed": 0,
    "scale_normalization": False,
    "source": "exact_bound_v1_geometry_cache_points",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def point_sha256(points):
    """The unchanged v1 geometry tensor hash, including shape and dtype."""
    points = points.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(points.shape), str(points.dtype))).encode()
                          + points.numpy().tobytes()).hexdigest()


def _points(points):
    if (not isinstance(points, torch.Tensor) or points.shape != (128, 3)
            or points.dtype != torch.float32 or not bool(torch.isfinite(points).all())):
        raise ValueError("policy shape requires finite float32 FPS128 points [128,3]")
    return points.detach().cpu().contiguous()


def _catalog(path):
    path = Path(path)
    digest = sha256_file(path)
    # Geometry binding is checked here; native USD bytes are independently
    # checked by the normal training asset loader before simulator creation.
    assets = load_assets(path, verify=False)
    if sha256_file(path) != digest:
        raise RuntimeError("object catalog changed while reading shape bindings")
    return assets, digest


def _matching_meshes(cached, bound):
    if not isinstance(cached, list) or len(cached) != len(bound):
        raise ValueError("FPS cache collision component count differs from target")
    for actual, expected in zip(cached, bound):
        for field, dtype in (("vertices", torch.float64), ("faces", torch.int64)):
            if not torch.equal(torch.as_tensor(actual[field], dtype=dtype),
                               torch.as_tensor(expected[field], dtype=dtype)):
                raise ValueError(f"FPS cache collision {field} differ from target")


def prepare_shape_bank(source_bundle, catalog, output):
    """Write a new bank exclusively, preserving the verified v1 point bytes."""
    source_bundle, catalog, output = map(Path, (source_bundle, catalog, output))
    sources, source_sha = _catalog(source_bundle)
    targets, catalog_sha = _catalog(catalog)
    if [a["object_asset"] for a in sources] != [a["object_asset"] for a in targets]:
        raise ValueError("shape source and current catalog asset names differ")
    entries = []
    for source, target in zip(sources, targets):
        for key in ("collision_hash", "exact_scale", "collision_meshes"):
            if source[key] != target[key]:
                raise ValueError(f"shape source and current catalog differ: {key}")
        binding = source.get("geometry_cache", {})
        cache_path = Path(binding["path"])
        if not cache_path.is_absolute():
            cache_path = source_bundle.parent / cache_path
        if sha256_file(cache_path) != binding["sha256"]:
            raise ValueError("FPS geometry cache SHA256 mismatch")
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if sha256_file(cache_path) != binding["sha256"]:
            raise RuntimeError("FPS geometry cache changed while reading")
        metadata = cached["metadata"]
        for key in ("frame", "point_count", "sampling", "candidate_count", "seed"):
            if metadata.get(key) != CONTRACT[key]:
                raise ValueError(f"FPS geometry metadata mismatch: {key}")
        for key in ("collision_hash", "exact_scale"):
            if metadata.get(key) != target[key]:
                raise ValueError(f"FPS geometry identity mismatch: {key}")
        _matching_meshes(cached["collision_meshes"], target["collision_meshes"])
        points = _points(cached["points"])
        digest = point_sha256(points)
        if metadata.get("point_hash") != digest:
            raise ValueError("FPS geometry point tensor hash mismatch")
        entries.append({"object_asset": target["object_asset"],
                        "collision_hash": target["collision_hash"],
                        "exact_scale": target["exact_scale"],
                        "point_sha256": digest, "points": points.tolist(),
                        "source_cache": {"path": str(cache_path.resolve()),
                                         "sha256": binding["sha256"]}})
    bank = {"schema": SCHEMA, "contract": deepcopy(CONTRACT),
            "catalog_sha256": catalog_sha, "source_bundle_sha256": source_sha,
            "source_bundle": str(source_bundle.resolve()), "assets": entries}
    bank["content_sha256"] = _digest(bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(bank, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return load_shape_bank(output, catalog)[1]


def load_shape_bank(path, catalog):
    """Return CPU points [8,128,3] and immutable-input identity for checkpoints."""
    path = Path(path)
    bank_sha = sha256_file(path)
    bank = json.loads(path.read_text(encoding="utf-8"))
    content_sha = bank.pop("content_sha256", None)
    if content_sha is None or _digest(bank) != content_sha:
        raise ValueError("FPS shape bank content hash mismatch")
    if bank.get("schema") != SCHEMA or bank.get("contract") != CONTRACT:
        raise ValueError("FPS shape bank schema/contract mismatch")
    assets, catalog_sha = _catalog(catalog)
    if bank.get("catalog_sha256") != catalog_sha:
        raise ValueError("FPS shape bank current object catalog identity mismatch")
    entries = bank.get("assets", [])
    if [entry.get("object_asset") for entry in entries] != [a["object_asset"] for a in assets]:
        raise ValueError("FPS shape bank canonical asset order mismatch")
    points = []
    for entry, asset in zip(entries, assets):
        for key in ("collision_hash", "exact_scale"):
            if entry.get(key) != asset[key]:
                raise ValueError(f"FPS shape bank object identity mismatch: {key}")
        point = _points(torch.tensor(entry["points"], dtype=torch.float32))
        if point_sha256(point) != entry.get("point_sha256"):
            raise ValueError("FPS shape bank point tensor hash mismatch")
        points.append(point)
    if sha256_file(path) != bank_sha:
        raise RuntimeError("FPS shape bank changed while reading")
    metadata = {"bank_sha256": bank_sha, "content_sha256": content_sha,
                "catalog_sha256": catalog_sha, "contract": deepcopy(CONTRACT),
                "assets": [a["object_asset"] for a in assets],
                "point_hashes": [entry["point_sha256"] for entry in entries]}
    return torch.stack(points), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = prepare_shape_bank(args.source_bundle, args.catalog, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
