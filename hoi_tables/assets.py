"""Bind existing table convex geometry, SDF/FPS cache and inertial assumptions.

Only object assets are inherited from the historical bundle. Its robot, labels,
data split, and native-acceptance claims do not apply to the new hw3 task.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil

import torch

from hoi_residual.assets import (_digest_json, _triangles, prepare_object_usd,
                                 sample_surface_points, sha256_file,
                                 validate_physical_profile)
from hoi_residual.shape import point_sha256
from .data import ASSETS

CATALOG_SCHEMA = "hoi_tables_object_catalog_v1"
SHAPE_SCHEMA = "hoi_tables_fps128_bank_v1"


def load_assets(path, *, verify=True):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = value.pop("catalog_content_sha256",None)
    if value.get("schema_version") != CATALOG_SCHEMA or declared != _digest_json(value):
        raise ValueError("tables object catalog schema/content mismatch")
    assets = deepcopy(value["assets"])
    if [a["object_asset"] for a in assets] != list(ASSETS) or [a["asset_id"] for a in assets] != [0,1]:
        raise ValueError("tables catalog requires canonical largetable/smalltable IDs 0/1")
    for asset in assets:
        validate_physical_profile(asset["physical_profile"])
        _triangles(asset)
        for field in ("native_asset","geometry_cache"):
            binding = asset[field] if isinstance(asset[field],dict) else {"path":asset[field],"sha256":asset[field+"_sha256"]}
            source = Path(binding["path"])
            if not source.is_absolute(): source = path.parent/source
            if field == "native_asset": asset[field] = str(source.resolve())
            else: asset[field]["path"] = str(source.resolve())
            if verify and sha256_file(source) != binding["sha256"]:
                raise ValueError(f"table {field} bytes changed")
    return assets


def load_geometry_cache(asset, device="cpu"):
    path = Path(asset["geometry_cache"]["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset["geometry_cache"]["sha256"]:
        raise ValueError("table SDF/FPS cache SHA256 mismatch")
    cache = torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    metadata = cache["metadata"]
    if metadata["collision_hash"] != asset["collision_hash"] or metadata["exact_scale"] != asset["exact_scale"]:
        raise ValueError("table cache collision identity differs from native asset")
    if len(cache["collision_meshes"]) != len(asset["collision_meshes"]):
        raise ValueError("table cache collision component count differs")
    for cached,bound in zip(cache["collision_meshes"],asset["collision_meshes"]):
        for key,dtype in (("vertices",torch.float64),("faces",torch.int64)):
            left,right = torch.as_tensor(cached[key],dtype=dtype),torch.as_tensor(bound[key],dtype=dtype)
            if left.shape != right.shape or not torch.equal(left,right):
                raise ValueError(f"table cache {key} differ from bound collision geometry")
    if metadata.get("frame") != "object_link_local_metres" or metadata.get("point_count") != 128:
        raise ValueError("unsupported table geometry sampling/frame")
    resolution = metadata.get("grid_resolution")
    if (type(resolution) is not int or resolution < 2 or
            tuple(cache["sdf_grid"].shape) != (resolution,resolution,resolution) or
            cache["grid_min"].shape != (3,) or cache["grid_max"].shape != (3,) or
            not (cache["grid_max"] > cache["grid_min"]).all() or
            metadata.get("grid_axis_order") != "zyx" or
            metadata.get("sdf_sign") != "negative_inside_collider_union"):
        raise ValueError("invalid table SDF grid contract")
    points = cache["points"]
    if points.dtype != torch.float32 or points.shape != (128,3) or not torch.isfinite(points).all() or point_sha256(points) != metadata["point_hash"]:
        raise ValueError("invalid bound table FPS128 points")
    for name in ("sdf_grid","grid_min","grid_max","bbox_diagonal","points"):
        if not torch.isfinite(cache[name]).all(): raise ValueError("nonfinite table geometry cache")
        cache[name] = cache[name].to(device)
    return cache


def prepare_assets(source_bundle, output):
    """Export a two-table catalog without recooking or rescaling geometry."""
    source_bundle,output = Path(source_bundle),Path(output)
    raw = source_bundle.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw)
    declared = source.pop("bundle_content_sha256",None)
    if declared != _digest_json(source): raise ValueError("historical source bundle content mismatch")
    selected = []
    for index,name in enumerate(ASSETS):
        matches = [a for a in source["assets"] if a.get("physical_evidence",{}).get("source_asset_id") == "omomo/"+name]
        if len(matches) != 1: raise ValueError(f"source bundle lacks unambiguous {name}")
        asset = deepcopy(matches[0])
        asset["source_asset_id"],asset["source_object_asset"] = asset["asset_id"],asset["object_asset"]
        asset["asset_id"],asset["object_asset"] = index,name
        asset["tables_binding"] = {"source_scale":.83,"native_geometry_already_scaled":True,
                                  "additional_runtime_mesh_scale":1.,"trajectory_scale_applied_by_source":True,
                                  "old_robot_or_contact_labels_reused":False}
        # The current robot/object pairing needs its own native smoke validation.
        asset["native_validation_status"] = "new_hw3_scene_smoke_required"
        asset["training_ready"] = False
        cache = asset["geometry_cache"]
        if not Path(cache["path"]).is_absolute(): cache["path"] = str(source_bundle.parent/cache["path"])
        if not Path(asset["native_asset"]).is_absolute(): asset["native_asset"] = str(source_bundle.parent/asset["native_asset"])
        validate_physical_profile(asset["physical_profile"])
        if sha256_file(asset["native_asset"]) != asset["native_asset_sha256"]:
            raise ValueError("historical table native asset SHA256 mismatch")
        load_geometry_cache(asset)
        destination = output/f"geometry_{name}.pt"
        shutil.copyfile(cache["path"],destination)
        if sha256_file(destination) != cache["sha256"]:
            raise RuntimeError("source geometry cache changed while copying")
        asset["source_geometry_cache"] = deepcopy(cache)
        asset["geometry_cache"] = {"path":str(destination.resolve()),"sha256":sha256_file(destination)}
        selected.append(asset)
    if sha256_file(source_bundle) != source_sha:
        raise RuntimeError("source object bundle changed during table preparation")
    catalog = {"schema_version":CATALOG_SCHEMA,"assets":selected,
               "source_bundle":{"path":str(source_bundle.resolve()),"sha256":source_sha},
               "hands":deepcopy(source.get("hands",{})),
               "protection":"object_only_reuse_no_old_robot_labels_or_native_acceptance",
               "physical_assumptions":"inherited_user_class_mass; geometry_derived_COM_and_inertia_not_measured"}
    catalog["catalog_content_sha256"] = _digest_json(catalog)
    path = output/"assets.json"
    path.write_text(json.dumps(catalog,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    bank = {"schema":SHAPE_SCHEMA,"catalog_sha256":sha256_file(path),
            "frame":"object_link_local_metres","scale_normalization":False,"assets":[]}
    for asset in selected:
        cache = load_geometry_cache(asset)
        bank["assets"].append({"asset_id":asset["asset_id"],"object_asset":asset["object_asset"],
                              "collision_hash":asset["collision_hash"],"points":cache["points"].tolist(),
                              "point_sha256":point_sha256(cache["points"])})
    bank["content_sha256"] = _digest_json(bank)
    (output/"shape_bank.json").write_text(json.dumps(bank,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return catalog


def load_shape_bank(path, catalog):
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    declared = bank.pop("content_sha256",None)
    if bank.get("schema") != SHAPE_SCHEMA or declared != _digest_json(bank) or bank["catalog_sha256"] != sha256_file(catalog):
        raise ValueError("table shape bank identity mismatch")
    assets = load_assets(catalog)
    if [a["object_asset"] for a in bank["assets"]] != [a["object_asset"] for a in assets]:
        raise ValueError("shape bank asset order differs")
    points = []
    for entry,asset in zip(bank["assets"],assets):
        point = torch.tensor(entry["points"],dtype=torch.float32)
        if point.shape != (128,3) or not torch.isfinite(point).all() or point_sha256(point) != entry["point_sha256"] or entry["collision_hash"] != asset["collision_hash"]:
            raise ValueError("table shape bank points differ")
        points.append(point)
    return torch.stack(points),{"bank_sha256":sha256_file(path),"content_sha256":declared,"catalog_sha256":sha256_file(catalog)}
