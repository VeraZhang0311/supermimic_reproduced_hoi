"""Read bound ULTRA object geometry without importing the former HOI robot.

The legacy bundle is used only as an immutable object-asset catalog. Its robot,
reference FK, rewards, train clips and native-validation claims are not reused.
The current GMT robot and physics must be supplied by the caller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


CATALOG_SCHEMA = "hoi_residual_ultra_object_catalog_v1"
EXPECTED_ASSETS = frozenset(
    f"{kind}_{scale}_{scale}_{scale}"
    for kind in ("largebox", "smallbox", "plasticbox", "suitcase")
    for scale in ("100", "080")
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_physical_profile(profile):
    """Return nominal mass, COM, inertia and native sliding coefficients."""
    mass = float(profile["mass"])
    com = torch.as_tensor(profile["com"], dtype=torch.float64)
    inertia = torch.as_tensor(profile["inertia"], dtype=torch.float64).reshape(3, 3)
    friction, ground = float(profile["friction"]), float(profile["ground_friction"])
    if com.shape != (3,) or not torch.isfinite(torch.cat(
            (com, inertia.flatten(), torch.tensor([mass, friction, ground])))).all():
        raise ValueError("object profile contains invalid/nonfinite mass, COM or inertia")
    if mass <= 0 or friction < 0 or ground < 0:
        raise ValueError("object mass must be positive and friction nonnegative")
    if not torch.allclose(inertia, inertia.T, rtol=0, atol=1e-10):
        raise ValueError("object inertia must be symmetric")
    eig = torch.linalg.eigvalsh(inertia)
    if (eig <= 0).any() or eig[-1] > eig[:2].sum() + 1e-8:
        raise ValueError("object inertia is not physically valid")
    return mass, com, inertia, friction, ground


def _triangles(asset):
    triangles = []
    for mesh in asset["collision_meshes"]:
        if mesh.get("frame") != "object_local":
            raise ValueError("object geometry must already be in the object local frame")
        vertices = torch.as_tensor(mesh["vertices"], dtype=torch.float64)
        faces = torch.as_tensor(mesh["faces"], dtype=torch.long)
        if (vertices.ndim != 2 or vertices.shape[-1] != 3 or
                faces.ndim != 2 or faces.shape[-1] != 3 or len(faces) == 0 or
                not torch.isfinite(vertices).all() or faces.min() < 0 or
                faces.max() >= len(vertices)):
            raise ValueError("invalid bound object triangle mesh")
        triangles.append(vertices[faces])
    if not triangles:
        raise ValueError("object has no bound collision geometry")
    return torch.cat(triangles)


def load_assets(bundle, *, verify=True):
    """Load all eight objects, verifying the catalog and (by default) USD bytes.

    ``verify=False`` supports local geometry inspection without access to remote
    USD paths. It still checks the entire bundle digest and all object metadata;
    ``prepare_object_usd`` always rechecks source USD identity.
    """
    path = Path(bundle)
    value = json.loads(path.read_text(encoding="utf-8"))
    schema = value.get("schema_version")
    if schema == "supermimic_hoi_v1_bundle_v1":
        hash_key = "bundle_content_sha256"
    elif schema == CATALOG_SCHEMA:
        hash_key = "catalog_content_sha256"
    else:
        raise ValueError("unsupported source object catalog schema")
    declared = value.pop(hash_key, None)
    if declared is None or _digest_json(value) != declared:
        raise ValueError("object catalog content hash mismatch")
    assets = deepcopy(value["assets"])
    names = [a["object_asset"] for a in assets]
    if len(assets) != 8 or set(names) != EXPECTED_ASSETS:
        raise ValueError("ULTRA catalog must contain the exact four classes/eight assets")
    if sorted(a["asset_id"] for a in assets) != list(range(8)):
        raise ValueError("object catalog asset IDs must be unique 0..7")
    for asset in assets:
        validate_physical_profile(asset["physical_profile"])
        _triangles(asset)
        source = Path(asset["native_asset"])
        if not source.is_absolute():
            source = path.parent / source
            asset["native_asset"] = str(source.resolve())
        if verify and sha256_file(source) != asset["native_asset_sha256"]:
            raise ValueError(f"native object asset bytes changed: {asset['object_asset']}")
    # The old bundle uses preparation order, while UltraMotionLib uses sorted
    # object names. Preserve that provenance but expose one shared ID space.
    assets.sort(key=lambda asset: asset["object_asset"])
    for asset_id, asset in enumerate(assets):
        asset.setdefault("source_asset_id", asset["asset_id"])
        asset["asset_id"] = asset_id
    return assets


def sample_surface_points(asset, num_points=1024, seed=0):
    """Deterministic area-weighted collision-facet samples, in meters.

    These describe the already-scaled native convex components; overlapping
    components can contribute interior facets. This is a shape descriptor, not
    an exact signed-distance query or a claim to reconstruct the visual surface.
    Sampling uses a private CPU generator and leaves the training RNG untouched.
    """
    if num_points <= 0:
        raise ValueError("surface sample count must be positive")
    triangles = _triangles(asset)
    area = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0]).norm(dim=-1) * .5
    keep = area > 1e-16
    triangles, area = triangles[keep], area[keep]
    if not len(area) or not torch.isfinite(area).all():
        raise ValueError("object collision geometry has no finite surface area")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    cdf = area.cumsum(0) / area.sum()
    strata = (torch.arange(num_points, dtype=torch.float64) +
              torch.rand(num_points, generator=generator, dtype=torch.float64)) / num_points
    selected = triangles[torch.searchsorted(cdf, strata).clamp_max(len(cdf) - 1)]
    uv = torch.rand(num_points, 2, generator=generator, dtype=torch.float64)
    root_u = uv[:, :1].sqrt()
    weights = torch.cat((1 - root_u, root_u * (1 - uv[:, 1:]), root_u * uv[:, 1:]), dim=1)
    return (selected * weights.unsqueeze(-1)).sum(1).to(torch.float32).contiguous()


def prepare_object_usd(asset, output):
    """Flatten a new USD, preserve bound hulls, author nominal inertial data.

    The caller must apply ``physical_profile`` friction to the native solver and
    perform native readback. This export does not certify new robot contacts.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdPhysics
    from scipy.spatial.transform import Rotation

    source = Path(asset["native_asset"])
    if sha256_file(source) != asset["native_asset_sha256"]:
        raise ValueError("native object asset requires a matching SHA256 binding")
    mass, com, inertia, _, _ = validate_physical_profile(asset["physical_profile"])
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"unable to open object USD {source}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "object.usda"
    stage.Flatten().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError("object USD must contain exactly one rigid body")
    mass_api = UsdPhysics.MassAPI.Apply(bodies[0])
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    values, axes = np.linalg.eigh(inertia.numpy())
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, values)))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    rigid = UsdPhysics.RigidBodyAPI(bodies[0])
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders or any(not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in colliders):
        raise ValueError("bound object must contain enabled native colliders")
    if len(colliders) != len(asset["collision_meshes"]):
        raise ValueError("object USD collider count differs from bound geometry")
    stage.GetRootLayer().Save()
    return path


def prepare_asset_library(bundle, output, *, num_points=1024, seed=0):
    """Export eight independent object USDs, shape tensors and a bound catalog."""
    assets = load_assets(bundle)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for asset in assets:
        directory = output / asset["object_asset"]
        usd = prepare_object_usd(asset, directory)
        points = sample_surface_points(asset, num_points, seed)
        point_path = directory / "surface_points.pt"
        torch.save(points, point_path)
        keep = ("asset_id", "source_asset_id", "object_asset", "exact_scale", "collision_hash",
                "collision_meshes", "physical_profile")
        records.append({**{key: asset[key] for key in keep},
                        "native_asset": str(usd.resolve()),
                        "native_asset_sha256": sha256_file(usd),
                        "source_native_asset": asset["native_asset"],
                        "source_native_asset_sha256": asset["native_asset_sha256"],
                        "surface_points": {"path": str(point_path.resolve()),
                                           "sha256": sha256_file(point_path),
                                           "tensor_sha256": hashlib.sha256(points.numpy().tobytes()).hexdigest(),
                                           "count": num_points, "seed": seed,
                                           "method": "area_stratified_convex_component_facets_v1"}})
    catalog = {"schema_version": CATALOG_SCHEMA,
               "source_bundle": {"path": str(Path(bundle).resolve()), "sha256": sha256_file(bundle)},
               "scope": "object geometry and nominal physical profiles only; no robot or motion data",
               "native_validation": "requires current GMT robot and solver smoke/readback",
               "assets": records}
    catalog["catalog_content_sha256"] = _digest_json(catalog)
    with (output / "assets.json").open("x", encoding="utf-8") as stream:
        json.dump(catalog, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = prepare_asset_library(args.bundle, args.output, num_points=args.points, seed=args.seed)
    print(json.dumps({"assets": len(result["assets"]), "catalog": str(args.output / "assets.json"),
                      "sha256": result["catalog_content_sha256"]}))


if __name__ == "__main__":
    main()
