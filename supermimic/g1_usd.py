"""Convert the verified hw3 MJCF to USD in an isolated, Kit-less process.

The standalone importer's PhysX schema plugins must be registered before
its USD types are loaded. Do not perform this conversion inside the active
Isaac Lab simulation process. No SimulationContext or Kit app is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


RECEIPT_NAME = "conversion_receipt.json"
RECEIPT_VERSION = "fengqiao_hw3_isolated_usd_v1"
IMPORT_OPTIONS = {
    "allow_self_collision": True,
    "merge_mesh": False,
    "run_multi_physics_conversion": False,
}
HULL_POLICY = {
    "attribute": "newton:maxHullVertices",
    "value": -1,
    "scope": "exact source robot collision mesh prims only",
    "mesh_collider_count": 33,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_identity(asset_dir):
    from .g1_assets import verify_assets

    receipt = verify_assets(asset_dir)
    xml = asset_dir / "g1_hw3.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"Missing verified hw3 MJCF: {xml}")
    return {
        "asset_identity": receipt["identity"],
        "mjcf_sha256": _sha(xml),
        "converter_helper_sha256": _sha(__file__),
        "physics_variant": "mujoco",
        "import_options": IMPORT_OPTIONS,
        "mesh_hull_policy": HULL_POLICY,
    }


def _within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"USD conversion path escapes its cache: {path}")
    return path


def _write_receipt(output_dir, generated, source, hull_report=None):
    """Called only after the worker checked the actual USD variant."""
    output_dir = Path(output_dir).resolve()
    generated = _within(generated, output_dir)
    if not generated.is_file():
        raise FileNotFoundError(f"Standalone importer returned no USD file: {generated}")
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"USD output must not contain symlinks: {path}")
        if path.is_file() and path.name not in (RECEIPT_NAME, RECEIPT_NAME + ".tmp"):
            files[path.relative_to(output_dir).as_posix()] = _sha(path)
    receipt = {
        "version": RECEIPT_VERSION,
        "source": source,
        "generated_usd": generated.relative_to(output_dir).as_posix(),
        "physics_variant": "mujoco",
        "mesh_hull_policy": hull_report,
        "files": files,
    }
    temporary = output_dir / (RECEIPT_NAME + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / RECEIPT_NAME)


def _read_receipt(output_dir, source):
    output_dir = Path(output_dir).resolve()
    path = output_dir / RECEIPT_NAME
    if not path.is_file():
        raise RuntimeError(f"USD conversion did not produce a completion receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("version") != RECEIPT_VERSION or receipt.get("source") != source:
        raise ValueError(f"Cached USD source/converter identity mismatch: {path}")
    if receipt.get("physics_variant") != "mujoco":
        raise ValueError(f"Cached USD is not the selected MuJoCo variant: {path}")
    hull_report = receipt.get("mesh_hull_policy")
    if not isinstance(hull_report, dict) or any(hull_report.get(k) != v for k, v in HULL_POLICY.items()):
        raise ValueError(f"Cached USD does not bind the unlimited robot mesh-hull policy: {path}")
    if len(hull_report.get("collider_paths", {})) != HULL_POLICY["mesh_collider_count"]:
        raise ValueError(f"Cached USD mesh-hull policy does not cover all source meshes: {path}")
    generated = _within(output_dir / receipt["generated_usd"], output_dir)
    files = receipt.get("files")
    if not isinstance(files, dict) or generated.relative_to(output_dir).as_posix() not in files:
        raise ValueError(f"Cached USD receipt does not bind its generated file: {path}")
    for name, digest in files.items():
        candidate = _within(output_dir / name, output_dir)
        if not candidate.is_file() or _sha(candidate) != digest:
            raise ValueError(f"Cached USD file missing or changed: {candidate}")
    return str(generated)


def _author_robot_hull_limits(stage, asset_dir):
    """Disable Newton's default 64-point hull simplification on this robot.

    Use -1, not 0: both skip Newton's positive-limit remeshing branch,
    but only -1 is valid when forwarded to MuJoCo's mesh compiler.

    The pinned Newton importer resolves this authored Int attribute directly
    (usd/schemas.py -> schema_resolver.py -> usd/_colliders.py). When the
    optional schema type is available apply it; otherwise its typed authored
    attribute is sufficient for that exact resolver. Never touch visual-only
    meshes, primitives, or unrelated scene shapes.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    from .g1_assets import verify_assets

    receipt = verify_assets(asset_dir)
    collision_names = set(receipt["collision_roles"])
    tree = ET.parse(Path(asset_dir) / "g1_hw3.xml")
    expected = {geom.get("name") for geom in tree.findall(".//geom")
                if geom.get("name") in collision_names and geom.get("mesh") is not None}
    if len(expected) != HULL_POLICY["mesh_collider_count"]:
        raise RuntimeError(f"Unexpected source robot mesh collider count: {len(expected)}")
    targets = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        matches = expected.intersection(str(prim.GetPath()).split("/"))
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous source collider identity in USD: {prim.GetPath()}")
        name = matches.pop()
        if name in targets:
            raise RuntimeError(f"Duplicate USD mesh collider for source shape {name}")
        targets[name] = str(prim.GetPath())
    if set(targets) != expected:
        raise RuntimeError(f"USD robot mesh hull policy missing source colliders: {sorted(expected-set(targets))}")
    schema = Usd.SchemaRegistry().GetTypeFromName("NewtonMeshCollisionAPI")
    registered = bool(schema)
    for path in targets.values():
        prim = stage.GetPrimAtPath(path)
        # Instance proxies cannot receive opinions. Deinstance only the
        # owning ancestor of this exact robot collider, then re-resolve it.
        if prim.IsInstanceProxy():
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.IsInstance():
                    ancestor.SetInstanceable(False)
                    break
                ancestor = ancestor.GetParent()
            prim = stage.GetPrimAtPath(path)
        if not prim or prim.IsInstanceProxy():
            raise RuntimeError(f"Cannot author the robot mesh hull limit: {path}")
        if registered and not prim.ApplyAPI(schema):
            raise RuntimeError(f"Cannot apply NewtonMeshCollisionAPI: {path}")
        attr = prim.CreateAttribute(HULL_POLICY["attribute"], Sdf.ValueTypeNames.Int,
                                    custom=not registered)
        if not attr.Set(HULL_POLICY["value"]) or attr.Get() != HULL_POLICY["value"]:
            raise RuntimeError(f"Cannot author unlimited robot hull vertices: {path}")
    return {**HULL_POLICY, "collider_paths": targets,
            "schema_applied": registered,
            "resolver": "pinned Newton authored newton:maxHullVertices -> mesh.maxhullvert; -1 skips Newton simplification and is valid in MuJoCo"}


def _verify_robot_hull_limits(stage, report):
    for path in report["collider_paths"].values():
        prim = stage.GetPrimAtPath(path)
        if not prim or prim.GetAttribute(HULL_POLICY["attribute"]).Get() != HULL_POLICY["value"]:
            raise RuntimeError(f"USD robot mesh hull limit did not persist: {path}")


def ensure_usd(asset_dir=None, *, timeout=900):
    """Return the real generated USD path, after checking source/output hashes.

    A partial/failed conversion is never accepted as an asset. Source or
    helper changes create a distinct cache; stale successful USD files are
    not used as a fallback. Existing source assets are never modified.
    """
    if asset_dir is None:
        from .g1_physics import ASSET_DIR
        asset_dir = ASSET_DIR
    asset_dir = Path(asset_dir).resolve()
    source = _source_identity(asset_dir)
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    cache_root = _within(asset_dir / "usd", asset_dir)
    cache_root.mkdir(exist_ok=True)
    output_dir = _within(cache_root / key, cache_root)
    if output_dir.exists():
        return _read_receipt(output_dir, source)
    output_dir.mkdir(exist_ok=False)
    log_path = cache_root / (key + ".log")
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "supermimic.g1_usd", "--worker",
               "--asset-dir", str(asset_dir), "--output-dir", str(output_dir)]
    try:
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=str(root), stdout=log,
                                    stderr=subprocess.STDOUT, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Isolated hw3 USD conversion failed; inspect {log_path}") from error
    if result.returncode:
        raise RuntimeError(f"Isolated hw3 USD converter exited {result.returncode}; inspect {log_path}")
    # Re-read the source too: do not accept output from concurrently changed inputs.
    if _source_identity(asset_dir) != source:
        raise ValueError("hw3 source changed during isolated USD conversion")
    return _read_receipt(output_dir, source)


def _worker(asset_dir, output_dir):
    # Bootstrap the pinned runtime's USD Python/library paths first. This does
    # not launch a simulator. Plugin registration MUST precede MJCFImporter.
    import isaaclab  # noqa: F401
    from pxr import Plug
    import isaacsim.asset.importer.utils.impl as importer_utils

    plugin_dir = Path(importer_utils.__file__).resolve().parent / "schemas"
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"Standalone importer schema directory is missing: {plugin_dir}")
    Plug.Registry().RegisterPlugins(str(plugin_dir))
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    from pxr import Usd

    asset_dir, output_dir = Path(asset_dir).resolve(), Path(output_dir).resolve()
    source = _source_identity(asset_dir)
    output_dir = _within(output_dir, asset_dir / "usd")
    if not output_dir.is_dir() or (output_dir / RECEIPT_NAME).exists():
        raise ValueError("USD worker requires a newly created cache without a completion receipt")
    config = MJCFImporterConfig(mjcf_path=str(asset_dir / "g1_hw3.xml"),
                                usd_path=str(output_dir), **IMPORT_OPTIONS)
    generated = MJCFImporter(config).import_mjcf()
    if not generated:
        raise RuntimeError("Standalone MJCF importer returned no generated USD path")
    generated = _within(Path(generated).resolve(), output_dir)
    if not generated.is_file():
        raise FileNotFoundError(f"Standalone MJCF importer returned a missing USD: {generated}")
    stage = Usd.Stage.Open(str(generated))
    if stage is None or not stage.GetDefaultPrim():
        raise RuntimeError(f"Generated USD has no valid default prim: {generated}")
    prim = stage.GetDefaultPrim()
    if "Physics" not in prim.GetVariantSets().GetNames():
        raise RuntimeError(f"Generated USD does not expose a Physics variant set: {generated}")
    physics = prim.GetVariantSets().GetVariantSet("Physics")
    if "mujoco" not in physics.GetVariantNames():
        raise RuntimeError(f"Generated USD has no MuJoCo Physics variant: {physics.GetVariantNames()}")
    if not physics.SetVariantSelection("mujoco"):
        raise RuntimeError("Could not select the generated USD MuJoCo Physics variant")
    hull_report = _author_robot_hull_limits(stage, asset_dir)
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Could not save MuJoCo variant selection: {generated}")
    stage = None
    check = Usd.Stage.Open(str(generated))
    selected = check.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
    if selected != "mujoco":
        raise RuntimeError(f"USD MuJoCo variant selection did not persist: {selected}")
    _verify_robot_hull_limits(check, hull_report)
    if _source_identity(asset_dir) != source:
        raise ValueError("hw3 source changed during USD import")
    _write_receipt(output_dir, generated, source, hull_report)
    print(json.dumps({"generated_usd": str(generated), "physics_variant": selected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.asset_dir is None or args.output_dir is None:
            parser.error("--worker requires --asset-dir and --output-dir")
        _worker(args.asset_dir, args.output_dir)
    else:
        if args.output_dir is not None:
            parser.error("--output-dir is only valid for --worker")
        print(ensure_usd(args.asset_dir))


if __name__ == "__main__":
    main()
