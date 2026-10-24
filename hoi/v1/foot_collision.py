"""Traceable ULTRA foot meshes layered over an immutable native hand asset.

USD authoring and CoACD never share an interpreter. The original PhysX VHACD
settings are recorded as provenance; they are not CoACD sampling parameters.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from hoi.hand_collision import RECIPE as HAND_RECIPE

FOOT_SCHEMA = "hoi_v1_ultra_g1_foot_mesh_overlay_v1"
FOOT_PROFILE = "ultra_g1_stl_coacd5v16_v1"
ULTRA_URDF_SHA256 = "1791b76b498fd60f4b29dfbf6cc82d2144df6ea739ef6d3cf32442168a3097a1"
FOOT_SHA = {"left": "c4092af943141d4d9f74232f3cfa345afc6565f46a077793b8ae0e68b39dc33f",
            "right": "4b66222ea56653e627711b56d0a8949b4920da5df091da0ceb343f54e884e3a5"}
RECIPE = dict(HAND_RECIPE)
ORIGINAL_VHACD = {"max_convex_hulls": 5, "max_num_vertices_per_ch": 16, "resolution": 60000}
LIMITATION = ("The original ULTRA URDF uses full foot STL collisions with PhysX VHACD "
              "5-hull/16-vertex/60000-voxel settings. This overlay uses byte-identical "
              "STLs and zero collision transforms, cooked by CoACD with 5-hull and "
              "16-vertex limits. It does not reproduce the original PhysX cooked "
              "vertices or map 60000 to a CoACD sampling resolution. Each explicit "
              "hull is imported as convexHull. Native compiled geometry and pair "
              "calibration require a separate new validation; no physics pass is inferred.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def resolve_mesh_uri(urdf, uri, override=None):
    """Resolve a local URDF mesh URI, with an explicit relocated-file override."""
    if override is not None:
        result = Path(override).resolve()
    else:
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("Remote file mesh authorities require an explicit local STL path")
            result = Path(url2pathname(unquote(parsed.path))).resolve()
        elif parsed.scheme == "package":
            folder = Path(urdf).resolve().parent
            candidates = [folder / unquote(parsed.path).lstrip("/"),
                          folder.parent / parsed.netloc / unquote(parsed.path).lstrip("/")]
            existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
            if len(existing) != 1:
                raise ValueError("Unresolved or ambiguous package mesh URI; supply the explicit STL path")
            result = existing[0]
        elif parsed.scheme:
            raise ValueError("Only local relative/file/package mesh URIs are supported")
        else:
            result = (Path(urdf).resolve().parent / uri).resolve()
    if not result.is_file():
        raise ValueError(f"Foot source STL is unavailable: {result}")
    return result


def original_definition(urdf):
    """Parse the original collision geometry, never substitute visual geometry."""
    root = ET.parse(urdf).getroot()
    result = {}
    for side in FOOT_SHA:
        name = side + "_ankle_roll_link"
        links = root.findall(f"link[@name='{name}']")
        if len(links) != 1 or len(links[0].findall("collision")) != 1:
            raise ValueError(f"ULTRA {side} foot must have one full-mesh collision")
        collision = links[0].find("collision")
        mesh = collision.find("geometry/mesh")
        if mesh is None or Path(urlparse(mesh.get("filename", "")).path).name != name + ".STL":
            raise ValueError(f"ULTRA {side} foot collision must bind its original STL")
        origin = collision.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
        if xyz != [0., 0., 0.] or rpy != [0., 0., 0.] or scale != [1., 1., 1.]:
            raise ValueError("ULTRA foot collision origin/rpy/scale changed")
        joints = [j for j in root.findall("joint") if j.find("child") is not None and j.find("child").get("link") == name]
        if len(joints) != 1 or joints[0].get("type") != "revolute":
            raise ValueError("ULTRA foot must retain its ankle-roll revolute joint")
        joint = joints[0]
        if joint.find("parent").get("link") != side + "_ankle_pitch_link":
            raise ValueError("Unexpected ULTRA foot parent link")
        mount = joint.find("origin")
        jxyz = [float(v) for v in mount.get("xyz", "0 0 0").split()]
        jrpy = [float(v) for v in mount.get("rpy", "0 0 0").split()]
        if jxyz != [0., 0., -.017558] or jrpy != [0., 0., 0.]:
            raise ValueError("ULTRA ankle-roll mounting transform changed")
        result[side] = {"link": name, "stl_filename": mesh.get("filename"), "collision_origin": xyz,
                        "collision_rpy": rpy, "mesh_scale": scale, "joint_origin": jxyz, "joint_rpy": jrpy}
    return result


def validate_hulls(cooked, side):
    if cooked.get("recipe") != RECIPE or cooked.get("stl_sha256") != FOOT_SHA[side]:
        raise ValueError("Foot cooked source/recipe differs from the versioned binding")
    pieces = cooked.get("pieces", [])
    if not 1 <= len(pieces) <= RECIPE["max_convex_hull"]:
        raise ValueError("Foot convex-hull count exceeds the original 5-hull limit")
    for piece in pieces:
        vertices, faces = piece.get("vertices", []), piece.get("faces", [])
        if not 4 <= len(vertices) <= RECIPE["max_ch_vertex"] or not faces:
            raise ValueError("Foot hull must contain 4–16 vertices and triangle faces")
        if any(len(v) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vertices):
            raise ValueError("Foot hull vertices must be finite 3D values")
        edges = {}
        for face in faces:
            if len(face) != 3 or len(set(face)) != 3 or any(type(i) is not int or not 0 <= i < len(vertices) for i in face):
                raise ValueError("Invalid foot hull triangle indices")
            for a, b in zip(face, face[1:] + face[:1]):
                edges[(a, b)] = edges.get((a, b), 0) + 1
        if any(count != 1 or edges.get((b, a), 0) != 1 for (a, b), count in edges.items()):
            raise ValueError("Foot hull triangles must form a closed consistently wound surface")
        volume6 = 0.
        for i, j, k in faces:
            a, b, c = vertices[i], vertices[j], vertices[k]
            volume6 += sum(a[d] * (b[(d + 1) % 3] * c[(d + 2) % 3] - b[(d + 2) % 3] * c[(d + 1) % 3]) for d in range(3))
        if abs(volume6) <= 1e-15:
            raise ValueError("Foot collision hull has zero enclosed volume")
    return pieces


def verify_stage_difference(before, after, disabled_paths, added_paths):
    """Allow exactly collisionEnabled=false on old foot spheres and new hulls."""
    disabled, added = set(disabled_paths), set(added_paths)
    if not disabled or not added or not disabled <= before.keys() or set(after) != set(before) | added:
        raise ValueError("Foot overlay removed/added unexpected prims")
    if any(not re.search(r"/(left|right)_ankle_roll_link/HOIUltraFootCollision\d+$", path) for path in added):
        raise ValueError("New foot collider lies outside its original ankle-roll body")
    if any(not re.search(r"/(left|right)_ankle_roll_link/", path) or before[path].get("type") != "Sphere" or
           "PhysicsCollisionAPI" not in before[path].get("schemas", []) for path in disabled):
        raise ValueError("Only original ankle-roll collision spheres may change")
    for path in added:
        prim = after[path]
        if (prim.get("type") != "Mesh" or "PhysicsCollisionAPI" not in prim.get("schemas", []) or
                "PhysicsMeshCollisionAPI" not in prim.get("schemas", []) or
                any(s in prim.get("schemas", []) for s in ("PhysicsRigidBodyAPI", "PhysicsMassAPI")) or
                prim.get("attributes", {}).get("physics:collisionEnabled") != "True" or
                prim.get("attributes", {}).get("physics:approximation") != "convexHull"):
            raise ValueError("Added foot hull must be an enabled convex mesh without a new rigid body or mass")
    for path, original in before.items():
        expected = deepcopy(original)
        if path in disabled:
            expected["attributes"]["physics:collisionEnabled"] = "False"
        if after[path] != expected:
            raise ValueError(f"Foot overlay changed an unrelated authored property: {path}")
    return {"passed": True, "original_prims_checked": len(before), "changed_existing_prim_paths": sorted(disabled),
            "added_collider_paths": sorted(added), "before_content_sha256": _digest(before), "after_content_sha256": _digest(after)}


def _usd_snapshot(stage):
    from pxr import Usd
    return {str(p.GetPath()): {"type": p.GetTypeName(), "schemas": list(p.GetAppliedSchemas()),
            "attributes": {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValueOpinion()},
            "relationships": {r.GetName(): [str(x) for x in r.GetTargets()] for r in p.GetRelationships()}}
            for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())}


def cook(side, stl, output):
    """Child-only CoACD entry; the USD-authoring interpreter never imports it."""
    if sha(stl) != FOOT_SHA[side]:
        raise ValueError("ULTRA foot STL bytes differ")
    import coacd
    import numpy as np
    import trimesh
    coacd.set_log_level("warn")
    mesh = trimesh.load_mesh(str(stl), process=True)
    pieces = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), **RECIPE)
    payload = {"recipe": RECIPE, "stl_sha256": sha(stl),
               "pieces": [{"vertices": v.tolist(), "faces": f.tolist()} for v, f in pieces]}
    validate_hulls(payload, side)
    _write(output, payload)


def prepare(base_robot_asset, original_urdf, output, *, left_stl=None, right_stl=None,
            expected_base_sha256=None, expected_hand_manifest_sha256=None):
    """Create a fresh overlay; no source USD, receipt, or original prim is edited."""
    from supermimic.physics import load_hand_asset
    base, urdf, output = Path(base_robot_asset).resolve(), Path(original_urdf).resolve(), Path(output).resolve()
    if (base.parent / "foot-asset.json").exists():
        raise ValueError("Base must be the original immutable hand overlay, not an existing foot overlay")
    base_report = load_hand_asset(base)
    original_hand = base.parent / "hand-asset.json"
    if expected_base_sha256 is not None and sha(base) != expected_base_sha256:
        raise ValueError("Bound base hand-overlay SHA256 differs")
    if expected_hand_manifest_sha256 is not None and sha(original_hand) != expected_hand_manifest_sha256:
        raise ValueError("Bound base hand receipt SHA256 differs")
    if sha(urdf) != ULTRA_URDF_SHA256:
        raise ValueError("Original ULTRA G1 URDF SHA256 differs")
    definitions = original_definition(urdf)
    paths = {side: resolve_mesh_uri(urdf, definitions[side]["stl_filename"], override)
             for side, override in (("left", left_stl), ("right", right_stl))}
    if any(sha(paths[side]) != FOOT_SHA[side] for side in FOOT_SHA):
        raise ValueError("Original ULTRA foot STL SHA256 differs")
    output.mkdir(parents=True, exist_ok=False)
    cooked = {}
    for side, stl in paths.items():
        path = output / (side + "-hulls.json")
        with (output / (side + "-cook.log")).open("x") as log:
            subprocess.run([sys.executable, "-X", "faulthandler", "-m", "hoi.v1.foot_collision",
                            "--cook", side, "--stl", str(stl), "--output", str(path)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        cooked[side] = json.loads(path.read_text())
        validate_hulls(cooked[side], side)
    # CoACD imports exist only in completed child interpreters.
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    source = Usd.Stage.Open(str(base))
    if source is None:
        raise ValueError("Cannot open the bound hand-overlay USD")
    before = _usd_snapshot(source)
    target = output / "g1_ultra_foot_hand_collision.usda"
    layer = Sdf.Layer.CreateNew(str(target))
    layer.subLayerPaths = [str(base)]
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath(source.GetDefaultPrim().GetPath()))
    sides, disabled_all, added_all = {}, [], []
    for side in FOOT_SHA:
        bodies = [p for p in stage.Traverse() if p.GetName() == definitions[side]["link"] and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError("Cannot bind a unique original ankle-roll rigid body")
        body = bodies[0]
        active = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                if prim.IsInstanceProxy() or not prim.IsA(UsdGeom.Sphere):
                    raise ValueError("Only the original noninstanced ankle-roll collision spheres may be disabled")
                if abs(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) - .005) > 1e-8:
                    raise ValueError("Original G1 foot collision sphere radius differs")
                active.append(prim)
        if len(active) != 4:
            raise ValueError("Expected exactly four original active foot spheres per ankle-roll body")
        disabled = [str(p.GetPath()) for p in active]
        for prim in active:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        parts = []
        for index, piece in enumerate(cooked[side]["pieces"]):
            path = body.GetPath().AppendChild(f"HOIUltraFootCollision{index}")
            collider = UsdGeom.Mesh.Define(stage, path)
            collider.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in piece["vertices"]])
            collider.CreateFaceVertexCountsAttr([3] * len(piece["faces"]))
            collider.CreateFaceVertexIndicesAttr([i for face in piece["faces"] for i in face])
            collider.CreateSubdivisionSchemeAttr("none")
            collider.CreatePurposeAttr("guide")
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
            parts.append({"path": str(path), "vertices": len(piece["vertices"]), "triangles": len(piece["faces"]),
                          "hull_sha256": _digest(piece)})
        sides[side] = {"link": definitions[side]["link"], "hull_count": len(parts), "disabled_colliders": disabled,
                       "parts": parts, "cooked_geometry": _record(output / (side + "-hulls.json"))}
        disabled_all.extend(disabled)
        added_all.extend(p["path"] for p in parts)
    after = _usd_snapshot(stage)
    difference = verify_stage_difference(before, after, disabled_all, added_all)
    _write(output / "foot-stage-before.json", before)
    _write(output / "foot-stage-after.json", after)
    difference.update(before_snapshot=_record(output / "foot-stage-before.json"), after_snapshot=_record(output / "foot-stage-after.json"))
    layer.Save()
    # This copy still describes the OLD hand overlay and retains its true
    # unchanged-source semantics. Shared loading uses foot-asset explicitly.
    shutil.copyfile(original_hand, output / "hand-asset.json")
    cook_recipe = {"algorithm": "CoACD", "recipe": RECIPE, "original_ultra_vhacd": ORIGINAL_VHACD,
                   "equivalent_to_physx_vhacd": False, "cooker_and_usd_imports_isolated": True}
    report = {"schema": FOOT_SCHEMA, "profile": FOOT_PROFILE, "original_urdf": _record(urdf),
              "source_meshes": {s: {**definitions[s], **_record(paths[s])} for s in FOOT_SHA},
              "cook": cook_recipe, "cook_sha256": _digest(cook_recipe), "limitations": LIMITATION,
              "overlay": _record(target), "base_robot_asset": {**_record(base), "hand_asset_manifest": _record(original_hand)},
              "copied_base_hand_manifest": _record(output / "hand-asset.json"), "sides": sides, "stage_diff": difference,
              "native_validation": "not_run", "formal_training_started": False}
    _write(output / "foot-asset.json", report)
    validate_overlay(target, base_hand_report=base_report)
    binding = {**_record(target), "hand_asset_manifest": _record(output / "hand-asset.json"),
               "foot_asset_manifest": _record(output / "foot-asset.json")}
    _write(output / "native-robot-asset.json", binding)
    return {**report, "native_robot_asset": binding}


def _read_record(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"Foot overlay bound file changed: {path}")
    return path


def validate_overlay(path, base_hand_report=None):
    """Verify the new overlay and old hand receipt without rewriting either."""
    path = Path(path).resolve()
    report = json.loads((path.parent / "foot-asset.json").read_text())
    if report.get("schema") != FOOT_SCHEMA or report.get("profile") != FOOT_PROFILE:
        raise ValueError("Unknown foot overlay schema/profile")
    if _read_record(report["overlay"]).resolve() != path:
        raise ValueError("Foot receipt belongs to a different robot overlay")
    base = _read_record(report["base_robot_asset"])
    original_hand = _read_record(report["base_robot_asset"]["hand_asset_manifest"])
    copied = _read_record(report["copied_base_hand_manifest"])
    if copied.resolve() != path.parent / "hand-asset.json" or copied.read_bytes() != original_hand.read_bytes():
        raise ValueError("Original hand receipt must be preserved byte-for-byte")
    if (base.parent / "foot-asset.json").exists():
        raise ValueError("Recursive foot overlays are not supported")
    from supermimic.physics import load_hand_asset
    checked_hand = load_hand_asset(base)
    if base_hand_report is not None and checked_hand != base_hand_report:
        raise ValueError("Supplied base hand validation differs from the unchanged source")
    if json.loads(original_hand.read_text()) != checked_hand:
        raise ValueError("Base hand receipt validation differs")
    urdf = _read_record(report["original_urdf"])
    if sha(urdf) != ULTRA_URDF_SHA256:
        raise ValueError("ULTRA foot source URDF identity differs")
    definitions = original_definition(urdf)
    expected_cook = {"algorithm": "CoACD", "recipe": RECIPE, "original_ultra_vhacd": ORIGINAL_VHACD,
                     "equivalent_to_physx_vhacd": False, "cooker_and_usd_imports_isolated": True}
    if report["cook"] != expected_cook or report["cook_sha256"] != _digest(expected_cook):
        raise ValueError("Unexpected foot cooking recipe or false PhysX equivalence")
    disabled, added = [], []
    for side in FOOT_SHA:
        source = report["source_meshes"][side]
        if sha(_read_record(source)) != FOOT_SHA[side] or any(source.get(k) != v for k, v in definitions[side].items()):
            raise ValueError("ULTRA foot source mesh/mount differs")
        item = report["sides"][side]
        pieces = validate_hulls(json.loads(_read_record(item["cooked_geometry"]).read_text()), side)
        if item["link"] != definitions[side]["link"] or item["hull_count"] != len(pieces) or len(item["parts"]) != len(pieces):
            raise ValueError("Foot side/hull count binding differs")
        if len(set(item["disabled_colliders"])) != 4 or any(
                f"/{side}_ankle_roll_link/" not in p for p in item["disabled_colliders"]):
            raise ValueError("Each foot must disable its four original collision spheres")
        for index, (part, piece) in enumerate(zip(item["parts"], pieces)):
            if (part["hull_sha256"] != _digest(piece) or part["vertices"] != len(piece["vertices"]) or
                    part["triangles"] != len(piece["faces"]) or not part["path"].endswith(
                        f"/{side}_ankle_roll_link/HOIUltraFootCollision{index}")):
                raise ValueError("Explicit foot hull data changed")
        disabled.extend(item["disabled_colliders"])
        added.extend(p["path"] for p in item["parts"])
    diff = report["stage_diff"]
    before = json.loads(_read_record(diff["before_snapshot"]).read_text())
    after = json.loads(_read_record(diff["after_snapshot"]).read_text())
    verified = verify_stage_difference(before, after, disabled, added)
    if any(diff.get(key) != value for key, value in verified.items()):
        raise ValueError("Foot overlay authored-property difference receipt changed")
    return report


def validate_foot_asset(robot_asset):
    if not isinstance(robot_asset, dict):
        return validate_overlay(robot_asset)
    path = _read_record(robot_asset)
    manifest = _read_record(robot_asset["foot_asset_manifest"])
    if manifest.resolve() != path.parent / "foot-asset.json":
        raise ValueError("Foot asset manifest is not adjacent to its robot overlay")
    hand = _read_record(robot_asset["hand_asset_manifest"])
    if hand.resolve() != path.parent / "hand-asset.json":
        raise ValueError("Copied hand manifest is not adjacent to its robot overlay")
    return validate_overlay(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cook", choices=tuple(FOOT_SHA))
    parser.add_argument("--stl", type=Path)
    parser.add_argument("--base-robot-asset", type=Path)
    parser.add_argument("--original-urdf", type=Path)
    parser.add_argument("--left-stl", type=Path)
    parser.add_argument("--right-stl", type=Path)
    parser.add_argument("--expected-base-sha256")
    parser.add_argument("--expected-hand-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cook:
        if args.stl is None:
            parser.error("--cook requires --stl")
        cook(args.cook, args.stl, args.output)
    else:
        if args.base_robot_asset is None or args.original_urdf is None:
            parser.error("overlay preparation requires --base-robot-asset and --original-urdf")
        result = prepare(args.base_robot_asset, args.original_urdf, args.output,
            left_stl=args.left_stl, right_stl=args.right_stl, expected_base_sha256=args.expected_base_sha256,
            expected_hand_manifest_sha256=args.expected_hand_manifest_sha256)
        print(json.dumps({"profile": FOOT_PROFILE, "native_robot_asset": result["native_robot_asset"],
                          "native_validation": result["native_validation"]}, allow_nan=False))


if __name__ == "__main__":
    main()
