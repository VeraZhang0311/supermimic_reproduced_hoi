"""Bind Ultra's original hand hulls to the existing HW3 articulation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from supermimic.g1_assets import SOURCE_URDF, verify_assets
from supermimic.g1_physics import ASSET_DIR

SIDES = ("left", "right")
PARTS = ("palm", "fingers", "thumb")
SCHEMA = "hoi_ultra_native_hand_asset_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geom_names(side):
    return tuple(f"hw3_{side}_wrist_yaw_link_{i}_rubber_hand_ultra_native_{i}" for i in range(5))


@dataclass(frozen=True)
class NativeHands:
    asset_dir: Path
    usd: Path
    identity: dict


def load_native_hands(binding):
    """Verify the prepared collision asset and exact standalone USD conversion."""
    path = Path(binding["path"] if isinstance(binding, dict) else binding).resolve(strict=True)
    data = json.loads(path.read_text())
    asset_dir = path.parent / "robot"
    asset = verify_assets(asset_dir)
    base = verify_assets(ASSET_DIR)
    if (data["schema"] != SCHEMA or data["base_asset_identity"] != base["identity"]
            or data["asset_identity"] != asset["identity"]
            or data["hands"] != {side: list(geom_names(side)) for side in SIDES}):
        raise ValueError("Ultra native hand asset does not bind the current HW3 parent")
    conversion_path = asset_dir / "usd/native_hands/conversion_receipt.json"
    if sha(conversion_path) != data["conversion_sha256"]:
        raise ValueError("Ultra native hand USD conversion receipt changed")
    conversion = json.loads(conversion_path.read_text())
    paths = conversion["mesh_hull_policy"]["collider_paths"]
    if (conversion["source"]["asset_identity"] != asset["identity"]
            or conversion["physics_variant"] != "mujoco"
            or any(name not in paths for side in SIDES for name in geom_names(side))):
        raise ValueError("Ultra native hand USD does not bind all ten collision hulls")
    for name, digest in conversion["files"].items():
        f = (conversion_path.parent / name).resolve(strict=True)
        if not f.is_relative_to(conversion_path.parent) or sha(f) != digest:
            raise ValueError(f"Ultra native hand USD dependency changed: {name}")
    usd = conversion_path.parent / conversion["generated_usd"]
    identity = {"path": str(path), "sha256": sha(path), "asset_identity": asset["identity"],
                "base_asset_identity": base["identity"], "source_ultra_usd_sha256": data["source_ultra_usd_sha256"]}
    if isinstance(binding, dict) and identity != binding:
        raise ValueError("Ultra native hand checkpoint asset binding changed")
    return NativeHands(asset_dir, usd, identity)


def _verified_legacy_hulls(legacy_binding, hand_cache):
    import isaaclab  # noqa: F401
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from .reward import bound_tensor

    binding = json.loads(Path(legacy_binding).read_text())
    for entry in (binding, binding["hand_asset_manifest"]):
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Historical Ultra hand USD or manifest changed")
    hands = bound_tensor(hand_cache)
    manifest = json.loads(Path(binding["hand_asset_manifest"]["path"]).read_text())
    stage = Usd.Stage.Open(binding["path"])
    for side in SIDES:
        record = manifest["hands"][side]
        if (len(record["parts"]) != 5 or len(hands[side]["collision_meshes"]) != 5
                or record["fixed_joint_origin"] != hands[side]["mount"]
                or record["mesh_scale"] != [1., 1., 1.]):
            raise ValueError("Unexpected original Ultra hand hull count, scale or mounting")
        for part, cache in zip(record["parts"], hands[side]["collision_meshes"], strict=True):
            prim = stage.GetPrimAtPath(part["path"])
            mesh = UsdGeom.Mesh(prim)
            xyz = np.asarray(mesh.GetPointsAttr().Get())
            cached = cache["vertices"].numpy()
            if (xyz.shape != (16, 3) or cached.shape != xyz.shape or cache["faces"].shape != (28, 3)
                    or list(mesh.GetFaceVertexCountsAttr().Get()) != [3] * 28
                    or UsdGeom.Xformable(prim).GetLocalTransformation() != Gf.Matrix4d(1.)
                    or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    or np.linalg.norm(xyz[:, None] - cached[None, :], axis=-1).min(1).max() > 1e-7):
                raise ValueError(f"Historical USD and frozen hand cache disagree: {part['path']}")
    return binding, hands


def _replace_colliders(robot, hands):
    import trimesh

    xml_path = robot / "g1_hw3.xml"
    urdf_path = robot / "source" / SOURCE_URDF
    mjcf, urdf = ET.parse(xml_path), ET.parse(urdf_path)
    meshes = mjcf.find("asset")
    for side in SIDES:
        wrist = f"{side}_wrist_yaw_link"
        body = mjcf.find(f".//body[@name='{wrist}']")
        link = urdf.find(f".//link[@name='{wrist}']")
        old = [g for g in body.findall("geom") if g.get("mesh") in {f"{side}_rubber_hand_{p}_hull" for p in PARTS}]
        old_urdf = [g for g in link.findall("collision")
                    if g.find("geometry/mesh") is not None and f"{side}_rubber_hand_" in g.find("geometry/mesh").get("filename")]
        palm, = [g for g in old if g.get("mesh") == f"{side}_rubber_hand_palm_hull"]
        mount = hands[side]["mount"]
        if (len(old) != 3 or len(old_urdf) != 3 or any(list(map(float, g.get("pos").split())) != mount for g in old)
                or palm.get("solref") is not None or palm.get("solimp") is not None):
            raise ValueError(f"Unexpected HW3 {side} hand collision layout or rigid material")
        # Retain the HW3 rigid palm defaults and friction priority, while
        # dropping the FQ finger/thumb compliance from the new rigid hulls.
        material = dict(palm.attrib)
        for g in old:
            body.remove(g)
        for g in old_urdf:
            link.remove(g)
        for index, (name, hull) in enumerate(zip(geom_names(side), hands[side]["collision_meshes"], strict=True)):
            mesh_name = f"{side}_rubber_hand_ultra_native_{index}"
            filename = f"{mesh_name}.STL"
            shape = trimesh.Trimesh(hull["vertices"].numpy(), hull["faces"].numpy(), process=False)
            if not shape.is_volume:
                raise ValueError(f"Original Ultra convex hull is not closed: {name}")
            (robot / "source/meshes/g1" / filename).write_bytes(shape.export(file_type="stl"))
            ET.SubElement(meshes, "mesh", name=mesh_name, content_type="model/stl",
                          file=f"source/meshes/g1/{filename}")
            ET.SubElement(body, "geom", {**material, "name": name, "mesh": mesh_name})
            collider = ET.SubElement(link, "collision", name=name)
            ET.SubElement(collider, "origin", xyz=material["pos"], rpy="0 0 0")
            ET.SubElement(ET.SubElement(collider, "geometry"), "mesh",
                          filename=f"package://robot_description/meshes/g1/{filename}")
    mjcf.write(xml_path, encoding="utf-8")
    urdf.write(urdf_path, encoding="utf-8")


def prepare(output, legacy_binding, geometry_bundle, base=ASSET_DIR):
    from .ultra import _binding

    output = Path(output).resolve()
    geometry_bundle = Path(geometry_bundle)
    bundle = json.loads(geometry_bundle.read_text())
    binding, hands = _verified_legacy_hulls(
        legacy_binding, _binding(bundle["hands"]["geometry_cache"], geometry_bundle.parent))
    base_record = verify_assets(base)
    output.mkdir(parents=True, exist_ok=False)
    robot = output / "robot"
    shutil.copytree(base, robot, ignore=shutil.ignore_patterns("usd"))
    _replace_colliders(robot, hands)
    from supermimic.g1_assets import load_model
    original, candidate = load_model(Path(base) / "g1_hw3.xml"), load_model(robot / "g1_hw3.xml")
    _audit_mjcf(original, candidate)
    asset = dict(base_record)
    asset["collision_roles"] = {name: role for name, role in asset["collision_roles"].items()
                                if not any(name.endswith(f"_{s}_rubber_hand_{p}_hull") for s in SIDES for p in PARTS)}
    asset["collision_roles"].update({name: "palm" for s in SIDES for name in geom_names(s)})
    asset["files"] = {p.relative_to(robot).as_posix(): sha(p) for p in sorted(robot.rglob("*"))
                      if p.is_file() and p != robot / "manifest.json"}
    asset["identity"] = hashlib.sha256(json.dumps(asset["files"], sort_keys=True).encode()).hexdigest()
    (robot / "manifest.json").write_text(json.dumps(asset, indent=2) + "\n")
    usd_output = robot / "usd/native_hands"
    usd_output.mkdir(parents=True)
    log = output / "usd_conversion.log"
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-m", "hoi_tables.native_hands", "--convert", str(robot), str(usd_output)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    receipt = {"schema": SCHEMA, "base_asset_identity": base_record["identity"],
               "asset_identity": asset["identity"], "source_ultra_usd_sha256": binding["sha256"],
               "source_hand_manifest_sha256": binding["hand_asset_manifest"]["sha256"],
               "hands": {side: list(geom_names(side)) for side in SIDES},
               "normal_material": "rigid HW3 palm defaults on all original Ultra hand hulls",
               "conversion_sha256": sha(usd_output / "conversion_receipt.json")}
    path = output / "variant.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return load_native_hands(path)


def _audit_mjcf(original, candidate):
    import numpy as np

    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                  "jnt_range", "jnt_stiffness", "dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(original, field), getattr(candidate, field), err_msg=field)
    old_names = {original.geom(i).name for i in range(original.ngeom)}
    new_names = {candidate.geom(i).name for i in range(candidate.ngeom)}
    removed = {name for name in old_names if any(name.endswith(f"_{s}_rubber_hand_{p}_hull") for s in SIDES for p in PARTS)}
    added = {name for s in SIDES for name in geom_names(s)}
    if len(removed) != 6 or old_names - new_names != removed or new_names - old_names != added:
        raise ValueError("Derived HW3 collider set changed outside the hands")
    for name in old_names - removed:
        i, j = original.geom(name).id, candidate.geom(name).id
        for field in ("geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size", "geom_contype",
                      "geom_conaffinity", "geom_priority", "geom_solref", "geom_solimp", "geom_friction"):
            np.testing.assert_array_equal(getattr(original, field)[i], getattr(candidate, field)[j], err_msg=f"{name}: {field}")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--convert":
        from supermimic import g1_usd

        g1_usd.HULL_POLICY = {**g1_usd.HULL_POLICY, "mesh_collider_count": 37}
        g1_usd._worker(*sys.argv[2:])
        return
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-binding", required=True)
    parser.add_argument("--geometry-bundle", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.legacy_binding, args.geometry_bundle).identity, indent=2))


if __name__ == "__main__":
    main()
