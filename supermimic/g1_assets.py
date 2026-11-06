"""Prepare immutable hw3 MJCF assets from the verified handoff tar (CPU only).

No Isaac Sim runtime is used. The Isaac Lab runtime can convert this MJCF
with its existing standalone importer when spawning the articulation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import xml.etree.ElementTree as ET

from .g1_physics import (ARCHIVE_MD5, ASSET_DIR, DEFAULT_JOINT_POS, FRICTION_LEVELS,
                         CONTACT_DAMPING, CONTACT_STIFFNESS, PROFILE, SOFT_IMPEDANCE,
                         PHYSICS_DT, SOURCE_COMMIT, CONTACT_CONE, CONTACT_IMPRATIO, motor_of, profile_contract, sha)
from .robot_metadata import G1_JOINT_NAMES

SOURCE_PREFIX = "GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/"
SOURCE_URDF = "urdf/g1/main_rubber_hand_meshcol_handcol_softfinger.urdf"


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


@contextmanager
def asset_cwd(directory):
    # MuJoCo's native file APIs on Windows do not accept all Unicode absolute
    # paths. Relative filenames under a Python-set cwd avoid that limitation.
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def load_model(path):
    import mujoco
    path = Path(path).resolve()
    with asset_cwd(path.parent):
        return mujoco.MjModel.from_xml_path(path.name)


def soft_solref(inverse_weight, stiffness=CONTACT_STIFFNESS, damping=CONTACT_DAMPING):
    # MuJoCo direct format: stiffness=dmax^2*k, damping=dmax*b.
    # For constant impedance d: F/x = stiffness/(invweight*(1-d)).
    # This normalization uses nominal translational body inverse weight,
    # not the configuration-dependent contact Jacobian effective mass.
    return (-stiffness * inverse_weight * (1 - SOFT_IMPEDANCE),
            -damping * inverse_weight * (1 - SOFT_IMPEDANCE))


def collider_role(name):
    if "rubber_hand_fingers_hull" in name or "rubber_hand_thumb_hull" in name:
        return "finger"
    if "rubber_hand_palm_hull" in name or "wrist_yaw_link" in name:
        return "palm"
    return "body"


def verify_assets(directory=ASSET_DIR):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing hw3 assets: run python -m supermimic.g1_assets --archive HANDOFF.tar.gz --output {directory}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("profile") != PROFILE or receipt.get("archive_md5") != ARCHIVE_MD5:
        raise ValueError("Unknown hw3 asset profile/source")
    if receipt["profile_contract"] != json.loads(json.dumps(profile_contract())):
        raise ValueError("hw3 parameters changed: prepare a new asset directory")
    for name, digest in receipt["files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or sha(directory.joinpath(*rel.parts)) != digest:
            raise ValueError(f"hw3 asset identity changed: {name}")
    identity = hashlib.sha256(json.dumps(receipt["files"], sort_keys=True).encode()).hexdigest()
    if identity != receipt["identity"]:
        raise ValueError("hw3 asset manifest identity mismatch")
    return receipt


def prepare(archive, output):
    import mujoco
    archive, output = Path(archive), Path(output)
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Handoff archive MD5 differs from Feng Qiao's receipt")
    output.mkdir(parents=True, exist_ok=False)
    # Extract only the selected URDF and its referenced assets. No links,
    # scripts, absolute paths or archive member permissions are executed.
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(SOURCE_PREFIX + SOURCE_URDF)
        source_bytes = tar.extractfile(member).read()
        source = ET.fromstring(source_bytes)
        wanted = {SOURCE_URDF}
        for mesh in source.findall(".//mesh"):
            raw = mesh.get("filename")
            if not raw.startswith("package://robot_description/"):
                raise ValueError(f"Unexpected mesh URI {raw}")
            wanted.add(raw.removeprefix("package://robot_description/"))
        for rel in sorted(wanted):
            p = PurePosixPath(rel)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(rel)
            m = tar.getmember(SOURCE_PREFIX + rel)
            if not m.isfile():
                raise ValueError(f"Expected real robot asset: {rel}")
            dst = output / "source" / Path(*p.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(tar.extractfile(m).read())
    receipt = compile_asset(output, source)
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    receipt.update(profile=PROFILE, archive_md5=ARCHIVE_MD5, source_commit=SOURCE_COMMIT,
                   profile_contract=profile_contract(), files=files,
                   identity=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    verify_assets(output)
    return receipt


def compile_asset(output, source):
    output = Path(output).resolve()
    with asset_cwd(output):
        return _compile_asset(Path("."), source)


def _compile_asset(output, source):
    import mujoco
    import numpy as np
    output = Path(output)
    # Relative references keep the prepared folder relocatable across hosts.
    for mesh in source.findall(".//mesh"):
        rel = mesh.get("filename").removeprefix("package://robot_description/")
        mesh.set("filename", "source/" + rel)
    old = source.find("mujoco")
    if old is not None:
        source.remove(old)
    ET.SubElement(ET.SubElement(source, "mujoco"), "compiler", meshdir=".",
                  discardvisual="false", fusestatic="false", strippath="false")
    collision_names = []
    for link in source.findall("link"):
        for i, geom in enumerate(link.findall("collision")):
            mesh = geom.find("geometry/mesh")
            stem = Path(mesh.get("filename")).stem if mesh is not None else f"primitive_{i}"
            name = f"hw3_{link.get('name')}_{i}_{stem}"
            geom.set("name", name)
            collision_names.append(name)
        for i, geom in enumerate(link.findall("visual")):
            geom.set("name", f"visual_{link.get('name')}_{i}")
    intermediate = output / "converted.urdf"
    ET.ElementTree(source).write(intermediate, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(intermediate))
    xml = output / "g1_hw3.xml"
    mujoco.mj_saveLastXML(str(xml), m)
    tree = ET.parse(xml)
    root = tree.getroot()
    # mj_saveLastXML serializes ~6 significant digits. Restore the importer's
    # double-precision inertial and joint/body-frame values before recompile.
    # Never infer masses from the replacement collision geometry.
    for body in root.findall(".//body"):
        bid = m.body(body.get("name")).id
        body.set("pos", numbers(m.body_pos[bid]))
        body.set("quat", numbers(m.body_quat[bid]))
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", format(float(m.body_mass[bid]), ".17g"))
            inertial.set("pos", numbers(m.body_ipos[bid]))
            inertial.set("quat", numbers(m.body_iquat[bid]))
            inertial.set("diaginertia", numbers(m.body_inertia[bid]))
    for joint in root.findall(".//joint"):
        jid = m.joint(joint.get("name")).id
        joint.set("pos", numbers(m.jnt_pos[jid]))
        joint.set("axis", numbers(m.jnt_axis[jid]))
        if m.jnt_limited[jid]:
            joint.set("range", numbers(m.jnt_range[jid]))
    pelvis = root.find("worldbody/body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("Missing floating pelvis")
    ET.SubElement(pelvis, "freejoint", name="floating_base_joint")
    compiler = root.find("compiler")
    compiler.set("inertiafromgeom", "false")
    compiler.set("meshdir", ".")
    ET.SubElement(root, "option", timestep=str(PHYSICS_DT), integrator="implicitfast", cone=CONTACT_CONE, impratio=str(CONTACT_IMPRATIO))
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in G1_JOINT_NAMES:
            cap, _, armature = motor_of(name)
            joint.set("armature", str(armature))
            joint.set("actuatorfrcrange", numbers([-cap, cap]))
            joint.set("damping", "0")
            joint.set("frictionloss", "0")  # runtime owns the native friction constraint
            joint.set("stiffness", "0")
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name not in collision_names:
            geom.set("contype", "0"); geom.set("conaffinity", "0")
            continue
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")  # matches FQ replace_cylinders_with_capsules=True
        geom.set("contype", "1"); geom.set("conaffinity", "1")
        geom.set("condim", "3")
        geom.set("friction", "1 0 0")
        geom.set("margin", "0"); geom.set("gap", "0")
        geom.set("priority", "1" if collider_role(name) != "body" else "0")
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(xml))
    if set(collision_names) != {m.geom(i).name for i in range(m.ngeom) if m.geom_contype[i]}:
        raise ValueError("URDF collision names/count changed during compilation")
    roles = {}
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name not in collision_names:
            continue
        role = collider_role(name); roles[name] = role
        if role == "finger":
            i = m.geom(name).id
            invweight = float(m.body_invweight0[m.geom_bodyid[i], 0])
            geom.set("solref", numbers(soft_solref(invweight)))
            geom.set("solimp", numbers([SOFT_IMPEDANCE, SOFT_IMPEDANCE, .001, .5, 2]))
    # Named direct torque actuators also allow independent MuJoCo CPU validation.
    actuators = ET.SubElement(root, "actuator")
    for name in G1_JOINT_NAMES:
        cap = motor_of(name)[0]
        ET.SubElement(actuators, "motor", name="torque_" + name, joint=name,
                      gear="1", ctrllimited="true", ctrlrange=numbers([-cap, cap]))
    tree.write(xml, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(xml))
    report = {"mujoco_version": mujoco.__version__, "collision_roles": roles,
              "source_link_count": len(source.findall("link")), "nq": m.nq, "nv": m.nv,
              "collision_count": len(collision_names), "total_mass_kg": float(m.body_mass.sum()),
              "soft_calibration_status": "mass-normalized adaptation; CPU probes and native validation required",
              "geometry_note": "same source meshes/transforms; MuJoCo convex hulls, not PhysX cooked triangles",
              "capsule_note": "source cylinders replaced by capsules, matching FQ spawner",
              "reference_note": "physics/reference/privileged features share source 38-link skeleton; 3283D privileged block"}
    if m.nq != 36 or m.nv != 35 or len([v for v in roles.values() if v == "finger"]) != 4:
        raise ValueError("Unexpected hw3 topology")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ASSET_DIR)
    args = parser.parse_args()
    receipt = prepare(args.archive, args.output)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("files", "collision_roles", "profile_contract")}, indent=2))


if __name__ == "__main__":
    main()
