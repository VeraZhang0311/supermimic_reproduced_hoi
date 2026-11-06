"""CPU physics probes for the G1 hw3 port. No policy, optimizer or training.

Run as `python -m supermimic.g1_validation --output NEW_REPORT.json`.
These are real MuJoCo CPU dynamics probes; they do not certify the Newton
USD import, GPU model updates or the PhysX-to-MuJoCo sim-to-sim gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .g1_assets import load_model, soft_solref, verify_assets, numbers
from .g1_physics import (ASSET_DIR, G1Motor, PHYSICS_DT, SOFT_IMPEDANCE, CONTACT_STIFFNESS,
                         CONTACT_DAMPING, TARGET_PAD_SLOPE_N_MM, CONTACT_CONE, CONTACT_IMPRATIO,
                         FRICTION_LEVELS, friction_threshold, physics_contract, motor_of)
from .g1_runtime import audit_model
from .robot_metadata import G1_JOINT_NAMES


def _pad_model(asset_dir, mass, role, stiffness, damping, side, part, sliding=False,
               cone=CONTACT_CONE, impratio=CONTACT_IMPRATIO):
    import trimesh
    if side not in ("left", "right") or part not in ("fingers", "thumb", "palm"):
        raise ValueError("Unknown hand collision part")
    mesh = Path(asset_dir) / f"source/meshes/g1/{side}_rubber_hand_{part}_hull.STL"
    vertices = trimesh.load(mesh, process=False).vertices
    lowest = float(vertices[:, 2].min())
    ref = numbers(soft_solref(1/mass, stiffness, damping)) if role == "finger" else ".005 1"
    imp = numbers([SOFT_IMPEDANCE, SOFT_IMPEDANCE, .001, .5, 2]) if role == "finger" else ".9 .95 .001 .5 2"
    slide_joint = '<joint name="slide" type="slide" axis="1 0 0"/>' if sliding else ""
    xml = f'''<mujoco><compiler angle="radian"/><option timestep="{PHYSICS_DT}" gravity="0 0 0" integrator="implicitfast" cone="{cone}" impratio="{impratio}"/>
    <asset><mesh name="finger" file="finger.stl"/></asset><worldbody>
    <geom name="plane" type="plane" size="1 1 .1" friction="1 0 0"/>
    <body pos="0 0 .03"><joint name="press" type="slide" axis="0 0 1"/>{slide_joint}
    <inertial mass="{mass}" pos="0 0 0" diaginertia=".01 .01 .01"/>
    <geom name="pad" type="mesh" mesh="finger" pos="0 0 {-lowest}" priority="1" solref="{ref}" solimp="{imp}" friction="1 0 0"/>
    </body></worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml, {"finger.stl": mesh.read_bytes()})


def _normal_force(model, data):
    force = np.zeros(6)
    normal = 0.
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        normal += float(force[0])
    return normal


def finger_probe(asset_dir=ASSET_DIR, mass=1., load=5., role="finger", stiffness=CONTACT_STIFFNESS,
                 damping=CONTACT_DAMPING, side="left", part="fingers", unload=False,
                 cone=CONTACT_CONE, impratio=CONTACT_IMPRATIO):
    """Load an actual hand hull against a plane at the production 5 ms step.

    Force / total penetration includes the geometrical offset between several
    contacts. The calibration criterion is the *incremental* 1--10 N slope,
    reported by soft_hand_probes(), rather than each force/secant value.
    """
    model = _pad_model(asset_dir, mass, role, stiffness, damping, side, part, cone=cone, impratio=impratio)
    data = mujoco.MjData(model)
    tail = []
    for _ in range(2000):
        data.qfrc_applied[0] = -load
        mujoco.mj_step(model, data)
        if _ >= 1800:
            tail.append(float(data.qvel[0]))
    penetration = max(0., -(.03 + float(data.qpos[0])))
    normal = _normal_force(model, data)
    if not np.isfinite(data.qpos).all() or data.ncon == 0:
        raise AssertionError("Finger press has no finite physical contact")
    report = {"mass_kg": mass, "load_N": load, "penetration_mm": penetration*1000,
              "secant_N_mm": load/max(penetration*1000, 1e-12), "normal_force_N": normal,
              "velocity_m_s": float(data.qvel[0]), "settled_velocity_max_m_s": max(map(abs, tail)),
              "contacts": data.ncon, "material": role, "side": side, "part": part,
              "cone": cone, "impratio": impratio,
              "contact_stiffness_N_m": stiffness if role == "finger" else None,
              "contact_damping_Ns_m": damping if role == "finger" else None}
    if unload:
        data.qfrc_applied[:] = 0
        force_max = 0.
        for _ in range(600):
            mujoco.mj_step(model, data)
            force_max = max(force_max, _normal_force(model, data))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise AssertionError("Nonfinite release response")
        report["release"] = {"normal_force_N": _normal_force(model, data),
                             "penetration_mm": max(0., -(.03 + float(data.qpos[0])))*1000,
                             "peak_normal_force_N": force_max, "velocity_m_s": float(data.qvel[0]),
                             "duration_s": 600*PHYSICS_DT}
    return report


def surface_friction_probe(asset_dir=ASSET_DIR, cone=CONTACT_CONE, impratio=CONTACT_IMPRATIO, check=True):
    """Physical hull-plane test of the fixed mu=1 hand material, not frictionloss."""
    results = []
    for lateral in (2., 7.):
        model = _pad_model(asset_dir, 1., "finger", CONTACT_STIFFNESS, CONTACT_DAMPING,
                           "left", "fingers", sliding=True, cone=cone, impratio=impratio)
        data = mujoco.MjData(model)
        for _ in range(2000):
            data.qfrc_applied[:] = [-5., 0.]
            mujoco.mj_step(model, data)
        for _ in range(200):
            data.qfrc_applied[:] = [-5., lateral]
            mujoco.mj_step(model, data)
        results.append({"lateral_load_N": lateral, "normal_force_N": _normal_force(model, data),
                        "slide_velocity_m_s": float(data.qvel[1]), "slide_distance_m": float(data.qpos[1])})
    if check and (abs(results[0]["slide_velocity_m_s"]) >= .001 or results[1]["slide_velocity_m_s"] < .1):
        raise AssertionError(f"Hand mu=1 stick/slip control failed: {results}")
    return {"mu": 1., "normal_load_N": 5., "cone": cone, "impratio": impratio, "results": results,
            "below_limit_velocity_target_m_s": .001,
            "note": "MuJoCo soft contacts allow finite low-load creep; above-limit load must slide"}


def drop_probe(asset_dir=ASSET_DIR, mass=1., cone=CONTACT_CONE, impratio=CONTACT_IMPRATIO):
    """3 cm gravity drop on the actual hull; report extrapolation honestly.

    A 4 kg load alone is 39.24 N and is outside the source's 0--10 N press
    calibration. Numerical convergence of that case is not physical fidelity.
    """
    model = _pad_model(asset_dir, mass, "finger", CONTACT_STIFFNESS, CONTACT_DAMPING,
                       "left", "fingers", cone=cone, impratio=impratio)
    model.opt.gravity[2] = -9.81
    data = mujoco.MjData(model)
    peak_force, peak_penetration, tail = 0., 0., []
    for i in range(2000):
        mujoco.mj_step(model, data)
        peak_force = max(peak_force, _normal_force(model, data))
        peak_penetration = max(peak_penetration, max(0., -(.03+float(data.qpos[0])))*1000)
        if i >= 1800:
            tail.append(abs(float(data.qvel[0])))
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    return {"mass_kg": mass, "initial_gap_m": .03, "gravity_m_s2": 9.81,
            "cone": cone, "impratio": impratio, "nominal_load_N": mass*9.81,
            "final_normal_force_N": _normal_force(model, data), "peak_normal_force_N": peak_force,
            "final_penetration_mm": max(0., -(.03+float(data.qpos[0])))*1000,
            "peak_penetration_mm": peak_penetration, "settled_velocity_max_m_s": max(tail), "finite": finite,
            "within_source_calibration_force_range": peak_force <= 10.,
            "scope": "numerical convergence probe; response above 10 N extrapolates the linear contact adaptation"}


def soft_hand_probes(asset_dir=ASSET_DIR):
    curves = []
    for side in ("left", "right"):
        for part in ("fingers", "thumb"):
            presses = [finger_probe(asset_dir, load=f, side=side, part=part) for f in (1., 5., 10.)]
            slope = 9. / (presses[-1]["penetration_mm"] - presses[0]["penetration_mm"])
            curves.append({"side": side, "part": part, "incremental_N_mm": slope, "presses": presses})
    masses = [finger_probe(asset_dir, mass=m, unload=True) for m in (.25, 1., 4.)]
    rigid = finger_probe(asset_dir, role="rigid_control")
    palm = finger_probe(asset_dir, role="palm", part="palm")
    return {"target_incremental_N_mm": TARGET_PAD_SLOPE_N_MM,
            "solver": {"cone": CONTACT_CONE, "impratio": CONTACT_IMPRATIO},
            "contact_stiffness_N_m": CONTACT_STIFFNESS, "contact_damping_Ns_m": CONTACT_DAMPING,
            "normalization": "nominal body inverse weight; contact Jacobian differs on the articulated robot",
            "target_scope": "left/right four-finger hulls pressed vertically against a plane",
            "thumb_scope": "same material as fingers, as in source; thumb curves reported without a matching-stiffness claim",
            "physx_equivalence": False,
            "curves": curves, "mass_release_probes": masses, "rigid_same_geometry_control": rigid,
            "actual_palm_probe": palm, "surface_friction": surface_friction_probe(asset_dir),
            "pyramidal_friction_baseline": surface_friction_probe(asset_dir, cone="pyramidal", impratio=1., check=False),
            "drop_sensitivity": [drop_probe(asset_dir, mass=m) for m in (.25, 1., 4.)]}


def friction_probe():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".005" gravity="0 0 0"/>
    <worldbody><body><joint name="j" type="hinge" axis="0 0 1" frictionloss=".893"/>
    <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = .3
    for _ in range(200):
        model.dof_frictionloss[0] = float(friction_threshold(torch.zeros(1, 29), torch.ones(1, 29))[0, 0])
        mujoco.mj_step(model, data)
    resting_velocity = float(data.qvel[0])
    data.qvel[0] = .5
    data.qfrc_applied[0] = 0
    model.dof_frictionloss[0] = float(friction_threshold(torch.full((1, 29), .5), torch.ones(1, 29))[0, 0])
    mujoco.mj_forward(model, data)
    force = float(data.qfrc_constraint[0])
    if abs(resting_velocity) > .01 or abs(force + .747) > .03:
        raise AssertionError(f"Native dry friction response differs: {resting_velocity}, {force}")
    return {"below_threshold_velocity_rad_s": resting_velocity, "moving_friction_Nm": force,
            "static_limit_Nm": .893, "dynamic_limit_Nm": .747,
            "note": "native soft constraint can creep; no duplicate explicit friction torque"}


def robot_probe(asset_dir=ASSET_DIR):
    # Add a real floor to the prepared model, leaving all robot physics intact.
    root = ET.parse(Path(asset_dir) / "g1_hw3.xml").getroot()
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    assets = {p.name: p.read_bytes() for p in (Path(asset_dir)/"source/meshes/g1").glob("*.STL")}
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="10 10 .1", friction=".3 0 0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    data = mujoco.MjData(model)
    from .g1_physics import DEFAULT_JOINT_POS
    import re
    q0 = np.array([next((v for p, v in DEFAULT_JOINT_POS.items() if re.fullmatch(p, n)), 0.) for n in G1_JOINT_NAMES])
    qids = [int(model.joint(n).qposadr[0]) for n in G1_JOINT_NAMES]
    vids = [int(model.joint(n).dofadr[0]) for n in G1_JOINT_NAMES]
    aids = [model.actuator("torque_"+n).id for n in G1_JOINT_NAMES]
    data.qpos[:7] = [0, 0, .76, 1, 0, 0, 0]
    data.qpos[qids] = q0
    motor = G1Motor(1, randomize=False)
    target = torch.tensor(q0, dtype=torch.float32).unsqueeze(0)
    motor.reset(torch.tensor([0]), target)
    peak_contacts, peak_torque, contact_impulse = 0, 0., 0.
    for _ in range(100):
        q = torch.tensor(data.qpos[qids], dtype=torch.float32).unsqueeze(0)
        qd = torch.tensor(data.qvel[vids], dtype=torch.float32).unsqueeze(0)
        tau = motor.compute(target, q, qd)
        data.ctrl[aids] = tau[0].numpy()
        model.dof_frictionloss[vids] = friction_threshold(qd, motor.friction_scale)[0].numpy()
        mujoco.mj_step(model, data)
        peak_contacts = max(peak_contacts, data.ncon)
        peak_torque = max(peak_torque, float(tau.abs().max()))
        force = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if model.geom_bodyid[c.geom1] == 0 or model.geom_bodyid[c.geom2] == 0:
                mujoco.mj_contactForce(model, data, i, force)
                contact_impulse += max(0, float(force[0])) * PHYSICS_DT
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise AssertionError("Nonfinite real-asset robot rollout")
    if peak_contacts == 0 or contact_impulse <= 0:
        raise AssertionError("No physical robot-floor contact in rollout")
    return {"physics_steps": 100, "duration_s": .5, "finite": True, "peak_contacts": peak_contacts,
            "peak_motor_torque_Nm": peak_torque, "floor_normal_impulse_Ns": contact_impulse,
            "final_root_height_m": float(data.qpos[2]), "scope": "untrained reset-pose PD smoke, not tracking success"}


def validate(asset_dir=ASSET_DIR):
    torch.set_num_threads(3)
    asset_dir = Path(asset_dir)
    verify_assets(asset_dir)
    model = load_model(asset_dir/"g1_hw3.xml")
    audit, _, _, _ = audit_model(model, asset_dir)
    soft = soft_hand_probes(asset_dir)
    check_soft_hand_probes(soft)
    return {"passed": True, "scope": "MuJoCo CPU only; no Isaac Lab/Newton GPU run or training",
            "mujoco_version": mujoco.__version__, "physics_contract": physics_contract(asset_dir),
            "compiled_asset": audit, "soft_hand": soft,
            "joint_friction": friction_probe(), "robot_rollout": robot_probe(asset_dir),
            "native_status": "pending Linux pinned Isaac Lab/Newton GPU smoke"}


def check_soft_hand_probes(report):
    """Validate only the experimentally defined response targets, not engine equivalence."""
    for curve in report["curves"]:
        if curve["part"] == "fingers" and not .5 <= curve["incremental_N_mm"] <= .6:
            raise AssertionError(f"Four-finger pad slope outside 0.5..0.6 N/mm: {curve}")
        if curve["incremental_N_mm"] <= 0:
            raise AssertionError("Hand press stiffness must be positive")
        for p in curve["presses"]:
            if abs(p["normal_force_N"]-p["load_N"]) > .03 or p["settled_velocity_max_m_s"] > .001:
                raise AssertionError(f"Hand press did not settle with correct load: {p}")
    for p in report["mass_release_probes"]:
        if abs(p["normal_force_N"]-p["load_N"]) > .03 or p["settled_velocity_max_m_s"] > .001:
            raise AssertionError(f"Mass-normalized hand press did not settle: {p}")
        release = p["release"]
        if release["normal_force_N"] > .01 or release["penetration_mm"] > .01:
            raise AssertionError(f"Unloaded pad did not separate: {release}")
    soft_at_5N = report["curves"][0]["presses"][1]["penetration_mm"]
    for rigid in (report["rigid_same_geometry_control"], report["actual_palm_probe"]):
        if rigid["penetration_mm"] >= soft_at_5N / 5:
            raise AssertionError("Rigid hand material is not substantially harder than fingers")
    low, high = report["surface_friction"]["results"]
    if abs(low["slide_velocity_m_s"]) >= .001 or high["slide_velocity_m_s"] < .1:
        raise AssertionError("Selected hand material/solver failed physical low-load/high-load slip criteria")
    for drop in report["drop_sensitivity"]:
        if not drop["finite"] or drop["settled_velocity_max_m_s"] > .001:
            raise AssertionError(f"Drop response did not converge numerically: {drop}")
        if abs(drop["final_normal_force_N"] - drop["nominal_load_N"]) > .05:
            raise AssertionError(f"Settled drop force does not balance weight: {drop}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new validation report path")
    report = validate(args.asset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "physics_contract"}, indent=2))


if __name__ == "__main__":
    main()
