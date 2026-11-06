"""Validate the real Isaac Lab/Newton G1 port without training or a policy.

Example in the pinned Linux GPU runtime:
  python -m supermimic.g1_native_smoke --num_envs 2 --steps 25 \
      --output .cache/hw3_native_smoke.json

Missing dependencies produce a non-passing 'unavailable' report, not a
successful mock. The output path must be new. No motion library, checkpoint,
optimizer, W&B run, queue submission or training entry point is loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


def _check_motor_delivery(physics, statistics):
    """Compare submitted motor torque with actual native solver inputs.

    The smoke has no user-applied external joint/body force. Gravity and
    passive/constraint forces are deliberately NOT added here. Depending on
    the Newton actuator route, user torque can reach qfrc_applied or
    qfrc_actuator; their sum must equal the single explicit hw3 command.
    """
    import torch
    data = physics.solver.mjw_data
    for name in ("qfrc_applied", "qfrc_actuator"):
        if not hasattr(data, name):
            raise RuntimeError(f"Native motor delivery cannot be verified: missing mjw_data.{name}")
    applied = physics.wp.to_torch(data.qfrc_applied)
    actuator = physics.wp.to_torch(data.qfrc_actuator)
    expected = physics.last_motor_effort
    if applied.ndim != 2 or actuator.shape != applied.shape or applied.shape[0] != physics.N:
        raise RuntimeError(f"Unsupported native torque telemetry layout: applied={tuple(applied.shape)}, actuator={tuple(actuator.shape)}")
    actual = applied[:, physics.dofs] + actuator[:, physics.dofs]
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()):
        raise RuntimeError("Native motor torque telemetry is nonfinite or has the wrong joint layout")
    error = float((actual - expected).abs().max())
    statistics["max_abs_error_Nm"] = max(statistics["max_abs_error_Nm"], error)
    statistics["checks"] += 1
    statistics["nonzero_command_components"] += int((expected.abs() > 1e-3).sum())
    statistics["peak_command_Nm"] = max(statistics["peak_command_Nm"], float(expected.abs().max()))
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-6,
                               msg=f"Explicit hw3 torque did not reach the native solver exactly once; max error {error} Nm")


def run(args, launch_simulation):
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate
    from isaaclab.sim import SimulationContext
    from newton import ModelFlags

    from .env import TrackingSceneCfg, make_sim
    from .g1_physics import CONTROL_DECIMATION, PHYSICS_DT
    from .g1_runtime import G1PhysicsRuntime

    device = args.device or "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the requested native GPU smoke")
    torch.manual_seed(0)
    cfg = make_sim(device, args)
    with launch_simulation(cfg, args):
        sim = SimulationContext(cfg)
        scene = instantiate(TrackingSceneCfg(num_envs=args.num_envs, env_spacing=0., replicate_physics=True))
        sim.reset()
        # Keep a full identity/geometry receipt even when the subsequent
        # strict audit fails. This is inspection output, never training data.
        model = sim.backend_manager._solver.mj_model
        newton_model = sim.backend_manager.get_model()
        # Save the exact compiled model for same-version CPU diagnosis. Use
        # MuJoCo's memory buffer API so Python's exclusive create enforces
        # no-overwrite semantics (also avoiding native Unicode path issues).
        binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, binary)
        with args.output.with_suffix(".mjb").open("xb") as stream:
            stream.write(binary.tobytes())
        _write_report(args.output.with_suffix(".model.json"), {
            "scope": "compiled native model before hw3 audit; no training",
            "mujoco_version": mujoco.__version__, "compiled_model_file": args.output.with_suffix(".mjb").name,
            "geoms": [{"name": model.geom(i).name, "body_id": int(model.geom_bodyid[i]),
                       "type": int(model.geom_type[i]), "contype": int(model.geom_contype[i]),
                       "conaffinity": int(model.geom_conaffinity[i]), "size": model.geom_size[i].tolist(),
                       "pos": model.geom_pos[i].tolist(), "quat": model.geom_quat[i].tolist()}
                      for i in range(model.ngeom)],
            "bodies": [{"name": model.body(i).name, "parent_id": int(model.body_parentid[i]),
                        "mass": float(model.body_mass[i]), "ipos": model.body_ipos[i].tolist(),
                        "inertia": model.body_inertia[i].tolist(), "iquat": model.body_iquat[i].tolist()}
                       for i in range(model.nbody)],
            "joints": [{"name": model.joint(i).name, "type": int(model.jnt_type[i]),
                        "qpos_adr": int(model.jnt_qposadr[i]), "dof_adr": int(model.jnt_dofadr[i]),
                        "range": model.jnt_range[i].tolist()} for i in range(model.njnt)],
            "newton_labels": {key: list(getattr(newton_model, key)) for key in
                              ("shape_label", "body_label", "joint_label")}})
        robot = scene["robot"]
        physics = G1PhysicsRuntime(sim, robot, scene, randomize=True)
        solver_options = physics.audit_solver_options()
        target = robot.data.default_joint_pos.torch.clone()
        initial_root = robot.data.root_link_pose_w.torch.clone()
        ids = torch.arange(args.num_envs, device=device)
        physics.reset(ids, target)
        # A small explicit command makes a zero-drive route fail even if the
        # reset pose happens to be held by joint friction without motor effort.
        target[:, robot.joint_names.index("left_elbow_joint")] += .01
        for field in (robot.data.joint_stiffness.torch, robot.data.joint_damping.torch):
            if not bool((field == 0).all()):
                raise AssertionError("A native servo is still active beside explicit hw3 PD")
        if sim.backend_manager._num_substeps != 1:
            raise AssertionError("Explicit motor updates must bracket every 5 ms solve")
        readback = physics.readback()
        counts = []
        max_constraint_effort = 0.
        partial_reset_verified = False
        external_material_verified = False
        delivery = {"checks": 0, "max_abs_error_Nm": 0., "nonzero_command_components": 0,
                    "peak_command_Nm": 0., "atol_Nm": 2e-4, "rtol": 2e-6,
                    "telemetry": "qfrc_applied + qfrc_actuator, named robot DOFs",
                    "probe_target_offset_rad": {"left_elbow_joint": .01}}
        def after_substep(_dt):
            _check_motor_delivery(physics, delivery)
            # Newton fills the live Warp timestep from its step(dt) argument;
            # verify that actual value after every 5 ms solve.
            physics.audit_solver_options()
        for step in range(args.steps):
            if step == args.steps // 2:
                # Isolated environment reset: no command history or sampled
                # delay/torque scale may leak to or change the other worlds.
                other_queue = physics.motor.queue[:, 1:].clone()
                other_delay = physics.motor.delay[1:].clone()
                other_scale = physics.motor.scale[1:].clone()
                reset_ids = ids[:1]
                robot.write_root_link_pose_to_sim_index(root_pose=initial_root[:1], env_ids=reset_ids)
                robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device), env_ids=reset_ids)
                robot.write_joint_position_to_sim_index(position=target[:1], env_ids=reset_ids)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 29, device=device), env_ids=reset_ids)
                scene.reset(reset_ids)
                physics.reset(reset_ids, target[:1])
                torch.testing.assert_close(physics.motor.queue[:, 1:], other_queue, atol=0, rtol=0)
                torch.testing.assert_close(physics.motor.delay[1:], other_delay, atol=0, rtol=0)
                torch.testing.assert_close(physics.motor.scale[1:], other_scale, atol=0, rtol=0)
                torch.testing.assert_close(physics.motor.queue[:, 0], target[0].expand(5, -1), atol=0, rtol=0)
                partial_reset_verified = True
                # A real backend property notification can reconstruct contact
                # fields. The hw3 hook must restore its own material afterward.
                sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
            before = physics.motor.tick
            physics.step_target(target, after_substep=after_substep)
            if physics.motor.tick != (before + CONTROL_DECIMATION) % 5:
                raise AssertionError("A policy interval did not execute four explicit motor updates")
            readback = physics.readback()
            if not bool(torch.isfinite(robot.data.root_link_pose_w.torch).all()):
                raise AssertionError("Nonfinite floating-base state")
            # MuJoCo Warp stores the total contact count across worlds in
            # nacon (not MuJoCo CPU's MjData.ncon).
            contacts = getattr(physics.solver.mjw_data, "nacon", None)
            if contacts is not None:
                counts.append(int(wp.to_torch(contacts).max()))
            constraint_effort = wp.to_torch(physics.solver.mjw_data.qfrc_constraint)
            if not bool(torch.isfinite(constraint_effort).all()):
                raise AssertionError("Nonfinite native constraint effort")
            max_constraint_effort = max(max_constraint_effort, float(constraint_effort.abs().max()))
        if not counts or max(counts) == 0 or max_constraint_effort == 0:
            raise AssertionError("Native smoke has no observed solved physical contact")
        # Verify restoring G1 contact rows leaves external material changes
        # intact. The world plane is deliberately changed for one final step.
        m = physics.solver.mj_model
        external_ids = [i for i in range(m.ngeom) if i not in physics.geom_ids.values() and int(m.geom_bodyid[i]) == 0]
        if not external_ids:
            raise AssertionError("Smoke scene has no independent world material")
        friction = wp.to_torch(physics.solver.mjw_model.geom_friction)
        external_id = external_ids[0]
        friction[:, external_id, 0] = .17
        physics.step_target(target, after_substep=after_substep)
        torch.testing.assert_close(friction[:, external_id, 0], torch.full_like(friction[:, external_id, 0], .17), atol=0, rtol=0)
        external_material_verified = True
        readback = physics.readback()
        if delivery["checks"] != (args.steps + 1) * CONTROL_DECIMATION or delivery["nonzero_command_components"] == 0:
            raise AssertionError("Native motor delivery was not checked for every nonzero-command substep")
        solver_options_after_steps = physics.audit_solver_options()
        # This smoke is not an assertion of tracking ability or PhysX parity.
        return {"passed": True, "scope": "real Isaac Lab/Newton/MuJoCo Warp; no training or policy",
                "device": device, "num_envs": args.num_envs, "control_intervals": args.steps + 1,
                "physics_dt_s": PHYSICS_DT, "motor_updates_per_interval": CONTROL_DECIMATION,
                "compiled_asset": physics.report, "native_readback": readback,
                "solver_options": solver_options,
                "solver_options_after_steps": solver_options_after_steps,
                "partial_reset_isolation": partial_reset_verified,
                "external_material_preservation": external_material_verified,
                "max_observed_contacts_all_worlds": max(counts) if counts else None,
                "max_constraint_effort": max_constraint_effort,
                "motor_delivery": delivery,
                "physics_contract": physics.contract,
                "training_started": False, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    basic, _ = parser.parse_known_args()
    if basic.output.exists():
        parser.error("Use a new native smoke report path")
    if basic.num_envs < 2 or basic.steps < 2:
        parser.error("Native isolation validation requires >=2 environments and >=2 steps")
    missing = [name for name in ("isaaclab", "isaaclab_newton", "newton", "warp", "mujoco_warp")
               if importlib.util.find_spec(name) is None]
    if missing:
        report = {"passed": False, "native_status": "unavailable", "missing_modules": missing,
                  "scope": "dependency check only; no native simulation performed", "training_started": False}
        _write_report(basic.output, report)
        print(json.dumps(report, indent=2))
        return 2
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    try:
        report = run(args, launch_simulation)
    except Exception as exc:
        report = {"passed": False, "native_status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "scope": "native smoke attempt, no training", "training_started": False}
        _write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return 1
    _write_report(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "physics_contract"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
