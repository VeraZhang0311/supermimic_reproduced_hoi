"""Native GMT collection and differentiation check, with zero optimizer updates.

Uses the actual training helper definitions without importing train.py, whose
top level starts training. A successful report requires the full native runtime.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


HELPERS = ("_wm_rollout", "_imag_step", "_critic_input", "_reward_term", "_wm_err")


def load_training_helpers(namespace, source=None):
    """Execute only named function definitions, never training top-level code."""
    source = Path(source or Path(__file__).with_name("train.py"))
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in HELPERS]
    if sorted(node.name for node in selected) != sorted(HELPERS):
        raise ValueError("Actual training helper definitions are missing or duplicated")
    if any(node.decorator_list for node in selected):
        raise ValueError("Preflight requires undecorated training helper definitions")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in HELPERS}


def parameter_snapshot(models):
    return {f"{name}.{key}": value.detach().cpu().clone()
            for name, model in models.items() for key, value in model.named_parameters()}


def snapshot_hash(snapshot):
    digest = hashlib.sha256()
    for name, value in sorted(snapshot.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_unchanged(models, before):
    import torch
    after = parameter_snapshot(models)
    if before.keys() != after.keys() or any(not torch.equal(before[key], after[key]) for key in before):
        raise AssertionError("A preflight changed model parameters")
    return snapshot_hash(after)


def check_gradients(model):
    import torch
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError("Missing or nonfinite gradients")
    peak = max(float(g.abs().max()) for g in gradients)
    if peak == 0:
        raise AssertionError("All gradients are zero")
    return peak


def projected_replay_bytes(steps=1000, num_envs=4096):
    import torch
    from .buffer import FIELDS
    return steps * num_envs * sum((shape if isinstance(shape, int) else 1) * torch.empty((), dtype=dtype).element_size()
                                  for shape, dtype in FIELDS.values())


def learning_checks(buffer, motion, mdp, privileged, models, device, compile_helpers=False, horizon=32):
    """Differentiate actual WM and imagination paths; do not construct optimizers."""
    import torch
    from .actuator_state import ACTUATOR_DIM, advance_actuator_state, normalize_actuator_state
    from .mdp import clamp_action, normalize_xi, unflatten_state

    wm, pi, critic = (models[name] for name in ("wm", "pi", "critic"))
    B, K, Hh = 2, 10, 10
    floors = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2),
                        torch.full((3,), 5e-2), torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(device)
    namespace = dict(torch=torch, wm=wm, pi=pi, critic=critic, mdp=mdp, motion=motion, privileged=privileged,
                     args=SimpleNamespace(wm_window=8), K=K, normalize_xi=normalize_xi,
                     clamp_action=clamp_action,
                     unflatten_state=unflatten_state, advance_actuator_state=advance_actuator_state,
                     normalize_actuator_state=normalize_actuator_state, ERR_FLOOR=floors)
    helpers = load_training_helpers(namespace)
    if compile_helpers:
        # Preserve training's compiled entry points, including their backward.
        for name in ("_wm_rollout", "_imag_step", "_critic_input"):
            helpers[name] = torch.compile(helpers[name])
    w = buffer.sample_windows(B, 8)
    predicted = helpers["_wm_rollout"](w["s"], w["a"], w["xi"], w["actuator"])
    wm_loss, _ = helpers["_wm_err"](predicted, w["s1"], w["mask"])
    if not bool(torch.isfinite(wm_loss)):
        raise AssertionError("Nonfinite supervised WM loss")
    wm_loss.backward()
    wm_peak = check_gradients(wm)
    wm.zero_grad(set_to_none=True)

    # A long enough clip is required; sample without crossing its end.
    for _ in range(100):
        s, h = buffer.sample_starts(128, Hh)
        ok = (s["f"] + horizon <= motion.clip_end[motion.clip_id[s["f"]]]).nonzero().flatten()
        if len(ok) >= B:
            ids = ok[:B]
            s, h = ({key: value[ids] for key, value in group.items()} for group in (s, h))
            break
    else:
        raise AssertionError("No valid replay starts for H32 imagination")
    S = unflatten_state(s["s"])
    xi, f, ap, ctx = s["xi"], s["f"], s["a_prev"], s["actuator"]
    xn = normalize_xi(xi)
    hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    qp, qdp = s["q_prev"], s["qd_prev"]
    goal = motion.goal(f, K)
    critic.norm.update(helpers["_critic_input"](S, f, xn, goal, ctx))

    # The WM head is zero initialized. Check inclusion/differentiability of
    # context directly instead of pretending its initial dynamics Jacobian
    # should be nonzero before a single learning update.
    context_probe = ctx.detach().clone().requires_grad_()
    features = wm.inputs(S, ap, xn, context_probe)
    feature_grad = torch.autograd.grad(features[:, -ACTUATOR_DIM:].sum(), context_probe)[0]
    if not bool(torch.isfinite(feature_grad).all()) or float(feature_grad.abs().max()) == 0:
        raise AssertionError("New actuator context does not reach WM inputs differentiably")
    action_probe = ap.detach().clone().requires_grad_()
    next_probe = advance_actuator_state(ctx, action_probe, xi)
    advance_grad = torch.autograd.grad(next_probe[:, :29].sum(), action_probe)[0]
    if not bool(torch.isfinite(advance_grad).all()) or float(advance_grad.abs().max()) == 0:
        raise AssertionError("Analytic next-target context severed action gradients")

    for model in (wm, critic):
        model.requires_grad_(False)
    try:
        alive = torch.ones(B, device=device)
        objective = torch.zeros(B, device=device)
        for t in range(horizon):
            old_action = ap
            S, ap, hist, f, goal, qp, qdp, ctx = helpers["_imag_step"](S, hist, ap, f, xi, xn, goal, qp, qdp, ctx)
            reward, terminated = helpers["_reward_term"](S, f, ap, old_action)
            objective = objective + (.99 ** t) * alive * reward
            alive = alive * (1 - terminated)
        value = critic(helpers["_critic_input"](S, f, xn, goal, ctx))
        actor_loss = -(objective + (.99 ** horizon) * alive * value).mean()
        if not bool(torch.isfinite(actor_loss)):
            raise AssertionError("Nonfinite imagined actor loss")
        actor_loss.backward()
        actor_peak = check_gradients(pi)
        if any(p.grad is not None for p in wm.parameters()):
            raise AssertionError("Frozen WM unexpectedly accumulated actor gradients")
    finally:
        for model in (wm, critic):
            model.requires_grad_(True)
    return {"batch": B, "horizon": horizon, "wm_window": 8, "compiled_actual_helpers": compile_helpers,
            "wm_loss": float(wm_loss.detach()), "wm_gradient_max": wm_peak,
            "actor_loss": float(actor_loss.detach()), "actor_gradient_max": actor_peak,
            "actuator_feature_gradient_max": float(feature_grad.abs().max()),
            "analytic_target_action_gradient_max": float(advance_grad.abs().max()),
            "initial_wm_output_head_zero": bool((wm.net[-1].weight == 0).all()),
            "initial_dynamics_context_jacobian_note": "Zero by design at initialization; feature and analytic-memory gradients checked separately"}


def run(args, launch_simulation):
    import torch
    import warp as wp
    import supermimic.networks as networks
    from .actuator_state import ACTUATOR_DIM, advance_actuator_state
    from .buffer import ReplayBuffer
    from .env import TrackingSim, make_sim
    from .kinematics import G1Kinematics
    from .mdp import MDP, OBS_DIM, INV_DIM, CRITIC_EXTRA, clamp_action, normalize_xi, unflatten_state
    from .motion import load_reference
    from .networks import WorldModel, Policy, Critic
    from .privileged import PRIVILEGED_DIM, PrivilegedFeatures, RobotKinematics38, checkpoint_contract, validate_checkpoint
    from .robot import default_joint_pos_list

    device = args.device or "cuda:0"
    torch.manual_seed(0)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    if device.startswith("cuda"):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    kin = G1Kinematics(device)
    motion = load_reference("lafan1", str(args.motion_dir), kin, device)
    if len(motion.names) != 40:
        raise AssertionError("Preflight requires all 40 LAFAN motions")
    teacher = RobotKinematics38(device=device)
    privileged = PrivilegedFeatures(teacher, motion)
    contract = checkpoint_contract(motion, teacher)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    models = {"wm": WorldModel().to(device),
              "pi": Policy(OBS_DIM, 730, hist=10, privileged_dim=PRIVILEGED_DIM).to(device),
              "critic": Critic(INV_DIM + CRITIC_EXTRA + 730 + ACTUATOR_DIM).to(device)}
    before = parameter_snapshot(models)
    before_hash = snapshot_hash(before)
    wm, pi = models["wm"], models["pi"]
    cfg = make_sim(device, args)
    with launch_simulation(cfg, args):
        env = TrackingSim(cfg, args.num_envs, motion, mdp, max_episode_steps=8)
        from .g1_native_smoke import _check_motor_delivery
        delivery = {"checks": 0, "max_abs_error_Nm": 0., "nonzero_command_components": 0, "peak_command_Nm": 0.}
        original_step_target = env.physics.step_target
        def checked_step_target(target, after_substep=None):
            def check(dt):
                _check_motor_delivery(env.physics, delivery)
                env.physics.audit_solver_options()
                if after_substep is not None:
                    after_substep(dt)
            return original_step_target(target, after_substep=check)
        env.physics.step_target = checked_step_target
        buffer = ReplayBuffer(48, args.num_envs, device)
        hist = torch.zeros(args.num_envs, 10, OBS_DIM, device=device)
        qp, qdp = (x.clone() for x in env.state()[4:6])
        dones, timeouts, max_context_error = 0, 0, 0.
        with torch.no_grad():
            for _ in range(40):
                S = env.state()
                obs = mdp.obs(S, env.a_prev, env.xi, env.f)
                fresh = env.ep_step == 0
                qp, qdp = torch.where(fresh[:, None], S[4], qp), torch.where(fresh[:, None], S[5], qdp)
                hist = torch.where(fresh[:, None, None], obs[:, None].expand_as(hist), torch.cat((hist[:, 1:], obs[:, None]), 1))
                goal = motion.goal(env.f, 10)
                direct = privileged(S, env.f, env.a_prev, env.xi, qp, qdp, env.actuator_state())
                pi.obs_norm.update(obs)
                pi.goal_norm.update(goal)
                pi.privileged_norm.update(direct)
                action = clamp_action(pi(hist, goal, direct) + .1 * torch.randn(args.num_envs, 29, device=device))
                out = env.step(action)
                expected = advance_actuator_state(out["actuator"], out["a"], out["xi"])
                torch.testing.assert_close(expected, out["actuator1"], atol=2e-6, rtol=2e-6)
                max_context_error = max(max_context_error, float((expected - out["actuator1"]).abs().max()))
                done = out["done"]
                dones += int(done.sum())
                timeouts += int((done & ~out["term"]).sum())
                if bool(done.any()):
                    reset_context = env.actuator_state()[done]
                    torch.testing.assert_close(reset_context[:, :29], env.state()[4][done], atol=0, rtol=0)
                    if torch.equal(reset_context, out["actuator1"][done]):
                        raise AssertionError("Terminal context unexpectedly equals post-reset context")
                out["q_prev"], out["qd_prev"] = qp, qdp
                buffer.add(out)
                qp, qdp = S[4].clone(), S[5].clone()
                S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
                wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), out["actuator"]))
                wm.out_norm.update(wm.targets(S0, S1))
        if not dones:
            raise AssertionError("No auto-reset transition was exercised")
        if delivery["checks"] != 40 * 4 or not delivery["nonzero_command_components"]:
            raise AssertionError("Native torque/timestep coverage incomplete")
        results = learning_checks(buffer, motion, mdp, privileged, models, device, args.compile)
        if not all(bool(torch.isfinite(part).all()) for part in env.state()):
            raise AssertionError("Nonfinite final native state")
        if bool((wp.to_torch(env.physics.solver.mjw_data.overflow) != 0).any()):
            raise AssertionError("Final native contact/constraint overflow")
        env.physics.audit_solver_options()
        # Only temporary checkpoint bytes; no training artifact or optimizer.
        legacy_rejected = False
        try:
            validate_checkpoint({}, contract)
        except ValueError:
            legacy_rejected = True
        if not legacy_rejected:
            raise AssertionError("Legacy checkpoint was accepted")
        with tempfile.TemporaryFile() as stream:
            torch.save({"gmt_contract": contract, **{name: model.state_dict() for name, model in models.items()}}, stream)
            stream.seek(0)
            checkpoint = torch.load(stream, map_location=device, weights_only=False)
            validate_checkpoint(checkpoint, contract)
            for name, model in models.items():
                model.load_state_dict(checkpoint[name], strict=True)
        after_hash = assert_unchanged(models, before)
        return {"passed": True, "scope": "native full-40 dataset collection, replay, actual GMT helper differentiation; no optimizer",
                "optimizer_updates": 0, "optimizer_constructed": False, "training_started": False,
                "motion_count": len(motion.names), "motion_manifest_sha256": motion.manifest_sha256,
                "device": device, "num_envs": args.num_envs, "collected_control_steps": 40,
                "done_transitions": dones, "timeout_only_transitions": timeouts,
                "native_motor_delivery": delivery,
                "context_transition_max_abs_error": max_context_error,
                "terminal_context_captured_before_reset": True, "learning_path": results,
                "parameters_unchanged_torch_equal": True, "parameters_sha256_before": before_hash,
                "parameters_sha256_after": after_hash, "legacy_checkpoint_rejected": True,
                "matching_checkpoint_roundtrip": True,
                "replay_bytes_T1000_N4096": projected_replay_bytes(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else None,
                "formal_4096_env_throughput_validated": False, "gmt_contract": contract}


def main():
    from .g1_native_smoke import _write_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--motion_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    basic, _ = parser.parse_known_args()
    if basic.output.exists() or basic.num_envs < 2:
        parser.error("Use a new report path and at least two environments")
    basic.output.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ("isaaclab", "isaaclab_newton", "newton", "warp", "mujoco_warp")
               if importlib.util.find_spec(name) is None]
    if missing:
        report = {"passed": False, "native_status": "unavailable", "missing_modules": missing,
                  "optimizer_updates": 0, "training_started": False}
        _write_report(basic.output, report)
        print(json.dumps(report, indent=2))
        return 2
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    try:
        report = run(args, launch_simulation)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report = {"passed": False, "native_status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "optimizer_updates": 0, "training_started": False}
        _write_report(args.output, report)
        return 1
    _write_report(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "gmt_contract"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
