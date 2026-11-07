"""Evaluate a deterministic GMT + residual checkpoint on the fixed canonical HOI80 panel.

Put the validated gmt26k_hoi80_zeroshot_20261106_03/source directory second on
PYTHONPATH for its hoi_zero_shot package. The first source is the residual
checkpoint's own code. Run one fresh simulator process per object asset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .data import ASSETS, sha256_file
from .evaluate import _write_json, frozen_residual


REWARD_TERMS = (
    "joint_pos", "joint_vel", "root_height", "root_rotation", "root_linear_vel",
    "root_angular_vel", "body_local", "body_global", "object_surface", "action_rate",
    "joint_limits", "joint_speed", "joint_accel",
)


def _verified_protocol():
    """Bind the same selection, scoring and initialization used for the GMT baseline."""
    import hoi_zero_shot
    package = Path(hoi_zero_shot.__file__).resolve().parent
    expected = {
        "data.py": "21b834f57da4e388a3c108ca8b34e108a698d034eb5077784421b3aa04138fc1",
        "metrics.py": "087742c554f3cf8de5edd83966c9e9479d6614eb1424983c11c5fc35d7700c17",
        "native.py": "450e066fb6dfc7cdb85c8b3b0ca9290556868bf9f19596f88c4977bc1bb53f2c",
        "ultra_kinematics.py": "c5547f78fedbab91e78ea9f9eeb1dc68b232e65f279a2c079a916b0539890710",
        "ultra_math.py": "274d172996fe40f3a5745cf6ea09d617448e2c33e8d15ad8f92ea0bd3b604f85",
    }
    for name, digest in expected.items():
        if sha256_file(package / name) != digest:
            raise ValueError(f"canonical HOI80 evaluator implementation differs: {name}")
    return expected


@torch.no_grad()
def collect(env, motion, mdp, privileged, teacher_kin, gmt, policy, points, audit):
    """Collect T+1 native states and T applied actions per lane after formal reset."""
    from .features import ObjectHistory, object_observation, oracle_features, torso_pose
    from supermimic.mdp import clamp_action, flatten_state
    from supermimic.rotations import mat_to_quat

    starts, ends = motion.clip_start, motion.clip_end
    steps = int((ends - starts).max())
    if len(starts) != env.N or bool((ends <= starts).any()) or not torch.equal(env.f, starts):
        raise ValueError("canonical lanes must start at the first state of their assigned clips")
    def storage(length, width):
        return torch.empty(length, env.N, width, device=env.device)
    states, objects = storage(steps + 1, 71), storage(steps + 1, 13)
    base_actions, residuals, actions = (storage(steps, 29) for _ in range(3))
    rewards = torch.empty(steps, env.N, device=env.device)
    terminated = torch.empty_like(rewards, dtype=torch.bool)
    reward_terms = storage(steps, len(REWARD_TERMS))
    surface_error = torch.empty_like(rewards)

    def record(t, state, obj):
        states[t] = flatten_state(state)
        p, rotation, v, w = obj
        quat = mat_to_quat(rotation)
        objects[t] = torch.cat((p, quat[:, 1:], quat[:, :1], v, w), -1)

    state, obj = env.state(), env.object_state()
    record(0, state, obj)
    qprev, qdprev = state[4].clone(), state[5].clone()
    obs = mdp.obs(state, env.a_prev, env.xi, env.f)
    history = obs[:, None].expand(-1, 10, -1).clone()
    object_history = ObjectHistory(object_observation(torso_pose(teacher_kin, state), obj))
    for t in range(steps):
        # Completed lanes continue independently. Their padded states/actions
        # are discarded; never ask their motion adapter for the next clip.
        env.f = torch.minimum(starts + t, ends - 1)
        direct = privileged(state, env.f, env.a_prev, env.xi, qprev, qdprev, env.actuator_state())
        base = gmt(history, motion.goal(env.f, 10), direct)
        oracle = oracle_features(direct, torso_pose(teacher_kin, state), obj, motion, env.f,
                                 env.object_context(), object_history.values)
        residual = policy.act(oracle, base, points, deterministic=True)
        action = clamp_action(base + residual)
        out = env.step(action)
        actions[t], base_actions[t], residuals[t] = action, base, residual
        rewards[t], terminated[t] = out["reward"], out["terminated"]
        torch.stack([out["terms"][name] for name in REWARD_TERMS], dim=-1, out=reward_terms[t])
        surface_error[t] = out["terms"]["object_surface_error_m"]
        qprev, qdprev = out["previous_q"], out["previous_qd"]
        state, obj = env.state(), env.object_state()
        record(t + 1, state, obj)
        audit.sample_contacts()
        obs = mdp.obs(state, env.a_prev, env.xi, env.f)
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        object_history.advance(object_observation(torso_pose(teacher_kin, state), obj))
    if not torch.equal(env.f, ends):
        raise RuntimeError("canonical formal rollouts failed to reach the exact final states")
    return {"robot_state_wxyz": states, "object_state_xyzw": objects, "action": actions,
            "base_action": base_actions, "residual_action": residuals,
            "residual_reward": rewards, "residual_termination": terminated,
            "object_surface_error_m": surface_error, "reward_terms": reward_terms,
            "object_surface": reward_terms[..., REWARD_TERMS.index("object_surface")],
            "action_rate": reward_terms[..., REWARD_TERMS.index("action_rate")]}


def score_lane(output, motion, lane, reference, rollout, virtual, mdp):
    from hoi_zero_shot.metrics import score
    from supermimic.mdp import MAX_STEP_REWARD, unflatten_state

    case = motion.cases[lane]
    steps = case["scored_control_steps"]
    arrays = {name: batch[:steps + int(name in ("robot_state_wxyz", "object_state_xyzw")), lane].cpu()
              for name, batch in rollout.items()}
    contributions = arrays["reward_terms"]
    if (contributions.shape != (steps, len(REWARD_TERMS)) or not torch.isfinite(contributions).all() or
            not torch.allclose(contributions.sum(-1), arrays["residual_reward"], rtol=1e-5, atol=5e-7)):
        raise ValueError(f"HOI80 reward contributions do not sum to the valid per-step rewards of {case['id']}")
    per_step = dict(zip(REWARD_TERMS, contributions.double().mean(0).tolist(), strict=True))
    state = unflatten_state(rollout["robot_state_wxyz"][:steps + 1, lane])
    body = virtual(state).cpu()
    metrics = score(body, arrays["object_state_xyzw"][:, :7], reference, dt=.02)
    action = rollout["action"][:steps, lane]
    frames = torch.arange(steps, device=action.device) + motion.clip_start[lane] + 1
    previous = torch.cat((torch.zeros_like(action[:1]), action[:-1]))
    post_state = tuple(value[1:] for value in state)
    author_reward = mdp.reward(post_state, frames, action, previous)
    author_termination = mdp.termination(post_state, frames)
    failed = author_termination.nonzero().flatten()
    first = int(failed[0]) + 1 if len(failed) else None
    # This is the original robot MDP statistic, distinct from the full-duration HOI return.
    author_normalized = float(author_reward[:first].sum() / (steps * MAX_STEP_REWARD))
    arrays.update(author_reward=author_reward.cpu(), author_termination=author_termination.cpu())
    target = output / (case["id"] + ".npz")
    with target.open("xb") as stream:
        np.savez_compressed(stream, **{key: value.numpy() for key, value in arrays.items()},
                            reward_term_names=np.array(REWARD_TERMS),
                            body_pos39=body.numpy(), reference=reference, time=np.arange(steps + 1) * .02,
                            body_names=np.array(virtual.body_names), joint_names=np.array(virtual.joint_names))
    clipped = arrays["action"] != (arrays["base_action"] + arrays["residual_action"])
    return dict(metrics, id=case["id"], object_asset=case["object_asset"], source_group=case["source_group"],
                canonical_index=case["canonical_index"], motion_sha256=case["motion_sha256"],
                reference_sha256=case["reference_sha256"], control_steps=steps, state_frames=steps + 1,
                full_duration=True, resets_within_clip=0, clipped_action_steps=int(clipped.any(-1).sum()),
                hoi_return=float(arrays["residual_reward"].sum()),
                reward_terms_per_step=per_step,
                mean_hoi_reward_per_step=float(arrays["residual_reward"].mean()),
                mean_object_surface_error_m=float(arrays["object_surface_error_m"].mean()),
                mean_object_surface_reward_per_step=float(arrays["object_surface"].mean()),
                mean_action_rate_reward_per_step=float(arrays["action_rate"].mean()),
                first_author_robot_failure_step=first, author_robot_normalized_return=author_normalized,
                rollout=target.name, rollout_sha256=sha256_file(target))


def summarize(cases):
    summary = {"clips": len(cases), "control_steps": sum(case["control_steps"] for case in cases),
            **{key + "_count": sum(case[key] for case in cases)
               for key in ("human_pass", "object_pass", "joint_pass")},
            **{key + "_rate": sum(case[key] for case in cases) / len(cases)
               for key in ("human_pass", "object_pass", "joint_pass")},
            "equal_clip_means": {
                key: sum(case[key] for case in cases) / len(cases) for key in
                ("mean_hoi_reward_per_step", "mean_object_surface_error_m",
                 "mean_object_surface_reward_per_step", "mean_action_rate_reward_per_step",
                 "author_robot_normalized_return")}}
    # Historical reports predate component collection. Never turn missing
    # terms into zeros or average over a subset of the panel.
    if any("reward_terms_per_step" in case for case in cases):
        if any(set(case.get("reward_terms_per_step", {})) != set(REWARD_TERMS) for case in cases):
            raise ValueError("HOI80 reward decomposition is incomplete across clips")
        summary["reward_terms_per_step"] = {
            key: sum(case["reward_terms_per_step"][key] for case in cases) / len(cases) for key in REWARD_TERMS}
    return summary


def load_teacher(args, motion, teacher_kin, device):
    from .policy import FrozenGMT
    from .shape import load_shape_bank
    import supermimic.networks as networks
    from supermimic.privileged import checkpoint_contract

    gmt = FrozenGMT(args.gmt_checkpoint, checkpoint_contract(motion, teacher_kin),
                    device=device, expected_iteration=args.gmt_iteration)
    networks.AMP_DTYPE = torch.bfloat16 if gmt.metadata["source_amp"] == "bf16" else None
    bank, shape = load_shape_bank(args.shape_bank, args.asset_bundle)
    catalog_sha = sha256_file(args.asset_bundle)
    asset_name = motion.object_assets[0 if args.asset is None else args.asset]
    policy, checkpoint, digest = frozen_residual(
        args.checkpoint, device, gmt_metadata=gmt.metadata,
        manifest_sha256=motion.selection_contract["ultra_manifest_sha256"],
        object_assets=motion.object_assets, asset_bundle_sha256=catalog_sha,
        selected_asset=asset_name, shape_metadata=shape)
    if (digest != args.checkpoint_sha256 or checkpoint["iteration"] != args.iteration
            or checkpoint.get("smoke") is not False or set(checkpoint["training_assets"]) != set(ASSETS)):
        raise ValueError("expected the specified formal eight-asset PPO checkpoint SHA and iteration")
    residual_before = checkpoint["residual"]  # Loaded on CPU, independent of live CUDA parameters/RMS.
    identity = {"schema": "residual_canonical_hoi80_v1", "checkpoint_sha256": digest,
                "training_reward_profile": checkpoint.get("experiment", {}).get("reward_profile", "resmimic"),
                "evaluation_reward_profile": "resmimic",
                "iteration": args.iteration, "gmt_checkpoint_sha256": gmt.metadata["checkpoint_sha256"],
                "gmt_iteration": gmt.metadata["iteration"], "canonical_manifest_sha256": motion.manifest_sha256,
                "ultra_manifest_sha256": motion.selection_contract["ultra_manifest_sha256"],
                "asset_catalog_sha256": catalog_sha, "shape_bank_sha256": shape["bank_sha256"]}
    return gmt, policy, bank, residual_before, identity


def preflight(args):
    from hoi_zero_shot.data import Canonical80Motion
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import RobotKinematics38

    protocol_files = _verified_protocol()
    motion = Canonical80Motion(args.manifest, args.canonical, G1Kinematics("cpu"), "cpu", asset=args.asset)
    gmt, policy, points, _, identity = load_teacher(args, motion, RobotKinematics38(device="cpu"), "cpu")
    gmt.assert_unchanged()
    if policy.training or any(p.requires_grad for p in policy.parameters()):
        raise RuntimeError("preflight residual must be frozen in eval mode")
    result = dict(identity, protocol_files_sha256=protocol_files, device="cpu",
                  clips=len(motion.cases), control_steps=int((motion.clip_len - 1).sum()),
                  shape_points=list(points.shape), physical_steps=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "preflight.json", result)
    print(json.dumps(result), flush=True)


def worker(args, launch_simulation):
    from hoi_zero_shot.data import Canonical80Motion
    from hoi_zero_shot.metrics import VirtualPHC
    from hoi_zero_shot.native import NativeSceneAuditor, initialize_canonical_object
    from .assets import load_assets
    from .env import HOISim
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.privileged import PrivilegedFeatures, RobotKinematics38
    from supermimic.robot import default_joint_pos_list

    protocol_files = _verified_protocol()
    device = args.device or "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(2024)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    kin = G1Kinematics(device)
    motion = Canonical80Motion(args.manifest, args.canonical, kin, device, asset=args.asset)
    asset_name = motion.object_assets[args.asset]
    teacher_kin = RobotKinematics38(device=device)
    privileged = PrivilegedFeatures(teacher_kin, motion)
    gmt, policy, bank, residual_before, identity = load_teacher(args, motion, teacher_kin, device)
    points = bank[args.asset].to(device)
    virtual = VirtualPHC(motion.legacy_urdf_path, motion.legacy_body_names, motion.legacy_joint_names, device)
    assets = {entry["object_asset"]: entry for entry in load_assets(args.asset_bundle)}
    asset = dict(assets[asset_name], asset_id=args.asset)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    references = [motion.load_reference(lane) for lane in range(len(motion.cases))]
    cfg = make_sim(device, args)
    cfg.physics.solver_cfg.nconmax, cfg.physics.solver_cfg.njmax = 512, 2048
    with launch_simulation(cfg, args), torch.no_grad():
        env = HOISim(cfg, len(references), motion, mdp, asset, output, eval_mode=True)
        env.step(torch.zeros(env.N, 29, device=device))
        auditor = NativeSceneAuditor(env)
        pre = auditor.audit(phase="unscored_solver_sync")
        curriculum = {key: getattr(env, key).clone() for key in ("bin_G", "bin_w_avg")}
        env.reset(torch.arange(env.N, device=device), motion.clip_start.clone())
        initialization = initialize_canonical_object(env, np.stack([ref[0] for ref in references]))
        env.refresh()
        body0 = virtual(env.state())
        physical = env.robot.data.body_link_pose_w.torch[..., :3] - env.scene.env_origins[:, None, :]
        common = virtual.audit_common_positions(body0, physical, list(env.robot.body_names))
        rollout = collect(env, motion, mdp, privileged, teacher_kin, gmt, policy, points, auditor)
        post = auditor.audit(phase="full_duration_complete")
        cases = [score_lane(output, motion, lane, ref, rollout, virtual, mdp)
                 for lane, ref in enumerate(references)]
        gmt.assert_unchanged()
        if any(not torch.equal(value.cpu(), residual_before[key]) for key, value in policy.state_dict().items()):
            raise RuntimeError("HOI80 evaluation changed residual parameters or RMS")
        if any(not torch.equal(value, getattr(env, key)) for key, value in curriculum.items()):
            raise RuntimeError("HOI80 evaluation changed the reset-sampling curriculum")
        if sha256_file(args.checkpoint) != identity["checkpoint_sha256"]:
            raise RuntimeError("evaluated residual checkpoint changed on disk")
    report = {**identity, "asset": asset_name,
              "protocol_files_sha256": protocol_files, "selection": motion.selection_contract,
              "initialization": initialization, "virtual39_identity": virtual.identity,
              "virtual39_native_positions": common, "physics_pre": pre, "physics_post": post,
              "optimizer_updates": 0, "normalizer_updates": 0, "domain_randomization": False,
              "parameters_and_buffers_unchanged": True, "reset_curriculum_unchanged": True,
              "deterministic": True, "full_duration_after_failure": True, "initial_state_scored": False,
              "summary": summarize(cases), "cases": cases}
    _write_json(output / "report.json", report)
    print(json.dumps({"asset": asset_name, **report["summary"]}), flush=True)


def aggregate(reports, canonical):
    if [r["asset"] for r in reports] != list(ASSETS):
        raise ValueError("HOI80 reports must cover the eight distinct assets in catalog order")
    keys = ("schema", "checkpoint_sha256", "iteration", "gmt_checkpoint_sha256", "gmt_iteration",
            "canonical_manifest_sha256", "ultra_manifest_sha256", "asset_catalog_sha256",
            "shape_bank_sha256", "protocol_files_sha256")
    identity = {key: reports[0][key] for key in keys}
    for key in ("training_reward_profile", "evaluation_reward_profile"):
        if key in reports[0]:
            identity[key] = reports[0][key]
    if any(any(report[key] != value for key, value in identity.items()) for report in reports):
        raise ValueError("HOI80 evaluation identity changed across asset reports")
    cases = sorted((case for report in reports for case in report["cases"]), key=lambda case: case["canonical_index"])
    if (len(cases) != 80 or len(canonical["cases"]) != 80 or
            any((case["canonical_index"], case["id"], case["object_asset"], case["scored_frames"])
                != (index, expected["id"], expected["object_asset"], expected["scored_control_steps"])
                for index, (case, expected) in enumerate(zip(cases, canonical["cases"])))):
        raise ValueError("HOI80 panel or scored duration differs from canonical80")
    return {**identity, "summary": summarize(cases), "cases": cases, "criteria": cases[0]["criteria"],
            "by_asset": {report["asset"]: report["summary"] for report in reports},
            "optimizer_updates": 0, "normalizer_updates": 0, "domain_randomization": False,
            "deterministic": True, "full_duration_after_failure": True, "initial_state_scored": False,
            "asset_reports": [f"asset_{i:02d}/report.json" for i in range(8)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "checkpoint-sha256", "gmt-checkpoint", "manifest", "canonical",
                 "asset-bundle", "shape-bank", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--gmt-iteration", type=int, default=4000)
    parser.add_argument("--asset", type=int, choices=range(8))
    parser.add_argument("--preflight", action="store_true", help="CPU model/data checks, writes preflight.json")
    if "--preflight" in sys.argv:
        parser.add_argument("--device", choices=("cpu",), default="cpu")
        preflight(parser.parse_args())
        return
    from .runtime import configure_warp_forward_only
    configure_warp_forward_only()
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.asset is not None:
        worker(args, launch_simulation)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for asset in range(8):
        target = root / f"asset_{asset:02d}"
        subprocess.run([sys.executable, "-m", "hoi_residual.evaluate_hoi80", *sys.argv[1:],
                        "--asset", str(asset), "--output", str(target)], check=True)
        reports.append(json.loads((target / "report.json").read_text(encoding="utf-8")))
    if sha256_file(args.canonical) != reports[0]["canonical_manifest_sha256"]:
        raise RuntimeError("canonical HOI80 manifest changed during evaluation")
    result = aggregate(reports, json.loads(Path(args.canonical).read_text(encoding="utf-8")))
    _write_json(root / "report.json", result)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
