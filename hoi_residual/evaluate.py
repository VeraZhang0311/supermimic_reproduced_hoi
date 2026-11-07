"""Deterministic, full-duration Protected eval for a frozen HOI teacher.

No optimizer is constructed and no normalizer is updated. Failure flags are
recorded while physics continues to the exact reference end. Each process has
one fixed object asset and evaluates clips sequentially in one environment.
Omit --asset to run all eight assets in separate sequential child processes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def validate_metadata(checkpoint, *, architecture, gmt_metadata, manifest_sha256,
                      object_assets, asset_bundle_sha256, selected_asset, shape_metadata):
    """Strict identities before any protected trajectory is evaluated."""
    if checkpoint.get("schema") != "hoi_residual_ppo_teacher_v2":
        raise ValueError("unsupported residual checkpoint schema")
    from .features import observation_contract
    if checkpoint.get("observation_contract") != observation_contract():
        raise ValueError("residual checkpoint observation contract mismatch")
    if (checkpoint.get("shape_bank_sha256") != shape_metadata['bank_sha256'] or
            checkpoint.get("shape_contract") != shape_metadata['contract']):
        raise ValueError("residual checkpoint FPS128 shape bank mismatch")
    from .contract import compatible_for_evaluation
    profile = checkpoint.get("experiment", {}).get("reward_profile", "resmimic")
    if not compatible_for_evaluation(checkpoint.get("residual_contract"), profile):
        raise ValueError("residual checkpoint learning implementation identity mismatch")
    if checkpoint.get("architecture") != architecture:
        raise ValueError("residual network/observation architecture mismatch")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("residual checkpoint ULTRA split manifest mismatch")
    if checkpoint.get("assets") != list(object_assets):
        raise ValueError("residual checkpoint exact asset order mismatch")
    if checkpoint.get("asset_bundle_sha256") != asset_bundle_sha256:
        raise ValueError("residual checkpoint object catalog identity mismatch")
    trained = checkpoint.get("training_assets")
    if (not isinstance(trained, list) or not trained or len(set(trained)) != len(trained)
            or not set(trained).issubset(object_assets) or selected_asset not in trained):
        raise ValueError("checkpoint did not train the requested asset; select a covered --asset")
    from .capacity import checkpoint_capacity
    checkpoint_capacity(checkpoint.get("capacity"))
    from .randomization import CONTRACT
    if checkpoint.get("domain_randomization") != CONTRACT:
        raise ValueError("residual checkpoint domain-randomization contract mismatch")
    source = checkpoint.get("gmt", {})
    for key in ("checkpoint_sha256", "iteration", "source_contract", "source_amp"):
        if key not in source or source[key] != gmt_metadata[key]:
            raise ValueError(f"residual checkpoint frozen GMT identity mismatch: {key}")
    # Reference-dataset fields alone can differ in the target observation contract.
    training_target = {key: value for key, value in source.get("target_contract", {}).items()
                       if key not in {"dataset", "motion_manifest_sha256"}}
    current_target = {key: value for key, value in gmt_metadata["target_contract"].items()
                      if key not in {"dataset", "motion_manifest_sha256"}}
    if not source.get("target_contract") or training_target != current_target:
        raise ValueError("residual checkpoint target robot/physics/observation contract mismatch")
    if not isinstance(checkpoint.get("iteration"), int) or checkpoint["iteration"] < 1:
        raise ValueError("residual checkpoint has no completed PPO update")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_residual(checkpoint_path, device, **identity):
    """Load the residual policy and RMS using the exact training identities."""
    import torch
    from .data import sha256_file
    from .features import ORACLE_DIM, SCHEMA
    from .policy import ResidualActorCritic

    policy = ResidualActorCritic(ORACLE_DIM, schema=SCHEMA).to(device)
    digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if sha256_file(checkpoint_path) != digest:
        raise RuntimeError("residual checkpoint changed during read")
    validate_metadata(checkpoint, architecture=policy.architecture(), **identity)
    weights = checkpoint.get("residual")
    expected = policy.state_dict()
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("residual tensor keys mismatch")
    for name, value in weights.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"nonfinite/invalid residual tensor: {name}")
    policy.load_state_dict(weights, strict=True)
    return policy.eval().requires_grad_(False), checkpoint, digest


def summarize(cases):
    from .rewards import MAX_POSITIVE_REWARD
    if not cases:
        raise ValueError("evaluation produced no cases")
    steps = sum(c["control_steps"] for c in cases)
    metrics = ("object_position_error_m", "object_orientation_error_rad", "object_surface_error_m",
               "root_position_error_m", "root_orientation_error_rad", "tracked_body_position_error_m")
    result = {"clips": len(cases), "control_steps": steps,
              "failure_clips": sum(c["failed"] for c in cases),
              "robot_failure_clips": sum(c["robot_failed"] for c in cases),
              "object_surface_failure_clips": sum(c["object_surface_failed"] for c in cases),
              "mean_normalized_return": sum(c["normalized_return"] for c in cases) / len(cases),
              "frame_weighted_normalized_return": sum(c["return"] for c in cases) / (steps * MAX_POSITIVE_REWARD)}
    for key in metrics:
        result[key] = {"frame_weighted_mean": sum(c[key]["mean"] * c["control_steps"] for c in cases) / steps,
                       "max": max(c[key]["max"] for c in cases)}
    return result


def rollout_clip(env, motion, mdp, privileged, gmt, policy, clip, teacher_kin, shape_points):
    """Exactly T actions and T+1 states; never reset after a failure flag."""
    import torch
    from .features import oracle_features, ObjectHistory, object_observation, torso_pose
    from .rewards import MAX_POSITIVE_REWARD
    from supermimic.mdp import clamp_action
    from supermimic.rotations import so3_log

    with torch.no_grad():
        ids = torch.zeros(1, dtype=torch.long, device=env.device)
        start = motion.clip_start[clip:clip + 1].clone()
        env.reset(ids, start)
        env.refresh()
        S = env.state()
        qprev, qdprev = S[4].clone(), S[5].clone()
        observation = mdp.obs(S, env.a_prev, env.xi, env.f)
        history = observation[:, None].expand(-1, 10, -1).clone()
        object_history = ObjectHistory(object_observation(torso_pose(teacher_kin,S),env.object_state()))
        steps = int(motion.clip_len[clip]) - 1
        if steps < 1:
            raise ValueError("evaluation clip has no complete control intervals")
        values = {name: [] for name in ("object_position_error_m", "object_orientation_error_rad",
                                       "object_surface_error_m", "root_position_error_m",
                                       "root_orientation_error_rad", "tracked_body_position_error_m")}
        total_return, terminated_steps, robot_steps, object_steps, clipped_steps = 0., 0, 0, 0, 0
        first_failure_step = None
        for tick in range(steps):
            S = env.state()
            direct = privileged(S, env.f, env.a_prev, env.xi, qprev, qdprev, env.actuator_state())
            base = gmt(history, motion.goal(env.f, 10), direct)
            oracle = oracle_features(direct, torso_pose(teacher_kin,S), env.object_state(), motion, env.f,
                                     env.object_context(), object_history.values)
            residual = policy.act(oracle, base, shape_points, deterministic=True)
            combined = base + residual
            action = clamp_action(combined)
            clipped_steps += int(not torch.equal(action, combined))
            out = env.step(action)
            qprev, qdprev = out["previous_q"], out["previous_qd"]
            actual, obj = env.state(), env.object_state()
            reference, obj_ref = motion.ref_state(env.f), motion.object_ref(env.f)
            robot_failed = bool(mdp.termination(actual, env.f).item())
            object_failed = bool((out["terms"]["object_surface_error_m"] > .3).item())
            failed = bool(out["terminated"].item())
            terminated_steps += int(failed)
            robot_steps += int(robot_failed)
            object_steps += int(object_failed)
            if failed and first_failure_step is None:
                first_failure_step = tick + 1
            total_return += float(out["reward"].item())
            errors = {
                "object_position_error_m": (obj[0] - obj_ref[0]).norm(dim=-1),
                "object_orientation_error_rad": so3_log(obj[1] @ obj_ref[1].transpose(-1, -2)).norm(dim=-1),
                "object_surface_error_m": out["terms"]["object_surface_error_m"],
                "root_position_error_m": (actual[0] - reference[0]).norm(dim=-1),
                "root_orientation_error_rad": so3_log(actual[1] @ reference[1].transpose(-1, -2)).norm(dim=-1),
                "tracked_body_position_error_m": (mdp.kin.forward(*actual)[0] - motion.reference_bodies(env.f)[0]).norm(dim=-1).mean(-1),
            }
            for key, value in errors.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"nonfinite evaluation metric: {key}")
                values[key].append(float(value.item()))
            next_observation = mdp.obs(actual, env.a_prev, env.xi, env.f)
            history = torch.cat((history[:, 1:], next_observation[:, None]), 1)
            object_history.advance(object_observation(torso_pose(teacher_kin,actual),obj))
        if int(env.f.item()) != int(motion.clip_end[clip]):
            raise RuntimeError("full-duration evaluation did not finish at the exact clip end")
        result = {"clip": clip, "motion": motion.entries[clip]["motion"],
                  "motion_sha256": motion.entries[clip]["motion_sha256"],
                  "source_group": motion.entries[clip]["source_group"],
                  "source_partition": motion.entries[clip]["partition"],
                  "object_asset": motion.entries[clip]["object_asset"],
                  "control_steps": steps, "state_frames": steps + 1, "duration_s": steps * .02,
                  "return": total_return, "normalized_return": total_return / (steps * MAX_POSITIVE_REWARD),
                  "failed": terminated_steps > 0, "robot_failed": robot_steps > 0,
                  "object_surface_failed": object_steps > 0,
                  "termination_steps": terminated_steps, "robot_termination_steps": robot_steps,
                  "object_surface_termination_steps": object_steps, "first_failure_step": first_failure_step,
                  "action_clipped_steps": clipped_steps, "resets_within_clip": 0,
                  "metric_alignment": "post-action actual and reference at t+1; initial state excluded from means"}
        for key, observations in values.items():
            result[key] = {"mean": sum(observations) / len(observations), "max": max(observations),
                           "final": observations[-1]}
        return result


def run_asset(args, launch_simulation):
    from .runtime import configure_warp_forward_only
    warp_runtime = configure_warp_forward_only()
    import torch
    from .assets import load_assets
    from .data import UltraMotionLib, sha256_file
    from .env import HOISim
    from .features import observation_contract
    from .shape import load_shape_bank
    from .policy import FrozenGMT
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import PrivilegedFeatures, RobotKinematics38, checkpoint_contract
    from supermimic.mdp import MDP
    from supermimic.robot import default_joint_pos_list
    import supermimic.networks as networks

    device = args.device or "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    kin = G1Kinematics(device)
    motion = UltraMotionLib(args.manifest, kin, device, partition=args.partition,
                            for_training=False, asset=args.asset, one_per_asset=args.smoke_eval)
    teacher_kin = RobotKinematics38(device=device)
    privileged = PrivilegedFeatures(teacher_kin, motion)
    gmt = FrozenGMT(args.gmt_checkpoint, checkpoint_contract(motion, teacher_kin),
                    device=device, expected_iteration=args.gmt_iteration)
    networks.AMP_DTYPE = torch.bfloat16 if gmt.metadata["source_amp"] == "bf16" else None
    shape_bank, shape_metadata = load_shape_bank(args.shape_bank,args.asset_bundle)
    shape_points = shape_bank[args.asset].to(device)
    policy, checkpoint, checkpoint_sha = frozen_residual(
        args.checkpoint, device, gmt_metadata=gmt.metadata, manifest_sha256=motion.manifest_sha256,
        object_assets=motion.object_assets, asset_bundle_sha256=sha256_file(args.asset_bundle),
        selected_asset=motion.object_assets[args.asset], shape_metadata=shape_metadata)
    before = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    assets = {entry["object_asset"]: entry for entry in load_assets(args.asset_bundle)}
    asset = dict(assets[motion.object_assets[args.asset]], asset_id=args.asset)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    cfg = make_sim(device, args)
    from .capacity import configure_capacity
    configure_capacity(cfg, checkpoint["capacity"])
    cases = []
    with launch_simulation(cfg, args):
        env = HOISim(cfg, 1, motion, mdp, asset, output, eval_mode=True)
        curriculum_before = {name: getattr(env, name).clone() for name in ("bin_G", "bin_w_avg")}
        for clip in range(len(motion.names)):
            result = rollout_clip(env, motion, mdp, privileged, gmt, policy, clip, teacher_kin, shape_points)
            cases.append(result)
            _write_json(output / f"case_{clip:05d}.json", result)
            print(json.dumps({"asset": args.asset, "case": clip + 1, "cases": len(motion.names),
                              "failed": result["failed"], "normalized_return": result["normalized_return"]}), flush=True)
        gmt.assert_unchanged()
        after = policy.state_dict()
        if set(after) != set(before) or any(not torch.equal(after[key].detach().cpu(), before[key]) for key in before):
            raise RuntimeError("evaluation changed residual parameters or normalizer state")
        if any(not torch.equal(getattr(env, key), value) for key, value in curriculum_before.items()):
            raise RuntimeError("evaluation changed the reset-sampling curriculum")
        if sha256_file(args.checkpoint) != checkpoint_sha:
            raise RuntimeError("residual checkpoint file changed during evaluation")
        report = {"passed": True, "scope": "train-only evaluation implementation smoke" if args.smoke_eval else "protected evaluation",
                  "partition": args.partition, "asset": asset["object_asset"], "checkpoint_sha256": checkpoint_sha,
                  "iteration": checkpoint["iteration"], "gmt_checkpoint_sha256": gmt.metadata["checkpoint_sha256"],
                  "checkpoint_training_smoke": checkpoint.get("smoke", False),
                  "training_reward_profile": checkpoint.get("experiment", {}).get("reward_profile", "resmimic"),
                  "evaluation_reward_profile": "resmimic",
                  "manifest_sha256": motion.manifest_sha256, "manifest_content_sha256": motion.manifest["content_sha256"],
                  "asset_bundle_sha256": sha256_file(args.asset_bundle), "data_selection": motion.selection_contract,
                  "residual_contract": checkpoint["residual_contract"], "warp_runtime": warp_runtime,
                  "observation_contract":observation_contract(),"shape":shape_metadata,
                  "normalizer_updates": 0, "optimizer_updates": 0, "world_model_constructed": False,
                  "all_model_parameters_and_buffers_unchanged": True, "deterministic": True,
                  "reset_curriculum_unchanged": True,
                  "domain_randomization": False, "full_duration_after_failure": True,
                  "summary": summarize(cases), "cases": cases,
                  "object_pose_metric_scope": "pose/surface tracking only; not a verified grasp-success metric",
                  "protected_history": motion.manifest["protection"],
                  "physics": {"dynamic_object": env.dynamic_object_evidence, "solver_options": env.physics.audit_solver_options()}}
        _write_json(output / "report.json", report)
    return report


def main():
    from .runtime import configure_warp_forward_only
    configure_warp_forward_only()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gmt-checkpoint", required=True)
    parser.add_argument("--gmt-iteration", type=int, default=4000)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-bundle", required=True)
    parser.add_argument("--shape-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset", type=int, choices=range(8))
    parser.add_argument("--partition", required=True, choices=("validation", "regression", "test", "protected_eval", "train"))
    parser.add_argument("--smoke-eval", action="store_true", help="one complete train clip per asset; never reads Protected tensors")
    parser.add_argument("--seed", type=int, default=0)
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.smoke_eval != (args.partition == "train"):
        parser.error("--smoke-eval requires --partition train; train evaluation requires --smoke-eval")
    if args.asset is not None:
        run_asset(args, launch_simulation)
        return
    import torch
    from .data import ASSETS
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if set(checkpoint.get("training_assets", [])) != set(ASSETS):
        parser.error("all-eight-asset evaluation requires a checkpoint trained on all eight assets; pass a covered --asset")
    del checkpoint
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for asset in range(8):
        output = root / f"asset_{asset:02d}"
        command = [sys.executable, "-m", "hoi_residual.evaluate", *sys.argv[1:],
                   "--asset", str(asset), "--output", str(output)]
        subprocess.run(command, check=True)
        reports.append(json.loads((output / "report.json").read_text(encoding="utf-8")))
    for key in ("checkpoint_sha256", "gmt_checkpoint_sha256", "manifest_sha256", "asset_bundle_sha256"):
        if len({report[key] for report in reports}) != 1:
            raise RuntimeError(f"evaluation identity changed between asset processes: {key}")
    cases = [case for report in reports for case in report["cases"]]
    _write_json(root / "report.json", {"passed": True, "partition": args.partition,
                                      "scope": "train-only evaluation implementation smoke" if args.smoke_eval else "protected evaluation",
                                      "assets": [report["asset"] for report in reports], "summary": summarize(cases),
                                      "checkpoint_sha256": reports[0]["checkpoint_sha256"],
                                      "training_reward_profile": reports[0]["training_reward_profile"],
                                      "evaluation_reward_profile": reports[0]["evaluation_reward_profile"],
                                      "normalizer_updates": 0, "optimizer_updates": 0,
                                      "full_duration_after_failure": True, "asset_reports": [str(root / f"asset_{i:02d}" / "report.json") for i in range(8)]})


if __name__ == "__main__":
    main()
