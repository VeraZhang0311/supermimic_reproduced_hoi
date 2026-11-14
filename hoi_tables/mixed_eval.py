"""Read-only protected tables20 evaluator and mixed-panel aggregation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from .experiments import TABLE_ASSETS
from .mixed_monitor import PROTOCOL, validate_request, _coverage
from .telemetry import atomic_json
from .trajectories import sha256


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table_asset(request, asset_id, *, state=None, record_trajectories=True):
    from hoi_residual.runtime import configure_warp_forward_only
    configure_warp_forward_only()
    import torch
    import supermimic.networks as shared_networks
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.privileged import PrivilegedFeatures, RobotKinematics38
    from supermimic.robot import default_joint_pos_list
    from hoi_residual.capacity import capacity_profile, configure_capacity
    from .assets import load_shape_bank
    from .data import TableMotionLib
    from .env import TableSim
    from .evaluate import state_dict_hash
    from .hoi80_batch import rollout_asset
    from .mixed import evaluation_checkpoint, read_experiment
    from .native_hands import load_native_hands
    from .networks import Policy
    from .prepare_contacts import _read_assets_and_hands
    from .reward import TablesReward
    from .table_trajectories import save_trajectory

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run native protected evaluation through Slurm")
    experiment = read_experiment(request["campaign"])
    if state is None:
        state = evaluation_checkpoint(experiment, request["checkpoints"]["mixed"], request["iteration"], request["checkpoint_sha256"]["mixed"])
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    shared_networks.AMP_DTYPE = torch.bfloat16
    native = load_native_hands(state["identity"]["native_hands"])
    assets, hands = _read_assets_and_hands(experiment.raw["table_assets"])
    points, _ = load_shape_bank(experiment.raw["table_shapes"], experiment.raw["table_assets"])
    kin, teacher = G1Kinematics(device), RobotKinematics38(device=device)
    motion = TableMotionLib(experiment.root / "tables/manifest.json", kin, device, partition="protected_eval",
                            for_training=False, asset=asset_id)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    reward = TablesReward(mdp, assets, hands, reference_contacts=experiment.root / "tables/reference_contacts.pt", native_hands=native)
    policy = Policy(experiment.config.actor_hidden).to(device).eval().requires_grad_(False)
    policy.load_state_dict(state["policy_state"], strict=True)
    before = state_dict_hash(policy.state_dict())
    if before != state["model_state_sha256"]["policy"]:
        raise RuntimeError("Protected table policy/RMS changed on load")
    output = Path(request["output"]) / "tables20" / TABLE_ASSETS[asset_id]
    output.mkdir(parents=True, exist_ok=False)
    cfg = make_sim(device, None)
    configure_capacity(cfg, capacity_profile(512, 2048, 64))
    env = TableSim(cfg, len(motion.entries), motion, mdp, assets[asset_id], output, teacher_kin=teacher,
                   reward=reward, eval_mode=True, max_episode_steps=int(motion.clip_len.max()) + 1,
                   solver_ls_iterations=200, native_hands=native)
    batch = rollout_asset(env, motion, mdp, PrivilegedFeatures(teacher, motion), policy, teacher,
                          points[asset_id:asset_id + 1].to(device), assets[asset_id], None)
    clips = []
    for index, entry in enumerate(motion.entries):
        clip, frames, controls = batch.clip(index, entry)
        clip["action"] = entry["action"]
        if record_trajectories:
            clip["trajectory_sha256"] = save_trajectory(output, clip, frames, controls, batch.names)
        clips.append(clip)
    after = state_dict_hash(policy.state_dict())
    if before != after or any(sha256(path) != request["checkpoint_sha256"][key]
                              for key, path in request["checkpoints"].items()):
        raise RuntimeError("Protected table checkpoint or policy changed during evaluation")
    atomic_json(output / "report.json", {"passed": True, "clips": clips, "asset": TABLE_ASSETS[asset_id],
                                         "policy_state_sha256_before": before, "policy_state_sha256_after": after,
                                         "model_state_sha256": state["model_state_sha256"],
                                         **{key: request[key] for key in ("iteration", "parent_run_id", "checkpoint_sha256")},
                                         **dict.fromkeys(("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates"), 0)})


def aggregate(request):
    from .evaluate import summarize
    from .evaluate_hoi80 import aggregate_reports
    from .table_trajectories import verify_trajectory

    table_cases, _ = validate_request(request)
    output = Path(request["output"])
    ultra = aggregate_reports(output / "hoi80", output / "hoi80/result.json", checkpoint_keys={"mixed"})
    table = [_read(output / "tables20" / name / "report.json") for name in TABLE_ASSETS]
    identity = {key: ultra[key] for key in ("iteration", "parent_run_id", "checkpoint_sha256", "model_state_sha256")}
    clips = []
    by_action = defaultdict(list)
    per_asset = {}
    for asset, report in zip(TABLE_ASSETS, table, strict=True):
        if (report["asset"] != asset or any(report[key] != value for key, value in identity.items())
                or not report["passed"] or report["policy_state_sha256_before"] != identity["model_state_sha256"]["policy"]
                or report["policy_state_sha256_before"] != report["policy_state_sha256_after"]
                or any(report[key] != 0 for key in ("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates"))):
            raise ValueError("Mixed table worker did not use the same frozen checkpoint as HOI80")
        rows = report["clips"]
        per_asset[asset] = {"metrics": summarize(rows)}
        clips.extend(rows)
        for row in rows:
            by_action[row["action"]].append(row)
    _coverage(clips, table_cases, table=True)
    for clip in clips:
        verify_trajectory(output / "tables20", clip, verify_payload=True)
    atomic_json(output / "tables20/result.json", {**identity, "passed": True, "clips": clips,
                "metrics": summarize(clips), "per_asset": per_asset,
                "per_action": {key: {"metrics": summarize(value), "clip_count": len(value)} for key, value in by_action.items()},
                **dict.fromkeys(("optimizer_updates", "normalizer_updates", "replay_updates", "rsi_updates"), 0)})
    atomic_json(output / "result.json", {**identity, "protocol": PROTOCOL, "passed": True,
                "reports": {name: sha256(output / name / "result.json") for name in ("hoi80", "tables20")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--table-asset", type=int, choices=range(2))
    action.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    request = _read(args.request)
    validate_request(request)
    if args.aggregate:
        aggregate(request)
    else:
        table_asset(request, args.table_asset)


if __name__ == "__main__":
    main()
