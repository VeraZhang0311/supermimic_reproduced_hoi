"""From-scratch tables teacher; smoke explicitly shortens the update schedule."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-bundle", required=True)
    parser.add_argument("--shape-bank", required=True)
    parser.add_argument("--reference-contacts", help="Optional complete, identity-bound exact reference contact cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capacity-only", action="store_true", help="Production-size zero-update memory/gradient validation")
    parser.add_argument("--asset", type=int, choices=(0, 1))
    parser.add_argument("--num-envs", type=int, default=4096, help="Global environment count across both asset workers")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--njmax", type=int, default=2048)
    parser.add_argument("--epa-horizon", type=int, default=64)
    args = parser.parse_args()
    if args.capacity_only and args.smoke:
        parser.error("Capacity validation uses the formal configuration, not smoke overrides")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU workloads through the scheduler")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world not in (1, 2) or (not args.smoke and world != 2):
        raise ValueError("Unified formal training requires exactly two equal asset workers")
    if args.asset is not None and world != 1:
        raise ValueError("Distributed workers select asset by rank")
    if world == 1 and args.asset is None:
        raise ValueError("Single-asset smoke requires --asset")
    if args.num_envs % world:
        raise ValueError("Global environment count must divide equally between assets")
    for variable in ("WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        if variable in os.environ:
            os.environ[variable] = str(Path(os.environ[variable]) / f"rank_{rank:02d}")
    from hoi_residual.runtime import configure_warp_forward_only
    warp_runtime = configure_warp_forward_only()
    import torch
    import torch.distributed as dist
    import supermimic.networks as shared_networks
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import RobotKinematics38
    from supermimic.robot import default_joint_pos_list
    from supermimic.mdp import MDP
    from supermimic.g1_physics import physics_contract
    from hoi_residual.capacity import capacity_profile, configure_capacity, capacity_readback
    from .assets import load_assets, load_shape_bank
    from .data import TableMotionLib
    from .config import TrainConfig
    from .networks import Policy, Critic, WorldModel, feature_schema
    from .learner import Learner
    from .replay import ReplayBuffer
    from .reward import TablesReward
    from .env import TableSim
    from .distributed import (broadcast_models, bind_distributed_statistics, synchronize_gradients,
                              synchronize_contact_counts, state_hash)
    device = "cuda:0" if args.shared_gpu or world == 1 else f"cuda:{rank}"
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo")
        if args.shared_gpu and torch.cuda.device_count() != 1:
            raise ValueError("Shared-GPU smoke requires exactly one scheduler-visible GPU")
    output = Path(args.output) / f"rank_{rank:02d}"
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    shared_networks.AMP_DTYPE = torch.bfloat16
    config = TrainConfig(num_envs=args.num_envs, iterations=args.iterations, seed=args.seed)
    if args.smoke:
        config = replace(config, experiment_kind="smoke", num_envs=args.num_envs,
                         iterations=3, buffer_steps=256, warmup_steps=64, wm_batches=1,
                         wm_batch=8*world, pi_batch=2*world, wm_only_iters=1,
                         critic_epochs=1, critic_minibatches=2, ckpt_every=3)
    global_config = config.validate()
    config = replace(config, num_envs=args.num_envs//world,
                     wm_batch=config.wm_batch//world, pi_batch=config.pi_batch//world).validate()
    kin, teacher_kin = G1Kinematics(device), RobotKinematics38(device=device)
    asset_id = rank if world == 2 else args.asset
    motion = TableMotionLib(args.manifest, kin, device, partition="train", for_training=True,
                            asset=asset_id, one_per_asset=args.smoke)
    assets = load_assets(args.asset_bundle)
    asset = assets[asset_id]
    points, shape_metadata = load_shape_bank(args.shape_bank, args.asset_bundle)
    class Shapes:
        def __init__(self): self.bank = points.to(device)
        def points(self, ids): return self.bank[ids]
    shapes = Shapes()
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    catalog = json.loads(Path(args.asset_bundle).read_text())
    hands = catalog["hands"]["geometry_cache"]
    reward = TablesReward(mdp, assets, hands, reference_contacts=args.reference_contacts)
    policy, wm, critic = (Policy(config.actor_hidden).to(device), WorldModel(config.wm_hidden, config.contact_hidden).to(device),
                          Critic(config.critic_hidden).to(device))
    if world > 1:
        broadcast_models(policy, wm, critic)
        bind_distributed_statistics(policy, wm, critic)
    torch.manual_seed(args.seed + rank)
    replay = ReplayBuffer(config.buffer_steps, config.num_envs, device, seed=args.seed+rank)
    learner = Learner(policy, wm, critic, motion, teacher_kin, reward, config, shapes=shapes, replay=replay,
                      gradient_sync=synchronize_gradients if world > 1 else None,
                      stats_sync=synchronize_contact_counts if world > 1 else None,
                      distributed_normalizers=world > 1)
    solver_capacity = capacity_profile(args.nconmax, args.njmax, args.epa_horizon)
    sim_cfg = make_sim(device, args)
    configure_capacity(sim_cfg, solver_capacity)
    env = TableSim(sim_cfg, config.num_envs, motion, mdp, asset, output, teacher_kin=teacher_kin,
                   reward=reward, max_episode_steps=32 if args.smoke else 500)
    print(json.dumps({"rank":rank,"phase":"native_scene_ready","asset":asset["object_asset"]}),flush=True)
    identity = {"schema": feature_schema(), "physics": physics_contract(), "warp_runtime": warp_runtime,
                "global_config": global_config.to_dict(), "worker_config": config.to_dict(),
                "asset": asset["object_asset"], "asset_id": asset_id, "world_size": world,
                "from_scratch": True, "base_checkpoint": None, "reward": reward.contract,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "asset_catalog_sha256": hashlib.sha256(Path(args.asset_bundle).read_bytes()).hexdigest(),
                "shape_bank": shape_metadata, "smoke": args.smoke,
                "schedule_scope": "smoke1WM+2joint" if args.smoke else "500WM_then_fixed_actor_critic"}
    _json(output/"config.json", identity)
    resets = 0
    collected_steps = 0
    collection_stats = {"reward_sum": 0., "transitions": 0, "terminated": 0, "done": 0,
                        "episode_steps_sum": 0., "episode_return_sum": 0., "episodes": 0, "terms": {}}
    def collect(steps):
        nonlocal resets, collected_steps
        for _ in range(steps):
            before = env.snapshot()
            action = learner.action(before)
            result = env.step(action)
            after = env.snapshot()  # preserve terminal state before reset
            transition = {**before, "a": action, "s_next": after["s"], "o_next": after["o"],
                          "actuator_next": after["actuator"], "next_f": after["f"],
                          "done": result["done"], "r": result["reward"],
                          "contact": result["contact"], "contact_valid": result["contact_valid"]}
            learner.observe(transition)
            collected_steps += 1
            collection_stats["reward_sum"] += float(result["reward"].sum())
            collection_stats["transitions"] += config.num_envs
            collection_stats["terminated"] += int(result["terminated"].sum())
            collection_stats["done"] += int(result["done"].sum())
            for key, value in result["terms"].items():
                collection_stats["terms"][key] = collection_stats["terms"].get(key, 0.) + float(value.sum())
            if args.smoke and collected_steps % 16 == 0:
                print(json.dumps({"rank":rank,"phase":"real_collection","steps":collected_steps,
                                  "reference_contact_frames_cached":len(reward._reference_contacts)}),flush=True)
            ids = result["done"].nonzero().flatten()
            if len(ids):
                collection_stats["episode_steps_sum"] += float(env.ep_step[ids].sum())
                collection_stats["episode_return_sum"] += float((env.ep_ret[ids] / (env.ep_step[ids] * .15)).sum())
                collection_stats["episodes"] += len(ids)
                env.reset(ids)
                resets += len(ids)
    initial = {name: state_hash(module) for name, module in (("pi", policy), ("wm", wm), ("critic", critic))}
    collect(config.warmup_steps)
    if args.capacity_only:
        from .capacity import validate_capacity, CapacityValidationError
        print(json.dumps({"rank":rank,"phase":"production_capacity_begin"}),flush=True)
        try:
            report = validate_capacity(learner, replay, env, global_config)
        except CapacityValidationError as error:
            _json(output/"capacity_report.json", error.report)
            raise
        _json(output/"capacity_report.json", report)
        print("TABLES_PRODUCTION_CAPACITY_PASS", flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    history, accumulator = [], 0.
    training_initial = {name: _parameter_hash(module) for name, module in (("pi", policy), ("wm", wm), ("critic", critic))}
    started = time.monotonic()
    for iteration in range(1, config.iterations+1):
        env.iteration = iteration
        accumulator += 1/config.utd
        count, accumulator = int(accumulator), accumulator % 1
        collect(count)
        before_update = {"pi": _parameter_hash(policy), "critic": _parameter_hash(critic)} if iteration == 1 and args.smoke else None
        metrics = learner.update(iteration)
        if args.smoke and iteration == 1:
            if before_update != {"pi": _parameter_hash(policy), "critic": _parameter_hash(critic)}:
                raise RuntimeError("WM-only update changed actor/critic weights")
        transitions = max(1, collection_stats["transitions"])
        metrics.update(real_steps=collected_steps * config.num_envs, resets=resets,
                       sim_rew=collection_stats["reward_sum"] / transitions,
                       term_frac=collection_stats["terminated"] / transitions,
                       reset_frac=collection_stats["done"] / transitions,
                       wall_hours=(time.monotonic()-started)/3600,
                       iterations_per_second=iteration/max(1e-6,time.monotonic()-started))
        if collection_stats["episodes"]:
            metrics.update(ep_len=collection_stats["episode_steps_sum"]/collection_stats["episodes"],
                           ep_ret_norm=collection_stats["episode_return_sum"]/collection_stats["episodes"])
        metrics.update({"reward_"+k:v/transitions for k,v in collection_stats["terms"].items()})
        for key in collection_stats:
            collection_stats[key] = {} if key == "terms" else 0
        if iteration % 50 == 0:
            rng = replay.generator.get_state()
            try:
                validation = learner.wm_validation(replay.sample_windows(min(256, config.wm_batch), 32, split="holdout"))
                for horizon, values in validation["horizons"].items():
                    metrics.update({f"wm_val_h{horizon}_{key}":value for key,value in values.items()})
                for key in ("contact_recall", "contact_precision", "contact_brier"):
                    metrics.update({f"wm_val_{key}_{channel}":value for channel,value in zip(validation["contact_channels"],validation[key])})
            finally:
                replay.generator.set_state(rng)
        history.append(metrics)
        with (output/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False)+"\n")
        print(json.dumps(metrics), flush=True)
        if iteration % config.ckpt_every == 0 or iteration == config.iterations:
            checkpoint = {"schema": "hoi_tables_checkpoint_v1", "identity": identity, "iteration": iteration,
                          "policy": policy.state_dict(), "wm": wm.state_dict(), "critic": critic.state_dict(),
                          "optimizers": {k:v.state_dict() for k,v in learner.optimizers.items()},
                          "sampler": env.sampler.state_dict(), "contact_positive": learner.contact_positive,
                          "contact_negative": learner.contact_negative,
                          "rng": {"torch":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state()},
                          "replay_resume": "fresh replay required; simulator is not serialized"}
            tmp = output/"checkpoint.tmp"
            torch.save(checkpoint, tmp)
            tmp.replace(output/f"checkpoint_{iteration:06d}.pt")
    final_hash = state_hash(policy, wm, critic)
    hashes = [None]*world
    if world > 1:
        dist.all_gather_object(hashes, final_hash)
    else:
        hashes[0] = final_hash
    changed = {name: _parameter_hash(module) != training_initial[name]
               for name,module in (("pi",policy),("wm",wm),("critic",critic))}
    if len(set(hashes)) != 1:
        raise RuntimeError(f"Shared model/RMS synchronization failed: {hashes}")
    if args.smoke and (not all(changed.values()) or resets < 1):
        raise RuntimeError(f"Incomplete real smoke evidence: updates={changed}, hashes={hashes}, resets={resets}")
    report = {"passed":True,"smoke":args.smoke,"asset":asset["object_asset"],"world_size":world,
              "model_parameters_and_normalizers_synchronized":len(set(hashes))==1,"model_hashes":hashes,
              "models_changed":changed,"resets":resets,"iterations":len(history),
              "horizon":config.horizon,"feature_schema":feature_schema(),
              "native_object":env.object_solver_report,"native_physics":env.physics_report,
              "native_contacts":env.contacts.evidence,"native_capacity":capacity_readback(env.physics),
              "elapsed_learning_seconds":time.monotonic()-started,"no_formal_training":args.smoke}
    _json(output/"report.json",report)
    print("TABLES_SMOKE_PASS" if args.smoke else "TABLES_TRAIN_COMPLETE",flush=True)
    if world>1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
