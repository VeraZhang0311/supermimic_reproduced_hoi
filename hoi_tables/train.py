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


def actor_backward_schedule(profile, workers_per_gpu, actor_ranks_per_turn=1):
    """Execution order only; batches, objective and optimizer steps are fixed."""
    if profile != "ultra" or workers_per_gpu == 1:
        return "parallel"
    return "serialized_per_gpu" if actor_ranks_per_turn == 1 else "grouped_per_gpu"


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--experiment-config")
    parser.add_argument("--profile", choices=("tables","ultra"))
    parser.add_argument("--manifest")
    parser.add_argument("--asset-bundle")
    parser.add_argument("--geometry-bundle")
    parser.add_argument("--native-hands", help="Verified original Ultra hand asset receipt")
    parser.add_argument("--shape-bank")
    parser.add_argument("--canonical")
    parser.add_argument("--helper-root")
    parser.add_argument("--workers-per-gpu", type=int)
    parser.add_argument("--actor-ranks-per-turn", type=int, help="Concurrent full H32 actor graphs per Ultra GPU")
    parser.add_argument("--reference-contacts", help="Optional complete, identity-bound exact reference contact cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--capacity-only", action="store_true", help="Production-size zero-update memory/gradient validation")
    parser.add_argument("--asset", type=int)
    parser.add_argument("--num-envs", type=int, help="Global environment count across all asset workers")
    parser.add_argument("--iterations", type=int)
    parser.add_argument('--checkpoint-every',type=int,help='Checkpoint IO cadence; Ultra defaults to100, tables to500')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--njmax", type=int, default=2048)
    parser.add_argument("--epa-horizon", type=int, default=64)
    parser.add_argument('--ls-iterations',type=int,help='Explicit line-search budget; all native flags remain fatal')
    parser.add_argument('--resume-bundle',help='Complete identity-bound parent checkpoint bundle')
    parser.add_argument('--recovery-validation-steps',type=int,default=0,help='Disposable restored-policy real-control stress before full capacity probe')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    from .experiments import resolve_arguments
    experiment, experiment_binding = resolve_arguments(args)
    if args.capacity_only and args.smoke:
        parser.error("Capacity validation uses the formal configuration, not smoke overrides")
    if args.recovery_validation_steps and (not args.capacity_only or not args.resume_bundle or args.recovery_validation_steps<4096):
        parser.error('Recovery stress requires restored full capacity mode and at least4096 real steps')
    if args.resume_bundle and args.smoke:parser.error('Continuation cannot change the formal learning configuration')
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU workloads through the scheduler")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    topology = experiment.topology(rank=rank, world=world, local_rank=int(os.environ.get("LOCAL_RANK",rank)),
                                   workers_per_gpu=args.workers_per_gpu, shared_gpu=args.shared_gpu,
                                   smoke=args.smoke, asset=args.asset)
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
    from .config import TrainConfig
    from .networks import Policy, Critic, WorldModel, feature_schema
    from .learner import Learner
    from .replay import ReplayBuffer
    from .env import TableSim
    from .distributed import (broadcast_models, bind_distributed_statistics, synchronize_gradients,
                              synchronize_contact_counts, state_hash, serialized_gpu_backward)
    experiment.topology(rank=rank, world=world, local_rank=int(os.environ.get("LOCAL_RANK",rank)),
                        workers_per_gpu=args.workers_per_gpu, shared_gpu=args.shared_gpu,
                        smoke=args.smoke, asset=args.asset, visible_devices=torch.cuda.device_count())
    device = f"cuda:{topology['device_index']}"
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
    config = TrainConfig(num_envs=args.num_envs, iterations=args.iterations, seed=args.seed,ckpt_every=args.checkpoint_every)
    if args.smoke:
        config = replace(config, experiment_kind="smoke", num_envs=args.num_envs,
                         iterations=3, buffer_steps=256, warmup_steps=64, wm_batches=1,
                         wm_batch=8*world, pi_batch=2*world, wm_only_iters=1,
                         critic_epochs=1, critic_minibatches=2, ckpt_every=3)
    global_config = config.validate()
    if any(getattr(config,name)%world for name in ("num_envs","wm_batch","pi_batch")):
        raise ValueError("Global environment and learning batches must divide equally across asset workers")
    config = replace(config, num_envs=args.num_envs//world,
                     wm_batch=config.wm_batch//world, pi_batch=config.pi_batch//world).validate()
    kin, teacher_kin = G1Kinematics(device), RobotKinematics38(device=device)
    asset_id = topology["asset_id"]
    motion = experiment.motion(args, kin, device, asset_id)
    assets, hands = experiment.load_assets(args)
    if tuple(assets[i]["object_asset"] for i in range(len(assets))) != experiment.assets:
        raise ValueError("Dataset profile and exact asset catalog differ")
    asset = assets[asset_id]
    points, shape_metadata = experiment.load_shape(args)
    class Shapes:
        def __init__(self): self.bank = points.to(device)
        def points(self, ids): return self.bank[ids]
    shapes = Shapes()
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    native_hands = None
    if args.native_hands:
        from .native_hands import load_native_hands
        native_hands = load_native_hands(args.native_hands)
    reward = experiment.reward(args, mdp, assets, hands, native_hands=native_hands)
    policy, wm, critic = (Policy(config.actor_hidden).to(device), WorldModel(config.wm_hidden, config.contact_hidden).to(device),
                          Critic(config.critic_hidden).to(device))
    if world > 1:
        broadcast_models(policy, wm, critic)
        bind_distributed_statistics(policy, wm, critic)
    torch.manual_seed(args.seed + rank)
    replay = ReplayBuffer(config.buffer_steps, config.num_envs, device, seed=args.seed+rank)
    actor_schedule = actor_backward_schedule(experiment.name, topology["workers_per_gpu"], args.actor_ranks_per_turn)
    actor_context = None
    if actor_schedule != "parallel":
        actor_context = lambda: serialized_gpu_backward(
            topology["workers_per_gpu"], ranks_per_turn=args.actor_ranks_per_turn)
    learner = Learner(policy, wm, critic, motion, teacher_kin, reward, config, shapes=shapes, replay=replay,
                      gradient_sync=synchronize_gradients if world > 1 else None,
                      stats_sync=synchronize_contact_counts if world > 1 else None,
                      distributed_normalizers=world > 1, actor_backward_context_factory=actor_context)
    solver_capacity = capacity_profile(args.nconmax, args.njmax, args.epa_horizon)
    sim_cfg = make_sim(device, args)
    configure_capacity(sim_cfg, solver_capacity)
    env = TableSim(sim_cfg, config.num_envs, motion, mdp, asset, output, teacher_kin=teacher_kin,
                   reward=reward, max_episode_steps=32 if args.smoke else 500, solver_ls_iterations=args.ls_iterations,
                   native_hands=native_hands)
    print(json.dumps({"rank":rank,"phase":"native_scene_ready","asset":asset["object_asset"]}),flush=True)
    if native_hands is not None:
        _json(output / "native_hand_contacts.json", env.contacts.evidence)
    identity = {"schema": feature_schema(), "physics": physics_contract(), "warp_runtime": warp_runtime,
                "global_config": global_config.to_dict(), "worker_config": config.to_dict(),
                "asset": asset["object_asset"], "asset_id": asset_id, "world_size": world,
                "from_scratch": True, "base_checkpoint": None, "reward": reward.contract,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "asset_catalog_sha256": hashlib.sha256(Path(args.asset_bundle).read_bytes()).hexdigest(),
                "shape_bank": shape_metadata, "smoke": args.smoke,
                "schedule_scope": "smoke1WM+2joint" if args.smoke else "500WM_then_fixed_actor_critic"}
    if experiment.name != "tables":
        identity["experiment"] = {"profile":experiment.name,"assets":list(experiment.assets),
                                  "workers_per_gpu":topology["workers_per_gpu"],
                                  "actor_backward_schedule":actor_schedule,
                                  "actor_ranks_per_turn":args.actor_ranks_per_turn,
                                  "geometry_bundle_sha256":hashlib.sha256(Path(args.geometry_bundle).read_bytes()).hexdigest(),
                                  "config_binding":experiment_binding,
                                  "evaluation":{"kind":experiment.evaluation,"every":2000,"owning_training_run":True}}
    if args.ls_iterations!=50 or args.resume_bundle:
        from .solver_health import policy as solver_policy
        identity['solver_health']=solver_policy(args.ls_iterations)
    if native_hands is not None:
        identity["physics"] = env.physics.contract
        identity["native_hands"] = native_hands.identity
    start_iteration=0
    resume_receipt=None
    if args.resume_bundle:
        from .resume import load_rank,restore,lineage
        checkpoint,parent=load_rank(args.resume_bundle,rank,identity)
        resume_receipt=restore(checkpoint,parent,learner,env)
        start_iteration=checkpoint['iteration']
        identity.update(from_scratch=False,base_checkpoint=lineage(parent))
        env.reset(torch.arange(env.N,device=env.device))
        del checkpoint
        _json(output/'resume_receipt.json',resume_receipt)
    _json(output/"config.json", identity)
    resets = 0
    collected_steps = 0
    collection_stats = {"reward_sum": 0., "transitions": 0, "terminated": 0, "done": 0,
                        "episode_steps_sum": 0., "episode_return_sum": 0., "episodes": 0, "terms": {}}
    def collect(steps, *, update_statistics=True):
        nonlocal resets, collected_steps
        for _ in range(steps):
            before = env.snapshot()
            action = learner.action(before,update_statistics=update_statistics)
            result = env.step(action)
            after = env.snapshot()  # preserve terminal state before reset
            transition = {**before, "a": action, "s_next": after["s"], "o_next": after["o"],
                          "actuator_next": after["actuator"], "next_f": after["f"],
                          "done": result["done"], "r": result["reward"],
                          "contact": result["contact"], "contact_valid": result["contact_valid"]}
            learner.observe(transition,update_statistics=update_statistics)
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
    collect(config.warmup_steps,update_statistics=not bool(args.resume_bundle))
    if resume_receipt:
        from .evaluate import state_dict_hash
        after_prefill={name:state_dict_hash(module.state_dict()) for name,module in (('policy',policy),('wm',wm),('critic',critic))}
        if after_prefill!=resume_receipt['models_and_rms_sha256']:raise RuntimeError('Fresh replay prefill changed restored model/RMS')
        _json(output/'resume_prefill.json',{'passed':True,'real_steps':config.warmup_steps,'optimizer_updates':0,
              'models_and_RMS_unchanged':True,'next_iteration':start_iteration+1,'repeat_wm_warmup':False})
    if args.capacity_only:
        from .capacity import validate_capacity, CapacityValidationError
        if args.recovery_validation_steps:
            from .recovery_checks import validate_native_guard
            _json(output/'guard_validation.json',validate_native_guard(device))
            from .evaluate import state_dict_hash
            stress_before={name:state_dict_hash(module.state_dict()) for name,module in (('policy',policy),('wm',wm),('critic',critic))}
            for tick in range(args.recovery_validation_steps):
                env.iteration=start_iteration+int(tick*config.utd)
                collect(1,update_statistics=False)
                if (tick+1)%256==0:print(json.dumps({'rank':rank,'phase':'recovery_native_stress','real_steps':tick+1}),flush=True)
            stress_after={name:state_dict_hash(module.state_dict()) for name,module in (('policy',policy),('wm',wm),('critic',critic))}
            if stress_before!=stress_after:raise RuntimeError('Recovery stress changed model/RMS')
            from hoi_residual.capacity import overflow_detail
            native=overflow_detail(env.physics)
            if native['overflow_world_count']:raise RuntimeError('Recovery stress has native flags')
            _json(output/'recovery_stress_report.json',{'passed':True,'real_control_steps':args.recovery_validation_steps,
                'physics_substeps':4*args.recovery_validation_steps,'environment_transitions':config.num_envs*args.recovery_validation_steps,
                'optimizer_updates':0,'models_and_RMS_unchanged':True,'native':native,'solver_health':env.solver_health})
        print(json.dumps({"rank":rank,"phase":"production_capacity_begin"}),flush=True)
        try:
            report = validate_capacity(learner, replay, env, global_config,
                                       expected_assets=experiment.assets, workers_per_gpu=topology["workers_per_gpu"])
        except CapacityValidationError as error:
            _json(output/"capacity_report.json", error.report)
            raise
        _json(output/"capacity_report.json", report)
        print("TABLES_PRODUCTION_CAPACITY_PASS", flush=True)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    history, accumulator = [], (start_iteration/config.utd)%1
    training_initial = {name: _parameter_hash(module) for name, module in (("pi", policy), ("wm", wm), ("critic", critic))}
    started = time.monotonic()
    for iteration in range(start_iteration+1, config.iterations+1):
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
                       iterations_per_second=(iteration-start_iteration)/max(1e-6,time.monotonic()-started))
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
