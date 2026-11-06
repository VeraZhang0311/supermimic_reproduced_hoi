"""Frozen privileged GMT + oracle residual PPO on real ULTRA object physics.

Use torchrun with a multiple of eight ranks for a unified eight-asset teacher.
A single asset requires an explicit --asset (or --smoke). This entry point
never reads Protected eval motion bytes or starts world-model training.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, default=str, allow_nan=False)


def run(args, launch_simulation):
    from .runtime import configure_warp_forward_only
    warp_runtime = configure_warp_forward_only()
    import torch
    import torch.distributed as dist
    import warp as wp
    from .assets import load_assets
    from .data import UltraMotionLib
    from .env import HOISim
    from .features import ORACLE_DIM, SCHEMA, BLOCKS, oracle_features
    from .policy import FrozenGMT, ResidualActorCritic, file_sha256
    from .ppo import PPO, PPOConfig, RolloutBatch
    from .randomization import CONTRACT as DR_CONTRACT
    from .contract import implementation_contract
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import PrivilegedFeatures, RobotKinematics38, checkpoint_contract
    from supermimic.mdp import MDP, clamp_action
    from supermimic.robot import default_joint_pos_list
    import supermimic.networks as networks

    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    device = args.device or 'cuda:0'
    if world > 1:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK',0))}"
    args.device = device
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    if args.asset is None and not args.smoke and (world < 8 or world % 8):
        raise ValueError('Unified eight-asset training requires a multiple of eight ranks; single asset needs --asset')
    if args.asset is not None and world > 1:
        raise ValueError('Distributed unified training assigns assets by rank; omit --asset')
    asset_id = args.asset if args.asset is not None else rank % 8
    root = Path(args.output)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    output = root / f'rank_{rank:02d}'
    output.mkdir(exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    kin = G1Kinematics(device)
    motion = UltraMotionLib(args.manifest, kin, device, partition='train', for_training=True,
                            asset=asset_id, one_per_asset=args.smoke)
    teacher_kin = RobotKinematics38(device=device)
    privileged = PrivilegedFeatures(teacher_kin,motion)
    contract = checkpoint_contract(motion,teacher_kin)
    gmt = FrozenGMT(args.gmt_checkpoint,contract,device=device,expected_iteration=args.gmt_iteration)
    networks.AMP_DTYPE = torch.bfloat16 if gmt.metadata['source_amp'] == 'bf16' else None
    policy = ResidualActorCritic(ORACLE_DIM,schema=SCHEMA).to(device)
    if world > 1:
        for value in policy.state_dict().values():
            dist.broadcast(value,0)
    # Identical initial networks, independent sampling streams per worker.
    torch.manual_seed(args.seed+rank)
    ppo = PPO(policy,PPOConfig(rollout_steps=args.rollout_steps,epochs=args.epochs,
                              minibatches=args.minibatches))
    asset_bundle_sha256 = file_sha256(args.asset_bundle)
    residual_contract = implementation_contract()
    assets = {a['object_asset']:a for a in load_assets(args.asset_bundle)}
    asset = dict(assets[motion.object_assets[asset_id]], asset_id=asset_id)
    mdp = MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=device))
    cfg = make_sim(device,args)
    # Allocation capacity only: the 121/134-hull ULTRA objects require more
    # contacts/constraints. All solver options, integration and robot stay GMT.
    cfg.physics.solver_cfg.nconmax = 512
    cfg.physics.solver_cfg.njmax = 2048
    write_json(output/'config.json',{'args':vars(args),'rank':rank,'world_size':world,
        'asset':asset['object_asset'],'asset_bundle_sha256':asset_bundle_sha256,
        'gmt':gmt.metadata,'network':policy.architecture(),
        'residual_contract':residual_contract,'warp_runtime':warp_runtime,
        'oracle_blocks':BLOCKS,'ppo':asdict(ppo.config),'domain_randomization':DR_CONTRACT,
        'data_selection':motion.selection_contract,
        'capacity':{'nconmax':512,'njmax':2048,'gmt_baseline':[128,500],
                    'reason':'extra dynamic ULTRA collision hulls; allocation only'},
        'world_model_constructed':False})
    with launch_simulation(cfg,args):
        env = HOISim(cfg,args.num_envs,motion,mdp,asset,output,
                     max_episode_steps=8 if args.smoke else 500)
        # Smoke forces a timeout to exercise final-state bootstrap and reset.
        env.refresh()
        state = env.state()
        qprev,qdprev = state[4].clone(),state[5].clone()
        obs = mdp.obs(state,env.a_prev,env.xi,env.f)
        history = obs[:,None].expand(-1,10,-1).clone()

        @torch.no_grad()
        def inputs():
            S = env.state()
            direct = privileged(S,env.f,env.a_prev,env.xi,qprev,qdprev,env.actuator_state())
            base = gmt(history,motion.goal(env.f,10),direct)
            oracle = oracle_features(direct,S,env.object_state(),motion,env.f,
                                     env.object_context(),env.asset_ids,env.bounds)
            return oracle,base

        oracle,base = inputs()
        init_actor = {n:p.detach().clone() for n,p in policy.actor.named_parameters()}
        init_critic = {n:p.detach().clone() for n,p in policy.critic.named_parameters()}
        resets, timeout_count, max_contacts, transitions = 0,0,0,0
        # Startup material writes are staged until the first pre-actuator
        # callback. Audit the actual solver only after that physics boundary.
        robot_physics_before = None
        initial_context = env.object_context().detach().cpu().tolist()
        # Check a subset reset leaves other worlds' controller and object DR intact.
        others = env.object_context()[1:].clone()
        queue = env.physics.motor.queue[:,1:].clone()
        env.reset(torch.tensor([0],device=device),env.f[:1].clone())
        if not torch.equal(others,env.object_context()[1:]) or not torch.equal(queue,env.physics.motor.queue[:,1:]):
            raise AssertionError('Subset reset changed other physical worlds')
        state = env.state()
        qprev,qdprev = state[4].clone(),state[5].clone()
        obs = mdp.obs(state,env.a_prev,env.xi,env.f)
        history = obs[:,None].expand(-1,10,-1).clone()
        oracle,base = inputs()
        norm_data = (oracle,base)
        started = time.monotonic()
        with (output/'metrics.jsonl').open('x',encoding='utf-8') as log:
            for iteration in range(1,args.iterations+1):
                policy.update_normalizer(*norm_data)
                rows = {k:[] for k in ('oracle','base_actions','actions','log_probs','values','rewards',
                                       'terminated','truncated','next_values')}
                for tick in range(args.rollout_steps):
                    sample = policy.sample(oracle,base)
                    combined = clamp_action(base+sample['actions'])
                    out = env.step(combined)
                    if robot_physics_before is None:
                        robot_physics_before = env.physics.readback()
                    qprev,qdprev = out['previous_q'],out['previous_qd']
                    new_obs = mdp.obs(env.state(),env.a_prev,env.xi,env.f)
                    history = torch.cat((history[:,1:],new_obs[:,None]),1)
                    next_oracle,next_base = inputs()
                    next_value = policy.value(next_oracle,next_base)
                    values = (oracle,base,sample['actions'],sample['log_prob'],sample['value'],out['reward'],
                              out['terminated'],out['truncated'],next_value)
                    for key,value in zip(rows,values):
                        rows[key].append(value.detach().clone())
                    timeout_count += int(out['truncated'].sum())
                    contacts = wp.to_torch(env.physics.solver.mjw_data.nacon)
                    max_contacts = max(max_contacts,int(contacts.max()))
                    ids = out['done'].nonzero().flatten()
                    final_transition = iteration == args.iterations and tick == args.rollout_steps-1
                    if len(ids) and not final_transition:
                        resets += len(ids)
                        env.reset(ids)
                        S = env.state()
                        qprev[ids],qdprev[ids] = S[4][ids],S[5][ids]
                        reset_obs = mdp.obs(S,env.a_prev,env.xi,env.f)
                        history[ids] = reset_obs[ids,None].expand(-1,10,-1)
                        oracle,base = inputs()
                    else:
                        oracle,base = next_oracle,next_base
                    transitions += args.num_envs
                batch = RolloutBatch(**{k:torch.stack(v) for k,v in rows.items()},
                                     normalizer_revision=policy.normalizer_revision)
                metrics = ppo.update(batch)
                norm_data = (batch.oracle,batch.base_actions)
                metrics.update(iteration=iteration,mean_reward=float(batch.rewards.mean()),
                               elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(metrics,allow_nan=False)+'\n')
                log.flush()
                print(json.dumps({'rank':rank,**metrics}),flush=True)
                if iteration % args.save_every == 0 or iteration == args.iterations:
                    gmt.assert_unchanged()
                    if rank == 0:
                        ck={'schema':'hoi_residual_ppo_teacher_v1','iteration':iteration,
                            'residual':policy.state_dict(),'optimizer':ppo.optimizer.state_dict(),
                            'architecture':policy.architecture(),'ppo_config':asdict(ppo.config),
                            'gmt':gmt.metadata,'manifest_sha256':motion.manifest_sha256,
                            'asset_bundle_sha256':asset_bundle_sha256,'domain_randomization':DR_CONTRACT,
                            'residual_contract':residual_contract,'warp_runtime':warp_runtime,
                            'capacity':{'nconmax':512,'njmax':2048},
                            'assets':motion.object_assets,'world_size':world,
                            'smoke':args.smoke,'data_scope':{'partition':'train','one_clip_per_asset':args.smoke},
                            'training_assets':([motion.object_assets[i] for i in sorted({r % 8 for r in range(world)})]
                                               if world>1 else [asset['object_asset']]),
                            'rng_state':torch.get_rng_state(), 'resume_semantics':'fresh simulator; not bitwise continuation'}
                        torch.save(ck,root/f'checkpoint_{iteration:06d}.pt')
        changed_actor = any(not torch.equal(p,init_actor[n]) for n,p in policy.actor.named_parameters())
        changed_critic = any(not torch.equal(p,init_critic[n]) for n,p in policy.critic.named_parameters())
        if not changed_actor or not changed_critic or max_contacts <= 0:
            raise AssertionError('No actor/critic learning update or no physical contact observed')
        if args.smoke and timeout_count == 0:
            raise AssertionError('Smoke did not exercise timeout bootstrapping')
        gmt.assert_unchanged()
        report = {'passed':True,'scope':'real IsaacLab/Newton/MuJoCo-Warp residual PPO smoke' if args.smoke else 'training',
            'asset':asset['object_asset'],'selected_train_clips':len(motion.names),
            'iterations':args.iterations,'transitions':transitions,'resets':resets,'timeouts':timeout_count,
            'actor_changed':changed_actor,'critic_changed':changed_critic,'gmt_weights_and_rms_unchanged':True,
            'world_model_constructed':False,'max_physical_contacts':max_contacts,
            'partial_reset_isolation':True,'dynamic_object':env.dynamic_object_evidence,
            'object_solver_checks':env.object_solver_checks,'object_solver_report':env.object_solver_report,
            'object_dr_native_readbacks':env.randomizer.readback_count,
            'initial_object_context':initial_context,'final_object_context':env.object_context().cpu().tolist(),
            'robot_physics_before':robot_physics_before,'robot_physics_after':env.physics.readback(),
            'solver_options':env.physics.audit_solver_options(),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-started,'performance_claim':'smoke does not establish task success'}
        write_json(output/'report.json',report)
    if world>1:
        dist.destroy_process_group()


def main():
    from .runtime import configure_warp_forward_only
    configure_warp_forward_only()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--asset-bundle',required=True)
    parser.add_argument('--gmt-checkpoint',required=True)
    parser.add_argument('--gmt-iteration',type=int,default=26000)
    parser.add_argument('--output',required=True)
    parser.add_argument('--asset',type=int,choices=range(8))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--num-envs',type=int,default=512)
    parser.add_argument('--iterations',type=int,default=100000)
    parser.add_argument('--rollout-steps',type=int,default=24)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--minibatches',type=int,default=4)
    parser.add_argument('--save-every',type=int,default=100)
    parser.add_argument('--seed',type=int,default=0)
    from isaaclab.app import add_launcher_args,launch_simulation
    add_launcher_args(parser)
    args=parser.parse_args()
    if min(args.num_envs,args.iterations,args.rollout_steps,args.epochs,args.minibatches,args.save_every)<1:
        parser.error('training sizes must be positive')
    if args.smoke and (args.num_envs<2 or args.rollout_steps<8 or args.iterations<2):
        parser.error('smoke requires >=2 envs, >=8 steps, >=2 updates')
    run(args,launch_simulation)


if __name__=='__main__':
    main()
