"""Standalone frozen-checkpoint one-clip A/B recorder; no training/W&B writes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp
import mujoco
from hoi_tables import evaluate
from supermimic.rotations import mat_to_quat
from normal_contact import FootTableNormalProbe


def digest_arrays(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        a = np.asarray(value)
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--variant', choices=('A','B'), required=True)
    args, remaining = parser.parse_known_args()
    root = args.audit_root
    root.mkdir(parents=True, exist_ok=False)
    original = evaluate.rollout_clip

    def run(env, motion, mdp, privileged, policy, ignored_clip, teacher, shape, asset):
        clip, = [i for i,e in enumerate(motion.entries) if e['clip_id'].endswith('/sub3_smalltable_027')]
        m, d = env.physics.solver.mj_model, env.physics.solver.mjw_data
        roles = env.contacts.roles.numpy()
        time_constant = None if args.variant=='A' else .01
        probe = FootTableNormalProbe(env.physics.contactmix, roles, d.naconmax, time_constant)
        policy_before = evaluate.state_dict_hash(policy.state_dict())
        fields = ('geom_type','geom_dataid','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity',
                  'geom_solref','geom_solimp','geom_solmix','geom_friction','geom_priority','geom_condim','geom_margin','geom_gap',
                  'mesh_vert','mesh_face','body_parentid','body_pos','body_quat','body_mass','body_ipos','body_inertia','body_iquat',
                  'jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','jnt_pos','jnt_axis','jnt_range','dof_armature','dof_damping')
        static_signature = digest_arrays({name:getattr(m,name) for name in fields})
        gpu_fields = ('geom_friction','geom_solref','geom_solimp','geom_priority','geom_condim','geom_margin','geom_gap')
        gpu_initial = {name:wp.to_torch(getattr(env.physics.solver.mjw_model,name)).cpu().numpy() for name in gpu_fields}
        gpu_signature = digest_arrays(gpu_initial)
        mujoco.mj_saveModel(m, str(root/'native_model.mjb'))
        geoms = []
        for i in np.flatnonzero(np.isin(roles,[2,3,4])):
            i=int(i)
            geoms.append({'id':i,'role':int(roles[i]),'name':m.geom(i).name,
                **{name:np.asarray(getattr(m,name)[i]).tolist() for name in ('geom_type','geom_dataid','geom_pos','geom_quat','geom_size')}})
        (root/'geometry.json').write_text(json.dumps({'geoms':geoms,'static_model_sha256':static_signature},indent=2))
        names = list(env.robot.body_names)
        ref_indices = [teacher.body_names.index(n) for n in names]
        frames, fine, actions, rewards, failures, contacts = [], [], [], [], [], []
        audit = {'target_contact_samples':0,'other_contact_samples':0,'target_solref_changes':0,
                 'other_solref_changes':0,'damping_ratio_changes':0,'max_target_time_constant_error':0.}
        original_reset, original_step = env.reset, env.step
        after_substep = env.contacts._after_substep
        initial_snapshot = None

        def capture(reference):
            pose = env.robot.data.body_link_pose_w.torch.detach().clone()
            positions = pose[0,:,:3]-env.scene.env_origins[0]
            state, obj = env.state(), env.object_state()
            assert torch.allclose(positions[names.index('pelvis')],state[0][0],atol=2e-5,rtol=0)
            pack = {'actual_pos':positions,'actual_quat':pose[0,:,[6,3,4,5]],
                    'actual_object_pos':obj[0][0],'actual_object_quat':mat_to_quat(obj[1])[0]}
            if reference:
                rp,rr,_,_=teacher.forward(motion.ref_state(env.f));ro=motion.object_ref(env.f)
                pack.update(reference_pos=rp[0,ref_indices],reference_quat=mat_to_quat(rr[0,ref_indices]),
                    reference_object_pos=ro[0][0],reference_object_quat=mat_to_quat(ro[1])[0],reference_frame=env.f[0])
            return {k:v.detach().cpu().numpy().copy() for k,v in pack.items()}

        def reset(*a, **kw):
            nonlocal initial_snapshot
            assert not frames
            result=original_reset(*a,**kw)
            snap=env.snapshot()
            initial_snapshot=digest_arrays({k:v.detach().cpu().numpy() for k,v in snap.items()
                if isinstance(v,torch.Tensor) and k not in ('episode_id',)})
            frames.append(capture(True));fine.append(capture(False))
            return result

        def step(action):
            actions.append(action[0].detach().cpu().numpy().copy())
            result=original_step(action)
            frames.append(capture(True));rewards.append(float(result['reward'].item()));failures.append(bool(result['terminated'].item()))
            return result

        def substep(dt):
            after_substep(dt)
            probe.ensure_installed()
            count=int(d.nacon.numpy()[0])
            pair=d.contact.geom.numpy()[:count]
            valid=(pair>=0).all(1)
            a,b=roles[np.maximum(pair[:,0],0)],roles[np.maximum(pair[:,1],0)]
            target=valid & (((a==2)&np.isin(b,[3,4]))|((b==2)&np.isin(a,[3,4])))
            before=probe.original.numpy()[:count];after=d.contact.solref.numpy()[:count]
            changed=np.any(before!=after,axis=1)
            audit['target_contact_samples']+=int(target.sum());audit['other_contact_samples']+=int((~target).sum())
            audit['target_solref_changes']+=int((changed&target).sum());audit['other_solref_changes']+=int((changed&~target).sum())
            audit['damping_ratio_changes']+=int((before[:,1]!=after[:,1]).sum())
            if target.any():
                assert np.allclose(before[target],[.02,1.],atol=1e-7,rtol=0)
                expected=.02 if args.variant=='A' else .01
                error=float(np.abs(after[target,0]-expected).max())
                audit['max_target_time_constant_error']=max(audit['max_target_time_constant_error'],error)
                dist=d.contact.dist.numpy();pos=d.contact.pos.numpy();normal=d.contact.frame.numpy();force=env.contacts.forces.numpy()
                for j in np.flatnonzero(target):
                    contacts.append({'substep':len(fine),'time_s':len(fine)*dt,'geoms':pair[j].tolist(),
                      'distance_m':float(dist[j]),'position':pos[j].tolist(),'frame':normal[j].tolist(),
                      'wrench':force[j].tolist(),'solref':after[j].tolist()})
            fine.append(capture(False))

        env.reset,env.step=reset,step
        env.contacts._after_substep=substep
        with torch.no_grad():
            metrics=original(env,motion,mdp,privileged,policy,clip,teacher,shape,asset)
        env.reset,env.step=original_reset,original_step
        env.contacts._after_substep=after_substep
        assert metrics['control_steps']==255 and len(frames)==256 and len(fine)==1021
        assert audit['other_solref_changes']==audit['damping_ratio_changes']==0
        assert audit['max_target_time_constant_error']<1e-7
        if args.variant=='B': assert audit['target_solref_changes']==audit['target_contact_samples']>0
        else: assert audit['target_solref_changes']==0
        assert policy_before==evaluate.state_dict_hash(policy.state_dict())
        final_signature=digest_arrays({name:getattr(m,name) for name in fields})
        source = Path(__file__).resolve().parent
        saved={k:np.stack([f[k] for f in frames]) for k in frames[0]}
        saved.update(names=np.array(names),time=np.arange(256)*.02,action=np.stack(actions),reward=np.array(rewards),failed=np.array(failures))
        np.savez_compressed(root/'rollout.npz',**saved)
        fine_saved={k:np.stack([f[k] for f in fine]) for k in fine[0]}
        fine_saved.update(names=np.array(names),time=np.arange(1021)*.005)
        np.savez_compressed(root/'substeps.npz',**fine_saved)
        assert all(np.isfinite(v).all() for k,v in saved.items() if k!='names')
        with gzip.open(root/'contacts.json.gz','wt') as stream:json.dump(contacts,stream)
        result={'variant':args.variant,'checkpoint_iteration':6000,'seed':0,'metrics':metrics,'audit':audit,
          'foot_table_time_constant_s':.02 if args.variant=='A' else .01,'damping_ratio':1.,
          'only_intervention':'foot-table contact solref[0]; no geom, friction, solimp, controller or hand changes',
          'static_model_sha256':static_signature,'final_model_sha256':final_signature,'initial_gpu_geom_fields_sha256':gpu_signature,
          'initial_snapshot_sha256':initial_snapshot,'policy_state_sha256':policy_before,
          'optimizer_updates':0,'normalizer_updates':0,'wandb_writes':0,'state_frames':256,'physics_state_frames':1021,
          'nominal_properties':True,'deterministic_policy':True,'no_failure_reset':True,
          'rollout_sha256':hashlib.sha256((root/'rollout.npz').read_bytes()).hexdigest(),
          'substeps_sha256':hashlib.sha256((root/'substeps.npz').read_bytes()).hexdigest(),
          'helper_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}}
        (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({'variant':args.variant,'passed':True,'contact_audit':audit,'state_frames':256}),flush=True)
        raise SystemExit(0)

    evaluate.rollout_clip=run
    sys.argv=[sys.argv[0],*remaining]
    evaluate.main()


if __name__=='__main__':main()
