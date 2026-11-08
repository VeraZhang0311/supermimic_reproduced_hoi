"""CPU distance queries at every saved 200Hz native pose; no FK or stepping."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    meta=json.loads((root/'geometry.json').read_text());report=json.loads((root/'report.json').read_text())
    assert digest(root/'substeps.npz')==report['substeps_sha256'] and digest(root/'rollout.npz')==report['rollout_sha256']
    fine=dict(np.load(root/'substeps.npz',allow_pickle=False));coarse=dict(np.load(root/'rollout.npz',allow_pickle=False))
    names=list(map(str,fine['names']))
    for field in ('actual_pos','actual_quat','actual_object_pos','actual_object_quat'):
        np.testing.assert_array_equal(fine[field][::4],coarse[field])
    m=mujoco.MjModel.from_binary_path(str(root/'native_model.mjb'));d=mujoco.MjData(m)
    rot=lambda q:Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]]).as_matrix()
    geoms=meta['geoms'];objects=[g for g in geoms if g['role']==2]
    feet={role:[g for g in geoms if g['role']==role] for role in (3,4)}
    for g in geoms:g['rotation']=rot(g['geom_quat'])
    depths=np.zeros((len(fine['time']),2),np.float64);pairs=np.full((len(depths),2,2),-1,np.int32)
    for frame in range(len(depths)):
        for g in geoms:
            if g['role']==2:p,R=fine['actual_object_pos'][frame],rot(fine['actual_object_quat'][frame])
            else:
                idx=names.index('left_ankle_roll_link' if g['role']==3 else 'right_ankle_roll_link')
                p,R=fine['actual_pos'][frame,idx],rot(fine['actual_quat'][frame,idx])
            d.geom_xpos[g['id']]=p+R@np.asarray(g['geom_pos'])
            d.geom_xmat[g['id']]=(R@g['rotation']).reshape(9)
        for col,role in enumerate((3,4)):
            for foot in feet[role]:
                a=foot['id']
                for obj in objects:
                    b=obj['id']
                    if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]+1e-6:continue
                    distance=mujoco.mj_geomDistance(m,d,a,b,1.,None)
                    if -distance>depths[frame,col]:depths[frame,col]=-distance;pairs[frame,col]=[a,b]
    np.savez_compressed(root/'penetration.npz',time=fine['time'],depth_m=depths,pair=pairs,
                        control_time=coarse['time'],control_depth_m=depths[::4])
    summaries={}
    for label,values,dt in (('physics_200hz',depths,.005),('video_50hz',depths[::4],.02)):
        x=values.max(1);idx=int(x.argmax())
        summaries[label]={'state_frames':len(x),'max_depth_mm':float(x.max()*1000),
            'mean_depth_mm':float(x.mean()*1000),'max_time_s':idx*dt,
            'left_max_mm':float(values[:,0].max()*1000),'right_max_mm':float(values[:,1].max()*1000),
            'integrated_depth_mm_s':float(((x[1:]+x[:-1])*.5).sum()*1000*dt),
            **{f'frames_over_{mm}mm':int((x>mm/1000).sum()) for mm in (0,1,5,10)}}
    report['penetration']=summaries
    (root/'analysis.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    reports={v:analyze(args.root/v) for v in ('A','B')}
    equal=('static_model_sha256','final_model_sha256','initial_gpu_geom_fields_sha256','initial_snapshot_sha256','policy_state_sha256')
    for key in equal:assert reports['A'][key]==reports['B'][key],key
    for r in reports.values():assert r['optimizer_updates']==r['normalizer_updates']==r['wandb_writes']==0
    result={'passed':True,'checkpoint_iteration':6000,'clip_id':'omomo/unitree_g1/sub3_smalltable_027',
        'same_initial_state_policy_geometry_motor_hand_and_geom_materials':True,
        'only_intervention':'foot-table contact solref time constant .02 -> .01; damping ratio, solimp, friction and all other pairs unchanged',
        'policy_inference':'frozen deterministic closed-loop policy in each rollout, not action replay',
        'geometry_measurement':'postintegration native 38-link/object poses, CPU mj_geomDistance at200Hz; video uses every fourth state',
        'variants':reports,'full_duration':True,'physics_steps_per_variant':1020,'state_frames_per_variant':256,
        'scope':'single full-clip A/B; no claim of bitwise simulator determinism or general training improvement'}
    a=reports['A']['penetration']['physics_200hz']['max_depth_mm'];b=reports['B']['penetration']['physics_200hz']['max_depth_mm']
    result['maximum_depth_reduction_percent']=None if a==0 else 100*(a-b)/a
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for v,color in (('A','#2667c9'),('B','#dc5d35')):
        values=np.load(args.root/v/'penetration.npz');x=values['depth_m'].max(1)*1000
        ax.plot(values['time'],x,label=f"{v}: {reports[v]['foot_table_time_constant_s']*1000:.0f}ms; max {x.max():.2f}mm",color=color)
    ax.axhline(10,color='gray',ls='--',lw=.8);ax.set(xlabel='Time (s)',ylabel='Maximum foot/table penetration (mm)',
        title='Tables checkpoint6000 / sub3_smalltable_027 / both feet, every5ms',xlim=(0,5.1),ylim=(0,None))
    ax.legend();ax.grid(alpha=.2);fig.savefig(args.root/'penetration_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'passed':True,'A':reports['A']['penetration'],'B':reports['B']['penetration'],
                     'max_reduction_percent':result['maximum_depth_reduction_percent']}),flush=True)


if __name__=='__main__':main()
