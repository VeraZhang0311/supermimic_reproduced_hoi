"""Verify the canonical 80 motion hashes and prepare read-only evaluation inputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import numpy as np
import torch
from .ultra_reference import UltraReference

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(inputs, output, repository, ultra_urdf):
    output.mkdir(parents=True, exist_ok=False)
    (output/'motions').mkdir()
    (output/'references').mkdir()
    (output/'metadata').mkdir()
    cases = json.loads((inputs/'evaluation_cases.json').read_text())['cases']
    assert len(cases)==80 and len({c['id'] for c in cases})==80
    assert len({c['source_group'] for c in cases})==80
    tree=ast.parse((repository/'supermimic/robot.py').read_text())
    names=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in n.targets)))
    assert sha(ultra_urdf)=='1791b76b498fd60f4b29dfbf6cc82d2144df6ea739ef6d3cf32442168a3097a1'
    shutil.copyfile(ultra_urdf,output/'metadata/ultra_g1.urdf')
    prepared=[]
    assets={}
    for case in cases:
        path=Path(case['motion'])
        assert sha(path)==case['motion_sha256'], case['id']
        raw=torch.load(path,map_location='cpu',weights_only=False)
        assert raw.shape==(case['source_frames'],630) and torch.isfinite(raw).all()
        reference=UltraReference(raw)
        count=int(np.floor(reference.duration/0.02))+1
        sampled=reference.sample(torch.arange(count,dtype=torch.float32)*0.02)
        asset=case['object_asset']
        fixture=inputs/'fixtures'/f'{asset}.pt'
        if asset not in assets:
            meta=torch.load(fixture,map_location='cpu',weights_only=False)['metadata']
            assert meta['object_names']==[asset] and set(meta['dof_names'])==set(names)
            usd=list((inputs/'object_cache'/asset).rglob(f'{asset}.usd'))
            assert len(usd)==1, usd
            physical=json.loads((inputs/'objects'/f'{asset}.json').read_text())
            assert abs(physical['mass_kg']-meta['object_mass'])<1e-7
            assets[asset]={'usd':str(usd[0]),'usd_sha256':sha(usd[0]),'fixture_sha256':sha(fixture),
                           'metadata':meta,'physics':physical,'case_ids':[]}
            (output/'metadata'/f'{asset}.json').write_text(json.dumps(assets[asset],indent=2))
        meta=assets[asset]['metadata']
        order=[meta['dof_names'].index(name) for name in names]
        # Original 60 Hz positions and xyzw quaternions; original MotionLib handles 50 Hz resampling.
        csv=output/'motions'/f"{case['id']}.csv"
        np.savetxt(csv,torch.cat((raw[:,:7],raw[:,13:42][:,order]),-1).numpy(),delimiter=',',fmt='%.10g')
        np.savez_compressed(output/'references'/f"{case['id']}.npz",reference=sampled.numpy(),
                            time=np.arange(count,dtype=np.float64)*.02)
        assets[asset]['case_ids'].append(case['id'])
        prepared.append({**case,'csv_sha256':sha(csv),'resampled_frames':count,
                         'duration_s':reference.duration,'scored_control_steps':count-1})
    manifest={'protocol':'hoi80_supermimic_zero_shot_v1','cases':prepared,'assets':assets,
              'source_cases':str(inputs/'evaluation_cases.json'),'source_cases_sha256':sha(inputs/'evaluation_cases.json'),
              'robot_joint_names':names,'control_dt':.02,'reference_fps':60.,
              'reference_time':'integer 50Hz ticks from zero, floor((source_frames-1)/60/0.02); no speed change',
              'policy_observation_dim':99,'goal_dim':730,'object_observation_channels':0,
              'physical_objects':True,'optimizer_updates':0,'normalization_updates':0,
              'body_reference_velocity':'Unchanged author MotionLib finite differences after 50Hz resampling',
              'ultra_urdf_sha256':sha(ultra_urdf)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'prepared':len(prepared),'assets':len(assets),'output':str(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--ultra-urdf',type=Path,required=True)
    a=p.parse_args();prepare(a.inputs,a.output,a.repository,a.ultra_urdf)
