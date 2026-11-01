"""Supplement named task screenshots using a verified, frozen native runtime.

This separate driver leaves the accepted source and all original evidence intact.
It selects reference reset frames, runs the same 8-env/64-step zero-action smoke,
and renders only saved post-integration state. It never trains a policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASKS = ('soccer/case2_kick_left', 'soccer/case3_kick_right')


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def select_frames(bank, entries, foot_columns):
    import torch
    by_name = {c['name']: c for c in entries}
    result = []
    for task in TASKS:
        candidates = []
        for i, name in enumerate(bank.names):
            c = by_name[name]
            if c['dataset'] != 'omnicontact' or c['task'] != task:
                continue
            start, end = int(bank.clip_start[i]), int(bank.clip_end[i])
            positive = ((bank.expected_contact[start:end+1, foot_columns] > .5) &
                        (bank.expected_valid[start:end+1, foot_columns] > .5)).any(-1)
            offsets = torch.nonzero(positive, as_tuple=False).flatten().tolist()
            offsets = [f for f in offsets if f >= 1]
            if offsets:
                arrival = start + offsets[len(offsets)//2]
                candidates.append({'task': task, 'clip_name': name, 'clip_index': i,
                                   'start_frame': arrival-1, 'arrival_frame': arrival,
                                   'arrival_local_frame': arrival-start,
                                   'selection': 'middle annotated positive-foot frame, start one control interval earlier'})
        if not candidates:
            raise ValueError('No outgoing positive-foot reference frame for '+task)
        candidates.sort(key=lambda x: x['clip_name'])
        # Four worlds per named task, deterministic clip identities. Repeats are
        # explicit when fewer than four clips exist; they never imply new clips.
        base = len(result)
        result.extend(dict(candidates[i % len(candidates)], environment=base+i)
                      for i in range(4))
    return result


def main():
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from supermimic.physics import add_physics_arguments, resolve_arguments
    from supermimic.kinematics import G1Kinematics
    from supermimic.robot import default_joint_pos_list
    from hoi.v1.native_batch import read_plan
    from hoi.v1.native import HOITrackingSim, make_hoi_sim
    from hoi.v1.prepare import load_bundle
    from hoi.v1.mdp import HOIMDP
    from hoi.v1.contacts import CHANNEL_INDEX
    from hoi.v1.privileged import configure_teacher
    from hoi.v1.native_model_debug import export_actual_model, enable_smoke_visual_capture
    from hoi.v1.native_input_contract import validate_transition_inputs
    from hoi.v1.native_acceptance import capture_native_runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--partition', choices=('train','protected_eval'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    add_physics_arguments(parser); add_launcher_args(parser)
    args = parser.parse_args()
    settings, _ = resolve_arguments(args)
    if not (settings.hand_collision and settings.self_collision):
        raise ValueError('Both original collision switches must stay enabled')
    plan = read_plan(args.plan)
    if (plan['execution']['num_envs'], plan['execution']['steps']) != (8,64):
        raise ValueError('Unexpected parent protocol')
    asset_id = 209
    parent = next(t for t in plan['tasks'] if t['partition']==args.partition and t['asset_id']==asset_id)
    from hoi.v1.native_batch import _completed
    accepted = _completed(plan,parent)
    if accepted is None:
        raise ValueError('Supplement requires the accepted parent asset task')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['SUPERMIMIC_ASSET_DIR'] = str(args.output/'robot_assets')
    torch.manual_seed(plan['execution']['seed'])
    import warp as wp
    wp.init()
    kin = G1Kinematics('cuda:0')
    bundle_path = plan['bundles'][args.partition]['path']
    bank, shapes, bundle = load_bundle(bundle_path, kin, device='cuda:0', require_train=False, asset_filter=[asset_id])
    asset = next(a for a in bundle['assets'] if int(a['asset_id'])==asset_id)
    selected = select_frames(bank,bundle['clips'],[CHANNEL_INDEX['left_foot_object'],CHANNEL_INDEX['right_foot_object']])
    if [r['environment'] for r in selected] != list(range(8)):
        raise ValueError('Selection must identify all eight worlds exactly')
    mdp = HOIMDP(kin,bank,torch.tensor(default_joint_pos_list(),device='cuda:0'),shapes)
    cfg = make_hoi_sim('cuda:0',args)
    cfg.physics.solver_cfg.save_to_mjcf = str(args.output/'compiled_model.generated.xml')
    with launch_simulation(cfg,args), torch.no_grad():
        env = HOITrackingSim(cfg,8,bank,mdp,asset,args.output,eval_mode=False,reset_noise=False,robot_asset=bundle['native_robot_asset'])
        export_actual_model(env.contacts.solver.mj_model,args.output,args.output/'compiled_model.generated.xml',recorder=env.contacts,
            scene_snapshot={**env.support_state(),'environment_origins':env.scene.env_origins,'reference_frames':env.f,'declared_support_ids':list(env.supports)})
        geometry,meshes = env.export_compiled_geometry(args.output/'compiled_geometry')
        env.verify_compiled_geometry(meshes,compiled_hands=geometry['hands'])
        pairs=env.calibrate_pair_probes(); supports=env.calibrate_support_probes()
        if not pairs['passed'] or not supports['passed']:
            raise ValueError('Original native calibration did not pass')
        configure_teacher(mdp,bundle,'cuda:0')
        frames=torch.tensor([r['start_frame'] for r in selected],device='cuda:0')
        env.reset(torch.arange(8,device='cuda:0'),frames)
        if not torch.equal(env.f,frames):raise ValueError('Target reset frame mismatch')
        visual=enable_smoke_visual_capture(env,args.output/'smoke_visual',args.output/'compiled_model.generated.xml')
        first=None
        for step in range(64):
            tr=env.step(torch.zeros(8,29,device='cuda:0'))
            if step==0:
                if not torch.equal(tr['f'],frames):raise ValueError('First integrated transition is not the requested task frame')
                first=validate_transition_inputs(tr,mdp,args.output/'first_transition_inputs.pt')
        if not visual['captured'] or not first['passed']:raise ValueError('Missing real dynamic/input evidence')
        runtime=capture_native_runtime(env.contacts,compilation=env.warp_compilation)
        physics=env.physics_report
        env.close()
    rendered=[]
    # Each requested task is shown once per partition; all eight worlds remain
    # in the bound state dump and first-transition input evidence.
    for index in (0,4):
        output=args.output/('render_env_'+str(index))
        subprocess.run([sys.executable,'-m','hoi.v1.native_snapshot_render','--manifest',
            str(args.output/'smoke_visual/compiled_model_manifest.json'),'--environment',str(index),'--output',str(output)],
            cwd=plan['execution']['source'],env=dict(os.environ,MUJOCO_GL='egl'),check=True)
        report=json.loads((output/'rendering.json').read_bytes())
        if report['saved_frame']!=selected[index]['arrival_frame'] or report['environment']!=index:
            raise ValueError('Rendered reference identity mismatch')
        if any(report[k]!=0 for k in ('physics_steps','fk_calls','policy_calls')):
            raise ValueError('Renderer executed dynamics or inference')
        rendered.append({'selection':selected[index],'preview':identity(output/'preview.png'),'rendering':identity(output/'rendering.json')})
    result={'schema':'hoi_named_task_dynamic_visual_supplement_v1','passed':True,'partition':args.partition,
        'plan':identity(args.plan),'parent_success':identity(accepted[0]/'SUCCESS.json'),'driver':identity(__file__),
        'source':plan['execution']['source'],'source_files':plan['execution']['source_files'],'bundle':plan['bundles'][args.partition],
        'seed':plan['execution']['seed'],'num_envs':8,'control_steps':64,'selected':selected,'rendered':rendered,
        'first_transition':identity(args.output/'first_transition_inputs.pt'),'snapshot':identity(args.output/'smoke_visual/compiled_model_manifest.json'),
        'pair_probes':pairs,'support_probes':supports,'physics':physics,'native_runtime':runtime,
        'training_started':False,'visual_review_completed':False,
        'scope':'named reference-task states after actual integration; no trained-policy task-success claim; original379 evidence unchanged'}
    (args.output/'SUCCESS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'partition':args.partition,'output':str(args.output),'tasks':list(TASKS),'training_started':False}),flush=True)


if __name__=='__main__':main()
