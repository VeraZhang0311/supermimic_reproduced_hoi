"""Publish verified setup facts and a compact workspace for this new project only."""
from pathlib import Path
import csv
import hashlib
import json
import sys

sys.path.insert(0,'/mnt/projects/ll/tianzizh/SuperMimic_HOI/tools/wandb-workspaces')
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

root=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
frames=[]
for path in sorted((root/'data/lafan1_g1').glob('*.csv')):
    count=0
    with path.open() as stream:
        for row in csv.reader(stream):
            assert len(row)==36,(path.name,len(row))
            count+=1
    frames.append({'name':path.name,'source_frames':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(frames)==40
assert sum(x['source_frames'] for x in frames)==264705
data={'clip_count':40,'source_frame_count':264705,'source_fps':30,'clips':frames}
(root/'provenance/data-audit.json').write_text(json.dumps(data,indent=2)+'\n')
project=json.loads((root/'provenance/wandb-project.json').read_text())
audit=json.loads((root/'provenance/lock-audit.json').read_text())
deviation=json.loads((root/'provenance/environment-deviation.json').read_text())
job=int((root/'provenance/gpu-job.txt').read_text().strip())
run=wandb.init(entity=project['entity'],project=project['project'],id='setup-a1508d4-20261022',
    name='GRASP-environment-and-data-verification',job_type='setup',resume='allow',mode='online',
    save_code=False,settings=wandb.Settings(disable_code=True),
    config={'stage':'setup','upstream_commit':project['upstream_commit'],
        'isaaclab_commit':'46d6cc3d7b115906fad12efd372fbf69f4e3d3ed','environment_deviation':deviation,
        'planned_training_iterations':14000,'planned_training_envs':4096,'planned_gpu':'one L40',
        'original_experiments_preserved':True,'checkpoint_uploads':False})
run.log({'setup/locked_pins_checked':audit['checked_distribution_count'],'setup/clip_count':40,
    'setup/source_frames':264705,'setup/assets_converted':1,'setup/native_gpu_validated':0})
run.summary['formal_gpu_job_id']=job
run.summary['formal_job_state_at_registration']='PENDING(Resources)'
run.summary['setup_status']='complete; waiting for native GPU gate'
run_url=run.url
run.finish()
def plot(title,x,*ys,bounds=(None,None)):
    return wr.LinePlot(title=title,x=x,y=list(ys),range_y=bounds,smoothing_type='none',aggregate=False)
workspace=ws.Workspace(entity=project['entity'],project=project['project'],name='Upstream reproduction - 核心指标',
    auto_generate_panels=False,settings=ws.WorkspaceSettings(x_axis='train/wall_h',group_by_prefix='first'),
    runset_settings=ws.RunsetSettings(filters="Config('stage') == 'formal'"),
    sections=[
        ws.Section(name='完整40动作评估 · 验收',is_open=True,panels=[
            plot('完整动作平均回报 · 目标 ≥0.82','eval/train_wall_h','eval/norm_return',bounds=(0,1)),
            plot('完整动作完成率 · 目标 100%','eval/train_wall_h','eval/completion_fraction',bounds=(0,1)),
            plot('失败动作数 · 目标 0/40','eval/train_wall_h','eval/n_failed',bounds=(0,40)),
        ]),
        ws.Section(name='实际仿真与训练速度',is_open=True,panels=[
            plot('真实仿真平均归一化奖励','train/wall_h','train/sim_rew'),
            plot('训练迭代速度 · iteration/s','train/wall_h','train/it_per_s'),
        ]),
        ws.Section(name='模型与优化诊断',is_open=False,panels=[
            plot('世界模型 loss / 8步误差','train/wall_h','train/wm_loss','train/wm_err8'),
            plot('Critic loss','train/wall_h','train/v_loss'),
            plot('想象回报与存活率 · 不等同于验收','train/wall_h','train/imag_rew','train/imag_survival'),
        ]),
    ])
receipt_path=root/'provenance/wandb-workspace.json'
if receipt_path.exists():
    old=json.loads(receipt_path.read_text())
    current=ws.Workspace.from_url(old['sdk_url'])
    assert current.project==project['project']
    current.sections=workspace.sections
    current.settings=workspace.settings
    current.runset_settings=workspace.runset_settings
    workspace=current
workspace.save()
loaded=ws.Workspace.from_url(workspace.url)
assert loaded.auto_generate_panels is False
assert sum(len(section.panels) for section in loaded.sections)==8
from urllib.parse import urlsplit,urlunsplit
base=urlsplit(run_url.rsplit('/runs/',1)[0])
url=urlunsplit((base.scheme,base.netloc,base.path,urlsplit(workspace.url).query,''))
receipt={'url':url,'sdk_url':workspace.url,'setup_run_url':run_url,'panels':8,'gpu_job_id':job,'native_gpu_validated':False,'source_frames':264705,'clip_count':40}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
