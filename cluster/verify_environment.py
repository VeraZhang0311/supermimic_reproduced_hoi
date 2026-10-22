"""Check every active exported lock pin, excluding only the documented CUDA substitution."""
from pathlib import Path
import importlib.metadata as metadata
import json
from packaging.requirements import Requirement

root=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
checks=[]
for line in (root/'provenance/grasp-cu128-requirements.txt').read_text().splitlines():
    if not line or line.startswith('#'): continue
    requirement=Requirement(line)
    if requirement.marker is not None and not requirement.marker.evaluate(): continue
    dist=metadata.distribution(requirement.name)
    if requirement.specifier:
        assert dist.version in requirement.specifier,(requirement.name,dist.version,str(requirement.specifier))
    if requirement.url and 'git+' in requirement.url:
        direct=json.loads(dist.read_text('direct_url.json'))
        expected=requirement.url.rsplit('@',1)[1]
        assert direct['vcs_info']['commit_id']==expected,(requirement.name,direct,expected)
    checks.append({'name':requirement.name,'version':dist.version})
assert metadata.version('torch')=='2.11.0+cu128'
assert metadata.version('warp-lang')=='1.17.0'
assert metadata.version('mujoco-warp')=='3.15.0'
assert metadata.version('mujoco')=='3.15.0'
assert not any(d.metadata['Name'].lower()=='isaacsim' for d in metadata.distributions())
report={'status':'all_active_exported_pins_verified','checked_distribution_count':len(checks),'distributions':checks,'gpu_validation':False}
(root/'provenance/lock-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='distributions'}))
