"""Keep the author's lock; replace only the CUDA stack for GRASP's old driver."""
from pathlib import Path
import importlib.metadata as metadata
import json

r=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
old=Path('/mnt/projects/ll/tianzizh/SuperMimic_HOI/envs/isaac-portable-py312/lib/python3.12/site-packages')
def cuda_package(name):
    name=name.lower().replace('_','-')
    return name in {'torch','torchvision','triton'} or name.startswith('nvidia-')
source=(r/'provenance/author-lock-requirements.txt').read_text().splitlines()
kept=[]
removed=[]
for line in source:
    name=line.split('==')[0].strip()
    if cuda_package(name):
        removed.append(line)
    elif not line.startswith(('--index-url','--extra-index-url','-e ','source/','./source/')):
        kept.append(line)
replacement=sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions(path=[str(old)]) if cuda_package(d.metadata['Name']))
assert 'torch==2.11.0+cu128' in [x.lower() for x in replacement]
assert replacement and removed
(r/'provenance/grasp-cu128-requirements.txt').write_text('\n'.join(kept+replacement)+'\n')
(r/'provenance/environment-deviation.json').write_text(json.dumps({
    'author_isaaclab_commit':'46d6cc3d7b115906fad12efd372fbf69f4e3d3ed',
    'reason':'GRASP L40 driver535 cannot run the author locked CUDA13 Torch build; use the already native-proven CUDA12.8 build in an independent environment.',
    'removed_from_author_lock':removed,'replacement_cuda_stack':replacement,
    'installation':'native isolated venv; Docker unavailable; no Isaac Sim/Kit runtime',
    'training_algorithm_modified':False},indent=2)+'\n')
print(json.dumps({'locked_non_cuda_requirements':len(kept),'cuda_replacements':replacement}))
