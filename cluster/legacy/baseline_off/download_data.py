"""Fetch precisely the public source used by the author's download_data.sh."""
from pathlib import Path
import urllib.request
import json
import hashlib
import time

r=Path('/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi')
base='https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/'
api='https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/'
def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=120) as response:
                return response.read()
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
def fetch(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        data=get(base+source)
        temporary=destination.with_suffix(destination.suffix+'.partial')
        temporary.write_bytes(data)
        temporary.replace(destination)
    return {'source':source,'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
records=[]
motions=json.loads(get(api+'g1'))
for entry in motions:
    if entry.get('type')=='file' and entry['path'].endswith('.csv'):
        records.append(fetch(entry['path'],r/'data/lafan1_g1'/Path(entry['path']).name))
assert len(records)==40,len(records)
records.append(fetch('robot_description/g1/g1_29dof_rev_1_0.urdf',r/'data/g1_description/g1_29dof_rev_1_0.urdf'))
for entry in json.loads(get(api+'robot_description/g1/meshes')):
    if entry.get('type')=='file':
        records.append(fetch(entry['path'],r/'data/g1_description/meshes'/Path(entry['path']).name))
(r/'provenance/download-manifest.json').write_text(json.dumps({'dataset':'lvhaidong/LAFAN1_Retargeting_Dataset','motions':40,'files':records},indent=2)+'\n')
print(json.dumps({'downloaded_files':len(records),'motion_count':40}))
