"""Prove that the only upstream scientific-source edit is the self-collision flag."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM='a1508d42c9d1699a52e8233ab16d08c488441ab9'
ISAACLAB='46d6cc3d7b115906fad12efd372fbf69f4e3d3ed'
FLAG_OFF=b'enabled_self_collisions=False'
FLAG_ON=b'enabled_self_collisions=True'

def validate():
    def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args])
    names=git('ls-tree','-r','--name-only',UPSTREAM,'supermimic','tools').decode().splitlines()
    expected_edits=[]
    for name in names:
        original=git('show',UPSTREAM+':'+name)
        actual=(ROOT/name).read_bytes()
        if name=='supermimic/robot.py':
            assert original.count(FLAG_OFF)==1
            expected=original.replace(FLAG_OFF,FLAG_ON)
        else: expected=original
        assert actual==expected, 'Unexpected scientific-source change: '+name
        if actual!=original: expected_edits.append(name)
    assert expected_edits==['supermimic/robot.py'],expected_edits
    tracked_now=set(git('ls-files','supermimic','tools').decode().splitlines())
    assert tracked_now==set(names), 'Unexpected added or removed upstream tools/core source'
    assert not git('status','--porcelain','--untracked-files=all','--','supermimic','tools').strip()
    assert subprocess.check_output(['git','-C',str(ROOT/'IsaacLab'),'rev-parse','HEAD'],text=True).strip()==ISAACLAB
    # Compare training defaults with the exact baseline integration manifest, using AST only.
    base=ROOT.with_name('supermimic_reproduced_hoi')
    tree=ast.parse((base/'cluster/run_pipeline.py').read_text())
    dictionaries=[n for n in ast.walk(tree) if isinstance(n,ast.Dict)]
    baseline_params=next(ast.literal_eval(n) for n in dictionaries if any(isinstance(k,ast.Constant) and k.value=='utd' for k in n.keys))
    parser_tree=ast.parse((ROOT/'supermimic/train.py').read_text())
    defaults={}
    for node in ast.walk(parser_tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant):
            default=next((kw.value for kw in node.keywords if kw.arg=='default'),None)
            if default is not None:
                try: defaults[node.args[0].value.lstrip('-')]=ast.literal_eval(default)
                except (ValueError,TypeError): pass
    for key,value in baseline_params.items():
        if key=='iterations': assert value==14000
        else: assert defaults[key]==value,(key,defaults.get(key),value)
    identity={'upstream_commit':UPSTREAM,'isaaclab_commit':ISAACLAB,
        'experiment_commit':git('rev-parse','HEAD').decode().strip(),
        'scientific_diff_files':expected_edits,'physical_self_collisions':True,
        'reward_source_sha256':hashlib.sha256((ROOT/'supermimic/mdp.py').read_bytes()).hexdigest(),
        'training_parameters':baseline_params,'training_parameters_match_baseline':True}
    (ROOT/'provenance/source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return identity

if __name__=='__main__': print(json.dumps(validate(),indent=2))
