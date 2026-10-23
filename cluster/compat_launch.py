"""Run unchanged upstream entry points with the supported old-driver graph fallback."""
from pathlib import Path
import datetime
import functools
import json
import os
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]

def install_graph_compatibility(mujoco_warp, supported):
    """Only select MuJoCo-Warp's graph scheduling path; solver settings stay intact."""
    original=mujoco_warp.put_model
    @functools.wraps(original)
    def put_model(*args,**kwargs):
        model=original(*args,**kwargs)
        previous=bool(model.opt.graph_conditional)
        if not supported:
            model.opt.graph_conditional=False
        receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'pid':os.getpid(),
            'conditional_graph_supported':bool(supported),
            'original_graph_conditional':previous,'effective_graph_conditional':bool(model.opt.graph_conditional),
            'solver_iterations':int(model.opt.iterations),'line_search_iterations':int(model.opt.ls_iterations),
            'reason':'MuJoCo-Warp supported nonconditional scheduling fallback for driver CUDA<12.4' if not supported else 'Driver supports original graph configuration'}
        path=ROOT/'provenance'/f"graph-compat-{os.environ.get('SLURM_JOB_ID','local')}-{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt,indent=2)+'\n')
        print('[GRASP compatibility] '+json.dumps(receipt),flush=True)
        return model
    mujoco_warp.put_model=put_model

def main():
    import warp as wp
    import mujoco_warp
    wp.init()
    install_graph_compatibility(mujoco_warp,wp.is_conditional_graph_supported())
    script=Path(sys.argv[1])
    if not script.is_absolute(): script=ROOT/script
    assert script.resolve().is_relative_to(ROOT)
    sys.argv=[str(script),*sys.argv[2:]]
    sys.path.insert(0,str(ROOT))
    runpy.run_path(str(script),run_name='__main__')

if __name__=='__main__': main()
