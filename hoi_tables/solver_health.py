"""Explicit numerical budget and fatal native diagnostics; never mask flags."""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 'hoi_native_solver_health_v1'
LIMIT_FLAGS = frozenset(('ITERATIONS', 'LS_ITERATIONS'))
CAPACITY_FLAGS = frozenset(('NEFC','NJMAX_NNZ','BROADPHASE','NARROWPHASE','CCD',
                            'HFIELD','CONTACT_MATCH','NVMAX','EPA_HORIZON','TACTILE'))


def policy(ls_iterations=50):
    if type(ls_iterations) is not int or not 50 <= ls_iterations <= 1000:
        raise ValueError('Native line-search budget must be an integer in [50,1000]')
    return {'schema':SCHEMA,'ls_iterations':ls_iterations,'fatal_on_all_native_flags':True,
            'clear_or_mask_flags':False,'physical_contact_or_motor_changes':False}


def identity_policy(identity):
    value = identity.get('solver_health', policy())
    if value != policy(value.get('ls_iterations')):
        raise ValueError('Invalid solver-health checkpoint contract')
    return value


def configure(env, ls_iterations=50):
    selected = policy(ls_iterations)
    observations = {}
    for label, model in (('template',env.physics.solver.mj_model),('warp',env.physics.solver.mjw_model)):
        opt = model.opt
        names=('ls_iterations','iterations')+ (('warn_overflow',) if hasattr(opt,'warn_overflow') else ())
        before = {name:int(getattr(opt,name)) for name in names}
        if before['ls_iterations'] > ls_iterations:
            raise ValueError('Refuse to lower an existing native line-search budget')
        opt.ls_iterations = ls_iterations
        assert all(int(getattr(opt,name))==before[name] for name in names if name!='ls_iterations')
        observations[label] = {'before':before,'after':{name:int(getattr(opt,name)) for name in before}}
    return {**selected,'readback':observations}


def classify(detail):
    names = set(detail.get('overflow_bit_counts',{}))
    reasons = {name for row in detail.get('overflow_worlds',[]) for name in row.get('reasons',[])}
    names |= reasons
    capacity, limits = names & CAPACITY_FLAGS, names & LIMIT_FLAGS
    unknown = names - CAPACITY_FLAGS - LIMIT_FLAGS
    known_mask = (1 << 12)-1
    unknown |= {'UNKNOWN_BITS'} if any(int(row['mask']) & ~known_mask for row in detail.get('overflow_worlds',[])) else set()
    if unknown:return 'unknown_native_failure'
    if capacity and limits:return 'capacity_and_solver_limit'
    if capacity:return 'capacity_overflow'
    if limits:return 'solver_iteration_limit'
    return 'unclassified_native_failure'


class NativeSolverFailure(RuntimeError):
    def __init__(self, report, path=None):
        self.report, self.path = report, path
        reasons=','.join(report['native']['overflow_bit_counts']) or 'unknown'
        super().__init__(f"Fatal native {report['category']}: {reasons}; diagnostic={path}")


def raise_failure(env, cause):
    """Called only after the unchanged GMT hard guard raises; no retry/reset."""
    from hoi_residual.capacity import overflow_detail
    detail = overflow_detail(env.physics)
    frames = []
    for row in detail.get('overflow_worlds',[]):
        world = row['world']; frame = int(env.f[world].item())
        clip = int(env.motion.clip_id[frame].item())
        frames.append({'world':world,'reference_frame_before_failed_control':frame,
                       'clip_id':env.motion.entries[clip]['clip_id']})
    report = {'schema':SCHEMA,'fatal':True,'category':classify(detail),'native':detail,
        'asset':env.asset['object_asset'],'iteration':int(env.iteration),'frames':frames,
        'completed_substeps_in_control':int(getattr(env.contacts,'substeps',0)),
        'solver_policy':getattr(env,'solver_health',policy()),'original_error':str(cause),
        'native_flags_cleared':False,'diagnostic_performed_retry_or_reset':False}
    directory = Path(env._solver_failure_output)
    path = directory/f'native_failure_iter_{int(env.iteration):06d}_{os.getpid()}.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'phase':'fatal_native_solver','category':report['category'],
                      'asset':report['asset'],'iteration':report['iteration'],'report':str(path)}),flush=True)
    raise NativeSolverFailure(report,str(path)) from cause
