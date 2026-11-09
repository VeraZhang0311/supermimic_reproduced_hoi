"""Disposable guard fault injection, isolated from the real simulator state."""
from __future__ import annotations


def validate_native_guard(device='cpu'):
    import torch
    from types import SimpleNamespace as NS
    from supermimic.g1_runtime import G1PhysicsRuntime
    native=torch.device(device).type=='cuda'
    if native:
        import warp as wp
        flag=wp.zeros(1,dtype=wp.int32,device=device)
        flags=wp.to_torch(flag)
        bridge=wp
    else:
        flags=torch.zeros(1,dtype=torch.int32);flag=flags
        bridge=NS(to_torch=lambda x:x)
    q=torch.zeros(1,29,device=device)
    qd=torch.zeros_like(q)
    fake=NS(contactmix=NS(ensure_installed=lambda:None),
        robot=NS(data=NS(joint_pos=NS(torch=q),joint_vel=NS(torch=qd)),
                 set_joint_effort_target_index=lambda **k:None),
        motor=NS(compute=lambda *a:torch.zeros_like(q)),update_friction=lambda x:None,
        scene=NS(write_data_to_sim=lambda:None,update=lambda dt:None),
        sim=NS(step=lambda **k:None),wp=bridge,solver=NS(mjw_data=NS(overflow=flag)))
    masks=[1<<i for i in range(12)]+[1<<30,(1<<0)|(1<<10)]
    for mask in masks:
        flags.fill_(mask)
        try:G1PhysicsRuntime.step_target(fake,q)
        except RuntimeError as error:
            if str(error)!='native contact/constraint buffer overflow':raise
        else:raise RuntimeError('Native guard ignored injected flag: '+str(mask))
        if int(flags.item())!=mask:raise RuntimeError('Native guard cleared an injected flag')
    # This isolated test array may be reset; no live training solver is touched.
    flags.zero_();q.fill_(float('nan'))
    try:G1PhysicsRuntime.step_target(fake,q)
    except FloatingPointError:pass
    else:raise RuntimeError('Native guard ignored nonfinite state')
    q.zero_();G1PhysicsRuntime.step_target(fake,q)
    return {'passed':True,'native_cuda_array':native,'fatal_masks':masks,
            'all_native_flags_remain_fatal':True,'nonfinite_remains_fatal':True,
            'flags_unchanged_after_failure':True,'real_simulator_modified':False}
