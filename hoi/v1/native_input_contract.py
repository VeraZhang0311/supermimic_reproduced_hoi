"""Read-only network input validation from one already collected native interval."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from supermimic.mdp import normalize_xi
from .contacts import CHANNELS
from .networks import Policy, WorldModel, Critic
from .privileged import TEACHER_PROFILE, PRIVILEGED_DIM
from .scene_context import scene_from_record, normalize_scene_context
from .state import unflatten_state, unflatten_object, object_features

SCHEMA = "hoi_native_transition_input_contract_v1"


def _check(value, shape, name):
    if tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"native input {name} requires finite {tuple(shape)}, got {tuple(value.shape)}")


@torch.no_grad()
def validate_transition_inputs(tr, mdp, output, *, require_cuda=True):
    """No optimizer/RMS updates or simulator calls; random validation outputs are discarded.

    WM consumes the actual pre-step state/action. Actor/critic consume the actual
    arrival state, with real pre-step teacher history and explicit repeated
    startup histories. The reference uses arrival-index contact semantics.
    """
    device=tr["s"].device; n=len(tr["f"])
    if require_cuda and device.type != "cuda":
        raise ValueError("native transition input acceptance requires actual CUDA tensors")
    S,O=unflatten_state(tr["s"]),unflatten_object(tr["o"])
    S1,O1=unflatten_state(tr["s1"]),unflatten_object(tr["o1"])
    motion=mdp.motion; f=tr["f"]
    f1=torch.minimum(f+1,motion.clip_end[motion.clip_id[f]])
    if not torch.equal(motion.asset_ids[f],tr["asset_id"]):
        raise ValueError("native transition asset and reference differ")
    if not hasattr(mdp,"privileged_features"):
        raise ValueError("native input validation requires the bound teacher FK adapter")
    for key,width in (("s",71),("s1",71),("o",13),("o1",13),("a",29),("a_prev",29),
                      ("xi",33),("xi_o",12),("contact",16),("contact_valid",16)):
        _check(tr[key],(n,width),key)
        if tr[key].device != device:raise ValueError("native transition tensors must share the actual state device")
    expected,valid=motion.expected(f1)
    for name,value in (("expected",expected),("valid",valid),("contact",tr["contact"]),("contact_valid",tr["contact_valid"])):
        _check(value,(n,16),name)
        if not bool(((value==0)|(value==1)).all()):raise ValueError(name+" must be binary")
    if bool((expected[~valid.bool()] != 0).any()):
        raise ValueError("unknown reference contact must remain masked zero")
    scene0=scene_from_record(S,tr);scene1=scene_from_record(S1,tr)
    obs=mdp.obs(S1,tr["a"],tr["xi"],f1);obj=mdp.object_obs(S1,O1,f1)
    goal=motion.goal(f1,10);object_goal=motion.object_goal(S1,O1,f1,10)
    points=mdp.shapes.points(tr["asset_id"])
    teacher=mdp.privileged_features(S1,O1,f1,tr["a"],tr["xi"],tr["xi_o"],S)
    legacy=torch.cat((mdp.legacy_critic_features(S1,f1,normalize_xi(tr["xi"])),goal),-1)
    critic_args=(legacy,object_goal,object_features(S1,O1),mdp.object_errors(S1,O1,f1),tr["xi_o"],torch.cat((expected,valid),-1),scene1,points)
    values={"robot_obs":(obs,99),"object_obs":(obj,18),"human_goal":(goal,730),"object_goal":(object_goal,470),
            "teacher":(teacher,PRIVILEGED_DIM),"scene_pre":(scene0,26),"scene_arrival":(scene1,26),"critic_legacy":(legacy,914)}
    for name,(value,width) in values.items():_check(value,(n,width),name)
    _check(points,(n,128,3),"object_points")
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=="cuda" else []
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=[torch.cuda.get_rng_state(d).clone() for d in cuda_devices]
    with torch.random.fork_rng(devices=cuda_devices):
        # Defaults are production architectures. These weights are never used
        # for actions, learning, acceptance scores or any simulator state.
        pi=Policy(actor_observation_profile=TEACHER_PROFILE).to(device).eval()
        wm=WorldModel().to(device).eval();critic=Critic().to(device).eval()
        modules=(pi,wm,critic)
        before=[{k:v.clone() for k,v in module.named_buffers()} for module in modules]
        for module in modules:module.requires_grad_(False)
        wm_input=wm.inputs(S,O,tr["a"],normalize_xi(tr["xi"]),tr["xi_o"],points,scene0)
        _,_,logits=wm.step(S,O,tr["a"],normalize_xi(tr["xi"]),tr["xi_o"],points,scene0)
        action=pi(obs[:,None].expand(-1,10,-1),goal,obj[:,None].expand(-1,10,-1),object_goal,points,scene_context=scene1,privileged=teacher)
        value=critic(*critic_args)
        _check(wm_input,(n,250),"wm_input");_check(logits,(n,16),"contact_logits")
        _check(action,(n,29),"teacher_action");_check(value,(n,),"critic_value")
        for module,old in zip(modules,before):
            if any(not torch.equal(old[k],v) for k,v in module.named_buffers()):
                raise RuntimeError("input validation modified a model/RMS buffer")
            normalized=normalize_scene_context(module.scene_norm,scene1).reshape(n,2,13)
            if bool((normalized[~tr["support_present"].bool()]!=0).any()):
                raise ValueError("absent support must remain all-zero after normalization")
    rng_preserved=torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(old,torch.cuda.get_rng_state(d)) for d,old in zip(cuda_devices,cuda_rng))
    if not rng_preserved:raise RuntimeError("input validation changed collector RNG")
    raw_keys=("s","s1","o","o1","a","a_prev","xi","xi_o","f","asset_id","contact","contact_valid","support_pose_xyzw","support_present")
    saved={key:tr[key].detach().cpu().clone() for key in raw_keys}
    saved.update(arrival_frame=f1.cpu(),expected_contact=expected.cpu(),expected_valid=valid.cpu(),scene_pre=scene0.cpu(),scene_arrival=scene1.cpu(),teacher_features=teacher.cpu())
    output=Path(output)
    with output.open("xb") as stream:torch.save(saved,stream)
    return {"schema":SCHEMA,"passed":True,"cuda_verified":device.type=="cuda","device":str(device),"num_envs":n,
            "artifact":{"path":str(output.resolve()),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()},
            "frames":f.cpu().tolist(),"arrival_frames":f1.cpu().tolist(),"asset_ids":tr["asset_id"].cpu().tolist(),
            "support_present":tr["support_present"].cpu().tolist(),"support_pose_xyzw":tr["support_pose_xyzw"].cpu().tolist(),
            "expected_contact":expected.cpu().tolist(),"expected_valid":valid.cpu().tolist(),
            "unknown_reference_slots":int((~valid.bool()).sum()),"unknown_coverage_observed":bool((~valid.bool()).any()),
            "dimensions":{**{name:width for name,(_,width) in values.items()},"wm_input":250,"contact_logits":16,"teacher_action":29,"critic_value":1},
            "channels":list(CHANNELS),"source":"first real smoke transition captured before autoreset; same interval, no extra physics",
            "history":"arrival actor histories repeat its observed current frame for startup; teacher previous state is actual pre-step state",
            "support_origin":"actual collector reset-held world pose relative to environment origin",
            "normalizer_updates":0,"optimizer_steps":0,"additional_physics_steps":0,"rng_preserved":True,
            "untrained_validation_only":True,"outputs_applied_to_simulator":False,"formal_training_started":False}
