"""Frozen direct-teacher evaluation on the exact canonical HOI80 panel.

Run eight independent asset workers, then aggregate. This module never creates
a learner, optimizer, replay or W&B writer. Canonical PHC39 scores and frozen
TablesReward survival are reported as separate measures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath

from .evaluate import (file_sha256, state_dict_hash, _canonical, _common_identity,
                       _model_specifications, _validate_state, _reward_identity,
                       rollout_clip, summarize as tracking_summary, FAILURE_DEFINITION,
                       SUCCESS_DEFINITION)
from .experiments import ULTRA_ASSETS as ASSETS
from .trajectories import SCHEMA as TRAJECTORY_SCHEMA, validate_trajectory

SCHEMA = "hoi_ultra_hoi80_evaluation_v1"
ASSET_SCHEMA = "hoi_ultra_hoi80_asset_evaluation_v1"
CANONICAL_SHA256 = "5c2832ba04c1e4061d9a1a1864f820f7ad949e1dd2ab982efff1dbf9511049a2"
PROTOCOL_FILES = {
    "data.py":"21b834f57da4e388a3c108ca8b34e108a698d034eb5077784421b3aa04138fc1",
    "metrics.py":"087742c554f3cf8de5edd83966c9e9479d6614eb1424983c11c5fc35d7700c17",
    "native.py":"450e066fb6dfc7cdb85c8b3b0ca9290556868bf9f19596f88c4977bc1bb53f2c",
    "ultra_kinematics.py":"c5547f78fedbab91e78ea9f9eeb1dc68b232e65f279a2c079a916b0539890710",
    "ultra_math.py":"274d172996fe40f3a5745cf6ea09d617448e2c33e8d15ad8f92ea0bd3b604f85",
}


def verified_protocol():
    import hoi_zero_shot
    package = Path(hoi_zero_shot.__file__).resolve().parent
    if any(file_sha256(package/name)!=digest for name,digest in PROTOCOL_FILES.items()):
        raise ValueError("Canonical HOI80 helper source differs from the frozen protocol")
    return dict(PROTOCOL_FILES)


def validate_source(training_source, runtime=None):
    source = Path(training_source).resolve()
    runtime = Path(runtime or Path(__file__).resolve().parents[1]).resolve()
    manifest_path = source/"SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("branch") != "hoi-retarget/tables-boxes-v1.0" or not manifest.get("files"):
        raise ValueError("Invalid frozen tables-boxes source manifest")
    scientific = {}
    for name,digest in manifest["files"].items():
        target = (source/name).resolve()
        if not target.is_relative_to(source) or not target.is_file() or file_sha256(target)!=digest:
            raise ValueError("Frozen training source changed: "+name)
        if name.endswith(".py") and name.split("/")[0] in ("hoi_tables","hoi_residual","supermimic"):
            if not (runtime/name).is_file() or file_sha256(runtime/name)!=digest:
                raise ValueError("Evaluation differs from frozen training science: "+name)
            scientific[name] = digest
    required = ("hoi_tables/networks.py","hoi_tables/learner.py","hoi_tables/train.py","hoi_tables/ultra.py",
                "hoi_tables/reward.py","hoi_tables/native_checks.py")
    if any(name not in scientific for name in required): raise ValueError("Frozen scientific manifest is incomplete")
    from .eval_monitor import source_tree_hash
    return {"training_source":str(source),"training_source_manifest_sha256":file_sha256(manifest_path),
            "matched_scientific_files":scientific,"evaluation_source":str(runtime),
            "evaluation_source_manifest_sha256":file_sha256(runtime/"SOURCE_MANIFEST.json") if (runtime/"SOURCE_MANIFEST.json").exists() else None,
            "evaluation_source_tree_sha256":source_tree_hash(runtime),"evaluation_module_sha256":file_sha256(__file__)}


def expected_clips(binding):
    cases = binding["cases"]
    clips = {}
    for case in cases:
        key = "ultra/" + PurePosixPath(case["motion"].replace("\\","/")).stem
        if key in clips: raise ValueError("Canonical80 duplicates a motion clip")
        clips[key] = {"asset":case["object_asset"],"state_frames":case["state_frames"],
                      "control_steps":case["control_steps"]}
    counts = [sum(c["asset"]==asset for c in clips.values()) for asset in ASSETS]
    if len(clips)!=80 or counts!=[5,15,6,14,4,16,11,9] or sum(c["control_steps"] for c in clips.values())!=23718:
        raise ValueError("Exact canonical80 membership, asset quotas or duration changed")
    return clips


def load_checkpoints(root, iteration, *, manifest, asset_bundle, geometry_bundle, shape_metadata,
                     reference_contacts, physics, source, smoke=False):
    """Require all eight same-iteration models and RMS tensors to be identical."""
    import torch
    from dataclasses import replace
    from .config import TrainConfig
    from .networks import feature_schema
    if type(iteration) is not int or iteration<1 or (not smoke and iteration%2000):
        raise ValueError("Scheduled HOI80 evaluation requires an exact2000-iteration milestone")
    identities, hashes, models, policy_state = [], {}, {}, None
    paths = {"manifest_sha256":manifest,"asset_catalog_sha256":asset_bundle}
    for rank,asset in enumerate(ASSETS):
        path = Path(root)/f"rank_{rank:02d}"/f"checkpoint_{iteration:06d}.pt"
        digest = file_sha256(path)
        checkpoint = torch.load(path,map_location="cpu",weights_only=True)
        if file_sha256(path)!=digest: raise RuntimeError("Checkpoint changed during frozen read")
        if checkpoint.get("schema")!="hoi_tables_checkpoint_v1" or checkpoint.get("iteration")!=iteration:
            raise ValueError("Direct-teacher checkpoint schema/iteration mismatch")
        identity = checkpoint.get("identity",{})
        if (identity.get("schema")!=feature_schema() or identity.get("asset")!=asset or identity.get("asset_id")!=rank
                or identity.get("world_size")!=8 or identity.get("from_scratch") is not True
                or identity.get("base_checkpoint") is not None or identity.get("smoke") is not False
                or identity.get("schedule_scope")!="500WM_then_fixed_actor_critic" or identity.get("physics")!=physics):
            raise ValueError("Ultra checkpoint physics/features/asset/direct-teacher protocol mismatch")
        experiment = identity.get("experiment",{})
        if (experiment.get("profile")!="ultra" or experiment.get("assets")!=list(ASSETS)
                or experiment.get("geometry_bundle_sha256")!=file_sha256(geometry_bundle)
                or experiment.get("evaluation")!={"kind":"canonical_hoi80","every":2000,"owning_training_run":True}):
            raise ValueError("Ultra checkpoint experiment/geometry protocol mismatch")
        if any(identity.get(name)!=file_sha256(path) for name,path in paths.items()) or identity.get("shape_bank")!=shape_metadata:
            raise ValueError("Ultra checkpoint data/asset/shape identity mismatch")
        if identity.get("reward",{}).get("frozen_reference_contacts",{}).get("file_sha256")!=file_sha256(reference_contacts):
            raise ValueError("Ultra checkpoint contact cache identity mismatch")
        config = TrainConfig(**identity["global_config"]).validate()
        if config.experiment_kind!="formal" or config.wm_only_iters!=500:
            raise ValueError("HOI80 requires a formal500-warmup checkpoint")
        if any(getattr(config,key)%8 for key in ("num_envs","wm_batch","pi_batch")):
            raise ValueError("Ultra global batches do not divide across all eight assets")
        worker = replace(config,num_envs=config.num_envs//8,wm_batch=config.wm_batch//8,pi_batch=config.pi_batch//8).to_dict()
        if _canonical(worker)!=_canonical(identity.get("worker_config")):
            raise ValueError("Ultra checkpoint worker batches differ from the global protocol")
        for name,specification in _model_specifications(config).items():
            _validate_state(checkpoint.get(name),specification,name)
            models.setdefault(name,[]).append(state_dict_hash(checkpoint[name]))
        if identities and _canonical(_common_identity(identity))!=_canonical(_common_identity(identities[0])):
            raise ValueError("Eight-rank checkpoint scientific identities differ")
        hashes[f"rank_{rank:02d}"] = digest
        identities.append(identity)
        if policy_state is None: policy_state = checkpoint["policy"]
        del checkpoint
    if any(len(set(values))!=1 for values in models.values()):
        raise ValueError("Eight-rank checkpoint models or RMS statistics are not synchronized")
    return {"policy_state":policy_state,"identity":identities[0],"iteration":iteration,
            "checkpoint_sha256":hashes,"model_state_sha256":{name:values[0] for name,values in models.items()},"source_identity":source}


class CanonicalRecorder:
    """Post-reset canonical COM-velocity initialization, then T+1 state capture."""
    def __init__(self, env, reference, teacher_kin, trajectory=None):
        self.env, self.reference, self.teacher_kin = env, reference, teacher_kin
        self.states, self.objects, self.initialization = [], [], None
        self.trajectory = trajectory

    def __getattr__(self, name): return getattr(self.env,name)

    def record(self):
        import torch
        from supermimic.mdp import flatten_state
        from supermimic.rotations import mat_to_quat
        state, obj = self.env.state(), self.env.object_state()
        q = mat_to_quat(obj[1])
        self.states.append(flatten_state(state)[0].detach().clone())
        self.objects.append(torch.cat((obj[0],q[:,1:],q[:,:1]),-1)[0].detach().clone())
        if self.trajectory is not None:
            self.trajectory.capture(self.env, state, obj)

    def reset(self, ids, frames):
        import numpy as np
        from hoi_zero_shot.native import initialize_canonical_object
        from hoi_residual.features import object_observation, torso_pose
        self.env.reset(ids,frames)
        self.initialization = initialize_canonical_object(self.env,np.stack([self.reference[0]]))
        self.env.refresh()
        self.env.history.reset(ids,object_observation(torso_pose(self.teacher_kin,self.env.state()),self.env.object_state()))
        self.record()

    def step(self, action):
        out = self.env.step(action)
        if self.trajectory is not None:
            self.trajectory.control(action, out)
        self.record()
        return out


def virtual_scorer(motion, canonical, asset, device):
    from hoi_zero_shot.metrics import VirtualPHC
    meta = canonical["assets"][asset]["metadata"]
    if set(meta["body_names"])!=set(motion.legacy_body_names) or meta["dof_names"]!=motion.legacy_joint_names:
        raise ValueError("Canonical39 body/DOF bindings differ")
    scorer = VirtualPHC(motion.legacy_urdf_path,meta["body_names"],meta["dof_names"],device)
    if scorer.identity["urdf_sha256"]!=motion.legacy_urdf_sha256: raise ValueError("Canonical scoring URDF changed")
    return scorer


def summarize(clips):
    result = tracking_summary(clips)
    result["tables_reward_survival_rate"] = result["success_rate"]
    result.update({key+"_rate":sum(c[key] for c in clips)/len(clips) for key in ("human_pass","object_pass","joint_pass")})
    return result


def aggregate_reports(root, output=None):
    reports = [json.loads((Path(root)/f"asset_{rank:02d}"/"report.json").read_text()) for rank in range(8)]
    common = ("iteration","checkpoint_sha256","model_state_sha256","parent_run_id","source_identity","smoke",
              "input_sha256","checkpoint_identity","manifest_sha256","canonical_sha256","eval_source_sha256",
              "expected_clip_ids","expected_clips","expected_control_steps","protocol_files_sha256",
              "policy_state_sha256_before","policy_state_sha256_after","trajectory_schema")
    expected = reports[0].get("expected_clips",{})
    if len(expected)!=80 or sum(c["control_steps"] for c in expected.values())!=23718:
        raise ValueError("HOI80 report has an invalid expected panel")
    if reports[0].get("canonical_sha256")!=CANONICAL_SHA256 or reports[0].get("protocol_files_sha256")!=PROTOCOL_FILES:
        raise ValueError("HOI80 canonical manifest/scoring implementation identity changed")
    if reports[0].get("trajectory_schema") != TRAJECTORY_SCHEMA:
        raise ValueError("HOI80 requires full native trajectory recording")
    if set(reports[0].get("checkpoint_sha256",{}))!={f"rank_{i:02d}" for i in range(8)}:
        raise ValueError("HOI80 requires all eight checkpoint hashes")
    for rank,report in enumerate(reports):
        count = sum(c["asset"]==ASSETS[rank] for c in expected.values())
        if (report.get("schema")!=ASSET_SCHEMA or report.get("passed") is not True or report.get("asset")!=ASSETS[rank]
                or report.get("clip_count")!=count or len(report.get("clips",[]))!=count):
            raise ValueError("Missing/failed/mislabeled canonical asset report")
        if any(name not in report for name in common) or (rank and any(report[name]!=reports[0][name] for name in common)):
            raise ValueError("HOI80 worker checkpoint/source/input identities differ")
        if report["policy_state_sha256_before"]!=report["policy_state_sha256_after"]:
            raise ValueError("Frozen policy or RMS changed")
        if any(report.get(key) is not True for key in ("no_normalizer_update","no_optimizer_update","full_reference_duration","no_failure_reset")):
            raise ValueError("HOI80 worker violated the frozen full-duration protocol")
        if any(report.get(key)!=0 for key in ("optimizer_updates","normalizer_updates","replay_updates","rsi_updates")):
            raise ValueError("HOI80 worker performed learning")
        if report.get("control_steps")!=sum(c["control_steps"] for c in report["clips"]):
            raise ValueError("HOI80 worker step total differs")
        if any(c["asset"]!=ASSETS[rank] for c in report["clips"]): raise ValueError("Wrong asset lane")
    clips = sorted((c for r in reports for c in r["clips"]),key=lambda c:c["canonical_index"])
    if (len(clips)!=80 or [c["canonical_index"] for c in clips]!=list(range(80))
            or sorted(c["clip_id"] for c in clips)!=sorted(expected) or sorted(expected)!=reports[0]["expected_clip_ids"]):
        raise ValueError("HOI80 coverage is missing, duplicated or outside the canonical panel")
    if [sum(c["asset"]==asset for c in clips) for asset in ASSETS]!=[5,15,6,14,4,16,11,9]:
        raise ValueError("HOI80 heterogeneous asset quotas changed")
    for c in clips:
        if (expected[c["clip_id"]]!={"asset":c["asset"],"state_frames":c["state_frames"],"control_steps":c["control_steps"]}
                or c["partition"]!="protected_eval" or c["control_steps"]!=c["reference_control_steps"]
                or c["control_steps"]!=c["state_frames"]-1 or c["scored_frames"]!=c["control_steps"]
                or c["first_scored_state_index"]!=1 or c["final_scored_state_index"]!=c["control_steps"]
                or c["final_frame"]!=c["reference_end_frame"] or c["start_frame"]!=c["reference_start_frame"]
                or c["full_reference_duration"] is not True or c["no_failure_reset"] is not True or c["reset_calls_during_rollout"]!=0
                or any(type(c.get(k)) is not bool for k in ("human_pass","object_pass","joint_pass"))
                or c["joint_pass"]!=(c["human_pass"] and c["object_pass"])):
            raise ValueError("HOI80 clip was truncated/reset/mis-scored")
    steps = sum(c["control_steps"] for c in clips)
    if steps!=23718 or steps!=reports[0]["expected_control_steps"]: raise ValueError("HOI80 control duration changed")
    for clip in clips:
        validate_trajectory(root, clip, reports[0], verify_payload=True)
    result = {key:reports[0][key] for key in common}
    result.update(schema=SCHEMA,passed=True,partition="protected_eval",clip_count=80,control_steps=steps,
                  no_normalizer_update=True,no_optimizer_update=True,full_reference_duration=True,no_failure_reset=True,
                  optimizer_updates=0,normalizer_updates=0,replay_updates=0,rsi_updates=0,
                  scheduled_evaluation=not result["smoke"],metrics=summarize(clips),clips=clips,
                  per_asset={asset:{"clip_count":sum(c["asset"]==asset for c in clips),
                                    "control_steps":sum(c["control_steps"] for c in clips if c["asset"]==asset),
                                    "metrics":summarize([c for c in clips if c["asset"]==asset])} for asset in ASSETS},
                  success_definition=SUCCESS_DEFINITION,failure_definition=deepcopy(FAILURE_DEFINITION),
                  primary_success_definition="Pinned canonical PHC39 human/object/joint pose gates; distinct from TablesReward survival",
                  domain_randomization=False,nominal_properties=True,policy_deterministic=True,tracked_body_count=14,
                  canonical_scoring_body_count=39)
    result["trajectory_count"] = len(clips)
    result["recorded_state_frames"] = sum(c["state_frames"] for c in clips)
    if not result["smoke"] and result["iteration"]%2000: raise ValueError("Non-milestone scheduled report")
    _canonical(result)
    if output is not None:
        with Path(output).open("x",encoding="utf-8") as stream: json.dump(result,stream,indent=2,allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--aggregate")
    for name in ("checkpoint-root","manifest","asset-bundle","geometry-bundle","shape-bank","reference-contacts",
                 "training-source","parent-run-id","canonical","helper-root"):
        parser.add_argument("--"+name)
    parser.add_argument("--output",required=True)
    parser.add_argument("--iteration",type=int)
    parser.add_argument("--asset",type=int,choices=range(8))
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--seed",type=int,default=0)
    args = parser.parse_args()
    if args.aggregate:
        result = aggregate_reports(args.aggregate,args.output)
        print(json.dumps({"schema":SCHEMA,"passed":True,"clip_count":80,"iteration":result["iteration"]}),flush=True)
        return
    for name in ("checkpoint_root","manifest","asset_bundle","geometry_bundle","shape_bank","reference_contacts",
                 "training_source","parent_run_id","iteration","asset"):
        if getattr(args,name) is None: parser.error("Missing --"+name.replace("_","-"))
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("Run native GPU evaluation through the scheduler")
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.training_source).resolve()) or output.is_relative_to(Path(args.checkpoint_root).resolve()):
        raise ValueError("Evaluation output must be independent of frozen source and training")
    source = validate_source(args.training_source)
    from hoi_residual.runtime import configure_warp_forward_only
    warp_runtime = configure_warp_forward_only()
    from .ultra import (load_ultra_assets,load_ultra_shape,HOI80TeacherMotion,make_ultra_reward,
                        canonical_binding,DEFAULT_CANONICAL)
    args.canonical = args.canonical or DEFAULT_CANONICAL
    binding = canonical_binding(args.manifest,args.canonical,helper_root=args.helper_root)
    expected = expected_clips(binding)
    protocol = verified_protocol()
    assets,hands = load_ultra_assets(args.asset_bundle,args.geometry_bundle)
    points,shape_metadata = load_ultra_shape(args.shape_bank,args.asset_bundle)
    from supermimic.g1_physics import physics_contract
    pair = load_checkpoints(args.checkpoint_root,args.iteration,manifest=args.manifest,asset_bundle=args.asset_bundle,
                            geometry_bundle=args.geometry_bundle,shape_metadata=shape_metadata,reference_contacts=args.reference_contacts,
                            physics=physics_contract(),source=source,smoke=args.smoke)
    import torch
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import PrivilegedFeatures,RobotKinematics38
    from supermimic.robot import default_joint_pos_list
    from supermimic.mdp import MDP,unflatten_state
    import supermimic.networks as shared_networks
    from hoi_residual.capacity import capacity_profile,configure_capacity,capacity_readback
    from hoi_zero_shot.metrics import score
    from .networks import Policy
    from .env import TableSim
    torch.cuda.set_device(args.device); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    shared_networks.AMP_DTYPE = torch.bfloat16
    kin,teacher_kin = G1Kinematics(args.device),RobotKinematics38(device=args.device)
    motion = HOI80TeacherMotion(args.manifest,kin,args.device,asset=args.asset,canonical=args.canonical,helper_root=args.helper_root)
    mdp = MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=args.device))
    reward = make_ultra_reward(mdp,assets,hands,reference_contacts=args.reference_contacts)
    if _reward_identity(reward.contract)!=_reward_identity(pair["identity"]["reward"]): raise ValueError("Frozen reward/contact geometry mismatch")
    policy = Policy(pair["identity"]["global_config"]["actor_hidden"]).to(args.device)
    policy.load_state_dict(pair.pop("policy_state"),strict=True); policy.eval().requires_grad_(False)
    before = state_dict_hash(policy.state_dict())
    if before!=pair["model_state_sha256"]["policy"]: raise RuntimeError("Policy/RMS changed during device load")
    output.mkdir(parents=True,exist_ok=False)
    sim_cfg = make_sim(args.device,args); configure_capacity(sim_cfg,capacity_profile(512,2048,64))
    env = TableSim(sim_cfg,1,motion,mdp,assets[args.asset],output,teacher_kin=teacher_kin,reward=reward,
                   eval_mode=True,max_episode_steps=int(motion.clip_len.max())+1)
    privileged = PrivilegedFeatures(teacher_kin,motion)
    shape = points[args.asset:args.asset+1].to(args.device)
    panel = json.loads(Path(args.canonical).read_text(encoding="utf-8"))
    virtual = virtual_scorer(motion,panel,ASSETS[args.asset],args.device)
    inputs = {name:file_sha256(getattr(args,name)) for name in ("manifest","asset_bundle","geometry_bundle","shape_bank","reference_contacts","canonical")}
    from .trajectories import NativeTrajectory
    trajectory_identity = {"iteration":args.iteration,"parent_run_id":args.parent_run_id,
                           "checkpoint_sha256":pair["checkpoint_sha256"],"model_state_sha256":pair["model_state_sha256"],
                           "source_identity":source,"canonical_sha256":CANONICAL_SHA256,"input_sha256":inputs,
                           "asset_bundle":str(Path(args.asset_bundle).resolve()),"robot_urdf":teacher_kin.urdf_path,
                           "robot_visual_urdf":str(Path(__file__).resolve().parents[1]/"data/g1_hw3/converted.urdf"),
                           "robot_mesh_root":str(Path(__file__).resolve().parents[1]/"data/g1_hw3")}
    clips = []
    for clip,case in enumerate(motion.cases):
        reference = motion.load_reference(clip)
        trajectory = NativeTrajectory(motion,teacher_kin)
        recorder = CanonicalRecorder(env,reference,teacher_kin,trajectory)
        record = rollout_clip(recorder,motion,mdp,privileged,policy,clip,teacher_kin,shape,assets[args.asset])
        states = unflatten_state(torch.stack(recorder.states))
        body = virtual(states)
        common = virtual.audit_common_positions(body,teacher_kin.forward(states)[0],teacher_kin.body_names)
        canonical_score = score(body,torch.stack(recorder.objects),reference,dt=.02)
        record.update(canonical_score,canonical_case_id=case["id"],canonical_index=case["canonical_index"],
                      motion_sha256=case["motion_sha256"],reference_sha256=case["reference_sha256"],
                      canonical_initialization=recorder.initialization,virtual39_common_fk=common)
        record["trajectory"] = trajectory.save(output,record,trajectory_identity,reference)
        validate_trajectory(output.parent,record,trajectory_identity,verify_payload=True)
        clips.append(record)
        print(json.dumps({"phase":"canonical_clip_complete","iteration":args.iteration,"clip_id":record["clip_id"],
                          "control_steps":record["control_steps"],"human_pass":record["human_pass"],
                          "object_pass":record["object_pass"],"joint_pass":record["joint_pass"]}),flush=True)
    after = state_dict_hash(policy.state_dict())
    if after!=before: raise RuntimeError("HOI80 changed policy/RMS tensors")
    if any(file_sha256(Path(args.checkpoint_root)/rank/f"checkpoint_{args.iteration:06d}.pt")!=digest for rank,digest in pair["checkpoint_sha256"].items()):
        raise RuntimeError("Evaluated checkpoint changed during rollout")
    report = {"schema":ASSET_SCHEMA,"passed":True,"partition":"protected_eval","asset":ASSETS[args.asset],"asset_id":args.asset,
              "iteration":args.iteration,"parent_run_id":args.parent_run_id,"checkpoint_sha256":pair["checkpoint_sha256"],
              "model_state_sha256":pair["model_state_sha256"],"source_identity":source,"smoke":args.smoke,
              "scheduled_evaluation":not args.smoke,"manifest_sha256":inputs["manifest"],"canonical_sha256":inputs["canonical"],
              "eval_source_sha256":source["evaluation_source_tree_sha256"],"input_sha256":inputs,
              "checkpoint_identity":_common_identity(pair["identity"]),"policy_state_sha256_before":before,"policy_state_sha256_after":after,
              "expected_clip_ids":sorted(expected),"expected_clips":expected,"expected_control_steps":23718,
              "protocol_files_sha256":protocol,"virtual39_identity":virtual.identity,"clip_count":len(clips),
              "control_steps":sum(c["control_steps"] for c in clips),"clips":clips,"metrics":summarize(clips),
              "no_normalizer_update":True,"no_optimizer_update":True,"full_reference_duration":True,"no_failure_reset":True,
              "optimizer_updates":0,"normalizer_updates":0,"replay_updates":0,"rsi_updates":0,
              "domain_randomization":False,"nominal_properties":True,"policy_deterministic":True,"seed":args.seed,
              "success_definition":SUCCESS_DEFINITION,"failure_definition":deepcopy(FAILURE_DEFINITION),
              "native_physics":env.physics_report,"native_object":env.object_solver_report,"native_contacts":env.contacts.evidence,
              "native_capacity":capacity_readback(env.physics),"warp_runtime":warp_runtime}
    report["trajectory_schema"] = TRAJECTORY_SCHEMA
    with (output/"report.json").open("x",encoding="utf-8") as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print("ULTRA_HOI80_ASSET_EVALUATION_PASS",flush=True)


if __name__=="__main__": main()
