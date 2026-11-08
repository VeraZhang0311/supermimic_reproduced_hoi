"""Independent full-duration protected evaluation of frozen tables teachers.

The evaluator reads both same-iteration rank checkpoints and an immutable
training-source snapshot. It creates no learner, replay, optimizer or RSI.
Asset workers run one nominal environment and score every reference interval,
continuing after failure. The aggregate report covers all22 protected clips.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path

ASSETS = ("largetable", "smalltable")
SCHEMA = "hoi_tables_protected_evaluation_v1"
ASSET_SCHEMA = "hoi_tables_protected_asset_evaluation_v1"
MAX_STEP_REWARD = .15
CONTROL_DT = .02
FAILURE_DEFINITION = {
    "robot_root_height_error_m": .25,
    "robot_reference_minus_actual_Rzz_absolute_difference": .8,
    "Rzz_threshold_units": "dimensionless matrix-entry difference, not radians",
    "object_origin_position_threshold_m": "max(1.0,2.0*object_collision_bbox_diagonal_m)",
    "comparison": "strict_greater", "reference_end_or_time_limit_is_failure": False,
}
SUCCESS_DEFINITION = "No frozen TablesReward termination flag over the complete reference; not a grasp or engineering-quality claim"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def state_dict_hash(state):
    import torch
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is empty")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Invalid/nonfinite frozen model tensor: " + name)
        value = tensor.detach().cpu().contiguous()
        digest.update(_canonical([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_source_compatibility(training_source, evaluation_source=None):
    """Check the frozen manifest and every existing scientific Python file."""
    source = Path(training_source).resolve()
    runtime = Path(evaluation_source or Path(__file__).resolve().parents[1]).resolve()
    manifest_path = source/"SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("branch") not in ("hoi-retarget/tables-v1.0", "hoi-retarget/tables-boxes-v1.0") or not manifest.get("files"):
        raise ValueError("Invalid frozen tables source manifest")
    scientific = {}
    for name, expected in manifest["files"].items():
        target = (source/name).resolve()
        if not target.is_relative_to(source) or not target.is_file() or file_sha256(target) != expected:
            raise ValueError("Frozen training source changed: " + name)
        if name.endswith(".py") and name.split("/")[0] in ("hoi_tables", "hoi_residual", "supermimic"):
            candidate = runtime/name
            if not candidate.is_file() or file_sha256(candidate) != expected:
                raise ValueError("Evaluation scientific source differs from frozen training: " + name)
            scientific[name] = expected
    if not scientific:
        raise ValueError("Frozen manifest contains no scientific source files")
    from .eval_monitor import source_tree_hash
    return {"training_source": str(source), "training_source_manifest_sha256": file_sha256(manifest_path),
            "matched_scientific_files": scientific, "evaluation_source": str(runtime),
            "evaluation_source_manifest_sha256": file_sha256(runtime/"SOURCE_MANIFEST.json") if (runtime/"SOURCE_MANIFEST.json").exists() else None,
            "evaluation_source_tree_sha256": source_tree_hash(runtime),
            "evaluation_module_sha256": file_sha256(__file__)}


def _reward_identity(contract):
    """Geometry bytes/transforms matter; checkout paths and worker filters do not."""
    from .prepare_contacts import _hands_identity
    result = deepcopy(contract)
    if "hand_geometry" in result:
        result["hand_geometry"] = _hands_identity(result["hand_geometry"])
    frozen = result.get("frozen_reference_contacts")
    if isinstance(frozen, dict):
        frozen.pop("selected_clips", None)
    return result


def _common_identity(identity):
    result = deepcopy(identity)
    result.pop("asset", None); result.pop("asset_id", None)
    result["reward"] = _reward_identity(result["reward"])
    return result


def _validate_state(state, specification, label):
    import torch
    if not isinstance(state, dict) or set(state) != set(specification):
        raise ValueError(label + " checkpoint tensor keys differ")
    for name, target in specification.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(label + " checkpoint tensor shape/dtype/finite mismatch: " + name)


def _model_specifications(config):
    import torch
    from .networks import Policy, WorldModel, Critic
    with torch.device("meta"):
        models = {"policy": Policy(config.actor_hidden), "wm": WorldModel(config.wm_hidden, config.contact_hidden),
                  "critic": Critic(config.critic_hidden)}
    return {name: model.state_dict() for name, model in models.items()}


def load_checkpoint_pair(checkpoint_root, iteration, *, manifest_path, asset_bundle, shape_metadata,
                         physics_identity, source_identity, reference_contacts, smoke=False):
    """Read two immutable snapshots, validate identity and synchronized full states."""
    import torch
    from .config import TrainConfig
    from .networks import feature_schema
    if type(iteration) is not int or iteration < 1 or (not smoke and iteration % 2000):
        raise ValueError("Scheduled evaluation requires an exact2000-iteration milestone")
    records, hashes, model_hashes = [], {}, {}
    for rank, asset in enumerate(ASSETS):
        path = Path(checkpoint_root)/f"rank_{rank:02d}"/f"checkpoint_{iteration:06d}.pt"
        before = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != before:
            raise RuntimeError("Checkpoint changed during frozen read")
        if checkpoint.get("schema") != "hoi_tables_checkpoint_v1" or checkpoint.get("iteration") != iteration:
            raise ValueError("Tables checkpoint schema/iteration mismatch")
        identity = checkpoint.get("identity", {})
        required = {"schema", "physics", "global_config", "worker_config", "asset", "asset_id", "world_size",
                    "from_scratch", "base_checkpoint", "reward", "manifest_sha256", "asset_catalog_sha256",
                    "shape_bank", "smoke", "schedule_scope"}
        if not required.issubset(identity):
            raise ValueError("Tables checkpoint identity is incomplete")
        if (identity["schema"] != feature_schema() or identity["asset"] != asset or identity["asset_id"] != rank
                or identity["world_size"] != 2 or identity["from_scratch"] is not True
                or identity["base_checkpoint"] is not None or identity["smoke"] is not False
                or identity["schedule_scope"] != "500WM_then_fixed_actor_critic"):
            raise ValueError("Tables checkpoint feature/asset/from-scratch protocol mismatch")
        if identity["physics"] != physics_identity:
            raise ValueError("Frozen checkpoint physics/source identity mismatch")
        if (identity["manifest_sha256"] != file_sha256(manifest_path)
                or identity["asset_catalog_sha256"] != file_sha256(asset_bundle)
                or identity["shape_bank"] != shape_metadata):
            raise ValueError("Frozen checkpoint data/asset/shape identity mismatch")
        frozen_contact = identity["reward"].get("frozen_reference_contacts", {})
        if frozen_contact.get("file_sha256") != file_sha256(reference_contacts):
            raise ValueError("Frozen checkpoint reference-contact cache identity mismatch")
        config = TrainConfig(**identity["global_config"]).validate()
        if config.experiment_kind != "formal" or config.wm_only_iters != 500:
            raise ValueError("Protected evaluation requires a formal500-warmup teacher")
        for name, spec in _model_specifications(config).items():
            _validate_state(checkpoint.get(name), spec, name)
            model_hashes.setdefault(name, []).append(state_dict_hash(checkpoint[name]))
        hashes[f"rank_{rank:02d}"] = before
        records.append(checkpoint)
    if _canonical(_common_identity(records[0]["identity"])) != _canonical(_common_identity(records[1]["identity"])):
        raise ValueError("Paired checkpoint scientific identities differ")
    if any(len(set(values)) != 1 for values in model_hashes.values()):
        raise ValueError("Paired checkpoint models or RMS statistics are not synchronized")
    return {"policy_state": records[0]["policy"], "identity": records[0]["identity"],
            "iteration": iteration, "checkpoint_sha256": hashes,
            "model_state_sha256": {name: values[0] for name, values in model_hashes.items()},
            "source_identity": source_identity, "checkpoint_root": str(Path(checkpoint_root).resolve())}


def validate_nominal_snapshot(env, snapshot, asset):
    """Check nominal parameters without modifying simulation or observations."""
    import torch
    if not env.eval_mode or env.sampler is not None or env.randomizer.enabled:
        raise ValueError("Protected evaluation requires nominal mode and no RSI sampler")
    xi, actuator, context = snapshot["xi"], snapshot["actuator"], snapshot["xi_o"]
    if (not torch.equal(xi[:, 1:], torch.zeros_like(xi[:, 1:]))
            or not torch.equal(xi[:, 0], torch.ones_like(xi[:, 0]))
            or not torch.equal(actuator[:, 29], torch.full_like(actuator[:, 29], 3))
            or not torch.equal(actuator[:, 30:], torch.ones_like(actuator[:, 30:]))):
        raise RuntimeError("Evaluation robot context/motor parameters are randomized")
    profile = asset["physical_profile"]
    inertia = torch.tensor(profile["inertia"], dtype=context.dtype, device=context.device)
    expected = torch.cat((context.new_tensor([profile["mass"]]), context.new_tensor(profile["com"]),
                          inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]], context.new_tensor([profile["friction"], 1.])))
    if not torch.allclose(context, expected[None].expand_as(context), atol=1e-6, rtol=1e-5):
        raise RuntimeError("Evaluation object physics context differs from nominal asset")


def rollout_clip(env, motion, mdp, privileged, policy, clip, teacher_kin, shape_points, asset):
    """Score exactly T-1 actions and T states; never reset after failure."""
    import torch
    with torch.no_grad():
        return _rollout_clip(env,motion,mdp,privileged,policy,clip,teacher_kin,shape_points,asset)


def _rollout_clip(env, motion, mdp, privileged, policy, clip, teacher_kin, shape_points, asset):
    import torch
    from hoi_residual.features import oracle_features, torso_pose
    from supermimic.mdp import clamp_action
    from supermimic.rotations import so3_log
    if motion.partition != "protected_eval" or motion.for_training or env.N != 1:
        raise ValueError("Evaluator requires one protected, non-training environment")
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
    steps = int(motion.clip_len[clip])-1
    if steps < 1 or end-start != steps:
        raise ValueError("Evaluation reference has no complete full-duration clip")
    env.reset(ids, torch.tensor([start], device=env.device))
    env.refresh()
    validate_nominal_snapshot(env, env.snapshot(), asset)
    initial_episode = env.episode_id.clone()
    names = ("object_position_error_m", "object_orientation_error_rad", "root_position_error_m",
             "root_orientation_error_rad", "tracked_body_position_error_m", "tracked_body_max_position_error_m")
    errors = {name: [] for name in names}
    total, first_failure, failure_steps, robot_failure_steps, object_failure_steps = 0., None, 0, 0, 0
    quality_steps, clipped_steps, terms = 0, 0, {}
    for tick in range(steps):
        snapshot = env.snapshot()
        S, O = env.state(), env.object_state()
        robot = privileged(S, snapshot["f"], snapshot["a_prev"], snapshot["xi"],
                           snapshot["q_prev"], snapshot["qd_prev"], snapshot["actuator"])
        oracle = oracle_features(robot, torso_pose(teacher_kin, S), O, motion, snapshot["f"],
                                 snapshot["xi_o"], snapshot["object_history"])
        raw = policy(oracle, shape_points)
        action = clamp_action(raw)
        clipped_steps += int(not torch.equal(action, raw))
        result = env.step(action)
        if not torch.equal(env.episode_id, initial_episode):
            raise RuntimeError("Protected evaluator reset an episode during its scored clip")
        if int(env.f.item()) != start+tick+1:
            raise RuntimeError("Protected evaluation reference advanced incorrectly")
        actual, obj, f = env.state(), env.object_state(), env.f
        reference, obj_ref = motion.ref_state(f), motion.object_ref(f)
        robot_failed = bool(mdp.termination(actual, f).item())
        object_failed = bool(((obj[0]-obj_ref[0]).norm(dim=-1) > env.reward_model.object_failure_threshold(f)).item())
        failed = bool(result["terminated"].item())
        if failed != (robot_failed or object_failed):
            raise RuntimeError("Evaluation failure flags differ from the frozen tables objective")
        if failed and first_failure is None: first_failure = tick+1
        failure_steps += int(failed); robot_failure_steps += int(robot_failed); object_failure_steps += int(object_failed)
        reward = float(result["reward"].item())
        if not math.isfinite(reward): raise FloatingPointError("Nonfinite protected evaluation reward")
        total += reward
        for key, value in result["terms"].items(): terms[key] = terms.get(key, 0.) + float(value.item())
        bp = mdp.kin.forward(*actual)[0]
        body_error = (bp-motion.reference_bodies(f)[0]).norm(dim=-1)
        values = {"object_position_error_m": (obj[0]-obj_ref[0]).norm(dim=-1),
                  "object_orientation_error_rad": so3_log(obj[1]@obj_ref[1].transpose(-1,-2)).norm(dim=-1),
                  "root_position_error_m": (actual[0]-reference[0]).norm(dim=-1),
                  "root_orientation_error_rad": so3_log(actual[1]@reference[1].transpose(-1,-2)).norm(dim=-1),
                  "tracked_body_position_error_m": body_error.mean(-1),
                  "tracked_body_max_position_error_m": body_error.amax(-1)}
        for name, value in values.items():
            number = float(value.item())
            if not math.isfinite(number): raise FloatingPointError("Nonfinite protected tracking metric: "+name)
            errors[name].append(number)
        quality_steps += int(values["tracked_body_max_position_error_m"].item()<.5
                             and values["object_position_error_m"].item()<.1
                             and values["object_orientation_error_rad"].item()<math.pi/6)
    if int(env.f.item()) != end:
        raise RuntimeError("Protected evaluation did not reach its exact final reference state")
    validate_nominal_snapshot(env, env.snapshot(), asset)
    entry = motion.entries[clip]
    free_steps = steps if first_failure is None else first_failure-1
    return {"clip_id": entry["clip_id"], "source_group": entry["source_group"], "asset": entry["object_asset"],
            "partition": "protected_eval", "state_frames": steps+1, "control_steps": steps,
            "reference_control_steps": steps, "reference_duration_s": steps*CONTROL_DT,
            "start_frame": start, "final_frame": int(env.f.item()),
            "reference_start_frame": start, "reference_end_frame": end,
            "full_reference_duration": True, "no_failure_reset": True, "reset_calls_during_rollout": 0,
            "failed": first_failure is not None, "first_failure_step": first_failure,
            "failure_free_steps": free_steps, "failure_free_seconds": free_steps*CONTROL_DT,
            "terminated_steps": failure_steps, "robot_failure_steps": robot_failure_steps,
            "object_failure_steps": object_failure_steps, "return": total,
            "normalized_return": total/(steps*MAX_STEP_REWARD),
            "tracking_quality_fraction": quality_steps/steps, "tracking_quality_all_steps": quality_steps==steps,
            "action_clipped_fraction": clipped_steps/steps,
            "reward_terms_mean": {name:value/steps for name,value in terms.items()},
            **{name:{"mean":sum(value)/steps,"max":max(value)} for name,value in errors.items()}}


def summarize(clips):
    if not clips: raise ValueError("Evaluation produced no clips")
    steps = sum(c["control_steps"] for c in clips)
    return {"success_rate": sum(not c["failed"] for c in clips)/len(clips),
            "mean_normalized_return": sum(c["normalized_return"] for c in clips)/len(clips),
            "frame_weighted_normalized_return": sum(c["return"] for c in clips)/(steps*MAX_STEP_REWARD),
            "mean_failure_free_seconds": sum(c["failure_free_seconds"] for c in clips)/len(clips),
            "tracking_quality_fraction": sum(c["tracking_quality_fraction"]*c["control_steps"] for c in clips)/steps,
            **{name:sum(c[name]["mean"]*c["control_steps"] for c in clips)/steps
               for name in ("object_position_error_m", "object_orientation_error_rad", "tracked_body_position_error_m")}}


def aggregate_reports(root, output=None):
    reports = [json.loads((Path(root)/f"asset_{rank:02d}"/"report.json").read_text()) for rank in range(2)]
    common = ("iteration", "checkpoint_sha256", "model_state_sha256", "parent_run_id", "source_identity", "smoke", "input_sha256", "checkpoint_identity",
              "manifest_sha256", "eval_source_sha256", "expected_clip_ids", "expected_control_steps", "expected_clips",
              "policy_state_sha256_before", "policy_state_sha256_after")
    for rank, report in enumerate(reports):
        if report.get("schema") != ASSET_SCHEMA or not report.get("passed") or report.get("asset") != ASSETS[rank]:
            raise ValueError("Missing/failed/mislabeled asset evaluation report")
        if report.get("clip_count") != 11 or len(report.get("clips", [])) != 11:
            raise ValueError("Asset evaluation must contain all11 protected clips")
        if any(c.get("asset")!=ASSETS[rank] for c in report["clips"]):
            raise ValueError("Protected clip asset differs from its worker")
        if any(name not in report for name in common):
            raise ValueError("Asset evaluation report lacks checkpoint/input/source identity")
        if any(report.get(name)!=0 for name in ("optimizer_updates","normalizer_updates","replay_updates","rsi_updates")):
            raise ValueError("Protected evaluation performed training updates")
        if report.get("control_steps")!=sum(c["control_steps"] for c in report["clips"]):
            raise ValueError("Asset evaluation step total differs from its clips")
        if any(report.get(key) is not True for key in ("no_normalizer_update", "no_optimizer_update", "full_reference_duration", "no_failure_reset")):
            raise ValueError("Asset evaluation protocol is not frozen/full-duration")
        if rank and any(report.get(key) != reports[0].get(key) for key in common):
            raise ValueError("Asset evaluation checkpoint/source/iteration identities differ")
        if report["policy_state_sha256_before"] != report["policy_state_sha256_after"]:
            raise ValueError("Frozen policy or RMS changed during evaluation")
    clips = sorted([c for r in reports for c in r["clips"]], key=lambda c:c["clip_id"])
    expected = reports[0]["expected_clip_ids"]
    if len(clips)!=22 or len(set(c["clip_id"] for c in clips))!=22 or sorted(c["clip_id"] for c in clips)!=sorted(expected):
        raise ValueError("Protected evaluation coverage is missing, duplicated or outside the22-clip partition")
    for clip in clips:
        expected_clip = reports[0]["expected_clips"].get(clip["clip_id"],{})
        if (clip["partition"]!="protected_eval" or clip["control_steps"]!=clip["reference_control_steps"]
                or clip["control_steps"]!=clip["state_frames"]-1 or clip["final_frame"]!=clip["reference_end_frame"]
                or clip["start_frame"]!=clip["reference_start_frame"] or not clip["full_reference_duration"]
                or not clip["no_failure_reset"] or clip["reset_calls_during_rollout"]!=0):
            raise ValueError("Protected clip was truncated, reset or scored on the wrong interval")
        if expected_clip!={"asset":clip["asset"],"state_frames":clip["state_frames"],"control_steps":clip["control_steps"]}:
            raise ValueError("Protected clip duration/asset differs from its frozen manifest")
    steps = sum(c["control_steps"] for c in clips)
    if steps != reports[0]["expected_control_steps"]:
        raise ValueError("Protected evaluation control-step coverage differs from its manifest")
    result = {key:reports[0][key] for key in common}
    result.update(schema=SCHEMA, passed=True, partition="protected_eval", clip_count=22, control_steps=steps,
                  no_normalizer_update=True, no_optimizer_update=True, full_reference_duration=True, no_failure_reset=True,
                  scheduled_evaluation=not result["smoke"], metrics=summarize(clips), clips=clips,
                  per_asset={name:{"clip_count":11,"control_steps":sum(c["control_steps"] for c in clips if c["asset"]==name),
                                   "metrics":summarize([c for c in clips if c["asset"]==name])} for name in ASSETS})
    result.update(success_definition=SUCCESS_DEFINITION,failure_definition=deepcopy(FAILURE_DEFINITION),
                  tracked_body_count=14,domain_randomization=False,nominal_properties=True,
                  policy_deterministic=True)
    if "tracking_quality_definition" in reports[0]:
        result["tracking_quality_definition"] = reports[0]["tracking_quality_definition"]
    if not result["smoke"] and result["iteration"]%2000:
        raise ValueError("A scheduled report cannot label a non-milestone checkpoint")
    _canonical(result)  # reject NaN/Inf before publishing a completed report
    if output is not None:
        with Path(output).open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--aggregate", help="Directory containing asset_00/01/report.json")
    for name in ("checkpoint-root", "manifest", "asset-bundle", "shape-bank", "reference-contacts", "training-source", "parent-run-id"):
        parser.add_argument("--"+name)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--asset", type=int, choices=(0,1))
    parser.add_argument("--smoke", action="store_true", help="Full22-clip validation only; never a scheduled W&B point")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.aggregate:
        report = aggregate_reports(args.aggregate, args.output)
        print(json.dumps({"schema":SCHEMA,"passed":True,"clip_count":report["clip_count"],"iteration":report["iteration"]}),flush=True)
        return
    for name in ("checkpoint_root", "manifest", "asset_bundle", "shape_bank", "reference_contacts", "training_source", "parent_run_id", "iteration", "asset"):
        if getattr(args,name) is None: parser.error("Missing --"+name.replace("_","-"))
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("Run native GPU evaluation through the scheduler")
    import torch
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.training_source).resolve()) or output.is_relative_to(Path(args.checkpoint_root).resolve()):
        raise ValueError("Evaluation output must be independent of frozen training source and active run")
    source = validate_source_compatibility(args.training_source)
    from hoi_residual.runtime import configure_warp_forward_only
    warp_runtime = configure_warp_forward_only()
    from supermimic.g1_physics import physics_contract
    from .assets import load_assets, load_shape_bank
    assets = load_assets(args.asset_bundle)
    points, shape_metadata = load_shape_bank(args.shape_bank, args.asset_bundle)
    pair = load_checkpoint_pair(args.checkpoint_root,args.iteration,manifest_path=args.manifest,
                                 asset_bundle=args.asset_bundle,shape_metadata=shape_metadata,
                                 physics_identity=physics_contract(),source_identity=source,
                                 reference_contacts=args.reference_contacts,smoke=args.smoke)
    from .data import TableMotionLib, read_manifest
    manifest = read_manifest(args.manifest)
    protected = [e for e in manifest["clips"] if e["partition"]=="protected_eval"]
    if len(protected)!=22 or any(sum(e["object_asset"]==a for e in protected)!=11 for a in ASSETS):
        raise ValueError("Expected exactly11 protected clips per table asset")
    from supermimic.env import make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.privileged import PrivilegedFeatures, RobotKinematics38
    from supermimic.robot import default_joint_pos_list
    from supermimic.mdp import MDP
    import supermimic.networks as shared_networks
    from hoi_residual.capacity import capacity_profile,configure_capacity,capacity_readback
    from .networks import Policy
    from .reward import TablesReward
    from .env import TableSim
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    shared_networks.AMP_DTYPE = torch.bfloat16
    kin, teacher_kin = G1Kinematics(args.device), RobotKinematics38(device=args.device)
    motion = TableMotionLib(args.manifest,kin,args.device,partition="protected_eval",for_training=False,asset=args.asset)
    mdp = MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=args.device))
    catalog = json.loads(Path(args.asset_bundle).read_text())
    hands = catalog["hands"]["geometry_cache"]
    reward = TablesReward(mdp,assets,hands,reference_contacts=args.reference_contacts)
    if _reward_identity(reward.contract)!=_reward_identity(pair["identity"]["reward"]):
        raise ValueError("Frozen checkpoint reward/contact/hand geometry contract mismatch")
    policy = Policy(pair["identity"]["global_config"]["actor_hidden"]).to(args.device)
    policy.load_state_dict(pair.pop("policy_state"),strict=True)
    policy.eval().requires_grad_(False)
    before = state_dict_hash(policy.state_dict())
    if before!=pair["model_state_sha256"]["policy"]: raise RuntimeError("Policy tensors changed during device load")
    output.mkdir(parents=True,exist_ok=False)
    sim_cfg = make_sim(args.device,args)
    configure_capacity(sim_cfg,capacity_profile(512,2048,64))
    env = TableSim(sim_cfg,1,motion,mdp,assets[args.asset],output,teacher_kin=teacher_kin,reward=reward,
                   eval_mode=True,max_episode_steps=int(motion.clip_len.max())+1)
    privileged = PrivilegedFeatures(teacher_kin,motion)
    shape = points[args.asset:args.asset+1].to(args.device)
    clips = []
    for clip in range(len(motion.entries)):
        record = rollout_clip(env,motion,mdp,privileged,policy,clip,teacher_kin,shape,assets[args.asset])
        clips.append(record)
        print(json.dumps({"phase":"protected_clip_complete","iteration":args.iteration,"clip_id":record["clip_id"],
                          "control_steps":record["control_steps"],"failed":record["failed"]}),flush=True)
    after = state_dict_hash(policy.state_dict())
    if after!=before: raise RuntimeError("Protected evaluation changed policy weights or RMS statistics")
    if any(file_sha256(Path(args.checkpoint_root)/rank/f"checkpoint_{args.iteration:06d}.pt")!=digest
           for rank,digest in pair["checkpoint_sha256"].items()):
        raise RuntimeError("Evaluated checkpoint pair changed during rollout")
    report = {"schema":ASSET_SCHEMA,"passed":True,"partition":"protected_eval","asset":ASSETS[args.asset],
              "asset_id":args.asset,"iteration":args.iteration,"parent_run_id":args.parent_run_id,
              "checkpoint_sha256":pair["checkpoint_sha256"],"model_state_sha256":pair["model_state_sha256"],
              "source_identity":source,"smoke":args.smoke,"scheduled_evaluation":not args.smoke,
              "manifest_sha256":file_sha256(args.manifest),"eval_source_sha256":source["evaluation_source_tree_sha256"],
              "input_sha256":{name:file_sha256(getattr(args,name)) for name in ("manifest","asset_bundle","shape_bank","reference_contacts")},
              "checkpoint_identity":_common_identity(pair["identity"]),
              "policy_state_sha256_before":before,"policy_state_sha256_after":after,
              "expected_clip_ids":sorted(e["clip_id"] for e in protected),
              "expected_clips":{e["clip_id"]:{"asset":e["object_asset"],"state_frames":e["state_frames"],
                                             "control_steps":e["state_frames"]-1} for e in protected},
              "expected_control_steps":sum(e["state_frames"]-1 for e in protected),
              "clip_count":len(clips),"control_steps":sum(c["control_steps"] for c in clips),"clips":clips,"metrics":summarize(clips),
              "no_normalizer_update":True,"no_optimizer_update":True,"full_reference_duration":True,"no_failure_reset":True,
              "optimizer_updates":0,"normalizer_updates":0,"replay_updates":0,"rsi_updates":0,
              "nominal_domain_randomization":True,"policy_deterministic":True,"seed":args.seed,
              "domain_randomization":False,"nominal_properties":True,
              "success_definition":SUCCESS_DEFINITION,"failure_definition":deepcopy(FAILURE_DEFINITION),
              "bitwise_simulator_determinism_claim":False,"tracked_body_count":14,
              "tracking_quality_definition":{"body_max_position_m":.5,"object_position_m":.1,"object_orientation_rad":math.pi/6,
                                             "meaning":"per-step engineering joint tracking; distinct from frozen failure/success"},
              "native_physics":env.physics_report,"native_object":env.object_solver_report,
              "native_contacts":env.contacts.evidence,"native_capacity":capacity_readback(env.physics),"warp_runtime":warp_runtime}
    with (output/"report.json").open("x",encoding="utf-8") as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print("TABLES_PROTECTED_ASSET_EVALUATION_PASS",flush=True)


if __name__=="__main__": main()
