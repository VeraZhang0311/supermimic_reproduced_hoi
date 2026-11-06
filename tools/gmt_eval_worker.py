"""Evaluate one immutable GMT checkpoint with its frozen evaluator function.

Run with the same kitless GPU Python/runtime as training, from an independent
GPU allocation. Importing this module needs only the standard library.
"""
from __future__ import annotations

import argparse
import ast
import __future__
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import sys
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checkpoint_iteration(path):
    match = re.fullmatch(r"ckpt_(\d{6})\.pt", Path(path).name)
    if match is None or int(match[1]) <= 0 or int(match[1]) % 2000:
        raise ValueError("Expected an exact positive 2000-iteration checkpoint filename")
    return int(match[1])


def validate_checkpoint_identity(checkpoint, iteration, expected_contract, dataset="lafan1"):
    if type(checkpoint.get("iteration")) is not int or checkpoint["iteration"] != iteration:
        raise ValueError("Checkpoint filename/metadata iteration mismatch")
    if checkpoint.get("gmt_contract") != expected_contract:
        raise ValueError("Frozen observation/physics/robot/dataset contract mismatch")
    if not isinstance(checkpoint.get("args"), dict) or checkpoint["args"].get("dataset") != dataset:
        raise ValueError("Checkpoint evaluation dataset mismatch")


def load_evaluate_function(path, namespace):
    """Compile only the frozen evaluate() AST; never execute its watcher/imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate"]
    if len(functions) != 1:
        raise ValueError("Frozen source must define exactly one evaluate function")
    code = compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    scope = dict(namespace)
    exec(code, scope)
    return scope["evaluate"]


def publish_json(path, value):
    """Atomically publish a complete result, refusing to replace any result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_metrics(metrics, names, expected_count=40):
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError(f"Nominal evaluation requires {expected_count} distinct complete clips")
    for key in ("norm_return", "norm_return_framew"):
        if type(metrics.get(key)) not in (float, int) or not math.isfinite(metrics[key]):
            raise ValueError("Evaluation returned a nonfinite metric")
    failed, count = metrics.get("failed"), metrics.get("n_failed")
    if (type(count) is not int or not isinstance(failed, list) or count != len(failed)
            or len(set(failed)) != count or not set(failed).issubset(names)):
        raise ValueError("Evaluation failed-clip identity/count mismatch")
    per_clip = metrics.get("per_clip")
    if not isinstance(per_clip, dict) or set(per_clip) != set(names):
        raise ValueError("Evaluation did not cover every complete clip")
    if not all(type(value) in (float, int) and math.isfinite(value) for value in per_clip.values()):
        raise ValueError("Evaluation returned a nonfinite per-clip metric")


def state_signature(policy):
    # No parameter-sized copies. Ordinary torch in-place updates increment
    # _version; replacing a normalizer buffer changes its storage identity.
    return {name: (value.data_ptr(), value._version, tuple(value.shape), str(value.dtype), str(value.device))
            for name, value in policy.state_dict().items()}


def validate_source_package(package, source):
    # supermimic is a namespace package (no __init__.py / __file__). Reject
    # a namespace that would permit fallback imports from a different checkout.
    locations = list(package.__path__)
    if not locations or any(not Path(path).resolve().is_relative_to(source) for path in locations):
        raise ValueError("supermimic was imported outside the requested frozen source")


def validate_boneseed_eval_library(full_directory, eval_directory, run_contract, expected_count=512):
    """Verify a complete-clips training subset without opening full states.bin.

    Hashes bind the small library to the training manifest; exact name, timing,
    offset and source lineage checks prohibit cropped or reprocessed clips.
    The selection is a fixed seed-zero subset of training, never held out.
    """
    full_directory, eval_directory = Path(full_directory).resolve(), Path(eval_directory).resolve()
    if full_directory == eval_directory:
        raise ValueError("Boneseed evaluation must use the separate small library")
    full_path, eval_path = full_directory / "manifest.json", eval_directory / "manifest.json"
    full_hash, eval_hash = sha256(full_path), sha256(eval_path)
    if run_contract.get("dataset") != "boneseed" or run_contract.get("motion_manifest_sha256") != full_hash:
        raise ValueError("Training full manifest differs from the run contract")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    selection_path = eval_directory / "selection.json"
    selection_hash = sha256(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (full.get("format") != "bones_seed_g1_float32_50hz_v1"
            or full.get("selection") != "all_original_filtered_clips"
            or evaluation.get("format") != full["format"]
            or evaluation.get("selection") != "eval_training_subset"
            or selection.get("kind") != "eval_training_subset"
            or evaluation.get("held_out") is not False or selection.get("held_out") is not False
            or evaluation.get("training_excludes_selection") is not False
            or selection.get("training_excludes_selection") is not False
            or evaluation.get("parent_manifest_sha256") != full_hash
            or selection.get("parent_manifest_sha256") != full_hash
            or evaluation.get("selection_sha256") != selection_hash):
        raise ValueError("Invalid Boneseed evaluation selection/parent/hash lineage")
    for key in ("joint_names", "state_layout", "dtype", "control_dt", "source_fps", "lineage"):
        if key not in full or evaluation.get(key) != full[key]:
            raise ValueError(f"Evaluation must preserve the full library metadata: {key}")
    full_clips, eval_clips = full.get("clips"), evaluation.get("clips")
    if not isinstance(full_clips, list) or not full_clips or not isinstance(eval_clips, list):
        raise ValueError("Invalid full/evaluation clip lists")
    full_names = [clip["name"] for clip in full_clips]
    names = [clip["name"] for clip in eval_clips]
    if (len(set(full_names)) != len(full_names) or len(names) != expected_count
            or len(set(names)) != expected_count or selection.get("names") != names
            or selection.get("population_clips") != len(full_names)
            or type(selection.get("seed")) is not int or selection["seed"] != 0
            or len(full_names) < expected_count):
        raise ValueError("Evaluation requires the fixed distinct complete clip selection")
    sorted_names = sorted(full_names)
    sampled = random.Random(0).sample(range(len(sorted_names)), expected_count)
    if selection.get("sampled_sorted_indices") != sampled or names != [sorted_names[i] for i in sampled]:
        raise ValueError("Evaluation selection differs from the frozen seed-zero sample")
    by_name, full_offset = {}, 0
    for index, clip in enumerate(full_clips):
        length = clip.get("frames")
        if type(length) is not int or length < 2 or clip.get("offset") != full_offset:
            raise ValueError("Training manifest has invalid clip lengths/offsets")
        by_name[clip["name"]] = (index, clip)
        full_offset += length
    if full.get("frames") != full_offset:
        raise ValueError("Training manifest frame count disagrees with its clips")
    eval_offset = 0
    for clip in eval_clips:
        parent_index, parent = by_name[clip["name"]]
        if (clip.get("offset") != eval_offset or clip.get("parent_clip_index") != parent_index
                or clip.get("parent_offset") != parent["offset"]):
            raise ValueError("Evaluation clip offset/parent identity mismatch")
        for key, value in parent.items():
            if key != "offset" and (key not in clip or clip[key] != value):
                raise ValueError(f"Evaluation changed complete training clip metadata: {key}")
        if (clip.get("fps") != 50 or not math.isclose(clip.get("duration", -1), (clip["frames"]-1)/50,
                                                      rel_tol=1e-12, abs_tol=1e-10)):
            raise ValueError("Evaluation clip timing must remain 50Hz")
        eval_offset += clip["frames"]
    if evaluation.get("frames") != eval_offset:
        raise ValueError("Evaluation manifest frame count disagrees with complete clips")
    return {"eval_manifest_sha256": eval_hash, "selection_sha256": selection_hash,
            "parent_manifest_sha256": full_hash, "names": names}


def evaluate_native(args):
    """GPU-only path. Setup matches the frozen script; weights never update."""
    import torch
    import supermimic
    source = args.source_root.resolve()
    validate_source_package(supermimic, source)
    from supermimic.privileged import (PRIVILEGED_DIM, PrivilegedFeatures, RobotKinematics38,
                                      checkpoint_contract, validate_checkpoint)
    # Validate the saved training identity on CPU before creating a simulator or
    # loading any reference onto a GPU. The full BONES state library stays shut.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    gmt_contract = json.loads((args.run_dir / "gmt_contract.json").read_text(encoding="utf-8"))
    validate_checkpoint(checkpoint, gmt_contract)
    training_args = json.loads((args.run_dir / "args.json").read_text(encoding="utf-8"))
    dataset = training_args.get("dataset")
    expected_count = getattr(args, "expected_clips", 40)
    iteration = checkpoint_iteration(args.checkpoint)
    validate_checkpoint_identity(checkpoint, iteration, gmt_contract, dataset=dataset)
    for name in ("dataset", "motion_dir", "goal_frames", "hist", "amp"):
        if checkpoint["args"].get(name) != training_args.get(name):
            raise ValueError("Run/checkpoint evaluation arguments disagree")
    eval_directory = getattr(args, "eval_motion_dir", None)
    evaluation_identity = {}
    if dataset == "boneseed":
        if eval_directory is None or expected_count != 512:
            raise ValueError("Boneseed evaluation requires the explicit fixed 512-clip library")
        evaluation_identity = validate_boneseed_eval_library(
            training_args["motion_dir"], eval_directory, gmt_contract, expected_count)
        motion_directory = eval_directory
    elif dataset == "lafan1":
        if eval_directory is not None or expected_count != 40:
            raise ValueError("LAFAN1 evaluation requires the original complete 40-clip library")
        motion_directory = training_args["motion_dir"]
    else:
        raise ValueError("Unsupported evaluation dataset")
    # A frozen evaluation must reuse a completed USD conversion. Fail instead
    # of allowing robot.py's import to initiate a new conversion in the source.
    from supermimic import g1_usd
    from supermimic.g1_physics import ASSET_DIR
    identity = g1_usd._source_identity(ASSET_DIR)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    g1_usd._read_receipt(ASSET_DIR / "usd" / key, identity)

    from isaaclab.app import launch_simulation
    import supermimic.networks as networks
    from supermimic.env import TrackingSim, make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, clamp_action
    from supermimic.motion import load_reference
    from supermimic.networks import Policy
    from supermimic.robot import default_joint_pos_list

    if not torch.cuda.is_available():
        raise RuntimeError("Native evaluation requires an allocated CUDA GPU")
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    kin = G1Kinematics(dev)
    motion = load_reference(dataset, motion_directory, kin, dev)
    if len(motion.names) != expected_count:
        raise ValueError("The reference library clip count differs from expected-clips")
    teacher_kin = RobotKinematics38(device=dev)
    privileged = PrivilegedFeatures(teacher_kin, motion)
    eval_contract = checkpoint_contract(motion, teacher_kin)
    if dataset == "boneseed":
        from supermimic.gmt_transfer import validate_contract_transfer
        if (eval_contract.get("dataset") != "boneseed"
                or eval_contract.get("motion_manifest_sha256") != evaluation_identity["eval_manifest_sha256"]
                or list(motion.names) != evaluation_identity["names"]):
            raise ValueError("Loaded eval library changed after lineage validation")
        validate_contract_transfer(checkpoint["gmt_contract"], eval_contract)
    else:
        validate_checkpoint(checkpoint, eval_contract)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    policies = []
    class GuardedPolicy(Policy):
        def load_state_dict(self, *a, **kw):
            result = super().load_state_dict(*a, **kw)
            self.evaluation_signature = state_signature(self)
            policies.append(self)
            return result
    namespace = dict(torch=torch, networks=networks, TrackingSim=TrackingSim, dev=dev,
                     Policy=GuardedPolicy, OBS_DIM=OBS_DIM, PRIVILEGED_DIM=PRIVILEGED_DIM,
                     validate_checkpoint=validate_checkpoint, gmt_contract=gmt_contract,
                     motion=motion, C=expected_count, mdp=mdp, MAX_STEP_REWARD=MAX_STEP_REWARD,
                     privileged=privileged, clamp_action=clamp_action)
    evaluate = load_evaluate_function(source / "supermimic/evaluate.py", namespace)
    sim_cfg = make_sim(dev, args)
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(sim_cfg, args.eval_envs, motion, mdp, eval_mode=True)
        metrics = evaluate(env, checkpoint)
    validate_metrics(metrics, motion.names, expected_count=expected_count)
    if not policies or any(state_signature(p) != p.evaluation_signature for p in policies):
        raise RuntimeError("Evaluation mutated policy parameters or normalizer buffers")
    if any(module.training for policy in policies for module in policy.modules()):
        raise RuntimeError("Policy was not in evaluation mode")
    if any(parameter.grad is not None for policy in policies for parameter in policy.parameters()):
        raise RuntimeError("Evaluation populated a parameter gradient")
    wall = checkpoint.get("wall_time")
    if type(wall) not in (float, int) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Checkpoint has invalid training wall time")
    if not isinstance(checkpoint.get("saved_at"), str):
        raise ValueError("Checkpoint has no save timestamp")
    return {
        "checkpoint_iteration": iteration, "norm_return": metrics["norm_return"],
        "norm_return_framew": metrics["norm_return_framew"], "n_failed": metrics["n_failed"],
        "n_total": expected_count, "failed": metrics["failed"], "metrics": metrics,
        "train_wall_h": wall/3600, "saved_at": checkpoint["saved_at"],
        "eval_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": contract_digest(gmt_contract),
        "eval_contract_sha256": contract_digest(eval_contract),
        "eval_manifest_sha256": motion.manifest_sha256,
        "selection_sha256": evaluation_identity.get("selection_sha256"),
        "evaluation_selection": "eval_training_subset" if dataset == "boneseed" else "complete_library",
        "held_out": False,
        "scope": {"training_updates": 0, "optimizer_steps": 0, "normalizer_updates": 0,
                  "nominal": True, "observation_noise": False, "action_noise": False, "domain_randomization": False},
        "args_identity": {name: checkpoint["args"].get(name) for name in
                          ("dataset", "goal_frames", "hist", "amp", "iterations")},
        "eval_envs": args.eval_envs,
        "cuda_device": {"name": torch.cuda.get_device_name(dev),
                        "capability": list(torch.cuda.get_device_capability(dev)),
                        "torch_version": str(torch.__version__)},
    }


def run_worker(args, native=evaluate_native):
    """Publish success only after evaluation and immutable checkpoint checks."""
    phase, before, failure_path = "validation", None, None
    try:
        args.source_root = Path(args.source_root).resolve()
        args.run_dir = Path(args.run_dir).resolve()
        args.checkpoint = Path(args.checkpoint).resolve()
        args.output = Path(args.output).resolve()
        if args.output.is_relative_to(args.source_root):
            raise ValueError("Evaluation output must be outside the frozen source")
        failure_path = Path(str(args.output)+".error.json")
        if args.output.exists():
            raise FileExistsError("Evaluation success receipt already exists")
        expected_count = getattr(args, "expected_clips", 40)
        if type(expected_count) is not int or expected_count < 1 or not 1 <= args.eval_envs <= expected_count:
            raise ValueError("eval-envs must be in 1..expected-clips; every selected clip is evaluated")
        iteration = checkpoint_iteration(args.checkpoint)
        if args.checkpoint.parent != args.run_dir:
            raise ValueError("Checkpoint must belong to the selected training run")
        before = sha256(args.checkpoint)
        phase = "native_evaluation"
        result = native(args)
        phase = "immutability"
        if sha256(args.checkpoint) != before:
            raise ValueError("Checkpoint changed during evaluation")
        if result.get("checkpoint_iteration") != iteration:
            raise ValueError("Evaluation returned a different checkpoint iteration")
        result.update(passed=True, checkpoint_sha256=before, checkpoint_path=str(args.checkpoint))
        phase = "publication"
        publish_json(args.output, result)
        print(json.dumps({"passed": True, "checkpoint_iteration": iteration, "output": str(args.output)}), flush=True)
        return 0
    except Exception as error:
        failure = {"passed": False, "phase": phase, "error_type": type(error).__name__,
                   "checkpoint_sha256": before,
                   "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                              for f in traceback.extract_tb(error.__traceback__)]}
        # Do not echo exception text, source lines, environment or credentials.
        if failure_path is not None:
            try:
                publish_json(failure_path, failure)
            except OSError:
                pass
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-envs", type=int, default=40)
    parser.add_argument("--eval-motion-dir", type=Path, help="fixed small Boneseed evaluation subset")
    parser.add_argument("--expected-clips", type=int, default=40)
    known, _ = parser.parse_known_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(known.source_root.resolve()))
    try:
        from isaaclab.app import add_launcher_args
        add_launcher_args(parser)
    except Exception as error:
        def bootstrap_failure(_):
            raise error
        return run_worker(known, native=bootstrap_failure)
    return run_worker(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
