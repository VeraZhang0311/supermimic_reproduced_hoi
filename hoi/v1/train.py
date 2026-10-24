"""Joint HOI training entry: real workers, joint WM, gated actor BPTT.

Use --preflight to inspect a concrete bundle/config without starting workers
or optimization. All runs, including continuation, require fresh output paths.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from .config import TrainConfig, bundle_identity, fresh_directory, write_json, source_identity
from .gates import WMThresholdProposal, require_native_evidence, validate_native_evidence, validate_wm_evidence
from .variants import geometry_input_enabled, reward_config, max_step_reward


def read_checkpoint(path):
    raw = Path(path).read_bytes()
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False), hashlib.sha256(raw).hexdigest()


def read_config(path=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    unknown = set(payload) - {field.name for field in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown training configuration: {sorted(unknown)}")
    config = TrainConfig(**payload).validate()
    # Changes to the reproduction's numerical training profile are explicit
    # smoke experiments, never silently labelled the default formal run.
    defaults = TrainConfig().to_dict()
    protected = set(defaults) - {"iterations", "seed", "ckpt_every", "eval_every", "log_every", "gate_every", "experiment_kind", "variant"}
    changed = {key: {"default": defaults[key], "actual": getattr(config, key)}
               for key in protected if getattr(config, key) != defaults[key]}
    if changed and config.experiment_kind != "smoke":
        raise ValueError("nondefault training settings require experiment_kind='smoke': " + str(sorted(changed)))
    return config, changed


def collect(learner, workers, steps):
    if not hasattr(learner, "sampling_bin_counts"):
        learner.sampling_bin_counts = torch.zeros(learner.motion.num_bins, dtype=torch.long, device=learner.device)
    values = []
    for _ in range(steps):
        snapshot = workers.snapshot()
        reset_frames = snapshot["f"][snapshot["ep_step"] == 0]
        learner.sampling_bin_counts += torch.bincount(learner.motion.frame_bin[reset_frames],
                                                     minlength=learner.motion.num_bins)
        action = learner.action(snapshot)
        transition = workers.step(action)
        learner.observe(transition)
        values.append({"human_reward_normalized": float(transition["r_human"].mean()) / .1,
                       "hoi_reward_normalized": float(transition["r"].mean()) / .15,
                       "objective_reward_normalized": float(transition["r"].mean()) / max_step_reward(learner.config.variant),
                       "terminations": int(transition["term"].sum()),
                       "actual_pair_positive": transition["contact"].sum(0).tolist()})
    return values


def train_loop(learner, workers, output, *, physics, data_identity, initialization, native_gate,
               evaluation_callback=None, tracker=None):
    """Run a fresh replay prefill even after model/optimizer continuation."""
    from .checkpoint import save_checkpoint
    config, output = learner.config, Path(output)
    if native_gate.get("passed") is not True:
        raise ValueError("native evidence must pass before collecting training replay")
    started = time.monotonic()
    source = source_identity()
    write_json(output / "source.json", source)
    collect(learner, workers, config.warmup_steps)
    write_json(output / "prefill.json", {"vector_steps": config.warmup_steps, "sim_steps": learner.sim_steps,
                                        "fresh_replay": True, "bitwise_continuation": False})
    accumulator = 0.
    policy_updates = wm_updates = 0
    evaluation_dispatches = []
    gate = {"passed": False, "blockers": ["WM validation has not run"]}
    proposal = WMThresholdProposal()
    last_checkpoint = None
    first_iteration = learner.iteration
    with (output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as metrics:
        for iteration in range(learner.iteration, config.iterations):
            lr = learner.set_lr(iteration)
            accumulator += 1 / config.utd
            count = int(accumulator)
            accumulator -= count
            real = collect(learner, workers, count)
            losses = []
            for _ in range(config.wm_batches):
                try:
                    losses.append(learner.wm_update())
                    wm_updates += 1
                except ValueError as exc:
                    # A sparse initial replay can lack reserved/fit windows.
                    # Collect more real data; do not substitute reference data.
                    if "replay coverage" not in str(exc) and "not enough replay" not in str(exc):
                        raise
                    break
            if iteration % config.gate_every == 0 or iteration + 1 == config.iterations:
                try:
                    validation = learner.wm_validation()
                    gate = validate_wm_evidence(validation, proposal)
                except ValueError as exc:
                    if "replay coverage" not in str(exc) and "not enough replay" not in str(exc):
                        raise
                    validation = {"status": "insufficient_real_holdout_coverage", "reason": str(exc)}
                    gate = {"passed": False, "blockers": [str(exc)], "proposal": asdict(proposal)}
                write_json(output / f"wm_gate_{iteration + 1:06d}.json", {"validation": validation, "gate": gate})
            actor = None
            if iteration >= config.wm_only_iters and gate["passed"]:
                actor = learner.policy_update()
                policy_updates += 1
            learner.iteration = iteration + 1
            # One early observation establishes finite updates after prefill,
            # including on continuation, without changing the logging cadence.
            if iteration == first_iteration or learner.iteration % config.log_every == 0 or learner.iteration == config.iterations:
                record = {"iteration": learner.iteration, "sim_steps": learner.sim_steps, "lr": lr,
                          "elapsed_seconds": time.monotonic() - started, "real": real,
                          "wm": losses, "actor": actor, "policy_gate": gate,
                          "wm_updates": wm_updates, "policy_updates": policy_updates,
                          "fit_contact_positive": learner.contact_positive.tolist(),
                          "fit_contact_negative": learner.contact_negative.tolist()}
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps({key: record[key] for key in ("iteration", "sim_steps", "wm_updates", "policy_updates")}), flush=True)
                if tracker is not None:
                    tracker.log_record(record)
            if (learner.iteration % config.ckpt_every == 0 or learner.iteration % config.eval_every == 0 or
                    learner.iteration == config.iterations):
                last_checkpoint = output / f"checkpoint_{learner.iteration:06d}.pt"
                save_checkpoint(last_checkpoint, learner.pi, learner.wm, learner.critic, learner.optimizers,
                                iteration=learner.iteration, physics=physics, data_identity=data_identity,
                                initialization=initialization,
                                extra={"config": config.to_dict(), "sim_steps": learner.sim_steps,
                                       "wm_updates_this_run": wm_updates, "policy_updates_this_run": policy_updates,
                                       "update_count_scope": "this output run since its fresh real replay prefill; excludes prior runs",
                                       "contact_positive": learner.contact_positive.cpu(),
                                       "contact_negative": learner.contact_negative.cpu(),
                                       "sampling_bin_counts": learner.sampling_bin_counts.cpu(),
                                       "source_identity": source,
                                       "native_gate": native_gate, "wm_gate": gate})
                if tracker is not None:
                    tracker.checkpoint(learner.iteration, wm_updates, policy_updates)
                if learner.iteration % config.eval_every == 0:
                    if evaluation_callback is None:
                        raise RuntimeError("periodic frozen HOI80 evaluation is due but no dispatcher is configured")
                    evaluation_dispatches.append(evaluation_callback(last_checkpoint, learner.iteration))
    summary = {"iteration": learner.iteration, "sim_steps": learner.sim_steps,
               "wm_updates": wm_updates, "policy_updates": policy_updates, "policy_gate": gate,
               "update_count_scope": "this output run since its fresh real replay prefill; excludes prior runs",
               "checkpoint": str(last_checkpoint), "experiment_kind": config.experiment_kind,
               "elapsed_seconds": time.monotonic() - started, "training_completed": True,
               "evaluation_dispatches": evaluation_dispatches,
               "convergence_assessed": False}
    write_json(output / "completion.json", summary)
    write_json(output / "sampling.json", {"definition": "actual episode-reset frames counted by compatible reference bin",
                                         "bin_counts": learner.sampling_bin_counts.tolist(),
                                         "bin_start": learner.motion.bin_start.tolist(),
                                         "bin_clip": learner.motion.bin_clip.tolist(),
                                         "bin_asset": learner.motion.bin_asset.tolist(),
                                         "clip_names": learner.motion.names})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-evidence", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-mjcf")
    parser.add_argument("--max-processes", type=int)
    parser.add_argument("--eval-bundle", type=Path, help="complete canonical HOI80 bundle for every 5000-iteration frozen evaluation")
    parser.add_argument("--evaluation-runtime", help="read-only runtime for independent evaluation Slurm jobs")
    parser.add_argument("--evaluation-isaaclab", help="read-only IsaacLab path for evaluation")
    parser.add_argument("--evaluation-launcher", type=Path, default=Path(__file__).resolve().parents[2] / "cluster/hoi_v1_evaluate.sbatch.sh")
    parser.add_argument("--preflight", action="store_true")
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--init-checkpoint", type=Path)
    init.add_argument("--resume", type=Path)
    init.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--legacy-hand-collision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy-self-collision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-initialization-physics-change", action="store_true")
    from supermimic.physics import add_physics_arguments, resolve_arguments
    add_physics_arguments(parser)
    args, launcher = parser.parse_known_args(argv)
    config, deviations = read_config(args.config)
    checkpoint, checkpoint_sha = read_checkpoint(args.resume) if args.resume else (None, None)
    settings, _ = resolve_arguments(args, checkpoint.get("physics") if checkpoint else None)
    physics = asdict(settings)
    if not (settings.hand_collision and settings.self_collision):
        parser.error("v1 design requires hand ON and self ON")
    if not args.preflight and not any((args.init_checkpoint, args.resume, args.from_scratch)):
        parser.error("choose explicit --init-checkpoint, --resume or --from-scratch")
    from .evaluate import _read_bundle_header
    bundle = _read_bundle_header(args.bundle)
    if bundle["partition"] != "train":
        raise ValueError("training may read only the frozen train partition")
    if bundle["scope"]["is_subset"] and config.experiment_kind != "smoke":
        raise ValueError("subset bundle requires an explicitly labelled smoke configuration")
    evaluation_header = None
    if args.eval_bundle:
        from .periodic_eval import validate_canonical80
        evaluation_header = _read_bundle_header(args.eval_bundle)
        validate_canonical80(bundle, evaluation_header)
    elif config.iterations >= config.eval_every:
        raise ValueError("--eval-bundle is required for automatic full HOI80 evaluation")
    required_assets = sorted({entry["asset_id"] for entry in bundle["clips"]})
    native = validate_native_evidence(args.native_evidence, expected_physics=physics,
                                      required_asset_ids=required_assets,
                                      expected_bundle_sha256=bundle["bundle_content_sha256"]) if args.native_evidence else {
                                          "passed": False, "blockers": ["--native-evidence is required"]}
    output = fresh_directory(args.output)
    identity = bundle_identity(bundle)
    plan = {"config": config.to_dict(), "smoke_deviations": deviations, "physics": physics,
            "data_identity": identity, "bundle": str(args.bundle.resolve()), "scope": bundle["scope"],
            "native_gate": native, "wm_gate_proposal": asdict(WMThresholdProposal()),
            "source_identity": source_identity(),
            "initialization_request": {"init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                                       "resume": str(args.resume) if args.resume else None,
                                       "from_scratch": args.from_scratch,
                                       "legacy_hand_collision": args.legacy_hand_collision,
                                       "legacy_self_collision": args.legacy_self_collision,
                                       "allow_physics_change": args.allow_initialization_physics_change},
            "launcher_args": launcher, "training_started": False}
    plan["periodic_evaluation"] = {"every_iterations": config.eval_every, "checkpoint_every": config.ckpt_every,
                                   "bundle": str(args.eval_bundle.resolve()) if args.eval_bundle else None,
                                   "bundle_content_sha256": evaluation_header["bundle_content_sha256"] if evaluation_header else None,
                                   "case_count": 80, "frozen": True, "independent_gpu_job": True}
    import os
    from .track import destination, _base_url
    tracking_enabled = os.environ.get("WANDB_SYNC", "0")
    if tracking_enabled not in ("0", "1"):
        raise ValueError("WANDB_SYNC must be 0 or 1")
    tracking_entity, tracking_project = destination()
    plan["tracking"] = {"enabled": tracking_enabled == "1", "entity": tracking_entity, "project": tracking_project,
                        "base_url": _base_url(), "resume": "never", "mode": "live_training" if tracking_enabled == "1" else "disabled",
                        "source_weight_raw_rollout_uploads": False}
    from .buffer import FIELDS
    import math
    per_transition = sum(math.prod((shape,) if isinstance(shape, int) else shape) * dtype.itemsize
                         for shape, dtype in FIELDS.values())
    plan["resource_proposal"] = {"required_static_groups": len(required_assets), "num_envs": config.num_envs,
                                 "replay_bytes": per_transition * config.buffer_steps * config.num_envs,
                                 "full_gpu_memory_and_throughput_validated": False,
                                 "cache_and_source_bytes_rechecked_on_load_bundle": True}
    write_json(output / "plan.json", plan)
    if args.preflight:
        print(json.dumps(plan, allow_nan=False))
        return plan
    require_native_evidence(args.native_evidence, expected_physics=physics,
                   required_asset_ids=required_assets,
                   expected_bundle_sha256=bundle["bundle_content_sha256"])
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("native HOI training requires the prepared Linux GPU runtime")
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import supermimic.networks as legacy_networks
    legacy_networks.AMP_DTYPE = torch.bfloat16 if config.amp == "bf16" else None
    from supermimic.kinematics import G1Kinematics
    from supermimic.robot_metadata import default_joint_pos_list
    from .prepare import load_bundle
    from .mdp import HOIMDP
    from .networks import Policy, WorldModel, Critic
    from .buffer import ReplayBuffer
    from .learner import Learner
    from .checkpoint import migrate_legacy, load_checkpoint
    from .workers import allocate_workers, resource_preflight, WorkerGroup
    kin = G1Kinematics(args.device, mjcf_path=args.robot_mjcf)
    motion, shapes, bundle = load_bundle(args.bundle, kin, device=args.device)
    mdp = HOIMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), shapes,
                 cfg=reward_config(config.variant))
    models = [model(geometry_input_enabled=geometry_input_enabled(config.variant)).to(args.device)
              for model in (Policy, WorldModel, Critic)]
    replay = ReplayBuffer(config.buffer_steps, config.num_envs, args.device)
    learner = Learner(*models, motion, mdp, shapes, replay, config)
    initialization = {"mode": "explicit_from_scratch"}
    if args.init_checkpoint:
        old, init_sha = read_checkpoint(args.init_checkpoint)
        declared = None
        if args.legacy_hand_collision is not None or args.legacy_self_collision is not None:
            declared = {"hand_collision": args.legacy_hand_collision, "self_collision": args.legacy_self_collision}
        initialization = migrate_legacy(old, *models, physics=physics, legacy_physics=declared,
                                        allow_physics_change=args.allow_initialization_physics_change)
        initialization.update(path=str(args.init_checkpoint.resolve()), sha256=init_sha)
        del old
    elif checkpoint:
        load_checkpoint(checkpoint, *models, learner.optimizers, physics=physics, data_identity=identity,
                        variant=config.variant)
        if checkpoint["extra"]["config"] != config.to_dict():
            previous = checkpoint["extra"]["config"]
            allowed = {"iterations", "log_every", "gate_every", "ckpt_every", "eval_every"}
            if any(previous[k] != v for k, v in config.to_dict().items() if k not in allowed):
                raise ValueError("continuation training profile differs")
        learner.iteration = checkpoint["iteration"]
        if config.iterations <= learner.iteration:
            raise ValueError("continuation target iterations must exceed checkpoint iteration")
        learner.sim_steps = checkpoint["extra"]["sim_steps"]
        for field in ("contact_positive", "contact_negative"):
            getattr(learner, field).copy_(checkpoint["extra"][field].to(args.device))
        if "sampling_bin_counts" in checkpoint["extra"]:
            learner.sampling_bin_counts = checkpoint["extra"]["sampling_bin_counts"].to(args.device)
        initialization = {"mode": "same_schema_continuation_fresh_replay", "path": str(args.resume.resolve()),
                          "sha256": checkpoint_sha, "origin": checkpoint["initialization"]}
    write_json(output / "initialization.json", initialization)
    specs = allocate_workers(bundle, motion, config.num_envs)
    resources = resource_preflight(specs, device=args.device, max_processes=args.max_processes)
    resources["replay_bytes"] = sum(t.numel() * t.element_size() for t in replay.data.values())
    write_json(output / "resources.json", resources)
    worker_config = {"bundle": str(args.bundle.resolve()), "output": str(output / "workers"), "physics": physics,
                     "variant": config.variant,
                     "robot_mjcf": args.robot_mjcf,
                     "seed": config.seed,
                     "device": args.device, "gamma": config.gamma, "launcher_args": launcher,
                     "max_processes": args.max_processes}
    evaluation_callback = None
    if args.eval_bundle:
        import os
        from .periodic_eval import schedule_evaluation
        environment = {"SUPERMIMIC_ROOT": str(Path(__file__).resolve().parents[2]),
                       "SUPERMIMIC_RUNTIME": args.evaluation_runtime or os.environ.get("SUPERMIMIC_RUNTIME", ""),
                       "ISAACLAB_PATH": args.evaluation_isaaclab or os.environ.get("ISAACLAB_PATH", "")}
        if any(not value or not Path(value).is_dir() for value in environment.values()):
            raise ValueError("automatic evaluation requires existing read-only runtime/IsaacLab paths")
        if not args.evaluation_launcher.is_file():
            raise FileNotFoundError(args.evaluation_launcher)
        def evaluation_callback(path, iteration):
            receipt = schedule_evaluation(path, iteration, args.bundle, args.eval_bundle, output / "evaluations",
                                          args.evaluation_launcher, environment)
            if receipt["status"] != "submitted":
                raise RuntimeError("periodic HOI80 submission was not confirmed; checkpoint and dispatch claim preserved: " +
                                   receipt["status"])
            return receipt
    from .track import live_training
    # Native children start before tracking's process-local SDK/auth context so
    # they never inherit credentials materialized by the tracking bootstrap.
    workers = WorkerGroup(specs, worker_config, device=args.device)
    try:
        write_json(output / "native_workers.json", workers.native_reports)
        with live_training(plan, output / "tracking") as tracker:
            return train_loop(learner, workers, output, physics=physics, data_identity=identity,
                              initialization=initialization, native_gate=native, evaluation_callback=evaluation_callback,
                              tracker=tracker)
    finally:
        workers.close()


if __name__ == "__main__":
    main()
