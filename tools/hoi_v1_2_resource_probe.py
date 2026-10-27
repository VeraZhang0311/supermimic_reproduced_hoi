"""Real production-size v1.2 GPU validation with disposable Adam updates.

This uses fresh models, real native replay, full batch4096/H32, the unchanged
production config, and actual optimizer kernels/state. It never saves models;
the independent formal run starts from scratch in a new process.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("SUPERMIMIC_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

import torch

from hoi.v1.config import TrainConfig, bundle_identity, fresh_directory, source_identity, write_json
from hoi.v1.contacts import CHANNELS, LABEL_SCHEMA
from hoi.v1.resource_probe import (
    _GpuSampler, full_active_wm_windows, initialize_cuda_memory_counters,
    summarize_gpu_samples, validate_real_start_mix,
)

ALLOCATOR_FLAG = "expandable_segments:True"
ALLOCATOR_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
MEMORY_STAT_KEYS = (
    "allocated_bytes.all.current", "allocated_bytes.all.peak",
    "reserved_bytes.all.current", "reserved_bytes.all.peak",
    "active_bytes.all.current", "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak", "num_alloc_retries", "num_ooms",
)


def parameter_digest(model):
    """Exclude mutable normalization buffers; no graph/tensor is retained."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def allocator_summary():
    flags = {key: os.environ.get(key) for key in ALLOCATOR_KEYS}
    assert all(value == ALLOCATOR_FLAG for value in flags.values()), flags
    snapshot = torch.cuda.memory_snapshot()
    expandable = [segment for segment in snapshot if segment.get("is_expandable") is True]
    return {"environment": flags, "backend": torch.cuda.memory.get_allocator_backend(),
            "snapshot_segment_count": len(snapshot), "expandable_segment_count": len(expandable),
            "expandable_segment_total_bytes": sum(int(segment["total_size"]) for segment in expandable)}


def memory_summary():
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    return {"parent_allocated_bytes": int(torch.cuda.memory_allocated()),
            "parent_reserved_bytes": int(torch.cuda.memory_reserved()),
            "parent_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "parent_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "all_process_gpu_used_bytes": int(total - free), "gpu_total_bytes": int(total),
            "allocator_statistics": {key: int(stats[key]) for key in MEMORY_STAT_KEYS if key in stats}}


def finite_tree(value):
    """Reject nonfinite diagnostic numbers without conflating quality failure."""
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def gradient_summary(model):
    count, nonzero, squared = 0, 0, 0.
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        assert bool(torch.isfinite(gradient).all()), "nonfinite diagnostic gradient"
        count += 1
        nonzero += int(bool((gradient != 0).any()))
        squared += float(gradient.float().square().sum())
    assert count > 0, "missing backward gradients"
    return {"finite": True, "gradient_tensor_count": count,
            "nonzero_gradient_tensor_count": nonzero, "clipped_gradient_l2": math.sqrt(squared)}


def fresh_state(learner, models):
    from supermimic.networks import RunningNorm
    norms = {}
    for model_name, model in models.items():
        for name, module in model.named_modules():
            if isinstance(module, RunningNorm):
                assert float(module.count) == 0
                assert bool((module.mean == 0).all()) and bool((module.var == 1).all())
                norms[model_name + "/" + name] = {"count": 0, "mean_all_zero": True, "variance_all_one": True}
    assert len(norms) == 10, norms
    assert all(not optimizer.state for optimizer in learner.optimizers.values())
    assert learner.iteration == 0 and learner.sim_steps == 0
    assert learner.contact_positive.shape == learner.contact_negative.shape == (5,)
    assert bool((learner.contact_positive == 0).all()) and bool((learner.contact_negative == 0).all())
    assert learner.buffer.size == 0 and learner.buffer.ptr == 0
    return {"mode": "explicit_from_scratch", "checkpoint_read": False,
            "iteration": 0, "sim_steps": 0, "replay_size": 0, "normalizers": norms,
            "adam_state_entries": {name: len(opt.state) for name, opt in learner.optimizers.items()},
            "contact_positive": learner.contact_positive.tolist(), "contact_negative": learner.contact_negative.tolist()}


def validate_environment():
    assert os.environ.get("WANDB_SYNC") == "0", "capacity diagnostic requires WANDB_SYNC=0"
    assert all(os.environ.get(key) == ALLOCATOR_FLAG for key in ALLOCATOR_KEYS)
    # Check names only; never serialize inherited credentials or their values.
    forbidden = ("INIT_CHECKPOINT", "RESUME_CHECKPOINT", "RESUME", "WANDB_API_KEY", "WANDB_ACCESS_TOKEN",
                 "WANDB_IDENTITY_TOKEN", "WANDB_IDENTITY_TOKEN_FILE", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
    assert not any(os.environ.get(key) for key in forbidden), "initialization/resume/credential environment is forbidden"
    for key in ("SUPERMIMIC_ROOT", "BUNDLE", "NATIVE_EVIDENCE", "RUN_DIR", "CONFIG"):
        assert os.environ.get(key), "missing required capacity diagnostic path: " + key
    assert Path(os.environ["SUPERMIMIC_ROOT"]).resolve() == ROOT
    assert Path(__file__).resolve().parent.parent == ROOT, "probe must reside under the isolated source root"
    assert CHANNELS == ("left_hand_object", "right_hand_object", "object_ground", "left_foot_object", "right_foot_object")
    assert LABEL_SCHEMA == "hoi_v1_pair_normal_impulse_v2"


def validate_production_config(config):
    from hoi.v1.actor_fusion import FUSION_ACTOR
    from hoi.v1.observations import MATCHING
    from hoi.v1.randomization import MATCHING_DR
    expected = {"num_envs": 4096, "buffer_steps": 1000, "pi_batch": 4096,
                "horizon": 32, "wm_batch": 2048, "wm_window": 8,
                "wm_batches": 10, "warmup_steps": 64, "critic_epochs": 5,
                "critic_minibatches": 4, "iterations": 20000, "wm_only_iters": 500,
                "actor_architecture": FUSION_ACTOR, "observation_profile": MATCHING,
                "domain_randomization_profile": MATCHING_DR}
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f"GPU validation preserves {key}={value}")


def fusion_gradient_summary(policy, *, require_live):
    result = {"fusion": gradient_summary(policy.fusion),
              "human_encoder": gradient_summary(policy.enc),
              "object_encoder": gradient_summary(policy.object_enc),
              "shape_encoder": gradient_summary(policy.shape_encoder),
              "decoder": gradient_summary(policy.dec),
              "projections": {name: gradient_summary(module)
                              for name, module in policy.fusion.projections.items()}}
    if require_live:
        for name in ("fusion", "human_encoder", "object_encoder", "shape_encoder", "decoder"):
            assert result[name]["clipped_gradient_l2"] > 0, name
        for name, summary in result["projections"].items():
            assert summary["clipped_gradient_l2"] > 0, name
    else:
        assert result["fusion"]["nonzero_gradient_tensor_count"] == 0
    return result


def normalization_digest(*models):
    from supermimic.networks import RunningNorm
    digest = hashlib.sha256()
    for model in models:
        for name, module in model.named_modules():
            if isinstance(module, RunningNorm):
                for field in ("mean", "var", "count"):
                    digest.update((name + "/" + field).encode())
                    digest.update(getattr(module, field).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    validate_environment()
    sys.path.insert(0, str(ROOT / "tools"))
    from hoi_v1_2_launch_train import verify_stage
    root, stage = verify_stage()
    from supermimic.physics import PhysicsSettings
    from hoi.v1.actor_fusion import FUSION_ACTOR, actor_architecture_contract
    from hoi.v1.evaluate import _read_bundle_header, _validate_contact_contract
    from hoi.v1.gates import require_native_evidence, validate_wm_evidence
    from hoi.v1.randomization import validate_dr_evidence
    from hoi.v1.networks import feature_schema
    from hoi.v1.train import read_config

    config, changes = read_config(Path(os.environ["CONFIG"]))
    assert not changes and config.to_dict() == stage["production_config"]
    validate_production_config(config)
    bundle_path = Path(os.environ["BUNDLE"]).resolve()
    header = _read_bundle_header(bundle_path)
    assert header["partition"] == "train" and header["scope"]["is_subset"] is False
    clips = header["clips"]
    assert len(clips) == 2651 and len({clip["source_group"] for clip in clips}) == 381
    assert sum(clip["state_frames"] for clip in clips) == 756524
    assert sum(clip["state_frames"] - 1 for clip in clips) == 753873
    assert len(header["assets"]) == 8 and {clip["asset_id"] for clip in clips} == set(range(8))
    physics = asdict(PhysicsSettings().export())
    native = require_native_evidence(Path(os.environ["NATIVE_EVIDENCE"]), expected_physics=physics,
        required_asset_ids=range(8), expected_bundle_sha256=header["bundle_content_sha256"])
    assert validate_dr_evidence(Path(os.environ["NATIVE_EVIDENCE"]), config.domain_randomization_profile)["passed"]
    output = fresh_directory(os.environ["RUN_DIR"])
    frozen_source = source_identity()
    report = {"schema": "hoi_v1_2_fusion_production_capacity_v1", "passed": False,
              "stage_sha256": os.environ["EXPECTED_STAGE_SHA256"], "source_identity": frozen_source,
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": config.to_dict(),
              "feature_schema": feature_schema(config.observation_profile, torso_mass_context=True,
                                                actor_architecture=config.actor_architecture),
              "actor_architecture_contract": actor_architecture_contract(config.actor_architecture, config.observation_profile),
              "contact_channels": list(CHANNELS), "contact_label_schema": LABEL_SCHEMA,
              "physics": physics, "data_identity": bundle_identity(header), "native_gate": native,
              "bundle_counts": {"clips": 2651, "source_groups": 381, "assets": 8,
                                "state_frames": 756524, "control_intervals": 753873},
              "actual_optimizer_steps": 0, "actual_optimizer_steps_by_model": {"wm": 0, "pi": 0, "critic": 0},
              "formal_training_started": False, "probe_models_discarded": True,
              "quality_thresholds_control_this_capacity_diagnostic": False,
              "native_evidence_required": True, "phases": []}
    write_json(output / "request.json", report)
    workers, sampler = None, _GpuSampler()
    started = time.monotonic()

    def phase(name, **values):
        torch.cuda.synchronize()
        record = {"name": name, "elapsed_seconds": time.monotonic() - started, **memory_summary(), **values}
        report["phases"].append(record)
        write_json(output / f"phase_{len(report['phases']):02d}_{name}.json", record)
        print(json.dumps({"phase": name, "elapsed_seconds": record["elapsed_seconds"],
                          "parent_peak_allocated_bytes": record["parent_peak_allocated_bytes"]}), flush=True)

    try:
        assert torch.cuda.is_available(), "prepared Linux CUDA/native runtime required"
        device = initialize_cuda_memory_counters("cuda:0")
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(device), "amp": config.amp}
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        import supermimic.networks as original_networks
        original_networks.AMP_DTYPE = torch.bfloat16 if config.amp == "bf16" else None
        from supermimic.kinematics import G1Kinematics
        from supermimic.robot_metadata import default_joint_pos_list
        from hoi.v1.prepare import load_bundle
        from hoi.v1.mdp import HOIMDP
        from hoi.v1.networks import Policy, WorldModel, Critic, frozen_parameters
        from hoi.v1.buffer import ReplayBuffer
        from hoi.v1.learner import Learner
        from hoi.v1.train import collect
        from hoi.v1.workers import allocate_workers, WorkerGroup, resource_preflight
        from hoi.v1.variants import reward_config

        with sampler:
            robot_mjcf = header["robot_fk"]["mjcf_path"]
            kin = G1Kinematics(device, mjcf_path=robot_mjcf)
            bank, shapes, bundle = load_bundle(bundle_path, kin, device=device)
            mdp = HOIMDP(kin, bank, torch.tensor(default_joint_pos_list(), device=device), shapes,
                         cfg=reward_config(config.variant))
            models = {"pi": Policy(observation_profile=config.observation_profile,
                                    actor_architecture=config.actor_architecture).to(device),
                      "wm": WorldModel(torso_mass_context=True).to(device),
                      "critic": Critic(torso_mass_context=True).to(device)}
            assert models["pi"].actor_architecture == FUSION_ACTOR
            assert models["pi"].dec[0].in_features == 1660
            assert sum(p.numel() for p in models["pi"].parameters()) == 11923229
            assert not bool(models["pi"].dec[0].weight[:, 1148:].any())
            replay = ReplayBuffer(config.buffer_steps, config.num_envs, device, torso_mass_context=True)
            learner = Learner(models["pi"], models["wm"], models["critic"], bank, mdp, shapes, replay, config)
            report["initialization"] = fresh_state(learner, models)
            before = {name: parameter_digest(model) for name, model in models.items()}
            report["parameter_sha256_before"] = before
            report["allocator_before"] = allocator_summary()
            assert report["allocator_before"]["backend"] == "native"
            assert report["allocator_before"]["expandable_segment_count"] > 0
            phase("fresh_models_full_replay_empty_adam",
                  replay_bytes=sum(value.numel() * value.element_size() for value in replay.data.values()),
                  adam_state_empty=True, initialization=report["initialization"])
            specs = allocate_workers(bundle, bank, config.num_envs)
            assert len(specs) == 8 and sum(spec.num_envs for spec in specs) == 4096
            report["resources"] = resource_preflight(specs, device=device, max_processes=8)
            worker_config = {"bundle": str(bundle_path), "output": str(output / "workers"),
                "physics": physics, "variant": config.variant, "robot_mjcf": robot_mjcf,
                "domain_randomization_profile": config.domain_randomization_profile,
                "seed": config.seed, "device": str(device), "gamma": config.gamma,
                "launcher_args": [], "max_processes": 8}
            workers = WorkerGroup(specs, worker_config, device=device)
            report["native_workers"] = workers.native_reports
            report["process_ids"] = [os.getpid()] + [process.pid for process in workers._processes]
            for native_worker in workers.native_reports:
                _validate_contact_contract(native_worker["contact"])
            phase("eight_native_groups_ready")
            collecting = time.monotonic()
            real = collect(learner, workers, config.warmup_steps)
            assert config.warmup_steps == 64 and replay.size == 64
            assert learner.iteration == 0 and learner.sim_steps == 64 * 4096
            assert finite_tree(real)
            assert all(len(row["actual_pair_positive"]) == 5 for row in real)
            phase("real_prefill_64", vector_steps=64, real_transitions=64 * 4096,
                  seconds=time.monotonic() - collecting,
                  fit_contact_positive=learner.contact_positive.tolist(), fit_contact_negative=learner.contact_negative.tolist(),
                  actual_positive_intervals=replay.data["contact"][:64].sum((0, 1)).tolist(),
                  actual_valid_intervals=replay.data["contact_valid"][:64].sum((0, 1)).tolist())
            del real
            validation = learner.wm_validation()
            assert validation["finite"] is True and finite_tree(validation)
            assert validation["real_executed_actions"] is True
            assert validation["scope"] == "training_reference_bins_reserved_from_wm_fit_not_eval80_or_test40"
            assert validation["contact_diagnostics"]["channels"] == list(CHANNELS)
            assert all(validation["horizon_reports"][str(h)]["valid_samples"] > 0 for h in (1, 8, 32))
            quality = validate_wm_evidence(validation)
            write_json(output / "real_holdout_diagnostics.json", {"validation": validation, "quality": quality,
                       "quality_controls_capacity_probe": False, "models_have_never_taken_optimizer_steps": True})
            phase("real_reserved_holdout_finite", quality_passed=quality["passed"], quality_is_diagnostic_only=True)

            calls = report["actual_optimizer_steps_by_model"]
            gradients = {name: [] for name in models}
            for name, optimizer in learner.optimizers.items():
                actual_step = optimizer.step
                def checked_step(*args, _name=name, _actual=actual_step, **kwargs):
                    assert not args and not kwargs, "production Adam closures/options unexpectedly changed"
                    gradients[_name].append(gradient_summary(models[_name]))
                    result = _actual()
                    calls[_name] += 1
                    report["actual_optimizer_steps"] = sum(calls.values())
                    return result
                optimizer.step = checked_step

            window, selection = full_active_wm_windows(replay, config.wm_batch, config.wm_window)
            assert window["contact"].shape == (2048, 8, 5)
            assert selection["active_rows_per_step"] == [2048] * 8
            valid_counts = (window["contact_valid"] & window["mask"].bool()[..., None]).sum((0, 1)).tolist()
            assert all(count > 0 for count in valid_counts)
            from unittest.mock import patch
            def selected_window(batch, length, *, split):
                assert (batch, length, split) == (2048, 8, "fit")
                return window
            with patch.object(replay, "sample_windows", new=selected_window):
                wm_loss = learner.wm_update()
            assert finite_tree(wm_loss)
            head = learner.wm.contact_head[-1]
            assert head.weight.shape[0] == 5 and head.weight.grad is not None
            row_grad = head.weight.grad.detach().float().norm(dim=-1).tolist()
            assert all(math.isfinite(value) and value > 0 for value in row_grad)
            phase("production_wm_2048x8_backward_and_adam", loss=wm_loss,
                  selected_real_windows=selection, valid_contact_counts=valid_counts,
                  contact_head_weight_gradient_norm_per_channel=row_grad,
                  finite_gradients=gradients["wm"][-1], actual_optimizer_steps=calls.copy(),
                  measured_wm_batches=1, production_wm_batches_per_iteration=config.wm_batches)
            del window

            dtype_observations = {"projection": set(), "attention": set(), "fusion_output": set()}
            hooks = [learner.pi.fusion.projections["robot_history"].register_forward_hook(
                        lambda module, args, value: dtype_observations["projection"].add(str(value.dtype))),
                     learner.pi.fusion.layers[0].self_attn.register_forward_hook(
                        lambda module, args, value: dtype_observations["attention"].add(str(value[0].dtype))),
                     learner.pi.fusion.register_forward_hook(
                        lambda module, args, value: dtype_observations["fusion_output"].add(str(value.dtype)))]
            try:
                for actor_update in (1, 2):
                    learner.wm.zero_grad(set_to_none=True)
                    learner.critic.zero_grad(set_to_none=True)
                    start, history = replay.sample_starts(config.pi_batch, config.hist, horizon=config.horizon, motion=bank)
                    report["real_start_mix"] = validate_real_start_mix(start, history, bank,
                        expected_assets=[spec.asset_id for spec in specs], batch_size=4096, horizon=32)
                    norms_before = normalization_digest(learner.pi, learner.wm, learner.critic)
                    frozen_before = {name: parameter_digest(models[name]) for name in ("wm", "critic")}
                    with frozen_parameters(learner.wm, learner.critic):
                        trajectory = learner.imagine(start, history)
                        assert bool(torch.isfinite(trajectory["objective"]))
                        assert all(value.shape == (4096, 5) and bool(torch.isfinite(value).all())
                                   for value in trajectory["contact"])
                        next_state = torch.cat((trajectory["states"][1][0], trajectory["states"][1][4],
                                                trajectory["objects"][1][0]), -1)
                        action_jacobian = torch.autograd.grad(next_state.sum(), trajectory["actions"][0],
                                                             retain_graph=True)[0]
                        assert bool(torch.isfinite(action_jacobian).all()) and bool(action_jacobian.abs().sum() > 0)
                        action_jacobian_l2 = float(action_jacobian.norm())
                        del next_state, action_jacobian
                        learner.optimizers["pi"].zero_grad(set_to_none=True)
                        (-trajectory["objective"]).backward()
                        actor_norm = torch.nn.utils.clip_grad_norm_(learner.pi.parameters(), 1.)
                        torch.nn.utils.clip_grad_value_(learner.pi.parameters(), .1)
                        assert bool(torch.isfinite(actor_norm))
                        fusion_gradients = fusion_gradient_summary(learner.pi, require_live=actor_update == 2)
                        assert all(parameter.grad is None for model in (learner.wm, learner.critic)
                                   for parameter in model.parameters())
                        assert normalization_digest(learner.pi, learner.wm, learner.critic) == norms_before
                        assert all(parameter_digest(models[name]) == digest for name, digest in frozen_before.items())
                        learner.optimizers["pi"].step()
                        assert bool(learner.pi.dec[0].weight[:, 1148:].any())
                        phase(f"production_actor_{actor_update}_4096x32_backward_and_adam",
                              objective=float(trajectory["objective"].detach()),
                              gradient_norm_before_clipping=float(actor_norm), fusion_gradients=fusion_gradients,
                              wm_and_critic_parameter_gradients_absent=True,
                              wm_action_to_next_state_jacobian_l2=action_jacobian_l2,
                              wm_critic_parameters_and_all_RMS_unchanged_during_actor=True,
                              finite_gradients=gradients["pi"][-1], actual_optimizer_steps=calls.copy())

                    target_rows, mini_rows, normalization_rows = [], [], []
                    original_norm = learner.critic.update_normalizers
                    def record_norm(*arguments):
                        assert all(len(value) == 135168 for value in arguments)
                        normalization_rows.append(int(len(arguments[0])))
                        return original_norm(*arguments)
                    def record_critic(module, arguments):
                        (mini_rows if torch.is_grad_enabled() else target_rows).append(int(len(arguments[0])))
                    hook = learner.critic.register_forward_pre_hook(record_critic)
                    try:
                        with patch.object(learner.critic, "update_normalizers", new=record_norm):
                            critic_loss = learner.critic_update(trajectory, start)
                    finally:
                        hook.remove()
                    assert math.isfinite(critic_loss)
                    assert normalization_rows == [135168]
                    assert target_rows == [32768, 32768, 32768, 32768, 4096]
                    assert mini_rows == [32768] * 20
                    phase(f"production_critic_{actor_update}_20_minibatch_backward_and_adam", loss=critic_loss,
                          target_forward_rows=target_rows, normalization_rows=normalization_rows,
                          training_forward_rows=mini_rows, epochs=config.critic_epochs,
                          minibatches_per_epoch=config.critic_minibatches,
                          finite_minibatch_gradients=gradients["critic"][-20:],
                          original_critic_update_called=True, actual_optimizer_steps=calls.copy(),
                          actor_trajectory_and_gradients_retained=True)
                    del trajectory, start, history
            finally:
                for hook in hooks:
                    hook.remove()
            assert calls == {"wm": 1, "pi": 2, "critic": 40}
            report["amp_observed_dtypes"] = {name: sorted(values) for name, values in dtype_observations.items()}
            if config.amp == "bf16":
                assert dtype_observations["projection"] == {"torch.bfloat16"}
                assert dtype_observations["attention"] == {"torch.bfloat16"}
            assert dtype_observations["fusion_output"] == {"torch.float32"}
            assert all(len(optimizer.state) == len(list(models[name].parameters()))
                       for name, optimizer in learner.optimizers.items())
            for optimizer in learner.optimizers.values():
                for state in optimizer.state.values():
                    for value in state.values():
                        assert not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
            after = {name: parameter_digest(model) for name, model in models.items()}
            assert all(after[name] != before[name] for name in models)
            assert learner.iteration == 0 and learner.sim_steps == 64 * 4096
            assert source_identity() == frozen_source == stage["source_identity"]
            report["allocator_after"] = allocator_summary()
            assert report["allocator_after"]["expandable_segment_count"] > 0
            memory = memory_summary()
            assert memory["allocator_statistics"].get("num_ooms", 0) == 0
            report.update(parameter_sha256_after=after, all_model_parameters_unchanged=False,
                          finite_backward_summaries=gradients, fusion_live_second_actor_backward=True,
                          final_adam_state_entries={name: len(opt.state) for name, opt in learner.optimizers.items()},
                          final_iteration=learner.iteration, final_sim_steps=learner.sim_steps,
                          final_memory=memory, actual_optimizer_steps=sum(calls.values()), passed=True)
            report["limitations"] = [
                "Disposable fresh-model production-size diagnostic; models and optimizer states are never saved or reused",
                "One WM batch, two actor horizons, forty critic minibatches; not a ten-WM-batch formal iteration throughput measurement",
                "The second actor backward includes allocated real Adam moments for all three models",
                "Fresh WM holdout quality is diagnostic only; no continuous quality gate or learned-HOI success claim",
                "Actual replay sampling uses replacement; validation does not establish interaction coverage or convergence",
            ]
    except BaseException as exc:
        report["error"] = {"exception_type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if torch.cuda.is_initialized():
            try:
                report["final_memory"] = memory_summary()
                report["allocator_after"] = allocator_summary()
            except BaseException as exc:
                report["final_memory_diagnostic_error_type"] = type(exc).__name__
        if workers is not None:
            workers.close()
        report["nvidia_smi"] = summarize_gpu_samples(sampler.samples, report.get("process_ids", [os.getpid()]), os.getpid())
        if report["passed"] and not report["nvidia_smi"]["assigned_device_uuid_verified"]:
            report["passed"] = False
            report["error"] = {"exception_type": "GPUOwnershipVerificationError",
                               "message": "nvidia-smi did not bind the actual parent process to exactly one GPU UUID"}
        if report["passed"] and report["nvidia_smi"]["has_other_job_processes_on_assigned_device"]:
            report["passed"] = False
            report["error"] = {"exception_type": "GPUIsolationVerificationError",
                               "message": "another CUDA-owning job shares the assigned physical GPU"}
        report["memory_measurement"] = {"global_samples_include_separate_simulator_processes": True,
            "nvidia_smi_interval_seconds": 1., "sampled_peak_is_lower_bound": True,
            "parent_allocator_peaks_are_exact": True, "tensor_arguments_retained_by_observers": False,
            "actual_adam_state_and_optimizer_kernels_executed": True}
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "resource_validation.json", report)
    assert report["passed"], report.get("error")
    print(json.dumps({"passed": report["passed"], "report": str(output / "resource_validation.json")}), flush=True)


if __name__ == "__main__":
    main()

