"""Validate all eight native DR groups, then the full v1.2 GPU training graph.

Run once in an independent source/output allocation. Successful diagnostics do
not launch training and their disposable models are never written or resumed.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get("SUPERMIMIC_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_report_workers(specs, worker_config):
    from hoi.v1.workers import WorkerGroup
    assert worker_config["device"] == "cuda:0"
    # Only the spawned native simulators need CUDA in this report-only phase.
    # Keep the controller's IPC/preflight on CPU so no controller CUDA context
    # survives to appear as an external owner during the subprocess probe.
    return WorkerGroup(specs, worker_config, device="cpu")


def main():
    from hoi_v1_2_launch_train import verify_stage
    root, stage = verify_stage()
    assert os.environ.get("WANDB_SYNC") == "0"
    import torch
    torch.set_num_threads(3)
    assert not torch.cuda.is_initialized(), "native report controller must stay CPU-only"
    from supermimic.physics import PhysicsSettings
    from hoi.v1.config import fresh_directory, write_json
    from hoi.v1.actor_fusion import actor_architecture_contract
    from hoi.v1.train import read_config
    from hoi.v1.evaluate import _read_bundle_header, validate_evaluation_identity
    from hoi.v1.workers import WorkerSpec, geometry_group
    from hoi.v1.gates import require_native_evidence
    from hoi.v1.randomization import MATCHING_DR, randomization_contract, validate_dr_evidence
    config, changes = read_config(root / "configs/hoi_v1_2_perception_fusion.json")
    assert not changes and config.to_dict() == stage["production_config"]
    train = _read_bundle_header(stage["train_bundle"])
    validate_evaluation_identity(train, _read_bundle_header(stage["eval_bundle"]))
    assert len(train["assets"]) == 8 and {int(a["asset_id"]) for a in train["assets"]} == set(range(8))
    physics = asdict(PhysicsSettings().export())
    assert require_native_evidence(stage["base_native_evidence"], expected_physics=physics,
               required_asset_ids=range(8), expected_bundle_sha256=train["bundle_content_sha256"])["passed"]
    output = fresh_directory(root / "results/v12_native_dr")
    specs = [WorkerSpec(int(a["asset_id"]), 4, 4, geometry_group(a)) for a in train["assets"]]
    reports = []
    for nominal in (False, True):
        print("NATIVE_DR_MODE", "nominal_eval" if nominal else "training_DR", flush=True)
        worker_config = {"bundle": stage["train_bundle"],
                  "output": str(output / ("nominal" if nominal else "randomized")),
                  "physics": physics, "domain_randomization_profile": MATCHING_DR, "device": "cuda:0",
                  "robot_mjcf": train["robot_fk"]["mjcf_path"], "seed": 0, "eval_mode": nominal,
                  "reset_noise": False, "variant": "hoi_v1", "max_processes": 8,
                  "validate_domain_randomization": True, "launcher_args": []}
        workers = native_report_workers(specs, worker_config)
        try:
            assert len(workers.native_reports) == 8
            for report in workers.native_reports:
                check = report["domain_randomization_validation"]
                assert check["passed"] and check["effective_MuJoCo_Warp_properties_verified"] and check["subset_reset_passed"]
            reports.extend(workers.native_reports)
            write_json(output / ("nominal_reports.json" if nominal else "randomized_reports.json"), workers.native_reports)
        finally:
            workers.close()
    assert len(reports) == 16
    assert not torch.cuda.is_initialized(), "native report controller unexpectedly initialized CUDA"
    evidence = json.loads(Path(stage["base_native_evidence"]).read_text())
    evidence["domain_randomization"] = {"contract": randomization_contract(MATCHING_DR),
        "native_property_readback_passed": True, "subset_reset_passed": True, "nominal_eval_passed": True,
        "source_identity": stage["source_identity"], "asset_ids": list(range(8)),
        "native_job_id": os.environ["SLURM_JOB_ID"],
        "reports": {name: {"path": str(output / name), "sha256": sha256(output / name)}
                    for name in ("randomized_reports.json", "nominal_reports.json")}}
    evidence_path = output / "native_evidence.json"
    write_json(evidence_path, evidence)
    assert validate_dr_evidence(evidence_path, MATCHING_DR)["passed"]
    print("V12_NATIVE_DR_ALL_8_ASSETS_BOTH_MODES_PASSED", flush=True)
    resource_path = root / "results/v12_resource/resource_validation.json"
    env = dict(os.environ, BUNDLE=stage["train_bundle"], NATIVE_EVIDENCE=str(evidence_path),
               CONFIG=str(root / "configs/hoi_v1_2_perception_fusion.json"),
               RUN_DIR=str(resource_path.parent))
    subprocess.run([sys.executable, str(root / "tools/hoi_v1_2_resource_probe.py")], cwd=root, env=env, check=True)
    resource = json.loads(resource_path.read_text())
    assert resource["passed"] and resource["actual_optimizer_steps"] == 43
    assert resource["actual_optimizer_steps_by_model"] == {"wm": 1, "pi": 2, "critic": 40}
    assert resource["fusion_live_second_actor_backward"] and resource["probe_models_discarded"]
    assert resource["config"] == stage["production_config"] and resource["source_identity"] == stage["source_identity"]
    assert resource["stage_sha256"] == os.environ["EXPECTED_STAGE_SHA256"]
    assert resource["actor_architecture_contract"] == actor_architecture_contract(config.actor_architecture, config.observation_profile)
    write_json(root / "results/v12_validation_complete.json", {
        "schema": "hoi_v1_2_full_gpu_validation_complete_1", "passed": True,
        "source_identity": stage["source_identity"], "stage_sha256": os.environ["EXPECTED_STAGE_SHA256"],
        "config": config.to_dict(), "actor_architecture_contract": resource["actor_architecture_contract"],
        "native_evidence": str(evidence_path), "native_evidence_sha256": sha256(evidence_path),
        "resource_report": str(resource_path), "resource_report_sha256": sha256(resource_path),
        "actual_optimizer_steps": resource["actual_optimizer_steps"],
        "actual_optimizer_steps_by_model": resource["actual_optimizer_steps_by_model"],
        "probe_models_discarded": True, "formal_training_started": False,
        "job_id": os.environ["SLURM_JOB_ID"]})
    print("V12_NATIVE_AND_PRODUCTION_GPU_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
