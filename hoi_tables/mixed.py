"""Ultra + tables input contracts and checkpoint initialization."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from hoi_residual.data import Indexed, UltraMotionLib, sha256_file
from supermimic.rotations import quat_to_mat
from .config import TrainConfig
from .experiments import TABLE_ASSETS, ULTRA_ASSETS

ASSETS = ULTRA_ASSETS + TABLE_ASSETS
PROPORTIONS = (1 / 16,) * 8 + (1 / 4,) * 2
SOURCE_ITERATION = 8000
CHECKPOINT_SCHEMA = "hoi_ultra_tables_checkpoint_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_hashes(experiment):
    paths = experiment.ultra["paths"]
    inputs = {"experiment": experiment.root / "experiment.json",
              "table_manifest": experiment.root / "tables/manifest.json",
              "table_contacts": experiment.root / "tables/reference_contacts.pt",
              "table_assets": experiment.raw["table_assets"], "table_shapes": experiment.raw["table_shapes"],
              "canonical": experiment.ultra["evaluation"]["canonical"],
              **{f"ultra_{name}": paths[name] for name in ("manifest", "asset_bundle", "geometry_bundle",
                                                            "shape_bank", "reference_contacts", "native_hands")}}
    return {key: sha256_file(path) for key, path in inputs.items()}


def evaluation_checkpoint(experiment, path, iteration, expected_hash, *, supplemental=False):
    from .evaluate import state_dict_hash, _validate_state, _model_specifications
    from .networks import feature_schema

    if sha256_file(path) != expected_hash:
        raise ValueError("Mixed evaluation checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    identity = state["identity"]
    cadence = experiment.config.ckpt_every if supplemental else 2000
    if (state["schema"] != CHECKPOINT_SCHEMA or state["iteration"] != iteration or iteration % cadence
            or identity["profile"] != "ultra_tables" or identity["assets"] != list(ASSETS)
            or identity["features"] != feature_schema() or identity["inputs"] != input_hashes(experiment)
            or identity["solver_ls_iterations"] != 200):
        raise ValueError("Mixed evaluation checkpoint protocol or input identity mismatch")
    models = {}
    for name, specification in _model_specifications(experiment.config).items():
        _validate_state(state[name], specification, name)
        models[name] = state_dict_hash(state[name])
    return {"policy_state": state["policy"], "identity": identity, "iteration": iteration,
            "checkpoint_sha256": {"mixed": expected_hash}, "model_state_sha256": models}


def read_experiment(root):
    root = Path(root).resolve()
    raw = read_json(root / "experiment.json")
    if (raw["schema"] != "hoi_ultra_tables_experiment_v1"
            or raw["sampling"] != {"ultra_boxes": .5, "largetable": .25, "smalltable": .25}
            or raw["evaluation"] != {"every": 2000, "suites": ["canonical_hoi80", "protected_tables20"],
                                      "owning_training_run": True}):
        raise ValueError("Mixed experiment sampling or evaluation protocol differs")
    settings = raw["training"]
    config = TrainConfig(num_envs=settings["num_envs"], iterations=settings["iterations"], seed=settings["seed"],
                         ckpt_every=settings["checkpoint_every"]).validate()
    if config.num_envs % 16 or config.ckpt_every != 100 or config.iterations <= SOURCE_ITERATION:
        raise ValueError("Mixed experiment needs exact lane counts and 100-iteration checkpoints")
    execution = raw["execution"]
    for key in ("actor_microbatch", "wm_microbatch"):
        if type(execution[key]) is not int or execution[key] < 1:
            raise ValueError("Microbatches must be positive integers")
    if (type(execution["overlap_collection"]) is not bool or type(execution.get("collector_cuda_graph", True)) is not bool
            or type(execution.get("compiled_actor", False)) is not bool
            or execution["target_seconds_per_iteration"] <= 0):
        raise ValueError("Invalid mixed pipeline execution settings")
    ultra = read_json(raw["ultra_experiment"])
    if ultra["profile"] != "ultra" or ultra["solver"]["ls_iterations"] != 200:
        raise ValueError("Mixed experiment must inherit the Ultra native solver settings")
    return SimpleNamespace(root=root, raw=raw, config=config, execution=execution, ultra=ultra,
                           lanes=tuple(int(config.num_envs * weight) for weight in PROPORTIONS))


class MixedMotion(UltraMotionLib):
    """One global frame address space built from the ten verified local banks."""

    def __init__(self, motions, kin, device):
        if len(motions) != len(ASSETS) or any(m.partition != "train" or not m.for_training for m in motions):
            raise ValueError("Mixed motion requires all ten training banks")
        self.partition, self.for_training, self.allowed_for_training = "train", True, True
        self.device, self.kin = device, kin
        self.states = torch.cat([m.states for m in motions]).to(device)
        self.objects = torch.cat([m.objects for m in motions]).to(device)
        self.num_frames = len(self.states)
        self.root_p, self.root_v, self.root_w = self.states[:, :3], self.states[:, 7:10], self.states[:, 10:13]
        self.q, self.qd = self.states[:, 13:42], self.states[:, 42:71]
        self.root_R = Indexed(lambda f: quat_to_mat(self.states[f, 3:7]), self.num_frames)
        self.root_v_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_v[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        self.root_w_local = Indexed(lambda f: (self.root_R[f].transpose(-1, -2) @ self.root_w[f].unsqueeze(-1)).squeeze(-1), self.num_frames)
        self.clip_len = torch.cat([m.clip_len for m in motions]).to(device)
        self.clip_end = self.clip_len.cumsum(0) - 1
        self.clip_id = torch.repeat_interleave(torch.arange(len(self.clip_len), device=device), self.clip_len)
        self.clip_asset_ids = torch.repeat_interleave(torch.arange(len(ASSETS), device=device),
                                                    torch.tensor([len(m.entries) for m in motions], device=device))


def components(experiment):
    from .assets import load_assets, load_shape_bank
    from .native_hands import load_native_hands
    from .ultra import load_ultra_assets, load_ultra_shape

    raw, paths = experiment.raw, experiment.ultra["paths"]
    boxes, box_hands = load_ultra_assets(paths["asset_bundle"], paths["geometry_bundle"])
    box_points, _ = load_ultra_shape(paths["shape_bank"], paths["asset_bundle"])
    tables = load_assets(raw["table_assets"])
    table_points, _ = load_shape_bank(raw["table_shapes"], raw["table_assets"])
    table_hands = read_json(raw["table_assets"])["hands"]["geometry_cache"]
    table_hands = {**table_hands, "path": str(Path(raw["table_assets"]).parent / table_hands["path"])}
    return (boxes, tables), (box_hands, table_hands), torch.cat((box_points, table_points)), load_native_hands(paths["native_hands"])


def load_source(experiment, models, native_hands):
    """Restore shared model/RMS and the box samplers, checking the evaluated 8000 lineage."""
    from .evaluate import state_dict_hash, _validate_state
    from .networks import feature_schema
    from .solver_health import identity_policy
    from supermimic.g1_physics import physics_contract

    request_path = experiment.raw["ultra_initialization"]
    request = read_json(request_path)
    evidence = read_json(Path(request_path).parent / "evaluation/result.json")
    if (request["iteration"] != SOURCE_ITERATION or evidence["iteration"] != SOURCE_ITERATION
            or not evidence["passed"] or evidence["clip_count"] != 80
            or evidence["checkpoint_sha256"] != request["checkpoint_sha256"]):
        raise ValueError("Missing verified Ultra checkpoint 8000 evaluation binding")
    samplers = []
    for index, asset in enumerate(ULTRA_ASSETS):
        rank = f"rank_{index:02d}"
        path = request["checkpoints"][rank]
        if sha256_file(path) != request["checkpoint_sha256"][rank]:
            raise ValueError(f"Ultra source checkpoint bytes changed for {asset}")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        identity = state["identity"]
        if (state["iteration"] != SOURCE_ITERATION or identity["asset"] != asset
                or identity["schema"] != feature_schema() or identity["native_hands"] != native_hands.identity
                or identity["experiment"]["profile"] != "ultra" or identity_policy(identity)["ls_iterations"] != 200
                or identity["physics"] != physics_contract(native_hands.asset_dir)):
            raise ValueError(f"Ultra checkpoint scientific identity differs for {asset}")
        for name, model in models.items():
            value = state[name]
            _validate_state(value, model.state_dict(), name)
            if state_dict_hash(value) != evidence["model_state_sha256"][name]:
                raise ValueError(f"Ultra model/RMS mismatch on {rank}: {name}")
            if index == 0:
                model.load_state_dict(value, strict=True)
        samplers.append({key: value.clone() if torch.is_tensor(value) else value
                         for key, value in state["sampler"].items()})
    return samplers, {"iteration": SOURCE_ITERATION, "checkpoint_sha256": request["checkpoint_sha256"],
                      "model_state_sha256": evidence["model_state_sha256"]}
