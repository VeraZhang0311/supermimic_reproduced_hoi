"""Dataset routing and execution topology for the unchanged direct teacher."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

TABLE_ASSETS = ("largetable", "smalltable")
ULTRA_ASSETS = tuple(sorted(f"{obj}_{scale}_{scale}_{scale}" for obj in
                           ("largebox", "smallbox", "plasticbox", "suitcase") for scale in ("080", "100")))
SCHEMA = "hoi_direct_teacher_experiment_v1"


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    assets: tuple
    evaluation: str

    @property
    def world_size(self): return len(self.assets)

    def topology(self, *, rank, world, local_rank, workers_per_gpu=1, shared_gpu=False,
                 smoke=False, asset=None, visible_devices=None):
        if type(workers_per_gpu) is not int or workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be a positive integer")
        if world not in (1, self.world_size) or (not smoke and world != self.world_size):
            raise ValueError(f"{self.name} formal training requires exactly {self.world_size} synchronized asset workers")
        if not (0 <= rank < world and 0 <= local_rank < world):
            raise ValueError("Invalid distributed worker rank")
        if world == 1:
            if not smoke or asset not in range(self.world_size):
                raise ValueError("Single-worker smoke requires an explicit valid asset")
        elif asset is not None:
            raise ValueError("Distributed workers select their exact asset by rank")
        if world % workers_per_gpu:
            raise ValueError("Asset workers must divide equally among visible GPUs")
        gpu = 0 if shared_gpu else local_rank // workers_per_gpu
        needed = 1 if shared_gpu else world // workers_per_gpu
        if visible_devices is not None and ((shared_gpu and visible_devices != 1) or visible_devices < needed):
            raise ValueError(f"Execution requires {needed} scheduler-visible GPUs")
        if shared_gpu and not smoke:
            raise ValueError("--shared-gpu is reserved for smoke; formal sharing uses explicit workers_per_gpu")
        return {"asset_id": rank if world > 1 else asset, "device_index": gpu,
                "workers_per_gpu": world if shared_gpu else workers_per_gpu,
                "required_visible_devices": needed}

    def load_assets(self, args):
        if self.name == "tables":
            from .assets import load_assets
            catalog = json.loads(Path(args.asset_bundle).read_text(encoding="utf-8"))
            return load_assets(args.asset_bundle), catalog["hands"]["geometry_cache"]
        from .ultra import load_ultra_assets
        return load_ultra_assets(args.asset_bundle, args.geometry_bundle)

    def load_shape(self, args):
        if self.name == "tables":
            from .assets import load_shape_bank
            return load_shape_bank(args.shape_bank, args.asset_bundle)
        from .ultra import load_ultra_shape
        return load_ultra_shape(args.shape_bank, args.asset_bundle)

    def motion(self, args, kin, device, asset_id):
        kwargs = dict(partition="train", for_training=True, asset=asset_id, one_per_asset=args.smoke)
        if self.name == "tables":
            from .data import TableMotionLib
            return TableMotionLib(args.manifest, kin, device, **kwargs)
        from .ultra import UltraTeacherMotion
        return UltraTeacherMotion(args.manifest, kin, device, canonical=args.canonical, helper_root=args.helper_root, **kwargs)

    def reward(self, args, mdp, assets, hands):
        if self.name == "tables":
            from .reward import TablesReward
            return TablesReward(mdp, assets, hands, reference_contacts=args.reference_contacts)
        from .ultra import make_ultra_reward
        return make_ultra_reward(mdp, assets, hands, reference_contacts=args.reference_contacts)


PROFILES = {"tables": ExperimentProfile("tables", TABLE_ASSETS, "protected_tables22"),
            "ultra": ExperimentProfile("ultra", ULTRA_ASSETS, "canonical_hoi80")}


def profile(name="tables"):
    try: return PROFILES[name]
    except KeyError as exc: raise ValueError("Unknown direct-teacher experiment profile") from exc


def resolve_arguments(args):
    """Fill only unspecified CLI values; JSON cannot change scientific math."""
    raw, binding = {}, None
    if args.experiment_config:
        path = Path(args.experiment_config)
        payload = path.read_bytes()
        raw = json.loads(payload)
        if raw.get("schema") != SCHEMA or set(raw)-{"schema","profile","paths","training","execution","evaluation","solver"}:
            raise ValueError("Invalid direct-teacher experiment schema/fields")
        for key in ("paths", "training", "execution", "evaluation"):
            if not isinstance(raw.get(key), dict): raise ValueError("Experiment requires " + key)
        allowed = {"paths":{"manifest","asset_bundle","shape_bank","reference_contacts","geometry_bundle","resume_bundle"},
                   "training":{"num_envs","iterations","seed"}, "execution":{"workers_per_gpu"},
                   "evaluation":{"kind","every","owning_training_run","canonical","helper_root"}}
        for key, names in allowed.items():
            if set(raw[key])-names: raise ValueError("Unsupported experiment fields: " + key)
        if not isinstance(raw.get('solver',{}),dict) or set(raw.get('solver',{}))-{'ls_iterations'}:
            raise ValueError('Invalid explicit numerical solver budget')
        binding = {"schema":SCHEMA,"config_sha256":hashlib.sha256(payload).hexdigest()}
    selected = profile(args.profile or raw.get("profile", "tables"))
    evaluation = raw.get("evaluation", {})
    if evaluation and (evaluation.get("kind") != selected.evaluation or evaluation.get("every") != 2000
                       or evaluation.get("owning_training_run") is not True):
        raise ValueError("Evaluation must use the matching full panel every2000 on the owning training run")
    defaults = {"num_envs":4096,"iterations":20000,"seed":0,"workers_per_gpu":1,"canonical":None,"helper_root":None,
                'resume_bundle':None,'ls_iterations':50}
    values = {**raw.get("paths",{}), **raw.get("training",{}), **raw.get("execution",{}),
              **{key:evaluation[key] for key in ("canonical","helper_root") if key in evaluation},**raw.get('solver',{})}
    for name in ("manifest","asset_bundle","shape_bank","reference_contacts","geometry_bundle", *defaults):
        if getattr(args, name, None) is None: setattr(args, name, values.get(name,defaults.get(name)))
    for name in ("manifest","asset_bundle","shape_bank"):
        if not getattr(args,name): raise ValueError("Missing --" + name.replace("_","-"))
    if selected.name == "ultra" and not args.geometry_bundle:
        raise ValueError("Ultra requires an explicit geometry bundle")
    for name in ("num_envs", "iterations", "workers_per_gpu"):
        if type(getattr(args,name)) is not int or getattr(args,name)<1: raise ValueError(name+" must be a positive integer")
    if type(args.seed) is not int or args.seed<0: raise ValueError("seed must be nonnegative")
    from .solver_health import policy
    policy(args.ls_iterations)
    if args.resume_bundle and selected.name!='ultra':raise ValueError('This continuation contract is Ultra-only')
    args.profile = selected.name
    if binding is not None:
        binding.update(profile=selected.name, assets=list(selected.assets), evaluation={"kind":selected.evaluation,"every":2000,"owning_training_run":True})
    return selected, binding
