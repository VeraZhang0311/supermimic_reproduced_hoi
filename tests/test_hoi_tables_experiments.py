"""Dataset routing preserves scientific defaults and exact asset ownership."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hoi_tables.config import TrainConfig
from hoi_tables.experiments import profile, resolve_arguments, SCHEMA, ULTRA_ASSETS
from hoi_tables.train import build_parser


class ExperimentTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args(["--output","unused",*extra])

    def test_old_tables_cli_and_identity_defaults(self):
        args = self.args("--manifest","m","--asset-bundle","a","--shape-bank","s")
        experiment,binding = resolve_arguments(args)
        self.assertEqual(experiment.assets,("largetable","smalltable"))
        self.assertIsNone(binding)
        self.assertEqual((args.num_envs,args.iterations,args.seed,args.workers_per_gpu),(4096,20000,0,1))
        self.assertEqual(experiment.topology(rank=1,world=2,local_rank=1,visible_devices=2)["device_index"],1)

    def config(self):
        return {"schema":SCHEMA,"profile":"ultra","paths":{"manifest":"m","asset_bundle":"a","shape_bank":"s",
                "geometry_bundle":"g","reference_contacts":"c"},"training":{"num_envs":4096,"iterations":20000,"seed":0},
                "execution":{"workers_per_gpu":2},"evaluation":{"kind":"canonical_hoi80","every":2000,"owning_training_run":True,
                "canonical":"canonical","helper_root":"helper"}}

    def resolve_json(self, value, *extra):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"experiment.json"; path.write_text(json.dumps(value))
            args = self.args("--experiment-config",str(path),*extra)
            experiment,binding = resolve_arguments(args)
            return args,experiment,binding

    def test_ultra_config_global_batches_and_explicit_four_gpu_mapping(self):
        args,experiment,binding = self.resolve_json(self.config())
        self.assertEqual(experiment.assets,ULTRA_ASSETS)
        self.assertEqual((args.canonical,args.helper_root),("canonical","helper"))
        cfg = TrainConfig(num_envs=args.num_envs,iterations=args.iterations,seed=args.seed)
        workers = [experiment.topology(rank=i,world=8,local_rank=i,workers_per_gpu=args.workers_per_gpu,visible_devices=4) for i in range(8)]
        self.assertEqual([w["asset_id"] for w in workers],list(range(8)))
        self.assertEqual([w["device_index"] for w in workers],[0,0,1,1,2,2,3,3])
        local = replace(cfg,num_envs=cfg.num_envs//8,wm_batch=cfg.wm_batch//8,pi_batch=cfg.pi_batch//8).validate()
        self.assertEqual((local.num_envs,local.wm_batch,local.pi_batch),(512,256,512))
        for name in cfg.to_dict():
            if name not in ("num_envs","wm_batch","pi_batch"): self.assertEqual(getattr(local,name),getattr(cfg,name))
        self.assertEqual(binding["profile"],"ultra")

    def test_cli_overrides_only_explicit_fields(self):
        args,_,_ = self.resolve_json(self.config(),"--iterations","22000","--workers-per-gpu","1")
        self.assertEqual((args.iterations,args.workers_per_gpu,args.num_envs),(22000,1,4096))

    def test_checkpoint_cadence_preserves_eval_and_supports_explicit_override(self):
        args,_,binding=self.resolve_json(self.config())
        self.assertEqual(args.checkpoint_every,100)
        self.assertEqual(binding['evaluation']['every'],2000)
        value=self.config();value['execution']['checkpoint_every']=100
        args,_,_=self.resolve_json(value,'--checkpoint-every','500')
        self.assertEqual(args.checkpoint_every,500)
        tables=self.args('--manifest','m','--asset-bundle','a','--shape-bank','s')
        resolve_arguments(tables);self.assertEqual(tables.checkpoint_every,500)
        for bad in (0,True,50,300,100.0):
            value=self.config();value['execution']['checkpoint_every']=bad
            with self.assertRaisesRegex(ValueError,'checkpoint_every'):self.resolve_json(value)

    def test_ultra_actor_concurrency_preserves_two_gpu_topology(self):
        value = self.config()
        value["execution"].update(workers_per_gpu=4, actor_ranks_per_turn=4)
        args, experiment, _ = self.resolve_json(value)
        self.assertEqual((args.actor_ranks_per_turn, experiment.world_size // args.workers_per_gpu), (4, 2))
        args, _, _ = self.resolve_json(value, "--actor-ranks-per-turn", "2")
        self.assertEqual(args.actor_ranks_per_turn, 2)
        for bad in (True, 0, 3, 8):
            value["execution"]["actor_ranks_per_turn"] = bad
            with self.assertRaisesRegex(ValueError, "actor_ranks_per_turn"):
                self.resolve_json(value)

    def test_single_l40_keeps_all_eight_assets_and_full_global_configuration(self):
        value=self.config();value['execution']['workers_per_gpu']=8
        args,experiment,_=self.resolve_json(value)
        workers=[experiment.topology(rank=i,world=8,local_rank=i,
                                    workers_per_gpu=args.workers_per_gpu,visible_devices=1) for i in range(8)]
        self.assertEqual([w['asset_id'] for w in workers],list(range(8)))
        self.assertEqual([w['device_index'] for w in workers],[0]*8)
        self.assertEqual([w['required_visible_devices'] for w in workers],[1]*8)
        cfg=TrainConfig(num_envs=args.num_envs,iterations=args.iterations,seed=args.seed).validate()
        self.assertEqual((cfg.num_envs,cfg.wm_batch,cfg.pi_batch,cfg.horizon,cfg.wm_only_iters),
                         (4096,2048,4096,32,500))

    def test_rejects_scientific_overrides_and_separate_eval_run(self):
        value = self.config(); value["training"]["wm_only_iters"] = 12
        with self.assertRaisesRegex(ValueError,"Unsupported"): self.resolve_json(value)
        value = self.config(); value["evaluation"]["owning_training_run"] = False
        with self.assertRaisesRegex(ValueError,"owning training"): self.resolve_json(value)

    def test_wrong_worker_count_asset_mapping_and_visible_gpus_rejected(self):
        experiment = profile("ultra")
        for kwargs in ({"world":2,"rank":0,"local_rank":0},
                       {"world":8,"rank":0,"local_rank":0,"asset":1},
                       {"world":8,"rank":7,"local_rank":7,"workers_per_gpu":2,"visible_devices":3},
                       {"world":8,"rank":0,"local_rank":0,"workers_per_gpu":3},
                       {"world":8,"rank":0,"local_rank":0,"shared_gpu":True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): experiment.topology(**kwargs)
        self.assertEqual(experiment.topology(world=1,rank=0,local_rank=0,asset=7,smoke=True,visible_devices=1)["asset_id"],7)

    def test_formal_capacity_requires_all_eight_actual_assets(self):
        from hoi_tables.capacity import _configuration
        global_cfg = TrainConfig()
        local = replace(global_cfg,num_envs=512,wm_batch=256,pi_batch=512)
        replay = SimpleNamespace(N=512,T=1000,size=64)
        learner = SimpleNamespace(config=local,device="cuda:3",buffer=replay,gradient_sync=lambda:None)
        env = SimpleNamespace(N=512,asset={"asset_id":7,"object_asset":ULTRA_ASSETS[7]})
        with patch("hoi_tables.capacity.dist.is_initialized",return_value=True),patch("hoi_tables.capacity.dist.get_world_size",return_value=8),\
             patch("hoi_tables.capacity.dist.get_rank",return_value=7),patch("hoi_tables.capacity.dist.get_backend",return_value="gloo"),\
             patch("hoi_tables.capacity.torch.cuda.get_device_properties",return_value=SimpleNamespace(name="NVIDIA L40")):
            self.assertEqual(_configuration(learner,replay,env,global_cfg,True,ULTRA_ASSETS,2),8)
            with self.assertRaisesRegex(ValueError,"exact profile asset"):
                _configuration(learner,replay,env,global_cfg,True,("largetable","smalltable"),1)
            env.asset["asset_id"]=6
            with self.assertRaisesRegex(ValueError,"own its exact"):
                _configuration(learner,replay,env,global_cfg,True,ULTRA_ASSETS,2)


if __name__=="__main__": unittest.main()
