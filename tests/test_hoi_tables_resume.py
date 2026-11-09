"""Continuation restores optimizer/RMS/RSI and refuses changed contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

from hoi_tables import resume
from hoi_tables.capacity import _hash_state
from hoi_tables.config import TrainConfig
from hoi_tables.experiments import ULTRA_ASSETS
from hoi_tables.recovery_checks import validate_native_guard
from hoi_tables.sampling import AdaptiveRSI
from hoi_tables.solver_health import classify,configure,policy
from tests.test_hoi_tables_capacity import Model
from tests.test_hoi_tables_sampling import motion


class RecoveryTests(unittest.TestCase):
    def test_original_native_guard_keeps_every_bit_and_nonfinite_fatal(self):
        self.assertTrue(validate_native_guard()['passed'])

    def test_only_explicit_ls_budget_changes(self):
        opts=[NS(ls_iterations=50,iterations=100,timestep=.005),
              NS(ls_iterations=50,iterations=100,timestep=.005,warn_overflow=4095)]
        env=NS(physics=NS(solver=NS(mj_model=NS(opt=opts[0]),mjw_model=NS(opt=opts[1]))))
        configure(env,200)
        self.assertEqual([(o.ls_iterations,o.iterations,o.timestep) for o in opts],[(200,100,.005)]*2)
        self.assertEqual(opts[1].warn_overflow,4095)
        with self.assertRaises(ValueError):configure(env,50)
        for invalid in (False,49,200.,1001):
            with self.assertRaises(ValueError):policy(invalid)
        for names,expected in ((['LS_ITERATIONS'],'solver_iteration_limit'),(['NEFC'],'capacity_overflow'),
             (['NEFC','LS_ITERATIONS'],'capacity_and_solver_limit'),(['FUTURE_BIT'],'unknown_native_failure')):
            self.assertEqual(classify({'overflow_bit_counts':dict.fromkeys(names,1)}),expected)
        self.assertEqual(classify({'overflow_worlds':[{'mask':1<<30}]}),'unknown_native_failure')

    def test_full_restore_and_absolute_schedule(self):
        def learner():
            value=NS(pi=Model(),wm=Model(),critic=Model(),device='cpu',iteration=0,
                     contact_positive=torch.zeros(5),contact_negative=torch.zeros(5))
            value.optimizers={k:torch.optim.Adam(getattr(value,k).parameters()) for k in ('pi','wm','critic')}
            return value
        parent=learner();env=NS(sampler=AdaptiveRSI(motion(),asset_index=0),iteration=0)
        parent.wm(torch.randn(4,3)).square().mean().backward();parent.optimizers['wm'].step()
        parent.wm.rms_count.fill_(100);parent.contact_positive.fill_(91)
        env.sampler.update([0],[False],[0.],elapsed_steps=[50])
        checkpoint={'iteration':500,'policy':deepcopy(parent.pi.state_dict()),'wm':deepcopy(parent.wm.state_dict()),
             'critic':deepcopy(parent.critic.state_dict()),'optimizers':{k:deepcopy(o.state_dict()) for k,o in parent.optimizers.items()},
             'sampler':deepcopy(env.sampler.state_dict()),'contact_positive':parent.contact_positive.clone(),
             'contact_negative':parent.contact_negative.clone(),'rng':{'torch':torch.get_rng_state().clone()}}
        value={'iteration':500,'parent_run_id':'parent00','bundle_sha256':'a'*64,
               'records':[{'rank':i,'sha256':str(i)*64} for i in range(8)]}
        child=learner();newenv=NS(sampler=AdaptiveRSI(motion(),asset_index=0),iteration=0)
        receipt=resume.restore(checkpoint,value,child,newenv)
        self.assertEqual(receipt['next_iteration'],501)
        self.assertEqual(child.wm.rms_count.item(),100)
        self.assertEqual(_hash_state(child.optimizers['wm'].state_dict()),_hash_state(checkpoint['optimizers']['wm']))
        self.assertEqual(_hash_state(newenv.sampler.state_dict()),_hash_state(checkpoint['sampler']))
        self.assertEqual(child.contact_positive.tolist(),[91]*5)
        self.assertTrue(torch.equal(torch.get_rng_state(),checkpoint['rng']['torch']))
        self.assertEqual(TrainConfig().stage(501),'actor_critic')
        self.assertLess(TrainConfig().lr_at(501),TrainConfig().lr_at(1))
        resume.validate_lineage({'from_scratch':False,'base_checkpoint':resume.lineage(value),'solver_health':policy(200)})
        for field,bad in (('repeat_wm_warmup',True),('iteration',501),('checkpoint_sha256',{})):
            base=resume.lineage(value);base[field]=bad
            with self.assertRaises(ValueError):resume.validate_lineage({'from_scratch':False,'base_checkpoint':base})

    def test_bound_bytes_and_changed_science_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'checkpoint.pt'
            identity={'from_scratch':True,'base_checkpoint':None,'asset':ULTRA_ASSETS[0],'asset_id':0,
                      'reward':{},'physics':{'unchanged':True},'experiment':{'config_binding':{'config_sha256':'old'}}}
            torch.save({'schema':'hoi_tables_checkpoint_v1','iteration':500,'identity':identity},cp)
            value={'schema':'hoi_resume_bundle_v1','passed':True,'world_size':8,'assets':list(ULTRA_ASSETS),
                'iteration':500,'parent_run_id':'parent00','replay_and_simulator_serialized':False,
                'records':[{'rank':i,'asset':asset,'path':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
                           for i,asset in enumerate(ULTRA_ASSETS)]}
            path=root/'bundle.json';path.write_text(json.dumps(value))
            current=deepcopy(identity);current['solver_health']=policy(200)
            current['experiment']['config_binding']['config_sha256']='reviewed new launcher'
            resume.load_rank(path,0,current,enforce_model_source=False)
            current['physics']={'changed':True}
            with self.assertRaisesRegex(ValueError,'scientific'):resume.load_rank(path,0,current,enforce_model_source=False)
            cp.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'bytes changed'):resume.load_rank(path,0,identity,enforce_model_source=False)

    def test_continuation_telemetry_preserves_absolute_iteration(self):
        from hoi_tables.telemetry import curated_config,JSONLReader
        value={'iteration':500,'parent_run_id':'parent00','bundle_sha256':'a'*64,
               'records':[{'rank':i,'sha256':str(i)*64} for i in range(8)]}
        metadata={'from_scratch':False,'resume':resume.lineage(value),'solver_health':policy(200),'start_iteration':501}
        config=curated_config(metadata)
        self.assertFalse(config['from_scratch'])
        self.assertEqual(config['resume']['parent_run_id'],'parent00')
        self.assertEqual(config['start_iteration'],501)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.jsonl'
            path.write_text(json.dumps({'iteration':501,'stage':'actor_critic','wm_total':1.})+'\n')
            reader=JSONLReader(path);self.assertEqual(reader.poll(),1)
            self.assertEqual(set(reader.pending),{501})
