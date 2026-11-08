"""Fresh-campaign safety and immutable input routing; no GPU or scheduler."""
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hoi_tables.launch import freeze, prepare
from hoi_tables.telemetry import curated_config


def fake_link(path,target,target_is_directory=False):
    # Native Windows symlink privileges are unrelated to launch correctness.
    if target_is_directory:path.mkdir()
    else:shutil.copyfile(target,path)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'repo'
        (self.source/'hoi_tables').mkdir(parents=True)
        file=self.source/'hoi_tables/fixture.py';file.write_text('value = 1\n')
        (self.source/'data/g1_hw3').mkdir(parents=True)
        self.manifest={'branch':'hoi-retarget/tables-boxes-v1.0','revision':'a'*40,
                       'files':{'hoi_tables/fixture.py':hashlib.sha256(file.read_bytes()).hexdigest()}}
        (self.source/'SOURCE_MANIFEST.json').write_text(json.dumps(self.manifest))
        self.inputs=self.root/'inputs';self.inputs.mkdir()
        values={'manifest':{'partitions':{'train':{'motions':2651,'source_groups':381}}},
                'asset_bundle':{},'shape_bank':{},'geometry_bundle':{},'reference_contacts':{},'canonical':{}}
        self.paths={}
        for key,value in values.items():
            p=self.inputs/(key+'.json');p.write_text(json.dumps(value));self.paths[key]=str(p)
        self.config={'schema':'hoi_direct_teacher_experiment_v1','profile':'ultra',
                     'paths':{k:v for k,v in self.paths.items() if k!='canonical'},
                     'training':{'num_envs':4096,'iterations':20000,'seed':0},
                     'execution':{'workers_per_gpu':1},
                     'evaluation':{'kind':'canonical_hoi80','every':2000,'owning_training_run':True,
                                   'canonical':self.paths['canonical'],'helper_root':'/frozen/helper'}}
        self.config_path=self.root/'experiment.json';self.config_path.write_text(json.dumps(self.config))

    def test_prepare_eight_assets_preserves_scientific_defaults_and_owning_run(self):
        campaign=self.root/'campaign'
        with patch.object(Path,'symlink_to',autospec=True,side_effect=fake_link):
            result=prepare(self.config_path,campaign,self.source)
        self.assertEqual((result['workers'],result['gpus']), (8,8))
        self.assertTrue(result['evaluation_owning_run']);self.assertFalse(result['gpu_job_started'])
        metadata=json.loads((campaign/'training/run_metadata.json').read_text())
        self.assertEqual(metadata['train_clips'],2651)
        self.assertEqual(metadata['train_source_groups'],381)
        self.assertNotIn('protected_eval',metadata)
        self.assertEqual(metadata['global_config']['wm_only_iters'],500)
        self.assertEqual(metadata['global_config']['horizon'],32)
        self.assertEqual(metadata['global_config']['wm_batch'],2048)
        self.assertEqual(metadata['global_config']['pi_batch'],4096)
        self.assertEqual(metadata['schema']['wm_input_dim'],313)
        self.assertEqual(metadata['schema']['wm_output_dim'],82)
        self.assertEqual(len(metadata['schema']['contact']),5)
        safe=curated_config(metadata)
        self.assertEqual(safe['experiment']['evaluation_role'],'canonical_development80')
        self.assertEqual(safe['branch'],'hoi-retarget/tables-boxes-v1.0')
        self.assertEqual(json.loads((campaign/'experiment.json').read_text())['evaluation']['every'],2000)

    def test_explicit_two_workers_per_gpu_keeps_eight_asset_workers(self):
        with patch.object(Path,'symlink_to',autospec=True,side_effect=fake_link):
            result=prepare(self.config_path,self.root/'shared',self.source,workers_per_gpu=2)
        self.assertEqual((result['workers'],result['workers_per_gpu'],result['gpus']),(8,2,4))

    def test_existing_campaign_is_never_overwritten(self):
        campaign=self.root/'existing';campaign.mkdir();sentinel=campaign/'training';sentinel.write_text('active')
        with self.assertRaises(FileExistsError):prepare(self.config_path,campaign,self.source)
        self.assertEqual(sentinel.read_text(),'active')

    def test_missing_fresh_contact_cache_fails_before_creating_campaign(self):
        Path(self.paths['reference_contacts']).unlink();campaign=self.root/'missing'
        with self.assertRaisesRegex(FileNotFoundError,'reference_contacts'):prepare(self.config_path,campaign,self.source)
        self.assertFalse(campaign.exists())

    def test_source_hash_drift_rejected_before_code_is_frozen(self):
        (self.source/'hoi_tables/fixture.py').write_text('value = 2\n')
        with self.assertRaisesRegex(ValueError,'frozen source changed'):freeze(self.source,self.root/'bad_source')
        self.assertFalse((self.root/'bad_source').exists())


if __name__=='__main__':unittest.main()
