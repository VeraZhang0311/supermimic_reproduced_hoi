"""Actual deterministic caches, parallel preparation and corruption rejection."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from hoi.data.curated import canonical_bytes, digest, load_json
from hoi.data.curated_bundle import _prepare_asset, prepare_curated_bundle
from hoi.data.curated_geometry import prepare_geometry_manifest, load_geometry_manifest
from hoi.v1.prepare import _bound_load, load_bundle
from tests import test_hoi_curated_data as fixtures


class PreparedGeometryTests(unittest.TestCase):
    setUp = fixtures.CuratedBundleTests.setUp

    def test_parallel_output_matches_existing_algorithm_and_resumes_without_rebuilding(self):
        recipe = load_json(self.recipe)
        for i in range(2):
            entry = deepcopy(recipe['assets'][0])
            entry['object_geometry_sha256'] = str(i)*64
            entry['exact_scale'] = [1+i*.2, 1, 1]
            entry['physical_profile']['mass'] = i+2
            recipe['assets'].append(entry)
        self.recipe.write_bytes(canonical_bytes(recipe))
        out = self.root/'parallel'
        manifest = prepare_geometry_manifest(self.recipe,out,workers=2,threads=1)
        self.assertEqual(len(manifest['assets']),3)
        for row,entry in zip(manifest['assets'],recipe['assets']):
            serial = self.root/f"serial_{row['asset_index']}"
            serial.mkdir()
            direct = _prepare_asset(entry,row['asset_index'],self.root,serial)
            a,b = _bound_load(direct['geometry_cache']),_bound_load(row['geometry_cache'])
            self.assertEqual(a['metadata'],b['metadata'])
            for key in ('points','sdf_grid','grid_min','grid_max','bbox_diagonal'):
                self.assertTrue(torch.equal(a[key],b[key]),key)
        # Even after an interrupted final manifest write, successes are reused.
        (out/'geometry_manifest.json').unlink()
        with patch('hoi.data.curated_geometry.ProcessPoolExecutor.submit',side_effect=AssertionError('must not rebuild')):
            resumed = prepare_geometry_manifest(self.recipe,out,workers=2,threads=1,resume=True)
        self.assertEqual(manifest,resumed)

    def test_two_full_partitions_share_verified_cache_without_recomputing(self):
        out = self.root/'prepared'
        prepare_geometry_manifest(self.recipe,out,threads=1)
        manifest = out/'geometry_manifest.json'
        with patch('hoi.v1.geometry.build_geometry_cache',side_effect=AssertionError('cache reuse must not recalculate')):
            train = prepare_curated_bundle(self.split,self.recipe,self.root/'train',self.kin,geometry_cache_manifest=manifest)
            evaluate = prepare_curated_bundle(self.split,self.recipe,self.root/'eval',self.kin,
                partition='protected_eval',geometry_cache_manifest=manifest)
        self.assertEqual(train['assets'][0]['geometry_cache'],evaluate['assets'][0]['geometry_cache'])
        bank,_,_ = load_bundle(self.root/'train/bundle.json',self.kin)
        self.assertEqual(bank.partition,'train')
        with self.assertRaisesRegex(ValueError,'only train'):
            load_bundle(self.root/'eval/bundle.json',self.kin)
        raw = Path(train['assets'][0]['geometry_cache']['path'])
        raw.write_bytes(raw.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'cache SHA256'):
            load_geometry_manifest(manifest,self.recipe)

    def test_recipe_and_geometry_identity_cannot_be_swapped(self):
        out = self.root/'prepared'
        prepare_geometry_manifest(self.recipe,out,threads=1)
        path = out/'geometry_manifest.json'
        original = path.read_bytes()
        manifest = load_json(path)
        manifest['assets'][0]['cache_key'] = '0'*64
        manifest['content_sha256'] = digest({k:v for k,v in manifest.items() if k != 'content_sha256'})
        path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(ValueError,'key or asset identity'):
            load_geometry_manifest(path,self.recipe)
        path.write_bytes(original)
        recipe = load_json(self.recipe)
        recipe['assets'][0]['physical_profile']['mass'] = 7
        self.recipe.write_bytes(canonical_bytes(recipe))
        with self.assertRaisesRegex(ValueError,'different native recipe'):
            load_geometry_manifest(path,self.recipe)

    def test_incomplete_manifest_and_duplicate_producer_are_rejected(self):
        out = self.root/'prepared'
        prepare_geometry_manifest(self.recipe,out,threads=1)
        path = out/'geometry_manifest.json'
        manifest = load_json(path)
        manifest['assets'] = []
        manifest['content_sha256'] = digest({k:v for k,v in manifest.items() if k != 'content_sha256'})
        path.write_bytes(canonical_bytes(manifest))
        with self.assertRaisesRegex(ValueError,'complete native recipe'):
            load_geometry_manifest(path,self.recipe)
        (out/'PREPARING.claim').write_text('active producer')
        with self.assertRaises(FileExistsError):
            prepare_geometry_manifest(self.recipe,out,resume=True)
        self.assertEqual((out/'PREPARING.claim').read_text(),'active producer')


if __name__ == '__main__':
    unittest.main()
