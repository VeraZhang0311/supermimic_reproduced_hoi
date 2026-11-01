"""Explicit packed-loader reuse retains validation and independent motion banks."""
import copy
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

from hoi.data.curated_bundle import prepare_curated_bundle
from hoi.data.curated import canonical_bytes, sha, write_split
from hoi.v1 import prepare
from hoi.v1.prepare import load_bundle
from tests import test_hoi_curated_data as fixtures
from tests.test_hoi_v1_geometry import BundleKinematics


class SharedShapesTests(unittest.TestCase):
    def setUp(self):
        fixtures.CuratedBundleTests.setUp(self)
        self.path = self.root / 'bundle' / 'bundle.json'
        self.bundle = prepare_curated_bundle(self.split, self.recipe, self.path.parent, self.kin)

    def test_explicit_reuse_skips_static_loads_and_keeps_motion_local(self):
        bank, shapes, original = load_bundle(self.path, self.kin, asset_filter=[0])
        static_paths = {item['geometry_cache']['path'] for item in self.bundle['assets']}
        static_paths.add(self.bundle['hands']['geometry_cache']['path'])
        with patch.object(prepare, '_bound_load', wraps=prepare._bound_load) as reader, \
                patch('hoi.v1.geometry.ShapeAssetBank', side_effect=AssertionError('registry rebuilt')):
            other, reused, loaded = load_bundle(self.path, BundleKinematics(), asset_filter=[0], shared_shapes=shapes)
        self.assertIs(reused, shapes)
        self.assertEqual(loaded, original)
        self.assertIsNot(other, bank)
        self.assertFalse(static_paths & {call.args[0]['path'] for call in reader.call_args_list})
        self.assertEqual(len(reader.call_args_list), 2)  # Reference and contacts are still loaded.
        self.assertEqual(other.num_frames, bank.num_frames)
        self.assertEqual(other.num_bins, bank.num_bins)
        self.assertTrue(torch.equal(other.bin_start, bank.bin_start))
        self.assertNotEqual(other.bin_start.data_ptr(), bank.bin_start.data_ptr())
        self.assertTrue(torch.equal(other.global_frame_index, bank.global_frame_index))
        self.assertTrue(torch.equal(other.expected_contact, bank.expected_contact))
        self.assertNotEqual(other.expected_contact.data_ptr(), bank.expected_contact.data_ptr())
        other.expected_contact.fill_(0)
        self.assertEqual(bank.expected_contact[1, 1].item(), 1.)
        _, separate, _ = load_bundle(self.path, self.kin)
        self.assertIsNot(separate, shapes)  # Baseline remains opt-in, no implicit global cache.
        self.assertNotEqual(separate._points.data_ptr(), shapes._points.data_ptr())

    def test_distinct_asset_filters_share_full_registry_but_not_frames_or_bins(self):
        # Add a second independent training source, keeping clip1 protected.
        source = self.root / 'clip2'
        shutil.copytree(self.root / 'clip0', source)
        geometry = self.root / 'second_geometry.json'
        geometry.write_bytes(canonical_bytes({'shapes': [{'kind': 'cube', 'size': [.2] * 3}],
                                               'basis': 'second_source'}))
        geometry_hash = sha(geometry)
        metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
        metadata.update(clip_id='clip2', object_geometry_path=str(geometry), object_geometry_sha256=geometry_hash)
        metadata['provenance']['source_group'] = 'sub2_box_000'
        (source / 'metadata.json').write_bytes(canonical_bytes(metadata))
        success = {'metadata_sha256': sha(source / 'metadata.json'), 'array_sha256': sha(source / 'labels.npz')}
        (source / 'SUCCESS.json').write_bytes(canonical_bytes(success))
        rows = [json.loads(line) for line in (self.selection / 'included.jsonl').read_text(encoding='utf-8').splitlines()]
        row = {**rows[0], 'clip_id': 'clip2', 'output': str(source)}
        rows.append(row)
        (self.selection / 'included.jsonl').write_text(''.join(canonical_bytes(r).decode() + '\n' for r in rows), encoding='utf-8')
        manifest = json.loads((self.selection / 'manifest.json').read_text(encoding='utf-8'))
        manifest['included_count'] = 3; manifest['included_by_dataset']['ultra'] = 3
        manifest['files']['included.jsonl']['sha256'] = sha(self.selection / 'included.jsonl')
        (self.selection / 'manifest.json').write_bytes(canonical_bytes(manifest))
        (self.selection / 'READY.json').write_bytes(canonical_bytes({'manifest_sha256': sha(self.selection / 'manifest.json')}))
        metadata_rows = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8'))
        metadata_rows.append({**row, **success, 'labels_sha256': success['array_sha256'],
                              'provenance': metadata['provenance'], 'object_geometry_sha256': geometry_hash,
                              'object_geometry_path': str(geometry)})
        snapshot = self.root / 'multi_metadata.json'; snapshot.write_bytes(canonical_bytes(metadata_rows))
        split = self.root / 'multi_split' / 'split.json'
        write_split(self.selection, snapshot, split.parent)
        recipe = json.loads(self.recipe.read_text(encoding='utf-8'))
        asset = copy.deepcopy(recipe['assets'][0])
        asset.update(object_asset='second', object_geometry_sha256=geometry_hash)
        asset['physical_profile']['mass'] = 2  # Separate native identity, not an alias.
        recipe['assets'].append(asset)
        self.recipe.write_bytes(canonical_bytes(recipe))
        path = self.root / 'multi_bundle' / 'bundle.json'
        bundle = prepare_curated_bundle(split, self.recipe, path.parent, self.kin)
        self.assertEqual(len(bundle['assets']), 2)
        banks = []
        shapes = None
        for asset_id in range(2):
            bank, registry, _ = load_bundle(path, BundleKinematics(), asset_filter=[asset_id], shared_shapes=shapes)
            if shapes is not None:
                self.assertIs(registry, shapes)
            shapes = registry; banks.append(bank)
            self.assertEqual(len(registry.assets), 2)
            self.assertEqual(bank.num_frames, 3)
            self.assertEqual(bank.num_bins, 1)
            self.assertEqual(bank.asset_ids.tolist(), [asset_id] * 3)
            self.assertEqual(bank.bin_asset.tolist(), [asset_id])
        self.assertFalse(set(banks[0].global_frame_index.tolist()) & set(banks[1].global_frame_index.tolist()))
        self.assertNotEqual(banks[0].q.data_ptr(), banks[1].q.data_ptr())

    def test_shared_queries_preserve_values_gradients_and_readonly_signature(self):
        _, shapes, _ = load_bundle(self.path, self.kin)
        _, separate, _ = load_bundle(self.path, self.kin)
        ids = torch.tensor([0, 0])
        points = torch.tensor([[[.18, .02, 0]], [[.02, -.15, .01]]], requires_grad=True)
        other_points = points.detach().clone().requires_grad_(True)
        self.assertFalse(shapes._grid_cache)
        actual = shapes.query(ids, points)
        expected = separate.query(ids, other_points)
        self.assertTrue(shapes._grid_cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward(); expected.sum().backward()
        torch.testing.assert_close(points.grad, other_points.grad, rtol=0, atol=0)
        for method in ('points', 'bbox_diagonal', 'physical_context'):
            torch.testing.assert_close(getattr(shapes, method)(ids), getattr(separate, method)(ids), rtol=0, atol=0)
        _, reused, _ = load_bundle(self.path, BundleKinematics(), shared_shapes=shapes)
        self.assertIs(reused, shapes)  # Lazy grid caching and backward are valid reads.

    def test_reuse_rehashes_static_and_dynamic_inputs_even_with_same_stat(self):
        _, shapes, _ = load_bundle(self.path, self.kin)
        records = [self.bundle['assets'][0]['geometry_cache'],
                   self.bundle['hands']['geometry_cache'],
                   self.bundle['clips'][0]['reference_cache'],
                   self.bundle['clips'][0]['contact_cache'],
                   self.bundle['native_robot_asset'],
                   self.bundle['reference_kinematics']['urdf']]
        source = Path(self.bundle['clips'][0]['resolved_source_output']) / 'labels.npz'
        records.append({'path': str(source)})
        for record in records:
            with self.subTest(path=record['path']):
                path = Path(record['path']); original = path.read_bytes(); stat = path.stat()
                modified = bytearray(original); modified[-1] ^= 1
                try:
                    path.write_bytes(modified)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    with self.assertRaises(ValueError):
                        load_bundle(self.path, self.kin, shared_shapes=shapes)
                finally:
                    path.write_bytes(original)
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_mutated_geometry_hands_physics_and_derived_tensors_rejected(self):
        mutations = [lambda s: s.geometries[0]['points'].add_(.1),
                     lambda s: s.geometries[0]['sdf_grid'].add_(.1),
                     lambda s: s.geometries[0]['metadata'].update(point_hash='changed'),
                     lambda s: s.assets[0]['physical_profile'].update(mass=99),
                     lambda s: s.hands.update(unbound=True),
                     lambda s: s._points.add_(.1),
                     lambda s: s._bbox.add_(.1),
                     lambda s: s._contexts.add_(.1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                _, shapes, _ = load_bundle(self.path, self.kin)
                mutate(shapes)
                with self.assertRaisesRegex(ValueError, 'read-only'):
                    load_bundle(self.path, self.kin, shared_shapes=shapes)

    def test_binding_rejects_other_bundle_device_fk_and_unbound_registry(self):
        _, shapes, _ = load_bundle(self.path, self.kin)
        bundle = copy.deepcopy(self.bundle); bundle['bundle_content_sha256'] = 'different'
        kin = BundleKinematics(); kin.body_names = ['changed']
        for bundle_arg, kin_arg, device in ((bundle, self.kin, 'cpu'),
                                          (self.bundle, self.kin, 'cuda:0'),
                                          (self.bundle, kin, 'cpu')):
            with self.subTest(device=device, body_names=kin_arg.body_names):
                with self.assertRaisesRegex(ValueError, 'same curated bundle'):
                    prepare._validate_shared_shapes(shapes, bundle_arg, kin_arg, device)
        del shapes._curated_share_binding
        with self.assertRaisesRegex(ValueError, 'same curated bundle'):
            load_bundle(self.path, self.kin, shared_shapes=shapes)

    def test_protected_eval_cannot_reuse_train_shapes_or_skip_partition_gate(self):
        _, shapes, _ = load_bundle(self.path, self.kin)
        path = self.root / 'evaluation' / 'bundle.json'
        prepare_curated_bundle(self.split, self.recipe, path.parent, self.kin, partition='protected_eval')
        with self.assertRaisesRegex(ValueError, 'only train'):
            load_bundle(path, self.kin, shared_shapes=shapes)
        with self.assertRaisesRegex(ValueError, 'same curated bundle'):
            load_bundle(path, self.kin, require_train=False, shared_shapes=shapes)


if __name__ == '__main__':
    unittest.main()
