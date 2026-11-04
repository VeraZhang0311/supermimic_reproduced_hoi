from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from hoi.data.protected_eval_subset import choose_clips, create_subset, digest


def fixture():
    spec = [('grail_sitting', 'sitting', 50, 5), ('hoi_retarget', 'unspecified', 288, 9),
            ('omnicontact', 'box/case1_carry', 156, 32),
            ('omnicontact', 'soccer/case1_kick_forward', 45, 1),
            ('omnicontact', 'soccer/case2_kick_left', 51, 1),
            ('omnicontact', 'soccer/case3_kick_right', 87, 1),
            ('omnicontact', 'soccer/case4_carry', 60, 1),
            ('omnicontact', 'soccer/case5_pickball_kick', 75, 1),
            ('ultra', 'unspecified', 273, 8)]
    rows = []
    for dataset, task, count, assets in spec:
        for i in range(count):
            rows.append(dict(dataset=dataset, task=task, clip_id=f'{task}:{i}',
                             asset_id=i % assets, source_group=f'{dataset}:{task}:{i}',
                             partition='protected_eval', state_frames=101))
    parent = dict(partition='protected_eval', clips=rows, manifest={'identity': 'frozen'}, scope={'is_subset': False})
    train = dict(partition='train', clips=[], manifest={'identity': 'frozen'}, scope={'is_subset': False})
    return train, parent


class ProtectedSubsetTests(unittest.TestCase):
    def test_quota_coverage_and_reproducibility(self):
        train, parent = fixture()
        before = deepcopy(parent)
        ids, strata = choose_clips(train, parent)
        self.assertEqual(len(ids), 109)
        self.assertEqual(len(set(ids)), 109)
        self.assertEqual([s['selected_clips'] for s in strata], [5, 29, 16, 4, 5, 9, 6, 8, 27])
        for dataset, count in [('grail_sitting', 5), ('hoi_retarget', 9), ('ultra', 8)]:
            self.assertEqual(len({parent['clips'][i]['asset_id'] for i in ids if parent['clips'][i]['dataset'] == dataset}), count)
        self.assertEqual(choose_clips(train, parent)[0], ids)
        self.assertNotEqual(choose_clips(train, parent, seed=123)[0], ids)
        self.assertEqual(parent, before)
        selected = {(parent['clips'][i]['dataset'], parent['clips'][i]['clip_id']) for i in ids}
        parent['clips'].reverse()
        self.assertEqual({(parent['clips'][i]['dataset'], parent['clips'][i]['clip_id'])
                          for i in choose_clips(train, parent)[0]}, selected)

    def test_leakage_and_duplicate_rejected(self):
        for kind in ['clip', 'group', 'duplicate', 'partition', 'split']:
            train, parent = fixture()
            c = deepcopy(parent['clips'][0])
            if kind in ('clip', 'group'):
                if kind == 'group': c['clip_id'] = 'different_augmentation'
                train['clips'].append(c)
            if kind == 'duplicate': parent['clips'].append(c)
            if kind == 'partition': parent['clips'][0]['partition'] = 'train'
            if kind == 'split': train['manifest'] = {'identity': 'other'}
            with self.subTest(kind=kind), self.assertRaises(ValueError): choose_clips(train, parent)

    def test_saved_rows_hashes_and_no_overwrite(self):
        train, parent = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in [('train', train), ('eval', parent)]:
                data['bundle_content_sha256'] = digest(data)
                (root/f'{name}.json').write_text(json.dumps(data), encoding='utf-8')
            result = create_subset(root/'train.json', root/'eval.json', root/'protect_eval_sm')
            rows = [json.loads(line) for line in (root/'protect_eval_sm/clips.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 109)
            for row, case in zip(rows, result['cases']):
                self.assertEqual(row, parent['clips'][case['parent_clip_index']])
                self.assertEqual(digest(row), case['parent_row_sha256'])
            self.assertFalse(result['training_allowed'])
            with self.assertRaises(FileExistsError): create_subset(root/'train.json', root/'eval.json', root/'protect_eval_sm')
            parent['clips'][0]['state_frames'] += 1
            (root/'eval.json').write_text(json.dumps(parent))
            with self.assertRaises(ValueError): create_subset(root/'train.json', root/'eval.json', root/'bad')
            self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
