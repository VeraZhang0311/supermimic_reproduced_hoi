import json
from pathlib import Path
import tempfile
import unittest

import torch

from hoi.data.protected_eval_subset import create_subset, digest
from hoi.v1.evaluation_subset import load_selection, asset_batches
from hoi.v1.evaluate import evaluate_group
from hoi.v1.workers import WorkerSpec
from tests.test_protected_eval_subset import fixture as subset_fixture
from tests.test_hoi_v1_evaluate import fixture, SyntheticWorkers


class SubsetEvaluationTests(unittest.TestCase):
    def test_saved_manifest_load_and_tamper_rejection(self):
        train, parent = subset_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in [('train', train), ('eval', parent)]:
                data['bundle_content_sha256'] = digest(data)
                (root/f'{name}.json').write_text(json.dumps(data))
            create_subset(root/'train.json', root/'eval.json', root/'sm')
            ids, report = load_selection(root/'sm/selection.json', train, parent)
            self.assertEqual(len(ids), 109)
            self.assertTrue(report['is_subset'])
            batches = asset_batches(parent, ids, 4)
            self.assertTrue(all(len(b) <= 4 for b in batches))
            self.assertEqual({i for b in batches for _, selected in b for i in selected}, ids)
            (root/'sm/clips.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                load_selection(root/'sm/selection.json', train, parent)

    def test_batch_plan_uses_only_selected_clips_and_duration_order(self):
        bundle = {'clips': [dict(asset_id=a, state_frames=n) for a,n in [(2,90),(0,11),(2,4),(1,22),(3,31)]]}
        self.assertEqual(asset_batches(bundle, {0,1,3,4}, 2), [[(0,[1]),(1,[3])],[(3,[4]),(2,[0])]])
        self.assertEqual(asset_batches(bundle, {0,1}, 1), [[(0,[1])],[(2,[0])]])
        with self.assertRaises(ValueError): asset_batches(bundle, None, 0)

    def test_mixed_asset_full_duration_alignment(self):
        policy, motion, mdp, shapes, fk, metadata = fixture()
        motion.asset_ids[motion.clip_start[1]:] = 1

        class MixedWorkers(SyntheticWorkers):
            specs = [WorkerSpec(0, 1, 1, 'a'), WorkerSpec(1, 1, 1, 'b')]
            def reset(self, starts):
                super().reset(starts)
                self.value['asset_id'] = self.motion.asset_ids[starts].clone()
                return self.snapshot()

        workers = MixedWorkers(motion, mdp)
        before = {k:v.clone() for k,v in policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            reports = evaluate_group(policy, motion, mdp, shapes, workers, [0,1], fk39=fk,
                                     joint_names=fk.dof_names, output=Path(tmp)/'batch', metadata=metadata,
                                     verify_tracked_fk=False)
            self.assertEqual([c['asset_id'] for c in reports], [0,1])
            self.assertEqual([c['control_steps'] for c in reports], [3,6])
            for c in reports:
                data = torch.load(c['rollout'], weights_only=True)
                self.assertEqual(len(data['states']['s']), c['control_steps']+1)
                self.assertTrue(c['full_reference_completed'])
            self.assertTrue(all(torch.equal(before[k],v) for k,v in policy.state_dict().items()))
        workers.specs.reverse()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'layout'):
            evaluate_group(policy, motion, mdp, shapes, workers, [0,1], fk39=fk,
                           joint_names=fk.dof_names, output=Path(tmp)/'bad', verify_tracked_fk=False)


if __name__ == '__main__':
    unittest.main()
