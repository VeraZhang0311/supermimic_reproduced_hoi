import unittest
from types import SimpleNamespace
import torch
from tools.hoi_native_task_visual import select_frames, TASKS

class SelectionTests(unittest.TestCase):
    def fixture(self):
        bank = SimpleNamespace(names=['z', 'a'], clip_start=torch.tensor([0,5]), clip_end=torch.tensor([4,9]),
            expected_contact=torch.zeros(10,16), expected_valid=torch.zeros(10,16))
        entries=[dict(name=n,dataset='omnicontact',task=t) for n,t in zip(bank.names,TASKS)]
        # Direction names must not force the corresponding anatomical foot.
        bank.expected_contact[2,13]=bank.expected_valid[2,13]=1
        bank.expected_contact[7,12]=bank.expected_valid[7,12]=1
        return bank,entries
    def test_both_categories_and_correct_arrival(self):
        bank,entries=self.fixture(); rows=select_frames(bank,entries,[12,13])
        self.assertEqual([r['environment'] for r in rows],list(range(8)))
        self.assertEqual([r['arrival_frame'] for r in rows],[2]*4+[7]*4)
        self.assertEqual([r['start_frame'] for r in rows],[1]*4+[6]*4)
        self.assertEqual([r['task'] for r in rows], [TASKS[0]]*4+[TASKS[1]]*4)
    def test_unknown_positive_not_used(self):
        bank,entries=self.fixture(); bank.expected_valid[2,13]=0
        with self.assertRaises(ValueError):select_frames(bank,entries,[12,13])
    def test_no_cross_clip_predecessor(self):
        bank,entries=self.fixture(); bank.expected_contact[7,12]=0
        bank.expected_contact[5,12]=bank.expected_valid[5,12]=1
        with self.assertRaises(ValueError):select_frames(bank,entries,[12,13])

if __name__=='__main__':unittest.main()
