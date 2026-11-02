"""Audit actual start frames, not reset choices or post-reset snapshots."""
import copy
import unittest

import torch

from hoi.v1.sampling_audit import ActualCollectionAudit
from hoi.v1.task_sampling import PersistentTaskSampler
from hoi.v1.workers import allocate_workers
from tests.test_hoi_task_sampling import population, motion_for, adaptive_state


class ActualCollectionAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.bundle = population()
        self.bank = motion_for(self.bundle)
        self.specs = allocate_workers(self.bundle, self.bank, 1000)
        self.frames = torch.cat([PersistentTaskSampler(self.bank, s.asset_id, s.num_envs).first_frames(
            torch.arange(s.num_envs), horizon=32) for s in self.specs])
        self.assets = torch.tensor([s.asset_id for s in self.specs for _ in range(s.num_envs)])

    def transition(self, frames=None, ep_step=None, done=None, term=None):
        n = len(self.frames)
        return {"f": self.frames.clone() if frames is None else frames.clone(), "asset_id": self.assets.clone(),
                "ep_step": torch.zeros(n,dtype=torch.int32) if ep_step is None else ep_step,
                "done": torch.zeros(n,dtype=torch.bool) if done is None else done,
                "term": torch.zeros(n,dtype=torch.bool) if term is None else term}

    def test_exact_collection_counts_and_group_details(self):
        audit = ActualCollectionAudit(self.bundle,self.specs)
        self.assertFalse(audit.report()["allocation_match"])
        for tick in range(3):
            audit.observe(self.transition(self.frames+tick,torch.full((1000,),tick,dtype=torch.int32)))
        report=audit.report(detail=True)
        self.assertTrue(report["allocation_match"])
        self.assertEqual(report["total_transitions"],3000)
        self.assertEqual(sum(report["reset_category_counts"].values()),1000)
        self.assertEqual(sum(report["clip_transition_counts"]),3000)
        self.assertEqual(sum(report["source_group_transition_counts"].values()),3000)
        self.assertEqual(report["category_counts"]["omnicontact/soccer"],450)
        for name,count in report["category_counts"].items():
            self.assertEqual(count,report["allocation"]["category_env_counts"][name]*3)

    def test_clip_start_boundary_maps_to_next_clip_and_terminal_rejected(self):
        audit=ActualCollectionAudit(self.bundle,self.specs)
        # Asset 0 contains ten clips. Its second clip begins exactly at the
        # previous clip's exclusive end; it must count as clip 1, not clip 0.
        frames=self.frames.clone()
        frames[self.assets==0]=self.bank.clip_start[1]
        audit.observe(self.transition(frames))
        report=audit.report(detail=True)
        self.assertEqual(report["clip_transition_counts"][0],0)
        self.assertEqual(report["clip_transition_counts"][1],int((self.assets==0).sum()))
        bad=frames.clone();bad[0]=self.bank.clip_end[1]
        with self.assertRaisesRegex(ValueError,"terminal-only"):
            audit.observe(self.transition(bad))
        self.assertEqual(audit.report()["total_transitions"],1000)

    def test_worker_local_frames_must_be_mapped_to_global_before_audit(self):
        audit=ActualCollectionAudit(self.bundle,self.specs)
        bad=self.frames.clone()
        row=int(torch.nonzero(self.assets==1)[0])
        # Valid local frame zero for asset 1 is invalid as central global frame.
        bad[row]=0
        with self.assertRaisesRegex(ValueError,"asset/frame"):
            audit.observe(self.transition(bad))
        self.assertEqual(audit.report()["total_transitions"],0)

    def test_lane_permutation_rejected_even_when_all_ratios_still_match(self):
        audit=ActualCollectionAudit(self.bundle,self.specs)
        audit.observe(self.transition())
        soccer_asset=next(r["asset_id"] for r in self.bank.sampling_records if r["category"]=="omnicontact/soccer")
        ids=torch.nonzero(self.assets==soccer_asset).flatten()
        bad=self.frames.clone()
        a,b=int(ids[0]),int(ids[-1])
        bad[a],bad[b]=bad[b].clone(),bad[a].clone()
        with self.assertRaisesRegex(ValueError,"persistent task lane"):
            audit.observe(self.transition(bad))
        self.assertEqual(audit.report()["total_transitions"],1000)

    def test_wrong_soccer_task_mass_rejected_before_counting(self):
        audit=ActualCollectionAudit(self.bundle,self.specs)
        soccer_asset=next(r["asset_id"] for r in self.bank.sampling_records if r["category"]=="omnicontact/soccer")
        ids=torch.nonzero(self.assets==soccer_asset).flatten()
        bad=self.frames.clone();bad[ids[0]]=bad[ids[-1]]
        with self.assertRaisesRegex(ValueError,"transition ratios"):
            audit.observe(self.transition(bad))
        self.assertFalse(audit.report()["allocation_match"])

    def test_different_episode_lengths_do_not_turn_reset_counts_into_transition_ratios(self):
        audit=ActualCollectionAudit(self.bundle,self.specs)
        n=len(self.frames)
        # Deliberately terminate all sitting rows on every tick; other rows
        # remain in their episodes. Their transition shares must stay 10%.
        sitting = torch.tensor([self.bank.sampling_records[int(self.bank.clip_id[f])]["category"]=="grail_sitting/sitting" for f in self.frames])
        for tick in range(5):
            ages=torch.full((n,),tick,dtype=torch.int32);ages[sitting]=0
            audit.observe(self.transition(self.frames,ages,sitting.clone(),sitting.clone()))
        report=audit.report()
        self.assertEqual(report["category_fractions"]["grail_sitting/sitting"],.1)
        resets=report["reset_category_counts"]
        self.assertGreater(resets["grail_sitting/sitting"]/sum(resets.values()),.3)
        self.assertEqual(report["completed_episodes"],500)
        self.assertEqual(report["terminations"],500)

    def test_shapes_types_negative_steps_and_out_of_range_fail_closed(self):
        corruptions=[("f",self.frames.float()),("asset_id",self.assets[:-1]),
                     ("ep_step",torch.full((1000,),-1,dtype=torch.int32)),
                     ("done",torch.zeros(1000)),("term",torch.zeros(1000,dtype=torch.int32))]
        out=self.frames.clone();out[0]=-1;corruptions.append(("f",out))
        for key,value in corruptions:
            with self.subTest(key=key):
                audit=ActualCollectionAudit(self.bundle,self.specs)
                tr=self.transition();tr[key]=value
                with self.assertRaises(ValueError):audit.observe(tr)
                self.assertEqual(audit.report()["total_transitions"],0)


if __name__=="__main__":
    unittest.main()
