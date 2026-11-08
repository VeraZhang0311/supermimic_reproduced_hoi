"""Actor serialization changes execution order, never the H32 update math."""
from contextlib import contextmanager
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hoi_tables.learner import _finite
from hoi_tables.networks import frozen_parameters
from hoi_tables.train import actor_backward_schedule
from tests.test_hoi_tables_learning import fixture, window


def original_policy_update(learner, start):
    """The pre-serialization algorithm, kept as an independent equivalence oracle."""
    optimizer = learner.optimizers["pi"]
    learner.optimizers["wm"].zero_grad(set_to_none=True)
    learner.optimizers["critic"].zero_grad(set_to_none=True)
    with frozen_parameters(learner.wm, learner.critic):
        trajectory = learner.imagine(start)
        loss = -trajectory["objective"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if learner.gradient_sync is not None: learner.gradient_sync(learner.pi)
        grad = nn.utils.clip_grad_norm_(learner.pi.parameters(), 1.)
        nn.utils.clip_grad_value_(learner.pi.parameters(), .1)
        if not bool(torch.isfinite(grad)): raise FloatingPointError("Nonfinite actor BPTT gradient norm")
        _finite(learner.pi,"actor")
        if any(p.grad is not None for module in (learner.wm,learner.critic) for p in module.parameters()):
            raise RuntimeError("Actor BPTT must not populate WM or critic parameter gradients")
        optimizer.step()
    critic_loss = learner.critic_update(trajectory,start)
    return {"pi_loss":float(loss.detach()),"v_loss":critic_loss,"pi_gradnorm":float(grad),
            "imag_return_per_step":float(trajectory["reward"].detach().mean()),
            "imag_survival":float(trajectory["terminal_survival"].detach().mean())}


class ActorMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def assert_tree_equal(self, first, second):
        if isinstance(first,torch.Tensor): self.assertTrue(torch.equal(first,second))
        elif isinstance(first,dict):
            self.assertEqual(set(first),set(second))
            for key in first: self.assert_tree_equal(first[key],second[key])
        elif isinstance(first,(tuple,list)):
            self.assertEqual(len(first),len(second))
            for a,b in zip(first,second): self.assert_tree_equal(a,b)
        else: self.assertEqual(first,second)

    def test_h32_parameters_gradients_adam_rms_metrics_and_rng_equal_original(self):
        original,start=fixture(B=2,H=32)
        scheduled,other_start=fixture(B=2,H=32)
        before_rms={name:deepcopy(model.state_dict()) for name,model in (("pi",scheduled.pi),("wm",scheduled.wm),("critic",scheduled.critic))}
        local_grads={"original":[],"scheduled":[]};events=[];active={"value":False}
        def sync(label):
            def apply(module):
                if label=="scheduled": self.assertFalse(active["value"])
                if module is (original.pi if label=="original" else scheduled.pi):
                    local_grads[label]=[p.grad.clone() for p in module.parameters()]
                # Emulate a nontrivial peer gradient contribution. This also
                # verifies clipping/Adam remain after the global gradient mean.
                for p in module.parameters():
                    if p.grad is not None: p.grad.mul_(.5).add_(.0001)
            return apply
        original.gradient_sync=sync("original");scheduled.gradient_sync=sync("scheduled")
        @contextmanager
        def context():
            self.assertFalse(active["value"]);active["value"]=True;events.append("enter")
            try: yield
            finally:
                self.assertTrue(all(p.requires_grad for model in (scheduled.wm,scheduled.critic) for p in model.parameters()))
                self.assertTrue(all(p.grad is None for model in (scheduled.wm,scheduled.critic) for p in model.parameters()))
                active["value"]=False;events.append("exit")
        scheduled.actor_backward_context_factory=context
        rng=torch.get_rng_state()
        expected=original_policy_update(original,start)
        expected_rng=torch.get_rng_state()
        torch.set_rng_state(rng)
        actual=scheduled.policy_update(other_start)
        self.assertEqual(actual,expected)
        self.assertEqual(events,["enter","exit"])
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assert_tree_equal(local_grads["original"],local_grads["scheduled"])
        for name in ("pi","wm","critic"):
            self.assert_tree_equal(getattr(original,name).state_dict(),getattr(scheduled,name).state_dict())
            self.assert_tree_equal(original.optimizers[name].state_dict(),scheduled.optimizers[name].state_dict())
            for key,value in before_rms[name].items():
                if "norm." in key: self.assertTrue(torch.equal(value,getattr(scheduled,name).state_dict()[key]))
        self.assertEqual(scheduled.config.horizon,32)
        self.assertEqual(scheduled.config.pi_batch,2)

    def test_only_local_forward_backward_are_inside_context(self):
        learner,start=fixture(H=4);active={"value":False};events=[]
        @contextmanager
        def context():
            active["value"]=True;events.append("enter")
            try: yield
            finally: active["value"]=False;events.append("exit")
        learner.actor_backward_context_factory=context
        imagine=learner.imagine
        def checked_imagine(values):
            self.assertTrue(active["value"])
            self.assertTrue(all(not p.requires_grad for m in (learner.wm,learner.critic) for p in m.parameters()))
            events.append("imagine")
            return imagine(values)
        learner.imagine=checked_imagine
        def synchronize(module): self.assertFalse(active["value"]);events.append("sync")
        learner.gradient_sync=synchronize
        for name in ("pi","wm","critic"):
            # Imagined forward must only read RMS. Calling any update would
            # invoke distributed statistics on a real shared-rank learner.
            getattr(learner,name).update_normalizers=lambda *a,**k: (_ for _ in ()).throw(AssertionError("actor RMS update"))
        old_clip_norm=nn.utils.clip_grad_norm_;old_clip_value=nn.utils.clip_grad_value_
        def clip_norm(*a,**k): self.assertFalse(active["value"]);return old_clip_norm(*a,**k)
        def clip_value(*a,**k): self.assertFalse(active["value"]);return old_clip_value(*a,**k)
        old_step=learner.optimizers["pi"].step
        def step(*a,**k): self.assertFalse(active["value"]);events.append("step");return old_step(*a,**k)
        learner.optimizers["pi"].step=step
        old_critic_update=learner.critic_update
        def critic_update(*a,**k): self.assertFalse(active["value"]);events.append("critic");return old_critic_update(*a,**k)
        learner.critic_update=critic_update
        with patch.object(nn.utils,"clip_grad_norm_",side_effect=clip_norm),patch.object(nn.utils,"clip_grad_value_",side_effect=clip_value),\
             patch("torch.distributed.all_reduce",side_effect=AssertionError("unexpected internal collective")),\
             patch("torch.distributed.barrier",side_effect=AssertionError("unexpected internal collective")):
            learner.policy_update(start)
        self.assertEqual(events[:6],["enter","imagine","exit","sync","step","critic"])
        self.assertEqual(events.count("sync"),1+learner.config.critic_epochs*learner.config.critic_minibatches)

    def test_exception_exits_context_and_restores_frozen_flags_without_step(self):
        learner,start=fixture(H=3);events=[]
        @contextmanager
        def context():
            events.append("enter")
            try: yield
            finally: events.append("exit")
        learner.actor_backward_context_factory=context
        before=deepcopy(learner.pi.state_dict())
        with patch.object(learner,"imagine",side_effect=RuntimeError("fixture forward failure")):
            with self.assertRaisesRegex(RuntimeError,"forward failure"): learner.policy_update(start)
        self.assertEqual(events,["enter","exit"])
        self.assertTrue(all(p.requires_grad for m in (learner.wm,learner.critic) for p in m.parameters()))
        self.assert_tree_equal(before,learner.pi.state_dict())
        self.assertFalse(learner.optimizers["pi"].state)

    def test_wm_only_and_readonly_actions_do_not_enter_actor_context(self):
        learner,start=fixture(H=3)
        def forbidden(): raise AssertionError("actor context entered outside actor update")
        learner.actor_backward_context_factory=forbidden
        report=learner.update(1,window=window(learner,start),start=start)
        self.assertEqual(report["stage"],"wm_only")
        learner.action({**start,"partition":"protected_eval"},exploration=False,update_statistics=False)

    def test_tables_default_and_single_worker_ultra_stay_parallel(self):
        self.assertEqual(actor_backward_schedule("tables",1),"parallel")
        self.assertEqual(actor_backward_schedule("tables",8),"parallel")
        self.assertEqual(actor_backward_schedule("ultra",1),"parallel")
        self.assertEqual(actor_backward_schedule("ultra",8),"serialized_per_gpu")
        learner,_=fixture(H=3)
        from contextlib import nullcontext
        self.assertIs(learner.actor_backward_context_factory,nullcontext)


if __name__=="__main__": unittest.main()
