"""CPU and injected CUDA RNG tests; no GPU or simulator is initialized."""
from contextlib import contextmanager
import unittest

import torch

from hoi.v1.worker_rng import AssetRNG, asset_seed


class FakeCUDA:
    def __init__(self):
        self.current = 0
        self.states = {0: torch.tensor([101]), 1: torch.tensor([202])}
        self.fail_seed = False
        self.fail_next_read = False

    @contextmanager
    def device(self, device):
        previous, self.current = self.current, torch.device(device).index
        try:
            yield
        finally:
            self.current = previous

    def get_rng_state(self, device):
        if self.fail_next_read:
            self.fail_next_read = False
            raise RuntimeError("injected state-read failure")
        return self.states[torch.device(device).index].clone()

    def set_rng_state(self, state, device):
        self.states[torch.device(device).index] = state.clone()

    def manual_seed(self, seed):
        self.states[self.current] = torch.tensor([seed])
        if self.fail_seed:
            raise RuntimeError("injected seed failure")

    def draw(self, device=1):
        result = int(self.states[device][0])
        self.states[device] += 1
        return result


def draws(generator=None):
    # Match the three draw families used by bin sampling/reset/DR.
    return (torch.rand(7, generator=generator),
            torch.empty(3, 2).uniform_(-.01, .01, generator=generator),
            torch.multinomial(torch.tensor([.2, .3, .5]), 9, replacement=True, generator=generator))


class AssetRNGTests(unittest.TestCase):
    def setUp(self):
        self.caller = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, self.caller)

    def assert_draws_equal(self, actual, expected):
        for got, want in zip(actual, expected):
            self.assertTrue(torch.equal(got, want))

    def test_seed_rule_and_invalid_identity(self):
        self.assertEqual(asset_seed(7, 209), 7 + 100003 * 209)
        for base, asset in ((True, 1), (-1, 0), (0, -1), (0, True), (2**64 - 1, 1)):
            with self.subTest(base=base, asset=asset), self.assertRaises(ValueError):
                asset_seed(base, asset)
        for device in ("cuda", "meta"):
            with self.assertRaises(ValueError):
                AssetRNG(1, device)

    def test_construction_and_interleaving_preserve_caller_and_independent_draws(self):
        streams = [AssetRNG(asset_seed(17, a)) for a in (0, 59)]
        expected = [torch.Generator().manual_seed(asset_seed(17, a)) for a in (0, 59)]
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))
        for index in (0, 1, 1, 0, 1, 0):
            caller = torch.get_rng_state().clone()
            with streams[index].activate():
                actual = draws()
            self.assert_draws_equal(actual, draws(expected[index]))
            self.assertTrue(torch.equal(torch.get_rng_state(), caller))
            torch.rand(3)  # Learner draws must not change either asset stream.

    def test_body_exception_advances_asset_but_restores_caller(self):
        stream = AssetRNG(123)
        expected = torch.Generator().manual_seed(123)
        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            with stream.activate():
                self.assert_draws_equal(draws(), draws(expected))
                raise RuntimeError("solver failed")
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))
        with stream.activate():
            self.assert_draws_equal(draws(), draws(expected))

    def test_cuda_seed_and_swaps_preserve_other_device_and_caller(self):
        cuda = FakeCUDA()
        a, b = AssetRNG(31, "cuda:1", cuda=cuda), AssetRNG(47, "cuda:1", cuda=cuda)
        self.assertEqual(cuda.current, 0)
        self.assertEqual(cuda.draw(0), 101)
        self.assertEqual(cuda.draw(1), 202)
        caller_gpu = cuda.states[1].clone()
        expected = torch.Generator().manual_seed(31)
        with a.activate():
            self.assertEqual(cuda.draw(), 31)
            self.assert_draws_equal(draws(), draws(expected))
            # A different asset can be nested; it must restore the outer stream.
            with b.activate():
                self.assertEqual(cuda.draw(), 47)
                torch.rand(5)
            self.assertEqual(cuda.draw(), 32)
            self.assert_draws_equal(draws(), draws(expected))
        self.assertTrue(torch.equal(cuda.states[1], caller_gpu))
        self.assertEqual(int(cuda.states[0][0]), 102)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))
        with a.activate():
            self.assertEqual(cuda.draw(), 33)
        with b.activate():
            self.assertEqual(cuda.draw(), 48)

    def test_cuda_seed_failure_restores_device_and_rng(self):
        cuda = FakeCUDA()
        cuda.fail_seed = True
        with self.assertRaisesRegex(RuntimeError, "seed failure"):
            AssetRNG(31, "cuda:1", cuda=cuda)
        self.assertEqual(cuda.current, 0)
        self.assertEqual(int(cuda.states[0][0]), 101)
        self.assertEqual(int(cuda.states[1][0]), 202)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))

    def test_cuda_read_failure_and_body_failure_restore_caller(self):
        cuda = FakeCUDA()
        stream = AssetRNG(31, "cuda:1", cuda=cuda)
        with self.assertRaisesRegex(RuntimeError, "state-read failure"):
            with stream.activate():
                cuda.draw()
                torch.rand(3)
                cuda.fail_next_read = True
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))
        self.assertEqual(int(cuda.states[1][0]), 202)
        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            with stream.activate():
                self.assertEqual(cuda.draw(), 31)
                raise RuntimeError("solver failed")
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))
        self.assertEqual(int(cuda.states[1][0]), 202)
        with stream.activate():
            self.assertEqual(cuda.draw(), 32)

    def test_same_asset_reentry_is_rejected_without_disturbing_outer_stream(self):
        stream = AssetRNG(123)
        expected = torch.Generator().manual_seed(123)
        with stream.activate():
            self.assert_draws_equal(draws(), draws(expected))
            with self.assertRaisesRegex(RuntimeError, "recursively"):
                with stream.activate():
                    self.fail("must not enter")
            self.assert_draws_equal(draws(), draws(expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), self.caller))


if __name__ == "__main__":
    unittest.main()
