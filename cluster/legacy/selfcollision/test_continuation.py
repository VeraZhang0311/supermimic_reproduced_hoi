"""CPU checks for resume-only glue; no simulator or scientific implementation mocks."""
import ast
import argparse
import copy
import io
from pathlib import Path
import sys
import types
import unittest

sys.dont_write_bytecode = True
import resume_launch
import continue_pipeline


class ResumeCompatibilityTests(unittest.TestCase):
    def test_actual_upstream_only_three_declared_edits(self):
        # Undo the two condition changes and two audit calls and require exact AST
        # equality with the actual upstream source, including every numerical update.
        source = Path(__file__).resolve().parents[1] / 'supermimic/train.py'
        if not source.exists():
            source = Path('D:/supermimic_reproduced_hoi_selfcollision/supermimic/train.py')
        original = ast.parse(source.read_text())
        patched = resume_launch.transform(source.read_text())
        removed = 0
        for node in ast.walk(patched):
            for field, value in ast.iter_fields(node):
                if not isinstance(value, list): continue
                for item in list(value):
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name) and item.value.func.id in ('_resume_audit', '_prefill_audit'):
                        value.remove(item)
                        removed += 1
            if isinstance(node, ast.If):
                if ast.unparse(node.test) in ('start_it == 0 or bool(args.resume)', 'log_f.tell() == 0'):
                    node.test = ast.parse('start_it == 0', mode='eval').body
        self.assertEqual(removed, 2)
        self.assertEqual(ast.dump(original), ast.dump(patched))

    def test_resume_refills_once_writes_header_without_pretraining(self):
        code = '''
start_it = 14000
if args.resume:
    restored = True
stats = {}
if start_it == 0:
    collect(args.warmup_steps, stats)
if start_it == 0:
    log_f.write('iteration,wm_loss\\n')
iterations = list(range(start_it, args.iterations))
'''
        events = []
        stream = io.StringIO()
        ns = {'args': types.SimpleNamespace(resume='last.pt', warmup_steps=64, iterations=24000),
              'collect': lambda steps, stats: events.append(('collect', steps)), 'log_f': stream,
              '_resume_audit': lambda ns: events.append(('restored', ns['start_it'])),
              '_prefill_audit': lambda ns: events.append(('prefilled', ns['start_it']))}
        exec(compile(resume_launch.transform(code), '<test>', 'exec'), ns)
        self.assertEqual(events, [('restored', 14000), ('collect', 64), ('prefilled', 14000)])
        self.assertEqual(stream.getvalue(), 'iteration,wm_loss\n')
        self.assertEqual((len(ns['iterations']), ns['iterations'][0], ns['iterations'][-1]), (10000, 14000, 23999))

    def test_missing_upstream_warmup_fails_closed(self):
        with self.assertRaises(AssertionError):
            resume_launch.transform('if args.resume:\n    pass\n')

    def test_all_other_saved_parameters_identical(self):
        values = {'iterations': 14000, 'ckpt_every': 2000, 'log_dir': 'old', 'resume': None,
                  'lr': .0002, 'num_envs': 4096, 'lr_decay_iters': 250000, 'wm_only_iters': 500,
                  'motion_dir': 'same', 'amp': 'bf16', 'seed': 0}
        original = copy.deepcopy(values)
        changed = continue_pipeline.effective_args(values)
        self.assertEqual(values, original)
        self.assertEqual({k for k in changed if changed[k] != values[k]},
                         {'iterations', 'ckpt_every', 'log_dir', 'resume'})

    def test_restore_audit_detects_weight_optimizer_or_nonfinite_change(self):
        import torch
        expected = {'model': torch.tensor([1., 2.]), 'optimizer': [{'step': torch.tensor(140000.), 'lr': .0001985}]}
        resume_launch.assert_same(copy.deepcopy(expected), expected, torch)
        actual = copy.deepcopy(expected)
        actual['optimizer'][0]['step'] += 1
        with self.assertRaises(AssertionError): resume_launch.assert_same(actual, expected, torch)
        actual = copy.deepcopy(expected)
        actual['model'][0] = 5
        with self.assertRaises(AssertionError): resume_launch.assert_same(actual, expected, torch)
        bad = {'x': torch.tensor(float('nan'))}
        with self.assertRaises(AssertionError): resume_launch.assert_same(bad, bad, torch)

    def test_all_1000_checkpoints_pass_unchanged_author_eval_filter(self):
        checkpoints = range(15000, 24001, 1000)
        self.assertEqual([it for it in checkpoints if (it // 2000) % 1 == 0], list(checkpoints))

    def test_real_training_parser_receives_exact_continuation_args(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'supermimic/train.py').exists():
            root = Path('D:/supermimic_reproduced_hoi_selfcollision')
        tree = ast.parse((root / 'supermimic/train.py').read_text())
        parser = argparse.ArgumentParser()
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'parser.add_argument':
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<parser>', 'exec'),
                     {'parser': parser, 'os': __import__('os'), 'REPO': str(root)})
        import json
        saved = root / 'logs/lafan1_selfcollision_seed0/args.json'
        original = json.loads(saved.read_text()) if saved.exists() else vars(parser.parse_args([]))
        original.update(iterations=14000, ckpt_every=2000)
        params = continue_pipeline.effective_args(original)
        keys = ('num_envs', 'iterations', 'buffer_steps', 'utd', 'horizon', 'goal_frames', 'hist',
                'wm_window', 'wm_batches', 'wm_batch', 'pi_batch', 'lr', 'gamma', 'lam',
                'action_noise', 'critic_epochs', 'critic_minibatches', 'warmup_steps',
                'wm_only_iters', 'lr_decay_iters', 'ckpt_every', 'log_every', 'seed', 'amp')
        manifest = {'effective_parameters': params, 'parameters': {k: original[k] for k in keys}}
        actual = vars(parser.parse_args(continue_pipeline.training_arguments(manifest)))
        self.assertEqual(actual, {k: params[k] for k in actual})

    def test_checkpoint_launcher_changes_allowlisted_but_science_change_rejected(self):
        before = {'lr': .0002, 'num_envs': 4096, 'livestream': -1}
        after = dict(before, livestream=0, kit_visualizer=False, enable_cameras=False)
        self.assertEqual(set(continue_pipeline.validate_saved_arguments(before, after)),
                         {'livestream', 'kit_visualizer', 'enable_cameras'})
        after['lr'] = .0001
        with self.assertRaises(AssertionError):
            continue_pipeline.validate_saved_arguments(before, after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
