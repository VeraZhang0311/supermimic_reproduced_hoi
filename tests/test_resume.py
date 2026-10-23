"""Exercise the real training entry's fresh-replay and new-file header branches."""
import ast
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


class ResumeTests(unittest.TestCase):
    def branch(self, predicate):
        source = Path(__file__).resolve().parents[1] / 'supermimic/train.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and predicate(node)]
        self.assertEqual(len(matches), 1)
        return compile(ast.Module(body=matches, type_ignores=[]), str(source), 'exec')

    def test_loaded_checkpoint_still_fills_fresh_replay(self):
        branch = self.branch(lambda n: ast.unparse(n.body[0]) == 'collect(args.warmup_steps, stats)')
        calls = []
        exec(branch, {'start_it': 14000, 'args': SimpleNamespace(resume='saved.pt', warmup_steps=64),
                      'stats': {}, 'collect': lambda steps, stats: calls.append(steps)})
        self.assertEqual(calls, [64])

    def test_resumed_log_gets_one_header_in_its_new_directory(self):
        branch = self.branch(lambda n: ast.unparse(n.body[0]).startswith("log_f.write('iteration,"))
        stream = io.StringIO()
        exec(branch, {'log_f': stream})
        exec(branch, {'log_f': stream})
        self.assertEqual(stream.getvalue().count('iteration,'), 1)


if __name__ == '__main__':
    unittest.main()
