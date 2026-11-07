import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import hoi80_watch


class WatchTests(unittest.TestCase):
    def test_milestones_backfill_partial_write_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train_123"
            work = root / "watch"
            train.mkdir()
            work.mkdir()
            for index in (500, 2000, 4000):
                with zipfile.ZipFile(train / f"checkpoint_{index:06}.pt", "w") as archive:
                    archive.writestr("archive/data.pkl", b"test archive")
            partial = train / "checkpoint_006000.pt"
            partial.write_bytes(b"PK")
            scheduler = Mock()
            with patch.object(hoi80_watch.subprocess, "run", return_value=Mock(stdout="999;cluster")) as submit:
                for _ in range(2):
                    hoi80_watch.dispatch(root, 123, root, work, scheduler)
                self.assertEqual(submit.call_count, 2)
                self.assertFalse((work / "iter_000500").exists())
                self.assertFalse((work / "iter_006000/dispatch.json").exists())
                request = work / "iter_002000/dispatch.json"
                record = json.loads(request.read_text())
                self.assertEqual((record["eval_job"], record["iteration"]), ("999", 2000))
                record.pop("eval_job")
                request.write_text(json.dumps(record))
                scheduler.find_jobs.return_value = ["999"]
                hoi80_watch.dispatch(root, 123, root, work, scheduler)
                self.assertEqual(submit.call_count, 2)
                self.assertEqual(json.loads(request.read_text())["eval_job"], "999")
                with zipfile.ZipFile(partial, "w") as archive:
                    archive.writestr("archive/data.pkl", b"finished")
                hoi80_watch.dispatch(root, 123, root, work, scheduler)
                self.assertEqual(submit.call_count, 3)


if __name__ == "__main__":
    unittest.main()
