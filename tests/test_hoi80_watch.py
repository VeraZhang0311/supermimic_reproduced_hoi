import json
from contextlib import nullcontext
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
    def test_resume_1800_persists_1000_cadence_and_binds_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, work = root/"train_123", root/"hoi80_123"
            (train/"rank_00").mkdir(parents=True)
            work.mkdir()
            cfg = {"resume":{"iteration":1800}, "experiment":{"reward_profile":"hoi_v1_1"},
                   "capacity":{"nconmax":512, "njmax":2048, "epa_horizon":64}}
            (train/"rank_00/config.json").write_text(json.dumps(cfg))
            for index in (1000, 1800, 1900, 2000, 2100, 3000):
                with zipfile.ZipFile(train/f"checkpoint_{index:06}.pt", "w") as archive:
                    archive.writestr("archive/data.pkl", b"checkpoint")
            scheduler = Mock()
            scheduler.status.return_value = {"state":"COMPLETED"}
            argv = ["hoi80_watch.py", str(root), "123", str(root)]
            with patch.object(hoi80_watch, "Slurm", return_value=scheduler), \
                    patch.object(hoi80_watch, "exclusive_lock", return_value=nullcontext()), \
                    patch.object(hoi80_watch.subprocess, "run", return_value=Mock(stdout="987;cluster")) as submit:
                with patch.object(sys, "argv", argv+["--after-iteration", "1800", "--interval", "1000"]):
                    hoi80_watch.main()
                with patch.object(sys, "argv", argv):
                    hoi80_watch.main()
                with patch.object(sys, "argv", argv+["--interval", "2000"]), self.assertRaisesRegex(ValueError, "schedule"):
                    hoi80_watch.main()
                with patch.object(sys, "argv", argv+["--after-iteration", "0"]), self.assertRaisesRegex(ValueError, "restored"):
                    hoi80_watch.main()
                self.assertEqual(submit.call_count, 2)
            self.assertEqual(sorted(x.name for x in work.glob("iter_*")), ["iter_002000", "iter_003000"])
            schedule = json.loads((work/"watcher.json").read_text())
            self.assertEqual((schedule["interval"],schedule["after_iteration"],schedule["first_milestone"]), (1000,1800,2000))
            for record in work.glob("iter_*/dispatch.json"):
                self.assertEqual(json.loads(record.read_text())["expected_training"], {"reward_profile":"hoi_v1_1","epa_horizon":64})

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
