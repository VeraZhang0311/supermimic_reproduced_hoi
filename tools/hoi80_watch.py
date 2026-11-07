"""Submit an independent Slurm evaluation for each complete 2000-iteration PPO checkpoint."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

from gmt_auto_eval import Slurm, TERMINAL, atomic_json, exclusive_lock, sha256


def milestones(train):
    for path in sorted(train.glob("checkpoint_*.pt")):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match and (iteration := int(match[1])) > 0 and iteration % 2000 == 0:
            yield iteration, path


def complete(path):
    # torch.save writes the ZIP central directory last. Ignore a checkpoint
    # still being written, then bind the dispatched work to its file hash.
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("/data.pkl") for name in archive.namelist())
    except (FileNotFoundError, zipfile.BadZipFile):
        return False


def dispatch(root, job, code, work, slurm):
    for iteration, checkpoint in milestones(root / f"train_{job}"):
        target = work / f"iter_{iteration:06d}"
        target.mkdir(exist_ok=True)
        request = target / "dispatch.json"
        if request.exists():
            record = json.loads(request.read_text())
            if record.get("eval_job"):
                continue
            matches = slurm.find_jobs(record["name"], record["created"])
            if len(matches) != 1:
                raise RuntimeError(f"Reconcile prior submission for {record['name']}: {matches}")
            record["eval_job"] = matches[0]
        else:
            if not complete(checkpoint):
                continue
            name = f"hoi80-{job}-{iteration}"
            record = {"iteration": iteration, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
                      "train_job": job, "name": name, "created": time.time()}
            atomic_json(request, record)
            reply = subprocess.run(["sbatch", "--parsable", f"--job-name={name}",
                f"--output={target}/slurm-%j.log", str(code / "cluster/hoi_residual_hoi80_eval.sbatch.sh"),
                str(root), str(code), str(request)], check=True, capture_output=True, text=True).stdout.strip()
            job_id = reply.split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unrecognized Slurm reply for {name}")
            record["eval_job"] = job_id
        atomic_json(request, record)
        print(f"Iteration {iteration}: evaluation job {record['eval_job']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("job", type=int)
    parser.add_argument("code", type=Path)
    args = parser.parse_args()
    work = args.root / f"hoi80_{args.job}"
    if not (args.root / f"train_{args.job}/rank_00/config.json").is_file():
        raise FileNotFoundError("Training directory missing")
    slurm = Slurm()
    with exclusive_lock(work):
        atomic_json(work / "watcher.json", {"train_job": args.job, "interval": 2000,
                                             "code": str(args.code), "started": time.time()})
        while True:
            dispatch(args.root, args.job, args.code, work, slurm)
            try:
                status = slurm.status(args.job)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                status = {"state": "UNKNOWN"}
            if status["state"] in TERMINAL:
                dispatch(args.root, args.job, args.code, work, slurm)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
