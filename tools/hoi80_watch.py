"""Submit one canonical HOI80 evaluation per scheduled absolute PPO iteration."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

from gmt_auto_eval import Slurm, TERMINAL, atomic_json, exclusive_lock, sha256


def milestones(train, interval=2000, after_iteration=0):
    if type(interval) is not int or interval < 1 or type(after_iteration) is not int or after_iteration < 0:
        raise ValueError("HOI80 interval must be positive and its exclusive starting iteration nonnegative")
    for path in sorted(train.glob("checkpoint_*.pt")):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match and (iteration := int(match[1])) > after_iteration and iteration % interval == 0:
            yield iteration, path


def complete(path):
    # torch.save writes the ZIP central directory last. Ignore a checkpoint
    # still being written, then bind the dispatched work to its file hash.
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("/data.pkl") for name in archive.namelist())
    except (FileNotFoundError, zipfile.BadZipFile):
        return False


def dispatch(root, job, code, work, slurm, *, interval=2000, after_iteration=0, expected_training=None):
    for iteration, checkpoint in milestones(root / f"train_{job}", interval, after_iteration):
        target = work / f"iter_{iteration:06d}"
        target.mkdir(exist_ok=True)
        request = target / "dispatch.json"
        if request.exists():
            record = json.loads(request.read_text())
            if record.get("eval_job"):
                continue
            matches = slurm.find_jobs(record["name"], record["created"])
            if len(matches) > 1:
                raise RuntimeError(f"Multiple submissions for {record['name']}: {matches}")
            if not matches:
                # A submitter can exit between sbatch acceptance and recording
                # its job ID. Allow for accounting delay without another submit.
                continue
            record["eval_job"] = matches[0]
        else:
            if not complete(checkpoint):
                continue
            name = f"hoi80-{job}-{iteration}"
            record = {"iteration": iteration, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
                      "train_job": job, "name": name, "created": time.time(),
                      "interval": interval, "after_iteration": after_iteration}
            if expected_training is not None:
                record["expected_training"] = expected_training
            atomic_json(request, record)
            try:
                reply = subprocess.run(["sbatch", "--parsable", f"--job-name={name}",
                    f"--output={target}/slurm-%j.log", str(code / "cluster/hoi_residual_hoi80_eval.sbatch.sh"),
                    str(root), str(code), str(request)], check=True, capture_output=True, text=True, timeout=60).stdout.strip()
            except subprocess.TimeoutExpired:
                print(f"Lost Slurm submit reply for {name}; reconciling by job name", flush=True)
                continue
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
    parser.add_argument("--interval", type=int, help="absolute update cadence; default 2000 or saved schedule")
    parser.add_argument("--after-iteration", type=int, help="exclusive lower bound; defaults to the restored update")
    args = parser.parse_args()
    work = args.root / f"hoi80_{args.job}"
    train_config = args.root / f"train_{args.job}/rank_00/config.json"
    slurm = Slurm()
    with exclusive_lock(work):
        while True:
            try:
                cfg = json.loads(train_config.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                if slurm.status(args.job)["state"] in TERMINAL:
                    raise RuntimeError("Training terminated before its configuration became available")
                time.sleep(15)
        started_after = (cfg.get("resume") or {}).get("iteration", 0)
        schedule_path = work / "watcher.json"
        previous = json.loads(schedule_path.read_text()) if schedule_path.is_file() else {}
        interval = args.interval if args.interval is not None else previous.get("interval", 2000)
        boundary = args.after_iteration if args.after_iteration is not None else previous.get("after_iteration", started_after)
        if interval < 1 or boundary < started_after or boundary < 0:
            raise ValueError("Invalid HOI80 cadence or boundary before the restored checkpoint")
        if (previous.get("train_job", args.job) != args.job or previous.get("interval", interval) != interval or
                previous.get("after_iteration", boundary) != boundary):
            raise ValueError("Cannot silently change an existing HOI80 dispatch schedule")
        training = {"reward_profile": cfg.get("experiment", {}).get("reward_profile", "resmimic"),
                    "epa_horizon": cfg["capacity"].get("epa_horizon", 24)}
        if previous.get("expected_training", training) != training:
            raise ValueError("HOI80 dispatch ledger belongs to a different training configuration")
        options = {"interval": interval, "after_iteration": boundary, "expected_training": training}
        atomic_json(schedule_path, {"train_job": args.job, **options, "code": str(args.code),
                                    "first_milestone": (boundary // interval + 1) * interval, "started": time.time()})
        while True:
            dispatch(args.root, args.job, args.code, work, slurm, **options)
            try:
                status = slurm.status(args.job)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                status = {"state": "UNKNOWN"}
            if status["state"] in TERMINAL:
                dispatch(args.root, args.job, args.code, work, slurm, **options)
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
