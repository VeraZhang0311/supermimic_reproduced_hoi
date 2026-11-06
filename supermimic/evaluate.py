"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=None, help="default: training args.json motion_dir")
parser.add_argument("--eval_envs", type=int, default=256, help="batch full-library evaluation without creating one environment per clip")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP, clamp_action
from supermimic.motion import load_reference
from supermimic.privileged import PRIVILEGED_DIM, PrivilegedFeatures, RobotKinematics38, checkpoint_contract, validate_checkpoint
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
if args.eval_envs <= 0:
    raise ValueError("eval_envs must be positive")
with open(os.path.join(args.run_dir, "args.json")) as stream:
    training_args = json.load(stream)
if args.motion_dir is None:
    args.motion_dir = training_args["motion_dir"]
motion = load_reference(training_args["dataset"], args.motion_dir, kin, dev)
teacher_kin = RobotKinematics38(device=dev)
privileged = PrivilegedFeatures(teacher_kin, motion)
gmt_contract = checkpoint_contract(motion, teacher_kin)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    validate_checkpoint(ck, gmt_contract)
    pi = Policy(OBS_DIM, 73 * K, hist=Hh, privileged_dim=PRIVILEGED_DIM).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    if record and C > env.N:
        raise ValueError("recording requires eval_envs >= clip count; evaluate the full large library without --record")
    ids = torch.arange(env.N, device=dev)
    ret_all, failed_all = [], []
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    for start in range(0, C, env.N):
        count = min(env.N, C-start)
        clips = (start + ids).clamp(max=C-1)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start[clips].clone())
        env.refresh()
        env.done_eval[count:] = True
        clip_end = motion.clip_end[clips]
        ret = torch.zeros(env.N, device=dev)
        hist = None
        q_prev, qd_prev = (x.clone() for x in env.state()[4:6])
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            direct = privileged(S, env.f, env.a_prev, env.xi, q_prev, qd_prev, env.actuator_state())
            a = clamp_action(pi(hist, motion.goal(env.f, K), direct))
            q_prev, qd_prev = S[4].clone(), S[5].clone()
            alive = ~env.done_eval.clone()
            if record:
                from supermimic.rotations import mat_to_quat
                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            out = env.step(a)
            ret += torch.where(alive, out["r"], 0.0)
            env.f = torch.minimum(env.f, clip_end - 1)  # keep f+1 in range for finished lanes
        ret_all.append(ret[:count])
        failed_all.append(env.failed_eval[:count].clone())
    ret, failed = torch.cat(ret_all), torch.cat(failed_all)
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(failed.sum().item()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, min(C, args.eval_envs), motion, mdp, eval_mode=True)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)
