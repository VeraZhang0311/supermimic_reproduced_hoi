"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--dataset", choices=["lafan1", "boneseed"], default="lafan1", help="default: complete 40-clip LAFAN1 G1 set")
parser.add_argument("--motion_dir", type=str, default=None, help="default: data/lafan1_g1; BONES option uses data/boneseed_g1")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "lafan1_privileged"))
parser.add_argument("--seed", type=int, default=0)
initialization = parser.add_mutually_exclusive_group()
initialization.add_argument("--resume", type=str, default=None)
initialization.add_argument("--transfer_from", type=str, default=None,
                            help="full-state cross-dataset fine-tune into a new run with fresh local counters")
parser.add_argument("--transfer_parent_iteration", type=int, default=None,
                    help="require this exact parent iteration for --transfer_from")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
add_launcher_args(parser)
args = parser.parse_args()
if args.transfer_parent_iteration is not None and args.transfer_from is None:
    parser.error("--transfer_parent_iteration requires --transfer_from")
if args.motion_dir is None:
    args.motion_dir = os.path.join(REPO, "data", "lafan1_g1" if args.dataset == "lafan1" else "boneseed_g1")
args.motion_dir = os.path.abspath(args.motion_dir)

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state, clamp_action
from supermimic.motion import load_reference
from supermimic.privileged import PRIVILEGED_DIM, PrivilegedFeatures, RobotKinematics38, checkpoint_contract, validate_checkpoint
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.actuator_state import ACTUATOR_DIM, advance_actuator_state, normalize_actuator_state

if args.transfer_from:
    from supermimic.gmt_transfer import normalize_transfer_launcher_args
    # KitLauncher records resolved display defaults in saved checkpoint args.
    # Set the equivalent explicit disabled values before strict transfer checks.
    vars(args).update(normalize_transfer_launcher_args(vars(args)))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
if not args.resume and os.path.exists(os.path.join(args.log_dir, "args.json")):
    raise ValueError("run already exists; use a new log_dir or explicit matching resume")
os.makedirs(args.log_dir, exist_ok=True)

kin = G1Kinematics(dev)
motion = load_reference(args.dataset, args.motion_dir, kin, dev)
print(f"[gmt] reference dataset={motion.dataset}, clips={len(motion.names)}, frames={motion.num_frames}; "
      f"CUDA allocated={torch.cuda.memory_allocated()/2**30:.3f} GiB", flush=True)
teacher_kin = RobotKinematics38(device=dev)
privileged = PrivilegedFeatures(teacher_kin, motion)
gmt_contract = checkpoint_contract(motion, teacher_kin)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
if not bool((motion.clip_len > HZ).any()):
    raise ValueError("no motion is long enough for the configured imagination horizon")
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM + ACTUATOR_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh, privileged_dim=PRIVILEGED_DIM).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
transfer, lr_iteration_offset = None, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev, weights_only=True)
    validate_checkpoint(ck, gmt_contract)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    transfer = ck.get("transfer")
    lr_iteration_offset = 0 if transfer is None else transfer["learning_rate_iteration_offset"]
elif args.transfer_from:
    from supermimic.gmt_transfer import load_transfer_checkpoint
    initial = load_transfer_checkpoint(args.transfer_from, gmt_contract, vars(args),
        wm=wm, pi=pi, critic=critic, opt_wm=opt_wm, opt_pi=opt_pi, opt_v=opt_v,
        run_dir=args.log_dir, additional_iterations=args.iterations,
        expected_parent_iteration=args.transfer_parent_iteration, map_location="cpu")
    start_it, wall_offset, sim_steps = initial["iteration"], initial["wall_time"], initial["sim_steps"]
    transfer, lr_iteration_offset = initial["transfer"], initial["learning_rate_iteration_offset"]
    json.dump(transfer, open(os.path.join(args.log_dir, "transfer.json"), "w"), indent=2)
    print(f"[gmt] full-state transfer from iteration {transfer['parent_iteration']}; "
          f"new updates={args.iterations}, wm_only_iters={args.wm_only_iters}, "
          f"LR schedule offset={lr_iteration_offset}", flush=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)
json.dump(gmt_contract, open(os.path.join(args.log_dir, "gmt_contract.json"), "w"), indent=2)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi, actuator):
    S = unflatten_state(s[:, 0])
    context = actuator[:, 0]
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]), context)
        context = advance_actuator_state(context, a[:, j], xi[:, j])
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, q_prev, qd_prev, actuator):
    a = clamp_action(pi(o_hist, g, privileged(S, f, a_prev, xi, q_prev, qd_prev, actuator)))
    S1 = wm.step(S, a, xi_n, actuator)
    actuator1 = advance_actuator_state(actuator, a, xi)
    f1 = f + 1
    o1 = mdp.obs(S1, a, xi, f1)
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), S[4], S[5], actuator1


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g, actuator):
    return torch.cat((mdp.critic_features(S, f, xi_n), g, normalize_actuator_state(actuator)), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"], w["actuator"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    q_prev, qd_prev = s["q_prev"], s["qd_prev"]
    actuator = s["actuator"]
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs, Actuators = [], [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g, q_prev, qd_prev, actuator = imag_step(S, o_hist, a_prev, f, xi, xi_n, g, q_prev, qd_prev, actuator)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        Actuators.append(actuator)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g, actuator))
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            actuator_c = torch.cat((s["actuator"].unsqueeze(0), torch.stack(Actuators).detach()), 0).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c, actuator_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
    print(f"[gmt] native collection N={args.num_envs}, replay_steps={args.buffer_steps}, H={HZ}; "
          f"CUDA allocated={torch.cuda.memory_allocated()/2**30:.3f} GiB", flush=True)
    obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
    q_prev, qd_prev = (x.clone() for x in env.state()[4:6])

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps, q_prev, qd_prev
        for _ in range(n_steps):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            fresh = env.ep_step == 0
            q_prev = torch.where(fresh[:, None], S[4], q_prev)
            qd_prev = torch.where(fresh[:, None], S[5], qd_prev)
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = motion.goal(env.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            direct = privileged(S, env.f, env.a_prev, env.xi, q_prev, qd_prev, env.actuator_state())
            pi.privileged_norm.update(direct)
            a = pi(obs_hist, g, direct)
            a = clamp_action(a + args.action_noise * torch.randn_like(a))
            out = env.step(a)
            out["q_prev"], out["qd_prev"] = q_prev, qd_prev
            buffer.add(out)
            q_prev, qd_prev = S[4].clone(), S[5].clone()
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"]), out["actuator"]))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
    # Checkpoints do not contain replay/simulator/RNG; resume needs fresh prefill.
    collect(args.warmup_steps, stats)
    log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
    if start_it == 0:
        log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
    for it in range(start_it, args.iterations):
        lr = lr_at(it + lr_iteration_offset)
        for o in (opt_wm, opt_pi, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(args.wm_batches):
            l, e8 = wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv = policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        if it == start_it:
            print(f"[gmt] first update complete: local_iteration={it+1}, wm_only_iters={args.wm_only_iters}; "
                  f"CUDA peak allocated={torch.cuda.max_memory_allocated()/2**30:.3f} GiB, "
                  f"peak reserved={torch.cuda.max_memory_reserved()/2**30:.3f} GiB", flush=True)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                  f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                  "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "gmt_contract": gmt_contract}  # fmt: skip
            if transfer is not None:
                ck["transfer"] = transfer
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "gmt_contract")}
            if transfer is not None:
                small["transfer"] = transfer
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
