"""Two-device training with fixed native lanes and a bounded collection pipeline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from .mixed import ASSETS, CHECKPOINT_SCHEMA, SOURCE_ITERATION
from .table_actions import TABLE_ACTIONS, TABLE_ACTION_LOG_EVERY, episode_metrics


def _models(config, device):
    from .networks import Critic, Policy, WorldModel
    return {"policy": Policy(config.actor_hidden).to(device),
            "wm": WorldModel(config.wm_hidden, config.contact_hidden).to(device),
            "critic": Critic(config.critic_hidden).to(device)}


def _put_json(path, value):
    from .telemetry import atomic_json
    atomic_json(path, value)


class Shapes:
    def __init__(self, points):
        self.bank = points

    def points(self, ids):
        return self.bank[ids]


def _reward_on_device(mdp, asset, hands, native, contacts):
    import torch
    from .geometry import query_grid
    from .reward import TablesReward

    class DeviceReward(TablesReward):
        def object_failure_threshold(self, frames):
            return self.threshold.expand_as(frames)

        def hand_distances(self, bodies, objects, frames):
            bp, bR = bodies[:2]
            world = bp[:, self.wrists, None] + (bR[:, self.wrists, None] @ self.probes.unsqueeze(-1)).squeeze(-1)
            local = (objects[1].transpose(-1, -2)[:, None, None] @
                     (world - objects[0][:, None, None]).unsqueeze(-1)).squeeze(-1)
            return query_grid(*self.grid, local)

    reward = DeviceReward(mdp, [asset], hands, native_hands=native)
    reward._frozen_contacts = contacts
    reward.probes = torch.stack([side["palm_points"] for side in reward.hands.values()]).to(mdp.motion.device)
    geom = reward.geometry[asset["asset_id"]]
    reward.grid = tuple(geom[key].to(mdp.motion.device) for key in ("sdf_grid", "grid_min", "grid_max"))
    reward.threshold = (geom["bbox_diagonal"] * 2.).clamp_min(1.).to(mdp.motion.device)
    return reward


class Collector:
    """The collection thread owns all native scenes, on a single CUDA device."""

    def __init__(self, experiment, output, geometry, samplers, start):
        import torch
        from isaaclab.app import add_launcher_args, launch_simulation
        from supermimic.env import make_sim
        from supermimic.kinematics import G1Kinematics
        from supermimic.mdp import MDP
        from supermimic.privileged import PrivilegedFeatures, RobotKinematics38
        from supermimic.robot import default_joint_pos_list
        from hoi_residual.capacity import capacity_profile, configure_capacity
        from .batched_fk import BatchedFK
        from .data import TableMotionLib
        from .env import TableSim
        from .mixed import MixedMotion
        from .networks import Policy
        from .prepare_contacts import load_contact_cache as load_table_contacts
        from .simulation_slot import SimulationSlots
        from .ultra import UltraTeacherMotion, load_contact_cache as load_ultra_contacts

        self.device = "cuda:0"
        torch.cuda.set_device(self.device)
        torch.cuda.manual_seed(experiment.config.seed)
        self.noise = torch.Generator(device=self.device).manual_seed(experiment.config.seed + 1)
        self.config = experiment.config
        catalogs, hands, points, native = geometry
        self.shapes = Shapes(points.to(self.device))
        self.policy = Policy(self.config.actor_hidden).to(self.device).eval().requires_grad_(False)
        self.kin = BatchedFK(G1Kinematics(self.device), privileged=False)
        self.teacher_kin = BatchedFK(RobotKinematics38(device=self.device), privileged=True)
        self.nominal = torch.tensor(default_joint_pos_list(), device=self.device)
        self.torso_id = self.teacher_kin.body_names.index("torso_link")
        parser = argparse.ArgumentParser()
        add_launcher_args(parser)
        args = parser.parse_args([])
        cfg = make_sim(self.device, args)
        cfg.physics.use_cuda_graph = experiment.execution.get("collector_cuda_graph", True)
        configure_capacity(cfg, capacity_profile(512, 2048, 64))
        self.launcher = launch_simulation(cfg, args)
        self.launcher.__enter__()
        self.slots = None
        self.motions, self.labels, self.envs, self.offsets = [], [], [], []
        self.action_ids = {}
        self.action_totals = torch.zeros((len(TABLE_ACTIONS), 3), device=self.device, dtype=torch.float64)
        try:
            self.slots = SimulationSlots(cfg)
            for index, name in enumerate(ASSETS):
                is_table = index >= 8
                local_assets, binding = catalogs[is_table], hands[is_table]
                paths = experiment.ultra["paths"]
                kwargs = {"asset": name}
                if is_table:
                    path = experiment.root / "tables/manifest.json"
                    motion = TableMotionLib(path, self.kin, self.device, **kwargs)
                    labels = load_table_contacts(experiment.root / "tables/reference_contacts.pt", motion,
                                                 local_assets, binding, native_hands=native)
                    self.action_ids[index] = torch.tensor([TABLE_ACTIONS.index(entry["action"]) for entry in motion.entries],
                                                          device=self.device)
                else:
                    evaluation = experiment.ultra["evaluation"]
                    motion = UltraTeacherMotion(paths["manifest"], self.kin, self.device, **kwargs,
                                                canonical=evaluation["canonical"], helper_root=evaluation["helper_root"])
                    labels = load_ultra_contacts(paths["reference_contacts"], motion, local_assets, binding, native_hands=native)
                self.offsets.append(sum(m.num_frames for m in self.motions))
                self.motions.append(motion)
                self.labels.append(labels[:2])
                self.kin.bind_reference(motion)
                asset = local_assets[index - 8 if is_table else index]
                mdp = MDP(self.kin, motion, self.nominal)
                reward = _reward_on_device(mdp, asset, binding, native, labels[:2])
                folder = output / name
                folder.mkdir(exist_ok=False)
                with self.slots.create() as slot:
                    env = TableSim(slot.cfg, experiment.lanes[index], motion, mdp, asset, folder,
                                   teacher_kin=self.teacher_kin, reward=reward, solver_ls_iterations=200,
                                   native_hands=native)
                    env.iteration = start
                    if samplers[index] is not None:
                        env.sampler.load_state_dict(samplers[index])
                    env.reset(torch.arange(env.N, device=self.device))
                self.envs.append((env, slot))
                print(json.dumps({"phase": "collector_scene_ready", "asset": name, "worlds": env.N}), flush=True)
            self.motion = MixedMotion(self.motions, self.kin, self.device)
            self.teacher_kin.bind_reference(self.motion)
            self.privileged = PrivilegedFeatures(self.teacher_kin, self.motion)
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.slots is not None:
                self.slots.close()
        finally:
            self.launcher.__exit__(None, None, None)

    def sampler_states(self):
        import torch
        states = [{key: value.cpu() if torch.is_tensor(value) else value
                   for key, value in env.sampler.state_dict().items()} for env, _ in self.envs]
        return {"samplers": states, "noise": self.noise.get_state().cpu(),
                "torch": torch.cuda.get_rng_state(self.device).cpu(), "table_action_totals": self.action_totals.cpu()}

    def collect(self, policy_state, iteration, steps):
        import torch
        from hoi_residual.features import oracle_features
        from supermimic.mdp import clamp_action
        from .batched_fk import privileged_call
        from .reward import TERM_NAMES
        from .v11.state import unflatten_object, unflatten_state

        started = perf_counter()
        torch.cuda.set_device(self.device)
        self.policy.load_state_dict(policy_state)
        rows, totals = [], None
        with torch.no_grad():
            embedding = self.policy.shape_encoder(self.shapes.bank)
            for _ in range(steps):
                snapshots = []
                for index, (env, slot) in enumerate(self.envs):
                    with slot.activate():
                        sample = env.snapshot()
                    sample["f"] += self.offsets[index]
                    sample["asset_id"].fill_(index)
                    snapshots.append(sample)
                before = {key: torch.cat([sample[key] for sample in snapshots]) for key in snapshots[0] if key != "partition"}
                S, O = unflatten_state(before["s"]), unflatten_object(before["o"])
                bodies = self.teacher_kin.forward(S)
                robot = privileged_call(self.privileged, S, before["f"], before["a_prev"], before["xi"], before["q_prev"],
                                        before["qd_prev"], before["actuator"], bodies=bodies)
                torso = bodies[0][..., self.torso_id, :], bodies[1][..., self.torso_id, :, :]
                oracle = oracle_features(robot, torso, O, self.motion, before["f"],
                                         before["xi_o"], before["object_history"])
                action = self.policy(oracle, None, shape_embedding=embedding[before["asset_id"]])
                noise = torch.randn(action.shape, device=self.device, generator=self.noise)
                action = clamp_action(action + self.config.action_noise * noise)
                after, stats = [], []
                for index, ((env, slot), local) in enumerate(zip(self.envs, action.split([env.N for env, _ in self.envs]), strict=True)):
                    with slot.activate():
                        env.iteration = iteration
                        result = env.step(local)
                        snapshot = env.snapshot()
                        after.append({"s_next": snapshot["s"], "o_next": snapshot["o"], "actuator_next": snapshot["actuator"],
                                      "next_f": snapshot["f"] + self.offsets[index], "done": result["done"],
                                      "r": result["reward"], "contact": result["contact"], "contact_valid": result["contact_valid"]})
                        ids = result["done"].nonzero().flatten()
                        returns = env.ep_ret[ids] / (env.ep_step[ids] * env.reward_model.max_step_reward)
                        stats.append(torch.stack((result["reward"].sum(), result["terminated"].sum(), result["done"].sum(),
                                                  env.ep_step[ids].sum(), returns.sum(),
                                                  *(result["terms"][name].sum() for name in TERM_NAMES))))
                        if len(ids):
                            if iteration > SOURCE_ITERATION and index in self.action_ids:
                                labels = self.action_ids[index][env.motion.clip_id[env.f[ids]]]
                                values = torch.stack((torch.ones_like(returns), result["terminated"][ids], returns), dim=1)
                                self.action_totals.index_add_(0, labels, values.to(self.action_totals))
                            env.reset(ids)
                row = {**before, "a": action, **{key: torch.cat([sample[key] for sample in after]) for key in after[0]}}
                rows.append(row)
                stats = torch.stack(stats)
                totals = stats if totals is None else totals + stats
        torch.cuda.synchronize(self.device)
        action_metrics = {}
        if iteration > SOURCE_ITERATION and iteration % TABLE_ACTION_LOG_EVERY == 0:
            action_metrics = episode_metrics(self.action_totals.tolist())
            self.action_totals.zero_()
        return rows, totals.cpu().tolist(), perf_counter() - started, action_metrics


def _learner(experiment, models, geometry, collector):
    import torch
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.privileged import RobotKinematics38
    from supermimic.robot import default_joint_pos_list
    from .batched_fk import BatchedFK
    from .learner import Learner
    from .mixed import MixedMotion
    from .replay import ReplayBuffer

    device = "cuda:1"
    (boxes, tables), hands, points, native = geometry
    kin = BatchedFK(G1Kinematics(device), privileged=False)
    teacher = BatchedFK(RobotKinematics38(device=device), privileged=True)
    motion = MixedMotion(collector.motions, kin, device)
    kin.bind_reference(motion)
    teacher.bind_reference(motion)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    assets = boxes + [{**a, "asset_id": i + 8} for i, a in enumerate(tables)]
    contacts = tuple(torch.cat(values).to(device) for values in zip(*collector.labels, strict=True))
    reward = SimpleNamespace(by_asset={
        index: _reward_on_device(mdp, asset, hands[index >= 8], native, contacts)
        for index, asset in enumerate(assets)
    })
    config = experiment.config
    buffer = ReplayBuffer(config.buffer_steps, config.num_envs, device, seed=config.seed, asset_env_counts=experiment.lanes)
    bank = points.to(device)
    return Learner(models["policy"], models["wm"], models["critic"], motion, teacher, reward, config,
                   shapes=Shapes(bank), replay=buffer, asset_points=bank,
                   actor_microbatch=experiment.execution["actor_microbatch"],
                   wm_microbatch=experiment.execution["wm_microbatch"], profile_updates=True,
                   compiled_actor=experiment.execution.get("compiled_actor", False))


def _ingest(learner, rows, update_statistics):
    import torch
    with torch.no_grad():
        for row in rows:
            local = {name: value.to(learner.device, non_blocking=True) for name, value in row.items()}
            if update_statistics:
                features = learner.features(local)
                learner.pi.update_normalizers(features)
                learner.critic.update_normalizers(features)
            learner.observe(local, update_statistics=update_statistics)


def _save(path, iteration, models, learner, collector, executor, identity):
    import torch
    collector_state = executor.submit(collector.sampler_states).result()
    temporary = path.with_suffix(".tmp")
    torch.save({"schema": CHECKPOINT_SCHEMA, "iteration": iteration, "identity": identity,
                **{key: value.state_dict() for key, value in models.items()}, "collector": collector_state,
                "optimizers": {key: value.state_dict() for key, value in learner.optimizers.items()},
                "contact_positive": learner.contact_positive, "contact_negative": learner.contact_negative,
                "learner_rng": torch.cuda.get_rng_state(learner.device), "replay_rng": learner.buffer.generator.get_state(),
                "replay_resume": "fresh replay and native scene reset required"}, temporary)
    temporary.replace(path)


def run(root, output, benchmark, *, collector_cuda_graph=None, compiled_actor=None):
    from .mixed import components, input_hashes, load_source, read_experiment
    from hoi_residual.runtime import configure_warp_forward_only
    configure_warp_forward_only()
    import torch
    import supermimic.networks as shared_networks
    from .evaluate import state_dict_hash
    from .networks import feature_schema
    from .reward import TERM_NAMES

    experiment = read_experiment(root)
    overrides = {key: value for key, value in (("collector_cuda_graph", collector_cuda_graph), ("compiled_actor", compiled_actor))
                 if value is not None}
    if overrides:
        if not benchmark:
            raise ValueError("Execution overrides are only for a disposable benchmark")
        experiment.execution = {**experiment.execution, **overrides}
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 2:
        raise RuntimeError("Mixed teacher requires two scheduler-visible CUDA devices")
    torch.set_num_threads(1)
    torch.cuda.set_device(1)
    torch.manual_seed(experiment.config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    shared_networks.AMP_DTYPE = torch.bfloat16
    output.mkdir(parents=True, exist_ok=not benchmark)
    if (output / "identity.json").exists() or (output / "metrics.jsonl").exists():
        raise FileExistsError("Mixed training requires a fresh output directory")
    geometry = components(experiment)
    models = _models(experiment.config, "cuda:1")
    samplers, lineage = load_source(experiment, models, geometry[-1])
    identity = {"profile": "ultra_tables", "assets": list(ASSETS), "lanes": list(experiment.lanes),
                "ultra_initialization": lineage, "native_hands": geometry[-1].identity,
                "global_config": asdict(experiment.config), "solver_ls_iterations": 200,
                "features": feature_schema(), "inputs": input_hashes(experiment), "execution": experiment.execution}
    _put_json(output / "identity.json", identity)
    if any(state_dict_hash(model.state_dict()) != lineage["model_state_sha256"][key] for key, model in models.items()):
        raise RuntimeError("Initialized network tensors differ from the Ultra checkpoint")
    _put_json(output / "initialization.json", {"passed": True, "optimizer_state": "fresh", "initial_iteration": SOURCE_ITERATION,
                                              "rsi": "restored Ultra bins, new table bins"})
    begin = SOURCE_ITERATION + 1
    last = SOURCE_ITERATION + benchmark if benchmark else experiment.config.iterations
    overlap = experiment.execution["overlap_collection"]
    timings = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        collector = executor.submit(Collector, experiment, output, geometry, samplers + [None, None], SOURCE_ITERATION).result()
        try:
            learner = _learner(experiment, models, geometry, collector)

            def submit(iteration, steps):
                # One immutable policy/RMS version per collection segment. The copy
                # is enqueued before the learner can modify its own parameter storage.
                state = {key: value.detach().to("cuda:0", copy=True) for key, value in models["policy"].state_dict().items()}
                torch.cuda.synchronize("cuda:0")
                return executor.submit(collector.collect, state, iteration, steps)

            prefill_seconds = 0.
            prefill_steps = experiment.config.warmup_steps
            for first in range(0, prefill_steps, 16):
                count = min(16, prefill_steps - first)
                rows, _, elapsed, _ = submit(SOURCE_ITERATION, count).result()
                _ingest(learner, rows, update_statistics=False)
                del rows
                prefill_seconds += elapsed
                print(json.dumps({"phase": "mixed_prefill_progress", "steps": first + count,
                                  "seconds": prefill_seconds}), flush=True)
            if any(state_dict_hash(model.state_dict()) != lineage["model_state_sha256"][key] for key, model in models.items()):
                raise RuntimeError("Mixed replay prefill changed initialized network tensors")
            print(json.dumps({"phase": "mixed_prefill_pass", "steps": experiment.config.warmup_steps,
                              "seconds": prefill_seconds, "lanes": list(experiment.lanes)}), flush=True)
            accumulator = 0.
            pending_rows = None
            for iteration in range(begin, last + 1):
                started = perf_counter()
                accumulator += 1 / experiment.config.utd
                steps = int(accumulator)
                accumulator -= steps
                transfer_started = perf_counter()
                future = submit(iteration, steps)
                if pending_rows is None:
                    pending_rows, _, _, _ = future.result()
                _ingest(learner, pending_rows, update_statistics=True)
                sync_seconds = perf_counter() - transfer_started
                metrics = learner.update(iteration)
                wait_started = perf_counter()
                rows, stats, collection_seconds, action_metrics = future.result()
                # Keep one collected segment pending. Its tensors are immutable and
                # will enter learner replay while the next segment is collected.
                pending_rows = rows if overlap and iteration != begin else None
                del rows
                wait_seconds = perf_counter() - wait_started
                transitions = experiment.config.num_envs * steps
                totals = [sum(asset[k] for asset in stats) for k in range(5)]
                metrics.update(collection_seconds=collection_seconds,
                               transfer_and_statistics_seconds=sync_seconds, collection_wait_seconds=wait_seconds,
                               real_steps=transitions, sim_rew=totals[0] / transitions, term_frac=totals[1] / transitions,
                               reset_frac=totals[2] / transitions, resets=int(totals[2]))
                metrics.update(action_metrics)
                metrics.update({f"reward_{name}": sum(row[column] for row in stats) / transitions
                                for column, name in enumerate(TERM_NAMES, start=5)})
                metrics["assets"] = {
                    asset: {"sim_rew": sums[0] / (steps * count), "term_frac": sums[1] / (steps * count),
                            "reset_frac": sums[2] / (steps * count), "real_steps": steps * count,
                            **({"ep_len": sums[3] / sums[2], "ep_ret_norm": sums[4] / sums[2]} if sums[2] else {})}
                    for asset, count, sums in zip(ASSETS, experiment.lanes, stats, strict=True)
                }
                for asset_id, value in metrics.pop("asset_wm_total").items():
                    metrics["assets"][ASSETS[asset_id]]["wm_total"] = value
                if totals[2]:
                    metrics.update(ep_len=totals[3] / totals[2], ep_ret_norm=totals[4] / totals[2])
                if iteration % 50 == 0:
                    generator_state = learner.buffer.generator.get_state()
                    try:
                        validation = learner.wm_validation(learner.buffer.sample_windows(256, 32, split="holdout"))
                        for asset_id, value in validation["asset_h32_object_rmse_m"].items():
                            metrics["assets"][ASSETS[asset_id]]["wm_val_h32_object_position_rmse_m"] = value
                        for horizon, values in validation["horizons"].items():
                            metrics.update({f"wm_val_h{horizon}_{key}": value for key, value in values.items()})
                        for key in ("contact_recall", "contact_precision", "contact_brier"):
                            metrics.update({f"wm_val_{key}_{channel}": value for channel, value in
                                            zip(validation["contact_channels"], validation[key], strict=True)})
                    finally:
                        learner.buffer.generator.set_state(generator_state)
                if not benchmark and (iteration % experiment.config.ckpt_every == 0 or iteration == last):
                    _save(output / f"checkpoint_{iteration:06d}.pt", iteration, models, learner, collector, executor, identity)
                metrics["iteration_seconds"] = perf_counter() - started
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                print(json.dumps(metrics, allow_nan=False), flush=True)
                if benchmark:
                    timings.append(metrics)
            if benchmark:
                _save(output / f"checkpoint_{last:06d}.pt", last, models, learner, collector, executor, identity)
                measured = timings[2:]
                medians = {key: median(row[key] for row in measured) for key in (
                    "iteration_seconds", "collection_seconds", "wm_seconds", "actor_seconds", "critic_seconds",
                    "transfer_and_statistics_seconds", "collection_wait_seconds")}
                elapsed = medians["iteration_seconds"]
                _put_json(output / "benchmark.json", {"passed": True, "iterations": benchmark, "measured_iterations": len(measured),
                          "median_seconds": elapsed, "stage_medians_seconds": medians,
                          "target_seconds": experiment.execution["target_seconds_per_iteration"],
                          "target_met": elapsed <= experiment.execution["target_seconds_per_iteration"],
                          "collector_peak_mib": torch.cuda.max_memory_reserved(0) / 2**20,
                          "learner_peak_mib": torch.cuda.max_memory_reserved(1) / 2**20, "lanes": list(experiment.lanes)})
        finally:
            executor.submit(collector.close).result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", type=int, default=0, help="Run disposable full-size updates in an isolated output")
    parser.add_argument("--collector-cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compiled-actor", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.benchmark and args.benchmark < 5:
        parser.error("Benchmark needs at least five full updates")
    run(args.campaign, args.output, args.benchmark, collector_cuda_graph=args.collector_cuda_graph,
        compiled_actor=args.compiled_actor)


if __name__ == "__main__":
    main()
