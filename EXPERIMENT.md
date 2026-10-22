# G1 self-collision comparison

Baseline: `/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/logs/lafan1_stock_seed0` (self-collision OFF).
This experiment: `/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/logs/lafan1_selfcollision_seed0` (ON).

The only scientific-source edit against upstream a1508d42c9d1699a52e8233ab16d08c488441ab9 is
`supermimic/robot.py`: `enabled_self_collisions=False` becomes `True`.
No self-collision reward penalty is added. Reward, observation, termination, model architecture,
normalization, data, seed, and all training hyperparameters remain identical to the OFF baseline.
Both jobs request one L40 with 24 CPUs / 96 GB RAM, 4096 environments, 500 WM warm-up updates,
14000 formal updates and full 40-motion evaluations every 2000 updates. Each starts from scratch.

The same pinned data, generated USD assets, IsaacLab checkout and installed CUDA12.8 environment
are borrowed read-only through symbolic links. Logs, replay, checkpoints, caches and receipts are
separate. No installers or asset conversion scripts run in this experiment. Both variants use
the same supported old-driver conditional-CUDA-graph fallback; physical solver parameters stay intact.

Before training, separate OFF and ON diagnostic processes inspect the effective USD attribute,
Newton collision filters, compiled MuJoCo masks/exclusions and finite physical steps. Importer and
MuJoCo exclusions for adjacent or welded bodies remain in effect. Enabling self-collision does not
mean forcing every pair of linked meshes to collide.

W&B private project: `tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction`.
Formal ON run ID: `selfcoll-lafan1-20261022-s0`; OFF: `stock-lafan1-20261022-s0`.
The existing dashboard's formal-run panels include both. Compare full-40 normalized return,
failed clips, actual simulator return and speed against iteration and training wall time.
Only curated config/metrics/evaluation summaries are uploaded; no checkpoints/data/source uploads.

One seed is a controlled preliminary comparison, not an estimate of cross-seed variability.
