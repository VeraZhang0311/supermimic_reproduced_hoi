# HOI v1.1 cluster workflow

The default entrypoint is `cluster/hoi_v1_1_combined.sbatch.sh`: one L40,
24 CPUs, 96 GiB host memory and a 96-hour allocation. It runs native DR checks
and the production-size backward-only capacity probe first, then starts a new
Python process for from-scratch 20k training. Validation has a 7200-second
timeout. Failure exits the job before training. Periodic canonical HOI80
evaluation remains a separate job with a two-hour limit.

The reusable Python helpers live here, rather than in ignored `data/` or `tmp/`:

| Module | Purpose |
| --- | --- |
| `stage` | Verify frozen files, config, observation contract and source identity |
| `validate` | Eight assets, training DR and nominal evaluation, then the capacity probe |
| `capacity_probe` | Full replay/prefill, WM/actor/critic backward checks; zero optimizer steps |
| `pipeline` | Check one allocation/GPU and successful evidence before handing off |
| `train` | Recheck evidence, run preflight, then start fresh training |

Use a new independent checkout, output and operations directory for each run.
The caller supplies these non-secret environment variables:

```text
SUPERMIMIC_ROOT
SUPERMIMIC_RUNTIME
ISAACLAB_PATH
OPERATIONS_ROOT
EXPECTED_STAGE_SHA256
EXPECTED_OPERATIONS_SHA256
```

`SUPERMIMIC_ROOT/data/v11_staging.json` is generated input, not source code.
It binds `root`, `files` (relative paths to SHA256, including these helpers),
`source_identity`, `production_config`, `actor_observation_contract`,
`train_bundle`, `eval_bundle`, `base_native_evidence`, `robot_mjcf`, `runtime`
and `isaaclab`. Build the config with `hoi.v1.train.read_config` and the contract
with `hoi.v1.observations.observation_contract`; record the actual immutable
bundle and asset bindings rather than copying paths from another experiment.

`OPERATIONS_ROOT/operations.json` binds its absolute `root`, the stage file's
`stage_sha256`, and a `files` SHA256 map for any auxiliary operations files.
`pipeline` creates exclusive phase receipts below `OPERATIONS_ROOT/run/`.
Neither manifest nor phase receipts, evidence, logs or checkpoints belong in Git.

The common runtime setup preserves both `expandable_segments:True` allocator
variables and OMP/MKL=3. W&B is disabled during validation; training uses a fresh
private run and only the repository's curated metrics. Supply the existing
tracking overlay and destination through the non-secret environment. Never
export credentials into Slurm submissions.

Standalone entrypoints `hoi_v1_1_validate.sbatch.sh` and
`hoi_v1_1_train.sbatch.sh` are also available. They use the same manifests and
hard checks. The combined entrypoint avoids a second training queue.

Packaging these entrypoints does not modify an already frozen or queued job.
The historical run's exact helpers and hashes remain in its immutable snapshot;
use the packaged modules for a newly staged run.

CPU checks:

```text
python -m unittest discover -s tests -v
python -m unittest hoi.test_reference hoi.test_hand_collision -v
```
