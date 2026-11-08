# Privileged tables and Ultra boxes teacher

Branch `hoi-retarget/tables-boxes-v1.0` (renamed from `hoi-retarget/tables-v1.0`) starts at
`codex/hoi-residual-ppo-teacher@78cd5b2c3530f9aa47b0c944436e59d5dc054093`.
The existing `supermimic/` and `hoi_residual/` implementations are unchanged.
The entry point is `python -m hoi_tables.train`. Profiles are
`configs/hoi_tables_v1.json` and `configs/hoi_ultra_boxes_v1.json`.
Both publish periodic evaluation into their owning training W&B run under
`eval/*`; neither creates an evaluation W&B run. The existing tables training
job continues unchanged. Ultra training has not been submitted.

## Scientific contract

- The current privileged GMT hw3 robot, soft hand, motor envelope, action
  scaling, delay, native joint friction and robot randomization are reused.
  Position targets are passed to the inherited runtime once; there is no
  second PD controller or residual/base policy.
- WM, direct privileged actor, critic, PointNets, optimizers and RMS statistics
  start from scratch. No pretrained checkpoint argument is accepted.
- The WM retains v1.1's 225D feature prefix and adds the approved normalized
  88D actuator context, for 313 inputs. It predicts the same 70 robot and
  12 object increments. The actuator state advances analytically.
- The five interval-impulse contact channels are left hand/object, right
  hand/object, object/ground, left foot/object and right foot/object.
  The threshold is strictly greater than 1 N, from all four 5 ms substeps.
  Foot/ground and self contacts are excluded.
- Actor and critic use current 3647D oracle features plus independent learned
  64D metric-shape encoders. The former 29D base action input is removed.
  Imagined features use predicted states and analytic actuator memory.
- The fourteen signed, weighted v1.1 reward terms delegate to the existing
  frozen reward implementation. Reference intent uses exact triangle gaps,
  penetration and intersection-unknown semantics on the current hand hulls.
  Calibrated palm probes are projected onto current immutable hand hulls;
  transforms and projection distances are recorded. Actual contacts come from
  the native recorder. The five prediction heads do not add foot rewards.
- v1.1 robot and bounding-box-dependent object termination are used, including
  the corresponding differentiable termination in imagination.
- Formal configuration enforces 500 WM-only iterations, then fixed joint
  updates without a quality gate. Original H32, layer sizes, batch settings,
  optimizers and UTD 0.4 are retained. Each iteration collects an accumulated
  1/UTD native control steps.
- Tables use two equal asset workers to update one shared model. Gradients are averaged
  before clipping, real-data RMS moments and contact counts are synchronized,
  and final full model/RMS hashes must agree. Global batch/environment counts
  are divided evenly across workers. Shared single-GPU mode uses CPU Gloo
  buffers; ordinary two-GPU mode assigns one device per worker.

## Data

`configs/hoi_tables_selection.json` freezes the original Excel hash and exact
226-ID whitelist: 113 large and 113 small tables. The requested exclusions are
`sub17_smalltable_047`, `sub1_largetable_049`, `sub6_largetable_036`.

The source is the immutable HOI-Retarget OMOMO G1 Parquet snapshot
`b5ac8125e497e6735d6314e63a0e1eedf6229336`. Its declared 30 Hz poses are sampled
once onto the 50 Hz grid. Quaternions are wxyz; joints are reordered by name;
velocities are derived after resampling. There is no trajectory rescaling.
The inherited table hulls already contain the source 0.83 geometry scaling;
their additional runtime scale remains 1.

Preparation `1511810` completed on GRASP and produced
`/mnt/projects/ll/tianzizh/hoi_tables_teacher_20261107_01/input`:

- `manifest.json`: 204 training clips (102 per object), 22 protected evaluation
  clips (11 per object); source-group and duplicate-trajectory partitions are
  disjoint.
- `reference_50hz.pt`: versioned, hash-bound robot/object references and preserved
  source annotations, with T+1 states and T outgoing intervals.
- `assets.json`, `shape_bank.json`, two geometry caches: exact native convex
  meshes, FPS128 and signed-distance fields.

The inherited large/small table masses are 6/4 kg. These are prior explicitly
chosen simulation parameters, with derived uniform-density inertias; they are
not source measurements. Collision components are 13/44 respectively. Only
object assets are reused; old robot labels, splits and native acceptance are
excluded. Robot DR follows current GMT; object DR follows the current branch
(mass scale 0.7–1.3, COM ±1 cm, friction 0.4–0.8).

## RSI and replay

Training starts uniformly through iteration 500. Iterations 501–700 ramp to
50% uniform coverage, 30% recent failures and 20% mastered snippets. The two
objects have equal global allocation. The sampler uses one-second bins,
failure EMA, bounded priority and pre-failure-bin credit. Mastery requires at
least ten adequately observed episodes and both success and tracking quality
EMA at least 0.8. Empty pools return their share to uniform sampling. Sampler
statistics, identity and random generator state are checkpointed.

Replay owns pre-reset final states, pre/next actuator state and object history.
WM windows cannot cross done, episode or asset boundaries. A separate
episode/environment-level diagnostic WM holdout is excluded from WM fitting
and its normalization/contact-balance statistics. Protected reference eval
cannot enter replay or update models, normalizers or curriculum.

## Commands and verification

Prepare new data in a fresh directory using the existing read-only runtime:

```bash
python -m hoi_tables.prepare --selection configs/hoi_tables_selection.json --output NEW_INPUT
```

For repeated training runs, exact reference contacts can be computed once on a
CPU allocation and passed as `--reference-contacts NEW_CONTACT_CACHE.pt`:

```bash
python -m hoi_tables.prepare_contacts --manifest NEW_INPUT/manifest.json \
  --asset-bundle NEW_INPUT/assets.json --output NEW_CONTACT_CACHE.pt
```

This cache covers all 226 reference clips, is keyed by clip ID, and verifies
the current robot, hand/object geometry, implementation and selection bytes.
Without it, the same exact contact targets are cached lazily in memory.
Full offline contact-cache generation has not been launched in this smoke.

Run a bounded GPU smoke through Slurm:

```bash
sbatch cluster/hoi_tables_smoke.sbatch.sh NEW_EXPERIMENT_ROOT FROZEN_SOURCE_ROOT
```

The smoke uses the full default hidden layers and H32, with two asset workers
sharing one L40. Its explicit smoke configuration has 16 total environments,
one WM-only iteration and two joint iterations, reduced batches, and fresh
replay. It checks warmup isolation, actual model parameter updates, exact
cross-worker model/RMS equality, resets, native object readback, collision
identity and contact/constraint overflow. These smoke checkpoints are not a
trained teacher. Formal warmup remains 500.

CPU checks:

```bash
python -m unittest discover -s tests -p 'test_hoi_tables*.py' -v
```

Tests cover timebase and split identities, geometry/cache tampering, all
fourteen reward contributions, five contact identities, terminal bookkeeping,
RSI weighting, replay boundaries, H32 actor gradients, finite differences and
two-worker RMS/gradient synchronization. Native acceptance must come from the
actual GPU report, not CPU mocks. Formal launch receipts are recorded below.

### Verified smoke

- Corrected native job **1512175**: Slurm `COMPLETED`, exit `0:0`, 2m11s,
  NVIDIA L40 (driver 535.216.01), one card shared by two asset workers.
- **63 tables CPU tests** passed both locally and inside the Linux allocation;
  **17 inherited actuator/physics/v1.1 reward tests** also passed locally.
- Full hidden layers, H32, 16 total native environments; one training reference
  per object for the smoke. The complete 226-clip dataset was prepared separately.
- One WM-only update left actor/critic parameters unchanged. Two subsequent
  joint updates changed actual WM, actor and critic parameters, with finite
  metrics. Both workers finished with identical model and RMS hash
  `bede579b533d50366e98aab7da98693a88d04add426cb22e80e87aea212d494b`.
- Large/small workers completed 17/16 resets. Actual native mass, COM, inertia,
  friction, free-joint/gravity, five-channel geometry identity and overflow
  checks passed. Relative inertia errors were 3.42e-7/3.09e-7, below 9.54e-7.
- Current 64-edge EPA allocation was read back as `[4096,64]` per worker.
  All 66 runtime Python source checksum checks passed after the smoke.
- Frozen source: `/mnt/projects/ll/tianzizh/hoi_tables_teacher_20261107_01/source_smoke02`;
  archive SHA256 `f0685c4b26c26ceeba00f82293c40929d94ae0a5ab8f5c8823980e4d21091144`.
- Report: `/mnt/projects/ll/tianzizh/hoi_tables_teacher_20261107_01/smoke_1512175/smoke_report.json`;
  local copy: `.cache/tables_smoke_source02/evidence/smoke_report.json`.

This verifies the small end-to-end execution path, not convergence, held-out
tracking performance or production 4096-environment capacity.

The first native smoke (1512131) exited at inertia readback before optimizer
updates. Entrywise `1e-8` absolute tolerance falsely rejected float32
principal-frame reconstruction near zero off-diagonal elements (observed
maximum errors 1.49e-7 and 3.73e-8 kg m²). The tables adapter therefore checks
each inertia matrix with a Frobenius error bound of eight float32 epsilons
times the expected matrix norm (relative 9.5367e-7). This is a numerical audit
change, not a physical-property change. Mass, COM, friction, gravity,
collider identity and overflow checks remain intact. Full-scale readback now
reconstructs the stored principal tensor in float64 and independently replays
the exact pinned Newton/Warp conversion kernel on the requested tensor in
private temporary arrays. Native versus independently encoded results must
agree within eight float32 epsilons per matrix. Both results must separately
remain within the inherited `2e-5` relative tolerance of the requested physical
tensor in Frobenius norm. This accounts for the fixed four-sweep native `eig3`
approximation without admitting stale or incorrect native writes. Kernel and
Warp SVD implementation hashes are pinned and included in audit evidence.
AST parity verifies every other audit operation is unchanged; no simulator
properties are written. Tests include the captured full-scale failure,
incorrect native tensors inside the physical bound, inaccurate reference
conversions, private-buffer plumbing and 40,000 profile-scaled tensors.

Checkpoints atomically save all models, optimizers, sampler, counts and Torch
RNG state per worker. They do not serialize the simulator or full replay; any
future resume must explicitly refill replay and must not claim bitwise
continuation. A resume CLI is not part of this initial from-scratch entry point.

## Authorized formal run

The subsequent user request authorizes formal training on L40 and registration
in the new private W&B project `hoiretarget-tables`, entity
`tianzizh-university-of-pennsylvania`. This supersedes the implementation-stage
statement above that only smoke was authorized.

The formal launcher requests two L40 GPUs, 24 CPU cores, 96 GiB host memory and
96 hours under `liu-compute / liu-account / ll-med`. It keeps 4096 environments,
the full 204 training clips, all original full-size batches/H32, 20,000 total
iterations and 500 WM-only iterations. Each GPU handles one exact table asset
and half the global batches; gradients and RMS moments remain shared.

Before optimization, a disposable process on the same allocation verifies full
native environment/replay storage, complete WM windows, full H32 actor/critic
backward passes and finite gradients with original Adam steps blocked. Real
Adam moments and conservative scratch space are allocated. Successful evidence
must show zero optimizer updates and unchanged model/RMS/optimizer state; the
process then exits before fresh from-scratch training processes are created.

CPU job `1512284` prepares complete exact reference-contact labels using 16
workers in `formal_inputs01`. Its frozen source and the training source bind
the same contact implementation bytes, reference manifest and robot geometry.

The independent `hoi_tables.telemetry` sidecar uses the existing read-only SDK
and compatibility paths. Only allowed scalar metrics, configuration/schema and
identity hashes are uploaded. Registration uses a fresh run ID; the sole
monitor resumes that ID, combines the two asset logs once per iteration and
records WM warmup, reward components and diagnostic metrics. No weights,
source code or complete rollouts are uploaded.

The first formal preflight (`1512347`, `formal_01`, W&B `bb0bff31`) stopped at
the initial native inertia audit with zero optimizer updates. It exposed a rare
float32 readback error in one of 2048 small-table worlds (relative 1.097e-6).
Its source, logs and failed W&B status are preserved. The precision-only
readback fix uses a fresh frozen source and a new `formal_02` run; the already
verified 226-clip contact cache is reused unchanged (SHA256
`480025137dd6b5a5f4f9022478372a172e6e1603acceae4c2e687f2830488b60`).

The second allocation (`1512383`, `formal_02`, W&B `4d8ab98a`) passed the
complete production-capacity probe with zero optimizer updates. Both workers
used distinct L40 cards and the exact full configuration. Peak Torch allocation
was 16.19 GiB, peak reservation 21.80 GiB; driver readback retained at least
10.79 GB free per card after conservative Adam/scratch reservations. Fresh
formal processes then completed five WM-only iterations, all uploaded to W&B,
before another native inertia diagnostic rejection (relative 1.1705e-6).
The failed run and its metrics are preserved. This isolates the remaining issue
to the native float32 spectral-conversion audit rather than production memory
capacity; the high-precision readback alone does not bound that conversion.

### Current formal submission

After the pinned-kernel audit correction, all **88 tables CPU tests passed**.
The local restricted run could not open its Gloo loopback socket; rerunning
the exact suite outside that restriction passed without test or code changes.
Shared `supermimic/` and `hoi_residual/` remain unchanged.

- Slurm **1512413**, 2×L40, source `source_train03`, output `formal_03`.
- Source archive SHA256
  `385190ced014ff1c9059daf47f967b64f46dc2a3c943037ba0a725fe241c1be7`;
  all 128 frozen files and all seven contact implementation identities verified.
- Private W&B project `hoiretarget-tables`, run `44ad3b72`, name
  `tables_v1-20261107-170411`:
  https://wandb.ai/tianzizh-university-of-pennsylvania/hoiretarget-tables/runs/44ad3b72
- Training job `1512413` is running on two L40s. Its scheduler start time is
  `2026-11-07T17:52:17`, with zero restarts. The complete production capacity
  probe passed before fresh training processes were constructed.
- Same-run server verification found continuous unique training iterations
  1 through 1242, including 73 events from the new unified writer; the run is
  running. This is a verification snapshot, not a live iteration counter.
- Training remains on immutable `source_train03`. Logging is supplied by the
  separate CPU-only source `source_unified01`, job `1513706`; GPU evaluation
  continues to use `source_eval01`. No model-training process was restarted.
- Local submission and registration receipts:
  `.cache/source_train03/evidence/`; cluster receipt: `formal03_submission.json`.

## Periodic evaluation in the owning training run

`hoi_tables.eval_monitor.Supervisor` schedules full protected22 rollouts every
2000 checkpoint iterations, from 2000 through 20000. It waits for both atomic
rank checkpoint files, dispatches one A10 evaluation at a time, and preserves
milestones if evaluation queues fall behind. Submission intentions and result
hashes are durable; lost scheduler replies are reconciled without duplicate
jobs. Failed evaluation stops further dispatch while training telemetry keeps
running. Available milestones are drained after training exits.

Each evaluation covers all 22 protected clips (eleven per table), all **6482
control intervals / 6504 states**, without failure resets. The longest clip
has 411 intervals. Policy actions are deterministic; nominal physical
parameters and seed zero are used. No optimizer, RMS, replay or RSI updates
occur. Complete checkpoint, model/RMS, physics, reward, data and source
identities are checked before and after evaluation. Success records survival
under the existing tables termination rule; tracking metrics are separate.

`hoi_tables.unified_monitor` is the only W&B SDK writer after registration.
It resumes the **owning training run**, merges native training JSONL and
validated evaluation reports, and uses explicit axes `iteration` and
`eval/iteration`. SDK steps are monotonic transport counters; checkpoint
iterations are never used as SDK steps for this merged stream. Semantic event
IDs, durable outbox records, server acknowledgements and the existing writer
locks protect restart/handoff behavior. Evaluation does not overwrite training
stage or status. Tables and Ultra use this same ownership protocol.

For current tables training, all `eval/*` scalars go to run **44ad3b72** in
`hoiretarget-tables`. The project workspace has a manual protected-eval section
with large/small table plots separated and automatic chart creation disabled.
The former independent eval watcher `1513606` was canceled before dispatch;
its empty W&B run `788bbb03` was verified empty and deleted. Its durable
evaluation ledger `protected_eval_2000_01` is reused by unified job `1513706`.
Only the old logging process and its SDK child were stopped during handoff;
the launcher and both original trainer PID/start-time pairs were preserved.

Validated deployment evidence:

- A10 job `1513451` completed the full protected22 rollout at checkpoint 500,
  explicitly marked smoke and excluded from scheduled evaluation publication.
- Policy/RMS SHA256 before and after:
  `f832ad858a7c77ddfa1a51519547dbe15b7966e61ccd01d47dc5bf547f75a824`.
- Aggregate smoke report SHA256:
  `28039232a7559ee0807bb94407f055e8d02dc9295c4822bf25abe53af013cf4b`.
- Unchanged GPU evaluator source `source_eval01`, 134 files; archive SHA256
  `1faeef8089a29c4f16429bb518af556b35ab7fadd5742bc4a4f383c329b7c3f4`;
  executable-tree SHA256
  `2aacd7af53e30f1c9a39b02d7d704165469e9c1c76429a4229c0ce3ee773d949`.
- The logging snapshot passed 31 Linux tests. Server verification checked all
  128 training, 134 evaluator and 137 logging manifest files unchanged.
- Cluster evidence: `protected_eval_attach01/{same_run_server_verified,
  legacy_writer_handoff,empty_eval_run_cleanup}.json`; unified state and outbox
  are in `unified_logging_01`. No scheduled point exists before checkpoint 2000.

## Ultra boxes configuration

`configs/hoi_ultra_boxes_v1.json` selects canonical Ultra **train** data:
**2651 motions / 381 source groups**, with eight exact metric assets:

| Object | Scales |
| --- | --- |
| largebox | 080, 100 |
| plasticbox | 080, 100 |
| smallbox | 080, 100 |
| suitcase | 080, 100 |

`080` means the existing `*_080_080_080` asset and `100` the existing
`*_100_100_100` asset. The adapter binds geometry and shape points by asset
name, checks the existing hull/SDF/FPS128 identities, and applies no additional
mesh scaling or geometry cooking. The established Ultra 60-to-50 Hz sampling
protocol is retained. Tables retain their 30-to-50 Hz protocol.

The current privileged GMT robot, motors, soft hands, joint friction, action
definition, 313D WM input, 82D output, five contact heads, privileged actor /
critic observations, 14 reward terms and optimization settings are unchanged.
Training remains global4096/H32/WM batch2048/policy batch4096, 20k iterations,
500 WM-only iterations and the existing adaptive RSI mixture. The 500-step
training timeout is unchanged.

The eight asset ranks share one model and synchronized statistics. Default
`workers_per_gpu=1` requests eight L40s; an explicit value of two uses four L40s
without changing the eight asset workers or global objective. The complete
production zero-update capacity probe must pass before fresh formal models
are constructed. Ultra capacity and full training have **not** been run.

Reference-contact labels must be recomputed for the current calibrated hands.
The old robot's contact labels cannot be reused. `hoi_tables.ultra` prepares
2651 training references plus 80 evaluation references, with strict data,
robot, geometry and implementation identities. These are reference labels,
not policy rollouts. The configuration names a new cache destination; the
full cache preparation has **not** been submitted.

Every 2000 iterations, `hoi_tables.hoi80_monitor.Supervisor` binds all eight
same-iteration checkpoints and runs the **complete canonical HOI80** panel.
Its eight asset counts are **5,15,6,14,4,16,11,9**, with **23718 controls /
23798 states**. The longest clip has **629 controls** and is evaluated in
full, beyond the training timeout. These 80 source groups belong to Ultra
validation and are a fixed development evaluation panel, not an independent
test set.

`evaluate_hoi80` uses the current direct policy and unchanged reward. It
reuses the pinned canonical initialization and PHC39/object scoring helpers,
not the residual teacher loader. Canonical human/object/joint pass rates are
reported separately from TablesReward survival, under
`eval/hoi80/{human_pass_rate,object_pass_rate,joint_pass_rate}`. Per-asset
metrics use `eval/assets/<exact_asset>/...`. All evaluation data are logged
into the corresponding **Ultra training run**, with `eval/iteration` as axis.
GPU workers have W&B disabled and never register a run.

The manual project workspace includes three canonical HOI80 plots in
`09 Eval - HOI80`, initially collapsed; existing training and protected22
panels are preserved. Automatic panel creation remains disabled.

Final validation: **155 tables/Ultra CPU tests passed**, including owning-run
logging/handoff, all-ten-milestone scheduling, eight-rank input/checkpoint
contracts, full HOI80 duration, source freezing and launch preparation. All
five new/updated scheduler scripts passed `bash -n`. Existing cluster Ultra
geometry, shape and canonical inputs passed read-only binding checks. This
does not constitute Ultra GPU capacity or policy-rollout validation.

Future launch procedure, using an isolated committed checkout on GRASP:

```bash
SRC=/absolute/committed/hoi-retarget-tables
CFG=$SRC/configs/hoi_ultra_boxes_v1.json
CAMPAIGN=/mnt/projects/ll/tianzizh/hoi_direct_ultra_v1_01
prep=$(sbatch --parsable "$SRC/cluster/hoi_ultra_prepare.sbatch.sh" "$SRC" "$CFG")
sbatch --dependency=afterok:$prep --gres=gpu:l40:8 \
  --output=/mnt/projects/ll/tianzizh/hoi_direct_ultra_v1_01-%j.log \
  "$SRC/cluster/hoi_direct_train.sbatch.sh" "$CAMPAIGN" "$SRC" "$CFG"
```

The destination cache and campaign must be new. To select four GPUs, pass
`--export=ALL,HOI_WORKERS_PER_GPU=2 --gres=gpu:l40:4` to the training submission.
`hoi_tables.launch` freezes source, links existing inputs read-only, writes
profile metadata, and rejects existing campaigns. The launcher requires a
healthy owning-run logger **and** evaluation scheduler before starting the
capacity probe. Existing runtimes and active tables sources remain read-only.
These commands are documented entry points; no Ultra training was submitted.
