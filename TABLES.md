# Privileged tables and Ultra boxes teacher

Tables training ended after scheduler preemption on2026-11-09; the user chose
not to resume it. Both rank checkpoint8500 files are retained. Automatic
preemption recovery is restricted to the Ultra profile and its existing owner.

Ultra preemption recovery uses `hoi_tables/preemption.py` and the three
`cluster/hoi_preemption_*.sbatch.sh` wrappers, staged separately from the frozen
training/evaluation science. An afterany CPU handler acts only on PREEMPTED;
application errors, OOM, cancellation and completion are never blindly retried.
It audits the latest complete synchronized eight-rank snapshot (or the original
checkpoint500 bundle), submits a held two-L40 attempt, arms its successor
handler, then releases it. All learner/RMS/Adam/RSI/contact/RNG state restores;
simulator/replay recreate with statistics-frozen prefill and no repeated warmup.
Each new GPU allocation passes the existing full-size zero-update capacity gate.

The owning W&B registration and canonical HOI80 evaluation ledger stay fixed.
Telemetry performs a graceful handoff. Paired historical metric rows are
retained; replayed iterations remain in each attempt's native logs until the
original history frontier is exceeded, avoiding conflicting W&B event payloads.
New checkpoints publish only as complete eight-rank sets. Unpaired metric tails
and incomplete newer checkpoints are archived, never deleted. Full HOI80 still
runs at absolute2000-iteration milestones and records trajectories into the same
training run's eval section. A running trainer is never requeued to test this.

Ultra recovery from checkpoint500 uses `configs/hoi_ultra_boxes_resume500_solver_v1.json`
and `cluster/hoi_ultra_resume500.sbatch.sh`. The immutable eight-rank bundle binds
the failed parent run, checkpoint bytes, source, all three networks/RMS, Adam
states, RSI, contact statistics and RNG. Robot/contact/motor code stays unchanged.
The model-local numerical line-search budget is raised from50 to200; every native
flag (including solver limits and unknown bits) and nonfinite state remains fatal.
Failures save decoded world/asset/frame diagnostics before stopping.

On two L40 GPUs, eight disposable workers first restore500, prefill64 real replay
steps without updating normalizers, collect4096 real stress steps each, test the
unchanged native hard guard, then run full-sized zero-update gradient/capacity
validation. Only an all-rank pass allows W&B registration and fresh processes
restoring the original500 again. Training starts at501, preserving the absolute
LR schedule and2000-iteration full HOI80 milestones, with trajectories and eval
scalars written into the owning continuation run. Simulator/replay are recreated;
this is not bitwise simulator continuation. The tables training node is excluded.

Branch `hoi-retarget/tables-boxes-v1.0` (renamed from `hoi-retarget/tables-v1.0`) starts at
`codex/hoi-residual-ppo-teacher@78cd5b2c3530f9aa47b0c944436e59d5dc054093`.
The existing `supermimic/` and `hoi_residual/` implementations are unchanged.
The entry point is `python -m hoi_tables.train`. Profiles are
`configs/hoi_tables_v1.json` and `configs/hoi_ultra_boxes_v1.json`.
Both publish periodic evaluation into their owning training W&B run under
`eval/*`; neither creates an evaluation W&B run. The existing tables training
job continues unchanged. The Ultra preparation/validation/training chain is
submitted; its current launch snapshot is recorded below.

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
  frozen reward implementation. Tables reference intent uses exact triangle
  gaps, penetration and intersection-unknown semantics on the current hulls.
  Ultra reuses the original v1.1 cached reference intent, as requested by the user.
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

The eight asset ranks share one logical model and synchronized statistics. The
Ultra configuration now defaults to `workers_per_gpu=4` on **two L40s** after
the user's explicit authorization following two failed single-card probes.
All eight asset workers and the same global objective are retained. Explicit
values of two or one select four or eight GPUs, respectively. The complete
production zero-update capacity probe must pass before fresh formal models
are constructed. A successful Ultra capacity pass is required; earlier failed
preflight attempts do not authorize formal optimizer updates.

Ultra reuses the **existing v1.1 reference-contact targets**. The previous
boxes runs already loaded `expected_contact` / `expected_valid` directly from
the frozen v1.1 bundles. New robot physics does not, by itself, require changing
those task-reference labels. The user explicitly selected reuse after the
initial current-hand recomputation was identified as an additional choice.

`hoi_tables.v11_reference_contacts` verifies and repackages train2651 and
canonical80 into one cache without running FK or exact geometry queries. It
binds each source bundle and label file by SHA, matches motion/source-group/
exact-asset/T+1 identities, preserves the original label values, and uses a
separate schema with `label_origin=v11_cached_reference_intent`. The labels
are not claimed to have been generated by the current hw3 hand geometry.
The current robot, projected palm probes, hand-distance calculation, reward
formulae and five-channel native WM contact targets remain unchanged.

The existing train bundle covers all 2651 training motion hashes and durations;
its sibling `bundle_eval80/bundle.json` matches all 80 canonical hashes and
durations. A 128-frame / 256-hand-channel audit found 18 differences in valid
masks between old and current-hand geometric labels, with zero positive/negative
differences where both were valid. This is a sample comparison, not proof of
whole-dataset equivalence. Reuse preserves the original reference intent.

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

Final validation: **175 tables/Ultra CPU tests passed**, including owning-run
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
sbatch --dependency=afterok:$prep --gres=gpu:l40:1 \
  --output=/mnt/projects/ll/tianzizh/hoi_direct_ultra_v1_01-%j.log \
  "$SRC/cluster/hoi_direct_train.sbatch.sh" "$CAMPAIGN" "$SRC" "$CFG"
```

The destination cache and campaign must be new. The default is one GPU. To explicitly select four GPUs, pass
`--export=ALL,HOI_WORKERS_PER_GPU=2 --gres=gpu:l40:4` to the training submission.
`hoi_tables.launch` freezes source, links existing inputs read-only, writes
profile metadata, and rejects existing campaigns. The launcher requires a
healthy owning-run logger **and** evaluation scheduler before starting the
capacity probe. Existing runtimes and active tables sources remain read-only.
These commands are reusable entry points; the current submission is below.

## Authorized Ultra launch snapshot

The user authorized contact preparation, L40 validation and subsequent fresh
Ultra training, with the existing tables training left running. Both datasets
use project **hoiretarget-tables**, with separate training runs. Ultra names
begin **ultra_boxes_v1-**; no evaluation-only run is created.

- Isolated deployment root:
  `/mnt/projects/ll/tianzizh/hoi_ultra_boxes_teacher_20261107_01`.
- Immutable staged source revision: `a03f15fb87203371f4165d64c67256383c66e5e5`;
  147 manifest files, all 46 shared Python files equal to current tables.
  Source archive SHA256:
  `e74351e0bb20e9fab0d0a6ab841d75facad7c9ff294e13720b58e1ed24e2ae00`.
- Initial CPU reference recomputation: **1513788**, 16 CPUs / 48 GiB on
  `ee-a6000-1.grasp.maas`. Input preflight passed all eight assets, train2651,
  canonical80 and 23718 canonical controls. All 16 workers are computing
  unchanged exact current-hand labels. A short timing job `1513816` passed;
  representative frames took 0.17–0.21 seconds each, so full preparation is
  expected to require hours. This small timing sample was not a completion ETA.
  It was subsequently canceled at the user's request to reuse existing labels.
- Original CPU attempt `1513778` was canceled during abnormally slow runtime
  initialization on `mp-2080ti-0`; no contact cache or optimizer update was
  produced. The pending training dependency was updated to the replacement.
- Original L40 request **1513779** used four GPUs as a parallel deployment
  choice, without single-GPU capacity evidence. After the user pointed out
  their prior single-L40 v1.1 Ultra training, this still-pending request was
  held and canceled before allocation. Contact preparation was not restarted.
- Initial single-card replacement **1513885** requested **1×L40**, 32 CPUs / 192 GiB / 96 hours,
  with **afterok:1513788** and `HOI_WORKERS_PER_GPU=8`. All eight asset workers
  map to CUDA0; global4096, WM batch2048, policy batch4096, H32, 500 WM-only
  iterations and 20k total remain unchanged. Twenty targeted routing/launch/
  capacity CPU tests passed. It was held while replacing the label dependency
  and frozen source with the verified-cache-reuse path. Actual single-L40
  capacity is still unmeasured.
- Its launcher refuses missing labels, freezes `formal_01/source`, registers
  a distinct Ultra training run in the shared project, and verifies the unified
  logger plus complete canonical80 scheduler before the zero-update native
  capacity probe. All eight capacity reports must pass before fresh training
  processes/models/RMS/optimizers/replay are constructed. No checkpoint resume.
- Every2000, full HOI80 results go to that Ultra run under `eval/*`, using
  `eval/iteration`. All eight checkpoints and all 80 original cases are bound.
- At this snapshot, the old full geometric recomputation is canceled;
  **L40 validation and formal optimizer updates have not begun**. Reuse packaging
  and its new submission receipt are recorded below. W&B registration occurs
  inside the L40 launcher, so a new run ID is not claimed yet.
- Root receipts: `staging.json`, `contact_prepare_submission.json`,
  `contact_prepare_retry.json`, `training_submission.json`, `label_timing.json`;
  live preparation log: `contacts02-1513788.log`. Naming change passed all
  12 telemetry tests; current tables sources/jobs were not changed.
- Historical GPU receipt: `single_l40_submission.json`; the original
  `training_submission.json` is retained as history. The frozen source config
  is overridden explicitly by `HOI_WORKERS_PER_GPU=8`; normalization writes
  eight workers per GPU into the eventual campaign configuration.
- Actual Slurm start estimate is **N/A** while labels are a dependency.
  A non-submitting single-GPU `sbatch --test-only` on November7, assuming
  labels ready immediately, predicted **November9 21:46 America/New_York**.
  This conditional scheduler estimate can move and is not a promised start.

## Existing-cache reuse and single-GPU actor memory schedule

The user then explicitly requested reuse of existing Ultra labels. Full
recomputation `1513788` was canceled, and the dependent old GPU request was
held and replaced. The train and eval bundles already contain all 2731 required
reference-label records with matching motion SHA and T+1 frame counts.

- Frozen reuse source `source_reuse01`, revision `2daa5f7`, 149 files; archive
  SHA256 `b20bb80a92f9246fb3ad891f0d7b960e6f46aca3159c98e067127b856cafb6a7`.
- Fast verification/repacking job **1514101** completed successfully in 64s.
  It copied existing values for train2651 and canonical80, totaling **780322
  state frames**; no geometric labels were recomputed.
- Cache:
  `/mnt/projects/ll/tianzizh/hoi_direct_ultra_inputs_v11_cached_v1/reference_contacts.pt`;
  SHA256 `b341769b9d28b1d6c7852fb1355eb3a09e5acc77aa0905b0471b8aeeb573d1e8`.
- CPU consumer job **1514102** passed all 16 real dataset/asset consumers,
  verifying the complete train/eval inventory, identical normalized reward
  provenance, original source values, correct local indexing and zero calls to
  the exact-reference-geometry kernel. No model/statistics/replay updates.
- Single-L40 preflight **1514103**, output `formal_01`, allocated successfully.
  It registered owning training run **30ff7c18** in `hoiretarget-tables`, name
  `ultra_boxes_v1-20261107-230612`; its unified logger was **1514106**. The
  canonical80 supervisor initialized and no separate evaluation run was made.
- That preflight hit CUDA OOM while eight ranks simultaneously retained their
  H32 actor graphs. The card exposed 44.32GiB; processes used about 4.2–6.1GiB
  each. The zero-update guard stopped it before formal training, with no
  checkpoints or optimizer updates. Its failed run/reports are retained.

The memory fix serializes only each shared GPU's **local actor forward/BPTT**.
All eight assets, 4096 native environments, full local/global learning batches,
H32, networks, reward formulae, observations, precision and optimizer counts are
unchanged. Each rank computes the same full local gradient; all ranks then
perform the original global gradient mean, clipping and Adam step together.
RMS updates and critic/WM updates retain their original locations. Unused CUDA
allocator segments are returned before the first turn and after each rank's
backward; live parameters/gradients are retained. Tables defaults to the old
parallel path and its running frozen source is untouched.

H32 CPU equivalence tests compare gradients, model/RMS tensors, optimizer
states, metrics and RNG against the original algorithm. Real 2/4-rank Gloo
tests also verify one actor graph per logical GPU and coordinated failure
propagation before any gradient synchronization or optimizer step. Runtime
metadata records `actor_backward_schedule=serialized_per_gpu`. The cached
v1.1 labels and their repacker code remain byte-identical, so they are reused
without another preparation job. A fresh single-L40 production probe must
pass before fresh formal training starts; current retry receipts follow.

The complete CPU suite passed **175 tests** after this execution-order change,
including the full H32 update/state/RNG equivalence and real Gloo scheduling
checks. Cache-reuse adapter/source bytes remain unchanged from the already
validated pack; the retry does not regenerate or repack reference labels.

Historical single-card retry snapshot (superseded by the two-card retry below):

- Source `source_memory01`, revision `28cb5e9`, 151 manifest files; archive
  SHA256 `ec5f5eeaa48210ebfae32221ac08aa76a0903c7ebbf8a3323f639785c488fea8`.
  All 56 physics/network/reward/reference-loader source files match
  `source_reuse01`; only execution scheduling and its metadata/tests changed.
- Fresh campaign `formal_02`, **Slurm1514391**, **one L40**, eight asset ranks,
  full global configuration. Existing cache SHA remains `b341769b…573d1e8`.
- It is pending GPU resources, with Slurm's current tentative start
  **November8 07:23:57 America/New_York**. This replaces the earlier
  hypothetical queue estimate. No reference-label dependency remains.
- The launcher must again pass the complete native zero-update production
  capacity probe before constructing fresh formal processes. Single-GPU
  capacity with the new memory schedule is **not yet verified**; formal
  training has not started. No automatic resource or batch/horizon reduction.
- Receipt: `memory_retry_submission.json`. Original failed preflight run
  `30ff7c18` remains the historical record; the retry registers a fresh
  `ultra_boxes_v1-...` training run on allocation, in `hoiretarget-tables`.
  Full HOI80 every2000 is logged into that owning training run.

## Two-L40 retry after the second capacity failure

Single-card job **1514391** / `formal_02` / W&B **acdb6db4** failed in
the native zero-update probe. All eight ranks completed ten WM backwards;
rank7 ran out of memory during its H32 actor turn. The other ranks propagated
that failure before any optimizer step. All eight reports confirm zero
optimizer updates and unchanged model/RMS tensors; no formal iteration or
checkpoint exists. The failed W&B run is retained as the failure record.

The user explicitly authorized **two L40s**. New job **1516579** requests two
cards, 32 CPUs, 192 GiB host memory and 96 hours; eight asset ranks now share
four ranks per GPU. Source `source_two_l40_01`, revision **f3c533b**, is frozen
separately, with output `formal_03`. All **146 Python/scheduler files** are
byte-identical to `source_memory01`; only the Ultra config's execution topology
changes. The cached labels, global4096/WM2048/policy4096/H32, 500 WM-only
iterations, physics, network and rewards remain unchanged.

The job must pass all eight full native capacity reports before fresh training
processes start. It registers a new `ultra_boxes_v1-...` run in
`hoiretarget-tables` on allocation; full HOI80 every2000 writes into that run.
At **November8 09:23 America/New_York**, the job is pending priority; Slurm's
tentative start is **November14 22:51:49 America/New_York**, subject to change.
There is no new W&B run yet. Tables job1512413/run44ad3b72 remains running with
restart count0. Deployment/status receipts: `two_l40_submission.json` and
`two_l40_status.json` under the Ultra campaign root. Thirteen CPU routing and
launch tests passed; two-card native capacity and formal training remain pending.

## Earlier public-queue candidate

On November8, the user requested an earlier two-L40 queue without affecting
tables. The account is authorized for `ll-med` and `normal`; other groups'
owner QoS values and `ll-high` are unavailable. Public `batch` adds the eight
real L40 cards at `dj-l40-0` beyond the three Liu nodes. All eligible L40 cards
were allocated when inspected; there was no immediately idle two-card slot.

Existing job **1516579** was moved to **batch**, preserving its original
09:22:21 submission/eligibility/accrual timestamps, two L40s, 32 CPUs, 192 GiB,
96-hour allocation and frozen source. No second training candidate was made.
The scheduler requested QoS list is `normal,ll-med`; only **normal** is
admitted by batch, with normal first. This valid configuration was observed
across scheduler cycles with reason **Priority**. A mixed-partition candidate
was not retained because it displayed a partition/QoS permission conflict.

At **November8 10:48 America/New_York**, the actual job's tentative start is
**November11 07:27:31 America/New_York**, compared with the original Liu-only
**November14 22:51:49**. The estimate can change. Public batch is preemptible
by its resource owner; it cannot preempt the tables job's `ll-med` allocation.
Tables **1512413** remains running, original start time and restart count0
unchanged. All151 frozen source files were reverified, the global training
configuration and existing label cache remain unchanged, and the GPU probe
and new W&B run have not started. The owning-run HOI80 chain is unchanged.

Current receipt: `queue_status_latest.json`; full before/after records and
resource/source checks: `queue_candidates_20261108_01/verified.json`, under
the Ultra campaign root. The original `two_l40_submission.json` remains the
historical submission receipt, rather than being overwritten after migration.

## Native trajectories from each periodic HOI80 evaluation

The user requested recording before Ultra starts. `evaluate_hoi80` now saves
all **80** cases during the original scheduled evaluation every2000 iterations,
covering **23798 T+1 state frames / 23718 control intervals** at50Hz. The
629-step case remains complete, including all post-failure states. Canonical
COM-velocity initialization is applied before the first recorded frame.

Each asset worker writes `trajectories/case_XXXXX.npz` and a JSON manifest
inside its independent evaluation directory. The aggregate `result.json`
indexes every file by relative path and SHA256. Saved fields include native
38-link robot world poses, object poses/velocities, physical reference38 poses,
71D actual/reference robot states, actions/rewards/failure flags, five-channel
contacts and validity, exact reference-frame indices and the complete50Hz
clock. Original canonical PHC39 source columns are also preserved separately.
Native rendering poses use **wxyz**; the raw canonical reference retains its
original **xyzw** layout. Coordinates retain their shared world frame, without
trajectory alignment or reconstruction of actual links from joint FK.

Manifests bind the owning run, all eight checkpoint hashes, model/RMS hashes,
source/input identities, body order, clip IDs/durations and visual-asset paths.
Missing, corrupted, substituted or truncated trajectories fail aggregation
and the owning-run monitor's result validation. The recorder changes no
policy/RMS/statistics and owns no simulation step or optimizer. Networks,
observations, rewards, physics, data, learning batches and evaluation scoring
remain unchanged. W&B continues receiving the same scalar `eval/*` metrics
on the owning training run; raw trajectories and videos stay on the cluster.

Deployment before Ultra starts: code revision **13fcab8**, frozen independently
at `source_hoi80_trajectories_01`. The existing pending job **1516579** was
temporarily held while its entry `source_two_l40_01` became a symlink to that
tested snapshot, then released. The old source's exact bytes remain under
`source_two_l40_01_before_trajectory_recording`. No cancellation/re-submission
or duplicate run was used. The original job ID/submission time and two-L40
request remain; eligibility/accrual timestamps may be reset by Slurm's hold.

All **179** tables CPU tests passed on the cluster (1517256), and again on
node-local scratch (1517274). Native GPU job **1517257** completed a four-control
interval / five-state-frame save/load/integrity smoke with the real38-link
robot. It made zero policy inferences/optimizer updates/W&B runs. The629-step
CPU case verifies complete post-failure recording and exact policy/RMS/RNG and
metric equivalence with recording disabled. This is an IO validation, not a
full80 policy evaluation or a two-L40 training-capacity claim; the original
eight-rank zero-update capacity prerequisite still applies at allocation.
All147 unchanged existing files, including learning/physics/configuration,
were checked byte-for-byte. CPU launch readthrough also preserved the original
global4096/WM2048/policy4096/H32/warmup500 and eight ranks over two GPUs.

Deployment receipt: Ultra root `hoi80_trajectory_update_01/installed.json`.
Tables1512413 retained its original start time and restart0 throughout.

The same Ultra job subsequently obtained two L40s at `dj-l40-0` on November8,
16:23:52 America/New_York. New owning run **d9189ee4** is named
`ultra_boxes_v1-20261108-162506` in `hoiretarget-tables`. All eight full native
zero-update capacity reports passed, the launcher saved `capacity_passed.json`,
and fresh formal processes have produced actual `wm_only` optimization rows
through iteration51. Warmup remains500. The formal frozen recorder bytes match
`source_hoi80_trajectories_01`; evaluations will use this complete recording
contract. Slurm allocation alone is not the evidence of optimization here.
Subsequent server-side W&B readback confirms `running`, iteration67,
stage`wm_only`, on the same owning run d9189ee4.

## Checkpoint6000 protected22 video delivery

The original scheduled22-case result remains unchanged (SHA256
`e0fe62eec9c8dd6d15884bcb3551a322ef505fc0e5fafb5c4aab39a51db21271`).
Independent A10 job **1517200** recorded the same frozen6000 checkpoint pair
under `videos_6000_01`, without policy/RMS/training updates or W&B writes.
All22 complete clips were rendered from saved native38-link/object poses,
with separately recorded physical reference poses:6504 states,6482 controls,
720p50,130.08 seconds. There is no actual-state FK reconstruction or physics
stepping during rendering. Contact simulation is not claimed bitwise
deterministic; both original evaluation and visualization rerun survived19/22.
The original scheduled W&B metrics were retained.

The all22, largetable, smalltable and22 individual videos plus gallery/previews
were delivered to local `results/tables_checkpoint6000_protected22_20261108`.
Checks include full video decoding, every clip/bundle checksum,66 preview
edge checks, all-frame camera projections and unchanged source/checkpoints/
original evaluation. Raw native recordings remain on the cluster.

## sub3_smalltable_027 foot/table penetration investigation

The user's checkpoint6000 video observation is a **real collision-volume
overlap**, not only artwork intersection. The original256-frame native-pose
recording SHA is `e7e7e38ffc60bda67643b645789de224755de20bf24512861f6056612b3aac11`.
At frame128 (2.56s), the right foot overlaps a convex table-leg segment by
**19.495463 mm**; the maximum is **22.199047 mm** at frame156 (3.12s).
There are158 saved states with overlap exceeding10mm. Frame0 is separated.

The original recording was queried against both the exact bound source
shapes and the actual native model exported from the same frozen evaluator.
These results agree. An independent SciPy Minkowski-hull / capsule-radius
calculation agrees within2.6e-9m at both sample frames and confirms that the
capsule axis segment itself intersects the convex leg. No actual-state FK,
trajectory alignment or physics replay was used for these measurements.

Actual native geometry contains **44 closed, positive-volume table convex
components**,3210 outward faces and zero inward faces. Both feet have seven
analytic capsules, with8–10mm radii, enabled collision masks and the expected
body-local transforms. Foot colliders therefore have no mesh winding to
reverse. Their local Z coverage is approximately[-.035,-.015]m; shoe artwork
extends to+.023584m. The unfilled upper artwork is a separate limitation,
and does not explain away the confirmed capsule/table overlap.

Measured foot/table native contact parameters are the default soft
`solref=(.02,1)` and `solimp=(.9,.95,.001,.5,2)`, with foot sliding friction1.
The table remains a4kg dynamic free rigid body; no attachment/following code
is present. Soft-contact interpenetration followed by frictional dragging is
the leading mechanism, rather than inverted mesh normals. A minimal **cluster
MuJoCo3.15** compliance fixture gives9.631N at22mm overlap with defaults and
37.240N with a stiffer pair probe. Those are fixture forces, **not measured
forces from the full robot**. Reducing object `gap` from.01 to0 gives identical
fixture forces/activation and is not credited as a fix.

Full GPU force tracing job1517277 exported the native model, then failed to
load a CCD CUDA module on the2080Ti (CUDA error200). It is not a successful
GPU rollout. Completed CPU geometry jobs1517290/1517294/1517309 and independent
fixture1517318 support the measurements above. A full-clip GPU A/B remains
necessary before deploying a contact fix; the proposed first comparison is
an isolated foot/table pair adjustment, retaining the robot/motor/soft-hand
contract. No physics or running training was modified during this inspection.

Cluster receipts: `collision_native_geometry_check_01/{geometry_summary.json,
native_geometry.json,contact_activation_fixture.json}` under the tables root.
Local figures/summaries: `results/tables_sub3_smalltable_027_collision_audit_20261108`.
The clear figures use the same original saved poses, with shoe artwork in
gray, physical capsules in green and the implicated convex leg segment in
orange. Original scheduled6000 result SHA is still `e0fe62ee…b21271`;
tables1512413 retained its original start time and restart0.

## Foot/table stiffness A/B: checkpoint6000, sub3_smalltable_027

User-authorized independent A10 job **1517402** completed with exit0 in11m03s.
It used the original frozen `source_eval01`, the unchanged6000 checkpoint
pair, and fresh seed0 deterministic closed-loop policy processes for A/B.
Both ran the full255 controls /5.10s without resets after failure or training
updates. The policy/RMS tensors and initial snapshots matched exactly.

**A** retains foot/table contact `solref=(.02,1)`; **B** changes only the
time-constant component of actual foot/table contacts to`.01`. Solimp,
friction, damping ratio and every hand, foot/ground and other contact remain
unchanged. The private callback is composed inside the existing hand owner;
no model geom/material arrays or shared source/environment are patched.
CPU/CUDA contact checks and CUDA graph replay confirm A's bitwise no-op,
B's pair isolation and preserved hand friction/normal response. Full-clip
auditing found zero unrelated solref/damping changes; B changed all107
observed foot/table contact samples and zero of27774 other samples.

All robot link/object native poses were saved every5ms (1021 states), plus
256 states and physical reference poses at50Hz for video. Penetration is
queried against the actual exported collision model, over both feet and all
44 object components. Initial/final native model hashes, initial GPU geom
fields, policy tensors and initial snapshot hashes match across A/B.

| Metric | A:20ms | B:10ms |
| --- | ---: | ---: |
| Maximum penetration at200Hz |23.7903mm|13.9211mm|
| Mean penetration across recorded states |3.27858mm|0.07962mm|
| Sampled time above1mm (state count ×5ms) |0.935s|0.065s|
| Sampled time above10mm |0.820s|0.015s|
| Integrated penetration |16.7372mm·s|0.40645mm·s|
| Normalized return |0.46119|0.42498|
| Failure triggered |no|no|

The maximum drops41.48% and sustained penetration drops substantially, but
B still has a13.9mm transient peak at1.96s. Both trials retain tracking-quality
fraction0.17647 and B's reward is lower. This single-clip result is not a
complete collision fix or evidence of a training improvement. The original
22-case evaluation/video is retained; fresh simulation reruns are not claimed
bitwise identical to that earlier video.

Independent CPU job **1517426** checks all geom pairs at each variant's
maximum, without bounding-sphere pruning, then verifies penetration with a
separate SciPy Minkowski-hull/capsule calculation. Errors are below9e-6mm.
Actual CUDA contact forces were recorded in this successful A10 experiment;
the previous2080Ti failure is not used as an evaluation result.

Experiment helpers are in `tools/tables_contact_ab/`; their frozen cluster
snapshot is `contact_ab_helpers_01`, archive SHA256
`96abf8d3a79e8250d5ad9415650f08784a1b2252d6def40845d15cb7a4be24c0`.
The independent measurement verifier has its own SHA in its result receipt.
Cluster outputs: tables root `contact_ab_6000_sub3_01`. Original source,
checkpoint pair, scheduled6000 result and experiment helper SHA checks all
passed. Tables1512413 and Ultra1516579 retained their RUNNING state, original
start time, node, allocation and restart0. No W&B writes were made.

Delivery: `results/tables6000_sub3_contact_AB_20261108`. Paired full-body and
foot/table close-up videos use the same fixed world camera per view, original
artwork and reference ghosts. A quarter-speed close-up uses those same saved
frames. All videos fully decode; rendering uses zero simulation steps or
policy inferences. Raw poses/contact traces stay on the cluster. B has not
been deployed to either running training job.
