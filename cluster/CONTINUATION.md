# ON continuation: 14000 -> 24000

User-requested continuation of `selfcoll-lafan1-20261022-s0`, using its full
14k `last.pt` (SHA256 `b933f0949de00484328423659a903437641be629845908eb85c0a7cd891fba15`).
Only total iterations (24000) and checkpoint/full40 evaluation interval (1000)
change. All model, optimizer, physics, data, seed and LR schedule settings remain
the same. The LR uses global iteration 14000 onward; its 250000-iteration decay
is not restarted. The 500 WM-only iterations are not repeated.

Run `sbatch cluster/continue.sbatch.sh`. The independent output folder is
`logs/lafan1_selfcollision_seed0_continue14000to24000`; the original 14k folder
is read-only to this pipeline. A full checkpoint snapshot is copied there and
verified before loading. Live equality checks verify all model, normalizer and
Adam state tensors before collecting new data. Original files are hashed and
checked again after completion. Existing output causes a hard stop, never an
overwrite or an automatic duplicate producer.

The original checkpoint does **not** save replay, simulator/contact state,
observation history, motion sampler statistics or RNG state. Resuming therefore
creates a fresh seed0 simulator/replay. Original `collect(64)` fills replay before
the first optimizer update, retaining its usual online normalizer updates. This
adds 262144 environment transitions, recorded in sim_steps. Without this fix,
upstream resume samples an empty replay buffer. `resume_launch.py` adapts only
this warm-up condition and empty CSV header handling, and inserts two audits.
The scientific `supermimic/` and `tools/` sources are untouched. This is complete
model/optimizer continuation, but is not bitwise uninterrupted training.

The original evaluator is unchanged (`--watch --every 1`): it evaluates all saved
checkpoints, so all ten checkpoints 15k through24k receive the same full40 test.
CSV curves append to the existing W&B run with `resume=must`, starting after14k.
Original W&B configuration/history is retained; effective continuation parameters
and limitations are added under `continuation_14000_to_24000`. A cloud resume
handshake must succeed before a training producer starts. No source, checkpoint
or raw-data artifacts are uploaded. New-session wall time includes the original
saved training wall time, as in upstream resume. Warm-up remains outside that
clock; extra evaluations contribute to elapsed training time.

Checks: `python cluster/test_continuation.py` (CPU); actual GPU restore and replay
audits are written into the continuation directory. `pipeline.json` marks this
session's completion independently of the old completed pipeline.
