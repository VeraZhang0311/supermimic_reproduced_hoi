# Cluster entry points

Use `train.sbatch.sh` for new unified runs. Supply a fresh development checkout
through `SUPERMIMIC_ROOT`, and a new `RUN_DIR`. `SELF_COLLISION` and `HAND_COLLISION`
default to 1. Pass the matching flags when resuming legacy checkpoints.

Historical OFF and ON pipelines, monitoring, setup and continuation code are
preserved verbatim in `legacy/baseline_off/` and `legacy/selfcollision/`.
They document fixed experiment paths and W&B identities; do not run them from
the development checkout. Existing experiment checkouts/results remain in place.

`compat_launch.py` is the shared official graph scheduling compatibility adapter;
it resolves its own checkout rather than writing into a historical experiment.
