# HOI80 zero-shot

Unified entry: `python -m hoi.run`. Both hand collisions and body self-collision are now ON by default.
Use `--no-self-collision` for the historical hand-only comparison, or both negative switches for the original baseline profile.
See [DEVELOPMENT.md](../DEVELOPMENT.md) for full commands, immutable outputs, checkpoint provenance and offline rendering.

Frozen policy inputs remain 99 proprioceptive channels and a 730-dimensional human reference goal.
Physical objects remain active; no object channels enter the actor. All80 full-duration rollouts are saved without failure resets.
Every result records requested physics and native compiled collision verification. New defaults constitute a new physics combination, not an existing experiment result.

HOI v1's separate CPU data preparation lives in [data/README.md](data/README.md).
Use `python -m hoi.data.prepare` for manifests and 60-to-50 Hz reference audits; this does not start training or change this zero-shot entry point.

The additive full-reference HOI implementation has separate [v1 entry points](v1/README.md) for geometry preparation, joint-model training, frozen evaluation and recorded-state rendering.
