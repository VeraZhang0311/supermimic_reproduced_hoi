> Historical hands-ON/self-OFF comparison. New unified defaults and commands: [DEVELOPMENT.md](../DEVELOPMENT.md).

# HOI80 paired rubber-hand collision comparison

Parent baseline: `hoi/zero_shot` at65a3b2a, job1457069. Keep its original output
and W&B run unchanged. This independent checkout evaluates the identical frozen
baseline OFF 14k checkpoint on the same80 prepared motions, with dynamic objects,
99D actor observations,730D human goals, original PD,50Hz control and unchanged
reward/termination/whole-clip diagnostics. No optimizer/normalizer updates, no
failure reset, full T+1 state / T control recordings for every case.

The only scientific change is adding fixed rubber-hand collision geometry. The
left/right STL bytes equal the original HOI/ULTRA meshes (SHA verified), mounted
at the existing hand links, left(.0415,.003,0),right(.0415,-.003,0) relative to
the wrist, identity rotation, unit scale. Existing visual meshes, wrists,
rigid bodies, inertias, joints and all original USD property values are preserved.
The overlay introduces only new collision Mesh prims under the two existing
fixed hand links; shared assets are never edited or recooked in place.

Original HOI in-memory PhysX cooking requested convexDecomposition with5 hulls,
16 vertices and60000 voxels; it did not save cooked hull vertices. We cook only
the byte-identical hand STLs with installed CoACD, cap at5 hulls/16 vertices,
seed0, using the recorded recipe, then import the resulting explicit convex
hulls. CoACD is not the original PhysX cooker and its sampling resolution is not
PhysX voxel resolution. The original hand contact/rest offsets were automatic,
so the new colliders retain this backend's defaults. Existing object CoACD,
solver100/line-search50, graph compatibility and all old colliders stay unchanged.

Before the formal80, independent OFF/ON16-step gates must show zero OFF hand
colliders and positive ON left/right compiled masks/pairs; CPU contact probes in
independent MjData must produce actual object contacts with both hands. Robot
compiled mass/inertia/COM/joint frames/limits must match between gates. Overflow
and nonfinite state checks remain active. Extra gates are excluded from results.

Outputs: `results/hoi80_stock14000_handcollision-JOBID`; new W&B run
`zeroshot-hoi80-stock14000-rubberhands-20261022`. Reports include per-case paired
deltas against the preserved baseline. Only curated config/metrics/tables sync to
W&B; weights, source, raw motions and rollouts remain on disk. The80 verified
rollouts are also downloaded to the independent D: checkout for later rendering.

CPU preflight1457362 and import-only diagnostic1457370 isolated a native crash:
loading CoACD/trimesh then pxr.Tf in one interpreter segfaults in the pinned
environment. Hand cooking now runs in a separate interpreter from USD authoring.
No dependencies, geometry recipe, model or simulation parameters are changed.
Failure logs/source snapshots remain in provenance/failed-cpu-1457362.
