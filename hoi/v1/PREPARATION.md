# HOI v1 references and collision recipes

Run this preparation with CPU Torch and MuJoCo. It does not launch IsaacLab or train a model. Source tensors, native assets and the legacy zero-shot inputs stay read-only; every output directory must be new.

First build and validate the canonical manifest using [the stage-one data instructions](../data/README.md). An `objects` entry that contains only `asset_id` and an identity-only status is accepted as an identity declaration: the canonical motion entry supplies the object name. It does not establish scale, a mesh transform, COM or native geometry. Those properties must be supplied by the collision recipe and verified by the native gate.

```bash
python -m hoi.data.prepare validate --manifest /new/data_manifest/manifest.json \
  --partition train --output /new/full_train_data_audit

python -m hoi.v1.prepare --manifest /new/data_manifest/manifest.json \
  --assets /new/compiled_geometry/assets.json --output /new/train_bundle \
  --partition train --jobs 8
```

`--robot-mjcf` can bind an explicitly located copy of the original G1 MJCF; the default remains the original repository path. Its SHA and original 14 reward-body names are saved in the bundle. This does not change the articulation, PD or the reward body set. `--limit` creates an explicitly labelled prefix subset for development. A subset cannot run as a formal experiment.

`--one-per-asset-resource-fixture` creates a separate, explicitly nonformal train fixture containing the first real motion for each exact asset binding. It keeps the canonical manifest unchanged and binds its original SHA; selected clips still pass ordinary membership and source-byte checks. The bundle is always marked `scope.is_subset=true`, `purpose=one_per_asset_resource_fixture` and cannot supply formal training. This flag is incompatible with `--limit` and nontrain partitions. It supports all-group memory/native pipeline checks before the full cache is ready. The CPU launcher enables it only through explicit `RESOURCE_FIXTURE=1`.

The recipe is a JSON object with `assets`, `hands` and optional `reference_kinematics` fields. The native geometry probe exports a concrete `assets.json` for one asset; a multi-asset recipe must include all required exact asset/scale groups and the same verified hand contract.

```bash
python -m hoi.v1.merge_recipes --manifest /new/data_manifest/manifest.json \
  --master /immutable/master_probe/compiled_geometry/assets.json \
  --objects /new/other_object_probe/compiled_geometry/assets.json /new/additional_probe/compiled_geometry/assets.json \
  --output /new/full_recipe/assets.json
```

Supply all remaining object exports. The merger requires exactly the train manifest's object keys and orders them by first occurrence in that frozen manifest. It takes hands, palm evidence and robot overlay only from the master, records every source recipe hash, and exclusively creates its output. The legacy fixture's asset indices can differ from this order; select and compare by `object_asset`, then use the new bundle's stable integer IDs. Merging is provenance work, not a substitute for testing each final bundle geometry group.

| Field | Required contents |
| --- | --- |
| `assets[].object_asset` | Canonical object identity used by the manifest |
| `assets[].binding_key` | Optional explicit identity for distinct exact-scale bindings; manifest `bindings.objects` can name it per motion |
| `assets[].exact_scale` | Positive scale3; retained explicitly even when already baked into vertices |
| `assets[].mesh_to_object` | Proper homogeneous 4×4 transform, in metres |
| `assets[].native_asset`, `native_asset_sha256` | Read-only, already scaled/transformed rigid-body USD and its byte hash |
| `assets[].physical_profile` | Positive mass, local COM3, symmetric positive definite physical COM inertia3×3, object friction and ground friction |
| `assets[].collision_meshes` | Explicit closed, consistently wound triangle meshes, supplied inline as vertices/faces or by hashed OBJ/STL/PT files |
| `assets[].collision_meshes[].frame` | `mesh` applies `exact_scale` then `mesh_to_object`; `object_local` means these are already baked and are not applied again |
| `assets[].distance_recipe` | Optional `grid_resolution`, `padding`, `candidate_count`, `seed`, `max_grid_error`; defaults 64, .08m, 4096, 0, .01m |
| `hands.left/right.source_stl` | Original rubber-hand STL path and the repository's known byte SHA |
| `hands.left/right.mount` | Original wrist fixed offset: left `[.0415,.003,0]`, right `[.0415,-.003,0]` |
| `hands.left/right.collision_meshes` | Actual compiled hand colliders expressed in the original hand-local frame |
| `hands.left/right.palm_points` | Actual inner-palm collider-surface points, before the fixed mounting transform |
| `hands.left/right.palm_selection_verified`, `palm_selection_evidence` | Versioned ray/source-face mapping, source STL/code/collider/probe hashes and visual-QA image hash; default verification is false |
| `reference_kinematics` | Optional original virtual URDF path/SHA plus the exact 39-body and 29-DOF name orders; required for full PHC evaluation |
| `native_robot_asset` | Exact robot/rubber-hand USD path/SHA from the calibrated native probe; its adjacent `hand-asset.json` receipt is also bound |

Relative paths resolve from the recipe directory. Preparation rehashes the source motion bytes and decodes those exact bytes, then performs the canonical 60→50 resampling. It does not resample a previously adapted 50Hz cache a second time. `state_frames=T+1`, `control_steps=T`; source velocity/contact fields remain separate from pose-derived reference velocity and geometric pair schedules.

Canonical preparation requires the exact cooked robot/hand overlay. The older export's `hand_runtime_report.usd`/`usd_sha256` is accepted as the same explicit binding. Workers reuse that frozen overlay: independent CoACD invocations were observed to produce different hulls under the same seed/recipe, so recooking each worker cannot establish equality with the cached reward geometry. The existing hand-asset receipt and shared source asset hashes are checked again when loading the overlay.

Shape candidates come from exposed collider-union surfaces, followed by deterministic FPS128. Coordinates remain object-local metres. Overlapping hulls' internal faces are rejected as shape candidates. Expected hand-object labels use collider triangle queries, not the 128 shape points or the source 39 net-contact flags: gaps ≤.02m are positive, ≥.04m negative, the interval between them unknown. Every detected edge-face intersection is conservatively unknown, including thin colliders crossing with all their vertices outside the opposite collider. Some boundary touches cannot be distinguished by that predicate and also remain unknown. The contact cache preserves `surface_intersection` separately from the exact gap and sampled vertex depth, and records this versioned mask method. Sampled vertex penetration is a bound rather than a global minimum-translation penetration depth.

The differentiable grid has the correct collider-union zero surface and sign. It uses the minimum of component signed distances, so interior magnitudes are a CSG surrogate. Beyond the grid it adds the distance to the grid box to the boundary value; that magnitude is approximate and preserves coordinate gradients. Metadata records these definitions, randomized/exposed-surface QA probe counts and measured maximum/RMS grid error. The same cached query and reward formula are used for real and imagined states.

CPU triangle queries use exact AABB branch-and-bound pruning. For signed component queries, the signed AABB distance is a lower bound on the contained collider's signed distance. For mesh-pair gaps, disjoint pairs are skipped only when their AABB gap cannot improve the current exact gap; every overlapping or touching AABB pair still contributes to the vertex-penetration audit. A conservative 1e-10m scheduling margin adds exact queries near equality. These branches preserve the original exact values and labels, including nested and intersecting components; they do not replace positive/contact queries with a proximity approximation.

Within each contact-preparation clip, immutable object/hand topology and unique edge indices are validated once and reused across the known rigid FK transforms. Every transformed vertex array still requires the same shape and finite coordinates. Public arbitrary-mesh query APIs keep full validation. The private path shares all signed triangle, edge-edge and edge-face arithmetic with the public path; it does not replace those calculations with a proxy. A real 221-state comparison against the frozen previous cache verified bitwise equality of gap, sampled penetration, intersection, expected labels and both validity masks.

Each clip and the bundle's per-asset/whole-partition summary report reference contact quality using `contacts[1:]`, the T interval endpoints. They count LR valid, positive, negative, total unknown, distance-gray unknown, detected intersections, vertex depth at least the severe threshold, and adjacent-valid onset/release endpoints. Unknown reason counts can overlap. These are geometric-intent counts, not actual pair labels or a subjective pass criterion; an all-unknown cache cannot be presented as sufficient reference coverage.

Parallel preparation only changes CPU scheduling. `--jobs` defaults to 1. With a larger explicit value, spawn workers each use one Torch thread, rehash/decode independent source files and write exclusively named clip caches. Manifest order determines output clip/global-frame order; completion order does not. Static collider meshes and FK are passed read-only; workers do not duplicate SDF grids. Grid construction stays once per asset before the clip pool. A failed worker prevents creation of `bundle.json`; existing partial audit files remain for inspection.

Full 2651-motion geometry preparation should first be timed on fixed train-only clips covering no-contact/contact/transport/release, using the real compiled hulls. Triangle-pair work scales with frames and collider complexity; process parallelism helps but is not a measured full-dataset throughput guarantee. Allocate CPU cores and memory explicitly, keep `--jobs` within that allocation, and record elapsed time and peak memory. Do not reduce labels, frame rates, geometry resolution or training settings to make the preparation fit.

Every bundle binds source manifest, recipes, motion/cache/contact/geometry bytes, scale/transforms, FK and optional virtual URDF. Loaders recheck the bindings. Training workers may load one asset's clips and map local frame IDs through `global_frame_index`; central training retains the stable full-bundle asset IDs. Evaluation retains the original resampled source630 reference for independent 39-body scoring. The HOI bank omits terminal-only reset bins, unlike the original motion-only library: every reset bin contains an outgoing control interval, and the final state belongs to the last valid bin.

Native readiness remains a separate gate. It requires matching compiled colliders, actual palm evidence, native mass/COM/inertia/friction/gravity readback, hand/self collisions and four-substep pair calibration including hand-ground negative controls. Exported colliders must come from the collision engine's `mjModel.mesh_graph` convex hulls. Independent `mj_geomDistance` GJK queries at two separated poses per hand must agree with cached surface gaps within 50µm, while the bundle-to-compiled mesh comparison retains its stricter 10µm tolerance. Native and reference robot FK must also agree within the versioned 50µm position and 5e-5 rotation-matrix tolerance. Comparing raw triangles with another copy of those triangles does not satisfy this gate. The WM policy gate uses an explicitly recorded train-only startup proposal with heldout real 1/8/32-step comparisons and actual-pair coverage/calibration. Passing that proposal is not an empirically calibrated success criterion or evidence that HOI interaction succeeds in the simulator.

```bash
python -m hoi.v1.gates --bundle /new/train_bundle/bundle.json \
  --native-report /new/group0/native.json --native-report /new/group1/native.json \
  --output /new/native_evidence
```

Repeat `--native-report` for all geometry IDs actually used by the bundle. This creates `native_evidence.json` binding each exact report's SHA and `gate.json` with explicit blockers. A blocked aggregation exits 1 and preserves its evidence; success exits 0. Training consumes the generated `native_evidence.json` through `--native-evidence`. Reports from a single-clip bundle do not automatically establish evidence for a different full-bundle identity.
