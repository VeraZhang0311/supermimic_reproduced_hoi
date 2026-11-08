# Tables6000 foot/table contact A/B

Independent evaluation of `sub3_smalltable_027` with the frozen tables6000
checkpoint and original evaluator. No training or W&B updates.

- A: the original foot/table `solref=(.02,1)`.
- B: replace only `contact.solref[0]` with `.01` on actual left/right-foot
  versus table contacts. Damping ratio, solimp, friction, geom materials,
  geometry, motors, soft hands and every other contact pair are unchanged.
- The model-local probe is composed inside the original hand callback;
  its owner and friction logic stay installed. Nothing is patched globally.

`verify_hook.py` checks real CPU and CUDA contacts, A's bitwise no-op,
hand/other-pair preservation, unmodified geom arrays, and CUDA graph replay.
`rollout.py` wraps the original frozen evaluator, selects this one full clip,
and saves all256 video frames plus1021 native physics states at200Hz. It
uses fresh deterministic closed-loop policy inference in each A/B process;
actions are not replayed. No actual pose is reconstructed through FK.

`analyze.py` queries the actual compiled native collision shapes at every
saved5ms boundary. Conservative bounding spheres only prune separated pairs.
It reports both feet's maximum penetration, time above thresholds and depth
integral, separately at200Hz and50Hz. It requires identical initial snapshots,
policy tensors and native model/geom configuration hashes between A and B.

`render.py` uses the exact display/camera libraries used for the earlier
checkpoint6000 videos (`display_library.py`, `camera_library.py`), now included
here with their original bytes. Its display models have zero generalized
coordinates and do not simulate physics. Both A/B panels use a shared fixed
world camera, real-time50Hz poses and the original artwork/reference colors.
It also creates a quarter-speed foot/table close-up from those same frames.

Stage this directory as an immutable helper snapshot on GRASP. The Slurm
script requires the existing frozen evaluation source, checkpoint pair and
bound assets under the tables campaign. It deliberately requests a separate
A10/12GiB/4CPU allocation; source runtimes are reused read-only. Invocation:

```bash
sbatch --qos=normal --output=/absolute/new/contact-ab-%j.log \
  /absolute/frozen/helpers/run.sbatch.sh \
  /absolute/frozen/helpers /absolute/new/output
```

Outputs include per-variant reports/pose recordings/contact traces,
`comparison.json`, `penetration_comparison.png` and paired videos. Full raw
recordings remain on the cluster; only videos, images and summary metadata
are copied for local delivery. A single-clip comparison is not a training
improvement claim or validation of a deployable physics change.
