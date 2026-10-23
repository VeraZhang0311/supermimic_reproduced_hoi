# HOI80 zero-shot

Evaluate baseline OFF's immutable `ckpt_014000.pt` on the existing canonical
80 OMOMO validation motions. No training, optimizer, or normalizer updates.
The original policy receives its unchanged 99-dimensional proprioceptive input
and 730-dimensional human reference goal. Dynamic objects collide with the robot
and ground; object pose, velocity, geometry and contact are absent from actor inputs.

The trained G1, PD, hands, self-collision OFF, analytic reward and termination
definitions remain unchanged. Object meshes and nominal mass/COM/inertia come
from the frozen HOI inputs. Newton's CoACD and MuJoCo-Warp replace the previous
PhysX/VHACD execution, so this is not identical ULTRA benchmark physics.
Actual object mass, COM, inertia, initial pose, compiled free joints and enabled
object/robot collision pairs must pass native checks for every asset.
MuJoCo-Warp collision/constraint capacity flags are checked after every control
step; an overflow preserves evidence and aborts before publishing evaluation.

Source motions are verified against all 80 hashes. Original 60Hz motions are
sampled on integer 50Hz ticks without changing their duration. Full references
play without failure resets. The author-normalized return stops accumulating at
the first author termination; the existing PHC and engineering object-pose
diagnostics score the entire reference. The object-pose diagnostic is not a
contact/grasp-success judge. The 39-body virtual FK must match the trained
robot's 14 common links before these diagnostics are used.

Each case gets a compressed NPZ, including actual root/joint states and velocities,
physical link poses, virtual39 body poses/velocities, dynamic object pose/COM
velocity, full HOI reference, action, actor observation and goal. State/reference
arrays have T+1 frames; action/observation/goal arrays have T frames and describe
the control taking state[t] to state[t+1]. Root quaternions are wxyz; object/body
quaternions are xyzw. Failures remain in the full recording for later rendering.
All files have SHA256 receipts. No weights, raw rollouts or source are uploaded.

`ultra_math.py`, `ultra_reference.py`, `ultra_kinematics.py` and `ultra_metrics.py`
reuse the user's existing ProtoMotions HOI diagnostic helpers without modification.
Original experiments and frozen ProtoMotions recipes are read-only.
