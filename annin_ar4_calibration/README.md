# annin_ar4_calibration

TCP (tool center point) calibration for the Annin AR4 robot arm, using a
**pivot-point method**: touch one fixed physical reference point from many
different orientations, and solve for the fixed translational offset between
`ee_link` and the true tool tip.

This package is **TF-only** — it never commands the robot. Jogging the arm
(with MoveIt/RViz, teleop, or any other means) is handled entirely outside
this package by `annin_ar4_driver`/`annin_ar4_moveit_config`. It also only
solves for a translational offset, not an orientation offset.

## Prerequisites

- `annin_ar4_driver` (real hardware) or `annin_ar4_gazebo` (simulation) running,
  so that the TF tree (`base_link -> ... -> ee_link`) is being published.
- A way to jog the arm, e.g. `annin_ar4_moveit_config`'s MoveIt/RViz interface.
- A pointed tool mounted at `ee_link`, and one fixed physical reference point
  in the workspace that the tool tip can repeatedly touch.

## Launching

```bash
ros2 launch annin_ar4_calibration tcp.launch.py
```

This starts three nodes: `collector`, `calibration`, and `visualization`.

## Procedure

1. Jog the arm so the tool tip touches the reference point, in some
   orientation.
2. Record the sample:

   ```bash
   ros2 service call /collector/collect_sample std_srvs/srv/Trigger {}
   ```

3. Repeat steps 1-2 at least `min_samples` times (default 30), spreading the
   orientations as widely as possible. Avoid capturing several
   near-duplicate/near-parallel orientations in a row — this weakens the
   least-squares fit and RANSAC's ability to reject outliers.
4. Compute the calibration:

   ```bash
   ros2 service call /calibration/compute_calibration std_srvs/srv/Trigger {}
   ```

   The response reports the offset (in meters), RMS error, and inlier count.

## Verifying

- View `/visualization/markers` (a `visualization_msgs/MarkerArray`) in RViz:
  blue points are the raw touch points, green/red points are the fitted
  points colored by inlier/outlier status, and the yellow sphere is the
  fitted reference point. A tight, mostly-green cluster indicates a good
  calibration.
- Check the new `tcp` TF frame (child of `ee_link`), e.g.
  `ros2 run tf2_ros tf2_echo base_link tcp`.

## Persistence & redoing

Samples and results are stored under `$ROS_HOME/annin_ar4_calibration/`
(`ROS_HOME` defaults to `~/.ros`):

- `samples.npy` — collected samples.
- `tcp_result.yaml` — the computed offset and fit statistics. The
  `calibration` node reloads this on startup and immediately re-broadcasts
  the `tcp` TF frame, without needing to recompute.

To start over:

```bash
ros2 service call /collector/reset_samples std_srvs/srv/Trigger {}
```

This clears `samples.npy`, first backing it up to a timestamped
`samples.npy.bak-<timestamp>` file.

## Consuming the result elsewhere

The calibration result is exposed as a live TF frame (`tcp`) only. Wiring it
into MoveIt as a planning tip frame (e.g. via the SRDF/URDF) is a possible
future step and is out of scope for this package.

---

# Hand-eye calibration (Intel RealSense)

Calibrates an Intel RealSense camera against the arm, in either configuration:

- **`eye_in_hand`**: camera mounted on the wrist, observing a ChArUco board (or
  ArUco marker) fixed in the workspace. Solves `ee_link -> camera_color_optical_frame`.
- **`eye_to_hand`**: camera fixed in the workspace, observing a ChArUco board
  (or ArUco marker) mounted on `ee_link`/J6. Solves `base_link -> camera_color_optical_frame`.

Unlike TCP calibration above, **this procedure moves the robot itself**
(`run_auto_sequence` drives the arm through a series of automatically
generated poses via MoveIt). Clear the workspace and know where the e-stop
is before running it, especially the first time with a new seed pose or new
`max_translation_m`/`max_orientation_deg` bounds.

## Prerequisites

- `annin_ar4_driver` (real hardware) or `annin_ar4_gazebo` (simulation), and
  `annin_ar4_moveit_config`'s `moveit.launch.py` running, so both TF and
  MoveIt's planning interface are available.
- `realsense2_camera` launched separately (this package does not launch it),
  publishing `Image`/`CameraInfo` on the default Jazzy topics and a static TF
  from wherever it's mounted down to `camera_color_optical_frame`.
- **`pymoveit2`**, cloned into your workspace `src/` (it isn't available via
  `rosdep`/apt - vcs-import it):
  ```bash
  cd ~/ros2_ws/src
  git clone https://github.com/AndrejOrsula/pymoveit2.git
  ```
- A printed ChArUco board (default target) or ArUco marker, measured with
  calipers and matched to `config/hand_eye_params.yaml`'s `charuco.square_length_m`/
  `marker_length_m` (printers can silently rescale a PDF export - this is the
  most common source of a scaled/wrong translation result). Mount it per
  `calibration_type`:
  - `eye_in_hand`: board/marker fixed somewhere in the workspace, camera
    mounted on the wrist.
  - `eye_to_hand`: board/marker mounted on `ee_link`/J6, camera fixed
    somewhere in the workspace with a clear view of the arm's workspace.

## Launching

```bash
ros2 launch annin_ar4_calibration hand_eye.launch.py calibration_type:=eye_in_hand target_type:=charuco
```

This starts four nodes: `target_detector`, `hand_eye_collector`,
`hand_eye_calibration`, and `hand_eye_visualization`.

## Procedure

1. Check `~/target_detector/debug_image` (e.g. in `rqt_image_view`) or the
   `calibration_target` TF frame in RViz to confirm the target is being
   detected.
2. Jog the arm (MoveIt/RViz) to one starting pose where the target is
   clearly visible, then capture it as the seed:
   ```bash
   ros2 service call /hand_eye_collector/capture_seed std_srvs/srv/Trigger {}
   ```
3. Run the automated sequence - the robot moves through generated poses by
   itself, capturing a sample at each one where the target stays visible:
   ```bash
   ros2 service call /hand_eye_collector/run_auto_sequence std_srvs/srv/Trigger {}
   ```
   The response reports how many poses were collected vs. skipped (and why).
   If it collected fewer than you'd like, you can re-run it (it keeps
   appending), or top up manually:
   ```bash
   ros2 service call /hand_eye_collector/collect_sample std_srvs/srv/Trigger {}
   ```
4. Compute the calibration:
   ```bash
   ros2 service call /hand_eye_calibration/compute_calibration std_srvs/srv/Trigger {}
   ```
   The response reports the transform, and a consistency RMS (translation in
   mm, rotation in degrees) - this measures internal self-consistency, not
   ground-truth accuracy (see note below).

## Verifying

- View `~/hand_eye_visualization/markers` (a `visualization_msgs/MarkerArray`)
  in RViz: blue points are collected sample positions, and (once computed) an
  RGB axis triad shows the computed camera pose.
- Check the resulting TF frame, e.g.
  `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame`.
- The consistency RMS reported above can't catch a self-consistent-but-wrong
  calibration (e.g. a sign/mirror error) - do one physical sanity check after
  your first calibration: does `tf2_echo` place the camera/target where they
  actually are?

## Persistence & redoing

Stored under `$ROS_HOME/annin_ar4_calibration/` (separate from TCP's files):

- `hand_eye_samples.npy` - collected paired samples.
- `hand_eye_result.yaml` - the computed transform and consistency stats. The
  `hand_eye_calibration` node reloads this on startup and immediately
  re-broadcasts the result TF, without needing to recompute.

```bash
ros2 service call /hand_eye_collector/reset_samples std_srvs/srv/Trigger {}
```

## Notes

- `calibration_type` determines both the physical mounting *and* which frame
  pair gets solved for (`ee_link -> camera` for `eye_in_hand`, `base_link ->
  camera` for `eye_to_hand`). It must be consistent between
  `hand_eye_collector` and `hand_eye_calibration` (the launch file's
  `calibration_type` argument sets both together); `target_detector` doesn't
  need it since it only publishes the target's pose in the camera frame,
  independent of calibration type.
- `cv2.calibrateHandEye` needs at least a couple of motions with non-parallel
  rotation axes; the default `max_orientation_deg: 35` random orbit sampling
  around your seed pose is meant to provide that automatically, but a sample
  set that's accidentally rotated about only one axis will give a poor result.

---

# Kinematic parameter identification (WP4)

Identifies small corrections to the robot's 6 joint origins (the real arm
deviates slightly from the CAD-derived values hardcoded in
`annin_ar4_description`'s `ar_macro.xacro`), using joint angles plus the
already-computed hand-eye extrinsic (above) as an external "ground truth"
measurement device.

**This only computes and persists a result YAML - it does not patch the live
`robot_description`/URDF.** See Notes below for why, and what it would take.

## Prerequisites

- Hand-eye calibration (above) must already be computed and persisted
  (`hand_eye_result.yaml`) - this feature depends on it as a known,
  trusted extrinsic. `compute_calibration` will fail clearly if it's missing.
- Everything hand-eye calibration itself needs (driver/gazebo, MoveIt,
  RealSense, a printed target) plus a tag rigidly mounted at whichever end of
  the chain matches your `calibration_type` (same physical setup as
  hand-eye's target, or a second tag - either is fine as long as
  `calibration_type` here matches how you actually computed
  `hand_eye_result.yaml`).

## Launching

```bash
ros2 launch annin_ar4_calibration kinematic.launch.py calibration_type:=eye_to_hand target_type:=charuco
```

This starts four nodes: `target_detector`, `kinematic_collector`,
`kinematic_calibration`, and `kinematic_visualization`. It does not launch
hand-eye's own nodes.

## Procedure

1. Check the target is being detected (same as hand-eye's step 1).
2. Run the automated sequence - the robot drives itself through randomized
   joint-space configurations spanning each joint's range (**clear the
   workspace and know where the e-stop is first** - same caution as
   hand-eye's `run_auto_sequence`, and more so: this sweeps much more of the
   joint range than hand-eye's local orbit):
   ```bash
   ros2 service call /kinematic_collector/run_auto_sequence std_srvs/srv/Trigger {}
   ```
   The response reports how many configs were collected vs. skipped, and
   whether every joint's observed angle range cleared the
   `min_joint_range_deg` gate (if not, it lists which joints came up short -
   re-run to top up, it keeps appending).
3. Compute the calibration:
   ```bash
   ros2 service call /kinematic_calibration/compute_calibration std_srvs/srv/Trigger {}
   ```
   The response reports RMS before/after and the condition number for each
   method run (`sequential`, `joint_trf`, and - as a secondary diagnostic
   only - `joint_lm`; see Notes).

## Verifying

- View `/kinematic_visualization/markers` in RViz: blue points are
  nominal-FK-predicted positions, green are corrected-FK-predicted, orange
  are reconstructed from the camera measurement directly - a tight
  green/orange overlap that's visibly separated from blue indicates the
  correction is doing something real.
- A live TF frame `<tool_frame>_kinematic_corrected` (child of `base_frame`)
  tracks the corrected FK at the robot's current pose, for a running visual
  comparison against the (nominal-URDF-driven) real `ee_link` frame.
- `kinematic_result.yaml` (see Persistence below) has full diagnostics:
  per-method RMS, condition number, and singular values.

## Persistence & redoing

Stored under `$ROS_HOME/annin_ar4_calibration/`:

- `kinematic_samples.npy` - collected `(joint_angles, camera->target pose)`
  samples.
- `kinematic_result.yaml` - per-joint corrections, mount offset, and
  diagnostics for every method run. `kinematic_calibration` reloads and logs
  (but does not TF-broadcast, since the correction is theta-dependent) this
  on startup.

```bash
ros2 service call /kinematic_collector/reset_samples std_srvs/srv/Trigger {}
```

## Notes

- **Parameterization**: each joint gets 4 free correction parameters (2
  translation + 2 rotation), not 6 - rotation about a joint's own axis is
  excluded because it's exactly equivalent to shifting that joint's constant
  homing offset (already handled separately, in
  `annin_ar4_driver/config/joint_offsets/`), and translation along a joint's
  own axis is excluded because it's exactly collinear with a specific
  direction in the next joint's (or the mount offset's) translation, for any
  data. **A direct consequence: this feature cannot detect or correct
  homing-offset errors** - if your homing calibration is off, WP4's output
  won't reflect it either way.
- **`eye_in_hand` is significantly worse-conditioned than `eye_to_hand`, not
  just missing joint_1.** Joint_1's correction is *exactly* unobservable for
  `eye_in_hand` (nothing theta-dependent separates it from the free
  `base_T_marker` nuisance parameter) and is fixed to zero. But empirically
  (see `test/test_kinematic_calibration.py`), the *rest* of the `eye_in_hand`
  problem is also considerably softer than `eye_to_hand` - condition numbers
  around 1e5-1e6 vs. ~3 in synthetic tests - so per-joint corrections and
  even the mount offset itself should be trusted less there. **Prefer
  `eye_to_hand` for this feature when you have the choice** of camera
  mounting; if only `eye_in_hand` is available, treat the reported condition
  number/singular values as a first-class result, not just a diagnostic
  footnote.
- **Joints 2/3 are the one exactly-parallel-axis pair in this arm** (checked
  programmatically at `kinematic_calibration` startup, logged as a warning
  if found - do not assume from axis *labels* in the xacro, joint_4 shares
  the same `±z` label but is not actually parallel). This near-singularity is
  mitigated with ridge regularization and bounds, not corrected exactly - the
  reported condition number/singular values surface it rather than silently
  absorbing it into an arbitrary split between the two joints.
- **`method: both` (default)** runs `sequential` and `joint_trf`, plus
  `joint_lm` as a secondary, explicitly non-unique diagnostic (unconstrained
  Levenberg-Marquardt isn't guaranteed to land at a meaningful point given
  the near-singularities above) - compare their RMS/condition numbers rather
  than trusting either blindly.
- **Scope boundary**: the result YAML is never automatically applied to the
  live URDF. Doing so safely would mean adding a new xacro-consumed
  correction-parameters YAML (analogous to
  `annin_ar4_driver/config/joint_offsets/`) and replacing the 6 hardcoded
  `origin xyz=".." rpy=".."` literals in `ar_macro.xacro` with substitutions
  from it, then regenerating `robot_description` and restarting
  `robot_state_publisher`/`move_group`/the hardware interface - a separate,
  riskier change to `annin_ar4_description` that's out of scope here.

---

# Running in simulation (no hardware)

All three work packages run against `annin_ar4_gazebo` instead of the real arm,
with a simulated camera and ChArUco board. Two reasons to care:

1. The pipeline can be developed and exercised with the arm unavailable.
2. Simulation supplies **exact ground truth**, which hardware cannot. Every
   recovered parameter can be scored against its true value.

```bash
# Terminal 1 - arm + camera + board + MoveIt + ground truth
ros2 launch annin_ar4_gazebo calibration_sim.launch.py \
    calibration_type:=eye_to_hand headless:=True

# Terminal 2 - the usual work packages, with use_sim_time:=True
ros2 launch annin_ar4_calibration hand_eye.launch.py \
    use_sim_time:=True calibration_type:=eye_to_hand
```

The procedures themselves are unchanged - same services, same order, same result
files. Only two things differ:

- **`use_sim_time:=True` is required.** It also layers
  `config/sim_overrides.yaml` on top of the hardware defaults (longer settle
  time, more candidate poses, board geometry matching the generated asset).
  Without it, TF lookups fail against Gazebo's clock.
- **`calibration_type` must match** what `calibration_sim.launch.py` was
  launched with - it decides where the camera and board are physically mounted.

See `annin_ar4_gazebo/README.md` for the ground-truth mechanism, the perturbation
files, and the simulation-specific pitfalls.

## Scoring a result against ground truth

`calibration_sim.launch.py` starts a `ground_truth` node. After any
`compute_calibration`:

```bash
ros2 service call /ground_truth/compare std_srvs/srv/Trigger {}
```

It writes `$ROS_HOME/annin_ar4_calibration/ground_truth.yaml` with the true
values *and* a `comparison` block, using the same keys and units as the result
files so one script can load both. That file is the recovered-versus-true table
for a run; archive it with `kinematic_result.yaml` and `perturbed_robot.urdf`.

Ground truth comes from two places. Hand-eye, mount-offset and TCP truths are
rigid transforms in the nominal URDF, published as `_gt` TF frames. Kinematic
corrections are the injected perturbation verbatim - `core/perturbation.py`
shares `core/kinematic_model.py`'s parameterization exactly, so no conversion
sits between the injected value and the recovered one.

## WP2 (TCP) in simulation

The manual procedure above works, but the simulated tool has no collision
geometry, so there is no contact feedback - lining the needle up with the
reference sphere is pure eyeballing, and 30 touches at 1-2 mm each would swamp
every other error term. Use the automated poser instead:

```bash
ros2 launch annin_ar4_calibration tcp.launch.py use_sim_time:=True
ros2 run annin_ar4_calibration tcp_sim_poser --ros-args \
    -p num_poses:=40 -p touch_noise_std_m:=0.0 \
    -p perturbation_file:=<abs path to the perturbation used for the sim>
ros2 service call /tcp_sim_poser/run_sequence std_srvs/srv/Trigger {}
ros2 service call /calibration/compute_calibration std_srvs/srv/Trigger {}
```

**Pass the same `perturbation_file` the simulation was launched with.** The node
solves IK against the *true* plant (nominal chain plus injected corrections) and
commands joint angles directly, so the true tip lands on the reference point. If
it placed the *nominal* tip there instead, the pivot equations would be exactly
satisfied in nominal coordinates, the solver would return the ground-truth offset
with zero residual, and the kinematic error would be invisible.

`touch_noise_std_m` injects a controlled "human touch error" - sweeping it gives
an accuracy-versus-touch-precision curve that a human cannot produce repeatably.

## What to expect

- **`eye_in_hand` results will look much worse than `eye_to_hand`.** That is not
  a defect - joint_1's correction is exactly unobservable and is fixed to zero,
  and the rest of the problem is far softer (see the WP4 notes above).
  Simulation is the right place to demonstrate that quantitatively: run both,
  report the condition numbers, treat `eye_to_hand` as the primary result.
- **Joints 2/3 are exactly parallel**, so the split between them is ill-posed.
  Expect per-joint error concentrated there, and compare the composed J2+J3
  correction rather than the individual ones.
- **With `zero.yaml` (the default) the recovered corrections should be ~0** and
  `rms_after ~= rms_before`. Run that baseline first: a large hand-eye error with
  no perturbation means a systematic problem (board scale, mirroring, intrinsics),
  not noise.
- Detected target pose agrees with ground truth to about **0.6 mm / 0.06°** at
  640x480, better at 1280x720. Nothing recovered below that is real.
