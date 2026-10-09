# annin_ar4_calibration

Calibration for the Annin AR4 robot arm in ROS 2, in three work packages:

| | Launch file | Nodes | Solves | Moves the arm |
|---|---|---|---|---|
| [**WP2 – TCP**](#tcp-calibration-wp2) | `tcp.launch.py` | `collector`, `calibration`, `visualization` | `ee_link → tcp` translation (pivot method) | no |
| [**WP3 – Hand-eye**](#hand-eye-calibration-wp3) | `hand_eye.launch.py` | `target_detector`, `hand_eye_collector`, `hand_eye_calibration`, `hand_eye_visualization` | `ee_link → camera` (eye-in-hand) or `base_link → camera` (eye-to-hand) | **yes** |
| [**WP4 – Kinematic**](#kinematic-parameter-identification-wp4) | `kinematic.launch.py` | `target_detector`, `kinematic_collector`, `kinematic_calibration`, `kinematic_visualization` | corrections to the six joint origins | **yes** |

WP4 uses the WP3 extrinsic as its measuring device, so run WP3 first, with the
same camera mounting. WP2 is independent of both.

Everything also runs against `annin_ar4_gazebo`, where the robot carries a known
kinematic error and every result can be scored against the truth
([Running in simulation](#running-in-simulation-no-hardware)). The
[benchmark harness](#benchmarking) repeats that over swept conditions and seeds.

## Safety

During `run_auto_sequence`, `hand_eye_collector` and `kinematic_collector`
drive the arm themselves (in simulation, so does `tcp_sim_poser`):

- Motions are planned by MoveIt and run at **full speed**: no velocity or
  acceleration scaling is applied.
- `kinematic_collector` sweeps most of every joint's range, not a small orbit
  around one pose.
- After repeated motion failures, `kinematic_collector` sends a recovery
  trajectory **straight to the joint trajectory controller, bypassing MoveIt —
  no planning and no collision checking** (see
  [What `run_auto_sequence` does](#what-run_auto_sequence-does)).

Clear the workspace and keep a hand on the e-stop on a first run.

## Prerequisites

- `annin_ar4_driver` (hardware) or `annin_ar4_gazebo` (simulation) running, so
  the TF tree `base_link → … → ee_link` is published.
- **WP3/WP4:** `annin_ar4_moveit_config`'s `moveit.launch.py` (the collectors
  plan through MoveIt), and `realsense2_camera` launched separately (this
  package does not start it), publishing `/camera/camera/color/image_raw` and
  `/camera/camera/color/camera_info`.
- **WP3/WP4:** [`pymoveit2`](https://github.com/AndrejOrsula/pymoveit2) in your
  workspace `src/`. It is not available via rosdep/apt. The collectors rely on
  it therefore it is included as a submodule.
- **WP3/WP4:** a printed ChArUco board (default: 5 × 7 squares of 35 mm with
  26 mm markers, `DICT_5X5_250`) or a single ArUco marker. **Measure the print**
  and set `charuco.square_length_m` / `charuco.marker_length_m` in
  `config/hand_eye_params.yaml` and `config/kinematic_params.yaml` to what you
  measured. Printers silently rescale PDFs, and a wrong scale scales every
  result without any error showing up.
- **WP2:** a pointed tool mounted at `ee_link`, and one fixed reference point in
  the workspace that the tip can touch repeatedly.

## Launch arguments

| Argument | Launch files | Default | Meaning |
|---|---|---|---|
| `use_sim_time` | all three | `False` | Set `True` against Gazebo. Also layers `config/sim_overrides.yaml` on top of the defaults (longer settle time, more candidates, the generated board's geometry). |
| `extra_params_file` | all three | `''` | A ROS 2 parameter YAML applied last: the simplest way to change any default described below. |
| `calibration_type` | `hand_eye`, `kinematic` | `eye_in_hand` / `eye_to_hand` | Camera mounting. **The two defaults differ.** |
| `target_type` | `hand_eye`, `kinematic` | `charuco` | `charuco` or `aruco`. |

Always pass `calibration_type` explicitly, and pass the same value to WP3 and
WP4: the hand-eye result records the mounting it was computed for, and WP4
refuses a result computed for the other one.

An `extra_params_file` is an ordinary parameter file, for example:

```yaml
kinematic_calibration:
  ros__parameters:
    min_samples: 40
```

## Files and topics

Samples and results live in `$ROS_HOME/annin_ar4_calibration/`
(`~/.ros/annin_ar4_calibration/` by default):

| | Samples | Result |
|---|---|---|
| WP2 | `samples.npy` | `tcp_result.yaml` |
| WP3 | `hand_eye_samples.npy` | `hand_eye_result.yaml` |
| WP4 | `kinematic_samples.npy` | `kinematic_result.yaml` |

- Every sample file has a `<file>.meta.yaml` describing how the samples were
  taken (frames, and for WP3/WP4 more). WP3 and WP4 also write
  `<file>.detections.csv`: the corner count and reprojection RMS of each
  detection.
- In simulation, `/ground_truth/compare` writes `ground_truth.yaml` next to them.
- The collectors **append**: re-running a sequence or `collect_sample` adds to
  the existing samples. To start over, call the collector's `reset_samples`
  service. It backs the samples up to `<file>.bak-<timestamp>` and deletes the
  `.detections.csv`.
- Each calibration node reloads its result on startup. The TCP and hand-eye
  nodes immediately re-broadcast their TF frame.

Topics worth knowing:

- `/target_detector/debug_image`: the camera image with the detection drawn
  (`publish_debug_image`, on by default).
- `/target_detector/detection_quality`: `[corner count, reprojection RMS px]`
  for each detection.
- `/visualization/markers`, `/hand_eye_visualization/markers`,
  `/kinematic_visualization/markers`: RViz markers, described per work package
  below.

---

# TCP calibration (WP2)

Pivot-point method: touch one fixed reference point with the tool tip from many
different orientations, and solve for the fixed translation between `ee_link`
and the tip. This work package only reads TF and never commands the arm — jog it
with MoveIt/RViz, teleop or any other means. It solves the translation only, not
an orientation.

## Launching

```bash
ros2 launch annin_ar4_calibration tcp.launch.py
```

This starts `collector`, `calibration` and `visualization`.

## Procedure

1. Jog the arm so the tool tip touches the reference point.
2. Record the sample:
   ```bash
   ros2 service call /collector/collect_sample std_srvs/srv/Trigger {}
   ```
3. Repeat at least `min_samples` times (**10** with the shipped
   `config/tcp_params.yaml`), spreading the tool orientations as widely as
   possible. Near-parallel orientations leave the offset poorly determined while
   the residual still looks small. Setting the collector's
   `min_orientation_change_deg` (default `0`, off) refuses a touch whose
   orientation is too close to the previous one.
4. Compute the calibration:
   ```bash
   ros2 service call /calibration/compute_calibration std_srvs/srv/Trigger {}
   ```
   The response reports the offset (in metres), the RMS residual of the
   inliers, and the inlier count.

## Verifying

- View `/visualization/markers` in RViz: blue points are the raw touch points,
  green/red points the fitted points coloured by inlier/outlier status, and the
  yellow sphere the fitted reference point. A tight, mostly green cluster means
  the fit is consistent.
- Check the `tcp` TF frame (child of `ee_link`), e.g.
  `ros2 run tf2_ros tf2_echo base_link tcp`.

## Persistence & redoing

`tcp_result.yaml` holds the offset and fit statistics. To start over:

```bash
ros2 service call /collector/reset_samples std_srvs/srv/Trigger {}
```

## Notes

- **RANSAC is on by default** (`ransac.enabled: true`): consensus sets from
  subsets of 3 touches, 100 iterations, a 2 mm threshold on the residual norm,
  and a final fit on the inliers. It only helps when some touches are genuinely
  bad, such as a slipped tip. In the simulated benchmark, which has no such
  outliers, it made the TCP *less* accurate in every condition — 1.91 against
  1.21 mm on average, 3.83 against 1.87 mm at 2 mm of touch noise. The residuals
  contain the arm's own kinematic error as well as touch error, and a 2 mm
  threshold then discards valid touches. Unless you expect slipped touches, turn
  it off:
  ```yaml
  calibration:
    ros__parameters:
      ransac:
        enabled: false
  ```
- With plain least squares, the benchmark's TCP error fell from 1.82 mm with 10
  touches to 0.86 mm with 40.

## Consuming the result elsewhere

The result is exposed as a live TF frame (`tcp`) only. Wiring it into MoveIt as
a planning tip frame (e.g. via the SRDF/URDF) can be done manually.

---

# Hand-eye calibration (WP3)

Calibrates the camera against the arm, in either mounting:

- **`eye_in_hand`**: camera on the wrist, observing a ChArUco board (or ArUco
  marker) fixed in the workspace. Solves `ee_link → camera_color_optical_frame`.
- **`eye_to_hand`**: camera fixed in the workspace, observing a board mounted on
  `ee_link`/J6. Solves `base_link → camera_color_optical_frame`.

**This moves the robot** — see [Safety](#safety).

## Launching

```bash
ros2 launch annin_ar4_calibration hand_eye.launch.py calibration_type:=eye_in_hand target_type:=charuco
```

This starts `target_detector`, `hand_eye_collector`, `hand_eye_calibration` and
`hand_eye_visualization`.

## Procedure

1. Confirm the target is detected: `/target_detector/debug_image` (e.g. in
   `rqt_image_view`) or the `calibration_target` TF frame in RViz.
2. Jog the arm to one pose where the target is clearly visible, and capture it
   as the seed:
   ```bash
   ros2 service call /hand_eye_collector/capture_seed std_srvs/srv/Trigger {}
   ```
3. Run the automated sequence:
   ```bash
   ros2 service call /hand_eye_collector/run_auto_sequence std_srvs/srv/Trigger {}
   ```
   The collector visits up to `num_candidate_poses` (20) poses around the seed —
   each offset by up to `max_translation_m` (30 mm) per axis and rotated by up to
   `max_orientation_deg` (35°) about a random axis — and records a sample
   wherever the target is detected after the arm has settled. It stops once it
   has `min_samples` (10). The response reports how many poses were collected
   and why the others were skipped. Re-running appends; you can also top up by
   hand:
   ```bash
   ros2 service call /hand_eye_collector/collect_sample std_srvs/srv/Trigger {}
   ```
4. Compute the calibration:
   ```bash
   ros2 service call /hand_eye_calibration/compute_calibration std_srvs/srv/Trigger {}
   ```
   It uses `hand_eye_method` (default `PARK`; also `TSAI`, `HORAUD`, `ANDREFF`
   and `DANIILIDIS`, all OpenCV's `calibrateHandEye`) and reports the transform
   plus a consistency RMS (mm and degrees). The consistency RMS measures internal
   self-consistency, not accuracy.

## Verifying

- View `/hand_eye_visualization/markers` in RViz: blue points are the collected
  sample positions, and once computed an RGB axis triad shows the camera pose.
- Check the TF frame, e.g.
  `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame`.
- The consistency RMS cannot catch a self-consistent but wrong result (e.g. a
  sign or mirror error). After your first calibration, check physically that
  `tf2_echo` puts the camera and the target where they actually are.

## Persistence & redoing

`hand_eye_result.yaml` holds the transform, the consistency statistics and the
`calibration_type` it was computed for. To start over:

```bash
ros2 service call /hand_eye_collector/reset_samples std_srvs/srv/Trigger {}
```

## Notes

- `calibration_type` decides both the physical mounting and which frame pair is
  solved for. The launch argument sets it for the collector and the calibration
  node together; `target_detector` does not need it.
- `cv2.calibrateHandEye` needs motions about several non-parallel rotation axes.
  The random orbit around the seed is meant to provide that; a sample set rotated
  about one axis only gives a poor result.
- **Choosing a method.** In the simulated benchmark, compared on the same data,
  Park, Tsai and Horaud were indistinguishable (2.6 mm with `eye_in_hand`,
  4.2 mm with `eye_to_hand`), Daniilidis was within 0.3 mm of them, and Andreff
  was clearly the least accurate (4.5 and 10.5 mm).
- **More poses** improved the `eye_in_hand` extrinsic steadily (Park: 3.4 mm with
  10 poses, 1.0 mm with 150), while the `eye_to_hand` extrinsic levelled off above
  3 mm. With `eye_to_hand`, this error carries straight into where WP4 places the
  arm (see the WP4 notes).

---

# Kinematic parameter identification (WP4)

Identifies small corrections to the six joint origins of the robot description
(the real arm deviates from the CAD values hardcoded in `annin_ar4_description`'s
`ar_macro.xacro`), using joint angles together with the WP3 extrinsic as the
external measuring device.

**It computes and stores a result YAML; it does not patch the live
`robot_description`/URDF.** See Notes for why, and what that would take.

## Prerequisites

- **WP3 computed for the same `calibration_type`.** `run_auto_sequence` already
  needs `hand_eye_result.yaml` to predict where the board will appear; it is not
  only needed by `compute_calibration`. (If the extrinsic is published on TF, as
  `ee_link` or `base_link → camera_color_optical_frame`, the collector uses that
  instead; the calibration node always uses the file.)
- Everything WP3 needs, plus the board rigidly mounted at the end of the chain
  that matches `calibration_type`: fixed in the workspace for `eye_in_hand`, on
  the flange for `eye_to_hand`.
- `target_size_m` (in `kinematic_collector`) set to the size of the board's
  square grid — default `[0.175, 0.245]` m, i.e. 5 × 7 squares of 35 mm. The
  visibility filter depends on it.

## Launching

```bash
ros2 launch annin_ar4_calibration kinematic.launch.py calibration_type:=eye_to_hand target_type:=charuco
```

This starts `target_detector`, `kinematic_collector`, `kinematic_calibration`
and `kinematic_visualization`. It does not start WP3's nodes.

## Procedure

1. Jog the arm to a pose where the target is detected (check
   `/target_detector/debug_image`) and leave it there.
2. Run the automated sequence. **This moves the arm across most of every joint's
   range** — see [Safety](#safety):
   ```bash
   ros2 service call /kinematic_collector/run_auto_sequence std_srvs/srv/Trigger {}
   ```
   [What it does](#what-run_auto_sequence-does) is described below. The response
   reports how many configurations were collected and attempted, why the others
   were skipped, the candidate yield, and — if the sequence stopped early — why.
   If the per-joint range gate was not met, it lists the joints that came up
   short; re-running appends.
3. Compute the calibration:
   ```bash
   ros2 service call /kinematic_calibration/compute_calibration std_srvs/srv/Trigger {}
   ```
   It needs at least `min_samples` (80) samples and 30° of movement on every
   joint. It runs `sequential` and `joint_trf` (the primary result) and, with
   `run_lm_diagnostic`, `joint_lm`. For each it reports the translational
   loop-closure RMS before → after and the condition number.

## What `run_auto_sequence` does

1. Waits up to `visibility_setup_timeout_sec` (60 s) for CameraInfo and for a
   detection stamped after the call.
2. Locates the board from that one detection and the current joint angles, and
   learns which face of the board carries the markers.
3. Draws `num_candidate_configs` (200) joint configurations uniformly within each
   joint's limits shrunk by `joint_margin_deg` (15°). It keeps only those from
   which a pinhole model predicts a usable view: the board centre and at least 3
   of its 4 corners inside the image with a 20 px margin, 0.10–1.20 m from the
   camera, at most 65° off-axis, printed side towards the camera (the
   `visibility_*` parameters).
4. Moves through them and records a sample wherever a detection stamped after
   the settle time (`settle_time_sec`) arrives, until it has `min_samples` (100)
   **and** every joint has moved through at least `min_joint_range_deg` (30°).
5. After `max_consecutive_motion_failures` (10) failed motions in a row, sends a
   recovery trajectory straight to the joint trajectory controller — clamping
   any joint that is outside its limits back inside the sampling range — and
   continues once the measured joint states are within limits. After
   `max_recovery_attempts` (2) recoveries, or if a recovery fails, the sequence
   stops and reports why. `max_sequence_duration_sec` (default `0`, off) caps the
   whole sequence.

## Verifying

- View `/kinematic_visualization/markers` in RViz: blue points are the flange
  positions predicted by the nominal model, green those predicted by the
  corrected model, orange those implied by the camera measurements. Close
  green/orange agreement shows that the model **fits** the measurements, not
  that it is accurate (see [Judging a result](#judging-a-result)).
- A live TF frame `<tool_frame>_kinematic_corrected` (child of `base_frame`)
  tracks the corrected forward kinematics at the current joint angles, for
  comparison with the nominal `ee_link`.
- `kinematic_result.yaml` holds every method's corrections, mount offset, RMS,
  condition number and singular values.

## Judging a result

The RMS that `compute_calibration` reports and the RViz overlay are in-sample
fit statistics. In the simulated benchmark, neither they nor a cross-validated
(held-out) RMS tracked the true accuracy of `eye_in_hand` calibrations: a
calibration could fit as well as any other and still be worse than no
calibration at all.

A better check is the **stability** of the result: re-solve on bootstrap
resamples of the samples and see how far the calibrated forward kinematics
moves. The calibration node does not compute this. The benchmark's offline
analysis does, on the samples you already have:

```bash
mkdir -p ~/wp4_check
cd ~/.ros/annin_ar4_calibration
cp kinematic_samples.npy kinematic_samples.npy.meta.yaml hand_eye_result.yaml ~/wp4_check/
ros2 param get /robot_state_publisher robot_description --hide-type > ~/wp4_check/nominal_robot.urdf
ros2 run annin_ar4_calibration benchmark_run --mode offline --run-dir ~/wp4_check
```

In `~/wp4_check/metrics.yaml`, each solver variant reports
`bootstrap.fk_spread_mm`: the RMS distance, over the collected configurations,
between the flange positions of the full-data calibration and of each of 20
resampled ones (with ridge 1e-6 by default).

- A spread above about **2 mm** means the solution is floating along directions
  the measurements do not constrain; reject it. In the benchmark this flagged 73
  of the 80 `eye_in_hand` calibrations with a true error above 5 mm, none of
  which came from `joint_trf`.
- A small spread does **not** prove accuracy. An error shared by every
  measurement — a biased hand-eye extrinsic, a mis-scaled board — and, with
  `eye_in_hand`, the arm's placement on its base (see Notes) are the same in
  every resample, so they never show up in it.

## Persistence & redoing

`kinematic_result.yaml` stores the primary method's corrections and mount
offset, plus the diagnostics of every method run. `kinematic_calibration`
reloads and logs it on startup, but does not broadcast it as TF, since the
correction depends on the joint angles. To start over:

```bash
ros2 service call /kinematic_collector/reset_samples std_srvs/srv/Trigger {}
```

## Notes

- **Parameterization.** Each joint gets 4 correction parameters (2 translation,
  2 rotation), not 6. Rotation about a joint's own axis is excluded because it
  is exactly equivalent to shifting that joint's homing offset, which is handled
  separately in `annin_ar4_driver/config/joint_offsets/`. Translation along the
  axis is excluded because it is exactly collinear with a direction of the next
  joint's translation (or of the mount offset), for any data. **As a
  consequence, WP4 cannot detect or correct homing-offset errors.**
- **Where the arm stands on its base.** Both mountings correct the arm's own
  geometry well, but the camera alone cannot pin down where the arm as a whole
  stands relative to `base_link`:
  - `eye_in_hand`: a rigid move of the whole arm is absorbed by the unknown
    board pose. Joint 1's corrections are therefore fixed to zero, and two
    further rigid motions — a vertical shift and a rotation about the vertical
    axis — remain unconstrained; only the ridge term and the bounds hold them.
    The placement keeps its uncalibrated error.
  - `eye_to_hand`: the fixed camera anchors the base frame, so the corrections
    of joints 1 and 2 place the arm wherever the hand-eye extrinsic says,
    including the extrinsic's error.

  In the simulated benchmark (`joint_trf`, ridge 1e-3; injected joint-origin
  errors with σ up to 2 mm and 0.1°), the flange position error went from 2.73
  to 1.85 mm with `eye_in_hand` and from 2.74 to 2.67 mm with `eye_to_hand`
  (0.74 mm given the exact extrinsic). With the placement removed, the geometry
  error went from about 1.5 mm to 0.41 and 0.32 mm respectively. Neither mounting
  is simply better: `eye_to_hand` is only as good as its hand-eye stage.
- **Conditioning.** `eye_to_hand` is well conditioned (condition number ≈ 24 at
  any ridge weight). The `eye_in_hand` condition number — about 1e3 at ridge
  1e-3, 3e4 at 1e-6 and 8e5 at 1e-9 — is set entirely by the two unconstrained
  directions above; apart from them the problem is as well conditioned as
  `eye_to_hand`.
- **Joints 2 and 3 are parallel** (`kinematic_calibration` logs a warning at
  startup). With translations along the joint axes already excluded, this pair
  does not create an unidentifiable direction: in the benchmark's identification
  Jacobian, the only near-zero singular values were the `eye_in_hand` placement
  directions.
- **Solvers.** `joint_trf` stays near the nominal placement at every ridge
  weight, because of its ±10 mm / ±3° bounds. `joint_lm` has no bounds: with weak
  regularization it drifts along the `eye_in_hand` placement directions while
  its residuals stay as small as `joint_trf`'s. In the benchmark, 58 of 60 of its
  `eye_in_hand` runs ended above 5 mm at ridge 1e-9, 27 % at 1e-6 and none at
  1e-3. Treat it as a diagnostic only. `sequential` stops at its cap of 10 outer
  iterations without meeting its tolerance, so `converged: false` is expected for
  it. The shipped `ridge_lambda` is 1e-6; the benchmark's reference
  configuration was `joint_trf` with 1e-3.
- **How many samples.** `kinematic_calibration` refuses fewer than `min_samples`
  (80). In the benchmark, 40 samples gave the same flange error as 150 (the
  placement dominates it), while more samples kept improving the geometry. To
  calibrate from fewer, lower `min_samples` with an `extra_params_file`.
- **Scope boundary.** The result YAML is never applied to the live URDF
  automatically. Doing so safely would mean adding an xacro-consumed
  correction-parameters YAML (analogous to `annin_ar4_driver/config/joint_offsets/`),
  replacing the six hardcoded `origin xyz=".." rpy=".."` literals in
  `ar_macro.xacro` with substitutions from it, then regenerating
  `robot_description` and restarting `robot_state_publisher`, `move_group` and
  the hardware interface — a separate, riskier change to `annin_ar4_description`.

---

# Running in simulation (no hardware)

All three work packages run against `annin_ar4_gazebo` instead of the real arm,
with a simulated camera and ChArUco board. Two reasons to care:

1. The pipeline can be developed and exercised while the arm is unavailable.
2. Simulation supplies **exact ground truth**, which hardware cannot: the spawned
   robot carries a known kinematic error while everything else keeps the nominal
   model, so every recovered parameter can be scored against its true value.

```bash
# Terminal 1 - arm + camera + board + MoveIt + ground truth
ros2 launch annin_ar4_gazebo calibration_sim.launch.py \
    calibration_type:=eye_to_hand headless:=True

# Terminal 2 - the usual work packages, with use_sim_time:=True
ros2 launch annin_ar4_calibration hand_eye.launch.py \
    use_sim_time:=True calibration_type:=eye_to_hand
```

The procedures are unchanged: same services, same order, same result files.
Two things differ:

- **`use_sim_time:=True` is required.** It also layers
  `config/sim_overrides.yaml` on top of the hardware defaults. Without it, TF
  lookups fail against Gazebo's clock.
- **`calibration_type` must match** what `calibration_sim.launch.py` was
  launched with; it decides where the camera and the board are mounted.

Useful `calibration_sim.launch.py` arguments:

| Argument | Default | |
|---|---|---|
| `calibration_type` | `eye_to_hand` | where the camera and the board are mounted |
| `perturbation_file` | `config/perturbations/zero.yaml` | known kinematic error baked into the spawned robot only (`example_1mm.yaml` is also shipped) |
| `image_noise_stddev` | `0.0` | Gaussian noise on the colour image (normalised intensity) |
| `camera_width`, `camera_height`, `camera_hfov` | `1280`, `720`, `1.2` rad | the simulated camera |
| `headless`, `rviz` | `False`, `True` | Gazebo GUI and RViz |
| `enable_depth` | `False` | adds a depth camera for MoveIt's octomap; the calibration does not use it |

See `annin_ar4_gazebo/README.md` for the ground-truth mechanism, the
perturbation files and the simulation-specific pitfalls.

## Scoring a result against ground truth

This section covers scoring **one** run by hand. For repeated runs, swept
conditions, error bars and aggregated tables, see [Benchmarking](#benchmarking).

`calibration_sim.launch.py` starts a `ground_truth` node. After any
`compute_calibration`:

```bash
ros2 service call /ground_truth/compare std_srvs/srv/Trigger {}
```

It writes `$ROS_HOME/annin_ar4_calibration/ground_truth.yaml` with the true
values *and* a `comparison` block, using the same keys and units as the result
files so one script can load both. For WP4 the comparison is per joint and for
the mount offset — diagnostics only. The task-space error (how far apart the
true and the calibrated model place the flange) comes from the benchmark's
offline analysis: run it on the data directory as in
[Judging a result](#judging-a-result), and with `ground_truth.yaml` present it
adds `truth.fk_agreement`.

Ground truth comes from two places. The hand-eye, mount-offset and TCP truths are
rigid transforms in the nominal URDF, published as `_gt` TF frames. The kinematic
corrections are the injected perturbation verbatim: `core/perturbation.py`
shares `core/kinematic_model.py`'s parameterization exactly, so no conversion
sits between the injected value and the recovered one.

## WP2 (TCP) in simulation

The manual procedure works, but the simulated tool has no collision geometry, so
there is no contact feedback. Lining the needle up with the reference sphere is
pure eyeballing, and 30 touches at 1–2 mm each would swamp every other error
term. Use the automated poser instead:

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

`touch_noise_std_m` injects a controlled "human touch error"; sweeping it gives
an accuracy-versus-touch-precision curve that a human cannot produce repeatably.
Other parameters: `reference_point` (default `[0.30, 0.10, 0.20]` m),
`max_tilt_deg` (35°) and `seed` (0, which fixes the touch sequence).


# Benchmarking

Every number the three work packages report after `compute_calibration` —
`rms_error`, `consistency_rms_m`, `rms_after_m` — is an **in-sample fit
statistic**, not an accuracy. Each measures how well a model reproduces the data
it was fitted to, and each falls as free parameters are added.

`annin_ar4_calibration/benchmark/` adds the layer above: swept conditions,
repeated seeds, held-out cross-validation, bootstrap spreads, and aggregation
into a long-format CSV plus figures and LaTeX tables. Its metrics are split into
those that need simulation's ground truth and those that do not, so the same
analysis applies to data from the physical arm.

```bash
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml --dry-run   # expand the grid, launch nothing
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml --limit 1   # one cell, to check the rig
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml --resume    # skips cells already collected and analysed
ros2 run annin_ar4_calibration benchmark_run --mode offline --run-dir <dir>   # analyse archived samples, no robot
ros2 run annin_ar4_calibration benchmark_aggregate                            # -> results.csv
ros2 run annin_ar4_calibration benchmark_report                               # -> figures and LaTeX tables
ros2 run annin_ar4_calibration benchmark_thesis_numbers --out-dir <dir> --figure-dir <dir> --refits
```

`benchmark_thesis_numbers` recomputes the numbers quoted in the thesis into
`thesis_numbers.json`; `--refits` adds the slower checks that re-solve
calibrations.

See [`annin_ar4_calibration/benchmark/README.md`](annin_ar4_calibration/benchmark/README.md)
for the design, the metric families and the pitfalls.

---

# Tests

With the ROS environment sourced, from this package's directory:

```bash
python3 -m pytest test/
```

The tests run without a robot, a simulator or a running ROS graph.
