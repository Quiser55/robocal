# annin_ar4_gazebo

Gazebo (gz-sim Harmonic) simulation of the AR4, in two flavours:

- **`gazebo.launch.py`** — the plain arm, no camera. Unchanged in behaviour.
- **`calibration_sim.launch.py`** — the arm plus a simulated RGB(-D) camera, a
  ChArUco board and a TCP tool, for running `annin_ar4_calibration`'s three
  work packages **with exact ground truth**.

---

## Quick start

```bash
# Terminal 1 - sim + MoveIt + camera + board + ground truth
ros2 launch annin_ar4_gazebo calibration_sim.launch.py \
    calibration_type:=eye_to_hand headless:=True

# Terminal 2 - one work package at a time (note use_sim_time)
ros2 launch annin_ar4_calibration tcp.launch.py       use_sim_time:=True
ros2 launch annin_ar4_calibration hand_eye.launch.py  use_sim_time:=True calibration_type:=eye_to_hand
ros2 launch annin_ar4_calibration kinematic.launch.py use_sim_time:=True calibration_type:=eye_to_hand

# after each compute_calibration, score it against the truth
ros2 service call /ground_truth/compare std_srvs/srv/Trigger {}
```

`calibration_type` sets the mounting for the whole rig, and must match what you
pass to the calibration launch files:

| | camera | ChArUco board |
|---|---|---|
| `eye_in_hand` | on the wrist | on a stand in the workspace |
| `eye_to_hand` | on a stand in the workspace | on the wrist |

`eye_to_hand` is much better conditioned for WP4 (kinematic identification) —
see the calibration package's README.

**Use `headless:=True` for anything that collects data.** The GUI is a second
full rendering client competing for the same GPU; on a modest machine it starves
the camera sensor's render thread. Measured here: 9 Hz headless at 640×480
versus 1.7 Hz with the GUI open.

---

## Ground truth: how the simulation knows the right answer

This is the reason to calibrate in simulation at all, and it is worth
understanding before trusting any number that comes out of it.

`robot_state_publisher` publishes the **nominal** URDF. Gazebo spawns a
**perturbed** copy of it, whose six joint origins carry a known error read from
`config/perturbations/*.yaml`. So:

- MoveIt, ros2_control and every calibration node see the nominal model — they
  are *estimating*, exactly as they would against hardware.
- The robot physically in the world really is slightly different.
- WP4 therefore has a real error to identify, and its true value is known
  exactly. Physical hardware can never provide that.

The parameterization is deliberately identical to
`annin_ar4_calibration/core/kinematic_model.py` (4 coefficients per joint, in
that joint's own `orthogonal_basis`), so **the numbers in the perturbation YAML
are directly comparable to `kinematic_result.yaml`'s recovered corrections with
no conversion**. `core/perturbation.py` reuses the solver's own basis function
rather than reimplementing it; if the two ever disagreed, every
recovered-versus-true comparison would be silently rotated within each joint's
plane with nothing to flag it. `test/test_perturbation.py` proves the round trip.

The hand-eye, mount-offset and TCP ground truths need no perturbation machinery:
they are rigid transforms in the nominal URDF, published as `_gt` TF frames and
read straight out of TF by the `ground_truth` node.

```bash
# reproducible perturbation, archived with the run
ros2 run annin_ar4_gazebo generate_perturbation.py --seed 7 \
    --t-sigma 0.001 --r-sigma-deg 0.1 -o my_experiment.yaml

ros2 launch annin_ar4_gazebo calibration_sim.launch.py \
    perturbation_file:=/abs/path/my_experiment.yaml image_noise_stddev:=0.005
```

Keep translations under ~2 mm and rotations under ~0.15°: the solver's default
bounds are 10 mm / 3°, so anything larger is clipped and the comparison stops
meaning anything, while below ~0.2 mm you are under the simulated measurement
floor.

Each run writes, into `$ROS_HOME/annin_ar4_calibration/`:

- `perturbed_robot.urdf` — the exact model that was spawned
- `ground_truth.yaml` — the truth, plus a `comparison` block after `~/compare`

Archive those two with `kinematic_result.yaml` and the run is fully reproducible.

---

## Layout

```
config/perturbations/     zero.yaml (unperturbed baseline) + example_1mm.yaml
models/charuco_5x7_35mm/  generated board texture, mesh and geometry manifest
scripts/                  generate_charuco_asset.py, generate_perturbation.py
worlds/calibration.world  empty.world + the Sensors system, lighting, ground,
                          and the TCP reference point
launch/calibration_sim.launch.py
launch/sim_launch_common.py   shared substitutions (also used by gazebo.launch.py)
```

### Regenerating the ChArUco board

```bash
ros2 run annin_ar4_gazebo generate_charuco_asset.py --square-length 0.020
```

The script refuses to write anything unless OpenCV can detect all the markers in
what it just rendered, so a bad texture cannot reach the simulation silently. It
also burns an `AR4 >>` sentinel into the quiet-zone margin: a mirrored ArUco
board is *undetectable with no error anywhere*, whereas mirrored text is obvious
at a glance in `rqt_image_view`.

**If you change the board geometry, change `annin_ar4_calibration`'s
`config/sim_overrides.yaml` to match.** The generator records what it used in
`models/<board>/board.yaml`. A mismatch scales every measured pose and the
calibration still converges — to a confidently wrong answer.

---

## Things that will bite you

**Everything must run on sim time.** `gz_ros2_control` publishes `/joint_states`
at sim time (starting near 0) while a default-configured node stamps at wall
clock (~1.7e9). Every TF lookup then fails or extrapolates wildly, and every
downstream symptom is a red herring. Both launch files here force
`use_sim_time`; the calibration launch files need `use_sim_time:=True` passed
explicitly.

**`GZ_SIM_RESOURCE_PATH` is not set by ROS or colcon.** sdformat rewrites
`package://X/...` mesh URIs to `model://X/...`, which gz resolves only through
that variable. Both launch files build it from `AMENT_PREFIX_PATH`. Without it
the arm loads with no geometry at all. (Pre-existing and unrelated: the repo's
mk5 mesh set has no `Link_5_Col.STL` / `Link_6_Col.STL`, so those two collision
meshes log errors and are skipped.)

**Do not attach the workspace-fixed half of the rig to the URDF's `world`
link.** It is tempting — `world` is special-cased by sdformat into a rigid
attachment, and `base_link` sits at the same place. But it gives the model a
second independent world-attached root, and gz then fails to put *any* of the
model's visuals into the sensor render scene: the camera sees the ground plane
and world models while the robot and the board are simply absent, with nothing
logged. `calibration_rig_macro.xacro` parents to `base_link` instead (identical
pose, since the base joint origin is identity).

**The camera's URDF frames are `_gt`-suffixed on purpose.** `hand_eye_calibration`
broadcasts a static transform whose child is `camera_color_optical_frame`. If
`robot_state_publisher` also published a link with that name, two publishers
would be writing one TF edge and tf2 would return whichever arrived last — a
wrong answer that looks entirely plausible. The image's *header* still carries
`camera_color_optical_frame`, via the gz sensor's `<gz_frame_id>`, which is why
none of the calibration code needed changing. In RViz the gap between
`sim_camera_color_optical_frame_gt` and `camera_color_optical_frame` is the
hand-eye error, live.

**`gz_frame_id` logs a warning.** `XML Element[gz_frame_id] ... not defined in
SDF. Copying[gz_frame_id] as children of [sensor]` is expected and benign —
sdformat preserves the element and gz-sensors honours it.

**Controller spawners need `--switch-timeout`.** The default is 5 s, and the
first activation lands while gz is still starting its render thread. The symptom
is `joint_state_broadcaster` sitting `inactive` forever, hence no
`/joint_states`, no TF, and every calibration node waiting on nothing.
`--controller-manager-timeout` does *not* cover this.

**Gravity is off** (`<gravity>0 0 0</gravity>`, inherited from `empty.world`).
The AR4 sim has no gravity compensation, so under load every pose would droop by
a systematic amount — which is exactly the kind of pose-dependent error WP4 is
trying to identify. Zero gravity keeps the injected perturbation the only error
source.

**Shadows are off and there are three lights.** The arm's own shadow sweeping
across the board causes intermittent detection dropouts that look exactly like
random measurement noise.

**`settle_time_sec` is wall clock.** Both collectors settle with `time.sleep()`,
which does not scale with the real-time factor. `sim_overrides.yaml` raises it to
2.0 s; if you run at a very low RTF, raise it further or samples get taken while
the arm is still moving.
