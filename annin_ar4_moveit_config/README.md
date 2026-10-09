# annin_ar4_moveit_config

MoveIt 2 configuration for the Annin AR4.

- `moveit.launch.py` - real `move_group` + RViz (+ optional `moveit_servo`).
- `demo.launch.py` - fake-hardware demo (`ros2_control` FakeSystem).
- `collision_avoidance.launch.py` - camera-based collision avoidance bringup
  (see below).

## Camera-based collision avoidance (octomap)

Feeds an Intel RealSense depth stream into MoveIt's occupancy-map monitor so
`move_group` (OMPL) **and** `moveit_servo` automatically avoid obstacles the
camera sees. The camera is placed in the robot frame using the hand-eye
extrinsic produced by `annin_ar4_calibration` (eye-to-hand /
`base_link -> camera_color_optical_frame`).

Pipeline: `aligned_depth_to_color/image_raw` -> `DepthImageOctomapUpdater`
(renders the robot model from the camera view and masks it, so the arm itself is
not voxelized) -> octomap in the planning scene (`octomap_frame: base_link`).

### Prerequisites

- `sudo apt install ros-jazzy-moveit-ros-perception` (provides the
  `DepthImageOctomapUpdater` plugin - not a default MoveIt dependency).
- A completed hand-eye calibration: run
  `ros2 launch annin_ar4_calibration hand_eye.launch.py calibration_type:=eye_to_hand`
  once so `hand_eye_result.yaml` exists.
- The arm brought up separately: `annin_ar4_driver` (real HW) or
  `annin_ar4_gazebo` (sim).
- A RealSense **D405** mounted ~0.3-0.5 m from the workspace (short-range
  module; tune `config/sensors_3d.yaml` clipping/range for other models).

### Launching

```bash
# All-in-one: RealSense + calibration TF re-broadcast + move_group with octomap
ros2 launch annin_ar4_moveit_config collision_avoidance.launch.py \
    calibration_type:=eye_to_hand moveit_servo:=True
```

Or wire only the octomap into an otherwise-normal MoveIt bringup (you start the
camera and calibration node yourself):

```bash
ros2 launch annin_ar4_moveit_config moveit.launch.py octomap:=True
```

### Verifying

1. `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame` resolves
   (calibration node running) before trusting obstacle placement.
2. `ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw` shows the
   depth stream, and `move_group` logs the `DepthImageOctomapUpdater` starting.
3. Place a box in the workspace: occupied voxels appear in RViz (MotionPlanning
   -> Scene Geometry) at the right spot and the arm stays un-voxelized. Plan to a
   goal on the far side of the box -> the path detours. Remove it -> voxels clear.
4. With `moveit_servo:=True`, jog toward the obstacle -> motion decelerates/stops
   at the proximity threshold (`config/moveit_servo.yaml`).

### Tuning

`config/sensors_3d.yaml`:
- `octomap_resolution` - voxel size (m) vs. CPU. 0.01 suits the short-range D405.
- `near/far_clipping_plane_distance`, `max_range` - match your camera's usable
  depth range and how far you want obstacles registered.
- `padding_scale`/`padding_offset` - grow the robot self-filter to avoid the arm
  bleeding into the octomap.

Accuracy depends on a good `hand_eye_result.yaml` and correct camera intrinsics
(`camera_info`).
