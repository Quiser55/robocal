"""All-in-one bringup for camera-based collision avoidance.

Starts, in one command (after the arm bringup is already running):

  1. realsense2_camera - the Intel RealSense depth camera, with aligned depth
     enabled and its own TF publishing DISABLED. Aligned depth is registered to
     camera_color_optical_frame, which the hand-eye calibration owns; letting the
     driver also publish that frame would create a duplicate-parent TF conflict.
  2. hand_eye_calibration (from annin_ar4_calibration) - reloads the saved
     hand_eye_result.yaml and re-broadcasts the static
     base_link -> camera_color_optical_frame transform, placing the camera (and
     therefore its octomap) correctly in the robot's frame.
  3. moveit.launch.py with octomap:=True - move_group with the
     DepthImageOctomapUpdater wired in (see config/sensors_3d.yaml), so planning
     avoids obstacles the camera sees.

Prerequisites:
  - `sudo apt install ros-jazzy-moveit-ros-perception` (octomap updater plugin).
  - A completed hand-eye calibration (hand_eye_result.yaml on disk); run
    `ros2 launch annin_ar4_calibration hand_eye.launch.py` once beforehand.
  - The arm itself brought up separately: `annin_ar4_driver` (real hardware) or
    `annin_ar4_gazebo` (simulation).

Example:
  ros2 launch annin_ar4_moveit_config collision_avoidance.launch.py \\
      calibration_type:=eye_to_hand moveit_servo:=True
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    calibration_type = LaunchConfiguration("calibration_type")
    ar_model = LaunchConfiguration("ar_model")
    moveit_servo = LaunchConfiguration("moveit_servo")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_camera = LaunchConfiguration("launch_camera")

    declared_arguments = [
        DeclareLaunchArgument(
            "calibration_type",
            default_value="eye_to_hand",
            choices=["eye_in_hand", "eye_to_hand"],
            description="Hand-eye configuration used when the saved calibration "
            "was produced. Must match hand_eye_result.yaml.",
        ),
        DeclareLaunchArgument(
            "ar_model",
            default_value="mk5",
            choices=["mk1", "mk2", "mk3", "mk4", "mk5"],
            description="Model of AR4.",
        ),
        DeclareLaunchArgument(
            "moveit_servo",
            default_value="False",
            choices=["True", "False"],
            description="Also run moveit_servo (real-time teleop that avoids the "
            "octomap; see config/moveit_servo.yaml).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="False",
            description="Use simulation time (set True with annin_ar4_gazebo).",
        ),
        DeclareLaunchArgument(
            "launch_camera",
            default_value="True",
            choices=["True", "False"],
            description="Launch realsense2_camera here. Set False if you start "
            "the camera driver yourself.",
        ),
    ]

    moveit_config_share = get_package_share_directory("annin_ar4_moveit_config")
    calibration_share = get_package_share_directory("annin_ar4_calibration")
    hand_eye_params = os.path.join(calibration_share, "config",
                                   "hand_eye_params.yaml")
    # RViz layout that also shows the live camera image alongside the octomap
    # voxels (both rviz configs render occupied voxels already).
    camera_rviz = os.path.join(moveit_config_share, "rviz",
                               "moveit_with_camera.rviz")

    # 1. RealSense: aligned depth on, its own TF off (calibration owns the
    #    camera frame), point cloud off (the DepthImageOctomapUpdater consumes
    #    the depth image directly).
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "false",
            "publish_tf": "false",
        }.items(),
        condition=IfCondition(launch_camera),
    )

    # 2. Re-broadcast the saved hand-eye extrinsic (base_link -> camera frame).
    hand_eye_calibration_node = Node(
        package="annin_ar4_calibration",
        executable="hand_eye_calibration",
        name="hand_eye_calibration",
        parameters=[
            hand_eye_params,
            {
                "calibration_type": calibration_type,
                "use_sim_time": use_sim_time,
            },
        ],
        output="screen",
    )

    # 3. MoveIt with the octomap pipeline enabled.
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_share, "launch", "moveit.launch.py")),
        launch_arguments={
            "octomap": "True",
            "ar_model": ar_model,
            "moveit_servo": moveit_servo,
            "use_sim_time": use_sim_time,
            "rviz_config_file": camera_rviz,
        }.items(),
    )

    return LaunchDescription(declared_arguments + [
        realsense_launch,
        hand_eye_calibration_node,
        moveit_launch,
    ])
