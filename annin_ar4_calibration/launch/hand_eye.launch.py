import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

# ROS 2 does not put a launch file's own directory on sys.path.
sys.path.insert(0, os.path.dirname(__file__))
from sim_params import sim_parameter_files  # noqa: E402


def launch_setup(context, *args, **kwargs):
    params_file = os.path.join(
        get_package_share_directory('annin_ar4_calibration'), 'config', 'hand_eye_params.yaml')
    params = [params_file] + sim_parameter_files(context)

    calibration_type = LaunchConfiguration('calibration_type')
    target_type = LaunchConfiguration('target_type')

    return [
        Node(
            package='annin_ar4_calibration',
            executable='target_detector',
            name='target_detector',
            parameters=params + [{'target_type': target_type}],
        ),
        Node(
            package='annin_ar4_calibration',
            executable='hand_eye_collector',
            name='hand_eye_collector',
            parameters=params + [{'calibration_type': calibration_type}],
        ),
        Node(
            package='annin_ar4_calibration',
            executable='hand_eye_calibration',
            name='hand_eye_calibration',
            parameters=params + [{'calibration_type': calibration_type}],
        ),
        Node(
            package='annin_ar4_calibration',
            executable='hand_eye_visualization',
            name='hand_eye_visualization',
            parameters=params,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'calibration_type', default_value='eye_in_hand',
            choices=['eye_in_hand', 'eye_to_hand'],
            description='eye_in_hand: camera on the wrist, target fixed in the workspace. '
                         'eye_to_hand: camera fixed in the workspace, target on ee_link.'),
        DeclareLaunchArgument(
            'target_type', default_value='charuco', choices=['charuco', 'aruco'],
            description='Calibration target type.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='False', choices=['True', 'False'],
            description='Set True when running against annin_ar4_gazebo. Also layers '
                        'config/sim_overrides.yaml on top of the hardware defaults.'),
        DeclareLaunchArgument(
            'extra_params_file', default_value='',
            description='Optional parameter file applied last, overriding everything '
                        'above it.'),

        SetParameter(name='use_sim_time', value=LaunchConfiguration('use_sim_time')),
        OpaqueFunction(function=launch_setup),
    ])
