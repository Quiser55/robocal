"""WP4: kinematic parameter identification.

Prerequisite: hand_eye.launch.py must already have been run and
compute_calibration already called successfully - this depends on
hand_eye_result.yaml existing (see core/paths.default_hand_eye_result_file
and the package README's kinematic-calibration section). It does not launch
hand-eye's own nodes.
"""
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
        get_package_share_directory('annin_ar4_calibration'), 'config', 'kinematic_params.yaml')
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
            executable='kinematic_collector',
            name='kinematic_collector',
            parameters=params,
        ),
        Node(
            package='annin_ar4_calibration',
            executable='kinematic_calibration',
            name='kinematic_calibration',
            parameters=params + [{'calibration_type': calibration_type}],
        ),
        Node(
            package='annin_ar4_calibration',
            executable='kinematic_visualization',
            name='kinematic_visualization',
            parameters=params,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'calibration_type', default_value='eye_to_hand',
            choices=['eye_in_hand', 'eye_to_hand'],
            description='Must match whichever hand-eye calibration was actually computed '
                         '(hand_eye_result.yaml). eye_in_hand fixes joint_1\'s correction to '
                         'zero - see README notes.'),
        DeclareLaunchArgument(
            'target_type', default_value='charuco', choices=['charuco', 'aruco'],
            description='Calibration target type (must match the physical tag in use).'),
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
