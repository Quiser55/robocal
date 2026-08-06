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
        get_package_share_directory('annin_ar4_calibration'), 'config', 'tcp_params.yaml')
    params = [params_file] + sim_parameter_files(context)

    return [
        Node(
            package='annin_ar4_calibration',
            executable='collector',
            name='collector',
            parameters=params,
        ),
        Node(
            package='annin_ar4_calibration',
            executable='calibration',
            name='calibration',
            parameters=params,
        ),
        Node(
            package='annin_ar4_calibration',
            executable='visualization',
            name='visualization',
            parameters=params,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
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
