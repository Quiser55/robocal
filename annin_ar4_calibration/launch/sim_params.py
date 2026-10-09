"""Shared helper for layering simulation parameter overrides"""
import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def sim_parameter_files(context) -> list:
    """Extra parameter files to append after a work package's own params"""
    files = []
    if LaunchConfiguration('use_sim_time').perform(context).lower() == 'true':
        files.append(os.path.join(
            get_package_share_directory('annin_ar4_calibration'),
            'config', 'sim_overrides.yaml'))

    extra = LaunchConfiguration('extra_params_file').perform(context)
    if extra:
        files.append(extra)
    return files
