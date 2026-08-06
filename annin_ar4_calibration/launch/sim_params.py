"""Shared helper for layering simulation parameter overrides.

Imported by the three calibration launch files, which are installed side by
side in ``share/annin_ar4_calibration/launch/``. ROS 2 does *not* put a launch
file's own directory on ``sys.path``, so each of them inserts it explicitly
before importing this module.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def sim_parameter_files(context) -> list:
    """Extra parameter files to append after a work package's own params.

    Later files win, so this layers on top rather than replacing anything.
    ``config/sim_overrides.yaml`` is applied whenever ``use_sim_time`` is true:
    the tuning it carries (settle time, sample counts, board geometry) is a
    consequence of running against Gazebo, so tying it to that one flag keeps
    a simulated run from silently using hardware collection settings.
    """
    files = []
    if LaunchConfiguration('use_sim_time').perform(context).lower() == 'true':
        files.append(os.path.join(
            get_package_share_directory('annin_ar4_calibration'),
            'config', 'sim_overrides.yaml'))

    extra = LaunchConfiguration('extra_params_file').perform(context)
    if extra:
        files.append(extra)
    return files
