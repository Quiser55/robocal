"""Filesystem locations for TCP calibration runtime data.

Runtime data (collected samples, computed results) must never default into
the installed ``share/`` directory, since that is read-only after
``colcon build``/``install``. Instead it lives under ``$ROS_HOME`` (the same
env var ROS 2 core tooling already uses for writable runtime state),
defaulting to ``~/.ros`` when unset.
"""
import os
from pathlib import Path


def data_dir() -> Path:
    ros_home = os.environ.get('ROS_HOME', os.path.join(os.path.expanduser('~'), '.ros'))
    path = Path(ros_home) / 'annin_ar4_calibration'
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_sample_file() -> str:
    return str(data_dir() / 'samples.npy')


def default_result_file() -> str:
    return str(data_dir() / 'tcp_result.yaml')


def default_hand_eye_sample_file() -> str:
    return str(data_dir() / 'hand_eye_samples.npy')


def default_hand_eye_result_file() -> str:
    return str(data_dir() / 'hand_eye_result.yaml')


def default_kinematic_sample_file() -> str:
    return str(data_dir() / 'kinematic_samples.npy')


def default_kinematic_result_file() -> str:
    return str(data_dir() / 'kinematic_result.yaml')


def default_ground_truth_file() -> str:
    return str(data_dir() / 'ground_truth.yaml')


def default_perturbed_urdf_file() -> str:
    """Where the simulation writes the URDF it actually spawned. Kept next to
    the samples/results (rather than in a temp file) so a run is reproducible
    and archivable after the fact."""
    return str(data_dir() / 'perturbed_robot.urdf')


def resolve(value: str, default_fn) -> str:
    """Empty-string param value means "use the shared default"."""
    return value if value else default_fn()
