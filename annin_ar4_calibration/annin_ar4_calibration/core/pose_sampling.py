"""Generates candidate calibration poses by randomly perturbing a seed pose.

Pure-numeric, no ROS message deps (mirrors geometry.py's pattern). Random
perturbation (rather than a hand-crafted grid) naturally yields the
non-parallel rotation axes cv2.calibrateHandEye needs for a well-conditioned
solve, without needing to hand-tune a systematic sweep.
"""
import numpy as np

from annin_ar4_calibration.core import geometry


def sample_orbit_poses(
        seed_R: np.ndarray, seed_t: np.ndarray, n: int,
        max_translation_m: float, max_orientation_deg: float,
        rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Draws n candidate poses around (seed_R, seed_t) by composing it with a
    random small rigid perturbation: a uniform translation offset within
    +/- max_translation_m per axis, and a rotation of a uniformly random axis
    by an angle uniform within +/- max_orientation_deg."""
    rng = rng or np.random.default_rng()
    max_angle_rad = np.radians(max_orientation_deg)

    poses = []
    for _ in range(n):
        axis = rng.normal(size=3)
        angle = rng.uniform(-max_angle_rad, max_angle_rad)
        dR = geometry.axis_angle_to_rotation_matrix(axis, angle)
        dt = rng.uniform(-max_translation_m, max_translation_m, size=3)
        R, t = geometry.compose_rt(seed_R, seed_t, dR, dt)
        poses.append((R, t))
    return poses
