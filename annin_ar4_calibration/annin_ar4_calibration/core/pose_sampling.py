"""Generates candidate calibration poses by randomly perturbing a seed pose"""
import numpy as np

from annin_ar4_calibration.core import geometry


def sample_orbit_poses(
        seed_R: np.ndarray, seed_t: np.ndarray, n: int,
        max_translation_m: float, max_orientation_deg: float,
        rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Draws n candidate poses around (seed_R, seed_t)"""
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
