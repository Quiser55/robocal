"""Generates candidate joint configurations spanning the workspace.

Pure-numeric, no ROS message deps (mirrors pose_sampling.py's pattern).
Kinematic parameter identification needs joint-space coverage across each
joint's full range (unlike hand-eye's local Cartesian orbit around a seed
pose), since several of the identifiability arguments in
core/kinematic_model.py rely on genuine per-joint angle diversity, not just
sample count.
"""
import numpy as np


def sample_joint_configs(
        joint_frames: list, n: int,
        margin_deg: float = 5.0,
        rng: np.random.Generator | None = None) -> list:
    """Draws n candidate joint configurations, each joint sampled
    independently and uniformly within its [lower, upper] limit (shrunk by
    margin_deg on each side to avoid commanding exact limit stops)."""
    rng = rng or np.random.default_rng()
    margin_rad = np.radians(margin_deg)

    configs = []
    for _ in range(n):
        theta = np.array([
            rng.uniform(jf.lower + margin_rad, jf.upper - margin_rad)
            if (jf.upper - jf.lower) > 2 * margin_rad
            else 0.5 * (jf.lower + jf.upper)
            for jf in joint_frames
        ])
        configs.append(theta)
    return configs


def observed_ranges_deg(thetas: np.ndarray) -> np.ndarray:
    """thetas: (N, 6) collected joint angles -> (6,) per-joint observed range
    (max - min) in degrees, used to gate dataset acceptance."""
    return np.degrees(thetas.max(axis=0) - thetas.min(axis=0))
