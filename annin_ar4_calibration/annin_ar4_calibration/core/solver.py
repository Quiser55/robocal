"""Pivot-point (translation-only) TCP calibration solver."""
from dataclasses import dataclass, field

import numpy as np

# Minimal subset size for RANSAC
DEFAULT_MIN_SUBSET_SIZE = 3
_SVD_DEGENERACY_EPS = 1e-6


@dataclass
class PivotResult:
    p_tool: np.ndarray
    p_base: np.ndarray
    rms_inliers: float
    rms_all: float
    inlier_mask: np.ndarray
    num_samples: int
    num_inliers: int
    used_ransac: bool
    degenerate_fallback: bool = False


def build_system(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """the stacked (3N, 6) least-squares system"""
    n = R.shape[0]
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i in range(n):
        A[3 * i:3 * i + 3, 0:3] = R[i]
        A[3 * i:3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i:3 * i + 3] = -t[i]
    return A, b


def residuals(R: np.ndarray, t: np.ndarray, p_tool: np.ndarray, p_base: np.ndarray) -> np.ndarray:
    """Per-sample residual norm ||R_i @ p_tool - p_base + t_i||"""
    predicted = np.einsum('nij,j->ni', R, p_tool) - p_base + t
    return np.linalg.norm(predicted, axis=1)


#: Pre-existing internal names, kept so nothing outside this module breaks.
_build_system = build_system
_residuals = residuals


def solve_pivot_linear(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R: (N,3,3), t: (N,3) -> (p_tool (3,), p_base (3,), residuals (N,))."""
    A, b = _build_system(R, t)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tool, p_base = x[0:3], x[3:6]
    residuals = _residuals(R, t, p_tool, p_base)
    return p_tool, p_base, residuals


def _is_degenerate(R_subset: np.ndarray, t_subset: np.ndarray) -> bool:
    A, _ = _build_system(R_subset, t_subset)
    s = np.linalg.svd(A, compute_uv=False)
    if s[0] <= 0:
        return True
    return (s[-1] / s[0]) < _SVD_DEGENERACY_EPS


def solve_pivot_ransac(
    R: np.ndarray,
    t: np.ndarray,
    threshold: float,
    iterations: int,
    min_subset_size: int = DEFAULT_MIN_SUBSET_SIZE,
    rng: np.random.Generator | None = None,
) -> PivotResult:
    n = R.shape[0]
    rng = rng or np.random.default_rng()

    best_mask = None
    best_count = -1
    best_rms = np.inf

    max_attempts = iterations * 10
    attempts = 0
    successful_draws = 0
    while successful_draws < iterations and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(n, size=min_subset_size, replace=False)
        R_subset, t_subset = R[idx], t[idx]
        if _is_degenerate(R_subset, t_subset):
            continue
        successful_draws += 1

        p_tool, p_base, _ = solve_pivot_linear(R_subset, t_subset)
        residuals = _residuals(R, t, p_tool, p_base)
        mask = residuals < threshold
        count = int(mask.sum())
        rms = float(
            np.sqrt(np.mean(residuals[mask] ** 2))) if count > 0 else np.inf

        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_mask = mask

    if best_mask is None or best_count < min_subset_size:
        # All draws were degenerate (or no inliers found) - fall back to a full-data least-squares fit
        p_tool, p_base, residuals = solve_pivot_linear(R, t)
        mask = np.ones(n, dtype=bool)
        rms_all = float(np.sqrt(np.mean(residuals ** 2)))
        return PivotResult(
            p_tool=p_tool, p_base=p_base, rms_inliers=rms_all, rms_all=rms_all,
            inlier_mask=mask, num_samples=n, num_inliers=n, used_ransac=True,
            degenerate_fallback=True,
        )

    # Final refit restricted to the winning inlier set.
    p_tool, p_base, _ = solve_pivot_linear(R[best_mask], t[best_mask])
    residuals_all = _residuals(R, t, p_tool, p_base)
    rms_inliers = float(np.sqrt(np.mean(residuals_all[best_mask] ** 2)))
    rms_all = float(np.sqrt(np.mean(residuals_all ** 2)))

    return PivotResult(
        p_tool=p_tool, p_base=p_base, rms_inliers=rms_inliers, rms_all=rms_all,
        inlier_mask=best_mask, num_samples=n, num_inliers=int(best_mask.sum()),
        used_ransac=True, degenerate_fallback=False,
    )


def solve_pivot(
    R: np.ndarray,
    t: np.ndarray,
    ransac_enabled: bool,
    threshold: float,
    iterations: int,
    min_subset_size: int = DEFAULT_MIN_SUBSET_SIZE,
    rng: np.random.Generator | None = None,
) -> PivotResult:
    """Single entry point used by calibration_node."""
    if ransac_enabled:
        return solve_pivot_ransac(R, t, threshold, iterations, min_subset_size, rng)

    p_tool, p_base, residuals = solve_pivot_linear(R, t)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    mask = np.ones(R.shape[0], dtype=bool)
    return PivotResult(
        p_tool=p_tool, p_base=p_base, rms_inliers=rms, rms_all=rms,
        inlier_mask=mask, num_samples=R.shape[0], num_inliers=R.shape[0],
        used_ransac=False,
    )
