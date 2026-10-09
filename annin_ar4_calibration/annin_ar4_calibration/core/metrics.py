"""Calibration quality metrics, split into two families.

**Family A - truth metrics.** Only available in simulation, where
``annin_ar4_gazebo`` spawns a perturbed robot whose true parameters are known
exactly.

**Family B - truth-free metrics.** Computable from the collected samples alone,
so they work identically in simulation and on the physical arm. These are the
only metrics available on hardware
"""
import math

import cv2
import numpy as np

from annin_ar4_calibration.core import geometry, kinematic_model, solver

SIM_DETECTION_FLOOR_MM = 0.6
SIM_DETECTION_FLOOR_DEG = 0.06


# Family A

def translation_error_mm(true_t: np.ndarray, est_t: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(est_t) - np.asarray(true_t)) * 1000.0)


def rotation_error_deg(true_R: np.ndarray, est_R: np.ndarray) -> float:
    rotvec = geometry.rotation_matrix_to_rotvec(est_R @ true_R.T)
    return float(math.degrees(np.linalg.norm(rotvec)))


def pose_error(true_R, true_t, est_R, est_t) -> dict:
    """Rigid-transform error, split into a translation and rotation"""
    return {
        'translation_error_mm': translation_error_mm(true_t, est_t),
        'rotation_error_deg': rotation_error_deg(true_R, est_R),
    }


def transform_dict(R: np.ndarray, t: np.ndarray) -> dict:
    """The `transform` block shape used by every result YAML"""
    quat = geometry.rotation_matrix_to_quat(R)
    return {
        'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
        'rotation': {'x': float(quat[0]), 'y': float(quat[1]),
                     'z': float(quat[2]), 'w': float(quat[3])},
    }


def rt_from_transform_dict(transform: dict) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `transform_dict`"""
    t = np.array([transform['translation']['x'], transform['translation']['y'],
                  transform['translation']['z']])
    q = transform['rotation']
    return geometry.quat_to_rotation_matrix(q['x'], q['y'], q['z'], q['w']), t


def per_joint_correction_error(
        true_corrections: dict, est_corrections: dict, joint_names) -> dict:
    per_joint = {}
    for name in joint_names:
        true_j = true_corrections.get(name, {}) or {}
        est_j = est_corrections.get(name, {}) or {}
        dt = (np.asarray(est_j.get('delta_t_perp', [0.0, 0.0]), dtype=float)
              - np.asarray(true_j.get('delta_t_perp', [0.0, 0.0]), dtype=float))
        dr = (np.asarray(est_j.get('delta_r_perp', [0.0, 0.0]), dtype=float)
              - np.asarray(true_j.get('delta_r_perp', [0.0, 0.0]), dtype=float))
        per_joint[name] = {
            'delta_t_perp_error_mm': [float(v * 1000.0) for v in dt],
            'delta_r_perp_error_mdeg': [float(math.degrees(v) * 1000.0) for v in dr],
            'delta_t_perp_error_norm_mm': float(np.linalg.norm(dt) * 1000.0),
            'delta_r_perp_error_norm_mdeg': float(math.degrees(np.linalg.norm(dr)) * 1000.0),
        }
    return per_joint


def fk_agreement_error(
        joint_frames: list, true_corrections: np.ndarray, est_corrections: np.ndarray,
        thetas: np.ndarray) -> dict:
    true_R, true_t = kinematic_model.forward_kinematics_batch(
        thetas, joint_frames, true_corrections)
    est_R, est_t = kinematic_model.forward_kinematics_batch(
        thetas, joint_frames, est_corrections)

    pos_err = np.linalg.norm(est_t - true_t, axis=1) * 1000.0
    rel_R = np.einsum('nij,njk->nik', true_R.transpose(0, 2, 1), est_R)
    rot_err = np.degrees(np.linalg.norm(
        geometry.batch_rotation_matrix_to_rotvec(rel_R), axis=1))

    return {
        'position_rms_mm': float(np.sqrt(np.mean(pos_err ** 2))),
        'position_max_mm': float(np.max(pos_err)),
        'position_mean_mm': float(np.mean(pos_err)),
        'rotation_rms_deg': float(np.sqrt(np.mean(rot_err ** 2))),
        'rotation_max_deg': float(np.max(rot_err)),
        'num_configs': int(thetas.shape[0]),
    }


def fk_geometry_error(
        joint_frames: list, true_corrections: np.ndarray, est_corrections: np.ndarray,
        thetas: np.ndarray) -> float:
    _, true_t = kinematic_model.forward_kinematics_batch(
        thetas, joint_frames, true_corrections)
    _, est_t = kinematic_model.forward_kinematics_batch(
        thetas, joint_frames, est_corrections)
    a = true_t - true_t.mean(axis=0)
    b = est_t - est_t.mean(axis=0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return float(np.sqrt(np.mean(np.sum((b @ R - a) ** 2, axis=1))) * 1000.0)


# Family B

def kfold_indices(n: int, k: int, rng: np.random.Generator | None = None,
                  shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Standard K-fold partition -> list of (train_idx, test_idx)"""
    if n < 2:
        raise ValueError(
            f'need at least 2 samples for cross-validation, got {n}')
    k = int(min(max(k, 2), n))
    idx = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng(0)).shuffle(idx)

    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((np.sort(train), np.sort(test)))
    return splits


def bootstrap_indices(n: int, num_resamples: int,
                      rng: np.random.Generator | None = None) -> list[np.ndarray]:
    rng = rng or np.random.default_rng(0)
    return [rng.integers(0, n, size=n) for _ in range(num_resamples)]


def summarize(values, ci: float = 95.0) -> dict:
    v = np.asarray(
        [x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {'n': 0}
    lo, hi = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    return {
        'n': int(v.size),
        'mean': float(np.mean(v)),
        'std': float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        'median': float(np.median(v)),
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'p95': float(np.percentile(v, 95.0)),
        f'ci{int(ci)}_lo': float(np.percentile(v, lo)),
        f'ci{int(ci)}_hi': float(np.percentile(v, hi)),
    }


def pivot_conditioning(R: np.ndarray, t: np.ndarray) -> dict:
    A, _ = solver.build_system(R, t)
    s = np.linalg.svd(A, compute_uv=False)
    smallest = max(float(s[-1]), 1e-12)
    return {
        'condition_number': float(s[0] / smallest),
        'singular_values': [float(v) for v in s],
        **orientation_diversity_deg(R),
    }


def orientation_diversity_deg(R: np.ndarray, max_pairs: int = 20000) -> dict:
    """Mean/min pairwise geodesic angle between sample orientations"""
    n = R.shape[0]
    quats = np.array([geometry.rotation_matrix_to_quat(Ri) for Ri in R])
    i, j = np.triu_indices(n, k=1)
    if i.size > max_pairs:
        sel = np.random.default_rng(0).choice(
            i.size, size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    dots = np.clip(np.abs(np.sum(quats[i] * quats[j], axis=1)), -1.0, 1.0)
    angles = np.degrees(2.0 * np.arccos(dots))
    return {
        'orientation_spread_mean_deg': float(np.mean(angles)) if angles.size else 0.0,
        'orientation_spread_max_deg': float(np.max(angles)) if angles.size else 0.0,
    }


def motion_axis_diversity(R: np.ndarray) -> dict:
    """Diversity of the *relative* rotation axes between consecutive robot poses"""
    n = R.shape[0]
    if n < 2:
        return {'axis_rank_ratio': 0.0, 'num_motions': 0, 'mean_motion_angle_deg': 0.0}

    axes, angles = [], []
    for i in range(n - 1):
        rel = R[i].T @ R[i + 1]
        rotvec = geometry.rotation_matrix_to_rotvec(rel)
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-6:
            continue
        axes.append(rotvec / angle)
        angles.append(math.degrees(angle))

    if len(axes) < 2:
        return {'axis_rank_ratio': 0.0, 'num_motions': len(axes),
                'mean_motion_angle_deg': float(np.mean(angles)) if angles else 0.0}

    s = np.linalg.svd(np.array(axes), compute_uv=False)
    return {
        'axis_rank_ratio': float(s[-1] / max(s[0], 1e-12)),
        'num_motions': len(axes),
        'mean_motion_angle_deg': float(np.mean(angles)),
    }


def reprojection_rms_px(
        object_points: np.ndarray, image_points: np.ndarray,
        rvec: np.ndarray, tvec: np.ndarray,
        camera_matrix: np.ndarray, dist_coeffs: np.ndarray | None) -> float:
    """RMS pixel distance between the detected target corners and where the
    solved pose says they should appear"""
    object_points = np.asarray(
        object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if object_points.shape[0] == 0:
        return float('nan')

    projected, _ = cv2.projectPoints(
        object_points, np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64), camera_matrix,
        dist_coeffs if dist_coeffs is not None else np.zeros(5))
    err = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
