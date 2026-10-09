"""Synthetic round-trip tests for the WP2 (pivot) and WP3 (hand-eye) solvers"""
import math

import numpy as np
import pytest

from annin_ar4_calibration.core import geometry, handeye_solver, solver


def _random_rotation(rng, max_angle_deg=180.0) -> np.ndarray:
    return geometry.axis_angle_to_rotation_matrix(
        rng.normal(size=3), math.radians(rng.uniform(-max_angle_deg, max_angle_deg)))


# WP2: pivot

def _synthesize_pivot(n=30, p_tool=(0.0, 0.01, 0.15), p_base=(0.30, 0.10, 0.20),
                      noise_std=0.0, rng=None):
    """Touches of one fixed reference point from n different orientations."""
    rng = rng or np.random.default_rng(0)
    p_tool = np.asarray(p_tool, float)
    p_base = np.asarray(p_base, float)
    R = np.array([_random_rotation(rng) for _ in range(n)])
    t = np.array([p_base - Ri @ p_tool for Ri in R])
    if noise_std > 0:
        t = t + rng.normal(scale=noise_std, size=t.shape)
    return R, t, p_tool, p_base


def test_pivot_recovers_offset_exactly_without_noise():
    R, t, p_tool, p_base = _synthesize_pivot()
    est_tool, est_base, residuals = solver.solve_pivot_linear(R, t)
    np.testing.assert_allclose(est_tool, p_tool, atol=1e-9)
    np.testing.assert_allclose(est_base, p_base, atol=1e-9)
    assert np.max(residuals) < 1e-9


def test_pivot_error_scales_with_touch_noise():
    errors = []
    for noise in (0.0002, 0.001, 0.003):
        R, t, p_tool, _ = _synthesize_pivot(
            n=30, noise_std=noise, rng=np.random.default_rng(1))
        est_tool, _, _ = solver.solve_pivot_linear(R, t)
        errors.append(np.linalg.norm(est_tool - p_tool))
    assert errors[0] < errors[1] < errors[2]


def test_pivot_residuals_are_public_and_evaluate_a_foreign_fit():
    R, t, p_tool, p_base = _synthesize_pivot(n=20)
    good = solver.residuals(R, t, p_tool, p_base)
    assert np.max(good) < 1e-9

    wrong = solver.residuals(
        R, t, p_tool + np.array([0.005, 0.0, 0.0]), p_base)
    assert np.all(wrong > 1e-4)


def test_pivot_held_out_residual_exceeds_in_sample_under_noise():
    from annin_ar4_calibration.core import metrics

    R, t, _, _ = _synthesize_pivot(
        n=40, noise_std=0.001, rng=np.random.default_rng(2))
    _, _, in_sample = solver.solve_pivot_linear(R, t)
    in_sample_rms = float(np.sqrt(np.mean(in_sample ** 2)))

    held_out = []
    for train, test in metrics.kfold_indices(40, 5, np.random.default_rng(0)):
        p_tool, p_base, _ = solver.solve_pivot_linear(R[train], t[train])
        res = solver.residuals(R[test], t[test], p_tool, p_base)
        held_out.append(float(np.sqrt(np.mean(res ** 2))))

    assert np.mean(held_out) > in_sample_rms


def test_pivot_ransac_rejects_a_gross_outlier():
    R, t, p_tool, _ = _synthesize_pivot(n=30, rng=np.random.default_rng(3))
    t[7] += np.array([0.05, -0.04, 0.03])       # one badly missed touch

    result = solver.solve_pivot_ransac(
        R, t, threshold=0.002, iterations=300, rng=np.random.default_rng(0))
    assert not result.inlier_mask[7], 'the gross outlier should not be an inlier'
    assert np.linalg.norm(result.p_tool - p_tool) < 1e-3

    # Plain least squares has no defence against it, which is why RANSAC exists.
    naive, _, _ = solver.solve_pivot_linear(R, t)
    assert np.linalg.norm(
        naive - p_tool) > np.linalg.norm(result.p_tool - p_tool)


def test_pivot_build_system_shape_matches_the_stacked_equations():
    R, t, _, _ = _synthesize_pivot(n=5)
    A, b = solver.build_system(R, t)
    assert A.shape == (15, 6)
    assert b.shape == (15,)


# WP3: hand-eye

def _synthesize_handeye(calibration_type, n=15, noise_t_std=0.0, noise_r_std=0.0,
                        single_axis=False, rng=None):
    """Paired (base->tool, camera->target) samples consistent with a known X"""
    rng = rng or np.random.default_rng(0)
    X_R = geometry.axis_angle_to_rotation_matrix(
        np.array([0.2, -0.8, 0.3]), math.radians(70))
    X_t = np.array([0.05, -0.03, 0.12])
    target_R = geometry.axis_angle_to_rotation_matrix(np.array([0.4, 0.1, 0.9]),
                                                      math.radians(-25))
    target_t = np.array([0.35, 0.12, 0.22])

    robot_R = np.empty((n, 3, 3))
    robot_t = np.empty((n, 3))
    for i in range(n):
        if single_axis:
            robot_R[i] = geometry.axis_angle_to_rotation_matrix(
                np.array([0.0, 0.0, 1.0]), rng.uniform(-1.0, 1.0))
        else:
            robot_R[i] = _random_rotation(rng, max_angle_deg=60.0)
        robot_t[i] = np.array([0.3, 0.0, 0.3]) + \
            rng.uniform(-0.08, 0.08, size=3)

    hand_R, hand_t = handeye_solver.hand_poses(
        calibration_type, robot_R, robot_t)
    # cam_T_target = X^-1 @ hand^-1 @ (fixed target pose)
    X_R_inv = X_R.T
    X_t_inv = -X_R_inv @ X_t
    hand_R_inv, hand_t_inv = geometry.batch_invert_rt(hand_R, hand_t)
    mid_R, mid_t = geometry.batch_compose_rt(
        hand_R_inv, hand_t_inv, target_R, target_t)
    cam_R, cam_t = geometry.batch_compose_rt(X_R_inv, X_t_inv, mid_R, mid_t)

    if noise_r_std > 0:
        cam_R = np.array([
            Ri @ geometry.axis_angle_to_rotation_matrix(
                rng.normal(size=3), rng.normal(scale=noise_r_std))
            for Ri in cam_R])
    if noise_t_std > 0:
        cam_t = cam_t + rng.normal(scale=noise_t_std, size=cam_t.shape)

    return robot_R, robot_t, cam_R, cam_t, X_R, X_t


@pytest.mark.parametrize('calibration_type', ['eye_in_hand', 'eye_to_hand'])
def test_handeye_recovers_known_extrinsic(calibration_type):
    robot_R, robot_t, cam_R, cam_t, X_R, X_t = _synthesize_handeye(
        calibration_type, n=20, rng=np.random.default_rng(4))

    fit = handeye_solver.solve_hand_eye(
        calibration_type, robot_R, robot_t, cam_R, cam_t, method='PARK')

    np.testing.assert_allclose(fit.t, X_t, atol=1e-6)
    rot_error = geometry.rotation_matrix_to_rotvec(fit.R.T @ X_R)
    assert np.linalg.norm(rot_error) < 1e-6
    assert fit.consistency_rms_m < 1e-9


@pytest.mark.parametrize('method', ['TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'])
def test_all_five_handeye_methods_agree_on_noiseless_data(method):
    robot_R, robot_t, cam_R, cam_t, X_R, X_t = _synthesize_handeye(
        'eye_to_hand', n=20, rng=np.random.default_rng(5))
    fit = handeye_solver.solve_hand_eye(
        'eye_to_hand', robot_R, robot_t, cam_R, cam_t, method=method)
    np.testing.assert_allclose(fit.t, X_t, atol=1e-4)
    assert np.linalg.norm(
        geometry.rotation_matrix_to_rotvec(fit.R.T @ X_R)) < 1e-4


def test_chain_poses_are_constant_for_the_true_extrinsic():
    robot_R, robot_t, cam_R, cam_t, X_R, X_t = _synthesize_handeye(
        'eye_to_hand', n=12, rng=np.random.default_rng(6))
    chain_R, chain_t = handeye_solver.chain_poses(
        'eye_to_hand', robot_R, robot_t, cam_R, cam_t, X_R, X_t)

    ref_q, ref_t = handeye_solver.chain_reference(chain_R, chain_t)
    rms_m, rms_deg = handeye_solver.chain_deviation(
        chain_R, chain_t, ref_q, ref_t)
    assert rms_m < 1e-9
    assert rms_deg < 1e-6


def test_chain_deviation_detects_a_wrong_extrinsic():
    robot_R, robot_t, cam_R, cam_t, X_R, X_t = _synthesize_handeye(
        'eye_to_hand', n=12, rng=np.random.default_rng(7))
    wrong_t = X_t + np.array([0.01, 0.0, 0.0])
    chain_R, chain_t = handeye_solver.chain_poses(
        'eye_to_hand', robot_R, robot_t, cam_R, cam_t, X_R, wrong_t)
    rms_m, _ = handeye_solver.chain_deviation(
        chain_R, chain_t, *handeye_solver.chain_reference(chain_R, chain_t))
    assert rms_m > 1e-4


def test_single_axis_motions_are_flagged_as_undiversified():
    from annin_ar4_calibration.core import metrics

    robot_R, robot_t, _cam_R, _cam_t, _X_R, _X_t = _synthesize_handeye(
        'eye_to_hand', n=15, single_axis=True, rng=np.random.default_rng(8))
    hand_R, _ = handeye_solver.hand_poses('eye_to_hand', robot_R, robot_t)
    assert metrics.motion_axis_diversity(hand_R)['axis_rank_ratio'] < 1e-6


def test_handeye_error_grows_with_measurement_noise():
    errors = []
    for noise in (0.0002, 0.001, 0.004):
        robot_R, robot_t, cam_R, cam_t, X_R, X_t = _synthesize_handeye(
            'eye_to_hand', n=25, noise_t_std=noise,
            noise_r_std=math.radians(noise * 100), rng=np.random.default_rng(10))
        fit = handeye_solver.solve_hand_eye(
            'eye_to_hand', robot_R, robot_t, cam_R, cam_t, method='PARK')
        errors.append(np.linalg.norm(fit.t - X_t))
    assert errors[0] < errors[-1]
