"""Synthetic round-trip tests for the WP4 kinematic-calibration core modules"""
import math

import numpy as np
import pytest

from annin_ar4_calibration.core import geometry, joint_sampling, kinematic_model, kinematic_solver


def _ar4_joint_frames():
    """Nominal joint origins/axes matching annin_ar4_description"""
    pi = math.pi
    specs = [
        ([0, 0, 0.092], [pi, 0, 0], [0, 0, 1]),
        ([0, 0.06415, -0.07778], [1.5708, 0, -1.5708], [0, 0, -1]),
        ([0, -0.305, 0], [0, 0, pi], [0, 0, -1]),
        ([0, 0, 0], [1.5708, 0, -1.5708], [0, 0, -1]),
        ([0, 0, -0.22294], [pi, 0, -1.5708], [1, 0, 0]),
        ([0, 0, 0.041], [0, 0, 0], [0, 0, 1]),
    ]
    return [
        kinematic_model.JointFrame(
            xyz0=np.array(xyz, dtype=float), rpy0=np.array(rpy, dtype=float),
            axis=np.array(axis, dtype=float), lower=-pi, upper=pi, name=f'joint_{i + 1}')
        for i, (xyz, rpy, axis) in enumerate(specs)
    ]


def _ground_truth_corrections(fix_joint1: bool) -> np.ndarray:
    rng = np.random.default_rng(42)
    corrections = rng.uniform(-1, 1,
                              size=(6, kinematic_model.PARAMS_PER_JOINT))
    corrections[:, 0:2] *= 0.003    # translation coefficients ~ +/- 3 mm
    # rotation coefficients ~ +/- 1.5 deg
    corrections[:, 2:4] *= math.radians(1.5)
    if fix_joint1:
        corrections[0, :] = 0.0
    return corrections


def _ground_truth_mount():
    mount_R = geometry.axis_angle_to_rotation_matrix(
        np.array([0.3, -0.7, 0.2]), math.radians(20))
    mount_t = np.array([0.04, -0.015, 0.025])
    return mount_R, mount_t


def _known_extrinsic():
    known_R = geometry.axis_angle_to_rotation_matrix(
        np.array([0.1, 0.2, -0.9]), math.radians(-95))
    known_t = np.array([0.5, 0.1, 0.4])
    return known_R, known_t


def _synthesize_samples(
        calibration_type, joint_frames, gt_corrections, mount_R, mount_t, known_R, known_t,
        n=150, noise_t_std=0.0, noise_r_std=0.0, rng=None):
    rng = rng or np.random.default_rng(7)
    thetas = joint_sampling.sample_joint_configs(
        joint_frames, n, margin_deg=5.0, rng=rng)

    theta_arr = np.array(thetas)
    cam_target_R = np.empty((n, 3, 3))
    cam_target_t = np.empty((n, 3))
    known_R_inv, known_t_inv = geometry.batch_invert_rt(
        known_R[None], known_t[None])
    known_R_inv, known_t_inv = known_R_inv[0], known_t_inv[0]

    for i, theta in enumerate(theta_arr):
        pred_R, pred_t = kinematic_model.forward_kinematics(
            theta, joint_frames, gt_corrections)
        if calibration_type == 'eye_to_hand':
            # cam_T_marker = known^-1 @ (pred @ mount)
            lhs_R, lhs_t = geometry.compose_rt(
                pred_R, pred_t, mount_R, mount_t)
            R, t = geometry.compose_rt(known_R_inv, known_t_inv, lhs_R, lhs_t)
        else:
            # cam_T_marker = (pred @ known)^-1 @ mount
            chain_R, chain_t = geometry.compose_rt(
                pred_R, pred_t, known_R, known_t)
            chain_R_inv = chain_R.T
            chain_t_inv = -chain_R_inv @ chain_t
            R, t = geometry.compose_rt(
                chain_R_inv, chain_t_inv, mount_R, mount_t)

        if noise_r_std > 0:
            noise_axis = rng.normal(size=3)
            noise_angle = rng.normal(scale=noise_r_std)
            R = R @ geometry.axis_angle_to_rotation_matrix(
                noise_axis, noise_angle)
        if noise_t_std > 0:
            t = t + rng.normal(scale=noise_t_std, size=3)

        cam_target_R[i], cam_target_t[i] = R, t

    return theta_arr, cam_target_R, cam_target_t


@pytest.mark.parametrize('solver_name', ['sequential', 'joint'])
def test_solver_recovers_ground_truth_noiseless_eye_to_hand(solver_name):
    calibration_type = 'eye_to_hand'
    joint_frames = _ar4_joint_frames()
    fix_joint1 = False
    gt_corrections = _ground_truth_corrections(fix_joint1)
    mount_R, mount_t = _ground_truth_mount()
    known_R, known_t = _known_extrinsic()

    theta, cam_target_R, cam_target_t = _synthesize_samples(
        calibration_type, joint_frames, gt_corrections, mount_R, mount_t, known_R, known_t)

    solve_fn = (kinematic_solver.solve_sequential if solver_name == 'sequential'
                else kinematic_solver.solve_joint)
    result = solve_fn(
        calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
        joint_frames, fix_joint1, ridge_lambda=1e-9)

    assert result.rms_after_m < 1e-6
    assert result.condition_number < 100, (
        'eye_to_hand should be well-conditioned - a large condition number here would mean '
        'something regressed in the parameterization/near-singularity assumptions')
    np.testing.assert_allclose(result.corrections, gt_corrections, atol=1e-4)
    np.testing.assert_allclose(result.mount_t, mount_t, atol=1e-4)

    mount_rot_err = geometry.rotation_matrix_to_rotvec(
        result.mount_R.T @ mount_R)
    assert np.linalg.norm(mount_rot_err) < 1e-3


@pytest.mark.parametrize('solver_name', ['sequential', 'joint'])
def test_solver_fits_data_eye_in_hand(solver_name):
    calibration_type = 'eye_in_hand'
    joint_frames = _ar4_joint_frames()
    fix_joint1 = True
    gt_corrections = _ground_truth_corrections(fix_joint1)
    mount_R, mount_t = _ground_truth_mount()
    known_R, known_t = _known_extrinsic()

    theta, cam_target_R, cam_target_t = _synthesize_samples(
        calibration_type, joint_frames, gt_corrections, mount_R, mount_t, known_R, known_t)

    solve_fn = (kinematic_solver.solve_sequential if solver_name == 'sequential'
                else kinematic_solver.solve_joint)
    result = solve_fn(
        calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
        joint_frames, fix_joint1, ridge_lambda=1e-9)

    assert result.rms_after_m < 1e-6
    np.testing.assert_array_equal(result.corrections[0, :], np.zeros(4))
    assert result.condition_number > 1000


def test_noisy_data_still_improves_rms():
    joint_frames = _ar4_joint_frames()
    calibration_type = 'eye_to_hand'
    fix_joint1 = False
    gt_corrections = _ground_truth_corrections(fix_joint1)
    mount_R, mount_t = _ground_truth_mount()
    known_R, known_t = _known_extrinsic()

    theta, cam_target_R, cam_target_t = _synthesize_samples(
        calibration_type, joint_frames, gt_corrections, mount_R, mount_t, known_R, known_t,
        n=150, noise_t_std=0.0005, noise_r_std=math.radians(0.2))

    result = kinematic_solver.solve_joint(
        calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
        joint_frames, fix_joint1)

    assert result.rms_after_m < result.rms_before_m
    assert result.rms_after_m < 0.001


def test_axis_alignment_flags_joint2_joint3_as_near_parallel():
    joint_frames = _ar4_joint_frames()
    dots = kinematic_model.axis_alignment_diagnostic(joint_frames)
    assert len(dots) == 5
    # joint_2/joint_3 (index 1) are exactly parallel in world frame
    assert dots[1] > 0.99
    others = [d for i, d in enumerate(dots) if i != 1]
    assert all(d < 0.99 for d in others)


def test_forward_kinematics_none_matches_zero_corrections():
    joint_frames = _ar4_joint_frames()
    theta = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    R1, t1 = kinematic_model.forward_kinematics(
        theta, joint_frames, corrections=None)
    zero_corr = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    R2, t2 = kinematic_model.forward_kinematics(
        theta, joint_frames, corrections=zero_corr)
    np.testing.assert_allclose(R1, R2, atol=1e-12)
    np.testing.assert_allclose(t1, t2, atol=1e-12)


def test_orthogonal_basis_is_unit_and_perpendicular_to_axis():
    for axis in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0.5, 0.5, 0.7071]):
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        basis = kinematic_model.orthogonal_basis(axis)
        assert basis.shape == (2, 3)
        for v in basis:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-10
            assert abs(np.dot(v, axis)) < 1e-10
        assert abs(np.dot(basis[0], basis[1])) < 1e-10
