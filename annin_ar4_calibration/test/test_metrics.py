"""Unit tests for core/metrics.py"""
import math

import numpy as np
import pytest

from annin_ar4_calibration.core import geometry, kinematic_model, metrics


def _random_rotation(rng) -> np.ndarray:
    return geometry.axis_angle_to_rotation_matrix(
        rng.normal(size=3), rng.uniform(-math.pi, math.pi))


# Family A

def test_pose_error_is_zero_for_identical_poses():
    rng = np.random.default_rng(0)
    R, t = _random_rotation(rng), rng.normal(size=3)
    error = metrics.pose_error(R, t, R, t)
    assert error['translation_error_mm'] == pytest.approx(0.0, abs=1e-9)
    assert error['rotation_error_deg'] == pytest.approx(0.0, abs=1e-9)


def test_pose_error_reports_known_offsets():
    R = np.eye(3)
    est_R = geometry.axis_angle_to_rotation_matrix(
        np.array([0, 0, 1.0]), math.radians(2.5))
    error = metrics.pose_error(R, np.zeros(
        3), est_R, np.array([0.003, 0.004, 0.0]))
    assert error['translation_error_mm'] == pytest.approx(5.0)      # 3-4-5
    assert error['rotation_error_deg'] == pytest.approx(2.5, abs=1e-6)


def test_transform_dict_round_trips():
    rng = np.random.default_rng(3)
    R, t = _random_rotation(rng), rng.normal(size=3)
    back_R, back_t = metrics.rt_from_transform_dict(
        metrics.transform_dict(R, t))
    np.testing.assert_allclose(back_R, R, atol=1e-12)
    np.testing.assert_allclose(back_t, t, atol=1e-12)


def test_per_joint_correction_error_handles_missing_joints():
    truth = {'joint_2': {'delta_t_perp': [
        0.001, 0.0], 'delta_r_perp': [0.0, 0.0]}}
    est = {'joint_2': {'delta_t_perp': [0.0, 0.0], 'delta_r_perp': [0.0, 0.0]}}
    out = metrics.per_joint_correction_error(
        truth, est, ['joint_1', 'joint_2'])
    # An absent joint means "no correction", not "no data".
    assert out['joint_1']['delta_t_perp_error_norm_mm'] == pytest.approx(0.0)
    assert out['joint_2']['delta_t_perp_error_norm_mm'] == pytest.approx(1.0)


def _ar4_like_frames():
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
            xyz0=np.array(xyz, float), rpy0=np.array(rpy, float),
            axis=np.array(axis, float), lower=-pi, upper=pi, name=f'joint_{i + 1}')
        for i, (xyz, rpy, axis) in enumerate(specs)
    ]


def test_fk_agreement_is_zero_for_identical_models():
    frames = _ar4_like_frames()
    rng = np.random.default_rng(1)
    corrections = rng.normal(scale=0.001, size=(
        6, kinematic_model.PARAMS_PER_JOINT))
    thetas = rng.uniform(-1.0, 1.0, size=(20, 6))
    out = metrics.fk_agreement_error(frames, corrections, corrections, thetas)
    assert out['position_rms_mm'] == pytest.approx(0.0, abs=1e-9)
    assert out['rotation_rms_deg'] == pytest.approx(0.0, abs=1e-9)
    assert out['num_configs'] == 20


def test_fk_agreement_grows_with_model_difference():
    frames = _ar4_like_frames()
    rng = np.random.default_rng(2)
    thetas = rng.uniform(-1.0, 1.0, size=(30, 6))
    zero = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))

    errors = []
    for scale in (0.0002, 0.001, 0.005):
        other = np.zeros_like(zero)
        other[:, 0:2] = scale
        errors.append(
            metrics.fk_agreement_error(frames, zero, other, thetas)['position_rms_mm'])
    assert errors[0] < errors[1] < errors[2]


def test_fk_agreement_is_blind_to_the_joint2_joint3_split():
    '''The parallel pair of joints 2 and 3 can be split apart in the model, but the tool's position 
    is only affected by their sum. The FK agreement metricmust not report a large error when the two joints'
    corrections are split between them, because the tool's position is unchanged'''
    frames = _ar4_like_frames()
    rng = np.random.default_rng(11)
    thetas = rng.uniform(-1.0, 1.0, size=(40, 6))
    a = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    a[1, 0:2] = [0.001, 0.0]
    b = np.zeros_like(a)
    b[2, 0:2] = [0.001, 0.0]

    split_apart = metrics.fk_agreement_error(
        frames, a, b, thetas)['position_rms_mm']
    versus_nominal = metrics.fk_agreement_error(
        frames, np.zeros_like(a), a, thetas)['position_rms_mm']

    assert split_apart < 4.0 * versus_nominal
    assert versus_nominal > 0.0


def test_fk_geometry_error_ignores_a_rigid_move_of_the_whole_arm():
    frames = _ar4_like_frames()
    rng = np.random.default_rng(5)
    thetas = rng.uniform(-1.0, 1.0, size=(40, 6))
    zero = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    moved = zero.copy()
    moved[0] = [0.002, -0.001, 0.003, 0.002]
    assert metrics.fk_agreement_error(frames, zero, moved, thetas)[
        'position_rms_mm'] > 1.0
    assert metrics.fk_geometry_error(
        frames, zero, moved, thetas) == pytest.approx(0.0, abs=1e-6)


def test_fk_geometry_error_never_exceeds_the_base_frame_error():
    frames = _ar4_like_frames()
    rng = np.random.default_rng(6)
    thetas = rng.uniform(-1.0, 1.0, size=(40, 6))
    for _ in range(5):
        a = rng.normal(scale=0.002, size=(6, kinematic_model.PARAMS_PER_JOINT))
        b = rng.normal(scale=0.002, size=(6, kinematic_model.PARAMS_PER_JOINT))
        geo = metrics.fk_geometry_error(frames, a, b, thetas)
        assert 0.0 < geo <= metrics.fk_agreement_error(frames, a, b, thetas)[
            'position_rms_mm'] + 1e-9


# Family B

def test_kfold_partitions_are_disjoint_and_cover():
    splits = metrics.kfold_indices(23, 5, np.random.default_rng(0))
    assert len(splits) == 5
    seen = []
    for train, test in splits:
        assert set(train).isdisjoint(
            set(test)), 'a sample leaked into its own training set'
        assert len(train) + len(test) == 23
        seen.extend(test.tolist())
    assert sorted(seen) == list(
        range(23)), 'every sample must be held out exactly once'


def test_kfold_clamps_to_leave_one_out_for_tiny_sets():
    splits = metrics.kfold_indices(3, 10, np.random.default_rng(0))
    assert len(splits) == 3
    assert all(len(test) == 1 for _train, test in splits)


def test_kfold_rejects_degenerate_sample_counts():
    with pytest.raises(ValueError):
        metrics.kfold_indices(1, 5)


def test_bootstrap_is_seed_reproducible():
    a = metrics.bootstrap_indices(10, 5, np.random.default_rng(7))
    b = metrics.bootstrap_indices(10, 5, np.random.default_rng(7))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert all(len(idx) == 10 for idx in a)


def test_summarize_reports_interval_and_ignores_non_finite():
    out = metrics.summarize([1.0, 2.0, 3.0, np.nan, None])
    assert out['n'] == 3
    assert out['mean'] == pytest.approx(2.0)
    assert out['median'] == pytest.approx(2.0)
    assert out['ci95_lo'] <= out['mean'] <= out['ci95_hi']


def test_summarize_handles_empty_input():
    assert metrics.summarize([])['n'] == 0


def test_pivot_conditioning_flags_near_parallel_orientations():
    rng = np.random.default_rng(5)
    p_tool = np.array([0.0, 0.0, 0.15])
    p_base = np.array([0.3, 0.1, 0.2])

    def build(spread_deg):
        R = np.array([
            geometry.axis_angle_to_rotation_matrix(
                rng.normal(size=3), math.radians(rng.uniform(-spread_deg, spread_deg)))
            for _ in range(30)])
        t = np.array([p_base - Ri @ p_tool for Ri in R])
        return R, t

    tight = metrics.pivot_conditioning(*build(1.0))
    wide = metrics.pivot_conditioning(*build(60.0))
    assert tight['condition_number'] > wide['condition_number']
    assert wide['orientation_spread_mean_deg'] > tight['orientation_spread_mean_deg']


def test_motion_axis_diversity_detects_single_axis_rotation():
    angles = np.linspace(0.1, 1.2, 12)
    one_axis = np.array([
        geometry.axis_angle_to_rotation_matrix(np.array([0.0, 0.0, 1.0]), a)
        for a in angles])

    rng = np.random.default_rng(9)
    all_axes = np.array([
        geometry.axis_angle_to_rotation_matrix(
            rng.normal(size=3), rng.uniform(0.2, 1.0))
        for _ in range(12)])

    assert metrics.motion_axis_diversity(one_axis)['axis_rank_ratio'] < 1e-6
    assert metrics.motion_axis_diversity(all_axes)['axis_rank_ratio'] > 0.1


def test_motion_axis_diversity_survives_a_stationary_sequence():
    identical = np.array([np.eye(3)] * 5)
    out = metrics.motion_axis_diversity(identical)
    assert out['num_motions'] == 0
    assert out['axis_rank_ratio'] == 0.0


def test_reprojection_rms_is_zero_for_perfectly_projected_points():
    camera_matrix = np.array([[600.0, 0.0, 320.0],
                              [0.0, 600.0, 240.0],
                              [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(4)
    object_points = np.column_stack([
        rng.uniform(-0.05, 0.05, 20), rng.uniform(-0.05, 0.05, 20), np.zeros(20)])
    rvec = np.array([0.05, -0.03, 0.2])
    tvec = np.array([0.01, -0.02, 0.5])

    import cv2
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, np.zeros(5))

    exact = metrics.reprojection_rms_px(
        object_points, projected, rvec, tvec, camera_matrix, np.zeros(5))
    assert exact == pytest.approx(0.0, abs=1e-9)

    # A known 2-pixel shift on every corner must read back as 2 px.
    shifted = projected.reshape(-1, 2) + np.array([2.0, 0.0])
    assert metrics.reprojection_rms_px(
        object_points, shifted, rvec, tvec, camera_matrix, np.zeros(5)
    ) == pytest.approx(2.0, abs=1e-6)


def test_reprojection_rms_is_nan_without_points():
    assert math.isnan(metrics.reprojection_rms_px(
        np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(3), np.zeros(3), np.eye(3), None))
