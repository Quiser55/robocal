"""Tests the candidate filter that decides whether a joint configuration is
worth commanding"""
import math

import numpy as np
import pytest

from annin_ar4_calibration.core import geometry, joint_sampling, kinematic_model

#: the collector's per-joint observed-range gate (min_joint_range_deg)
RANGE_GATE_DEG = 30.0


def _ar4_joint_frames():
    """Nominal AR4 joint origins/axes with the real URDF limits"""
    pi = math.pi
    specs = [
        ([0, 0, 0.092], [pi, 0, 0], [0, 0, 1], -2.7925, 2.7925),
        ([0, 0.06415, -0.07778], [1.5708, 0, -1.5708], [0, 0, -1], -0.7330, 1.5708),
        ([0, -0.305, 0], [0, 0, pi], [0, 0, -1], -1.5533, 0.9076),
        ([0, 0, 0], [1.5708, 0, -1.5708], [0, 0, -1], -pi, pi),
        ([0, 0, -0.22294], [pi, 0, -1.5708], [1, 0, 0], -1.8326, 1.8326),
        ([0, 0, 0.041], [0, 0, 0], [0, 0, 1], -pi, pi),
    ]
    return [
        kinematic_model.JointFrame(
            xyz0=np.array(xyz, dtype=float), rpy0=np.array(rpy, dtype=float),
            axis=np.array(axis, dtype=float), lower=lo, upper=hi, name=f'joint_{i + 1}')
        for i, (xyz, rpy, axis, lo, hi) in enumerate(specs)
    ]


def _camera():
    """1280x720 colour camera on the tool"""
    return joint_sampling.CameraModel(
        R=np.eye(3), t=np.array([0.0002, -0.0708, 0.0201]), on_tool=True,
        fx=900.0, fy=900.0, cx=640.0, cy=360.0, width=1280, height=720)


def _camera_pose(theta, joint_frames, camera):
    R_base_tool, t_base_tool = kinematic_model.forward_kinematics(
        np.asarray(theta, dtype=float), joint_frames)
    return geometry.compose_rt(R_base_tool, t_base_tool, camera.R, camera.t)


def _target_facing_camera(theta, joint_frames, camera, distance_m=0.4, tilt_deg=0.0):
    """A board placed `distance_m` along the camera's optical axis at `theta`"""
    R_cam, t_cam = _camera_pose(theta, joint_frames, camera)
    centre = t_cam + R_cam @ np.array([0.0, 0.0, distance_m])

    R = R_cam @ geometry.axis_angle_to_rotation_matrix(
        np.array([0.0, 1.0, 0.0]), math.pi)
    R = R @ geometry.axis_angle_to_rotation_matrix(
        np.array([0.0, 1.0, 0.0]), math.radians(tilt_deg))

    size = (0.175, 0.245)
    # place the corner-origin board so its centre sits at `centre`
    origin = centre - R @ np.array([0.5 * size[0], 0.5 * size[1], 0.0])
    return joint_sampling.TargetModel(
        R=R, t=origin, on_tool=False, size_m=size, origin_at_center=False)


def test_board_in_front_of_the_camera_is_visible():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)
    target = _target_facing_camera(theta, jf, cam)

    assert joint_sampling.visible_mask(theta, jf, cam, target)[0]


def test_board_behind_the_camera_is_not_visible():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)
    R_cam, t_cam = _camera_pose(theta, jf, cam)

    behind = _target_facing_camera(theta, jf, cam)
    behind.t = t_cam + R_cam @ np.array([0.0, 0.0, -0.4])

    assert not joint_sampling.visible_mask(theta, jf, cam, behind)[0]


def test_rotating_the_base_away_hides_the_board():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)

    away = np.zeros(6)
    away[0] = 2.5
    assert not joint_sampling.visible_mask(away, jf, cam, target)[0]


def test_range_gate_rejects_a_board_beyond_max_range():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)
    limits = joint_sampling.VisibilityLimits(max_range_m=0.5)

    near = _target_facing_camera(theta, jf, cam, distance_m=0.4)
    far = _target_facing_camera(theta, jf, cam, distance_m=0.9)

    assert joint_sampling.visible_mask(theta, jf, cam, near, limits)[0]
    assert not joint_sampling.visible_mask(theta, jf, cam, far, limits)[0]


def test_incidence_gate_rejects_an_edge_on_board():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)
    limits = joint_sampling.VisibilityLimits(max_view_angle_deg=65.0)

    facing = _target_facing_camera(theta, jf, cam, tilt_deg=0.0)
    edge_on = _target_facing_camera(theta, jf, cam, tilt_deg=80.0)

    assert joint_sampling.visible_mask(theta, jf, cam, facing, limits)[0]
    assert not joint_sampling.visible_mask(theta, jf, cam, edge_on, limits)[0]


def test_a_board_turned_away_is_rejected_once_the_marker_face_is_known():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)

    R_cam, t_cam = _camera_pose(theta, jf, cam)
    facing = _target_facing_camera(theta, jf, cam)
    centred = joint_sampling.TargetModel(
        R=facing.R, t=t_cam + R_cam @ np.array([0.0, 0.0, 0.4]), on_tool=False,
        size_m=facing.size_m, origin_at_center=True)
    turned = joint_sampling.TargetModel(
        R=centred.R @ geometry.axis_angle_to_rotation_matrix(
            np.array([1.0, 0.0, 0.0]), math.pi),
        t=centred.t, on_tool=False, size_m=centred.size_m, origin_at_center=True)

    # with the marker face unknown, both are accepted - the old behaviour
    assert joint_sampling.visible_mask(theta, jf, cam, centred)[0]
    assert joint_sampling.visible_mask(theta, jf, cam, turned)[0]

    # once the face is known, only the one presenting it survives
    for model in (centred, turned):
        model.normal_sign = _sign_of(theta, jf, cam, centred)
    assert joint_sampling.visible_mask(theta, jf, cam, centred)[0]
    assert not joint_sampling.visible_mask(theta, jf, cam, turned)[0]


def _sign_of(theta, joint_frames, camera, target):
    R_cam, t_cam = _camera_pose(theta, joint_frames, camera)
    centre = target.R @ target.corner_points().mean(axis=0) + target.t
    R_cam_target = R_cam.T @ target.R
    t_cam_target = R_cam.T @ (centre - t_cam)
    return joint_sampling.infer_normal_sign(R_cam_target, t_cam_target)


def test_infer_normal_sign_recovers_the_face_a_detection_was_taken_from():
    jf = _ar4_joint_frames()
    cam = _camera()
    theta = np.zeros(6)
    R_cam, t_cam = _camera_pose(theta, jf, cam)

    for flip in (False, True):
        R = np.eye(3) if not flip else geometry.axis_angle_to_rotation_matrix(
            np.array([1.0, 0.0, 0.0]), math.pi)
        t_cam_target = np.array([0.0, 0.0, 0.4])
        R_cam_target = R
        sign = joint_sampling.infer_normal_sign(R_cam_target, t_cam_target)
        expected = 1.0 if not flip else -1.0
        assert sign == expected
    del R_cam, t_cam


def test_sampled_configurations_all_pass_the_filter():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)

    configs, stats = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, rng=np.random.default_rng(0))

    assert stats.kept == 40
    assert not stats.exhausted
    assert joint_sampling.visible_mask(
        np.array(configs), jf, cam, target).all()


def test_filtering_keeps_the_per_joint_diversity_the_solver_needs():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)

    configs, _ = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, rng=np.random.default_rng(0))

    ranges = joint_sampling.observed_ranges_deg(np.array(configs))
    assert (ranges > RANGE_GATE_DEG).all(), ranges


def test_unfiltered_sampling_is_the_bug_this_replaces():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)

    raw = np.array(joint_sampling.sample_joint_configs(
        jf, 3000, rng=np.random.default_rng(0)))
    assert joint_sampling.visible_mask(raw, jf, cam, target).mean() < 0.10


def test_exhausting_the_draw_budget_reports_a_shortfall():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)
    target.t = np.array([50.0, 50.0, 50.0])

    configs, stats = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, rng=np.random.default_rng(0), max_draws=2048)

    assert configs == []
    assert stats.exhausted
    assert stats.drawn == 2048
    assert 'draw budget exhausted' in stats.summary()


def test_sampling_bounds_respect_the_limit_margin():
    jf = _ar4_joint_frames()
    margin = math.radians(5.0)
    lo, hi = joint_sampling.sampling_bounds(jf, margin)

    for i, frame in enumerate(jf):
        assert lo[i] == pytest.approx(frame.lower + margin)
        assert hi[i] == pytest.approx(frame.upper - margin)

    draws = np.array(joint_sampling.sample_joint_configs(
        jf, 200, margin_deg=5.0, rng=np.random.default_rng(3)))
    assert (draws >= lo).all() and (draws <= hi).all()


def test_a_joint_too_narrow_to_shrink_collapses_to_its_midpoint():
    narrow = kinematic_model.JointFrame(
        xyz0=np.zeros(3), rpy0=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]),
        lower=-0.01, upper=0.01, name='narrow')

    lo, hi = joint_sampling.sampling_bounds([narrow], math.radians(5.0))
    assert lo[0] == hi[0] == pytest.approx(0.0)

    draws = np.array(joint_sampling.sample_joint_configs([narrow], 10))
    assert draws == pytest.approx(0.0)


def _collector_bounds_stub(joint_frames, margin_deg):
    lo, hi = joint_sampling.sampling_bounds(
        joint_frames, math.radians(margin_deg))
    return lo, hi


def test_recovery_target_pulls_an_out_of_bounds_joint_back_inside():
    jf = _ar4_joint_frames()
    lo, hi = _collector_bounds_stub(jf, 5.0)

    stuck = np.zeros(6)
    stuck[5] = 3.24647
    clamped = np.clip(stuck, lo, hi)

    assert clamped[5] < jf[5].upper
    assert clamped[5] == pytest.approx(hi[5])
    assert not np.allclose(
        clamped, stuck), 'recovery must actually move joint_6'


def test_recovery_target_is_a_real_move_when_nothing_is_out_of_bounds():
    jf = _ar4_joint_frames()
    lo, hi = _collector_bounds_stub(jf, 5.0)

    inside = 0.5 * (lo + hi) + 0.1
    clamped = np.clip(inside, lo, hi)
    assert np.allclose(
        clamped, inside), 'nothing should have been clamped here'

    fallback = 0.5 * (lo + hi)
    assert not np.allclose(fallback, inside)
    assert ((fallback >= lo) & (fallback <= hi)).all()


def _look_at(eye, at, up=np.array([0.0, 0.0, 1.0])):
    z = at - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _eye_to_hand_rig():
    jf = _ar4_joint_frames()
    eye = np.array([0.7, 0.0, 0.45])
    cam = joint_sampling.CameraModel(
        R=_look_at(eye, np.array([0.0, 0.0, 0.3])), t=eye, on_tool=False,
        fx=900.0, fy=900.0, cx=640.0, cy=360.0, width=1280, height=720)
    target = joint_sampling.TargetModel(
        R=np.eye(3), t=np.array([0.0, -0.05, 0.03]), on_tool=True,
        size_m=(0.175, 0.245), origin_at_center=True)
    return jf, cam, target, joint_sampling.VisibilityLimits(
        min_range_m=0.15, max_range_m=1.5)


def test_eye_to_hand_sampling_finds_configurations_that_show_the_board():
    jf, cam, target, limits = _eye_to_hand_rig()

    configs, stats = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, limits=limits, rng=np.random.default_rng(0))

    assert stats.kept == 40
    assert joint_sampling.visible_mask(
        np.array(configs), jf, cam, target, limits).all()
    ranges = joint_sampling.observed_ranges_deg(np.array(configs))
    assert (ranges > RANGE_GATE_DEG).all(), ranges


def test_eye_to_hand_rejects_configurations_that_turn_the_board_away():
    jf, cam, target, limits = _eye_to_hand_rig()

    configs, _ = joint_sampling.sample_visible_joint_configs(
        jf, 20, cam, target, limits=limits, rng=np.random.default_rng(0))

    turned = np.array(configs)
    turned[:, 4] = np.clip(turned[:, 4] + math.pi, jf[4].lower, jf[4].upper)
    assert not joint_sampling.visible_mask(
        turned, jf, cam, target, limits).all()


@pytest.mark.parametrize('camera_on_tool, target_on_tool', [(True, True), (False, False)])
def test_a_mounting_pair_with_no_moving_frame_is_rejected(camera_on_tool, target_on_tool):
    jf = _ar4_joint_frames()
    cam = joint_sampling.CameraModel(
        R=np.eye(3), t=np.zeros(3), on_tool=camera_on_tool,
        fx=900.0, fy=900.0, cx=640.0, cy=360.0, width=1280, height=720)
    target = joint_sampling.TargetModel(
        R=np.eye(3), t=np.zeros(3), on_tool=target_on_tool)

    with pytest.raises(ValueError, match='exactly one'):
        joint_sampling.visible_mask(np.zeros(6), jf, cam, target)


@pytest.mark.parametrize('eye_in_hand', [True, False])
def test_locating_the_board_from_one_detection_round_trips(eye_in_hand):
    jf = _ar4_joint_frames()
    theta = np.array([0.3, 0.2, -0.4, 0.5, -0.6, 0.7])
    R_base_tool, t_base_tool = kinematic_model.forward_kinematics(theta, jf)

    # hand-eye extrinsic: tool -> camera (eye_in_hand) or base -> camera
    R_mount = geometry.axis_angle_to_rotation_matrix(
        np.array([0.2, -0.5, 0.3]), 0.9)
    t_mount = np.array([0.02, -0.07, 0.03]
                       ) if eye_in_hand else np.array([0.7, 0.1, 0.45])

    R_truth = geometry.axis_angle_to_rotation_matrix(
        np.array([0.1, 0.8, -0.2]), 1.3)
    t_truth = np.array([0.35, -0.12, 0.28]
                       ) if eye_in_hand else np.array([0.0, -0.05, 0.03])

    if eye_in_hand:
        R_base_cam, t_base_cam = geometry.compose_rt(
            R_base_tool, t_base_tool, R_mount, t_mount)
    else:
        R_base_cam, t_base_cam = R_mount, t_mount
    R_base_target, t_base_target = (
        geometry.compose_rt(R_base_tool, t_base_tool, R_truth, t_truth)
        if not eye_in_hand else (R_truth, t_truth))
    R_cam_target = R_base_cam.T @ R_base_target
    t_cam_target = R_base_cam.T @ (t_base_target - t_base_cam)

    R_out, t_out = joint_sampling.locate_target(
        eye_in_hand, R_base_tool, t_base_tool, R_mount, t_mount, R_cam_target, t_cam_target)

    assert t_out == pytest.approx(t_truth, abs=1e-9)
    assert R_out == pytest.approx(R_truth, abs=1e-9)


MEASURED_OVERSHOOT_DEG = 11.0


def _configured_joint_margin_deg():
    import os
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'config', 'kinematic_params.yaml')
    with open(path) as f:
        return yaml.safe_load(f)['kinematic_collector']['ros__parameters']['joint_margin_deg']


def test_the_joint_margin_absorbs_the_arm_overshoot():
    assert _configured_joint_margin_deg() > MEASURED_OVERSHOOT_DEG


def test_the_configured_margin_still_leaves_ample_joint_coverage():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)

    configs, stats = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, margin_deg=_configured_joint_margin_deg(),
        rng=np.random.default_rng(0))

    assert stats.kept == 40
    ranges = joint_sampling.observed_ranges_deg(np.array(configs))
    assert (ranges > 2 * RANGE_GATE_DEG).all(), ranges


def test_recovery_clamp_lands_inside_the_hard_limits_by_the_full_margin():
    jf = _ar4_joint_frames()
    margin = _configured_joint_margin_deg()
    lo, hi = joint_sampling.sampling_bounds(jf, math.radians(margin))

    stuck = np.array([j.upper + 0.05 for j in jf])
    clamped = np.clip(stuck, lo, hi)

    for frame, value in zip(jf, clamped):
        assert frame.lower < value < frame.upper
        assert (frame.upper - value) >= math.radians(margin) - 1e-9


def test_eye_to_hand_sampling_only_returns_front_facing_poses():
    jf, cam, target, limits = _eye_to_hand_rig()

    target.normal_sign = 0.0
    unsigned, _ = joint_sampling.sample_visible_joint_configs(
        jf, 200, cam, target, limits=limits, rng=np.random.default_rng(0))

    target.normal_sign = 1.0
    signed, _ = joint_sampling.sample_visible_joint_configs(
        jf, 200, cam, target, limits=limits, rng=np.random.default_rng(0))

    def front_fraction(configs):
        thetas = np.array(configs)
        R_bt, t_bt = kinematic_model.forward_kinematics_batch(thetas, jf)
        R_bT = R_bt @ target.R
        t_bT = np.einsum('nij,j->ni', R_bt, target.t) + t_bt
        centre = np.einsum('nij,j->ni', R_bT,
                           target.corner_points().mean(axis=0)) + t_bT
        in_cam = np.einsum('ji,nj->ni', cam.R, centre - cam.t)
        normal = np.einsum('ji,nj->ni', cam.R, R_bT[:, :, 2])
        view = in_cam / np.linalg.norm(in_cam, axis=1)[:, None]
        return float((np.einsum('ni,ni->n', normal, view) > 0).mean())

    assert front_fraction(signed) == 1.0
    assert front_fraction(
        unsigned) < 1.0, 'unsigned must be the permissive case'


def test_eye_in_hand_still_finds_candidates_with_the_face_constraint():
    jf = _ar4_joint_frames()
    cam = _camera()
    target = _target_facing_camera(np.zeros(6), jf, cam)
    target.normal_sign = _sign_of(np.zeros(6), jf, cam, target)

    configs, stats = joint_sampling.sample_visible_joint_configs(
        jf, 40, cam, target, margin_deg=_configured_joint_margin_deg(),
        rng=np.random.default_rng(0))

    assert stats.kept == 40 and not stats.exhausted
    ranges = joint_sampling.observed_ranges_deg(np.array(configs))
    assert (ranges > 2 * RANGE_GATE_DEG).all(), ranges
