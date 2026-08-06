"""Kinematic parameter identification solver.

Uses the WP3 hand-eye extrinsic (base_T_cam for eye_to_hand, ee_T_cam for
eye_in_hand - already computed and trusted) as a known measurement device,
together with paired (joint_angles, camera->target pose) samples, to jointly
identify:

- per-joint corrections (see core/kinematic_model.py for the restricted
  4-params-per-joint parameterization and why it's restricted),
- a fixed 6-DOF "mount offset" nuisance transform: ee_T_marker (the tag's
  mount offset on the arm) for eye_to_hand, or base_T_marker (the tag's fixed
  pose in the workspace) for eye_in_hand.

Two solve strategies are provided so they can be compared directly (WP4
explicitly asks to test "iterative" vs "sequential" approaches):

- solve_joint: a single nonlinear least-squares over all unknowns at once
  (scipy.optimize.least_squares, 'trf' with bounds by default - see module
  docstring discussion in the design plan for why 'lm' is offered only as a
  secondary, non-unique diagnostic given the near-singular J2/J3 axis pair).
- solve_sequential: alternates a closed-form mount-offset estimate (rotation
  averaging via geometry.mean_rotation_markley + linear translation
  least-squares) with a bounded Gauss-Newton refinement of just the
  kinematic corrections (mount offset held fixed), for a few outer
  iterations.

For eye_in_hand, joint 1's correction block is exactly unobservable (nothing
theta-dependent sits between it and the free base_T_marker nuisance
parameter - see core/kinematic_model.py) and must be fixed to zero by the
caller (`fix_joint1=True`).

All residual/mount-offset computation below is vectorized over the sample
batch (N, ...) rather than looped in Python - with a numerical Jacobian,
scipy.optimize.least_squares needs O(n_params) residual evaluations per
iteration, so a per-sample Python loop here would make compute_calibration
far too slow to be a synchronous Trigger service call.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from annin_ar4_calibration.core import geometry, kinematic_model

#: characteristic lever-arm length (~ arm reach), used to bring rotation-vector
#: residual components (radians) into meter-comparable units alongside
#: translation residual components, for balanced least-squares weighting.
LEVER_ARM_M = 0.4


@dataclass
class KinematicCalibrationResult:
    corrections: np.ndarray    # (6, PARAMS_PER_JOINT) per-joint correction coefficients
    mount_R: np.ndarray         # (3,3)
    mount_t: np.ndarray         # (3,)
    mount_role: str             # 'ee_T_marker' (eye_to_hand) | 'base_T_marker' (eye_in_hand)
    rms_before_m: float
    rms_after_m: float
    method: str
    num_samples: int
    condition_number: float
    singular_values: np.ndarray
    converged: bool
    iterations: int
    fixed_joint1: bool


def _mount_role(calibration_type: str) -> str:
    return 'ee_T_marker' if calibration_type == 'eye_to_hand' else 'base_T_marker'


def _free_kin_count(fix_joint1: bool) -> int:
    n_joints = 5 if fix_joint1 else 6
    return n_joints * kinematic_model.PARAMS_PER_JOINT


def _corrections_from_free(x_kin: np.ndarray, fix_joint1: bool) -> np.ndarray:
    corrections = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    if fix_joint1:
        corrections[1:6, :] = x_kin.reshape(5, kinematic_model.PARAMS_PER_JOINT)
    else:
        corrections[0:6, :] = x_kin.reshape(6, kinematic_model.PARAMS_PER_JOINT)
    return corrections


def _kin_bounds(fix_joint1: bool, bounds_translation_m: float,
                bounds_rotation_rad: float) -> tuple:
    n_joints = 5 if fix_joint1 else 6
    per_joint_upper = np.array([
        bounds_translation_m, bounds_translation_m,
        bounds_rotation_rad, bounds_rotation_rad,
    ])
    upper = np.tile(per_joint_upper, n_joints)
    return -upper, upper


def _mount_offset_batch(
        calibration_type: str, pred_R: np.ndarray, pred_t: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        meas_target_R: np.ndarray, meas_target_t: np.ndarray) -> tuple:
    """Per-sample estimate of the constant mount-offset unknown X, given the
    current (possibly corrected) FK, batched over N samples. eye_to_hand:
    X = ee_T_marker, estimated as predicted_ee^-1 @ (base_T_cam @
    cam_T_marker). eye_in_hand: X = base_T_marker, estimated directly as
    predicted_ee @ ee_T_cam @ cam_T_marker (no inverse - X sits at the
    chain's far, un-anchored end)."""
    meas_R, meas_t = geometry.batch_compose_rt(known_R, known_t, meas_target_R, meas_target_t)
    if calibration_type == 'eye_to_hand':
        pred_R_inv = pred_R.transpose(0, 2, 1)
        pred_t_inv = -np.einsum('nij,nj->ni', pred_R_inv, pred_t)
        return geometry.batch_compose_rt(pred_R_inv, pred_t_inv, meas_R, meas_t)
    elif calibration_type == 'eye_in_hand':
        return geometry.batch_compose_rt(pred_R, pred_t, meas_R, meas_t)
    else:
        raise ValueError(f'Unknown calibration_type {calibration_type!r}')


def reconstruct_measured_ee(
        calibration_type: str, mount_R: np.ndarray, mount_t: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray) -> tuple:
    """Given a solved mount offset and the known WP3 hand-eye extrinsic,
    reconstructs the "ground truth" base_frame -> ee_link pose implied by one
    (or a batch of) camera->target measurement(s) - for comparing against
    nominal/corrected FK predictions (e.g. in kinematic_visualization_node).
    Accepts either single (3,3)/(3,) or batched (N,3,3)/(N,3) cam_target_R/t."""
    meas_R, meas_t = geometry.batch_compose_rt(known_R, known_t, cam_target_R, cam_target_t)
    if calibration_type == 'eye_to_hand':
        # meas = base_T_marker, mount = ee_T_marker -> base_T_ee = meas @ mount^-1
        mount_R_inv = mount_R.T
        mount_t_inv = -mount_R_inv @ mount_t
        return geometry.batch_compose_rt(meas_R, meas_t, mount_R_inv, mount_t_inv)
    elif calibration_type == 'eye_in_hand':
        # meas = ee_T_marker, mount = base_T_marker -> base_T_ee = mount @ meas^-1
        meas_R_inv = meas_R.swapaxes(-1, -2)
        meas_t_inv = -np.einsum('...ij,...j->...i', meas_R_inv, meas_t)
        return geometry.batch_compose_rt(mount_R, mount_t, meas_R_inv, meas_t_inv)
    else:
        raise ValueError(f'Unknown calibration_type {calibration_type!r}')


def _lhs_rhs_batch(
        calibration_type: str, pred_R: np.ndarray, pred_t: np.ndarray,
        mount_R: np.ndarray, mount_t: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        meas_target_R: np.ndarray, meas_target_t: np.ndarray) -> tuple:
    """Both sides of the loop-closure equation, batched over N samples,
    expressed as base_frame -> marker transforms: predicted (via FK + mount
    offset) vs. measured (via the known WP3 extrinsic + the camera->target
    detections)."""
    n = pred_R.shape[0]
    meas_R, meas_t = geometry.batch_compose_rt(known_R, known_t, meas_target_R, meas_target_t)
    if calibration_type == 'eye_to_hand':
        lhs_R, lhs_t = geometry.batch_compose_rt(pred_R, pred_t, mount_R, mount_t)
        return lhs_R, lhs_t, meas_R, meas_t
    elif calibration_type == 'eye_in_hand':
        rhs_R, rhs_t = geometry.batch_compose_rt(pred_R, pred_t, meas_R, meas_t)
        mount_R_b = np.broadcast_to(mount_R, (n, 3, 3))
        mount_t_b = np.broadcast_to(mount_t, (n, 3))
        return mount_R_b, mount_t_b, rhs_R, rhs_t
    else:
        raise ValueError(f'Unknown calibration_type {calibration_type!r}')


def _pose_residual_batch(
        lhs_R: np.ndarray, lhs_t: np.ndarray, rhs_R: np.ndarray, rhs_t: np.ndarray) -> np.ndarray:
    trans_res = lhs_t - rhs_t
    rel_R = np.einsum('nij,njk->nik', rhs_R.transpose(0, 2, 1), lhs_R)
    rot_res = geometry.batch_rotation_matrix_to_rotvec(rel_R) * LEVER_ARM_M
    return np.concatenate([trans_res, rot_res], axis=1)


def _mount_offset_closed_form(
        calibration_type: str, theta: np.ndarray, corrections: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list) -> tuple:
    pred_R, pred_t = kinematic_model.forward_kinematics_batch(theta, joint_frames, corrections)
    Xi_R, Xi_t = _mount_offset_batch(
        calibration_type, pred_R, pred_t, known_R, known_t, cam_target_R, cam_target_t)
    quats = np.array([geometry.rotation_matrix_to_quat(R) for R in Xi_R])
    mount_R = geometry.mean_rotation_markley(quats)
    mount_t = Xi_t.mean(axis=0)
    return mount_R, mount_t


def _rms_pose_error(
        calibration_type: str, theta: np.ndarray, corrections: np.ndarray,
        mount_R: np.ndarray, mount_t: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list) -> float:
    pred_R, pred_t = kinematic_model.forward_kinematics_batch(theta, joint_frames, corrections)
    lhs_R, lhs_t, rhs_R, rhs_t = _lhs_rhs_batch(
        calibration_type, pred_R, pred_t, mount_R, mount_t,
        known_R, known_t, cam_target_R, cam_target_t)
    sq_errs = np.sum((lhs_t - rhs_t) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq_errs)))


def _residual_vector(
        x: np.ndarray, calibration_type: str, theta: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list, fix_joint1: bool, ridge_lambda: float) -> np.ndarray:
    n_kin = _free_kin_count(fix_joint1)
    x_kin = x[:n_kin]
    mount_R = geometry.rotvec_to_rotation_matrix(x[n_kin:n_kin + 3])
    mount_t = x[n_kin + 3:n_kin + 6]
    corrections = _corrections_from_free(x_kin, fix_joint1)

    pred_R, pred_t = kinematic_model.forward_kinematics_batch(theta, joint_frames, corrections)
    lhs_R, lhs_t, rhs_R, rhs_t = _lhs_rhs_batch(
        calibration_type, pred_R, pred_t, mount_R, mount_t,
        known_R, known_t, cam_target_R, cam_target_t)
    residuals = _pose_residual_batch(lhs_R, lhs_t, rhs_R, rhs_t)

    ridge = np.sqrt(ridge_lambda) * x_kin
    return np.concatenate([residuals.ravel(), ridge])


def _numerical_jacobian(residual_fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    f0 = residual_fn(x)
    J = np.zeros((f0.size, x.size))
    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = eps
        J[:, j] = (residual_fn(x + dx) - f0) / eps
    return J


def _jacobian_diagnostics(residual_fn, x: np.ndarray) -> tuple:
    J = _numerical_jacobian(residual_fn, x)
    singular_values = np.linalg.svd(J, compute_uv=False)
    smallest = max(singular_values[-1], 1e-12)
    condition_number = float(singular_values[0] / smallest)
    return condition_number, singular_values


def _baseline_rms(
        calibration_type: str, theta: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list) -> tuple:
    """RMS with nominal (uncorrected) kinematics, best-fit mount offset only
    - the fair baseline both solvers report improvement against."""
    zero_corr = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    mount_R, mount_t = _mount_offset_closed_form(
        calibration_type, theta, zero_corr, known_R, known_t,
        cam_target_R, cam_target_t, joint_frames)
    rms = _rms_pose_error(
        calibration_type, theta, zero_corr, mount_R, mount_t,
        known_R, known_t, cam_target_R, cam_target_t, joint_frames)
    return rms, mount_R, mount_t


def solve_sequential(
        calibration_type: str, theta: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list, fix_joint1: bool,
        ridge_lambda: float = 1e-6,
        bounds_translation_m: float = 0.01, bounds_rotation_deg: float = 3.0,
        max_outer_iters: int = 10, tol: float = 1e-9) -> KinematicCalibrationResult:
    rms_before, mount_R, mount_t = _baseline_rms(
        calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t, joint_frames)

    n_kin = _free_kin_count(fix_joint1)
    x_kin = np.zeros(n_kin)
    lb, ub = _kin_bounds(fix_joint1, bounds_translation_m, np.radians(bounds_rotation_deg))

    outer_iters_run = 0
    converged = False
    for outer in range(max_outer_iters):
        outer_iters_run += 1
        prev_x_kin = x_kin.copy()

        # (a) closed-form mount-offset estimate, kinematic corrections held fixed
        corrections = _corrections_from_free(x_kin, fix_joint1)
        mount_R, mount_t = _mount_offset_closed_form(
            calibration_type, theta, corrections, known_R, known_t,
            cam_target_R, cam_target_t, joint_frames)

        # (b) bounded Gauss-Newton refinement of kinematic corrections, mount fixed
        mount_rotvec = geometry.rotation_matrix_to_rotvec(mount_R)

        def _resid_kin_only(xk, _mount_rotvec=mount_rotvec, _mount_t=mount_t):
            x_full = np.concatenate([xk, _mount_rotvec, _mount_t])
            return _residual_vector(
                x_full, calibration_type, theta, known_R, known_t,
                cam_target_R, cam_target_t, joint_frames, fix_joint1, ridge_lambda)

        res = least_squares(
            _resid_kin_only, x_kin, bounds=(lb, ub), method='trf', x_scale='jac', max_nfev=20)
        x_kin = res.x

        if np.linalg.norm(x_kin - prev_x_kin) < tol:
            converged = True
            break

    corrections = _corrections_from_free(x_kin, fix_joint1)
    mount_R, mount_t = _mount_offset_closed_form(
        calibration_type, theta, corrections, known_R, known_t,
        cam_target_R, cam_target_t, joint_frames)
    rms_after = _rms_pose_error(
        calibration_type, theta, corrections, mount_R, mount_t,
        known_R, known_t, cam_target_R, cam_target_t, joint_frames)

    mount_rotvec = geometry.rotation_matrix_to_rotvec(mount_R)
    x_final = np.concatenate([x_kin, mount_rotvec, mount_t])

    def _resid_full(x):
        return _residual_vector(
            x, calibration_type, theta, known_R, known_t,
            cam_target_R, cam_target_t, joint_frames, fix_joint1, ridge_lambda)

    condition_number, singular_values = _jacobian_diagnostics(_resid_full, x_final)

    return KinematicCalibrationResult(
        corrections=corrections, mount_R=mount_R, mount_t=mount_t,
        mount_role=_mount_role(calibration_type),
        rms_before_m=rms_before, rms_after_m=rms_after, method='sequential',
        num_samples=theta.shape[0], condition_number=condition_number,
        singular_values=singular_values, converged=converged,
        iterations=outer_iters_run, fixed_joint1=fix_joint1)


def solve_joint(
        calibration_type: str, theta: np.ndarray,
        known_R: np.ndarray, known_t: np.ndarray,
        cam_target_R: np.ndarray, cam_target_t: np.ndarray,
        joint_frames: list, fix_joint1: bool,
        ridge_lambda: float = 1e-6,
        bounds_translation_m: float = 0.01, bounds_rotation_deg: float = 3.0,
        optimizer: str = 'trf') -> KinematicCalibrationResult:
    rms_before, mount_R0, mount_t0 = _baseline_rms(
        calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t, joint_frames)

    n_kin = _free_kin_count(fix_joint1)
    mount_rotvec0 = geometry.rotation_matrix_to_rotvec(mount_R0)
    x0 = np.concatenate([np.zeros(n_kin), mount_rotvec0, mount_t0])

    def _resid_full(x):
        return _residual_vector(
            x, calibration_type, theta, known_R, known_t,
            cam_target_R, cam_target_t, joint_frames, fix_joint1, ridge_lambda)

    if optimizer == 'trf':
        kin_lb, kin_ub = _kin_bounds(
            fix_joint1, bounds_translation_m, np.radians(bounds_rotation_deg))
        lb = np.concatenate([kin_lb, np.full(6, -np.inf)])
        ub = np.concatenate([kin_ub, np.full(6, np.inf)])
        res = least_squares(_resid_full, x0, bounds=(lb, ub), method='trf', x_scale='jac')
    elif optimizer == 'lm':
        # Unconstrained: not guaranteed unique near the J2/J3 near-singular
        # direction - report as a secondary diagnostic only, not the primary
        # 'joint' result. See module docstring.
        res = least_squares(_resid_full, x0, method='lm', x_scale='jac')
    else:
        raise ValueError(f'Unknown optimizer {optimizer!r}')

    x_final = res.x
    x_kin = x_final[:n_kin]
    mount_R = geometry.rotvec_to_rotation_matrix(x_final[n_kin:n_kin + 3])
    mount_t = x_final[n_kin + 3:n_kin + 6]
    corrections = _corrections_from_free(x_kin, fix_joint1)

    rms_after = _rms_pose_error(
        calibration_type, theta, corrections, mount_R, mount_t,
        known_R, known_t, cam_target_R, cam_target_t, joint_frames)
    condition_number, singular_values = _jacobian_diagnostics(_resid_full, x_final)

    return KinematicCalibrationResult(
        corrections=corrections, mount_R=mount_R, mount_t=mount_t,
        mount_role=_mount_role(calibration_type),
        rms_before_m=rms_before, rms_after_m=rms_after,
        method=f'joint_{optimizer}',
        num_samples=theta.shape[0], condition_number=condition_number,
        singular_values=singular_values, converged=bool(res.success),
        iterations=int(res.nfev), fixed_joint1=fix_joint1)
