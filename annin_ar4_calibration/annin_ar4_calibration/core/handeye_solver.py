"""Hand-eye calibration solver: wraps cv2.calibrateHandEye for both the
eye_in_hand and eye_to_hand configurations.

cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
returns (R_cam2gripper, t_cam2gripper), i.e. tool_frame -> camera_optical_frame.
Feeding it the robot samples as-is (base_frame -> tool_frame) directly gives
the eye_in_hand answer. For eye_to_hand (camera fixed, target mounted on the
gripper), feeding the *inverted* robot samples (tool_frame -> base_frame)
instead gives base_frame -> camera_optical_frame directly - a documented
trick also used by easy_handeye2's own OpenCV backend. No further TF
composition is needed in either case; both results are already expressed
directly in the frame that gets published (see core/paths.py callers).
"""
from dataclasses import dataclass

import cv2
import numpy as np

from annin_ar4_calibration.core import geometry

_METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeResult:
    R: np.ndarray             # (3,3) parent_frame -> camera_optical_frame
    t: np.ndarray             # (3,)
    parent_frame_role: str    # 'tool_frame' (eye_in_hand) | 'base_frame' (eye_to_hand)
    consistency_rms_m: float
    consistency_rms_deg: float
    num_samples: int
    method: str


def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """Mean of unit quaternions (x,y,z,w rows), sign-aligned to the first
    sample first to avoid double-cover cancellation."""
    aligned = quats.copy()
    for i in range(1, aligned.shape[0]):
        if np.dot(aligned[0], aligned[i]) < 0:
            aligned[i] = -aligned[i]
    mean = aligned.mean(axis=0)
    return mean / np.linalg.norm(mean)


def _consistency(chain_R: np.ndarray, chain_t: np.ndarray) -> tuple[float, float]:
    """chain_R (N,3,3) / chain_t (N,3) are per-sample poses of the quantity
    that should be constant across all samples if the calibration is correct
    (see solve_hand_eye's docstring for what that quantity is per
    calibration_type). Returns (translation RMS spread in meters, rotation
    RMS spread in degrees)."""
    mean_t = chain_t.mean(axis=0)
    rms_m = float(np.sqrt(np.mean(np.linalg.norm(chain_t - mean_t, axis=1) ** 2)))

    quats = np.array([geometry.rotation_matrix_to_quat(R) for R in chain_R])
    mean_q = _mean_quaternion(quats)
    angles = np.array([geometry.quat_angle_deg(q, mean_q) for q in quats])
    rms_deg = float(np.sqrt(np.mean(angles ** 2)))
    return rms_m, rms_deg


def solve_hand_eye(
        calibration_type: str,
        robot_R: np.ndarray, robot_t: np.ndarray,
        target_R: np.ndarray, target_t: np.ndarray,
        method: str = 'PARK') -> HandEyeResult:
    """robot_R/robot_t: (N,3,3)/(N,3) samples of base_frame -> tool_frame.
    target_R/target_t: (N,3,3)/(N,3) samples of camera_optical_frame -> target,
    captured at the same instants. Both are produced by
    geometry.batch_samples_to_rt on the two halves of a stored (N,14) sample
    row.

    For 'eye_in_hand', the returned transform is directly tool_frame ->
    camera_optical_frame, and the consistency check verifies that the
    target's pose in base_frame (where it's physically fixed) comes out the
    same for every sample. For 'eye_to_hand', the returned transform is
    directly base_frame -> camera_optical_frame, and the consistency check
    verifies that the target's pose in tool_frame (where it's physically
    mounted) comes out the same for every sample.
    """
    if calibration_type == 'eye_in_hand':
        hand_R, hand_t = robot_R, robot_t
        parent_frame_role = 'tool_frame'
    elif calibration_type == 'eye_to_hand':
        hand_R, hand_t = geometry.batch_invert_rt(robot_R, robot_t)
        parent_frame_role = 'base_frame'
    else:
        raise ValueError(f'Unknown calibration_type {calibration_type!r}')

    R_out, t_out = cv2.calibrateHandEye(
        list(hand_R), list(hand_t), list(target_R), list(target_t),
        method=_METHODS[method])
    t_out = t_out.reshape(3)

    n = robot_R.shape[0]
    chain_R = np.empty((n, 3, 3))
    chain_t = np.empty((n, 3))
    for i in range(n):
        cam_to_target = (target_R[i], target_t[i])
        parent_to_target = geometry.compose_rt(R_out, t_out, *cam_to_target)
        chain_R[i], chain_t[i] = geometry.compose_rt(
            hand_R[i], hand_t[i], *parent_to_target)

    rms_m, rms_deg = _consistency(chain_R, chain_t)

    return HandEyeResult(
        R=R_out, t=t_out, parent_frame_role=parent_frame_role,
        consistency_rms_m=rms_m, consistency_rms_deg=rms_deg,
        num_samples=n, method=method)
