"""Hand-eye calibration solver: wraps cv2.calibrateHandEye for both the eye_in_hand and eye_to_hand configurations."""
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
    # 'tool_frame' (eye_in_hand) | 'base_frame' (eye_to_hand)
    parent_frame_role: str
    consistency_rms_m: float
    consistency_rms_deg: float
    num_samples: int
    method: str


def _mean_quaternion(quats: np.ndarray) -> np.ndarray:
    """Mean of unit quaternions (x,y,z,w rows)"""
    aligned = quats.copy()
    for i in range(1, aligned.shape[0]):
        if np.dot(aligned[0], aligned[i]) < 0:
            aligned[i] = -aligned[i]
    mean = aligned.mean(axis=0)
    return mean / np.linalg.norm(mean)


def chain_reference(chain_R: np.ndarray, chain_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a chain-pose sequence, as (mean quaternion, mean translation)."""
    quats = np.array([geometry.rotation_matrix_to_quat(R) for R in chain_R])
    return _mean_quaternion(quats), chain_t.mean(axis=0)


def chain_deviation(chain_R: np.ndarray, chain_t: np.ndarray,
                    ref_q: np.ndarray, ref_t: np.ndarray) -> tuple[float, float]:
    """RMS deviation , as (translation in metres, rotation in degrees)."""
    rms_m = float(
        np.sqrt(np.mean(np.linalg.norm(chain_t - ref_t, axis=1) ** 2)))
    quats = np.array([geometry.rotation_matrix_to_quat(R) for R in chain_R])
    angles = np.array([geometry.quat_angle_deg(q, ref_q) for q in quats])
    rms_deg = float(np.sqrt(np.mean(angles ** 2)))
    return rms_m, rms_deg


def _consistency(chain_R: np.ndarray, chain_t: np.ndarray) -> tuple[float, float]:
    """poses chain_R (N,3,3) / chain_t (N,3) Returns (translation RMS spread in meters,
      rotation RMS spread in degrees)."""
    return chain_deviation(chain_R, chain_t, *chain_reference(chain_R, chain_t))


def hand_poses(calibration_type: str, robot_R: np.ndarray, robot_t: np.ndarray) -> tuple:
    """The pose sequence `cv2.calibrateHandEye` is actually fed, given raw robot samples:"""
    if calibration_type == 'eye_in_hand':
        return robot_R, robot_t
    if calibration_type == 'eye_to_hand':
        return geometry.batch_invert_rt(robot_R, robot_t)
    raise ValueError(f'Unknown calibration_type {calibration_type!r}')


def chain_poses(
        calibration_type: str, robot_R: np.ndarray, robot_t: np.ndarray,
        target_R: np.ndarray, target_t: np.ndarray,
        X_R: np.ndarray, X_t: np.ndarray) -> tuple:
    hand_R, hand_t = hand_poses(calibration_type, robot_R, robot_t)
    parent_R, parent_t = geometry.batch_compose_rt(
        X_R, X_t, target_R, target_t)
    return geometry.batch_compose_rt(hand_R, hand_t, parent_R, parent_t)


def solve_hand_eye(
        calibration_type: str,
        robot_R: np.ndarray, robot_t: np.ndarray,
        target_R: np.ndarray, target_t: np.ndarray,
        method: str = 'PARK') -> HandEyeResult:
    """robot_R/robot_t: (N,3,3)/(N,3) samples of base_frame -> tool_frame.
    target_R/target_t: (N,3,3)/(N,3) samples of camera_optical_frame -> target"""
    hand_R, hand_t = hand_poses(calibration_type, robot_R, robot_t)
    parent_frame_role = ('tool_frame' if calibration_type == 'eye_in_hand'
                         else 'base_frame')

    R_out, t_out = cv2.calibrateHandEye(
        list(hand_R), list(hand_t), list(target_R), list(target_t),
        method=_METHODS[method])
    t_out = t_out.reshape(3)

    chain_R, chain_t = chain_poses(
        calibration_type, robot_R, robot_t, target_R, target_t, R_out, t_out)
    rms_m, rms_deg = _consistency(chain_R, chain_t)

    return HandEyeResult(
        R=R_out, t=t_out, parent_frame_role=parent_frame_role,
        consistency_rms_m=rms_m, consistency_rms_deg=rms_deg,
        num_samples=robot_R.shape[0], method=method)
