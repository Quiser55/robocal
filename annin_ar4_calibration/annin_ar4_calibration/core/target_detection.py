"""Pure-OpenCV ChArUco/ArUco target detection and pose estimation.

No ROS message dependencies (mirrors geometry.py's pure-numeric-core pattern).
Detection always returns the target's pose as (R, t) expressed in the camera's
own optical frame (camera_optical_frame -> target), ready to hand to
handeye_solver / geometry.rt_to_transform.

OpenCV API note: Ubuntu 24.04's apt python3-opencv is 4.6.0, which only has the
*legacy* Charuco API (CharucoBoard_create/interpolateCornersCharuco/
estimatePoseCharucoBoard/Dictionary_get/DetectorParameters_create). OpenCV
>=4.7 replaced this with a class-based CharucoBoard/CharucoDetector API. Both
branches are implemented behind the same build_target()/detect() entry
points, selected once at import time via _LEGACY_ARUCO - only the legacy
branch has been exercised, since that's what's actually installed here; treat
the new-API branch as best-effort/untested.
"""
from dataclasses import dataclass

import cv2
import numpy as np

_LEGACY_ARUCO = hasattr(cv2.aruco, 'CharucoBoard_create')


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray  # (3,3)
    dist_coeffs: np.ndarray    # (N,)


@dataclass
class TargetConfig:
    target_type: str  # 'charuco' | 'aruco'
    dictionary_name: str = 'DICT_5X5_250'
    # charuco board geometry
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    min_charuco_corners: int = 6
    # single-marker fallback mode
    marker_id: int = 0
    aruco_marker_length_m: float = 0.05


@dataclass
class DetectionResult:
    found: bool
    R: np.ndarray | None = None       # (3,3), camera_optical_frame -> target
    t: np.ndarray | None = None       # (3,)
    num_points: int = 0
    debug_image: np.ndarray | None = None


def _get_dictionary(dictionary_name: str):
    dict_id = getattr(cv2.aruco, dictionary_name)
    if _LEGACY_ARUCO:
        return cv2.aruco.Dictionary_get(dict_id)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def build_target(cfg: TargetConfig):
    """Builds the dictionary/board objects once, to be reused across frames."""
    dictionary = _get_dictionary(cfg.dictionary_name)
    if cfg.target_type == 'charuco':
        if _LEGACY_ARUCO:
            board = cv2.aruco.CharucoBoard_create(
                cfg.squares_x, cfg.squares_y, cfg.square_length_m, cfg.marker_length_m,
                dictionary)
        else:
            board = cv2.aruco.CharucoBoard(
                (cfg.squares_x, cfg.squares_y), cfg.square_length_m, cfg.marker_length_m,
                dictionary)
        return ('charuco', dictionary, board, cfg)
    elif cfg.target_type == 'aruco':
        return ('aruco', dictionary, None, cfg)
    raise ValueError(f'Unknown target_type {cfg.target_type!r}')


def _detect_markers(image_gray, dictionary):
    if _LEGACY_ARUCO:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            image_gray, dictionary, parameters=params)
    else:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _rejected = detector.detectMarkers(image_gray)
    return corners, ids


def _detect_charuco(image_gray, intrinsics, dictionary, board, cfg, draw_debug, debug_image):
    corners, ids = _detect_markers(image_gray, dictionary)
    if ids is None or len(ids) == 0:
        return DetectionResult(found=False)

    if draw_debug:
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)

    if _LEGACY_ARUCO:
        count, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image_gray, board,
            cameraMatrix=intrinsics.camera_matrix, distCoeffs=intrinsics.dist_coeffs)
        if ch_ids is None or count < cfg.min_charuco_corners:
            return DetectionResult(found=False, num_points=int(count))

        if draw_debug:
            cv2.aruco.drawDetectedCornersCharuco(debug_image, ch_corners, ch_ids)

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board, intrinsics.camera_matrix, intrinsics.dist_coeffs,
            None, None)
        if not ok:
            return DetectionResult(found=False, num_points=int(count))
        num_points = int(count)
    else:
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _marker_corners, _marker_ids = detector.detectBoard(image_gray)
        if ch_corners is None or len(ch_corners) < cfg.min_charuco_corners:
            return DetectionResult(found=False)
        obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, intrinsics.camera_matrix, intrinsics.dist_coeffs)
        if not ok:
            return DetectionResult(found=False)
        num_points = len(ch_corners)

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec).reshape(3)

    if draw_debug:
        cv2.drawFrameAxes(
            debug_image, intrinsics.camera_matrix, intrinsics.dist_coeffs, rvec, tvec,
            cfg.square_length_m)

    return DetectionResult(
        found=True, R=R, t=t, num_points=num_points,
        debug_image=debug_image if draw_debug else None)


def _detect_aruco_marker(image_gray, intrinsics, dictionary, cfg, draw_debug, debug_image):
    corners, ids = _detect_markers(image_gray, dictionary)
    if ids is None:
        return DetectionResult(found=False)

    matches = np.where(ids.flatten() == cfg.marker_id)[0]
    if len(matches) == 0:
        return DetectionResult(found=False)
    idx = int(matches[0])

    if draw_debug:
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)

    if _LEGACY_ARUCO:
        rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], cfg.aruco_marker_length_m, intrinsics.camera_matrix,
            intrinsics.dist_coeffs)
        rvec, tvec = rvecs[0], tvecs[0]
    else:
        half = cfg.aruco_marker_length_m / 2.0
        obj_points = np.array([
            [-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0],
        ])
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, corners[idx], intrinsics.camera_matrix, intrinsics.dist_coeffs)
        if not ok:
            return DetectionResult(found=False)

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec).reshape(3)

    if draw_debug:
        cv2.drawFrameAxes(
            debug_image, intrinsics.camera_matrix, intrinsics.dist_coeffs, rvec, tvec,
            cfg.aruco_marker_length_m)

    return DetectionResult(
        found=True, R=R, t=t, num_points=4, debug_image=debug_image if draw_debug else None)


def detect(
        image_bgr: np.ndarray, intrinsics: CameraIntrinsics, target,
        draw_debug: bool = False) -> DetectionResult:
    """target: the opaque object returned by build_target()."""
    target_type, dictionary, board, cfg = target
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    debug_image = image_bgr.copy() if draw_debug else None

    if target_type == 'charuco':
        return _detect_charuco(
            image_gray, intrinsics, dictionary, board, cfg, draw_debug, debug_image)
    return _detect_aruco_marker(image_gray, intrinsics, dictionary, cfg, draw_debug, debug_image)
