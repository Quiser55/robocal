"""Generates candidate joint configurations spanning the workspace,
 and filters them for predicted visibility of the target.
 """
from dataclasses import dataclass

import numpy as np

from annin_ar4_calibration.core import geometry, kinematic_model


def sampling_bounds(joint_frames: list, margin_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint (lower, upper) draw interval:  joint limits shrunk by `margin_rad` on each side"""
    lo = np.empty(len(joint_frames))
    hi = np.empty(len(joint_frames))
    for i, jf in enumerate(joint_frames):
        if (jf.upper - jf.lower) > 2 * margin_rad:
            lo[i], hi[i] = jf.lower + margin_rad, jf.upper - margin_rad
        else:
            lo[i] = hi[i] = 0.5 * (jf.lower + jf.upper)
    return lo, hi


def sample_joint_configs(
        joint_frames: list, n: int,
        margin_deg: float = 5.0,
        rng: np.random.Generator | None = None) -> list:
    rng = rng or np.random.default_rng()
    lo, hi = sampling_bounds(joint_frames, np.radians(margin_deg))

    configs = []
    for _ in range(n):
        theta = np.array([
            rng.uniform(lo[i], hi[i]) if hi[i] > lo[i] else lo[i]
            for i in range(len(joint_frames))
        ])
        configs.append(theta)
    return configs


def _draw_batch(lo: np.ndarray, hi: np.ndarray, n: int,
                rng: np.random.Generator) -> np.ndarray:
    """(n, J) uniform draw within [lo, hi] per joint, vectorized"""
    return lo + (hi - lo) * rng.random((n, len(lo)))


@dataclass
class Mounting:
    """A frame's placement, relative either to the tool or to the base.

    Exactly one of the camera and the target moves with the arm, and which one
    is the difference between the two calibration types:

    - ``eye_in_hand``: the camera rides on the tool (`on_tool=True`, mount =
      tool -> camera) and the board is fixed in the workspace.
    - ``eye_to_hand``: the camera is fixed in the workspace (`on_tool=False`,
      mount = base -> camera) and the board rides on the tool."""
    R: np.ndarray
    t: np.ndarray
    on_tool: bool


@dataclass
class CameraModel(Mounting):
    """Pinhole intrinsics plus the camera's mounting (see `Mounting`)"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class TargetModel(Mounting):
    """Where the board is (see `Mounting`) and how big it is"""
    size_m: tuple = (0.175, 0.245)
    origin_at_center: bool = False
    #: Which face of the board carries the markers
    normal_sign: float = 0.0

    def corner_points(self) -> np.ndarray:
        """(4, 3) board corners in the target frame, counter-clockwise."""
        w, h = self.size_m
        if self.origin_at_center:
            x0, x1, y0, y1 = -0.5 * w, 0.5 * w, -0.5 * h, 0.5 * h
        else:
            x0, x1, y0, y1 = 0.0, w, 0.0, h
        return np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])


def locate_target(eye_in_hand: bool,
                  R_base_tool: np.ndarray, t_base_tool: np.ndarray,
                  R_mount_cam: np.ndarray, t_mount_cam: np.ndarray,
                  R_cam_target: np.ndarray, t_cam_target: np.ndarray,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Locate the board from a single detection taken at a known configuration"""
    if eye_in_hand:
        R_base_cam, t_base_cam = geometry.compose_rt(
            R_base_tool, t_base_tool, R_mount_cam, t_mount_cam)
        return geometry.compose_rt(R_base_cam, t_base_cam, R_cam_target, t_cam_target)

    R_tool_base, t_tool_base = R_base_tool.T, -R_base_tool.T @ t_base_tool
    R_tool_cam, t_tool_cam = geometry.compose_rt(
        R_tool_base, t_tool_base, R_mount_cam, t_mount_cam)
    return geometry.compose_rt(R_tool_cam, t_tool_cam, R_cam_target, t_cam_target)


@dataclass
class VisibilityLimits:
    """Thresholds for `visible_mask`"""
    min_range_m: float = 0.10
    max_range_m: float = 1.20
    max_view_angle_deg: float = 65.0
    edge_margin_px: float = 20.0
    #: fraction of the four board corners that must project into the image
    min_corner_fraction: float = 0.75


@dataclass
class SamplingStats:
    requested: int = 0
    drawn: int = 0
    #: configurations the mask passed
    visible: int = 0
    kept: int = 0
    exhausted: bool = False

    @property
    def yield_fraction(self) -> float:
        return self.visible / self.drawn if self.drawn else 0.0

    def summary(self) -> str:
        return (f'{self.kept}/{self.requested} visible configs from {self.drawn} draws '
                f'({100.0 * self.yield_fraction:.1f}% yield)'
                + ('; draw budget exhausted' if self.exhausted else ''))


def visible_mask(thetas: np.ndarray, joint_frames: list, camera: CameraModel,
                 target: TargetModel,
                 limits: VisibilityLimits | None = None) -> np.ndarray:
    limits = limits or VisibilityLimits()
    if camera.on_tool == target.on_tool:
        raise ValueError(
            'exactly one of the camera and the target must be mounted on the tool '
            f'(camera.on_tool={camera.on_tool}, target.on_tool={target.on_tool}); '
            'eye_in_hand puts the camera there, eye_to_hand the target')

    thetas = np.asarray(thetas, dtype=float)
    if thetas.ndim == 1:
        thetas = thetas[None, :]
    if thetas.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    R_base_tool, t_base_tool = kinematic_model.forward_kinematics_batch(
        thetas, joint_frames)
    n = thetas.shape[0]

    def _place(mounting):
        """base -> frame for every candidate"""
        if not mounting.on_tool:
            return (np.broadcast_to(mounting.R, (n, 3, 3)),
                    np.broadcast_to(mounting.t, (n, 3)))
        return (R_base_tool @ mounting.R,
                np.einsum('nij,j->ni', R_base_tool, mounting.t) + t_base_tool)

    R_base_cam, t_base_cam = _place(camera)
    R_base_target, t_base_target = _place(target)

    # Board corners + centre, in the base frame.
    corners_target = target.corner_points()
    points_target = np.vstack(
        [corners_target, corners_target.mean(axis=0)])   # (5, 3)
    points_base = (np.einsum('nij,kj->nki', R_base_target, points_target)
                   # (N, 5, 3)
                   + t_base_target[:, None, :])

    # Into each candidate's camera frame: p_cam = R_base_cam^T (p_base - t_base_cam)
    # (N, 5, 3)
    delta = points_base - t_base_cam[:, None, :]
    points_cam = np.einsum('nji,nkj->nki', R_base_cam,
                           delta)                  # (N, 5, 3)

    z = points_cam[..., 2]
    in_front = z > 1e-6
    # avoid /0
    safe_z = np.where(in_front, z, 1.0)
    u = camera.fx * points_cam[..., 0] / safe_z + camera.cx
    v = camera.fy * points_cam[..., 1] / safe_z + camera.cy

    m = limits.edge_margin_px
    in_image = (in_front
                & (u >= m) & (u <= camera.width - m)
                & (v >= m) & (v <= camera.height - m))

    centre_cam = points_cam[:, 4, :]
    centre_range = np.linalg.norm(centre_cam, axis=1)
    range_ok = ((centre_range >= limits.min_range_m)
                & (centre_range <= limits.max_range_m))

    corner_fraction = in_image[:, :4].mean(axis=1)
    framing_ok = in_image[:, 4] & (
        corner_fraction >= limits.min_corner_fraction)

    normal_base = R_base_target[:, :, 2]
    normal_cam = np.einsum('nji,nj->ni', R_base_cam, normal_base)
    view_dir = centre_cam / np.maximum(centre_range, 1e-9)[:, None]
    cos_incidence = np.einsum('ni,ni->n', normal_cam, view_dir)
    cos_incidence = (target.normal_sign * cos_incidence if target.normal_sign
                     else np.abs(cos_incidence))
    angle_ok = cos_incidence >= np.cos(np.radians(limits.max_view_angle_deg))

    return range_ok & framing_ok & angle_ok


def infer_normal_sign(R_cam_target: np.ndarray, t_cam_target: np.ndarray) -> float:
    """Learn which board face carries the markers, from one real detection"""
    distance = float(np.linalg.norm(t_cam_target))
    if distance < 1e-9:
        return 1.0
    cos = float(np.dot(np.asarray(R_cam_target)[
                :, 2], np.asarray(t_cam_target) / distance))
    return -1.0 if cos < 0 else 1.0


def sample_visible_joint_configs(
        joint_frames: list, n: int, camera: CameraModel, target: TargetModel,
        limits: VisibilityLimits | None = None,
        margin_deg: float = 5.0,
        rng: np.random.Generator | None = None,
        max_draws: int = 200_000,
        chunk_size: int = 4096) -> tuple[list, SamplingStats]:
    """Rejection-samples n joint configurations from which the board is predicted visible"""
    rng = rng or np.random.default_rng()
    lo, hi = sampling_bounds(joint_frames, np.radians(margin_deg))

    stats = SamplingStats(requested=n)
    kept: list = []
    while len(kept) < n and stats.drawn < max_draws:
        size = min(chunk_size, max_draws - stats.drawn)
        batch = _draw_batch(lo, hi, size, rng)
        stats.drawn += size

        accepted = batch[visible_mask(
            batch, joint_frames, camera, target, limits)]
        stats.visible += len(accepted)
        for theta in accepted[:n - len(kept)]:
            kept.append(theta)

    stats.kept = len(kept)
    stats.exhausted = len(kept) < n
    return kept, stats


def observed_ranges_deg(thetas: np.ndarray) -> np.ndarray:
    """thetas: (N, 6) collected joint angles -> (6,) per-joint observed range
    (max - min) in degrees, used to gate dataset acceptance."""
    return np.degrees(thetas.max(axis=0) - thetas.min(axis=0))
