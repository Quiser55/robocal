"""forward-kinematics model with a small correction parameterization per joint"""
from dataclasses import dataclass

import numpy as np
from xml.etree import ElementTree

from annin_ar4_calibration.core import geometry

#: number of free correction parameters per joint (2 translation + 2 rotation)
PARAMS_PER_JOINT = 4


@dataclass
class JointFrame:
    xyz0: np.ndarray  # (3,) nominal origin translation
    rpy0: np.ndarray  # (3,) nominal origin rpy (roll, pitch, yaw), radians
    axis: np.ndarray  # (3,) unit rotation axis, in the joint's own local frame
    lower: float
    upper: float
    name: str


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def parse_urdf_chain(urdf_xml: str, joint_names: list) -> list:
    """Parse ``<joint>`` origin/axis/limit for the given joint names out of an
    (already xacro-expanded) URDF XML string. Returns a list of JointFrame in
    the order of ``joint_names``.
    """
    root = ElementTree.fromstring(urdf_xml)
    joints_by_name = {j.get('name'): j for j in root.findall('joint')}

    frames = []
    for name in joint_names:
        joint = joints_by_name.get(name)
        if joint is None:
            raise ValueError(f"URDF has no joint named '{name}'")

        origin = joint.find('origin')
        if origin is not None:
            xyz0 = np.array([float(v)
                            for v in origin.get('xyz', '0 0 0').split()])
            rpy0 = np.array([float(v)
                            for v in origin.get('rpy', '0 0 0').split()])
        else:
            xyz0 = np.zeros(3)
            rpy0 = np.zeros(3)

        axis_elem = joint.find('axis')
        if axis_elem is not None:
            axis = np.array([float(v)
                            for v in axis_elem.get('xyz', '0 0 1').split()])
        else:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)

        limit = joint.find('limit')
        lower = float(limit.get('lower')) if limit is not None else -np.pi
        upper = float(limit.get('upper')) if limit is not None else np.pi

        frames.append(JointFrame(
            xyz0=xyz0, rpy0=rpy0, axis=axis, lower=lower, upper=upper, name=name))
    return frames


def orthogonal_basis(axis: np.ndarray) -> np.ndarray:
    """Return a (2,3) array of two unit vectors orthogonal to `axis` and to
    each other, spanning the plane perpendicular to it."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(
        axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.stack([u, v])


def correction_transform(
        delta_t_perp: np.ndarray, delta_r_perp: np.ndarray, basis: np.ndarray) -> tuple:
    """(2,)/(2,) correction coefficients in `basis` -> (R, t) of dT."""
    t = delta_t_perp[0] * basis[0] + delta_t_perp[1] * basis[1]
    rotvec = delta_r_perp[0] * basis[0] + delta_r_perp[1] * basis[1]
    R = geometry.rotvec_to_rotation_matrix(rotvec)
    return R, t


def forward_kinematics(
        theta: np.ndarray, joint_frames: list, corrections: np.ndarray = None) -> tuple:
    """theta: (6,) joint angles. corrections: (6, 4) per-joint
    [delta_tx_perp, delta_ty_perp, delta_rx_perp, delta_ry_perp] coefficients
    in each joint's own orthogonal basis  or None forthe nominal model. 
    Returns (R, t) of base_frame -> the frame at the end of the chain"""
    R_acc = np.eye(3)
    t_acc = np.zeros(3)
    for i, jf in enumerate(joint_frames):
        R0 = _rpy_to_matrix(jf.rpy0)
        t0 = jf.xyz0

        if corrections is not None:
            basis = orthogonal_basis(jf.axis)
            dR, dt = correction_transform(
                corrections[i, 0:2], corrections[i, 2:4], basis)
        else:
            dR, dt = np.eye(3), np.zeros(3)

        R_joint = geometry.axis_angle_to_rotation_matrix(jf.axis, theta[i])

        R_step = R0 @ dR @ R_joint
        t_step = R0 @ dt + t0

        R_acc, t_acc = geometry.compose_rt(R_acc, t_acc, R_step, t_step)
    return R_acc, t_acc


def _axis_angle_fixed_axis_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: (N,) angles -> (N,3,3) rotation matrices."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    K2 = K @ K
    sin_t = np.sin(angles)
    cos_t = 1.0 - np.cos(angles)
    return (np.eye(3)[None, :, :] + sin_t[:, None, None] * K[None, :, :]
            + cos_t[:, None, None] * K2[None, :, :])


def forward_kinematics_batch(
        thetas: np.ndarray, joint_frames: list, corrections: np.ndarray = None) -> tuple:
    """Vectorized forward_kinematics over a batch of joint-angle samples"""
    N = thetas.shape[0]
    R_acc = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    t_acc = np.zeros((N, 3))

    for i, jf in enumerate(joint_frames):
        R0 = _rpy_to_matrix(jf.rpy0)
        t0 = jf.xyz0

        if corrections is not None:
            basis = orthogonal_basis(jf.axis)
            dR, dt = correction_transform(
                corrections[i, 0:2], corrections[i, 2:4], basis)
        else:
            dR, dt = np.eye(3), np.zeros(3)

        R0dR = R0 @ dR
        R_joint = _axis_angle_fixed_axis_batch(jf.axis, thetas[:, i])
        R_step = np.einsum('ij,njk->nik', R0dR, R_joint)
        t_step = np.broadcast_to(R0 @ dt + t0, (N, 3))

        R_acc, t_acc = geometry.batch_compose_rt(R_acc, t_acc, R_step, t_step)
    return R_acc, t_acc


def axis_alignment_diagnostic(joint_frames: list) -> list:
    """Pairwise |dot(world_axis_i, world_axis_{i+1})| for consecutive joints"""
    world_axes = []
    R_acc = np.eye(3)
    for jf in joint_frames:
        R_acc = R_acc @ _rpy_to_matrix(jf.rpy0)
        world_axes.append(R_acc @ jf.axis)
        # Rot(axis, theta=0) is the identity, so it doesn't affect R_acc here,
        # but at nonzero theta later joints' world axes would depend on it -
        # this diagnostic is deliberately evaluated at the nominal theta=0
        # pose only (parallel-axis pairs stay parallel at every theta anyway,
        # by construction: a rotation about a shared axis doesn't change it).

    return [
        float(abs(np.dot(world_axes[i], world_axes[i + 1])))
        for i in range(len(world_axes) - 1)
    ]
