"""Pure-numpy geometry helpers for rigid transforms, rotation matrices, quaternions, and rotation vectors."""
import math

import numpy as np
from geometry_msgs.msg import Transform, TransformStamped


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a quaternion (x, y, z, w order)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 *
         (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)],
    ])


def transform_to_rt(transform: Transform) -> tuple[np.ndarray, np.ndarray]:
    """Transform -> (R (3,3), t (3,))."""
    q = transform.rotation
    t = transform.translation
    R = quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
    return R, np.array([t.x, t.y, t.z])


def transform_stamped_to_rt(ts: TransformStamped) -> tuple[np.ndarray, np.ndarray]:
    return transform_to_rt(ts.transform)


def rt_to_transform(t: np.ndarray, R: np.ndarray | None = None) -> Transform:
    """Build a Transform msg from a translation (and optional rotation, default identity)."""
    transform = Transform()
    transform.translation.x = float(t[0])
    transform.translation.y = float(t[1])
    transform.translation.z = float(t[2])
    if R is None:
        transform.rotation.x = 0.0
        transform.rotation.y = 0.0
        transform.rotation.z = 0.0
        transform.rotation.w = 1.0
    else:
        transform.rotation = _rotation_matrix_to_quat_msg(R)
    return transform


def _rotation_matrix_to_quat_msg(R: np.ndarray):
    from geometry_msgs.msg import Quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = Quaternion()
    q.x, q.y, q.z, q.w = x, y, z, w
    return q


def batch_samples_to_rt(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,7) rows [tx,ty,tz,qx,qy,qz,qw] -> (R (N,3,3), t (N,3))."""
    t = samples[:, 0:3]
    q = samples[:, 3:7]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    qx, qy, qz, qw = (q / n).T

    N = samples.shape[0]
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R, t


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angle in degrees between two unit quaternions (x,y,z,w)."""
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return math.degrees(2.0 * math.acos(dot))


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion, (x, y, z, w) order."""
    q = _rotation_matrix_to_quat_msg(R)
    return np.array([q.x, q.y, q.z, q.w])


def batch_invert_rt(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched rigid-transform inverse. R: (N,3,3), t: (N,3) -> (R_inv, t_inv)."""
    R_inv = R.transpose(0, 2, 1)
    t_inv = -np.einsum('nij,nj->ni', R_inv, t)
    return R_inv, t_inv


def compose_rt(
        R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-transform composition: applies transform 2 then transform 1"""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula. axis need not be pre-normalized."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (axis * angle, radians), via
    the matrix logarithm. Exact (not a small-angle approximation)."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    if angle > math.pi - 1e-8:
        B = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(B)))
        axis = B[k, :] / math.sqrt(max(B[k, k], 1e-12))
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * math.sin(angle))
    return axis * angle


def batch_compose_rt(
        R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Like compose_rt, but broadcasts over a leading batch dimension: any of
    R1/t1/R2/t2 may be (3,3)/(3,) (a single fixed transform) or (N,3,3)/(N,3)
    (per-sample), following standard numpy broadcasting."""
    R = np.einsum('...ij,...jk->...ik', R1, R2)
    t = np.einsum('...ij,...j->...i', R1, t2) + t1
    return R, t


def batch_rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation_matrix_to_rotvec over a batch (N,3,3) -> (N,3).
    Optimized for small-to-moderate angles"""
    trace = np.trace(R, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    small = angle < 1e-8
    # avoid /0; masked out below
    sin_angle = np.where(small, 1.0, np.sin(angle))
    axis_unnorm = np.stack([
        R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] -
        R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1],
    ], axis=1)
    rotvec = (axis_unnorm / (2.0 * sin_angle)[:, None]) * angle[:, None]
    rotvec[small] = 0.0

    near_pi = angle > (math.pi - 1e-8)
    for idx in np.nonzero(near_pi)[0]:
        rotvec[idx] = rotation_matrix_to_rotvec(R[idx])
    return rotvec


def mean_rotation_markley(quats: np.ndarray) -> np.ndarray:
    """Markley's quaternion-averaging method (eigenvector of the mean outer
    product matrix)"""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    A = (q.T @ q) / q.shape[0]
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    return quat_to_rotation_matrix(*q_mean)


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, radians) -> 3x3 rotation matrix."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    return axis_angle_to_rotation_matrix(rotvec / angle, angle)
