"""Inject a *known* kinematic error into a URDF, for simulation ground truth.

This is the inverse of what WP4 does. `kinematic_solver` estimates per-joint
corrections from measurements; this module takes corrections that are *given*
and bakes them into a URDF, so a simulator can be driven by a robot whose real
kinematics differ from the nominal model by an exactly-known amount.

The point is that in Gazebo the *spawned* model carries the perturbation while
``robot_state_publisher``/MoveIt/every calibration node keep the nominal
description. WP4 then has something real to identify, and every recovered
parameter can be compared against its true value - which physical hardware can
never provide.

**The parameterization is deliberately identical to `kinematic_model`.** Each
joint's nominal frame is ``T0 = Trans(xyz0) @ Rot(rpy0)``, and the correction
``dT = Trans(delta_t) @ Rot(delta_r)`` is inserted as ``T0 @ dT @ Rot(axis,
theta)`` with ``delta_t``/``delta_r`` restricted to the 2D subspace orthogonal
to the joint's own axis (see that module's docstring for why). A URDF joint
renders as ``T0' @ Rot(axis, theta)``, so writing back ``T0' = T0 @ dT`` makes
the two algebraically identical - not approximately, exactly.

Two consequences worth stating explicitly:

- ``<axis>`` is expressed in the child link's frame and ``dT`` is applied
  *before* ``Rot(axis, theta)``, so the axis vector is unchanged and must not
  be touched.
- The ground-truth correction vector is therefore *literally* the numbers in
  the perturbation YAML - no transformation is needed to compare them against
  `kinematic_result.yaml`'s ``corrections``.

`orthogonal_basis` is reused from `kinematic_model` rather than reimplemented,
and that is load-bearing: the basis is an arbitrary-but-deterministic choice,
so if the injector and the solver ever computed it differently, every
recovered-vs-true comparison would be silently rotated within each joint's
plane with nothing to flag it.
"""
import numpy as np
import yaml
from xml.etree import ElementTree

from annin_ar4_calibration.core import kinematic_model

#: the arm joints a perturbation applies to, in chain order
DEFAULT_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

#: top-level key of a perturbation YAML file
ROOT_KEY = 'kinematic_perturbation'


def load(path: str, joint_names: list = None) -> np.ndarray:
    """Read a perturbation YAML -> (N, 4) corrections array.

    The column layout is `kinematic_model`'s: per joint,
    ``[delta_t_perp[0], delta_t_perp[1], delta_r_perp[0], delta_r_perp[1]]``,
    coefficients in that joint's own `orthogonal_basis`. Rotations are
    returned in radians regardless of the file's declared units.

    Joints absent from the file are left at zero, so a file listing only
    ``joint_3`` is a valid single-joint perturbation.
    """
    joint_names = joint_names or DEFAULT_JOINT_NAMES
    with open(path, 'r') as f:
        document = yaml.safe_load(f) or {}

    spec = document.get(ROOT_KEY)
    if spec is None:
        raise ValueError(f"{path}: missing top-level '{ROOT_KEY}' key")

    units = spec.get('units', {})
    rotation_in_degrees = str(units.get('rotation', 'rad')).startswith('deg')
    joints = spec.get('joints') or {}

    unknown = set(joints) - set(joint_names)
    if unknown:
        raise ValueError(
            f"{path}: unknown joint name(s) {sorted(unknown)}; expected a subset of "
            f"{joint_names}. A typo here would silently perturb nothing.")

    corrections = np.zeros((len(joint_names), kinematic_model.PARAMS_PER_JOINT))
    for i, name in enumerate(joint_names):
        entry = joints.get(name) or {}
        corrections[i, 0:2] = np.asarray(entry.get('delta_t_perp', [0.0, 0.0]), dtype=float)
        delta_r = np.asarray(entry.get('delta_r_perp', [0.0, 0.0]), dtype=float)
        corrections[i, 2:4] = np.radians(delta_r) if rotation_in_degrees else delta_r
    return corrections


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> URDF (roll, pitch, yaw), radians.

    Exact inverse of `kinematic_model._rpy_to_matrix`, i.e. the fixed-axis
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` convention URDF uses.
    """
    cos_pitch = np.hypot(R[0, 0], R[1, 0])
    if cos_pitch < 1e-9:
        # Gimbal lock (pitch = +/-pi/2): only (roll - yaw) is determined, so
        # yaw is pinned to 0 and folded into roll. Reconstructs R exactly.
        return np.array([
            np.arctan2(-R[1, 2], R[1, 1]),
            np.arctan2(-R[2, 0], cos_pitch),
            0.0,
        ])
    return np.array([
        np.arctan2(R[2, 1], R[2, 2]),
        np.arctan2(-R[2, 0], cos_pitch),
        np.arctan2(R[1, 0], R[0, 0]),
    ])


def perturbed_origin(
        xyz0: np.ndarray, rpy0: np.ndarray, axis: np.ndarray, params: np.ndarray,
) -> tuple:
    """One joint's nominal origin + its 4 correction coefficients -> the
    perturbed origin ``(xyz, rpy)`` such that ``Trans(xyz) @ Rot(rpy) ==
    Trans(xyz0) @ Rot(rpy0) @ dT``."""
    R0 = kinematic_model._rpy_to_matrix(np.asarray(rpy0, dtype=float))
    basis = kinematic_model.orthogonal_basis(np.asarray(axis, dtype=float))
    dR, dt = kinematic_model.correction_transform(params[0:2], params[2:4], basis)
    return R0 @ dt + np.asarray(xyz0, dtype=float), matrix_to_rpy(R0 @ dR)


def apply_to_urdf(
        urdf_xml: str, corrections: np.ndarray, joint_names: list = None,
        tf_prefix: str = '') -> str:
    """Rewrite the given joints' ``<origin>`` in an (already xacro-expanded)
    URDF so the robot it describes has the given kinematic error. Everything
    else - joint limits, axes, links, ros2_control blocks, gazebo plugins - is
    passed through untouched.

    Note ``ElementTree`` drops XML comments on the way out. The result is
    machine-consumed (spawned into Gazebo), so this is cosmetic only.
    """
    joint_names = joint_names or DEFAULT_JOINT_NAMES
    root = ElementTree.fromstring(urdf_xml)
    joints_by_name = {j.get('name'): j for j in root.findall('joint')}

    for i, name in enumerate(joint_names):
        prefixed = tf_prefix + name
        joint = joints_by_name.get(prefixed)
        if joint is None:
            raise ValueError(f"URDF has no joint named '{prefixed}'")

        origin = joint.find('origin')
        if origin is None:
            origin = ElementTree.SubElement(joint, 'origin')
        xyz0 = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        rpy0 = [float(v) for v in origin.get('rpy', '0 0 0').split()]

        axis_elem = joint.find('axis')
        axis = [float(v) for v in (
            axis_elem.get('xyz', '0 0 1') if axis_elem is not None else '0 0 1').split()]

        xyz, rpy = perturbed_origin(xyz0, rpy0, axis, corrections[i])
        origin.set('xyz', ' '.join(f'{v:.12g}' for v in xyz))
        origin.set('rpy', ' '.join(f'{v:.12g}' for v in rpy))

    return ElementTree.tostring(root, encoding='unicode')


def to_yaml_dict(corrections: np.ndarray, joint_names: list = None, **metadata) -> dict:
    """Corrections array -> the YAML structure `load` reads back. Used by the
    generator script and by the ground-truth dump."""
    joint_names = joint_names or DEFAULT_JOINT_NAMES
    return {
        ROOT_KEY: {
            'units': {'translation': 'm', 'rotation': 'rad'},
            **metadata,
            'joints': {
                name: {
                    'delta_t_perp': [float(v) for v in corrections[i, 0:2]],
                    'delta_r_perp': [float(v) for v in corrections[i, 2:4]],
                }
                for i, name in enumerate(joint_names)
            },
        }
    }
