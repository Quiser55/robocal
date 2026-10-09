"""Round-trip tests for core/perturbation.py - the simulation ground-truth path"""
import math
import textwrap

import numpy as np
import pytest

from annin_ar4_calibration.core import kinematic_model, perturbation

JOINT_NAMES = perturbation.DEFAULT_JOINT_NAMES

# Nominal joint origins/axes matching annin_ar4_description's ar_macro.xacro
# (joint_1..joint_6). Same values as test_kinematic_calibration.py's fixture.
AR4_SPECS = [
    ([0, 0, 0.092], [math.pi, 0, 0], [0, 0, 1]),
    ([0, 0.06415, -0.07778], [1.5708, 0, -1.5708], [0, 0, -1]),
    ([0, -0.305, 0], [0, 0, math.pi], [0, 0, -1]),
    ([0, 0, 0], [1.5708, 0, -1.5708], [0, 0, -1]),
    ([0, 0, -0.22294], [math.pi, 0, -1.5708], [1, 0, 0]),
    ([0, 0, 0.041], [0, 0, 0], [0, 0, 1]),
]


def _nominal_urdf(tf_prefix: str = '') -> str:
    joints = '\n'.join(
        textwrap.dedent(f"""\
          <joint name="{tf_prefix}{name}" type="revolute">
            <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
            <parent link="{tf_prefix}link_{i}"/>
            <child link="{tf_prefix}link_{i + 1}"/>
            <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
          </joint>""")
        for i, (name, (xyz, rpy, axis)) in enumerate(zip(JOINT_NAMES, AR4_SPECS))
    )
    links = '\n'.join(
        f'  <link name="{tf_prefix}link_{i}"/>' for i in range(len(AR4_SPECS) + 1))
    return f'<robot name="ar4">\n{links}\n{joints}\n</robot>'


def _random_corrections(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corrections = rng.uniform(-1, 1,
                              size=(6, kinematic_model.PARAMS_PER_JOINT))
    # translation coefficients ~ +/- 2 mm
    corrections[:, 0:2] *= 0.002
    # rotation coefficients ~ +/- 0.15 deg
    corrections[:, 2:4] *= math.radians(0.15)
    return corrections


def test_matrix_to_rpy_inverts_rpy_to_matrix():
    rng = np.random.default_rng(1)
    for rpy in rng.uniform(-math.pi, math.pi, size=(200, 3)):
        R = kinematic_model._rpy_to_matrix(rpy)
        assert np.allclose(kinematic_model._rpy_to_matrix(perturbation.matrix_to_rpy(R)), R,
                           atol=1e-12)


def test_matrix_to_rpy_at_gimbal_lock():
    '''pitch = +/-pi/2 leaves only (roll - yaw) determined; the recovered rpy
    must still reconstruct the same matrix.'''
    for pitch in (math.pi / 2, -math.pi / 2):
        for roll, yaw in ((0.3, 0.0), (0.0, 0.7), (-1.1, 2.0)):
            R = kinematic_model._rpy_to_matrix(np.array([roll, pitch, yaw]))
            assert np.allclose(
                kinematic_model._rpy_to_matrix(perturbation.matrix_to_rpy(R)), R, atol=1e-9)


def test_urdf_roundtrip_matches_correction_model():
    """The property the whole ground-truth story rests on."""
    nominal = _nominal_urdf()
    corrections = _random_corrections()
    perturbed = perturbation.apply_to_urdf(nominal, corrections)

    nominal_frames = kinematic_model.parse_urdf_chain(nominal, JOINT_NAMES)
    perturbed_frames = kinematic_model.parse_urdf_chain(perturbed, JOINT_NAMES)

    rng = np.random.default_rng(11)
    for theta in rng.uniform(-2.0, 2.0, size=(50, 6)):
        R_baked, t_baked = kinematic_model.forward_kinematics(
            theta, perturbed_frames, None)
        R_model, t_model = kinematic_model.forward_kinematics(
            theta, nominal_frames, corrections)
        assert np.allclose(t_baked, t_model, atol=1e-12)
        assert np.allclose(R_baked, R_model, atol=1e-12)


def test_zero_corrections_leave_kinematics_unchanged():
    nominal = _nominal_urdf()
    perturbed = perturbation.apply_to_urdf(nominal, np.zeros((6, 4)))

    nominal_frames = kinematic_model.parse_urdf_chain(nominal, JOINT_NAMES)
    perturbed_frames = kinematic_model.parse_urdf_chain(perturbed, JOINT_NAMES)
    for a, b in zip(nominal_frames, perturbed_frames):
        assert np.allclose(a.xyz0, b.xyz0, atol=1e-12)
        # rpy is re-derived through a matrix round trip, so compare the matrices
        assert np.allclose(kinematic_model._rpy_to_matrix(a.rpy0),
                           kinematic_model._rpy_to_matrix(b.rpy0), atol=1e-12)


def test_apply_preserves_axis_and_limits():
    nominal = _nominal_urdf()
    perturbed = perturbation.apply_to_urdf(nominal, _random_corrections())

    nominal_frames = kinematic_model.parse_urdf_chain(nominal, JOINT_NAMES)
    perturbed_frames = kinematic_model.parse_urdf_chain(perturbed, JOINT_NAMES)
    for a, b in zip(nominal_frames, perturbed_frames):
        assert np.allclose(a.axis, b.axis, atol=1e-15)
        assert a.lower == b.lower and a.upper == b.upper


def test_apply_honours_tf_prefix():
    nominal = _nominal_urdf(tf_prefix='left_')
    corrections = _random_corrections()
    perturbed = perturbation.apply_to_urdf(
        nominal, corrections, tf_prefix='left_')

    prefixed = [f'left_{name}' for name in JOINT_NAMES]
    nominal_frames = kinematic_model.parse_urdf_chain(nominal, prefixed)
    perturbed_frames = kinematic_model.parse_urdf_chain(perturbed, prefixed)

    theta = np.array([0.1, -0.4, 0.7, 1.1, -0.2, 0.5])
    R_baked, t_baked = kinematic_model.forward_kinematics(
        theta, perturbed_frames, None)
    R_model, t_model = kinematic_model.forward_kinematics(
        theta, nominal_frames, corrections)
    assert np.allclose(t_baked, t_model, atol=1e-12)
    assert np.allclose(R_baked, R_model, atol=1e-12)


def test_apply_rejects_missing_joint():
    with pytest.raises(ValueError, match='no joint named'):
        perturbation.apply_to_urdf('<robot name="empty"/>', np.zeros((6, 4)))


def test_load_roundtrips_through_to_yaml_dict(tmp_path):
    import yaml
    corrections = _random_corrections()
    path = tmp_path / 'p.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(perturbation.to_yaml_dict(
            corrections, generated_by='test'), f)
    assert np.allclose(perturbation.load(str(path)), corrections, atol=1e-15)


def test_load_accepts_degrees_and_partial_specs(tmp_path):
    path = tmp_path / 'deg.yaml'
    path.write_text(textwrap.dedent("""\
        kinematic_perturbation:
          units: {translation: m, rotation: deg}
          joints:
            joint_3: {delta_t_perp: [0.001, 0.0], delta_r_perp: [0.1, 0.0]}
        """))
    corrections = perturbation.load(str(path))
    assert corrections.shape == (6, 4)
    assert np.allclose(corrections[2], [0.001, 0.0, math.radians(0.1), 0.0])
    assert np.allclose(np.delete(corrections, 2, axis=0), 0.0)


def test_load_rejects_unknown_joint_name(tmp_path):
    path = tmp_path / 'typo.yaml'
    path.write_text(
        'kinematic_perturbation:\n  joints:\n    joint_7: {delta_t_perp: [1, 0]}\n')
    with pytest.raises(ValueError, match='unknown joint name'):
        perturbation.load(str(path))
