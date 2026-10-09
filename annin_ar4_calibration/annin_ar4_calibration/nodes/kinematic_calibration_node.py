"""Solves and persists kinematic parameter corrections (WP4), using the WP3
hand-eye calibration result as the trusted external measurement device"""
import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from annin_ar4_calibration.core import (
    geometry, joint_sampling, kinematic_model, kinematic_solver, paths, storage)
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description

RESULT_KEY = 'kinematic_calibration'
REQUIRED_KEYS = ('corrections', 'mount_offset', 'calibration_type')
HAND_EYE_KEY = 'hand_eye_calibration'
HAND_EYE_REQUIRED_KEYS = ('transform', 'parent_frame', 'child_frame')


class KinematicCalibrationNode(Node):

    def __init__(self):
        super().__init__('kinematic_calibration')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('calibration_type', 'eye_to_hand')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('hand_eye_result_file', '')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')
        self.declare_parameter('min_samples', 80)
        self.declare_parameter('min_joint_range_deg', 30.0)
        # 'sequential' | 'joint' | 'both'
        self.declare_parameter('method', 'both')
        self.declare_parameter('ridge_lambda', 1e-6)
        self.declare_parameter('bounds.translation_m', 0.01)
        self.declare_parameter('bounds.rotation_deg', 3.0)
        self.declare_parameter('run_lm_diagnostic', True)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.calibration_type = self.get_parameter('calibration_type').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.hand_eye_result_file = paths.resolve(
            self.get_parameter('hand_eye_result_file').value, paths.default_hand_eye_result_file)
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_kinematic_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_kinematic_result_file)
        self.min_samples = self.get_parameter('min_samples').value
        self.min_joint_range_deg = self.get_parameter(
            'min_joint_range_deg').value
        self.method = self.get_parameter('method').value
        self.ridge_lambda = self.get_parameter('ridge_lambda').value
        self.bounds_translation_m = self.get_parameter(
            'bounds.translation_m').value
        self.bounds_rotation_deg = self.get_parameter(
            'bounds.rotation_deg').value
        self.run_lm_diagnostic = self.get_parameter('run_lm_diagnostic').value

        urdf_xml = fetch_robot_description(self)
        self.joint_frames = kinematic_model.parse_urdf_chain(
            urdf_xml, self.arm_joint_names)
        axis_alignment = kinematic_model.axis_alignment_diagnostic(
            self.joint_frames)
        for i, dot in enumerate(axis_alignment):
            if dot > 0.98:
                self.get_logger().warning(
                    f'joint_{i + 1}/joint_{i + 2} axes are near-parallel '
                    f'(|dot|={dot:.4f}) - expect a poorly-conditioned direction there, '
                    f'mitigated by ridge regularization/bounds, not corrected exactly.')

        self.create_service(Trigger, '~/compute_calibration', self._compute_cb)

        self._log_existing_result()

        self.get_logger().info(
            f'Kinematic calibration ready: calibration_type={self.calibration_type}, '
            f'sample_file={self.sample_file}, result_file={self.result_file}')

    def _fix_joint1(self) -> bool:
        return self.calibration_type == 'eye_in_hand'

    def _log_existing_result(self) -> None:
        try:
            result = storage.load_result(
                self.result_file, key=RESULT_KEY, required_keys=REQUIRED_KEYS)
        except storage.ResultFileError as exc:
            self.get_logger().warning(f'Ignoring existing result file: {exc}')
            return
        if result is None:
            return
        self.get_logger().info(
            f'Loaded existing kinematic result from {self.result_file} '
            f'(primary_method={result.get("primary_method")})')

    def _load_hand_eye_result(self):
        result = storage.load_result(
            self.hand_eye_result_file, key=HAND_EYE_KEY, required_keys=HAND_EYE_REQUIRED_KEYS)
        if result is None:
            raise storage.ResultFileError(
                f'No hand-eye calibration result at {self.hand_eye_result_file} - run '
                f'hand_eye.launch.py and compute_calibration first (WP4 depends on WP3).')
        if result.get('calibration_type') != self.calibration_type:
            raise storage.ResultFileError(
                f'calibration_type mismatch: hand_eye_result.yaml was computed with '
                f'{result.get("calibration_type")!r}, but this node is configured with '
                f'{self.calibration_type!r}. Relaunch with '
                f'calibration_type:={result.get("calibration_type")}, or recompute '
                f'the hand-eye calibration for {self.calibration_type!r}.')
        transform = result['transform']
        known_R = geometry.quat_to_rotation_matrix(
            transform['rotation']['x'], transform['rotation']['y'],
            transform['rotation']['z'], transform['rotation']['w'])
        known_t = np.array([
            transform['translation']['x'], transform['translation']['y'],
            transform['translation']['z'],
        ])
        return known_R, known_t

    def _result_dict(self, result: kinematic_solver.KinematicCalibrationResult) -> dict:
        mount_quat = geometry.rotation_matrix_to_quat(result.mount_R)
        return {
            'method': result.method,
            'rms_before_m': result.rms_before_m,
            'rms_after_m': result.rms_after_m,
            'converged': result.converged,
            'iterations': result.iterations,
            'num_samples': result.num_samples,
            'fixed_joint1': result.fixed_joint1,
            'condition_number': result.condition_number,
            'singular_values': [float(v) for v in result.singular_values],
            'corrections': {
                self.arm_joint_names[i]: {
                    'delta_t_perp': [float(v) for v in result.corrections[i, 0:2]],
                    'delta_r_perp': [float(v) for v in result.corrections[i, 2:4]],
                }
                for i in range(6)
            },
            'mount_offset': {
                'role': result.mount_role,
                'translation': {
                    'x': float(result.mount_t[0]), 'y': float(result.mount_t[1]),
                    'z': float(result.mount_t[2]),
                },
                'rotation': {
                    'x': float(mount_quat[0]), 'y': float(mount_quat[1]),
                    'z': float(mount_quat[2]), 'w': float(mount_quat[3]),
                },
            },
        }

    def _compute_cb(self, request, response):
        try:
            known_R, known_t = self._load_hand_eye_result()
        except storage.ResultFileError as exc:
            response.success = False
            response.message = str(exc)
            return response

        try:
            samples = storage.load_samples(self.sample_file, ncols=13)
        except storage.SampleFileError as exc:
            response.success = False
            response.message = str(exc)
            return response

        if samples.shape[0] < self.min_samples:
            response.success = False
            response.message = (
                f'Need at least {self.min_samples} samples, have {samples.shape[0]}')
            return response

        theta = samples[:, 0:6]
        ranges = joint_sampling.observed_ranges_deg(theta)
        if np.any(ranges < self.min_joint_range_deg):
            response.success = False
            response.message = (
                f'Per-joint observed range gate not met (need >= '
                f'{self.min_joint_range_deg} deg): '
                f'{dict(zip(self.arm_joint_names, np.round(ranges, 1).tolist()))}')
            return response

        cam_target_R, cam_target_t = geometry.batch_samples_to_rt(
            samples[:, 6:13])
        fix_joint1 = self._fix_joint1()

        results = {}
        if self.method in ('sequential', 'both'):
            results['sequential'] = kinematic_solver.solve_sequential(
                self.calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
                self.joint_frames, fix_joint1, ridge_lambda=self.ridge_lambda,
                bounds_translation_m=self.bounds_translation_m,
                bounds_rotation_deg=self.bounds_rotation_deg)
        if self.method in ('joint', 'both'):
            results['joint_trf'] = kinematic_solver.solve_joint(
                self.calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
                self.joint_frames, fix_joint1, ridge_lambda=self.ridge_lambda,
                bounds_translation_m=self.bounds_translation_m,
                bounds_rotation_deg=self.bounds_rotation_deg, optimizer='trf')
            if self.run_lm_diagnostic:
                results['joint_lm_diagnostic'] = kinematic_solver.solve_joint(
                    self.calibration_type, theta, known_R, known_t, cam_target_R, cam_target_t,
                    self.joint_frames, fix_joint1, ridge_lambda=self.ridge_lambda,
                    optimizer='lm')

        primary_key = 'joint_trf' if 'joint_trf' in results else 'sequential'
        primary = results[primary_key]

        result_dict = {
            RESULT_KEY: {
                'calibration_type': self.calibration_type,
                'base_frame': self.base_frame,
                'tool_frame': self.tool_frame,
                'arm_joint_names': list(self.arm_joint_names),
                'primary_method': primary_key,
                'corrections': self._result_dict(primary)['corrections'],
                'mount_offset': self._result_dict(primary)['mount_offset'],
                'methods': {name: self._result_dict(res) for name, res in results.items()},
                'timestamp': datetime.datetime.now(datetime.timezone.utc)
                .strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
        }
        storage.save_result(self.result_file, result_dict)

        summary = ', '.join(
            f'{name}: rms {res.rms_before_m * 1000:.2f}->{res.rms_after_m * 1000:.2f} mm, '
            f'cond={res.condition_number:.1f}'
            for name, res in results.items())
        response.success = True
        response.message = f'Calibration complete ({samples.shape[0]} samples). {summary}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = KinematicCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
