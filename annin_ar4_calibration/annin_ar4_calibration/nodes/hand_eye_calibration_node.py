"""Solves and publishes the hand-eye calibration transform: tool_frame ->
camera_optical_frame for eye_in_hand, or base_frame -> camera_optical_frame
for eye_to_hand"""
import datetime

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from annin_ar4_calibration.core import geometry, handeye_solver, paths, storage

RESULT_KEY = 'hand_eye_calibration'
REQUIRED_KEYS = ('transform', 'parent_frame', 'child_frame')


class HandEyeCalibrationNode(Node):

    def __init__(self):
        super().__init__('hand_eye_calibration')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('camera_optical_frame',
                               'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'calibration_target')
        self.declare_parameter('calibration_type', 'eye_in_hand')
        self.declare_parameter('hand_eye_method', 'PARK')
        self.declare_parameter('min_samples', 10)
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.camera_optical_frame = self.get_parameter(
            'camera_optical_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.calibration_type = self.get_parameter('calibration_type').value
        self.hand_eye_method = self.get_parameter('hand_eye_method').value
        self.min_samples = self.get_parameter('min_samples').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_hand_eye_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_hand_eye_result_file)

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_service(Trigger, '~/compute_calibration', self._compute_cb)

        self._load_and_republish_existing_result()

        self.get_logger().info(
            f'Hand-eye calibration ready: calibration_type={self.calibration_type}, '
            f'sample_file={self.sample_file}, result_file={self.result_file}')

    def _parent_frame(self) -> str:
        return self.tool_frame if self.calibration_type == 'eye_in_hand' else self.base_frame

    def _broadcast(self, R: np.ndarray, t: np.ndarray) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._parent_frame()
        msg.child_frame_id = self.camera_optical_frame
        msg.transform = geometry.rt_to_transform(t, R)
        self.tf_broadcaster.sendTransform(msg)

    def _broadcast_from_stored(self, transform: dict) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._parent_frame()
        msg.child_frame_id = self.camera_optical_frame
        msg.transform.translation.x = transform['translation']['x']
        msg.transform.translation.y = transform['translation']['y']
        msg.transform.translation.z = transform['translation']['z']
        msg.transform.rotation.x = transform['rotation']['x']
        msg.transform.rotation.y = transform['rotation']['y']
        msg.transform.rotation.z = transform['rotation']['z']
        msg.transform.rotation.w = transform['rotation']['w']
        self.tf_broadcaster.sendTransform(msg)

    def _load_and_republish_existing_result(self) -> None:
        try:
            result = storage.load_result(
                self.result_file, key=RESULT_KEY, required_keys=REQUIRED_KEYS)
        except storage.ResultFileError as exc:
            self.get_logger().warning(f'Ignoring existing result file: {exc}')
            return
        if result is None:
            return
        self._broadcast_from_stored(result['transform'])
        self.get_logger().info(
            f'Loaded existing calibration from {self.result_file}, '
            f're-broadcasting {result["parent_frame"]} -> {result["child_frame"]}')

    def _compute_cb(self, request, response):
        try:
            samples = storage.load_samples(self.sample_file, ncols=14)
        except storage.SampleFileError as exc:
            response.success = False
            response.message = str(exc)
            return response

        if samples.shape[0] < self.min_samples:
            response.success = False
            response.message = (
                f'Need at least {self.min_samples} samples, have {samples.shape[0]}')
            return response

        meta = storage.load_meta(self.sample_file)
        if meta and meta.get('calibration_type') != self.calibration_type:
            self.get_logger().warning(
                f'Sample file was captured with calibration_type='
                f'{meta.get("calibration_type")}, but this node is configured with '
                f'calibration_type={self.calibration_type}')

        robot_R, robot_t = geometry.batch_samples_to_rt(samples[:, 0:7])
        target_R, target_t = geometry.batch_samples_to_rt(samples[:, 7:14])

        result = handeye_solver.solve_hand_eye(
            self.calibration_type, robot_R, robot_t, target_R, target_t,
            method=self.hand_eye_method)

        quat = geometry.rotation_matrix_to_quat(result.R)
        result_dict = {
            RESULT_KEY: {
                'calibration_type': self.calibration_type,
                'base_frame': self.base_frame,
                'tool_frame': self.tool_frame,
                'camera_optical_frame': self.camera_optical_frame,
                'target_frame': self.target_frame,
                'parent_frame': self._parent_frame(),
                'child_frame': self.camera_optical_frame,
                'transform': {
                    'translation': {
                        'x': float(result.t[0]), 'y': float(result.t[1]),
                        'z': float(result.t[2]),
                    },
                    'rotation': {
                        'x': float(quat[0]), 'y': float(quat[1]),
                        'z': float(quat[2]), 'w': float(quat[3]),
                    },
                },
                'consistency_rms_m': result.consistency_rms_m,
                'consistency_rms_deg': result.consistency_rms_deg,
                'num_samples': result.num_samples,
                'method': result.method,
                'timestamp': datetime.datetime.now(datetime.timezone.utc)
                .strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
        }
        storage.save_result(self.result_file, result_dict)
        self._broadcast(result.R, result.t)

        response.success = True
        response.message = (
            f'Calibration complete: {self._parent_frame()} -> {self.camera_optical_frame}, '
            f't=[{result.t[0]:.4f}, {result.t[1]:.4f}, {result.t[2]:.4f}] m, '
            f'consistency_rms={result.consistency_rms_m * 1000:.2f} mm / '
            f'{result.consistency_rms_deg:.2f} deg, samples={result.num_samples}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
