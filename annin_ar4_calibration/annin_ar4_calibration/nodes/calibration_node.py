"""Solves and publishes the tool_frame -> tcp_frame static TCP offset."""
import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from annin_ar4_calibration.core import geometry, paths, solver, storage


class CalibrationNode(Node):

    def __init__(self):
        super().__init__('calibration')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('tcp_frame', 'tcp')
        self.declare_parameter('min_samples', 30)
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')
        self.declare_parameter('ransac.enabled', True)
        self.declare_parameter('ransac.threshold', 0.002)
        self.declare_parameter('ransac.iterations', 100)
        self.declare_parameter('ransac.min_subset_size', solver.DEFAULT_MIN_SUBSET_SIZE)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.tcp_frame = self.get_parameter('tcp_frame').value
        self.min_samples = self.get_parameter('min_samples').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_result_file)
        self.ransac_enabled = self.get_parameter('ransac.enabled').value
        self.ransac_threshold = self.get_parameter('ransac.threshold').value
        self.ransac_iterations = self.get_parameter('ransac.iterations').value
        self.ransac_min_subset_size = self.get_parameter('ransac.min_subset_size').value

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_service(Trigger, '~/compute_calibration', self._compute_cb)

        self._load_and_republish_existing_result()

        self.get_logger().info(
            f'Calibration ready: {self.tool_frame} -> {self.tcp_frame}, '
            f'sample_file={self.sample_file}, result_file={self.result_file}')

    def _broadcast(self, offset: np.ndarray) -> None:
        from geometry_msgs.msg import TransformStamped
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.tool_frame
        msg.child_frame_id = self.tcp_frame
        msg.transform = geometry.rt_to_transform(offset)
        self.tf_broadcaster.sendTransform(msg)

    def _load_and_republish_existing_result(self) -> None:
        try:
            result = storage.load_result(self.result_file)
        except storage.ResultFileError as exc:
            self.get_logger().warning(f'Ignoring existing result file: {exc}')
            return
        if result is None:
            return
        offset = np.array([result['offset']['x'], result['offset']['y'], result['offset']['z']])
        self._broadcast(offset)
        self.get_logger().info(
            f'Loaded existing calibration from {self.result_file}, '
            f're-broadcasting {self.tool_frame} -> {self.tcp_frame}')

    def _compute_cb(self, request, response):
        try:
            samples = storage.load_samples(self.sample_file)
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
        if meta and (meta.get('base_frame') != self.base_frame
                     or meta.get('tool_frame') != self.tool_frame):
            self.get_logger().warning(
                f'Sample file was captured with base_frame={meta.get("base_frame")}, '
                f'tool_frame={meta.get("tool_frame")}, but this node is configured with '
                f'base_frame={self.base_frame}, tool_frame={self.tool_frame}')

        R, t = geometry.batch_samples_to_rt(samples)
        result = solver.solve_pivot(
            R, t, self.ransac_enabled, self.ransac_threshold, self.ransac_iterations,
            self.ransac_min_subset_size)

        if result.degenerate_fallback:
            self.get_logger().warning(
                'RANSAC could not find a non-degenerate sample subset; '
                'falling back to a full-data least-squares fit')

        result_dict = {
            'tcp_calibration': {
                'base_frame': self.base_frame,
                'tool_frame': self.tool_frame,
                'tcp_frame': self.tcp_frame,
                'offset': {
                    'x': float(result.p_tool[0]),
                    'y': float(result.p_tool[1]),
                    'z': float(result.p_tool[2]),
                },
                'base_point': {
                    'x': float(result.p_base[0]),
                    'y': float(result.p_base[1]),
                    'z': float(result.p_base[2]),
                },
                'rms_error': result.rms_inliers,
                'rms_all_samples': result.rms_all,
                'num_samples': result.num_samples,
                'num_inliers': result.num_inliers,
                'ransac_enabled': self.ransac_enabled,
                'ransac_threshold': self.ransac_threshold,
                'timestamp': datetime.datetime.now(datetime.timezone.utc)
                .strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
        }
        storage.save_result(self.result_file, result_dict)
        self._broadcast(result.p_tool)

        response.success = True
        response.message = (
            f'Calibration complete: offset=[{result.p_tool[0]:.4f}, '
            f'{result.p_tool[1]:.4f}, {result.p_tool[2]:.4f}] m, '
            f'rms_inliers={result.rms_inliers:.5f} m, samples={result.num_samples}, '
            f'inliers={result.num_inliers}/{result.num_samples}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
