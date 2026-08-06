"""Publishes RViz markers for captured TCP calibration samples and results.

Reads only from the sample/result files (no TF lookups), so it stays useful
for reviewing previously captured data even when the robot/TF tree isn't
currently running.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from annin_ar4_calibration.core import geometry, paths, storage


class VisualizationNode(Node):

    def __init__(self):
        super().__init__('visualization')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_result_file)
        publish_rate_hz = self.get_parameter('publish_rate_hz').value

        self.publisher = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish_markers)

    def _publish_markers(self):
        try:
            samples = storage.load_samples(self.sample_file)
        except storage.SampleFileError as exc:
            self.get_logger().warning(f'Could not load samples: {exc}', throttle_duration_sec=5.0)
            return

        try:
            result = storage.load_result(self.result_file)
        except storage.ResultFileError as exc:
            self.get_logger().warning(f'Could not load result: {exc}', throttle_duration_sec=5.0)
            result = None

        meta = storage.load_meta(self.sample_file)
        if meta and meta.get('base_frame') != self.base_frame:
            self.get_logger().warning(
                f'Samples were captured with base_frame={meta.get("base_frame")}, '
                f'but this node is configured with base_frame={self.base_frame}',
                throttle_duration_sec=5.0)

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        n = samples.shape[0]

        raw_points = Marker()
        raw_points.header.frame_id = self.base_frame
        raw_points.header.stamp = now
        raw_points.ns = 'tcp_calibration'
        raw_points.id = 0
        raw_points.type = Marker.POINTS
        raw_points.action = Marker.ADD
        raw_points.scale.x = 0.006
        raw_points.scale.y = 0.006
        raw_points.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
        raw_points.points = [_point(row[0], row[1], row[2]) for row in samples]
        markers.markers.append(raw_points)

        status_text = f'{n} sample(s) captured'

        if result is not None and n > 0:
            offset = np.array([result['offset']['x'], result['offset']['y'], result['offset']['z']])
            base_point = result.get('base_point', {'x': 0.0, 'y': 0.0, 'z': 0.0})
            base_point = np.array([base_point['x'], base_point['y'], base_point['z']])
            threshold = result.get('ransac_threshold', result.get('rms_error', 0.002))

            R, t = geometry.batch_samples_to_rt(samples)
            fitted = np.einsum('nij,j->ni', R, offset) + t
            errors = np.linalg.norm(fitted - base_point, axis=1)

            fitted_points = Marker()
            fitted_points.header.frame_id = self.base_frame
            fitted_points.header.stamp = now
            fitted_points.ns = 'tcp_calibration'
            fitted_points.id = 1
            fitted_points.type = Marker.POINTS
            fitted_points.action = Marker.ADD
            fitted_points.scale.x = 0.006
            fitted_points.scale.y = 0.006
            fitted_points.points = [_point(p[0], p[1], p[2]) for p in fitted]
            fitted_points.colors = [
                ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9) if err < threshold
                else ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.9)
                for err in errors
            ]
            markers.markers.append(fitted_points)

            base_point_marker = Marker()
            base_point_marker.header.frame_id = self.base_frame
            base_point_marker.header.stamp = now
            base_point_marker.ns = 'tcp_calibration'
            base_point_marker.id = 2
            base_point_marker.type = Marker.SPHERE
            base_point_marker.action = Marker.ADD
            base_point_marker.pose.position = _point(base_point[0], base_point[1], base_point[2])
            base_point_marker.pose.orientation.w = 1.0
            base_point_marker.scale.x = 0.015
            base_point_marker.scale.y = 0.015
            base_point_marker.scale.z = 0.015
            base_point_marker.color = ColorRGBA(r=1.0, g=0.9, b=0.0, a=1.0)
            markers.markers.append(base_point_marker)

            num_inliers = result.get('num_inliers', n)
            rms = result.get('rms_error', 0.0)
            status_text = f'{n} samples, {num_inliers} inliers, rms={rms * 1000:.2f} mm'

        text_marker = Marker()
        text_marker.header.frame_id = self.base_frame
        text_marker.header.stamp = now
        text_marker.ns = 'tcp_calibration'
        text_marker.id = 3
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        if n > 0:
            centroid = samples[:, 0:3].mean(axis=0)
            text_marker.pose.position = _point(centroid[0], centroid[1], centroid[2] + 0.05)
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.03
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = status_text
        markers.markers.append(text_marker)

        self.publisher.publish(markers)


def _point(x: float, y: float, z: float):
    from geometry_msgs.msg import Point
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
