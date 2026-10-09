"""Publishes RViz markers for captured hand-eye calibration samples and results"""
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from annin_ar4_calibration.core import geometry, paths, storage

RESULT_KEY = 'hand_eye_calibration'
REQUIRED_KEYS = ('transform', 'parent_frame', 'child_frame')
AXIS_LENGTH_M = 0.05


class HandEyeVisualizationNode(Node):

    def __init__(self):
        super().__init__('hand_eye_visualization')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_hand_eye_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_hand_eye_result_file)
        publish_rate_hz = self.get_parameter('publish_rate_hz').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish_markers)

    def _publish_markers(self):
        try:
            samples = storage.load_samples(self.sample_file, ncols=14)
        except storage.SampleFileError as exc:
            self.get_logger().warning(
                f'Could not load samples: {exc}', throttle_duration_sec=5.0)
            return

        try:
            result = storage.load_result(
                self.result_file, key=RESULT_KEY, required_keys=REQUIRED_KEYS)
        except storage.ResultFileError as exc:
            self.get_logger().warning(
                f'Could not load result: {exc}', throttle_duration_sec=5.0)
            result = None

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        n = samples.shape[0]

        sample_points = Marker()
        sample_points.header.frame_id = self.base_frame
        sample_points.header.stamp = now
        sample_points.ns = 'hand_eye_calibration'
        sample_points.id = 0
        sample_points.type = Marker.POINTS
        sample_points.action = Marker.ADD
        sample_points.scale.x = 0.006
        sample_points.scale.y = 0.006
        sample_points.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)
        sample_points.points = [
            _point(row[0], row[1], row[2]) for row in samples]
        markers.markers.append(sample_points)

        status_text = f'{n} sample(s) captured'
        camera_axis_note = ''

        if result is not None:
            transform = result['transform']
            R = geometry.quat_to_rotation_matrix(
                transform['rotation']['x'], transform['rotation']['y'],
                transform['rotation']['z'], transform['rotation']['w'])
            t = np.array([
                transform['translation']['x'], transform['translation']['y'],
                transform['translation']['z']])

            calibration_type = result.get('calibration_type', 'eye_to_hand')
            if calibration_type == 'eye_in_hand':
                camera_R, camera_t, ok = self._compose_with_live_tf(R, t)
                if not ok:
                    camera_axis_note = ' (camera pose marker unavailable: TF not live)'
            else:
                camera_R, camera_t, ok = R, t, True

            if ok:
                markers.markers.append(_axis_marker(
                    camera_R, camera_t, self.base_frame, id_offset=1))

            rms_m = result.get('consistency_rms_m', 0.0)
            rms_deg = result.get('consistency_rms_deg', 0.0)
            status_text = (
                f'{n} samples, calibration_type={calibration_type}, '
                f'consistency rms={rms_m * 1000:.2f} mm / {rms_deg:.2f} deg{camera_axis_note}')

        text_marker = Marker()
        text_marker.header.frame_id = self.base_frame
        text_marker.header.stamp = now
        text_marker.ns = 'hand_eye_calibration'
        text_marker.id = 10
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        if n > 0:
            centroid = samples[:, 0:3].mean(axis=0)
            text_marker.pose.position = _point(
                centroid[0], centroid[1], centroid[2] + 0.05)
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.03
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = status_text
        markers.markers.append(text_marker)

        self.publisher.publish(markers)

    def _compose_with_live_tf(self, tool_to_camera_R, tool_to_camera_t):
        """eye_in_hand only: compose the stored tool_frame -> camera transform"""
        try:
            ts = self.tf_buffer.lookup_transform(
                self.base_frame, self.tool_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, False
        base_to_tool_R, base_to_tool_t = geometry.transform_stamped_to_rt(ts)
        R, t = geometry.compose_rt(
            base_to_tool_R, base_to_tool_t, tool_to_camera_R, tool_to_camera_t)
        return R, t, True


def _point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _axis_marker(R: np.ndarray, t: np.ndarray, frame_id: str, id_offset: int) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = 'hand_eye_calibration'
    marker.id = id_offset
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.003
    marker.pose.orientation.w = 1.0

    origin = _point(t[0], t[1], t[2])
    axis_colors = [
        ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
        ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
    ]
    for axis_idx in range(3):
        tip = t + R[:, axis_idx] * AXIS_LENGTH_M
        marker.points.append(origin)
        marker.points.append(_point(tip[0], tip[1], tip[2]))
        marker.colors.append(axis_colors[axis_idx])
        marker.colors.append(axis_colors[axis_idx])
    return marker


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeVisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
