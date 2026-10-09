"""Publishes RViz markers for captured kinematic-calibration samples"""
import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from annin_ar4_calibration.core import geometry, kinematic_model, kinematic_solver, paths, storage
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description

RESULT_KEY = 'kinematic_calibration'
REQUIRED_KEYS = ('corrections', 'mount_offset', 'calibration_type')
HAND_EYE_KEY = 'hand_eye_calibration'
HAND_EYE_REQUIRED_KEYS = ('transform', 'parent_frame', 'child_frame')


def _corrections_from_result(result: dict) -> np.ndarray:
    corrections = np.zeros((6, kinematic_model.PARAMS_PER_JOINT))
    for i, name in enumerate(result['arm_joint_names']):
        entry = result['corrections'].get(name)
        if entry is None:
            continue
        corrections[i, 0:2] = entry['delta_t_perp']
        corrections[i, 2:4] = entry['delta_r_perp']
    return corrections


def _mount_from_result(result: dict) -> tuple:
    mount = result['mount_offset']
    mount_R = geometry.quat_to_rotation_matrix(
        mount['rotation']['x'], mount['rotation']['y'],
        mount['rotation']['z'], mount['rotation']['w'])
    mount_t = np.array([mount['translation']['x'], mount['translation']['y'],
                        mount['translation']['z']])
    return mount_R, mount_t


class KinematicVisualizationNode(Node):

    def __init__(self):
        super().__init__('kinematic_visualization')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('sample_file', '')
        self.declare_parameter('result_file', '')
        self.declare_parameter('hand_eye_result_file', '')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_kinematic_sample_file)
        self.result_file = paths.resolve(
            self.get_parameter('result_file').value, paths.default_kinematic_result_file)
        self.hand_eye_result_file = paths.resolve(
            self.get_parameter('hand_eye_result_file').value, paths.default_hand_eye_result_file)
        publish_rate_hz = self.get_parameter('publish_rate_hz').value

        try:
            urdf_xml = fetch_robot_description(self)
            self.joint_frames = kinematic_model.parse_urdf_chain(
                urdf_xml, self.arm_joint_names)
        except RuntimeError as exc:
            self.get_logger().warning(
                f'Could not fetch robot_description at startup ({exc}) - nominal/corrected FK '
                f'markers will be unavailable until this succeeds on a later attempt.')
            self.joint_frames = None

        self._latest_joint_state = None
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self._joint_state_cb, 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.publisher = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish)

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _current_theta(self):
        msg = self._latest_joint_state
        if msg is None:
            return None
        try:
            indices = [msg.name.index(n) for n in self.arm_joint_names]
        except ValueError:
            return None
        return np.array([msg.position[i] for i in indices])

    def _publish(self):
        try:
            samples = storage.load_samples(self.sample_file, ncols=13)
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

        known = None
        if result is not None:
            try:
                hand_eye = storage.load_result(
                    self.hand_eye_result_file, key=HAND_EYE_KEY,
                    required_keys=HAND_EYE_REQUIRED_KEYS)
            except storage.ResultFileError:
                hand_eye = None
            if hand_eye is not None:
                transform = hand_eye['transform']
                known_R = geometry.quat_to_rotation_matrix(
                    transform['rotation']['x'], transform['rotation']['y'],
                    transform['rotation']['z'], transform['rotation']['w'])
                known_t = np.array([transform['translation']['x'], transform['translation']['y'],
                                    transform['translation']['z']])
                known = (known_R, known_t)

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        n = samples.shape[0]
        status_text = f'{n} sample(s) captured'

        if n > 0 and self.joint_frames is not None:
            theta = samples[:, 0:6]

            nominal_points = np.array([
                kinematic_model.forward_kinematics(
                    theta[i], self.joint_frames)[1]
                for i in range(n)
            ])
            markers.markers.append(_points_marker(
                nominal_points, self.base_frame, 0,
                ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.9)))

            if result is not None and known is not None:
                corrections = _corrections_from_result(result)
                mount_R, mount_t = _mount_from_result(result)
                cam_target_R, cam_target_t = geometry.batch_samples_to_rt(
                    samples[:, 6:13])
                known_R, known_t = known

                corrected_points = np.array([
                    kinematic_model.forward_kinematics(
                        theta[i], self.joint_frames, corrections)[1]
                    for i in range(n)
                ])
                markers.markers.append(_points_marker(
                    corrected_points, self.base_frame, 1,
                    ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)))

                measured_points = np.array([
                    kinematic_solver.reconstruct_measured_ee(
                        result['calibration_type'], mount_R, mount_t, known_R, known_t,
                        cam_target_R[i], cam_target_t[i])[1]
                    for i in range(n)
                ])
                markers.markers.append(_points_marker(
                    measured_points, self.base_frame, 2,
                    ColorRGBA(r=1.0, g=0.6, b=0.0, a=0.9)))

                methods = result.get('methods', {})
                primary = methods.get(result.get('primary_method'), {})
                status_text = (
                    f'{n} samples, primary={result.get("primary_method")}, '
                    f'rms {primary.get("rms_before_m", 0.0) * 1000:.2f}->'
                    f'{primary.get("rms_after_m", 0.0) * 1000:.2f} mm, '
                    f'cond={primary.get("condition_number", 0.0):.1f}')

                self._broadcast_live_corrected_tf(corrections)

        text_marker = Marker()
        text_marker.header.frame_id = self.base_frame
        text_marker.header.stamp = now
        text_marker.ns = 'kinematic_calibration'
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

    def _broadcast_live_corrected_tf(self, corrections: np.ndarray) -> None:
        theta = self._current_theta()
        if theta is None:
            return
        R, t = kinematic_model.forward_kinematics(
            theta, self.joint_frames, corrections)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = f'{self.tool_frame}_kinematic_corrected'
        msg.transform = geometry.rt_to_transform(t, R)
        self.tf_broadcaster.sendTransform(msg)


def _point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _points_marker(
        points: np.ndarray, frame_id: str, marker_id: int, color: ColorRGBA) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = 'kinematic_calibration'
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 0.006
    marker.scale.y = 0.006
    marker.color = color
    marker.points = [_point(p[0], p[1], p[2]) for p in points]
    return marker


def main(args=None):
    rclpy.init(args=args)
    node = KinematicVisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
