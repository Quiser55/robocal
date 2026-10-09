"""Detects a ChArUco board or single ArUco marker in a camera color image and
publishes its pose as a live TF frame, for hand-eye calibration sample collection"""
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray
from tf2_ros import TransformBroadcaster

from annin_ar4_calibration.core import geometry, target_detection

#: Layout of the ~/detection_quality message's `data` field
DETECTION_QUALITY_FIELDS = ('num_points', 'reprojection_rms_px')


class TargetDetectorNode(Node):

    def __init__(self):
        super().__init__('target_detector')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        self.declare_parameter('target_type', 'charuco')
        self.declare_parameter('aruco_dictionary', 'DICT_5X5_250')
        self.declare_parameter('charuco.squares_x', 5)
        self.declare_parameter('charuco.squares_y', 7)
        self.declare_parameter('charuco.square_length_m', 0.035)
        self.declare_parameter('charuco.marker_length_m', 0.026)
        self.declare_parameter('charuco.min_charuco_corners', 6)
        self.declare_parameter('aruco.marker_id', 1)
        self.declare_parameter('aruco.marker_length_m', 0.05)
        self.declare_parameter('target_frame', 'calibration_target')
        self.declare_parameter('publish_debug_image', True)

        self.target_frame = self.get_parameter('target_frame').value
        self.publish_debug_image = self.get_parameter(
            'publish_debug_image').value

        cfg = target_detection.TargetConfig(
            target_type=self.get_parameter('target_type').value,
            dictionary_name=self.get_parameter('aruco_dictionary').value,
            squares_x=self.get_parameter('charuco.squares_x').value,
            squares_y=self.get_parameter('charuco.squares_y').value,
            square_length_m=self.get_parameter(
                'charuco.square_length_m').value,
            marker_length_m=self.get_parameter(
                'charuco.marker_length_m').value,
            min_charuco_corners=self.get_parameter(
                'charuco.min_charuco_corners').value,
            marker_id=self.get_parameter('aruco.marker_id').value,
            aruco_marker_length_m=self.get_parameter(
                'aruco.marker_length_m').value,
        )
        self.target = target_detection.build_target(cfg)

        self.bridge = CvBridge()
        self.intrinsics = None

        self.tf_broadcaster = TransformBroadcaster(self)
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(Image, '~/debug_image', 1)
        # Per-frame detection quality
        self.quality_pub = self.create_publisher(
            Float32MultiArray, '~/detection_quality', 1)

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._info_cb, 10)
        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._image_cb, 10)

        self.get_logger().info(
            f'Target detector ready: target_type={cfg.target_type}, '
            f'target_frame={self.target_frame}')

    def _info_cb(self, msg: CameraInfo) -> None:
        self.intrinsics = target_detection.CameraIntrinsics(
            camera_matrix=np.array(msg.k).reshape(3, 3),
            dist_coeffs=np.array(msg.d))

    def _image_cb(self, msg: Image) -> None:
        if self.intrinsics is None:
            self.get_logger().warning(
                'No CameraInfo received yet, skipping frame', throttle_duration_sec=5.0)
            return

        image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = target_detection.detect(
            image_bgr, self.intrinsics, self.target, draw_debug=self.publish_debug_image)

        if result.found:
            ts = TransformStamped()
            ts.header.stamp = msg.header.stamp
            ts.header.frame_id = msg.header.frame_id
            ts.child_frame_id = self.target_frame
            ts.transform = geometry.rt_to_transform(result.t, result.R)
            self.tf_broadcaster.sendTransform(ts)

            quality = Float32MultiArray()
            quality.data = [float(result.num_points),
                            float(result.reprojection_rms_px)]
            self.quality_pub.publish(quality)

        if self.publish_debug_image:
            debug_image = result.debug_image if result.debug_image is not None else image_bgr
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
