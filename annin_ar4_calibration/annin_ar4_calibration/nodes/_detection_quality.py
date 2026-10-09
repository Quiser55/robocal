"""Shared helper: record how good the camera measurement was at the moment
each sample was captured"""
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray

from annin_ar4_calibration.core import storage


class DetectionQualityRecorder:
    """Latches the latest `~/detection_quality` message and writes it into the
    sample file's sidecar on demand."""

    def __init__(self, node, topic: str):
        self._node = node
        self._latest: tuple[float, float] | None = None
        node.create_subscription(
            Float32MultiArray, topic, self._cb, 1, callback_group=ReentrantCallbackGroup())

    def _cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            self._latest = (float(msg.data[0]), float(msg.data[1]))

    def record(self, sample_file: str, sample_count: int) -> None:
        """`sample_count` is what `storage.append_sample` returned, i.e. the
        new total - so the row's index is one less."""
        if self._latest is None:
            return
        num_points, reprojection_rms_px = self._latest
        try:
            storage.append_detection(
                sample_file, sample_count - 1, num_points, reprojection_rms_px)
        except OSError as exc:
            # A diagnostic sidecar must never take a collection run down.
            self._node.get_logger().warning(
                f'Could not record detection quality: {exc}')
