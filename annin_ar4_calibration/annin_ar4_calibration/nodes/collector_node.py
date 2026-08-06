"""Collects base_frame -> tool_frame TF samples for TCP pivot calibration.

Does not command the robot. The user jogs the arm by whatever means they
already have (MoveIt/RViz, teleop, ...) so the tool tip touches a fixed
physical reference point, then calls the collect_sample service. This node
only reads TF and persists samples to disk.
"""
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener

from annin_ar4_calibration.core import geometry, paths, storage


class CollectorNode(Node):

    def __init__(self):
        super().__init__('collector')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)
        self.declare_parameter('min_orientation_change_deg', 0.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_sample_file)
        self.tf_lookup_timeout_sec = self.get_parameter('tf_lookup_timeout_sec').value
        self.min_orientation_change_deg = self.get_parameter('min_orientation_change_deg').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_service(Trigger, '~/collect_sample', self._collect_cb)
        self.create_service(Trigger, '~/reset_samples', self._reset_cb)

        self.get_logger().info(
            f'Collector ready: {self.base_frame} -> {self.tool_frame}, '
            f'sample_file={self.sample_file}')

    def _lookup(self):
        return self.tf_buffer.lookup_transform(
            self.base_frame, self.tool_frame, rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout_sec))

    def _collect_cb(self, request, response):
        try:
            ts = self._lookup()
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            response.success = False
            response.message = f'TF lookup {self.base_frame} -> {self.tool_frame} failed: {exc}'
            return response

        t = ts.transform.translation
        q = ts.transform.rotation
        row = np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w])

        if self.min_orientation_change_deg > 0.0:
            existing = storage.load_samples(self.sample_file)
            if existing.shape[0] > 0:
                prev_q = existing[-1, 3:7]
                delta_deg = geometry.quat_angle_deg(prev_q, row[3:7])
                if delta_deg < self.min_orientation_change_deg:
                    response.success = False
                    response.message = (
                        f'Orientation too similar to last sample (delta={delta_deg:.2f} deg), '
                        'move to a more distinct orientation before capturing')
                    return response

        try:
            storage.save_meta(
                self.sample_file, {'base_frame': self.base_frame, 'tool_frame': self.tool_frame})
            count = storage.append_sample(self.sample_file, row)
        except storage.SampleFileError as exc:
            response.success = False
            response.message = str(exc)
            return response

        response.success = True
        response.message = (
            f'Recorded sample {count} (total {count}): '
            f't=[{t.x:.4f}, {t.y:.4f}, {t.z:.4f}]')
        return response

    def _reset_cb(self, request, response):
        backup_path = storage.reset_samples(self.sample_file, keep_backup=True)
        response.success = True
        if backup_path:
            response.message = f'Cleared samples (backup: {backup_path})'
        else:
            response.message = 'Cleared samples (no prior samples existed)'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
