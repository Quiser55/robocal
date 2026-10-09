"""Drives the robot through automatically-generated calibration poses,
capturing paired (robot pose, detected target pose) samples for hand-eye calibration"""
import time

import numpy as np
import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException, LookupException,
                     TransformListener)

from annin_ar4_calibration.core import geometry, paths, pose_sampling, storage
from annin_ar4_calibration.nodes import _motion
from annin_ar4_calibration.nodes._detection_quality import DetectionQualityRecorder


class HandEyeCollectorNode(Node):

    def __init__(self):
        super().__init__('hand_eye_collector')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('camera_optical_frame',
                               'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'calibration_target')
        self.declare_parameter('calibration_type', 'eye_in_hand')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)
        self.declare_parameter('motion_timeout_sec',
                               _motion.DEFAULT_MOTION_TIMEOUT_SEC)
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('num_candidate_poses', 20)
        self.declare_parameter('min_samples', 10)
        self.declare_parameter('max_translation_m', 0.03)
        self.declare_parameter('max_orientation_deg', 35.0)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter(
            'detection_quality_topic', '/target_detector/detection_quality')

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.camera_optical_frame = self.get_parameter(
            'camera_optical_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.calibration_type = self.get_parameter('calibration_type').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_hand_eye_sample_file)
        self.tf_lookup_timeout_sec = self.get_parameter(
            'tf_lookup_timeout_sec').value
        self.motion_timeout_sec = self.get_parameter(
            'motion_timeout_sec').value
        self.num_candidate_poses = self.get_parameter(
            'num_candidate_poses').value
        self.min_samples = self.get_parameter('min_samples').value
        self.max_translation_m = self.get_parameter('max_translation_m').value
        self.max_orientation_deg = self.get_parameter(
            'max_orientation_deg').value
        self.settle_time_sec = self.get_parameter('settle_time_sec').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detection_quality = DetectionQualityRecorder(
            self, self.get_parameter('detection_quality_topic').value)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.get_parameter('arm_joint_names').value,
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name='ar_manipulator',
            callback_group=ReentrantCallbackGroup(),
            # Plan and execute inside move_group (a single action), instead of
            # pymoveit2's default of planning client-side and executing separately
            use_move_group_action=True,
        )

        self.seed_R = None
        self.seed_t = None
        self.rng = np.random.default_rng()

        service_cb_group = ReentrantCallbackGroup()
        self.create_service(
            Trigger, '~/capture_seed', self._capture_seed_cb, callback_group=service_cb_group)
        self.create_service(
            Trigger, '~/run_auto_sequence', self._run_auto_sequence_cb,
            callback_group=service_cb_group)
        self.create_service(
            Trigger, '~/collect_sample', self._collect_sample_cb,
            callback_group=service_cb_group)
        self.create_service(
            Trigger, '~/reset_samples', self._reset_cb, callback_group=service_cb_group)

        self.get_logger().info(
            f'Hand-eye collector ready: calibration_type={self.calibration_type}, '
            f'sample_file={self.sample_file}. This node moves the robot during '
            f'run_auto_sequence - clear the workspace first.')

    # TF helpers

    def sample_stamp(self):
        return self.get_clock().now()

    def _lookup_robot_pose(self, when=None):
        return self.tf_buffer.lookup_transform(
            self.base_frame, self.tool_frame, when or rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout_sec))

    def _lookup_target_pose(self, when=None):
        return self.tf_buffer.lookup_transform(
            self.camera_optical_frame, self.target_frame, when or rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout_sec))

    def _append_paired_sample(self, robot_ts, target_ts) -> int:
        r_t, r_q = robot_ts.transform.translation, robot_ts.transform.rotation
        t_t, t_q = target_ts.transform.translation, target_ts.transform.rotation
        row = np.array([
            r_t.x, r_t.y, r_t.z, r_q.x, r_q.y, r_q.z, r_q.w,
            t_t.x, t_t.y, t_t.z, t_q.x, t_q.y, t_q.z, t_q.w,
        ])
        storage.save_meta(self.sample_file, {
            'base_frame': self.base_frame, 'tool_frame': self.tool_frame,
            'camera_optical_frame': self.camera_optical_frame,
            'target_frame': self.target_frame, 'calibration_type': self.calibration_type,
        })
        count = storage.append_sample(self.sample_file, row, ncols=14)
        self.detection_quality.record(self.sample_file, count)
        return count

    # Services

    def _capture_seed_cb(self, request, response):
        try:
            ts = self._lookup_robot_pose()
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            response.success = False
            response.message = f'TF lookup {self.base_frame} -> {self.tool_frame} failed: {exc}'
            return response

        self.seed_R, self.seed_t = geometry.transform_stamped_to_rt(ts)
        response.success = True
        response.message = (
            f'Captured seed pose: t=[{self.seed_t[0]:.4f}, {self.seed_t[1]:.4f}, '
            f'{self.seed_t[2]:.4f}]. Call run_auto_sequence next.')
        return response

    def _collect_sample_cb(self, request, response):
        try:
            robot_ts = self._lookup_robot_pose()
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            response.success = False
            response.message = f'TF lookup {self.base_frame} -> {self.tool_frame} failed: {exc}'
            return response

        try:
            target_ts = self._lookup_target_pose()
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            response.success = False
            response.message = (
                f'Target not currently visible ({self.camera_optical_frame} -> '
                f'{self.target_frame} TF lookup failed): {exc}')
            return response

        count = self._append_paired_sample(robot_ts, target_ts)
        response.success = True
        response.message = f'Recorded manual sample {count} (total {count})'
        return response

    def _reset_cb(self, request, response):
        backup_path = storage.reset_samples(
            self.sample_file, keep_backup=True, ncols=14)
        storage.reset_detections(self.sample_file)
        response.success = True
        if backup_path:
            response.message = f'Cleared samples (backup: {backup_path})'
        else:
            response.message = 'Cleared samples (no prior samples existed)'
        return response

    def _run_auto_sequence_cb(self, request, response):
        if self.seed_R is None:
            response.success = False
            response.message = (
                'No seed pose captured yet - jog the arm to a good starting pose '
                '(target visible) and call capture_seed first')
            return response

        candidates = pose_sampling.sample_orbit_poses(
            self.seed_R, self.seed_t, self.num_candidate_poses,
            self.max_translation_m, self.max_orientation_deg, rng=self.rng)

        collected = 0
        skipped_unreachable = 0
        skipped_not_visible = 0

        for R, t in candidates:
            if collected >= self.min_samples:
                break

            quat = geometry.rotation_matrix_to_quat(R)
            self.moveit2.move_to_pose(
                position=(float(t[0]), float(t[1]), float(t[2])),
                quat_xyzw=(float(quat[0]), float(quat[1]),
                           float(quat[2]), float(quat[3])),
                frame_id=self.base_frame)
            moved, reason = _motion.wait_for_motion(
                self.moveit2, self.motion_timeout_sec)
            if not moved:
                skipped_unreachable += 1
                self.get_logger().warning(f'Candidate pose skipped: {reason}')
                continue

            time.sleep(self.settle_time_sec)

            try:
                stamp = self.sample_stamp()
                robot_ts = self._lookup_robot_pose(stamp)
                target_ts = self._lookup_target_pose(stamp)
            except (LookupException, ConnectivityException, ExtrapolationException):
                skipped_not_visible += 1
                self.get_logger().warning(
                    'No fresh target detection at candidate pose, skipping')
                continue

            self._append_paired_sample(robot_ts, target_ts)
            collected += 1

        attempted = collected + skipped_unreachable + skipped_not_visible
        response.success = collected > 0
        response.message = (
            f'collected {collected}/{attempted} attempted poses '
            f'({skipped_unreachable} unreachable, {skipped_not_visible} target not visible)')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCollectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
