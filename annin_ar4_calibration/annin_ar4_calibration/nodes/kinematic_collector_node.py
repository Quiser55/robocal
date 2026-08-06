"""Drives the robot through randomized joint-space configurations spanning
the workspace, capturing paired (joint angles, detected target pose) samples
for kinematic parameter identification (WP4).

Unlike hand-eye's collector (a small Cartesian orbit around one seed pose),
this node needs genuine per-joint angle diversity across each joint's full
range - several of WP4's identifiability arguments (see
core/kinematic_model.py) depend on that, not just on sample count. So
candidates are sampled directly in joint space (core/joint_sampling.py)
rather than perturbing a single seed pose.

Like hand_eye_collector_node.py, this node commands the robot directly via
MoveIt (through pymoveit2) during run_auto_sequence - see the package
README's safety note before running it.

NOTE (pymoveit2 API): move_to_configuration()/wait_until_executed() are used
exactly as pymoveit2's own examples do. Verify this against whatever
pymoveit2 revision is actually vcs-imported into the workspace if this needs
adjusting (see the same caveat in hand_eye_collector_node.py).
"""
import time

import numpy as np
import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException, LookupException,
                      TransformListener)

from annin_ar4_calibration.core import joint_sampling, kinematic_model, paths, storage
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description


class KinematicCollectorNode(Node):

    def __init__(self):
        super().__init__('kinematic_collector')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('camera_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'calibration_target')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('num_candidate_configs', 200)
        self.declare_parameter('min_samples', 100)
        self.declare_parameter('min_joint_range_deg', 30.0)
        self.declare_parameter('joint_margin_deg', 5.0)
        self.declare_parameter('settle_time_sec', 0.5)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_kinematic_sample_file)
        self.tf_lookup_timeout_sec = self.get_parameter('tf_lookup_timeout_sec').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.num_candidate_configs = self.get_parameter('num_candidate_configs').value
        self.min_samples = self.get_parameter('min_samples').value
        self.min_joint_range_deg = self.get_parameter('min_joint_range_deg').value
        self.joint_margin_deg = self.get_parameter('joint_margin_deg').value
        self.settle_time_sec = self.get_parameter('settle_time_sec').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._latest_joint_state = None
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self._joint_state_cb, 10)

        urdf_xml = fetch_robot_description(self)
        self.joint_frames = kinematic_model.parse_urdf_chain(urdf_xml, self.arm_joint_names)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name='ar_manipulator',
            callback_group=ReentrantCallbackGroup(),
        )

        self.rng = np.random.default_rng()

        service_cb_group = ReentrantCallbackGroup()
        self.create_service(
            Trigger, '~/run_auto_sequence', self._run_auto_sequence_cb,
            callback_group=service_cb_group)
        self.create_service(
            Trigger, '~/collect_sample', self._collect_sample_cb,
            callback_group=service_cb_group)
        self.create_service(
            Trigger, '~/reset_samples', self._reset_cb, callback_group=service_cb_group)

        self.get_logger().info(
            f'Kinematic collector ready: sample_file={self.sample_file}. This node moves the '
            f'robot during run_auto_sequence - clear the workspace first.')

    # -- TF/joint-state helpers --

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _current_theta(self):
        """Returns (6,) joint angles in arm_joint_names order, or None if no
        /joint_states message has been received yet or it's missing a joint."""
        msg = self._latest_joint_state
        if msg is None:
            return None
        try:
            indices = [msg.name.index(n) for n in self.arm_joint_names]
        except ValueError:
            return None
        return np.array([msg.position[i] for i in indices])

    def _lookup_target_pose(self):
        return self.tf_buffer.lookup_transform(
            self.camera_optical_frame, self.target_frame, rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout_sec))

    def _append_sample(self, theta: np.ndarray, target_ts) -> int:
        t_t, t_q = target_ts.transform.translation, target_ts.transform.rotation
        row = np.concatenate([theta, [t_t.x, t_t.y, t_t.z, t_q.x, t_q.y, t_q.z, t_q.w]])
        storage.save_meta(self.sample_file, {
            'base_frame': self.base_frame, 'tool_frame': self.tool_frame,
            'camera_optical_frame': self.camera_optical_frame,
            'target_frame': self.target_frame,
            'arm_joint_names': list(self.arm_joint_names),
        })
        return storage.append_sample(self.sample_file, row, ncols=13)

    # -- Services --

    def _collect_sample_cb(self, request, response):
        theta = self._current_theta()
        if theta is None:
            response.success = False
            response.message = 'No /joint_states received yet for all arm_joint_names'
            return response

        try:
            target_ts = self._lookup_target_pose()
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            response.success = False
            response.message = (
                f'Target not currently visible ({self.camera_optical_frame} -> '
                f'{self.target_frame} TF lookup failed): {exc}')
            return response

        count = self._append_sample(theta, target_ts)
        response.success = True
        response.message = f'Recorded manual sample {count} (total {count})'
        return response

    def _reset_cb(self, request, response):
        backup_path = storage.reset_samples(self.sample_file, keep_backup=True, ncols=13)
        response.success = True
        if backup_path:
            response.message = f'Cleared samples (backup: {backup_path})'
        else:
            response.message = 'Cleared samples (no prior samples existed)'
        return response

    def _run_auto_sequence_cb(self, request, response):
        candidates = joint_sampling.sample_joint_configs(
            self.joint_frames, self.num_candidate_configs,
            margin_deg=self.joint_margin_deg, rng=self.rng)

        collected_thetas = []
        skipped_unreachable = 0
        skipped_not_visible = 0

        for theta in candidates:
            if len(collected_thetas) >= self.min_samples and self._range_gate_passes(
                    np.array(collected_thetas)):
                break

            self.moveit2.move_to_configuration(
                joint_positions=[float(v) for v in theta], joint_names=self.arm_joint_names)
            moved = self.moveit2.wait_until_executed()
            if not moved:
                skipped_unreachable += 1
                self.get_logger().warning(
                    'Candidate configuration unreachable or failed to execute, skipping')
                continue

            time.sleep(self.settle_time_sec)

            actual_theta = self._current_theta()
            try:
                target_ts = self._lookup_target_pose()
            except (LookupException, ConnectivityException, ExtrapolationException):
                skipped_not_visible += 1
                self.get_logger().warning('Target not visible at candidate pose, skipping')
                continue

            if actual_theta is None:
                skipped_not_visible += 1
                continue

            self._append_sample(actual_theta, target_ts)
            collected_thetas.append(actual_theta)

        attempted = len(collected_thetas) + skipped_unreachable + skipped_not_visible
        collected = len(collected_thetas)
        range_ok = self._range_gate_passes(np.array(collected_thetas)) if collected else False

        response.success = collected >= self.min_samples and range_ok
        message = (
            f'collected {collected}/{attempted} attempted configs '
            f'({skipped_unreachable} unreachable, {skipped_not_visible} target not visible)')
        if collected and not range_ok:
            shortfall = self._range_shortfall(np.array(collected_thetas))
            message += f'; per-joint range gate not met (< {self.min_joint_range_deg} deg): ' \
                       f'{shortfall}'
        response.message = message
        return response

    def _range_gate_passes(self, thetas: np.ndarray) -> bool:
        if thetas.shape[0] == 0:
            return False
        ranges = joint_sampling.observed_ranges_deg(thetas)
        return bool(np.all(ranges >= self.min_joint_range_deg))

    def _range_shortfall(self, thetas: np.ndarray) -> dict:
        ranges = joint_sampling.observed_ranges_deg(thetas)
        return {
            name: round(float(r), 1)
            for name, r in zip(self.arm_joint_names, ranges)
            if r < self.min_joint_range_deg
        }


def main(args=None):
    rclpy.init(args=args)
    node = KinematicCollectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
