"""Drives the robot through randomized joint-space configurations spanning
the workspace, capturing paired (joint angles, detected target pose) samples
for kinematic parameter identification (WP4)"""
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from pymoveit2 import MoveIt2
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, JointState
from std_srvs.srv import Trigger
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException, LookupException,
                     TransformListener)

from annin_ar4_calibration.core import (
    geometry, joint_sampling, kinematic_model, paths, storage)
from annin_ar4_calibration.nodes import _motion
from annin_ar4_calibration.nodes._detection_quality import DetectionQualityRecorder
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description


class KinematicCollectorNode(Node):

    def __init__(self):
        super().__init__('kinematic_collector')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter('camera_optical_frame',
                               'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'calibration_target')
        self.declare_parameter('sample_file', '')
        self.declare_parameter('tf_lookup_timeout_sec', 1.0)
        self.declare_parameter('motion_timeout_sec',
                               _motion.DEFAULT_MOTION_TIMEOUT_SEC)
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('num_candidate_configs', 200)
        self.declare_parameter('min_samples', 100)
        self.declare_parameter('min_joint_range_deg', 30.0)
        self.declare_parameter('joint_margin_deg', 5.0)
        self.declare_parameter('settle_time_sec', 0.5)
        self.declare_parameter(
            'detection_quality_topic', '/target_detector/detection_quality')

        #  Visibility gating (see core/joint_sampling.py)
        self.declare_parameter('camera_info_topic',
                               '/camera/camera/color/camera_info')
        self.declare_parameter('target_size_m', [0.175, 0.245])
        self.declare_parameter('target_origin_at_center', False)
        self.declare_parameter('visibility_min_range_m', 0.10)
        self.declare_parameter('visibility_max_range_m', 1.20)
        self.declare_parameter('visibility_max_view_angle_deg', 65.0)
        self.declare_parameter('visibility_edge_margin_px', 20.0)
        self.declare_parameter('visibility_min_corner_fraction', 0.75)
        self.declare_parameter('visibility_max_draws', 200000)
        self.declare_parameter('visibility_setup_timeout_sec', 60.0)
        self.declare_parameter('calibration_type', 'eye_in_hand')
        self.declare_parameter('hand_eye_result_file', '')

        # Circuit breaker (see _run_auto_sequence_cb)
        self.declare_parameter('max_consecutive_motion_failures', 10)
        self.declare_parameter('max_recovery_attempts', 2)
        self.declare_parameter(
            'recovery_action_name', '/joint_trajectory_controller/follow_joint_trajectory')
        self.declare_parameter('recovery_duration_sec', 5.0)
        self.declare_parameter('recovery_settle_timeout_sec', 20.0)
        self.declare_parameter('max_sequence_duration_sec', 0.0)

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.camera_optical_frame = self.get_parameter(
            'camera_optical_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.sample_file = paths.resolve(
            self.get_parameter('sample_file').value, paths.default_kinematic_sample_file)
        self.tf_lookup_timeout_sec = self.get_parameter(
            'tf_lookup_timeout_sec').value
        self.motion_timeout_sec = self.get_parameter(
            'motion_timeout_sec').value
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.num_candidate_configs = self.get_parameter(
            'num_candidate_configs').value
        self.min_samples = self.get_parameter('min_samples').value
        self.min_joint_range_deg = self.get_parameter(
            'min_joint_range_deg').value
        self.joint_margin_deg = self.get_parameter('joint_margin_deg').value
        self.settle_time_sec = self.get_parameter('settle_time_sec').value
        self.target_size_m = tuple(self.get_parameter('target_size_m').value)
        self.target_origin_at_center = self.get_parameter(
            'target_origin_at_center').value
        self.visibility_limits = joint_sampling.VisibilityLimits(
            min_range_m=self.get_parameter('visibility_min_range_m').value,
            max_range_m=self.get_parameter('visibility_max_range_m').value,
            max_view_angle_deg=self.get_parameter(
                'visibility_max_view_angle_deg').value,
            edge_margin_px=self.get_parameter(
                'visibility_edge_margin_px').value,
            min_corner_fraction=self.get_parameter('visibility_min_corner_fraction').value)
        self.visibility_max_draws = self.get_parameter(
            'visibility_max_draws').value
        self.visibility_setup_timeout_sec = self.get_parameter(
            'visibility_setup_timeout_sec').value
        self.calibration_type = self.get_parameter('calibration_type').value
        self.hand_eye_result_file = paths.resolve(
            self.get_parameter('hand_eye_result_file').value,
            paths.default_hand_eye_result_file)
        self.max_consecutive_motion_failures = self.get_parameter(
            'max_consecutive_motion_failures').value
        self.max_recovery_attempts = self.get_parameter(
            'max_recovery_attempts').value
        self.recovery_duration_sec = self.get_parameter(
            'recovery_duration_sec').value
        self.recovery_settle_timeout_sec = self.get_parameter(
            'recovery_settle_timeout_sec').value
        self.max_sequence_duration_sec = self.get_parameter(
            'max_sequence_duration_sec').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detection_quality = DetectionQualityRecorder(
            self, self.get_parameter('detection_quality_topic').value)

        self._latest_joint_state = None
        self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self._joint_state_cb, 10)

        self._latest_camera_info = None
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self._camera_info_cb, 10)

        urdf_xml = fetch_robot_description(self)
        self.joint_frames = kinematic_model.parse_urdf_chain(
            urdf_xml, self.arm_joint_names)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name='ar_manipulator',
            callback_group=ReentrantCallbackGroup(),
            # Plan and execute inside move_group (a single action), instead of
            # pymoveit2's default of planning client-side and executing separately
            use_move_group_action=True,
        )

        self.recovery_client = ActionClient(
            self, FollowJointTrajectory,
            self.get_parameter('recovery_action_name').value,
            callback_group=ReentrantCallbackGroup())

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

    # TF/joint-state helpers

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._latest_camera_info = msg

    def _load_camera_extrinsic(self):
        """The hand-eye transform: tool -> camera for eye_in_hand, base ->
        camera for eye_to_hand. Returns ``(R, t, None)`` or ``(None, None,
        reason)``"""
        parent = self.tool_frame if self.calibration_type == 'eye_in_hand' else self.base_frame
        try:
            ts = self.tf_buffer.lookup_transform(
                parent, self.camera_optical_frame, rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec))
            R, t = geometry.transform_stamped_to_rt(ts)
            return R, t, None
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

        try:
            result = storage.load_result(
                self.hand_eye_result_file, key='hand_eye_calibration',
                required_keys=('transform', 'parent_frame', 'child_frame'))
        except storage.ResultFileError as exc:
            return None, None, f'hand-eye result at {self.hand_eye_result_file} unreadable ({exc})'

        if result is None:
            return None, None, (
                f'{parent} -> {self.camera_optical_frame} is not on TF and there is no '
                f'hand-eye result at {self.hand_eye_result_file} - run hand_eye.launch.py '
                f'and compute_calibration first (WP4 depends on WP3)')
        if result.get('calibration_type') != self.calibration_type:
            return None, None, (
                f'calibration_type mismatch: {self.hand_eye_result_file} was computed with '
                f'{result.get("calibration_type")!r}, this node is configured with '
                f'{self.calibration_type!r}')

        transform = result['transform']
        rot, trans = transform['rotation'], transform['translation']
        R = geometry.quat_to_rotation_matrix(
            rot['x'], rot['y'], rot['z'], rot['w'])
        return R, np.array([trans['x'], trans['y'], trans['z']]), None

    def _await_visibility_inputs(self):
        """Block until the camera pipeline is actually live, or the setup budget
        runs out. Returns ``(R_cam_target, t_cam_target, theta, None)`` or
        ``(None, None, None, reason)`` """
        deadline = time.time() + self.visibility_setup_timeout_sec
        info_topic = self.get_parameter('camera_info_topic').value
        reason = f'timed out after {self.visibility_setup_timeout_sec:.0f}s'

        while time.time() < deadline:
            if self._latest_camera_info is None:
                reason = f"no CameraInfo received on '{info_topic}'"
                time.sleep(0.2)
                continue
            try:
                ts = self._lookup_target_pose(self.get_clock().now())
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                reason = f'no current detection of {self.target_frame} ({exc})'
                continue

            theta = self._current_theta()
            if theta is None:
                reason = 'no /joint_states received yet for all arm_joint_names'
                time.sleep(0.2)
                continue

            R_cam_target, t_cam_target = geometry.transform_stamped_to_rt(ts)
            return R_cam_target, t_cam_target, theta, None

        return None, None, None, reason

    def _build_visibility_models(self):
        R_mount_cam, t_mount_cam, why = self._load_camera_extrinsic()
        if R_mount_cam is None:
            return None, None, why

        R_cam_target, t_cam_target, theta, why = self._await_visibility_inputs()
        if R_cam_target is None:
            return None, None, why

        info = self._latest_camera_info
        eye_in_hand = self.calibration_type == 'eye_in_hand'
        camera = joint_sampling.CameraModel(
            R=R_mount_cam, t=t_mount_cam, on_tool=eye_in_hand,
            fx=float(info.k[0]), fy=float(info.k[4]),
            cx=float(info.k[2]), cy=float(info.k[5]),
            width=int(info.width), height=int(info.height))

        R_base_tool, t_base_tool = kinematic_model.forward_kinematics(
            theta, self.joint_frames)
        R_target, t_target = joint_sampling.locate_target(
            eye_in_hand, R_base_tool, t_base_tool, R_mount_cam, t_mount_cam,
            R_cam_target, t_cam_target)

        normal_sign = joint_sampling.infer_normal_sign(
            R_cam_target, t_cam_target)
        self.get_logger().info(
            f'Target marker face: board {"+Z" if normal_sign > 0 else "-Z"} '
            f'(learned from the seed detection)')

        target = joint_sampling.TargetModel(
            R=R_target, t=t_target, on_tool=not eye_in_hand,
            size_m=self.target_size_m, origin_at_center=self.target_origin_at_center,
            normal_sign=normal_sign)
        return camera, target, None

    # Recovery

    def _safe_configuration(self, theta: np.ndarray) -> tuple[np.ndarray, str]:
        lo, hi = joint_sampling.sampling_bounds(
            self.joint_frames, np.radians(self.joint_margin_deg))
        clamped = np.clip(theta, lo, hi)
        if np.allclose(clamped, theta):
            return 0.5 * (lo + hi), 'mid-range'
        out = [jf.name for jf, a, b in zip(
            self.joint_frames, theta, clamped) if a != b]
        return clamped, f"clamped {', '.join(out)} back inside limits"

    def _out_of_bounds(self, theta: np.ndarray) -> list:
        """Names of joints outside their URDF limits"""
        return [jf.name for jf, v in zip(self.joint_frames, theta)
                if not (jf.lower <= v <= jf.upper)]

    def _wait_until_within_limits(self, timeout_sec: float) -> tuple[bool, str]:
        """Poll /joint_states until every joint is back inside its limits"""
        deadline = time.time() + timeout_sec
        offenders = ['(no /joint_states)']
        while time.time() < deadline:
            theta = self._current_theta()
            if theta is not None:
                offenders = self._out_of_bounds(theta)
                if not offenders:
                    return True, ''
            time.sleep(0.1)
        return False, f'still outside limits after {timeout_sec:.0f}s: {", ".join(offenders)}'

    def _recover_to_safe_configuration(self) -> tuple[bool, str]:
        """Bring the arm back to a plannable state after a run of motion
        failures, bypasses MoveIt"""
        theta = self._current_theta()
        if theta is None:
            return False, 'no /joint_states to recover from'

        safe, how = self._safe_configuration(theta)
        self.get_logger().warning(
            f'Recovering to a safe configuration ({how})')
        accepted, why = _motion.send_direct_trajectory(
            self.recovery_client, self.arm_joint_names, safe,
            duration_sec=self.recovery_duration_sec,
            timeout_sec=self.recovery_duration_sec + 25.0)
        if not accepted:
            return False, why

        return self._wait_until_within_limits(
            self.recovery_duration_sec + self.recovery_settle_timeout_sec)

    def _current_theta(self):
        """Returns (6,) joint angles in arm_joint_names order, or None"""
        msg = self._latest_joint_state
        if msg is None:
            return None
        try:
            indices = [msg.name.index(n) for n in self.arm_joint_names]
        except ValueError:
            return None
        return np.array([msg.position[i] for i in indices])

    def _lookup_target_pose(self, when=None):
        return self.tf_buffer.lookup_transform(
            self.camera_optical_frame, self.target_frame, when or rclpy.time.Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout_sec))

    def _append_sample(self, theta: np.ndarray, target_ts) -> int:
        t_t, t_q = target_ts.transform.translation, target_ts.transform.rotation
        row = np.concatenate(
            [theta, [t_t.x, t_t.y, t_t.z, t_q.x, t_q.y, t_q.z, t_q.w]])
        storage.save_meta(self.sample_file, {
            'base_frame': self.base_frame, 'tool_frame': self.tool_frame,
            'camera_optical_frame': self.camera_optical_frame,
            'target_frame': self.target_frame,
            'arm_joint_names': list(self.arm_joint_names),
        })
        count = storage.append_sample(self.sample_file, row, ncols=13)
        self.detection_quality.record(self.sample_file, count)
        return count

    # Services

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
        backup_path = storage.reset_samples(
            self.sample_file, keep_backup=True, ncols=13)
        storage.reset_detections(self.sample_file)
        response.success = True
        if backup_path:
            response.message = f'Cleared samples (backup: {backup_path})'
        else:
            response.message = 'Cleared samples (no prior samples existed)'
        return response

    def _run_auto_sequence_cb(self, request, response):
        camera, target, why = self._build_visibility_models()
        if camera is None:
            response.success = False
            response.message = (
                f'Cannot generate candidate configurations after waiting '
                f'{self.visibility_setup_timeout_sec:.0f}s for the camera pipeline: {why}. '
                f'This sequence filters candidates by whether the target would be in frame, '
                f'which needs the camera intrinsics and a live detection - check that the '
                f'detector is publishing and that the arm is at a pose where it sees the '
                f'board, then call this again.')
            return response

        candidates, stats = joint_sampling.sample_visible_joint_configs(
            self.joint_frames, self.num_candidate_configs, camera, target,
            limits=self.visibility_limits, margin_deg=self.joint_margin_deg,
            rng=self.rng, max_draws=self.visibility_max_draws)
        self.get_logger().info(f'Candidate generation: {stats.summary()}')

        if not candidates:
            response.success = False
            response.message = (
                f'No candidate configuration can see the target ({stats.summary()}). '
                f'Check that the board is within visibility_max_range_m '
                f'({self.visibility_limits.max_range_m} m) of the arm, that '
                f'target_size_m {list(self.target_size_m)} matches the real board, and '
                f'that the marker face learned from the seed detection (board '
                f'{"+Z" if target.normal_sign > 0 else "-Z"}) is the printed one - a '
                f'sign learned from a spurious detection rejects every pose.')
            return response

        deadline = (time.time() + self.max_sequence_duration_sec
                    if self.max_sequence_duration_sec > 0 else None)

        collected_thetas = []
        skipped_unreachable = 0
        skipped_not_visible = 0
        consecutive_failures = 0
        recovery_attempts = 0
        aborted = None

        for theta in candidates:
            if len(collected_thetas) >= self.min_samples and self._range_gate_passes(
                    np.array(collected_thetas)):
                break

            if deadline is not None and time.time() > deadline:
                aborted = (f'time budget of {self.max_sequence_duration_sec:.0f}s exhausted '
                           f'before the sample target was met')
                break

            self.moveit2.move_to_configuration(
                joint_positions=[float(v) for v in theta], joint_names=self.arm_joint_names)
            moved, reason = _motion.wait_for_motion(
                self.moveit2, self.motion_timeout_sec)
            if not moved:
                skipped_unreachable += 1
                consecutive_failures += 1
                self.get_logger().warning(
                    f'Candidate configuration skipped: {reason}')

                # Circuit breaker
                if consecutive_failures >= self.max_consecutive_motion_failures:
                    if recovery_attempts >= self.max_recovery_attempts:
                        aborted = (
                            f'{consecutive_failures} consecutive motion failures with '
                            f'{recovery_attempts} recovery attempts already spent')
                        break
                    recovery_attempts += 1
                    self.get_logger().warning(
                        f'{consecutive_failures} consecutive motion failures - recovery '
                        f'attempt {recovery_attempts}/{self.max_recovery_attempts}')
                    recovered, recovery_why = self._recover_to_safe_configuration()
                    if not recovered:
                        aborted = f'motion recovery failed: {recovery_why}'
                        break
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            time.sleep(self.settle_time_sec)

            actual_theta = self._current_theta()
            try:
                # Require a detection stamped after the settle, so the joint
                # angles just read and the image cannot come from different
                # arm poses.
                target_ts = self._lookup_target_pose(self.get_clock().now())
            except (LookupException, ConnectivityException, ExtrapolationException):
                skipped_not_visible += 1
                self.get_logger().warning(
                    'No fresh target detection at candidate pose, skipping')
                continue

            if actual_theta is None:
                skipped_not_visible += 1
                continue

            self._append_sample(actual_theta, target_ts)
            collected_thetas.append(actual_theta)

        attempted = len(collected_thetas) + \
            skipped_unreachable + skipped_not_visible
        collected = len(collected_thetas)
        range_ok = self._range_gate_passes(
            np.array(collected_thetas)) if collected else False

        response.success = collected >= self.min_samples and range_ok and aborted is None
        message = (
            f'collected {collected}/{attempted} attempted configs '
            f'({skipped_unreachable} unreachable, {skipped_not_visible} target not visible); '
            f'candidates: {stats.summary()}')
        if aborted:
            message += f'; sequence aborted early: {aborted}'
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
