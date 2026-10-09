"""Drives the arm so the *true* tool tip touches the reference point, for
automated WP2 (TCP pivot) data collection in simulation"""
import time

import numpy as np
import rclpy
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.optimize import least_squares
from std_srvs.srv import Trigger

from annin_ar4_calibration.core import geometry, kinematic_model, perturbation
from annin_ar4_calibration.nodes import _motion
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description

#: metres-per-radian weight making the orientation residual commensurate with
#: the position residual. Position is the hard constraint - a touch is accepted
#: only within `position_tolerance_m` (0.05 mm) - while the requested
#: orientation is a soft goal, drawn at random purely to spread the touches.
#: Weighting them 10:1 let the solver buy orientation with position and miss
#: the tolerance; 100:1 does not, and measurably improves the conditioning it
#: was there to protect.
#:
#: It cannot go to zero. With no orientation term at all the solver returns the
#: same tool axis every time: orientation spread 0 deg, smallest singular value
#: 0, condition number 1e13 - a rank-deficient pivot fit that still reports a
#: tiny residual, which is exactly the failure `pivot_conditioning` exists to
#: catch.
LEVER_ARM_M = 0.01


class TcpSimPoserNode(Node):

    def __init__(self):
        super().__init__('tcp_sim_poser')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('move_group', 'ar_manipulator')
        # Must match calibration.world's tcp_reference model pose
        self.declare_parameter('reference_point', [0.30, 0.10, 0.20])
        self.declare_parameter('tool_length', 0.15)
        self.declare_parameter('perturbation_file', '')
        self.declare_parameter('num_poses', 40)
        self.declare_parameter('motion_timeout_sec',
                               _motion.DEFAULT_MOTION_TIMEOUT_SEC)
        self.declare_parameter('touch_noise_std_m', 0.0)
        self.declare_parameter('max_tilt_deg', 35.0)
        self.declare_parameter('position_tolerance_m', 5.0e-5)
        self.declare_parameter('ik_restarts', 10)
        self.declare_parameter('settle_time_sec', 2.0)
        self.declare_parameter('joint_margin_deg', 5.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('collect_service', '/collector/collect_sample')

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.arm_joint_names = list(
            self.get_parameter('arm_joint_names').value)
        self.reference_point = np.array(
            self.get_parameter('reference_point').value, dtype=float)
        self.tool_offset = np.array(
            [0.0, 0.0, self.get_parameter('tool_length').value])
        self.num_poses = self.get_parameter('num_poses').value
        self.motion_timeout_sec = self.get_parameter(
            'motion_timeout_sec').value
        self.touch_noise_std = self.get_parameter('touch_noise_std_m').value
        self.max_tilt = np.radians(self.get_parameter('max_tilt_deg').value)
        self.position_tolerance = self.get_parameter(
            'position_tolerance_m').value
        self.ik_restarts = int(self.get_parameter('ik_restarts').value)
        self.settle_time = self.get_parameter('settle_time_sec').value
        self.joint_margin = np.radians(
            self.get_parameter('joint_margin_deg').value)
        pose_seed, ik_seed = np.random.SeedSequence(
            self.get_parameter('seed').value).spawn(2)
        self.rng = np.random.default_rng(pose_seed)
        self.ik_rng = np.random.default_rng(ik_seed)

        urdf = fetch_robot_description(self)
        self.joint_frames = kinematic_model.parse_urdf_chain(
            urdf, self.arm_joint_names)

        perturbation_file = self.get_parameter('perturbation_file').value
        if perturbation_file:
            self.corrections = perturbation.load(
                perturbation_file, self.arm_joint_names)
            self.get_logger().info(f'Solving IK against the perturbed plant '
                                   f'({perturbation_file})')
        else:
            self.corrections = None
            self.get_logger().warning(
                'No perturbation_file given - solving IK against the nominal model. '
                'That is correct only for an unperturbed (zero.yaml) run; with a '
                'perturbed robot it would place the nominal tip on the target and '
                'hide the very error WP2 is meant to reveal.')

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name=self.get_parameter('move_group').value,
            callback_group=callback_group,
            # Plan and execute inside move_group (a single action), instead of
            # pymoveit2's default of planning client-side and executing separately
            use_move_group_action=True,
        )
        self.collect_client = self.create_client(
            Trigger, self.get_parameter('collect_service').value,
            callback_group=callback_group)

        self.create_service(Trigger, '~/run_sequence', self._run_cb,
                            callback_group=callback_group)
        self.get_logger().info(
            f'TCP sim poser ready: reference_point={self.reference_point.tolist()}, '
            f'tool_length={self.tool_offset[2]} m, num_poses={self.num_poses}. '
            f'Call ~/run_sequence to start.')

    # solving

    def _true_pose(self, theta: np.ndarray):
        return kinematic_model.forward_kinematics(theta, self.joint_frames, self.corrections)

    def _solve_ik(self, R_desired: np.ndarray, target: np.ndarray, seed: np.ndarray):
        """Find joint angles putting the TRUE tool tip on `target` with roughly
        the requested tool orientation. 6 residuals, 6 unknowns, bounded by the
        URDF's own joint limits."""
        def residual(theta):
            R, t = self._true_pose(theta)
            position_error = t + R @ self.tool_offset - target
            orientation_error = geometry.rotation_matrix_to_rotvec(
                R_desired.T @ R)
            return np.concatenate([position_error, LEVER_ARM_M * orientation_error])

        lower = np.array(
            [jf.lower + self.joint_margin for jf in self.joint_frames])
        upper = np.array(
            [jf.upper - self.joint_margin for jf in self.joint_frames])

        def attempt(start):
            solution = least_squares(
                residual, np.clip(start, lower, upper), bounds=(lower, upper),
                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            return solution.x, float(np.linalg.norm(residual(solution.x)[0:3]))

        theta, error = attempt(seed)
        restarts = 0
        while error > self.position_tolerance and restarts < self.ik_restarts:
            theta, error = attempt(self.ik_rng.uniform(lower, upper))
            restarts += 1
        if restarts:
            self.get_logger().debug(
                f'IK needed {restarts} restart(s), final residual {error * 1e6:.1f} um')
        return theta, error

    def _sample_orientation(self) -> np.ndarray:
        """A tool orientation tilted away from straight-down by up to max_tilt"""
        # Nominal: tool +Z pointing down (-Z world), i.e. rotate pi about X.
        base = geometry.axis_angle_to_rotation_matrix(
            np.array([1.0, 0.0, 0.0]), np.pi)
        tilt_axis = self.rng.normal(size=3)
        tilt_axis[2] = 0.0
        norm = np.linalg.norm(tilt_axis)
        if norm < 1e-9:
            tilt_axis = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        tilt = geometry.axis_angle_to_rotation_matrix(
            tilt_axis / norm, self.rng.uniform(-self.max_tilt, self.max_tilt))
        spin = geometry.axis_angle_to_rotation_matrix(
            np.array([0.0, 0.0, 1.0]), self.rng.uniform(-np.pi, np.pi))
        return tilt @ base @ spin

    # driving

    def _run_cb(self, request, response):
        collected = skipped_ik = skipped_motion = skipped_collect = 0
        seed = np.zeros(len(self.arm_joint_names))

        if not self.collect_client.wait_for_service(timeout_sec=5.0):
            response.success = False
            response.message = (
                f'{self.collect_client.srv_name} not available - is tcp.launch.py running?')
            return response

        for index in range(self.num_poses):
            R_desired = self._sample_orientation()
            target = self.reference_point + (
                self.rng.normal(0.0, self.touch_noise_std, 3)
                if self.touch_noise_std > 0.0 else 0.0)

            label = f'pose {index + 1}/{self.num_poses}'
            theta, position_error = self._solve_ik(R_desired, target, seed)
            if position_error > self.position_tolerance:
                skipped_ik += 1
                self.get_logger().warning(
                    f'{label}: SKIP unreachable - IK residual '
                    f'{position_error * 1e3:.2f} mm > {self.position_tolerance * 1e3:.2f} mm '
                    f'tolerance')
                continue
            seed = theta

            self.get_logger().info(f'{label}: executing (ik residual '
                                   f'{position_error * 1e6:.1f} um)')
            self.moveit2.move_to_configuration(
                joint_positions=theta.tolist(), joint_names=self.arm_joint_names)
            moved, reason = _motion.wait_for_motion(
                self.moveit2, self.motion_timeout_sec)
            if not moved:
                skipped_motion += 1
                self.get_logger().warning(f'{label}: SKIP {reason}')
                continue

            time.sleep(self.settle_time)

            future = self.collect_client.call_async(Trigger.Request())
            while rclpy.ok() and not future.done():
                time.sleep(0.02)
            result = future.result()
            if result is not None and result.success:
                collected += 1
                self.get_logger().info(
                    f'{label}: collected ({collected} so far)')
            else:
                skipped_collect += 1
                # The collector's own reason for refusing the touch
                reason = 'no response' if result is None else (
                    result.message or 'no reason given')
                self.get_logger().warning(
                    f'{label}: SKIP collector refused - {reason}')

        response.success = collected > 0
        response.message = (
            f'Collected {collected}/{self.num_poses} touches '
            f'(skipped: {skipped_ik} unreachable, {skipped_motion} motion failures, '
            f'{skipped_collect} collector rejections). Now call '
            f'/calibration/compute_calibration.')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TcpSimPoserNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
