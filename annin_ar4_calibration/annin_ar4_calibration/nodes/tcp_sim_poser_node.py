"""Drives the arm so the *true* tool tip touches the reference point, for
automated WP2 (TCP pivot) data collection in simulation.

The README's manual procedure - jog with the RViz marker until the needle
visually meets the sphere, then call collect_sample - works in simulation too,
and should stay the documented fallback. But the simulated tool has no
collision geometry, so there is no contact feedback: it is pure eyeballing, and
30 touches at 1-2 mm each would dominate every other error term and bury the
injected perturbation entirely. This node removes that error source.

**The critical subtlety is which tip gets placed on the target.** The obvious
implementation - ask MoveIt for a pose that puts the tip on the reference point
- places the *nominal* tip there. The pivot equations would then be exactly
satisfied in nominal coordinates, the solver would recover the ground-truth
offset with zero residual, and the kinematic perturbation would be completely
invisible. So this node solves inverse kinematics against the TRUE plant
(nominal joint frames plus the injected corrections) and commands the resulting
joint angles directly, bypassing MoveIt's nominal IK. The nominal tip then
misses the target by exactly the kinematic error, which is the signal WP2 is
supposed to see.

    ros2 run annin_ar4_calibration tcp_sim_poser --ros-args \
        -p num_poses:=40 -p touch_noise_std_m:=0.0 \
        -p perturbation_file:=<...>/example_1mm.yaml

`touch_noise_std_m` injects a controlled "human touch error"; sweeping it gives
a WP2 accuracy-versus-touch-precision curve that a human simply cannot produce
repeatably.
"""
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
from annin_ar4_calibration.nodes._robot_description import fetch_robot_description

#: metres-per-radian weight making the orientation residual commensurate with
#: the position residual. Roughly the arm's wrist scale - the exact value only
#: trades position accuracy against orientation spread.
LEVER_ARM_M = 0.1


class TcpSimPoserNode(Node):

    def __init__(self):
        super().__init__('tcp_sim_poser')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('move_group', 'ar_manipulator')
        # Must match calibration.world's tcp_reference model pose. base_link
        # coincides with the world origin, so these are the same numbers.
        self.declare_parameter('reference_point', [0.30, 0.10, 0.20])
        self.declare_parameter('tool_length', 0.15)
        self.declare_parameter('perturbation_file', '')
        self.declare_parameter('num_poses', 40)
        self.declare_parameter('touch_noise_std_m', 0.0)
        self.declare_parameter('max_tilt_deg', 35.0)
        self.declare_parameter('position_tolerance_m', 5.0e-5)
        self.declare_parameter('settle_time_sec', 2.0)
        self.declare_parameter('joint_margin_deg', 5.0)
        self.declare_parameter('seed', 0)
        self.declare_parameter('collect_service', '/collector/collect_sample')

        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.arm_joint_names = list(self.get_parameter('arm_joint_names').value)
        self.reference_point = np.array(self.get_parameter('reference_point').value, dtype=float)
        self.tool_offset = np.array([0.0, 0.0, self.get_parameter('tool_length').value])
        self.num_poses = self.get_parameter('num_poses').value
        self.touch_noise_std = self.get_parameter('touch_noise_std_m').value
        self.max_tilt = np.radians(self.get_parameter('max_tilt_deg').value)
        self.position_tolerance = self.get_parameter('position_tolerance_m').value
        self.settle_time = self.get_parameter('settle_time_sec').value
        self.joint_margin = np.radians(self.get_parameter('joint_margin_deg').value)
        self.rng = np.random.default_rng(self.get_parameter('seed').value)

        urdf = fetch_robot_description(self)
        self.joint_frames = kinematic_model.parse_urdf_chain(urdf, self.arm_joint_names)

        # The nominal chain plus the injected error IS the simulated robot.
        # Without this the whole exercise measures nothing.
        perturbation_file = self.get_parameter('perturbation_file').value
        if perturbation_file:
            self.corrections = perturbation.load(perturbation_file, self.arm_joint_names)
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

    # ---------------------------------------------------------------- solving

    def _true_pose(self, theta: np.ndarray):
        return kinematic_model.forward_kinematics(theta, self.joint_frames, self.corrections)

    def _solve_ik(self, R_desired: np.ndarray, target: np.ndarray, seed: np.ndarray):
        """Find joint angles putting the TRUE tool tip on `target` with roughly
        the requested tool orientation. 6 residuals, 6 unknowns, bounded by the
        URDF's own joint limits."""
        def residual(theta):
            R, t = self._true_pose(theta)
            position_error = t + R @ self.tool_offset - target
            orientation_error = geometry.rotation_matrix_to_rotvec(R_desired.T @ R)
            return np.concatenate([position_error, LEVER_ARM_M * orientation_error])

        lower = np.array([jf.lower + self.joint_margin for jf in self.joint_frames])
        upper = np.array([jf.upper - self.joint_margin for jf in self.joint_frames])
        solution = least_squares(
            residual, np.clip(seed, lower, upper), bounds=(lower, upper),
            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        return solution.x, float(np.linalg.norm(residual(solution.x)[0:3]))

    def _sample_orientation(self) -> np.ndarray:
        """A tool orientation tilted away from straight-down by up to max_tilt.

        Orientation spread is what makes the pivot problem well posed at all -
        a set of near-parallel touches gives a rank-deficient fit no matter how
        many samples it contains.
        """
        # Nominal: tool +Z pointing down (-Z world), i.e. rotate pi about X.
        base = geometry.axis_angle_to_rotation_matrix(np.array([1.0, 0.0, 0.0]), np.pi)
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

    # ----------------------------------------------------------------- driving

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

            theta, position_error = self._solve_ik(R_desired, target, seed)
            if position_error > self.position_tolerance:
                skipped_ik += 1
                continue
            seed = theta

            self.moveit2.move_to_configuration(
                joint_positions=theta.tolist(), joint_names=self.arm_joint_names)
            if not self.moveit2.wait_until_executed():
                skipped_motion += 1
                continue

            # Wall-clock on purpose: matches the collectors, and a sim-time
            # sleep inside a service callback can deadlock the executor.
            time.sleep(self.settle_time)

            future = self.collect_client.call_async(Trigger.Request())
            while rclpy.ok() and not future.done():
                time.sleep(0.02)
            if future.result() is not None and future.result().success:
                collected += 1
            else:
                skipped_collect += 1

            self.get_logger().info(
                f'pose {index + 1}/{self.num_poses}: collected={collected} '
                f'(ik residual {position_error * 1e6:.1f} um)')

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
