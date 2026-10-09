"""Records the simulation's ground truth, and scores the calibration results against it"""
import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from annin_ar4_calibration.core import geometry, metrics, paths, perturbation, storage

RESULT_KEY = 'ground_truth'

_transform_dict = metrics.transform_dict
_pose_error = metrics.pose_error
_rt_from_result = metrics.rt_from_transform_dict


class GroundTruthNode(Node):

    def __init__(self):
        super().__init__('ground_truth')

        self.declare_parameter('calibration_type', 'eye_to_hand')
        self.declare_parameter('ar_model', '')
        self.declare_parameter('perturbation_file', '')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tool_frame', 'ee_link')
        self.declare_parameter(
            'camera_gt_frame', 'sim_camera_color_optical_frame_gt')
        self.declare_parameter('board_gt_frame', 'charuco_board_frame_gt')
        self.declare_parameter('tcp_tip_gt_frame', 'tcp_tip_gt')
        self.declare_parameter('arm_joint_names',
                               list(perturbation.DEFAULT_JOINT_NAMES))
        self.declare_parameter('hand_eye_result_file', '')
        self.declare_parameter('kinematic_result_file', '')
        self.declare_parameter('tcp_result_file', '')
        self.declare_parameter('output_file', '')
        self.declare_parameter('tf_settle_sec', 3.0)

        self.calibration_type = self.get_parameter('calibration_type').value
        self.ar_model = self.get_parameter('ar_model').value
        self.perturbation_file = self.get_parameter('perturbation_file').value
        self.base_frame = self.get_parameter('base_frame').value
        self.tool_frame = self.get_parameter('tool_frame').value
        self.camera_gt_frame = self.get_parameter('camera_gt_frame').value
        self.board_gt_frame = self.get_parameter('board_gt_frame').value
        self.tcp_tip_gt_frame = self.get_parameter('tcp_tip_gt_frame').value
        self.arm_joint_names = list(
            self.get_parameter('arm_joint_names').value)
        self.hand_eye_result_file = paths.resolve(
            self.get_parameter('hand_eye_result_file').value,
            paths.default_hand_eye_result_file)
        self.kinematic_result_file = paths.resolve(
            self.get_parameter('kinematic_result_file').value,
            paths.default_kinematic_result_file)
        self.tcp_result_file = paths.resolve(
            self.get_parameter('tcp_result_file').value, paths.default_result_file)
        self.output_file = paths.resolve(
            self.get_parameter('output_file').value, paths.default_ground_truth_file)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_service(Trigger, '~/compare', self._compare_cb)
        self.create_service(Trigger, '~/write', self._write_cb)

        self._startup_timer = self.create_timer(
            float(self.get_parameter('tf_settle_sec').value), self._startup_write)

        self.get_logger().info(
            f'Ground truth ready: calibration_type={self.calibration_type}, '
            f'perturbation_file={self.perturbation_file or "(none)"}, '
            f'output_file={self.output_file}')

    # truth

    def _lookup(self, parent: str, child: str):
        ts = self.tf_buffer.lookup_transform(
            parent, child, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0))
        return geometry.transform_stamped_to_rt(ts)

    def _hand_eye_parent(self) -> str:
        return self.tool_frame if self.calibration_type == 'eye_in_hand' else self.base_frame

    def _mount_parent(self) -> str:
        return self.base_frame if self.calibration_type == 'eye_in_hand' else self.tool_frame

    def collect(self) -> dict:
        document = {
            'calibration_type': self.calibration_type,
            'ar_model': self.ar_model,
            'perturbation_file': self.perturbation_file,
            'timestamp': datetime.datetime.now(datetime.timezone.utc)
            .strftime('%Y-%m-%dT%H:%M:%SZ'),
            'note': ('Simulation ground truth. Kinematic corrections are the injected '
                     'perturbation verbatim (same parameterization as '
                     'kinematic_result.yaml); the transforms are read from the nominal '
                     'URDF via the _gt TF frames.'),
        }

        if self.perturbation_file:
            try:
                corrections = perturbation.load(
                    self.perturbation_file, self.arm_joint_names)
            except (OSError, ValueError) as exc:
                self.get_logger().error(
                    f'Could not read perturbation file: {exc}')
            else:
                document['kinematic_calibration'] = {
                    'arm_joint_names': self.arm_joint_names,
                    'corrections': perturbation.to_yaml_dict(
                        corrections, self.arm_joint_names)[perturbation.ROOT_KEY]['joints'],
                }

        try:
            parent = self._hand_eye_parent()
            R, t = self._lookup(parent, self.camera_gt_frame)
            document['hand_eye_calibration'] = {
                'parent_frame': parent,
                'child_frame': 'camera_color_optical_frame',
                'gt_frame': self.camera_gt_frame,
                'transform': _transform_dict(R, t),
            }
        except Exception as exc:
            self.get_logger().warning(
                f'No hand-eye ground truth ({self.camera_gt_frame}): {exc}')

        try:
            parent = self._mount_parent()
            R, t = self._lookup(parent, self.board_gt_frame)
            document.setdefault('kinematic_calibration', {})['mount_offset'] = {
                'role': 'base_T_marker' if self.calibration_type == 'eye_in_hand'
                        else 'ee_T_marker',
                'parent_frame': parent,
                'gt_frame': self.board_gt_frame,
                **_transform_dict(R, t),
            }
        except Exception as exc:
            self.get_logger().warning(
                f'No mount-offset ground truth ({self.board_gt_frame}): {exc}')

        try:
            _, t = self._lookup(self.tool_frame, self.tcp_tip_gt_frame)
            document['tcp_calibration'] = {
                'parent_frame': self.tool_frame,
                'gt_frame': self.tcp_tip_gt_frame,
                # The pivot solver is translation-only, so only the offset is truth.
                'offset': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
            }
        except Exception as exc:
            self.get_logger().warning(
                f'No TCP ground truth ({self.tcp_tip_gt_frame}): {exc}')

        return document

    # comparison

    def _compare_hand_eye(self, truth: dict) -> dict | None:
        result = storage.load_result(
            self.hand_eye_result_file, key='hand_eye_calibration',
            required_keys=('transform', 'parent_frame', 'child_frame'))
        if result is None:
            return None
        if result['parent_frame'] != truth['parent_frame']:
            return {'error': f"result is {result['parent_frame']}-relative but ground truth "
                    f"is {truth['parent_frame']}-relative; calibration_type "
                    f"almost certainly differs between the run and this node"}
        true_R, true_t = _rt_from_result(truth['transform'])
        est_R, est_t = _rt_from_result(result['transform'])
        return {**_pose_error(true_R, true_t, est_R, est_t),
                'consistency_rms_mm': float(result.get('consistency_rms_m', 0.0) * 1000.0),
                'num_samples': result.get('num_samples')}

    def _compare_kinematic(self, truth: dict) -> dict | None:
        result = storage.load_result(
            self.kinematic_result_file, key='kinematic_calibration',
            required_keys=('corrections', 'mount_offset', 'calibration_type'))
        if result is None:
            return None

        comparison = {'calibration_type': result['calibration_type']}
        if 'corrections' in truth:
            comparison['per_joint'] = metrics.per_joint_correction_error(
                truth['corrections'], result['corrections'], self.arm_joint_names)

        if 'mount_offset' in truth:
            true_R, true_t = _rt_from_result(truth['mount_offset'])
            est_R, est_t = _rt_from_result(result['mount_offset'])
            comparison['mount_offset'] = _pose_error(
                true_R, true_t, est_R, est_t)

        method = result.get('methods', {}).get(
            result.get('primary_method', ''), {})
        comparison['primary_method'] = result.get('primary_method')
        for key in ('rms_before_m', 'rms_after_m'):
            if key in method:
                comparison[key[:-2] + '_mm'] = method[key] * 1000.0
        if 'condition_number' in method:
            comparison['condition_number'] = method['condition_number']
        return comparison

    def _compare_tcp(self, truth: dict) -> dict | None:
        result = storage.load_result(self.tcp_result_file)
        if result is None:
            return None
        true_t = np.array(
            [truth['offset']['x'], truth['offset']['y'], truth['offset']['z']])
        est_t = np.array(
            [result['offset']['x'], result['offset']['y'], result['offset']['z']])
        return {
            'translation_error_mm': metrics.translation_error_mm(true_t, est_t),
            'fitted_base_point': result.get('base_point'),
            'rms_error_mm': float(result.get('rms_error', 0.0) * 1000.0),
            'num_samples': result.get('num_samples'),
        }

    # hooks

    def _write(self) -> dict:
        document = self.collect()
        storage.save_result(self.output_file, {RESULT_KEY: document})
        return document

    def _startup_write(self) -> None:
        self._startup_timer.cancel()
        try:
            self._write()
            self.get_logger().info(f'Wrote ground truth to {self.output_file}')
        except Exception as exc:
            self.get_logger().error(f'Could not write ground truth: {exc}')

    def _write_cb(self, request, response):
        try:
            self._write()
        except Exception as exc:
            response.success = False
            response.message = f'Failed: {exc}'
            return response
        response.success = True
        response.message = f'Wrote ground truth to {self.output_file}'
        return response

    def _compare_cb(self, request, response):
        truth = self._write()
        comparison = {}
        summary = []

        for label, truth_key, compare_fn in (
                ('hand_eye', 'hand_eye_calibration', self._compare_hand_eye),
                ('kinematic', 'kinematic_calibration', self._compare_kinematic),
                ('tcp', 'tcp_calibration', self._compare_tcp)):
            if truth_key not in truth:
                continue
            try:
                entry = compare_fn(truth[truth_key])
            except storage.ResultFileError as exc:
                self.get_logger().warning(f'{label}: {exc}')
                continue
            if entry is None:
                continue
            comparison[label] = entry
            if 'translation_error_mm' in entry:
                summary.append(
                    f'{label} {entry["translation_error_mm"]:.2f} mm')

        storage.save_result(self.output_file, {
                            RESULT_KEY: truth, 'comparison': comparison})

        response.success = bool(comparison)
        response.message = (
            f'Compared {len(comparison)} result(s) against ground truth '
            f'({", ".join(summary) if summary else "see file"}); wrote {self.output_file}'
            if comparison else
            'No calibration results found yet - run a work package and call its '
            'compute_calibration first.')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
