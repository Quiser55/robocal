"""Turn one run's archived samples into metrics. No ROS runtime, no robot.

This is the module the hardware path reuses verbatim. Collection differs
between Gazebo and a physical AR4 - one is scripted, the other partly manual,
and only one has ground truth - but once the samples are on disk the analysis
is identical, which is the whole reason the benchmark can claim to measure the
same thing in both places.

It also means solver comparisons are nearly free. Choosing between five
hand-eye methods, three WP4 optimizers or a range of ridge values does not need
the robot again; it needs this module run again over samples already collected.

Every work package reports:

- **in-sample fit** - what the solvers already report, kept only so the gap
  against the held-out number is visible;
- **held-out prediction error** - K-fold, the accuracy proxy that works
  without ground truth and that rises rather than falls under overfitting;
- **bootstrap spread** - error bars on the estimate itself;
- **conditioning** - how well-posed the problem was for the data collected;
- **truth error** - only when a `ground_truth.yaml` is present, i.e. only in
  simulation.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
import yaml

from annin_ar4_calibration.benchmark import spec
from annin_ar4_calibration.core import (geometry, handeye_solver, kinematic_model,
                                        kinematic_solver, metrics, perturbation, solver,
                                        storage)

DATA_SUBDIR = os.path.join('ros_home', 'annin_ar4_calibration')
DEFAULT_WP4_BOOTSTRAP = 20


class RunArtifacts:
    """The files one run left behind, loaded lazily."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.data_dir = os.path.join(run_dir, DATA_SUBDIR)
        if not os.path.isdir(self.data_dir):
            self.data_dir = run_dir
        self.manifest = self._yaml(os.path.join(run_dir, 'manifest.yaml'))
        gt = self._yaml(os.path.join(self.data_dir, 'ground_truth.yaml'))
        self.ground_truth = (gt or {}).get('ground_truth')
        self.nominal_urdf_path = os.path.join(run_dir, 'nominal_robot.urdf')

    @staticmethod
    def _yaml(path: str):
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return None

    def path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    def result(self, name: str, key: str):
        try:
            return storage.load_result(self.path(name), key=key, required_keys=())
        except storage.ResultFileError:
            return None

    def joint_frames(self, joint_names) -> list | None:
        if not os.path.exists(self.nominal_urdf_path):
            return None
        with open(self.nominal_urdf_path) as f:
            return kinematic_model.parse_urdf_chain(f.read(), list(joint_names))

    def detections(self, sample_file: str) -> dict:
        rows = storage.load_detections(sample_file)
        if not rows:
            return {}
        values = [r['reprojection_rms_px'] for r in rows.values()
                  if np.isfinite(r['reprojection_rms_px'])]
        points = [r['num_points'] for r in rows.values()]
        out = {'num_detections': len(rows)}
        if values:
            out['reprojection_rms_px'] = metrics.summarize(values)
        if points:
            out['num_points_mean'] = float(np.mean(points))
        return out


def analyze_wp2(art: RunArtifacts, variants: list[dict], analysis: dict) -> dict:
    sample_file = art.path('samples.npy')
    samples = storage.load_samples(sample_file, ncols=7)
    n = samples.shape[0]
    if n < 4:
        return {'skipped': f'only {n} TCP samples (need >= 4)'}

    R, t = geometry.batch_samples_to_rt(samples)
    out = {
        'num_samples': n,
        'conditioning': metrics.pivot_conditioning(R, t),
        'variants': {},
    }

    truth_offset = None
    if art.ground_truth and 'tcp_calibration' in art.ground_truth:
        o = art.ground_truth['tcp_calibration']['offset']
        truth_offset = np.array([o['x'], o['y'], o['z']])

    for variant in variants:
        ransac = bool(variant.get('ransac', True))
        entry = {}

        result = solver.solve_pivot(
            R, t, ransac_enabled=ransac,
            threshold=float(variant.get('threshold', 0.002)),
            iterations=int(variant.get('iterations', 500)),
            rng=np.random.default_rng(analysis.get('bootstrap_seed', 0)))
        entry['in_sample'] = {
            'rms_inliers_mm': result.rms_inliers * 1000.0,
            'rms_all_mm': result.rms_all * 1000.0,
            'num_inliers': result.num_inliers,
            'degenerate_fallback': result.degenerate_fallback,
        }
        entry['offset_m'] = [float(v) for v in result.p_tool]

        if analysis.get('kfold'):
            fold_rms = []
            for train, test in metrics.kfold_indices(
                    n, int(analysis['kfold']),
                    np.random.default_rng(analysis.get('kfold_seed', 0))):
                p_tool, p_base, _ = solver.solve_pivot_linear(R[train], t[train])
                res = solver.residuals(R[test], t[test], p_tool, p_base)
                fold_rms.append(float(np.sqrt(np.mean(res ** 2))) * 1000.0)
            entry['held_out_rms_mm'] = metrics.summarize(fold_rms)

        if analysis.get('bootstrap'):
            offsets = []
            rng = np.random.default_rng(analysis.get('bootstrap_seed', 0))
            for idx in metrics.bootstrap_indices(n, int(analysis['bootstrap']), rng):
                p_tool, _, _ = solver.solve_pivot_linear(R[idx], t[idx])
                offsets.append(p_tool)
            offsets = np.array(offsets)
            centre = offsets.mean(axis=0)
            entry['bootstrap'] = {
                'std_mm': [float(v * 1000.0) for v in offsets.std(axis=0, ddof=1)],
                'radius_mm': metrics.summarize(
                    np.linalg.norm(offsets - centre, axis=1) * 1000.0),
            }

        if truth_offset is not None:
            entry['truth'] = {
                'translation_error_mm': metrics.translation_error_mm(
                    truth_offset, result.p_tool)}

        out['variants'][spec.variant_label(variant)] = entry

    return out


def _handeye_held_out(calibration_type, robot_R, robot_t, target_R, target_t,
                      method, n, analysis) -> dict:
    trans, rot = [], []
    for train, test in metrics.kfold_indices(
            n, int(analysis['kfold']), np.random.default_rng(analysis.get('kfold_seed', 0))):
        fit = handeye_solver.solve_hand_eye(
            calibration_type, robot_R[train], robot_t[train],
            target_R[train], target_t[train], method=method)
        train_chain = handeye_solver.chain_poses(
            calibration_type, robot_R[train], robot_t[train],
            target_R[train], target_t[train], fit.R, fit.t)
        ref_q, ref_t = handeye_solver.chain_reference(*train_chain)

        test_chain = handeye_solver.chain_poses(
            calibration_type, robot_R[test], robot_t[test],
            target_R[test], target_t[test], fit.R, fit.t)
        rms_m, rms_deg = handeye_solver.chain_deviation(*test_chain, ref_q, ref_t)
        trans.append(rms_m * 1000.0)
        rot.append(rms_deg)
    return {'translation_mm': metrics.summarize(trans),
            'rotation_deg': metrics.summarize(rot)}


def analyze_wp3(art: RunArtifacts, variants: list[dict], analysis: dict) -> dict:
    sample_file = art.path('hand_eye_samples.npy')
    samples = storage.load_samples(sample_file, ncols=14)
    n = samples.shape[0]
    if n < 4:
        return {'skipped': f'only {n} hand-eye samples (need >= 4)'}

    meta = storage.load_meta(sample_file) or {}
    calibration_type = meta.get('calibration_type', 'eye_to_hand')

    robot_R, robot_t = geometry.batch_samples_to_rt(samples[:, 0:7])
    target_R, target_t = geometry.batch_samples_to_rt(samples[:, 7:14])

    out = {
        'num_samples': n,
        'calibration_type': calibration_type,
        'conditioning': metrics.motion_axis_diversity(
            handeye_solver.hand_poses(calibration_type, robot_R, robot_t)[0]),
        'detection': art.detections(sample_file),
        'variants': {},
    }

    truth = None
    if art.ground_truth and 'hand_eye_calibration' in art.ground_truth:
        truth = metrics.rt_from_transform_dict(
            art.ground_truth['hand_eye_calibration']['transform'])

    for variant in variants:
        method = variant.get('method', 'PARK')
        entry = {'method': method}
        try:
            fit = handeye_solver.solve_hand_eye(
                calibration_type, robot_R, robot_t, target_R, target_t, method=method)
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            out['variants'][spec.variant_label(variant)] = entry
            continue

        entry['in_sample'] = {
            'consistency_rms_mm': fit.consistency_rms_m * 1000.0,
            'consistency_rms_deg': fit.consistency_rms_deg,
        }
        entry['transform'] = metrics.transform_dict(fit.R, fit.t)

        if analysis.get('kfold'):
            entry['held_out'] = _handeye_held_out(
                calibration_type, robot_R, robot_t, target_R, target_t, method, n, analysis)

        if analysis.get('bootstrap'):
            translations, rotations = [], []
            rng = np.random.default_rng(analysis.get('bootstrap_seed', 0))
            for idx in metrics.bootstrap_indices(n, int(analysis['bootstrap']), rng):
                try:
                    b = handeye_solver.solve_hand_eye(
                        calibration_type, robot_R[idx], robot_t[idx],
                        target_R[idx], target_t[idx], method=method)
                except Exception:
                    continue
                translations.append(b.t)
                rotations.append(metrics.rotation_error_deg(fit.R, b.R))
            if translations:
                translations = np.array(translations)
                entry['bootstrap'] = {
                    'translation_std_mm': [
                        float(v * 1000.0) for v in translations.std(axis=0, ddof=1)],
                    'translation_radius_mm': metrics.summarize(
                        np.linalg.norm(translations - translations.mean(axis=0), axis=1)
                        * 1000.0),
                    'rotation_deg': metrics.summarize(rotations),
                }

        if truth is not None:
            entry['truth'] = metrics.pose_error(truth[0], truth[1], fit.R, fit.t)

        out['variants'][spec.variant_label(variant)] = entry

    return out


def _wp4_solve(method, calibration_type, theta, known_R, known_t,
               cam_R, cam_t, joint_frames, fix_joint1, ridge_lambda):
    if method == 'sequential':
        return kinematic_solver.solve_sequential(
            calibration_type, theta, known_R, known_t, cam_R, cam_t,
            joint_frames, fix_joint1, ridge_lambda=ridge_lambda)
    optimizer = 'lm' if method.endswith('lm') else 'trf'
    return kinematic_solver.solve_joint(
        calibration_type, theta, known_R, known_t, cam_R, cam_t,
        joint_frames, fix_joint1, ridge_lambda=ridge_lambda, optimizer=optimizer)


def _wp4_held_out(method, calibration_type, theta, known_R, known_t, cam_R, cam_t,
                  joint_frames, fix_joint1, ridge_lambda, n, analysis) -> dict:
    corrected, nominal = [], []
    for train, test in metrics.kfold_indices(
            n, int(analysis['kfold']), np.random.default_rng(analysis.get('kfold_seed', 0))):
        try:
            fit = _wp4_solve(method, calibration_type, theta[train], known_R, known_t,
                             cam_R[train], cam_t[train], joint_frames, fix_joint1,
                             ridge_lambda)
        except Exception:
            continue
        corrected.append(kinematic_solver.evaluate(
            calibration_type, theta[test], known_R, known_t, cam_R[test], cam_t[test],
            joint_frames, fit.corrections, fit.mount_R, fit.mount_t) * 1000.0)
        nom_R, nom_t = kinematic_solver.fit_mount_offset(
            calibration_type, theta[train], known_R, known_t,
            cam_R[train], cam_t[train], joint_frames)
        nominal.append(kinematic_solver.evaluate(
            calibration_type, theta[test], known_R, known_t, cam_R[test], cam_t[test],
            joint_frames, None, nom_R, nom_t) * 1000.0)

    if not corrected:
        return {'error': 'every cross-validation fold failed to solve'}
    improvement = [nom - cor for nom, cor in zip(nominal, corrected)]
    return {
        'corrected_rms_mm': metrics.summarize(corrected),
        'nominal_rms_mm': metrics.summarize(nominal),
        'improvement_mm': metrics.summarize(improvement),
        'generalizes': bool(np.mean(improvement) > 0),
    }


def analyze_wp4(art: RunArtifacts, variants: list[dict], analysis: dict) -> dict:
    sample_file = art.path('kinematic_samples.npy')
    samples = storage.load_samples(sample_file, ncols=13)
    n = samples.shape[0]
    if n < 12:
        return {'skipped': f'only {n} kinematic samples (need >= 12)'}

    meta = storage.load_meta(sample_file) or {}
    joint_names = meta.get('arm_joint_names') or list(perturbation.DEFAULT_JOINT_NAMES)

    joint_frames = art.joint_frames(joint_names)
    if joint_frames is None:
        return {'skipped': f'no nominal_robot.urdf in {art.run_dir}; WP4 cannot be '
                           f'scored without the nominal chain the corrections are '
                           f'relative to'}

    theta = samples[:, 0:6]
    cam_R, cam_t = geometry.batch_samples_to_rt(samples[:, 6:13])

    hand_eye = art.result('hand_eye_result.yaml', 'hand_eye_calibration')
    if not hand_eye or 'transform' not in hand_eye:
        return {'skipped': 'no hand_eye_result.yaml; WP4 depends on WP3'}
    calibration_type = hand_eye.get('calibration_type', 'eye_to_hand')
    fix_joint1 = calibration_type == 'eye_in_hand'

    extrinsics = {'estimated': metrics.rt_from_transform_dict(hand_eye['transform'])}
    truth_kin = (art.ground_truth or {}).get('kinematic_calibration') or {}
    truth_he = (art.ground_truth or {}).get('hand_eye_calibration')
    if truth_he:
        extrinsics['truth'] = metrics.rt_from_transform_dict(truth_he['transform'])

    true_corrections = None
    if 'corrections' in truth_kin:
        true_corrections = perturbation.corrections_from_dict(
            truth_kin['corrections'], joint_names)

    out = {
        'num_samples': n,
        'calibration_type': calibration_type,
        'fixed_joint1': fix_joint1,
        'axis_alignment': kinematic_model.axis_alignment_diagnostic(joint_frames),
        'joint_range_deg': [
            float(np.degrees(theta[:, j].max() - theta[:, j].min())) for j in range(6)],
        'detection': art.detections(sample_file),
        'variants': {},
    }

    bootstrap_n = int(analysis.get(
        'bootstrap_wp4', min(int(analysis.get('bootstrap', 0) or 0), DEFAULT_WP4_BOOTSTRAP)))

    for variant in variants:
        method = variant.get('method', 'joint_trf')
        ridge_lambda = float(variant.get('ridge_lambda', 1e-6))

        for source, (known_R, known_t) in extrinsics.items():
            label = f'{spec.variant_label(variant)}|extrinsic={source}'
            entry = {'method': method, 'ridge_lambda': ridge_lambda,
                     'extrinsic_source': source}
            try:
                fit = _wp4_solve(method, calibration_type, theta, known_R, known_t,
                                 cam_R, cam_t, joint_frames, fix_joint1, ridge_lambda)
            except Exception as exc:
                entry['error'] = f'{type(exc).__name__}: {exc}'
                out['variants'][label] = entry
                continue

            entry['in_sample'] = {
                'rms_before_mm': fit.rms_before_m * 1000.0,
                'rms_after_mm': fit.rms_after_m * 1000.0,
                'converged': bool(fit.converged),
            }
            entry['conditioning'] = {
                'condition_number': fit.condition_number,
                'singular_value_min': float(fit.singular_values[-1]),
                'singular_value_max': float(fit.singular_values[0]),
            }

            if analysis.get('kfold'):
                entry['held_out'] = _wp4_held_out(
                    method, calibration_type, theta, known_R, known_t, cam_R, cam_t,
                    joint_frames, fix_joint1, ridge_lambda, n, analysis)

            if bootstrap_n:
                agreements = []
                rng = np.random.default_rng(analysis.get('bootstrap_seed', 0))
                for idx in metrics.bootstrap_indices(n, bootstrap_n, rng):
                    try:
                        b = _wp4_solve(method, calibration_type, theta[idx], known_R, known_t,
                                       cam_R[idx], cam_t[idx], joint_frames, fix_joint1,
                                       ridge_lambda)
                    except Exception:
                        continue
                    agreements.append(metrics.fk_agreement_error(
                        joint_frames, fit.corrections, b.corrections, theta)['position_rms_mm'])
                if agreements:
                    entry['bootstrap'] = {'fk_spread_mm': metrics.summarize(agreements)}

            if true_corrections is not None:
                truth_entry = {
                    'fk_agreement': metrics.fk_agreement_error(
                        joint_frames, true_corrections, fit.corrections, theta),
                    'per_joint': metrics.per_joint_correction_error(
                        truth_kin['corrections'],
                        _corrections_to_dict(fit.corrections, joint_names),
                        joint_names),
                }
                if 'mount_offset' in truth_kin:
                    true_R, true_t = metrics.rt_from_transform_dict(truth_kin['mount_offset'])
                    truth_entry['mount_offset'] = metrics.pose_error(
                        true_R, true_t, fit.mount_R, fit.mount_t)
                entry['truth'] = truth_entry

            out['variants'][label] = entry

    return out


def _corrections_to_dict(corrections: np.ndarray, joint_names) -> dict:
    return perturbation.to_yaml_dict(
        corrections, list(joint_names))[perturbation.ROOT_KEY]['joints']


# entry point

_ANALYZERS = {'wp2': analyze_wp2, 'wp3': analyze_wp3, 'wp4': analyze_wp4}


def analyze_run(run_dir: str, run_spec: 'spec.RunSpec | None' = None,
                write: bool = True) -> dict:
    """Analyze one run directory and write `metrics.yaml` into it."""
    art = RunArtifacts(run_dir)
    if run_spec is None:
        manifest_spec = (art.manifest or {}).get('spec')
        run_spec = spec.RunSpec.from_dict(manifest_spec) if manifest_spec else None

    if run_spec is not None:
        work_packages = list(run_spec.work_packages)
        variants_cfg = run_spec.variants
        analysis = run_spec.analysis
    else:
        # No manifest: infer from which sample files actually exist. For Hardware datasets.
        work_packages = [
            wp for wp, name in (('wp2', 'samples.npy'), ('wp3', 'hand_eye_samples.npy'),
                                ('wp4', 'kinematic_samples.npy'))
            if os.path.exists(art.path(name))]
        variants_cfg = {wp: spec.DEFAULT_VARIANTS[wp] for wp in work_packages}
        analysis = dict(spec.DEFAULT_ANALYSIS)

    document = {
        'run_id': os.path.basename(os.path.normpath(run_dir)),
        'has_ground_truth': art.ground_truth is not None,
        'analysis': dict(analysis),
        'work_packages': {},
    }
    if run_spec is not None:
        document['spec'] = run_spec.to_dict()

    for wp in work_packages:
        variants = spec.variant_combinations(
            variants_cfg.get(wp) or spec.DEFAULT_VARIANTS[wp])
        try:
            document['work_packages'][wp] = _ANALYZERS[wp](art, variants, analysis)
        except (storage.SampleFileError, FileNotFoundError) as exc:
            document['work_packages'][wp] = {'skipped': str(exc)}
        except Exception as exc:
            document['work_packages'][wp] = {
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            }

    if write:
        storage.save_result(os.path.join(run_dir, 'metrics.yaml'), document)
    return document


def analyze_all(runs_dir: str, force: bool = False) -> list[str]:
    """Analyze every run directory under `runs_dir`. Returns the paths done."""
    done = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if not force and os.path.exists(os.path.join(run_dir, 'metrics.yaml')):
            continue
        analyze_run(run_dir)
        done.append(run_dir)
    return done
