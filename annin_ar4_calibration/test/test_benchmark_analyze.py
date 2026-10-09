"""End-to-end tests for the benchmark's offline path, with no Gazebo and no robot"""
import math
import os

import numpy as np
import pytest
import yaml

from annin_ar4_calibration.benchmark import aggregate, analyze, spec
from annin_ar4_calibration.core import geometry, metrics, perturbation, storage

# Reuses the WP4 sample synthesizer the existing solver tests already have
from test_kinematic_calibration import (_ar4_joint_frames, _ground_truth_corrections,
                                        _ground_truth_mount, _known_extrinsic,
                                        _synthesize_samples)


def _urdf_from_frames(frames) -> str:
    """Minimal URDF carrying just what `parse_urdf_chain` reads back."""
    joints = []
    for jf in frames:
        joints.append(
            f'  <joint name="{jf.name}" type="revolute">\n'
            f'    <origin xyz="{" ".join(f"{v:.12g}" for v in jf.xyz0)}" '
            f'rpy="{" ".join(f"{v:.12g}" for v in jf.rpy0)}"/>\n'
            f'    <axis xyz="{" ".join(f"{v:.12g}" for v in jf.axis)}"/>\n'
            f'    <limit lower="{jf.lower}" upper="{jf.upper}" effort="1" velocity="1"/>\n'
            f'  </joint>')
    return '<robot name="test">\n' + '\n'.join(joints) + '\n</robot>'


def _pose_rows(R, t) -> np.ndarray:
    quats = np.array([geometry.rotation_matrix_to_quat(Ri) for Ri in R])
    return np.hstack([t, quats])


def _write_pivot_samples(data_dir, p_tool, n=30, noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)
    p_base = np.array([0.30, 0.10, 0.20])
    R = np.array([
        geometry.axis_angle_to_rotation_matrix(
            rng.normal(size=3), rng.uniform(-math.pi, math.pi)) for _ in range(n)])
    t = np.array([p_base - Ri @ p_tool for Ri in R])
    if noise_std:
        t = t + rng.normal(scale=noise_std, size=t.shape)
    np.save(os.path.join(data_dir, 'samples.npy'), _pose_rows(R, t))


def _write_handeye_samples(data_dir, calibration_type, X_R, X_t, n=20,
                           noise_t_std=0.0, seed=0):
    from annin_ar4_calibration.core import handeye_solver
    rng = np.random.default_rng(seed)
    target_R = geometry.axis_angle_to_rotation_matrix(np.array([0.4, 0.1, 0.9]),
                                                      math.radians(-25))
    target_t = np.array([0.35, 0.12, 0.22])

    robot_R = np.array([
        geometry.axis_angle_to_rotation_matrix(
            rng.normal(size=3), rng.uniform(-1.0, 1.0))
        for _ in range(n)])
    robot_t = np.array([[0.3, 0.0, 0.3]] * n) + \
        rng.uniform(-0.08, 0.08, size=(n, 3))

    hand_R, hand_t = handeye_solver.hand_poses(
        calibration_type, robot_R, robot_t)
    hand_R_inv, hand_t_inv = geometry.batch_invert_rt(hand_R, hand_t)
    mid_R, mid_t = geometry.batch_compose_rt(
        hand_R_inv, hand_t_inv, target_R, target_t)
    X_R_inv = X_R.T
    cam_R, cam_t = geometry.batch_compose_rt(
        X_R_inv, -X_R_inv @ X_t, mid_R, mid_t)
    if noise_t_std:
        cam_t = cam_t + rng.normal(scale=noise_t_std, size=cam_t.shape)

    sample_file = os.path.join(data_dir, 'hand_eye_samples.npy')
    np.save(sample_file, np.hstack([_pose_rows(robot_R, robot_t),
                                    _pose_rows(cam_R, cam_t)]))
    storage.save_meta(sample_file, {'calibration_type': calibration_type})


def _write_kinematic_samples(data_dir, calibration_type, frames, corrections,
                             mount_R, mount_t, X_R, X_t, n=60,
                             noise_t_std=0.0, noise_r_std=0.0, seed=0):
    theta, cam_R, cam_t = _synthesize_samples(
        calibration_type, frames, corrections, mount_R, mount_t, X_R, X_t,
        n=n, noise_t_std=noise_t_std, noise_r_std=noise_r_std,
        rng=np.random.default_rng(seed))
    sample_file = os.path.join(data_dir, 'kinematic_samples.npy')
    np.save(sample_file, np.hstack([theta, _pose_rows(cam_R, cam_t)]))
    storage.save_meta(sample_file, {
        'arm_joint_names': list(perturbation.DEFAULT_JOINT_NAMES)})


def build_run(tmp_path, *, work_packages=('wp2', 'wp3', 'wp4'),
              calibration_type='eye_to_hand', with_ground_truth=True,
              kinematic_noise_t=0.0, kinematic_noise_r=0.0,
              pivot_noise=0.0, handeye_noise=0.0, n_kinematic=60, seed=0,
              extrinsic_error_m=0.0) -> str:
    """Write a synthetic run directory in `collect.py`'s exact layout."""
    run_dir = str(tmp_path)
    data_dir = os.path.join(run_dir, analyze.DATA_SUBDIR)
    os.makedirs(data_dir, exist_ok=True)

    frames = _ar4_joint_frames()
    with open(os.path.join(run_dir, 'nominal_robot.urdf'), 'w') as f:
        f.write(_urdf_from_frames(frames))

    fix_joint1 = calibration_type == 'eye_in_hand'
    corrections = _ground_truth_corrections(fix_joint1)
    mount_R, mount_t = _ground_truth_mount()
    X_R, X_t = _known_extrinsic()
    p_tool = np.array([0.0, 0.01, 0.15])

    if 'wp2' in work_packages:
        _write_pivot_samples(
            data_dir, p_tool, noise_std=pivot_noise, seed=seed)
    if 'wp3' in work_packages:
        _write_handeye_samples(data_dir, calibration_type, X_R, X_t,
                               noise_t_std=handeye_noise, seed=seed)
    if 'wp4' in work_packages:
        _write_kinematic_samples(
            data_dir, calibration_type, frames, corrections, mount_R, mount_t,
            X_R, X_t, n=n_kinematic, noise_t_std=kinematic_noise_t,
            noise_r_std=kinematic_noise_r, seed=seed)

        # WP4 reads its "known" extrinsic from WP3's result file
        result_X_t = X_t + np.array([extrinsic_error_m, 0.0, 0.0])
        storage.save_result(os.path.join(data_dir, 'hand_eye_result.yaml'), {
            'hand_eye_calibration': {
                'calibration_type': calibration_type,
                'transform': metrics.transform_dict(X_R, result_X_t),
            }})

    if with_ground_truth:
        truth = {'ground_truth': {
            'tcp_calibration': {'offset': {'x': float(p_tool[0]), 'y': float(p_tool[1]),
                                           'z': float(p_tool[2])}},
            'hand_eye_calibration': {'transform': metrics.transform_dict(X_R, X_t)},
            'kinematic_calibration': {
                'corrections': perturbation.to_yaml_dict(
                    corrections)[perturbation.ROOT_KEY]['joints'],
                'mount_offset': metrics.transform_dict(mount_R, mount_t),
            },
        }}
        storage.save_result(os.path.join(data_dir, 'ground_truth.yaml'), truth)

    run_spec = spec.RunSpec(
        sweep='unit', mode='sim', work_packages=list(work_packages), seed=seed,
        collection=dict(spec.DEFAULT_COLLECTION),
        analysis={'kfold': 3, 'bootstrap': 5, 'bootstrap_wp4': 3,
                  'bootstrap_seed': 0, 'kfold_seed': 0},
        variants={'wp2': {'ransac': [False]}, 'wp3': {'method': ['PARK']},
                  'wp4': {'method': ['sequential'], 'ridge_lambda': [1e-9]}})
    storage.save_result(os.path.join(run_dir, 'manifest.yaml'), {
        'run_id': run_spec.run_id, 'outcome': 'ok', 'spec': run_spec.to_dict()})
    return run_dir


# structure

def test_analyze_run_covers_every_work_package(tmp_path):
    run_dir = build_run(tmp_path)
    document = analyze.analyze_run(run_dir)

    assert document['has_ground_truth'] is True
    assert os.path.exists(os.path.join(run_dir, 'metrics.yaml'))
    for wp in ('wp2', 'wp3', 'wp4'):
        data = document['work_packages'][wp]
        assert 'skipped' not in data, f'{wp} skipped: {data.get("skipped")}'
        assert 'error' not in data, f'{wp} errored: {data.get("error")}'
        assert data['variants'], f'{wp} produced no variant results'


def test_metrics_yaml_is_plain_yaml(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp2',))
    analyze.analyze_run(run_dir)
    with open(os.path.join(run_dir, 'metrics.yaml')) as f:
        reloaded = yaml.safe_load(f)
    assert reloaded['work_packages']['wp2']['variants']


def test_analysis_works_without_a_manifest(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp2', 'wp3'))
    os.remove(os.path.join(run_dir, 'manifest.yaml'))

    document = analyze.analyze_run(run_dir)
    assert set(document['work_packages']) == {'wp2', 'wp3'}


def test_hardware_style_run_reports_family_b_but_no_truth(tmp_path):
    run_dir = build_run(tmp_path, with_ground_truth=False)
    document = analyze.analyze_run(run_dir)
    assert document['has_ground_truth'] is False

    for wp in ('wp2', 'wp3', 'wp4'):
        for entry in document['work_packages'][wp]['variants'].values():
            assert 'truth' not in entry
        assert document['work_packages'][wp]['variants']

    wp3 = list(document['work_packages']['wp3']['variants'].values())[0]
    assert wp3['held_out']['translation_mm']['n'] == 3
    assert 'bootstrap' in wp3


def test_wp4_skips_cleanly_without_the_nominal_urdf(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp4',))
    os.remove(os.path.join(run_dir, 'nominal_robot.urdf'))
    document = analyze.analyze_run(run_dir)
    assert 'nominal_robot.urdf' in document['work_packages']['wp4']['skipped']


def test_wp4_skips_cleanly_without_a_hand_eye_result(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp4',))
    os.remove(os.path.join(
        run_dir, analyze.DATA_SUBDIR, 'hand_eye_result.yaml'))
    document = analyze.analyze_run(run_dir)
    assert 'WP4 depends on WP3' in document['work_packages']['wp4']['skipped']


# metric behaviour

def test_noiseless_run_recovers_ground_truth(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp2', 'wp3'))
    document = analyze.analyze_run(run_dir)

    wp2 = list(document['work_packages']['wp2']['variants'].values())[0]
    assert wp2['truth']['translation_error_mm'] < 1e-6
    wp3 = list(document['work_packages']['wp3']['variants'].values())[0]
    assert wp3['truth']['translation_error_mm'] < 1e-3


def test_held_out_error_grows_with_measurement_noise(tmp_path):
    held_out = []
    for i, noise in enumerate((0.0002, 0.001, 0.004)):
        run_dir = build_run(tmp_path / f'noise{i}', work_packages=('wp3',),
                            handeye_noise=noise, seed=1)
        document = analyze.analyze_run(run_dir)
        entry = list(document['work_packages']['wp3']['variants'].values())[0]
        held_out.append(entry['held_out']['translation_mm']['mean'])
    assert held_out[0] < held_out[1] < held_out[2]


def test_held_out_error_tracks_true_error(tmp_path):
    held_out, truth = [], []
    for i, noise in enumerate((0.0002, 0.001, 0.004)):
        run_dir = build_run(tmp_path / f'track{i}', work_packages=('wp3',),
                            handeye_noise=noise, seed=2)
        entry = list(analyze.analyze_run(run_dir)['work_packages']['wp3']
                     ['variants'].values())[0]
        held_out.append(entry['held_out']['translation_mm']['mean'])
        truth.append(entry['truth']['translation_error_mm'])

    assert np.corrcoef(held_out, truth)[0, 1] > 0.9


def test_wp4_reports_generalization_against_the_nominal_model(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp4',),
                        kinematic_noise_t=1e-5, n_kinematic=80, seed=3)
    entry = list(analyze.analyze_run(run_dir)['work_packages']['wp4']
                 ['variants'].values())[0]

    held_out = entry['held_out']
    assert held_out['generalizes'] is True
    assert held_out['corrected_rms_mm']['mean'] < held_out['nominal_rms_mm']['mean']


def test_wp4_scores_both_extrinsic_sources(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp4',),
                        extrinsic_error_m=0.004, n_kinematic=80, seed=4)
    variants = analyze.analyze_run(run_dir)['work_packages']['wp4']['variants']

    labels = list(variants)
    assert any(label.endswith('extrinsic=estimated') for label in labels)
    assert any(label.endswith('extrinsic=truth') for label in labels)

    estimated = next(v for k, v in variants.items() if k.endswith('estimated'))
    truth = next(v for k, v in variants.items() if k.endswith('truth'))
    # A 4 mm error in the extrinsic must show up as worse WP4 accuracy.
    assert (estimated['truth']['fk_agreement']['position_rms_mm']
            > truth['truth']['fk_agreement']['position_rms_mm'])


def test_detection_sidecar_is_picked_up_when_present(tmp_path):
    run_dir = build_run(tmp_path, work_packages=('wp3',))
    sample_file = os.path.join(
        run_dir, analyze.DATA_SUBDIR, 'hand_eye_samples.npy')
    for i in range(20):
        storage.append_detection(sample_file, i, 24, 0.3 + 0.01 * i)

    document = analyze.analyze_run(run_dir)
    detection = document['work_packages']['wp3']['detection']
    assert detection['num_detections'] == 20
    assert detection['reprojection_rms_px']['mean'] == pytest.approx(
        0.395, abs=1e-6)


def test_detection_block_is_empty_without_a_sidecar(tmp_path):
    """Archived runs collected before the sidecar existed must still analyse."""
    run_dir = build_run(tmp_path, work_packages=('wp3',))
    assert analyze.analyze_run(
        run_dir)['work_packages']['wp3']['detection'] == {}


# aggregation

def test_aggregate_emits_tidy_rows_for_a_run(tmp_path):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir()
    run_dir = build_run(runs_dir / 'run-a')
    analyze.analyze_run(run_dir)

    out_csv = str(tmp_path / 'results.csv')
    count = aggregate.aggregate(str(runs_dir), out_csv)
    assert count > 0

    rows = [r for r in _read_csv(out_csv)]
    assert {r['work_package'] for r in rows} >= {'wp2', 'wp3', 'wp4'}
    families = {r['family'] for r in rows}
    assert 'truth' in families and 'truth_free' in families
    assert any(r['metric'] ==
               'truth.fk_agreement.position_rms_mm' for r in rows)
    assert all(r['run_id'] for r in rows)


def test_aggregate_records_failed_runs_rather_than_dropping_them(tmp_path):
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir()
    failed = runs_dir / 'run-failed'
    failed.mkdir()
    storage.save_result(str(failed / 'manifest.yaml'), {
        'run_id': 'run-failed', 'outcome': 'not_ready',
        'spec': spec.RunSpec(sweep='unit', mode='sim', work_packages=['wp4'], seed=9,
                             collection=dict(spec.DEFAULT_COLLECTION)).to_dict()})

    out_csv = str(tmp_path / 'results.csv')
    aggregate.aggregate(str(runs_dir), out_csv)
    rows = _read_csv(out_csv)
    assert any(r['metric'] == 'outcome' and r['value']
               == 'not_ready' for r in rows)


def _read_csv(path):
    import csv
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# sweeps

def test_shipped_sweeps_expand(tmp_path):
    sweeps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'annin_ar4_calibration', 'benchmark', 'sweeps')
    for name in sorted(os.listdir(sweeps_dir)):
        if not name.endswith('.yaml'):
            continue
        specs = spec.load_sweep(os.path.join(sweeps_dir, name))
        assert specs, f'{name} expanded to nothing'
        assert len({s.run_id for s in specs}) == len(specs), \
            f'{name} produced duplicate run_ids - two cells would share a directory'


def test_coupled_grid_axis_keeps_resolutions_paired(tmp_path):
    path = tmp_path / 'coupled.yaml'
    path.write_text(
        'name: t\nmode: sim\nwork_packages: [wp3]\nseeds: [1]\n'
        'grid:\n'
        '  _resolution:\n'
        '    - {camera_width: 640, camera_height: 480}\n'
        '    - {camera_width: 1280, camera_height: 720}\n')
    specs = spec.load_sweep(str(path))
    pairs = {(s.collection['camera_width'],
              s.collection['camera_height']) for s in specs}
    assert pairs == {(640, 480), (1280, 720)}


def test_solver_settings_in_the_grid_are_rejected(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('name: t\nmode: sim\nwork_packages: [wp3]\nseeds: [1]\n'
                    'grid:\n  method: [PARK, TSAI]\n')
    with pytest.raises(ValueError, match='variants'):
        spec.load_sweep(str(path))


def test_run_id_ignores_variants_but_tracks_collection(tmp_path):
    base = spec.RunSpec(sweep='s', mode='sim', work_packages=['wp3'], seed=1,
                        collection=dict(spec.DEFAULT_COLLECTION))
    same_data = spec.RunSpec(sweep='s', mode='sim', work_packages=['wp3'], seed=1,
                             collection=dict(spec.DEFAULT_COLLECTION),
                             variants={'wp3': {'method': ['TSAI']}})
    other_data = spec.RunSpec(sweep='s', mode='sim', work_packages=['wp3'], seed=1,
                              collection={**spec.DEFAULT_COLLECTION,
                                          'image_noise_stddev': 0.01})
    assert base.run_id == same_data.run_id
    assert base.run_id != other_data.run_id


def test_work_packages_are_ordered_so_wp4_follows_wp3(tmp_path):
    path = tmp_path / 'order.yaml'
    path.write_text(
        'name: t\nmode: sim\nwork_packages: [wp4, wp3]\nseeds: [1]\n')
    assert spec.load_sweep(str(path))[0].work_packages == ['wp3', 'wp4']
