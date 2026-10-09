"""Tests for the benchmark report generator"""
import csv

import pytest

from annin_ar4_calibration.benchmark import report


def _write_csv(path, rows):
    columns = ['run_id', 'sweep', 'mode', 'seed', 'work_package', 'variant',
               'calibration_type', 'num_samples', 'metric', 'value', 'unit', 'family']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in columns})


def _wp4_rows(calibration_type, true_error, held_out=0.7, n=3):
    """One WP4 run per seed, for a single variant."""
    out = []
    for seed in range(n):
        common = dict(run_id=f'r{calibration_type}{seed}', sweep='sim_wp4', mode='sim',
                      seed=seed, work_package='wp4',
                      variant='method=joint_lm|ridge_lambda=1e-09|extrinsic=estimated',
                      calibration_type=calibration_type, num_samples=40, unit='mm')
        out.append({**common, 'metric': 'truth.fk_agreement.position_rms_mm',
                    'value': true_error, 'family': 'truth'})
        out.append({**common, 'metric': 'held_out.corrected_rms_mm.mean',
                    'value': held_out, 'family': 'truth_free'})
        out.append({**common, 'metric': 'in_sample.rms_after_mm',
                    'value': 0.6, 'family': 'truth_free'})
        out.append({**common, 'metric': 'held_out.nominal_rms_mm.mean',
                    'value': 2.2, 'family': 'truth_free'})
        out.append({**common, 'metric': 'conditioning.condition_number',
                    'value': 8.3e5 if calibration_type == 'eye_in_hand' else 25.0,
                    'family': 'truth_free'})
    return out


@pytest.fixture
def two_mounting_csv(tmp_path):
    path = tmp_path / 'results.csv'
    _write_csv(path, _wp4_rows('eye_in_hand', 23.4) +
               _wp4_rows('eye_to_hand', 2.7))
    return str(path)


def test_a_table_is_emitted_per_camera_mounting(two_mounting_csv, tmp_path):
    out = tmp_path / 'report'
    report.build_report(two_mounting_csv, str(out))

    assert (out / 'wp4_summary_eye_in_hand.tex').exists()
    assert (out / 'wp4_summary_eye_to_hand.tex').exists()
    assert not (
        out / 'wp4_summary.tex').exists(), 'the pooled table must not also be written'


def test_each_table_reports_only_its_own_mounting(two_mounting_csv, tmp_path):
    out = tmp_path / 'report'
    report.build_report(two_mounting_csv, str(out))

    in_hand = (out / 'wp4_summary_eye_in_hand.tex').read_text()
    to_hand = (out / 'wp4_summary_eye_to_hand.tex').read_text()

    assert '23.4' in in_hand and '2.7' not in in_hand
    assert '2.7' in to_hand and '23.4' not in to_hand
    # n must be the per-mounting count, not the pooled one
    assert '& 3 &' in in_hand and '& 3 &' in to_hand


def test_a_single_mounting_still_produces_one_unsuffixed_table(tmp_path):
    path = tmp_path / 'results.csv'
    _write_csv(path, _wp4_rows('eye_to_hand', 2.7))
    out = tmp_path / 'report'
    report.build_report(str(path), str(out))

    assert (out / 'wp4_summary.tex').exists()
    assert not (out / 'wp4_summary_eye_to_hand.tex').exists()


def test_overfitting_figure_is_emitted_per_mounting(two_mounting_csv, tmp_path):
    out = tmp_path / 'report'
    report.build_report(two_mounting_csv, str(out))

    assert (out / 'wp4_overfitting_eye_in_hand.pdf').exists()
    assert (out / 'wp4_overfitting_eye_to_hand.pdf').exists()


def test_no_figure_is_silently_skipped_on_complete_data(two_mounting_csv, tmp_path):
    out = tmp_path / 'report'
    produced = report.build_report(two_mounting_csv, str(out))

    unexpected = [s for s in produced['skipped'] if 'no data' not in s]
    assert not unexpected, unexpected


def test_conditioning_figure_survives_either_matplotlib_boxplot_spelling(
        two_mounting_csv, tmp_path):
    out = tmp_path / 'report'
    produced = report.build_report(two_mounting_csv, str(out))

    assert (out / 'wp4_conditioning.pdf').exists()
    assert not any('conditioning' in s for s in produced['skipped'])


def test_calibration_types_reports_nothing_to_split_on_when_uniform(two_mounting_csv):
    rows = report.load_csv(two_mounting_csv)
    assert report.calibration_types(rows, 'wp4') == [
        'eye_in_hand', 'eye_to_hand']

    single = report.select(rows, calibration_type='eye_in_hand')
    assert report.calibration_types(single, 'wp4') == []


# sweep resume

def _run_dir(tmp_path, name, *, collected=True, analyzed=True):
    from annin_ar4_calibration.benchmark import collect
    d = tmp_path / 'runs' / name
    d.mkdir(parents=True)
    if collected:
        (d / 'manifest.yaml').write_text(f'outcome: {collect.OUTCOME_OK}\n')
    if analyzed:
        (d / 'metrics.yaml').write_text('wp4: {}\n')
    return str(d)


def test_resume_skips_only_cells_that_are_both_collected_and_analyzed(tmp_path):
    from annin_ar4_calibration.benchmark import runner

    done = _run_dir(tmp_path, 'done')
    failed = _run_dir(tmp_path, 'failed', collected=False, analyzed=False)
    collected_only = _run_dir(tmp_path, 'collected_only', analyzed=False)

    def resumable(d):
        return runner._already_collected(d) and runner._already_analyzed(d)

    assert resumable(done)
    assert not resumable(failed)
    assert not resumable(collected_only), \
        'collected but never analyzed means no row in results.csv - must re-run'


def test_a_failed_cell_is_not_treated_as_collected(tmp_path):
    from annin_ar4_calibration.benchmark import runner

    d = tmp_path / 'runs' / 'bad'
    d.mkdir(parents=True)
    (d / 'manifest.yaml').write_text('outcome: collection_failed\n')
    (d / 'metrics.yaml').write_text('wp4: {}\n')

    assert not runner._already_collected(str(d))


def test_resume_is_off_by_default():
    """The default must stay re-analyze-everything: that is what lets a changed
    variant or analysis setting take effect without re-collecting."""
    from annin_ar4_calibration.benchmark import runner

    args = runner.build_parser().parse_args(['--sweep', 'sim_wp4.yaml'])
    assert args.resume is False
    assert args.force is False

    assert runner.build_parser().parse_args(
        ['--sweep', 'sim_wp4.yaml', '--resume']).resume is True
