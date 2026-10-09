# aggregates many runs `metrics.yaml` into one tidy CSV.

from __future__ import annotations

import argparse
import csv
import os
import sys

import yaml

from annin_ar4_calibration.benchmark import spec
from annin_ar4_calibration.core import paths

CONDITION_COLUMNS = list(spec.COLLECTION_KEYS)

BASE_COLUMNS = ['run_id', 'sweep', 'mode', 'seed', 'work_package', 'variant',
                'num_samples']
TAIL_COLUMNS = ['metric', 'value', 'unit', 'family']

#: Statistic to extract from a `metrics.summarize` block. The mean plus the percentile interval, so a plot can draw a point and a ribbon without the CSV carrying every raw resample.
SUMMARY_STATS = ('mean', 'std', 'median', 'ci95_lo', 'ci95_hi', 'n')


def _unit_of(name: str) -> str:
    for suffix, unit in (('_mm', 'mm'), ('_deg', 'deg'), ('_mdeg', 'mdeg'),
                         ('_px', 'px'), ('_m', 'm'), ('_sec', 's')):
        if name.endswith(suffix):
            return unit
    return ''


def _emit(rows, base, metric, value, family):
    if value is None:
        return
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        return
    rows.append({**base, 'metric': metric, 'value': value,
                 'unit': _unit_of(metric), 'family': family})


def _emit_summary(rows, base, prefix, summary: dict, family):
    if not isinstance(summary, dict):
        return
    for stat in SUMMARY_STATS:
        if stat in summary:
            _emit(rows, base, f'{prefix}.{stat}', summary[stat], family)


def _flatten(rows, base, node, prefix, family):
    if not isinstance(node, dict):
        return
    if 'mean' in node and 'n' in node:
        _emit_summary(rows, base, prefix, node, family)
        return
    for key, value in node.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            _flatten(rows, base, value, name, family)
        elif isinstance(value, (int, float, bool)):
            _emit(rows, base, name, value, family)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (int, float, bool)):
                    _emit(rows, base, f'{name}[{i}]', item, family)


def rows_for_run(run_dir: str) -> list[dict]:
    metrics_path = os.path.join(run_dir, 'metrics.yaml')
    manifest_path = os.path.join(run_dir, 'manifest.yaml')

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

    run_spec = manifest.get('spec') or {}
    collection = run_spec.get('collection') or {}
    common = {
        'run_id': run_spec.get('run_id') or os.path.basename(os.path.normpath(run_dir)),
        'sweep': run_spec.get('sweep', ''),
        'mode': run_spec.get('mode', ''),
        'seed': run_spec.get('seed', ''),
        **{key: collection.get(key, '') for key in CONDITION_COLUMNS},
    }

    rows: list[dict] = []
    outcome = manifest.get('outcome')
    if outcome and outcome != 'ok':
        rows.append({**common, 'work_package': '', 'variant': '', 'num_samples': '',
                     'metric': 'outcome', 'value': outcome, 'unit': '', 'family': 'status'})
        return rows

    if not os.path.exists(metrics_path):
        return rows
    with open(metrics_path) as f:
        document = yaml.safe_load(f) or {}

    for wp, data in (document.get('work_packages') or {}).items():
        wp_base = {**common, 'work_package': wp,
                   'num_samples': data.get('num_samples', '')}

        if 'skipped' in data or 'error' in data:
            rows.append({**wp_base, 'variant': '', 'metric': 'outcome',
                         'value': 'skipped' if 'skipped' in data else 'error',
                         'unit': '', 'family': 'status'})
            continue
        run_base = {**wp_base, 'variant': '-'}
        for block, family in (('conditioning', 'truth_free'), ('detection', 'truth_free')):
            _flatten(rows, run_base, data.get(block) or {}, block, family)
        for i, dot in enumerate(data.get('axis_alignment') or []):
            _emit(rows, run_base, f'axis_alignment[{i}]', dot, 'truth_free')
        for i, rng in enumerate(data.get('joint_range_deg') or []):
            _emit(rows, run_base, f'joint_range_deg[{i}]', rng, 'truth_free')

        for variant, entry in (data.get('variants') or {}).items():
            base = {**wp_base, 'variant': variant}
            if 'error' in entry:
                rows.append({**base, 'metric': 'outcome', 'value': 'solver_error',
                             'unit': '', 'family': 'status'})
                continue
            for block, family in (('in_sample', 'truth_free'),
                                  ('held_out', 'truth_free'),
                                  ('bootstrap', 'truth_free'),
                                  ('conditioning', 'truth_free'),
                                  ('truth', 'truth')):
                _flatten(rows, base, entry.get(block) or {}, block, family)
            # WP2's held-out block is a bare summary rather than a dict of them.
            _emit_summary(rows, base, 'held_out', entry.get('held_out_rms_mm') or {},
                          'truth_free')
    return rows


def aggregate(runs_dir: str, out_csv: str) -> int:
    rows: list[dict] = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if os.path.isdir(run_dir):
            rows.extend(rows_for_run(run_dir))

    columns = BASE_COLUMNS + CONDITION_COLUMNS + TAIL_COLUMNS
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in columns})
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='benchmark_aggregate', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=str(paths.data_dir() / 'benchmark'))
    parser.add_argument('--runs-dir', default=None)
    parser.add_argument('--csv', default=None, help='default: <out-dir>/results.csv')
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir or os.path.join(args.out_dir, 'runs')
    if not os.path.isdir(runs_dir):
        print(f'no runs directory at {runs_dir}', file=sys.stderr)
        return 1

    out_csv = args.csv or os.path.join(args.out_dir, 'results.csv')
    count = aggregate(runs_dir, out_csv)
    print(f'wrote {count} rows to {out_csv}')
    print(f'next: benchmark_report --csv {out_csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
