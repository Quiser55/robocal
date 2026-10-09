""" Turns results.csv into thesis figures and LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

from annin_ar4_calibration.core import metrics, paths

FIGURE_SIZE = (6.0, 3.7)

AXIS_LABELS = {
    'touch_noise_std_m': 'touch precision $\\sigma$ [m]',
    'image_noise_stddev': 'image noise $\\sigma$ [normalised]',
    'perturbation_sigma_t_mm': 'injected joint-origin error $\\sigma$ [mm]',
    'perturbation_sigma_r_deg': 'injected joint-origin rotation $\\sigma$ [deg]',
    'num_samples': 'number of samples collected',
    'num_poses': 'number of touches',
    'camera_width': 'image width [px]',
    'seed': 'seed',
}


def axis_label(column: str) -> str:
    return AXIS_LABELS.get(column, column.replace('_', ' '))


def _configure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'font.size': 9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })
    return plt


# CSV utils

def load_csv(path: str) -> list[dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        try:
            row['value'] = float(row['value'])
        except (TypeError, ValueError):
            row['value'] = None
    return rows


def select(rows, **filters) -> list[dict]:
    out = []
    for row in rows:
        for key, want in filters.items():
            got = row.get(key)
            if isinstance(want, (list, tuple, set)):
                if got not in want:
                    break
            elif got != want:
                break
        else:
            out.append(row)
    return out


def series(rows, x_column: str, metric: str, **filters):
    """Rows -> {x_value: [metric values]}, numeric x, sorted."""
    grouped = defaultdict(list)
    for row in select(rows, metric=metric, **filters):
        if row['value'] is None:
            continue
        try:
            x = float(row[x_column])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[x].append(row['value'])
    return dict(sorted(grouped.items()))


def _mean_ci(values):
    summary = metrics.summarize(values)
    if not summary.get('n'):
        return None
    return summary['mean'], summary.get('ci95_lo', summary['mean']), \
        summary.get('ci95_hi', summary['mean'])


def distinct(rows, column: str) -> list[str]:
    return sorted({row[column] for row in rows if row.get(column) not in (None, '')})


def calibration_types(rows, wp: str | None = None) -> list[str]:
    subset = select(rows, work_package=wp) if wp else rows
    found = distinct(subset, 'calibration_type')
    return found if len(found) > 1 else []


def _suffix(calibration_type: str | None) -> str:
    return f'_{calibration_type}' if calibration_type else ''


def _title_suffix(calibration_type: str | None) -> str:
    return f' ({calibration_type.replace("_", "-")})' if calibration_type else ''


# figures

def _floor_line(ax, value=metrics.SIM_DETECTION_FLOOR_MM, label='detection floor'):
    ax.axhline(value, color='0.4', linestyle=':', linewidth=1.0)
    ax.annotate(label, xy=(0.01, value), xycoords=('axes fraction', 'data'),
                va='bottom', fontsize=7, color='0.35')


def _plot_curve(plt, out_path, curves, xlabel, ylabel, title, floor=None, logy=False):
    """curves: {label: {x: [values]}}. Mean line plus 95% CI ribbon."""
    drawn = False
    fig, ax = plt.subplots()
    for label, data in curves.items():
        xs, means, los, his = [], [], [], []
        for x, values in sorted(data.items()):
            stats = _mean_ci(values)
            if stats is None:
                continue
            xs.append(x)
            means.append(stats[0])
            los.append(stats[1])
            his.append(stats[2])
        if not xs:
            continue
        drawn = True
        line, = ax.plot(xs, means, marker='o', markersize=3.5, label=label)
        ax.fill_between(xs, los, his, alpha=0.15, color=line.get_color())

    if not drawn:
        plt.close(fig)
        return False

    if floor is not None:
        _floor_line(ax, floor)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend(fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def figure_error_vs_condition(plt, rows, out_dir, x_column, metric, wp,
                              ylabel, title, filename, split_by='calibration_type'):
    wp_rows = select(rows, work_package=wp)
    curves = {}
    for value in distinct(wp_rows, split_by) or ['all']:
        subset = wp_rows if value == 'all' else select(
            wp_rows, **{split_by: value})
        data = series(subset, x_column, metric)
        if data:
            curves[value] = data
    if not curves:
        return None
    path = os.path.join(out_dir, filename)
    return path if _plot_curve(plt, path, curves, axis_label(x_column), ylabel, title,
                               floor=metrics.SIM_DETECTION_FLOOR_MM) else None


def figure_held_out_vs_truth(plt, rows, out_dir):
    pairs_by_wp = {
        'wp2': ('held_out.mean', 'truth.translation_error_mm'),
        'wp3': ('held_out.translation_mm.mean', 'truth.translation_error_mm'),
        'wp4': ('held_out.corrected_rms_mm.mean',
                'truth.fk_agreement.position_rms_mm'),
    }

    def _points(subset):
        points = defaultdict(list)
        for wp, (held_metric, truth_metric) in pairs_by_wp.items():
            held = {(r['run_id'], r['variant']): r['value']
                    for r in select(subset, work_package=wp, metric=held_metric)
                    if r['value'] is not None}
            truth = {(r['run_id'], r['variant']): r['value']
                     for r in select(subset, work_package=wp, metric=truth_metric)
                     if r['value'] is not None}
            for key in held.keys() & truth.keys():
                points[wp].append((held[key], truth[key]))
        return points

    panels = calibration_types(rows) or [None]
    per_panel = [(ct, _points(select(rows, calibration_type=ct) if ct else rows))
                 for ct in panels]
    per_panel = [(ct, pts) for ct, pts in per_panel if any(pts.values())]
    if not per_panel:
        return None

    fig, axes = plt.subplots(
        1, len(per_panel), sharex=True, sharey=True,
        figsize=(FIGURE_SIZE[0] * min(len(per_panel), 2), FIGURE_SIZE[1]))
    axes = [axes] if len(per_panel) == 1 else list(axes)

    for ax, (calibration_type, points) in zip(axes, per_panel):
        limit = 0.0
        for wp, values in sorted(points.items()):
            if not values:
                continue
            xs, ys = zip(*values)
            ax.scatter(xs, ys, s=12, alpha=0.6, label=wp.upper())
            limit = max(limit, max(xs), max(ys))
        if limit > 0:
            ax.plot([0, limit], [0, limit], color='0.5',
                    linewidth=0.8, linestyle='--')
            ax.annotate('y = x', xy=(limit, limit), fontsize=7, color='0.4',
                        ha='right', va='bottom')
        ax.set_xlabel(
            'held-out prediction error [mm]\n(available on hardware)')
        ax.set_title(calibration_type.replace('_', '-') if calibration_type
                     else 'all mountings', fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('true error vs. ground truth [mm]\n(simulation only)')
    fig.suptitle('Does the truth-free metric track the truth?')
    path = os.path.join(out_dir, 'held_out_vs_truth.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_overfitting(plt, rows, out_dir, calibration_type=None):
    wp4 = select(rows, work_package='wp4')
    if calibration_type:
        wp4 = select(wp4, calibration_type=calibration_type)
    if not wp4:
        return None

    by_variant_in = defaultdict(list)
    by_variant_out = defaultdict(list)
    for row in select(wp4, metric='in_sample.rms_after_mm'):
        if row['value'] is not None:
            by_variant_in[row['variant']].append(row['value'])
    for row in select(wp4, metric='held_out.corrected_rms_mm.mean'):
        if row['value'] is not None:
            by_variant_out[row['variant']].append(row['value'])

    variants = sorted(by_variant_in.keys() & by_variant_out.keys())
    if not variants:
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(variants) + 2), 3.7))
    positions = range(len(variants))
    ax.bar([p - 0.2 for p in positions],
           [metrics.summarize(by_variant_in[v]).get('mean', 0)
            for v in variants],
           width=0.4, label='in-sample RMS')
    ax.bar([p + 0.2 for p in positions],
           [metrics.summarize(by_variant_out[v]).get('mean', 0)
            for v in variants],
           width=0.4, label='held-out RMS')
    ax.set_xticks(list(positions))
    ax.set_xticklabels([v.replace('|', '\n')
                       for v in variants], fontsize=6, rotation=0)
    ax.set_ylabel('loop-closure RMS [mm]')
    ax.set_title('WP4: in-sample fit vs. held-out prediction'
                 + _title_suffix(calibration_type))
    ax.legend(fontsize=7)
    path = os.path.join(
        out_dir, f'wp4_overfitting{_suffix(calibration_type)}.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_conditioning(plt, rows, out_dir):
    """WP4 condition number by camera mounting"""
    groups = defaultdict(list)
    for row in select(rows, work_package='wp4', metric='conditioning.condition_number'):
        if row['value'] and row['value'] > 0:
            groups[row.get('calibration_type') or 'unknown'].append(
                math.log10(row['value']))
    if not groups:
        return None

    labels = sorted(groups)
    fig, ax = plt.subplots()
    try:
        ax.boxplot([groups[k] for k in labels], tick_labels=labels)
    except TypeError:
        ax.boxplot([groups[k] for k in labels], labels=labels)
    ax.set_ylabel(r'$\log_{10}$ condition number')
    path = os.path.join(out_dir, 'wp4_conditioning.pdf')
    fig.savefig(path)
    plt.close(fig)
    return path


# LaTeX tables

def _latex_escape(text: str) -> str:
    for char, repl in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'),
                       ('&', r'\&'), ('#', r'\#')):
        text = text.replace(char, repl)
    return text


def _format_number(value: float) -> str:
    if value != value:
        return '--'
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-3):
        return f'{value:.3g}'
    return f'{value:.3f}'


def latex_table(caption: str, label: str, headers: list, body_rows: list) -> str:
    align = 'l' + 'r' * (len(headers) - 1)
    lines = [
        r'\begin{table}[htbp]', r'  \centering',
        f'  \\caption{{{caption}}}', f'  \\label{{tab:{label}}}',
        f'  \\begin{{tabular}}{{{align}}}', r'    \toprule',
        '    ' + ' & '.join(_latex_escape(str(h)) for h in headers) + r' \\',
        r'    \midrule',
    ]
    for row in body_rows:
        cells = []
        for value in row:
            cells.append(_format_number(value) if isinstance(value, float)
                         else _latex_escape(str(value)))
        lines.append('    ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


SUMMARY_METRICS = {
    'wp2': [('held_out.mean', 'held-out [mm]'),
            ('in_sample.rms_all_mm', 'in-sample [mm]'),
            ('truth.translation_error_mm', 'true err [mm]')],
    'wp3': [('held_out.translation_mm.mean', 'held-out [mm]'),
            ('held_out.rotation_deg.mean', 'held-out [deg]'),
            ('in_sample.consistency_rms_mm', 'consistency [mm]'),
            ('truth.translation_error_mm', 'true err [mm]'),
            ('truth.rotation_error_deg', 'true err [deg]')],
    'wp4': [('held_out.corrected_rms_mm.mean', 'held-out [mm]'),
            ('held_out.nominal_rms_mm.mean', 'nominal held-out [mm]'),
            ('in_sample.rms_after_mm', 'in-sample [mm]'),
            ('truth.fk_agreement.position_rms_mm', 'true FK err [mm]'),
            ('conditioning.condition_number', 'cond')],
}


def table_for_wp(rows, wp: str, calibration_type: str | None = None) -> str | None:
    wp_rows = select(rows, work_package=wp)
    if calibration_type:
        wp_rows = select(wp_rows, calibration_type=calibration_type)
    variants = [v for v in distinct(wp_rows, 'variant') if v != '-']
    if not variants:
        return None

    headers = ['variant', 'n'] + [h for _m, h in SUMMARY_METRICS[wp]]
    body = []
    for variant in variants:
        cells = [variant.replace('|', ', ')]
        counts = 0
        values = []
        for metric, _heading in SUMMARY_METRICS[wp]:
            picked = [r['value'] for r in select(wp_rows, variant=variant, metric=metric)
                      if r['value'] is not None]
            summary = metrics.summarize(picked)
            counts = max(counts, summary.get('n', 0))
            values.append(summary.get('mean'))
        if counts == 0:
            continue
        cells.append(counts)
        cells.extend(v if v is not None else float('nan') for v in values)
        body.append(cells)

    if not body:
        return None
    mounting = (f' for the {calibration_type.replace("_", "-")} mounting'
                if calibration_type else '')
    return latex_table(
        caption=(f'{wp.upper()} calibration quality{mounting}, mean over $n$ runs. '
                 f'Held-out columns are available on hardware; true-error columns '
                 f'require simulation.'),
        label=f'{wp}-summary{_suffix(calibration_type)}'.replace('_', '-'),
        headers=headers, body_rows=body)


# entry point

def build_report(csv_path: str, out_dir: str) -> dict:
    plt = _configure_matplotlib()
    rows = load_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    produced = {'figures': [], 'tables': [], 'skipped': []}

    figure_specs = [
        ('wp2_error_vs_touch_noise',
         dict(x_column='touch_noise_std_m', metric='held_out.mean', wp='wp2',
              ylabel='held-out pivot residual [mm]',
              title='WP2: TCP accuracy vs. touch precision',
              filename='wp2_error_vs_touch_noise.pdf', split_by='mode')),
        ('wp2_error_vs_num_poses',
         dict(x_column='num_poses', metric='held_out.mean', wp='wp2',
              ylabel='held-out pivot residual [mm]',
              title='WP2: how many touches are enough?',
              filename='wp2_error_vs_num_poses.pdf', split_by='mode')),
        ('wp3_error_vs_image_noise',
         dict(x_column='image_noise_stddev', metric='held_out.translation_mm.mean',
              wp='wp3', ylabel='held-out target-pose error [mm]',
              title='WP3: hand-eye accuracy vs. image noise',
              filename='wp3_error_vs_image_noise.pdf')),
        ('wp3_error_vs_num_samples',
         dict(x_column='num_samples', metric='held_out.translation_mm.mean', wp='wp3',
              ylabel='held-out target-pose error [mm]',
              title='WP3: how many poses are enough?',
              filename='wp3_error_vs_num_samples.pdf')),
        ('wp4_error_vs_num_samples',
         dict(x_column='num_samples', metric='held_out.corrected_rms_mm.mean', wp='wp4',
              ylabel='held-out loop-closure RMS [mm]',
              title='WP4: kinematic identification vs. sample count',
              filename='wp4_error_vs_num_samples.pdf')),
        ('wp4_error_vs_perturbation',
         dict(x_column='perturbation_sigma_t_mm',
              metric='truth.fk_agreement.position_rms_mm', wp='wp4',
              ylabel='true FK position error [mm]',
              title='WP4: recovery vs. injected error magnitude',
              filename='wp4_error_vs_perturbation.pdf')),
    ]

    for name, kwargs in figure_specs:
        try:
            path = figure_error_vs_condition(plt, rows, out_dir, **kwargs)
        except Exception as exc:
            produced['skipped'].append(f'{name}: {type(exc).__name__}: {exc}')
            continue
        (produced['figures'].append(path) if path
         else produced['skipped'].append(f'{name}: no data'))

    for name, fn in (('held_out_vs_truth', figure_held_out_vs_truth),
                     ('wp4_conditioning', figure_conditioning)):
        try:
            path = fn(plt, rows, out_dir)
        except Exception as exc:
            produced['skipped'].append(f'{name}: {type(exc).__name__}: {exc}')
            continue
        (produced['figures'].append(path) if path
         else produced['skipped'].append(f'{name}: no data'))

    for calibration_type in calibration_types(rows, 'wp4') or [None]:
        name = f'wp4_overfitting{_suffix(calibration_type)}'
        try:
            path = figure_overfitting(plt, rows, out_dir, calibration_type)
        except Exception as exc:
            produced['skipped'].append(f'{name}: {type(exc).__name__}: {exc}')
            continue
        (produced['figures'].append(path) if path
         else produced['skipped'].append(f'{name}: no data'))

    for wp in ('wp2', 'wp3', 'wp4'):
        for calibration_type in calibration_types(rows, wp) or [None]:
            label = f'{wp}{_suffix(calibration_type)}'
            table = table_for_wp(rows, wp, calibration_type)
            if table is None:
                produced['skipped'].append(f'table {label}: no data')
                continue
            path = os.path.join(
                out_dir, f'{wp}_summary{_suffix(calibration_type)}.tex')
            with open(path, 'w') as f:
                f.write(table)
            produced['tables'].append(path)

    return produced


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='benchmark_report', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--out-dir', default=str(paths.data_dir() / 'benchmark'))
    parser.add_argument('--csv', default=None,
                        help='default: <out-dir>/results.csv')
    parser.add_argument('--report-dir', default=None,
                        help='default: <out-dir>/report')
    args = parser.parse_args(argv)

    csv_path = args.csv or os.path.join(args.out_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f'no results CSV at {csv_path}; run benchmark_aggregate first',
              file=sys.stderr)
        return 1

    report_dir = args.report_dir or os.path.join(args.out_dir, 'report')
    produced = build_report(csv_path, report_dir)

    for path in produced['figures']:
        print(f'figure  {path}')
    for path in produced['tables']:
        print(f'table   {path}')
    for reason in produced['skipped']:
        print(f'skipped {reason}')
    print(f'\n{len(produced["figures"])} figure(s), {len(produced["tables"])} table(s) '
          f'in {report_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
