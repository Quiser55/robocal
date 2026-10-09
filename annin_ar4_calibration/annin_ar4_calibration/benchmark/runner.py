"""Execute a sweep, or re-analyze data already collected.

Three modes, sharing one analysis path:

- ``--mode sim`` drives Gazebo
- ``--mode hardware`` drives a live arm that is already up (driver + MoveIt + camera launched by the operator)
- ``--mode offline`` re-analyzes archived runs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from annin_ar4_calibration.benchmark import analyze, collect, spec, trace
from annin_ar4_calibration.core import paths

#: Estimate used to calculate the total time a sweep will take, so the operator can check it before starting.
MINUTES_PER_SIM_RUN = 6

DEFAULT_TIMEOUTS = {
    'ready': 180.0,     # sim spawn + controllers + first detection
    'collect': 3600.0,  # a full auto sequence over ~100 configs
    'service': 300.0,   # compute_calibration and friends
}


def parse_timeouts(overrides: list[str] | None) -> dict:
    timeouts = dict(DEFAULT_TIMEOUTS)
    for item in overrides or []:
        key, _, value = item.partition('=')
        if key not in timeouts:
            raise SystemExit(
                f'unknown timeout {key!r}; choose from {sorted(timeouts)}')
        try:
            timeouts[key] = float(value)
        except ValueError:
            raise SystemExit(f'--timeout {item}: {value!r} is not a number')
    return timeouts


def default_out_dir() -> str:
    return str(paths.data_dir() / 'benchmark')


def resolve_sweep(name: str) -> str:
    """Accept either a path or a bare sweep name"""
    if os.path.exists(name):
        return name
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('annin_ar4_calibration'), 'sweeps', name))
    except Exception:
        pass
    candidates.append(os.path.join(os.path.dirname(__file__), 'sweeps', name))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'no sweep {name!r}; looked in {[name] + candidates}')


def _run_dir(out_dir: str, run_spec: spec.RunSpec) -> str:
    return os.path.join(out_dir, 'runs', run_spec.run_id)


def _already_analyzed(run_dir: str) -> bool:
    """True when `analyze_run` has already written its output for this cell."""
    return os.path.exists(os.path.join(run_dir, 'metrics.yaml'))


def _already_collected(run_dir: str) -> bool:
    manifest = os.path.join(run_dir, 'manifest.yaml')
    if not os.path.exists(manifest):
        return False
    import yaml
    try:
        with open(manifest) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return False
    return (data.get('manifest') or data).get('outcome') == collect.OUTCOME_OK


def cmd_sweep(args) -> int:
    sweep_path = resolve_sweep(args.sweep)
    specs = spec.load_sweep(sweep_path)
    if args.mode:
        for run_spec in specs:
            run_spec.mode = args.mode
    if args.limit:
        specs = specs[:args.limit]

    out_dir = args.out_dir or default_out_dir()
    print(f'{len(specs)} run(s) from {sweep_path} -> {out_dir}')
    if specs and specs[0].mode == 'sim' and not args.dry_run:
        minutes = len(specs) * MINUTES_PER_SIM_RUN * len(specs[0].work_packages)
        print(f'estimated {minutes / 60:.0f} h at ~{MINUTES_PER_SIM_RUN} min per work '
              f'package; use --limit N to try a subset first')

    failures = 0
    for i, run_spec in enumerate(specs, start=1):
        run_dir = _run_dir(out_dir, run_spec)
        print(f'[{i}/{len(specs)}] {run_spec.run_id}  ({run_spec.label()})')

        if args.resume and not args.force and _already_collected(run_dir) \
                and _already_analyzed(run_dir):
            print('    already complete, skipping (--resume)')
            continue

        if _already_collected(run_dir) and not args.force:
            print('    already collected, re-analyzing only')
        else:
            started = time.time()
            manifest = collect.collect_run(
                run_spec, run_dir, parse_timeouts(args.timeout),
                dry_run=args.dry_run, keep_alive=args.keep_alive)
            outcome = manifest.get('outcome')
            print(f'    {outcome} in {time.time() - started:.0f}s')
            if outcome not in (collect.OUTCOME_OK, 'dry_run'):
                failures += 1
                print(f'    {manifest.get("error", "")}')
                trace.debug(f'logs for this run: {run_dir}/*.log')
                continue

        if args.dry_run:
            continue
        document = analyze.analyze_run(run_dir, run_spec)
        _print_summary(document)

    print(f'\ndone: {len(specs) - failures}/{len(specs)} collected successfully')
    print(f'next: benchmark_aggregate --out-dir {out_dir}')
    return 1 if failures == len(specs) and specs else 0


def cmd_offline(args) -> int:
    out_dir = args.out_dir or default_out_dir()
    runs_dir = args.runs_dir or os.path.join(out_dir, 'runs')

    if args.run_dir:
        targets = [args.run_dir]
    else:
        if not os.path.isdir(runs_dir):
            print(f'no runs directory at {runs_dir}', file=sys.stderr)
            return 1
        targets = [os.path.join(runs_dir, n) for n in sorted(os.listdir(runs_dir))
                   if os.path.isdir(os.path.join(runs_dir, n))]

    if not targets:
        print(f'no run directories under {runs_dir}', file=sys.stderr)
        return 1

    for run_dir in targets:
        print(f'analyzing {os.path.basename(run_dir)}')
        _print_summary(analyze.analyze_run(run_dir))
    print(f'\nnext: benchmark_aggregate --out-dir {out_dir}')
    return 0


def _print_summary(document: dict) -> None:
    for wp, data in (document.get('work_packages') or {}).items():
        if 'skipped' in data:
            print(f'    {wp}: skipped - {data["skipped"]}')
            continue
        if 'error' in data:
            print(f'    {wp}: error - {data["error"]}')
            continue
        for label, entry in (data.get('variants') or {}).items():
            bits = []
            held_out = entry.get('held_out') or entry.get('held_out_rms_mm')
            if isinstance(held_out, dict):
                if 'corrected_rms_mm' in held_out:
                    bits.append(f'held-out {held_out["corrected_rms_mm"].get("mean", 0):.3f} mm '
                                f'(nominal {held_out["nominal_rms_mm"].get("mean", 0):.3f})')
                elif 'translation_mm' in held_out:
                    bits.append(
                        f'held-out {held_out["translation_mm"].get("mean", 0):.3f} mm')
                elif 'mean' in held_out:
                    bits.append(f'held-out {held_out["mean"]:.3f} mm')
            truth = entry.get('truth')
            if isinstance(truth, dict):
                if 'translation_error_mm' in truth:
                    bits.append(f'true err {truth["translation_error_mm"]:.3f} mm')
                elif 'fk_agreement' in truth:
                    bits.append(
                        f'true FK err {truth["fk_agreement"]["position_rms_mm"]:.3f} mm')
            if bits:
                print(f'    {wp} [{label}]: ' + ', '.join(bits))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='benchmark_run', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['sim', 'hardware', 'offline'], default=None,
                        help="default: the sweep file's own `mode:` (usually sim)")
    parser.add_argument('--sweep', help='sweep YAML (required for sim/hardware)')
    parser.add_argument('--out-dir', default=None,
                        help=f'default: {default_out_dir()}')
    parser.add_argument('--runs-dir', default=None,
                        help='offline mode: directory of run dirs to re-analyze')
    parser.add_argument('--run-dir', default=None,
                        help='offline mode: a single run directory')
    parser.add_argument('--limit', type=int, default=0,
                        help='only execute the first N runs of the sweep')
    parser.add_argument('--force', action='store_true',
                        help='re-collect cells that already completed successfully')
    parser.add_argument('--resume', action='store_true',
                        help='skip cells that are already collected AND analyzed, '
                             'instead of re-analyzing them; failed cells are still '
                             'retried, so this is how you re-run a sweep to fill in '
                             'only what did not finish')
    parser.add_argument('--debug', action='store_true',
                        help='stream child process logs, tighten the progress '
                             'heartbeat, and report each step as it completes')
    parser.add_argument('--timeout', action='append', metavar='NAME=SECONDS',
                        help=f'override a timeout {sorted(DEFAULT_TIMEOUTS)}; '
                             'repeatable')
    parser.add_argument('--keep-alive', action='store_true',
                        help='on failure, leave Gazebo and the nodes running for '
                             'inspection instead of tearing them down')
    parser.add_argument('--dry-run', action='store_true',
                        help='expand the sweep and write manifests without launching '
                             'anything - use this to check a grid before an overnight run')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trace.enable(args.debug)

    if args.mode == 'offline':
        return cmd_offline(args)
    if not args.sweep:
        parser.error('--sweep is required for sim/hardware mode')
    return cmd_sweep(args)


if __name__ == '__main__':
    sys.exit(main())
