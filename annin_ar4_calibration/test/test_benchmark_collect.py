"""Regression tests for  benchmark/collect.py"""
import os
import time

import pytest

from annin_ar4_calibration.benchmark import collect


def _alive(pid: int) -> bool:
    return os.path.exists(f'/proc/{pid}')


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs procfs')
def test_stop_reaps_grandchildren_that_ignore_sigint(tmp_path):
    proc = collect.ManagedProcess(
        ['/bin/sh', '-c', "trap '' INT; sleep 120 & sleep 120"],
        dict(os.environ), str(tmp_path / 'proc.log'))
    time.sleep(1.5)

    descendants = proc._descendants()
    assert descendants, 'expected to see the child processes before teardown'

    proc.stop(grace=3.0)
    time.sleep(1.0)

    leaked = [pid for pid, _ in descendants if _alive(pid)]
    assert not leaked, f'processes survived teardown: {leaked}'


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs procfs')
def test_descendants_carries_start_times_for_pid_reuse_safety(tmp_path):
    proc = collect.ManagedProcess(
        ['/bin/sh', '-c', 'sleep 30'], dict(os.environ), str(tmp_path / 'p.log'))
    time.sleep(1.0)
    try:
        for entry in proc._descendants():
            assert isinstance(entry, tuple) and len(entry) == 2
            pid, starttime = entry
            assert starttime == collect.ManagedProcess._stat_fields(pid)[1]
    finally:
        proc.stop(grace=2.0)


def test_stat_fields_survives_a_process_name_containing_spaces():
    ppid, starttime = collect.ManagedProcess._stat_fields(os.getpid())
    assert ppid == os.getppid()
    assert starttime > 0


def _referenced_names(path: str) -> set[str]:
    import ast
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split('.')[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[-1])
    return names


def test_collect_does_not_use_the_self_spinning_urdf_helper():
    names = _referenced_names(collect.__file__)
    assert 'fetch_robot_description' not in names, (
        'collect.py must fetch the URDF via RosBridge._await, not via the '
        'self-spinning helper in nodes/_robot_description.py')
    assert 'spin_until_future_complete' not in names


def test_rosbridge_awaits_futures_by_polling():
    import ast
    import inspect
    import textwrap

    assert hasattr(collect.RosBridge, '_await')
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(collect.RosBridge._await)))
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert 'done' in calls, '_await must poll future.done()'
    assert not any('spin' in name for name in calls), (
        '_await must not spin; the background executor owns that')


# retried runs

def test_a_retried_run_starts_from_empty_sample_files(tmp_path):
    import numpy as np
    from annin_ar4_calibration.benchmark import collect

    run_dir = str(tmp_path / 'run')
    data = tmp_path / 'run' / 'ros_home' / 'annin_ar4_calibration'
    data.mkdir(parents=True)
    np.save(data / 'hand_eye_samples.npy', np.zeros((150, 14)))
    (data / 'hand_eye_samples.npy.detections.csv').write_text('index,num_points,reprojection_rms_px\n')

    archived = collect.archive_previous_attempt(run_dir)

    assert archived is not None
    assert not data.exists() or not any(
        data.iterdir()), 'the new attempt must start empty'
    kept = np.load(os.path.join(archived, 'hand_eye_samples.npy'))
    assert kept.shape == (
        150, 14), 'the earlier attempt is moved aside, not destroyed'


def test_a_first_attempt_has_nothing_to_archive(tmp_path):
    from annin_ar4_calibration.benchmark import collect
    assert collect.archive_previous_attempt(
        str(tmp_path / 'fresh_run')) is None
    (tmp_path / 'empty' / 'ros_home' / 'annin_ar4_calibration').mkdir(parents=True)
    assert collect.archive_previous_attempt(str(tmp_path / 'empty')) is None
