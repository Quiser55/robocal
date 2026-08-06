"""Persistence for collected samples and computed calibration results.

Samples are stored as a plain (N, 7) float64 numpy array, columns
[tx, ty, tz, qx, qy, qz, qw]. Results are stored as YAML. Both are written
atomically (temp file + os.replace) so a killed process can never leave a
corrupt file behind - a reader always sees either the previous complete file
or the new complete one.
"""
import datetime
import os

import numpy as np
import yaml

SAMPLE_COLUMNS = ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')


class SampleFileError(RuntimeError):
    pass


class ResultFileError(RuntimeError):
    pass


def _atomic_write_bytes(path: str, write_fn) -> None:
    tmp_path = f'{path}.tmp'
    write_fn(tmp_path)
    os.replace(tmp_path, path)


def load_samples(path: str, ncols: int = 7) -> np.ndarray:
    if not os.path.exists(path):
        return np.empty((0, ncols), dtype=np.float64)
    try:
        samples = np.load(path)
    except Exception as exc:
        raise SampleFileError(f'Could not read sample file {path}: {exc}') from exc

    if samples.ndim != 2 or samples.shape[1] != ncols:
        raise SampleFileError(
            f'Sample file {path} has unexpected shape {samples.shape}, expected (N, {ncols})')
    if not np.all(np.isfinite(samples)):
        raise SampleFileError(f'Sample file {path} contains non-finite values')
    return samples.astype(np.float64)


def append_sample(path: str, row: np.ndarray, ncols: int = 7) -> int:
    samples = load_samples(path, ncols=ncols)
    samples = np.vstack([samples, row.reshape(1, ncols)])

    def _write(tmp_path):
        np.save(tmp_path, samples)
        # np.save appends a .npy suffix if not already present
        if not tmp_path.endswith('.npy') and os.path.exists(tmp_path + '.npy'):
            os.replace(tmp_path + '.npy', tmp_path)

    _atomic_write_bytes(path, _write)
    return samples.shape[0]


def reset_samples(path: str, keep_backup: bool = True, ncols: int = 7) -> str | None:
    """Overwrites the sample file with an empty array. Returns the backup path
    (or None if there was nothing to back up / keep_backup is False)."""
    backup_path = None
    if os.path.exists(path):
        if keep_backup:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            backup_path = f'{path}.bak-{timestamp}'
            os.replace(path, backup_path)
        else:
            os.remove(path)

    empty = np.empty((0, ncols), dtype=np.float64)

    def _write(tmp_path):
        np.save(tmp_path, empty)
        if not tmp_path.endswith('.npy') and os.path.exists(tmp_path + '.npy'):
            os.replace(tmp_path + '.npy', tmp_path)

    _atomic_write_bytes(path, _write)
    return backup_path


def save_result(path: str, result: dict) -> None:
    def _write(tmp_path):
        with open(tmp_path, 'w') as f:
            yaml.safe_dump(result, f, default_flow_style=False, sort_keys=False)

    _atomic_write_bytes(path, _write)


def load_result(
        path: str, key: str = 'tcp_calibration',
        required_keys: tuple = ('offset', 'tool_frame', 'tcp_frame'),
) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ResultFileError(f'Could not read result file {path}: {exc}') from exc

    if not isinstance(data, dict) or key not in data:
        raise ResultFileError(f"Result file {path} is missing the '{key}' key")
    result = data[key]
    missing = [k for k in required_keys if k not in result]
    if missing:
        raise ResultFileError(f'Result file {path} is missing required keys: {missing}')
    return result


def _meta_path(sample_file: str) -> str:
    return f'{sample_file}.meta.yaml'


def save_meta(sample_file: str, meta: dict) -> None:
    path = _meta_path(sample_file)
    if os.path.exists(path):
        return

    def _write(tmp_path):
        with open(tmp_path, 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False)

    _atomic_write_bytes(path, _write)


def load_meta(sample_file: str) -> dict | None:
    path = _meta_path(sample_file)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None
