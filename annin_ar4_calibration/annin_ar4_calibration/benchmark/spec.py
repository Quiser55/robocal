"""Sweep definitions: a YAML grid expands into the individual runs to execute.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field

import yaml

#: Work packages, in the order they must run when several share one session.
#: WP4 consumes WP3's hand_eye_result.yaml, so it can never precede it.
WORK_PACKAGE_ORDER = ('wp2', 'wp3', 'wp4')

#: Parameters that describe the physical experiment. Everything here feeds the
#: run_id hash; anything not here is free to change without re-collecting.
COLLECTION_KEYS = (
    'calibration_type',
    'perturbation_file',
    'perturbation_sigma_t_mm',
    'perturbation_sigma_r_deg',
    'image_noise_stddev',
    'camera_width',
    'camera_height',
    'num_samples',
    'num_candidates',
    'touch_noise_std_m',
    'num_poses',
    'ar_model',
)

DEFAULT_COLLECTION = {
    'calibration_type': 'eye_to_hand',
    'perturbation_file': '',
    'perturbation_sigma_t_mm': 1.0,
    'perturbation_sigma_r_deg': 0.1,
    'image_noise_stddev': 0.0,
    'camera_width': 1280,
    'camera_height': 720,
    'num_samples': 100,
    'num_candidates': 400,
    'touch_noise_std_m': 0.0,
    'num_poses': 40,
    'ar_model': 'mk5',
}

DEFAULT_ANALYSIS = {
    'kfold': 5,
    'bootstrap': 100,
    'bootstrap_seed': 0,
    'kfold_seed': 0,
}

#: Applied when a sweep declares no `variants:` block of its own.
DEFAULT_VARIANTS = {
    'wp2': {'ransac': [True, False]},
    'wp3': {'method': ['TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS']},
    'wp4': {'method': ['sequential', 'joint_trf', 'joint_lm'],
            'ridge_lambda': [1e-6]},
}


@dataclass
class RunSpec:
    """One collection: a single (condition, seed) cell of a sweep."""
    sweep: str
    mode: str                      # 'sim' | 'hardware'
    work_packages: list
    seed: int
    collection: dict
    analysis: dict = field(default_factory=lambda: dict(DEFAULT_ANALYSIS))
    variants: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        payload = {
            'sweep': self.sweep,
            'mode': self.mode,
            'work_packages': list(self.work_packages),
            'seed': int(self.seed),
            'collection': {k: self.collection.get(k) for k in sorted(COLLECTION_KEYS)},
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        digest = hashlib.sha1(blob).hexdigest()[:10]
        return f'{self.sweep}-s{self.seed:03d}-{digest}'

    def label(self) -> str:
        parts = [f'seed={self.seed}']
        for key in COLLECTION_KEYS:
            value = self.collection.get(key)
            if value != DEFAULT_COLLECTION.get(key):
                parts.append(f'{key}={value}')
        return ', '.join(parts)

    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'sweep': self.sweep,
            'mode': self.mode,
            'work_packages': list(self.work_packages),
            'seed': int(self.seed),
            'collection': dict(self.collection),
            'analysis': dict(self.analysis),
            'variants': dict(self.variants),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RunSpec':
        return cls(
            sweep=data['sweep'], mode=data['mode'],
            work_packages=list(data['work_packages']), seed=int(data['seed']),
            collection=dict(data['collection']),
            analysis=dict(data.get('analysis') or DEFAULT_ANALYSIS),
            variants=dict(data.get('variants') or {}))


def _ordered_work_packages(names) -> list:
    unknown = [n for n in names if n not in WORK_PACKAGE_ORDER]
    if unknown:
        raise ValueError(f'unknown work package(s) {unknown}; expected {WORK_PACKAGE_ORDER}')
    return [wp for wp in WORK_PACKAGE_ORDER if wp in names]


def load_sweep(path: str) -> list[RunSpec]:
    """Expand a sweep YAML into one RunSpec per (grid point, seed)."""
    with open(path) as f:
        document = yaml.safe_load(f) or {}

    name = document.get('name') or os.path.splitext(os.path.basename(path))[0]
    mode = document.get('mode', 'sim')
    work_packages = _ordered_work_packages(document.get('work_packages') or ['wp3', 'wp4'])
    seeds = document.get('seeds') or [0]
    fixed = document.get('fixed') or {}
    grid = document.get('grid') or {}
    analysis = {**DEFAULT_ANALYSIS, **(document.get('analysis') or {})}

    variants = document.get('variants')
    if variants is None:
        variants = {wp: DEFAULT_VARIANTS[wp] for wp in work_packages}

    grid_keys = sorted(grid)
    axes = []
    for key in grid_keys:
        values = grid[key] if isinstance(grid[key], list) else [grid[key]]
        if key.startswith('_'):
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}: coupled grid axis '{key}' must be a list of mappings, "
                        f'got {type(value).__name__}')
            axes.append(list(values))
        else:
            axes.append([{key: value} for value in values])

    declared = set(fixed)
    for axis in axes:
        for point in axis:
            declared |= set(point)
    unknown = declared - set(COLLECTION_KEYS)
    if unknown:
        raise ValueError(
            f'{path}: unknown collection parameter(s) {sorted(unknown)}. Solver settings '
            f'belong under `variants:` - putting them in `grid:` would re-collect data '
            f'that only needs re-solving. Known keys: {sorted(COLLECTION_KEYS)}')

    specs = []
    for combo in itertools.product(*axes) if axes else [()]:
        point = {k: v for part in combo for k, v in part.items()}
        collection = {**DEFAULT_COLLECTION, **fixed, **point}
        for seed in seeds:
            specs.append(RunSpec(
                sweep=name, mode=mode, work_packages=work_packages, seed=int(seed),
                collection=collection, analysis=analysis, variants=variants))
    return specs


def variant_combinations(variants_for_wp: dict) -> list[dict]:
    if not variants_for_wp:
        return [{}]
    keys = sorted(variants_for_wp)
    values = [v if isinstance(v, list) else [v] for v in (variants_for_wp[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def variant_label(variant: dict) -> str:
    if not variant:
        return 'default'
    return '|'.join(f'{k}={variant[k]}' for k in sorted(variant))
