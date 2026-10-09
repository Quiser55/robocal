"""tests that launch files and config YAMLs are consistent with each other,
 and that the calibration_type argument is not set in YAML anywhere"""
import ast
import os
import re

import pytest
import yaml

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PACKAGE_DIR, 'config')
LAUNCH_DIR = os.path.join(PACKAGE_DIR, 'launch')

LAUNCH_FILES = ('tcp.launch.py', 'hand_eye.launch.py', 'kinematic.launch.py')
CONFIG_FILES = ('tcp_params.yaml', 'hand_eye_params.yaml', 'kinematic_params.yaml',
                'sim_overrides.yaml')


def _launch_overridden_parameters(path: str) -> set[str]:
    """Parameter names a launch file passes as a dict"""
    tree = ast.parse(open(path).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
    return names


def _yaml_parameter_names(path: str) -> dict[str, set[str]]:
    """{node_name: {parameter names}} for a ROS 2 params file."""
    document = yaml.safe_load(open(path)) or {}
    out = {}
    for node_name, block in document.items():
        if isinstance(block, dict) and 'ros__parameters' in block:
            out[node_name] = set(block['ros__parameters'] or {})
    return out


@pytest.mark.parametrize('launch_file', LAUNCH_FILES)
def test_launch_overrides_are_not_also_set_in_config_yaml(launch_file):
    overridden = _launch_overridden_parameters(
        os.path.join(LAUNCH_DIR, launch_file))
    # 'ros__parameters' shows up as a dict key in some launch idioms
    overridden.discard('ros__parameters')
    if not overridden:
        pytest.skip(f'{launch_file} passes no parameter dicts')

    collisions = []
    for config_file in CONFIG_FILES:
        path = os.path.join(CONFIG_DIR, config_file)
        if not os.path.exists(path):
            continue
        for node_name, params in _yaml_parameter_names(path).items():
            if node_name == '/**':
                continue    # wildcard vs wildcard resolves by file order; fine
            for clash in overridden & params:
                collisions.append(f'{config_file}:{node_name}:{clash}')

    assert not collisions, (
        f'{launch_file} overrides {sorted(overridden)} via a `/**` parameter dict, but '
        f'these are also set under specific node keys: {collisions}. Which one wins is '
        f'not stable - remove them from the YAML and let the launch argument decide.')


def test_calibration_type_is_absent_from_every_config_file():
    offenders = []
    for config_file in CONFIG_FILES:
        path = os.path.join(CONFIG_DIR, config_file)
        if not os.path.exists(path):
            continue
        for node_name, params in _yaml_parameter_names(path).items():
            if 'calibration_type' in params:
                offenders.append(f'{config_file}:{node_name}')
    assert not offenders, (
        f'calibration_type must come from the launch argument only, but is set in '
        f'{offenders}. eye_in_hand and eye_to_hand solve for different frame pairs; '
        f'picking the wrong one still converges and still reports a plausible RMS.')


def test_launch_files_still_declare_the_arguments_they_override():
    for launch_file in ('hand_eye.launch.py', 'kinematic.launch.py'):
        source = open(os.path.join(LAUNCH_DIR, launch_file)).read()
        for argument in ('calibration_type', 'target_type'):
            assert re.search(rf"DeclareLaunchArgument\(\s*'{argument}'", source), (
                f'{launch_file} must declare {argument} - the config YAML no longer '
                f'provides a default for it')


def test_kinematic_calibration_rejects_a_hand_eye_type_mismatch():
    source = open(os.path.join(
        PACKAGE_DIR, 'annin_ar4_calibration', 'nodes',
        'kinematic_calibration_node.py')).read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != '_load_hand_eye_result':
            continue
        raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
        assert len(raises) >= 2, (
            '_load_hand_eye_result must raise on a calibration_type mismatch, not '
            'just on a missing file')
        warnings = [n for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'warning']
        assert not warnings, 'a mismatch must be fatal, not a warning'
        return
    pytest.fail('_load_hand_eye_result not found')
