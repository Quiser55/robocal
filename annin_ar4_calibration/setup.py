from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'annin_ar4_calibration'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
    # colcon picks its Python testing step by looking for a 'pytest' test
    # dependency here. Without it, `colcon test` falls back to
    # `python -m unittest`, which collects none of these pytest-style tests and
    # reports "NO TESTS RAN" as a failure - so the suite only ever ran by hand.
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Jonas Hils',
    description='Calibration package for Annin Ar4 robot.',
    license="MIT",
    entry_points={
        'console_scripts': [
            'collector = annin_ar4_calibration.nodes.collector_node:main',
            'calibration = annin_ar4_calibration.nodes.calibration_node:main',
            'visualization = annin_ar4_calibration.nodes.visualization_node:main',
            'target_detector = annin_ar4_calibration.nodes.target_detector_node:main',
            'hand_eye_collector = annin_ar4_calibration.nodes.hand_eye_collector_node:main',
            'hand_eye_calibration = annin_ar4_calibration.nodes.hand_eye_calibration_node:main',
            'hand_eye_visualization = '
            'annin_ar4_calibration.nodes.hand_eye_visualization_node:main',
            'kinematic_collector = '
            'annin_ar4_calibration.nodes.kinematic_collector_node:main',
            'kinematic_calibration = '
            'annin_ar4_calibration.nodes.kinematic_calibration_node:main',
            'kinematic_visualization = '
            'annin_ar4_calibration.nodes.kinematic_visualization_node:main',
            # Simulation-only (annin_ar4_gazebo); no-ops against real hardware.
            'ground_truth = annin_ar4_calibration.nodes.ground_truth_node:main',
            'tcp_sim_poser = annin_ar4_calibration.nodes.tcp_sim_poser_node:main',
            # Benchmark harness. Plain CLIs, not ROS nodes - benchmark_run
            # launches the nodes above as subprocesses, and the other two never
            # touch ROS at all.
            'benchmark_run = annin_ar4_calibration.benchmark.runner:main',
            'benchmark_aggregate = annin_ar4_calibration.benchmark.aggregate:main',
            'benchmark_report = annin_ar4_calibration.benchmark.report:main',
            'benchmark_thesis_numbers = annin_ar4_calibration.benchmark.thesis_numbers:main',
        ],
    },
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'),
    glob(os.path.join('launch', '*.launch.py'))
    # sim_params.py is imported by the three launch files, so it has to sit
    # beside them in share/ - the *.launch.py glob alone would not install it.
    + glob(os.path.join('launch', 'sim_params.py'))),
    (os.path.join('share', package_name, 'config'),
    glob(os.path.join('config', '*.yaml'))),
    # Sweep definitions are data, not code - installed into share/ so
    # `benchmark_run --sweep <name>.yaml` works from an installed workspace,
    # not just from a source checkout.
    (os.path.join('share', package_name, 'sweeps'),
    glob(os.path.join(package_name, 'benchmark', 'sweeps', '*.yaml'))),
    ],
)