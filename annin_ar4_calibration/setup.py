from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'annin_ar4_calibration'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
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
    ],
)