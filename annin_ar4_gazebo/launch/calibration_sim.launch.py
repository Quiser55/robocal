"""Bring up the AR4 in Gazebo with a camera and ChArUco board, for running
annin_ar4_calibration's WP2/WP3/WP4 in simulation with known ground truth.

    ros2 launch annin_ar4_gazebo calibration_sim.launch.py \
        calibration_type:=eye_to_hand \
        perturbation_file:=<share>/annin_ar4_gazebo/config/perturbations/example_1mm.yaml

Then run one work package at a time in a second terminal, e.g.

    ros2 launch annin_ar4_calibration hand_eye.launch.py \
        use_sim_time:=True calibration_type:=eye_to_hand

The ground-truth trick, in one paragraph: robot_state_publisher publishes the
NOMINAL description, and Gazebo spawns a PERTURBED copy of it whose six joint
origins carry a known error from `perturbation_file`. Everything that estimates
- MoveIt, ros2_control, every calibration node - therefore sees the nominal
model, while the physical robot really is slightly different. WP4 then has a
genuine error to identify whose true value is known exactly, which is the one
thing physical hardware can never provide. With the default zero.yaml the two
models coincide and the run is an unperturbed baseline.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(__file__))
from sim_launch_common import (  # noqa: E402
    ControllerConfigSubstitution,
    PerturbedUrdfSubstitution,
    gz_resource_path_value,
)

# gz-side topic namespace, matching sim_camera_macro.xacro's <topic> elements.
GZ_CAMERA_NS = '/sim_camera'
# ROS-side topic names the calibration package and sensors_3d.yaml already
# expect (the realsense2_camera defaults on Jazzy). Remapped, not renamed, so
# nothing downstream has to know it is talking to a simulator.
ROS_COLOR_IMAGE = '/camera/camera/color/image_raw'
ROS_COLOR_INFO = '/camera/camera/color/camera_info'
ROS_DEPTH_IMAGE = '/camera/camera/aligned_depth_to_color/image_raw'


def launch_setup(context, *args, **kwargs):
    calibration_type = LaunchConfiguration('calibration_type')
    ar_model = LaunchConfiguration('ar_model')
    tf_prefix = LaunchConfiguration('tf_prefix')
    enable_depth = LaunchConfiguration('enable_depth').perform(context).lower() == 'true'

    controllers_file = ControllerConfigSubstitution(
        PathJoinSubstitution(
            [FindPackageShare('annin_ar4_driver'), 'config', 'controllers.yaml']),
        tf_prefix=tf_prefix)

    # The nominal description: what everything that *estimates* gets to see.
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([
            FindPackageShare('annin_ar4_description'), 'urdf', 'ar_gazebo.urdf.xacro']), ' ',
        'ar_model:=', ar_model, ' ',
        'tf_prefix:=', tf_prefix, ' ',
        'simulation_controllers:=', controllers_file, ' ',
        # The rig follows the calibration type: eye_in_hand puts the camera on
        # the wrist and the board in the workspace, eye_to_hand the reverse.
        'mount_mode:=', calibration_type, ' ',
        'camera_width:=', LaunchConfiguration('camera_width'), ' ',
        'camera_height:=', LaunchConfiguration('camera_height'), ' ',
        'camera_hfov:=', LaunchConfiguration('camera_hfov'), ' ',
        'camera_rate:=', LaunchConfiguration('camera_rate'), ' ',
        'image_noise_stddev:=', LaunchConfiguration('image_noise_stddev'), ' ',
        'enable_depth:=', LaunchConfiguration('enable_depth'), ' ',
        'tool_length:=', LaunchConfiguration('tool_length'),
    ])

    # gz_ros2_control does not read the spawned SDF - it blocks until the
    # controller_manager's ResourceManager initialises from this node's
    # transient-local /robot_description topic. So the nominal model is also
    # what supplies ros2_control's interfaces and joint limits, while the
    # perturbed SDF supplies only the physical geometry. Since only <origin> is
    # rewritten and never <limit>, the two agree on every limit.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        # ParameterValue(..., value_type=str) is required: launch otherwise
        # tries to parse the URDF as YAML and fails.
        parameters=[{'robot_description': ParameterValue(
            robot_description_content, value_type=str)}],
    )

    # Server-only (-s) plus offscreen rendering. The GUI is a second full
    # rendering client competing for the same GPU, which on a modest machine
    # starves the Sensors system's render thread - the camera then publishes
    # stale frames that never show the spawned robot. Headless is also what you
    # want for long unattended collection runs.
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    gz_args = ['-r -v 4 --physics-engine gz-physics-bullet-featherstone-plugin ']
    if headless:
        gz_args.insert(0, '-s --headless-rendering ')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('ros_gz_sim'), '/launch', '/gz_sim.launch.py']),
        launch_arguments={
            'gz_args': gz_args + [LaunchConfiguration('world')],
            'on_exit_shutdown': 'True',
        }.items())

    # -file, not -topic: -topic would read robot_description, which is
    # deliberately the *nominal* model. -file over -string because the expanded
    # AR4 URDF is tens of kB and does not belong on argv.
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', ar_model, '-file', PerturbedUrdfSubstitution(
            robot_description_content,
            LaunchConfiguration('perturbation_file'),
            LaunchConfiguration('perturbed_urdf_out'),
            tf_prefix)],
        output='screen',
    )

    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        f'{GZ_CAMERA_NS}/color/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
        f'{GZ_CAMERA_NS}/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    ]
    remappings = [
        (f'{GZ_CAMERA_NS}/color/image_raw', ROS_COLOR_IMAGE),
        (f'{GZ_CAMERA_NS}/color/camera_info', ROS_COLOR_INFO),
    ]
    if enable_depth:
        # gz publishes 32FC1 in metres; MoveIt's DepthImageOctomapUpdater
        # branches on 16UC1 and otherwise takes the float path, so this is
        # accepted natively. Note the real RealSense publishes 16UC1 in
        # *millimetres* on this topic - a 1000x trap for anything downstream
        # that assumes the hardware encoding.
        bridge_args.append(
            f'{GZ_CAMERA_NS}/depth/image_raw@sensor_msgs/msg/Image[gz.msgs.Image')
        remappings.append((f'{GZ_CAMERA_NS}/depth/image_raw', ROS_DEPTH_IMAGE))

    # One parameter_bridge for everything: ros_gz_image's image_bridge would be
    # a second process that still cannot carry CameraInfo. Its ROS publishers
    # are RELIABLE/KEEP_LAST(10), which matches target_detector's plain
    # create_subscription default exactly - no QoS tuning needed.
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        arguments=bridge_args,
        remappings=remappings,
        output='screen',
    )

    # --switch-timeout matters here and is easy to miss: the default is 5 s,
    # and the *first* controller activation lands while gz is still bringing up
    # its rendering thread, so the switch service does not answer in time. The
    # symptom is joint_state_broadcaster sitting "inactive" forever - hence no
    # /joint_states, no TF, and every calibration node waiting on nothing.
    # --controller-manager-timeout does not cover this; it only waits for the
    # manager to appear.
    spawners = [
        Node(package='controller_manager', executable='spawner',
             arguments=[name, '-c', '/controller_manager',
                        '--controller-manager-timeout', '60',
                        '--switch-timeout', '60'])
        for name in ('joint_state_broadcaster', 'joint_trajectory_controller',
                     'gripper_controller')
    ]

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('annin_ar4_moveit_config'), 'launch', 'moveit.launch.py'])]),
        condition=IfCondition(LaunchConfiguration('launch_moveit')),
        launch_arguments={
            'use_sim_time': 'True',
            'ar_model': ar_model,
            'tf_prefix': tf_prefix,
            'moveit_servo': LaunchConfiguration('moveit_servo'),
            'octomap': LaunchConfiguration('enable_depth'),
        }.items())

    ground_truth = Node(
        package='annin_ar4_calibration',
        executable='ground_truth',
        name='ground_truth',
        output='screen',
        parameters=[{
            'calibration_type': calibration_type,
            'ar_model': ar_model,
            'perturbation_file': LaunchConfiguration('perturbation_file'),
            'tool_frame': [tf_prefix, 'ee_link'],
            'base_frame': [tf_prefix, 'base_link'],
        }],
    )

    return [robot_state_publisher, gazebo, spawn_robot, gz_bridge,
            *spawners, moveit, ground_truth]


def generate_launch_description():
    perturbations = PathJoinSubstitution(
        [FindPackageShare('annin_ar4_gazebo'), 'config', 'perturbations'])

    declared_arguments = [
        DeclareLaunchArgument(
            'calibration_type', default_value='eye_to_hand',
            choices=['eye_in_hand', 'eye_to_hand'],
            description='Camera/board mounting. eye_to_hand (camera fixed, board on '
                        'the wrist) is far better conditioned for WP4; eye_in_hand '
                        'matches hand_eye_params.yaml\'s default.'),
        DeclareLaunchArgument('ar_model', default_value='mk5',
                              choices=['mk1', 'mk2', 'mk3', 'mk4', 'mk5'],
                              description='Model of AR4'),
        DeclareLaunchArgument('tf_prefix', default_value='',
                              description='Prefix for AR4 tf_tree'),
        DeclareLaunchArgument(
            'perturbation_file',
            default_value=PathJoinSubstitution([perturbations, 'zero.yaml']),
            description='Known kinematic error baked into the SPAWNED robot only. '
                        'The default zero.yaml is an unperturbed baseline run; use '
                        'example_1mm.yaml (or generate your own with '
                        'scripts/generate_perturbation.py) for a WP4 experiment.'),
        DeclareLaunchArgument(
            'perturbed_urdf_out', default_value='',
            description='Where to write the URDF actually spawned. Empty means '
                        '$ROS_HOME/annin_ar4_calibration/perturbed_robot.urdf. Archive '
                        'it alongside the results - it is the record of what the '
                        '"real" robot was for that run.'),
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution([
                FindPackageShare('annin_ar4_gazebo'), 'worlds', 'calibration.world']),
            description='Must declare the gz Sensors system, or the camera '
                        'produces no images at all.'),
        DeclareLaunchArgument(
            'image_noise_stddev', default_value='0.0',
            description='Gaussian colour-image noise. 0.0 is a noiseless baseline; '
                        '~0.005 gives a plausible measurement floor.'),
        DeclareLaunchArgument('camera_width', default_value='1280'),
        DeclareLaunchArgument('camera_height', default_value='720'),
        DeclareLaunchArgument('camera_hfov', default_value='1.2',
                              description='Horizontal field of view, radians'),
        DeclareLaunchArgument('camera_rate', default_value='15.0'),
        DeclareLaunchArgument('tool_length', default_value='0.15',
                              description='WP2 tool length along ee_link +Z. This is '
                                          'the TCP ground truth.'),
        DeclareLaunchArgument(
            'enable_depth', default_value='False', choices=['True', 'False'],
            description='Add a depth camera and feed MoveIt\'s octomap. Not needed '
                        'by any of the three calibration work packages.'),
        DeclareLaunchArgument(
            'headless', default_value='False', choices=['True', 'False'],
            description='Run gz server-only with offscreen rendering. Strongly '
                        'recommended for data collection: the GUI competes with the '
                        'camera sensor for the GPU.'),
        DeclareLaunchArgument('launch_moveit', default_value='True',
                              choices=['True', 'False']),
        DeclareLaunchArgument('moveit_servo', default_value='False',
                              choices=['True', 'False']),
    ]

    return LaunchDescription([
        # Before gz starts: sdformat rewrites package:// mesh URIs to model://,
        # which gz resolves only through this variable. Without it the arm and
        # the ChArUco board load with no geometry at all.
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path_value()),
        # Applies to every node in this description and everything it includes,
        # so nothing can accidentally run on wall-clock time.
        SetParameter(name='use_sim_time', value=True),
        *declared_arguments,
        OpaqueFunction(function=launch_setup),
    ])
