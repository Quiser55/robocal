import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, os.path.dirname(__file__))
from sim_launch_common import ControllerConfigSubstitution, gz_resource_path_value  # noqa: E402

# This launch file only ever runs against Gazebo, so sim time is not optional.
# Without it robot_state_publisher stamps TF with the wall clock while
# gz_ros2_control publishes /joint_states at sim time, and every TF lookup
# either fails or extrapolates wildly.
USE_SIM_TIME = {'use_sim_time': True}


def generate_launch_description():
    ar_model_arg = DeclareLaunchArgument("ar_model",
                                         default_value="mk5",
                                         choices=["mk1", "mk2", "mk3", "mk4", "mk5"],
                                         description="Model of AR4")
    ar_model_config = LaunchConfiguration("ar_model")
    tf_prefix_arg = DeclareLaunchArgument("tf_prefix",
                                          default_value="",
                                          description="Prefix for AR4 tf_tree")
    tf_prefix = LaunchConfiguration("tf_prefix")

    initial_joint_controllers = ControllerConfigSubstitution(
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_driver"), "config", "controllers.yaml"
        ]),
        tf_prefix=tf_prefix)

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_description"),
            "urdf",
            "ar_gazebo.urdf.xacro",
        ]),
        " ",
        "ar_model:=",
        ar_model_config,
        " ",
        "tf_prefix:=",
        tf_prefix,
        " ",
        "simulation_controllers:=",
        initial_joint_controllers,
    ])
    robot_description = {"robot_description": robot_description_content}

    # gz_ros2_control does not read the spawned SDF: it blocks until the
    # controller_manager's ResourceManager initializes from the transient-local
    # /robot_description topic published here. So this node is a prerequisite
    # for the controllers coming up, not just a TF publisher.
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, USE_SIM_TIME],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", "-c", "/controller_manager",
            # --switch-timeout (default 5 s) is separate from the manager
            # timeout: the first activation races gz's rendering startup,
            # and without it joint_state_broadcaster silently stays inactive.
            "--controller-manager-timeout", "60", "--switch-timeout", "60"
        ],
        parameters=[USE_SIM_TIME],
    )

    # There may be other controllers of the joints, but this is the initially-started one
    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller", "-c", "/controller_manager",
            # --switch-timeout (default 5 s) is separate from the manager
            # timeout: the first activation races gz's rendering startup,
            # and without it joint_state_broadcaster silently stays inactive.
            "--controller-manager-timeout", "60", "--switch-timeout", "60"
        ],
        parameters=[USE_SIM_TIME],
    )

    gripper_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller", "-c", "/controller_manager",
            # --switch-timeout (default 5 s) is separate from the manager
            # timeout: the first activation races gz's rendering startup,
            # and without it joint_state_broadcaster silently stays inactive.
            "--controller-manager-timeout", "60", "--switch-timeout", "60"
        ],
        parameters=[USE_SIM_TIME],
    )

    # Gazebo nodes
    world = os.path.join(get_package_share_directory('annin_ar4_gazebo'),
                         'worlds', 'empty.world')

    # Bridge
    gazebo_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        parameters=[USE_SIM_TIME],
        output='screen')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch", "/gz_sim.launch.py"]),
        launch_arguments={
            'gz_args':
            f'-r -v 4 --physics-engine gz-physics-bullet-featherstone-plugin {world}',
            'on_exit_shutdown': 'True'
        }.items())

    # Spawn robot
    gazebo_spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-name", ar_model_config, "-topic", "robot_description"],
        parameters=[USE_SIM_TIME],
        output="screen",
    )

    return LaunchDescription([
        # Must precede gz starting, or package:// mesh URIs resolve to nothing.
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path_value()),
        ar_model_arg,
        tf_prefix_arg,
        gazebo_bridge,
        gazebo,
        gazebo_spawn_robot,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        initial_joint_controller_spawner_started,
        gripper_joint_controller_spawner_started,
    ])
