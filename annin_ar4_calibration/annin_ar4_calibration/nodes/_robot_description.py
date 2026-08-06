"""Shared helper for nodes that need the live, already-xacro-expanded URDF.

robot_state_publisher exposes ``robot_description`` only as a parameter on
itself (not a topic, unlike ROS1) - this fetches it once via a
GetParameters service call, the standard ROS2 idiom.
"""
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node


def fetch_robot_description(node: Node, timeout_sec: float = 10.0) -> str:
    client = node.create_client(GetParameters, '/robot_state_publisher/get_parameters')
    if not client.wait_for_service(timeout_sec=timeout_sec):
        raise RuntimeError(
            '/robot_state_publisher/get_parameters not available - is robot_state_publisher '
            'running?')
    request = GetParameters.Request()
    request.names = ['robot_description']
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if future.result() is None or not future.result().values:
        raise RuntimeError('Failed to fetch robot_description from robot_state_publisher')
    return future.result().values[0].string_value
