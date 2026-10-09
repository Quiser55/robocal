"""Shared helper for waiting on a MoveIt2 motion from inside a spun node"""
import time

from pymoveit2 import MoveIt2State

DEFAULT_MOTION_TIMEOUT_SEC = 120.0

_POLL_SEC = 0.02


def wait_for_motion(moveit2, timeout_sec: float = DEFAULT_MOTION_TIMEOUT_SEC,
                    ) -> tuple[bool, str]:
    """Wait for the motion just requested. Returns ``(succeeded, reason)``"""
    if moveit2.query_state() == MoveIt2State.IDLE:
        return False, 'motion never started (move action server unavailable)'

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if moveit2.query_state() == MoveIt2State.IDLE:
            if moveit2.motion_suceeded:     # pymoveit2 spells it this way
                return True, ''
            return False, 'motion failed'
        time.sleep(_POLL_SEC)

    return False, f'motion did not finish within {timeout_sec:.0f}s'


def _await_future(future, deadline: float):
    """Poll a future to completion without spinning the node"""
    while not future.done():
        if time.time() > deadline:
            return None
        time.sleep(_POLL_SEC)
    return future.result()


def send_direct_trajectory(action_client, joint_names: list, positions,
                           duration_sec: float = 4.0,
                           timeout_sec: float = 30.0) -> tuple[bool, str]:
    """Command one joint configuration straight to the trajectory controller,
    bypassing MoveIt entirely. Returns ``(accepted, reason)``"""
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    if not action_client.wait_for_server(timeout_sec=min(timeout_sec, 10.0)):
        name = getattr(action_client, '_action_name',
                       'the trajectory controller')
        return False, f'trajectory action server {name} unavailable'

    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [0.0] * len(joint_names)
    point.time_from_start.sec = int(duration_sec)
    point.time_from_start.nanosec = int((duration_sec % 1.0) * 1e9)

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory()
    goal.trajectory.joint_names = list(joint_names)
    goal.trajectory.points = [point]

    goal_handle = _await_future(
        action_client.send_goal_async(goal), time.time() + timeout_sec)
    if goal_handle is None:
        return False, 'recovery goal was not acknowledged before the timeout'
    if not goal_handle.accepted:
        return False, 'recovery goal rejected by the trajectory controller'
    return True, ''
