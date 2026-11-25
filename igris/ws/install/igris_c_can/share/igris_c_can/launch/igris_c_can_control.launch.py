import os
import sys

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription

sys.path.append(os.path.dirname(__file__))
from launch_common import (
    load_robot_params_as_ros_dict,
    resolve_robot_config_path,
)

# Node name of the parameter server provided by this package. The control node
# fetches its configuration from this node via its "param_server" parameter.
PARAM_SERVER_NODE = "igris_c_can_param_server"


def _create_actions(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    global_parameters = resolve_robot_config_path(config_file)
    ros_params = load_robot_params_as_ros_dict(global_parameters)

    # Parameter server. Must be available before the control node performs its
    # parameter lookup, so it is added first.
    param_server_node = Node(
        package="igris_c_can",
        executable="igris_c_can_param_server",
        name=PARAM_SERVER_NODE,
        parameters=[ros_params],
        output="screen",
        output_format="   param: {line}",
        respawn=True,
        respawn_delay=2.0,
    )

    # CAN control node. "param_server" points it at this package's parameter
    # server instead of the igris_c_controller one.
    can_control_node = Node(
        package="igris_c_can",
        executable="igris_c_can_control_node",
        name="igris_c_can_control_node",
        parameters=[{"param_server": PARAM_SERVER_NODE, "dummy": False}],
        output="screen",
        output_format="      can: {line}",
        respawn=True,
        respawn_delay=2.0,
    )

    return [param_server_node, can_control_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="",
                description="Path to robot config yaml (installed: /robros/configs/robot.yaml, dev: <repo>/configs/robot/dev_robot.yaml)",
            ),
            OpaqueFunction(function=_create_actions),
        ]
    )
