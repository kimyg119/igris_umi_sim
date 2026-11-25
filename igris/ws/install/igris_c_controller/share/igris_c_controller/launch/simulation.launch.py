import os
import sys

from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription

sys.path.append(os.path.dirname(__file__))
from launch_common import (
    cyclonedds_env_actions,
    load_robot_params_as_ros_dict,
    prepare_model_params,
    resolve_robot_config_path,
)


def _create_actions(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    global_parameters = resolve_robot_config_path(config_file, os.path.basename(__file__))

    ros_params = prepare_model_params(load_robot_params_as_ros_dict(global_parameters))

    joystick_enabled = bool(ros_params.get("igris_c.joystick.enabled", False))

    actions = cyclonedds_env_actions(ros_params)

    # Zero all shared-memory segments once before any module starts.
    shm_zero = Node(
        package="igris_c_shm",
        executable="shm_zero_node",
        name="shm_zero_node",
        output="screen",
        output_format=" shm_zero: {line}",
    )

    module_nodes = [
        Node(
            package="igris_c_controller",
            executable="igris_c_param_server",
            name="igris_c_param_server",
            parameters=[ros_params],
            output="screen",
            output_format="    param: {line}",
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="igris_c_mujoco",
            executable="igris_c_mujoco",
            name="igris_c_mujoco",
            output="screen",
            output_format="   mujoco: {line}",
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="igris_c_can",
            executable="igris_c_can_dummy_node",
            name="igris_c_can_dummy_node",
            output="screen",
            output_format="      can: {line}",
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="igris_c_hand",
            executable="igris_c_hand_dummy_node",
            name="igris_c_hand_dummy_node",
            output="screen",
            output_format="     hand: {line}",
            respawn=True,
            respawn_delay=2.0,
        ),
        Node(
            package="igris_c_bridge",
            executable="bridge_sim",
            name="igris_c_bridge",
            output="screen",
            output_format="   bridge: {line}",
            parameters=[{"simulation": True}],
            respawn=True,
            respawn_delay=2.0,
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="igris_c_controller",
                    executable="sim",
                    name="sim",
                    output="screen",
                    output_format="      sim: {line}",
                    respawn=True,
                    respawn_delay=2.0,
                ),
            ],
        ),
    ]

    # Optional joystick teleop receiver. Like the other nodes it reads its config
    # (transport / device_path / network domain+namespace) from the param server
    # (robot.yaml), so no per-node launch arguments are passed.
    if joystick_enabled:
        module_nodes.append(
            Node(
                package="igris_c_joystick",
                executable="joystick_publisher",
                name="igris_c_joystick",
                output="screen",
                output_format=" joystick: {line}",
                respawn=True,
                respawn_delay=2.0,
            )
        )

    # Start modules only after the zeroer exits successfully; abort the launch otherwise.
    def _start_modules_after_zero(event, context):
        if event.returncode != 0:
            return [
                LogInfo(msg=f"shm_zero_node failed (rc={event.returncode}); aborting launch."),
                Shutdown(reason="shm zero-init failed"),
            ]
        return module_nodes

    actions.append(shm_zero)
    actions.append(
        RegisterEventHandler(
            OnProcessExit(target_action=shm_zero, on_exit=_start_modules_after_zero)
        )
    )

    return actions


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
