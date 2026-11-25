import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()

    global_parameters = os.path.join(
        get_package_share_directory("igris_c_controller"), "config", "params.yaml"
    )

    global_param_node = Node(
        package="igris_c_controller",
        executable="igris_c_param_server",
        name="igris_c_param_server",
        parameters=[global_parameters],
    )

    ld.add_action(global_param_node)

    return ld
