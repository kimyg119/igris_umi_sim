import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            name="urdf_package",
            default_value="igris_c_can",
            description="Pacage name containing URDF",
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            name="urdf_path",
            default_value="urdf/igris_c_motors.urdf",
            description="Path to the URDF file",
        )
    )

    def launch_setup(context, *args, **kwargs):
        urdf_pkg = LaunchConfiguration("urdf_package").perform(context)
        urdf_file = LaunchConfiguration("urdf_path").perform(context)
        urdf_full_path = os.path.join(get_package_share_directory(urdf_pkg), urdf_file)
        with open(urdf_full_path, "r") as file:
            urdf_context = file.read()
            # Do something with urdf_context
            print(urdf_context)

        return [
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
                parameters=[{"use_gui": True, "rate": 100}],
                remappings=[("/joint_states", "/motor/target_joint_states")],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": urdf_context}],
                remappings=[("/joint_states", "/motor/target_joint_states")],
            ),
            Node(
                package="igris_c_can",
                executable="test_joint_node",
                name="test_joint_node",
                output="screen",
            ),
            # Node(
            #     package='rviz2',
            #     executable='rviz2',
            #     name='rviz2',
            #     output='screen',
            # )
        ]

    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld
