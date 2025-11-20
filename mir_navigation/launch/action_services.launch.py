from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        # 1) observe_obstacle_server
        Node(
            package='mir_navigation',
            executable='observe_obstacle_server',
            name='observe_obstacle_server',
            output='screen',
            emulate_tty=True,
            prefix='xterm -e',
        ),

        # 2) push_geometry_server
        Node(
            package='mir_navigation',
            executable='push_geometry_server',
            name='push_geometry_server',
            output='screen',
            emulate_tty=True,
            prefix='xterm -e',
        ),

        # 3) push_manipulation_action_server
        Node(
            package='mir_navigation',
            executable='push_manipulation_action_server',
            name='push_manipulation_action_server',
            output='screen',
            emulate_tty=True,
            prefix='xterm -e',
        ),

        # 4) visibility_action
        Node(
            package='mir_navigation',
            executable='visibility_action',
            name='visibility_action',
            output='screen',
            emulate_tty=True,
            prefix='xterm -e',
        ),
    ])
