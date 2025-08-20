"""
MiR Robot Gazebo Launch File


Author: Dikshant Thakur
Date: 05/Aug/2025
"""


import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
    SetLaunchConfiguration, ExecuteProcess, RegisterEventHandler
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder



def generate_launch_description():
    """
    Generate the launch description for MiR robot simulation.
    
    Returns:
        LaunchDescription: Complete launch configuration
    """
    
    # ============================================================================
    # PACKAGE DIRECTORIES
    # ============================================================================
    mir_description_dir = get_package_share_directory('mir_description')
    mir_gazebo_dir = get_package_share_directory('mir_gazebo')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    
    # ============================================================================
    # LAUNCH ARGUMENTS DECLARATION
    # ============================================================================
    
    # Robot namespace and positioning
    declare_namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace to push all topics into.'
    )
    
    declare_robot_x_arg = DeclareLaunchArgument(
        'robot_x',
        default_value='0.0',
        description='Spawning position of robot (x coordinate)'
    )
    
    declare_robot_y_arg = DeclareLaunchArgument(
        'robot_y',
        default_value='0.0',
        description='Spawning position of robot (y coordinate)'
    )
    
    declare_robot_yaw_arg = DeclareLaunchArgument(
        'robot_yaw',
        default_value='0.0',
        description='Spawning orientation of robot (yaw angle in radians)'
    )
    
    # Simulation settings
    declare_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty',
        description='Choose simulation world. Available worlds: empty, maze'
    )
    
    declare_verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Set to true to enable verbose mode for Gazebo.'
    )
    
    declare_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "false" to run headless Gazebo simulation.'
    )
    
    # Interface options
    declare_teleop_arg = DeclareLaunchArgument(
        'teleop_enabled',
        default_value='true',
        description='Set to true to enable teleop for manual robot control.'
    )
    
    # declare_rviz_arg = DeclareLaunchArgument(
    #     'rviz_enabled',
    #     default_value='true',
    #     description='Set to true to launch RViz visualization.'
    # )
    
    # declare_rviz_config_arg = DeclareLaunchArgument(
    #     'rviz_config_file',
    #     default_value=os.path.join(mir_description_dir, 'rviz', 'mir_visu_full.rviz'),
    #     description='Define RViz config file to be used.'
    # )   
    # ============================================================================
    # DUAL RVIZ CONFIGURATION
    # ============================================================================
    declare_nav_rviz_arg = DeclareLaunchArgument(
        'nav_rviz_enabled',
        default_value='true',
        description='Enable navigation RViz window'
    )

    declare_moveit_rviz_arg = DeclareLaunchArgument(
        'moveit_rviz_enabled', 
        default_value='true',
        description='Enable MoveIt RViz window'
    )

    declare_nav_rviz_config_arg = DeclareLaunchArgument(
        'nav_rviz_config',
        default_value=os.path.join(
            get_package_share_directory('mir_navigation'),  #
            'rviz',
            'mir_nav.rviz'
        ),
        description='Navigation RViz config file'
    )

    declare_moveit_rviz_config_arg = DeclareLaunchArgument(
        'moveit_rviz_config',
        default_value=os.path.join(
            get_package_share_directory("moveit_config"), "config", "moveit.rviz"
        ),
        description='MoveIt RViz config file'
    )
    
    # ============================================================================
    # MOVEIT CONFIGURATION
    # ============================================================================
    moveit_config = (
        MoveItConfigsBuilder("custom_robot", package_name="moveit_config")
        .robot_description(file_path="config/mir_250.urdf.xacro")
        .robot_description_semantic(file_path="config/mir_250.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, 
            publish_robot_description_semantic=True, 
            publish_planning_scene=True
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )
    
    # ============================================================================
    # ROBOT DESCRIPTION
    # ============================================================================
    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("mir_description"), 
            "urdf", 
            "mir.urdf.xacro"
        ]),
    ])
    
    # ============================================================================
    # HELPER FUNCTIONS - FOR MULTIPLE ROBOTS 
    # ============================================================================
    def process_namespace(context):
        """
        Process namespace to create proper robot name for spawning.
        
        Args:
            context: Launch context containing configurations
            
        Returns:
            list: SetLaunchConfiguration action for robot_name
        """
        robot_name = "mir_robot"
        try:
            namespace = context.launch_configurations['namespace']
            if namespace:
                robot_name = namespace + '/' + robot_name
        except KeyError:
            pass
        return [SetLaunchConfiguration('robot_name', robot_name)]
    
    # ============================================================================
    # GAZEBO WORLD LAUNCH
    # ============================================================================
    launch_gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': LaunchConfiguration('verbose'),
            'gui': LaunchConfiguration('gui'),
            'world': [mir_gazebo_dir, '/worlds/', LaunchConfiguration('world'), '.world']
        }.items()
    )
    
    # ============================================================================
    # ROBOT STATE PUBLISHER - Ye node robot k structure aur joints mtlb unki positions
    #  ki real-time information publish karta hai.
    # ============================================================================
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[moveit_config.robot_description],
        namespace=LaunchConfiguration('namespace'),
    )
    
    # ============================================================================
    # GAZEBO ROBOT SPAWNING
    # ============================================================================
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', LaunchConfiguration('robot_name'),
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('robot_x'),
            '-y', LaunchConfiguration('robot_y'),
            '-z', '0.0',
            '-Y', LaunchConfiguration('robot_yaw'),
            '-b'  # Bond node to gazebo model
        ],
        namespace=LaunchConfiguration('namespace'),
        output='screen'
    )
    
    # ============================================================================
    # CONTROLLER SPAWNERS
    # ============================================================================
    
    # Joint State Broadcaster - controller ko spawn karta hai, real joint data collect karta hai,
    # and publishes joint states

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", 
            "--controller-manager", 
            "/controller_manager"
        ],
        output="screen",
    )
    
    # Differential Drive Controller - handles mobile base movement
    diff_drive_spawner = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', 
            '--set-state', 'active', 'diff_cont'
        ],
        output='screen'
    )
    
    # Arm Trajectory Controller - handles manipulator movement
    arm_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller", 
            "--controller-manager", 
            "/controller_manager"
        ],
        output="screen",
    )


    # ========== GRIPPER CONTROLLER SPAWNER (NEW) ==========
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller", 
            "--controller-manager", 
            "/controller_manager"
        ],
        output="screen",
    )
    
    # ============================================================================
    # MOVEIT MOVE GROUP NODE
    # ============================================================================
    use_sim_time = {"use_sim_time": True}
    config_dict = moveit_config.to_dict()
    config_dict.update(use_sim_time)
    
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[config_dict],
        arguments=["--ros-args", "--log-level", "info"],
    )
    # ============================================================================
    # NAV2 VISUALIZATION - MOVEIT
    # ============================================================================
    launch_nav_rviz = Node(
        condition=IfCondition(LaunchConfiguration('nav_rviz_enabled')),
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav',
        output='screen',
        arguments=['-d', LaunchConfiguration('nav_rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[('/initialpose', '/initialpose'), ('/goal_pose', '/goal_pose')]
    )












    # ============================================================================
    # RVIZ VISUALIZATION - MOVEIT
    # ============================================================================
    # moveit_rviz_config_path = os.path.join(
    #     get_package_share_directory("moveit_config"),
    #     "config",
    #     "moveit.rviz"
    # )
    
    launch_moveit_rviz = Node(
        condition=IfCondition(LaunchConfiguration('moveit_rviz_enabled')),
        package='rviz2',
        executable='rviz2',
        name='rviz2_moveit',
        output='screen',
        arguments=['-d', LaunchConfiguration('moveit_rviz_config')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
    ]
)
    
    # ============================================================================
    # TELEOP CONTROL
    # ============================================================================
    launch_teleop = Node(
        condition=IfCondition(LaunchConfiguration("teleop_enabled")),
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        arguments=['-r', '/cmd_vel:=/diff_cont/cmd_vel_unstamped'],
        prefix='xterm -e'
    )
    
    # ============================================================================
    # ADDITIONAL LAUNCH INCLUDES
    # ============================================================================
    launch_mir_gazebo_common = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mir_gazebo_dir, 'launch', 'include', 'mir_gazebo_common.py')
        )
    )
    
    # ============================================================================
    # EVENT HANDLERS FOR SEQUENTIAL CONTROLLER STARTUP
    # ============================================================================
    
    # Start joint state broadcaster after robot is spawned
    delayed_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=spawn_robot,  # Robot spawn(node) hone ka wait karo,
            on_start=[joint_state_broadcaster_spawner], #spawn_robot command executed → IMMEDIATELY → broadcaster starts
        )
    )
    
    # Start differential drive controller after joint state broadcaster
    delayed_diff_drive_spawner = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,  # Joint state broadcaster start hone ka wait karo
            on_exit=[diff_drive_spawner], # broadcaster starting... → broadcaster READY/FINISHED → diff_drive starts
        )
    )
    
    # Start arm controller after differential drive controller
    delayed_arm_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=diff_drive_spawner,  # ExecuteProcess
            on_exit=[arm_trajectory_controller_spawner],
        )
    )
    # ========== DELAYED GRIPPER CONTROLLER (NEW) ==========

    delayed_gripper_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_trajectory_controller_spawner,  # Arm controller ke baad
            on_exit=[gripper_controller_spawner],  # Gripper controller start karo
        )
    )
    
    # ============================================================================
    # LAUNCH DESCRIPTION ASSEMBLY
    # ============================================================================
    ld = LaunchDescription()
    
    # Add namespace processing function
    ld.add_action(OpaqueFunction(function=process_namespace))
    
    # Add all launch arguments
    ld.add_action(declare_namespace_arg)
    ld.add_action(declare_robot_x_arg)
    ld.add_action(declare_robot_y_arg)
    ld.add_action(declare_robot_yaw_arg) 
    ld.add_action(declare_sim_time_arg)
    ld.add_action(declare_world_arg)
    ld.add_action(declare_verbose_arg)
    ld.add_action(declare_teleop_arg)
    # ld.add_action(declare_rviz_arg)
    # ld.add_action(declare_rviz_config_arg)
    ld.add_action(declare_nav_rviz_arg)
    ld.add_action(declare_moveit_rviz_arg) 
    ld.add_action(declare_nav_rviz_config_arg)
    ld.add_action(declare_moveit_rviz_config_arg)
    ld.add_action(launch_nav_rviz)
    ld.add_action(launch_moveit_rviz)
    ld.add_action(declare_gui_arg)
    
    # Add core launch components
    ld.add_action(launch_gazebo_world)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_robot)
    ld.add_action(launch_mir_gazebo_common)
    
    # Add controllers with proper sequencing
    ld.add_action(delayed_joint_state_broadcaster_spawner)
    ld.add_action(delayed_diff_drive_spawner)
    ld.add_action(delayed_arm_controller)
    ld.add_action(delayed_gripper_controller)
    
    # Add MoveIt and visualization
    ld.add_action(move_group_node)

    # ld.add_action(launch_rviz)
    ld.add_action(launch_teleop)
    
    return ld