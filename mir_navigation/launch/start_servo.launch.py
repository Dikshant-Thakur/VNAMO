import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. MoveIt Config Load
    moveit_config = MoveItConfigsBuilder("mir_250", package_name="moveit_config").to_moveit_configs()

    # 2. Parameters ko Hardcode Karein (No YAML File Needed)
    # Isse "panda_arm" wala error hamesha ke liye khatam ho jayega
    servo_params = {
        "moveit_servo": {
            "use_gazebo": True,
            "status_topic": "servo_node/status",
            
            # CRITICAL: Group Name
            "move_group_name": "ur5_manip",
            
            # Frames
            "planning_frame": "ur_base_link",
            "ee_frame": "ur_tool0",
            "robot_link_command_frame": "ur_base_link",
            
            # Output Topic
            "command_out_topic": "/joint_group_velocity_controller/commands",
            "command_out_type": "std_msgs/Float64MultiArray",
            
            # Publish Settings
            "publish_joint_positions": False,
            "publish_joint_velocities": True,
            "publish_joint_accelerations": False,
            "publish_period": 0.033,
            
            # Incoming Commands
            "incoming_command_timeout": 0.1,
            
            # Safety & Collision (Loose settings for PID)
            "check_collisions": False,        # PID ke liye band rakhein
            "lower_singularity_threshold": 5.0,
            "hard_stop_singularity_threshold": 10.0,
            
            # Scales
            "linear_scale": 1.0,
            "rotational_scale": 1.0,
            "joint_scale": 1.0,
        }
    }

    return LaunchDescription([
        Node(
            package="moveit_servo",
            executable="servo_node_main",
            name="moveit_servo", # Node name must match param key
            parameters=[
                moveit_config.to_dict(),
                servo_params, 
                {"moveit_manage_controllers": False},
            ],
            output="screen",
        )
    ])