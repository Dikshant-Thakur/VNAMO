#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time

# --- FIX: Point aur Vector3 add kiya gaya hai ---
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, TwistStamped, Point, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from control_msgs.action import FollowJointTrajectory
from nav2_msgs.action import NavigateToPose

import tf2_ros
import tf2_geometry_msgs

class SimpleRodGrab(Node):
    def __init__(self):
        super().__init__("simple_rod_grab")

        self.planning_frame = "ur_base_link"
        self.ee_link = "ur_tool0"
        
        # Target
        self.rod_x = 1.475
        self.rod_y = 0.021
        self.rod_z = 0.5
        
        # Gripper
        self.gripper_length = 0.15
        self.gripper_open_val = 0.25
        self.gripper_close_val = -0.25
        self.gripper_joint_names = ["gripper_soft_robotics_gripper_left_finger_joint1", "gripper_soft_robotics_gripper_right_finger_joint1"]

        # Publishers
        self.collision_pub = self.create_publisher(CollisionObject, "collision_object", 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        # Clients
        self.nav2_ac = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.gripper_ac = ActionClient(self, FollowJointTrajectory, "gripper_controller/follow_joint_trajectory")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("[INIT] PID Servo Grasper Ready.")

    def _execute_pid_approach(self, target_pose_planning: PoseStamped) -> bool:
        self.get_logger().info("[PID] Starting Velocity Control Approach...")
        
        tx = target_pose_planning.pose.position.x
        ty = target_pose_planning.pose.position.y
        tz = target_pose_planning.pose.position.z
        
        Kp = 4.0
        max_speed = 0.20
        stop_dist = 0.015
        timeout = 25.0
        
        start_time = time.time()
        
        while rclpy.ok():
            if (time.time() - start_time) > timeout:
                self.get_logger().error("[PID] Timeout!")
                self._stop_robot()
                return False

            try:
                if not self.tf_buffer.can_transform(self.planning_frame, self.ee_link, Time()):
                    rclpy.spin_once(self, timeout_sec=0.01); continue
                
                tf = self.tf_buffer.lookup_transform(self.planning_frame, self.ee_link, Time())
                cx, cy, cz = tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z

                ex, ey, ez = tx - cx, ty - cy, tz - cz
                dist = math.sqrt(ex**2 + ey**2 + ez**2)

                if dist < stop_dist:
                    self.get_logger().info(f"[PID] Success! Gap: {dist*100:.2f} cm")
                    self._stop_robot()
                    return True

                vx, vy, vz = Kp * ex, Kp * ey, Kp * ez

                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                if speed > max_speed:
                    scale = max_speed / speed
                    vx, vy, vz = vx*scale, vy*scale, vz*scale

                msg = TwistStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.planning_frame
                msg.twist.linear.x = vx; msg.twist.linear.y = vy; msg.twist.linear.z = vz
                self.vel_pub.publish(msg)
                
                rclpy.spin_once(self, timeout_sec=0.02)

            except Exception as e:
                self.get_logger().warn(f"[PID] Error: {e}")
                self._stop_robot(); return False
        return False

    def _stop_robot(self):
        msg = TwistStamped(); msg.header.frame_id = self.planning_frame
        for _ in range(5): self.vel_pub.publish(msg); time.sleep(0.02)

    def execute_sequence(self) -> bool:
        if not self.wait_for_tf("map", self.planning_frame): return False
        self.get_logger().info("[SEQ] Started")

        target_wrist_x = self.rod_x - self.gripper_length
        target_wrist_y = self.rod_y
        
        self.update_rod_obstacle("ADD")

        self.get_logger().info("[SEQ] Navigating...")
        if not self._navigate_closer(target_wrist_x, target_wrist_y): return False

        self._execute_gripper(self.gripper_open_val)
        
        self.update_rod_obstacle("REMOVE"); time.sleep(1.0)
        
        t_pose = PoseStamped()
        t_pose.header.frame_id = "map"; t_pose.header.stamp = Time().to_msg()
        t_pose.pose.position.x = target_wrist_x
        t_pose.pose.position.y = target_wrist_y
        t_pose.pose.position.z = 0.5
        t_pose.pose.orientation = Quaternion(x=0.0, y=-0.707, z=0.0, w=0.707)

        if not self.wait_for_tf(self.planning_frame, "map"): return False
        try:
            target_planning = self.tf_buffer.transform(t_pose, self.planning_frame, timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f"[SEQ] TF Error: {e}")
            return False
        
        if not self._execute_pid_approach(target_planning): return False

        self.get_logger().info("[SEQ] GRIPPING...")
        self._execute_gripper(self.gripper_close_val)
        return True

    def wait_for_tf(self, t, s, timeout=5.0):
        start = time.time()
        while (time.time() - start) < timeout:
            if self.tf_buffer.can_transform(t, s, Time()): return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def update_rod_obstacle(self, op="ADD"):
        co = CollisionObject(); co.header.frame_id = "map"; co.id = "target_rod"
        if op == "ADD":
            co.operation = CollisionObject.ADD
            cyl = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[1.0, 0.03])
            # FIX: Use Point instead of Vector3
            pose = Pose(position=Point(x=self.rod_x, y=self.rod_y, z=self.rod_z), orientation=Quaternion(w=1.0))
            co.primitives = [cyl]; co.primitive_poses = [pose]
        else: co.operation = CollisionObject.REMOVE
        for _ in range(3): self.collision_pub.publish(co); rclpy.spin_once(self, timeout_sec=0.05)

    def _navigate_closer(self, tx, ty):
        if not self.nav2_ac.wait_for_server(5.0): return False
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"; goal.pose.header.stamp = self.get_clock().now().to_msg()
        try:
            rclpy.spin_once(self, timeout_sec=0.1)
            tf = self.tf_buffer.lookup_transform("map", "base_link", Time())
            bx, by = tf.transform.translation.x, tf.transform.translation.y
            dx, dy = bx-tx, by-ty
            dist = math.hypot(dx, dy); ux, uy = dx/dist, dy/dist
            goal.pose.pose.position.x = tx + ux * 0.75; goal.pose.pose.position.y = ty + uy * 0.75
            yaw = math.atan2(-dy, -dx)
            goal.pose.pose.orientation = Quaternion(z=math.sin(yaw/2), w=math.cos(yaw/2))
        except: return False
        self.nav2_ac.send_goal_async(goal); time.sleep(8.0)
        return True

    def _execute_gripper(self, pos):
        traj = JointTrajectory(); traj.joint_names = self.gripper_joint_names
        pt = JointTrajectoryPoint(); pt.positions = [float(pos)]*2; pt.time_from_start.sec = 1
        traj.points.append(pt)
        self.gripper_ac.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)); time.sleep(1.0)

def main(args=None):
    rclpy.init(args=args); node = SimpleRodGrab()
    if node.execute_sequence(): node.get_logger().info("[MAIN] SUCCESS")
    else: node.get_logger().error("[MAIN] FAILED")
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()