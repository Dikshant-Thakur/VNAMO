#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# 👉 tumhara actual action type
from mir_navigation.action import ManipulateObstacle


@dataclass
class ObstacleInfo:
    name: str


class ManipulateObstacleServer(Node):
    """
    Simple push-only manipulation server:

    - Planner already brought robot to pre-manip pose.
    - Goal: obstacle_name + push_dir (direction info only).
    - Node does:
        1) slow forward to 'contact' (time-based placeholder),
        2) push forward for fixed distance (from odom),
        3) retract back fixed distance,
    - Returns: success + message.
    """

    def __init__(self):
        super().__init__('manipulate_obstacle_server')

        # ---------- Parameters (tune as needed) ----------
        # speeds
        self.declare_parameter('contact_approach_speed', 0.05)  # m/s
        self.declare_parameter('push_speed',             0.12)  # m/s
        self.declare_parameter('retract_speed',          0.10)  # m/s

        # times / distances
        self.declare_parameter('contact_approach_time_s', 3.0)   # seconds (fake contact)
        self.declare_parameter('push_distance_m',         1.0)   # how far to push
        self.declare_parameter('retract_distance_m',      0.5)   # how far to go back

        self.declare_parameter('contact_timeout_s',       20.0)
        self.declare_parameter('push_timeout_s',          120.0)
        self.declare_parameter('retract_timeout_s',       20.0)

        # (optional) known obstacles (abhi sirf names, pose/size use nahi)
        self._obstacles: Dict[str, ObstacleInfo] = {
            "test_box": ObstacleInfo(name="test_box"),
        }

        # ---------- State ----------
        self.odom_pose: Optional[Tuple[float, float, float]] = None

        # ---------- ROS I/O ----------
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Action server
        self._action_server = ActionServer(
            self,
            ManipulateObstacle,
            'manipulate_obstacle',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info('[MANIP] ManipulateObstacleServer up (push-only, cmd_vel based).')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose
        # yaw not really needed, but store anyway
        orientation = msg.pose.pose.orientation
        yaw = self._yaw_from_quat(orientation)
        self.odom_pose = (p.position.x, p.position.y, yaw)

    def goal_callback(self, goal_request: ManipulateObstacle.Goal) -> GoalResponse:
        """Check if obstacle_name looks valid (optional) and push_dir is non-empty."""
        name = goal_request.obstacle_name
        push_dir = goal_request.push_dir

        if name and name not in self._obstacles:
            # Optional: you can also accept unknown names; then remove this check
            self.get_logger().warn(f"[MANIP] Unknown obstacle '{name}'. Rejecting goal.")
            return GoalResponse.REJECT

        if push_dir not in ["+X", "-X", "+Y", "-Y", ""]:
            self.get_logger().warn(f"[MANIP] Weird push_dir '{push_dir}'. Still accepting, but check planner.")
            # You can ACCEPT here, just warning:
            # return GoalResponse.REJECT

        self.get_logger().info(
            f"[MANIP] Accepted goal: obstacle='{name}', dir='{push_dir}'. "
            f"(Assuming robot already at pre-manip pose.)"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info('[MANIP] Cancel request received.')
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute_callback(self, goal_handle):
        """
        Simple pipeline:
          1) Wait for odom.
          2) Slow forward for contact_approach_time_s (fake contact).
          3) Push forward for push_distance_m (using odom).
          4) Retract backward for retract_distance_m.
        """
        result = ManipulateObstacle.Result()
        feedback = ManipulateObstacle.Feedback()

        name = goal_handle.request.obstacle_name
        push_dir = goal_handle.request.push_dir

        self.get_logger().info(
            f"[MANIP] Starting push for obstacle='{name}', dir='{push_dir}'."
        )

        # 1) Wait for odom
        if not await self._wait_for_odom(goal_handle):
            msg = "[MANIP] No odom data. Aborting."
            self.get_logger().warn(msg)
            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        # 2) Approach contact
        feedback.state = "approach_contact"
        goal_handle.publish_feedback(feedback)

        ok_contact = await self._approach_contact(goal_handle)
        if not ok_contact:
            msg = "[MANIP] Contact approach failed."
            self.get_logger().warn(msg)
            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            self._stop_robot()
            goal_handle.canceled()
            result.success = False
            result.message = "Cancelled after contact approach."
            return result

        # 3) Push
        feedback.state = "pushing"
        goal_handle.publish_feedback(feedback)

        ok_push = await self._do_push(goal_handle)
        if not ok_push:
            msg = "[MANIP] Push failed."
            self.get_logger().warn(msg)
            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            self._stop_robot()
            goal_handle.canceled()
            result.success = False
            result.message = "Cancelled after push."
            return result

        # 4) Retract
        feedback.state = "retract"
        goal_handle.publish_feedback(feedback)

        await self._retract(goal_handle)

        feedback.state = "done"
        goal_handle.publish_feedback(feedback)

        msg = "[MANIP] Push manipulation finished."
        self.get_logger().info(msg)
        result.success = True
        result.message = msg
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _wait_for_odom(self, goal_handle, timeout_s: float = 5.0) -> bool:
        t0 = time.monotonic()
        while rclpy.ok():
            if self.odom_pose is not None:
                return True
            if goal_handle.is_cancel_requested:
                return False
            if (time.monotonic() - t0) > timeout_s:
                return False
            await asyncio.sleep(0.1)
        return False

    async def _approach_contact(self, goal_handle) -> bool:
        """
        Fake contact approach:
          - move forward at contact_approach_speed
          - for contact_approach_time_s (or until timeout / cancel)
        """
        speed = float(self.get_parameter('contact_approach_speed').get_parameter_value().double_value)
        contact_time = float(self.get_parameter('contact_approach_time_s').get_parameter_value().double_value)
        timeout = float(self.get_parameter('contact_timeout_s').get_parameter_value().double_value)

        self.get_logger().info(f"[MANIP] Approach contact: ~{contact_time:.1f}s at {speed:.2f} m/s.")

        cmd = Twist()
        cmd.linear.x = speed

        t0 = time.monotonic()
        t_contact = t0 + contact_time

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                return False

            now = time.monotonic()
            if now >= t_contact:
                break
            if (now - t0) > timeout:
                self.get_logger().warn("[MANIP] Approach timeout.")
                self._stop_robot()
                return False

            self.cmd_vel_pub.publish(cmd)
            await asyncio.sleep(0.05)

        self._stop_robot()
        return True

    async def _do_push(self, goal_handle) -> bool:
        """
        Push forward for push_distance_m using odom.
        """
        if self.odom_pose is None:
            self.get_logger().warn("[MANIP] No odom for push.")
            return False

        push_dist = float(self.get_parameter('push_distance_m').get_parameter_value().double_value)
        speed = float(self.get_parameter('push_speed').get_parameter_value().double_value)
        timeout = float(self.get_parameter('push_timeout_s').get_parameter_value().double_value)

        self.get_logger().info(
            f"[MANIP] Pushing forward: distance={push_dist:.2f} m at {speed:.2f} m/s."
        )

        x0, y0, _ = self.odom_pose
        cmd = Twist()
        cmd.linear.x = speed

        t0 = time.monotonic()
        moved = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._stop_robot()
                return False

            self.cmd_vel_pub.publish(cmd)
            await asyncio.sleep(0.05)

            if self.odom_pose is None:
                continue

            x, y, _ = self.odom_pose
            moved = math.hypot(x - x0, y - y0)

            if moved >= push_dist:
                break

            if (time.monotonic() - t0) > timeout:
                self.get_logger().warn("[MANIP] Push timeout.")
                self._stop_robot()
                return False

        self._stop_robot()
        self.get_logger().info(f"[MANIP] Push done, moved ~{moved:.2f} m.")
        return True

    async def _retract(self, goal_handle) -> None:
        """
        Go backward for retract_distance_m using odom.
        """
        if self.odom_pose is None:
            self.get_logger().warn("[MANIP] No odom for retract.")
            return

        dist = float(self.get_parameter('retract_distance_m').get_parameter_value().double_value)
        speed = float(self.get_parameter('retract_speed').get_parameter_value().double_value)
        timeout = float(self.get_parameter('retract_timeout_s').get_parameter_value().double_value)

        self.get_logger().info(
            f"[MANIP] Retracting: distance={dist:.2f} m at {speed:.2f} m/s."
        )

        x0, y0, _ = self.odom_pose
        cmd = Twist()
        cmd.linear.x = -speed

        t0 = time.monotonic()
        moved = 0.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                break

            self.cmd_vel_pub.publish(cmd)
            await asyncio.sleep(0.05)

            if self.odom_pose is None:
                continue

            x, y, _ = self.odom_pose
            moved = math.hypot(x - x0, y - y0)

            if moved >= dist:
                break

            if (time.monotonic() - t0) > timeout:
                self.get_logger().warn("[MANIP] Retract timeout.")
                break

        self._stop_robot()

    def _stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    @staticmethod
    def _yaw_from_quat(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = ManipulateObstacleServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
