#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.action import ActionClient, ActionServer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import time
from tf_transformations import quaternion_from_euler
from mir_navigation.action import ManipulateBox


class PushActionServer(Node):
    def __init__(self):
        super().__init__('push_action_server')
        self.get_logger().set_level(LoggingSeverity.DEBUG)
        self.get_logger().debug('Node initialized')

        self._action_server = ActionServer(
            self,
            ManipulateBox,
            'manipulate_box',
            self.execute_callback)

        self.current_goal_handle = None
        self.loop_active = False
        self.occ_grid = None
        self.robot_pose_stamped = None
        self.last_goal = None
        self.push_search_active = False
        self.push_executed = False

        self.obstacle_center = None
        self.obs_length = None
        self.obs_width  = None
        self.push_gap   = 0.5
        self.robot_width= 0.5

        self.status_timer = self.create_timer(1.0, self.print_waiting_status)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, qos)

        self.vel_pub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)

        self.create_timer(1.0, self.try_push_if_blocked)

    def execute_callback(self, goal_handle):
        self.get_logger().info(f"Received manipulation goal: x={goal_handle.request.x}, y={goal_handle.request.y}, length={goal_handle.request.length}, width={goal_handle.request.width}")
        
        # Reset all state variables at the start
        self.obstacle_center = (goal_handle.request.x, goal_handle.request.y)
        self.obs_length = goal_handle.request.length
        self.obs_width  = goal_handle.request.width
        self.loop_active = True
        self.push_executed = False
        self.push_search_active = False  # Ensure this is reset
        self.last_goal = None  # Reset last goal to allow new searches
        self.current_goal_handle = goal_handle

        feedback = ManipulateBox.Feedback()
        feedback.status = "Waiting to find valid push point..."
        goal_handle.publish_feedback(feedback)

        # Wait for push execution to complete
        while not self.push_executed and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)

        result = ManipulateBox.Result()
        result.success = self.push_executed
        
        # Clean up state variables BEFORE completing the goal
        self.loop_active = False
        self.current_goal_handle = None
        self.obstacle_center = None
        self.obs_length = None
        self.obs_width = None
        self.push_search_active = False
        self.last_goal = None
        
        # Complete the goal
        goal_handle.succeed()
        self.get_logger().info("[ACTION] Goal completed, returning result.")
        return result

    def print_waiting_status(self):
        if not self.loop_active:
            self.get_logger().info("Waiting for the action goal...")

    def map_cb(self, msg: OccupancyGrid):
        self.occ_grid = msg

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose   = msg.pose.pose
        self.robot_pose_stamped = ps

    def try_push_if_blocked(self):
        if not self.loop_active:
            return
        if self.obstacle_center is None or self.obs_length is None or self.obs_width is None:
            self.get_logger().debug('Obstacle info not yet received')
            return
        self.get_logger().debug('Timer fired: trying push search')
        if not self.occ_grid or not self.robot_pose_stamped:
            self.get_logger().debug('Map ya amcl_pose missing, skip kar raha hu')
            return

        if self.push_search_active or self.push_executed:
            self.get_logger().debug('Push already handled, skipping...')
            return

        self.push_search_active = True
        sides = ['right','left','up','down']

        for side in sides:
            pt = self.find_free_point(side)
            if not pt:
                self.get_logger().debug(f'({side}) no free point')
                continue

            if pt == self.last_goal:
                self.get_logger().debug('Same goal as before, skip')
                self.push_search_active = False
                return

            self.last_goal = pt
            self.get_logger().info(f'Found push point at {pt}, sending Nav2 goal')
            self.send_nav2_goal(pt)
            return

        self.get_logger().warn('Koi valid push point nahi mila')
        self.push_search_active = False

    def execute_push_motion(self):
        self.get_logger().info("Executing push motion")
        twist = Twist()
        twist.linear.x = 0.2
        push_duration = 10  # seconds

        if self.current_goal_handle:
            feedback = ManipulateBox.Feedback()
            feedback.status = "Pushing object..."
            self.current_goal_handle.publish_feedback(feedback)

        # 1. Push Forward
        start_time = time.time_ns()
        duration_ns = int(push_duration * 1e9)
        while time.time_ns() - start_time < duration_ns:
            self.vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

        # 2. Reverse a bit
        twist.linear.x = -0.2  # Reverse speed
        twist.angular.z = 0.0
        reverse_duration = 5.0  # seconds
        start_reverse = time.time_ns()
        reverse_ns = int(reverse_duration * 1e9)
        while time.time_ns() - start_reverse < reverse_ns:
            self.vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

        # 3. Rotate in-place
        twist.linear.x = 0.0
        twist.angular.z = 0.5  # Rotate CCW
        rotation_duration = 5.2  # seconds (for 180 degrees at 0.5 rad/s)
        start_turn = time.time_ns()
        turn_duration_ns = int(rotation_duration * 1e9)
        while time.time_ns() - start_turn < turn_duration_ns:
            self.vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 4. Stop
        twist.angular.z = 0.0
        twist.linear.x = 0.0
        self.vel_pub.publish(twist)
        self.push_executed = True

    def is_direction_valid(self, side):
        if self.occ_grid is None:
            return False

        cx, cy = self.obstacle_center
        l, w = self.obs_length, self.obs_width
        res = self.occ_grid.info.resolution
        ox = self.occ_grid.info.origin.position.x
        oy = self.occ_grid.info.origin.position.y
        W = self.occ_grid.info.width
        H = self.occ_grid.info.height
        data = self.occ_grid.data

        if side == 'right':
            xmin = cx + l/2
            xmax = xmin + l
            ymin = cy - w/2
            ymax = cy + w/2
        elif side == 'left':
            xmax = cx - l/2
            xmin = xmax - l
            ymin = cy - w/2
            ymax = cy + w/2
        elif side == 'up':
            ymin = cy + w/2
            ymax = ymin + l
            xmin = cx - l/2
            xmax = cx + l/2
        elif side == 'down':
            ymax = cy - w/2
            ymin = ymax - l
            xmin = cx - l/2
            xmax = cx + l/2
        else:
            return False

        i_min = max(int((xmin - ox) / res), 0)
        i_max = min(int((xmax - ox) / res), W)
        j_min = max(int((ymin - oy) / res), 0)
        j_max = min(int((ymax - oy) / res), H)

        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                if data[j * W + i] > 50:
                    return False
        return True

    def find_free_point(self, side, max_d=3.0, step=0.1):
        if not self.is_direction_valid(side):
            self.get_logger().info(f"Direction '{side}' blocked — skipping")
            return None

        x0, y0 = self.obstacle_center
        res = self.occ_grid.info.resolution
        ox = self.occ_grid.info.origin.position.x
        oy = self.occ_grid.info.origin.position.y

        if side == 'up':
            dx, dy = 0, 1
            start = self.obs_width/2 + self.push_gap + self.robot_width/2
        elif side == 'down':
            dx, dy = 0, -1
            start = self.obs_width/2 + self.push_gap + self.robot_width/2
        elif side == 'right':
            dx, dy = 1, 0
            start = self.obs_length/2 + self.push_gap + self.robot_width/2
        else:
            dx, dy = -1, 0
            start = self.obs_length/2 + self.push_gap + self.robot_width/2

        for d in np.arange(start, max_d, step):
            px = x0 + dx * d
            py = y0 + dy * d
            i = int((px - ox) / res)
            j = int((py - oy) / res)
            if not (0 <= i < self.occ_grid.info.width and 0 <= j < self.occ_grid.info.height):
                continue
            if self.occ_grid.data[j * self.occ_grid.info.width + i] > 50:
                continue
            return (px, py)
        return None

    def send_nav2_goal(self, pt):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.robot_pose_stamped.header.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = pt[0]
        goal.pose.pose.position.y = pt[1]

        dx = self.obstacle_center[0] - pt[0]
        dy = self.obstacle_center[1] - pt[1]
        theta = math.atan2(dy, dx)
        qx, qy, qz, qw = quaternion_from_euler(0, 0, theta)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 server unavailable')
            self.push_search_active = False
            return

        if self.current_goal_handle:
            feedback = ManipulateBox.Feedback()
            feedback.status = "Navigating to push location..."
            self.current_goal_handle.publish_feedback(feedback)

        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self.on_nav_response)

    def on_nav_response(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().warn('Nav2 rejected goal')
            self.push_search_active = False
            return
        self.get_logger().debug('Nav2 accepted, waiting for result...')
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(self.on_nav_result)

    def on_nav_result(self, fut):
        status = fut.result().status
        if status == 4:
            self.get_logger().info('Reached push point successfully')
            self.execute_push_motion()
        else:
            self.get_logger().warn(f'Nav2 failed with status {status}')
        self.push_search_active = False

def main(args=None):
    rclpy.init(args=args)
    node = PushActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()