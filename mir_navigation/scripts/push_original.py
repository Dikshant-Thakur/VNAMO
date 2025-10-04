#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Push Orchestrator (Nav2)
- Sequential run() loop (no timer callbacks)
- Acceptance timeout and execution timeout are handled cleanly via futures.
- Result wait uses a robust wall-clock loop (monotonic) to avoid false timeouts.
- Yaw alignment check removed (as requested).
"""

import math
import time
from typing import Optional, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Quaternion, Twist


from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import tf2_ros
from scipy.spatial.transform import Rotation as R


# ---------- tiny helpers ----------
def yaw_from_quat(q: Quaternion) -> float:
    s = 2.0 * (q.w * q.z + q.x * q.y)
    c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(s, c)

def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0; q.y = 0.0
    return q

def rot_local_to_world(cx, cy, yaw, lx, ly):
    return (cx + lx*math.cos(yaw) - ly*math.sin(yaw),
            cy + lx*math.sin(yaw) + ly*math.cos(yaw))

def point_in_oriented_rect(px, py, cx, cy, yaw, half_L, half_W) -> bool:
    dx, dy = px-cx, py-cy
    lx =  dx*math.cos(yaw) + dy*math.sin(yaw)
    ly = -dx*math.sin(yaw) + dy*math.cos(yaw)
    return (-half_L <= lx <= half_L) and (-half_W <= ly <= half_W)

def world_to_cell(x, y, ox, oy, res):  # (row, col)
    return (int(math.floor((y-oy)/res)), int(math.floor((x-ox)/res)))

def cell_center(rc, ox, oy, res):
    r, c = rc
    return (ox + (c+0.5)*res, oy + (r+0.5)*res)

def clamp_angle(a: float) -> float:
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


# ---------- main node ----------
class PushOrchestrator(Node):
    """
    Plan (+X/-X/+Y/-Y) -> Nav2 pre-manip pose -> Step-wise push.

    Contract:
      - Goal must be ACCEPTED within `nav_timeout_accept_s` else try next dir.
      - Reaching/execution governed by `nav_result_timeout_s`.
    """
    def __init__(self):
        super().__init__('push_orchestrator_nav2')

        # ---- RViz L* viz params/publisher ----
        self.declare_parameter('viz_enabled', True)
        self.declare_parameter('viz_topic', '/lstar_viz')

        viz_topic = self.get_parameter('viz_topic').get_parameter_value().string_value
        self.viz_pub = self.create_publisher(MarkerArray, viz_topic, 10)
        self._viz_last_pub = 0.0  # throttling

        #tf setup
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Params
        self.declare_parameter('global_costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('lethal_threshold', 253)
        self.declare_parameter('treat_unknown_as_blocked', True)

        self.declare_parameter('robot_length', 0.90)
        self.declare_parameter('robot_width',  0.50)
        self.declare_parameter('buffer_m',     0.10)
        self.declare_parameter('dir_order', ['+X','-X','+Y','-Y'])

        self.declare_parameter('standoff_m',   0.50)
        self.declare_parameter('contact_gap_m',0.06)
        # self.declare_parameter('yaw_align_deg', 12.0)  # removed: not used anymore

        self.declare_parameter('box_x', 6.5)
        self.declare_parameter('box_y', 0.5)
        self.declare_parameter('box_length', 0.50)
        self.declare_parameter('box_width',  2.0)

        self.declare_parameter('nav_timeout_accept_s', 20.0)
        self.declare_parameter('nav_result_timeout_s', 180.0)

        self.declare_parameter('cmd_vel_topic', '/diff_cont/cmd_vel_unstamped')
        self.declare_parameter('cmd_speed', 0.15)
        self.declare_parameter('step_m', 0.30)
        self.declare_parameter('contact_ratio_thresh', 0.30)
        self.declare_parameter('contact_hold_s', 0.25)
        self.declare_parameter('prox_thresh_m', 0.06)
        self.declare_parameter('settle_s', 0.30)

        # Subs / pubs / clients
        topic = self.get_parameter('global_costmap_topic').get_parameter_value().string_value
        self.create_subscription(OccupancyGrid, topic, self._map_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        self.create_subscription(Odometry, '/diff_cont/odom', self._odom_cb, 50)

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').get_parameter_value().string_value, 10
        )
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # State
        self.grid: Optional[np.ndarray] = None
        self.info = None
        self.robot_pose: Optional[Tuple[float,float,float]] = None
        self.odom_pose: Optional[Tuple[float,float,float]]  = None
        self.odom_vx_filt = 0.0
        self._ema_alpha = 0.3

        self.box: Dict[str,float] = {
            'x': self.get_parameter('box_x').get_parameter_value().double_value,
            'y': self.get_parameter('box_y').get_parameter_value().double_value,
            'L': self.get_parameter('box_length').get_parameter_value().double_value,
            'W': self.get_parameter('box_width').get_parameter_value().double_value,
        }
        self.robot_L = float(self.get_parameter('robot_length').get_parameter_value().double_value)
        self.robot_W = float(self.get_parameter('robot_width').get_parameter_value().double_value)

        self.dir_order = list(self.get_parameter('dir_order').get_parameter_value().string_array_value or ['+X','-X','+Y','-Y'])

        self.get_logger().info("Push orchestrator up. Waiting for costmap + amcl...")

    # ----- callbacks -----
    def _map_cb(self, msg: OccupancyGrid):
        self.info = msg.info
        self.grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.robot_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        self.odom_vx_filt = (1.0 - self._ema_alpha)*self.odom_vx_filt + self._ema_alpha*vx
        p = msg.pose.pose
        self.odom_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    # ----- utils -----
    def _occ_bool(self):
        leth = int(self.get_parameter('lethal_threshold').get_parameter_value().integer_value)
        occ = (self.grid >= leth)
        if self.get_parameter('treat_unknown_as_blocked').get_parameter_value().bool_value:
            occ = np.logical_or(occ, (self.grid == 255) | (self.grid < 0))
        return occ

    def _rect_free(self, occ, center, yaw, L, W) -> bool:
        ox, oy, res = self.info.origin.position.x, self.info.origin.position.y, self.info.resolution
        hL, hW = 0.5*L, 0.5*W
        corners = [rot_local_to_world(center[0], center[1], yaw, sx, sy)
                   for sx in (-hL, hL) for sy in (-hW, hW)]
        xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
        r0, c0 = world_to_cell(min(xs), min(ys), ox, oy, res)
        r1, c1 = world_to_cell(max(xs), max(ys), ox, oy, res)
        H, Wg = occ.shape
        r0, r1 = max(0, min(H-1, r0)), max(0, min(H-1, r1))
        c0, c1 = max(0, min(Wg-1, c0)), max(0, min(Wg-1, c1))
        for r in range(min(r0,r1), max(r0,r1)+1):
            for c in range(min(c0,c1), max(c0,c1)+1):
                x, y = cell_center((r,c), ox, oy, res)
                if point_in_oriented_rect(x, y, center[0], center[1], yaw, hL, hW):
                    if occ[r, c]:
                        return False
        return True
    
    
    def _viz_clear(self):
        if not hasattr(self, 'viz_pub'):
            return
        m = Marker(); m.action = Marker.DELETEALL
        arr = MarkerArray(); arr.markers.append(m)
        self.viz_pub.publish(arr)

    def _viz_selected_lstar(self, dir_tag: str, Ls: float):
        if not self.get_parameter('viz_enabled').get_parameter_value().bool_value:
            return
        if self.grid is None or self.robot_pose is None:
            return

        bx, by, bl, bw = self.box['x'], self.box['y'], self.box['L'], self.box['W']
        xmin, xmax = bx - 0.5*bl, bx + 0.5*bl
        ymin, ymax = by - 0.5*bw, by + 0.5*bw

        if dir_tag in ('+X','-X'): Wdir = max(self.robot_W, bw)
        else:                      Wdir = max(self.robot_L, bl)

        if dir_tag == '+X':   yaw_dir, sx, sy = 0.0,        xmin, by
        elif dir_tag == '-X': yaw_dir, sx, sy = math.pi,    xmax, by
        elif dir_tag == '+Y': yaw_dir, sx, sy = math.pi/2., bx,   ymin
        else:                 yaw_dir, sx, sy = -math.pi/2.,bx,   ymax

        ax, ay = math.cos(yaw_dir), math.sin(yaw_dir)
        frame = 'map'; now = self.get_clock().now().to_msg()
        arr = MarkerArray(); mid = 0

        # Box (context)
        m_box = Marker(); m_box.header.frame_id = frame; m_box.header.stamp = now
        m_box.ns = "lstar_sel"; m_box.id = mid; mid += 1
        m_box.type = Marker.CUBE; m_box.action = Marker.ADD
        m_box.pose.position.x = bx; m_box.pose.position.y = by
        m_box.pose.orientation = quat_from_yaw(0.0)
        m_box.scale.x = bl; m_box.scale.y = bw; m_box.scale.z = 0.02
        m_box.color = ColorRGBA(r=0.1, g=0.6, b=1.0, a=0.25)
        arr.markers.append(m_box)

        # Selected L* rectangle
        if Ls > 0.0:
            cx = sx + 0.5*Ls*ax; cy = sy + 0.5*Ls*ay
            m_ls = Marker(); m_ls.header.frame_id = frame; m_ls.header.stamp = now
            m_ls.ns = "lstar_sel"; m_ls.id = mid; mid += 1
            m_ls.type = Marker.CUBE; m_ls.action = Marker.ADD
            m_ls.pose.position.x = cx; m_ls.pose.position.y = cy
            m_ls.pose.orientation = quat_from_yaw(yaw_dir)
            m_ls.scale.x = Ls; m_ls.scale.y = Wdir; m_ls.scale.z = 0.05
            m_ls.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.65)
            arr.markers.append(m_ls)

        # Label
        m_txt = Marker(); m_txt.header.frame_id = frame; m_txt.header.stamp = now
        m_txt.ns = "lstar_sel"; m_txt.id = mid; mid += 1
        m_txt.type = Marker.TEXT_VIEW_FACING; m_txt.action = Marker.ADD
        tx = sx + (0.2 + max(0.0, Ls))*ax; ty = sy + (0.2 + max(0.0, Ls))*ay
        m_txt.pose.position.x = tx; m_txt.pose.position.y = ty
        m_txt.scale.z = 0.25
        m_txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)  # floats!
        m_txt.text = f"{dir_tag}  L*={Ls:.2f} m"
        arr.markers.append(m_txt)

        self._viz_clear()
        self.viz_pub.publish(arr)


    def _robot_extremes(self, rx, ry, ryaw):
        hL, hW = 0.5*self.robot_L, 0.5*self.robot_W
        pts = [rot_local_to_world(rx, ry, ryaw, sx, sy) for sx in (-hL,hL) for sy in (-hW,hW)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return max(xs), min(xs), min(ys), max(ys)  # x_front, x_back, y_right, y_left
    def _box_extents(self):
        bl, bw = self.box['L'], self.box['W']
        #xmin,xmax, ymin, ymax
        return (self.box['x'] - 0.5*bl, self.box['x'] + 0.5*bl,
                self.box['y'] - 0.5*bw, self.box['y'] + 0.5*bw)

    def _dir_unit_and_yaw(self, tag):
        if tag == '+X': return (1.0,0.0), 0.0
        if tag == '-X': return (-1.0,0.0), math.pi
        if tag == '+Y': return (0.0,1.0), math.pi/2.0
        if tag == '-Y': return (0.0,-1.0), -math.pi/2.0
        return (1.0,0.0), 0.0

    def _recompute_Lstar(self, tag, K, buf, xf, xb, yl, yr, xmin, xmax, ymin, ymax) -> float:
        if tag == '+X': return max(0.0, (xf + K + buf) - xmin)
        if tag == '-X': return max(0.0, xmax - (xb - K - buf))
        if tag == '+Y': return max(0.0, (yl + K + buf) - ymin)
        if tag == '-Y': return max(0.0, ymax - (yr - K - buf))
        return 0.0

    def _premanip_pose(self, dtag) -> Tuple[float,float,float]:
        s   = float(self.get_parameter('standoff_m').get_parameter_value().double_value)
        gap = float(self.get_parameter('contact_gap_m').get_parameter_value().double_value)
        d   = s + 0.5*self.robot_L + gap
        xmin, xmax, ymin, ymax = self._box_extents()
        bx, by = self.box['x'], self.box['y']
        if dtag == '+X':  return (xmin - d, by, 0.0)
        if dtag == '-X':  return (xmax + d, by, math.pi)
        if dtag == '+Y':  return (bx, ymin - d, math.pi/2.0)
        # -Y
        return (bx, ymax + d, -math.pi/2.0)

    # ----- nav2 helpers -----
    def _build_nav_goal(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x; ps.pose.position.y = y
        ps.pose.orientation = quat_from_yaw(yaw)
        goal = NavigateToPose.Goal()
        goal.pose = ps
        return goal

    def _send_and_wait_accept(self, goal) -> Optional[object]:
        """Wait ONLY for ACCEPTANCE within nav_timeout_accept_s (robust, wall-clock)."""
        T = float(self.get_parameter('nav_timeout_accept_s').get_parameter_value().double_value)

        self.get_logger().info(f"[DEBUG] Waiting for nav server (timeout={T}s)...")
        if not self.nav_client.wait_for_server(timeout_sec=T):
            self.get_logger().warn("[NAV] Action server not available")
            return None

        self.get_logger().info("[DEBUG] Server ready, sending goal...")
        send_future = self.nav_client.send_goal_async(goal)

        # Manual wait using wall-clock (monotonic) + spin_once
        t0 = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if send_future.done():
                break
            if (time.monotonic() - t0) >= T:
                self.get_logger().warn(f"[NAV] Goal send/accept wait timed out after {T:.1f}s")
                return None

        if not send_future.done():
            self.get_logger().warn("[NAV] Goal send future not completed")
            return None

        gh = send_future.result()
        if not gh or not gh.accepted:
            self.get_logger().warn("[NAV] Goal NOT accepted")
            return None

        self.get_logger().info("[NAV] Goal accepted ✅")
        return gh

    def _wait_nav_result(self, goal_handle) -> Optional[int]:
        """Wait for execution result up to nav_result_timeout_s using a robust wall-clock loop."""
        T = float(self.get_parameter('nav_result_timeout_s').get_parameter_value().double_value)
        result_future = goal_handle.get_result_async()

        t0 = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            # Got result?
            if result_future.done():
                try:
                    result = result_future.result()
                    return getattr(result, "status", None)
                except Exception as e:
                    self.get_logger().warn(f"[NAV] Exception reading result: {e}")
                    return None

            # Timed out?
            if (time.monotonic() - t0) >= T:
                self.get_logger().warn(f"[NAV] Result wait timeout (> {T:.1f}s). Canceling.")
                try:
                    cancel_future = goal_handle.cancel_goal_async()
                    # wait briefly for cancel to settle to avoid races with next goal
                    t_cancel0 = time.monotonic()
                    while rclpy.ok() and not cancel_future.done() and (time.monotonic() - t_cancel0) < 2.0:
                        rclpy.spin_once(self, timeout_sec=0.05)
                except Exception as e:
                    self.get_logger().warn(f"[NAV] Cancel request error: {e}")
                return None

        # Node shutting down
        return None

    def _publish_vx(self, v: float):
        msg = Twist(); msg.linear.x = v; msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def _project_along_dir(self, plan_dir: str, x: float, y: float) -> float:
        (ax, ay), _ = self._dir_unit_and_yaw(plan_dir)
        return ax * x + ay * y
    
    # --- Add this function inside the class PushOrchestrator ---


    def _go_side_peek_obstacle_side(self,
                                plan_dir: str,
                                reach_wait_s: float = 15.0,
                                peek_forward: float = 0.25,
                                extra_buffer_m: float = None) -> bool:
        """
        Make two NAV goals *anchored at the obstacle side* (not robot side):
        1) Anchor = box center nudged 'peek_forward' along push direction
        2) Left/Right offsets so robot stands beside the box (outside corridor)
        3) Try nearer first; wait up to 'reach_wait_s' to reach; on timeout cancel & try other
        4) Return True if any reached; else False
        """
        # --- Direction basis ---
        (_, _), yaw_dir = self._dir_unit_and_yaw(plan_dir)
        fx, fy = math.cos(yaw_dir), math.sin(yaw_dir)           # forward (push dir)
        ux, uy = -math.sin(yaw_dir), math.cos(yaw_dir)          # left (perp)

        # --- Box center (anchor near obstacle) ---
        bx, by = self.box['x'], self.box['y']                   # already in your class
        ax = bx + peek_forward * fx
        ay = by + peek_forward * fy

        # --- Corridor width per push axis ---
        if plan_dir in ('+X', '-X'):
            corridor_W = max(self.robot_W, self.box['W'])
        else:  # '+Y' / '-Y'
            corridor_W = max(self.robot_L, self.box['L'])

        # --- Safe side offset (corridor edge + half robot + buffer) ---
        buf_param = float(self.get_parameter('buffer_m').get_parameter_value().double_value)
        buf = extra_buffer_m if (extra_buffer_m is not None) else buf_param
        offset = 0.5 * corridor_W + 0.5 * self.robot_W + buf

        # --- Left/Right targets facing down the corridor ---
        targets = {
            'left':  (ax + offset * ux, ay + offset * uy, yaw_dir),
            'right': (ax - offset * ux, ay - offset * uy, yaw_dir),
        }

        # --- Try nearer first (from current robot pose) ---
        rx, ry = (self.robot_pose[0], self.robot_pose[1]) if self.robot_pose else (ax, ay)
        order = sorted(targets.items(), key=lambda kv: (kv[1][0]-rx)**2 + (kv[1][1]-ry)**2)

        for tag, (tx, ty, tyaw) in order:
            self.get_logger().info(
                f"[SCOUT] Obstacle-side {tag}: target=({tx:.2f},{ty:.2f},{math.degrees(tyaw):.1f}deg), "
                f"offset={offset:.2f} m, peek_fwd={peek_forward:.2f} m"
            )
            goal = self._build_nav_goal(tx, ty, tyaw)
            gh = self._send_and_wait_accept(goal)
            if gh is None:
                self.get_logger().warn(f"[SCOUT] {tag}: goal not accepted.")
                continue

            # --- Wait up to reach_wait_s to *reach*; if not, cancel & try other ---
            result_future = gh.get_result_async()
            t0 = time.monotonic()
            reached = False
            while rclpy.ok() and (time.monotonic() - t0) < max(0.0, reach_wait_s):
                rclpy.spin_once(self, timeout_sec=0.05)
                if result_future.done():
                    try:
                        result = result_future.result()
                        status = getattr(result, "status", None)
                    except Exception as e:
                        self.get_logger().warn(f"[SCOUT] {tag}: result read error: {e}")
                        status = None
                    if status == GoalStatus.STATUS_SUCCEEDED:
                        self.get_logger().info(f"[SCOUT] {tag}: reached ✅")
                        reached = True
                    else:
                        self.get_logger().warn(f"[SCOUT] {tag}: status={status}")
                    break

            if not reached and (time.monotonic() - t0) >= reach_wait_s:
                self.get_logger().warn(f"[SCOUT] {tag}: timeout {reach_wait_s:.1f}s → cancel.")
                try:
                    cancel_future = gh.cancel_goal_async()
                    t_cancel0 = time.monotonic()
                    while rclpy.ok() and not cancel_future.done() and (time.monotonic() - t_cancel0) < 2.0:
                        rclpy.spin_once(self, timeout_sec=0.05)
                except Exception as e:
                    self.get_logger().warn(f"[SCOUT] {tag}: cancel error: {e}")

            if reached:
                return True

        self.get_logger().warn("[SCOUT] Both obstacle-side goals failed → ABORT.")
        return False



    def _approach_contact(self, plan_dir: str, K: float):
        """
        Softly drive forward from pre-manip until contact is confirmed.
        Returns (ok: bool, s_origin: float, cap_from_contact: float)
        Contact is considered when: gap <= prox_thresh AND velocity ratio <= contact_ratio_thresh
        for at least contact_hold_s seconds.
        """
        if self.robot_pose is None or self.odom_pose is None:
            self.get_logger().warn("[PUSH] Missing robot/odom for approach. Abort.")
            return False, None, None

        # ---- Parameters ----
        cmd_v   = float(self.get_parameter('cmd_speed').get_parameter_value().double_value)
        prox_t  = float(self.get_parameter('prox_thresh_m').get_parameter_value().double_value)          # e.g., 0.06 m
        drop_r  = float(self.get_parameter('contact_ratio_thresh').get_parameter_value().double_value)   # e.g., 0.30
        hold_s  = float(self.get_parameter('contact_hold_s').get_parameter_value().double_value)         # e.g., 0.25 s
        settle  = float(self.get_parameter('settle_s').get_parameter_value().double_value)               # e.g., 0.30 s
        buf     = float(self.get_parameter('buffer_m').get_parameter_value().double_value)

        # gentle approach speed; normalize ratio by this
        approach_v = max(0.05, 0.5 * cmd_v)
        watchdog_s = 20.0  # timeout for approach loop

        self.get_logger().info("[PUSH] Approaching until contact...")
        t0 = time.monotonic()
        contact_on = False
        contact_since = None

        while rclpy.ok():

            rclpy.spin_once(self, timeout_sec=0.01)

            # ---- Safety: corridor around the box must remain free ----
            occ = self._occ_bool()
            xmin, xmax, ymin, ymax = self._box_extents()
            if plan_dir == '+X':
                center=(xmax+0.5*K, self.box['y']); yaw=0.0;       W=self.robot_W
            elif plan_dir == '-X':
                center=(xmin-0.5*K, self.box['y']); yaw=math.pi;    W=self.robot_W
            elif plan_dir == '+Y':
                center=(self.box['x'], ymax+0.5*K); yaw=math.pi/2.0;W=self.robot_L
            else:  # '-Y'
                center=(self.box['x'], ymin-0.5*K); yaw=-math.pi/2.0;W=self.robot_L
            if not self._rect_free(occ, center, yaw, K, W):
                self._stop()
                self.get_logger().warn("[PUSH] Corridor blocked during approach. Abort.")
                return False, None, None

            # ---- Use ODOM pose for proximity during approach (more responsive than AMCL) ----
            pose_src = self.odom_pose if self.odom_pose is not None else self.robot_pose
            
            rx, ry, ryaw = pose_src
            xf, xb, yr, yl = self._robot_extremes(rx, ry, ryaw)
            xmin, xmax, ymin, ymax = self._box_extents()
            gap = self._gap_to_contact(plan_dir, xf, xb, yl, yr, xmin, xmax, ymin, ymax)
            # self.get_logger().info(f"[DEBUG] Robot pose: rx={rx:.3f}, ry={ry:.3f}, ryaw={math.degrees(ryaw):.1f}deg")
            # self.get_logger().info(f"[DEBUG] Robot extremes: xb={xb:.3f}, xf={xf:.3f}, yr={yr:.3f}, yl={yl:.3f}")
            # self.get_logger().info(f"[DEBUG] Box extents: xmin={xmin:.3f}, xmax={xmax:.3f}, ymin={ymin:.3f}, ymax={ymax:.3f}")
            # self.get_logger().info(f"[DEBUG] Gap calculation: xmin-xf = {xmin:.3f} - {xf:.3f} = {gap:.3f}")

            # ---- Velocity drop ratio normalized by *approach_v* ----
            current_cmd = max(approach_v, 1e-3)
            ratio = self.odom_vx_filt / current_cmd

            prox_ok = (gap <= prox_t)

            if not (prox_ok):
                # keep creeping forward
                self._publish_vx(approach_v)
                self.get_logger().info(f"[PUSH] Approaching... gap={gap:.3f} m")
            else:
                # require the state to hold for hold_s to avoid false positives
                now = time.time()
                if not contact_on:
                    contact_on = True
                    contact_since = now
                elif (now - contact_since) >= hold_s:
                    # Contact confirmed
                    self._stop()
                    time.sleep(settle)

                    # Distance origin from contact point (project along plan_dir)
                    (ax, ay), _ = self._dir_unit_and_yaw(plan_dir)
                    s_origin = ax * self.odom_pose[0] + ay * self.odom_pose[1]

                    # Recompute L* from (preferably) map-frame pose for accuracy.
                    # Use AMCL (robot_pose) if available; otherwise fall back to odom_pose.
                    rx_map, ry_map, ryaw_map = self.robot_pose if self.robot_pose is not None else self.odom_pose
                    xf_c, xb_c, yr_c, yl_c = self._robot_extremes(rx_map, ry_map, ryaw_map)
                    xmin, xmax, ymin, ymax = self._box_extents()
                    Ls_contact = self._recompute_Lstar(plan_dir, K, buf, xf_c, xb_c, yl_c, yr_c, xmin, xmax, ymin, ymax)
                    
                    cap = Ls_contact + 0.20

                    try:
                        self._viz_selected_lstar(plan_dir, Ls_contact)
                    except Exception as e:
                        self.get_logger().warn(f"[VIZ] contact L* publish failed: {e}")

                    self.get_logger().info(
                        f"[PUSH] Contact confirmed. gap={gap:.3f} m, ratio={ratio:.3f} | "
                        f"L*={Ls_contact:.3f} m, cap={cap:.3f} m (from contact)"
                    )
                    return True, s_origin, cap

            # ---- Watchdog timeout ----
            if (time.monotonic() - t0) > watchdog_s:
                self._stop()
                self.get_logger().warn("[PUSH] Approach watchdog timeout. Abort.")
                return False, None, None

        return False, None, None



    def _stop(self):
        self._publish_vx(0.0)
    def _brake_and_settle(self, duration_s: float, hz: int = 30) -> None:
        """Publish zero Twist at a steady rate to actively brake & settle."""
        dt = 1.0 / max(1, hz)
        t_end = time.monotonic() + max(0.0, duration_s)
        zero = Twist()
        while time.monotonic() < t_end and rclpy.ok():
            self.cmd_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(dt)

    def _gap_to_contact(self, tag, xf, xb, yl, yr, xmin, xmax, ymin, ymax):
        if tag == '+X': return (xmin - xf)
        if tag == '-X': return (xb - xmax)
        if tag == '+Y': return (ymin - yl)
        if tag == '-Y': return (yr - ymax)
        return 1e9

    # ---------- planning ----------
    def _compute_plan(self) -> Optional[Dict]:
        # quick viz tick so RViz updates even if plan fails/succeeds
        if self.grid is None or self.robot_pose is None:
            return None
        occ = self._occ_bool()
        rx, ry, ryaw = self.robot_pose
        x_back, x_front, y_right, y_left = self._robot_extremes(rx, ry, ryaw)
        xmin, xmax, ymin, ymax = self._box_extents()

        K = max(1.5*self.robot_L, 2.0)
        W_fb = max(self.robot_W, self.box['W'])
        W_lr = max(self.robot_L, self.box['L'])

        def corridor_ok(tag: str) -> bool:
            if tag == '+X':
                center=(xmax+0.5*K, self.box['y']); yaw=0.0; W=W_fb
            elif tag == '-X':
                center=(xmin-0.5*K, self.box['y']); yaw=math.pi; W=W_fb
            elif tag == '+Y':
                center=(self.box['x'], ymax+0.5*K); yaw=math.pi/2.0; W=W_lr
            else:
                center=(self.box['x'], ymin-0.5*K); yaw=-math.pi/2.0; W=W_lr
            return self._rect_free(occ, center, yaw, K, W)

        chosen = None
        for d in self.dir_order:
            self.get_logger().info(f"[PLAN] Testing corridor {d} ...")
            if corridor_ok(d):
                chosen = d
                self.get_logger().info(f"[PLAN] Corridor {d} FREE ✅")
                break
            else:
                self.get_logger().info(f"[PLAN] Corridor {d} BLOCKED ❌")

        if chosen is None:
            return None

        buf = float(self.get_parameter('buffer_m').get_parameter_value().double_value)
        Ls  = self._recompute_Lstar(chosen, K, buf, x_front, x_back, y_left, y_right, xmin, xmax, ymin, ymax)
        cap = Ls + 0.20
        Wdir = W_fb if chosen in ['+X','-X'] else W_lr
        try:
            self._viz_selected_lstar(chosen, Ls)
        except Exception as e:
            self.get_logger().warn(f"[VIZ] selected L* publish failed: {e}")
        return {'dir': chosen, 'K': K, 'W_dir': Wdir, 'L_star': Ls, 'cap': cap}

    # ---------- pushing ----------
    # --- Add this function inside the class PushOrchestrator ---

    def _correct_orientation_inplace(self, plan_dir: str) -> bool:
        """
        Rotate in place until base_link yaw in /map == y_ref.
        Uses direct TF(map->base_link) for y_raw (robust),
        and allows axis keywords for y_ref (+X,+Y,-X,-Y).
        """
        import math, time
        from geometry_msgs.msg import Twist
        from scipy.spatial.transform import Rotation as R

        def wrap(a): return math.atan2(math.sin(a), math.cos(a))
        rad = math.radians; deg = math.degrees

        # ---- Tune once
        tol   = rad(0.4)     # stop within 0.4°
        kp    = 0.20         # P gain
        w_max = 0.5          # rad/s
        w_min = 0.06         # anti-stiction min speed
        timeout = 15.0
        dt = 0.03
        plate_yaw_offset_deg = getattr(self, "plate_yaw_offset_deg", -0.5)  # optional

        # ---- y_ref from axis keywords or existing plan_dir

            # fallback to your existing mapping
        try:
            (_, _), y_ref = self._dir_unit_and_yaw(plan_dir)
        except Exception:
            self.get_logger().warn(f"[ORIENT] plan_dir='{plan_dir}' unknown; default +X")
            y_ref = 0.0

        # apply visual/tool offset if any
        y_ref = wrap(y_ref + rad(plate_yaw_offset_deg))

        self.get_logger().info(
            f"[ORIENT] target='{plan_dir}' y_ref={deg(y_ref):.2f}° (plate_off={plate_yaw_offset_deg:.2f}°)"
        )

        t0 = time.monotonic()
        while rclpy.ok():
            # ---- y_raw directly from TF(map->base_link)
            try:
                tf_mb = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                q = tf_mb.transform.rotation
                y_raw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
            except Exception as ex:
                self.get_logger().warn(f"[ORIENT] TF map->base_link failed: {ex}")
                return False

            e = wrap(y_ref - y_raw)
            if abs(e) <= tol:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    f"[ORIENT] aligned ✅ y_ref={deg(y_ref):.2f}° y_raw={deg(y_raw):.2f}° err={deg(e):.2f}° "
                    f"t={(time.monotonic()-t0):.2f}s"
                )
                return True

            # P-control with caps + min speed
            w = kp * e
            if abs(w) > w_max: w = math.copysign(w_max, w)
            if abs(w) < w_min: w = math.copysign(w_min, w)

            cmd = Twist()
            cmd.angular.z = w            # yaw rate around base_link z
            self.cmd_pub.publish(cmd)

            if (time.monotonic() - t0) > timeout:
                self.cmd_pub.publish(Twist())
                self.get_logger().warn(f"[ORIENT] timeout; last err={deg(e):.2f}°")
                return False

            try: rclpy.spin_once(self, timeout_sec=0.01)
            except Exception: pass
            time.sleep(dt)



    # def _correct_orientation_inplace(self, plan_dir: str) -> bool:
    #     """
    #     Simple closed-loop: keep rotating in place until yaw ≈ target.
    #     No extra parameters; hardcoded small tolerance & speeds.
    #     """

    #     try:
    #         tf = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())  # map <- odom
    #     except Exception as ex:
    #         self.get_logger().warn(f"[ORIENT] TF lookup failed map<-odom: {ex}")
    #         return False

    #     # target yaw from your existing helper
    #     _, yaw_ref = self._dir_unit_and_yaw(plan_dir)

    #     # local helper to wrap angle to [-pi, pi]
    #     def wrap(a: float) -> float:
    #         return math.atan2(math.sin(a), math.cos(a))

    #     # choose pose source (prefer odom if available)
    #     pose = self.odom_pose if self.odom_pose is not None else self.robot_pose
    #     if pose is None:
    #         self.get_logger().warn("[ORIENT] No pose available.")
    #         return False
        
    #     tol = math.radians(0.2)   # 0.2°
    #     kp  = 0.2
    #     w_max = 0.5
    #     w_min = 0.08

    #     # entry log
    #     self.get_logger().info(
    #         f"[ORIENT] start dir={plan_dir} | yaw_ref={math.degrees(yaw_ref):.2f}deg "
    #         f"| tol={math.degrees(tol):.2f}deg kp={kp:.2f} w_max={w_max:.2f} w_min={w_min:.2f} src={'ODOM' if self.odom_pose is not None else 'MAP'}"
    #     )

    #     t0 = time.monotonic()
    #     timeout_s = 15.0
    #     dt = 0.03
    #     tick = 0

    #     while True:
    #         pose = self.odom_pose if self.odom_pose is not None else self.robot_pose
    #         if pose is None:
    #             self.cmd_pub.publish(Twist())
    #             self.get_logger().warn("[ORIENT] pose lost → abort.")
    #             return False

    #         # get fresh map<-odom each tick (AMCL may update this)
    #         try:
    #             tf = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
    #             q = tf.transform.rotation  # xyzw
    #             yaw_off = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
    #         except Exception as ex:
    #             self.get_logger().warn(f"[ORIENT] TF lookup failed: {ex}")
    #             yaw_off = 0.0  # fail-safe 
 
    #         # compute yaw_now in /map
    #         if self.odom_pose is not None:
    #             odom_yaw = self.odom_pose[2]
    #             yaw_now  = (odom_yaw + yaw_off + math.pi) % (2*math.pi) - math.pi
    #         else:
    #             # pose is already in /map
    #             yaw_now = pose[2]

    #         e = wrap(yaw_ref - yaw_now)

    #         if abs(e) <= tol:
    #             self.cmd_pub.publish(Twist())
    #             self.get_logger().info(f"[ORIENT] aligned | final_err={math.degrees(e):.2f}deg | t={(time.monotonic()-t0):.2f}s")
    #             return True

    #         w_raw = kp * e
    #         clipped = abs(w_raw) > w_max
    #         w = max(min(w_raw, w_max), -w_max)
    #         hit_min = abs(w) < w_min
    #         if hit_min:
    #             w = math.copysign(w_min, w)

    #         cmd = Twist(); cmd.angular.z = w
    #         self.cmd_pub.publish(cmd)

    #         if tick % 3 == 0:
    #             self.get_logger().info(
    #                 f"[ORIENT] yaw={math.degrees(yaw_now):.2f}deg "
    #                 f"err={math.degrees(e):.2f}deg w_cmd={w:.3f} "
    #                 f"clip={clipped} min={hit_min}"
    #             )
    #         tick += 1 

    #         try:
    #             rclpy.spin_once(self, timeout_sec=0.01)
    #         except Exception:
    #             pass
    #         time.sleep(dt)

    #         # watchdog
    #         if (time.monotonic() - t0) > timeout_s:
    #             self.cmd_pub.publish(Twist())
    #             self.get_logger().warn(f"[ORIENT] timeout {timeout_s:.1f}s; last_err={math.degrees(e):.2f}deg")
    #             return False

        # tol = math.radians(0.2)   # ~2°
        # kp  = 0.2                 # simple proportional
        # w_max = 0.5               # rad/s cap
        # w_min = 0.08              # minimum to break static friction

        # t0 = time.monotonic()
        # timeout_s = 15.0
        # dt = 0.03                 # ~33 Hz

        # while True:
        #     # refresh latest yaw
        #     pose = self.odom_pose if self.odom_pose is not None else self.robot_pose
        #     if pose is None:
        #         # stop and fail gracefully
        #         z = Twist(); self.cmd_pub.publish(z)
        #         return False

        #     yaw_now = pose[2]
        #     e = wrap(yaw_ref - yaw_now)

        #     # done?
        #     if abs(e) <= tol:
        #         z = Twist()  # stop
        #         self.cmd_pub.publish(z)
        #         return True

        #     # simple P -> omega, with clamp + minimum magnitude
        #     w = kp * e
        #     if w >  w_max: w =  w_max
        #     if w < -w_max: w = -w_max
        #     if abs(w) < w_min:
        #         w = math.copysign(w_min, w)

        #     cmd = Twist()
        #     cmd.linear.x = 0.0
        #     cmd.angular.z = w
        #     self.cmd_pub.publish(cmd)

        #     # let callbacks run a bit & pace the loop
        #     try:
        #         rclpy.spin_once(self, timeout_sec=1.0)
        #     except Exception:
        #         pass
        #     time.sleep(dt)

            # # simple watchdog
            # if (time.monotonic() - t0) > timeout_s:
            #     z = Twist(); self.cmd_pub.publish(z)
            #     self.get_logger().warn("[ORIENT] Timeout; aborting.")
            #     return False
            


    def _do_push_from_contact(self, plan_dir: str, K: float, s_origin: float, cap: float) -> bool:
        """
        Distance-based push from *contact*: push until progress reaches 'cap' OR gate clears.
        - Ramp down near target to avoid overshoot
        - Active zero-hold braking on stop
        - L* computed against a 'virtual box' that moves with progress so it monotonically decreases
        """
        # Preconditions
        if self.robot_pose is None or self.odom_pose is None:
            self.get_logger().warn("[PUSH] Missing robot/odom pose.")
            return False
        if not self._correct_orientation_inplace(plan_dir):
            self.get_logger().warn("[PUSH] Orientation correction failed.")
            return False

        # Params
        cmd_v  = float(self.get_parameter('cmd_speed').get_parameter_value().double_value)  # forward m/s
        buf    = float(self.get_parameter('buffer_m').get_parameter_value().double_value)   # safety buffer
        settle = float(self.get_parameter('settle_s').get_parameter_value().double_value)   # brake hold
        # Local tuning (no new params required)
        BRAKE_ZONE = 0.12     # start ramp-down within 12 cm
        MIN_V      = 0.05     # don't go slower than this until final few mm
        V_EPS      = 0.02     # optional velocity gate (if you have filtered odom vel)

        # Direction basis and target along that axis
        (ax, ay), yaw_dir = self._dir_unit_and_yaw(plan_dir)
        s_target = s_origin + cap
        self.get_logger().info(f"[PUSH] Distance-mode: target +{cap:.3f} m from contact")

        # --- snapshot the box extents at contact; we'll *virtually* move them with progress ---
        xmin0, xmax0, ymin0, ymax0 = self._box_extents()

        # Main push loop
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            # Safety corridor free check (same K/W logic you already use)
            occ = self._occ_bool()
            xmin, xmax, ymin, ymax = self._box_extents()
            if plan_dir == '+X':
                center=(xmax+0.5*K, self.box['y']); yaw=0.0;        W=self.robot_W
            elif plan_dir == '-X':
                center=(xmin-0.5*K, self.box['y']); yaw=math.pi;     W=self.robot_W
            elif plan_dir == '+Y':
                center=(self.box['x'], ymax+0.5*K); yaw=math.pi/2.0; W=self.robot_L
            else:  # '-Y'
                center=(self.box['x'], ymin-0.5*K); yaw=-math.pi/2.0; W=self.robot_L
            if not self._rect_free(occ, center, yaw, K, W):
                self._stop()
                self.get_logger().warn("[PUSH] Corridor blocked → abort.")
                return False

            # Progress from contact (ODOM projection along plan_dir)
            ox, oy = self.odom_pose[0], self.odom_pose[1]
            s_now = ax * ox + ay * oy
            progress = max(0.0, s_now - s_origin)     # >= 0
            dist_left = max(0.0, s_target - s_now)    # how much of 'cap' remains

            # --- compute L* against a *moving* (virtual) box so it shrinks correctly ---
            # shift extents by progress along the push axis only
            if plan_dir in ('+X', '-X'):
                dx = ax * progress          # ax = ±1 → signed shift
                xmin_v = xmin0 + dx; xmax_v = xmax0 + dx
                ymin_v = ymin0;  ymax_v = ymax0
            else:
                dy = ay * progress          # ay = ±1 → signed shift
                xmin_v = xmin0;  xmax_v = xmax0
                ymin_v = ymin0 + dy; ymax_v = ymax0 + dy

            rx, ry, ryaw = self.robot_pose  # if your L* uses map pose, keep it; box is virtually moved
            xf, xb, yr, yl = self._robot_extremes(rx, ry, ryaw)
            Ls_now = self._recompute_Lstar(plan_dir, K, buf, xf, xb, yl, yr,
                                        xmin_v, xmax_v, ymin_v, ymax_v)

            # Early-clear gate: if L* <= 0, stop with brake and finish
            if Ls_now <= 0.0:
                self._stop()
                self._brake_and_settle(settle, hz=30)
                # Apply the *actual* moved distance to real box state once
                self.box['x'] += ax * progress
                self.box['y'] += ay * progress
                self.get_logger().info("[PUSH] Gate clear! ✅")
                return True

            # Target-distance reached → stop with brake and finish
            if dist_left <= 0.0:
                self._stop()
                self._brake_and_settle(settle, hz=30)
                # Clamp to 'cap' in case of mm-scale overshoot
                actual = min(cap, progress)
                self.box['x'] += ax * actual
                self.box['y'] += ay * actual
                self.get_logger().info(f"[PUSH] DONE: moved={actual:.3f} m (target={cap:.3f} m)")
                return True

            # --- Approach control: ramp down near target to avoid overshoot ---
            if dist_left < BRAKE_ZONE:
                # Linear ramp; keep a minimum until the last millimeters
                v_cmd = max(MIN_V, cmd_v * (dist_left / BRAKE_ZONE))
                if dist_left < 0.01:  # last centimeter: allow even slower
                    v_cmd = min(v_cmd, 0.02)
            else:
                v_cmd = cmd_v

            # Push forward
            self._publish_vx(v_cmd)

            # Optional: print for debugging like your logs
            if int(time.time() * 10) % 2 == 0:  # ~5 Hz
                self.get_logger().info(f"[PUSH] Pushing... dist_left={dist_left:.3f} m, L*={Ls_now:.3f} m")
            try:
                    self._viz_selected_lstar(plan_dir, Ls_now)
            except Exception as e:
                self.get_logger().warn(f"[VIZ] live L* publish failed: {e}")

        # Fallback
        self._stop()
        return False
    
    def _retract_from_contact(self, plan_dir: str, backoff_m: float = 0.5) -> bool:
        """
        After a successful push, move *backwards along plan_dir* by 'backoff_m' meters
        so a small clearance is created between robot and the pushed object.
        Assumes robot is still oriented along 'plan_dir'.
        """
        if self.odom_pose is None:
            self.get_logger().warn("[RETRACT] Missing odom pose.")
            return False

        # speeds & timing (no new params required)
        cmd_v   = float(self.get_parameter('cmd_speed').get_parameter_value().double_value)
        settle  = float(self.get_parameter('settle_s').get_parameter_value().double_value)
        v_back  = max(0.05, min(0.25, 0.6 * cmd_v))   # conservative backoff speed
        BRAKE_ZONE = 0.08                              # start ramp-down in last 8 cm

        # axis projection along the *same* plan_dir; going backward means negative body-x
        (ax, ay), _ = self._dir_unit_and_yaw(plan_dir)
        s_start  = ax * self.odom_pose[0] + ay * self.odom_pose[1]
        s_target = s_start - backoff_m                 # move opposite to prior push

        self.get_logger().info(f"[RETRACT] Backing off {backoff_m:.3f} m from contact line")

        # simple distance-bounded loop
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            ox, oy = self.odom_pose[0], self.odom_pose[1]
            s_now  = ax * ox + ay * oy
            dist_left = s_now - s_target  # how much more we need to go backward

            # reached target (we backed up enough)
            if dist_left <= 0.0:
                self._stop()
                self._brake_and_settle(settle, hz=30)
                self.get_logger().info("[RETRACT] Done.")
                return True

            # ramp down near target to avoid overshoot
            if dist_left < BRAKE_ZONE:
                v_cmd = max(0.03, v_back * (dist_left / BRAKE_ZONE))
            else:
                v_cmd = v_back

            # negative vx → move backward in robot frame (orientation unchanged)
            self._publish_vx(-v_cmd)

            # optional light logging (comment out if noisy)
            if int(time.time() * 10) % 3 == 0:
                self.get_logger().info(f"[RETRACT] dist_left={dist_left:.3f} m")

        self._stop()
        return False


    # ---------- run loop ----------
    def run(self):
        # Wait for costmap + amcl
        while rclpy.ok() and (self.grid is None or self.robot_pose is None):
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info("[SYS] Ready (costmap + amcl).")

        # PLAN
        self.get_logger().info("[PLAN] Searching corridor...")
        plan = self._compute_plan()
        if plan is None:
            self.get_logger().warn("[PLAN] No direction free. ABORT.")
            return
        self.get_logger().info(f"[PLAN] Chosen {plan['dir']}, K={plan['K']:.2f}, W={plan['W_dir']:.2f}, L*={plan['L_star']:.2f}, cap={plan['cap']:.2f}")

        # NAV (try chosen first, then cycle others on accept failure / result failure)
        start_idx = self.dir_order.index(plan['dir'])
        for off in range(len(self.dir_order)):
            dtag = self.dir_order[(start_idx + off) % len(self.dir_order)]
            x,y,yaw = self._premanip_pose(dtag)
            self.get_logger().info(f"[NAV] Trying {dtag} pre-manip: ({x:.2f},{y:.2f},{math.degrees(yaw):.1f}deg)")
            goal = self._build_nav_goal(x,y,yaw)

            gh = self._send_and_wait_accept(goal)
            if gh is None:
                self.get_logger().info("[NAV] Acceptance failed. Trying next direction.")
                continue

            status = self._wait_nav_result(gh)
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f"[NAV] Navigation failed (status={status}). Trying next direction.")
                continue

            # (Yaw alignment check removed as requested)
            self.get_logger().info(f"[NAV] Reached pre-manip for {dtag} ✅")
            pre_x, pre_y, pre_yaw = x, y, yaw

            ok_view = self._go_side_peek_obstacle_side(
                plan['dir'],            # chosen push direction
                reach_wait_s=15.0,      # wait to *reach* each side for up to 15 s
                peek_forward=0.25,      # nudge anchor slightly ahead of box face
                extra_buffer_m=0.12     # keep robot outside the corridor edge
            )
            if not ok_view:
                self.get_logger().warn("[SCOUT] Side-peek both points failed → trying next direction.")
                continue
            # Return to pre-manip before starting approach
            goal_back = self._build_nav_goal(pre_x, pre_y, pre_yaw)
            gh_back   = self._send_and_wait_accept(goal_back)
            if gh_back is None or self._wait_nav_result(gh_back) != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn("[NAV] Could not return to pre-manip. Trying next direction.")
                continue
                
            # LOCK final direction and push
            ok, s_origin, cap_from_contact = self._approach_contact(plan['dir'], plan['K'])
            if not ok:
                self.get_logger().warn("[PUSH] Could not confirm contact. Trying next direction.")
                continue
            # Use cap from contact recompute; fallback to plan cap if None
            cap_final = cap_from_contact if cap_from_contact is not None else plan['cap']
            ok = self._do_push_from_contact(plan['dir'], plan['K'], s_origin, cap_final)
            if ok:
                self._retract_from_contact(plan['dir'], backoff_m=0.5)

            if not ok:
                self.get_logger().warn("[PUSH] Push failed. Trying next direction.")
                continue

            return

        self.get_logger().warn("[NAV] All directions failed. ABORT.")


# ---------- main ----------
def main():
    rclpy.init()
    node = PushOrchestrator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node._stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
