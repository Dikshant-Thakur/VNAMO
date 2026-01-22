#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Push Geometry Server (DETERMINISTIC VERSION)

This node exposes only two services:
  - compute_side_peek_points (ComputeSidePeekPoints)
  - compute_pre_manip_pose (ComputePreManipPose)

DETERMINISTIC MODE:
  - Uses hardcoded lookup table for obstacle_name → direction, pre-manip pose, side-peek pose
  - No dynamic corridor calculation
  - Planner sends obstacle_name, service returns fixed values
"""
import time
import math
from typing import Optional, Tuple, Dict

import numpy as np
from shapely import box
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA

from mir_navigation.srv import ComputeSidePeekPoints, ComputePreManipPose


# ---------- tiny helpers ----------

def yaw_from_quat(q: Quaternion) -> float:
    s = 2.0 * (q.w * q.z + q.x * q.y)
    c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(s, c)


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


def rot_local_to_world(cx: float, cy: float, yaw: float, lx: float, ly: float) -> Tuple[float, float]:
    """Rotate + translate a local (lx, ly) around (cx, cy, yaw) → world (x, y)."""
    return (
        cx + lx * math.cos(yaw) - ly * math.sin(yaw),
        cy + lx * math.sin(yaw) + ly * math.cos(yaw),
    )


def point_in_oriented_rect(
    px: float,
    py: float,
    cx: float,
    cy: float,
    yaw: float,
    half_L: float,
    half_W: float,
) -> bool:
    """Check if (px,py) lies in rectangle centered at (cx,cy,yaw) with size (2*half_L,2*half_W)."""
    dx, dy = px - cx, py - cy
    lx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (-half_L <= lx <= half_L) and (-half_W <= ly <= half_W)


def world_to_cell(x: float, y: float, ox: float, oy: float, res: float) -> Tuple[int, int]:
    """Convert world (x,y) to (row, col) index in the grid."""
    r = int(math.floor((y - oy) / res))
    c = int(math.floor((x - ox) / res))
    return r, c


def cell_center(rc: Tuple[int, int], ox: float, oy: float, res: float) -> Tuple[float, float]:
    r, c = rc
    return (ox + (c + 0.5) * res, oy + (r + 0.5) * res)


class PushGeometryServer(Node):
    """
    Deterministic geometry server for push planning.

    Uses hardcoded lookup table:
      obstacle_name → direction, pre-manip pose, side-peek pose
    """

    def __init__(self) -> None:
        super().__init__("push_geometry_server")

        # ============================================================
        # DETERMINISTIC LOOKUP TABLE (HARDCODED)
        # ============================================================
        # Format:
        #   obstacle_name: {
        #       "dir": "+X" / "+Y" / "-X" / "-Y",
        #       "pre_manip": (x, y, yaw),
        #       "side_peek": (x, y, yaw),
        #       "l_star": float (usable corridor length)
        #   }
        # ============================================================
        self.deterministic_poses = {
            "unit_box_0": {
                "dir": "+X",
                "pre_manip": (4.8, -0.21, 0.0),           # (x, y, yaw) - yaw=0 for +X
                "side_peek": (4.55, -2.25, 0.0),          # (x, y, yaw) - facing +X
                "l_star": 2.0,                             # usable corridor length (adjust as needed)
            },
            "unit_box_1": {
                "dir": "+X",
                "pre_manip": (3.52, -3.13, 0.0),          # (x, y, yaw) - yaw=0 for +X
                "side_peek": (4.54, -1.775, 0.0),         # (x, y, yaw) - facing +X
                "l_star": 2.0,
            },
            "unit_box": {
                "dir": "+Y",
                "pre_manip": (2.15, -0.44, math.pi / 2.0),  # (x, y, yaw) - yaw=π/2 for +Y
                "side_peek": (4.54, 1.3, math.pi / 2.0),    # (x, y, yaw) - facing +Y
                "l_star": 2.0,
            },
        }
        # ============================================================

        # --- Services ---
        self.side_peek_srv = self.create_service(
            ComputeSidePeekPoints,
            "compute_side_peek_points",
            self._handle_compute_side_peek_points,
        )

        self.pre_manip_srv = self.create_service(
            ComputePreManipPose,
            "compute_pre_manip_pose",
            self._handle_compute_pre_manip_pose,
        )

        # --- Parameters ---
        # Topics
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("amcl_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/diff_cont/odom")

        # Costmap handling
        self.declare_parameter("lethal_threshold", 150)
        self.declare_parameter("treat_unknown_as_blocked", True)

        # Corridor + box + robot geometry
        self.declare_parameter("robot_length", 0.90)
        self.declare_parameter("robot_width", 0.50)
        self.declare_parameter("buffer_m", 0.10)
        self.declare_parameter("dir_order", ["+X", "-X", "+Y", "-Y"])

        self.declare_parameter("standoff_m", 0.50)
        self.declare_parameter("contact_gap_m", 0.06)

        self.declare_parameter("box_x", 6.5)
        self.declare_parameter("box_y", 0.5)
        self.declare_parameter("box_length", 0.50)
        self.declare_parameter("box_width", 2.0)


        self.movable_obstacles = {

            "unit_box": {
                "x": 2.15, "y": 1.05, "length": 3.28, "width": 0.96,
                "marker": {
                    "x": 2.15, "y": 1.05, "z": 0.27,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            },

            "unit_box_0": {
                "x": 4.80, "y": -0.21, "length": 1.0, "width": 3.12,
                "marker": {
                    "x": 4.80, "y": -0.21, "z": 0.33,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            },

            "unit_box_1": {
                "x": 4.79, "y": -3.13, "length": 0.66, "width": 1.21,
                "marker": {
                    "x": 4.79, "y": -3.13, "z": 0.30,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            }
        }

        # Debug / visualization
        self.declare_parameter("debug_corridor", True)
        self.declare_parameter("viz_enabled", True)
        self.declare_parameter("viz_topic", "/lstar_viz")
        self.declare_parameter("corridor_debug_topic", "/corridor_debug")

        # --- Publishers for RViz debugging ---
        viz_topic = (
            self.get_parameter("viz_topic").get_parameter_value().string_value
        )
        self.viz_pub = self.create_publisher(MarkerArray, viz_topic, 10)

        corr_topic = (
            self.get_parameter("corridor_debug_topic").get_parameter_value().string_value
        )
        self.corridor_pub = self.create_publisher(MarkerArray, corr_topic, 10)

        self._viz_last_pub = 0.0  # throttling (if you want later)

        # --- Subscribers ---
        costmap_topic = (
            self.get_parameter("global_costmap_topic").get_parameter_value().string_value
        )
        amcl_topic = self.get_parameter("amcl_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.create_subscription(OccupancyGrid, costmap_topic, self._map_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, amcl_topic, self._amcl_cb, 10
        )
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 50)

        # --- Internal state ---
        self.grid: Optional[np.ndarray] = None
        self.info = None

        self.robot_pose: Optional[Tuple[float, float, float]] = None  # from AMCL
        self.odom_pose: Optional[Tuple[float, float, float]] = None   # from odom (optional)
        self.odom_vx_filt: float = 0.0
        self._ema_alpha: float = 0.3

        self.robot_L = float(
            self.get_parameter("robot_length").get_parameter_value().double_value
        )
        self.robot_W = float(
            self.get_parameter("robot_width").get_parameter_value().double_value
        )


        # order in which directions are tested for free corridor
        self.dir_order = list(
            self.get_parameter("dir_order")
            .get_parameter_value()
            .string_array_value
            or ["-X", "+X", "+Y", "-Y"]
        )

        self.get_logger().info("[PUSH_GEOM] Node ready (DETERMINISTIC MODE - hardcoded poses).")
        self.get_logger().info(f"[PUSH_GEOM] Available obstacles: {list(self.deterministic_poses.keys())}")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.info = msg.info
        self.grid = np.array(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width
        )

    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose
        self.robot_pose = (
            p.position.x,
            p.position.y,
            yaw_from_quat(p.orientation),
        )

    def _odom_cb(self, msg: Odometry) -> None:
        vx = msg.twist.twist.linear.x
        self.odom_vx_filt = (1.0 - self._ema_alpha) * self.odom_vx_filt + self._ema_alpha * vx
        p = msg.pose.pose
        self.odom_pose = (
            p.position.x,
            p.position.y,
            yaw_from_quat(p.orientation),
        )

    # ------------------------------------------------------------------
    # Service callbacks (DETERMINISTIC - hardcoded lookup)
    # ------------------------------------------------------------------
    def _handle_compute_side_peek_points(self, request, response):
        """
        DETERMINISTIC VERSION:
        Planner → obstacle_name
        Node   → returns hardcoded side-peek pose from lookup table
        """
        obstacle_name = request.obstacle_name

        self.get_logger().info(f"[SIDE_PEEK] Received request for obstacle: '{obstacle_name}'")

        # Lookup in deterministic table
        if obstacle_name not in self.deterministic_poses:
            self.get_logger().warn(
                f"[SIDE_PEEK] Unknown obstacle '{obstacle_name}'. "
                f"Available: {list(self.deterministic_poses.keys())}"
            )
            response.success = False
            response.n = 0
            return response

        # Get hardcoded values
        data = self.deterministic_poses[obstacle_name]
        direction = data["dir"]
        side_peek = data["side_peek"]  # (x, y, yaw)
        l_star = data["l_star"]

        self.get_logger().info(
            f"[SIDE_PEEK] DETERMINISTIC lookup: "
            f"obstacle='{obstacle_name}', dir='{direction}', "
            f"side_peek=({side_peek[0]:.2f}, {side_peek[1]:.2f}, {math.degrees(side_peek[2]):.1f}°)"
        )

        # Fill response (single direction, single side-peek pose)
        response.success = True
        response.n = 1

        response.dirs = [direction]

        # For deterministic mode, left and right are same (single pose)
        response.left_x = [float(side_peek[0])]
        response.left_y = [float(side_peek[1])]
        response.left_yaw = [float(side_peek[2])]

        response.right_x = [float(side_peek[0])]
        response.right_y = [float(side_peek[1])]
        response.right_yaw = [float(side_peek[2])]

        response.l_star = [float(l_star)]

        return response

    def _handle_compute_pre_manip_pose(self, request, response):
        """
        DETERMINISTIC VERSION:
        Planner → obstacle_name
        Node   → returns hardcoded pre-manip pose from lookup table
        """
        obstacle_name = request.obstacle_name

        self.get_logger().info(f"[PRE_MANIP] Received request for obstacle: '{obstacle_name}'")

        # Lookup in deterministic table
        if obstacle_name not in self.deterministic_poses:
            self.get_logger().warn(
                f"[PRE_MANIP] Unknown obstacle '{obstacle_name}'. "
                f"Available: {list(self.deterministic_poses.keys())}"
            )
            response.success = False
            return response

        # Get hardcoded values
        data = self.deterministic_poses[obstacle_name]
        pre_manip = data["pre_manip"]  # (x, y, yaw)
        direction = data["dir"]

        self.get_logger().info(
            f"[PRE_MANIP] DETERMINISTIC lookup: "
            f"obstacle='{obstacle_name}', dir='{direction}', "
            f"pre_manip=({pre_manip[0]:.2f}, {pre_manip[1]:.2f}, {math.degrees(pre_manip[2]):.1f}°)"
        )

        response.success = True
        response.pre_x = float(pre_manip[0])
        response.pre_y = float(pre_manip[1])
        response.pre_yaw = float(pre_manip[2])

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PushGeometryServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()