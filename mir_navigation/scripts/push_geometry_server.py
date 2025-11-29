#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Push Geometry Server

This node exposes only two services:
  - compute_side_peek_points (ComputeSidePeekPoints)
  - compute_pre_manip_pose (ComputePreManipPose)

It does NOT:
  - send Nav2 goals
  - publish cmd_vel
  - call VisibilityCheck
  - perform any push motion

It only:
  - listens to the global costmap + AMCL (and optionally odom)
  - computes a free corridor (+X/-X/+Y/-Y) around the box
  - returns side-peek poses and pre-manip pose based on that corridor
"""
import time
import math
from typing import Optional, Tuple, Dict

import numpy as np
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
    Minimal geometry server for push planning.

    Responsibilities:
      - Subscribe to global costmap + AMCL + (optionally odom)
      - Maintain robot + box state
      - Implement:
          * _compute_plan()              → choose free corridor (+X/-X/+Y/-Y)
          * _compute_side_peek_targets() → two side-peek poses + dir
          * _premanip_pose()             → pre-manipulation pose
      - Expose results via two ROS2 services.
    """

    def __init__(self) -> None:
        super().__init__("push_geometry_server")

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
        # self.declare_parameter("dir_order", ["+Y", "-Y", "+X", "-X"])

        self.declare_parameter("standoff_m", 0.50)
        self.declare_parameter("contact_gap_m", 0.06)

        self.declare_parameter("box_x", 6.5)
        self.declare_parameter("box_y", 0.5)
        self.declare_parameter("box_length", 0.50)
        self.declare_parameter("box_width", 2.0)

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

        self.box: Dict[str, float] = {
            "x": self.get_parameter("box_x").get_parameter_value().double_value,
            "y": self.get_parameter("box_y").get_parameter_value().double_value,
            "L": self.get_parameter("box_length").get_parameter_value().double_value,
            "W": self.get_parameter("box_width").get_parameter_value().double_value,
        }

        # order in which directions are tested for free corridor
        self.dir_order = list(
            self.get_parameter("dir_order")
            .get_parameter_value()
            .string_array_value
            or ["+X", "-X", "+Y", "-Y"]
        )

        self.get_logger().info("[PUSH_GEOM] Node ready (services + subscriptions up).")

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
    # Costmap utilities
    # ------------------------------------------------------------------
    def _occ_bool(self) -> np.ndarray:
        """Return boolean occupancy grid (True = blocked)."""
        if self.grid is None:
            return None  # type: ignore

        leth = int(
            self.get_parameter("lethal_threshold").get_parameter_value().integer_value
        )
        occ = self.grid >= leth

        if (
            self.get_parameter("treat_unknown_as_blocked")
            .get_parameter_value()
            .bool_value
        ):
            # Unknown in nav2 costmap is usually 255; we also guard <0 just in case
            occ = np.logical_or(occ, (self.grid == 255) | (self.grid < 0))

        if (
            self.get_parameter("debug_corridor")
            .get_parameter_value()
            .bool_value
        ):
            blocked = int(np.count_nonzero(occ))
            total = int(occ.size)
            self.get_logger().info(
                f"[DBG] occ_bool: grid.shape={occ.shape}, blocked={blocked}/{total}, leth={leth}"
            )

        return occ

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _viz_corridor_debug(
        self,
        center: Tuple[float, float],
        yaw: float,
        L: float,
        W: float,
        hit_cells: Optional[list],
    ) -> None:
        """Draw corridor + hit cells in RViz."""
        if not self.get_parameter("viz_enabled").get_parameter_value().bool_value:
            return
        if not self.get_parameter("debug_corridor").get_parameter_value().bool_value:
            return
        if self.grid is None or self.info is None:
            return

        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        base_id = 4000

        # Corridor rectangle
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = now
        m.ns = "corridor_dbg"
        m.id = base_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(center[0])
        m.pose.position.y = float(center[1])
        m.pose.orientation = quat_from_yaw(yaw)
        m.scale.x = float(L)
        m.scale.y = float(W)
        m.scale.z = 0.05
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)
        arr.markers.append(m)

        # Hits, if any
        if hit_cells:
            mid = base_id + 1
            for (hx, hy) in hit_cells:
                hm = Marker()
                hm.header = m.header
                hm.ns = "corridor_hits"
                hm.id = mid
                mid += 1
                hm.type = Marker.SPHERE
                hm.action = Marker.ADD
                hm.pose.position.x = float(hx)
                hm.pose.position.y = float(hy)
                hm.scale.x = hm.scale.y = hm.scale.z = 0.06
                hm.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
                arr.markers.append(hm)

        self.corridor_pub.publish(arr)

    def _viz_clear(self) -> None:
        """Clear markers (if you want to reset RViz displays)."""
        if self.viz_pub is None:
            return
        arr = MarkerArray()
        self.viz_pub.publish(arr)

    def _viz_selected_lstar(self, chosen: str, Ls: float) -> None:
        """Publish simple text marker for selected L*."""
        if not self.get_parameter("viz_enabled").get_parameter_value().bool_value:
            return
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = now
        m.ns = "lstar_selected"
        m.id = 5000
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(self.box["x"])
        m.pose.position.y = float(self.box["y"])
        m.scale.z = 0.2
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        m.text = f"{chosen} L*={Ls:.2f}m"
        arr.markers.append(m)
        self.viz_pub.publish(arr)

    # ------------------------------------------------------------------
    # Geometry helpers for robot + box
    # ------------------------------------------------------------------
    def _robot_extremes(self, rx: float, ry: float, ryaw: float) -> Tuple[float, float, float, float]:
        """
        Returns:
            x_front, x_back, y_right, y_left
        """
        hL, hW = 0.5 * self.robot_L, 0.5 * self.robot_W
        pts = [
            rot_local_to_world(rx, ry, ryaw, sx, sy)
            for sx in (-hL, hL)
            for sy in (-hW, hW)
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_front = max(xs)
        x_back = min(xs)
        y_right = min(ys)
        y_left = max(ys)
        return x_front, x_back, y_right, y_left

    def _box_extents(self) -> Tuple[float, float, float, float]:
        """
        Returns:
            xmin, xmax, ymin, ymax of the box rectangle.
        """
        bl = self.box["L"]
        bw = self.box["W"]
        return (
            self.box["x"] - 0.5 * bl,
            self.box["x"] + 0.5 * bl,
            self.box["y"] - 0.5 * bw,
            self.box["y"] + 0.5 * bw,
        )

    def _dir_unit_and_yaw(self, tag: str) -> Tuple[Tuple[float, float], float]:
        """Map '+X' etc → (unit_vector, yaw)."""
        if tag == "+X":
            return (1.0, 0.0), 0.0
        if tag == "-X":
            return (-1.0, 0.0), math.pi
        if tag == "+Y":
            return (0.0, 1.0), math.pi / 2.0
        if tag == "-Y":
            return (0.0, -1.0), -math.pi / 2.0
        # default
        return (1.0, 0.0), 0.0

    def _recompute_Lstar(
        self,
        tag: str,
        K: float,
        buf: float,
        x_front: float,
        x_back: float,
        y_left: float,
        y_right: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> float:
        if tag == "+X":
            return max(0.0, (x_front + K + buf) - xmin)
        if tag == "-X":
            return max(0.0, xmax - (x_back - K - buf))
        if tag == "+Y":
            return max(0.0, (y_left + K + buf) - ymin)
        if tag == "-Y":
            return max(0.0, ymax - (y_right - K - buf))
        return 0.0

    # ------------------------------------------------------------------
    # Core planning: choose free corridor
    # ------------------------------------------------------------------
    def _rect_free(
        self,
        occ: np.ndarray,
        center: Tuple[float, float],
        yaw: float,
        L: float,
        W: float,
    ) -> bool:
        """
        Check if oriented rectangle is free of occupied cells.

        Steps:
          1) Compute bounding box in world + grid
          2) Iterate cells, test if cell center is inside oriented rect
          3) If any such cell is occupied → blocked
          4) Optionally visualize hits in RViz
        """
        if self.info is None:
            return False

        ox = self.info.origin.position.x
        oy = self.info.origin.position.y
        res = self.info.resolution

        hL, hW = 0.5 * L, 0.5 * W

        # rectangle corners in world coordinates (for bounding box)
        corners = [
            rot_local_to_world(center[0], center[1], yaw, sx, sy)
            for sx in (-hL, hL)
            for sy in (-hW, hW)
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]

        # bounding box in grid coordinates
        r0, c0 = world_to_cell(min(xs), min(ys), ox, oy, res)
        r1, c1 = world_to_cell(max(xs), max(ys), ox, oy, res)

        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(occ.shape[0] - 1, r1)
        c1 = min(occ.shape[1] - 1, c1)

        debug = (
            self.get_parameter("debug_corridor").get_parameter_value().bool_value
        )

        hit_cells = [] if debug else None
        hit = False

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x, y = cell_center((r, c), ox, oy, res)
                if point_in_oriented_rect(x, y, center[0], center[1], yaw, hL, hW):
                    if occ[r, c]:
                        if debug:
                            hit = True
                            hit_cells.append((x, y))  # type: ignore[arg-type]
                        else:
                            return False

        if debug:
            self._viz_corridor_debug(center, yaw, L, W, hit_cells)
            state = "FREE" if not hit_cells else f"BLOCKED ({len(hit_cells)} hits)"
            self.get_logger().info(
                f"[DBG] _rect_free center=({center[0]:.2f},{center[1]:.2f}) "
                f"yaw={math.degrees(yaw):.1f}deg L={L:.2f} W={W:.2f} -> {state}"
            )
            return not hit

        return True

    def _compute_plan(self) -> Optional[Dict]:
        """
        Choose a free corridor direction around the box.

        Returns:
            None  -> costmap / pose missing, or no free corridor
            dict -> {
                'dir':   '+X' / '-X' / '+Y' / '-Y',
                'K':     corridor length,
                'W_dir': corridor width (depends on axis),
                'L_star': usable free length,
                'cap':   L_star + margin,
            }
        """
        if self.grid is None or (self.robot_pose is None and self.odom_pose is None):
            self.get_logger().info(
                f"[DBG_PLAN_INPUT] robot_pose(amcl)={self.robot_pose}, odom_pose={self.odom_pose}, "
                f"box(L,W)=({self.box['L']:.2f},{self.box['W']:.2f}) x,y=({self.box['x']:.2f},{self.box['y']:.2f})"
            )

            return None

        occ = self._occ_bool()
        if occ is None:
            return None

        if self.odom_pose is not None:
            rx, ry, ryaw = self.odom_pose
        else:
            rx, ry, ryaw = self.robot_pose

        x_front, x_back, y_right, y_left = self._robot_extremes(rx, ry, ryaw)
        xmin, xmax, ymin, ymax = self._box_extents()

        K = max(1.5 * self.robot_L, 2.0)
        W_fb = max(self.robot_W, self.box["W"])
        W_lr = max(self.robot_L, self.box["L"])

        def corridor_ok(tag: str) -> bool:
            if tag == "+X":
                center = (xmax + 0.5 * K, self.box["y"])
                yaw = 0.0
                W = W_fb
            elif tag == "-X":
                center = (xmin - 0.5 * K, self.box["y"])
                yaw = math.pi
                W = W_fb
            elif tag == "+Y":
                center = (self.box["x"], ymax + 0.5 * K)
                yaw = math.pi / 2.0
                W = W_lr
            else:  # "-Y"
                center = (self.box["x"], ymin - 0.5 * K)
                yaw = -math.pi / 2.0
                W = W_lr
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

        buf = float(self.get_parameter("buffer_m").get_parameter_value().double_value)
        Ls = self._recompute_Lstar(
            chosen, K, buf, x_front, x_back, y_left, y_right, xmin, xmax, ymin, ymax
        )
        cap = Ls + 0.20
        W_dir = W_fb if chosen in ["+X", "-X"] else W_lr

        try:
            self._viz_selected_lstar(chosen, Ls)
        except Exception as e:
            self.get_logger().warn(f"[VIZ] selected L* publish failed: {e}")

        return {"dir": chosen, "K": K, "W_dir": W_dir, "L_star": Ls, "cap": cap}
    


    def _wait_for_costmap_and_pose(self, timeout_sec: float = 1.0, sleep_sec: float = 0.05) -> bool:
        """
        Wait until costmap AND (AMCL pose OR odom pose) is available.
        Returns True if available within timeout, else False.
        """
        t0 = time.time()
        while (time.time() - t0) < timeout_sec:
            if self.grid is not None and (self.robot_pose is not None or self.odom_pose is not None):
                return True
            time.sleep(sleep_sec)
        return False



    # ------------------------------------------------------------------
    # Side-peek targets using _compute_plan
    # ------------------------------------------------------------------
    def _compute_side_peek_targets(
        self,
        box_x: float,
        box_y: float,
        box_l: float,
        box_w: float,
        peek_forward: float = 0.25,
        extra_buffer_m: float = 0.5,
    ):
        """
        Use global costmap + box info to compute, for **all free corridors**:
          - corridor dir  (+X / -X / +Y / -Y)
          - LEFT / RIGHT side-peek poses beside the box.

        Returns:
            None  -> no costmap/pose or no free corridor in any direction
            list[dict] -> each dict = {
                'dir': str,
                'left':  (x, y, yaw),
                'right': (x, y, yaw),
                'l_star': float,
            }
        """

        # --- wait for costmap + pose ---
        wait_sec = 5.0  # agar chaho to param bana sakte ho
        if not self._wait_for_costmap_and_pose(timeout_sec=wait_sec):
            self.get_logger().warn(
                f"[SIDE_PEEK] No costmap/pose after waiting {wait_sec:.2f}s → cannot compute side-peek targets."
            )
            return None

        if self.robot_pose is None and self.odom_pose is not None:
            self.get_logger().warn("[SIDE_PEEK] AMCL pose missing, using ODOM fallback.")

        # --- update box state (so geometry helpers same data use karein) ---
        self.box["x"] = float(box_x)
        self.box["y"] = float(box_y)
        self.box["L"] = float(box_l)
        self.box["W"] = float(box_w)

        # --- occupancy + robot / box extents ---
        occ = self._occ_bool()
        if occ is None:
            return None

        if self.odom_pose is not None:
            rx, ry, ryaw = self.odom_pose
        else:
            rx, ry, ryaw = self.robot_pose

        x_front, x_back, y_right, y_left = self._robot_extremes(rx, ry, ryaw)
        xmin, xmax, ymin, ymax = self._box_extents()

        # corridor length + width (same logic as _compute_plan)
        K = max(1.5 * self.robot_L, 2.0)
        W_fb = max(self.robot_W, self.box["W"])
        W_lr = max(self.robot_L, self.box["L"])

        def corridor_ok(tag: str):
            """Return (center, yaw, W_dir) if corridor free, else None."""
            if tag == "+X":
                center = (xmax + 0.5 * K, self.box["y"])
                yaw = 0.0
                W = W_fb
            elif tag == "-X":
                center = (xmin - 0.5 * K, self.box["y"])
                yaw = math.pi
                W = W_fb
            elif tag == "+Y":
                center = (self.box["x"], ymax + 0.5 * K)
                yaw = math.pi / 2.0
                W = W_lr
            elif tag == "-Y":
                center = (self.box["x"], ymin - 0.5 * K)
                yaw = -math.pi / 2.0
                W = W_lr
            else:
                return None

            if self._rect_free(occ, center, yaw, K, W):
                return center, yaw, W
            return None

        buf_param = float(self.get_parameter("buffer_m").get_parameter_value().double_value)
        targets = []

        # dir_order param se direction ordering aa rahi hai (+X, -X, +Y, -Y ...)
        for d in self.dir_order:
            self.get_logger().info(f"[PLAN] Testing corridor {d} ...")
            info = corridor_ok(d)
            if info is None:
                self.get_logger().info(f"[PLAN] Corridor {d} BLOCKED ❌")
                continue

            center, yaw_dir, W_dir = info
            self.get_logger().info(f"[PLAN] Corridor {d} FREE ✅")

            # usable free length L* (same helper as _compute_plan)
            L_star = self._recompute_Lstar(
                d, K, buf_param,
                x_front, x_back, y_left, y_right,
                xmin, xmax, ymin, ymax,
            )
            if L_star <= 0.0:
                self.get_logger().info(
                    f"[PLAN] Corridor {d} has non-positive L* ({L_star:.3f}) → skipping."
                )
                continue

            # basis vectors for this dir
            (_, _), yaw_dir = self._dir_unit_and_yaw(d)
            fx, fy = math.cos(yaw_dir), math.sin(yaw_dir)   # forward (push direction)
            ux, uy = -math.sin(yaw_dir), math.cos(yaw_dir)  # left (perpendicular)

            # anchor point near obstacle (box center pushed slightly along push axis)
            bx, by = box_x, box_y
            ax = bx + peek_forward * fx
            ay = by + peek_forward * fy

            # corridor width depends on axis
            if d in ("+X", "-X"):
                corridor_w = max(self.robot_W, box_w)
            else:
                corridor_w = max(self.robot_L, box_l)

            buf_eff = extra_buffer_m if (extra_buffer_m is not None) else buf_param
            offset = 0.5 * corridor_w + 0.5 * self.robot_W + buf_eff

            left = (ax + offset * ux, ay + offset * uy, yaw_dir)
            right = (ax - offset * ux, ay - offset * uy, yaw_dir)

            self.get_logger().info(
                f"[SIDE_PEEK] dir={d}, "
                f"L*={L_star:.3f} m, "
                f"left=({left[0]:.2f},{left[1]:.2f},{math.degrees(left[2]):.1f}deg), "
                f"right=({right[0]:.2f},{right[1]:.2f},{math.degrees(right[2]):.1f}deg)"
            )

            targets.append(
                {
                    "dir": d,
                    "left": left,
                    "right": right,
                    "l_star": float(L_star),
                }
            )

        if not targets:
            self.get_logger().warn(
                "[SIDE_PEEK] No free corridor in any direction → cannot compute side-peek targets."
            )
            return None

        return targets




    # ------------------------------------------------------------------
    # Pre-manip pose (pure geometry)
    # ------------------------------------------------------------------
    def _premanip_pose(self, dtag: str) -> Tuple[float, float, float]:
        """
        Compute pre-manip pose for the selected corridor direction.

        Uses parameters:
          - standoff_m
          - contact_gap_m
          - robot_length
          - box geometry
        """
        s = float(self.get_parameter("standoff_m").get_parameter_value().double_value)
        gap = float(self.get_parameter("contact_gap_m").get_parameter_value().double_value)
        d = s + 0.5 * self.robot_L + gap

        xmin, xmax, ymin, ymax = self._box_extents()
        bx, by = self.box["x"], self.box["y"]

        if dtag == "+X":
            return (xmin - d, by, 0.0)
        if dtag == "-X":
            return (xmax + d, by, math.pi)
        if dtag == "+Y":
            return (bx, ymin - d, math.pi / 2.0)
        # "-Y"
        return (bx, ymax + d, -math.pi / 2.0)

    # ------------------------------------------------------------------
    # Service callbacks (public API)
    # ------------------------------------------------------------------
    def _handle_compute_side_peek_points(self, request, response):
        """
        Planner → box info.
        Node   → returns ALL valid side-peek poses for all free dirs.
        """
        result_list = self._compute_side_peek_targets(
            box_x=request.box_x,
            box_y=request.box_y,
            box_l=request.box_l,
            box_w=request.box_w,
        )

        if not result_list:
            self.get_logger().warn("[SIDE_PEEK] Could not compute any side-peek targets.")
            response.success = False
            response.n = 0
            return response

        n = len(result_list)
        self.get_logger().info(f"[SIDE_PEEK] Computed {n} side-peek dir candidates.")

        response.success = True
        response.n = n

        # Arrays clear / fill (ROS will init empty arrays by default)
        response.dirs = []
        response.left_x = []
        response.left_y = []
        response.left_yaw = []
        response.right_x = []
        response.right_y = []
        response.right_yaw = []
        response.l_star = []

        for t in result_list:
            response.dirs.append(t["dir"])

            lx, ly, lyaw = t["left"]
            rx, ry, ryaw = t["right"]

            response.left_x.append(float(lx))
            response.left_y.append(float(ly))
            response.left_yaw.append(float(lyaw))

            response.right_x.append(float(rx))
            response.right_y.append(float(ry))
            response.right_yaw.append(float(ryaw))

            response.l_star.append(float(t["l_star"]))

        return response

    def _handle_compute_pre_manip_pose(self, request, response):
        """
        Planner → box_x, box_y, box_l, box_w, dir.
        Node   → uses _premanip_pose to compute pre-manip pose.
        """
        # self.box["x"] = float(request.box_x)
        # self.box["y"] = float(request.box_y)
        # self.box["l"] = float(request.box_l)
        # self.box["w"] = float(request.box_w)
        self.box["x"] = float(request.box_x)
        self.box["y"] = float(request.box_y)
        self.box["L"] = float(request.box_l)
        self.box["W"] = float(request.box_w)


        dtag = request.dir  # "+X" / "-X" / "+Y" / "-Y"

        try:
            px, py, pyaw = self._premanip_pose(dtag)
        except Exception as e:
            self.get_logger().warn(
                f"[PRE_MANIP] Failed to compute pre-manip pose: {e}"
            )
            response.success = False
            return response

        self.get_logger().info(
            f"[PRE_MANIP] dir={dtag}, pose=({px:.2f},{py:.2f},{math.degrees(pyaw):.1f}deg)"
        )

        response.success = True
        response.pre_x = float(px)
        response.pre_y = float(py)
        response.pre_yaw = float(pyaw)
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
