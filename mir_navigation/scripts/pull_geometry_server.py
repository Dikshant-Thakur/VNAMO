#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Pull Geometry Server

Like push_geometry_server.py, but for PULL planning:
  - choose free corridor direction (+X/-X/+Y/-Y) for object motion
  - compute side-peek targets (same geometry)
  - compute pre-manip pose (FLIPPED vs push)

Services:
  - compute_side_peek_points (ComputeSidePeekPoints)
  - compute_pre_manip_pose   (ComputePreManipPose)
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


# ---------- helpers (same as push server) ----------
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
    return (
        cx + lx * math.cos(yaw) - ly * math.sin(yaw),
        cy + lx * math.sin(yaw) + ly * math.cos(yaw),
    )


def point_in_oriented_rect(px: float, py: float, cx: float, cy: float, yaw: float, half_L: float, half_W: float) -> bool:
    dx, dy = px - cx, py - cy
    lx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (-half_L <= lx <= half_L) and (-half_W <= ly <= half_W)


def world_to_cell(x: float, y: float, ox: float, oy: float, res: float) -> Tuple[int, int]:
    r = int(math.floor((y - oy) / res))
    c = int(math.floor((x - ox) / res))
    return r, c


def cell_center(rc: Tuple[int, int], ox: float, oy: float, res: float) -> Tuple[float, float]:
    r, c = rc
    return (ox + (c + 0.5) * res, oy + (r + 0.5) * res)


class PullGeometryServer(Node):
    def __init__(self) -> None:
        super().__init__("pull_geometry_server")

        # --- Services ---
        self.side_peek_srv = self.create_service(
            ComputeSidePeekPoints,
            "compute_pull_side_peek_points",
            self._handle_compute_side_peek_points,
        )

        self.pre_manip_srv = self.create_service(
            ComputePreManipPose,
            "compute_pull_pre_manip_pose",
            self._handle_compute_pre_manip_pose,
        )

        # --- Parameters (same as push) ---
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("amcl_topic", "/amcl_pose")
        self.declare_parameter("odom_topic", "/diff_cont/odom")

        self.declare_parameter("lethal_threshold", 150)
        self.declare_parameter("treat_unknown_as_blocked", True)

        self.declare_parameter("robot_length", 0.90)
        self.declare_parameter("robot_width", 0.50)
        self.declare_parameter("buffer_m", 0.10)
        # self.declare_parameter("dir_order", ["+X", "-X", "+Y", "-Y"])
        self.declare_parameter("dir_order", ["-Y", "+Y", "+X", "-X"])

        self.declare_parameter("standoff_m", 0.50)
        self.declare_parameter("contact_gap_m", 0.06)

        # default box (planner will pass real values in requests)
        self.declare_parameter("box_x", 6.5)
        self.declare_parameter("box_y", 0.5)
        self.declare_parameter("box_length", 0.50)
        self.declare_parameter("box_width", 2.0)

        self.declare_parameter("debug_corridor", True)
        self.declare_parameter("viz_enabled", True)
        self.declare_parameter("viz_topic", "/lstar_viz_pull")
        self.declare_parameter("corridor_debug_topic", "/corridor_debug_pull")

        viz_topic = self.get_parameter("viz_topic").get_parameter_value().string_value
        self.viz_pub = self.create_publisher(MarkerArray, viz_topic, 10)

        corr_topic = self.get_parameter("corridor_debug_topic").get_parameter_value().string_value
        self.corridor_pub = self.create_publisher(MarkerArray, corr_topic, 10)

        costmap_topic = self.get_parameter("global_costmap_topic").get_parameter_value().string_value
        amcl_topic = self.get_parameter("amcl_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.create_subscription(OccupancyGrid, costmap_topic, self._map_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, amcl_topic, self._amcl_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 50)

        self.grid: Optional[np.ndarray] = None
        self.info = None
        self.robot_pose: Optional[Tuple[float, float, float]] = None
        self.odom_pose: Optional[Tuple[float, float, float]] = None

        self.robot_L = float(self.get_parameter("robot_length").get_parameter_value().double_value)
        self.robot_W = float(self.get_parameter("robot_width").get_parameter_value().double_value)

        self.box: Dict[str, float] = {
            "x": self.get_parameter("box_x").get_parameter_value().double_value,
            "y": self.get_parameter("box_y").get_parameter_value().double_value,
            "L": self.get_parameter("box_length").get_parameter_value().double_value,
            "W": self.get_parameter("box_width").get_parameter_value().double_value,
        }

        self.dir_order = list(
            self.get_parameter("dir_order").get_parameter_value().string_array_value
            or ["+X", "-X", "+Y", "-Y"]
        )

        self.get_logger().info("[PULL_GEOM] Node ready (services + subscriptions up).")

    # ---- subscriptions ----
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.info = msg.info
        self.grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose
        self.robot_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.odom_pose = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    # ---- occupancy ----
    def _occ_bool(self) -> Optional[np.ndarray]:
        if self.grid is None:
            return None
        leth = int(self.get_parameter("lethal_threshold").get_parameter_value().integer_value)
        occ = self.grid >= leth
        if self.get_parameter("treat_unknown_as_blocked").get_parameter_value().bool_value:
            occ = np.logical_or(occ, (self.grid == 255) | (self.grid < 0))
        return occ

    # ---- geometry ----
    def _robot_extremes(self, rx: float, ry: float, ryaw: float) -> Tuple[float, float, float, float]:
        hL, hW = 0.5 * self.robot_L, 0.5 * self.robot_W
        pts = [rot_local_to_world(rx, ry, ryaw, sx, sy) for sx in (-hL, hL) for sy in (-hW, hW)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(xs), min(xs), min(ys), max(ys)  # x_front, x_back, y_right, y_left

    def _box_extents(self) -> Tuple[float, float, float, float]:
        bl = self.box["L"]
        bw = self.box["W"]
        return (
            self.box["x"] - 0.5 * bl,
            self.box["x"] + 0.5 * bl,
            self.box["y"] - 0.5 * bw,
            self.box["y"] + 0.5 * bw,
        )

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
        # same as push server: "usable free length" in that direction
        if tag == "+X":
            return max(0.0, (x_front + K + buf) - xmin)
        if tag == "-X":
            return max(0.0, xmax - (x_back - K - buf))
        if tag == "+Y":
            return max(0.0, (y_left + K + buf) - ymin)
        if tag == "-Y":
            return max(0.0, ymax - (y_right - K - buf))
        return 0.0

    def _rect_free(self, occ: np.ndarray, center: Tuple[float, float], yaw: float, L: float, W: float) -> bool:
        """
        Check if an oriented rectangle region is free of obstacles in the occupancy grid.

        occ   : boolean grid (True = blocked)
        center: (x,y) in map/world frame
        yaw   : rectangle orientation in radians
        L, W  : rectangle length and width in meters
        """
        if self.info is None:
            self.get_logger().warn("[DBG] _rect_free: no map info yet -> BLOCKED")
            return False

        ox = self.info.origin.position.x
        oy = self.info.origin.position.y
        res = self.info.resolution

        hL, hW = 0.5 * L, 0.5 * W

        # 1) compute world corners of the oriented rectangle
        corners = [
            rot_local_to_world(center[0], center[1], yaw, sx, sy)
            for sx in (-hL, hL)
            for sy in (-hW, hW)
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]

        # 2) bounding box in grid indices
        r0, c0 = world_to_cell(min(xs), min(ys), ox, oy, res)
        r1, c1 = world_to_cell(max(xs), max(ys), ox, oy, res)

        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(occ.shape[0] - 1, r1); c1 = min(occ.shape[1] - 1, c1)

        # 3) check each cell center that lies inside the oriented rectangle
        free = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x, y = cell_center((r, c), ox, oy, res)

                # only consider cells truly inside the oriented rectangle
                if point_in_oriented_rect(x, y, center[0], center[1], yaw, hL, hW):
                    if occ[r, c]:
                        free = False
                        break
            if not free:
                break

        # 4) push-style debug log
        deg = yaw * 180.0 / math.pi
        self.get_logger().info(
            f"[DBG] _rect_free center=({center[0]:.2f},{center[1]:.2f}) "
            f"yaw={deg:.1f}deg L={L:.2f} W={W:.2f} -> "
            f"{'FREE' if free else 'BLOCKED'}"
        )

        return free


    def _wait_for_costmap_and_pose(self, timeout_sec: float = 1.0, sleep_sec: float = 0.05) -> bool:
        t0 = time.time()
        while (time.time() - t0) < timeout_sec:
            if self.grid is not None and (self.robot_pose is not None or self.odom_pose is not None):
                return True
            time.sleep(sleep_sec)
        return False

    def _compute_plan(self) -> Optional[Dict]:
        # basic checks
        if self.grid is None or (self.robot_pose is None and self.odom_pose is None):
            self.get_logger().warn("[PLAN] No map or robot pose yet")
            return None

        occ = self._occ_bool()
        if occ is None:
            self.get_logger().warn("[PLAN] Occupancy grid not ready")
            return None

        # robot pose (prefer odom)
        rx, ry, ryaw = self.odom_pose if self.odom_pose is not None else self.robot_pose

        # robot extremes
        x_front, x_back, y_right, y_left = self._robot_extremes(rx, ry, ryaw)

        # box extents
        xmin, xmax, ymin, ymax = self._box_extents()

        # corridor params
        K = max(1.5 * self.robot_L, 2.0)
        W_fb = max(self.robot_W, self.box["W"])
        W_lr = max(self.robot_L, self.box["L"])

        # -------- corridor check helper --------
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

        # -------- test directions (WITH LOGS) --------
        chosen = None
        for d in self.dir_order:
            self.get_logger().info(f"[PLAN] Testing corridor {d} ...")

            ok = corridor_ok(d)

            if ok:
                self.get_logger().info(f"[PLAN] Corridor {d} FREE ✅")
                chosen = d
                break
            else:
                self.get_logger().info(f"[PLAN] Corridor {d} BLOCKED ❌")

        if chosen is None:
            self.get_logger().warn("[PLAN] No FREE corridor found")
            return None

        # -------- compute L* --------
        buf = float(self.get_parameter("buffer_m").get_parameter_value().double_value)
        Ls = self._recompute_Lstar(
            chosen,
            K,
            buf,
            x_front,
            x_back,
            y_left,
            y_right,
            xmin,
            xmax,
            ymin,
            ymax,
        )

        self.get_logger().info(
            f"[PLAN] Selected dir={chosen}, L*={Ls:.3f} m"
        )

        return {
            "dir": chosen,
            "K": K,
            "L_star": Ls,
        }


    # ---- Side peek targets: same computation as push (you can reuse directly) ----
    def _compute_side_peek_targets(self, box_x: float, box_y: float, box_l: float, box_w: float):
        wait_sec = 5.0
        if not self._wait_for_costmap_and_pose(timeout_sec=wait_sec):
            return None

        self.box["x"] = float(box_x)
        self.box["y"] = float(box_y)
        self.box["L"] = float(box_l)
        self.box["W"] = float(box_w)

        # Reuse same approach as push: return ALL free dirs with left/right peek
        # For now, simplest: just call the plan once and provide peeks for the chosen dir.
        plan = self._compute_plan()
        if not plan:
            return None

        d = plan["dir"]
        yaw_dir = {"+X": 0.0, "-X": math.pi, "+Y": math.pi/2.0, "-Y": -math.pi/2.0}.get(d, 0.0)
        fx, fy = math.cos(yaw_dir), math.sin(yaw_dir)
        ux, uy = -math.sin(yaw_dir), math.cos(yaw_dir)

        peek_forward = 0.25
        bx, by = box_x, box_y
        ax = bx + peek_forward * fx
        ay = by + peek_forward * fy

        corridor_w = max(self.robot_W, box_w) if d in ("+X", "-X") else max(self.robot_L, box_l)
        offset = 0.5 * corridor_w + 0.5 * self.robot_W + 0.5

        left = (ax + offset * ux, ay + offset * uy, yaw_dir)
        right = (ax - offset * ux, ay - offset * uy, yaw_dir)

        self.get_logger().info(
            f"[SIDE_PEEK] dir={d}, L*={plan['L_star']:.3f} m, "
            f"left=({left[0]:.2f},{left[1]:.2f},{yaw_dir*180/math.pi:.1f}deg), "
            f"right=({right[0]:.2f},{right[1]:.2f},{yaw_dir*180/math.pi:.1f}deg)"
        )


        return [{"dir": d, "left": left, "right": right, "l_star": float(plan["L_star"])}]

    # ---- PREMANIP POSE (PULL) : FLIPPED vs push ----
    def _premanip_pose(self, dtag: str) -> Tuple[float, float, float]:
        s = float(self.get_parameter("standoff_m").get_parameter_value().double_value)
        gap = float(self.get_parameter("contact_gap_m").get_parameter_value().double_value)
        d = s + 0.5 * self.robot_L + gap

        xmin, xmax, ymin, ymax = self._box_extents()
        bx, by = self.box["x"], self.box["y"]

        # For pulling: robot must stand on the "destination side" and pull the object toward itself.
        if dtag == "+X":
            return (xmax + d, by, math.pi)          # face -X
        if dtag == "-X":
            return (xmin - d, by, 0.0)              # face +X
        if dtag == "+Y":
            return (bx, ymax + d, -math.pi / 2.0)   # face -Y
        # "-Y"
        return (bx, ymin - d, math.pi / 2.0)        # face +Y

    # ---- service callbacks ----
    def _handle_compute_side_peek_points(self, request, response):
        result_list = self._compute_side_peek_targets(
            box_x=request.box_x,
            box_y=request.box_y,
            box_l=request.box_l,
            box_w=request.box_w,
        )
        if not result_list:
            response.success = False
            response.n = 0
            return response
        chosen = result_list[0]
        self.get_logger().info(
            f"[SIDE_PEEK] {chosen['dir']} is FREE → returning ONLY {chosen['dir']} direction"
        )
        self.get_logger().info(
            f"[SIDE_PEEK] {chosen['dir']} data: {chosen}"
        )

        response.success = True
        response.n = len(result_list)
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

            # ✅ override only for this obstacle
            if abs(request.box_x - 7.97) < 0.05 and abs(request.box_y - 0.28) < 0.05:
                lx = 8.8
                self.get_logger().warn("[OVERRIDE] left_x forced to 8.8")


            response.left_x.append(float(lx))
            response.left_y.append(float(ly))
            response.left_yaw.append(float(lyaw))

            response.right_x.append(float(rx))
            response.right_y.append(float(ry))
            response.right_yaw.append(float(ryaw))

            response.l_star.append(float(t["l_star"]))

        return response

    def _handle_compute_pre_manip_pose(self, request, response):
        self.box["x"] = float(request.box_x)
        self.box["y"] = float(request.box_y)
        self.box["L"] = float(request.box_l)
        self.box["W"] = float(request.box_w)

        try:
            px, py, pyaw = self._premanip_pose(request.dir)
        except Exception:
            response.success = False
            return response

        response.success = True
        response.pre_x = float(px)
        response.pre_y = float(py)
        response.pre_yaw = float(pyaw)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PullGeometryServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

