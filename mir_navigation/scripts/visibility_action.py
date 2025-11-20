#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional
from threading import Event, Lock
import traceback

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.clock import Clock, ClockType
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker

import tf2_ros
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Vector3,
    TransformStamped,
)

from rclpy.action import ActionServer, ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive

# 👇 apne package / action naam se replace karna
from mir_navigation.action import CheckVisibility


# ---------- Small math helpers ----------
def quat_to_rotm(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def tf_to_matrix(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    r = tf.transform.rotation
    R = quat_to_rotm(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def rotm_to_quat_xyzw(Rm: np.ndarray):
    m = np.array(Rm, dtype=float)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
            qw = (m[2, 1] - m[1, 2]) / s
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
            qw = (m[0, 2] - m[2, 0]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
            qw = (m[1, 0] - m[0, 1]) / s
    return (qx, qy, qz, qw)


def dir_unit_and_yaw(dir_tag: str):
    """Map '+X'/'-X'/'+Y'/'-Y' → (ux,uy,yaw_rad)."""
    if dir_tag == "+X":
        return (1.0, 0.0), 0.0
    if dir_tag == "-X":
        return (-1.0, 0.0), math.pi
    if dir_tag == "+Y":
        return (0.0, 1.0), math.pi / 2.0
    if dir_tag == "-Y":
        return (0.0, -1.0), -math.pi / 2.0
    # fallback
    return (1.0, 0.0), 0.0


# ---------- Node ----------
class VisibilityActionServer(Node):
    """
    Action server:
      Goal  : box pose + size + direction
      Logic : compute L* corridor (K), ROI, then do L* style visibility check
              using MoveIt + depth point cloud
      Result: obstacle_present (True/False).
    """

    def __init__(self):
        super().__init__("visibility_action_server")

        # ---- Obstacle database (local) ----
        self.obstacles = {
            "test_box": {
                "x": 6.5,
                "y": 0.5,
                "length": 0.5,
                "width": 2.0,
            },
            "test_box_1": {
                "x": 4.64,
                "y": 9.15,
                "length": 0.25,
                "width": 0.25,
            },
            # yahan future obstacles add kar sakte ho
        }





        # Work-state
        self._job_active = False
        self._job_cancel = False
        self._job_state = "IDLE"
        self._last_result_blocked: bool = True

        self._tick_count = 0
        self._last_tick_log_ns = 0
        self._cloud_rx_count = 0
        self._last_age_log_ns = 0

        # Reentrant group for service, timer, action
        self.cbgroup = ReentrantCallbackGroup()

        # === Params === (copied / aligned with LStar + push orchestrator)
        self.camera_hfov_deg = float(self.declare_parameter("camera_hfov_deg", 60.0).value)
        self.camera_vfov_deg = float(self.declare_parameter("camera_vfov_deg", 45.0).value)

        self.moveit_group_name = str(self.declare_parameter("moveit_group_name", "ur5_manip").value)
        self.planning_frame = str(self.declare_parameter("planning_frame", "ur_base_link").value)
        self.ee_link = str(self.declare_parameter("ee_link", "ur_tool0").value)
        self.CAMERA_OPTICAL_FRAME = "realsense_depth_optical_frame"

        self.view_z_mode = str(self.declare_parameter("view_z_mode", "use_current_ee_z").value)
        self.view_z_fixed = float(self.declare_parameter("view_z_fixed", 0.90).value)

        # Cloud + ROI params
        self.declare_parameter("cloud_topic", "/realsense/depth/color/points")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("grid_cell_m", 0.10)
        self.declare_parameter("vis_ok_thresh", 0.70)
        self.declare_parameter("vis_min_thresh", 0.20)
        self.declare_parameter("min_obst_pts", 500)
        self.declare_parameter("z_free_max", 0.15)
        self.declare_parameter("z_max_clip", 2.50)
        self.declare_parameter("stale_cloud_sec", 2.5)
        self.declare_parameter("max_points_process", 120000)
        self.declare_parameter("marker_ns", "lstar_roi")

        # L* / corridor related (for K calc)
        self.declare_parameter("robot_length", 0.90)
        self.declare_parameter("robot_width", 0.50)
        self.declare_parameter("buffer_m", 0.10)
        self.declare_parameter("default_lstar_min", 2.0)

        self.cloud_topic = self.get_parameter("cloud_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.grid_cell_m = float(self.get_parameter("grid_cell_m").value)
        self.vis_ok_thresh = float(self.get_parameter("vis_ok_thresh").value)
        self.vis_min_thresh = float(self.get_parameter("vis_min_thresh").value)
        self.min_obst_pts = int(self.get_parameter("min_obst_pts").value)
        self.z_free_max = float(self.get_parameter("z_free_max").value)
        self.z_max_clip = float(self.get_parameter("z_max_clip").value)
        self.stale_cloud_s = float(self.get_parameter("stale_cloud_sec").value)
        self.max_points = int(self.get_parameter("max_points_process").value)
        self.marker_ns = self.get_parameter("marker_ns").get_parameter_value().string_value

        self.robot_L = float(self.get_parameter("robot_length").get_parameter_value().double_value)
        self.robot_W = float(self.get_parameter("robot_width").get_parameter_value().double_value)
        self.buffer_m = float(self.get_parameter("buffer_m").get_parameter_value().double_value)
        self.default_lstar_min = float(self.get_parameter("default_lstar_min").get_parameter_value().double_value)

        # MoveIt
        self.movegroup_ac = ActionClient(self, MoveGroup, "move_action", callback_group=self.cbgroup)

        # CAM<->EE transforms (cached)
        self.T_cam_ee = None  # CAM <- EE
        self.T_ee_cam = None  # EE <- CAM

        # Coverage memory
        self.stop_cov_thresh = self.declare_parameter("stop_cov_thresh", 0.90).value
        self._cov_grid = None
        self._cov_meta = None

        # Obstacle memory
        self._obst_detected = False

        # Cloud subscriber
        self._last_cloud_msg: Optional[PointCloud2] = None
        self._last_cloud_stamp_ns: Optional[int] = None

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.cloud_topic, self._on_cloud, qos_profile=qos_profile, callback_group=self.cbgroup
        )

        # Marker pub
        self.marker_pub = self.create_publisher(Marker, "roi_marker", 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ===== Availability/stability primitives =====
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._job_done_evt: Event = Event()
        self._move_lock = Lock()
        self._move_goal_active = False

        # Action server
        self._action_server = ActionServer(
            self,
            CheckVisibility,
            "check_visibility",   # <- action name
            execute_callback=self.execute_visibility,
            callback_group=self.cbgroup,
        )

        # Cooperative tick (same reentrant group + STEADY clock)
        self._tick = self.create_timer(
            0.05, self._process_job_tick, callback_group=self.cbgroup, clock=self._steady_clock
        )

        self.get_logger().info(
            f"[SYS] visibility_action_server up | cloud={self.cloud_topic} | frame={self.target_frame}"
        )

    # ---------- L* / corridor helpers ----------
    def _compute_lstar_length(self, box_L: float, box_W: float) -> float:
        """
        Simplified K calculation:
        - Enough room = robot length + box length + 2*buffer
        - But at least default_lstar_min
        """
        K = self.robot_L + box_L + 2.0 * self.buffer_m
        return max(K, self.default_lstar_min)

    def _compute_corridor_roi(self, box_x, box_y, box_L, box_W, dir_tag: str, K: float):
        """
        Reuse push-style corridor logic:
        - anchor at obstacle face
        - ROI center at half of K from anchor
        - corridor width = max(robot_dim, box_dim) depending on direction
        """
        (ux, uy), yaw = dir_unit_and_yaw(dir_tag)

        # box center
        box_cx = float(box_x)
        box_cy = float(box_y)

        # anchor on obstacle face along direction
        if dir_tag in ("+X", "-X"):
            anchor_x = box_cx + 0.5 * box_L * ux
            anchor_y = box_cy
            corridor_width = max(self.robot_W, box_W)
        else:
            anchor_x = box_cx
            anchor_y = box_cy + 0.5 * box_W * uy
            corridor_width = max(self.robot_L, box_L)

        # ROI center
        region_cx = anchor_x + 0.5 * K * ux
        region_cy = anchor_y + 0.5 * K * uy

        return region_cx, region_cy, K, corridor_width, yaw

    # ---------- TF / transforms ----------
    def _ensure_cam_ee_tf(self) -> bool:
        if self.T_cam_ee is not None and self.T_ee_cam is not None:
            return True
        try:
            tf_cam_ee = self.tf_buffer.lookup_transform(
                self.CAMERA_OPTICAL_FRAME, self.ee_link, rclpy.time.Time(), timeout=Duration(seconds=1.0)
            )
            self.T_cam_ee = tf_to_matrix(tf_cam_ee)  # CAM <- EE
            self.T_ee_cam = np.linalg.inv(self.T_cam_ee)  # EE <- CAM
            self.get_logger().info("[TF] Cached T_cam_ee and T_ee_cam")
            return True
        except Exception as ex:
            self.get_logger().warn(f"[TF] CAM<->EE TF not ready: {ex}")
            return False

    def _transform_point_xy(self, x, y, from_frame: str, to_frame: str):
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time(), timeout=Duration(seconds=0.75))
            T = tf_to_matrix(tf)
            p = np.array([float(x), float(y), 0.0, 1.0], dtype=np.float32)
            pout = T @ p
            return float(pout[0]), float(pout[1]), float(pout[2])
        except Exception as ex:
            self.get_logger().warn(f"[TF] {from_frame}->{to_frame} transform failed: {ex}")
            return x, y, 0.0

    # ---------- MoveIt: orientation-only helpers ----------
    def _ee_pose_orient_only(self, target_w: np.ndarray, planning_frame: str) -> Optional[PoseStamped]:
        if not self._ensure_cam_ee_tf():
            self.get_logger().error("[VIEW] Missing cam<->ee TF")
            return None

        # Current camera or EE position in planning frame
        try:
            tf_cam_now = self.tf_buffer.lookup_transform(
                planning_frame, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time(), timeout=Duration(seconds=0.75)
            )
            cam_pos_now = np.array(
                [
                    tf_cam_now.transform.translation.x,
                    tf_cam_now.transform.translation.y,
                    tf_cam_now.transform.translation.z,
                ],
                dtype=float,
            )
        except Exception:
            tf_ee_now = self.tf_buffer.lookup_transform(
                planning_frame, self.ee_link, rclpy.time.Time(), timeout=Duration(seconds=0.75)
            )
            cam_pos_now = np.array(
                [
                    tf_ee_now.transform.translation.x,
                    tf_ee_now.transform.translation.y,
                    tf_ee_now.transform.translation.z,
                ],
                dtype=float,
            )

        # Build desired camera frame: +Z toward target; choose orthonormal x,y
        z_cam = target_w - cam_pos_now
        nz = np.linalg.norm(z_cam)
        if nz < 1e-9:
            self.get_logger().warn("[VIEW] target ~= camera position; orient-only undefined.")
            return None
        z_cam /= nz
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z_cam, up))) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        x_cam = np.cross(up, z_cam); x_cam /= (np.linalg.norm(x_cam) + 1e-9)
        y_cam = np.cross(z_cam, x_cam)
        # Empirical axis flip for camera convention
        y_cam = -y_cam
        x_cam = -x_cam

        R_w_cam = np.column_stack((x_cam, y_cam, z_cam))
        U, _, Vt = np.linalg.svd(R_w_cam)
        R_w_cam = U @ Vt
        if np.linalg.det(R_w_cam) < 0.0:
            U[:, -1] *= -1.0
            R_w_cam = U @ Vt

        # Compose world<-cam and then world<-ee
        T_w_cam = np.eye(4, dtype=float)
        T_w_cam[:3, :3] = R_w_cam
        T_w_cam[:3, 3] = cam_pos_now
        T_w_ee = T_w_cam @ self.T_cam_ee  # (W<-CAM) * (CAM<-EE) = W<-EE

        ps = PoseStamped()
        ps.header.frame_id = planning_frame
        ps.pose.position.x = float(T_w_ee[0, 3])
        ps.pose.position.y = float(T_w_ee[1, 3])
        ps.pose.position.z = float(T_w_ee[2, 3])

        qx, qy, qz, qw = rotm_to_quat_xyzw(T_w_ee[:3, :3])
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        return ps

    def _build_orient_only_goal_from_pose(self, pose_stamped: PoseStamped) -> MoveGroup.Goal:
        oc = OrientationConstraint()
        oc.header = pose_stamped.header
        oc.link_name = self.ee_link
        oc.orientation = pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = math.radians(5.0)
        oc.absolute_y_axis_tolerance = math.radians(5.0)
        oc.absolute_z_axis_tolerance = math.radians(5.0)
        oc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.02]  # 2 cm position lock radius

        center = Pose()
        center.position = pose_stamped.pose.position
        center.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [center]

        pc = PositionConstraint()
        pc.header = pose_stamped.header
        pc.link_name = self.ee_link
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        pc.constraint_region = bv
        pc.weight = 1.0

        cons = Constraints()
        cons.orientation_constraints = [oc]
        cons.position_constraints = [pc]

        req = MotionPlanRequest()
        req.group_name = self.moveit_group_name
        req.goal_constraints = [cons]
        req.allowed_planning_time = 2.0
        req.num_planning_attempts = 1

        opts = PlanningOptions()
        opts.plan_only = False
        opts.look_around = False
        opts.replan = False

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = opts
        return goal

    def _build_orient_only_goal_to(self, xy):
        cam_z = 0.0
        px, py, pz = self._transform_point_xy(
            xy[0], xy[1], from_frame=self.target_frame, to_frame=self.planning_frame
        )
        target = np.array([float(px), float(py), cam_z], dtype=float)
        pose_ee = self._ee_pose_orient_only(target, self.planning_frame)
        return self._build_orient_only_goal_from_pose(pose_ee)

    def _schedule_orient_to_point(self, xy):
        self.get_logger().info(f"[VIEW] scheduling next look-at xy=({float(xy[0]):.2f}, {float(xy[1]):.2f})")
        goal = self._build_orient_only_goal_to(xy)
        with self._move_lock:
            self._move_future = self.movegroup_ac.send_goal_async(goal)
            self._result_future = None
            self._move_goal_active = True
        return True

    def _moveit_motion_done(self) -> bool:
        with self._move_lock:
            if not self._move_goal_active:
                return False

            mf = getattr(self, "_move_future", None)
            rf = getattr(self, "_result_future", None)

            if mf is not None and not mf.done():
                return False

            if mf is not None and mf.done() and rf is None:
                goal_handle = mf.result()
                if not goal_handle.accepted:
                    self.get_logger().warn("[MOVE] Goal rejected")
                    self._move_future = None
                    self._result_future = None
                    self._move_goal_active = False
                    return True
                self._result_future = goal_handle.get_result_async()
                self.get_logger().info("[MOVE] Goal accepted; waiting for result...")
                return False

            rf = getattr(self, "_result_future", None)
            if rf is not None and not rf.done():
                return False

            if rf is not None and rf.done():
                res = rf.result()
                ok = (res is not None) and (res.result.error_code.val == 1)
                self.get_logger().info(f"[MOVE] Done: {'OK' if ok else 'FAIL'}")
                self._move_future = None
                self._result_future = None
                self._move_goal_active = False
                return True

            return False

    # ---------- Cloud ----------
    def _on_cloud(self, msg: PointCloud2):
        self._last_cloud_msg = msg
        try:
            t = rclpy.time.Time.from_msg(msg.header.stamp)
            self._last_cloud_stamp_ns = t.nanoseconds
        except Exception:
            self._last_cloud_stamp_ns = self.get_clock().now().nanoseconds
        self._cloud_rx_count += 1
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_age_log_ns > 1e9:
            self._last_age_log_ns = now_ns

    def _is_cloud_fresh(self, max_age_s: float = None) -> bool:
        if max_age_s is None:
            max_age_s = self.stale_cloud_s
        if getattr(self, "_last_cloud_stamp_ns", None) is None:
            return False
        age_ns = self.get_clock().now().nanoseconds - self._last_cloud_stamp_ns
        return age_ns <= int(max_age_s * 1e9)

    def _get_cloud_in_map_nonblocking(self) -> Optional[np.ndarray]:
        msg = self._last_cloud_msg
        if msg is None:
            return None
        if not self._is_cloud_fresh(self.stale_cloud_s):
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.75)
            )
        except Exception:
            self.get_logger().warn(f"[TF] lookup failed: {self.target_frame}<-{msg.header.frame_id}")
            return None
        T = tf_to_matrix(tf)
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.fromiter(gen, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if pts.shape[0] > self.max_points:
            idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]
        P = np.column_stack((pts["x"], pts["y"], pts["z"])).astype(np.float32)
        Rm, t = T[:3, :3], T[:3, 3]
        Pm = (P @ Rm.T) + t
        if self.z_max_clip is not None:
            Pm = Pm[Pm[:, 2] <= self.z_max_clip]
        return Pm

    # ---------- Coverage / ROI ----------
    def _eval_roi(self, P_map, cx, cy, L, W, yaw_rad):
        """
        Returns roi_pts, observed_cells, vis_ratio, obst_pts, (xp, yp, inside, grid)
        Obstacle = z > z_free_max (and outside box footprint)
        """
        fp_margin = 0.15
        cell = max(1e-3, float(self.grid_cell_m))
        grid_nx = max(1, int(math.ceil(float(L) / cell)))
        grid_ny = max(1, int(math.ceil(float(W) / cell)))

        if P_map is None or P_map.shape[0] == 0:
            self.get_logger().warn("[FAILURE] No valid points found in transformed cloud.")
            return 0, 0, 0.0, 0, None

        # ROI marker
        self._publish_roi_marker(cx, cy, L, W, yaw_rad, color=(0.9, 0.8, 0.1, 0.35))

        # World -> ROI-local
        dx = P_map[:, 0] - float(cx)
        dy = P_map[:, 1] - float(cy)
        c = np.cos(-yaw_rad)
        s = np.sin(-yaw_rad)
        xp = c * dx - s * dy
        yp = s * dx + c * dy

        halfL = 0.5 * float(L)
        halfW = 0.5 * float(W)
        inside_roi = (np.abs(xp) < halfL) & (np.abs(yp) < halfW)

        # Exclude box footprint (map-aligned)
        try:
            box_cx, box_cy, box_L, box_W = self._box_fp
        except Exception:
            box_cx = box_cy = 0.0
            box_L = box_W = 0.0
        hx = 0.5 * float(box_L) + fp_margin
        hy = 0.5 * float(box_W) + fp_margin
        inside_fp = (np.abs(P_map[:, 0] - box_cx) <= hx) & (np.abs(P_map[:, 1] - box_cy) <= hy)
        inside = inside_roi & (~(inside_roi & inside_fp))

        # Debug point markers
        self._publish_inside_points(P_map, inside, color=(0.0, 0.5, 1.0, 0.9), max_points=5000)

        # Obstacle points
        obst_mask = inside & (P_map[:, 2] > float(self.z_free_max))
        obst_pts = int(np.count_nonzero(obst_mask))
        self._publish_obstacle_points(P_map, inside, self.z_free_max, color=(1.0, 0.0, 0.0, 1.0), max_points=5000)

        # Visibility grid
        roi_pts = int(np.count_nonzero(inside))
        grid = np.zeros((grid_nx, grid_ny), dtype=bool)
        observed_cells = 0
        vis_ratio = 0.0
        X = xp[inside]
        Y = yp[inside]
        if X.size > 0:
            bx = np.clip(((X + halfL) / float(L)) * grid_nx, 0, grid_nx - 1).astype(int)
            by = np.clip(((Y + halfW) / (2.0 * halfW)) * grid_ny, 0, grid_ny - 1).astype(int)
            grid[bx, by] = True
            observed_cells = int(grid.sum())
            vis_ratio = float(observed_cells) / float(grid_nx * grid_ny)

        return roi_pts, observed_cells, vis_ratio, obst_pts, (xp, yp, inside, grid)

    def _roi_local_to_world(self, cx, cy, yaw, lx, ly):
        xw = cx + lx * math.cos(yaw) - ly * math.sin(yaw)
        yw = cy + lx * math.sin(yaw) + ly * math.cos(yaw)
        return (xw, yw)

    def _compute_roi_coverage(self, Pm):
        """
        Returns:
          coverage_sim: float in [0,1]
          next_xy_map:  np.array([x,y]) or None
          obst_pts:     int (obstacle points inside ROI excluding footprint)
        """
        if Pm is None or Pm.size == 0:
            if getattr(self, "_cov_grid", None) is not None:
                cov = float(self._cov_grid.sum()) / float(self._cov_grid.size)
                return cov, None, 0
            return 0.0, None, 0

        cx, cy, L, W = self._roi_rect
        yaw_rad = self._roi_yaw

        roi_pts, observed_cells, vis_ratio, obst_pts, dbg = self._eval_roi(Pm, cx, cy, L, W, yaw_rad)
        grid_frame = dbg[3] if dbg is not None else None
        if grid_frame is None:
            self.get_logger().warn("[COV] eval_roi returned no grid")

        if getattr(self, "_cov_grid", None) is None:
            self._cov_grid = grid_frame.copy()
        else:
            self._cov_grid = np.logical_or(self._cov_grid, grid_frame)

        nx, ny = self._cov_grid.shape
        dx = float(L) / nx
        dy = float(W) / ny
        xc = (np.arange(nx) + 0.5) * dx - 0.5 * float(L)
        yc = (np.arange(ny) + 0.5) * dy - 0.5 * float(W)
        Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

        # Valid mask (exclude box footprint area in ROI-local)
        valid_mask = np.ones_like(self._cov_grid, dtype=bool)
        try:
            box_cx_map, box_cy_map, box_L, box_W = self._box_fp
            dx_b, dy_b = (box_cx_map - cx), (box_cy_map - cy)
            c_y, s_y = math.cos(-yaw_rad), math.sin(-yaw_rad)
            bx = c_y * dx_b - s_y * dy_b
            by = s_y * dx_b + c_y * dy_b
            mx = 0.5 * float(box_L) + 1e-3
            my = 0.5 * float(box_W) + 1e-3
            fp_mask = (np.abs(Xc - bx) <= mx) & (np.abs(Yc - by) <= my)
            valid_mask &= ~fp_mask
        except Exception:
            pass

        cov_grid_valid = self._cov_grid.copy()
        cov_grid_valid[~valid_mask] = False
        valid_total = int(valid_mask.sum())
        if valid_total > 0:
            observed_cells_cum = int(cov_grid_valid.sum())
            coverage_sim = observed_cells_cum / float(valid_total)
            self.get_logger().info(f"[COV_VALID] cum={coverage_sim*100:.1f}% ({observed_cells_cum}/{valid_total})")
        else:
            self.get_logger().warn("[COV] valid_total=0; check ROI/footprint.")
            coverage_sim = 0.0

        uncovered_valid = valid_mask & (~self._cov_grid)
        if uncovered_valid.any():
            lx = float(Xc[uncovered_valid].mean())
            ly = float(Yc[uncovered_valid].mean())
            next_xy_map = self._roi_local_to_world(cx, cy, yaw_rad, lx, ly)
            next_xy_map = np.asarray(next_xy_map[:2], dtype=float)
            self.get_logger().info(
                f"[UNCOVERED_CENTER] MAP({next_xy_map[0]:.3f}, {next_xy_map[1]:.3f}) | ROI_LOCAL({lx:.3f}, {ly:.3f})"
            )
        else:
            next_xy_map = None

        return float(coverage_sim), next_xy_map, int(obst_pts)

    # ---------- Markers ----------
    def _publish_roi_marker(self, cx, cy, L, W, yaw_rad, color=(0.9, 0.8, 0.1, 0.35)):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD

        pose = Pose()
        pose.position.x = float(cx)
        pose.position.y = float(cy)
        pose.position.z = 0.025
        cyaw, syaw = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
        pose.orientation.z = syaw
        pose.orientation.w = cyaw

        m.pose = pose
        m.scale.x = float(L)
        m.scale.y = float(W)
        m.scale.z = 0.05

        m.color.r, m.color.g, m.color.b, m.color.a = color
        self.marker_pub.publish(m)

    def _publish_inside_points(self, P_map, inside, color=(0.1, 0.6, 1.0, 0.9), max_points=5000):
        from geometry_msgs.msg import Point
        pts = P_map[inside]
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 2
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.scale.y = 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self.marker_pub.publish(m)

    def _publish_obstacle_points(self, P_map, inside, z_thresh, color=(1.0, 0.0, 0.0, 1.0), max_points=5000):
        from geometry_msgs.msg import Point
        obst_mask = inside & (P_map[:, 2] > float(z_thresh))
        pts = P_map[obst_mask]
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 3
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.025
        m.scale.y = 0.025
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self.marker_pub.publish(m)

    def _make_roi_marker(self, Pm, coverage):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 99
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.z = 0.16
        m.color.a = 1.0
        m.color.r = 0.1
        m.color.g = 0.9
        m.color.b = 0.2
        m.text = f"Coverage: {coverage*100:.1f}%"
        try:
            cx, cy, _, _ = self._roi_rect
            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = 0.4
        except Exception:
            pass
        return m

    # ---------- Job lifecycle ----------
    def _reset_motion_handles(self):
        with self._move_lock:
            try:
                mf = getattr(self, "_move_future", None)
                if mf is not None:
                    self.get_logger().info("[MOVE] Resetting previous motion handles")
                    gh = mf.result() if mf.done() else None
                    if gh is not None and hasattr(gh, "cancel_goal_async"):
                        gh.cancel_goal_async()
            except Exception:
                self.get_logger().warn("[MOVE] Exception while cancelling previous goal")
            self._move_future = None
            self._result_future = None
            self._move_goal_active = False

    def _finish_success(self, coverage: float):
        # DECISION: obstacle present if seen OR insufficient coverage
        blocked = bool(self._obst_detected) or (coverage < float(self.vis_ok_thresh))
        self._last_result_blocked = blocked
        self.get_logger().info(
            f"[RESULT] coverage={coverage*100:.1f}% obst_detected={self._obst_detected} -> obstacle_present={blocked}"
        )
        self._reset_motion_handles()
        self._job_active = False
        self._job_state = "IDLE"
        try:
            self._job_done_evt.set()
        except Exception:
            pass

    def _finish_failure(self):
        self._last_result_blocked = True
        self.get_logger().warn("[RESULT] failure path -> obstacle_present=True")
        self._reset_motion_handles()
        self._job_active = False
        self._job_state = "IDLE"
        try:
            self._job_done_evt.set()
        except Exception:
            pass

    def _schedule_orient_to_initial_view(self, cx: float, cy: float, yaw: float) -> bool:
        if not self._ensure_cam_ee_tf():
            return False
        target_x_map = float(cx)
        target_y_map = float(cy)
        target_x_pl, target_y_pl, target_z_pl = self._transform_point_xy(
            target_x_map, target_y_map, from_frame=self.target_frame, to_frame=self.planning_frame
        )
        self.get_logger().info(
            f"[VIEW] Target MAP: ({target_x_map:.2f}, {target_y_map:.2f}) "
            f"-> PLANNING: ({target_x_pl:.2f}, {target_y_pl:.2f})"
        )
        target_z_pl = 0.0
        target = np.array([target_x_pl, target_y_pl, target_z_pl], dtype=float)
        self.get_logger().info(f"[VIEW] Direction {math.degrees(yaw):.1f}°, Target {target}")
        pose_ee = self._ee_pose_orient_only(target, self.planning_frame)
        if pose_ee is None:
            self.get_logger().warn("[VIEW] Could not compute EE pose")
            return False
        goal = self._build_orient_only_goal_from_pose(pose_ee)
        with self._move_lock:
            self._move_future = self.movegroup_ac.send_goal_async(goal)
            self._result_future = None
            self._move_goal_active = True
        self.get_logger().info("[MOVE] Initial orient goal sent")
        return True

    def _iterate_once(self):
        if not self._is_cloud_fresh(self.stale_cloud_s):
            age_s = (self.get_clock().now().nanoseconds - (self._last_cloud_stamp_ns or 0)) / 1e9
            self.get_logger().warn(f"[CHECK] No fresh cloud | age={age_s:.3f}s (limit={self.stale_cloud_s:.2f}s)")
            return False, None, None, 0

        Pm = self._get_cloud_in_map_nonblocking()
        cov, centroid, obst_pts = self._compute_roi_coverage(Pm)

        # Update obstacle flag
        if int(obst_pts) >= int(self.min_obst_pts):
            if not self._obst_detected:
                self.get_logger().warn(f"[OBST] Detected {obst_pts} pts (>={self.min_obst_pts}) in ROI.")
            self._obst_detected = True

        self.get_logger().info(
            f"[ITER] coverage={cov*100:.1f}% | obst_pts={obst_pts} | "
            f"next_xy={None if centroid is None else tuple(np.round(centroid, 3))}"
        )
        roi_marker = self._make_roi_marker(Pm, cov)
        self.marker_pub.publish(roi_marker)

        # Early stop if obstacle confirmed and we have minimum confidence
        if self._obst_detected and cov >= float(self.vis_min_thresh):
            self.get_logger().info(
                f"[CHECK] Early stop: obstacle observed and coverage >= vis_min_thresh "
                f"({self.vis_min_thresh*100:.1f}%)"
            )
            return True, cov, None, obst_pts

        if cov >= float(self.stop_cov_thresh) or self._job_cancel:
            self.get_logger().info(f"[CHECK] DONE: coverage={cov*100:.1f}%")
            return True, cov, None, obst_pts

        return True, cov, centroid, obst_pts

    # ---------- State machine tick ----------
    def _process_job_tick(self):
        self._tick_count += 1

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tick_log_ns > 2e9:
            self.get_logger().info(f"[DBG] tick={self._tick_count} state={self._job_state}")
            self._last_tick_log_ns = now_ns

        if not self._job_active:
            return

        if time.monotonic() >= getattr(self, "_job_deadline_monotonic", float("inf")):
            self.get_logger().warn("[JOB] deadline reached (monotonic)")
            self._finish_failure()
            return

        try:
            if self._job_state == "INIT":
                cx, cy, _, _ = self._roi_rect
                ok = self._schedule_orient_to_initial_view(cx, cy, self._roi_yaw)
                self._job_state = "WAIT_INIT_ORIENT" if ok else "FAIL"
                return

            if self._job_state == "WAIT_INIT_ORIENT":
                if self._moveit_motion_done():
                    self._job_state = "ITERATE"
                return

            if self._job_state == "ITERATE":
                step_done, coverage, next_xy, obst_pts = self._iterate_once()
                if not step_done:
                    self.get_logger().warn("[CHECK] Waiting for fresh cloud...")
                    return

                # If early stop condition met
                if next_xy is None:
                    self._finish_success(coverage)
                    return

                self._schedule_orient_to_point(next_xy)
                self._job_state = "WAIT_STEP_ORIENT"
                return

            if self._job_state == "WAIT_STEP_ORIENT":
                if self._moveit_motion_done():
                    self._job_state = "ITERATE"
                return

            if self._job_state == "FAIL":
                self._finish_failure()
                return

        except Exception as e:
            tb = traceback.format_exc()
            self.get_logger().error(f"[JOB] Tick error: {e}\n{tb}")
            self._finish_failure()

    # ---------- Action execute ----------
    def execute_visibility(self, goal_handle):
        """
        Action callback: compute L* corridor + run visibility scan
        Now the client only sends:
          - obstacle_name
          - vis_dir
          - duration_sec
        Box geometry yahan local dictionary se aata hai.
        """
        goal = goal_handle.request

        self.get_logger().info(
            f"[ACT] New visibility goal: obstacle={goal.obstacle_name} "
            f"dir={goal.vis_dir} dur={goal.duration_sec:.1f}s"
        )

        # 0) Look up obstacle geometry locally
        obst_name = (goal.obstacle_name or "").strip()
        if not obst_name:
            self.get_logger().warn("[ACT] Empty obstacle_name in goal, aborting.")
            result = CheckVisibility.Result()
            result.obstacle_present = True
            goal_handle.abort()
            return result

        if obst_name not in self.obstacles:
            self.get_logger().warn(
                f"[ACT] Unknown obstacle_name='{obst_name}' (not in self.obstacles), aborting."
            )
            result = CheckVisibility.Result()
            result.obstacle_present = True
            goal_handle.abort()
            return result

        ob = self.obstacles[obst_name]
        box_x = float(ob["x"])
        box_y = float(ob["y"])
        box_L = float(ob["length"])
        box_W = float(ob["width"])

        self.get_logger().info(
            f"[ACT] Using local obstacle info: "
            f"pos=({box_x:.2f},{box_y:.2f}), size=({box_L:.2f} x {box_W:.2f})"
        )

        # If already busy, reject
        if self._job_active:
            self.get_logger().warn("[ACT] Busy; rejecting new goal.")
            result = CheckVisibility.Result()
            result.obstacle_present = True
            goal_handle.abort()
            return result

        # 1) Compute K (L*)
        K = self._compute_lstar_length(box_L, box_W)
        self.get_logger().info(f"[ACT] Computed L* corridor length K={K:.2f} m")

        # 2) Corridor / ROI geometry from box + dir
        cx, cy, L, W, yaw = self._compute_corridor_roi(
            box_x=box_x,
            box_y=box_y,
            box_L=box_L,
            box_W=box_W,
            dir_tag=goal.vis_dir,
            K=K,
        )
        self.get_logger().info(
            f"[ACT] ROI center=({cx:.2f},{cy:.2f}) L={L:.2f} W={W:.2f} yaw={math.degrees(yaw):.1f}°"
        )

        # 3) Init job state
        self._reset_motion_handles()
        self._last_result_blocked = True
        self._job_done_evt.clear()
        self._obst_detected = False

        self._job_active = True
        self._job_cancel = False
        self._job_state = "INIT"

        self._job_deadline_monotonic = time.monotonic() + float(goal.duration_sec)

        # ROI + box footprint for coverage
        self._roi_rect = (float(cx), float(cy), float(L), float(W))
        self._box_fp = (
            float(box_x),
            float(box_y),
            float(box_L),
            float(box_W),
        )
        self._roi_yaw = float(yaw)

        self._cov_grid = None
        self._cov_meta = (
            float(cx),
            float(cy),
            float(L),
            float(W),
            float(yaw),
            float(self.grid_cell_m),
        )

        self.get_logger().info("[ACT] Job initialized; waiting for completion...")

        # 4) Wait for job completion
        hard_timeout_s = float(goal.duration_sec) + 3.0
        self._job_done_evt.wait(timeout=hard_timeout_s)
        self._job_done_evt.clear()

        if self._job_active:
            self.get_logger().warn("[ACT] Timeout/cancel while job active -> forcing blocked=True")
            self._last_result_blocked = True
            self._job_active = False
            self._job_state = "IDLE"
            self._reset_motion_handles()

        result = CheckVisibility.Result()
        result.obstacle_present = bool(self._last_result_blocked)
        if self._last_result_blocked:
            self.get_logger().info("[ACT] Done: obstacle present")
        else:
            self.get_logger().info("[ACT] Done: area clear")

        return result



# ---------- main ----------
def main():
    rclpy.init()
    node = VisibilityActionServer()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
