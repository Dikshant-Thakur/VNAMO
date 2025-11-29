#!/usr/bin/env python3
"""
Normal file converted to Action-based fixed-marker observer.
Removed:
- ArUco detection / camera subscribers
- Nav2 navigation
- /start_push trigger

Added:
- ObserveObstacle ActionServer
- Fixed marker pose params (same as copy file)
- Action result same as copy file

MoveIt rotate-in-place + YOLO + home return kept from normal file.
"""

import math
import time
import threading
from typing import Optional, Tuple
from rclpy.time import Time
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from geometry_msgs.msg import PoseStamped, Pose, Vector3
from std_msgs.msg import String, Bool

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    JointConstraint,
)
from shape_msgs.msg import SolidPrimitive

import yolo_detector as yp

# same as copy file
from mir_navigation.action import ObserveObstacle


class ArucoNavigator(Node):
    # --------------------------- INIT ---------------------------
    def __init__(self) -> None:
        super().__init__("observe_obstacle_action_node")

        # ---------------- State flags ----------------
        self.alignment_active = False
        self.sent_goal = False
        self.armed = False
        self.home_goal_active = False

        # For action bookkeeping
        self._current_goal_handle: Optional["ServerGoalHandle"] = None
        self._observation_done_event: Optional[threading.Event] = None
        self._observation_success: bool = False
        self._observation_error: str = ""
        self.last_label: str = ""


        # Tolerances (keep normal values)
        self.ORI_TOL = 0.17
        self.POS_TOL = 0.10


        # ---------------- Alignment retry config ----------------
        # (ori_tol [rad], pos_tol [m]) – first strict, then more relaxed
        self.alignment_tolerance_levels = [
            (self.ORI_TOL, self.POS_TOL),               # 0.05 rad, 0.01 m
            (3.0 * self.ORI_TOL, 3.0 * self.POS_TOL),   # 0.15 rad, 0.03 m
            (7.0 * self.ORI_TOL, 7.0 * self.POS_TOL),   # 0.35 rad, 0.07 m
        ]
        self.alignment_retry_index = 0
        self._last_alignment_target: Optional[PoseStamped] = None

        # Frames / Move group same as normal
        self.BASE_FRAME = "ur_base_link"
        self.TOOL_LINK = "ur_tool0"
        self.CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"
        self.MOVE_GROUP = "ur5_manip"

        # ---------------- State flags ----------------
        self.alignment_active = False
        self.sent_goal = False
        self.armed = False
        self.home_goal_active = False

        # Action bookkeeping
        self._current_goal_handle = None
        self._observation_done_event: Optional[threading.Event] = None
        self._observation_success = False
        self._observation_error = ""
        self.last_label = ""
        self.yolo_seen = False

        # ---------------- Fixed marker params (same as copy) ----------------
        # self.declare_parameter("marker_xyz", [6.25, 0.5, 0.25])
        self.declare_parameter("marker_xyz", [6.25, 0.6, 0.5])
        self.declare_parameter("marker_q_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("approach_yaw_deg", None)

        m_xyz = self.get_parameter("marker_xyz").value
        m_q = self.get_parameter("marker_q_xyzw").value

        self.marker_pose_map = PoseStamped()
        self.marker_pose_map.header.frame_id = "map"
        self.marker_pose_map.pose.position.x = float(m_xyz[0])
        self.marker_pose_map.pose.position.y = float(m_xyz[1])
        self.marker_pose_map.pose.position.z = float(m_xyz[2])
        self.marker_pose_map.pose.orientation.x = float(m_q[0])
        self.marker_pose_map.pose.orientation.y = float(m_q[1])
        self.marker_pose_map.pose.orientation.z = float(m_q[2])
        self.marker_pose_map.pose.orientation.w = float(m_q[3])

        # ---------------- TF setup (same helper style as normal) ----------------
        self.tf_timeout = Duration(seconds=0.5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_cache = {}
        self._tf_cache_ttl = 0.2

        # ---------------- MoveIt & YOLO ----------------
        self.moveit_action_client = ActionClient(self, MoveGroup, "/move_action")
        self.yolo_node = yp.YoloDetector()
        self.yolo_node.set_enabled(False)

        self.create_subscription(Bool, "/yolo/detected", self._on_yolo_detected, 10)
        self.create_subscription(String, "/yolo/label", self._on_yolo_label, 10)

        # ---------------- Action server ----------------
        self._action_server = ActionServer(
            self,
            ObserveObstacle,
            "observe_obstacle",
            execute_callback=self._execute_observe_obstacle,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
        )

        self.get_logger().info("ObserveObstacle Action server initialized (fixed marker, MoveIt + YOLO).")

    # ============================================================
    #                         ACTION API
    # ============================================================

    def _on_goal(self, goal_request: ObserveObstacle.Goal) -> GoalResponse:
        if self._current_goal_handle is not None:
            self.get_logger().warn("Rejecting ObserveObstacle goal: already running.")
            return GoalResponse.REJECT
        self.get_logger().info("ObserveObstacle goal accepted.")
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        self.get_logger().info("ObserveObstacle cancel requested.")
        return CancelResponse.ACCEPT

    def _execute_observe_obstacle(self, goal_handle):
        self._current_goal_handle = goal_handle
        self._observation_done_event = threading.Event()
        self._observation_success = False
        self._observation_error = ""
        self.last_label = ""
        self.yolo_seen = False

        # reset gates
        self.armed = False
        self.home_goal_active = False
        self.sent_goal = False
        self.alignment_active = False

        obstacle_name = getattr(goal_handle.request, "obstacle_name", "")
        self.get_logger().info(f"[OBS_ACT] Start observation for obstacle='{obstacle_name}'")

        # 1) Wait TF ready
        if not self._wait_for_tf_ready(timeout_sec=5.0):
            msg = "TF not ready, aborting."
            self.get_logger().error(msg)
            self._fail_observation(msg)
        else:
            # 2) Start alignment (normal function)
            try:
                self.align_camera_to_marker_pose(self.marker_pose_map)
            except Exception as e:
                msg = f"Failed to start alignment: {e}"
                self.get_logger().error(msg)
                self._fail_observation(msg)

        # 3) Wait pipeline completion
        self._observation_done_event.wait()

        result = ObserveObstacle.Result()
        result.success = bool(self._observation_success)
        result.label = self.last_label or ""
        if self._observation_success:
            result.message = f"Observation completed successfully. Label='{result.label}'"
            goal_handle.succeed()
        else:
            result.message = self._observation_error or "Observation failed."
            goal_handle.abort()

        self.get_logger().info(
            f"[OBS_ACT] Done | success={result.success} label='{result.label}' msg='{result.message}'"
        )

        self._current_goal_handle = None
        return result

    # ============================================================
    #               NORMAL FILE KE HELPERS (UNCHANGED)
    # ============================================================

    def _normalize(self, v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v, 0.0
        return v / n, n

    def _look_at_cameraZ_worldUp(self, cam_world_p: np.ndarray, target_world_p: np.ndarray) -> np.ndarray:
        f, _ = self._normalize(target_world_p - cam_world_p)
        up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f, up_w)) > 0.99:
            up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(up_w, f); r, _ = self._normalize(r)
        u = np.cross(f, r);   u, _ = self._normalize(u)
        R_w_c = np.column_stack((r, u, f))
        if np.dot(R_w_c[:, 1], up_w) > 0.0:
            R_w_c[:, 0] *= -1.0
            R_w_c[:, 1] *= -1.0
        return R_w_c

    def _yaw_pitch_toolX_quat(self, dx: float, dy: float, dz: float):
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        roll = 0.0
        q = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False).as_quat()
        return q

    def _tf_lookup_cached(self, target: str, source: str):
        key = (target, source)
        now = time.time()
        if key in self._tf_cache:
            ts, tfm = self._tf_cache[key]
            if now - ts < self._tf_cache_ttl:
                return tfm
        tfm = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=self.tf_timeout)
        self._tf_cache[key] = (now, tfm)
        return tfm
    

    def _alignment_goal_response_cb(self, fut):
        """
        First callback after sending MoveIt alignment goal.
        Checks whether the goal was accepted, then attaches the final result callback.
        """
        try:
            goal_handle = fut.result()
        except Exception as e:
            msg = f"Alignment goal response error: {e}"
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return

        # Goal rejected by MoveIt
        if not goal_handle or not goal_handle.accepted:
            msg = "MoveIt alignment goal rejected"
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return

        # Goal accepted → wait for result (this will enter retry logic)
        self.get_logger().info("MoveIt alignment goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._alignment_result_cb)


    def _alignment_result_cb(self, fut):
        try:
            res = fut.result().result
            code = int(res.error_code.val)

            if code == 1:
                # Success → reset retry state and go to YOLO
                self.get_logger().info(
                    "Alignment succeeded → enabling YOLO and arming detection."
                )
                self.alignment_retry_index = 0
                self.sent_goal = False
                self._reset_and_arm_yolo()
                return

            # ------------------ Non-success case ------------------
            self.get_logger().warn(
                f"Alignment attempt {self.alignment_retry_index + 1} failed: error_code={code}"
            )

            max_attempts = len(getattr(self, "alignment_tolerance_levels", []))

            # Can we retry with more relaxed tolerances?
            if (
                self._last_alignment_target is not None
                and max_attempts > 0
                and self.alignment_retry_index + 1 < max_attempts
            ):
                self.alignment_retry_index += 1
                self.sent_goal = False
                ori_tol, pos_tol = self.alignment_tolerance_levels[self.alignment_retry_index]
                self.get_logger().info(
                    "Retrying alignment with relaxed tolerances "
                    f"(attempt {self.alignment_retry_index + 1}/{max_attempts}, "
                    f"ori_tol={ori_tol:.3f} rad, pos_tol={pos_tol:.3f} m)..."
                )
                self._send_moveit_alignment_goal(self._last_alignment_target)
                return

            # --------- All retries used → SOFT FALLBACK ----------
            self.get_logger().warn(
                "All alignment attempts failed "
                f"(last error_code={code}). "
                "Proceeding with soft fallback: YOLO from current pose (no MoveIt alignment)."
            )
            self.alignment_retry_index = 0
            self.sent_goal = False

            # Soft fallback = try observation from current pose
            self._reset_and_arm_yolo()

        except Exception as exc:
            msg = f"Alignment result error: {exc}"
            self.get_logger().error(msg)
            self._fail_observation(msg)


    # ============================================================
    #                     ALIGNMENT (NORMAL)
    # ============================================================


    def _send_moveit_alignment_goal(self, target_pose_stamped: PoseStamped):
        """
        Build and send MoveIt MotionPlanRequest to rotate-in-place at current tool position
        while constraining orientation and a small position tolerance sphere.
        """

        # Pick tolerances for this attempt based on retry index
        try:
            ori_tol, pos_tol = self.alignment_tolerance_levels[self.alignment_retry_index]
        except (AttributeError, IndexError):
            # Fallback to base tolerances if something goes wrong
            ori_tol, pos_tol = self.ORI_TOL, self.POS_TOL

        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        # Make planning a bit more robust
        req.allowed_planning_time = 3.0
        req.num_planning_attempts = 5
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = float(ori_tol)
        oc.absolute_y_axis_tolerance = float(ori_tol)
        oc.absolute_z_axis_tolerance = float(ori_tol)
        oc.weight = 1.0

        # Position constraint (small sphere around current tool position)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(pos_tol)]

        center_pose = Pose()
        center_pose.position = target_pose_stamped.pose.position
        center_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [center_pose]

        pc = PositionConstraint()
        pc.header = target_pose_stamped.header
        pc.link_name = self.TOOL_LINK
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        pc.constraint_region = bv
        pc.weight = 1.0

        goal_constraint = Constraints()
        goal_constraint.orientation_constraints = [oc]
        goal_constraint.position_constraints = [pc]
        req.goal_constraints = [goal_constraint]

        goal_msg = MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False

        self.get_logger().info(
            "Sending MoveIt alignment goal to /move_action "
            f"(attempt {self.alignment_retry_index + 1}, "
            f"ori_tol={ori_tol:.3f} rad, pos_tol={pos_tol:.3f} m)..."
        )
        self.sent_goal = True
        fut = self.moveit_action_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._alignment_goal_response_cb)


    def align_camera_to_marker_pose(self, marker_pose_map: PoseStamped) -> None:
        """
        Compute orientation so that the CAMERA looks at the marker, convert that
        into a TOOL orientation, then send a MoveIt alignment goal.

        - Uses map→base, base→tool, base→camera, (tool→camera) TFs
        - Stores target for retry + soft-fallback pipeline
        """
        if self.sent_goal:
            self.get_logger().warn(
                "Alignment already in progress, ignoring new request."
            )
            return

        if not self._tf_ready():
            raise RuntimeError("TF not ready, cannot plan alignment.")

        # ---------- 1) Refresh marker pose stamp + transform to base ----------
        marker_pose_map_latest = PoseStamped()
        # marker_pose_map_latest.header.stamp = self.get_clock().now().to_msg()
        marker_pose_map_latest.header.stamp = Time().to_msg()
        marker_pose_map_latest.header.frame_id = (
            marker_pose_map.header.frame_id or "map"
        )
        marker_pose_map_latest.pose = marker_pose_map.pose

        marker_pose_base = self.tf_buffer.transform(
            marker_pose_map_latest, self.BASE_FRAME, timeout=self.tf_timeout
        )

        # ---------- 2) Lookup tool + camera in base frame ----------
        t_tool = self._tf_lookup_cached(self.BASE_FRAME, self.TOOL_LINK)
        t_cam = self._tf_lookup_cached(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)

        tool_p = np.array(
            [
                t_tool.transform.translation.x,
                t_tool.transform.translation.y,
                t_tool.transform.translation.z,
            ],
            dtype=np.float64,
        )
        cam_p = np.array(
            [
                t_cam.transform.translation.x,
                t_cam.transform.translation.y,
                t_cam.transform.translation.z,
            ],
            dtype=np.float64,
        )
        marker_p = np.array(
            [
                marker_pose_base.pose.position.x,
                marker_pose_base.pose.position.y,
                marker_pose_base.pose.position.z,
            ],
            dtype=np.float64,
        )

        # ---------- 3) Decide TOOL orientation ----------
        # Try proper camera-based look-at using tool→camera TF
        try:
            t_tool_cam = self._tf_lookup_cached(
                self.TOOL_LINK, self.CAMERA_OPTICAL_FRAME
            )

            # R_tool_cam: rotation from TOOL frame to CAMERA frame
            R_t_c = R.from_quat(
                [
                    t_tool_cam.transform.rotation.x,
                    t_tool_cam.transform.rotation.y,
                    t_tool_cam.transform.rotation.z,
                    t_tool_cam.transform.rotation.w,
                ]
            ).as_matrix()
            # Inverse: CAMERA→TOOL
            R_c_t = R_t_c.T

            # Desired CAMERA orientation in base (Z axis looks towards marker)
            R_w_c_des = self._look_at_cameraZ_worldUp(cam_p, marker_p)

            # Convert desired camera orientation to TOOL orientation in base:
            # R_w_t_des = R_w_c_des * R_c_t
            R_w_t_des = R_w_c_des @ R_c_t

            q_des = R.from_matrix(R_w_t_des).as_quat()
        except Exception as exc:
            # Fallback: only use tool position, make tool +X roughly point to marker
            self.get_logger().warn(
                f"Failed to use tool→camera TF for look-at, "
                f"falling back to simple toolX orientation: {exc}"
            )
            v_tool_to_marker = marker_p - tool_p
            q_des = self._yaw_pitch_toolX_quat(
                v_tool_to_marker[0],
                v_tool_to_marker[1],
                v_tool_to_marker[2],
            )

        # ---------- 4) Log debug info ----------
        self.get_logger().info(
            "Planning MoveIt alignment:\n"
            f"  Marker (base) : [{marker_p[0]:.3f}, {marker_p[1]:.3f}, {marker_p[2]:.3f}]\n"
            f"  Tool  (base)  : [{tool_p[0]:.3f}, {tool_p[1]:.3f}, {tool_p[2]:.3f}]\n"
            f"  Cam   (base)  : [{cam_p[0]:.3f}, {cam_p[1]:.3f}, {cam_p[2]:.3f}]\n"
            f"  Desired quat  : [{q_des[0]:.4f}, {q_des[1]:.4f}, "
            f"{q_des[2]:.4f}, {q_des[3]:.4f}]"
        )

        # ---------- 5) Build target pose in BASE frame ----------
        target_pose_base = PoseStamped()
        target_pose_base.header.stamp = self.get_clock().now().to_msg()
        target_pose_base.header.frame_id = self.BASE_FRAME
        target_pose_base.pose.position.x = float(tool_p[0])
        target_pose_base.pose.position.y = float(tool_p[1])
        target_pose_base.pose.position.z = float(tool_p[2])
        target_pose_base.pose.orientation.x = float(q_des[0])
        target_pose_base.pose.orientation.y = float(q_des[1])
        target_pose_base.pose.orientation.z = float(q_des[2])
        target_pose_base.pose.orientation.w = float(q_des[3])

        # ---------- 6) Init retry state + send first MoveIt goal ----------
        self.alignment_retry_index = 0
        self._last_alignment_target = target_pose_base
        self.sent_goal = False  # will be set True inside _send_moveit_alignment_goal
        self._send_moveit_alignment_goal(target_pose_base)

    def _reset_and_arm_yolo(self):
        self.armed = True
        self.home_goal_active = False
        try:
            self.yolo_node.set_enabled(True)
        except Exception as e:
            self.get_logger().warn(f"Failed to enable YOLO: {e}")
            self._fail_observation(str(e))

    def send_rotate_in_place_goal(self, target_pose_stamped: PoseStamped) -> None:
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            msg = "MoveIt2 action server not available"
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_y_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_z_axis_tolerance = float(self.ORI_TOL)
        oc.weight = 1.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(self.POS_TOL)]

        center_pose = Pose()
        center_pose.position = target_pose_stamped.pose.position
        center_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [center_pose]

        pc = PositionConstraint()
        pc.header = target_pose_stamped.header
        pc.link_name = self.TOOL_LINK
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        pc.constraint_region = bv
        pc.weight = 1.0

        goal_constraint = Constraints()
        goal_constraint.orientation_constraints = [oc]
        goal_constraint.position_constraints = [pc]
        req.goal_constraints = [goal_constraint]

        goal_msg.request = req
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False

        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._moveit_goal_response_callback)

    def _moveit_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            msg = "MoveIt2 goal rejected"
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return
        rf = goal_handle.get_result_async()
        rf.add_done_callback(self._moveit_result_callback)

    def _moveit_result_callback(self, future) -> None:
        try:
            result = future.result().result
            self.alignment_active = False
            if result.error_code.val != 1:
                msg = f"Rotate-in-place failed: code={result.error_code.val}"
                self.get_logger().error(msg)
                self.sent_goal = False
                self._fail_observation(msg)
        except Exception as e:
            self.sent_goal = False
            self._fail_observation(str(e))

    # ============================================================
    #                    YOLO + HOME RETURN (NORMAL)
    # ============================================================

    def _on_yolo_detected(self, msg: Bool):
        if not msg.data:
            return
        if not self.armed or self.home_goal_active:
            return

        self.armed = False
        try:
            self.yolo_node.set_enabled(False)
        except Exception:
            pass

        self.home_goal_active = True
        self._send_return_to_home_goal()

    def _on_yolo_label(self, msg: String):
        self.last_label = msg.data or ""

    def _send_return_to_home_goal(self):
        joint_names = ["ur_shoulder_pan_joint","ur_shoulder_lift_joint",
                       "ur_elbow_joint","ur_wrist_1_joint","ur_wrist_2_joint","ur_wrist_3_joint"]
        home_position = [-1.57, -1.57, -1.57, -0.3, 1.57, 0.0]

        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP
        cs = Constraints()
        for name, val in zip(joint_names, home_position):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = jc.tolerance_below = 0.03
            jc.weight = 1.0
            cs.joint_constraints.append(jc)

        req.goal_constraints = [cs]
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False

        fut = self.moveit_action_client.send_goal_async(goal)
        fut.add_done_callback(self._home_position_goal_response_cb)

    def _home_position_goal_response_cb(self, fut):
        gh = fut.result()
        if not gh or not gh.accepted:
            msg = "Home pose MoveIt goal rejected"
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._home_position_result_cb)

    def _home_position_result_cb(self, fut):
        try:
            res = fut.result().result
            if res.error_code.val == 1:
                self.get_logger().info("Manipulator returned to home pose. Observation complete.")
                self._succeed_observation()
            else:
                msg = f"Home return failed: code={res.error_code.val}"
                self.get_logger().error(msg)
                # SOFT: treat as warning if we already have a label
                if self.last_label:
                    self.get_logger().warn(
                        "Home failed but label is present → "
                        "marking observation as success with warning."
                    )
                    self._succeed_observation()
                    self._observation_error = msg  # optional: keep as info text
                else:
                    self._fail_observation(msg)
        except Exception as exc:
            msg = f"Home return result error: {exc}"
            self.get_logger().error(msg)
            self._fail_observation(msg)


    # ============================================================
    #                    OBSERVATION RESULT UTILS
    # ============================================================

    def _succeed_observation(self):
        self._observation_success = True
        if self._observation_done_event:
            self._observation_done_event.set()

    def _fail_observation(self, msg: str):
        self._observation_success = False
        self._observation_error = msg
        if self._observation_done_event:
            self._observation_done_event.set()

    # ============================================================
    #                        TF READY HELPERS
    # ============================================================

    def _tf_ready(self) -> bool:
        try:
            _ = self._tf_lookup_cached(self.BASE_FRAME, self.TOOL_LINK)
            _ = self._tf_lookup_cached(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)
            return True
        except Exception:
            return False

    def _wait_for_tf_ready(self, timeout_sec: float = 5.0) -> bool:
        end_time = self.get_clock().now() + Duration(seconds=timeout_sec)
        while self.get_clock().now() < end_time:
            if self._tf_ready():
                return True
            time.sleep(0.1)
        return self._tf_ready()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigator()
    yolo_node = node.yolo_node

    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    exec.add_node(yolo_node)

    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec.shutdown()
        yolo_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
