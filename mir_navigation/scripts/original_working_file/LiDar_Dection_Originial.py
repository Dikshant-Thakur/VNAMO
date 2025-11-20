#!/usr/bin/env python3
"""
FixedMarkerNavigator (no Nav2, no live ArUco)
- Reuses your EXISTING stack: rclpy, MoveIt2 ActionClient (/move_action), YOLO node (yolo_detector.YoloDetector), same frames.
- Hardcoded (param) marker pose in MAP frame → compute orientation-only goal (rotate-in-place at current tool position) → enable YOLO until first detection → return to home.

NOTES
- Frames kept as-is: ur_base_link, ur_tool0, realsense_color_optical_frame.
- No moveit_commander; we use moveit_msgs + ActionClient just like your current file.
- Minimal changes; only what’s needed to drop Nav2 & live ArUco.
"""

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, Pose, Vector3

import tf2_ros
from scipy.spatial.transform import Rotation as R
import tf2_geometry_msgs  # noqa: F401 (registers conversions)

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


class FixedMarkerNavigator(Node):
    def __init__(self) -> None:
        super().__init__("fixed_marker_navigator")

        # ---------------- Frames & group (UNCHANGED) ----------------
        self.BASE_FRAME = "ur_base_link"
        self.TOOL_LINK = "ur_tool0"
        self.CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"
        self.MOVE_GROUP = "ur5_manip"

        # ---------------- Planning tolerances (reuse) ---------------
        self.ORI_TOL = 0.17  # rad (~10 deg)
        self.POS_TOL = 0.10  # m (sphere radius)

        # ---------------- State flags ----------------
        self.alignment_active = False
        self.sent_goal = False
        self.armed = False
        self.home_goal_active = False
        self.push_started = False

        # ---------------- Params ----------------
        # Hardcoded marker pose IN MAP FRAME (x, y, z). Orientation optional (w=1 by default)
        # self.declare_parameter("marker_xyz", [1.0, 0.0, 0.9])
        # self.declare_parameter("marker_q_xyzw", [0.0, 0.0, 0.0, 1.0])
        # Optionally override with a fixed yaw (deg) for tool orientation; if None use look-at logic
        self.declare_parameter("approach_yaw_deg", None)

        m_xyz = self.get_parameter("marker_xyz").get_parameter_value().double_array_value
        m_q = self.get_parameter("marker_q_xyzw").get_parameter_value().double_array_value
        if len(m_xyz) != 3 or len(m_q) != 4:
            raise ValueError("marker_xyz must be [x,y,z] and marker_q_xyzw must be [qx,qy,qz,qw]")

        self.marker_pose_map = PoseStamped()
        self.marker_pose_map.header.frame_id = "map"
        self.marker_pose_map.pose.position.x = float(m_xyz[0])
        self.marker_pose_map.pose.position.y = float(m_xyz[1])
        self.marker_pose_map.pose.position.z = float(m_xyz[2])
        self.marker_pose_map.pose.orientation.x = float(m_q[0])
        self.marker_pose_map.pose.orientation.y = float(m_q[1])
        self.marker_pose_map.pose.orientation.z = float(m_q[2])
        self.marker_pose_map.pose.orientation.w = float(m_q[3])

        yaw_param = self.get_parameter("approach_yaw_deg").get_parameter_value()
        self.approach_yaw_deg = yaw_param.double_value if hasattr(yaw_param, "double_value") else None

        # ---------------- TF setup (reuse) ----------------
        self.tf_timeout = Duration(seconds=0.5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_cache = {}
        self._tf_cache_ttl = 0.2

        # ---------------- MoveIt & YOLO (existing tools) ------------
        self.moveit_action_client = ActionClient(self, MoveGroup, "/move_action")
        self.yolo_node = yp.YoloDetector()
        self.yolo_node.set_enabled(False)

        # YOLO feedback
        self.yolo_seen = False
        self.create_subscription(Bool, "/yolo/detected", self._on_yolo_detected, 10)
        self.create_subscription(String, "/yolo/label", self._on_yolo_label, 10)
        self.start_push_pub = self.create_publisher(Bool, "/start_push", 10)

        # Kick main once TF is likely warm
        self.create_timer(0.5, self._kickoff)
        self.get_logger().info("Node initialized (No Nav2 | Fixed marker | MoveIt rotate-in-place)")

    # ---------------------- Kickoff ----------------------
    def _kickoff(self):
        if self.sent_goal:
            return
        if not self._tf_ready():
            return
        self.get_logger().info("Planning to hardcoded marker (orientation-only)…")
        try:
            self.align_camera_to_marker_pose(self.marker_pose_map)
        except Exception as e:
            self.get_logger().error(f"Kickoff failed: {e}")

    # ------------------ ALIGNMENT (MOVEIT) -----------------
    def align_camera_to_marker_pose(self, marker_pose_map: PoseStamped) -> None:
        if self.sent_goal:
            return
        # transforms in base frame
        t_tool = self._tf_lookup_cached(self.BASE_FRAME, self.TOOL_LINK)
        t_cam = self._tf_lookup_cached(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)

        marker_pose_map_latest = PoseStamped()
        marker_pose_map_latest.header.frame_id = "map"
        marker_pose_map_latest.header.stamp = self.get_clock().now().to_msg()
        marker_pose_map_latest.pose = marker_pose_map.pose

        marker_pose_base = self.tf_buffer.transform(
            marker_pose_map_latest, self.BASE_FRAME, timeout=self.tf_timeout
        )

        tool_p = np.array([
            t_tool.transform.translation.x,
            t_tool.transform.translation.y,
            t_tool.transform.translation.z,
        ], dtype=np.float64)
        cam_p = np.array([
            t_cam.transform.translation.x,
            t_cam.transform.translation.y,
            t_cam.transform.translation.z,
        ], dtype=np.float64)
        marker_p = np.array([
            marker_pose_base.pose.position.x,
            marker_pose_base.pose.position.y,
            marker_pose_base.pose.position.z,
        ], dtype=np.float64)

        # Orientation to look-at marker: prefer camera optical Z, else tool +X
        use_camera_tf = False
        try:
            if self.approach_yaw_deg is None:
                t_tool_cam = self._tf_lookup_cached(self.TOOL_LINK, self.CAMERA_OPTICAL_FRAME)
                R_t_c = R.from_quat([
                    t_tool_cam.transform.rotation.x,
                    t_tool_cam.transform.rotation.y,
                    t_tool_cam.transform.rotation.z,
                    t_tool_cam.transform.rotation.w,
                ]).as_matrix()
                R_c_t = R_t_c.T
                R_w_c_des = self._look_at_cameraZ_worldUp(cam_p, marker_p)
                R_w_t_des = R_w_c_des @ R_c_t
                use_camera_tf = True
            else:
                # fixed yaw around world Z at current tool pos
                yaw = math.radians(float(self.approach_yaw_deg))
                R_w_t_des = R.from_euler('z', yaw).as_matrix()
        except Exception as e:
            self.get_logger().warning(f"Camera TF not available, fallback tool +X: {e}")
            v = marker_p - tool_p
            q = self._yaw_pitch_toolX_quat(v[0], v[1], v[2])
            R_w_t_des = R.from_quat(q).as_matrix()

        q_des = R.from_matrix(R_w_t_des).as_quat()  # x,y,z,w

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.BASE_FRAME
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = float(tool_p[0])
        target_pose.pose.position.y = float(tool_p[1])
        target_pose.pose.position.z = float(tool_p[2])
        target_pose.pose.orientation.x = float(q_des[0])
        target_pose.pose.orientation.y = float(q_des[1])
        target_pose.pose.orientation.z = float(q_des[2])
        target_pose.pose.orientation.w = float(q_des[3])

        self.get_logger().info(
            "MoveIt Goal Preview:
"
            f"  Group          : {self.MOVE_GROUP}
"
            f"  Link           : {self.TOOL_LINK}
"
            f"  Base Frame     : {self.BASE_FRAME}
"
            f"  Use Camera TF  : {use_camera_tf}
"
            f"  Target Pos (m) : x={tool_p[0]:.4f}, y={tool_p[1]:.4f}, z={tool_p[2]:.4f}
"
            f"  Target Q (xyzw): [{q_des[0]:.5f}, {q_des[1]:.5f}, {q_des[2]:.5f}, {q_des[3]:.5f}]
"
            f"  Ori Tol (rad)  : {self.ORI_TOL:.3f}
"
            f"  Pos Tol (m)    : {self.POS_TOL:.3f}"
        )

        self._reset_and_arm_yolo()
        self.send_rotate_in_place_goal(target_pose)
        self.sent_goal = True
        self.alignment_active = True

    def send_rotate_in_place_goal(self, target_pose_stamped: PoseStamped) -> None:
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveIt2 action server not available")
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        # Orientation constraint (tight)
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_y_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_z_axis_tolerance = float(self.ORI_TOL)
        oc.weight = 1.0

        # Position constraint: sphere around current tool pos (rotate-in-place)
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

        self.get_logger().info(
            "MoveIt Request:
"
            f"  goal_constraints: 1 (orientation + position)
"
            f"  oc.link_name     : {oc.link_name}
"
            f"  oc.tolerances    : [{oc.absolute_x_axis_tolerance:.3f}, {oc.absolute_y_axis_tolerance:.3f}, {oc.absolute_z_axis_tolerance:.3f}]
"
            f"  pc.radius (m)    : {sphere.dimensions[0]:.3f}
"
            f"  plan_only        : {goal_msg.planning_options.plan_only}"
        )

        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._moveit_goal_response_callback)

    def _moveit_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.alignment_active = False
            self.get_logger().error("MoveIt2 goal rejected")
            self.sent_goal = False
            return
        self.get_logger().info("MoveIt2 goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._moveit_result_callback)

    def _moveit_result_callback(self, future) -> None:
        try:
            result = future.result().result
            self.alignment_active = False
            if result.error_code.val == 1:
                self.get_logger().info("Camera aligned to marker (rotation complete)")
            else:
                self.get_logger().error(f"Rotate-in-place failed: error_code={result.error_code.val}")
                self.sent_goal = False  # allow retry if needed
        except Exception as e:
            self.alignment_active = False
            self.get_logger().error(f"Result callback error: {e}")
            self.sent_goal = False

    # ---------------- YOLO gating & home return ---------------
    def _reset_and_arm_yolo(self):
        self.armed = True
        self.home_goal_active = False
        self.push_started = False
        try:
            self.yolo_node.set_enabled(True)
        except Exception as e:
            self.get_logger().warn(f"Failed to enable YOLO: {e}")

    def _on_yolo_detected(self, msg: Bool):
        if not msg.data:
            return
        if not self.armed or self.home_goal_active:
            return
        self.armed = False
        try:
            self.yolo_node.set_enabled(False)
        except Exception as e:
            self.get_logger().warn(f"Failed to disable YOLO: {e}")
        if not self.home_goal_active:
            self.home_goal_active = True
            self.get_logger().info("[YOLO] Detection True → returning manip to home pose")
            self._send_return_to_home_goal()

    def _on_yolo_label(self, msg: String):
        label = msg.data or ""
        if label:
            self.get_logger().info(f"[YOLO] Label: {label}")

    def _send_return_to_home_goal(self):
        joint_names = [
            "ur_shoulder_pan_joint",
            "ur_shoulder_lift_joint",
            "ur_elbow_joint",
            "ur_wrist_1_joint",
            "ur_wrist_2_joint",
            "ur_wrist_3_joint",
        ]
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
            self.get_logger().error("Home pose MoveIt goal rejected")
            return
        self.get_logger().info("Home pose MoveIt goal accepted")
        rf = gh.get_result_async()
        rf.add_done_callback(self._home_position_result_cb)

    def _home_position_result_cb(self, fut):
        try:
            res = fut.result().result
            if res.error_code.val == 1:
                if not self.push_started:
                    self.push_started = True
                    self.get_logger().info("Manipulator returned to home → starting push")
                    self.start_push_pub.publish(Bool(data=True))
            else:
                self.get_logger().error(f"Home return failed: code={res.error_code.val}")
        except Exception as e:
            self.get_logger().error(f"Home return result error: {e}")

    # ---------------- TF helpers (reuse) ---------------------
    def _tf_ready(self) -> bool:
        try:
            _ = self._tf_lookup_cached(self.BASE_FRAME, self.TOOL_LINK)
            _ = self._tf_lookup_cached(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)
            return True
        except Exception:
            return False

    def _tf_lookup_cached(self, target_frame: str, source_frame: str):
        key = (target_frame, source_frame)
        now = self.get_clock().now().nanoseconds * 1e-9
        if key in self._tf_cache:
            t_stamp, t_val = self._tf_cache[key]
            if (now - t_stamp) <= self._tf_cache_ttl:
                return t_val
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        self._tf_cache[key] = (now, t)
        return t

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
        s = np.cross(f, up_w)
        s, _ = self._normalize(s)
        u = np.cross(s, f)
        R_w_c = np.eye(3)
        R_w_c[:, 0] = s
        R_w_c[:, 1] = u
        R_w_c[:, 2] = f
        return R_w_c

    def _yaw_pitch_toolX_quat(self, vx: float, vy: float, vz: float) -> np.ndarray:
        v = np.array([vx, vy, vz], dtype=np.float64)
        v, n = self._normalize(v)
        if n < 1e-6:
            return np.array([0, 0, 0, 1], dtype=np.float64)
        yaw = math.atan2(v[1], v[0])
        pitch = math.atan2(v[2], max(1e-6, math.sqrt(v[0] ** 2 + v[1] ** 2)))
        q = R.from_euler("zyx", [yaw, -pitch, 0.0]).as_quat()
        return q


def main(args=None):
    rclpy.init(args=args)
    node = FixedMarkerNavigator()
    yolo_node = node.yolo_node

    exec = MultiThreadedExecutor(num_threads=2)
    exec.add_node(node)
    exec.add_node(yolo_node)

    try:
        exec.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        exec.shutdown()
        yolo_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

