#!/usr/bin/env python3
"""
Refactored: ArUco-driven navigation + MoveIt rotate-in-place camera alignment.
- Cleaner structure (setup_* sections, small helpers)
- Fewer repeated TF lookups (simple caching)
- ROS2 timers instead of ad-hoc threads
- Consistent logging (no emojis)
- Defensive guards and early returns to avoid nested try/except

Assumptions kept from original file:
- Nav2 stack active
- MoveIt2 action server exposed at /move_action
- Frames: ur_base_link, ur_tool0, realsense_color_optical_frame, map
- Topics: realsense color compressed image + camera_info
- YOLO side-node available as yolo_detector.YoloDetector
"""

import math
import time
import threading  # only for clean shutdown conjunction with MultiThreadedExecutor
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, Pose, Vector3, Twist
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from scipy.spatial.transform import Rotation as R

# TF geometry helpers must be imported to register converters
import tf2_geometry_msgs  # noqa: F401

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

from nav2_simple_commander import robot_navigator
from nav2_simple_commander import robot_navigator as robot_nav

import yolo_detector as yp


class ArucoNavigator(Node):
    # --------------------------- INIT ---------------------------
    def __init__(self) -> None:
        super().__init__("aruco_detector_moveit_node")
        self.bridge = CvBridge()

        # Params & frames
        self.BASE_FRAME = "ur_base_link"
        self.TOOL_LINK = "ur_tool0"
        self.CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"
        self.MOVE_GROUP = "ur5_manip"

        #YOLO FEEDBACK
        self.yolo_seen = False
        self.yolo_label = ""
        self.yolo_detect_sub = self.create_subscription(
            Bool, "/yolo/detected", self._on_yolo_detected, 10
        )
        self.yolo_label_sub = self.create_subscription(
            String, "/yolo/label", self._on_yolo_label, 10
        )




        # Push button to start pushing
        self.start_push_pub = self.create_publisher(Bool, "/start_push", 10)

        # Planning tolerances
        self.ORI_TOL = 0.17  # rad (~10 deg)
        self.POS_TOL = 0.10  # m (sphere radius)

        # Nav approach distance multiplier
        self.marker_distance_offset_scale = 2.0

        # State flags
        self.navigation_active = False
        self.alignment_active = False
        self.sent_goal = False
        self.marker_pose_map: Optional[PoseStamped] = None

        # Camera calib
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.5  # meters

        # Timers/utility
        self._zero_vel_timer = None

        self.setup_publishers_subscribers()
        self.setup_tf()
        self.setup_nav2()
        self.setup_moveit()
        self.setup_yolo()

        # __init__ ke end ke paas (publishers/subscribers ban chuke hon), yeh teen flags add karo:
        self.armed = False              # first detection ko allow?
        self.home_goal_active = False   # home goal already bhej diya?
        self.push_started = False       # push already trigger ho chuka?


        self.get_logger().info("Node initialized (Nav2 + MoveIt rotate-in-place)")

    # ---------------------- SETUP SECTIONS ----------------------
    def setup_publishers_subscribers(self) -> None:
        # pubs
        self.cmd_pub = self.create_publisher(Twist, "/diff_cont/cmd_vel_unstamped", 10)
        self.detection_pub = self.create_publisher(Bool, "/aruco_detected", 10)
        self.marker_info_pub = self.create_publisher(String, "/marker_info", 10)

        # subs
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/realsense/camera/color/image_raw/compressed",
            self.compressed_image_callback,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            "/realsense/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )

    def setup_tf(self) -> None:
        self.tf_timeout = Duration(seconds=0.5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # simple transform cache (key -> (stamp_sec, transform))
        self._tf_cache = {}
        self._tf_cache_ttl = 0.2  # seconds

    def setup_nav2(self) -> None:
        self.navigator = robot_navigator.BasicNavigator()
        self.navigator.waitUntilNav2Active()

    def setup_moveit(self) -> None:
        self.moveit_action_client = ActionClient(self, MoveGroup, "/move_action")

    def setup_yolo(self) -> None:
        self.yolo_node = yp.YoloDetector()
        self.yolo_node.set_enabled(False)

    # --------------------- CAMERA CALIBRATION -------------------
    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.image_width = int(msg.width)
        self.image_height = int(msg.height)
        self.get_logger().info("Camera calibration received")


    # ------------------------- HELPERS -------------------------
    def _normalize(self, v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v, 0.0
        return v / n, n

    def _look_at_cameraZ_worldUp(self, cam_world_p: np.ndarray, target_world_p: np.ndarray) -> np.ndarray:
        """Return desired world->camera rotation so camera +Z looks at target; keep roll upright to world +Z."""
        f, _ = self._normalize(target_world_p - cam_world_p)  # forward
        up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f, up_w)) > 0.99:  # nearly parallel to up -> change helper up
            up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(up_w, f); r, _ = self._normalize(r)
        u = np.cross(f, r);   u, _ = self._normalize(u)
        R_w_c = np.column_stack((r, u, f))
        # keep camera up not inverted wrt world up
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

    def _start_zero_vel_hold(self, duration_sec: float = 2.0, rate_hz: float = 20.0) -> None:
        """Publish zero velocity for a short duration using a ROS2 timer."""
        if self._zero_vel_timer is not None:
            self._zero_vel_timer.cancel()
            self._zero_vel_timer = None

        period = 1.0 / float(rate_hz)
        stop_time = self.get_clock().now() + rclpy.time.Duration(seconds=duration_sec)
        twist = Twist()  # all zeros

        def _tick():
            if self.get_clock().now() >= stop_time:
                if self._zero_vel_timer:
                    self._zero_vel_timer.cancel()
                    self._zero_vel_timer = None
                return
            self.cmd_pub.publish(twist)

        self._zero_vel_timer = self.create_timer(period, _tick)

    # ----------------------- NAVIGATION ------------------------
    def navigate_to_marker_front(self, map_x: float, map_y: float, marker_id: int, rvec: np.ndarray) -> None:
        try:
            # face normal from rvec in camera frame -> transform to map and project
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            face_normal_z_unit = rotation_matrix[:, 2] / np.linalg.norm(rotation_matrix[:, 2])

            z_cam = Vector3Stamped()
            z_cam.header.frame_id = self.CAMERA_OPTICAL_FRAME
            z_cam.header.stamp = rclpy.time.Time().to_msg()
            z_cam.vector.x, z_cam.vector.y, z_cam.vector.z = [float(x) for x in face_normal_z_unit]

            map_normal = self.tf_buffer.transform(z_cam, "map", timeout=self.tf_timeout)
            mx, my, mz = map_normal.vector.x, map_normal.vector.y, map_normal.vector.z
            sign = -1.0 if face_normal_z_unit[2] > 0.0 else 1.0

            goal_x = map_x + sign * mx * self.marker_distance_offset_scale
            goal_y = map_y + sign * my * self.marker_distance_offset_scale

            face_angle = math.atan2(-my, -mx)
            qz = math.sin(face_angle / 2.0)
            qw = math.cos(face_angle / 2.0)

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = float(goal_x)
            goal_pose.pose.position.y = float(goal_y)
            goal_pose.pose.position.z = 0.0
            goal_pose.pose.orientation.x = 0.0
            goal_pose.pose.orientation.y = 0.0
            goal_pose.pose.orientation.z = float(qz)
            goal_pose.pose.orientation.w = float(qw)

            self.navigator.goToPose(goal_pose)
            self.navigation_active = True
            self.get_logger().info(f"Navigating to marker {marker_id} front: ({goal_x:.2f}, {goal_y:.2f})")
            # poll using a timer rather than creating many timers
            self.create_timer(1.0, self._check_navigation_status)
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")

    def _check_navigation_status(self) -> None:
        if not self.navigation_active:
            return
        if not self.navigator.isTaskComplete():
            return
        result = self.navigator.getResult()
        self.navigation_active = False
        if result == robot_nav.TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation completed, starting camera alignment")
            if self.marker_pose_map and not self.sent_goal:
                self.align_camera_to_marker_pose(self.marker_pose_map)
        else:
            self.get_logger().warning(f"Navigation failed with result: {result}")

    # ------------------ ALIGNMENT (MOVEIT) ---------------------
    def align_camera_to_marker_pose(self, marker_pose_map: PoseStamped) -> None:
        if self.sent_goal:
            return
        try:
            # transforms in base frame
            t_tool = self._tf_lookup_cached(self.BASE_FRAME, self.TOOL_LINK)
            t_cam = self._tf_lookup_cached(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)

            marker_pose_map_latest = PoseStamped()
            marker_pose_map_latest.header.frame_id = "map"
            marker_pose_map_latest.header.stamp = rclpy.time.Time().to_msg()
            marker_pose_map_latest.pose = marker_pose_map.pose

            marker_pose_base = self.tf_buffer.transform(marker_pose_map_latest, self.BASE_FRAME, timeout=self.tf_timeout)

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

            # Try using camera optical axis for look-at
            use_camera_tf = False
            try:
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
            except Exception as e:
                self.get_logger().warning(f"Camera TF not available, falling back to tool +X aiming: {e}")
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
                "MoveIt Goal Preview:\n"
                f"  Group          : {self.MOVE_GROUP}\n"
                f"  Link           : {self.TOOL_LINK}\n"
                f"  Base Frame     : {self.BASE_FRAME}\n"
                f"  Use Camera TF  : {use_camera_tf}\n"
                f"  Target Pos (m) : x={tool_p[0]:.4f}, y={tool_p[1]:.4f}, z={tool_p[2]:.4f}\n"
                f"  Target Q (xyzw): [{q_des[0]:.5f}, {q_des[1]:.5f}, {q_des[2]:.5f}, {q_des[3]:.5f}]\n"
                f"  Ori Tol (rad)  : {self.ORI_TOL:.3f}\n"
                f"  Pos Tol (m)    : {self.POS_TOL:.3f}"
            )
            self._reset_and_arm_yolo()
            self.send_rotate_in_place_goal(target_pose)
            self.sent_goal = True
            self.alignment_active = True
        except Exception as e:
            self.get_logger().error(f"Align computation failed: {e}")


    def _reset_and_arm_yolo(self):
        # next cycle ke liye system ready
        self.armed = True
        self.home_goal_active = False
        self.push_started = False
        # yolo ko ON karo (ab first detection accept hoga)
        try:
            self.yolo_node.set_enabled(True)
        except Exception as e:
            self.get_logger().warn(f"Failed to enable YOLO: {e}")

    def send_rotate_in_place_goal(self, target_pose_stamped: PoseStamped) -> None:
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveIt2 action server not available")
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        #One thing to note - to control ee we have to define the PositionConstraint and OrientationConstraint
        #for the ee link - in our case ur_tool0
        # Orientation constraint (tight)
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_y_axis_tolerance = float(self.ORI_TOL)
        oc.absolute_z_axis_tolerance = float(self.ORI_TOL)
        oc.weight = 1.0

        # Position constraint: sphere around current tool pos (rotate in place)
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
            "MoveIt Request:\n"
            f"  goal_constraints: 1 (orientation + position)\n"
            f"  oc.link_name     : {oc.link_name}\n"
            f"  oc.tolerances    : [{oc.absolute_x_axis_tolerance:.3f}, {oc.absolute_y_axis_tolerance:.3f}, {oc.absolute_z_axis_tolerance:.3f}]\n"
            f"  pc.radius (m)    : {sphere.dimensions[0]:.3f}\n"
            f"  plan_only        : {goal_msg.planning_options.plan_only}"
        )

        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._moveit_goal_response_callback)
    
    # def _on_yolo_detected(self, msg: Bool):
    #     self.yolo_seen = bool(msg.data)
    #     if self.yolo_seen:
    #         self.get_logger().info("[YOLO] Detection True → returning manip to home pose")
    #         try:
    #             self.yolo_node.set_enabled(False)   # YOLO disable so it won’t trigger again
    #         except Exception:
    #             pass
    #         self._send_return_to_home_goal()

    def _on_yolo_detected(self, msg: Bool):
        if not msg.data:
            return

        # 1) guard: sirf first True accept karo
        if not self.armed or self.home_goal_active:
            return

        # 2) turant gate band karo (race window chhoti)
        self.armed = False
        try:
            self.yolo_node.set_enabled(False)
        except Exception as e:
            self.get_logger().warn(f"Failed to disable YOLO: {e}")

        # 3) ek hi home-goal bhejo
        if not self.home_goal_active:
            self.home_goal_active = True
            self.get_logger().info("[YOLO] Detection True → returning manip to home pose")
            self._send_return_to_home_goal()

    def _send_return_to_home_goal(self):
        joint_names = ["ur_shoulder_pan_joint","ur_shoulder_lift_joint",
                   "ur_elbow_joint","ur_wrist_1_joint","ur_wrist_2_joint","ur_wrist_3_joint"]
        home_position = [-1.57, -1.57, -1.57, -0.3, 1.57, 0.0]  # radians
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP
        cs = Constraints()
        #TO control each joints we have to define a JointConstraint for each joint
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
                # success
                if not self.push_started:
                    self.push_started = True
                    self.get_logger().info("Manipulator returned to home ✅ → starting push")
                    self.start_push_pub.publish(Bool(data=True))
            else:
                # failure
                self.get_logger().error(f"Home return failed: code={res.error_code.val}")
        except Exception as e:
            self.get_logger().error(f"Home return result error: {e}")

    
    def _on_yolo_label(self, msg: String):
        self.yolo_label = msg.data or ""
        if self.yolo_label:
            self.get_logger().info(f"[YOLO] Label detected: {self.yolo_label}")

    def _moveit_goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.alignment_active = False
            self.get_logger().error("MoveIt2 goal rejected")
            return
        self.get_logger().info("MoveIt2 goal accepted")
        # try:
        #     self.yolo_node.set_enabled(True)
        # except Exception as e:
        #     self.get_logger().error(f"Enable YOLO failed: {e}")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._moveit_result_callback)

    

    # def _moveit_result_callback(self, future) -> None:
    #     try:
    #         result = future.result().result
    #         self.alignment_active = False
    #         if result.error_code.val == 1:
    #             self.get_logger().info("Camera aligned to marker (rotation complete)")
    #             #self._start_zero_vel_hold(duration_sec=2.0, rate_hz=20.0)
    #         else:
    #             self.get_logger().error(f"Rotate-in-place failed: error_code={result.error_code.val}")
    #     except Exception as e:
    #         self.alignment_active = False
    #         self.get_logger().error(f"Result callback error: {e}")

    def _moveit_result_callback(self, future) -> None:
        try:
            result = future.result().result
            self.alignment_active = False
            if result.error_code.val == 1:
                self.get_logger().info("Camera aligned to marker (rotation complete)")
            else:
                self.get_logger().error(f"Rotate-in-place failed: error_code={result.error_code.val}")
                self.sent_goal = False  # <-- allow a reattempt on next ArUco update
        except Exception as e:
            self.alignment_active = False
            self.get_logger().error(f"Result callback error: {e}")
            self.sent_goal = False  # <-- also reset here

    # -------------------- IMAGE / DETECTION --------------------
    def compressed_image_callback(self, msg: CompressedImage) -> None:
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error("Failed to decode compressed image")
                return

            # detect aruco
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            detection_msg = Bool(data=False)
            if ids is not None:
                detection_msg.data = True
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

                if self.camera_matrix is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    for i, marker_id in enumerate(ids.flatten()):
                        # marker center in image (draw only, not used in logic)
                        center = corners[i][0].mean(axis=0)
                        cv2.circle(cv_image, (int(center[0]), int(center[1])), 6, (0, 0, 255), 2)

                        try:
                            camera_point = PointStamped()
                            camera_point.header.frame_id = self.CAMERA_OPTICAL_FRAME
                            camera_point.header.stamp = rclpy.time.Time().to_msg()
                            camera_point.point.x = float(tvecs[i][0][0])
                            camera_point.point.y = float(tvecs[i][0][1])
                            camera_point.point.z = float(tvecs[i][0][2])

                            map_point = self.tf_buffer.transform(camera_point, "map", timeout=self.tf_timeout)

                            map_pose = PoseStamped()
                            map_pose.header = map_point.header
                            map_pose.pose.position.x = map_point.point.x
                            map_pose.pose.position.y = map_point.point.y
                            map_pose.pose.position.z = map_point.point.z
                            map_pose.pose.orientation.w = 1.0

                            self.marker_pose_map = map_pose

                            if not self.navigation_active and not self.alignment_active and not self.sent_goal:
                                self.navigate_to_marker_front(
                                    map_pose.pose.position.x,
                                    map_pose.pose.position.y,
                                    marker_id,
                                    rvecs[i],
                                )
                        except Exception as e:
                            self.get_logger().error(f"Transform failed: {e}")

            self.detection_pub.publish(detection_msg)
            cv2.imshow("ArUco Detection + MoveIt rotate-in-place", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    # ------------------------- CLEANUP -------------------------
    def destroy_node(self) -> bool:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


# ------------------------------ MAIN ------------------------------

def main(args=None):
    rclpy.init(args=args)
    detector_node = ArucoNavigator()
    yolo_node = detector_node.yolo_node

    exec = MultiThreadedExecutor(num_threads=2)  # camera + YOLO parallel
    exec.add_node(detector_node)
    exec.add_node(yolo_node)

    try:
        exec.spin()
    except KeyboardInterrupt:
        detector_node.get_logger().info("Shutting down")
    finally:
        exec.shutdown()
        yolo_node.destroy_node()
        detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()