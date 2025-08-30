#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage
from rclpy.qos import qos_profile_sensor_data
import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped, Pose, Vector3
import tf2_geometry_msgs
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from scipy.spatial.transform import Rotation as R
import math


class SimpleArucoLookAt(Node):
    def __init__(self):
        super().__init__('simple_aruco_look_at')

        # --- Parameters you might want to tweak ---
        self.BASE_FRAME = "ur_base_link"
        self.TOOL_LINK  = "ur_tool0"
        self.CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"  # must exist in TF
        self.MOVE_GROUP = "ur5_manip"
        self.ORI_TOL = 0.1  # radians (~6 deg)
        self.POS_TOL = 0.02 # meters (2 cm) to rotate in place
        self.MARKER_SIZE_M = 0.5
        self.require_camera_tf = True  # try to aim camera +Z; if TF missing and this is True, will error out

        self.bridge = CvBridge()
        self.moveit_action_client = ActionClient(self, MoveGroup, '/move_action')

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # One-shot guard
        self.sent_goal = False

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/realsense/camera/color/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/realsense/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.get_logger().info('🤖 Simple ArUco Look-At (single-shot, rotate-in-place) started')

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info('✅ Camera calibration received')

    # ---------- Math helpers ----------
    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-9:
            return v, 0.0
        return v / n, n

    def _look_at_cameraZ_worldUp(self, from_world, to_world):
        """
        Build a rotation matrix that orients a camera frame such that:
        - camera +Z points from 'from_world' to 'to_world'
        - roll is 'upright' w.r.t. world up (0,0,1)
        Returns 3x3 rotation matrix R_w_c (world->camera)
        """
        f, _ = self._normalize(to_world - from_world)  # forward (+Z of camera)
        up_w = np.array([0.0, 0.0, 1.0])

        # Handle near-parallel forward/up
        if abs(np.dot(f, up_w)) > 0.99:
            up_w = np.array([0.0, 1.0, 0.0])

        r = np.cross(up_w, f); r, _ = self._normalize(r)  # right
        u = np.cross(f, r);    u, _ = self._normalize(u)  # corrected up

        R_w_c = np.column_stack((r, u, f))  # columns: right, up, forward

        # ---- Uprightness enforcement (anti-upside-down) ----
        # Make sure the camera's "up" (column 1) generally aligns with world up.
        # If it's inverted, flip roll by 180° (flip right & up).
        if np.dot(R_w_c[:, 1], up_w) < 0.0:
            R_w_c[:, 0] *= -1.0
            R_w_c[:, 1] *= -1.0
        return R_w_c

    def _yaw_pitch_toolX(self, dx, dy, dz):
        """
        Extrinsic ZYX: yaw about world Z, then pitch about world Y, roll=0.
        Tool +X will point toward the marker.
        """
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        roll = 0.0
        q = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False).as_quat()  # x,y,z,w
        return q, (yaw, pitch, roll)

    # ---------- Core logic ----------
    def look_at_marker(self, marker_pose_base: PoseStamped):
        """
        Compute orientation target and send a single rotate-in-place goal.
        """
        if self.sent_goal:
            return

        try:
            # Current tool pose (for position anchoring)
            t_tool = self.tf_buffer.lookup_transform(self.BASE_FRAME, self.TOOL_LINK, rclpy.time.Time())
            tool_p = np.array([
                t_tool.transform.translation.x,
                t_tool.transform.translation.y,
                t_tool.transform.translation.z
            ], dtype=np.float64)

            marker_p = np.array([
                marker_pose_base.pose.position.x,
                marker_pose_base.pose.position.y,
                marker_pose_base.pose.position.z
            ], dtype=np.float64)

            v = marker_p - tool_p
            dx, dy, dz = float(v[0]), float(v[1]), float(v[2])

            # Try to aim the actual camera optical axis (+Z) if TF is available
            use_camera_tf = False
            R_w_t_des = None
            try:
                t_tool_cam = self.tf_buffer.lookup_transform(self.TOOL_LINK, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time())
                # Rotation tool->camera
                R_t_c = R.from_quat([
                    t_tool_cam.transform.rotation.x,
                    t_tool_cam.transform.rotation.y,
                    t_tool_cam.transform.rotation.z,
                    t_tool_cam.transform.rotation.w
                ]).as_matrix()
                R_c_t = R_t_c.T

                # Desired camera orientation in world
                R_w_c_des = self._look_at_cameraZ_worldUp(tool_p, marker_p)
                # Convert to tool desired orientation in world
                R_w_t_des = R_w_c_des @ R_c_t
                use_camera_tf = True
            except Exception as e:
                if self.require_camera_tf:
                    self.get_logger().error(f'❌ Needed TF {self.TOOL_LINK}->{self.CAMERA_OPTICAL_FRAME} not found: {e}')
                    return
                else:
                    self.get_logger().warn('⚠️ Camera TF not available — falling back to tool +X aiming')

            if not use_camera_tf:
                # Fallback: aim tool +X with yaw/pitch (extrinsic)
                q_xyzw, _ = self._yaw_pitch_toolX(dx, dy, dz)
                R_w_t_des = R.from_quat(q_xyzw).as_matrix()

            # Convert to quaternion for goal
            q_des = R.from_matrix(R_w_t_des).as_quat()  # x,y,z,w

            # Build target pose at current position, new orientation
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
                f"🎯 Marker@({marker_p[0]:.3f},{marker_p[1]:.3f},{marker_p[2]:.3f}) "
                f"EE@({tool_p[0]:.3f},{tool_p[1]:.3f},{tool_p[2]:.3f})"
            )
            self.get_logger().info(
                "🧭 Desired q(x,y,z,w)=("
                f"{target_pose.pose.orientation.x:.4f},"
                f"{target_pose.pose.orientation.y:.4f},"
                f"{target_pose.pose.orientation.z:.4f},"
                f"{target_pose.pose.orientation.w:.4f})"
            )

            self.send_rotate_in_place_goal(target_pose)
            self.sent_goal = True

        except Exception as e:
            self.get_logger().error(f'❌ Look-at computation failed: {str(e)}')

    def send_rotate_in_place_goal(self, target_pose: PoseStamped):
        """
        Send a MoveIt goal that keeps position within a small sphere and matches orientation.
        """
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('❌ MoveIt2 action server not available')
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        # --- Orientation constraint (tight) ---
        oc = OrientationConstraint()
        oc.header = target_pose.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose.pose.orientation
        oc.absolute_x_axis_tolerance = self.ORI_TOL
        oc.absolute_y_axis_tolerance = self.ORI_TOL
        oc.absolute_z_axis_tolerance = self.ORI_TOL
        oc.weight = 1.0

        # --- Position constraint (tiny sphere at current pos) ---
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.POS_TOL]  # radius

        center_pose = Pose()
        center_pose.position = target_pose.pose.position  # same as current EE
        center_pose.orientation.w = 1.0  # irrelevant for sphere

        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [center_pose]

        pc = PositionConstraint()
        pc.header = target_pose.header
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

        self.get_logger().info("📤 Sending single rotate-in-place goal…")
        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected')
            return
        self.get_logger().info('✅ Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        try:
            result = future.result().result
            if result.error_code.val == 1:
                self.get_logger().info('✅ Rotation complete (success)')
            else:
                self.get_logger().error(f'❌ Rotate-in-place failed: error_code={result.error_code.val}')
        except Exception as e:
            self.get_logger().error(f'❌ Result callback error: {e}')

    def image_callback(self, msg: CompressedImage):
        if self.sent_goal:
            return  # single-shot

        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None or self.camera_matrix is None:
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is None or len(ids) == 0:
                return

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.MARKER_SIZE_M, self.camera_matrix, self.dist_coeffs
            )

            # Use the first marker found
            camera_point = PointStamped()
            camera_point.header.frame_id = self.CAMERA_OPTICAL_FRAME
            camera_point.header.stamp = rclpy.time.Time().to_msg()
            camera_point.point.x = float(tvecs[0][0][0])
            camera_point.point.y = float(tvecs[0][0][1])
            camera_point.point.z = float(tvecs[0][0][2])

            # Transform to base
            marker_base_point = self.tf_buffer.transform(
                camera_point,
                self.BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            marker_pose_base = PoseStamped()
            marker_pose_base.header = marker_base_point.header
            marker_pose_base.pose.position = marker_base_point.point
            marker_pose_base.pose.orientation.w = 1.0

            self.look_at_marker(marker_pose_base)

        except Exception as e:
            self.get_logger().error(f'❌ Image callback error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleArucoLookAt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down…')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
