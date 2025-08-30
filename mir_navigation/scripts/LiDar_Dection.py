#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import threading, time

import cv2
import numpy as np

from rclpy.executors import MultiThreadedExecutor
import yolo_detector as yp 

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import String, Bool
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, Pose, Vector3, Twist
import tf2_geometry_msgs

from nav2_simple_commander import robot_navigator
from nav2_simple_commander import robot_navigator as robot_nav
from nav2_simple_commander.robot_navigator import TaskResult

# MoveIt2 Action imports
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

from scipy.spatial.transform import Rotation as R
import math

class ArucoDetectorWithMoveIt(Node):
    def __init__(self):
        super().__init__('aruco_detector_moveit_node')
        self.bridge = CvBridge()
        self.navigator = robot_navigator.BasicNavigator()
        self.navigator.waitUntilNav2Active()

        self.yolo_node = yp.YoloDetector()   # YOLO node construct
        self.yolo_node.set_enabled(False)    # start me band (order gate)

        # === Frames & planning params (tweak as needed) ===
        self.BASE_FRAME = "ur_base_link"
        self.TOOL_LINK  = "ur_tool0"
        self.CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"
        self.MOVE_GROUP = "ur5_manip"

        # Planning tolerances
        self.ORI_TOL = 0.17    # rad (~6 deg)
        self.POS_TOL = 0.1     # m (sphere radius for rotate-in-place)

        #Stop the robot
        self.cmd_pub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)

        # TF behaviour
        self.require_camera_tf = True  # if False, fall back to tool+X yaw/pitch aiming
        self.tf_timeout = rclpy.duration.Duration(seconds=0.5)
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=30.0)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.moveit_action_client = ActionClient(self, MoveGroup, '/move_action')
        self.marker_distance_offset_scale = 2.0
        self.navigation_active = False
        self.alignment_active = False
        self.sent_goal = False        # single alignment goal guard after Nav2
        self.marker_pose_map = None

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.5

        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/realsense/camera/color/image_raw/compressed',
            self.compressed_image_callback,
            qos_profile_sensor_data
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/realsense/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.detection_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.marker_info_pub = self.create_publisher(String, '/marker_info', 10)
        self.detection_count = 0

        self.alignment_tolerance = 0.05
        self.get_logger().info('🤖 ArUco Detector with MoveIt2 (Nav2 + camera look-at rotate-in-place)')

    # ---------------- Mobile Base stop ----------------
    def hold_base_still_async(self, duration_sec: float = 2.0, rate_hz: float = 20.0):
        """UR5e motion ke baad base ko actively 0 vel pe hold karta hai."""
        def _run():
            twist = Twist()  # sab default 0: linear=0, angular=0
            period = 1.0 / float(rate_hz)
            stop_t = time.time() + duration_sec
            while time.time() < stop_t and rclpy.ok():
                self.cmd_pub.publish(twist)
                time.sleep(period)
        threading.Thread(target=_run, daemon=True).start()

    # ---------------- Camera calib ----------------
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info('✅ Camera calibration received')

    # ---------------- Nav2 front approach ----------------
    def navigate_to_marker_front(self, map_x, map_y, marker_id, rvecs):
        try:
            rotation_matrix, _ = cv2.Rodrigues(rvecs)
            face_z_axis = rotation_matrix[:, 2]
            face_normal_z_unit = face_z_axis / np.linalg.norm(face_z_axis)

            z = Vector3Stamped()
            z.header.frame_id = self.CAMERA_OPTICAL_FRAME
            latest_time = self.tf_buffer.get_latest_common_time(self.CAMERA_OPTICAL_FRAME, "map")
            z.header.stamp = latest_time.to_msg()
            z.vector.x, z.vector.y, z.vector.z = (
                float(face_normal_z_unit[0]),
                float(face_normal_z_unit[1]),
                float(face_normal_z_unit[2]),
            )
            map_normal = self.tf_buffer.transform(z, "map", timeout=self.tf_timeout)
            map_normal_x = map_normal.vector.x
            map_normal_y = map_normal.vector.y
            sign = -1.0 if face_normal_z_unit[2] > 0 else 1.0

            goal_x = map_x + sign * map_normal_x * self.marker_distance_offset_scale
            goal_y = map_y + sign * map_normal_y * self.marker_distance_offset_scale
            face_angle = math.atan2(-map_normal_y, -map_normal_x)
            qz = math.sin(face_angle / 2.0)
            qw = math.cos(face_angle / 2.0)

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = goal_x
            goal_pose.pose.position.y = goal_y
            goal_pose.pose.position.z = 0.0
            goal_pose.pose.orientation.x = 0.0
            goal_pose.pose.orientation.y = 0.0
            goal_pose.pose.orientation.z = qz
            goal_pose.pose.orientation.w = qw

            self.navigator.goToPose(goal_pose)
            self.navigation_active = True
            self.get_logger().info(f'🎯 Navigating to marker {marker_id} front: ({goal_x:.2f}, {goal_y:.2f})')
            self.create_timer(1.0, self.check_navigation_status)
        except Exception as e:
            self.get_logger().error(f'❌ Navigation failed: {str(e)}')

    def check_navigation_status(self):
        if not self.navigation_active:
            return
        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            if result == robot_navigator.TaskResult.SUCCEEDED:
                self.get_logger().info('✅ Navigation completed! Starting camera pose alignment...')
                self.navigation_active = False
                if self.marker_pose_map and not self.sent_goal:
                    self.align_camera_to_marker_pose(self.marker_pose_map)
            else:
                self.get_logger().warn(f'⚠️ Navigation failed with result: {result}')
                self.navigation_active = False

    # ---------------- Look-at helpers ----------------
    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-9:
            return v, 0.0
        return v / n, n

    def _look_at_cameraZ_worldUp(self, cam_world_p, target_world_p):
        """
        Build rotation matrix so camera +Z looks at target with roll 'upright' w.r.t world up (0,0,1).
        Returns R_w_c (world->camera) as 3x3.
        """
        f, _ = self._normalize(target_world_p - cam_world_p)   # unit forward vector from cam to target.
        up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f, up_w)) > 0.99: #check if forward is nearly parallel to world up, 0.99 means angle is less than ~8.1 degrees. 0.142 radians.
            up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64) #change to Y from z up.

        r = np.cross(up_w, f); r, _ = self._normalize(r)       # camera +X (right)
        u = np.cross(f, r);   u, _ = self._normalize(u)        # camera +Y (up)

        R_w_c = np.column_stack((r, u, f))                     # columns: right, up, forward

        # ---- Uprightness enforcement (anti-upside-down) ----
        # Keep camera's up (~column 1) aligned with world up; if inverted, flip roll 180°.
        if np.dot(R_w_c[:, 1], up_w) > 0.0: #+y(cam) should be oppoesite to +z(world) for seedhe image.     
            R_w_c[:, 0] *= -1.0
            R_w_c[:, 1] *= -1.0
        return R_w_c

    def _yaw_pitch_toolX_quat(self, dx, dy, dz):
        """Fallback: aim tool +X via extrinsic ZYX (world yaw→pitch, roll=0)."""
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))  # dz<0 => negative pitch (tilt down)
        roll = 0.0
        q = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False).as_quat()  # x,y,z,w
        return q, (yaw, pitch, roll)

    # ---------------- Final alignment (integrated) ----------------
    def align_camera_to_marker_pose(self, marker_pose_map: PoseStamped):
        """Rotate in place so the camera optical axis (+Z) looks at the ArUco center. Single goal."""
        if self.sent_goal:
            return
        try:
            # Use consistent "latest" timestamps for TF lookups
            t_now_tool_base = self.tf_buffer.get_latest_common_time(self.BASE_FRAME, self.TOOL_LINK)
            t_now_cam_base  = self.tf_buffer.get_latest_common_time(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME)
            t_now_map_base  = self.tf_buffer.get_latest_common_time("map", self.BASE_FRAME)

            t_tool = self.tf_buffer.lookup_transform(self.BASE_FRAME, self.TOOL_LINK, t_now_tool_base, timeout=self.tf_timeout)
            t_cam  = self.tf_buffer.lookup_transform(self.BASE_FRAME, self.CAMERA_OPTICAL_FRAME, t_now_cam_base, timeout=self.tf_timeout)

            marker_pose_map_latest = PoseStamped()
            marker_pose_map_latest.header.frame_id = "map"
            marker_pose_map_latest.header.stamp = t_now_map_base.to_msg()
            marker_pose_map_latest.pose = marker_pose_map.pose

            marker_pose_base = self.tf_buffer.transform(
                marker_pose_map_latest, self.BASE_FRAME, timeout=self.tf_timeout
            )

            tool_p = np.array([
                t_tool.transform.translation.x,
                t_tool.transform.translation.y,
                t_tool.transform.translation.z
            ], dtype=np.float64)
            cam_p = np.array([
                t_cam.transform.translation.x,
                t_cam.transform.translation.y,
                t_cam.transform.translation.z
            ], dtype=np.float64)
            marker_p = np.array([
                marker_pose_base.pose.position.x,
                marker_pose_base.pose.position.y,
                marker_pose_base.pose.position.z
            ], dtype=np.float64)

            # Preferred: aim actual camera +Z at marker with upright roll
            use_camera_tf = False
            R_w_t_des = None
            ypr_deg = (0.0, 0.0, 0.0)

            try:
                t_tool_cam = self.tf_buffer.lookup_transform(
                    self.TOOL_LINK, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time(), timeout=self.tf_timeout
                )
                R_t_c = R.from_quat([
                    t_tool_cam.transform.rotation.x,
                    t_tool_cam.transform.rotation.y,
                    t_tool_cam.transform.rotation.z,
                    t_tool_cam.transform.rotation.w
                ]).as_matrix()
                R_c_t = R_t_c.T  # inverse

                R_w_c_des = self._look_at_cameraZ_worldUp(cam_p, marker_p)   # world->camera desired (upright)
                R_w_t_des = R_w_c_des @ R_c_t                                 # world->tool desired
                use_camera_tf = True

                ypr = R.from_matrix(R_w_t_des).as_euler('ZYX', degrees=True)
                ypr_deg = (float(ypr[0]), float(ypr[1]), float(ypr[2]))

            except Exception as e:
                if self.require_camera_tf:
                    self.get_logger().error(f'❌ Missing TF {self.TOOL_LINK}->{self.CAMERA_OPTICAL_FRAME}: {e}')
                    return
                else:
                    self.get_logger().warn('⚠️ Camera TF not available — fallback to tool +X aiming')
                    v = marker_p - tool_p
                    q, ypr_rad = self._yaw_pitch_toolX_quat(v[0], v[1], v[2])
                    ypr_deg = tuple([float(math.degrees(a)) for a in ypr_rad])
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
                "📋 MoveIt Goal Preview:\n"
                f"  Group          : {self.MOVE_GROUP}\n"
                f"  Link           : {self.TOOL_LINK}\n"
                f"  Base Frame     : {self.BASE_FRAME}\n"
                f"  Use Camera TF  : {use_camera_tf}\n"
                f"  Target Pos (m) : x={tool_p[0]:.4f}, y={tool_p[1]:.4f}, z={tool_p[2]:.4f}\n"
                f"  Target Q (xyzw): [{q_des[0]:.5f}, {q_des[1]:.5f}, {q_des[2]:.5f}, {q_des[3]:.5f}]\n"
                f"  Target YPR(deg): yaw={ypr_deg[0]:.2f}, pitch={ypr_deg[1]:.2f}, roll={ypr_deg[2]:.2f}\n"
                f"  Ori Tol (rad)  : {self.ORI_TOL:.3f}\n"
                f"  Pos Tol (m)    : {self.POS_TOL:.3f}"
            )

            self.send_rotate_in_place_goal(target_pose)
            self.sent_goal = True
            self.alignment_active = True

        except Exception as e:
            self.get_logger().error(f'❌ Align computation failed: {str(e)}')

    # ---------------- MoveIt: rotate-in-place constraints ----------------
    def send_rotate_in_place_goal(self, target_pose_stamped: PoseStamped):
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('❌ MoveIt2 action server not available!')
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        # Orientation constraint (tight)
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = self.ORI_TOL
        oc.absolute_y_axis_tolerance = self.ORI_TOL
        oc.absolute_z_axis_tolerance = self.ORI_TOL
        oc.weight = 1.0

        # Position constraint: small sphere around current tool pos (rotate in place)
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.POS_TOL]  # radius

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
            "🧾 MoveIt Request:\n"
            f"  goal_constraints: 1 (ori+pos)\n"
            f"  oc.link_name     : {oc.link_name}\n"
            f"  oc.tolerances    : [{oc.absolute_x_axis_tolerance:.3f}, {oc.absolute_y_axis_tolerance:.3f}, {oc.absolute_z_axis_tolerance:.3f}]\n"
            f"  pc.radius (m)    : {sphere.dimensions[0]:.3f}\n"
            f"  plan_only        : {goal_msg.planning_options.plan_only}"
        )

        self.get_logger().info("📤 Sending single rotate-in-place goal…")
        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.moveit_goal_response_callback)

    def moveit_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.alignment_active = False
            self.get_logger().error('❌ MoveIt2 goal rejected')
            return
        self.get_logger().info('✅ MoveIt2 goal accepted')
        try:
            self.yolo_node.set_enabled(True)
        except Exception as e:
            self.get_logger().error(f'Enable YOLO failed: {e}')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.moveit_result_callback)

    def moveit_result_callback(self, future):
        try:
            result = future.result().result
            self.alignment_active = False
            if result.error_code.val == 1:
                self.get_logger().info('✅ Camera aligned to marker (rotation complete)')
                # 👉 UR5e complete → base ko 0-vel pe actively hold karo
                self.hold_base_still_async(duration_sec=2.0, rate_hz=20.0)
            else:
                self.get_logger().error(f'❌ Rotate-in-place failed: error_code={result.error_code.val}')
        except Exception as e:
            self.alignment_active = False
            self.get_logger().error(f'❌ Result callback error: {e}')

    # ---------------- Camera feed / detection ----------------
    def compressed_image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error('❌ Failed to decode compressed image!')
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            detection_msg = Bool(data=False)

            if self.image_width and self.image_height:
                camera_center_x = self.image_width // 2
                camera_center_y = self.image_height // 2
            else:
                camera_center_x = cv_image.shape[1] // 2
                camera_center_y = cv_image.shape[0] // 2

            if ids is not None:
                detection_msg.data = True
                self.detection_count += 1
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

                if self.camera_matrix is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    for i, marker_id in enumerate(ids.flatten()):
                        center = corners[i][0].mean(axis=0)
                        center_x, center_y = int(center[0]), int(center[1])
                        cv2.circle(cv_image, (center_x, center_y), 8, (0, 0, 255), 3)

                        try:
                            camera_point = PointStamped()
                            camera_point.header.frame_id = self.CAMERA_OPTICAL_FRAME
                            camera_point.header.stamp = rclpy.time.Time().to_msg()  # "latest"
                            camera_point.point.x = float(tvecs[i][0][0])
                            camera_point.point.y = float(tvecs[i][0][1])
                            camera_point.point.z = float(tvecs[i][0][2])

                            map_point = self.tf_buffer.transform(
                                camera_point, "map", timeout=self.tf_timeout
                            )

                            map_pose = PoseStamped()
                            map_pose.header = map_point.header
                            map_pose.pose.position.x = map_point.point.x
                            map_pose.pose.position.y = map_point.point.y
                            map_pose.pose.position.z = map_point.point.z
                            map_pose.pose.orientation.w = 1.0

                            self.marker_pose_map = map_pose

                            if not self.navigation_active and not self.alignment_active and not self.sent_goal:
                                self.navigate_to_marker_front(map_pose.pose.position.x,
                                                              map_pose.pose.position.y,
                                                              marker_id, rvecs[i])
                        except Exception as e:
                            self.get_logger().error(f'❌ Transform failed: {str(e)}')

            self.detection_pub.publish(detection_msg)
            cv2.imshow('ArUco Detection + MoveIt rotate-in-place', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'❌ Image callback error: {str(e)}')

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    detector_node = ArucoDetectorWithMoveIt()
    yolo_node = detector_node.yolo_node
    exec = MultiThreadedExecutor(num_threads=2)  # camera+YOLO parallel
    exec.add_node(detector_node)
    exec.add_node(yolo_node)

    try:
        exec.spin()
    except KeyboardInterrupt:
        detector_node.get_logger().info('🛑 Shutting down...')
    finally:
        exec.shutdown()
        yolo_node.destroy_node()
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()