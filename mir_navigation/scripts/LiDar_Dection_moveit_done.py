#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import String, Bool
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped

from nav2_simple_commander import robot_navigator
import tf2_geometry_msgs
from nav2_simple_commander.robot_navigator import TaskResult

# MoveIt2 Action imports
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, PlanningOptions, Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive



import math

class ArucoDetectorWithMoveIt(Node):
    def __init__(self):
        super().__init__('aruco_detector_moveit_node')
        self.bridge = CvBridge()
        self.navigator = robot_navigator.BasicNavigator()
        self.navigator.waitUntilNav2Active()

        self.moveit_action_client = ActionClient(self, MoveGroup, '/move_action')
        self.marker_distance_offset_scale = 1.5
        self.navigation_active = False
        self.alignment_active = False
        self.marker_pose_map = None

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.5

        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        self.alignment_tolerance = 0.05 # meters (5cm bot-bot error bhi modify kar sakte ho)
        self.get_logger().info('🤖 ArUco Detector with MoveIt2 Pose Goal Started!')

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info('✅ Camera calibration received!')

    def navigate_to_marker_front(self, map_x, map_y, marker_id, rvecs):
        try:
            rotation_matrix, _ = cv2.Rodrigues(rvecs)
            face_z_axis = rotation_matrix[:, 2]
            face_normal_z_unit = face_z_axis / np.linalg.norm(face_z_axis)

            z = Vector3Stamped()
            z.header.frame_id = "realsense_color_optical_frame"
            latest_time = self.tf_buffer.get_latest_common_time("realsense_color_optical_frame", "map")
            z.header.stamp = latest_time.to_msg()
            z.vector.x, z.vector.y, z.vector.z = float(face_normal_z_unit[0]), float(face_normal_z_unit[1]), float(face_normal_z_unit[2])
            map_normal = self.tf_buffer.transform(z, "map")
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
            self.get_logger().info(f'🎯 Navigating to marker {marker_id} front position: ({goal_x:.2f}, {goal_y:.2f})')
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
                # Now pose alignment
                if self.marker_pose_map:
                    self.align_camera_to_marker_pose(self.marker_pose_map)
            else:
                self.get_logger().warn(f'⚠️ Navigation failed with result: {result}')
                self.navigation_active = False
    def align_camera_to_marker_pose(self, marker_pose_map):
        try:
            
            marker_pose_base = self.tf_buffer.transform(marker_pose_map, "ur_base_link")
            
            # 🔍 DEBUG: Values print karo
            self.get_logger().info(f'Original marker pose (map): x={marker_pose_map.pose.position.x:.3f}, y={marker_pose_map.pose.position.y:.3f}, z={marker_pose_map.pose.position.z:.3f}')
            self.get_logger().info(f'Transformed pose (base): x={marker_pose_base.pose.position.x:.3f}, y={marker_pose_base.pose.position.y:.3f}, z={marker_pose_base.pose.position.z:.3f}')
            
            target_pose = PoseStamped()
            target_pose.header.frame_id = "ur_base_link"
            target_pose.pose.position.x = marker_pose_base.pose.position.x
            target_pose.pose.position.y = marker_pose_base.pose.position.y - 1.8
            target_pose.pose.position.z = marker_pose_base.pose.position.z 
            
            # 🔍 Final target values
            self.get_logger().info(f'Final target: x={target_pose.pose.position.x:.3f}, y={target_pose.pose.position.y:.3f}, z={target_pose.pose.position.z:.3f}')
            self.send_pose_goal_moveit2(target_pose)
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            return


    def send_pose_goal_moveit2(self, target_pose_stamped):
        if not self.moveit_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('❌ MoveIt2 action server not available!')
            return

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "ur5_manip" # <<-- modify if your group_name is different!

        # Position constraint (sphere of radius epsilon at target)
        pc = PositionConstraint()
        pc.header = target_pose_stamped.header
        pc.link_name = "ur_tool0" # <<-- set your end-effector link here!
        pc.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.05])
        )
        pc.constraint_region.primitive_poses.append(target_pose_stamped.pose) # Target position for constraint
        pc.weight = 1.0 #Constraint Importance

        # Optional: Orientation constraint (here: have a fixed orientation, generally look-at quaternion)
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = "ur_tool0" # <<-- same here!
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = 0.35 # radians
        oc.absolute_y_axis_tolerance = 0.35
        oc.absolute_z_axis_tolerance = 0.5
        oc.weight = 1.0

        goal_constraint = Constraints()
        goal_constraint.position_constraints = [pc]
        goal_constraint.orientation_constraints = [oc]
        req.goal_constraints = [goal_constraint]

        goal_msg.request = req
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 2

        self.get_logger().info("📤 Sending MoveIt2 pose goal...")
        future = self.moveit_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.moveit_goal_response_callback)

    def moveit_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ MoveIt2 pose goal rejected!')
            return
        self.get_logger().info('✅ MoveIt2 pose goal accepted!')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.moveit_result_callback)

    def moveit_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == 1:
            self.get_logger().info('✅ End-effector aligned with marker pose!')
        else:
            self.get_logger().error(f'❌ MoveIt2 execution failed with error: {result.error_code.val}')

    def compressed_image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error('❌ Failed to decode compressed image!')
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            detection_msg = Bool()
            detection_msg.data = False

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

                for i, marker_id in enumerate(ids.flatten()):
                    center = corners[i][0].mean(axis=0)
                    center_x, center_y = int(center[0]), int(center[1])
                    cv2.circle(cv_image, (center_x, center_y), 8, (0, 0, 255), 3)
                    # Pose estimation
                    if self.camera_matrix is not None:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                        )
                        distance = np.linalg.norm(tvecs[i])
                        try:
                            camera_point = PointStamped()
                            camera_point.header.frame_id = "realsense_color_optical_frame"
                            latest_time = self.tf_buffer.get_latest_common_time("realsense_color_optical_frame", "map")
                            camera_point.header.stamp = latest_time.to_msg()
                            camera_point.point.x = float(tvecs[i][0][0])
                            camera_point.point.y = float(tvecs[i][0][1])
                            camera_point.point.z = float(tvecs[i][0][2])
                            map_point = self.tf_buffer.transform(camera_point, "map")

                            map_pose = PoseStamped()
                            map_pose.header = map_point.header
                            map_pose.pose.position.x = map_point.point.x
                            map_pose.pose.position.y = map_point.point.y
                            map_pose.pose.position.z = map_point.point.z
                            # Orientation: not estimated from marker here (assume camera Z towards marker)
                            map_pose.pose.orientation.x = 0.0
                            map_pose.pose.orientation.y = 0.0
                            map_pose.pose.orientation.z = 0.0
                            map_pose.pose.orientation.w = 1.0

                            self.marker_pose_map = map_pose

                            if not self.navigation_active and not self.alignment_active:
                                self.navigate_to_marker_front(map_pose.pose.position.x, map_pose.pose.position.y, marker_id, rvecs[i])

                        except Exception as e:
                            self.get_logger().error(f'❌ Transform failed: {str(e)}')

            self.detection_pub.publish(detection_msg)
            cv2.imshow('ArUco Detection with MoveIt2 Pose', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'❌ Image callback error: {str(e)}')

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    detector_node = ArucoDetectorWithMoveIt()
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info('🛑 Shutting down ArUco Detector with MoveIt2...')
    finally:
        cv2.destroyAllWindows()
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
