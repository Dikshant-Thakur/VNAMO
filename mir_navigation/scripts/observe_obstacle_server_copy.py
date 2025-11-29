#!/usr/bin/env python3
"""
FixedMarkerNavigator + ObserveObstacle Action

- Reuses existing stack:
    * rclpy
    * MoveIt2 ActionClient (/move_action, moveit_msgs/MoveGroup)
    * YOLO node (yolo_detector.YoloDetector)
    * TF2 (map → base → tool → camera)

- Behavior (when ObserveObstacle goal is accepted):

    1. Wait for TF to be ready
    2. Compute orientation to look at a fixed marker pose in map frame
    3. Send MoveIt goal (rotate-in-place) to point camera at marker
    4. Enable YOLO
    5. Wait for first YOLO detection (Bool + label)
    6. Disable YOLO
    7. Return manipulator to home pose
    8. Finish action with result:
       - success: bool
       - label: string (from YOLO)
       - message: debug text

- NOTE:
    * This version DOES NOT start push operation (no /start_push).
    * Result of the pipeline is an action result, not a topic trigger.
"""

import math
import threading
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.action import ActionServer, CancelResponse, GoalResponse

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, Vector3

from std_msgs.msg import Bool, String

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    BoundingVolume,
    JointConstraint,
    PlanningOptions,
)
from shape_msgs.msg import SolidPrimitive

import yolo_detector as yp

# Replace this import with your actual action package
# e.g. from your_pkg_name.action import ObserveObstacle
from mir_navigation.action import ObserveObstacle


class FixedMarkerNavigator(Node):
    """
    Node that:
      * Hosts an ObserveObstacle action server
      * Uses MoveIt + YOLO + TF to observe a fixed marker
      * Returns to home and reports YOLO label in the action result
    """

    BASE_FRAME = "ur_base_link"
    TOOL_LINK = "ur_tool0"
    CAMERA_OPTICAL_FRAME = "realsense_color_optical_frame"
    MOVE_GROUP = "ur5_manip"

    # Tolerances used for orientation/position constraints
    ORI_TOL = 0.05    # rad
    POS_TOL = 0.01    # m

    def __init__(self) -> None:
        super().__init__("fixed_marker_navigator")


        # --- FORCE SIM TIME FOR THIS NODE ---
        self.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True
            )
        ])
        self.get_logger().info(f"[SIM_TIME] FixedMarkerNavigator use_sim_time={self.get_parameter('use_sim_time').value}")




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

        # ---------------- Parameters ----------------
        # Hardcoded marker pose IN MAP FRAME (x, y, z) and quaternion [x, y, z, w]
        # These can be overridden from launch.
        #I think we need to create dictionary for multiple markers with obstacle name information.
        self.declare_parameter("marker_xyz", [6.25, 0.5, 0.25])
        self.declare_parameter("marker_q_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("approach_yaw_deg", None)

        m_xyz_param = self.get_parameter("marker_xyz").get_parameter_value()
        m_q_param = self.get_parameter("marker_q_xyzw").get_parameter_value()
        m_xyz = m_xyz_param.double_array_value
        m_q = m_q_param.double_array_value

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

        yaw_param = self.get_parameter("approach_yaw_deg")
        if yaw_param.type_ == rclpy.parameter.Parameter.Type.DOUBLE:
            self.approach_yaw_deg = float(yaw_param.value)
        else:
            self.approach_yaw_deg = None

        self.get_logger().info(f"[DBG_PARAM] approach_yaw_deg = {self.approach_yaw_deg}")
        # ParameterValue layout can differ (double_value / string_value), be defensive
        self.approach_yaw_deg = getattr(yaw_param, "double_value", None)

        # ---------------- TF setup ----------------
        self.tf_timeout = Duration(seconds=0.5)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_cache = {}
        self._tf_cache_ttl = 0.2  # seconds

        # ---------------- MoveIt & YOLO ----------------
        self.moveit_action_client = rclpy.action.ActionClient(self, MoveGroup, "/move_action")
        self.yolo_node = yp.YoloDetector()

        # --- FORCE SIM TIME FOR YOLO NODE TOO ---
        self.yolo_node.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True
            )
        ])
        self.get_logger().info("[SIM_TIME] YOLO node use_sim_time=True")
        self.yolo_node.set_enabled(False)

        # YOLO feedback topics
        self.yolo_seen = False
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

        # Small timer just to warm TF and log readiness
        self.create_timer(1.0, self._warmup_log)

        self.get_logger().info(
            "FixedMarkerNavigator with ObserveObstacle action server initialized "
            "(No Nav2 | Fixed marker | MoveIt rotate-in-place)."
        )

    # ============================================================
    #                         ACTION API
    # ============================================================

    def _on_goal(self, goal_request: ObserveObstacle.Goal) -> GoalResponse:
        if self._current_goal_handle is not None:
            self.get_logger().warn("Rejecting new ObserveObstacle goal: already processing one.")
            return GoalResponse.REJECT
        self.get_logger().info("ObserveObstacle goal received and accepted.")
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        self.get_logger().info("ObserveObstacle cancellation requested.")
        # We don't implement detailed cancel handling for now
        return CancelResponse.ACCEPT

    def _execute_observe_obstacle(self, goal_handle):
        """
        Main execution thread for ObserveObstacle.action.

        This callback is executed in a separate thread by rclpy's ActionServer.
        Here we:
          1. Reset state / events
          2. Align camera to marker pose
          3. Enable YOLO and wait for detection + home return
          4. Build and return ObserveObstacle.Result
        """
        self._current_goal_handle = goal_handle
        self._observation_done_event = threading.Event()
        self._observation_success = False
        self._observation_error = ""
        self.last_label = ""
        self.yolo_seen = False
        self.home_goal_active = False
        self.armed = False
        self.sent_goal = False

        # Optional: we can use goal fields in the future (e.g. obstacle name)
        obstacle_name = getattr(goal_handle.request, "obstacle_name", "")
        self.get_logger().info(
            f"[OBS_ACT] Starting observation pipeline for obstacle='{obstacle_name}'"
        )

        # 1) Wait for TF to be ready
        if not self._wait_for_tf_ready(timeout_sec=5.0):
            msg = "TF not ready, aborting observation."
            self.get_logger().error(msg)
            self._observation_error = msg
            self._observation_success = False
            self._observation_done_event.set()
        else:
            # 2) Kick alignment pipeline (async via MoveIt callbacks)
            try:
                self.align_camera_to_marker_pose(self.marker_pose_map)
            except Exception as exc:
                msg = f"Failed to start alignment pipeline: {exc}"
                self.get_logger().error(msg)
                self._observation_error = msg
                self._observation_success = False
                self._observation_done_event.set()

        # 3) Wait until callbacks mark observation done
        #    (Either success or error is set)
        if self._observation_done_event is not None:
            # You can add a timeout if you want to protect against hangs
            self._observation_done_event.wait()

        # Build result message
        result = ObserveObstacle.Result()
        result.success = bool(self._observation_success)
        result.label = self.last_label or ""
        if self._observation_success:
            result.message = f"Observation completed successfully. Label='{result.label}'"
            goal_handle.succeed()
        else:
            err = self._observation_error or "Observation failed."
            result.message = err
            goal_handle.abort()

        self.get_logger().info(
            f"[OBS_ACT] Finished ObserveObstacle goal | success={result.success} | "
            f"label='{result.label}' | msg='{result.message}'"
        )

        # Clear current goal handle
        self._current_goal_handle = None
        return result

    # ============================================================
    #                    HIGH LEVEL PIPELINE
    # ============================================================

    def _warmup_log(self):
        """Just logs once TF becomes available."""
        if self._tf_ready():
            self.get_logger().debug("TF ready (map→base→tool→camera).")
        # Timer is cheap, leaving it running is fine.

    # ------------------ ALIGNMENT (MOVEIT) ------------------

    def align_camera_to_marker_pose(self, marker_pose_map: PoseStamped):
        """
        Compute orientation so that camera/tool looks at marker and send MoveIt goal.

        This uses:
          - map → base transform
          - base → tool transform
          - base → camera transform (optional)
        """
        if self.sent_goal:
            self.get_logger().warn("Alignment already in progress, ignoring new request.")
            return

        if not self._tf_ready():
            raise RuntimeError("TF not ready, cannot plan alignment.")

        # Refresh marker pose stamp
        marker_pose_map_latest = PoseStamped()
        marker_pose_map_latest.header.stamp = self.get_clock().now().to_msg()
        marker_pose_map_latest.header.frame_id = marker_pose_map.header.frame_id
        marker_pose_map_latest.pose = marker_pose_map.pose

        # Transform marker pose into base frame
        marker_pose_base = self.tf_buffer.transform(
            marker_pose_map_latest, self.BASE_FRAME, timeout=self.tf_timeout
        )

        # Lookup tool and camera in base frame
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


        v_cam_to_marker = marker_p - cam_p
        v_tool_to_marker = marker_p - tool_p

        self.get_logger().info(
            "[DBG_POS] marker_p(base)=(%.3f, %.3f, %.3f), cam_p=(%.3f, %.3f, %.3f), tool_p=(%.3f, %.3f, %.3f)"
            % (marker_p[0], marker_p[1], marker_p[2],
            cam_p[0], cam_p[1], cam_p[2],
            tool_p[0], tool_p[1], tool_p[2])
        )

        self.get_logger().info(
            "[DBG_VEC] v_cam_to_marker=(%.3f, %.3f, %.3f), v_tool_to_marker=(%.3f, %.3f, %.3f)"
            % (v_cam_to_marker[0], v_cam_to_marker[1], v_cam_to_marker[2],
            v_tool_to_marker[0], v_tool_to_marker[1], v_tool_to_marker[2])
        )

        # Decide orientation:
        # - If approach_yaw_deg is set, use that yaw + some default pitch
        # - Otherwise, use "look-at" direction from camera/tool to marker
        use_camera_tf = False
        if self.approach_yaw_deg is None:
            try:
                t_cam_tool = self._tf_lookup_cached(self.CAMERA_OPTICAL_FRAME, self.TOOL_LINK)

                                # ========== DEBUG: TF Quaternion ==========
                self.get_logger().info(
                    "[DBG_TF] cam->tool quat = [%.4f, %.4f, %.4f, %.4f]"
                    % (t_cam_tool.transform.rotation.x,
                    t_cam_tool.transform.rotation.y,
                    t_cam_tool.transform.rotation.z,
                    t_cam_tool.transform.rotation.w)
                )

                self.get_logger().info(
                    "[DBG_TF] base->cam trans = (%.3f, %.3f, %.3f)"
                    % (t_cam.transform.translation.x,
                    t_cam.transform.translation.y,
                    t_cam.transform.translation.z)
                )

                self.get_logger().info(
                    "[DBG_TF] base->tool trans = (%.3f, %.3f, %.3f)"
                    % (t_tool.transform.translation.x,
                    t_tool.transform.translation.y,
                    t_tool.transform.translation.z)
                )

                R_t_c = R.from_quat(
                    [
                        t_cam_tool.transform.rotation.x,
                        t_cam_tool.transform.rotation.y,
                        t_cam_tool.transform.rotation.z,
                        t_cam_tool.transform.rotation.w,
                    ]
                ).as_matrix()
                use_camera_tf = True
            except Exception as exc:
                self.get_logger().warn(f"Failed to get tool→camera TF, fallback to toolX: {exc}")

        # Vector from camera to marker
        v_cam_to_marker = marker_p - cam_p
        # Vector from tool to marker
        v_tool_to_marker = marker_p - tool_p

        if self.approach_yaw_deg is not None:
            # Simple yaw-based orientation around Z axis
            yaw_rad = math.radians(self.approach_yaw_deg)
            q_des = R.from_euler("z", yaw_rad).as_quat()
        else:
            if use_camera_tf:
                # 1) Desired CAMERA orientation in base frame
                q_cam_des = self._lookat_cameraZ_quat(v_cam_to_marker)
                # ========== DEBUG: Desired Camera Rotation ==========
                R_base_cam_des = R.from_quat(q_cam_des)
                cam_fwd_des = R_base_cam_des.apply([0, 0, 1])   # optical Z-axis

                v_cam_norm, _ = self._normalize(v_cam_to_marker)

                self.get_logger().info(
                    "[DBG_CAM_DES] q_cam_des = [%.4f, %.4f, %.4f, %.4f]"
                    % (q_cam_des[0], q_cam_des[1], q_cam_des[2], q_cam_des[3])
                )

                self.get_logger().info(
                    "[DBG_CAM_DES] cam_fwd_des(base) = (%.3f, %.3f, %.3f)"
                    % (cam_fwd_des[0], cam_fwd_des[1], cam_fwd_des[2])
                )

                angle_cam = math.degrees(math.acos(np.clip(np.dot(cam_fwd_des, v_cam_norm), -1.0, 1.0)))
                self.get_logger().info("[DBG_CAM_DES] angle(cam_fwd_des vs marker_vec) = %.2f deg" % angle_cam)


                    # 2) Current tool->camera rotation (from TF)
                q_cam_tool = [
                    t_cam_tool.transform.rotation.x,
                    t_cam_tool.transform.rotation.y,
                    t_cam_tool.transform.rotation.z,
                    t_cam_tool.transform.rotation.w,
                ]

                # 3) Convert: R_base_tool_des = R_base_cam_des * inv(R_cam_tool)
                R_base_cam_des = R.from_quat(q_cam_des)
                R_cam_tool = R.from_quat(q_cam_tool)
                R_tool_cam = R_cam_tool.inv()
                R_base_tool_des = R_base_cam_des * R_tool_cam
 

                # 4) Final TOOL orientation to send MoveIt
                q_des = R_base_tool_des.as_quat()
                # ========== DEBUG: Final Tool Quaternion & Predicted Camera Forward ==========
                R_base_tool = R.from_quat(q_des)

                R_cam_tool = R.from_quat([
                    t_cam_tool.transform.rotation.x,
                    t_cam_tool.transform.rotation.y,
                    t_cam_tool.transform.rotation.z,
                    t_cam_tool.transform.rotation.w
                ])
                R_tool_cam = R_cam_tool.inv()

                # predict camera forward after tool rotates
                R_base_cam_pred = R_base_tool * R_tool_cam
                cam_fwd_pred = R_base_cam_pred.apply([0, 0, 1])

                angle_pred = math.degrees(math.acos(np.clip(np.dot(cam_fwd_pred, v_cam_norm), -1.0, 1.0)))

                self.get_logger().info(
                    "[DBG_TOOL_DES] q_tool_des = [%.4f, %.4f, %.4f, %.4f]"
                    % (q_des[0], q_des[1], q_des[2], q_des[3])
                )

                self.get_logger().info(
                    "[DBG_CAM_PRED] cam_fwd_pred(base) = (%.3f, %.3f, %.3f), angle_pred = %.2f deg"
                    % (cam_fwd_pred[0], cam_fwd_pred[1], cam_fwd_pred[2], angle_pred)
                )

            else:
                # Make tool X point to marker
                self.get_logger().warn("Using tool frame to compute look-at orientation.")
                q_des = self._yaw_pitch_toolX_quat(*v_tool_to_marker)

        target_pose_base = PoseStamped()
        target_pose_base.header.frame_id = self.BASE_FRAME
        target_pose_base.pose.position.x = float(tool_p[0])
        target_pose_base.pose.position.y = float(tool_p[1])
        target_pose_base.pose.position.z = float(tool_p[2])
        target_pose_base.pose.orientation.x = float(q_des[0])
        target_pose_base.pose.orientation.y = float(q_des[1])
        target_pose_base.pose.orientation.z = float(q_des[2])
        target_pose_base.pose.orientation.w = float(q_des[3])

        self.get_logger().info(
            "Planning MoveIt alignment:\n"
            f"  Marker (base) : [{marker_p[0]:.3f}, {marker_p[1]:.3f}, {marker_p[2]:.3f}]\n"
            f"  Tool (base)   : [{tool_p[0]:.3f}, {tool_p[1]:.3f}, {tool_p[2]:.3f}]\n"
            f"  Cam  (base)   : [{cam_p[0]:.3f}, {cam_p[1]:.3f}, {cam_p[2]:.3f}]\n"
            f"  Desired quat  : [{q_des[0]:.4f}, {q_des[1]:.4f}, {q_des[2]:.4f}, {q_des[3]:.4f}]"
        )

        self._send_moveit_alignment_goal(target_pose_base)

    def _send_moveit_alignment_goal(self, target_pose_stamped: PoseStamped):
        """
        Build and send MoveIt MotionPlanRequest to rotate-in-place at current tool position
        while constraining orientation and a small position tolerance sphere.
        """
        req = MotionPlanRequest()
        req.group_name = self.MOVE_GROUP

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header = target_pose_stamped.header
        oc.link_name = self.TOOL_LINK
        oc.orientation = target_pose_stamped.pose.orientation
        oc.absolute_x_axis_tolerance = self.ORI_TOL
        oc.absolute_y_axis_tolerance = self.ORI_TOL
        oc.absolute_z_axis_tolerance = self.ORI_TOL
        oc.weight = 1.0

        # Position constraint (small sphere around current tool position)
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

        goal_msg = MoveGroup.Goal()
        goal_msg.request = req
        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False

        self.get_logger().info("Sending MoveIt alignment goal to /move_action...")
        self.sent_goal = True
        fut = self.moveit_action_client.send_goal_async(goal_msg)
        fut.add_done_callback(self._alignment_goal_response_cb)

    def _alignment_goal_response_cb(self, fut):
        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            msg = "Alignment MoveIt goal rejected."
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return
        self.get_logger().info("Alignment MoveIt goal accepted.")
        rf = goal_handle.get_result_async()
        rf.add_done_callback(self._alignment_result_cb)

    def _alignment_result_cb(self, fut):
        try:
            res = fut.result().result
            if res.error_code.val == 1:
                self.get_logger().info("Alignment succeeded → enabling YOLO and arming detection.")
                self._reset_and_arm_yolo()
            else:
                msg = f"Alignment failed: error_code={res.error_code.val}"
                self.get_logger().error(msg)
                self._fail_observation(msg)
        except Exception as exc:
            msg = f"Alignment result error: {exc}"
            self.get_logger().error(msg)
            self._fail_observation(msg)

    # -------------------- YOLO handling --------------------

    def _reset_and_arm_yolo(self):
        self.yolo_seen = False
        self.armed = True
        self.home_goal_active = False
        try:
            self.yolo_node.set_enabled(True)
            self.get_logger().info("[YOLO] Enabled and waiting for detection...")
        except Exception as exc:
            msg = f"Failed to enable YOLO: {exc}"
            self.get_logger().warn(msg)
            self._fail_observation(msg)

    def _on_yolo_detected(self, msg: Bool):
        if not msg.data:
            return
        if not self.armed or self.home_goal_active:
            return
        self.armed = False
        self.yolo_seen = True

        try:
            self.yolo_node.set_enabled(False)
            self.get_logger().info("[YOLO] Detection True → disabling YOLO and returning to home pose.")
        except Exception as exc:
            self.get_logger().warn(f"Failed to disable YOLO: {exc}")

        if not self.home_goal_active:
            self.home_goal_active = True
            self._send_return_to_home_goal()

    def _on_yolo_label(self, msg: String):
        label = msg.data or ""
        if label:
            self.last_label = label
            self.get_logger().info(f"[YOLO] Label received: {label}")

    # ----------------- Return to home pose -----------------

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

        self.get_logger().info("Sending MoveIt home pose goal to /move_action...")
        fut = self.moveit_action_client.send_goal_async(goal)
        fut.add_done_callback(self._home_position_goal_response_cb)

    def _home_position_goal_response_cb(self, fut):
        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            msg = "Home pose MoveIt goal rejected."
            self.get_logger().error(msg)
            self._fail_observation(msg)
            return
        self.get_logger().info("Home pose MoveIt goal accepted.")
        rf = goal_handle.get_result_async()
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
                self._fail_observation(msg)
        except Exception as exc:
            msg = f"Home return result error: {exc}"
            self.get_logger().error(msg)
            self._fail_observation(msg)

    # ============================================================
    #                      OBSERVATION RESULT UTILS
    # ============================================================

    def _succeed_observation(self):
        self._observation_success = True
        if self._observation_done_event is not None:
            self._observation_done_event.set()

    def _fail_observation(self, msg: str):
        self._observation_success = False
        self._observation_error = msg
        if self._observation_done_event is not None:
            self._observation_done_event.set()

    # ============================================================
    #                         TF HELPERS
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
            rclpy.sleep(0.1)
        return self._tf_ready()

    def _tf_lookup_cached(self, target_frame: str, source_frame: str):
        key = (target_frame, source_frame)
        now = self.get_clock().now().nanoseconds * 1e-9
        t_cached = self._tf_cache.get(key)
        if t_cached is not None:
            t_stamp, transform = t_cached
            if now - t_stamp < self._tf_cache_ttl:
                return transform

        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=self.tf_timeout,
        )
        self._tf_cache[key] = (now, transform)
        return transform

    # ============================================================
    #                  ORIENTATION MATH HELPERS
    # ============================================================

    @staticmethod
    def _normalize(v: np.ndarray) -> Tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return v, 0.0
        return v / n, n

    def _lookat_cameraZ_quat(self, v_cam_to_marker: np.ndarray) -> np.ndarray:
        f, n = self._normalize(v_cam_to_marker)
        if n < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # world up
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # if f is too close to up, pick different up
        if abs(np.dot(f, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # build right-handed basis
        s = np.cross(up, f)
        s, _ = self._normalize(s)

        u = np.cross(f, s)

        R_w_c = np.column_stack((s, u, f))

        # force right-handed: det must be +1
        if np.linalg.det(R_w_c) < 0:
            s = -s
            R_w_c = np.column_stack((s, u, f))

        return R.from_matrix(R_w_c).as_quat().astype(np.float64)



    def _yaw_pitch_toolX_quat(self, vx: float, vy: float, vz: float) -> np.ndarray:
        """
        Build quaternion so that tool +X roughly points towards marker.
        """
        v = np.array([vx, vy, vz], dtype=np.float64)
        v, n = self._normalize(v)
        if n < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        yaw = math.atan2(v[1], v[0])
        pitch = math.atan2(v[2], max(1e-6, math.sqrt(v[0] ** 2 + v[1] ** 2)))
        q = R.from_euler("zyx", [yaw, -pitch, 0.0]).as_quat()
        return q.astype(np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = FixedMarkerNavigator()
    yolo_node = node.yolo_node

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.add_node(yolo_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down FixedMarkerNavigator.")
    finally:
        executor.shutdown()
        yolo_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()