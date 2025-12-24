#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Pose, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

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
from moveit_msgs.srv import GetCartesianPath

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from rclpy.duration import Duration
from rclpy.time import Time


class SimpleRodGrab(Node):
    """
    Pipeline:
      1) Map-frame pre-grasp pose -> transform to planning_frame
      2) Move to pre-grasp
      3) Open gripper
      4) Move +Y in planning_frame
      5) Close gripper
    """

    def __init__(self):
        super().__init__("simple_rod_grab")

        # Use simulation time by default (can be overridden via ROS params)
        # self.declare_parameter("use_sim_time", True)

        # --- Parameters ---
        self.moveit_group_name = (
            self.declare_parameter("moveit_group_name", "ur5_manip")
            .get_parameter_value()
            .string_value
        )
        self.planning_frame = (
            self.declare_parameter("planning_frame", "ur_base_link")
            .get_parameter_value()
            .string_value
        )
        self.ee_link = (
            self.declare_parameter("ee_link", "ur_tool0")
            .get_parameter_value()
            .string_value
        )

        # Pre-grasp pose GIVEN IN /map FRAME (tumhare numbers yahan)
        self.pre_map_x = (
            self.declare_parameter("pre_map_x", 0.614 )
            .get_parameter_value()
            .double_value
        )
        self.pre_map_y = (
            self.declare_parameter("pre_map_y",  -0.091 )
            .get_parameter_value()
            .double_value
        )
        self.pre_map_z = (
            self.declare_parameter("pre_map_z", 1.437 )
            .get_parameter_value()
            .double_value
        )

        # Approach offset in planning_frame (+Y direction)
        self.approach_delta_y = (
            self.declare_parameter("approach_delta_y", 0.05)
            .get_parameter_value()
            .double_value
        )

                # Gripper config (controller = gripper_controller)
        self.gripper_action_name = (
            self.declare_parameter(
                "gripper_action_name", "gripper_controller/follow_joint_trajectory"
            )
            .get_parameter_value()
            .string_value
        )

        # FollowJointTrajectory gripper_controller controls BOTH finger joints
        self.gripper_joint_names = [
            "gripper_soft_robotics_gripper_left_finger_joint1",
            "gripper_soft_robotics_gripper_right_finger_joint1",
        ]

        self.gripper_open = (
            self.declare_parameter("gripper_open", 0.25)
            .get_parameter_value()
            .double_value
        )
        self.gripper_close = (
            self.declare_parameter("gripper_close", -0.25)
            .get_parameter_value()
            .double_value
        )


        # --- TF ---
        # Keep a short history in the TF buffer to make lookups more robust
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Action clients ---
        self.move_action_name = (
            self.declare_parameter("move_action_name", "move_action")
            .get_parameter_value()
            .string_value
        )
        self.movegroup_ac = ActionClient(self, MoveGroup, self.move_action_name)
        self.gripper_ac = ActionClient(
            self, FollowJointTrajectory, self.gripper_action_name
        )
                # Cartesian path service (for linear EE motion)
        # NOTE: topic name may be remapped in your bringup; adjust if needed.
        self.cartesian_client = self.create_client(
            GetCartesianPath, "compute_cartesian_path"
        )

        # Direct FollowJointTrajectory client for the arm
        self.arm_traj_ac = ActionClient(
            self,
            FollowJointTrajectory,
            "joint_trajectory_controller/follow_joint_trajectory",
        )

        self.get_logger().info(
            f"[INIT] group={self.moveit_group_name}, planning_frame={self.planning_frame}, ee_link={self.ee_link}"
        )
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        frame_id: Optional[str] = None,
    ) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = frame_id or self.planning_frame
        # ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.stamp = Time().to_msg()  # ya 0 time bhi rakh sakte ho
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = float(z)
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        return ps

    def _transform_pose(
        self, pose_stamped: PoseStamped, target_frame: str, timeout_sec: float = 1.0
    ) -> Optional[PoseStamped]:
        """Transform pose_stamped to target_frame using TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                pose_stamped.header.frame_id,
                Time(),  # rclpy.time.Time object
                timeout=Duration(seconds=timeout_sec),  # rclpy.duration.Duration
            )

        except TransformException as ex:
            self.get_logger().error(
                f"[TF] Failed to transform {pose_stamped.header.frame_id} -> {target_frame}: {ex}"
            )
            return None

        # ❗ Option A: do_transform_pose simple Pose pe chalega, PoseStamped pe nahi
        pose_out = do_transform_pose(pose_stamped.pose, tf)

        out = PoseStamped()
        out.header = pose_stamped.header
        out.header.frame_id = target_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose = pose_out
        return out

    
    def _wait_for_tf(
        self,
        target_frame: str,
        source_frame: str,
        max_wait_sec: float = 10.0,
        check_interval_sec: float = 0.2,
    ) -> bool:
        """Actively wait until a TF transform between source_frame and target_frame is available."""
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time) < Duration(seconds=max_wait_sec):
            try:
                # We only care that the transform exists; we discard the value here.
                self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.5),
                )
                self.get_logger().info(
                    f"[TF] Transform available: {source_frame} -> {target_frame}"
                )
                return True
            except TransformException as ex:
                self.get_logger().warn(
                    f"[TF] Waiting for transform {source_frame} -> {target_frame}: {ex}"
                )
                rclpy.spin_once(self, timeout_sec=check_interval_sec)

        self.get_logger().error(
            f"[TF] Transform {source_frame} -> {target_frame} not available after {max_wait_sec:.1f}s"
        )
        return False

    def _build_pose_goal(
        self, pose_stamped: PoseStamped, relaxed: bool = False
    ) -> MoveGroup.Goal:
        if relaxed:
            ori_tol_deg = 15.0
            pos_radius_m = 0.08
            plan_time = 3.0
        else:
            ori_tol_deg = 5.0
            pos_radius_m = 0.02
            plan_time = 2.0

        # Orientation constraint
        oc = OrientationConstraint()
        oc.header = pose_stamped.header
        oc.link_name = self.ee_link
        oc.orientation = pose_stamped.pose.orientation
        tol_rad = math.radians(ori_tol_deg)
        oc.absolute_x_axis_tolerance = tol_rad
        oc.absolute_y_axis_tolerance = tol_rad
        oc.absolute_z_axis_tolerance = tol_rad
        oc.weight = 1.0

        # Position constraint = small sphere around desired EE position
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_radius_m]

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
        req.allowed_planning_time = plan_time
        req.num_planning_attempts = 1

        opts = PlanningOptions()
        opts.plan_only = False
        opts.look_around = False
        opts.replan = False

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = opts
        return goal

    def _execute_moveit_pose(
        self, pose_stamped: PoseStamped, relaxed: bool = False, timeout_sec: float = 20.0
    ) -> bool:
        if not self.movegroup_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[MOVE] MoveGroup action server not available")
            return False

        goal = self._build_pose_goal(pose_stamped, relaxed=relaxed)
        self.get_logger().info(
            f"[MOVE] Sending goal to ({pose_stamped.pose.position.x:.3f}, "
            f"{pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f})"
        )

        send_future = self.movegroup_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout_sec)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("[MOVE] Goal rejected")
            return False

        self.get_logger().info("[MOVE] Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        res = result_future.result()
        if res is None:
            self.get_logger().error("[MOVE] Result future returned None")
            return False

        ok = getattr(res.result.error_code, "val", 0) == 1
        self.get_logger().info(f"[MOVE] Done: {'OK' if ok else 'FAIL'}")
        return bool(ok)


    def _execute_linear_approach(
        self,
        start_pose: PoseStamped,
        delta_y: float,
        eef_step: float = 0.01,
        jump_threshold: float = 0.0,
        timeout_sec: float = 20.0,
    ) -> bool:
        """Move EE in a (approximately) straight line along +Y by delta_y."""

        # 1) Wait for Cartesian path service
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("[CART] compute_cartesian_path service not available")
            return False

        # 2) Build GetCartesianPath request
        req = GetCartesianPath.Request()
        req.group_name = self.moveit_group_name
        req.link_name = self.ee_link
        req.max_step = float(eef_step)        # max EE step (meters) between points
        req.jump_threshold = float(jump_threshold)
        req.avoid_collisions = True

        req.header.frame_id = self.planning_frame
        req.header.stamp = self.get_clock().now().to_msg()

        # Start state = current robot state (leave default / empty -> MoveIt uses current)
        # Waypoints: start_pose, then same pose with y += delta_y
        start = Pose()
        start.position = start_pose.pose.position
        start.orientation = start_pose.pose.orientation

        end = Pose()
        end.position.x = start.position.x
        end.position.y = start.position.y + float(delta_y)
        end.position.z = start.position.z
        end.orientation = start.orientation

        req.waypoints = [start, end]

        self.get_logger().info(
            f"[CART] Requesting linear approach of {delta_y:.3f} m along +Y "
            f"in frame {self.planning_frame}"
        )

        # 3) Call service
        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        res = future.result()
        if res is None:
            self.get_logger().error("[CART] No response from compute_cartesian_path")
            return False

        if not res.solution.joint_trajectory.points:
            self.get_logger().error(
                f"[CART] Empty trajectory, fraction={res.fraction:.3f}"
            )
            return False

        if res.fraction < 0.99:
            self.get_logger().warn(
                f"[CART] Cartesian path incomplete: fraction={res.fraction:.3f}"
            )
            # Decide: you can return False here if you want strict behavior
            # return False

        traj = res.solution.joint_trajectory

        # 4) Execute the joint trajectory on the arm controller
        if not self.arm_traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "[CART] Arm FollowJointTrajectory action server not available"
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"[CART] Executing Cartesian trajectory with "
            f"{len(traj.points)} points on joints={traj.joint_names}"
        )

        send_future = self.arm_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout_sec)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("[CART] Cartesian goal rejected by controller")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)

        result = result_future.result()
        if result is None:
            self.get_logger().error("[CART] No result from arm trajectory action")
            return False

        self.get_logger().info("[CART] Cartesian approach completed")
        return True


    def _execute_gripper(self, position: float, duration: float = 1.0) -> bool:
        """FollowJointTrajectory for BOTH gripper joints."""
        if not self.gripper_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[GRIP] Gripper action server not available")
            return False

        traj = JointTrajectory()
        traj.joint_names = list(self.gripper_joint_names)

        pt = JointTrajectoryPoint()
        # Same position on both fingers (symmetrical gripper)
        pt.positions = [float(position)] * len(self.gripper_joint_names)
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"[GRIP] Sending gripper position={position:.3f} to joints={self.gripper_joint_names}"
        )
        send_future = self.gripper_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("[GRIP] Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=10.0)

        res = result_future.result()
        if res is None:
            self.get_logger().error("[GRIP] No result from gripper action")
            return False

        self.get_logger().info("[GRIP] Gripper action finished")
        return True

    # ------------------------------------------------------------------
    # Main sequence
    # ------------------------------------------------------------------
    def execute_sequence(self) -> bool:
        self.get_logger().info("[SEQ] Starting rod grab sequence")

        # 1) Pre-grasp pose in /map frame (tumhare numbers)
        pre_pose_map = self._make_pose(
            self.pre_map_x,
            self.pre_map_y,
            self.pre_map_z,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
            frame_id="map",
        )

        # Transform /map -> planning_frame (e.g. ur_base_link)
        pre_pose_planning = self._transform_pose(
            pre_pose_map, self.planning_frame, timeout_sec=1.0
        )
        if pre_pose_planning is None:
            self.get_logger().error("[SEQ] Failed to transform pre-grasp pose to planning_frame")
            return False

        # 1) Move to pre-grasp (planning_frame)
        if not self._execute_moveit_pose(pre_pose_planning, relaxed=False):
            self.get_logger().error("[SEQ] Failed to reach pre-grasp pose")
            return False

        # 2) Open gripper
        if not self._execute_gripper(self.gripper_open):
            self.get_logger().error("[SEQ] Failed to open gripper")
            return False

        # 3) Approach along +Y in planning_frame
        approach_pose = PoseStamped()
        approach_pose.header.frame_id = self.planning_frame
        approach_pose.header.stamp = self.get_clock().now().to_msg()
        approach_pose.pose = pre_pose_planning.pose

        approach_pose.pose.position.y += self.approach_delta_y

        # 3) Linear approach along +Y in planning_frame (Cartesian path)
        if not self._execute_linear_approach(pre_pose_planning, self.approach_delta_y):
            self.get_logger().error("[SEQ] Failed to do linear approach motion")
            return False

        # 4) Close gripper
        if not self._execute_gripper(self.gripper_close):
            self.get_logger().error("[SEQ] Failed to close gripper")
            return False

        self.get_logger().info("[SEQ] Rod grab sequence completed successfully ✅")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRodGrab()

    # Wait until the transform map -> planning_frame is available.
    node.get_logger().info(
        f"[TF] Waiting for transform map -> {node.planning_frame} before starting sequence..."
    )
    if not node._wait_for_tf(node.planning_frame, "map", max_wait_sec=10.0):
        node.get_logger().error("[MAIN] TF not ready, aborting sequence.")
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        ok = node.execute_sequence()
        if not ok:
            node.get_logger().error("[MAIN] Sequence failed")
        else:
            node.get_logger().info("[MAIN] Sequence SUCCESS")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
