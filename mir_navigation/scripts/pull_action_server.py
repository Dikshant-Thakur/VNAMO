#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point, Twist
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

from moveit_msgs.msg import CollisionObject, Constraints, OrientationConstraint, PositionConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import MoveGroup

from control_msgs.action import FollowJointTrajectory
from nav2_msgs.action import NavigateToPose

from sensor_msgs.msg import JointState

# link attacher imports
from linkattacher_msgs.srv import AttachLink, DetachLink

import tf2_ros
from scipy.spatial.transform import Rotation as R

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import ApplyPlanningScene


class SimpleRodGrab(Node):
    """
    Focus: PRE-GRASP -> GRASP (linear cartesian) -> CLOSE GRIPPER -> ATTACH -> PULL BACK -> DETACH -> HOME
    """

    def __init__(self):
        super().__init__("simple_rod_grab")

        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE

        # Inflation params (for safer pre-grasp planning)
        self.inflate_xy = 0.10
        self.inflate_z  = 0.05

        # Absolute topics (no namespace surprises)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", qos)
        self.collision_pub = self.create_publisher(CollisionObject, "/collision_object", qos)
        self.apply_scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        # --- Gazebo LinkAttacher ---
        self.attach_cli = self.create_client(AttachLink, "/ATTACHLINK")
        self.detach_cli = self.create_client(DetachLink, "/DETACHLINK")

        self.attach_model_1 = "mir_robot"
        self.attach_link_1  = "gripper_soft_robotics_left_finger_link1"
        self.attach_model_2 = "Table"
        self.attach_link_2  = "link"

        self.grip_allowance = 0.05

        # Frames / MoveIt
        self.moveit_group = "ur5_manip"
        self.planning_frame = "ur_base_link"
        self.ee_link = "ur_tool0"

        # Table in MAP
        self.table_x = 0.0
        self.table_y = 0.0
        self.table_size_x = 0.65
        self.table_size_y = 0.65
        self.table_top_thickness = 0.012
        self.table_center_z = 1.0

        # Nav/footprint
        self.base_length = 0.73
        self.base_margin = 0.05
        self.front_clearance = 0.5

        # Grasp geometry
        self.pre_dist = 0.25
        self.inside_grasp = -0.015

        self._last_js = None
        self._js_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

        self.arm_joint_names = [
            "ur_shoulder_pan_joint",
            "ur_shoulder_lift_joint",
            "ur_elbow_joint",
            "ur_wrist_1_joint",
            "ur_wrist_2_joint",
            "ur_wrist_3_joint",
        ]

        # Gripper
        self.gripper_open_cmd = 0.40
        self.gripper_close_cmd = -0.20

        # Cartesian sampling
        self.cart_max_step = 0.005
        self.tf_max_age_s = 0.30

        # ROS I/O
        self.marker_pub = self.create_publisher(Marker, "debug_markers", qos)

        self.nav2_ac = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.movegroup_ac = ActionClient(self, MoveGroup, "move_action")
        self.gripper_ac = ActionClient(self, FollowJointTrajectory, "gripper_controller/follow_joint_trajectory")
        self.arm_traj_ac = ActionClient(self, FollowJointTrajectory, "joint_trajectory_controller/follow_joint_trajectory")
        self.cartesian_client = self.create_client(GetCartesianPath, "compute_cartesian_path")

        # Direct base velocity control (for straight back pull)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("[INIT] SimpleRodGrab READY (PRE->GRASP->CLOSE->ATTACH->PULL->DETACH->HOME)")

    # ---------------- Small utils ----------------
    def _now(self):
        return self.get_clock().now().to_msg()

    def _quat_yaw(self, q: Quaternion) -> float:
        return R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def _wait_tf(self, target, source, timeout_s=3.0) -> bool:
        return self.tf_buffer.can_transform(
            target, source, rclpy.time.Time(),
            timeout=Duration(seconds=float(timeout_s))
        )

    def log_tf_age_and_pose(self, target="map", source="base_link"):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
            now = self.get_clock().now()
            tf_time = rclpy.time.Time.from_msg(tf.header.stamp)
            age = (now - tf_time).nanoseconds / 1e9
            yaw = self._quat_yaw(tf.transform.rotation)
            self.get_logger().info(
                f"[TF] {target}->{source} age={age:.3f}s "
                f"pos=({tf.transform.translation.x:.3f},{tf.transform.translation.y:.3f}) yaw={yaw:.3f}"
            )
            if age > self.tf_max_age_s:
                self.get_logger().warn(f"[TF] Transform is stale (age>{self.tf_max_age_s:.2f}s).")
            return tf, age
        except Exception as e:
            self.get_logger().error(f"[TF] Missing {target}->{source}: {e}")
            return None, None

    # ---------------- Markers ----------------
    def publish_marker(self, x, y, z, mid, r, g, b, scale=0.05):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self._now()
        m.ns = "table_debug"
        m.id = int(mid)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.lifetime.sec = 0
        m.lifetime.nanosec = 0
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale)
        m.scale.y = float(scale)
        m.scale.z = float(scale)
        m.color.a = 1.0
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        self.marker_pub.publish(m)

    # ---------------- Orientation (face table) ----------------
    def get_orientation_facing_table(self, robot_x, robot_y):
        dx = self.table_x - robot_x
        dy = self.table_y - robot_y
        yaw = math.atan2(dy, dx)
        r_yaw = R.from_euler("z", yaw, degrees=False)
        r_pitch = R.from_euler("y", 90, degrees=True)  # keep gripper horizontal
        qx, qy, qz, qw = (r_yaw * r_pitch).as_quat()
        return Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))

    # ---------------- Collision object publish ----------------
    def update_obstacle(self, inflate: bool = False):
        slab = CollisionObject()
        slab.header.frame_id = "map"
        slab.id = "table_slab"
        slab.operation = CollisionObject.ADD

        slab_box = SolidPrimitive()
        slab_box.type = SolidPrimitive.BOX

        if inflate:
            sx = self.table_size_x + self.inflate_xy
            sy = self.table_size_y + self.inflate_xy
            sz = self.inflate_z
            center_z = self.table_center_z + (sz - 0.008) / 2.0
        else:
            sx = self.table_size_x
            sy = self.table_size_y
            sz = 0.008
            center_z = self.table_center_z

        slab_box.dimensions = [sx, sy, sz]

        slab_pose = Pose()
        slab_pose.position.x = self.table_x
        slab_pose.position.y = self.table_y
        slab_pose.position.z = center_z
        slab_pose.orientation.w = 1.0

        slab.primitives = [slab_box]
        slab.primitive_poses = [slab_pose]

        tag = "INFLATED" if inflate else "REAL"
        self.get_logger().info(
            f"[SCENE] apply table_slab ({tag}) center=({self.table_x:.3f},{self.table_y:.3f},{center_z:.3f}) "
            f"dims=({sx:.3f},{sy:.3f},{sz:.3f})"
        )

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(slab)

        for _ in range(10):
            self.scene_pub.publish(ps)
            time.sleep(0.05)

        if not self.apply_scene.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("[SCENE] /apply_planning_scene not available (move_group running?)")
        else:
            req = ApplyPlanningScene.Request()
            req.scene = ps
            fut = self.apply_scene.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            ok = bool(fut.result().success) if fut.result() else False
            self.get_logger().info(f"[SCENE] apply_planning_scene success={ok}")

    def get_current_ee_orientation_in_map(self) -> Quaternion:
        rclpy.spin_once(self, timeout_sec=0.05)
        if not self._wait_tf("map", self.ee_link, timeout_s=2.0):
            raise RuntimeError(f"TF not ready: map -> {self.ee_link}")
        tf = self.tf_buffer.lookup_transform("map", self.ee_link, rclpy.time.Time())
        return tf.transform.rotation

    # ---------------- Transform map->planning pose ----------------
    def pose_map_to_planning(self, pose_map: PoseStamped) -> PoseStamped:
        tf_pm = self.tf_buffer.lookup_transform(self.planning_frame, "map", rclpy.time.Time())
        t = tf_pm.transform.translation
        q = tf_pm.transform.rotation
        rot = R.from_quat([q.x, q.y, q.z, q.w])

        px, py, pz = pose_map.pose.position.x, pose_map.pose.position.y, pose_map.pose.position.z
        rx, ry, rz = rot.apply([px, py, pz])

        q_map = R.from_quat([
            pose_map.pose.orientation.x,
            pose_map.pose.orientation.y,
            pose_map.pose.orientation.z,
            pose_map.pose.orientation.w
        ])
        q_plan = (R.from_quat([q.x, q.y, q.z, q.w]) * q_map).as_quat()

        out = PoseStamped()
        out.header.frame_id = self.planning_frame
        out.header.stamp = self._now()
        out.pose.position.x = float(rx + t.x)
        out.pose.position.y = float(ry + t.y)
        out.pose.position.z = float(rz + t.z)
        out.pose.orientation = Quaternion(
            x=float(q_plan[0]), y=float(q_plan[1]), z=float(q_plan[2]), w=float(q_plan[3])
        )
        return out

    # ---------------- Nav2 ----------------
    def navigate_to_face(self, face, clearance):
        hx, hy = self.table_size_x / 2.0, self.table_size_y / 2.0

        if face == "+X":
            fx, fy, nx, ny = self.table_x + hx, self.table_y, 1.0, 0.0
        elif face == "-X":
            fx, fy, nx, ny = self.table_x - hx, self.table_y, -1.0, 0.0
        elif face == "+Y":
            fx, fy, nx, ny = self.table_x, self.table_y + hy, 0.0, 1.0
        else:
            fx, fy, nx, ny = self.table_x, self.table_y - hy, 0.0, -1.0

        dist = (self.base_length / 2.0) + self.base_margin + clearance
        tx, ty = fx + nx * dist, fy + ny * dist
        yaw = math.atan2(-ny, -nx)

        self.get_logger().info(f"[NAV] face={face} clearance={clearance:.2f} goal=({tx:.3f},{ty:.3f}) yaw={yaw:.3f}")

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self._now()
        goal.pose.pose.position = Point(x=float(tx), y=float(ty), z=0.0)
        goal.pose.pose.orientation = Quaternion(
            z=float(math.sin(yaw / 2.0)), w=float(math.cos(yaw / 2.0))
        )

        self.nav2_ac.wait_for_server()
        send_fut = self.nav2_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error("[NAV] goal rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        status = res_fut.result().status
        ok = (status == 4)  # SUCCEEDED
        self.get_logger().info(f"[NAV] done status={status} ok={ok}")
        return ok

    # ---------------- MoveIt pose plan+execute ----------------
    def move_to_pose(self, pose_planning: PoseStamped):
        goal = MoveGroup.Goal()
        goal.request.group_name = self.moveit_group
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0

        oc = OrientationConstraint()
        oc.header = pose_planning.header
        oc.link_name = self.ee_link
        oc.orientation = pose_planning.pose.orientation
        oc.absolute_x_axis_tolerance = 0.10
        oc.absolute_y_axis_tolerance = 0.10
        oc.absolute_z_axis_tolerance = 3.14
        oc.weight = 1.0

        constraints = Constraints()
        constraints.orientation_constraints = [oc]

        pc = PositionConstraint()
        pc.header = pose_planning.header
        pc.link_name = self.ee_link
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.01, 0.01, 0.01])
        region_pose = Pose(position=pose_planning.pose.position, orientation=Quaternion(w=1.0))
        pc.constraint_region.primitives = [box]
        pc.constraint_region.primitive_poses = [region_pose]
        pc.weight = 1.0
        constraints.position_constraints = [pc]

        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False

        self.movegroup_ac.wait_for_server()
        fut = self.movegroup_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("[MOVEIT] goal rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        code = res_fut.result().result.error_code.val
        ok = (code == MoveItErrorCodes.SUCCESS)
        self.get_logger().info(f"[MOVEIT] result error_code={code} ok={ok}")
        return ok

    # ---------------- Cartesian pre->grasp ----------------
    def execute_cartesian_to(self, target_planning: PoseStamped):
        if not self._wait_tf(self.planning_frame, self.ee_link):
            self.get_logger().error("[CART] TF not ready for EE")
            return False

        cur_tf = self.tf_buffer.lookup_transform(self.planning_frame, self.ee_link, rclpy.time.Time())
        start = Pose()
        start.position.x = cur_tf.transform.translation.x
        start.position.y = cur_tf.transform.translation.y
        start.position.z = cur_tf.transform.translation.z
        start.orientation = cur_tf.transform.rotation

        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.group_name = self.moveit_group
        req.link_name = self.ee_link
        req.waypoints = [start, target_planning.pose]
        req.max_step = float(self.cart_max_step)
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        self.get_logger().info(
            f"[CART] start=({start.position.x:.3f},{start.position.y:.3f},{start.position.z:.3f}) "
            f"target=({target_planning.pose.position.x:.3f},{target_planning.pose.position.y:.3f},{target_planning.pose.position.z:.3f}) "
            f"max_step={req.max_step:.3f}"
        )

        self.cartesian_client.wait_for_service()
        fut = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()

        if not res.solution.joint_trajectory.points:
            self.get_logger().error("[CART] failed: no trajectory points (likely collision/IK)")
            return False

        if res.fraction < 0.70:
            self.get_logger().error(f"[CART] incomplete fraction={res.fraction:.3f} -> abort (likely collision)")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = res.solution.joint_trajectory
        self.arm_traj_ac.wait_for_server()
        send_fut = self.arm_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error("[ARM] trajectory rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info("[ARM] cartesian execute done")
        return True

    # ---------------- Joint state helpers ----------------
    def _on_joint_states(self, msg: JointState):
        self._last_js = msg

    def _get_arm_joint_positions(self):
        if self._last_js is None:
            return None
        name_to_pos = {n: p for n, p in zip(self._last_js.name, self._last_js.position)}
        try:
            return [float(name_to_pos[n]) for n in self.arm_joint_names]
        except KeyError:
            return None

    def rotate_wrist3_abs(self, deg: float, duration_s: float = 1.5) -> bool:
        t0 = time.time()
        while self._last_js is None and time.time() - t0 < 2.0:
            rclpy.spin_once(self, timeout_sec=0.05)

        cur = self._get_arm_joint_positions()
        if cur is None:
            self.get_logger().error("[WRIST3] No joint_states for UR joints. Check arm_joint_names.")
            return False

        cur[5] = math.radians(float(deg))

        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = cur
        pt.time_from_start.sec = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f"[WRIST3] Move ur_wrist_3_joint -> {deg:+.1f} deg")
        self.arm_traj_ac.wait_for_server()
        send_fut = self.arm_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error("[WRIST3] Trajectory rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        return True

    # ---------------- Gripper ----------------
    def operate_gripper(self, cmd_value: float, label: str):
        label_u = label.upper()
        if label_u == "OPEN":
            return self.operate_gripper_one_shot(float(cmd_value), label="OPEN", duration_s=0.8)
        if label_u == "CLOSE":
            target = float(cmd_value) + float(self.grip_allowance)
            self.get_logger().info(
                f"[GRIP] CLOSE_FAST target={target:.3f} (cmd={cmd_value:.3f} allowance={self.grip_allowance:.3f})"
            )
            return self.operate_gripper_one_shot(target, label="CLOSE_FAST", duration_s=0.6)

        return self.operate_gripper_one_shot(float(cmd_value), label=label_u, duration_s=0.8)

    def operate_gripper_one_shot(self, cmd_value: float, label: str, duration_s: float = 1.0):
        traj = JointTrajectory()
        traj.joint_names = [
            "gripper_soft_robotics_gripper_left_finger_joint1",
            "gripper_soft_robotics_gripper_right_finger_joint1",
        ]

        pt = JointTrajectoryPoint()
        pt.positions = [float(cmd_value), float(cmd_value)]
        pt.time_from_start.sec = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(f"[GRIP] {label} cmd={cmd_value:.3f}")
        self.gripper_ac.wait_for_server()
        send_fut = self.gripper_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        time.sleep(max(0.2, duration_s))
        return True

    # ---------------- Attach/Detach ----------------
    def attach_sheet_now(self) -> bool:
        if not self.attach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("[ATTACH] /ATTACHLINK service not available")
            return False

        req = AttachLink.Request()
        req.model1_name = self.attach_model_1
        req.link1_name  = self.attach_link_1
        req.model2_name = self.attach_model_2
        req.link2_name  = self.attach_link_2

        self.get_logger().info(
            f"[ATTACH] Request: ({req.model1_name}::{req.link1_name}) <-> ({req.model2_name}::{req.link2_name})"
        )

        fut = self.attach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error("[ATTACH] Service call failed (no response)")
            return False

        self.get_logger().info("[ATTACH] done ✅")
        return True

    def detach_sheet_now(self) -> bool:
        if not self.detach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("[DETACH] /DETACHLINK service not available")
            return False

        req = DetachLink.Request()
        req.model1_name = self.attach_model_1
        req.link1_name  = self.attach_link_1
        req.model2_name = self.attach_model_2
        req.link2_name  = self.attach_link_2

        self.get_logger().info(
            f"[DETACH] Request: ({req.model1_name}::{req.link1_name}) X ({req.model2_name}::{req.link2_name})"
        )

        fut = self.detach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            self.get_logger().error("[DETACH] Service call failed (no response)")
            return False

        self.get_logger().info("[DETACH] done ✅")
        return True

    # ---------------- Base straight reverse ----------------
    def stop_base(self):
        tw = Twist()
        tw.linear.x = 0.0
        tw.angular.z = 0.0
        self.cmd_vel_pub.publish(tw)

    def pull_back_straight(self, distance_m: float = 2.0, speed_mps: float = 0.15, timeout_s: float = 30.0) -> bool:
        """
        Move straight backwards (no turning) using /cmd_vel.
        Distance measured in map frame by projecting displacement onto initial backward direction.
        """
        # Ensure TF updates
        rclpy.spin_once(self, timeout_sec=0.05)
        if not self._wait_tf("map", "base_link", timeout_s=2.0):
            self.get_logger().error("[PULL] TF not ready map->base_link")
            return False

        tf0 = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        x0 = tf0.transform.translation.x
        y0 = tf0.transform.translation.y
        yaw0 = self._quat_yaw(tf0.transform.rotation)

        # Backward direction in map
        bx = -math.cos(yaw0)
        by = -math.sin(yaw0)

        self.get_logger().info(
            f"[PULL] Start ({x0:.3f},{y0:.3f}) yaw={yaw0:.3f} back_dir=({bx:.3f},{by:.3f}) "
            f"target_dist={distance_m:.2f} speed={speed_mps:.2f}"
        )

        t_start = time.time()
        rate_hz = 20.0
        dt = 1.0 / rate_hz

        tw = Twist()
        tw.linear.x = -abs(speed_mps)
        tw.angular.z = 0.0

        last_log = 0.0
        while True:
            if time.time() - t_start > timeout_s:
                self.get_logger().error("[PULL] Timeout reached, stopping base")
                self.stop_base()
                return False

            # Drive
            self.cmd_vel_pub.publish(tw)

            # Update TF
            rclpy.spin_once(self, timeout_sec=0.0)
            try:
                tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            except Exception:
                time.sleep(dt)
                continue

            x = tf.transform.translation.x
            y = tf.transform.translation.y
            dx = x - x0
            dy = y - y0

            progress = dx * bx + dy * by  # projection onto backward direction

            if time.time() - last_log > 0.5:
                self.get_logger().info(f"[PULL] progress={progress:.3f}m / {distance_m:.3f}m pos=({x:.3f},{y:.3f})")
                last_log = time.time()

            if progress >= distance_m:
                self.get_logger().info("[PULL] Target distance reached ✅ stopping")
                self.stop_base()
                time.sleep(0.3)
                return True

            time.sleep(dt)

    # ---------------- Arm Home ----------------
    def go_arm_home(self, duration_s: float = 3.0) -> bool:
        """
        Move manipulator to user-provided home joint configuration.
        """
        home = {
            "ur_shoulder_pan_joint": -1.57,
            "ur_shoulder_lift_joint": -1.57,
            "ur_elbow_joint": -1.57,
            "ur_wrist_1_joint": -0.3,
            "ur_wrist_2_joint": 1.57,
            "ur_wrist_3_joint": 0.0,
        }

        positions = [float(home[n]) for n in self.arm_joint_names]

        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = int(duration_s)
        pt.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1e9)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info("[HOME] Moving arm to HOME joint config...")
        self.arm_traj_ac.wait_for_server()
        send_fut = self.arm_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh.accepted:
            self.get_logger().error("[HOME] Trajectory rejected")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info("[HOME] Arm at HOME ✅")
        return True

    # ---------------- Main sequence ----------------
    def execute_sequence(self):
        self.get_logger().info("[SEQ] Starting PRE -> GRASP -> CLOSE -> ATTACH -> PULL -> DETACH -> HOME")

        # 1) Publish inflated table collision
        self.update_obstacle(inflate=True)

        # 2) Wait for TF
        self.get_logger().info("[TF] Waiting for map -> base_link TF...")
        t0 = time.time()
        timeout = 25.0
        last_log = 0.0

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time()):
                self.get_logger().info("[TF] map->base_link available ✅")
                break
            if time.time() - last_log > 0.5:
                self.get_logger().info("[TF] still waiting (map/base_link not ready yet)...")
                last_log = time.time()

        if not self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time()):
            self.get_logger().error("[TF] Timeout waiting for map->base_link. Abort.")
            return

        # 3) Base pose
        tf_base, _ = self.log_tf_age_and_pose("map", "base_link")
        if tf_base is None:
            self.get_logger().error("[SEQ] Abort: TF lookup failed.")
            return

        bx = tf_base.transform.translation.x
        by = tf_base.transform.translation.y

        # 4) Face selection
        dx = bx - self.table_x
        dy = by - self.table_y
        scores = {"+X": dx, "-X": -dx, "+Y": dy, "-Y": -dy}
        chosen = max(scores, key=lambda k: scores[k])

        self.get_logger().info(
            "[FACE] scores " + " ".join([f"{k}={v:+.3f}" for k, v in scores.items()]) + f" -> selected={chosen}"
        )

        # 5) NAV2 to face
        if not self.navigate_to_face(chosen, self.front_clearance):
            self.get_logger().error("[SEQ] Abort: NAV failed")
            return

        # 6) Edge pts
        hx, hy = self.table_size_x / 2.0, self.table_size_y / 2.0
        edge_pts = {
            "+X": (self.table_x + hx, self.table_y, 1.0, 0.0),
            "-X": (self.table_x - hx, self.table_y, -1.0, 0.0),
            "+Y": (self.table_x, self.table_y + hy, 0.0, 1.0),
            "-Y": (self.table_x, self.table_y - hy, 0.0, -1.0),
        }
        fx, fy, nx, ny = edge_pts[chosen]

        z_target = float(self.table_center_z)
        pre_x = fx + nx * self.pre_dist
        pre_y = fy + ny * self.pre_dist
        grasp_x = fx - nx * self.inside_grasp
        grasp_y = fy - ny * self.inside_grasp

        self.get_logger().info(
            f"[PTS] z_target={z_target:.3f} face={chosen} "
            f"face_center=({fx:.3f},{fy:.3f}) n=({nx:.1f},{ny:.1f}) "
            f"pre=({pre_x:.3f},{pre_y:.3f},{z_target:.3f}) "
            f"grasp=({grasp_x:.3f},{grasp_y:.3f},{z_target:.3f})"
        )

        # Markers
        self.publish_marker(self.table_x, self.table_y, z_target + 0.030, mid=0, r=1.0, g=0.0, b=0.0, scale=0.06)
        self.publish_marker(pre_x, pre_y, z_target + 0.015, mid=1, r=0.0, g=1.0, b=0.0, scale=0.05)
        self.publish_marker(grasp_x, grasp_y, z_target + 0.015, mid=2, r=0.0, g=0.0, b=1.0, scale=0.05)
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

        # 9) Orientation
        tf_base2, _ = self.log_tf_age_and_pose("map", "base_link")
        if tf_base2 is None:
            self.get_logger().error("[SEQ] Abort: TF lost after NAV")
            return

        q_orient = self.get_orientation_facing_table(
            tf_base2.transform.translation.x,
            tf_base2.transform.translation.y
        )

        # 10) OPEN gripper (explicit)
        self.operate_gripper(self.gripper_open_cmd, "OPEN")

        # 11) MoveIt PRE
        pose_pre = PoseStamped()
        pose_pre.header.frame_id = "map"
        pose_pre.header.stamp = self._now()
        pose_pre.pose.position = Point(x=float(pre_x), y=float(pre_y), z=float(z_target))
        pose_pre.pose.orientation = q_orient

        self.get_logger().info("[MOVE] MoveIt -> PRE")
        if not self.move_to_pose(self.pose_map_to_planning(pose_pre)):
            self.get_logger().error("[SEQ] Abort: failed to reach PRE")
            return

        # 11.5) Wrist rotate
        if not self.rotate_wrist3_abs(90.0):
            self.get_logger().error("[SEQ] Abort: failed to rotate wrist_3")
            return

        # Lock current EE orientation
        try:
            ee_q_map = self.get_current_ee_orientation_in_map()
            self.get_logger().info(
                f"[EE] post-wrist ori(map) q=({ee_q_map.x:.3f},{ee_q_map.y:.3f},{ee_q_map.z:.3f},{ee_q_map.w:.3f})"
            )
        except Exception as e:
            self.get_logger().error(f"[EE] Failed to read current EE orientation in map: {e}")
            return

        # 12) Cartesian PRE -> GRASP
        pose_grasp = PoseStamped()
        pose_grasp.header.frame_id = "map"
        pose_grasp.header.stamp = self._now()
        pose_grasp.pose.position = Point(x=float(grasp_x), y=float(grasp_y), z=float(z_target))
        pose_grasp.pose.orientation = ee_q_map

        self.update_obstacle(inflate=False)

        self.get_logger().info("[MOVE] Cartesian PRE -> GRASP (linear)")
        if not self.execute_cartesian_to(self.pose_map_to_planning(pose_grasp)):
            self.get_logger().error("[SEQ] Abort: failed to reach GRASP")
            return

        # 13) CLOSE + ATTACH
        self.operate_gripper(self.gripper_close_cmd, "CLOSE")
        if not self.attach_sheet_now():
            self.get_logger().warn("[ATTACH] attach failed, but continuing")

        # 14) PULL BACK straight ~2m (no turning)
        self.get_logger().info("[SEQ] Pulling back straight ~2.0m ...")
        ok_pull = self.pull_back_straight(distance_m=2.0, speed_mps=0.15, timeout_s=40.0)
        if not ok_pull:
            self.get_logger().warn("[SEQ] Pull-back failed/timeout, continuing to detach/home anyway")

        # 15) DETACH
        if not self.detach_sheet_now():
            self.get_logger().warn("[DETACH] detach failed")

        # (optional) open gripper after detach
        self.operate_gripper(self.gripper_open_cmd, "OPEN")

        # 16) ARM HOME
        if not self.go_arm_home(duration_s=3.0):
            self.get_logger().warn("[HOME] failed to reach home")

        self.get_logger().info("[SEQ] DONE ✅ (Grip + Pull + Detach + Home)")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRodGrab()

    node.execute_sequence()

    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
