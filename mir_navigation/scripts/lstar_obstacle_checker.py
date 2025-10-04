#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from std_msgs.msg import Header




import tf2_ros
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, Pose, Vector3, Twist, TransformStamped

from mir_navigation.srv import VisibilityCheck  # <-- your .srv

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from scipy.spatial.transform import Rotation as R

# === MOVEIT2 (ROS2) OFFICIAL APIS ===
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup

from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

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

def quat_to_rotm(qx, qy, qz, qw):
    """Quaternion (xyzw) -> 3x3 rotation matrix."""
    # normalized assumed
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)


def tf_to_matrix(tf: TransformStamped) -> np.ndarray:
    """TransformStamped -> 4x4 homogeneous transform (parent <- child)."""
    t = tf.transform.translation
    r = tf.transform.rotation
    R = quat_to_rotm(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class LStarObstacleChecker(Node):
    """
    Service-only slave:
      - Subscribes to a PointCloud2 (param: ~/cloud_topic).
      - On /lstar/check_area request:
          * Draw ROI Marker in RViz.
          * For 'duration_sec' window, transform latest cloud to /map and evaluate:
                - ROI points
                - Visibility vis = observed / expected
                - Decide blocked True/False (simple thresholds)
          * Logs: [VIS OK]/[VIS PARTIAL]/[VIS NONE]/[FAILURE]
          * Returns bool obstacle_present.
    """

    def __init__(self):
        super().__init__('lstar_obstacle_checker')

        #--- Callback group for Multithreading ---
        self.cbgroup = ReentrantCallbackGroup()


        # === CAMERA / VIEW / MOVEIT PARAMS ===
        self.camera_hfov_deg = float(self.declare_parameter('camera_hfov_deg', 60.0).value)
        self.camera_vfov_deg = float(self.declare_parameter('camera_vfov_deg', 45.0).value)

        # MoveIt group & frames
        self.moveit_group_name = str(self.declare_parameter('moveit_group_name', 'ur5_manip').value)
        self.planning_frame    = str(self.declare_parameter('planning_frame',    'ur_base_link').value)
        self.ee_link           = str(self.declare_parameter('ee_link',           'ur_tool0').value)  # end-effector link
        self.CAMERA_OPTICAL_FRAME = "realsense_depth_optical_frame"


        # --- Cache CAM<-EE (or EE<-CAM) transform once at startup (after tf_buffer is ready) ---
        self.T_cam_ee = None  # 4x4
        self.T_ee_cam = None  # 4x4


        # Z handling for the look-at point
        self.view_z_mode  = str(self.declare_parameter('view_z_mode', 'use_current_ee_z').value)  # 'use_current_ee_z'|'fixed'
        self.view_z_fixed = float(self.declare_parameter('view_z_fixed', 0.90).value)

        # === MOVEIT Clients ===
        # MoveGroup action client (plan+execute)
        self.movegroup_ac = ActionClient(self, MoveGroup, 'move_action', callback_group=self.cbgroup)

        # IK service client
        # self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')




        # ---- Parameters (tweak if needed) ----
        self.declare_parameter('cloud_topic', '/realsense/depth/color/points')
        self.declare_parameter('target_frame', 'map')
        # self.declare_parameter('grid_nx', 20)           # visibility grid X cells
        # self.declare_parameter('grid_ny', 10)           # visibility grid Y cells
        self.declare_parameter('grid_cell_m', 0.10)  # each cell ~10 cm (tune as you like)
        self.declare_parameter('vis_ok_thresh', 0.70)   # >= => OK
        self.declare_parameter('vis_min_thresh', 0.20)  # < => NONE
        self.declare_parameter('min_obst_pts', 500)       # >= => obstacle
        self.declare_parameter('z_free_max', 0.15)      # <= => free band upper (m, absolute z in map)
        self.declare_parameter('z_max_clip', 2.50)      # ignore points above this (m)
        self.declare_parameter('stale_cloud_sec', 1.0)  # no cloud newer than this => FAILURE
        self.declare_parameter('max_points_process', 120000)  # limit for speed
        self.declare_parameter('marker_ns', 'lstar_roi')

        self.cloud_topic   = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.target_frame  = self.get_parameter('target_frame').get_parameter_value().string_value
        # self.grid_nx       = int(self.get_parameter('grid_nx').value)
        # self.grid_ny       = int(self.get_parameter('grid_ny').value)
        self.vis_ok_thresh = float(self.get_parameter('vis_ok_thresh').value)
        self.vis_min_thresh= float(self.get_parameter('vis_min_thresh').value)
        self.min_obst_pts  = int(self.get_parameter('min_obst_pts').value)
        self.z_free_max    = float(self.get_parameter('z_free_max').value)
        self.z_max_clip    = float(self.get_parameter('z_max_clip').value)
        self.stale_cloud_s = float(self.get_parameter('stale_cloud_sec').value)
        self.max_points    = int(self.get_parameter('max_points_process').value)
        self.marker_ns     = self.get_parameter('marker_ns').get_parameter_value().string_value
        self.grid_cell_m = float(self.get_parameter('grid_cell_m').value)


        # ---- Subscribers / Publishers ----
        self._last_cloud_msg: Optional[PointCloud2] = None
        self._last_cloud_stamp_ns: Optional[int] = None
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.cloud_topic, self._on_cloud, 10,
            callback_group=self.cbgroup
        )

        self.marker_pub = self.create_publisher(Marker, 'roi_marker', 10)

        # ---- TF Buffer ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Service server ----
        self.srv = self.create_service(
            VisibilityCheck, '/lstar/check_area', self._on_check_area,
            callback_group=self.cbgroup
        )
        self.get_logger().info(f"[SYS] lstar_obstacle_checker up | cloud={self.cloud_topic} | frame={self.target_frame}")

    def _unit(self, v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def _rotmat_to_quat_xyzw(self, Rm):
        m = np.array(Rm, dtype=float)
        t = np.trace(m)
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2,1] - m[1,2]) / s
            qy = (m[0,2] - m[2,0]) / s
            qz = (m[1,0] - m[0,1]) / s
        else:
            i = int(np.argmax([m[0,0], m[1,1], m[2,2]]))
            if i == 0:
                s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
                qx = 0.25 * s
                qy = (m[0,1] + m[1,0]) / s
                qz = (m[0,2] + m[2,0]) / s
                qw = (m[2,1] - m[1,2]) / s
            elif i == 1:
                s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
                qx = (m[0,1] + m[1,0]) / s
                qy = 0.25 * s
                qz = (m[1,2] + m[2,1]) / s
                qw = (m[0,2] - m[2,0]) / s
            else:
                s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
                qx = (m[0,2] + m[2,0]) / s
                qy = (m[1,2] + m[2,1]) / s
                qz = 0.25 * s
                qw = (m[1,0] - m[0,1]) / s
        return (qx, qy, qz, qw)

    def _look_at_quat_xyzw(self, cam_pos, target, up=(0,0,1)):
        f = self._unit(np.array(target) - np.array(cam_pos))  # forward
        upv = self._unit(np.array(up, dtype=float))
        if abs(float(np.dot(f, upv))) > 0.99:
            upv = np.array([0,1,0], dtype=float)
        y = self._unit(np.cross(upv, f))   # right
        z = np.cross(f, y)                 # corrected up
        # Forward = +X basis
        Rm = np.column_stack((f, y, z))
        return self._rotmat_to_quat_xyzw(Rm)




    
    def _ensure_cam_ee_tf(self) -> bool:
        """Cache CAM<-EE and EE<-CAM transforms if not available yet."""
        if self.T_cam_ee is not None and self.T_ee_cam is not None:
            return True
        try:
            tf_cam_ee = self.tf_buffer.lookup_transform(
                self.CAMERA_OPTICAL_FRAME,  # target
                self.ee_link,               # source
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
            self.T_cam_ee = tf_to_matrix(tf_cam_ee)
            self.T_ee_cam = np.linalg.inv(self.T_cam_ee)
            self.get_logger().info("[TF] Cached T_cam_ee and T_ee_cam")
            return True
        except Exception as ex:
            self.get_logger().warn(f"[TF] CAM<->EE TF not ready: {ex}")
            return False

    

    def _plan_execute_joint_goal(self, js: JointState) -> bool:
        if not self.movegroup_ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("[MOVEIT] MoveGroup action server not available")
            return False

        # Build joint constraints for the group's joints from IK solution
        jc_list = []
        js_map = {n: p for n, p in zip(js.name, js.position)}
        # NOTE: if you know your group's joint names, list them explicitly for stability
        for name, pos in js_map.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.03
            jc.tolerance_below = 0.03
            jc.weight = 1.0
            jc_list.append(jc)

        constraints = Constraints()
        constraints.joint_constraints = jc_list

        req = MotionPlanRequest()
        req.group_name = self.moveit_group_name
        req.num_planning_attempts = 1
        req.allowed_planning_time = 1.5
        req.goal_constraints = [constraints]

        opts = PlanningOptions()
        opts.plan_only = False
        opts.look_around = False
        opts.replan = False

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = opts

        send = self.movegroup_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=3.0)
        gh = send.result()
        if gh is None:
            self.get_logger().error("[MOVEIT] goal handle none")
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=10.0)
        res = res_fut.result()
        if not res or res.result.error_code.val < 0:
            self.get_logger().error(f"[MOVEIT] planning/execution failed code={getattr(res.result.error_code, 'val', 'NA')}")
            return False
        return True
    
    def _transform_point_xy(self, x, y, from_frame: str, to_frame: str):
        try:
            tf = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            T = tf_to_matrix(tf)  # 4x4
            p = np.array([float(x), float(y), 0.0, 1.0], dtype=np.float32)
            pout = T @ p
            return (float(pout[0]), float(pout[1]), float(pout[2]))
        except Exception as ex:
            self.get_logger().warn(f"[TF] {from_frame}->{to_frame} transform failed: {ex}")
            return (x, y, 0.0)  # fallback
        



    

    def _ee_pose_orient_only(self, target_w: np.ndarray, planning_frame: str) -> Optional[PoseStamped]:
        """
        Sirf orientation set karo taaki camera target ko dekhe,
        EE/camera position JAHAN HAI WAHIN rahe.
        """
        # Ensure T_ee_cam cached
        if self.T_ee_cam is None:
            self.get_logger().error("[VIEW] Missing T_ee_cam; call _ensure_cam_ee_tf() before orient-only.")
            return None

        # 1) Current camera & EE poses (planning_frame me)
        try:
            self.get_logger().info("[VIEW] Getting current camera pose...")
            tf_cam_now = self.tf_buffer.lookup_transform(
                planning_frame, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            cam_pos_now = np.array([
                tf_cam_now.transform.translation.x,
                tf_cam_now.transform.translation.y,
                tf_cam_now.transform.translation.z
            ], dtype=float)
        except Exception:
            # fallback to EE pose
            self.get_logger().warn("[VIEW] Camera TF failed; falling back to EE pose for position lock.")
            tf_ee_now = self.tf_buffer.lookup_transform(
                planning_frame, self.ee_link, rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
            cam_pos_now = np.array([
                tf_ee_now.transform.translation.x,
                tf_ee_now.transform.translation.y,
                tf_ee_now.transform.translation.z
            ], dtype=float)

        # 2) Naya camera rotation: +Z target ki taraf
        z_cam = target_w - cam_pos_now #Δz = 0
        self.get_logger().info(f"[VIEW] Orienting camera at target {target_w} from {cam_pos_now}")
        nz = np.linalg.norm(z_cam) #Vectror length from camera to target
        if nz < 1e-9:
            self.get_logger().warn("[VIEW] target ~= camera position; orient-only undefined.")
            return None
        z_cam = z_cam / nz #Unit vector in the direction from camera to target

        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z_cam, up))) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        x_cam = np.cross(up, z_cam); x_cam /= (np.linalg.norm(x_cam) + 1e-9)
        y_cam = np.cross(z_cam, x_cam)
        # Hit & Trial se ulti kr dee axis kyunki camera frame me aisa hai.
        y_cam = -y_cam; 
        x_cam = -x_cam

        R_w_cam = np.column_stack((x_cam, y_cam, z_cam)) #rotation matrix of desuired camera frame in world frame
        U, _, Vt = np.linalg.svd(R_w_cam); R_w_cam = U @ Vt
        if np.linalg.det(R_w_cam) < 0.0:
            U[:, -1] *= -1.0; R_w_cam = U @ Vt

        # 3) CAMERA pose = same translation, new rotation
        T_w_cam = np.eye(4, dtype=float)
        T_w_cam[0:3, 0:3] = R_w_cam
        T_w_cam[0:3, 3]   = cam_pos_now  # <-- POSITION LOCKED

        # 4) EE pose from camera pose
        # T_w_ee = T_w_cam @ self.T_ee_cam
        T_w_ee = T_w_cam @ self.T_cam_ee

        ps = PoseStamped()
        ps.header.frame_id = planning_frame
        ps.pose.position.x = float(T_w_ee[0,3])
        ps.pose.position.y = float(T_w_ee[1,3])
        ps.pose.position.z = float(T_w_ee[2,3])

        q_ee = R.from_matrix(T_w_ee[0:3,0:3]).as_quat()  # xyzw
        ps.pose.orientation.x = float(q_ee[0])
        ps.pose.orientation.y = float(q_ee[1])
        ps.pose.orientation.z = float(q_ee[2])
        ps.pose.orientation.w = float(q_ee[3])
        return ps
    


    # def _plan_execute_orient_only(self, pose_stamped: PoseStamped, pos_lock_tol: float = 0.02) -> bool:
    #     """
    #     EE position ko chhote box (±pos_lock_tol) me lock karo,
    #     aur orientation ko pose_stamped.orientation pe align karao.
    #     """
    #     if not self.movegroup_ac.wait_for_server(timeout_sec=1.0):
    #         self.get_logger().error("[MOVEIT] MoveGroup server not available")
    #         return False

    #     # Position lock around the SAME position
    #     pc = PositionConstraint()
    #     pc.header.frame_id = pose_stamped.header.frame_id
    #     pc.link_name = self.ee_link
    #     from geometry_msgs.msg import Pose as GeoPose
    #     lock_pose = GeoPose()
    #     lock_pose.position = pose_stamped.pose.position
    #     pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
    #                                                     dimensions=[2*pos_lock_tol,
    #                                                                 2*pos_lock_tol,
    #                                                                 2*pos_lock_tol])]
    #     pc.constraint_region.primitive_poses = [lock_pose]
    #     pc.weight = 1.0

    #     # Orientation goal (~3 deg tolerance)
    #     oc = OrientationConstraint()
    #     oc.header.frame_id = pose_stamped.header.frame_id
    #     oc.link_name = self.ee_link
    #     oc.orientation = pose_stamped.pose.orientation
    #     oc.absolute_x_axis_tolerance = math.radians(3.0)
    #     oc.absolute_y_axis_tolerance = math.radians(3.0)
    #     oc.absolute_z_axis_tolerance = math.radians(3.0)
    #     oc.weight = 1.0

    #     cons = Constraints()
    #     cons.position_constraints = [pc]
    #     cons.orientation_constraints = [oc]

    #     req = MotionPlanRequest()
    #     req.group_name = self.moveit_group_name
    #     req.goal_constraints = [cons]
    #     req.allowed_planning_time = 2.0
    #     req.num_planning_attempts = 2

    #     opts = PlanningOptions()
    #     opts.plan_only = False
    #     opts.look_around = False
    #     opts.replan = False

    #     goal = MoveGroup.Goal()
    #     goal.request = req
    #     goal.planning_options = opts

    #     send = self.movegroup_ac.send_goal_async(goal)
    #     rclpy.spin_until_future_complete(self, send, timeout_sec=3.0)
    #     gh = send.result()
    #     if gh is None:
    #         return False

    #     res_fut = gh.get_result_async()
    #     rclpy.spin_until_future_complete(self, res_fut, timeout_sec=10.0)
    #     res = res_fut.result()
    #     return bool(res) and (res.result.error_code.val >= 0)

    def _plan_execute_orient_only(
        self,
        target_pose_stamped: PoseStamped,
        pos_tol: float = 0.02,           # ~2 cm lock radius (relax if controller is picky)
        ori_tol_deg: float = 5.0,        # ~10° orientation tolerance
        plan_time: float = 2.0,
        attempts: int = 2,
        wait_aclient_sec: float = 2.0,
        wait_send_sec: float = 30.0,
        wait_result_sec: float = 10.0,
    ) -> bool:
        """
        Rotate-in-place: keep EE near its current position (±pos_tol sphere),
        while aligning orientation to target_pose_stamped.orientation.
        Uses Pose() for primitive_poses (no GeoPose).
        """
        try:
            # 0) Ensure action server is up
            if not self.movegroup_ac.wait_for_server(timeout_sec=wait_aclient_sec):
                self.get_logger().error("[MOVEIT] MoveGroup action server not available")
                return False

            # 1) --- Orientation constraint ---
            oc = OrientationConstraint()
            oc.header = target_pose_stamped.header
            oc.link_name = self.ee_link
            oc.orientation = target_pose_stamped.pose.orientation
            oc.absolute_x_axis_tolerance = math.radians(ori_tol_deg)
            oc.absolute_y_axis_tolerance = math.radians(ori_tol_deg)
            oc.absolute_z_axis_tolerance = math.radians(ori_tol_deg)
            oc.weight = 1.0

            # 2) --- Position lock: sphere around the *same* EE position ---
            #    Center = target_pose_stamped.position (hum xyz ko same rakhna chahte hain)
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [float(pos_tol)]   # radius (meters)

            center_pose = Pose()
            center_pose.position = target_pose_stamped.pose.position
            center_pose.orientation.w = 1.0  # identity (orientation irrelevant for a sphere)

            bv = BoundingVolume()
            bv.primitives = [sphere]
            bv.primitive_poses = [center_pose]

            pc = PositionConstraint()
            pc.header = target_pose_stamped.header
            pc.link_name = self.ee_link
            pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
            pc.constraint_region = bv
            pc.weight = 1.0

            goal_cons = Constraints()
            goal_cons.orientation_constraints = [oc]
            goal_cons.position_constraints = [pc]

            # 3) --- Motion plan request ---
            req = MotionPlanRequest()
            req.group_name = self.moveit_group_name
            req.goal_constraints = [goal_cons]
            req.allowed_planning_time = float(plan_time)
            req.num_planning_attempts = int(attempts)

            # 4) --- Planning options (execute, not plan-only) ---
            opts = PlanningOptions()
            opts.plan_only = False
            opts.look_around = False
            opts.replan = False

            goal = MoveGroup.Goal()
            goal.request = req
            goal.planning_options = opts

            # 5) --- Send + wait ---
            send_fut = self.movegroup_ac.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut, timeout_sec=wait_send_sec)
            gh = send_fut.result()
            if gh is None:
                self.get_logger().error("[MOVEIT] send_goal_async returned None")
                return False

            res_fut = gh.get_result_async()
            rclpy.spin_until_future_complete(self, res_fut, timeout_sec=wait_result_sec)
            res = res_fut.result()
            if not res:
                self.get_logger().error("[MOVEIT] No result from MoveGroup")
                return False

            ok = (res.result.error_code.val >= 0)
            if not ok:
                self.get_logger().error(f"[MOVEIT] planning/execution failed code={res.result.error_code.val}")
            else:
                self.get_logger().info("[MOVEIT] orientation-only execution OK")
            return bool(ok)

        except Exception as ex:
            self.get_logger().error(f"[MOVEIT] exception in _plan_execute_orient_only: {ex}")
            return False


    
    # ------------------- Service callback -------------------

    def _on_check_area(self, req: VisibilityCheck.Request, resp: VisibilityCheck.Response):
        # ---- Pre-check: TF & cloud availability ----
        cloud_frame = self._last_cloud_msg.header.frame_id if self._last_cloud_msg else None
        if cloud_frame:
            ok = self.tf_buffer.can_transform(self.target_frame, cloud_frame, rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=2.0))
            if not ok:
                self.get_logger().warn(f"[FAILURE] TF not ready {self.target_frame}<-{cloud_frame} (pre-check).")
                self._diagnose_tf_issue(self.target_frame, cloud_frame, cloud_stamp=self._last_cloud_msg.header.stamp)
                resp = VisibilityCheck.Response()
                resp.obstacle_present = True  # safe
                return resp
        else:
            self.get_logger().warn("[FAILURE] No cloud received yet (no frame_id available) during pre-check.")
            resp = VisibilityCheck.Response()
            resp.obstacle_present = True
            return resp

        # ---- Parse request ----
        cx = float(req.center_x)
        cy = float(req.center_y)
        L  = float(req.length)
        W  = float(req.width)
        yaw= float(req.yaw)
        dur= float(req.duration_sec)
        self._box_fp = (req.box_center_x, req.box_center_y, req.box_length, req.box_width)


        self.get_logger().info(f"[CHECK] Request: center=({cx:.2f},{cy:.2f}) L={L:.2f} W={W:.2f} yaw={math.degrees(yaw):.1f}° dur={dur:.1f}s")

        # === VIEW ACQUISITION: orientation-only (lock position) ===
        try:
            # 0) Ensure CAM<->EE TF cached
            if not self._ensure_cam_ee_tf():
                self.get_logger().error("[VIEW] Missing CAM<->EE TF; cannot proceed.")
                resp.obstacle_present = True
                return resp

            # 1) ROI centre ko planning_frame me lao
            cx_pf, cy_pf, z_pf = (cx, cy, 0.0)
            if self.target_frame != self.planning_frame:
                cx_pf, cy_pf, z_pf = self._transform_point_xy(cx, cy, self.target_frame, self.planning_frame)
                # /map frame cx,cy ko /ur_base_link me dal diya.

            # 2) Target Z choose (tumhare hi params follow)
            if self.view_z_mode == 'fixed':
                z_target = self.view_z_fixed
                self.get_logger().info(f"[VIEW] Using fixed z={z_target:.3f}m as target z.")
            else:
                try:
                    tf_cam = self.tf_buffer.lookup_transform(
                        self.planning_frame, self.CAMERA_OPTICAL_FRAME, rclpy.time.Time(),
                        timeout=Duration(seconds=0.5)
                    )
                    z_target = float(tf_cam.transform.translation.z)
                    self.get_logger().info(f"[VIEW] Using current camera z={z_target:.3f}m as target z.")
                except Exception:
                    z_target = self.view_z_fixed
                    self.get_logger().warn(f"[VIEW] Camera TF failed; falling back to fixed z={z_target:.3f}m.")

            # target = np.array([cx_pf, cy_pf, z_target], dtype=float)
            target = np.array([cx_pf, cy_pf, 0.0], dtype=float)


            # 3) Sirf orientation wala EE pose banao (xyz same, sirf q change)
            ee_pose = self._ee_pose_orient_only(target, self.planning_frame)
            if ee_pose is None:
                self.get_logger().error("[VIEW] orient-only pose build failed.")
                resp.obstacle_present = True
                return resp

            # 4) Planner: position lock (tiny box) + orientation goal
            if not self._plan_execute_orient_only(ee_pose, pos_tol=0.02):
                self.get_logger().error("[VIEW] Orientation-only failed.")
                resp.obstacle_present = True
                return resp

            time.sleep(0.4)  # small settle

        except Exception as ex:
            self.get_logger().error(f"[VIEW] exception: {ex}")
            resp.obstacle_present = True
            return resp

        # === Visibility evaluation loop (unchanged) ===
        t0 = self.get_clock().now().nanoseconds
        end_ns = t0 + int(dur * 1e9)

        blocked_final = True  # safe default unless proven clear
        failure_flag = False

        while rclpy.ok() and self.get_clock().now().nanoseconds < end_ns:
            # publish marker (yellow translucent)
            # self._publish_roi_marker(cx, cy, L, W, yaw, color=(0.9, 0.8, 0.1, 0.35))

            # Pull cloud in map
            Pm = self._get_cloud_in_map()
            if Pm is None:
                failure_flag = True
                self.get_logger().warn("[FAILURE] No usable pointcloud (stale or TF error).")
                time.sleep(0.05)
                rclpy.spin_once(self, timeout_sec=0.5)
                continue

            roi_pts, observed_cells, vis, obst_pts = self._eval_roi(Pm, cx, cy, L, W, yaw)

            if roi_pts == 0:
                blocked = True
                self.get_logger().warn(f"[VIS NONE] vis={vis*100:.0f}% | ROI pts=0 | blocked={blocked}")
            else:
                if obst_pts >= self.min_obst_pts:
                    blocked = True
                    self.get_logger().warn(f"[VIS OK] vis={vis*100:.0f}% | ROI pts={roi_pts} | obst_pts={obst_pts} | blocked={blocked}")
                elif vis < self.vis_min_thresh:
                    blocked = True
                    self.get_logger().warn(f"[VIS PARTIAL] vis={vis*100:.0f}% | ROI pts={roi_pts} | Suggest small rotate/step | blocked={blocked}")
                elif vis >= self.vis_ok_thresh:
                    blocked = False
                    self.get_logger().info(f"[VIS OK] vis={vis*100:.0f}% | ROI pts={roi_pts} | blocked={blocked}")
                else:
                    blocked = True
                    self.get_logger().warn(f"[VIS PARTIAL] vis={vis*100:.0f}% | ROI pts={roi_pts} | blocked={blocked}")

            blocked_final = blocked

            if not blocked_final and vis >= self.vis_ok_thresh:
                break

            time.sleep(0.05)
            rclpy.spin_once(self, timeout_sec=0.5)

        # Final decision & final marker
        color = (0.1, 0.8, 0.2, 0.35) if not blocked_final else (0.9, 0.1, 0.1, 0.35)
        self._publish_roi_marker(cx, cy, L, W, yaw, color=color)

        if failure_flag:
            self.get_logger().warn("[FAILURE] Returning blocked=True due to stale cloud/TF issues.")

        resp.obstacle_present = bool(blocked_final)
        self.get_logger().info(f"[RESULT] obstacle_present={resp.obstacle_present}")
        return resp







    # ------------------- Cloud callback -------------------

    def _on_cloud(self, msg: PointCloud2):
        self._last_cloud_msg = msg
        self._last_cloud_stamp_ns = self.get_clock().now().nanoseconds
        self.get_logger().debug(f"[CLOUD] received {msg.width}x{msg.height} points in frame {msg.header.frame_id} with timestamp {self._last_cloud_stamp_ns}")


    # ------------------- FAILURE TF REPORT -------------------

    def _diagnose_tf_issue(self, target_frame: str, source_frame: str, cloud_stamp=None):
        """
        Print actionable reasons for TF failure.
        """
        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9

        # Cloud stamp (if known)
        cloud_age_s = None
        if cloud_stamp is not None:
            # cloud_stamp can be rclpy.time.Time or builtin_interfaces/Time
            try:
                if hasattr(cloud_stamp, 'nanoseconds'):
                    cloud_ts = cloud_stamp.nanoseconds / 1e9
                else:
                    cloud_ts = cloud_stamp.sec + cloud_stamp.nanosec * 1e-9
                cloud_age_s = max(0.0, now_s - cloud_ts)
            except Exception:
                cloud_age_s = None

        # 1) Quick existence check (0 timeout)
        can_now = self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.0))
        if not can_now:
            self.get_logger().warn(f"[TF] {target_frame} <- {source_frame}: transform not available *now* (likely frames not yet in tree or not connected).")
            self.get_logger().warn(f"[TF] Hints: is AMCL/map up (map->odom)? robot_state_publisher up (base_link->camera)?")
            return

        # 2) If "can_now" true but lookup still fails later, it's usually timing/extrapolation.
        #    Try to categorize by attempting at cloud time (if provided) and at 'latest'.
        # Try cloud time
        try:
            if cloud_stamp is not None:
                _ = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time.from_msg(cloud_stamp))
                # If this succeeds, we wouldn't be here. Just in case:
                self.get_logger().info(f"[TF] Transform OK at cloud time; earlier failure was transient.")
                return
        except Exception as ex_cloud:
            msg = str(ex_cloud)
            if 'extrapolation' in msg.lower():
                self.get_logger().warn(f"[TF] Extrapolation at cloud time (stamp off). Cloud age≈{cloud_age_s:.2f}s. Consider increasing stale_cloud_sec or sync clocks (sim_time/NTP).")
            elif 'connectivity' in msg.lower():
                self.get_logger().warn(f"[TF] Connectivity issue between {target_frame} and {source_frame} at cloud time (tree split?).")
            elif 'does not exist' in msg.lower() or 'could not find' in msg.lower():
                self.get_logger().warn(f"[TF] Frame missing in tree at cloud time: {msg}")
            else:
                self.get_logger().warn(f"[TF] Lookup at cloud time failed: {msg}")

        # Try latest time (time=0)
        try:
            _ = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            # If this works, problem is purely timestamp/extrapolation.
            self.get_logger().warn(f"[TF] Transform exists at latest, but not at cloud time. Cloud age≈{cloud_age_s:.2f}s. Increase stale window or ensure synchronized time.")
        except Exception as ex_now:
            msg = str(ex_now)
            if 'extrapolation' in msg.lower():
                self.get_logger().warn(f"[TF] Extrapolation at latest (clock jumping or wrong sim_time).")
            elif 'connectivity' in msg.lower():
                self.get_logger().warn(f"[TF] Connectivity issue at latest: tree parts not connected.")
            elif 'does not exist' in msg.lower() or 'could not find' in msg.lower():
                self.get_logger().warn(f"[TF] Missing frame(s) at latest: {msg}")
            else:
                self.get_logger().warn(f"[TF] Lookup at latest failed: {msg}")

        # Extra hints
        if cloud_age_s is not None:
            self.get_logger().warn(f"[TF] Cloud age: {cloud_age_s:.2f}s | stale_cloud_sec={self.stale_cloud_s:.2f}. If age > stale, raise stale_cloud_sec during startup.")
        use_sim = self.get_parameter('use_sim_time').get_parameter_value().bool_value if self.has_parameter('use_sim_time') else False
        self.get_logger().warn(f"[TF] use_sim_time={use_sim}. Ensure all nodes share same clock (gazebo/sim vs real).")


    # ------------------- Helper: publish ROI marker -------------------

    def _publish_roi_marker(self, cx, cy, L, W, yaw_rad, color=(0.9, 0.8, 0.1, 0.35)):
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Pose

        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD

        # ROI pose in map
        pose = Pose()
        pose.position.x = float(cx)
        pose.position.y = float(cy)
        pose.position.z = 0.025  # thin slab, 5 cm above ground
        # yaw -> quaternion (z-rotation only)
        cyaw, syaw = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
        pose.orientation.z = syaw
        pose.orientation.w = cyaw

        m.pose = pose
        m.scale.x = float(L)
        m.scale.y = float(W)
        m.scale.z = 0.05  # 5 cm thick

        m.color.r, m.color.g, m.color.b, m.color.a = color
        self.marker_pub.publish(m)


    def _publish_obstacle_points(self, P_map, inside, z_thresh, color=(1.0, 0.0, 0.0, 1.0), max_points=5000):
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        import numpy as np

        # Mask: ROI ke andar + z > threshold
        obst_mask = inside & (P_map[:, 2] > float(z_thresh))
        pts = P_map[obst_mask]

        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 3                   # <- alag ID for obstacles
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.025
        m.scale.y = 0.025
        m.color.r, m.color.g, m.color.b, m.color.a = color

        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
        self.marker_pub.publish(m)

    # ------------------- Helper: pull latest cloud in /map -------------------

    def _get_cloud_in_map(self) -> Optional[np.ndarray]:
        msg = self._last_cloud_msg
        if msg is None:
            return None

        # stale check
        now_ns = self.get_clock().now().nanoseconds
        if self._last_cloud_stamp_ns is None or (now_ns - self._last_cloud_stamp_ns) > int(self.stale_cloud_s * 1e9):
            self.get_logger().warn("[FAILURE] No recent cloud received (stale).")
            return None

        # TF: map <- cloud_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().warn(f"[FAILURE] TF lookup {self.target_frame}<-{msg.header.frame_id} failed: {ex}")
            self._diagnose_tf_issue(self.target_frame, msg.header.frame_id, cloud_stamp=msg.header.stamp)
            return None

        T = tf_to_matrix(tf)

        # Extract XYZ (limit points for speed)
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True) #gen = [(1.0, 2.0, 0.5), (1.1, 2.1, 0.6), (1.2, 2.2, 0.7), ...]
        #Only x,y,z fields, skip NaNs from camera. 
        pts = np.fromiter(gen, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]) #numpy array with shape (N,) and fields x,y,z. f4 - 32 float
        if pts.size == 0:
            self.get_logger().warn("[FAILURE] No valid points found in cloud.")
            return np.empty((0, 3), dtype=np.float32)

        if pts.shape[0] > self.max_points:
            idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[idx]

        P = np.column_stack((pts['x'], pts['y'], pts['z'])).astype(np.float32)

        # Transform: map <- cloud
        # P_map = R*P + t
        R = T[:3, :3]
        t = T[:3, 3]
        Pm = (P @ R.T) + t  # (N,3)
        if Pm.size == 0:
            self.get_logger().warn("[FAILURE] No valid points found in transformed cloud.")


        # z clip (optional)
        if self.z_max_clip is not None:
            Pm = Pm[Pm[:, 2] <= self.z_max_clip]

        return Pm


    def _publish_inside_points(self, P_map, inside, color=(0.1, 0.6, 1.0, 0.9), max_points=5000):
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        import numpy as np

        # ROI ke andar ke saare points
        pts = P_map[inside]

        # --- subsample BEFORE building Marker.points ---
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = self.marker_ns
        m.id = 2                     # ROI cube (id=1) se different
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.02             # point size (m)
        m.scale.y = 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = color

        # build points
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]

        # (optional) lifetime set karna ho to:
        # m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        self.marker_pub.publish(m)

        # Agar zero points: publish empty karna hi enough hota hai; warna:
        # if pts.shape[0] == 0:
        #     m.action = Marker.DELETE
        #     self.marker_pub.publish(m)



    # ------------------- Core evaluator -------------------

    def _eval_roi(self, P_map, cx, cy, L, W, yaw_rad):
        """
        Returns (roi_pts, observed_cells, vis_ratio, obst_pts)

        Changes vs before:
        - Forward-only strip: x' ∈ [eps_forward, L], |y'| ≤ W/2
        - EXCLUDE points inside the box footprint (axis-aligned) with small margin
        - Obstacle = z > self.z_free_max (unchanged)
        """
        import numpy as np
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point

        # ---- Tunables (simple) ----
        fp_margin   = 0.15    # 15 cm: dilate footprint to absorb sensor edge/noise
        # grid_nx, grid_ny = 20, 10   # for visibility metric (same spirit as before)
        cell = max(1e-3, float(self.grid_cell_m))  # safety
        grid_nx = max(1, int(math.ceil(float(L) / cell)))
        grid_ny = max(1, int(math.ceil(float(W) / cell)))

        # ---- Early outs ----
        if P_map is None or P_map.shape[0] == 0:
            self.get_logger().warn("[FAILURE] No valid points found in transformed cloud.")
            return 0, 0, 0.0, 0

        # ---- 0) Draw ROI marker (your existing helper) ----
        try:
            self._publish_roi_marker(cx, cy, L, W, yaw_rad, color=(0.9, 0.8, 0.1, 0.35))
        except Exception:
            pass

        # ---- 1) Transform points to ROI local frame (de-rotate by yaw) ----
        dx = P_map[:, 0] - float(cx) #Depth point offset from center of ROI centre (x cordinate)
        dy = P_map[:, 1] - float(cy) #Depth point offset from center of ROI centre (y cordinate)
        c = np.cos(-yaw_rad); s = np.sin(-yaw_rad)   # align ROI with axes, rotation matrix elements
        # 2D rotation matrix kyunki z axis pe rotation hai
        xp =  c * dx - s * dy                        # push-axis coord
        yp =  s * dx + c * dy                        # cross-axis coord

        # ---- 2) Forward-only ROI (NOT symmetric) ----
        halfL = 0.5 * float(L) #L - ROI length
        halfW = 0.5 * float(W) #W - ROI width
        inside_roi = (np.abs(xp) < halfL) & (np.abs(yp) < halfW)


        # ---- 3) Box footprint EXCLUSION (axis-aligned, map frame) ----
        # NOTE: service callback must set self._box_fp = (cx, cy, L, W) before calling _eval_roi
        try:
            box_cx, box_cy, box_L, box_W = self._box_fp
            self.get_logger().info(f"[FOOTPRINT] Excluding box footprint at ({box_cx:.2f},{box_cy:.2f}) L={box_L:.2f} W={box_W:.2f}")
        except Exception:
            # fallback: no exclusion if not provided
            self.get_logger().warn("Box footprint not provided")
            box_cx = box_cy = 0.0; box_L = box_W = 0.0

        hx = 0.5 * float(box_L) + fp_margin
        hy = 0.5 * float(box_W) + fp_margin
        inside_fp = (np.abs(P_map[:,0] - box_cx) <= hx) & (np.abs(P_map[:,1] - box_cy) <= hy)


        # inside = inside_roi & (~inside_fp)
        overlap = np.logical_and(inside_roi, inside_fp)
        inside = inside_roi & (~overlap)


        # ---- 4) Publish ROI-inside points (blue) if your helper exists ----
        try:
            self._publish_inside_points(P_map, inside, color=(0.0, 0.5, 1.0, 0.9), max_points=5000)
        except Exception:
            # minimal inline publisher (optional): ignore if you already have helper
            pass


        # # ---- 5) Obstacle points = z > z_free_max (same threshold) ----
        obst_mask = inside & (P_map[:, 2] > float(self.z_free_max))
        obst_pts  = int(np.count_nonzero(obst_mask))

        # red debug publish if your helper exists
        try:
            self._publish_obstacle_points(P_map, inside, self.z_free_max, color=(1.0, 0.0, 0.0, 1.0), max_points=5000)
        except Exception:
            pass

        # ---- 6) Visibility metric (grid coverage) ----
        roi_pts = int(np.count_nonzero(inside)) #how many points are inside the ROI

        observed_cells = 0
        vis_ratio = 0.0
        try:
            # Use only points we kept (inside)
            X = xp[inside]; Y = yp[inside]
            if X.size > 0:
                # shift & scale to 0..1, then to bins
                bx = np.clip(((X + halfL) / float(L)) * grid_nx, 0, grid_nx - 1).astype(int)
                by = np.clip(((Y + halfW) / (2.0 * halfW)) * grid_ny, 0, grid_ny - 1).astype(int)

                grid = np.zeros((grid_nx, grid_ny), dtype=bool)
                grid[bx, by] = True
                observed_cells = int(grid.sum())
                vis_ratio = float(observed_cells) / float(grid_nx * grid_ny)
        except Exception:
            observed_cells = 0
            vis_ratio = 0.0

        return roi_pts, observed_cells, vis_ratio, obst_pts



def main():
    rclpy.init()
    node = LStarObstacleChecker()
    try:
        executor = MultiThreadedExecutor(num_threads=2)  # 2 ya 4 bhi rakh sakte ho
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
