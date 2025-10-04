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
from geometry_msgs.msg import TransformStamped
from mir_navigation.srv import VisibilityCheck  # <-- your .srv

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


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


        # ---- Parameters (tweak if needed) ----
        self.declare_parameter('cloud_topic', '/realsense/depth/color/points')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('grid_nx', 20)           # visibility grid X cells
        self.declare_parameter('grid_ny', 10)           # visibility grid Y cells
        self.declare_parameter('vis_ok_thresh', 0.70)   # >= => OK
        self.declare_parameter('vis_min_thresh', 0.20)  # < => NONE
        self.declare_parameter('min_obst_pts', 5)       # >= => obstacle
        self.declare_parameter('z_free_max', 0.35)      # <= => free band upper (m, absolute z in map)
        self.declare_parameter('z_max_clip', 2.50)      # ignore points above this (m)
        self.declare_parameter('stale_cloud_sec', 1.0)  # no cloud newer than this => FAILURE
        self.declare_parameter('max_points_process', 120000)  # limit for speed
        self.declare_parameter('marker_ns', 'lstar_roi')

        self.cloud_topic   = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.target_frame  = self.get_parameter('target_frame').get_parameter_value().string_value
        self.grid_nx       = int(self.get_parameter('grid_nx').value)
        self.grid_ny       = int(self.get_parameter('grid_ny').value)
        self.vis_ok_thresh = float(self.get_parameter('vis_ok_thresh').value)
        self.vis_min_thresh= float(self.get_parameter('vis_min_thresh').value)
        self.min_obst_pts  = int(self.get_parameter('min_obst_pts').value)
        self.z_free_max    = float(self.get_parameter('z_free_max').value)
        self.z_max_clip    = float(self.get_parameter('z_max_clip').value)
        self.stale_cloud_s = float(self.get_parameter('stale_cloud_sec').value)
        self.max_points    = int(self.get_parameter('max_points_process').value)
        self.marker_ns     = self.get_parameter('marker_ns').get_parameter_value().string_value

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
        m = Marker()
        m.header = Header()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.target_frame
        m.ns = self.marker_ns
        m.id = 1
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(cx)
        m.pose.position.y = float(cy)
        m.pose.position.z = 0.0

        # yaw around Z
        cyaw = math.cos(yaw_rad * 0.5)
        syaw = math.sin(yaw_rad * 0.5)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = syaw
        m.pose.orientation.w = cyaw

        # thin prism for 2D ROI
        m.scale.x = float(L)
        m.scale.y = float(W)
        m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()  # refresh during window
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

    # ------------------- Core evaluator -------------------

    def _eval_roi(self, P_map: np.ndarray, cx, cy, L, W, yaw_rad) -> Tuple[int, int, float, int]:
        """
        Returns:
          roi_points_count, observed_cells, vis_ratio, obstacle_points
        """
        if P_map.size == 0:
            return 0, 0, 0.0, 0

        # Shift to center and rotate by -yaw to align ROI with axes
        c = np.array([cx, cy, 0.0], dtype=np.float32)
        P = P_map - c  # translate

        cyaw, syaw = math.cos(-yaw_rad), math.sin(-yaw_rad)
        Rz = np.array([[cyaw, -syaw, 0.0],
                       [syaw,  cyaw, 0.0],
                       [0.0,   0.0,  1.0]], dtype=np.float32)
        Pl = (P @ Rz.T)  # local ROI frame

        # In-rectangle mask
        hx, hy = L * 0.5, W * 0.5
        inside = (np.abs(Pl[:, 0]) <= hx) & (np.abs(Pl[:, 1]) <= hy)
        Pl_in = Pl[inside]
        if Pl_in.size == 0:
            return 0, 0, 0.0, 0

        # ROI points count
        roi_pts = Pl_in.shape[0]

        # Visibility grid (observed cells)
        nx, ny = self.grid_nx, self.grid_ny
        # Map local x,y in [-hx,hx]x[-hy,hy] -> indices [0..nx-1], [0..ny-1]
        ix = np.floor((Pl_in[:, 0] + hx) / (L / max(1, nx)) ).astype(np.int32)
        iy = np.floor((Pl_in[:, 1] + hy) / (W / max(1, ny)) ).astype(np.int32)
        ix = np.clip(ix, 0, max(0, nx-1))
        iy = np.clip(iy, 0, max(0, ny-1))
        observed = set((int(x), int(y)) for x, y in zip(ix, iy))
        observed_cells = len(observed)
        expected_total = max(1, nx * ny)
        vis = observed_cells / expected_total

        # Obstacle points: points with height above free band
        # (use absolute Z in map frame threshold z_free_max)
        # We have local Pl_in z equal to P_map z (rotation about Z)
        obst_pts = np.count_nonzero(P_map[inside, 2] > self.z_free_max)

        return roi_pts, observed_cells, vis, obst_pts

    # ------------------- Service callback -------------------

    def _on_check_area(self, req: VisibilityCheck.Request, resp: VisibilityCheck.Response):
        cloud_frame = self._last_cloud_msg.header.frame_id if self._last_cloud_msg else None
        if cloud_frame:
            ok = self.tf_buffer.can_transform(self.target_frame, cloud_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=2.0))
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
        
        cx = float(req.center_x)
        cy = float(req.center_y)
        L  = float(req.length)
        W  = float(req.width)
        yaw= float(req.yaw)
        dur= float(req.duration_sec)

        self.get_logger().info(f"[CHECK] Request: center=({cx:.2f},{cy:.2f}) L={L:.2f} W={W:.2f} yaw={math.degrees(yaw):.1f}° dur={dur:.1f}s")

        t0 = self.get_clock().now().nanoseconds
        end_ns = t0 + int(dur * 1e9)

        # show ROI marker each cycle
        blocked_final = True  # safe default unless proven clear
        failure_flag = False

        while rclpy.ok() and self.get_clock().now().nanoseconds < end_ns:
            # publish marker (yellow translucent)
            self._publish_roi_marker(cx, cy, L, W, yaw, color=(0.9, 0.8, 0.1, 0.35))

            # Pull cloud
            Pm = self._get_cloud_in_map()
            if Pm is None:
                failure_flag = True
                self.get_logger().warn("[FAILURE] No usable pointcloud (stale or TF error).")
                time.sleep(0.05)
                rclpy.spin_once(self, timeout_sec=0.5)
                continue

            roi_pts, observed_cells, vis, obst_pts = self._eval_roi(Pm, cx, cy, L, W, yaw)

            # Logs in the same style you like
            if roi_pts == 0:
                # nothing seen inside ROI
                blocked = True
                self.get_logger().warn(f"[VIS NONE] vis={vis*100:.0f}% | ROI pts=0 | blocked={blocked}")
            else:
                # decide with simple thresholds
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
                    # between min and ok => still conservative
                    blocked = True
                    self.get_logger().warn(f"[VIS PARTIAL] vis={vis*100:.0f}% | ROI pts={roi_pts} | blocked={blocked}")

            blocked_final = blocked  # keep latest decision

            # Early-exit policy: if we have a confident clear, we can return early.
            if not blocked_final and vis >= self.vis_ok_thresh:
                break

            time.sleep(0.05)
            rclpy.spin_once(self, timeout_sec=0.5)

        # Final decision & final marker (green if clear, red if blocked)
        color = (0.1, 0.8, 0.2, 0.35) if not blocked_final else (0.9, 0.1, 0.1, 0.35)
        self._publish_roi_marker(cx, cy, L, W, yaw, color=color)

        if failure_flag:
            self.get_logger().warn("[FAILURE] Returning blocked=True due to stale cloud/TF issues.")

        resp.obstacle_present = bool(blocked_final)
        self.get_logger().info(f"[RESULT] obstacle_present={resp.obstacle_present}")
        return resp

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
