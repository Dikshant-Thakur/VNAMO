#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L* Region Obstacle Checker (RGB-D)
- Standalone ROS2 node that tells if *any* obstacle points exist in the L* corridor
  extending from a given box face along a chosen push direction.
- Uses PointCloud2 from an RGB-D camera, transforms points to "map", and checks
  if they fall inside a 2D oriented rectangle (with optional height band).

This is a *concept-only* file, not integrated with other nodes. Wire it up later.

Basic I/O
---------
Subs:
  - /camera/depth/points (sensor_msgs/PointCloud2)  [configurable via param: pointcloud_topic]
TF:
  - camera frame -> map frame (tf2 must provide this)

Pubs:
  - lstar_obstacle_present (std_msgs/Bool): True if any valid points lie inside the L* region
  - lstar_points_in_roi (sensor_msgs/PointCloud2): filtered cloud (optional; param enable_debug_cloud)
  - lstar_region_marker (visualization_msgs/Marker): rectangle outline for RViz (optional)

Core idea
---------
For a box centered at (box_x, box_y) with size (box_length, box_width), and a
chosen push direction {+X,-X,+Y,-Y}, define the L* corridor starting at the box face
and extending `L_star` meters along that direction, with corridor width `corridor_width`.
If any point from the RGB-D point cloud (transformed to map frame) lies within that
oriented rectangle (and within [z_min, z_max] height), we flag "obstacle present".

Notes
-----
- This file does NOT read costmaps or call Nav2; it only inspects the RGB-D geometry.
- For performance, points are sub-sampled (param: voxel_stride) and clipped by [z_min, z_max].
- The "corridor_width" should be at least max(robot_width, box_width) for X-push, and
  at least max(robot_length, box_length) for Y-push — tweak per your setup.
- You can set L_star manually via params for now. Later you can compute it dynamically.
"""

import math
from typing import Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_py as tf2
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Duration


from collections import deque


# ----------------- tiny geometry helpers (match push orchestrator style) -----------------
def rot_local_to_world(cx, cy, yaw, lx, ly):
    return (cx + lx*math.cos(yaw) - ly*math.sin(yaw),
            cy + lx*math.sin(yaw) + ly*math.cos(yaw))

def point_in_oriented_rect(px, py, cx, cy, yaw, half_L, half_W) -> bool:
    dx, dy = px-cx, py-cy
    lx =  dx*math.cos(yaw) + dy*math.sin(yaw)
    ly = -dx*math.sin(yaw) + dy*math.cos(yaw)
    return (-half_L <= lx <= half_L) and (-half_W <= ly <= half_W)

def dir_unit_and_yaw(tag: str) -> Tuple[Tuple[float,float], float]:
    if tag == '+X': return (1.0,0.0), 0.0
    if tag == '-X': return (-1.0,0.0), math.pi
    if tag == '+Y': return (0.0,1.0), math.pi/2.0
    if tag == '-Y': return (0.0,-1.0), -math.pi/2.0
    return (1.0,0.0), 0.0

# ----------------- PointCloud helpers -----------------
def _pc2_to_xyz(np_from_iter) -> np.ndarray:
    """Convert an iterator of points (x,y,z) into an Nx3 numpy array, ignoring NaNs."""
    pts = np.array(list(np_from_iter), dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0,3)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    # Drop NaNs
    pts = pts[~np.isnan(pts).any(axis=1)]
    return pts

def _iterate_points_from_pc2(msg: PointCloud2, fields=('x','y','z')):
    """Yield (x,y,z) from a PointCloud2 quickly without importing pcl."""
    # Determine offsets
    field_map = {f.name: f for f in msg.fields}
    if not all(f in field_map for f in fields):
        return  # missing fields
    x_off = field_map['x'].offset
    y_off = field_map['y'].offset
    z_off = field_map['z'].offset
    step = msg.point_step
    data = msg.data  # bytes
    for i in range(0, msg.width * msg.height * step, step):
        x = np.frombuffer(data, dtype=np.float32, count=1, offset=i+x_off)[0]
        y = np.frombuffer(data, dtype=np.float32, count=1, offset=i+y_off)[0]
        z = np.frombuffer(data, dtype=np.float32, count=1, offset=i+z_off)[0]
        yield (x, y, z)

# ----------------- Main Node -----------------
class LStarObstacleProbe(Node):
    def __init__(self):
        super().__init__('lstar_obstacle_probe')


        self.first_cloud_received = False # Track if we've received the first point cloud

        self._decide_timer = self.create_timer(0.10, self._tick_decide)  # 10 Hz decision loop



        # --- Visibility/Obstacle params (ADD THESE) ---
        self.declare_parameter('grid_nx', 10)               # corridor ko nx x ny cells me split karo
        self.declare_parameter('grid_ny', 4)
        self.declare_parameter('vis_hi_thresh', 0.60)       # ≥60% -> good visibility
        self.declare_parameter('vis_lo_thresh', 0.30)       # 30–60% -> partial; <30% -> none
        self.declare_parameter('min_pts_obstacle', 25)      # itne points se zyada -> obstacle
        self.declare_parameter('min_cluster_side_m', 0.20)  # optional: density guard (approx cell size)
        self.declare_parameter('frames_smooth', 3)          # smoothing: last N frames union
        self.declare_parameter('stale_timeout_s', 0.5)      # itne sec me cloud na aaya -> Unknown

        self._roi_obs_cells_hist = deque(maxlen=int(self.get_parameter('frames_smooth').value))
        self._last_cloud_time = None  # rclpy.time.Time when last cloud processed

        # --- Parameters ---
        # Geometry
        self.declare_parameter('box_x', 6.5)
        self.declare_parameter('box_y', 0.5)
        self.declare_parameter('box_length', 0.50)
        self.declare_parameter('box_width',  2.0)

        self.declare_parameter('plan_dir', '+X')        # '+X' | '-X' | '+Y' | '-Y'
        self.declare_parameter('L_star', 0.60)          # meters: corridor length to check
        self.declare_parameter('corridor_width', 1.00)  # meters: corridor width (across)

        # Height band for 3D filtering
        self.declare_parameter('z_min', 0.02)           # ignore floor hits
        self.declare_parameter('z_max', 1.50)           # cap upper range

        # RGB-D input & TF
        self.declare_parameter('pointcloud_topic', '/realsense/depth/color/points')
        self.declare_parameter('camera_frame', 'realsense_depth_optical_frame')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('tf_timeout_s', 0.50)

        # Performance / Debug
        self.declare_parameter('voxel_stride', 4)       # take every Nth point
        self.declare_parameter('enable_debug_cloud', True)
        self.declare_parameter('enable_marker', True)

        # --- Publishers ---
        self.flag_pub  = self.create_publisher(Bool, 'lstar_obstacle_present', 10)
        self.dbg_cloud = self.create_publisher(PointCloud2, 'lstar_points_in_roi',  5)
        self.marker_pub= self.create_publisher(Marker, 'lstar_region_marker', 5)

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(sec=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscriptions ---
        topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        self.create_subscription(PointCloud2, topic, self._pc_cb, 10)

        # Render region marker at ~1 Hz
        self.marker_timer = self.create_timer(1.0, self._publish_region_marker)

        self.get_logger().info("L* obstacle node started")

    # -------- utilities --------
    def _box_face_anchor(self, L_star: float, plan_dir: str) -> Tuple[float,float,float,float,float]:
        """
        Return (cx, cy, yaw, half_L, half_W) for the oriented rectangle representing the L* region.
        The rectangle is centered at the middle of the corridor (halfway along length), starting
        flush with the relevant box face, extending L_star along plan_dir.
        """
        (ax, ay), yaw = dir_unit_and_yaw(plan_dir)
        bx = float(self.get_parameter('box_x').get_parameter_value().double_value)
        by = float(self.get_parameter('box_y').get_parameter_value().double_value)
        bl = float(self.get_parameter('box_length').get_parameter_value().double_value)
        bw = float(self.get_parameter('box_width').get_parameter_value().double_value)
        cw = float(self.get_parameter('corridor_width').get_parameter_value().double_value)

        # Box face center on the push side
        if plan_dir == '+X':
            face_x = bx + 0.5*bl; face_y = by
        elif plan_dir == '-X':
            face_x = bx - 0.5*bl; face_y = by
        elif plan_dir == '+Y':
            face_x = bx; face_y = by + 0.5*bw
        else: # '-Y'
            face_x = bx; face_y = by - 0.5*bw

        # Corridor center = face + (L*/2) * forward_dir
        cx = face_x + 0.5*L_star * ax
        cy = face_y + 0.5*L_star * ay

        half_L = 0.5 * L_star
        half_W = 0.5 * cw
        return cx, cy, yaw, half_L, half_W

    def _transform_points(self, pts_cam: np.ndarray, src_frame: str, dst_frame: str, timeout_s: float) -> Optional[np.ndarray]:
        """Transform Nx3 points from src_frame -> dst_frame using TF; returns Nx3 in dst frame, or None."""
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                dst_frame, src_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=timeout_s)
            )
        except Exception as e:
            self.get_logger().warn(f"[TF] lookup failed {src_frame}->{dst_frame}: {e}")
            return None

        # Build rotation (quaternion) and translation
        t = tf.transform.translation
        q = tf.transform.rotation
        # Rotation matrix from quaternion
        wx, wy, wz, ww = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(wy*wy + wz*wz),     2*(wx*wy - wz*ww),     2*(wx*wz + wy*ww)],
            [    2*(wx*wy + wz*ww), 1 - 2*(wx*wx + wz*wz),     2*(wy*wz - wx*ww)],
            [    2*(wx*wz - wy*ww),     2*(wy*wz + wx*ww), 1 - 2*(wx*wx + wy*wy)]
        ], dtype=np.float32)
        tvec = np.array([t.x, t.y, t.z], dtype=np.float32)

        # Apply transform: p_map = R * p_cam + t
        pts_map = (R @ pts_cam.T).T + tvec
        return pts_map

    # -------- main callbacks --------
    def _pc_cb(self, msg: PointCloud2):
        if not self.first_cloud_received:
            self.get_logger().info("✅ First PointCloud received, probe active now.")
            self.first_cloud_received = True

        stride   = int(self.get_parameter('voxel_stride').value)
        cam_frame = self.get_parameter('camera_frame').value
        map_frame = self.get_parameter('target_frame').value
        tf_to     = float(self.get_parameter('tf_timeout_s').value)

        # Convert cloud -> numpy (camera frame) with optional stride
        all_pts_iter = _iterate_points_from_pc2(msg)
        if stride > 1:
            all_pts_iter = (p for i, p in enumerate(all_pts_iter) if (i % stride == 0))
        pts_cam = _pc2_to_xyz(all_pts_iter)
        if pts_cam.size == 0:
            return

        # Transform to map
        pts_map = self._transform_points(pts_cam, cam_frame, map_frame, tf_to)
        if pts_map is None or pts_map.size == 0:
            return

        # --- Z-band filter (floor speckle cut) ---
        zmin = float(self.get_parameter('z_min').value)
        zmax = float(self.get_parameter('z_max').value)
        mask_z = (pts_map[:,2] >= zmin) & (pts_map[:,2] <= zmax)
        pts_map = pts_map[mask_z]
        if pts_map.size == 0:
            return

        # Cache + time
        self.last_pts_map = pts_map
        self._last_cloud_time = self.get_clock().now()

    def _publish_flag(self, present: bool):
        msg = Bool(data=present)
        self.flag_pub.publish(msg)
    def _world_to_roi_local(self, x, y, cx, cy, yaw):
        # inverse of rot_local_to_world: rotate/translate world->ROI
        dx, dy = x - cx, y - cy
        lx =  dx*math.cos(yaw) + dy*math.sin(yaw)
        ly = -dx*math.sin(yaw) + dy*math.cos(yaw)
        return lx, ly

    def _cell_index_from_local(self, lx, ly, hL, hW, nx, ny):
        # local ROI coords in [-hL,+hL] x [-hW,+hW] -> cell (ix,iy)
        if not (-hL <= lx <= hL and -hW <= ly <= hW):
            return None
        # normalize to [0,1)
        ux = (lx + hL) / (2*hL + 1e-9)
        uy = (ly + hW) / (2*hW + 1e-9)
        ix = min(nx-1, max(0, int(ux * nx)))
        iy = min(ny-1, max(0, int(uy * ny)))
        return (ix, iy)

    def _tick_decide(self):
        # Stale-data guard
        now = self.get_clock().now()
        stale_t = float(self.get_parameter('stale_timeout_s').value)
        if (self._last_cloud_time is None) or ((now - self._last_cloud_time).nanoseconds * 1e-9 > stale_t):
            # No fresh data -> Unknown visibility
            # Publish False for obstacle, but log unknown visibility
            self._publish_flag(False)
            return

        # ROI geometry
        plan_dir = self.get_parameter('plan_dir').get_parameter_value().string_value
        L_star   = float(self.get_parameter('L_star').get_parameter_value().double_value)
        cx, cy, yaw, hL, hW = self._box_face_anchor(L_star, plan_dir)

        nx = int(self.get_parameter('grid_nx').value)
        ny = int(self.get_parameter('grid_ny').value)
        map_frame = self.get_parameter('target_frame').value

        pts = getattr(self, 'last_pts_map', None)
        if pts is None or pts.size == 0:
            self._publish_flag(False)
            return

        # Filter points: inside oriented ROI
        inside_mask = []
        # Fast local transform + bounds check
        lxly = []
        for p in pts:
            lx, ly = self._world_to_roi_local(p[0], p[1], cx, cy, yaw)
            lxly.append((lx, ly))
            inside_mask.append((-hL <= lx <= hL) and (-hW <= ly <= hW))
        inside_mask = np.array(inside_mask, dtype=bool)
        roi_pts = pts[inside_mask]
        lxly = np.array(lxly, dtype=np.float32)[inside_mask]

        # Cells observed this frame
        observed = set()
        for (lx, ly) in lxly:
            idx = self._cell_index_from_local(lx, ly, hL, hW, nx, ny)
            if idx is not None:
                observed.add(idx)

        # Smooth across last N frames
        self._roi_obs_cells_hist.append(observed)
        observed_union = set().union(*list(self._roi_obs_cells_hist)) if self._roi_obs_cells_hist else observed

        # Expected = full corridor cells (your assumption)
        expected_total = nx * ny
        observed_count = len(observed_union)
        vis = (observed_count / max(1, expected_total))

        # Thresholding
        vis_hi = float(self.get_parameter('vis_hi_thresh').value)
        vis_lo = float(self.get_parameter('vis_lo_thresh').value)
        min_pts = int(self.get_parameter('min_pts_obstacle').value)

        if vis >= vis_hi:
            # Good visibility: decide obstacle
            # Optionally: light density check -> at least 'min_pts' points inside ROI
            is_blocked = (roi_pts.shape[0] >= min_pts)
            self._publish_flag(is_blocked)
            # Optional log (once per second you can throttle if needed)
            self.get_logger().info(f"[VIS OK] vis={vis*100:.0f}% | ROI pts={roi_pts.shape[0]} | blocked={is_blocked}")
            # Debug cloud of ROI points
            if bool(self.get_parameter('enable_debug_cloud').value):
                self._publish_debug_cloud(roi_pts, map_frame)

        elif vis_lo <= vis < vis_hi:
            # Partial visibility: ask for better angle; obstacle unknown -> be conservative (no-block or unknown)
            self._publish_flag(False)
            self.get_logger().warn(f"[VIS PARTIAL] vis={vis*100:.0f}% | Suggest small rotate/step to center ROI")
        else:
            # No visibility
            self._publish_flag(False)
            self.get_logger().warn(f"[VIS NONE] vis={vis*100:.0f}% | No decision (move to get view)")


    # -------- debug outputs --------
    def _publish_debug_cloud(self, pts_map: np.ndarray, frame: str):
        # craft minimal PointCloud2 with x,y,z
        pc = PointCloud2()
        pc.header.frame_id = frame
        pc.height = 1
        pc.width = pts_map.shape[0]
        pc.is_bigendian = False
        pc.is_dense = True
        pc.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc.point_step = 12
        pc.row_step = pc.point_step * pc.width
        pc.data = pts_map.astype(np.float32).tobytes()
        self.dbg_cloud.publish(pc)

    def _publish_region_marker(self):
        if not bool(self.get_parameter('enable_marker').get_parameter_value().bool_value):
            return

        plan_dir = self.get_parameter('plan_dir').get_parameter_value().string_value
        L_star   = float(self.get_parameter('L_star').get_parameter_value().double_value)
        cx, cy, yaw, hL, hW = self._box_face_anchor(L_star, plan_dir)

        # Four corners in world frame
        corners = [rot_local_to_world(cx, cy, yaw, sx, sy)
                   for sx in (-hL, hL) for sy in (-hW, hW)]  # (-L,-W),(L,-W),(L,W),(-L,W)

        # Build a LINE_STRIP rectangle
        m = Marker()
        m.header.frame_id = self.get_parameter('target_frame').get_parameter_value().string_value
        m.ns = "lstar_region"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03  # line width (m)
        # Default color (cyan-like): r,g,b,a
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 0.8
        # Close the loop by repeating the first point
        loop = corners + [corners[0]]
        for (x,y) in loop:
            p = Point(); p.x = x; p.y = y; p.z = 0.05
            m.points.append(p)
        self.marker_pub.publish(m)

# ----------------- main -----------------
def main():
    rclpy.init()
    node = LStarObstacleProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
