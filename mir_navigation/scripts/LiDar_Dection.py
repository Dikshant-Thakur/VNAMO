#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import random
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class SmartBlockingRegionNode(Node):
    """
    Fast obstacle detection for Nav2 global path corridor.

    Key optimizations:
      - Zero-copy style where possible & fewer dtype conversions
      - Vectorized expansion: unique() + isin() instead of per-label loops
      - Single-pass area computations with cached resolution^2
      - Lighter logging & short-circuits when nothing to publish
      - Robust path mask via cv2.polylines (continuous path)
    """

    def __init__(self):
        super().__init__('blocking_region_detector')
        self.get_logger().info("OpenCV Optimized Expansion Node started!")

        # ---- Tunables / Parameters ----
        self.robot_width = self.declare_parameter('robot_width', 0.5).value
        self.safety_width = self.declare_parameter('safety_width', 0.2).value

        # Minimum blocking-cluster area to consider (m^2)
        self.min_cluster_area = self.declare_parameter('min_cluster_area_m2', 0.2).value

        # Publish full (whole) obstacle regions that the blocking cluster belongs to
        self.detect_whole_obstacles = self.declare_parameter('detect_whole_obstacles', True).value
        self.min_whole_obstacle_area = self.declare_parameter('min_whole_obstacle_area_m2', 0.3).value

        # Cost thresholds
        self.blocking_threshold = self.declare_parameter('blocking_threshold', 90).value
        self.expansion_threshold = self.declare_parameter('expansion_threshold', 80).value

        # Morphology kernel for cleaning (configurable)
        ksize = self.declare_parameter('morph_kernel', 7).value
        ksize = int(max(3, ksize) | 1)  # odd >= 3
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # State
        self.global_path = None
        self.costmap = None
        random.seed(42)

        # Colors reused per publish
        self._palette = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9), ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
            ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9), ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9),
            ColorRGBA(r=0.5, g=0.0, b=0.5, a=0.9), ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.9),
            ColorRGBA(r=0.8, g=0.8, b=0.0, a=0.9), ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.9)
        ]

        # ROS I/O
        self.create_subscription(Path, '/plan', self._path_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)

        self.block_pub = self.create_publisher(MarkerArray, '/blocking_clusters', 10)
        self.whole_pub = self.create_publisher(MarkerArray, '/whole_obstacles', 10)
        self.corridor_pub = self.create_publisher(Marker, '/swept_corridor', 10)

    # -------------------- Callbacks --------------------

    def _path_cb(self, msg: Path):
        self.global_path = msg
        if self.costmap is not None:
            self._process()

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg
        if self.global_path is not None:
            self._process()

    # -------------------- Core Pipeline --------------------

    def _process(self):
        try:
            info = self.costmap.info
            h, w, res = info.height, info.width, info.resolution
            ox, oy = info.origin.position.x, info.origin.position.y
            res_sq = res * res

            if h == 0 or w == 0:
                self.get_logger().warn("Empty costmap dims; skip.")
                return

            # 1) Build masks
            path_mask = self._path_mask(h, w, res, ox, oy)
            corridor = self._corridor_mask(path_mask, res)

            # 2) Prepare costmap array (normalize unknowns to 0)
            cm = np.frombuffer(self.costmap.data, dtype=np.int8).reshape((h, w)).astype(np.int16, copy=False)
            # Treat unknown (-1) as free; set <0 -> 0 to avoid accidental high costs
            np.maximum(cm, 0, out=cm)

            # 3) Detect obstacles
            whole_masks, block_clusters = self._detect(cm, corridor, res_sq)

            # 4) Filter + Publish
            if corridor is not None:
                self._publish_corridor(corridor, info, ox, oy, res)

            if block_clusters:
                self._publish_all(whole_masks, block_clusters, info, ox, oy, res, res_sq)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

    # -------------------- Mask Builders --------------------

    def _path_mask(self, h: int, w: int, res: float, ox: float, oy: float) -> np.ndarray:
        """Robust 1px path polyline to avoid gaps between sparse waypoints."""
        mask = np.zeros((h, w), np.uint8)
        if not self.global_path.poses:
            return mask

        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in self.global_path.poses], dtype=np.float32)
        grid_xy = ((pts - np.array([ox, oy], dtype=np.float32)) / res).astype(np.int32)
        if len(grid_xy) >= 2:
            # OpenCV expects (x, y)
            poly = grid_xy.reshape(1, -1, 2)
            cv2.polylines(mask, [poly], isClosed=False, color=1, thickness=1, lineType=cv2.LINE_8)
        else:
            # Single point fallback
            x, y = grid_xy[0]
            if 0 <= x < w and 0 <= y < h:
                mask[y, x] = 1
        return mask

    def _corridor_mask(self, path_mask: np.ndarray, res: float) -> np.ndarray:
        rad = max(1, int(((self.robot_width * 0.5) + self.safety_width) / res))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        corridor = cv2.dilate(path_mask, k, iterations=1)
        return corridor

    # -------------------- Detection --------------------

    def _detect(self, cm: np.ndarray, corridor: np.ndarray, res_sq: float):
        """
        Returns:
            whole_masks: list[np.ndarray uint8] - expanded full obstacles for each blocking cluster
            block_clusters: list[np.ndarray uint8] - raw blocking clusters intersecting path corridor
        """
        # Blocking candidates intersecting the corridor
        blocking_mask = (cm >= self.blocking_threshold).astype(np.uint8, copy=False)
        intersect = cv2.bitwise_and(blocking_mask, corridor)

        # Clean intersections (morph close->open once; cheap but effective)
        if self._kernel is not None:
            intersect = cv2.morphologyEx(intersect, cv2.MORPH_CLOSE, self._kernel, iterations=1)
            intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, self._kernel, iterations=1)

        # Connected components over intersections -> blocking clusters
        n_block, block_labels, block_stats, _ = cv2.connectedComponentsWithStats(intersect, connectivity=8)
        if n_block <= 1:
            return [], []  # only background

        # Expansion universe: all high-cost (>= expansion_threshold)
        expansion_mask = (cm >= self.expansion_threshold).astype(np.uint8, copy=False)
        if self._kernel is not None:
            expansion_mask = cv2.morphologyEx(expansion_mask, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        # Label expansion regions once
        _, exp_labels = cv2.connectedComponents(expansion_mask, connectivity=8)

        whole_masks = []
        block_clusters = []

        # Iterate blocking components only; vectorized expansion via label sets
        for label_idx in range(1, n_block):
            pix = block_stats[label_idx, cv2.CC_STAT_AREA]
            area_m2 = pix * res_sq
            if area_m2 < self.min_cluster_area:
                continue

            # Boolean mask for this blocking cluster
            cluster = (block_labels == label_idx)

            # Expansion: find which expansion-labels intersect this cluster
            # NOTE: unique labels of expansion under cluster -> combine using isin
            intersecting_labels = np.unique(exp_labels[cluster])
            if intersecting_labels.size and intersecting_labels[0] == 0:
                intersecting_labels = intersecting_labels[1:]  # drop background

            if intersecting_labels.size == 0:
                # No parent expansion region? Keep cluster alone as whole (rare but safe)
                whole = cluster.astype(np.uint8)
            else:
                whole = np.isin(exp_labels, intersecting_labels, assume_unique=False).astype(np.uint8)

            whole_masks.append(whole)
            block_clusters.append(cluster.astype(np.uint8))

        return whole_masks, block_clusters

    # -------------------- Publishing --------------------

    def _publish_all(self, whole_masks, block_clusters, info, ox, oy, res, res_sq):
        # Filter by min whole obstacle area
        filtered_whole = []
        filtered_block = []

        for wmask, bmask in zip(whole_masks, block_clusters):
            area_m2 = float(np.count_nonzero(wmask)) * res_sq
            if area_m2 >= self.min_whole_obstacle_area:
                filtered_whole.append(wmask)
                filtered_block.append(bmask)

        if not filtered_block:
            # Nothing valid -> still announce that we found none
            self.get_logger().info("No valid blocking clusters after filtering.")
            return

        # Colors
        n = len(filtered_block)
        cluster_colors = []
        whole_colors = []
        for i in range(n):
            if i < len(self._palette):
                c = self._palette[i]
            else:
                c = ColorRGBA(r=random.uniform(0.3, 1.0),
                              g=random.uniform(0.3, 1.0),
                              b=random.uniform(0.3, 1.0),
                              a=0.9)
            cluster_colors.append(c)
            whole_colors.append(ColorRGBA(r=c.r * 0.5, g=c.g * 0.5, b=c.b * 0.5, a=0.4))

        # Common header timestamp for consistency
        stamp = self.get_clock().now().to_msg()

        # Publish blocking clusters
        block_msgs = []
        for i, cl in enumerate(filtered_block):
            if cl is None or cl.dtype == bool:
                cl = cl.astype(np.uint8)
            if np.count_nonzero(cl) == 0:
                continue
            m = self._make_marker("blocking_clusters", i, Marker.CUBE_LIST, 0.08, stamp)
            m.color = cluster_colors[i]
            m.scale.x = m.scale.y = res
            m.points = self._coords_to_points(np.argwhere(cl > 0), ox, oy, res, 0.08)
            block_msgs.append(m)

        if block_msgs:
            self.block_pub.publish(MarkerArray(markers=block_msgs))
            self.get_logger().info(f"Published {len(block_msgs)} blocking clusters.")
        else:
            self.get_logger().info("No blocking clusters to publish after post-check.")

        # Publish whole obstacles (optional)
        if self.detect_whole_obstacles:
            whole_msgs = []
            for i, wm in enumerate(filtered_whole):
                if np.count_nonzero(wm) == 0:
                    continue
                m = self._make_marker("whole_obstacles", i, Marker.CUBE_LIST, 0.05, stamp)
                m.color = whole_colors[i]
                m.scale.x = m.scale.y = res
                m.points = self._coords_to_points(np.argwhere(wm > 0), ox, oy, res, 0.05)
                whole_msgs.append(m)
            if whole_msgs:
                self.whole_pub.publish(MarkerArray(markers=whole_msgs))
                self.get_logger().info(f"Published {len(whole_msgs)} whole obstacles.")

    def _publish_corridor(self, corridor: np.ndarray, info, ox, oy, res: float):
        pts = np.argwhere(corridor > 0)
        if pts.size == 0:
            return
        m = self._make_marker("swept_corridor", 0, Marker.CUBE_LIST, 0.03, self.get_clock().now().to_msg())
        m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)
        m.scale.x = m.scale.y = res
        m.points = self._coords_to_points(pts, ox, oy, res, 0.03)
        self.corridor_pub.publish(m)

    # -------------------- Helpers --------------------

    def _make_marker(self, ns: str, mid: int, mtype: int, z: float, stamp):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.scale.z = z
        return m

    @staticmethod
    def _coords_to_points(coords_rc: np.ndarray, ox: float, oy: float, res: float, z: float):
        """
        coords_rc: Nx2 array of [row, col] indices
        Returns list[geometry_msgs/Point]
        """
        if coords_rc.size == 0:
            return []

        # Convert (row, col) -> world (x, y)
        cols = coords_rc[:, 1].astype(np.float32)
        rows = coords_rc[:, 0].astype(np.float32)
        xs = ox + cols * res + res * 0.5
        ys = oy + rows * res + res * 0.5

        # Build list of Points
        zf = float(z)
        return [Point(x=float(x), y=float(y), z=zf) for x, y in zip(xs, ys)]

# -------------------- Main --------------------

def main(args=None):
    rclpy.init(args=args)
    node = SmartBlockingRegionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
