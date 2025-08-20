#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import random
from collections import deque
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class SmartBlockingRegionNode(Node):
    def __init__(self):
        super().__init__('blocking_region_detector')
        self.get_logger().info("OpenCV Optimized Expansion Node started!")

        self.robot_width = self.declare_parameter('robot_width', 0.5).value
        self.safety_width = self.declare_parameter('safety_width', 0.2).value
        self.min_cluster_area = self.declare_parameter('min_cluster_area_m2', 0.2).value
        self.detect_whole_obstacles = self.declare_parameter('detect_whole_obstacles', True).value
        self.min_whole_obstacle_area = self.declare_parameter('min_whole_obstacle_area_m2', 0.3).value
        self.blocking_threshold = self.declare_parameter('blocking_threshold', 90).value
        self.expansion_threshold = self.declare_parameter('expansion_threshold', 80).value

        self._colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9), ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
            ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9), ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9),
            ColorRGBA(r=0.5, g=0.0, b=0.5, a=0.9), ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.9),
            ColorRGBA(r=0.8, g=0.8, b=0.0, a=0.9), ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.9)]

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        self.global_path = None
        self.costmap = None
        self.cluster_colors = []
        self.whole_colors = []
        random.seed(42)

        self.create_subscription(Path, '/plan', self.path_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.costmap_cb, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/blocking_clusters', 10)
        self.whole_pub = self.create_publisher(MarkerArray, '/whole_obstacles', 10)
        self.corridor_pub = self.create_publisher(Marker, '/swept_corridor', 10)

    def path_cb(self, msg):
        self.global_path = msg
        self.try_process()

    def costmap_cb(self, msg):
        self.costmap = msg
        self.try_process()

    def try_process(self):
        if self.global_path and self.costmap:
            self.process()

    def process(self):
        try:
            info = self.costmap.info
            h,w,res = info.height, info.width, info.resolution
            ox, oy = info.origin.position.x, info.origin.position.y

            path_mask = self.create_path_mask(h,w,res,ox,oy)
            corridor = self.create_corridor_mask(path_mask,res)
            costmap_arr = np.frombuffer(self.costmap.data, dtype=np.int8).reshape((h,w)).astype(np.int16)

            masks, clusters = self.detect_obstacles_optimized(costmap_arr, corridor)
            self.filter_publish(masks, clusters, info, ox, oy, res)
            self.publish_corridor(corridor, info, ox, oy, res)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def create_path_mask(self,h,w,res,ox,oy):
        mask = np.zeros((h,w), np.uint8)
        if not self.global_path.poses: return mask
        pos = np.array([[p.pose.position.x, p.pose.position.y] for p in self.global_path.poses])
        gc = ((pos - [ox,oy]) / res).astype(int)
        vm = (gc[:,0]>=0) & (gc[:,0]<w) & (gc[:,1]>=0) & (gc[:,1]<h)
        gc = gc[vm]
        mask[gc[:,1], gc[:,0]] = 1
        self.get_logger().info(f"Path mask points: {len(gc)}")
        return mask

    def create_corridor_mask(self, path_mask, res):
        rad = max(1, int(((self.robot_width/2) + self.safety_width)/res))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))
        corridor = cv2.dilate(path_mask, kernel)
        self.get_logger().info(f"Corridor size: {np.count_nonzero(corridor)}")
        return corridor

    def detect_obstacles_optimized(self, costmap_arr, corridor_mask):
        """OpenCV optimized obstacle detection with connected components"""
        # Step 1: Find blocking obstacles in corridor
        blocking_mask = (costmap_arr >= self.blocking_threshold).astype(np.uint8)
        intersect = cv2.bitwise_and(blocking_mask, corridor_mask)
        intersect = cv2.morphologyEx(intersect, cv2.MORPH_CLOSE, self._kernel)
        intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, self._kernel)

        # Step 2: Get blocking clusters using connected components
        n_block, block_labels, block_stats, _ = cv2.connectedComponentsWithStats(intersect, 8)
        resolution_sq = self.costmap.info.resolution ** 2

        # Step 3: Create expansion mask for all high-occupancy areas
        expansion_mask = (costmap_arr >= self.expansion_threshold).astype(np.uint8)
        expansion_mask = cv2.morphologyEx(expansion_mask, cv2.MORPH_CLOSE, self._kernel)
        
        # Step 4: Get all expansion regions
        n_exp, exp_labels = cv2.connectedComponents(expansion_mask, 8)

        masks, valid_clusters = [], []
        
        for i in range(1, n_block):
            area = block_stats[i, cv2.CC_STAT_AREA] * resolution_sq
            if area < self.min_cluster_area: continue
            
            cluster = (block_labels == i).astype(np.uint8)
            valid_clusters.append(cluster)

            # Find which expansion components intersect with this blocking cluster
            whole = self.opencv_expand_region(cluster, exp_labels, n_exp)
            whole = cv2.morphologyEx(whole, cv2.MORPH_CLOSE, self._kernel)
            masks.append(whole)

            blocking_area = np.count_nonzero(cluster) * resolution_sq
            whole_area = np.count_nonzero(whole) * resolution_sq
            self.get_logger().info(f"OpenCV Obs {len(masks)} blocking={blocking_area:.3f} whole={whole_area:.3f} ratio={whole_area/max(blocking_area,1e-6):.1f}")
        
        return masks, valid_clusters

    def opencv_expand_region(self, blocking_cluster, expansion_labels, num_labels):
        """Fast OpenCV-based region expansion using connected components"""
        result = blocking_cluster.copy().astype(bool)
        
        # Find which expansion components intersect with blocking cluster
        intersecting_labels = set()
        for label in range(1, num_labels):
            component_mask = (expansion_labels == label)
            if np.any(cv2.bitwise_and(blocking_cluster, component_mask.astype(np.uint8))):
                intersecting_labels.add(label)
        
        # Combine all intersecting expansion components
        for label in intersecting_labels:
            component_mask = (expansion_labels == label)
            result = np.logical_or(result, component_mask)
        
        return result.astype(np.uint8)

    def assign_colors(self,num):
        self.cluster_colors, self.whole_colors = [],[]
        for i in range(num):
            if i<len(self._colors): c = self._colors[i]
            else: c = ColorRGBA(r=random.uniform(.3,1), g=random.uniform(.3,1), b=random.uniform(.3,1), a=0.9)
            self.cluster_colors.append(c)
            wc = ColorRGBA(r=c.r*0.5, g=c.g*0.5, b=c.b*0.5, a=0.4)
            self.whole_colors.append(wc)

    def filter_publish(self,masks, clusters, info, ox, oy, res):
        valid_m, valid_c = [], []
        for m,c in zip(masks, clusters):
            area = np.count_nonzero(m)*(res**2)
            if area>=self.min_whole_obstacle_area:
                valid_m.append(m)
                valid_c.append(c)
        if not valid_m: return
        self.assign_colors(len(valid_m))
        self.publish_clusters(valid_c, info, ox, oy, res)
        if self.detect_whole_obstacles:
            self.publish_wholes(valid_m, info, ox, oy, res)

    def publish_clusters(self, clusters, info, ox, oy, res):
        markers = []
        for i, cl in enumerate(clusters):
            if np.count_nonzero(cl)==0: continue
            m = self.create_marker("blocking_clusters", i, Marker.CUBE_LIST, 0.08)
            m.color = self.cluster_colors[i] if i < len(self.cluster_colors) else ColorRGBA(r=1.0,g=0.0,b=0.0,a=0.9)
            m.scale.x = m.scale.y = res
            pts = self.coords_to_points(np.column_stack(np.where(cl>0)), ox, oy, res, 0.08)
            m.points = pts
            markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info(f"Published {len(markers)} OpenCV clusters")

    def publish_wholes(self, masks, info, ox, oy, res):
        markers = []
        for i,m in enumerate(masks):
            mkr = self.create_marker("whole_obstacles", i, Marker.CUBE_LIST, 0.05)
            mkr.color = self.whole_colors[i] if i < len(self.whole_colors) else ColorRGBA(r=0.0,g=0.8,b=0.0,a=0.4)
            mkr.scale.x = mkr.scale.y = res
            pts = self.coords_to_points(np.column_stack(np.where(m>0)), ox, oy, res, 0.05)
            mkr.points = pts
            markers.append(mkr)
        self.whole_pub.publish(MarkerArray(markers=markers))
        self.get_logger().info(f"Published {len(markers)} OpenCV obstacles")

    def publish_corridor(self, corridor, info, ox, oy, res):
        pts = np.column_stack(np.where(corridor>0))
        if len(pts)==0:
            self.get_logger().warn("No corridor points")
            return
        m = self.create_marker("swept_corridor", 0, Marker.CUBE_LIST, 0.03)
        m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)
        m.scale.x = m.scale.y = res
        m.points = self.coords_to_points(pts, ox, oy, res, 0.03)
        self.corridor_pub.publish(m)
        self.get_logger().info(f"Published corridor {len(pts)} pts")

    def create_marker(self, ns, mid, mtype, z):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.scale.z = z
        return m

    def coords_to_points(self, coords, ox, oy, res, z):
        if len(coords)==0: return []
        wc = np.column_stack([ox + coords[:,1]*res + res/2, oy + coords[:,0]*res + res/2, np.full(len(coords), z)])
        return [Point(x=float(x), y=float(y), z=float(z)) for x,y,z in wc]


def main(args=None):
    rclpy.init(args=args)
    node = SmartBlockingRegionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
