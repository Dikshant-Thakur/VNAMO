#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_point


class LocalCostmapHighCostVisualizer(Node):
    def __init__(self):
        super().__init__('local_costmap_highcost_visualizer')
        self.get_logger().info("Local Costmap High-Cost Visualizer started!")

        # Publisher for visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/costmap_highcost_visualization', 10)

        # TF buffer and listener for frame transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber to local costmap
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)

    def _costmap_cb(self, msg: OccupancyGrid):
        try:
            info = msg.info
            pos = info.origin.position
            orient = info.origin.orientation

            data = np.array(msg.data, dtype=np.int8).reshape((info.height, info.width))

            all_points = []           # sab points (for full map visuals)
            high_cost_points = []     # sirf cost >= 80 ke liye highlight

            for row in range(info.height):
                for col in range(info.width):
                    cost = data[row, col]

                    world_x = float(info.origin.position.x + (col + 0.5) * info.resolution)
                    world_y = float(info.origin.position.y + (row + 0.5) * info.resolution)

                    point = Point(x=world_x, y=world_y, z=0.01)

                    all_points.append(point)  # add all points

                    if cost >= 80:
                        high_cost_points.append(point)

            stamp = self.get_clock().now().to_msg()

            markers = []

            # Sab points light gray me
            if all_points:
                markers.append(self._create_marker(0, "all_points", all_points,
                                                ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.4),
                                                info.resolution, stamp))

            # High cost points red me highlight
            if high_cost_points:
                markers.append(self._create_marker(1, "high_cost_points", high_cost_points,
                                                ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9),
                                                info.resolution, stamp))

            marker_array = MarkerArray(markers=markers)
            self.marker_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")

    def _create_marker(self, marker_id, namespace, points, color, resolution, stamp):
        marker = Marker()
        marker.header.frame_id = "map"  # Publish markers in 'map' frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = float(resolution)
        marker.scale.z = 0.02
        marker.color = color
        marker.points = points
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmapHighCostVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
