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


class LocalCostmapVisualizer(Node):
    def __init__(self):
        super().__init__('local_costmap_visualizer')
        self.get_logger().info("Local Costmap Ditto Visualizer started!")

        # Publisher for visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/costmap_visualization', 10)

        # TF buffer and listener for frame transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber to local costmap
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)

    def _costmap_cb(self, msg: OccupancyGrid):
        """Callback to visualize entire costmap with transformation from 'odom' to 'map' frame"""
        try:
            info = msg.info
            pos = info.origin.position
            orient = info.origin.orientation
            self.get_logger().info(
                f"Origin position: x={pos.x}, y={pos.y}, z={pos.z}; "
                f"orientation: x={orient.x}, y={orient.y}, z={orient.z}, w={orient.w}"
            )

            # Try to get transform from odom to map
            try:
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    'map',
                    msg.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as ex:
                self.get_logger().error(f"TF Transform error: {ex}")
                return

            data = np.array(msg.data, dtype=np.int8).reshape((info.height, info.width))

            markers = []
            free_points = []
            low_cost_points = []
            med_cost_points = []
            high_cost_points = []
            obstacle_points = []
            unknown_points = []

            for row in range(info.height):
                for col in range(info.width):
                    cost = data[row, col]

                    world_x = float(info.origin.position.x + (col + 0.5) * info.resolution)
                    world_y = float(info.origin.position.y + (row + 0.5) * info.resolution)

                    pt = Point(x=world_x, y=world_y, z=0.01)

                    pt_stamped = PointStamped()
                    pt_stamped.header = msg.header
                    pt_stamped.point = pt

                    try:
                        pt_transformed = do_transform_point(pt_stamped, transform)
                        transformed_point = pt_transformed.point
                    except Exception as e:
                        self.get_logger().error(f"Point transform error: {e}")
                        continue

                    if cost == -1:
                        unknown_points.append(transformed_point)
                    elif cost == 0:
                        free_points.append(transformed_point)
                    elif cost < 50:
                        low_cost_points.append(transformed_point)
                    elif cost < 80:
                        med_cost_points.append(transformed_point)
                    elif cost < 100:
                        high_cost_points.append(transformed_point)
                    else:
                        obstacle_points.append(transformed_point)

            stamp = self.get_clock().now().to_msg()

            if free_points:
                markers.append(self._create_marker(0, "free_space", free_points,
                                                   ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.3),
                                                   info.resolution, stamp))
            if low_cost_points:
                markers.append(self._create_marker(1, "low_cost", low_cost_points,
                                                   ColorRGBA(r=1.0, g=1.0, b=0.5, a=0.5),
                                                   info.resolution, stamp))
            if med_cost_points:
                markers.append(self._create_marker(2, "med_cost", med_cost_points,
                                                   ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7),
                                                   info.resolution, stamp))
            if high_cost_points:
                markers.append(self._create_marker(3, "high_cost", high_cost_points,
                                                   ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9),
                                                   info.resolution, stamp))
            if obstacle_points:
                markers.append(self._create_marker(4, "obstacles", obstacle_points,
                                                   ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
                                                   info.resolution, stamp))
            if unknown_points:
                markers.append(self._create_marker(5, "unknown", unknown_points,
                                                   ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6),
                                                   info.resolution, stamp))

            marker_array = MarkerArray(markers=markers)
            self.marker_pub.publish(marker_array)

            self.get_logger().info(f"Published transformed costmap visualization: "
                                   f"Free={len(free_points)}, Low={len(low_cost_points)}, Med={len(med_cost_points)}, "
                                   f"High={len(high_cost_points)}, Obstacles={len(obstacle_points)}, Unknown={len(unknown_points)}")

        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")

    def _create_marker(self, marker_id, namespace, points, color, resolution, stamp):
        marker = Marker()
        marker.header.frame_id = "map"  # Publish markers in map frame
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
    node = LocalCostmapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
