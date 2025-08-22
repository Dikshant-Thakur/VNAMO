#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, qos_profile_sensor_data

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # CvBridge
        self.bridge = CvBridge()
        
        # ArUco Setup - आपकी same dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Marker size (adjust करें जरूरत के हिसाब से)
        self.marker_size = 0.2  # 20cm
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 
            '/realsense/camera/color/image_raw',  # Change topic according to your camera
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/realsense/camera/color/camera_info',  # Change topic according to your camera
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.marker_info_pub = self.create_publisher(String, '/marker_info', 10)
        
        # Status variables
        self.markers_detected = False
        self.detection_count = 0
        
        self.get_logger().info('🎯 ArUco Detector Node Started!')
        self.get_logger().info('📏 Dictionary: DICT_6X6_250')
        self.get_logger().info('📐 Marker Size: 20cm')
        self.get_logger().info('🎥 Waiting for camera feed...')
    
    def camera_info_callback(self, msg):
        """Get camera calibration parameters"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('✅ Camera calibration received!')
    
    def image_callback(self, msg):
        """Main detection callback"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
            
            # Detection flag
            detection_msg = Bool()
            detection_msg.data = False
            
            if ids is not None:
                # Markers detected!
                detection_msg.data = True
                self.markers_detected = True
                self.detection_count += 1
                
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Process each detected marker
                for i, marker_id in enumerate(ids.flatten()):
                    # Get marker center
                    center = corners[i][0].mean(axis=0)
                    center_x, center_y = int(center[0]), int(center[1])
                    
                    # Draw marker info
                    cv2.putText(cv_image, f"ID: {marker_id}", 
                               (center_x-30, center_y-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Log detection
                    self.get_logger().info(f'🎯 DETECTED! Marker ID: {marker_id} at ({center_x}, {center_y})')
                    
                    # Pose estimation (if camera is calibrated)
                    if self.camera_matrix is not None:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                        )
                        
                        # Calculate distance
                        distance = np.linalg.norm(tvecs[i])
                        
                        # Draw distance info
                        cv2.putText(cv_image, f"Dist: {distance:.2f}m", 
                                   (center_x-40, center_y+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        self.get_logger().info(f'📏 Distance: {distance:.2f}m')
                        
                        # Publish detailed marker info
                        info_msg = String()
                        info_msg.data = f"ID:{marker_id},X:{center_x},Y:{center_y},Dist:{distance:.2f}"
                        self.marker_info_pub.publish(info_msg)
                
                # Status display
                cv2.putText(cv_image, f"Found: {len(ids)} markers", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(cv_image, f"Total detections: {self.detection_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            else:
                # No markers detected
                detection_msg.data = False
                cv2.putText(cv_image, "No markers detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Always publish detection status
            self.detection_pub.publish(detection_msg)
            
            # Display image (optional - comment out if not needed)
            cv2.putText(cv_image, "ArUco Detection - DICT_6X6_250", 
                       (10, cv_image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('ArUco Detection', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'❌ Error in image callback: {str(e)}')
    
    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    
    # Create and run the node
    detector_node = ArucoDetectorNode()
    
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info('🛑 Shutting down ArUco Detector...')
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
