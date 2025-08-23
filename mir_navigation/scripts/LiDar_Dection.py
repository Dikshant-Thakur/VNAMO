#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, CompressedImage  # CompressedImage added
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, qos_profile_sensor_data
#qos_profile_sensor_data - ready-made recipe hai sensor data ke liye!
# QoSProfile -  aapko custom settings banane deta hai (qos_profile_sensor_data - bhi custom predefined h)
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        
        # CvBridge
        self.bridge = CvBridge()
        
        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_size = 0.5 # 20cm


        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # COMPRESSED IMAGE SUBSCRIBER - CHANGED
        self.image_sub = self.create_subscription(
            CompressedImage,  # Changed from Image
            '/realsense/camera/color/image_raw/compressed',  # Compressed topic
            self.compressed_image_callback,  # New callback
            qos_profile_sensor_data
        )
        
        # Camera info (same)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/realsense/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers (same)
        self.detection_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.marker_info_pub = self.create_publisher(String, '/marker_info', 10)
        
        self.markers_detected = False
        self.detection_count = 0
        
        self.get_logger().info('🎯 ArUco Detector Node Started (COMPRESSED)!')
        self.get_logger().info('📏 Dictionary: DICT_6X6_250')
        self.get_logger().info('🎥 Using COMPRESSED images...')

    def camera_info_callback(self, msg):
        """Get camera calibration parameters"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3) # (k stands for "camera matrix" means it contains intrinsic parameters)
            #fx, fy - how much the camera zooms in X and Y directions
            #cx, cy - the center point of the camera image
            self.dist_coeffs = np.array(msg.d) # distortion coefficients
            #camera lens ki mathematical prescription हैं जो बताती हैं कि lens कितना और कैसे image को bend/टेढ़ा करता है
            self.get_logger().info('✅ Camera calibration received!')

    def compressed_image_callback(self, msg):
        """Handle COMPRESSED images"""
        try:
            # DECODE COMPRESSED IMAGE - KEY STEP , Only for compressed images
            np_arr = np.frombuffer(msg.data, np.uint8) #msg data is in bytes format, so we convert it to a numpy array of type uint8
            #frombuffer: Bytes → NumPy array
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Convert numpy array to BGR(cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error('❌ Failed to decode compressed image!')
                return
            
            # बाकी सब same detection logic
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            #Convert BGR image to grayscale for better detection
            
            # Detect ArUco markers in gray scale image
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            ) #aruco_params - tells the accuracy
            
            # Detection flag
            detection_msg = Bool()
            detection_msg.data = False
            
            if ids is not None:
                detection_msg.data = True
                self.markers_detected = True
                self.detection_count += 1
                
                # Draw detected markers - Green squares and ID number
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Process each detected marker
                for i, marker_id in enumerate(ids.flatten()):
                    # Get marker center
                    center = corners[i][0].mean(axis=0) #axis=0 - mean of x's and y's separately
                    center_x, center_y = int(center[0]), int(center[1])
                    #Pixel values of center
                    
                    # Draw marker ID - redundant
                    # cv2.putText(cv_image, f"ID: {marker_id}", 
                    #            (center_x-30, center_y-20),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    #Draw center point - red dot
                    cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Log detection
                    self.get_logger().info(f'🎯 DETECTED! Marker ID: {marker_id} at ({center_x}, {center_y})')
                    
                    # Pose estimation (if camera is calibrated)
                    if self.camera_matrix is not None:
                        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                        )
                        # rvecs और tvecs ArUco marker की 3D position और orientation बताते हैं camera के relative में , 
                        # marker k center k relative mein
                        # rvecs - rotation vector
                        # tvecs - translation vector

                        # Calculate distance FIRST
                        distance = np.linalg.norm(tvecs[i])
                        # np.linalg.norm() = Euclidean Distance, 3D
                        
                        self.get_logger().info(f'📏 Distance: {distance:.2f}m')

                        try:
                            # Create point in camera frame
                            camera_point = PointStamped()
                            camera_point.header.frame_id = "realsense_color_optical_frame"
                            latest_time = self.tf_buffer.get_latest_common_time("realsense_color_optical_frame", "map")
                            #camera_point.header.stamp = latest_time.to_msg()  # Use latest common time
                            # camera_point.header.stamp = self.get_clock().now().to_msg()
                            camera_point.header.stamp = rclpy.time.Time().to_msg()  # Use latest available
                            camera_point.point.x = float(tvecs[i][0][0])  # Fixed: nested array access
                            camera_point.point.y = float(tvecs[i][0][1])  # Fixed: nested array access  
                            camera_point.point.z = float(tvecs[i][0][2])  # Fixed: nested array access
                            
                            # Transform to map frame
                            map_point = self.tf_buffer.transform(camera_point, "map")
                            
                            # Map coordinates
                            map_x = map_point.point.x
                            map_y = map_point.point.y
                            map_z = map_point.point.z
                            
                            self.get_logger().info(f'📍 Map coordinates: X={map_x:.2f}, Y={map_y:.2f}, Z={map_z:.2f}')
                            
                            # Publish detailed marker info with map coordinates
                            info_msg = String()
                            info_msg.data = f"ID:{marker_id},CamX:{center_x},CamY:{center_y},MapX:{map_x:.2f},MapY:{map_y:.2f},Dist:{distance:.2f}"
                            self.marker_info_pub.publish(info_msg)
                            
                        except Exception as e:
                            self.get_logger().error(f'❌ Transform failed: {str(e)}')
                            # Fallback to camera coordinates only
                            info_msg = String()
                            info_msg.data = f"ID:{marker_id},X:{center_x},Y:{center_y},Dist:{distance:.2f}"
                            self.marker_info_pub.publish(info_msg)

                        # Draw distance info
                        cv2.putText(cv_image, f"Dist: {distance:.2f}m", 
                                   (center_x-40, center_y+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Status display
                cv2.putText(cv_image, f"Found: {len(ids)} markers (COMPRESSED)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(cv_image, f"Total detections: {self.detection_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            else:
                # No markers detected
                detection_msg.data = False
                cv2.putText(cv_image, "No markers detected (COMPRESSED)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Always publish detection status
            self.detection_pub.publish(detection_msg)
            
            # Display image
            cv2.putText(cv_image, "ArUco Detection - COMPRESSED", 
                       (10, cv_image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('ArUco Detection', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'❌ Compressed image callback error: {str(e)}')

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    detector_node = ArucoDetectorNode()
    
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info('🛑 Shutting down ArUco Detector...')
    finally:
        cv2.destroyAllWindows()
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()