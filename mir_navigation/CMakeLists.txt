cmake_minimum_required(VERSION 3.5)
project(mir_navigation)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(navigation2 REQUIRED)
find_package(slam_toolbox REQUIRED)
find_package(action_msgs REQUIRED)

# ArUco Detection Dependencies
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)

#YOLO
find_package(vision_msgs REQUIRED)



## Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ManipulateObstacle.action"
  "action/CheckVisibility.action"
  "action/ObserveObstacle.action"
  "srv/ComputePreManipPose.srv"
  "srv/ComputeSidePeekPoints.srv"
  DEPENDENCIES builtin_interfaces action_msgs std_msgs geometry_msgs
)

# Install resources

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config launch rviz maps behavior_trees
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/Planner_explanation.py
  DESTINATION lib/${PROJECT_NAME}
)

# observe_obstacle_server
install(PROGRAMS scripts/observe_obstacle_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME observe_obstacle_server
)

# push_geometry_server
install(PROGRAMS scripts/push_geometry_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME push_geometry_server
)

# push_manipulation_action_server
install(PROGRAMS scripts/push_manipulation_action_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME push_manipulation_action_server
)

# lstar_obstacle_checker -> visibility_action
install(PROGRAMS scripts/visibility_action.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME visibility_action
)


install(PROGRAMS scripts/yolo_detector.py
  DESTINATION lib/${PROJECT_NAME}
)



# Tests (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies - UPDATE THIS
ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  rclpy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  OpenCV
  image_transport
  tf2_ros
  tf2_geometry_msgs
  vision_msgs
)

ament_package()
