cmake_minimum_required(VERSION 3.5)
project(gripper_description)

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY urdf models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
