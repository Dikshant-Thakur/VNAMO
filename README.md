This repository contains a ROS 2 Humble package for simulating a **MiR 250 (Mobile Industrial Robot)** equipped with a **Universal Robots (UR5e)** arm and a **Robotiq Gripper**.

The project integrates **Navigation2**, **SLAM Toolbox**, and **YOLOv8** for autonomous navigation and object detection in a Gazebo environment.

## Key Features
* **Base Setup:** Derived from the robust [mir250_robot_ros2](https://github.com/Rudresh172/mir250_robot_ros2) structure.
* **Sensor Fusion:** Merges dual laser scanners (Front & Back) using `ira_laser_tools` for 360° coverage.
* **Perception:** Real-time object detection using **YOLOv8** (CUDA-accelerated) via `ros2_run`.
* **Manipulation:** UR5e arm control with MoveIt and `ros2_control`.
* **Synchronization:** Implements `twist_stamper` to fix TF synchronization issues for velocity commands.

---

## Installation

### 1. Preliminaries (ROS 2)
Ensure you have [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) installed on Ubuntu 22.04.

### 2. Setup Workspace
Create a workspace and clone this repository:
```bash
mkdir -p ~/ros2_ws_simulation/src
cd ~/ros2_ws_simulation/src

# Clone this repository
git clone <https://github.com/Dikshant-Thakur/VNAMO.git> .
```
### 3. Install Dependencies
This project requires both ROS 2 packages and Python libraries for YOLO/AI.
Step A: Import External Repos (via vcs)
```bash
# Ensure vcstool is installed
sudo apt install python3-vcstool

# Import dependencies defined in ros2.repos (e.g., mir_robot, ur_description)
vcs import < ros2.repos . --recursive
```

Step B: Install ROS Dependencies (rosdep)
```bash
cd ~/ros2_ws_simulation
sudo apt update
sudo apt install -y python3-rosdep
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```
Step C: Install Python Libraries (YOLO & Vision)
```bash
pip install ultralytics opencv-python opencv-contrib-python
# Install PyTorch with CUDA support (verify your CUDA version)
pip3 install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```


### 4. Build the Project
```bash
cd ~/ros2_ws_simulation
colcon build --symlink-install
source install/setup.bash
```
Usage
1. Launch Simulation (Gazebo)
```bash
ros2 launch mir_gazebo mir_gazebo_launch.py world:=maze rviz_config_file:=$(ros2 pkg prefix mir_navigation)/share/mir_navigation/rviz/mir_nav.rviz
```
2. Navigation & Mapping
To start SLAM (Mapping):
```bash
ros2 launch mir_navigation mapping.py use_sim_time:=true slam_params_file:=$(ros2 pkg prefix mir_navigation)/share/mir_navigation/config/mir_mapping_async_sim.yaml
```
To start Navigation (AMCL + Nav2):
```bash
ros2 launch mir_navigation amcl.py use_sim_time:=true map:=$(ros2 pkg prefix mir_navigation)/share/mir_navigation/maps/maze.yaml
```

Acknowledgements & Credits

* **Base Project Inspiration:** [mir250_robot_ros2](https://github.com/Rudresh172/mir250_robot_ros2) by Rudresh172.
* **MiR Descriptions:** [mir_robot](https://github.com/DFKI-NI/mir_robot) by DFKI-NI.
* **Universal Robots:** [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)[cite: 21].
* **Laser Tools:** [dual_laser_merger](https://github.com/pradyum/dual_laser_merger)[cite: 2].

## Video - Glimpse of Computer Vision + Navigation of Mir + Ur5e.
https://github.com/user-attachments/assets/84e8e1ee-dfe1-44b1-96c8-f438b1279440

