# 🚀 Planner Architectures & Branching Strategy

This repository hosts two distinct planning algorithms across different branches. Specifically, the `recursive` branch introduces a hierarchical approach to problem-solving compared to the iterative nature of `master`.

### Comparison Table

| Feature | `master` Branch (Iterative) | `recursive` Branch (Hierarchical) |
| :--- | :--- | :--- |
| **Core Logic** | **Sequential / Reactive** | **Recursive / Nested** |
| **Obstacle Handling** | Detects obstacle $\rightarrow$ Aborts current plan $\rightarrow$ Generates a completely new plan for the new state. | Detects obstacle $\rightarrow$ Pauses current plan $\rightarrow$ Creates a **Child Plan** to remove the obstacle $\rightarrow$ Resumes Parent Plan. |
| **Solving Style** | Linear Replanning | **Recursive Recovery Strategy** (Backward Chaining) |

---

### 🧠 Logic Flow Visualization

#### 1. Iterative Approach (`master`)
> Make Plan A $\rightarrow$ Hit Obstacle $\rightarrow$  **Discard Plan A** $\rightarrow$ Make Plan B from scratch.

#### 2. Recursive Approach (`recursive`)
> Plan A (blocked by X) $\rightarrow$  **Pause Plan A** $\rightarrow$ Create **Child Plan B** (to remove X) $\rightarrow$  Execute B $\rightarrow$  **Resume Plan A**.


## VANAMO: Visibility-Aware Navigation Among Movable Obstacles
VANAMO is a ROS 2 based framework for mobile manipulation that solves the Navigation Among Movable Obstacles (NAMO) problem. Unlike traditional NAMO approaches, VANAMO introduces Visibility Constraints—ensuring that the robot verifies the target space is free before attempting to move an obstacle.

The system utilizes an AND/OR Graph (AOG) architecture with Graph Network Search (GNS) to dynamically switch between Navigation, Observation, Visibility Checks, and Manipulation (Push/Pull) behaviors.

## System Architecture
The system is designed as a hierarchical planner that orchestrates various Action Servers and Service Nodes.
High-Level Logic Flow
1. Global Navigation: The robot attempts to reach a goal using standard Nav2.
2. Failure Analysis: If the path is blocked, the planner identifies the blocking obstacle.
3. Observation: The robot positions its arm to detect the object (YOLOv8) and determines if it is Push_Movable or Pull_Movable.
4. Visibility Check: Before manipulation, the robot performs a "Look-Before-You-Sweep" check using PointClouds to ensure the area behind the object (L* corridor) is empty.
5. Manipulation: Depending on the obstacle type push/pull action will be done.
6. Resume: The graph updates, and the robot continues to the goal.
7. 
 ### Planner Architecture
 <img width="3160" height="1800" alt="Blank diagram" src="https://github.com/user-attachments/assets/b5c0d371-4576-4206-99db-b4c50f7fe0b6" />

### Action Stack Architecture
<img width="720" height="631" alt="Action2Stack" src="https://github.com/user-attachments/assets/d250a6ee-7648-4f6e-89ce-5aecd4e32665" />


### Recursive Plan (Full View)
> **Note:** Image is very wide. Click the image or the link below to zoom in.

[![Recursive Plan Graph](https://github.com/user-attachments/assets/63052793-6d4a-4fbf-9ab4-844d659232e4)](https://github.com/user-attachments/assets/63052793-6d4a-4fbf-9ab4-844d659232e4)

[🔍 **View Full Resolution Graph (High Quality)**](https://github.com/user-attachments/assets/63052793-6d4a-4fbf-9ab4-844d659232e4)

### Video
https://github.com/user-attachments/assets/20f50f2e-d379-417e-9556-66506fc4f5e7


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

