## 🌳 Branching Strategy: Planner Architectures

This repository hosts two distinct planning algorithms across different branches. Here is how they differ:

| Feature | `master` Branch (Iterative) | `recursive` Branch (Hierarchical) |
| :--- | :--- | :--- |
| **Core Logic** | **Sequential / Reactive** | **Recursive / Nested** |
| **Obstacle Handling** | Detects obstacle $\rightarrow$ Aborts current plan $\rightarrow$ Generates a completely new plan for the new state. | Detects obstacle $\rightarrow$ Pauses current plan $\rightarrow$ Creates a **Child Plan** to remove the obstacle $\rightarrow$ Resumes Parent Plan. |
| **Solving Style** | Linear Replanning | **Hierarchical Sub-goal Decomposition** (Backward Chaining) |
| **Complexity** | Low (Good for simple, changing environments) | High (Good for complex, inter-dependent tasks) |

---
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

 ### Planner Architecture
 <img width="3160" height="1800" alt="Blank diagram" src="https://github.com/user-attachments/assets/b5c0d371-4576-4206-99db-b4c50f7fe0b6" />


### Action Stack Architecture
<img width="720" height="631" alt="Action2Stack" src="https://github.com/user-attachments/assets/d250a6ee-7648-4f6e-89ce-5aecd4e32665" />

### Planner Scenarios (Non-recursive Nature)

<details>
  <summary><strong>❌ Plan 1 -> Failure</strong> (Click to Expand)</summary>
  <img src="https://github.com/user-attachments/assets/10f66d62-7f69-4d5f-be03-9f491c1789b5" width="450" />
</details>

<details>
  <summary><strong>❌ Plan 2 (For Red Box - Push Obstacle) -> Fail</strong></summary>
  <img src="https://github.com/user-attachments/assets/828e5173-f6e5-4107-a3da-fcc30b758d8a" width="450" />
</details>

<details>
  <summary><strong>❌ Plan 3 (For Chair - Pull Obstacle) -> Fail</strong></summary>
  <img src="https://github.com/user-attachments/assets/5fcd1835-bff9-4d7f-af07-f3d924461e00" width="450" />
</details>

### ✅ Plan 4 (For Red Box - Push Obstacle)
<img src="https://github.com/user-attachments/assets/828e5173-f6e5-4107-a3da-fcc30b758d8a" width="450" />

### Video
https://github.com/user-attachments/assets/92790f41-5e22-4a15-99f7-1fbb54a67bbd


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

