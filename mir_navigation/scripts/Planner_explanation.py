#!/usr/bin/env python3
from turtle import speed

from matplotlib.pyplot import step
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
import math
from enum import Enum
import time
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from graphviz import Digraph
import os
from datetime import datetime
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor   #SINGLE-THREADING for cleaner execution
from rclpy.callback_groups import ReentrantCallbackGroup #Callback group for reentrant callbacks
import threading # Import the threading library
from mir_navigation.action import ObserveObstacle, CheckVisibility, ManipulateObstacle, PullTrigger
from mir_navigation.srv import ComputeSidePeekPoints, ComputePreManipPose









# Utility function for graph visualization - define before classes
def visualize_graph_structure_graphviz(graph, filename="graph_output"):
    """Visualize AND/OR graph using Graphviz"""
    dot = Digraph(comment="AND/OR Graph")
    # Add nodes
    for name, node in graph.nodes.items():
        dot.node(name, f"{name}\n({node.type})", shape="circle")
    # Add arcs (edges)
    for i, arc in enumerate(graph.hyperarcs):
        parent = arc.parent.name
        for child in arc.children:
            dot.edge(child.name, parent, label=arc.action)
    # Save and render
    try:
        dot.render(filename, view=True, format='png')
        print(f"Graph rendered to {filename}.png")
    except Exception as e:
        print(f"Graph visualization failed: {e}")


class PlanningState(Enum):
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    UPDATING = "UPDATING"
    GOAL_REACHED = "GOAL_REACHED"
    FAILED = "FAILED"


class ExecutionResult:
    """Result of executing a hyperarc"""
    def __init__(self, success, new_position=None, map_updates=None, 
                 failure_type=None, blocked_region=None, 
                 blocking_obstacle=None, failure_context=None):
        self.success = success
        self.new_position = new_position
        self.map_updates = map_updates or []
        self.failure_type = failure_type  # "VISIBILITY_ISSUE" or "MANIPULATION_ISSUE"
        self.blocked_region = blocked_region  # For visibility issues
        self.blocking_obstacle = blocking_obstacle  # For manipulation issues
        self.failure_context = failure_context  # Additional context


    def __str__(self):
        if self.success:
            return "ExecutionResult(SUCCESS)"
        else:
            return f"ExecutionResult(FAILED: {self.failure_type})"

class AOGNode:
    """Node in an AND/OR Graph"""
    def __init__(self, name, node_type):
        self.name = name
        self.type = node_type
        self.status = "PENDING"  # PENDING, ACHIEVED
        self.incoming_arcs = []
        self.outgoing_arcs = []
        
    def add_incoming_arc(self, arc):
        self.incoming_arcs.append(arc)
        
    def add_outgoing_arc(self, arc):
        self.outgoing_arcs.append(arc)


class HyperArc:
    """Hyperarc in an AND/OR Graph"""
    def __init__(self, parent, children, action, action_params=None, condition=None):
        self.parent = parent
        self.children = children
        self.action = action
        self.action_params = action_params or {}
        self.condition = condition or (lambda: True)
        self.status = "PENDING"  # PENDING, READY, EXECUTING, SUCCEEDED, FAILED
        self.cost = 1.0  # Default cost


class AOGGraph:
    """AND/OR Graph structure"""
    def __init__(self):
        self.nodes = {}
        self.hyperarcs = []
        
    def add_node(self, name, node_type):
        """Add a node to the graph"""
        node = AOGNode(name, node_type)
        self.nodes[name] = node
        return node
        
    def add_hyperarc(self, parent, children, action, action_params=None, condition=None):
        """Add a hyperarc to the graph"""
        if action_params is None:
            action_params = {}
            
        arc = HyperArc(parent, children, action, action_params, condition)
        self.hyperarcs.append(arc)
        parent.add_incoming_arc(arc)    
        for child in children:
            child.add_outgoing_arc(arc)
        return arc
        
    def get_node(self, name):
        """Get a node by name"""
        return self.nodes.get(name)


class AOGModule:
    """AOG Module - Handles graph structure and creation"""
    def __init__(self):
        self.node_counter = 0
        
    def create_initial_graph(self):
        """Create initial AND/OR graph with basic structure"""
        graph = AOGGraph()
        
        # Core nodes
        n_final = graph.add_node("N_FINAL", node_type="final")
        n_current_config = graph.add_node("N_CURRENT_CONFIG", node_type="config")

        n_current_config.status = "ACHIEVED"   # Robot starts here
        
        # Direct navigation arc from current config to final goal
        arc_navigate = graph.add_hyperarc(
            parent=n_final,
            children=[n_current_config],
            action="NAVIGATE_TO_FINAL_GOAL",
            action_params={"node": n_final},  # Pass node object itself
            condition=lambda: True
        )
        
        self._update_initial_arc_statuses(graph)
        return graph
    
    def _update_initial_arc_statuses(self, graph):
        """Set initial arc statuses based on node states"""
        for arc in graph.hyperarcs:
            all_children_achieved = all(child.status == "ACHIEVED" for child in arc.children)
            if all_children_achieved and arc.condition():
                arc.status = "READY"
            else:
                arc.status = "PENDING"
    
    def expand_graph_for_visibility(self, obstacle):
        """
        Build the visibility sub-graph using the existing AOGGraph API.

        N_OBSERVATION
        --V0:NAVIGATE_TO_VIS_POINT-->  N_VIS_POINT
        --V1:VISIBILITY_ACTION------->  N_VIS_DONE
        """
        new_graph = AOGGraph()

        # Nodes (use your existing node_type strings)
        n_observation = new_graph.add_node("N_OBSERVATION", node_type="navigation")
        n_vis_point   = new_graph.add_node("N_VIS_POINT",   node_type="navigation")
        n_vis_done    = new_graph.add_node("N_VIS_DONE",    node_type="flag")

        # V0: go to visibility point from observation
        new_graph.add_hyperarc(
            parent=n_vis_point,
            children=[n_observation],
            action="NAVIGATE_TO_VIS_POINT",
            action_params={"obstacle": obstacle},
            condition=lambda: True,
        )

        # V1: run visibility action → mark done
        new_graph.add_hyperarc(
            parent=n_vis_done,
            children=[n_vis_point],
            action="VISIBILITY_ACTION",
            action_params={"obstacle": obstacle},
            condition=lambda: True,
        )

        # Initialize arc statuses using your existing utility
        self._update_initial_arc_statuses(new_graph)
        return new_graph


    def expand_graph_for_manipulation(self, obstacle, q_goal, include_final=True):

        """Manip + Visibility structure (existing AOGGraph API)."""
        new_graph = AOGGraph()

        # Nodes
        n_current          = new_graph.add_node("N_CURRENT_CONFIG",    node_type="config"); n_current.status = "ACHIEVED"
        n_q_observation    = new_graph.add_node("N_Q_OBSERVATION",    node_type="navigation")
        n_observation_done = new_graph.add_node("N_OBSERVATION_DONE", node_type="flag")
        n_vis_point        = new_graph.add_node("N_VIS_POINT",         node_type="navigation")
        n_vis_done         = new_graph.add_node("N_VIS_DONE",          node_type="flag")
        n_q_pre_manip      = new_graph.add_node("N_Q_PRE_MANIP",      node_type="navigation")
        n_manip_done       = new_graph.add_node("N_MANIPULATION_DONE", node_type="flag")

        # Main path: current -> observation point
        new_graph.add_hyperarc(
            parent=n_q_observation,
            children=[n_current],
            action="NAVIGATE_TO_OBSERVATION_POINT",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )

        # Observe (YOLO placeholder) -> observation done
        new_graph.add_hyperarc(
            parent=n_observation_done,
            children=[n_q_observation],
            action="YOLO_OBSERVE_ACTION",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )

        # Visibility branch: observation -> vis point -> vis done
        new_graph.add_hyperarc(
            parent=n_vis_point,
            children=[n_q_observation],
            action="NAVIGATE_TO_VIS_POINT",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )
        new_graph.add_hyperarc(
            parent=n_vis_done,
            children=[n_vis_point],
            action="VISIBILITY_ACTION",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )

        # AND-gate: need OBS_DONE & VIS_DONE → go to PRE_MANIP
        new_graph.add_hyperarc(
            parent=n_q_pre_manip,
            children=[n_observation_done, n_vis_done],
            action="NAVIGATE_TO_PRE_MANIP_POINT",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )

        # Do manipulation → manipulation done
        new_graph.add_hyperarc(
            parent=n_manip_done,
            children=[n_q_pre_manip],
            action="MANIPULATE_OBSTACLE",
            action_params={"obstacle": obstacle},
            condition=lambda: True
        )

        # Final navigation (optional) → N_FINAL
        if include_final:
            n_final = new_graph.add_node("N_FINAL", node_type="final")
            new_graph.add_hyperarc(
                parent=n_final,
                children=[n_manip_done],
                action="NAVIGATE_TO_FINAL_GOAL",
                action_params={"q_goal": q_goal},
                condition=lambda: True
            )

        # Initialize arc statuses based on children (existing utility)
        self._update_initial_arc_statuses(new_graph)
        return new_graph



class GNSModule:
    """GNS Module - Handles graph execution and feedback"""
    def __init__(self, node):
        self.execution_history = []
        self.node = node

    def find_best_executable_arc(self, graph):
        """Find the best executable hyperarc in the graph"""
        self.node.get_logger().info("--- GNS: Searching for READY arcs ---")
        
        # Debug: Print all arcs and their statuses
        self.node.get_logger().info(f"DEBUG: Graph has {len(graph.hyperarcs)} total arcs:")
        for i, arc in enumerate(graph.hyperarcs):
            children_names = [child.name for child in arc.children]
            children_statuses = [child.status for child in arc.children]
            self.node.get_logger().info(f"  Arc {i}: {arc.action} - Status: {arc.status}")
            self.node.get_logger().info(f"    Children: {children_names} - Statuses: {children_statuses}")
        
        executable_arcs = [arc for arc in graph.hyperarcs if arc.status == "READY"]
        
        if not executable_arcs:
            self.node.get_logger().warn("GNS: No READY arcs found.")
            return None
        
        # Sabse kam cost waala arc chuno
        best_arc = min(executable_arcs, key=lambda arc: arc.cost)
        self.node.get_logger().info(f"GNS: Found and selected READY arc: '{best_arc.action}'")
        return best_arc
    

    def update_arc_success(self, graph, arc):
        """Update graph state after successful execution"""
        arc.status = "SUCCEEDED"
        arc.parent.status = "ACHIEVED"
        self.update_dependent_arcs(graph, arc.parent)

    def update_dependent_arcs(self, graph, achieved_node):
        """Update arcs that depend on the achieved node"""
        # For each hyperarc in the graph
        for arc in graph.hyperarcs:
            # If the arc has the achieved node as a child
            if achieved_node in arc.children:
                # Check if all children are now achieved
                all_children_achieved = True
                for child in arc.children:
                    if child.status != "ACHIEVED":
                        all_children_achieved = False
                        break
                
                # If all children achieved, mark arc as ready
                if all_children_achieved:
                    arc.status = "READY"


class MotionPlanner:
    """Motion Planning implementation - Interface to Nav2"""
    def __init__(self, node, callback_group):
        print("DEBUG: MotionPlanner initialized")
        self.node = node
        self.nav_client = ActionClient(node, NavigateToPose, 'navigate_to_pose', callback_group=callback_group)
        self.current_execution = None 
        self.obstacle_dirs: dict[str, str] = {} #For storing obstacle directions
        self.obstacle_manip_mode: dict[str, str] = {} # For storing obstacle manipulation modes
        self.obstacle_lstar = {}

        self.side_peek_client = node.create_client(
            ComputeSidePeekPoints,
            "compute_side_peek_points"
        )

        self.pre_manip_client = node.create_client(
            ComputePreManipPose,
            "compute_pre_manip_pose",
            callback_group=callback_group,
        )
        
        # Pull trigger action client (new server)
        self.pull_client = ActionClient(
            node,
            PullTrigger,
            'pull_trigger',   # IMPORTANT: yahi name tum server side pe bhi rakhna
            callback_group=callback_group,
        )

        # --- PULL geometry clients (NEW) ---
        self.pull_side_peek_client = node.create_client(
            ComputeSidePeekPoints,
            "compute_pull_side_peek_points"
        )
        self.pull_pre_manip_client = node.create_client(
            ComputePreManipPose,
            "compute_pull_pre_manip_pose",
            callback_group=callback_group,
        )

        while not self.pull_side_peek_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("[Planner] Waiting for 'compute_pull_side_peek_points' service...")

        while not self.pull_pre_manip_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("[Planner] Waiting for 'compute_pull_pre_manip_pose' service...")



        self.observe_client = ActionClient(
            node,
            ObserveObstacle,
            'observe_obstacle',
            callback_group=callback_group,
        )
        self.visibility_client = ActionClient(
            node,
            CheckVisibility,
            'check_visibility',
            callback_group=callback_group,
        )
        
        self.manip_client = ActionClient(
            node,
            ManipulateObstacle,
            'manipulate_obstacle',   # jo bhi tumne server pe naam rakha hai
            callback_group=callback_group,
        )


        while not self.side_peek_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("[Planner] Waiting for 'compute_side_peek_points' service...")

        while not self.pre_manip_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("[Planner] Waiting for 'compute_pre_manip_pose' service...")
        
        # Hardcoded movable obstacles (for thesis simplicity)
        self.movable_obstacles = {

            "unit_box": {
                "x": 2.15, "y": 1.05, "length": 3.28, "width": 0.96,
                "marker": {
                    "x": 2.15, "y": 1.05, "z": 0.27,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            },

            "unit_box_0": {
                "x": 4.80, "y": -0.21, "length": 1.0, "width": 3.12,
                "marker": {
                    "x": 4.80, "y": -0.21, "z": 0.33,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            },

            "unit_box_1": {
                "x": 4.79, "y": -3.13, "length": 0.66, "width": 1.21,
                "marker": {
                    "x": 4.79, "y": -3.13, "z": 0.30,
                    "qx": 0.0, "qy": 0.0, "qz": 1.0, "qw": 0.0
                }
            }
        }

            # Add observation offsets
        self.obs_forward_offset = 3.0  # meters in front of obstacle
        self.obs_lateral_offset = 0.0  # centered laterally
        self.obs_safety_margin = 0.35  # safety margin
        self.obs_min_distance = 0.75  # minimum distance

    def compute_observation_pose(self, obstacle_name):
        """Compute an observation pose in front of the obstacle marker (using marker yaw).

        Convention:
        - Marker orientation defines obstacle frame in map.
        - "Forward" is marker +X axis projected onto ground (map XY).
        - Observation goal is: marker_xy + forward*d + left*lateral
        - Robot yaw faces the marker (look-at).
        """
        import math
        from geometry_msgs.msg import PoseStamped

        log = self.node.get_logger()

        if obstacle_name not in self.movable_obstacles:
            log.error(f"[OBS] Unknown obstacle: {obstacle_name}")
            return None

        ob = self.movable_obstacles[obstacle_name]
        marker = ob.get("marker", None)
        if not marker:
            log.error(f"[OBS] No marker pose found for obstacle '{obstacle_name}'")
            return None

        try:
            mx = float(marker["x"])
            my = float(marker["y"])
            qx = float(marker.get("qx", 0.0))
            qy = float(marker.get("qy", 0.0))
            qz = float(marker.get("qz", 0.0))
            qw = float(marker.get("qw", 1.0))
        except Exception as e:
            log.error(f"[OBS] Bad marker data for '{obstacle_name}': {e}")
            return None

        # Normalize quaternion (robustness)
        qnorm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if qnorm < 1e-9:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx /= qnorm
            qy /= qnorm
            qz /= qnorm
            qw /= qnorm

        # Forward direction = marker +X axis in map (projected to XY)
        # Rotation matrix first column (R00, R10) corresponds to body +X in world XY:
        fx = 1.0 - 2.0 * (qy*qy + qz*qz)          # R00
        fy = 2.0 * (qx*qy + qz*qw)                # R10

        # Normalize XY; fallback if degenerate
        fn = math.hypot(fx, fy)
        if fn < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx /= fn
            fy /= fn

        # Left direction = marker +Y axis in map (projected to XY) for lateral offset
        # Rotation matrix second column (R01, R11) corresponds to body +Y:
        lx = 2.0 * (qx*qy - qz*qw)                # R01
        ly = 1.0 - 2.0 * (qx*qx + qz*qz)          # R11
        ln = math.hypot(lx, ly)
        if ln < 1e-6:
            # If degenerate, just use perpendicular to forward
            lx, ly = -fy, fx
        else:
            lx /= ln
            ly /= ln

        # Distance selection
        fixed_d = float(getattr(self, "obs_forward_offset", 0.0) or 0.0)
        fixed_lat = float(getattr(self, "obs_lateral_offset", 0.0) or 0.0)

        safety = float(getattr(self, "obs_safety_margin", 0.35) or 0.35)
        cammin = float(getattr(self, "obs_min_distance", 0.75) or 0.75)
        length = float(ob.get("length", 0.5) or 0.5)

        d_auto = 0.5 * length + safety + cammin
        d = fixed_d if fixed_d > 0.0 else d_auto
        lateral = fixed_lat  # can be 0

        # Observation position: in front (+forward) and optionally to the left (+lateral)
        x_obs = mx + d * fx + lateral * lx
        y_obs = my + d * fy + lateral * ly

        # Face the marker (look-at): yaw from obs -> marker
        yaw_obs = math.atan2(my - y_obs, mx - x_obs)

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x_obs
        pose.pose.position.y = y_obs
        pose.pose.position.z = 0.0

        # yaw -> quaternion
        half = 0.5 * yaw_obs
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(half)
        pose.pose.orientation.w = math.cos(half)

        log.info(
            f"[OBS] compute_observation_pose('{obstacle_name}') "
            f"marker=({mx:.2f},{my:.2f}) forward=({fx:.3f},{fy:.3f}) "
            f"d={d:.2f} lat={lateral:.2f} -> obs=({x_obs:.2f},{y_obs:.2f}) yaw={yaw_obs:.2f}rad"
        )
        return pose

            

    

    def _nav_to_pose(self, pose_stamped, timeout_sec=30.0, max_attempts=1, strict_timeout=False):
        """
        Send a NavigateToPose goal to Nav2 and wait for result.

        - pose_stamped : PoseStamped (MAP frame)
        - strict_timeout=True  -> hard timeout (no dynamic)
        - strict_timeout=False -> dynamic timeout allowed
        """
        import math
        import rclpy
        from nav2_msgs.action import NavigateToPose
        from action_msgs.msg import GoalStatus

        log = self.node.get_logger()

        for attempt in range(1, max_attempts + 1):

            # --------- 0) timeout estimate ----------
            min_timeout = float(timeout_sec) if timeout_sec is not None else 10.0
            k_factor = 3.0
            nominal_speed = 0.30  # m/s

            # current robot pose
            try:
                rx, ry, _ = self.node.planner.q_current
            except Exception:
                rx, ry = 0.0, 0.0

            gx = float(pose_stamped.pose.position.x)
            gy = float(pose_stamped.pose.position.y)

            straight_dist = math.hypot(gx - rx, gy - ry)
            path_length_est = max(straight_dist, 0.5)

            if nominal_speed > 1e-6:
                dyn_timeout = k_factor * (path_length_est / nominal_speed)
            else:
                dyn_timeout = min_timeout

            # 🔐 strict vs dynamic
            if strict_timeout:
                timeout_effective = min_timeout
                log.info(
                    f"[NAV] (attempt {attempt}/{max_attempts}) "
                    f"STRICT timeout enabled -> {timeout_effective:.1f}s"
                )
            else:
                timeout_effective = max(min_timeout, dyn_timeout)
                log.info(
                    f"[NAV] (attempt {attempt}/{max_attempts}) "
                    f"Computed dynamic timeout for goal "
                    f"(x={gx:.2f}, y={gy:.2f}) -> {timeout_effective:.1f}s "
                    f"(min={min_timeout:.1f}, dist≈{path_length_est:.2f} m)"
                )

            # 1) Wait for action server
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                log.error("[NAV] NavigateToPose action server not available")
                return False

            # 2) Build goal
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = pose_stamped

            log.info(
                f"[NAV] (attempt {attempt}/{max_attempts}) Sending goal to Nav2: "
                f"x={gx:.2f}, y={gy:.2f}"
            )

            # 3) Send goal
            send_goal_future = self.nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=5.0)

            if not send_goal_future.done():
                log.warn("[NAV] Timed out waiting for goal handle")
                return False

            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                log.warn("[NAV] Goal rejected by Nav2")
                return False

            # 4) Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_effective)

            if not result_future.done():
                log.warn("[NAV] Timed out waiting for Nav2 result, canceling goal")
                try:
                    cancel_future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self.node, cancel_future, timeout_sec=2.0)
                except Exception:
                    pass

                if attempt < max_attempts:
                    log.warn("[NAV] Retrying same goal once more...")
                    continue
                else:
                    log.warn("[NAV] Final attempt timed out, giving up")
                    return False

            # 5) Result received
            result = result_future.result()
            status = result.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                log.info("[NAV] NavigateToPose SUCCEEDED ✅")
                return True

            log.warn(f"[NAV] NavigateToPose FAILED with status={status}, not retrying")
            return False

        return False






    def execute_navigation(self, label, params):
        """
        Execute navigation-type hyperarcs.
        Implements:
          - NAVIGATE_TO_OBSERVATION_POINT
          - NAVIGATE_TO_VIS_POINT
          - NAVIGATE_TO_PRE_MANIP_POINT
          - NAVIGATE_TO_FINAL_GOAL
        """
        log = self.node.get_logger()

        try:
            # ------------------------------------------------------------------
            # 1) NAVIGATE_TO_OBSERVATION_POINT
            # ------------------------------------------------------------------
            if label == "NAVIGATE_TO_OBSERVATION_POINT":
                obstacle = params.get("obstacle", None)
                if not obstacle:
                    log.error("[NAV] NAVIGATE_TO_OBSERVATION_POINT: missing 'obstacle' param")
                    return ExecutionResult(success=False,
                                           failure_type="INVALID_PARAMS")

                # 1) Compute observation pose
                pose = self.compute_observation_pose(obstacle)
                if pose is None:
                    log.error(f"[NAV] Failed to compute observation pose for '{obstacle}'")
                    return ExecutionResult(success=False,
                                           failure_type="COMPUTE_OBS_POSE_FAILED",
                                           failure_context={"obstacle": obstacle})

                # 2) Send goal to Nav2
                timeout_sec = float(getattr(self, "obs_nav_timeout_sec", 40.0))
                log.info(f"[NAV] Going to OBS pose for '{obstacle}' (timeout={timeout_sec:.1f}s)")
                ok = self._nav_to_pose(pose, timeout_sec=timeout_sec)

                # 3) Result check
                if ok:
                    log.info(f"[NAV] Reached observation pose for '{obstacle}' ✅")
                    return ExecutionResult(
                        success=True,
                        new_position=[pose.pose.position.x,
                                      pose.pose.position.y,
                                      0.0]
                    )
                else:
                    log.warn(f"[NAV] Failed to reach observation pose for '{obstacle}' ❌")
                    # failure_type, ctx = self._analyze_navigation_failure(
                    #     nav_result=None,
                    #     goal_pose=pose,
                    # )
                    return ExecutionResult(
                        success=False,
                        failure_type=failure_type,
                        failure_context=ctx,
                        blocking_obstacle=ctx.get("blocking_obstacle"),
                    )

            # ------------------------------------------------------------------
            # 2) NAVIGATE_TO_VIS_POINT  (multi-dir, multi-point)
            # ------------------------------------------------------------------

            elif label == "NAVIGATE_TO_VIS_POINT":
                log.info("[NAV] Starting NAVIGATE_TO_VIS_POINT")

                obstacle = params.get("obstacle")
                if not obstacle:
                    log.warn("[NAV_VIS] No obstacle specified in params")
                    return ExecutionResult(success=False, failure_type="NO_OBSTACLE_NAME")

                mode = self.obstacle_manip_mode.get(obstacle, "Push_Movable")
                geom_client = self.side_peek_client if mode == "Push_Movable" else self.pull_side_peek_client
                log.info(f"[NAV_VIS] Using geometry mode={mode} for obstacle={obstacle}")




                obstacle = params.get("obstacle")
                if obstacle is None:
                    log.warn("[NAV_VIS] No obstacle specified in params")
                    return ExecutionResult(success=False, failure_type="NO_OBSTACLE_NAME")
                
                if obstacle not in self.movable_obstacles:
                    log.warn(f"[NAV_VIS] Obstacle '{obstacle}' not found in movable_obstacles")
                    return ExecutionResult(success=False, failure_type="UNKNOWN_OBSTACLE")
                
                ob = self.movable_obstacles[obstacle]
                box_x = float(ob["x"])
                box_y = float(ob["y"])
                box_l = float(ob["length"])
                box_w = float(ob["width"])

                # ---- call push geometry service ----
                req = ComputeSidePeekPoints.Request()
                req.obstacle_name = obstacle
                req.box_x = box_x
                req.box_y = box_y
                req.box_l = box_l
                req.box_w = box_w

                log.info("[NAV_VIS] Requesting side-peek points from push node...")
                future = geom_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future)
                res = future.result()

                if (not res) or (not res.success) or res.n == 0:
                    log.warn("[NAV_VIS] Push node could not compute side-peek points")
                    return ExecutionResult(success=False, failure_type="NO_VIS_POINTS")

                log.info(
                    f"[NAV_VIS] Received {res.n} visibility directions from push node "
                    f"(dir_order already applied on push side)."
                )

                rx, ry, _ = self.node.planner.q_current

                def dist2(p):
                    return (p[0] - rx) ** 2 + (p[1] - ry) ** 2

                best_pose = None
                best_dir = None
                best_lstar = None
                last_goal_pose = None

                # ---- outer loop: directions ----
                for i in range(res.n):
                    dtag = res.dirs[i]
                    l = (res.left_x[i],  res.left_y[i],  res.left_yaw[i])
                    r = (res.right_x[i], res.right_y[i], res.right_yaw[i])
                    Ls = float(res.l_star[i])

                    log.info(
                        f"[NAV_VIS] Candidate dir[{i}]={dtag} | "
                        f"L*={Ls:.2f} | LEFT={l[0]:.2f},{l[1]:.2f} | RIGHT={r[0]:.2f},{r[1]:.2f}"
                    )

                    # per-dir: pehle nearest, fir doosra
                    if dist2(l) <= dist2(r):
                        candidates = [("left", l), ("right", r)]
                    else:
                        candidates = [("right", r), ("left", l)]

                    dir_success = False

                    # ---- inner loop: points in this direction (2 max) ----
                    for side_name, (cx, cy, cyaw) in candidates:
                        log.info(
                            f"[NAV_VIS] Trying dir={dtag}, side={side_name} "
                            f"with side-peek ({cx:.2f}, {cy:.2f}, {math.degrees(cyaw):.1f}°)"
                        )
                        vis_pose = self._create_pose_stamped([cx, cy, cyaw])
                        last_goal_pose = vis_pose

                        nav_ok = self._nav_to_pose(vis_pose)

                        if nav_ok:
                            log.info(
                                f"[NAV_VIS] Successfully reached vis-point for dir={dtag}, side={side_name} ✅"
                            )
                            best_pose = (cx, cy, cyaw)
                            best_dir = dtag
                            best_lstar = Ls
                            dir_success = True
                            break  # is direction ke liye aur point try nahi karna

                        else:
                            log.warn(
                                f"[NAV_VIS] Navigation to vis-point FAILED for "
                                f"dir={dtag}, side={side_name}; trying other side (if any)."
                            )

                    if dir_success:
                        # direction mil gayi → outer loop se bhi nikal jao
                        break

                    log.warn(
                        f"[NAV_VIS] Direction {dtag} fully failed (both left & right). "
                        "Trying next direction (if any)."
                    )

                # ---- all directions failed ----
                if best_pose is None or best_dir is None:
                    log.warn("[NAV_VIS] All candidate visibility points FAILED.")
                    failure_type = "NAV_VIS_ALL_FAILED"
                    ctx = {}
                    if last_goal_pose is not None:
                        failure_type, ctx = self._analyze_navigation_failure(
                            nav_result=None,
                            goal_pose=last_goal_pose,
                        )

                    return ExecutionResult(
                        success=False,
                        failure_type=failure_type,
                        failure_context=ctx,
                        blocking_obstacle=ctx.get("blocking_obstacle"),
                    )



                self.obstacle_dirs[obstacle] = best_dir
                self.obstacle_lstar[obstacle] = float(best_lstar)

                log.info(f"[NAV_VIS] Stored vis_dir={best_dir} for obstacle={obstacle}")
                log.info(f"[NAV_VIS] Stored L*={best_lstar:.2f} for obstacle={obstacle}")
                log.info(
                    f"[NAV_VIS] Final selected vis-point = "
                    f"({best_pose[0]:.2f}, {best_pose[1]:.2f}, {math.degrees(best_pose[2]):.1f}°)"
                )

                return ExecutionResult(
                    success=True,
                    new_position=list(best_pose)
                )


            # ------------------------------------------------------------------
            # 3) NAVIGATE_TO_PRE_MANIP_POINT
            # ------------------------------------------------------------------
            elif label == "NAVIGATE_TO_PRE_MANIP_POINT":
                log.info("[NAV] Starting NAVIGATE_TO_PRE_MANIP_POINT")

                obstacle = params.get("obstacle")
                if not obstacle:
                    log.warn("[NAV_PRE] No obstacle specified in params")
                    return ExecutionResult(success=False, failure_type="NO_OBSTACLE_NAME")
                

                mode = self.obstacle_manip_mode.get(obstacle, "Push_Movable")

                # ✅ PULL CASE: pre-manip navigation skip
                if mode == "Pull_Movable":
                    log.info(f"[PRE_MANIP][PULL] Skipping NAVIGATE_TO_PRE_MANIP_POINT for obstacle={obstacle} (directly go to MANIPULATE_OBSTACLE)")
                    return ExecutionResult(success=True)

                pre_client = self.pre_manip_client if mode == "Push_Movable" else self.pull_pre_manip_client
                log.info(f"[PRE_MANIP] Using geometry mode={mode} for obstacle={obstacle}")


                log.info(f"[PRE_MANIP] Using geometry mode={mode} for obstacle={obstacle}")


                # 1. Obstacle from params
                obstacle = params.get("obstacle")
                if obstacle is None:
                    log.warning("[PRE_MANIP] No obstacle specified in params")
                    return ExecutionResult(success=False, failure_type="NO_OBSTACLE_NAME")

                # 2. Obstacle info
                if obstacle not in self.movable_obstacles:
                    log.warning(f"[PRE_MANIP] Obstacle '{obstacle}' not found in movable_obstacles")
                    return ExecutionResult(success=False, failure_type="UNKNOWN_OBSTACLE")

                ob = self.movable_obstacles[obstacle]
                box_x = float(ob["x"])
                box_y = float(ob["y"])
                box_l = float(ob["length"])
                box_w = float(ob["width"])

                # 3. Direction (vis_dir) reuse from previous step
                vis_dir = self.obstacle_dirs.get(obstacle)
                if vis_dir is None:
                    log.warning(f"[PRE_MANIP] No stored vis_dir for obstacle={obstacle}")
                    return ExecutionResult(success=False, failure_type="NO_VIS_DIR")

                log.info(f"[PRE_MANIP] Using vis_dir={vis_dir} for obstacle={obstacle}")

                # 4. Call push node to get pre-manip pose
                req = ComputePreManipPose.Request()
                req.obstacle_name = obstacle
                req.box_x = box_x
                req.box_y = box_y
                req.box_l = box_l
                req.box_w = box_w
                req.dir   = vis_dir

                log.info("[PRE_MANIP] Requesting pre-manip pose from push node...")
                future = pre_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future)
                res = future.result()

                if (not res) or (not res.success):
                    log.warning("[PRE_MANIP] Push node could not compute pre-manip pose")
                    return ExecutionResult(success=False, failure_type="NO_PRE_MANIP_POSE")

                px, py, pyaw = res.pre_x, res.pre_y, res.pre_yaw
                log.info(f"[PRE_MANIP] Target pre-manip pose = ({px:.2f}, {py:.2f}, {math.degrees(pyaw):.1f}deg)")

                # 5. Navigate using planner's own Nav2 wrapper
                pre_pose = self._create_pose_stamped([px, py, pyaw])
                nav_ok = self._nav_to_pose(pre_pose)
                if not nav_ok:
                    log.warning("[PRE_MANIP] Navigation to pre-manip pose failed")
                    failure_type, ctx = self._analyze_navigation_failure(
                        nav_result=None,
                        goal_pose=pre_pose,
                    )
                    return ExecutionResult(
                        success=False,
                        failure_type=failure_type,
                        failure_context=ctx,
                        blocking_obstacle=ctx.get("blocking_obstacle"),
                    )

                log.info("[PRE_MANIP] Successfully reached pre-manip pose!")
                return ExecutionResult(
                    success=True,
                    new_position=[px, py, pyaw]
                )

            # ------------------------------------------------------------------
            # 4) NAVIGATE_TO_FINAL_GOAL
            # ------------------------------------------------------------------
            elif label == "NAVIGATE_TO_FINAL_GOAL":
                log.info("[NAV] Starting NAVIGATE_TO_FINAL_GOAL")

                # Option 1: node-based goals (initial graph)
                node_obj = params.get("node")

                if node_obj is not None:
                    try:
                        goal_x, goal_y, goal_yaw = node_obj.position
                        log.info(f"[NAV_FINAL] Using node.position = ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.1f}°)")
                    except Exception:
                        log.error("[NAV_FINAL] node.position missing or invalid")
                        return ExecutionResult(success=False, failure_type="INVALID_FINAL_NODE")
                
                else:
                    # Option 2: direct q_goal (manipulation graph)
                    q_goal = params.get("q_goal")
                    if not q_goal or len(q_goal) < 3:
                        log.error("[NAV_FINAL] No node or valid q_goal provided")
                        return ExecutionResult(success=False, failure_type="NO_FINAL_NODE")

                    goal_x, goal_y, goal_yaw = q_goal
                    log.info(f"[NAV_FINAL] Using q_goal = ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.1f}°)")

                # Build pose
                final_pose = self._create_pose_stamped([goal_x, goal_y, goal_yaw])
                # FINAL GOAL: strict navigation
                nav_ok = self._nav_to_pose(
                    final_pose,
                    timeout_sec=30.0,   # ✅ fixed 30 seconds
                    max_attempts=1,      # ✅ sirf ek hi attempt
                    strict_timeout=True
                )


                if not nav_ok:
                    log.warning("[NAV_FINAL] Navigation to final goal failed")
                    failure_type, ctx = self._analyze_navigation_failure(
                        nav_result=None,
                        goal_pose=final_pose,
                    )
                    return ExecutionResult(
                        success=False,
                        failure_type=failure_type,
                        failure_context=ctx,
                        blocking_obstacle=ctx.get("blocking_obstacle"),
                    )

                log.info("[NAV_FINAL] Successfully reached final goal!")
                return ExecutionResult(
                    success=True,
                    new_position=[goal_x, goal_y, goal_yaw]
                )


            # ------------------------------------------------------------------
            # Unknown label
            # ------------------------------------------------------------------
            else:
                log.error(f"[NAV] Unknown navigation label: {label}")
                return ExecutionResult(success=False, failure_type="UNKNOWN_LABEL",
                                       failure_context={"label": label})

        except Exception as e:
            log.error(f"[NAV] Exception in execute_navigation({label}): {e}")
            return ExecutionResult(success=False,
                                   failure_type="EXCEPTION",
                                   failure_context={"err": str(e), "label": label})


    
         
    def execute_observation(self, label, params):
        """
        Base structure for all observation-related actions.
        Handles:
        - YOLO_OBSERVE_ACTION
        - VISIBILITY_ACTION
        Currently: Only logs + dummy success result.
        """
        log = self.node.get_logger()
        # Extract params (if passed)
        obstacle = params.get("obstacle")
        node = self.node
        log.info(
            f"[OBS] execute_observation | label={label} | obstacle={obstacle}"
            )


        # Log entry
        log.info(
            f"[OBS] execute_observation called | label={label} | "
            f"obstacle={obstacle} | node={node} "
        )

        try:
            # -------------------------------
            # CASE 1: YOLO OBSERVE ACTION
            # -------------------------------
            if label == "YOLO_OBSERVE_ACTION":
                # 1) Basic checks
                if not obstacle:
                    log.error("[OBS][YOLO] Missing 'obstacle' in params.")
                    return ExecutionResult(
                        success=False,
                        failure_type="INVALID_PARAMS",
                        failure_context="YOLO_OBSERVE_ACTION requires 'obstacle' in params",
                    )
                log.info(
                    f"[OBS][YOLO] Starting YOLO observation for obstacle={obstacle}"
                )
                # 2) Wait for observation action server
                if not self.observe_client.wait_for_server(timeout_sec=5.0):
                    log.error("[OBS][YOLO] observe_obstacle action server not available.")
                    return ExecutionResult(
                        success=False,
                        failure_type="OBS_ACTION_UNAVAILABLE",
                        failure_context="observe_obstacle action server not available",
                    )
                
                # 3) Build goal
                goal = ObserveObstacle.Goal()
                goal.obstacle_name = obstacle
                
                 # 4) Send goal
                try:
                    send_goal_future = self.observe_client.send_goal_async(goal)
                    rclpy.spin_until_future_complete(self.node, send_goal_future)
                    goal_handle = send_goal_future.result()
                except Exception as exc:
                    log.error(f"[OBS][YOLO] Exception while sending goal: {exc!r}")
                    return ExecutionResult(
                        success=False,
                        failure_type="OBS_GOAL_SEND_ERROR",
                        failure_context=str(exc),
                    )
                
                if goal_handle is None or not goal_handle.accepted:
                    log.error("[OBS][YOLO] ObserveObstacle goal was rejected.")
                    return ExecutionResult(
                        success=False,
                        failure_type="OBS_GOAL_REJECTED",
                    )

                log.info("[OBS][YOLO] ObserveObstacle goal accepted. Waiting for result...")


                # 5) Wait for result
                try:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(node, result_future)
                    action_result = result_future.result().result
                except Exception as exc:
                    log.error(f"[OBS][YOLO] Exception while waiting for result: {exc!r}")
                    return ExecutionResult(
                        success=False,
                        failure_type="OBS_RESULT_ERROR",
                        failure_context=str(exc),
                    )
                
                # 6) Map action result → ExecutionResult
                if action_result.success:
                    mode = (action_result.label or "").strip()
                    low = mode.lower()
                    if low in ("pullable_movable", "pull_movable", "pull_movable", "pullable", "pull_movable"):
                        mode = "Pull_Movable"
                    elif low in ("push_movable", "pushable_movable", "pushable"):
                        mode = "Push_Movable"
                    log.info(
                        f"[OBS][YOLO] Observation succeeded | "
                        f"label='{mode}' | msg='{action_result.message}'"
                    )
                    if mode not in ("Push_Movable", "Pull_Movable"):
                        log.error(f"[OBS][YOLO] Unknown manipulation mode '{mode}' for obstacle={obstacle}")
                        return ExecutionResult(
                            success=False,
                            failure_type="UNKNOWN_MANIP_MODE",
                            failure_context={"obstacle": obstacle, "label": mode},
                        )
                    self.obstacle_manip_mode[obstacle] = mode
                    log.info(
                        f"[OBS][YOLO] Stored manip mode '{mode}' for obstacle={obstacle}"
                    )
                    return ExecutionResult(success=True)
                else:
                    log.error(
                        f"[OBS][YOLO] Observation failed for obstacle={obstacle} | "
                        f"label='{action_result.label}' | msg='{action_result.message}'"
                    )
                    return ExecutionResult(
                        success=False,
                        failure_type="OBSERVATION_FAILED",
                        failure_context={"obstacle": obstacle, "message": action_result.message},
                    )


            # -------------------------------
            # CASE 2: VISIBILITY ACTION
            # -------------------------------
            elif label == "VISIBILITY_ACTION":
                if not obstacle:
                    log.error("[OBS][VIS] Missing 'obstacle' in params.")
                    return ExecutionResult(
                        success=False,
                        failure_type="INVALID_PARAMS",
                        failure_context="VISIBILITY_ACTION requires 'obstacle' in params",
                    )

                # Direction: prefer explicit vis_dir in params, otherwise use stored from NAVIGATE_TO_VIS_POINT
                vis_dir_final = self.obstacle_dirs.get(obstacle)
                if not vis_dir_final:
                    log.error(
                        f"[OBS][VIS] No visibility direction for obstacle={obstacle} "
                        "(run NAVIGATE_TO_VIS_POINT first?)"
                    )
                    return ExecutionResult(
                        success=False,
                        failure_type="NO_VIS_DIR",
                        failure_context={"obstacle": obstacle},
                    )

                log.info(
                    f"[OBS][VIS] Starting visibility check | "
                    f"obstacle={obstacle} | vis_dir={vis_dir_final}"
                )

                # Wait for visibility action server
                if not self.visibility_client.wait_for_server(timeout_sec=5.0):
                    log.error("[OBS][VIS] check_visibility action server not available.")
                    return ExecutionResult(
                        success=False,
                        failure_type="VIS_ACTION_UNAVAILABLE",
                    )

                # Build goal: ONLY name + dir + duration
                goal = CheckVisibility.Goal()
                goal.obstacle_name = obstacle
                goal.vis_dir = vis_dir_final
                goal.duration_sec = float(getattr(self, "vis_duration_sec", 100.0))

                # Send goal
                try:
                    send_goal_future = self.visibility_client.send_goal_async(goal)
                    rclpy.spin_until_future_complete(self.node, send_goal_future)
                    goal_handle = send_goal_future.result()
                except Exception as exc:
                    log.error(f"[OBS][VIS] Exception while sending goal: {exc!r}")
                    return ExecutionResult(
                        success=False,
                        failure_type="VIS_GOAL_SEND_ERROR",
                        failure_context=str(exc),
                    )

                if goal_handle is None or not goal_handle.accepted:
                    log.error("[OBS][VIS] CheckVisibility goal was rejected.")
                    return ExecutionResult(
                        success=False,
                        failure_type="VIS_GOAL_REJECTED",
                        failure_context={"obstacle": obstacle},
                    )

                log.info("[OBS][VIS] Goal accepted. Waiting for result...")

                # Wait for result
                try:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(self.node, result_future)
                    action_result = result_future.result().result
                except Exception as exc:
                    log.error(f"[OBS][VIS] Exception while waiting for result: {exc!r}")
                    return ExecutionResult(
                        success=False,
                        failure_type="VIS_RESULT_ERROR",
                        failure_context=str(exc),
                    )

                blocked = bool(getattr(action_result, "obstacle_present", True))
                resolved_name = (getattr(action_result, "obstacle_name", "") or "").strip()

                if blocked:
                    log.warning(f"[OBS][VIS] Visibility check: OBSTACLE PRESENT for {obstacle}")

                    # ✅ IMPORTANT: blocked => planner ko manipulate-issue do
                    return ExecutionResult(
                        success=False,
                        failure_type="MANIPULATION_ISSUE",
                        blocking_obstacle=None,  # abhi unknown, resolver pick karega
                        failure_context={
                            "reason": "OBSTACLE_IN_LSTAR_REGION",
                            "hint_obstacle": resolved_name,  # ✅ THIS IS THE KEY (hint)
                        },
                    )

                log.info(f"[OBS][VIS] Visibility check: area clear for {obstacle}")
                return ExecutionResult(success=True)



            # -------------------------------
            # UNKNOWN LABEL (safety)
            # -------------------------------
            else:
                log.error(f"[OBS] Unknown observation label received: {label}")
                return ExecutionResult(
                    success=False,
                    failure_type="UNKNOWN_OBSERVATION_LABEL"
                )

        except Exception as exc:
            log.error(f"[OBS] Exception in execute_observation: {exc!r}")
            return ExecutionResult(
                success=False,
                failure_type="OBSERVATION_ERROR",
                failure_context=str(exc)
            )

    def execute_manipulation(self, label, params):
        """
        MANIPULATE_OBSTACLE ke liye high-level dispatcher.
        YOLO se stored mode (Push_Movable / pullable_movable) ke basis pe
        alag pipeline select karega.
        """
        log = self.node.get_logger()
        obstacle = params.get("obstacle")

        if label != "MANIPULATE_OBSTACLE":
            log.error(f"[MANIP] Unknown manipulation label: {label}")
            return ExecutionResult(
                success=False,
                failure_type="UNKNOWN_MANIP_LABEL",
                failure_context={"label": label},
            )

        if not obstacle:
            log.error("[MANIP] MANIPULATE_OBSTACLE: missing 'obstacle' param")
            return ExecutionResult(
                success=False,
                failure_type="INVALID_PARAMS",
                failure_context="MANIPULATE_OBSTACLE requires 'obstacle' in params",
            )

        # 1) Check obstacle known hai ya nahi
        if obstacle not in self.movable_obstacles:
            log.error(f"[MANIP] Unknown obstacle '{obstacle}' (not in movable_obstacles)")
            return ExecutionResult(
                success=False,
                failure_type="UNKNOWN_OBSTACLE",
                failure_context={"obstacle": obstacle},
            )

        # 2) YOLO se aaya hua mode padho
        mode = self.obstacle_manip_mode.get(obstacle)
        if mode is None:
            log.error(
                f"[MANIP] No manipulation mode stored for obstacle={obstacle} "
                f"(YOLO_OBSERVE_ACTION shayad run nahi hua?)"
            )
            return ExecutionResult(
                success=False,
                failure_type="NO_MANIP_MODE",
                failure_context={"obstacle": obstacle},
            )

        log.info(f"[MANIP] Manipulating obstacle={obstacle} with mode='{mode}'")

        # 3) Branch: pushable vs pullable
        if mode == "Push_Movable":
            if mode == "Push_Movable":
            # 3.1 direction (vis_dir) pehle NAVIGATE_TO_VIS_POINT me store hui thi
                push_dir = self.obstacle_dirs.get(obstacle)
                if push_dir is None:
                    log.error(
                        f"[MANIP] No stored push_dir / vis_dir for obstacle={obstacle} "
                        f"(NAVIGATE_TO_VIS_POINT shayad run nahi hua?)"
                    )
                    return ExecutionResult(
                        success=False,
                        failure_type="NO_PUSH_DIR",
                        failure_context={"obstacle": obstacle},
                    )

            log.info(f"[MANIP][PUSH] Using push_dir={push_dir} for obstacle={obstacle}")

            # 3.2 wait for action server
            if not self.manip_client.wait_for_server(timeout_sec=5.0):
                log.error("[MANIP][PUSH] manipulate_obstacle action server not available")
                return ExecutionResult(
                    success=False,
                    failure_type="MANIP_ACTION_UNAVAILABLE",
                )

            # 3.3 goal banao
            goal = ManipulateObstacle.Goal()
            goal.obstacle_name = obstacle
            goal.push_dir = push_dir
            goal.push_dist_m = self.obstacle_lstar.get(obstacle, 1.0)

            # 3.4 goal send karo
            try:
                send_goal_future = self.manip_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self.node, send_goal_future)
                goal_handle = send_goal_future.result()
            except Exception as exc:
                log.error(f"[MANIP][PUSH] Exception while sending goal: {exc!r}")
                return ExecutionResult(
                    success=False,
                    failure_type="MANIP_GOAL_SEND_ERROR",
                    failure_context=str(exc),
                )

            if goal_handle is None or not goal_handle.accepted:
                log.error("[MANIP][PUSH] ManipulateObstacle goal was rejected.")
                return ExecutionResult(
                    success=False,
                    failure_type="MANIP_GOAL_REJECTED",
                    failure_context={"obstacle": obstacle},
                )

            log.info("[MANIP][PUSH] Goal accepted. Waiting for result...")

            # 3.5 result ka wait
            try:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self.node, result_future)
                action_result = result_future.result().result
            except Exception as exc:
                log.error(f"[MANIP][PUSH] Exception while waiting for result: {exc!r}")
                return ExecutionResult(
                    success=False,
                    failure_type="MANIP_RESULT_ERROR",
                    failure_context=str(exc),
                )

            # 3.6 action result → ExecutionResult
            if getattr(action_result, "success", False):
                msg = getattr(action_result, "message", "")
                log.info(f"[MANIP][PUSH] Manipulation succeeded | msg='{msg}'")
                return ExecutionResult(success=True)
            else:
                msg = getattr(action_result, "message", "")
                log.error(f"[MANIP][PUSH] Manipulation failed | msg='{msg}'")
                return ExecutionResult(
                    success=False,
                    failure_type="PUSH_FAILED",
                    failure_context={"obstacle": obstacle, "message": msg},
                )

        elif mode == "Pull_Movable":
            log.info(f"[MANIP][PULL] Running PULL strategy for obstacle={obstacle}")

            # 1) wait for pull action server
            if not self.pull_client.wait_for_server(timeout_sec=5.0):
                log.error("[MANIP][PULL] pull_trigger action server not available")
                return ExecutionResult(
                    success=False,
                    failure_type="PULL_ACTION_UNAVAILABLE",
                    failure_context={"obstacle": obstacle},
                )

            # 2) build goal (boolean trigger only)
            goal = PullTrigger.Goal()
            goal.trigger = True

            # 3) send goal
            try:
                send_goal_future = self.pull_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self.node, send_goal_future)
                goal_handle = send_goal_future.result()
            except Exception as exc:
                log.error(f"[MANIP][PULL] Exception while sending goal: {exc!r}")
                return ExecutionResult(
                    success=False,
                    failure_type="PULL_GOAL_SEND_ERROR",
                    failure_context=str(exc),
                )

            if goal_handle is None or not goal_handle.accepted:
                log.error("[MANIP][PULL] PullTrigger goal was rejected.")
                return ExecutionResult(
                    success=False,
                    failure_type="PULL_GOAL_REJECTED",
                    failure_context={"obstacle": obstacle},
                )

            log.info("[MANIP][PULL] Goal accepted. Waiting for result...")

            # 4) wait result
            try:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self.node, result_future)
                action_result = result_future.result().result
            except Exception as exc:
                log.error(f"[MANIP][PULL] Exception while waiting for result: {exc!r}")
                return ExecutionResult(
                    success=False,
                    failure_type="PULL_RESULT_ERROR",
                    failure_context=str(exc),
                )

            # 5) map action result -> ExecutionResult
            ok = bool(getattr(action_result, "success", False))
            if ok:
                log.info("[MANIP][PULL] Pull succeeded ✅")
                return ExecutionResult(success=True)
            else:
                log.error("[MANIP][PULL] Pull failed ❌")
                return ExecutionResult(
                    success=False,
                    failure_type="PULL_FAILED",
                    failure_context={"obstacle": obstacle},
                )


    def _point_to_segment_distance(self, px, py, ax, ay, bx, by) -> float:
        """
        Distance from point P(px,py) to segment AB(ax,ay)->(bx,by).
        """
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            # A and B are same point
            return math.hypot(px - ax, py - ay)

        t = (apx * abx + apy * aby) / ab2  # projection factor
        if t < 0.0:
            cx, cy = ax, ay
        elif t > 1.0:
            cx, cy = bx, by
        else:
            cx = ax + t * abx
            cy = ay + t * aby

        return math.hypot(px - cx, py - cy)    


    def _nearest_movable_obstacle(
        self,
        goal_pose,
        max_robot_dist: float = 3.0,
        corridor_half_width: float = 1.2,
        path_stride: int = 1,
    ):
        """
        Path-based 'first blocking along path' obstacle picker.

        Criteria:
        1) Obstacle robot ke paas ho (dist_robot <= max_robot_dist)
        2) Obstacle Nav2 ke actual planned path corridor me ho
        3) Path ke order me sabse pehle jo obstacle milta hai -> wahi blocker

        Returns:
            obstacle_name (str) or None
        """

        log = self.node.get_logger()

        # ---------------- robot pose ----------------
        planner = getattr(self.node, "planner", None)
        if planner is None or not hasattr(planner, "q_current"):
            log.warn("[NAV_FAIL] planner/q_current not available; cannot infer blocking obstacle")
            return None

        rx, ry, _ = planner.q_current
        log.info(f"[NAV_FAIL][DBG] q_current(rx,ry)=({rx:.3f},{ry:.3f})")

        # ---------------- nav2 path ----------------
        nav_path = getattr(self.node, "nav_path", None) or []
        if len(nav_path) < 2:
            log.warn("[NAV_FAIL] nav_path empty/too short; cannot do path-based blocking check")
            return None

        stride = max(1, int(path_stride))
        p0x, p0y = nav_path[0]
        pmx, pmy = nav_path[len(nav_path) // 2]
        plx, ply = nav_path[-1]
        log.info(
            "[NAV_FAIL][DBG] nav_path sample: "
            f"p0=({p0x:.3f},{p0y:.3f}) mid=({pmx:.3f},{pmy:.3f}) last=({plx:.3f},{ply:.3f}) "
            f"len={len(nav_path)} stride={stride}"
        )

        # quick sanity: robot -> path sampled min distance
        try:
            dmin_robot_path = min(
                math.hypot(px - rx, py - ry) for (px, py) in nav_path[::stride]
            )
            log.info(f"[NAV_FAIL][DBG] min_dist(robot->path_sampled)={dmin_robot_path:.3f}m")
        except Exception as e:
            log.warn(f"[NAV_FAIL][DBG] could not compute min_dist(robot->path): {e}")

        # ---------------- movable obstacles ----------------
        movable = getattr(self, "movable_obstacles", None) or {}
        if not movable:
            log.warn("[NAV_FAIL] No movable_obstacles available")
            return None

        # Debug each obstacle distances (robot + path min)
        for name, ob in movable.items():
            try:
                ox, oy = float(ob["x"]), float(ob["y"])
            except Exception:
                log.warn(f"[NAV_FAIL][DBG] obs='{name}' invalid x/y; raw={ob}")
                continue

            d_robot = math.hypot(ox - rx, oy - ry)
            try:
                d_path_min = min(math.hypot(px - ox, py - oy) for (px, py) in nav_path[::stride])
            except Exception:
                d_path_min = float("nan")

            log.info(
                f"[NAV_FAIL][DBG] obs='{name}' pos=({ox:.3f},{oy:.3f}) "
                f"d_robot={d_robot:.3f} d_path_min(sampled)={d_path_min:.3f} "
                f"thr_robot={max_robot_dist:.2f} thr_corr={corridor_half_width:.2f}"
            )

        # Find closest waypoint index to robot -> start checking from there
        def d2(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        i_r = min(range(len(nav_path)), key=lambda i: d2(nav_path[i], (rx, ry)))
        cpx, cpy = nav_path[i_r]
        d_closest = math.hypot(cpx - rx, cpy - ry)

        log.info(
            f"[NAV_FAIL] Path-based blocking search | "
            f"start_idx={i_r}/{len(nav_path)} | "
            f"closest_wp=({cpx:.3f},{cpy:.3f}) dist={d_closest:.3f} | "
            f"max_robot_dist={max_robot_dist:.2f} | corridor={corridor_half_width:.2f} | stride={stride}"
        )

        # Throttled scan debug
        dbg_every_n = 30  # log every ~30 scanned points (after stride applied)

        scanned = 0
        near_hits = 0
        corridor_hits = 0

        # Iterate path forward from robot's closest point
        for i in range(i_r, len(nav_path) - 1, stride):
            ax, ay = nav_path[i]
            bx, by = nav_path[i + 1]

            scanned += 1

            do_dbg = (scanned % dbg_every_n == 0)

            if do_dbg:
                log.info(f"[NAV_FAIL][DBG] scanning path_idx={i} wp=({ax:.3f},{ay:.3f})")

            # Check each obstacle for corridor + near-robot condition
            for name, ob in movable.items():
                try:
                    ox = float(ob["x"])
                    oy = float(ob["y"])
                except Exception:
                    # already logged above, keep quiet here
                    continue

                dist_robot = math.hypot(ox - rx, oy - ry)
                if dist_robot > max_robot_dist:
                    continue

                near_hits += 1

                dist_path = self._point_to_segment_distance(ox, oy, ax, ay, bx, by)
                if dist_path <= corridor_half_width:
                    corridor_hits += 1
                    log.info(
                        f"[NAV_FAIL] First blocking obstacle found: '{name}' "
                        f"at path_idx={i} | dist_robot={dist_robot:.2f} | dist_path={dist_path:.2f}"
                    )
                    return name

                if do_dbg:
                    # show near-robot but not corridor
                    log.info(
                        f"[NAV_FAIL][DBG] near-but-not-corridor obs='{name}' "
                        f"dist_robot={dist_robot:.2f} dist_path={dist_path:.2f} "
                        f"(corr_thr={corridor_half_width:.2f})"
                    )

        log.warn(
            f"[NAV_FAIL] No blocking obstacle found along Nav2 path corridor | "
            f"scanned_pts={scanned} near_hits={near_hits} corridor_hits={corridor_hits}"
        )
        return None


    def _analyze_navigation_failure(self, nav_result, goal_pose):
        """
        Nav failure ko do category me todta hai:

        - MANIPULATION_ISSUE:
                path-based search se koi blocking movable obstacle mila
        - NAVIGATION_ISSUE:
                path clear (no blocking obstacle), phir bhi Nav2 fail/timeout

        Taaki:
        * obstacle-issue ho to manipulation pipeline chale
        * pure Nav2 issue ho to usko navigation failure treat kiya ja sake
        """
        log = self.node.get_logger()

        # ---- path-based blocking search ----
        blocking_obstacle = None
        try:
            blocking_obstacle = self._nearest_movable_obstacle(goal_pose)
        except Exception as e:
            log.warn(f"[NAV_FAIL] _nearest_movable_obstacle failed: {e!r}")
            blocking_obstacle = None

        # ---- classify failure ----
        if blocking_obstacle:
            failure_type = "MANIPULATION_ISSUE"
            reason = "BLOCKING_OBSTACLE_ON_PATH"
            log.warn(f"[NAV_FAIL] Blocking obstacle detected: '{blocking_obstacle}'")
        else:
            failure_type = "NAVIGATION_ISSUE"
            reason = "NAV2_FAILURE_NO_BLOCKING_OBSTACLE"
            log.warn("[NAV_FAIL] No blocking obstacle found along Nav2 path corridor")

        # nav_result se status (agar mila ho)
        nav_status = None
        try:
            if nav_result is not None:
                nav_status = getattr(nav_result, "status", None)
        except Exception:
            nav_status = None

        failure_context = {
            "reason": reason,
            "blocking_obstacle": blocking_obstacle,
            "nav_status": nav_status,
            "goal": {
                "x": getattr(goal_pose.pose.position, "x", None),
                "y": getattr(goal_pose.pose.position, "y", None),
                "z": getattr(goal_pose.pose.position, "z", None),
            },
        }

        return failure_type, failure_context



    def _create_pose_stamped(self, position):
        """Create PoseStamped message from position"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(position[0])
        pose_stamped.pose.position.y = float(position[1])
        pose_stamped.pose.position.z = 0.0
        
        # Convert theta to quaternion
        theta = float(position[2]) if len(position) > 2 else 0.0
        q = self._yaw_to_quaternion(theta)
        pose_stamped.pose.orientation = q
        return pose_stamped

    def _yaw_to_quaternion(self, yaw):
        """Convert yaw angle (theta) to quaternion (geometry_msgs/Quaternion)"""
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q



class VANAMOPlanner:
    """Main VANAMO planning algorithm"""
    def __init__(self, node, callback_group):
        self.node = node
        self.q_current = [0.0, 0.0, 0.0]  # Current robot pose
        #self.q_goal = [2.13, -4.77, 0.0]   # Goal pose
        self.q_goal = [2.88, 8.23, 0.0]   # Goal pose
        self.graph_id_counter = 0
        self.visited_states = set()
        self.graph_network = []
        self.current_graph_index = 0
        self.position_threshold = 0.5

        # --------- Recovery Mode ----------
        self.arc_retry_counts = {}
        self.max_arc_retries = 15

        # --------- META GRAPH (recursive view) ----------
        self.meta_dot = Digraph(comment="Recursive Plan Graph (Meta)")
        self.meta_dot.attr(rankdir="LR")  # left-to-right

        self.plan_uid_counter = 0
        self.current_plan_uid = None  # which plan cluster we are in

        # child_plan_uid -> info (parent_uid, reason, obstacle, resume_arc)
        self.plan_links = []

        # output file (single growing diagram)
        self.meta_graph_filename = "plan_graph_recursive"



        #lstar
        self.obstacle_lstar: dict[str, float] = {}

        
        # Thread safety for planning steps
        self.planning_lock = threading.Lock()
        self.is_planning_active = False
        
        # Initialize modules
        self.aog_module = AOGModule()
        self.motion_planner = MotionPlanner(node, callback_group)
        self.gns_module = GNSModule(node)
        
        # Planning state
        self.planning_state = PlanningState.INITIALIZING
        self.current_executing_arc = None

        # Parent arcs active flags
        self.parent2_active = False
        self.plan_stack = []

        # Detect obstacles in order
        self.obstacle_order = ["unit_box", "unit_box_0", "unit_box_1"]
        self.current_obstacle_index = 0


    
    

    def _meta_add_plan_cluster(self, aog_graph, plan_uid: int, title: str):
        """
        Add one plan (AOGGraph) as a Graphviz cluster into self.meta_dot.
        IMPORTANT: node IDs are prefixed with P{plan_uid}_ to avoid collisions.
        """
        cluster_name = f"cluster_P{plan_uid}"
        prefix = f"P{plan_uid}_"

        with self.meta_dot.subgraph(name=cluster_name) as c:
            c.attr(label=f"{title} (P{plan_uid})", style="rounded")
            c.attr(color="gray")

            # Nodes
            for name, node in aog_graph.nodes.items():
                node_id = prefix + name
                c.node(node_id, f"{name}\n({node.type})", shape="circle")

            # Hyperarcs as edges (child -> parent like your existing viz)
            for arc in aog_graph.hyperarcs:
                parent_id = prefix + arc.parent.name
                for child in arc.children:
                    child_id = prefix + child.name
                    c.edge(child_id, parent_id, label=arc.action)

        # A "plan anchor" node for connecting parent->child cleanly
        anchor_id = f"PLAN_P{plan_uid}"
        self.meta_dot.node(anchor_id, f"PLAN P{plan_uid}", shape="box")
        # connect anchor to this plan's final node (optional, but helps)
        if "N_FINAL" in aog_graph.nodes:
            self.meta_dot.edge(anchor_id, prefix + "N_FINAL", style="dashed")

    def _meta_link_plans(self, parent_uid: int, child_uid: int, label: str):
        """Connect parent plan to child plan in meta graph."""
        self.meta_dot.edge(f"PLAN_P{parent_uid}", f"PLAN_P{child_uid}", label=label)

    def _meta_render(self):
        """Render ONE single PNG that grows over time."""
        try:
            self.meta_dot.render(self.meta_graph_filename, view=True, format="png")
            self.node.get_logger().info(f"[META_GRAPH] Rendered {self.meta_graph_filename}.png")
        except Exception as e:
            self.node.get_logger().error(f"[META_GRAPH] Render failed: {e}")

        
    def set_goal(self, goal_pose):
        """Set new goal and reset planning"""
        self.q_goal = goal_pose
        self.planning_state = PlanningState.INITIALIZING
        self.graph_network.clear()
        self.current_graph_index = 0
        self.visited_states.clear()

    def planning_step(self):
        """Single planning step - called by timer"""
        # Thread safety: Only one planning step should run at a time
        with self.planning_lock:
            if self.is_planning_active:
                #self.node.get_logger().info("Planning step already in progress, skipping")
                return "SKIP"
            self.is_planning_active = True
        
        # Thread debugging info
        active_threads = threading.active_count()
        current_thread = threading.current_thread().name
        all_threads = [t.name for t in threading.enumerate()]
        
        self.node.get_logger().info(f"DEBUG: planning_step() called - State: {self.planning_state.value}")
        self.node.get_logger().info(f"DEBUG: THREAD INFO - Active: {active_threads}, Current: {current_thread}, All: {all_threads}")
        

        try:
            if self.planning_state == PlanningState.INITIALIZING:
                self.node.get_logger().info("DEBUG: Calling _initialize_planning()")
                result = self._initialize_planning()
            elif self.planning_state == PlanningState.PLANNING:
                self.node.get_logger().info("DEBUG: Calling _execute_planning_step()")
                result = self._execute_planning_step()
            elif self.planning_state == PlanningState.GOAL_REACHED:
                self.node.get_logger().info("DEBUG: Goal reached, returning SUCCESS")
                result = "SUCCESS"
            elif self.planning_state == PlanningState.FAILED:
                self.node.get_logger().info("[RECOVERY] FAILED state - resuming failed arc")
                
                if self.current_executing_arc:
                    self.current_executing_arc.status = "READY"
                
                self.planning_state = PlanningState.PLANNING
                result = "CONTINUE"
            else:
                self.node.get_logger().warn(f"DEBUG: Unknown planning state: {self.planning_state}")
                result = "FAILURE"
                
        except Exception as e:
            self.node.get_logger().error(f"Planning step failed: {e}")
            self.planning_state = PlanningState.FAILED
            result = "FAILURE"
            
        finally:
            with self.planning_lock:
                self.is_planning_active = False
            #self.node.get_logger().info(f"DEBUG: is_planning_active set to {self.is_planning_active} in finally block")
            
                
        return result
    
    def _initialize_planning(self):
        """Initialize planning with first graph"""
        # Create initial AND/OR graph
        root_graph = self.aog_module.create_initial_graph()
        n_final = root_graph.get_node("N_FINAL")
        n_final.position = self.q_goal
        self.graph_network.append(root_graph)
        self.current_graph_index = 0

        # Add this line for visualization
        # ---- META graph: add ROOT plan cluster ----
        self.plan_uid_counter += 1
        self.current_plan_uid = self.plan_uid_counter  # root uid

        self._meta_add_plan_cluster(root_graph, self.current_plan_uid, title="ROOT PLAN")
        self._meta_render()


        self.planning_state = PlanningState.PLANNING
        self.node.get_logger().info("VANAMO planning initialized")
        return "CONTINUE"

    def _execute_planning_step(self):
        """Execute one planning step with DFS + resume semantics"""

        self.node.get_logger().info(
            f"[STEP] Active graph={self.current_graph_index}, "
            f"stack_depth={len(self.plan_stack)}, "
            f"parent2_active={self.parent2_active}"
        )

        # 0) Goal reached
        if self.is_goal_reached():
            self.node.get_logger().info("[STEP] Global goal reached")
            self.planning_state = PlanningState.GOAL_REACHED
            return "SUCCESS"
        

        # 1) DFS unwind: child plan finished
        if self.task_completed() and self.plan_stack:
            ctx = self.plan_stack.pop()

            self.node.get_logger().info(
                f"[DFS-UNWIND] Child plan finished. "
                f"Returning to graph={ctx['graph_index']} | "
                f"Retry arc={ctx['resume_arc'].action}"
            )

            self.current_graph_index = ctx["graph_index"]
            self.q_current = ctx["resume_pose"]
            self.current_plan_uid = ctx.get("plan_uid", self.current_plan_uid)
            ctx["resume_arc"].status = "READY"

            self.node.get_logger().info(
                f"[DFS-UNWIND] Pose restored={self.q_current}, arc set READY"
            )

            return self._execute_planning_step()

        # 2) Normal planning
        current_graph = self.graph_network[self.current_graph_index]

        best_arc = self.gns_module.find_best_executable_arc(current_graph)

        if not best_arc:
            self.node.get_logger().warn(
                f"[GNS] No executable arcs in graph {self.current_graph_index}"
            )
            if not self._expand_current_graph(current_graph):
                self.node.get_logger().error("[GNS] Graph expansion failed")
                self.planning_state = PlanningState.FAILED
                return "FAILURE"
            return "CONTINUE"

        if best_arc.status == "EXECUTING":
            self.node.get_logger().info(
                f"[SKIP] Arc already EXECUTING: {best_arc.action}"
            )
            return "CONTINUE"

        # 3) Execute arc
        self.node.get_logger().info(
            f"[EXEC] Executing arc={best_arc.action} "
            f"on graph={self.current_graph_index}"
        )

        best_arc.status = "EXECUTING"
        self.current_executing_arc = best_arc

        execution_result = self._execute_hyperarc(best_arc)

        # 4) SUCCESS
        if execution_result.success:
            self.node.get_logger().info(
                f"[SUCCESS] Arc succeeded: {best_arc.action}"
            )

            best_arc.status = "COMPLETED"
            self.gns_module.update_arc_success(current_graph, best_arc)
            self._update_environment(execution_result)
            self.current_executing_arc = None

            if self.is_goal_reached():
                self.node.get_logger().info(
                    "[SUCCESS] Goal reached after arc execution"
                )
                self.planning_state = PlanningState.GOAL_REACHED
                return "SUCCESS"

            return self._execute_planning_step()

        # 5) FAILURE
        self.node.get_logger().warn(
            f"[FAIL] Arc failed: {best_arc.action} | "
            f"type={execution_result.failure_type}"
        )


        if execution_result.failure_type == "MANIPULATION_ISSUE":
            best_arc.status = "READY"

            # ROOT → Parent Plan 2
            if self.current_graph_index == 0 and not self.parent2_active:
                self.node.get_logger().warn(
                    "[ROOT->PARENT2] Root plan blocked. "
                    "Creating Parent Plan 2"
                )

                self.plan_stack.append({
                    "graph_index": 0,
                    "resume_arc": best_arc,
                    "resume_pose": list(self.q_current),
                    "plan_uid": self.current_plan_uid,
                })

                self.parent2_active = True

                self.node.get_logger().info(
                    f"[STACK] Pushed root context | depth={len(self.plan_stack)}"
                )

                self._handle_manipulation_issue(
                    execution_result,
                    include_final=True
                )
                return self._execute_planning_step()

            # DFS child creation
            self.node.get_logger().warn(
                "[DFS] Parent Plan 2 blocked. Creating child manipulation plan"
            )

            self.plan_stack.append({
                "graph_index": self.current_graph_index,
                "resume_arc": best_arc,
                "resume_pose": list(self.q_current),
                "plan_uid": self.current_plan_uid,
            })

            self.node.get_logger().info(
                f"[STACK] Child pushed | depth={len(self.plan_stack)}"
            )

            self._handle_manipulation_issue(
                execution_result,
                include_final=False
            )
            return self._execute_planning_step()

        # Non-manipulation failure

        arc_id = f"{best_arc.action}_{best_arc.parent.name}"
        
        # Initialize or increment retry count
        if arc_id not in self.arc_retry_counts:
            self.arc_retry_counts[arc_id] = 0
        self.arc_retry_counts[arc_id] += 1
        
        retry_count = self.arc_retry_counts[arc_id]
        
        self.node.get_logger().warn(
            f"[RECOVERY] Non-manipulation failure | "
            f"Arc: {best_arc.action} | "
            f"Retry: {retry_count}/{self.max_arc_retries}"
        )
        
        # Check if we should retry
        if retry_count < self.max_arc_retries:
            # RECOVERY: Arc ko READY mark karo
            best_arc.status = "READY"
            self.current_executing_arc = None
            
            self.node.get_logger().info(
                f"[RECOVERY] Arc '{best_arc.action}' marked READY for retry"
            )
            return self._execute_planning_step() 
        else:
            # Max retries exceeded - actually fail
            self.node.get_logger().error(
                f"[FATAL] Max retries ({self.max_arc_retries}) exceeded for "
                f"arc '{best_arc.action}'. Planning FAILED."
            )
            best_arc.status = "FAILED"
            self.planning_state = PlanningState.FAILED
            self.current_executing_arc = None
            return "FAILURE"

    def _execute_hyperarc(self, arc):
            """
            Dispatch a hyperarc to the right executor based on its action label.
            Returns an ExecutionResult(success: bool, ...).

            Handled labels:
            - NAVIGATE_TO_OBSERVATION_POINT
            - YOLO_OBSERVE_ACTION
            - NAVIGATE_TO_VIS_POINT
            - VISIBILITY_ACTION
            - NAVIGATE_TO_PRE_MANIP_POINT
            - MANIPULATE_OBSTACLE
            - NAVIGATE_TO_FINAL_GOAL
            """
            label  = getattr(arc, "action", None)
            params = getattr(arc, "action_params", None) or {}

            log = self.node.get_logger()

            if not label:
                log.error("[AOG] Missing action label on hyperarc")
                return ExecutionResult(
                    success=False,
                    failure_type="MISSING_LABEL",
                    failure_context={"err": "missing_label"},
                )

            NAVIGATION = {
                "NAVIGATE_TO_OBSERVATION_POINT",
                "NAVIGATE_TO_VIS_POINT",
                "NAVIGATE_TO_PRE_MANIP_POINT",
                "NAVIGATE_TO_FINAL_GOAL",
            }
            OBSERVATION = {
                "YOLO_OBSERVE_ACTION",
                "VISIBILITY_ACTION",
            }
            MANIPULATION = {
                "MANIPULATE_OBSTACLE",
            }

            log.info(f"[AOG] Action start: {label} | params={params}")
            try:
                # 👉 Motion-related actions go through MotionPlanner
                if label in NAVIGATION:
                    result = self.motion_planner.execute_navigation(label, params)
                elif label in OBSERVATION:
                    result = self.motion_planner.execute_observation(label, params)
                elif label in MANIPULATION:
                    result = self.motion_planner.execute_manipulation(label, params)
                else:
                    log.error(f"[AOG] Unknown action '{label}'")
                    return ExecutionResult(
                        success=False,
                        failure_type="UNKNOWN_ACTION",
                        failure_context={"err": "unknown_action", "action": label},
                    )

                ok = bool(getattr(result, "success", False))
                log.info(f"[AOG] Action end  : {label} | success={ok}")
                return result

            except Exception as e:
                log.error(f"[AOG] Action error: {label} | {e}")
                return ExecutionResult(
                    success=False,
                    failure_type="ACTION_EXCEPTION",
                    failure_context={"err": str(e), "action": label},
                )

    
    def _update_environment(self, result):
        """Update environment based on execution results"""
        if result.new_position:  
            self.q_current = result.new_position
            
        if result.map_updates:
            self.node.get_logger().info(f"Map updated with {len(result.map_updates)} changes")
    
    def calculate_viewpoint(self, blocked_region, current_pose, sensor_range=5.0):
        """Calculate best viewpoint to observe blocked region"""
        if not blocked_region:
            return None
        
        # Simple viewpoint calculation - position robot to see blocked region
        # In real implementation, consider obstacles, sensor constraints, etc.
        
        target_x, target_y = blocked_region[0], blocked_region[1]
        current_x, current_y = current_pose[0], current_pose[1]
        
        # Calculate direction from current to blocked region
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return None
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Position viewpoint at sensor_range distance from blocked region
        viewpoint_x = target_x - dx * (sensor_range * 0.8)
        viewpoint_y = target_y - dy * (sensor_range * 0.8)
        
        return [viewpoint_x, viewpoint_y, 0.0]  # x, y, theta
    
    def _handle_manipulation_issue(self, result, include_final: bool):
        if result.new_position:
            self.q_current = result.new_position

        # ============================================================
        # DETERMINISTIC: Use fixed obstacle order
        # ============================================================
        if self.current_obstacle_index >= len(self.obstacle_order):
            self.node.get_logger().error("All obstacles exhausted! No more obstacles to handle.")
            self.planning_state = PlanningState.FAILED
            return False

        obstacle = self.obstacle_order[self.current_obstacle_index]
        self.current_obstacle_index += 1

        self.node.get_logger().info(
            f"[DETERMINISTIC] Using obstacle '{obstacle}' "
            f"(index {self.current_obstacle_index}/{len(self.obstacle_order)})"
        )
        # ============================================================

        manip_graph = self.aog_module.expand_graph_for_manipulation(
            obstacle,
            self.q_goal,
            include_final=include_final
        )
        self.graph_network.append(manip_graph)
        self.current_graph_index = len(self.graph_network) - 1

        # ---- META graph: add CHILD/PARENT2 plan cluster + link ----
        parent_uid = self.current_plan_uid  # jis plan se spawn hua

        self.plan_uid_counter += 1
        child_uid = self.plan_uid_counter
        self.current_plan_uid = child_uid   # now we are in this new plan

        # Title decide (optional)
        plan_title = "PARENT PLAN 2" if include_final else "CHILD PLAN"
        self._meta_add_plan_cluster(manip_graph, child_uid, title=plan_title)

        # label: obstacle + reason (best effort)
        ctx = getattr(result, "failure_context", {}) or {}
        reason = ctx.get("reason", result.failure_type)
        self._meta_link_plans(parent_uid, child_uid, label=f"{reason} | obstacle={obstacle}")

        self._meta_render()
        return True
    


    def _handle_manipulation_issue(self, result, include_final: bool):
        if result.new_position:
            self.q_current = result.new_position

        # ============================================================
        # DETERMINISTIC: Use fixed obstacle order
        # ============================================================
        if self.current_obstacle_index >= len(self.obstacle_order):
            self.node.get_logger().error("All obstacles exhausted! No more obstacles to handle.")
            self.planning_state = PlanningState.FAILED
            return False

        obstacle = self.obstacle_order[self.current_obstacle_index]
        self.current_obstacle_index += 1

        self.node.get_logger().info(
            f"[DETERMINISTIC] Using obstacle '{obstacle}' "
            f"(index {self.current_obstacle_index}/{len(self.obstacle_order)})"
        )
        # ============================================================

        manip_graph = self.aog_module.expand_graph_for_manipulation(
            obstacle,
            self.q_goal,
            include_final=include_final
        )
        self.graph_network.append(manip_graph)
        self.current_graph_index = len(self.graph_network) - 1

        # ---- META graph: add CHILD/PARENT2 plan cluster + link ----
        parent_uid = self.current_plan_uid  # jis plan se spawn hua

        self.plan_uid_counter += 1
        child_uid = self.plan_uid_counter
        self.current_plan_uid = child_uid   # now we are in this new plan

        # Title decide (optional)
        plan_title = "PARENT PLAN 2" if include_final else "CHILD PLAN"
        self._meta_add_plan_cluster(manip_graph, child_uid, title=plan_title)

        # label: obstacle + reason (best effort)
        ctx = getattr(result, "failure_context", {}) or {}
        reason = ctx.get("reason", result.failure_type)
        self._meta_link_plans(parent_uid, child_uid, label=f"{reason} | obstacle={obstacle}")

        self._meta_render()
        return True
        

    
    
    def _expand_current_graph(self, graph):
        """Try to expand current graph or switch to another"""
        # For now, just fail if no executable arcs
        # In future, could try other expansion strategies
        return False
    
    def is_goal_reached(self):
        """Global goal reached ONLY when Parent Plan with N_FINAL completes"""
        if not self.graph_network:
            return False
        
        current_graph = self.graph_network[self.current_graph_index]
        final_node = current_graph.get_node("N_FINAL")
        
        # No N_FINAL = Child Plan = NOT global goal
        if final_node is None:
            return False
        
        return final_node.status == "ACHIEVED"
    
    def at_goal_position(self):
        """Check if robot is at goal position"""
        distance = math.sqrt(
            (self.q_current[0] - self.q_goal[0])**2 + 
            (self.q_current[1] - self.q_goal[1])**2
        )
        return distance < self.position_threshold
    
    def task_completed(self) -> bool:
        """Current plan (child or parent) completed?"""
        if not self.graph_network:
            return False
        
        current_graph = self.graph_network[self.current_graph_index]
        
        # Check N_FINAL first (Parent Plans)
        final_node = current_graph.get_node("N_FINAL")
        if final_node:
            return final_node.status == "ACHIEVED"
        
        # No N_FINAL = Child Plan, check N_MANIPULATION_DONE
        manip_done = current_graph.get_node("N_MANIPULATION_DONE")
        return bool(manip_done and manip_done.status == "ACHIEVED")

    
    def _get_state_key(self, position, target):
        """Create unique key for state to detect cycles"""
        return f"{position}_{target}"


class VANAMOPlannerNode(Node):
    """ROS2 Node for VANAMO Planner"""
    
    def __init__(self):
        super().__init__('vanamo_planner')

        self.manipulate_box_client = ActionClient(self, ManipulateObstacle, 'manipulate_box')



        self.nav_path = []  # Global path waypoints
        self.path_sub = self.create_subscription(
        Path,
        '/plan',
        self.path_callback,
        10
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        
        #publisher for manipulation issues
        self.manip_issue_pub = self.create_publisher(String, 'vanamo/manipulation_issue', qos)

        # Create a ReentrantCallbackGroup
        self.callback_group = ReentrantCallbackGroup()
        


        # Declare parameters
        self.declare_parameter('planning_frequency', 1.0)
        self.declare_parameter('position_threshold', 0.5)
        self.latest_occupancy_grid = None
        
        # Initialize planner
        self.planner = VANAMOPlanner(self, self.callback_group)
        
        # Set up planning timer
        planning_freq = self.get_parameter('planning_frequency').get_parameter_value().double_value
        self.planning_timer = self.create_timer(1.0/planning_freq, self.planning_callback, callback_group=self.callback_group)
        
        # Publishers for monitoring and visualization
        self.status_pub = self.create_publisher(String, 'vanamo/status', qos)
        self.graph_info_pub = self.create_publisher(String, 'vanamo/graph_info', qos)
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos
        )
        
        
        # Subscribers for goal and robot state
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'vanamo/goal',
            self.goal_callback,
            10
        )
        
        # self.pose_sub = self.create_subscription(
        #     PoseStamped,
        #     'robot_pose',
        #     self.pose_callback,
        #     10
        # )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10
        )

        self.get_logger().info("VANAMO Planner Node initialized")




    def path_callback(self, msg):
        self.nav_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f"[DEBUG] Received nav_path with {len(self.nav_path)} waypoints")


    def map_callback(self, msg):
        """Store the latest occupancy grid from map topic"""
        self.latest_occupancy_grid = msg
        self.get_logger().debug("Received new occupancy grid map")
        
    def planning_callback(self):
        """Main planning loop callback"""
        try:
            # Thread debugging info
            active_threads = threading.active_count()
            current_thread = threading.current_thread().name
            all_threads = [t.name for t in threading.enumerate()]
            #self.get_logger().info(f"CALLBACK DEBUG: Active threads: {active_threads}, Current: {current_thread}")
            #self.get_logger().info(f"CALLBACK DEBUG: All threads: {all_threads}")
            
            result = self.planner.planning_step()
            
            if result in ["SUCCESS", "FAILURE"]:     
                self.get_logger().info(f"Planning completed with result: {result}")
            
            # Publish status
            status_msg = String()
            status_msg.data = f"State: {self.planner.planning_state.value}, Result: {result}" 
            self.status_pub.publish(status_msg)
            
            # Publish graph info
            if self.planner.graph_network:
                graph_info = self._get_graph_info()
                graph_msg = String()
                graph_msg.data = graph_info
                self.graph_info_pub.publish(graph_msg) 


            # Timer status check - FOR DEBUGGING
            #self.get_logger().info(f"TIMER DEBUG: Timer cancelled: {self.planning_timer.is_canceled()}")
            #self.get_logger().info("CALLBACK DEBUG: Callback completed successfully")
                
        except Exception as e:
            self.get_logger().error(f"Planning callback failed: {e}")
            # Thread debug info on exception
            active_threads = threading.active_count()
            current_thread = threading.current_thread().name
            all_threads = [t.name for t in threading.enumerate()]
            self.get_logger().error(f"EXCEPTION DEBUG: Active threads: {active_threads}, Current: {current_thread}")
            self.get_logger().error(f"EXCEPTION DEBUG: All threads: {all_threads}")
            # Timer status on exception
            self.get_logger().error(f"TIMER DEBUG: Timer cancelled: {self.planning_timer.is_canceled()}")

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self._quaternion_to_yaw(msg.pose.pose.orientation)
        # Store old position for comparison
        old_pos = self.planner.q_current.copy()
        
        # Update planner
        # self.planner.q_current = [x, y, yaw]


    def goal_callback(self, msg):
        """Handle new goal messages"""
        goal_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            self._quaternion_to_yaw(msg.pose.orientation)
        ]
        
        self.get_logger().info(f"New goal received: {goal_pose}")
        self.planner.set_goal(goal_pose)
    
    # def pose_callback(self, msg):
    #     """Handle robot pose updates"""
    #     new_pose = [
    #         msg.pose.position.x,
    #         msg.pose.position.y,
    #         self._quaternion_to_yaw(msg.pose.orientation)
    #     ]
        
    #     # Update planner with current pose
    #     self.planner.q_current = new_pose
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        AMCL pose is in 'map' frame (usually). This should match Nav2 path frame.
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        x = p.x
        y = p.y
        yaw = self._quaternion_to_yaw(q)

        self.planner.q_current = [x, y, yaw]
        self.get_logger().info(f"[POSE][AMCL] q_current=({x:.3f},{y:.3f})")


    
    def _get_graph_info(self):
        """Get current graph information for monitoring"""
        if not self.planner.graph_network:
            return "No active graphs"
        
        current_graph = self.planner.graph_network[self.planner.current_graph_index]
        
        # Count nodes and arcs by status
        node_status_count = {}
        arc_status_count = {}
        
        for node in current_graph.nodes.values():
            status = node.status
            node_status_count[status] = node_status_count.get(status, 0) + 1
        
        for arc in current_graph.hyperarcs:
            status = arc.status
            arc_status_count[status] = arc_status_count.get(status, 0) + 1
        
        info = f"Graph {self.planner.current_graph_index + 1}/{len(self.planner.graph_network)} | "
        info += f"Nodes: {node_status_count} | "
        info += f"Arcs: {arc_status_count}"
        
        return info
    
    def _quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        
        # Extract yaw from quaternion
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def set_goal_programmatically(self, x, y, theta=0.0):
        """Set goal programmatically for testing"""
        goal_pose = [x, y, theta]
        self.get_logger().info(f"Setting goal programmatically: {goal_pose}")
        self.planner.set_goal(goal_pose)
        
    def get_planner_status(self):
        """Get current planner status for external monitoring"""
        return {
            'state': self.planner.planning_state.value,
            'current_pose': self.planner.q_current,
            'goal_pose': self.planner.q_goal,
            'active_graphs': len(self.planner.graph_network),
            'current_graph': self.planner.current_graph_index,
            'goal_reached': self.planner.is_goal_reached()
        }


def main(args=None):
    """Main function to run the VANAMO planner"""
    rclpy.init(args=args)
    
    # SingleThreadedExecutor banayein for cleaner thread management
    # Ye ek hi thread use karta hai, more predictable behavior
    executor = SingleThreadedExecutor()
    node = VANAMOPlannerNode()
    
    # Node ko executor me add karein
    executor.add_node(node)
    
    try:
        print("DEBUG: Starting SingleThreadedExecutor...")
        # Thread debugging info at startup
        active_threads = threading.active_count()
        current_thread = threading.current_thread().name
        all_threads = [t.name for t in threading.enumerate()]
        print(f"STARTUP DEBUG: Active threads: {active_threads}, Current: {current_thread}")
        print(f"STARTUP DEBUG: All threads: {all_threads}")
        
        # Sirf executor ko spin karein. Ye node ke saare events handle karega.
        executor.spin() #Yahan main thread single thread ko puri trh se control kr deta h. 
    except KeyboardInterrupt:
        print("DEBUG: KeyboardInterrupt received, shutting down.")
    finally:
        # Safai se band karein
        print("DEBUG: Shutting down executor and node.")
        # Final thread debugging info
        active_threads = threading.active_count()
        current_thread = threading.current_thread().name
        all_threads = [t.name for t in threading.enumerate()]
        print(f"SHUTDOWN DEBUG: Active threads: {active_threads}, Current: {current_thread}")
        print(f"SHUTDOWN DEBUG: All threads: {all_threads}")
        
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__': 
    main()


def visualize_graph_structure(graph):
    """Helper function to visualize graph structure for debugging"""
    print("=== Graph Structure ===")
    print("Nodes:")
    for name, node in graph.nodes.items():
        print(f"  {name}: {node.type} - {node.status}")
    
    print("Hyperarcs:")
    for i, arc in enumerate(graph.hyperarcs):
        children_names = [child.name for child in arc.children]
        print(f"  Arc {i}: {arc.parent.name} <- {children_names}")
        print(f"    Action: {arc.action}")
        print(f"    Status: {arc.status}")
        print(f"    Params: {arc.action_params}")


def debug_planner_state(planner):
    """Debug function to print current planner state"""
    print("=== VANAMO Planner State ===")
    print(f"Current Position: {planner.q_current}")
    print(f"Goal Position: {planner.q_goal}")
    print(f"Planning State: {planner.planning_state}")
    print(f"Active Graphs: {len(planner.graph_network)}")
    print(f"Current Graph Index: {planner.current_graph_index}")
    print(f"Goal Reached: {planner.is_goal_reached()}")
    
    if planner.graph_network:
        print(f"\nCurrent Graph Details:")
        current_graph = planner.graph_network[planner.current_graph_index]
        visualize_graph_structure(current_graph)