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
from geometry_msgs.msg import Twist
from graphviz import Digraph
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor   #SINGLE-THREADING for cleaner execution
from rclpy.callback_groups import ReentrantCallbackGroup #Callback group for reentrant callbacks
import threading # Import the threading library
from mir_navigation.action import ObserveObstacle, CheckVisibility, ManipulateObstacle
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


    def expand_graph_for_manipulation(self, obstacle, q_goal):
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
        n_final            = new_graph.add_node("N_FINAL",             node_type="final")

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
        "test_box": {
            "x": 6.5, "y": 0.5, "length": 0.5, "width": 2.0,
            "marker": {  # map frame pose
            "x": 6.25, "y": 0.5, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0  # I have to verify the x,y,z axis of marker.
            }
        },
        "test_box_1": {
            "x": 4.64, "y": 9.15, "length": 0.25, "width": 0.25,
            "marker": {
            "x": 4.60, "y": 9.10, "z": 0.88,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0        ## I have to verify the x,y,z axis of marker.
            }
        }
        }

            # Add observation offsets
        self.obs_forward_offset = 2.0  # meters in front of obstacle
        self.obs_lateral_offset = 0.0  # centered laterally
        
        
    def compute_observation_pose(self, obstacle_name):
        """Compute observation pose for given obstacle"""
        if obstacle_name not in self.movable_obstacles:
            self.node.get_logger().error(f"[OBS] Unknown obstacle: {obstacle_name}")
            return None
            
        ob = self.movable_obstacles[obstacle_name]
        marker = ob.get("marker", {})
        
        # Calculate angle to face the obstacle (from offset position)
        # For now, assume obstacle is at 0 yaw (can be extended)
        yaw = 0.0  # Default yaw
        
        try:
            mx = float(marker["x"]);  my = float(marker["y"]);  # mz = float(marker["z"])  # (unused here)
            qx = float(marker["qx"]); qy = float(marker["qy"])
            qz = float(marker["qz"]); qw = float(marker["qw"])
        except Exception:
            self.node.get_logger().error(f"[OBS] Missing marker pose for obstacle '{obstacle_name}'")
            return None
        
            # 1) Marker +Z (face normal) rotated into map frame, XY projection
    #    R[:,2] (third column) of quaternion rotation matrix:
        fx = 2.0 * (qx * qz + qy * qw)
        fy = 2.0 * (qy * qz - qx * qw)
        # (fz = 1 - 2*(qx*qx + qy*qy))  # not needed; we work on the ground plane

        # normalize XY forward; fallback to +X if degenerate
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx /= norm; fy /= norm
        fx = -fx
        fy = -fy

        # 2) Choose standoff distance d (robust default if fixed offset not set)
        fixed = float(getattr(self, "obs_forward_offset", 0.0) or 0.0)
        safety = float(getattr(self, "obs_safety", 0.25))
        cammin = float(getattr(self, "camera_min_range", 0.35))
        length = float(ob.get("length", 0.5))
        d_auto = 0.5 * length + safety + cammin
        d = fixed if fixed > 0.0 else d_auto

        # 3) Observation position = marker_xy + d * forward(+Z_projected)
        x_obs = mx + d * fx
        y_obs = my + d * fy

        # 4) Face the marker (look-at): yaw from obs -> marker
        yaw_obs = math.atan2(my - y_obs, mx - x_obs)

        # 5) Build pose (planar goal; z = 0 for base)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose.position.x = x_obs
        pose_stamped.pose.position.y = y_obs
        pose_stamped.pose.position.z = 0.0

        q = self._yaw_to_quaternion(yaw_obs)  # your existing helper
        pose_stamped.pose.orientation = q
        return pose_stamped
            

    

    def _nav_to_pose(self, pose_stamped: PoseStamped, timeout_sec: float | None = None) -> bool:
        """
        Send a NavigateToPose goal to Nav2 and wait for result.

        Behaviour:
        - Sirf RESULT TIMEOUT ke case mein same goal ko retry karega (max 2 attempts).
        - Agar Nav2 hard failure status de ya goal reject ho, to retry nahi karega.
        """
        log = self.node.get_logger()

        max_attempts = 2  # timeout par itni baar retry

        for attempt in range(1, max_attempts + 1):
            # --------- 0) dynamic timeout estimate ----------
            min_timeout = float(timeout_sec) if timeout_sec is not None else 10.0
            k_factor = 3.0          # slack factor
            nominal_speed = 0.30    # m/s (assumed nav speed)

            # current robot pose (agar available ho)
            try:
                rx, ry, _ = self.node.planner.q_current
            except Exception:
                rx, ry = 0.0, 0.0

            gx = float(pose_stamped.pose.position.x)
            gy = float(pose_stamped.pose.position.y)

            dx = gx - rx
            dy = gy - ry
            straight_dist = math.hypot(dx, dy)

            path_length_est = max(straight_dist, 0.5)

            if nominal_speed > 1e-6:
                dyn_timeout = k_factor * (path_length_est / nominal_speed)
            else:
                dyn_timeout = min_timeout

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
                f"x={pose_stamped.pose.position.x:.2f}, "
                f"y={pose_stamped.pose.position.y:.2f}"
            )

            # 3) Send goal asynchronously
            send_goal_future = self.nav_client.send_goal_async(goal_msg)

            # 4) Wait for goal handle
            rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=timeout_effective)
            if not send_goal_future.done():
                log.warn(f"[NAV] (attempt {attempt}) Timed out waiting for goal handle")
                # yahan retry ka matlab nahi banta, server hi respond nahi kar raha
                return False

            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                log.warn(f"[NAV] (attempt {attempt}) NavigateToPose goal rejected by server")
                return False  # hard failure, no retry

            # 5) Wait for result with SAME effective timeout
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=timeout_effective)
            if not result_future.done():
                # 👉 ye hai RESULT TIMEOUT case jisme hum retry karna chahte hain
                log.warn(f"[NAV] (attempt {attempt}) Timed out waiting for Nav2 result, canceling goal")
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass

                if attempt < max_attempts:
                    log.warn("[NAV] Timeout occurred, retrying same goal once more...")
                    continue  # next attempt
                else:
                    log.warn("[NAV] Timeout occurred again on last attempt, giving up.")
                    return False

            # yahan aaya matlab Nav2 ne result diya
            result = result_future.result()
            status = result.status

            if status == GoalStatus.STATUS_SUCCEEDED:
                log.info(f"[NAV] NavigateToPose SUCCEEDED ✅ (attempt {attempt})")
                return True
            else:
                # 👉 hard Nav2 failure: ABORTED / CANCELED / UNKNOWN etc.
                log.warn(f"[NAV] NavigateToPose FAILED with status={status} (attempt {attempt}), not retrying.")
                return False

        # theoretically yahan nahi aana chahiye
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
                req.box_x = box_x
                req.box_y = box_y
                req.box_l = box_l
                req.box_w = box_w

                log.info("[NAV_VIS] Requesting side-peek points from push node...")
                future = self.side_peek_client.call_async(req)
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

                # ---- success: store dir + L* for push ----
                bx, by, byaw = best_pose
                vis_dir = best_dir
                self.obstacle_dirs[obstacle] = vis_dir
                self.obstacle_lstar[obstacle] = float(best_lstar)

                log.info(f"[NAV_VIS] Stored vis_dir={vis_dir} for obstacle={obstacle}")
                log.info(f"[NAV_VIS] Stored L*={best_lstar:.2f} for obstacle={obstacle}")
                log.info(
                    f"[NAV_VIS] Final selected vis-point = "
                    f"({bx:.2f}, {by:.2f}, {math.degrees(byaw):.1f}°)"
                )

                return ExecutionResult(
                    success=True,
                    new_position=[bx, by, byaw]
                )


            # ------------------------------------------------------------------
            # 3) NAVIGATE_TO_PRE_MANIP_POINT
            # ------------------------------------------------------------------
            elif label == "NAVIGATE_TO_PRE_MANIP_POINT":
                log.info("[NAV] Starting NAVIGATE_TO_PRE_MANIP_POINT")

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
                req.box_x = box_x
                req.box_y = box_y
                req.box_l = box_l
                req.box_w = box_w
                req.dir   = vis_dir

                log.info("[PRE_MANIP] Requesting pre-manip pose from push node...")
                future = self.pre_manip_client.call_async(req)
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
                nav_ok = self._nav_to_pose(final_pose)

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
                    log.info(
                        f"[OBS][YOLO] Observation succeeded | "
                        f"label='{mode}' | msg='{action_result.message}'"
                    )
                    if mode not in ("Push_Movable", "pullable_movable"):
                        log.error(
                            f"[OBS][YOLO] Unknown manipulation mode '{mode}' "
                            f"for obstacle={obstacle}"
                        )
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
                if blocked:
                    log.warning(f"[OBS][VIS] Visibility check: OBSTACLE PRESENT for {obstacle}")
                else:
                    log.info(f"[OBS][VIS] Visibility check: area clear for {obstacle}")

                # Abhi ke liye: action execution success treat karte hain,
                # 'blocked' ko context me store kar dete hain.
                return ExecutionResult(
                    success=True,
                    failure_context={"obstacle": obstacle, "blocked": blocked},
                )

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
            

        elif mode == "pullable_movable":
            # 👉 Yaha PULL strategy (alag motion, maybe different action)
            log.info(f"[MANIP] Running PULL strategy for obstacle={obstacle}")
        else:
            log.error(f"[MANIP] Unexpected manip mode '{mode}' for obstacle={obstacle}")
            return ExecutionResult(
                success=False,
                failure_type="UNKNOWN_MANIP_MODE",
                failure_context={"obstacle": obstacle, "mode": mode},
            )

    def _navigation_failure_result(self, goal_pose, raw_failure_type, extra_context=None):
        """Invoke analyzer and package a standard manipulation-issue failure."""
        failure_type, failure_context = self._analyze_navigation_failure(None, goal_pose)
        failure_context["raw_failure_type"] = raw_failure_type
        if extra_context:
            failure_context.update(extra_context)

        return ExecutionResult(
            success=False,
            failure_type=failure_type,
            failure_context=failure_context,
        )

    def _nearest_movable_obstacle(
        self,
        goal_pose,
        max_robot_dist: float = 3.0,
        corridor_half_width: float = 0.8,
        path_stride: int = 3,
    ):

        """
        Path-based 'first blocking along path' obstacle picker.

        Criteria:
        1) Obstacle robot ke paas ho (dist_robot <= max_robot_dist)
        2) Obstacle Nav2 ke actual planned path corridor me ho
        3) Path ke order me sabse pehle jo obstacle milta hai -> wahi blocker

        Args:
            goal_pose: PoseStamped of goal (not heavily used here, but kept for API consistency).
            max_robot_dist: Robot se max distance jiske andar obstacle "near robot" maana jayega.
            corridor_half_width: Path ke around corridor half width (meters).
            path_stride: Waypoints skip factor (1 = every point, 2 = every 2nd point, etc.)

        Returns:
            obstacle_name (str) or None
        """
        log = self.node.get_logger()

        # -------------- robot pose --------------
        planner = getattr(self.node, "planner", None)
        if planner is None or not hasattr(planner, "q_current"):
            log.warn("[NAV_FAIL] planner/q_current not available; cannot infer blocking obstacle")
            return None

        rx, ry, _ = planner.q_current

        # -------------- nav2 path --------------
        nav_path = getattr(self.node, "nav_path", None) or []
        if len(nav_path) < 2:
            log.warn("[NAV_FAIL] nav_path empty/too short; cannot do path-based blocking check")
            return None

        # -------------- movable obstacles --------------
        movable = getattr(self, "movable_obstacles", None) or {}
        if not movable:
            log.warn("[NAV_FAIL] No movable_obstacles available")
            return None

        # Find closest waypoint index to robot -> start checking from there
        def d2(a, b):
            return (a[0] - b[0])**2 + (a[1] - b[1])**2

        i_r = min(range(len(nav_path)), key=lambda i: d2(nav_path[i], (rx, ry)))

        log.info(
            f"[NAV_FAIL] Path-based blocking search | "
            f"start_idx={i_r}/{len(nav_path)} | "
            f"max_robot_dist={max_robot_dist:.2f} | corridor={corridor_half_width:.2f}"
        )

        # Iterate path forward from robot's closest point
        for i in range(i_r, len(nav_path), max(1, int(path_stride))):
            px, py = nav_path[i]

            # Check each obstacle for corridor + near-robot condition
            for name, ob in movable.items():
                try:
                    ox = float(ob["x"])
                    oy = float(ob["y"])
                except Exception:
                    log.warn(f"[NAV_FAIL] Obstacle '{name}' has invalid x/y; skipping")
                    continue

                # (1) near robot check
                dist_robot = math.hypot(ox - rx, oy - ry)
                if dist_robot > max_robot_dist:
                    continue

                # (2) corridor check: obstacle close to this path waypoint
                dist_path = math.hypot(ox - px, oy - py)
                if dist_path <= corridor_half_width:
                    log.info(
                        f"[NAV_FAIL] First blocking obstacle found: '{name}' "
                        f"at path_idx={i} | dist_robot={dist_robot:.2f} | dist_path={dist_path:.2f}"
                    )
                    return name

        log.warn("[NAV_FAIL] No blocking obstacle found along Nav2 path corridor")
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
                self.node.get_logger().info("DEBUG: Planning failed, returning FAILURE")
                result = "FAILURE"
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
        visualize_graph_structure_graphviz(root_graph, f"plan_graph_{len(self.graph_network)}")


        self.planning_state = PlanningState.PLANNING
        self.node.get_logger().info("VANAMO planning initialized")
        return "CONTINUE"
    
    def _execute_planning_step(self):
        """Execute one planning step"""
        if self.is_goal_reached():
            self.planning_state = PlanningState.GOAL_REACHED
            self.node.get_logger().info("SHOULD RETURN SUCCESS")
            #self.planning_step()  # Trigger next planning step
            return "SUCCESS"

        if self.at_goal_position() == True:
            self.node.get_logger().info("DEBUG: Already at goal position")


        if self.task_completed() == True:
            self.node.get_logger().info("DEBUG: Task already completed")

        
        
        
        current_graph = self.graph_network[self.current_graph_index]
        
        # Find executable arcs using GNS
        best_arc = self.gns_module.find_best_executable_arc(current_graph)
        
        if not best_arc:
            self.node.get_logger().warn("No executable arcs found - planning failed")
            # No executable arcs - try to expand graph or fail
            if not self._expand_current_graph(current_graph):
                self.planning_state = PlanningState.FAILED
                return "FAILURE"
            return "CONTINUE"

        # Check if this arc is already being executed (race condition protection)
        if best_arc.status == "EXECUTING":
            self.node.get_logger().info(f"Arc {best_arc.action} already being executed, skipping")
            return "CONTINUE"

        # Execute the selected arc immediately (synchronous)
        best_arc.status = "EXECUTING"
        self.current_executing_arc = best_arc
        self.node.get_logger().info(f"Executing arc: {best_arc.action}")
        
        # Execute the hyperarc (synchronous)
        execution_result = self._execute_hyperarc(best_arc)
        
        # Process result
        if execution_result.success:
            self.node.get_logger().info("DEBUG: Arc execution SUCCESS - updating graph")
            best_arc.status = "COMPLETED"  # Mark as completed
            self.gns_module.update_arc_success(current_graph, best_arc)
            self._update_environment(execution_result)
            self.node.get_logger().info(f"Arc execution succeeded: {best_arc.action}")
            
            # Check if goal is reached after successful execution
            if self.is_goal_reached():
                self.node.get_logger().info("DEBUG: Goal reached after arc execution!, SHOULD RETURN SUCCESS")
                self.planning_state = PlanningState.GOAL_REACHED
                #return self._execute_planning_step() 
                return "SUCCESS"
            else:
                self.planning_state = PlanningState.PLANNING
                self.current_executing_arc = None
                # Immediately trigger next planning step after successful execution
                self.node.get_logger().info("DEBUG: Immediately executing next planning step after success")
                return self._execute_planning_step()  # Recursive call for immediate execution
                self.get_logger().info(f"CALLBACK DEBUG: planning_callback called, state: {self.planner.planning_state}")
                #return "CONTINUE"  # Return to continue planning normally
                

                
        else:
            self.node.get_logger().info("DEBUG: Arc execution FAILED")
            best_arc.status = "FAILED"
            self.node.get_logger().warn(f"Arc execution failed: {execution_result.failure_type}")
            
            # Handle failure - DON'T set planning state to FAILED yet for manipulation issues
            if execution_result.failure_type == "MANIPULATION_ISSUE":
                self.node.get_logger().info("DEBUG: Handling manipulation issue - keeping state as PLANNING")
                success = self._handle_manipulation_issue(execution_result)
                if success:
                    self.planning_state = PlanningState.PLANNING
                    self.current_executing_arc = None
                    self.node.get_logger().info("DEBUG: After manipulation handling - State: PLANNING, Graph: {}/{}".format(
                        self.current_graph_index + 1, len(self.graph_network)))
                    
                    # Immediately trigger next planning step after manipulation graph creation
                    self.node.get_logger().info("DEBUG: Immediately executing next planning step after manipulation")
                    return self._execute_planning_step()  # Recursive call for immediate execution
                    return "CONTINUE"

                else:
                    self.node.get_logger().info("DEBUG: Manipulation handling failed - setting state to FAILED")
                    self.planning_state = PlanningState.FAILED
                    return "FAILURE"
            else:
                self.node.get_logger().info("DEBUG: Non-manipulation failure - setting state to FAILED")
                self.planning_state = PlanningState.FAILED
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
    
    def _handle_visibility_issue(self, result):
        """Handle visibility problems by adding observation nodes"""
        if result.new_position:  
            self.q_current = result.new_position
        
        # Calculate viewpoint
        q_view = self.calculate_viewpoint(result.blocked_region, self.q_current)
        
        if q_view is None:
            self.planning_state = PlanningState.FAILED
            return False
        
        # Check for cycles
        state_key = self._get_state_key(self.q_current, q_view)
        if state_key in self.visited_states:
            self.planning_state = PlanningState.FAILED
            return False
        self.visited_states.add(state_key)
        
        # Create new graph for visibility
        visibility_graph = self.aog_module.expand_graph_for_visibility(q_view, self.q_goal)
        
        # Add to graph network
        self.graph_network.append(visibility_graph)
        self.current_graph_index = len(self.graph_network) - 1
        
        # === Add this for visualization ===
        visualize_graph_structure_graphviz(visibility_graph, f"plan_graph_{len(self.graph_network)}")
        
        self.node.get_logger().info(f"Created visibility graph with viewpoint: {q_view}")
        return True
    
        # Spatial reasoning utility functions
    


    def _handle_manipulation_issue(self, result):
        if result.new_position:
            self.q_current = result.new_position

        # Find which obstacle is blocking
        obstacle = getattr(result, 'blocking_obstacle', None)
        if not obstacle:
            self.node.get_logger().error("No blocking obstacle found for manipulation")
            self.planning_state = PlanningState.FAILED
            return False
        
        self.node.get_logger().info(f"Handling manipulation issue with obstacle: {obstacle}")

        # Get obstacle position
        if obstacle not in self.motion_planner.movable_obstacles:
            self.node.get_logger().error(f"Unknown obstacle: {obstacle}")
            self.planning_state = PlanningState.FAILED
            return False

        manip_graph = self.aog_module.expand_graph_for_manipulation(obstacle, self.q_goal)


        self.graph_network.append(manip_graph)
        self.current_graph_index = len(self.graph_network) - 1

        # === Add this for visualization ===
        visualize_graph_structure_graphviz(manip_graph, f"plan_graph_{len(self.graph_network)}")

        self.node.get_logger().info(f"Created manipulation graph for obstacle: {obstacle}")
        
        # DEBUG: Check planning state after graph creation
        self.node.get_logger().info(f"DEBUG: Planning state after manipulation graph creation: {self.planning_state}")
        self.node.get_logger().info(f"DEBUG: Current graph index: {self.current_graph_index}")
        self.node.get_logger().info(f"DEBUG: Total graphs: {len(self.graph_network)}")
        
        return True
    

    
    
    def _expand_current_graph(self, graph):
        """Try to expand current graph or switch to another"""
        # For now, just fail if no executable arcs
        # In future, could try other expansion strategies
        return False
    
    def is_goal_reached(self):
        """Check if goal is reached"""
        return self.task_completed() #and self.at_goal_position()
    
    def at_goal_position(self):
        """Check if robot is at goal position"""
        distance = math.sqrt(
            (self.q_current[0] - self.q_goal[0])**2 + 
            (self.q_current[1] - self.q_goal[1])**2
        )
        return distance < self.position_threshold
    
    def task_completed(self):
        """Check if task is completed based on graph state"""
        if not self.graph_network:
            return False
        current_graph = self.graph_network[self.current_graph_index]
        final_node = current_graph.get_node("N_FINAL")
        return final_node and final_node.status == "ACHIEVED"
    
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
        
        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/diff_cont/odom',
            self.odom_callback, 
            10, callback_group=self.callback_group
        )


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
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            'robot_pose',
            self.pose_callback,
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
        self.planner.q_current = [x, y, yaw]



    def goal_callback(self, msg):
        """Handle new goal messages"""
        goal_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            self._quaternion_to_yaw(msg.pose.orientation)
        ]
        
        self.get_logger().info(f"New goal received: {goal_pose}")
        self.planner.set_goal(goal_pose)
    
    def pose_callback(self, msg):
        """Handle robot pose updates"""
        new_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            self._quaternion_to_yaw(msg.pose.orientation)
        ]
        
        # Update planner with current pose
        self.planner.q_current = new_pose
    
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
