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
from mir_navigation.action import ManipulateBox








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
    
    def expand_graph_for_visibility(self, q_view, q_goal):
        """Create a new graph for handling visibility"""
        new_graph = AOGGraph()
        
        # Current configuration node
        n_current = new_graph.add_node("N_CURRENT_CONFIG", node_type="config")
        n_current.status = "ACHIEVED"
        
        # Viewpoint navigation node
        n_view = new_graph.add_node("N_Q_VIEW", node_type="navigation")
        
        # Navigation to viewpoint arc
        arc_to_view = new_graph.add_hyperarc(
            parent=n_view,
            children=[n_current],
            action="NAVIGATE_TO_VIEW",
            action_params={"q_view": q_view},
            condition=lambda: True
        )
        
        # Observation node
        n_observation = new_graph.add_node("N_OBSERVATION", node_type="observation")

        # Observation arc
        arc_observe = new_graph.add_hyperarc(
            parent=n_observation,
            children=[n_view],
            action="OBSERVE_AND_UPDATE_MAP",
            action_params={"q_view": q_view},
            condition=lambda: True
        )
        
        # Direct navigation node
        n_direct = new_graph.add_node("N_DIRECT", node_type="navigation")

        # Updated navigation arc
        arc_direct = new_graph.add_hyperarc(
            parent=n_direct,
            children=[n_observation],
            action="NAVIGATE_TO_FINAL_GOAL",
            action_params={"q_goal": q_goal},
            condition=lambda: True
        )
        
        # Final node
        n_final = new_graph.add_node("N_FINAL", node_type="final")

        # Final arc
        arc_final = new_graph.add_hyperarc(
            parent=n_final,
            children=[n_direct],
            action="REACH_FINAL_GOAL",
            action_params={},
            condition=lambda: True
        )
        
        self._update_initial_arc_statuses(new_graph)
        return new_graph

    def expand_graph_for_manipulation(self, obstacle, q_goal):
        """Create a new graph for handling manipulation"""
        new_graph = AOGGraph()
        
        # Current configuration node
        n_current = new_graph.add_node("N_CURRENT_CONFIG", node_type="config")  
        n_current.status = "ACHIEVED"
        
        # Manipulation pose node
        n_q_manip = new_graph.add_node("N_Q_MANIP", node_type="navigation")
        
        # Navigation to manipulation pose arc
        arc_to_manip = new_graph.add_hyperarc(
            parent=n_q_manip,
            children=[n_current],
            action="NAVIGATE_TO_MANIPULATION_POSE",
            action_params={"obstacle": obstacle},
            condition=lambda: True
            
        )
        
        # Manipulation node
        n_manipulation = new_graph.add_node("N_MANIPULATION", node_type="manipulation")
        
        # Manipulation arc
        arc_manipulate = new_graph.add_hyperarc(
            parent=n_manipulation,
            children=[n_q_manip],
            action="MANIPULATE_OBSTACLE",
            condition=lambda: True
        )

        
        # Final node
        n_final = new_graph.add_node("N_FINAL", node_type="final")
        
        # Connect to final goal
        arc_final = new_graph.add_hyperarc(
            parent=n_final,
            children=[n_manipulation],
            action="NAVIGATE_TO_FINAL_GOAL",
            action_params={"q_goal": q_goal},
            condition=lambda: True
        )
        
        # IMPORTANT: Update arc statuses after creating all arcs
        self._update_initial_arc_statuses(new_graph)
        print(f"DEBUG: Created manipulation graph with {len(new_graph.hyperarcs)} arcs")
        for i, arc in enumerate(new_graph.hyperarcs):
            print(f"  Arc {i}: {arc.action} - Status: {arc.status}")
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
        
        # Hardcoded movable obstacles (for thesis simplicity)
        self.movable_obstacles = {
            "test_box": {"x": 6.5, "y": 0.5, "length": 0.5, "width": 2.0},
            "test_box_1": {"x": 4.64, "y": 9.15, "length": 0.25, "width": 0.25}
        }

            

    def execute_navigation(self, params):
        """Execute a navigation action via Nav2"""
        target_pose = params.get("target_pose")
        action_label = params.get("action_label", "") 

        if action_label == "NAVIGATE_TO_MANIPULATION_POSE":
            obstacle = params.get("obstacle")
            obs_info = self.movable_obstacles.get(obstacle)
            if not obs_info:
                self.node.get_logger().error(f"Unknown obstacle: {obstacle}")
                return ExecutionResult(success=False, failure_type="INVALID_OBSTACLE")
            self.node.get_logger().info(f"[MANIPULATION] Action call for obstacle at {obs_info}")
            success = self.node.call_manipulate_box_service(obs_info)
            return ExecutionResult(success=success)

        if not target_pose:
            return ExecutionResult(success=False, failure_type="INVALID_PARAMS")
        
        print(f"DEBUG: Sending goal to Nav2: {target_pose}")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._create_pose_stamped(target_pose)
        
        # Send goal to Nav2
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            print("DEBUG: Nav2 action server not available!")
            return ExecutionResult(success=False, failure_type="NAV2_UNAVAILABLE")
        
        print("DEBUG: Nav2 action server available, sending goal...")
        future = self.nav_client.send_goal_async(goal_msg)
    
        # Wait for goal handle (goal acceptance response)
        print("DEBUG: Waiting for goal acceptance...")
        rclpy.spin_until_future_complete(self.node, future, timeout_sec = 10)
        
        if not future.done():
            print("DEBUG: Goal send timeout!")
            return ExecutionResult(success=False, failure_type="GOAL_SEND_TIMEOUT")
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("DEBUG: Goal REJECTED by action server")
            return ExecutionResult(success=False, failure_type="GOAL_REJECTED")
        
        print("DEBUG: Goal ACCEPTED by action server!")

        # Wait for result
        print("DEBUG: Waiting for navigation result...")
        result_future = goal_handle.get_result_async()
        #rclpy.spin_until_future_complete(self.node, result_future)  # Reduced timeout for testing\
        start_time = time.time()
        timeout_sec = 15.0
    
        while not result_future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        result = result_future.result()
        print(f"DEBUG: Navigation result status: {result}")
        
        if result and hasattr(result, 'status'):
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                print("DEBUG: Navigation SUCCEEDED!")
                return ExecutionResult(success=True)
            else:
                print(f"DEBUG: Navigation FAILED with status: {result.status}")
                # Get both failure_type AND blocking_obstacle
                failure_type, blocking_obstacle = self._analyze_navigation_failure(target_pose, self.node.planner.q_current)
                return ExecutionResult(
                    success=False, 
                    failure_type=failure_type,
                    blocking_obstacle=blocking_obstacle
                )
        else:
            print(f"DEBUG: Navigation FAILED with status: {result.status}")
            # Get both failure_type AND blocking_obstacle in one call
            failure_type, blocking_obstacle = self._analyze_navigation_failure(target_pose, self.node.planner.q_current)
            return ExecutionResult(
                success=False, 
                failure_type=failure_type,
                blocking_obstacle=blocking_obstacle  # Pass obstacle info
            )
    
         
    def execute_observation(self, params):
        """Execute an observation action"""
        q_view = params.get("q_view")
        if not q_view:
            return ExecutionResult(success=False, failure_type="INVALID_PARAMS")
        
        # Navigate to viewpoint first
        nav_result = self.execute_navigation({"target_pose": q_view})
        if not nav_result.success:
            return nav_result
        
        # Simulate observation (in real implementation, trigger sensors)
        self.node.get_logger().info("Performing observation...")
        
        # Simulate map update (simplified)
        map_updates = [{"position": q_view, "status": "observed"}]
        
        return ExecutionResult(
            success=True, 
            new_position=q_view,
            map_updates=map_updates
        )
    
      
    
    def _analyze_navigation_failure(self, target_pose,current_pose):
        """Analyze why navigation failed and determine failure type"""
        # For testing: Force a manipulation issue to test the fix
        # Check if target is near known obstacles
        target_x, target_y = target_pose[0], target_pose[1]

        # Check robot's current position
        current_x, current_y = current_pose[0], current_pose[1]
        # Distance to target
        target_distance = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
        print(f"DEBUG: Robot-target distance: {target_distance:.2f}m")
    
        min_distance = float('inf')
        blocking_obstacle = None  # <-- FIX: Initialize blocking_obstacle

        # Check distance to movable obstacles
        for obstacle_name, obstacle_pos in self.movable_obstacles.items():
            # Distance from target to obstacle
            target_obs_distance = math.sqrt((target_x - obstacle_pos["x"])**2 + 
                                        (target_y - obstacle_pos["y"])**2)
            
            # Distance from robot to obstacle
            robot_obs_distance = math.sqrt((current_x - obstacle_pos["x"])**2 + 
                                        (current_y - obstacle_pos["y"])**2)
            
            print(f"DEBUG: Obstacle {obstacle_name} - Target distance: {target_obs_distance:.2f}m, Robot distance: {robot_obs_distance:.2f}m")
            
            # If target is near obstacle AND robot got close to obstacle
            if target_obs_distance < 10 and robot_obs_distance < 15:
                print(f"DEBUG: Navigation blocked by movable obstacle: {obstacle_name}")
                if robot_obs_distance < min_distance:
                    min_distance = robot_obs_distance
                    blocking_obstacle = obstacle_name
        if blocking_obstacle:
            print(f"DEBUG: Navigation blocked by: {blocking_obstacle}")
            return "MANIPULATION_ISSUE", blocking_obstacle
        else:
            return "VISIBILITY_ISSUE", None

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
        """Execute a hyperarc through motion planning"""
        self.node.get_logger().info(f"DEBUG: Executing arc: {arc.action}")
        self.node.get_logger().info(f"DEBUG: Arc action_params: {arc.action_params}")
        
        if arc.action.startswith("NAVIGATE"):
            if "NAVIGATE_TO_FINAL_GOAL" in arc.action:
                params = {"target_pose": self.q_goal}
                self.node.get_logger().info(f"DEBUG: Using final goal: {self.q_goal}")
            elif "VIEW" in arc.action:
                params = {"target_pose": arc.action_params.get("q_view")}
                self.node.get_logger().info(f"DEBUG: Using view pose: {arc.action_params.get('q_view')}")
            else:
                params = {"target_pose": self.q_goal, "action_label": arc.action, **arc.action_params}
                self.node.get_logger().info(f"DEBUG: Using default goal: {self.q_goal}")
            return self.motion_planner.execute_navigation(params)
            
        elif arc.action.startswith("OBSERVE"):
            return self.motion_planner.execute_observation(arc.action_params)
            
        elif arc.action.startswith("MANIPULATE"):
            return ExecutionResult(
            success=True,
            new_position=self.node.planner.q_current,
            map_updates=[]
            )  # We already achieved this in the service call.

        return ExecutionResult(success=False, failure_type="UNKNOWN_ACTION")
    
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

        self.manipulate_box_client = ActionClient(self, ManipulateBox, 'manipulate_box')



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

    def call_manipulate_box_service(self, obstacle_info):
        if not self.manipulate_box_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("manipulate_box action server unavailable!")
            return False

        goal = ManipulateBox.Goal()
        goal.x = float(obstacle_info["x"])
        goal.y = float(obstacle_info["y"])
        goal.length = float(obstacle_info["length"])
        goal.width = float(obstacle_info["width"])

        send_goal_future = self.manipulate_box_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("ManipulateBox goal rejected!")
            return False

        self.get_logger().info("ManipulateBox goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info("Manipulation succeeded via action.")
            return True
        else:
            self.get_logger().warn("Manipulation failed via action.")
            return False


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
