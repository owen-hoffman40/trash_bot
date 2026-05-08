#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image
import numpy as np
import casadi as ca
import math
import message_filters
from pathlib import Path as FilePath
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


#We use this repository for basis of MPC code: https://github.com/astomodynamics/casadi_mpc/blob/master/casadi_mpc/casadi_mpc.py
class MPCController(Node):
    def __init__(self, goal=None, initial_pose=None, obstacles=None):
        super().__init__('casadi_mpc_node') 

        # Declare and get the robot_id parameter
        self.declare_parameter("robot_id", "j100_0857")
        self.robot_id = self.get_parameter("robot_id").value
        self.goal_tolerance = 0.30

        # MPC parameters
        self.horizon = 20       # MPC prediction horizon (number of control intervals)
        self.dt = 0.1           # Discrete time step [s]

        # States: [x, y, theta] and Controls: [v, omega]
        # Cost weights
        self.Q = ca.diag([0.1, 0.1, 0.0])
        self.R = ca.diag([0.01, 0.01])
        self.Qf = ca.diag([0.0, 0.0, 0.0])  # Terminal cost weight

        # Box constraints for state and inputs.
        self.x_min = -ca.inf
        self.x_max = ca.inf
        self.y_min = -ca.inf
        self.y_max = ca.inf
        self.theta_min = -ca.pi
        self.theta_max = ca.pi
        self.v_min = -0.5
        self.v_max = 0.5
        self.omega_min = -ca.pi
        self.omega_max = ca.pi
        self.front_angle_deg = 45.0
        self.obs_to_close = False

        # Current state (initialized to zeros)
        self.current_state = ca.DM.zeros(3)
        if initial_pose is None:
            self.initial_pose = None  # Latched once from first odometry message
        else:
            self.initial_pose = ca.DM(initial_pose)
        self.goal_reached = False

        # Default goal state: [x, y, theta]
        if goal is None:
            self.goal_state = ca.DM([5.0, 0.0, 0.0])
        else:
            self.goal_state = ca.DM(goal)

        self.is_current_pose_received = False
        if goal is None:
            self.is_goal_pose_received = False
        else:
            self.is_goal_pose_received = True

        # Obstacle parameters:
        self.obstacle_centers = []
        if obstacles is None:   
            self.get_logger().warn("No obstacles found! MPC will run without obstacle avoidance.")
        else:
            for obs in obstacles:
                print(f"Obstacle at x={obs['x']:.2f}, y={obs['y']:.2f}, radius={obs['radius']:.2f}")
                self.obstacle_centers.append((obs['x'], obs['y']))

        print(obstacles, "THESE ARE THER OBSTACLES ")
        # if self.initial_pose is not None:
        #     self.obstacle_centers = [(self.initial_pose[0] + 1, self.initial_pose[1] - 0.2)]
        # FIXME: if you want to add obstacles, add them here, i.e.
        # self.obstacle_centers = [(0.762, 2.54), (2.794, 3.429), (0.762, 4.318)]
        self.obstacle_radius = .50

        # Scan proximity check parameters
        self.min_range_m = 0.15
        self.max_range_m = 1.0
        self.lateral_limit_m = 0.70
        self.latest_scan = None
        self.close_obstacle_threshold_m = 0.48

        # Set up the MPC problem formulation using CasADi.
        self.setup_mpc()

        self.state_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.current_state_callback,
            10)
        
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10)

        self.control_pub = self.create_publisher(
            Twist,
            '/commands/velocity',
            10)
        
        self.local_path_pub = self.create_publisher(
            Path,
            '/local_path',
            10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/trashbot/depth_scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        
        self.control_timer = self.create_timer(0.1, self.control_callback)
        self.obj_timer = self.create_timer(0.1, self.obstacle_callback)

        self.get_logger().info(
            f'CasadiMPCNode for robot_id "{self.robot_id}" initialized with goal '
            f'[{float(self.goal_state[0]):.2f}, {float(self.goal_state[1]):.2f}, {float(self.goal_state[2]):.2f}] '
            f'and tolerance {self.goal_tolerance:.2f} m'
        )

    def setup_mpc(self):
        # ---------------------------------------------------------------------
        # 1. Define Symbolic Variables for State and Control
        # ---------------------------------------------------------------------
        x = ca.SX.sym('x', 3)  # state: [x, y, theta]
        u = ca.SX.sym('u', 2)  # control: [v, omega]

        # Continuous dynamics:
        x_dot = ca.vertcat(u[0] * ca.cos(x[2]),
                           u[0] * ca.sin(x[2]),
                           u[1])
        # Discrete-time dynamics via Euler integration:
        x_next = x + self.dt * x_dot
        f = ca.Function('f', [x, u], [x_next])
        self.f = f

        # ---------------------------------------------------------------------
        # 2. Define Decision Variables and Parameters
        # ---------------------------------------------------------------------
        X = ca.SX.sym('X', 3, self.horizon + 1)  # states at time steps 0,...,horizon
        U = ca.SX.sym('U', 2, self.horizon)       # controls at time steps 0,...,horizon-1
        self.X = X
        self.U = U

        # Parameters: initial state P and reference (goal) state ref.
        P = ca.SX.sym('P', 3)
        ref = ca.SX.sym('ref', 3)
        self.P = P
        self.ref = ref

        # ---------------------------------------------------------------------
        # 3. Build the Cost Function
        # ---------------------------------------------------------------------
        cost = 0
        for k in range(self.horizon):
            cost += ca.mtimes((X[:, k] - ref).T, ca.mtimes(self.Q, (X[:, k] - ref))) \
                    + ca.mtimes(U[:, k].T, ca.mtimes(self.R, U[:, k]))
        cost += ca.mtimes((X[:, self.horizon] - ref).T, ca.mtimes(self.Qf, (X[:, self.horizon] - ref)))

        # ---------------------------------------------------------------------
        # 4. Build the Constraints
        # ---------------------------------------------------------------------
        g_eq = []
        # (a) Initial condition: X[:, 0] = P.
        g_eq.append(X[:, 0] - P)
        # (b) Dynamics constraints: for k = 0,...,horizon-1.
        for k in range(self.horizon):
            g_eq.append(X[:, k+1] - f(X[:, k], U[:, k]))
        g_eq = ca.vertcat(*g_eq)

        # (B) Obstacle avoidance constraints.
        g_obs_list = []
        for k in range(self.horizon + 1):
            for (obs_x, obs_y) in self.obstacle_centers:
                obs_constraint = (X[0, k] - obs_x)**2 + (X[1, k] - obs_y)**2 - self.obstacle_radius**2
                g_obs_list.append(obs_constraint)
        g_obs = ca.vertcat(*g_obs_list)

        # Combine constraints.
        g_total = ca.vertcat(g_eq, g_obs)
        self.g_total = g_total

        # ---------------------------------------------------------------------
        # 5. Formulate the NLP
        # ---------------------------------------------------------------------
        Z = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        self.Z = Z

        # Parameter vector: [P; ref]
        params = ca.vertcat(P, ref)
        nlp = {'x': Z, 'f': cost, 'g': g_total, 'p': params}

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-6,
            'print_time': False
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Save constraint dimensions for later use.
        self.n_eq = 3 + 3 * self.horizon
        self.n_obs = len(self.obstacle_centers) * (self.horizon + 1)

    def current_state_callback(self, msg):
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        _, _, current_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.current_state = ca.DM([current_x, current_y, current_yaw])
        if self.initial_pose is None:
            self.initial_pose = (current_x, current_y, current_yaw)
            self.get_logger().info(
                f'Initial pose saved: x={current_x:.2f}, y={current_y:.2f}, theta={current_yaw:.2f}'
            )
        self.is_current_pose_received = True
        self.get_logger().info(
            f'Received state: x={current_x:.2f}, y={current_y:.2f}, theta={current_yaw:.2f}'
        )

    def path_callback(self, msg):
        self.path = msg
        self.ref_path = ca.DM([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])

    def goal_pose_callback(self, msg):
        # Extract goal position and orientation.
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        _, _, goal_yaw = self.euler_from_quaternion(msg.pose.orientation)
        self.goal_state = ca.DM([goal_x, goal_y, goal_yaw])
        self.is_goal_pose_received = True
        self.get_logger().info(
            f'Updated goal state to: x={goal_x:.2f}, y={goal_y:.2f}, theta={goal_yaw:.2f}'
        )

    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    
    def control_callback(self):
        # Check if pose and goal are received:
        if not self.is_current_pose_received or not self.is_goal_pose_received:
            self.get_logger().info(f'Current or Goal pose not received')
            return
        
        if self.obs_to_close:
            self.get_logger().warn(f'Obstacle too close, stopping robot')
            control_msg = Twist()
            control_msg.linear.x = 0.0
            control_msg.angular.z = 0.0
            self.control_pub.publish(control_msg)
            return

        # If the robot reaches the goal, reset is_goal_pose_received
        current_xy = np.array(self.current_state[:2].full()).flatten()
        goal_xy = np.array(self.goal_state[:2].full()).flatten()
        dist_to_goal = np.linalg.norm(current_xy - goal_xy)
        self.get_logger().info(
            f'Distance to goal: {dist_to_goal:.3f} m (tol={self.goal_tolerance:.2f} m)',
            throttle_duration_sec=1.0
        )
        yaw_diff = abs(float(self.current_state[2]) - float(self.goal_state[2]))
        self.get_logger().info(
            f'Yaw difference to goal: {yaw_diff:.3f} rad',
            throttle_duration_sec=1.0
        )
        if dist_to_goal < self.goal_tolerance and yaw_diff < 0.1:
            # Return zero control
            self.goal_reached = True
            control_msg = Twist()
            control_msg.linear.x = 0.0
            control_msg.angular.z = 0.0
            self.control_pub.publish(control_msg)

            self.is_goal_pose_received = False
            self.get_logger().info(f'Reached goal. Goal pose reset')
            return
        #START HERE, COMMENT OUT IF WE DO NOT WANT THIS BEHAVIOR
        elif dist_to_goal < self.goal_tolerance and yaw_diff >= 0.1:
            self.get_logger().info(f'Close to goal position but still adjusting orientation.')
            if (float(self.current_state[2]) - float(self.goal_state[2])) > 0:
                control_msg = Twist()
                control_msg.linear.x = 0.0
                control_msg.angular.z = -1.0
                self.control_pub.publish(control_msg)
                self.get_logger().info(
                    f'Published velocities: linear={control_msg.linear.x:.2f}, angular={control_msg.angular.z:.2f}'
                )
                return
            else:
                control_msg = Twist()
                control_msg.linear.x = 0.0
                control_msg.angular.z = 1.0 
                self.control_pub.publish(control_msg)
                self.get_logger().info(
                    f'Published velocities: linear={control_msg.linear.x:.2f}, angular={control_msg.angular.z:.2f}'
                )
                return
        
        # 1. Create an Initial Guess for the Decision Variables.
        x0_val = np.array(self.current_state.full().flatten())
        goal_val = np.array(self.goal_state.full().flatten())
        X0 = np.zeros((3, self.horizon + 1))
        for i in range(3):
            X0[i, :] = np.linspace(x0_val[i], goal_val[i], self.horizon + 1)
        U0 = np.zeros((2, self.horizon))
        Z0 = np.concatenate((X0.reshape(-1, order='F'), U0.reshape(-1, order='F')))

        # 2. Build Variable Bounds.
        lbx_states = []
        ubx_states = []
        for _ in range(self.horizon + 1):
            lbx_states.extend([self.x_min, self.y_min, self.theta_min])
            ubx_states.extend([self.x_max, self.y_max, self.theta_max])
        lbx_controls = []
        ubx_controls = []
        for _ in range(self.horizon):
            lbx_controls.extend([self.v_min, self.omega_min])
            ubx_controls.extend([self.v_max, self.omega_max])
        lbx = lbx_states + lbx_controls
        ubx = ubx_states + ubx_controls

        # 3. Build Constraint Bounds.
        lbg_eq = [0.0] * self.n_eq
        ubg_eq = [0.0] * self.n_eq
        lbg_obs = [0.0] * self.n_obs
        ubg_obs = [1e20] * self.n_obs
        lbg_total = lbg_eq + lbg_obs
        ubg_total = ubg_eq + ubg_obs

        # 4. Solve the NLP.
        p_val = np.concatenate((x0_val, goal_val))
        sol = self.solver(
            x0=Z0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg_total,
            ubg=ubg_total,
            p=p_val
        )
        Z_opt = sol['x'].full().flatten()

        # Extract the state and control trajectories.
        n_states = 3 * (self.horizon + 1)
        X_opt = Z_opt[:n_states].reshape((3, self.horizon+1), order='F')
        U_opt = Z_opt[n_states:].reshape((2, self.horizon), order='F')

        # 5. Publish the First Control Input.
        control_msg = Twist()
        control_msg.linear.x = float(U_opt[0, 0])
        control_msg.angular.z = float(U_opt[1, 0])
        self.control_pub.publish(control_msg)
        self.get_logger().info(
            f'Published velocities: linear={U_opt[0, 0]:.2f}, angular={U_opt[1, 0]:.2f}'
        )

        # 6. Publish the Predicted Local Path.
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"  # Adjust frame as needed
        for i in range(self.horizon + 1):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = X_opt[0, i]
            pose_stamped.pose.position.y = X_opt[1, i]
            pose_stamped.pose.position.z = 0.0
            theta = X_opt[2, i]
            # Convert yaw to quaternion (assuming roll=pitch=0)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(theta/2.0)
            pose_stamped.pose.orientation.w = math.cos(theta/2.0)
            path_msg.poses.append(pose_stamped)
        self.local_path_pub.publish(path_msg)
        self.get_logger().info('Published local predicted path.')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg 

    def obstacle_callback(self):
        if self.latest_scan is None:
            return
        print("Checking for obstacles using latest scan data...")
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        for i, range_m in enumerate(self.latest_scan.ranges):
            if not math.isfinite(range_m):
                continue
            if range_m < self.min_range_m or range_m > self.max_range_m:
                continue

            angle = angle_min + i * angle_increment
            if abs(math.degrees(angle)) > self.front_angle_deg:
                continue

            y_robot = range_m * math.sin(angle)
            if abs(y_robot) > self.lateral_limit_m:
                continue

            if range_m < self.close_obstacle_threshold_m:
                self.obs_to_close = True
                self.get_logger().warn(f'Obstacle at {range_m:.2f} m! Stopping.')
                return

        self.obs_to_close = False


def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()