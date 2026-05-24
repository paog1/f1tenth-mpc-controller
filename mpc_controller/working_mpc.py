import rclpy
from rclpy.node import Node
import numpy as np
import osqp
from scipy import sparse
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import pandas as pd

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('N', 20),                  # Prediction horizon (reduced for performance)
                ('Ts', 0.1),                # Sampling time
                ('speed_max', 10.0),         # Max speed
                ('steer_max', 0.4),         # Max steering
                ('wheelbase', 0.325),       # Wheelbase
                ('lf', 0.178),              # Front wheel to CoG
                ('lr', 0.147),              # Rear wheel to CoG
                ('q_pos', 5.0),            # Position tracking weight
                ('q_angle', 20.0),         # Increased angle tracking weight
                ('q_vel', 1.0),            # Velocity tracking weight
                ('r_accel', 0.5),          # Acceleration control weight
                ('r_steer', 2.0),          # Steering control weight
                ('safety_distance', 0.2),  # Increased safety distance
                ('lookahead_distance', 2.0),# Reduced lookahead distance
                ('min_speed', 0.3),        # Minimum speed
                ('trajectory_color_r', 0.0),
                ('trajectory_color_g', 1.0),
                ('trajectory_color_b', 0.0)
            ]
        )

        # Initialize variables
        self.N = self.get_parameter('N').value
        self.Ts = self.get_parameter('Ts').value
        self.nx = 6  # State: [x, y, yaw, vx, vy, omega]
        self.nu = 2  # Control: [throttle, steering]

        # ROS interfaces
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.trajectory_pub = self.create_publisher(Marker, '/mpc_trajectory', 10)
        self.ref_trajectory_pub = self.create_publisher(Marker, '/ref_trajectory', 10)

        # State and control variables
        self.current_pose = np.zeros(self.nx)
        self.last_control = np.zeros(self.nu)
        self.lidar_ranges = None
        self.centerline = None
        self.problem_setup = False

        # Initialize QP problem
        self.setup_osqp_problem()

        self.get_logger().info("MPC Controller Initialized")

    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)
        front_ranges = np.concatenate([self.lidar_ranges[-45:], self.lidar_ranges[:45]])
        min_distance = np.min(front_ranges[front_ranges > 0.1])
        
        if min_distance < self.get_parameter('safety_distance').value:
            self.publish_control(0.0, 0.0)
            self.get_logger().warn(f"Emergency stop! Obstacle at {min_distance:.2f}m")

    def odom_callback(self, msg):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        self.current_pose[2] = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose[3] = msg.twist.twist.linear.x
        self.current_pose[4] = msg.twist.twist.linear.y
        self.current_pose[5] = msg.twist.twist.angular.z

        if self.centerline is None:
            self.initialize_centerline()
        
        ref_trajectory = self.get_reference_trajectory()
        control = self.solve_mpc(ref_trajectory)
        self.publish_control(control[0], control[1])

    def initialize_centerline(self):
        """Load centerline from CSV or create oval track"""
        csv_path = "/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/SaoPaulo_centerline.csv"
        try:
            df = pd.read_csv(csv_path, comment='#', names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
            self.centerline = np.column_stack([df['x_m'], df['y_m']])
            self.get_logger().info(f"Loaded centerline with {len(self.centerline)} points")
            self.visualize_centerline()
        except Exception as e:
            self.get_logger().error(f"Failed to load centerline: {str(e)}")
            exit(1)


    def solve_mpc(self, ref_trajectory):
        """Solve MPC problem"""
        try:
            P, q, A, l, u = self.build_qp_matrices(ref_trajectory)
            
            # Update QP problem
            self.problem.update(Px=sparse.triu(P).data, q=q, Ax=A.data, l=l, u=u)
            result = self.problem.solve()
            
            if result.info.status_val == 1:  # OSQP_SOLVED
                predicted_states = result.x[:self.nx*(self.N+1)].reshape(self.N+1, self.nx)
                controls = result.x[-self.nu*self.N:].reshape(self.N, self.nu)
                
                self.visualize_trajectory(predicted_states, self.trajectory_pub, [0.0, 1.0, 0.0])
                return controls[0]  # Return first control action
            else:
                self.get_logger().warn(f"MPC solve failed with status {result.info.status}")
                return self.pure_pursuit_fallback(ref_trajectory)
                
        except Exception as e:
            self.get_logger().error(f"MPC error: {str(e)}")
            return self.pure_pursuit_fallback(ref_trajectory)
            
    def get_reference_trajectory(self):
        """Get smoothed reference trajectory with velocity profile"""
        if self.centerline is None:
            return np.zeros((self.N, self.nx))
            
        # Find closest point
        distances = np.linalg.norm(self.centerline - self.current_pose[:2], axis=1)
        closest_idx = np.argmin(distances)
        
        # Extract a longer segment of the centerline
        lookahead_points = int(self.get_parameter('lookahead_distance').value * 5)  # More points for smoothing
        centerline_segment = np.concatenate([
            self.centerline[closest_idx:closest_idx+lookahead_points],
            self.centerline[:max(0, (closest_idx+lookahead_points)-len(self.centerline))]
        ])
        
        # Smooth the segment (moving average)
        window_size = 5
        if len(centerline_segment) > window_size:
            kernel = np.ones(window_size) / window_size
            smooth_x = np.convolve(centerline_segment[:,0], kernel, mode='valid')
            smooth_y = np.convolve(centerline_segment[:,1], kernel, mode='valid')
            centerline_segment = np.column_stack([smooth_x, smooth_y])
        
        # Create reference trajectory
        ref_trajectory = np.zeros((self.N, self.nx))
        segment_length = len(centerline_segment)
        
        for i in range(self.N):
            # Select point along the segment
            idx = min(int(i * segment_length / self.N), segment_length - 1)
            next_idx = min(idx + 1, segment_length - 1)
            
            ref_trajectory[i, :2] = centerline_segment[idx]
            ref_trajectory[i, 2] = math.atan2(
                centerline_segment[next_idx, 1] - centerline_segment[idx, 1],
                centerline_segment[next_idx, 0] - centerline_segment[idx, 0]
            )
            
            # Adaptive speed based on curvature
            if i > 0:
                dx = ref_trajectory[i,0] - ref_trajectory[i-1,0]
                dy = ref_trajectory[i,1] - ref_trajectory[i-1,1]
                dtheta = ref_trajectory[i,2] - ref_trajectory[i-1,2]
                curvature = abs(dtheta) / math.sqrt(dx**2 + dy**2)
                
                # Reduce speed for high curvature
                max_speed = self.get_parameter('speed_max').value
                min_speed = self.get_parameter('min_speed').value
                ref_trajectory[i,3] = max(min_speed, 
                                        max_speed * (1.0 - 0.8 * min(1.0, curvature)))
            else:
                ref_trajectory[i,3] = self.get_parameter('speed_max').value * 0.7
        
        # Visualize reference trajectory
        self.visualize_trajectory(ref_trajectory, self.ref_trajectory_pub, [0.0, 0.0, 1.0])
        
        return ref_trajectory

    def setup_osqp_problem(self):
        """Initialize OSQP problem"""
        n_var = self.nx*(self.N+1) + self.nu*self.N
        n_con = self.nx*(self.N+1) + self.nu*self.N  # Added control constraints
        
        # Initialize matrices
        P = sparse.csc_matrix((n_var, n_var))
        q = np.zeros(n_var)
        A = sparse.csc_matrix((n_con, n_var))
        l = np.zeros(n_con)
        u = np.zeros(n_con)
        
        self.problem = osqp.OSQP()
        self.problem.setup(P, q, A, l, u, verbose=False, warm_start=True)
        self.problem_setup = True

    def build_qp_matrices(self, ref_trajectory):
        """Build QP matrices for MPC"""
        n_var = self.nx*(self.N+1) + self.nu*self.N
        n_con = self.nx*(self.N+1) + self.nu*self.N  # Added control constraints
        
        # Weight matrices
        Q = sparse.diags([
            self.get_parameter('q_pos').value,
            self.get_parameter('q_pos').value,
            self.get_parameter('q_angle').value,
            self.get_parameter('q_vel').value,
            0.1,  # vy weight
            0.1   # omega weight
        ])
        
        R = sparse.diags([
            self.get_parameter('r_accel').value,
            self.get_parameter('r_steer').value
        ])
        
        # Construct big Q and R matrices
        Q_big = sparse.kron(sparse.eye(self.N+1), Q)
        R_big = sparse.kron(sparse.eye(self.N), R)
        P = sparse.block_diag([Q_big, R_big], format='csc')
        
        # Reference vector
        ref_vec = np.zeros(self.nx*(self.N+1))
        for i in range(min(self.N+1, len(ref_trajectory))):
            ref_vec[i*self.nx:(i+1)*self.nx] = ref_trajectory[i]
        q = -Q_big @ ref_vec
        q = np.concatenate([q, np.zeros(self.nu*self.N)])
        
        # Dynamics matrices (bicycle model)
        dt = self.Ts
        L = self.get_parameter('wheelbase').value
        v = max(self.current_pose[3], 0.1)  # Avoid division by zero
        
        # Continuous-time A matrix
        Ac = np.zeros((self.nx, self.nx))
        Ac[0, 2] = -v * np.sin(self.current_pose[2])  # x_dot depends on yaw
        Ac[1, 2] = v * np.cos(self.current_pose[2])   # y_dot depends on yaw
        Ac[2, 5] = 1.0                               # yaw_dot = omega
        
        # Continuous-time B matrix
        Bc = np.zeros((self.nx, self.nu))
        Bc[3, 0] = 1.0  # throttle affects acceleration
        Bc[5, 1] = v/L  # steering affects yaw rate
        
        # Discretize (Euler method)
        Ad = sparse.eye(self.nx) + Ac * dt
        Bd = Bc * dt
        
        # Dynamics constraints
        A_dyn = sparse.kron(sparse.eye(self.N+1), -sparse.eye(self.nx)) + \
                sparse.kron(sparse.eye(self.N+1, k=-1), Ad)
        A_ctrl = sparse.hstack([
            sparse.csc_matrix((self.nx*(self.N+1), self.nx*(self.N+1))),
            sparse.kron(sparse.eye(self.N+1, k=-1)[1:], Bd)
        ])
        A_eq = sparse.hstack([A_dyn, A_ctrl])
        
        # Control constraints
        A_control = sparse.hstack([
            sparse.csc_matrix((self.nu*self.N, self.nx*(self.N+1))),
            sparse.eye(self.nu*self.N)
        ])
        
        # Combine all constraints
        A = sparse.vstack([A_eq, A_control], format='csc')
        
        # Bounds
        l_eq = np.zeros(self.nx*(self.N+1))
        u_eq = np.zeros(self.nx*(self.N+1))
        l_eq[:self.nx] = -self.current_pose
        u_eq[:self.nx] = -self.current_pose
        
        l_control = np.tile([0, -self.get_parameter('steer_max').value], self.N)
        u_control = np.tile([1.0, self.get_parameter('steer_max').value], self.N)
        
        l = np.concatenate([l_eq, l_control])
        u = np.concatenate([u_eq, u_control])
        
        return P, q, A, l, u

    def pure_pursuit_fallback(self, ref_trajectory):
        """Pure pursuit fallback controller"""
        lookahead = self.get_parameter('lookahead_distance').value
        closest_idx = np.argmin(np.linalg.norm(ref_trajectory[:, :2] - self.current_pose[:2], axis=1))
        target_idx = min(closest_idx + int(lookahead/0.1), len(ref_trajectory)-1)
        target_point = ref_trajectory[target_idx, :2]
        
        # Transform target to vehicle coordinates
        dx = target_point[0] - self.current_pose[0]
        dy = target_point[1] - self.current_pose[1]
        target_local = np.array([
            dx * np.cos(self.current_pose[2]) + dy * np.sin(self.current_pose[2]),
            -dx * np.sin(self.current_pose[2]) + dy * np.cos(self.current_pose[2])
        ])
        
        # Calculate curvature
        L = self.get_parameter('wheelbase').value
        alpha = np.arctan2(target_local[1], target_local[0])
        steering = np.arctan2(2 * L * np.sin(alpha), lookahead)
        
        # Limit steering and maintain speed
        steering = np.clip(steering, 
                         -self.get_parameter('steer_max').value,
                         self.get_parameter('steer_max').value)
        throttle = 0.5  # Moderate speed
        
        return np.array([throttle, steering])

    def visualize_centerline(self):
        """Publish centerline as marker points"""
        if self.centerline is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "centerline"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0  # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0  # Red color

        # Add all centerline points
        for point in self.centerline:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0  # Flat on the ground
            marker.points.append(p)
    
        self.trajectory_pub.publish(marker)

    def visualize_trajectory(self, trajectory, publisher, color):
        """Visualize trajectory as marker array"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        for i in range(0, len(trajectory), 2):  # Skip points for performance
            point = Point()
            point.x = float(trajectory[i, 0])
            point.y = float(trajectory[i, 1])
            point.z = 0.0
            marker.points.append(point)
        
        publisher.publish(marker)

    def publish_control(self, throttle, steering):
        """Publish control commands"""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Apply limits
        throttle = np.clip(throttle, 0.0, 1.0)
        steering = np.clip(steering, 
                         -self.get_parameter('steer_max').value,
                         self.get_parameter('steer_max').value)
        
        # Convert throttle to speed
        min_speed = self.get_parameter('min_speed').value
        max_speed = self.get_parameter('speed_max').value
        speed = min_speed + throttle * (max_speed - min_speed)
        
        msg.drive.speed = speed
        msg.drive.steering_angle = steering
        self.drive_pub.publish(msg)
        
        self.get_logger().info(
            f"Control: Speed={speed:.2f} m/s, Steering={math.degrees(steering):.1f}°"
        )

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    controller = MPCController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()