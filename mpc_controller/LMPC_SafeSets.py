#!/usr/bin/env python3
"""
LMPC for F1TENTH (ROS 2 / rclpy)

Reworked to be closer to the local LMPC structure used in the Berkeley racing papers:
- local convex safe set selected around the CURRENT state (not z_t)
- local convex Q-function (terminal linear cost J^T lambda)
- affine time-varying model from local regression on recent successful laps
- stage costs on handling states and inputs
- improved Frenet projection with local segment projection and continuity
- periodic interpolation fixed across track wrap-around
- safer terminal candidate update

Added:
- Pure Pursuit seeding phase
- default long seeding (20 laps)
- better RViz visualization in GLOBAL frame
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import osqp
import pandas as pd
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile
from scipy import sparse

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


def sat(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Track:
    s: np.ndarray
    kappa: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None
    w_left: Optional[np.ndarray] = None
    w_right: Optional[np.ndarray] = None

    @property
    def L(self) -> float:
        return float(self.s[-1]) if len(self.s) else 1.0

    def _interp_pair(self, arr: np.ndarray, s_query: float) -> float:
        L = self.L if self.L > 0 else 1.0
        s_wrapped = s_query % L
        idx = int(np.searchsorted(self.s, s_wrapped))

        if idx == 0:
            i0 = len(self.s) - 1
            i1 = 0
            s0 = float(self.s[i0]) - L
            s1 = float(self.s[i1])
        elif idx >= len(self.s):
            i0 = len(self.s) - 1
            i1 = 0
            s0 = float(self.s[i0])
            s1 = float(self.s[i1]) + L
        else:
            i0 = idx - 1
            i1 = idx
            s0 = float(self.s[i0])
            s1 = float(self.s[i1])

        sq = s_wrapped
        if sq < s0:
            sq += L

        a0, a1 = float(arr[i0]), float(arr[i1])
        if abs(s1 - s0) < 1e-9:
            return a0

        w = (sq - s0) / (s1 - s0)
        return (1.0 - w) * a0 + w * a1

    def kappa_at(self, s_query: float) -> float:
        return float(self._interp_pair(self.kappa, s_query))

    def widths_at(self, s_query: float, margin: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
        if self.w_left is None or self.w_right is None:
            return None, None
        wl = float(self._interp_pair(self.w_left, s_query))
        wr = float(self._interp_pair(self.w_right, s_query))
        return max(0.0, wl - margin), max(0.0, wr - margin)


class FrenetProjector:
    def __init__(self, track: Track):
        self.track = track
        self.has_xy = (track.x is not None) and (track.y is not None) and (track.theta is not None)
        self.last_idx = 0

    def _segment_project(self, px: float, py: float, i0: int, i1: int) -> Tuple[float, float, float, int]:
        tx0 = float(self.track.x[i0])
        ty0 = float(self.track.y[i0])
        tx1 = float(self.track.x[i1])
        ty1 = float(self.track.y[i1])

        vx = tx1 - tx0
        vy = ty1 - ty0
        seg2 = vx * vx + vy * vy

        t = 0.0 if seg2 < 1e-10 else sat(((px - tx0) * vx + (py - ty0) * vy) / seg2, 0.0, 1.0)

        xproj = tx0 + t * vx
        yproj = ty0 + t * vy

        dx = px - xproj
        dy = py - yproj
        dist2 = dx * dx + dy * dy

        s0 = float(self.track.s[i0])
        s1 = float(self.track.s[i1])
        if i1 == 0 and i0 == len(self.track.s) - 1:
            s1 += self.track.L
        s_proj = s0 + t * (s1 - s0)

        th0 = float(self.track.theta[i0])
        th1 = float(self.track.theta[i1])
        dth = wrap_pi(th1 - th0)
        th = th0 + t * dth

        nx, ny = -math.sin(th), math.cos(th)
        ey = (px - xproj) * nx + (py - yproj) * ny

        return dist2, s_proj % self.track.L, ey, i0

    def project(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        if not self.has_xy:
            return x, y, yaw

        n = len(self.track.s)
        if n < 2:
            return 0.0, 0.0, 0.0

        best = None

        for off in range(-25, 26):
            i0 = (self.last_idx + off) % n
            i1 = (i0 + 1) % n
            cand = self._segment_project(x, y, i0, i1)
            if best is None or cand[0] < best[0]:
                best = cand

        if best is None or best[0] > 1.5**2:
            for i0 in range(n):
                i1 = (i0 + 1) % n
                cand = self._segment_project(x, y, i0, i1)
                if best is None or cand[0] < best[0]:
                    best = cand

        _, s_cl, e_y, seg_idx = best
        self.last_idx = seg_idx

        th = float(self.track._interp_pair(self.track.theta, s_cl))
        e_psi = wrap_pi(yaw - th)

        return float(s_cl), float(e_y), float(e_psi)

    def frenet_to_global(self, s: float, ey: float) -> Tuple[float, float]:
        if not self.has_xy:
            return s, ey
        xc = float(self.track._interp_pair(self.track.x, s))
        yc = float(self.track._interp_pair(self.track.y, s))
        th = float(self.track._interp_pair(self.track.theta, s))
        nx, ny = -math.sin(th), math.cos(th)
        return float(xc + nx * ey), float(yc + ny * ey)


class LMPCF1(Node):
    def __init__(self):
        super().__init__('lmpc_f1tenth')
        qos = QoSProfile(depth=10)

        # self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/columbia_small_centerline.csv')
        self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/columbia_small_centerline_smooth.csv')
        # Path to the track centerline CSV file
        # self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_centerline.csv')
        # Namespace for the node
        self.declare_parameter('namespace', '')
        # Time step for the control loop
        self.declare_parameter('dt', 0.1)

        # Prediction horizon for the MPC
        self.declare_parameter('N', 12)
        # Number of nearest neighbors for regression
        self.declare_parameter('K', 15)
        # self.declare_parameter('K', 20)
        # Number of neighbors for probability density estimation
        # self.declare_parameter('P_neighbors', 400)
        self.declare_parameter('P_neighbors', 80)

        # Bandwidth for kernel density estimation
        # self.declare_parameter('h_bandwidth', 80.0)
        self.declare_parameter('h_bandwidth', 10.0)


        # Longitudinal velocity bounds (min, max)
        self.declare_parameter('vx_bounds', [0.8, 4.0])
        # Maximum lateral velocity
        self.declare_parameter('vy_abs_max', 2.0)
        # Maximum yaw rate
        self.declare_parameter('wz_abs_max', 4.5)
        # Maximum heading error
        self.declare_parameter('e_psi_abs_max', 1.0)
        # Maximum lateral error
        self.declare_parameter('e_y_abs_max', 1.0)

        # Steering angle bounds (min, max)
        self.declare_parameter('delta_bounds', [-0.30, 0.30])
        # Acceleration bounds (min, max)
        self.declare_parameter('a_bounds', [-2.5, 2.5])
        # Maximum steering rate
        self.declare_parameter('delta_rate_max', 1.3)
        # Maximum acceleration rate
        self.declare_parameter('a_rate_max', 5.0)

        # Regularization weight for control input changes
        self.declare_parameter('rho_du', 0.8)
        # Regularization weight for steering angle
        self.declare_parameter('rho_delta', 1.0) # 0.25
        # Regularization weight for acceleration
        self.declare_parameter('rho_a', 10.0) # 0.08

        # Weight for lateral velocity in the cost function
        self.declare_parameter('q_vy', 1.0) # 0.2
        # Weight for yaw rate in the cost function
        self.declare_parameter('q_wz', 1.0) # 0.15
        # Weight for heading error in the cost function
        self.declare_parameter('q_epsi', 1.0) # 4.0
        # Weight for lateral error in the cost function
        self.declare_parameter('q_ey', 100.0) # 6

        # Trust factor for lateral error
        self.declare_parameter('trust_ey', 0.25)
        # Trust factor for heading error
        self.declare_parameter('trust_epsi', 0.2)
        # Trust factor for longitudinal velocity
        self.declare_parameter('trust_vx', 0.8)

        # Wheelbase of the vehicle
        self.declare_parameter('wheelbase', 0.33)
        # Sign for steering direction
        self.declare_parameter('steer_sign', 1.0)

        # Feedforward gain for control
        self.declare_parameter('ff_gain', 1.0)
        # Initial lateral gain for pure pursuit
        self.declare_parameter('ky_seed', 0.8)
        # Initial heading gain for pure pursuit
        self.declare_parameter('kpsi_seed', 0.6)

        # Enable seed laps for initialization
        self.declare_parameter('do_seed_laps', True)
        # Number of pure pursuit laps
        self.declare_parameter('pp_laps', 1)
        # Number of LTI MPC laps
        self.declare_parameter('lti_mpc_laps', 2)
        # Initial speed for seed laps
        self.declare_parameter('seed_speed', 1.5)

        # Lookahead distance for pure pursuit
        self.declare_parameter('pp_lookahead', 1.6)
        # Minimum lookahead distance
        self.declare_parameter('pp_min_lookahead', 0.8)
        # Maximum lookahead distance
        self.declare_parameter('pp_max_lookahead', 3.0)
        # Minimum speed for pure pursuit
        self.declare_parameter('pp_speed_min', 1.0)
        # Maximum speed for pure pursuit
        self.declare_parameter('pp_speed_max', 3.0)
        # Gain for lookahead adjustment based on speed
        self.declare_parameter('pp_k_lookahead', 0.35)

        # Margin for lateral error
        self.declare_parameter('ey_margin', 0.3)
        # Fraction of the lap for wrapping
        self.declare_parameter('lap_wrap_frac', 0.5)
        # Window size for wrapping in meters
        self.declare_parameter('wrap_window_m', 3.0)
        # Minimum steps per lap
        self.declare_parameter('min_lap_steps', 150)

        # Enable soft terminal constraints
        self.declare_parameter('soft_terminal', True)
        # Weight for terminal constraints
        self.declare_parameter('terminal_weight', 400.0)

        # Previous steering angle
        self.declare_parameter('prev_delta', 0.0)

        # Mode for memory management
        self.declare_parameter('memory_mode', 'buckets')
        # Number of bins for memory
        self.declare_parameter('s_bins', 220)
        # Number of samples to keep per bin
        self.declare_parameter('per_bin_keep', 15)
        # self.declare_parameter('per_bin_keep', 4)

        # Threshold for failure fraction
        self.declare_parameter('fail_frac_thresh', 0.08)
        # Threshold for wall collision fraction
        self.declare_parameter('wall_frac_thresh', 0.12)

        # Frame for visualization
        self.declare_parameter('viz_frame', 'map')
        # Namespace for visualization topics
        self.declare_parameter('viz_ns', 'mpc')

        # Maximum lateral acceleration
        self.declare_parameter('ay_max', 1.0)
        # Minimum speed for turning
        self.declare_parameter('v_turn_min', 1.0)

        # Fraction of previous lap for wrapping
        self.declare_parameter('wrap_prev_frac', 0.85)
        # Fraction of current lap for wrapping
        self.declare_parameter('wrap_now_frac', 0.15)
        # Tolerance for lateral error during wrapping
        self.declare_parameter('wrap_ey_tol', 0.5)
        # Tolerance for heading error during wrapping
        self.declare_parameter('wrap_epsi_tol', 0.6)

        # Center position for finish line
        self.declare_parameter('finish_s_center', 1.0)
        # Half-width of the finish line
        self.declare_parameter('finish_s_halfwidth', 1.5)
        # Minimum forward speed to finish
        self.declare_parameter('finish_min_forward_speed', 0.2)
        # Require valid pose to finish
        self.declare_parameter('finish_require_pose_ok', True)

        # Paper consisent LMPC additions
        self.declare_parameter('memory_laps', 2)               # use l=j-2 ... j-1 (Table I)
        self.declare_parameter('K_per_lap', 20)                # K neighbours per lap (Table I)
        self.declare_parameter('ridge_reg', 1e-3)              # stronger ridge for LS stability
        self.declare_parameter('ss_global_fallback_debug', False)
        self.declare_parameter('osqp_reuse', True)
        self.declare_parameter('finish_arm_s_min', 10.0)       # metres along s away from finish before arming

        # Curvature-based vx cap handling (feasibility controls)
        self.declare_parameter('vx_cap_enable', True)          # disable to remove cap from constraints
        self.declare_parameter('vx_cap_brake_feasible', True)  # reshape cap so braking can satisfy it
        self.declare_parameter('vx_cap_tol', 0.05)             # m/s tolerance for k=0 inclusion
        self.declare_parameter('vx_cap_soft', False)           # soft cap via slack variables
        self.declare_parameter('vx_cap_slack_weight', 2000.0)  # penalty on slack

        self.finish_latch = False
        self.finish_armed = False

        self.debug_counter = 0
        self.last_lap_time = None
        self.fallback_countdown = 0

        # Candidate terminal state and prediction buffers (paper eq. (10) and ATV candidate)
        self.z_t = None
        self.prev_x_pred = None   # shape (N+1,6) from last LMPC/LTV solve
        self.prev_u_pred = None   # shape (N,2)   from last LMPC/LTV solve
        # OSQP caching (optional performance patch)
        self._osqp_cache = {}     # key -> dict(prob, P_pattern, A_pattern)


        self.get_logger().info("LMPC READY")

        self.ns = str(self.get_parameter('namespace').value)
        self.dt = float(self.get_parameter('dt').value)

        self.N = int(self.get_parameter('N').value)
        self.K = int(self.get_parameter('K').value)
        self.Pn = int(self.get_parameter('P_neighbors').value)
        self.h = float(self.get_parameter('h_bandwidth').value)

        self.vx_min, self.vx_max = self.get_parameter('vx_bounds').value
        self.vy_abs_max = float(self.get_parameter('vy_abs_max').value)
        self.wz_abs_max = float(self.get_parameter('wz_abs_max').value)
        self.epsi_abs_max = float(self.get_parameter('e_psi_abs_max').value)
        self.ey_abs_max = float(self.get_parameter('e_y_abs_max').value)

        self.delta_min, self.delta_max = self.get_parameter('delta_bounds').value
        self.a_min, self.a_max = self.get_parameter('a_bounds').value
        self.delta_rate_max = float(self.get_parameter('delta_rate_max').value)
        self.a_rate_max = float(self.get_parameter('a_rate_max').value)

        self.rho_du = float(self.get_parameter('rho_du').value)
        self.rho_delta = float(self.get_parameter('rho_delta').value)
        self.rho_a = float(self.get_parameter('rho_a').value)

        self.q_vy = float(self.get_parameter('q_vy').value)
        self.q_wz = float(self.get_parameter('q_wz').value)
        self.q_epsi = float(self.get_parameter('q_epsi').value)
        self.q_ey = float(self.get_parameter('q_ey').value)

        self.trust_ey = float(self.get_parameter('trust_ey').value)
        self.trust_epsi = float(self.get_parameter('trust_epsi').value)
        self.trust_vx = float(self.get_parameter('trust_vx').value)

        self.L_wb = float(self.get_parameter('wheelbase').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)

        self.ff_gain = float(self.get_parameter('ff_gain').value)
        self.ky_seed = float(self.get_parameter('ky_seed').value)
        self.kpsi_seed = float(self.get_parameter('kpsi_seed').value)

        self.do_seed = bool(self.get_parameter('do_seed_laps').value)
        self.pp_laps = int(self.get_parameter('pp_laps').value)
        self.lti_mpc_laps = int(self.get_parameter('lti_mpc_laps').value)
        self.seed_speed = float(self.get_parameter('seed_speed').value)

        self.pp_lookahead = float(self.get_parameter('pp_lookahead').value)
        self.pp_min_lookahead = float(self.get_parameter('pp_min_lookahead').value)
        self.pp_max_lookahead = float(self.get_parameter('pp_max_lookahead').value)
        self.pp_speed_min = float(self.get_parameter('pp_speed_min').value)
        self.pp_speed_max = float(self.get_parameter('pp_speed_max').value)
        self.pp_k_lookahead = float(self.get_parameter('pp_k_lookahead').value)

        self.ey_margin = float(self.get_parameter('ey_margin').value)
        self.lap_wrap_frac = float(self.get_parameter('lap_wrap_frac').value)
        self.wrap_window = float(self.get_parameter('wrap_window_m').value)
        self.min_lap_steps = int(self.get_parameter('min_lap_steps').value)

        self.soft_terminal = bool(self.get_parameter('soft_terminal').value)
        self.terminal_weight = float(self.get_parameter('terminal_weight').value)

        self.prev_delta = float(self.get_parameter('prev_delta').value)

        self.memory_mode = str(self.get_parameter('memory_mode').value)
        self.s_bins = int(self.get_parameter('s_bins').value)
        self.per_bin_keep = int(self.get_parameter('per_bin_keep').value)

        self.fail_frac_thresh = float(self.get_parameter('fail_frac_thresh').value)
        self.wall_frac_thresh = float(self.get_parameter('wall_frac_thresh').value)

        self.viz_frame = str(self.get_parameter('viz_frame').value)
        self.viz_ns = str(self.get_parameter('viz_ns').value)

        self.ay_max = float(self.get_parameter('ay_max').value)
        self.v_turn_min = float(self.get_parameter('v_turn_min').value)

        self.wrap_prev_frac = float(self.get_parameter('wrap_prev_frac').value)
        self.wrap_now_frac = float(self.get_parameter('wrap_now_frac').value)
        self.wrap_ey_tol = float(self.get_parameter('wrap_ey_tol').value)
        self.wrap_epsi_tol = float(self.get_parameter('wrap_epsi_tol').value)

        self.finish_s_center = float(self.get_parameter('finish_s_center').value)
        self.finish_s_halfwidth = float(self.get_parameter('finish_s_halfwidth').value)
        self.finish_min_forward_speed = float(self.get_parameter('finish_min_forward_speed').value)
        self.finish_require_pose_ok = bool(self.get_parameter('finish_require_pose_ok').value)
        self.memory_laps = int(self.get_parameter('memory_laps').value)
        self.K_per_lap = int(self.get_parameter('K_per_lap').value)
        self.ridge_reg = float(self.get_parameter('ridge_reg').value)
        self.ss_global_fallback_debug = bool(self.get_parameter('ss_global_fallback_debug').value)
        self.osqp_reuse = bool(self.get_parameter('osqp_reuse').value)
        self.finish_arm_s_min = float(self.get_parameter('finish_arm_s_min').value)

        self.vx_cap_enable = bool(self.get_parameter('vx_cap_enable').value)
        self.vx_cap_brake_feasible = bool(self.get_parameter('vx_cap_brake_feasible').value)
        self.vx_cap_tol = float(self.get_parameter('vx_cap_tol').value)
        self.vx_cap_soft = bool(self.get_parameter('vx_cap_soft').value)
        self.vx_cap_slack_weight = float(self.get_parameter('vx_cap_slack_weight').value)

        self.track = self._load_track(str(self.get_parameter('track_csv').value))
        self.get_logger().info(
            f"[TRACK] L={self.track.L:.3f} | points={len(self.track.s)} | "
            f"s0={self.track.s[0]:.3f} | s_end={self.track.s[-1]:.3f}"
        )
        self.frenet = FrenetProjector(self.track)

        self.x_meas = np.zeros(6)
        self.have_pose = False

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0

        self.speed_cmd = 0.0
        self.u_prev = np.zeros(2)

        self.prev_s = None
        self.wrap_thresh = max(0.5, self.lap_wrap_frac * self.track.L)

        self.z_t = None
        self.prev_osqp_x = None
        self.prev_osqp_y = None

        self.laps: List[dict] = []
        self.curr_traj_x: List[np.ndarray] = []
        self.curr_traj_u: List[np.ndarray] = []

        self.buckets: Dict[int, List[dict]] = {i: [] for i in range(self.s_bins)}

        self.qp_fail_in_lap = 0
        self.near_wall_in_lap = 0
        self.best_lap_time = None

        # self.Q_knn = np.diag([0.15, 0.6, 0.5, 1.0, 0.0, 1.2])
        self.Q_knn = np.diag([0.1, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.Q_ss = np.diag([0.20, 0.20, 0.10, 1.5, 0.0, 2.0])

        topic_cmd = f'{self.ns}/drive' if self.ns else 'drive'
        topic_odom = f'{self.ns}/odom' if self.ns else 'odom'

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, topic_cmd, qos)
        self.sub_odom = self.create_subscription(Odometry, topic_odom, self.odom_cb, qos)

        self.safe_set_pub = self.create_publisher(Marker, "/safe_set", 1)
        self.traj_pub = self.create_publisher(Marker, "/curr_traj", 1)

        self.add_on_set_parameters_callback(self._on_param_change)
        self.timer = self.create_timer(self.dt, self.control_loop)

    def _init_z_from_last_lap(self) -> Optional[np.ndarray]:
        """Paper eq. (10) initialisation: z0^j ≈ x_N^{j-1} from previous trajectory."""
        if len(self.laps) < 1:
            return None
        X = self.laps[-1]['x']
        idx = int(min(self.N, X.shape[0] - 1))
        z = X[idx].copy()
        # keep s in [0,L) (your state uses modulo s)
        z[4] = float(z[4] % self.track.L)
        return z

    def _on_param_change(self, params):
        try:
            for p in params:
                if p.name == 'rho_du':
                    self.rho_du = float(p.value)
                elif p.name == 'rho_delta':
                    self.rho_delta = float(p.value)
                elif p.name == 'rho_a':
                    self.rho_a = float(p.value)
                elif p.name == 'q_vy':
                    self.q_vy = float(p.value)
                elif p.name == 'q_wz':
                    self.q_wz = float(p.value)
                elif p.name == 'q_epsi':
                    self.q_epsi = float(p.value)
                elif p.name == 'q_ey':
                    self.q_ey = float(p.value)
                elif p.name == 'trust_ey':
                    self.trust_ey = float(p.value)
                elif p.name == 'trust_epsi':
                    self.trust_epsi = float(p.value)
                elif p.name == 'trust_vx':
                    self.trust_vx = float(p.value)
                elif p.name == 'seed_laps':
                    self.seed_laps = int(p.value)
                elif p.name == 'pp_laps':
                    self.pp_laps = int(p.value)
                elif p.name == 'lti_mpc_laps':
                    self.lti_mpc_laps = int(p.value)
                elif p.name == 'pp_lookahead':
                    self.pp_lookahead = float(p.value)
            return SetParametersResult(successful=True)
        except Exception:
            return SetParametersResult(successful=False)

    def _load_track(self, path: str) -> Track:
        df = pd.read_csv(path)

        # COLUMBIA SMALL ENABLE
        df = df.iloc[::-1].reset_index(drop=True) 

        def canon(name: str) -> str:
            return name.strip().lstrip('#').strip().lower()

        cols = {canon(c): c for c in df.columns}

        def get_any(names):
            for n in names:
                if n in cols:
                    return df[cols[n]].to_numpy(dtype=float)
            return None

        s = get_any(['s', 's_m', 'arc_length', 'arclength'])
        x = get_any(['x', 'x_m'])
        y = get_any(['y', 'y_m'])
        theta = get_any(['theta', 'theta_rad', 'heading'])
        kappa = get_any(['kappa', 'curvature', 'k'])
        w_left = get_any(['w_tr_left', 'w_tr_left_m', 'w_left', 'width_left'])
        w_right = get_any(['w_tr_right', 'w_tr_right_m', 'w_right', 'width_right'])

        if s is None and x is not None and y is not None:
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            s = np.cumsum(np.hypot(dx, dy))
            s -= s[0]

        if theta is None and x is not None and y is not None:
            gx = np.gradient(x)
            gy = np.gradient(y)
            theta = np.unwrap(np.arctan2(gy, gx))

        if kappa is None and theta is not None and s is not None:
            dth = np.gradient(theta)
            ds = np.gradient(s)
            ds[np.abs(ds) < 1e-6] = 1e-6
            kappa = dth / ds

        order = np.argsort(s)
        s = s[order] - s[order][0]
        kappa = kappa[order]
        x = x[order] if x is not None else None
        y = y[order] if y is not None else None
        theta = theta[order] if theta is not None else None
        if w_left is not None:
            w_left = w_left[order]
        if w_right is not None:
            w_right = w_right[order]

        return Track(
            s=s.astype(float),
            kappa=kappa.astype(float),
            x=x.astype(float) if x is not None else None,
            y=y.astype(float) if y is not None else None,
            theta=theta.astype(float) if theta is not None else None,
            w_left=w_left.astype(float) if w_left is not None else None,
            w_right=w_right.astype(float) if w_right is not None else None
        )

    def odom_cb(self, msg: Odometry):
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        s_cur, e_y, e_psi = self.frenet.project(px, py, yaw)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)

        self.pose_x = px
        self.pose_y = py
        self.pose_yaw = yaw

        self.x_meas = np.array([vx, vy, wz, e_psi, s_cur, e_y], dtype=float)
        self.have_pose = True

    def epanechnikov(self, u: float) -> float:
        return 0.75 * (1.0 - u * u) if abs(u) < 1.0 else 0.0
    
    def wrapped_s_error(self, s_a: float, s_b: float) -> float:
        L = self.track.L
        return ((s_a - s_b + 0.5 * L) % L) - 0.5 * L

    def lti_matrices(self, x_ref: np.ndarray, A_dyn: np.ndarray, B_dyn: np.ndarray, c_dyn: np.ndarray):
        vx, vy, wz, epsi, s, ey = x_ref

        kappa = self.track.kappa_at(s)
        den = max(1e-3, 1.0 - kappa * ey)
        cos_e = math.cos(epsi)
        sin_e = math.sin(epsi)

        f_epsi = wz - ((vx * cos_e - vy * sin_e) / den) * kappa
        f_s = (vx * cos_e - vy * sin_e) / den
        f_ey = vx * math.sin(epsi) + vy * math.cos(epsi)

        dfepsi = np.zeros(6)
        dfepsi[0] = -(cos_e / den) * kappa
        dfepsi[1] = (sin_e / den) * kappa
        dfepsi[2] = 1.0
        dfepsi[3] = -((-vx * sin_e - vy * cos_e) / den) * kappa
        dfepsi[5] = -((vx * cos_e - vy * sin_e) * kappa * kappa) / (den * den)

        dfs = np.zeros(6)
        dfs[0] = cos_e / den
        dfs[1] = -sin_e / den
        dfs[3] = (-vx * sin_e - vy * cos_e) / den
        dfs[5] = (vx * cos_e - vy * sin_e) * kappa / (den * den)

        dfey = np.zeros(6)
        dfey[0] = math.sin(epsi)
        dfey[1] = math.cos(epsi)
        dfey[3] = vx * math.cos(epsi) - vy * math.sin(epsi)

        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        C = np.zeros(6)

        # learned global LTI dynamics
        A[0:3, 0:3] = A_dyn
        B[0:3, :] = B_dyn
        C[0:3] = c_dyn

        # Frenet kinematics linearized around x_ref
        A[3, :] = np.eye(6)[3, :] + self.dt * dfepsi
        A[4, :] = np.eye(6)[4, :] + self.dt * dfs
        A[5, :] = np.eye(6)[5, :] + self.dt * dfey

        C[3] = self.dt * (f_epsi - dfepsi @ x_ref)
        C[4] = self.dt * (f_s - dfs @ x_ref)
        C[5] = self.dt * (f_ey - dfey @ x_ref)

        return A, B, C

    def make_vx_caps_braking_feasible(self, vx_caps: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Ensure vx_caps[k] is not lower than what is reachable under:
          - a_min (max braking)
          - a_rate_max (ramp limit)
          - dt
        This prevents future-horizon infeasibility when caps drop too fast.
        """
        if vx_caps is None:
            return None
        vx_caps = np.array(vx_caps, dtype=float).copy()

        a = float(self.u_prev[1])         # previously applied acceleration command
        vx = float(x0[0])                 # current measured speed

        # k=0: never demand vx below current measurement (within tolerance)
        vx_caps[0] = max(vx_caps[0], vx + float(self.vx_cap_tol))

        for k in range(1, len(vx_caps)):
            # ramp-limited braking toward a_min
            a = max(self.a_min, a - self.a_rate_max * self.dt)
            vx = max(self.vx_min, vx + a * self.dt)
            # raise cap if it is below reachable speed
            vx_caps[k] = max(vx_caps[k], vx)

        return vx_caps

    def pure_pursuit_control(self, x0: np.ndarray) -> Tuple[float, float]:
        vx = float(x0[0])
        s = float(x0[4])

        lookahead = self.pp_lookahead + self.pp_k_lookahead * max(0.0, vx)
        lookahead = sat(lookahead, self.pp_min_lookahead, self.pp_max_lookahead)

        s_target = (s + lookahead) % self.track.L

        xt = float(self.track._interp_pair(self.track.x, s_target))
        yt = float(self.track._interp_pair(self.track.y, s_target))

        dx = xt - self.pose_x
        dy = yt - self.pose_y

        alpha = wrap_pi(math.atan2(dy, dx) - self.pose_yaw)
        Ld = max(0.4, math.hypot(dx, dy))

        delta = math.atan2(2.0 * self.L_wb * math.sin(alpha), Ld)
        delta = sat(self.steer_sign * delta, self.delta_min, self.delta_max)

        kap = abs(self.track.kappa_at(s))
        if kap < 1e-4:
            v_target = self.pp_speed_max
        else:
            v_target = min(self.pp_speed_max, math.sqrt(max(0.25, self.ay_max / kap)))
        v_target = max(self.pp_speed_min, v_target)

        a = sat((v_target - self.speed_cmd) / self.dt, self.a_min, self.a_max)
        return float(delta), float(a)
    
    def build_tracking_reference(self, x0: np.ndarray):
        x_ref_seq = np.zeros((self.N + 1, 6))

        s_ref = float(x0[4])

        for k in range(self.N + 1):
            kap = abs(self.track.kappa_at(s_ref))
            if kap < 1e-4:
                vref = self.pp_speed_max
            else:
                vref = min(self.pp_speed_max, math.sqrt(max(0.25, self.ay_max / kap)))
            vref = max(self.pp_speed_min, vref)

            x_ref_seq[k, 0] = vref
            x_ref_seq[k, 1] = 0.0
            x_ref_seq[k, 2] = 0.0
            x_ref_seq[k, 3] = 0.0
            x_ref_seq[k, 4] = s_ref
            x_ref_seq[k, 5] = 0.0

            s_ref = (s_ref + vref * self.dt) % self.track.L

        return x_ref_seq

    def _collect_recent_samples_for_regression(self, purpose: str = "ATV"):
        """
        purpose="LTI": use 1 PP lap + last 2 LTI laps (bootstrap fit)
        purpose="ATV": use last `memory_laps` non-PP laps if available (paper l=j-2)
        """
        if len(self.laps) < 1:
            return np.empty((0, 6)), np.empty((0, 2)), np.empty((0, 6))

        laps = self.laps
        pp = [lap for lap in laps if lap.get('mode', 'PP') == 'PP']
        lti = [lap for lap in laps if lap.get('mode', '') == 'LTI_MPC']
        non_pp = [lap for lap in laps if lap.get('mode', 'PP') != 'PP']

        chosen = []
        if purpose.upper() == "LTI":
            # Keep at most 1 PP lap for seed, then last 2 LTI laps
            if len(pp) > 0:
                chosen.append(pp[0])
            if len(lti) > 0:
                chosen.extend(lti[-2:])
            if len(chosen) == 0:
                chosen = laps[-1:]
        else:
            # ATV/LMPC: prefer last `memory_laps` non-PP laps; fallback to last laps
            if len(non_pp) > 0:
                chosen = non_pp[-self.memory_laps:]
            else:
                chosen = laps[-self.memory_laps:]

        X, U, Xn = [], [], []
        for lap in chosen:
            X.append(lap['x'][:-1])
            U.append(lap['u'])
            Xn.append(lap['x'][1:])

        return np.vstack(X), np.vstack(U), np.vstack(Xn)

    def fit_lti_model_from_laps(self):
        X, U, Xn = self._collect_recent_samples_for_regression(purpose="LTI")

        if X.shape[0] < 30:
            return None, None, None

        # affine model:
        # [vx_{k+1}, vy_{k+1}, wz_{k+1}] = A_dyn [vx,vy,wz] + B_dyn [delta,a] + c_dyn
        #
        # error dynamics propagated with kinematic equations around x_ref online:
        # epsi_{k+1}, s_{k+1}, ey_{k+1}

        Phi = np.column_stack([
            X[:, 0],   # vx
            X[:, 1],   # vy
            X[:, 2],   # wz
            U[:, 0],   # delta
            U[:, 1],   # a
            np.ones(X.shape[0])
        ])

        Y = np.column_stack([
            Xn[:, 0],  # vx+
            Xn[:, 1],  # vy+
            Xn[:, 2],  # wz+
        ])

        # regularized least squares
        reg = self.ridge_reg * np.eye(Phi.shape[1])
        Theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)   # (6 x 3)

        # unpack
        # each output uses: [vx, vy, wz, delta, a, 1]
        theta_vx = Theta[:, 0]
        theta_vy = Theta[:, 1]
        theta_wz = Theta[:, 2]

        A_dyn = np.zeros((3, 3))
        B_dyn = np.zeros((3, 2))
        c_dyn = np.zeros(3)

        A_dyn[0, :] = theta_vx[0:3]
        A_dyn[1, :] = theta_vy[0:3]
        A_dyn[2, :] = theta_wz[0:3]

        B_dyn[0, :] = theta_vx[3:5]
        B_dyn[1, :] = theta_vy[3:5]
        B_dyn[2, :] = theta_wz[3:5]

        c_dyn[0] = theta_vx[5]
        c_dyn[1] = theta_vy[5]
        c_dyn[2] = theta_wz[5]

        return A_dyn, B_dyn, c_dyn

    def build_regression_sets(self, x_ref: np.ndarray):
        X, U, Xn = self._collect_recent_samples_for_regression(purpose="ATV")
        if X.shape[0] == 0:
            return (
                np.empty((0, 5)), np.empty((0,)),
                np.empty((0, 5)), np.empty((0,)),
                np.empty((0, 5)), np.empty((0,)),
                np.empty((0, 6))
            )

        dif = X - x_ref.reshape(1, -1)
        d2 = np.einsum('ij,jk,ik->i', dif, self.Q_knn, dif)
        idx = np.argsort(np.sqrt(np.maximum(0.0, d2)))[:min(self.Pn, len(d2))]

        Xp = X[idx]
        Up = U[idx]
        Xp1 = Xn[idx]

        Phi_vx = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 1], np.ones(len(idx))])
        y_vx = Xp1[:, 0]

        Phi_vy = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 0], np.ones(len(idx))])
        y_vy = Xp1[:, 1]

        Phi_wz = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 0], np.ones(len(idx))])
        y_wz = Xp1[:, 2]

        return Phi_vx, y_vx, Phi_vy, y_vy, Phi_wz, y_wz, Xp

    def weighted_ls(self, Phi: np.ndarray, y: np.ndarray, x_ref: np.ndarray, X_samples: np.ndarray) -> np.ndarray:
        if Phi.shape[0] == 0:
            return np.zeros(Phi.shape[1])

        dif = X_samples - x_ref.reshape(1, -1)
        dist2 = np.einsum('ij,jk,ik->i', dif, self.Q_knn, dif)
        dist = np.sqrt(np.maximum(0.0, dist2))
        uu = dist / max(1e-6, self.h)

        w = np.array([self.epanechnikov(ui) for ui in uu], dtype=float) + 1e-6
        W = np.diag(w)

        H = Phi.T @ W @ Phi + self.ridge_reg * np.eye(Phi.shape[1])
        b = Phi.T @ W @ y

        return np.linalg.solve(H, b)

    def atv_matrices(self, xbar_seq: np.ndarray):
        A_list, B_list, C_list = [], [], []

        if self.debug_counter % 10 == 0:
            self.get_logger().info(f"[ATVDBG] Pn={self.Pn} h={self.h:.1f} ridge={1e-6}")

        for k in range(self.N):
            xref = xbar_seq[k]

            Phi_vx, y_vx, Phi_vy, y_vy, Phi_wz, y_wz, Xp = self.build_regression_sets(xref)

            gamma_vx = self.weighted_ls(Phi_vx, y_vx, xref, Xp) if Phi_vx.size else np.zeros(5)
            gamma_vy = self.weighted_ls(Phi_vy, y_vy, xref, Xp) if Phi_vy.size else np.zeros(5)
            gamma_wz = self.weighted_ls(Phi_wz, y_wz, xref, Xp) if Phi_wz.size else np.zeros(5)

            vx, vy, wz, epsi, s, ey = xref
            kappa = self.track.kappa_at(s)
            den = max(1e-3, (1.0 - kappa * ey))
            cos_e = math.cos(epsi)
            sin_e = math.sin(epsi)

            f_epsi = wz - ((vx * cos_e - vy * sin_e) / den) * kappa
            f_s = (vx * cos_e - vy * sin_e) / den
            f_ey = vx * math.sin(epsi) + vy * math.cos(epsi)

            dfepsi = np.zeros(6)
            dfepsi[0] = -(cos_e / den) * kappa
            dfepsi[1] = (sin_e / den) * kappa
            dfepsi[2] = 1.0
            dfepsi[3] = -((-vx * sin_e - vy * cos_e) / den) * kappa
            dfepsi[5] = -((vx * cos_e - vy * sin_e) * kappa * kappa) / (den * den)

            dfs = np.zeros(6)
            dfs[0] = cos_e / den
            dfs[1] = -sin_e / den
            dfs[3] = (-vx * sin_e - vy * cos_e) / den
            dfs[5] = (vx * cos_e - vy * sin_e) * kappa / (den * den)

            dfey = np.zeros(6)
            dfey[0] = math.sin(epsi)
            dfey[1] = math.cos(epsi)
            dfey[3] = vx * math.cos(epsi) - vy * math.sin(epsi)

            A = np.zeros((6, 6))
            B = np.zeros((6, 2))
            C = np.zeros(6)

            A[0, 0:3] = gamma_vx[0:3]
            A[1, 0:3] = gamma_vy[0:3]
            A[2, 0:3] = gamma_wz[0:3]

            B[0, 1] = gamma_vx[3]  
            B[1, 0] = gamma_vy[3]  
            B[2, 0] = gamma_wz[3]  

            C[0] = gamma_vx[4]
            C[1] = gamma_vy[4]
            C[2] = gamma_wz[4]

            A[3, :] = np.eye(6)[3, :] + self.dt * dfepsi
            A[4, :] = np.eye(6)[4, :] + self.dt * dfs
            A[5, :] = np.eye(6)[5, :] + self.dt * dfey

            C[3] = self.dt * (f_epsi - dfepsi @ xref)
            C[4] = self.dt * (f_s - dfs @ xref)
            C[5] = self.dt * (f_ey - dfey @ xref)

            A_list.append(A)
            B_list.append(B)
            C_list.append(C)

        return A_list, B_list, C_list

    def build_local_safe_set(self, x_query: np.ndarray):
        if not any(self.buckets.values()):
            return np.zeros((6, 0)), np.zeros((6, 0)), np.zeros((0,)), np.zeros((0,), dtype=int)

        # use last `memory_laps` laps (paper l=j-2 ... j-1)
        if len(self.laps) == 0:
            return np.zeros((6, 0)), np.zeros((6, 0)), np.zeros((0,)), np.zeros((0,), dtype=int)
        lap_hi = len(self.laps) - 1
        lap_lo = max(0, lap_hi - (self.memory_laps - 1))
        use_laps = list(range(lap_lo, lap_hi + 1))

        ds_bin = self.track.L / max(1, self.s_bins)
        s_q = float(x_query[4] % self.track.L)
        qbin = int(math.floor(s_q / ds_bin)) % self.s_bins

        # collect local candidates (±6 bins) then select K_per_lap per lap by |Δs|
        D_cols, S_cols, J_list = [], [], []
        for lap_idx in use_laps:
            picks = []
            for off in range(-6, 7):
                picks.extend([it for it in self.buckets[(qbin + off) % self.s_bins] if it.get('lap_idx', -1) == lap_idx])

            # debug-only global fallback
            if (not picks) and self.ss_global_fallback_debug:
                for b in self.buckets.values():
                    picks.extend([it for it in b if it.get('lap_idx', -1) == lap_idx])

            if not picks:
                continue

            # s-only distance (paper D=diag(0,0,0,0,1,0))
            ds = np.array([abs(self.wrapped_s_error(it['s'], s_q)) for it in picks], dtype=float)
            order = np.argsort(ds)[:min(self.K_per_lap, len(ds))]

            sel = [picks[i] for i in order]
            # pad to fixed K_per_lap to keep m constant for solver reuse
            while len(sel) < self.K_per_lap:
                sel.append(sel[-1])

            for it in sel:
                D_cols.append(it['x'])
                S_cols.append(it['x_next'])
                J_list.append(float(it['J']))

        if len(D_cols) == 0:
            return np.zeros((6, 0)), np.zeros((6, 0)), np.zeros((0,)), np.zeros((0,), dtype=int)

        D = np.stack(D_cols, axis=1)
        S = np.stack(S_cols, axis=1)
        J = np.array(J_list, dtype=float)


        # if self.debug_counter % 50 == 0:
        #     self.get_logger().info(f"[SSDBG] safe-set size={self.m} using laps={[i for i in range(len(self.laps)-2,len(self.laps)) if i>=0]}")

        return D, S, J, np.zeros(D.shape[1], dtype=int)


    def update_z(self, x_query: np.ndarray, lambda_prev: np.ndarray) -> np.ndarray:
        _, S, _, _ = self.build_local_safe_set(x_query)
        return x_query.copy() if S.shape[1] == 0 else S @ lambda_prev

    def osqp_mats(self, A_list, B_list, C_list, x0, ss_query, ey_low_seq=None, ey_high_seq=None, vx_high_seq=None, xbar_seq=None):
        n_x, n_u, N = 6, 2, self.N

        Dmat, _, Jvec, _ = self.build_local_safe_set(ss_query)
        m = Dmat.shape[1]

        # if self.debug_counter % 20 == 0:
        #     total_bucket_points = sum(len(b) for b in self.buckets.values())
        #     self.get_logger().info(
        #         f"Safe set size: {m} | total memory points: {total_bucket_points} | stored laps: {len(self.laps)}"
        #     )

        if m == 0:
            self.get_logger().warn("⚠️ SAFE SET EMPTY → fallback to current state")
            Dmat = x0.reshape(-1, 1)
            Jvec = np.array([0.0], dtype=float)
            m = 1

        Xn = (N + 1) * n_x
        Un = N * n_u
        En = n_x if self.soft_terminal else 0
        Zn_base = Xn + Un + m + En

        use_vx_cap = (vx_high_seq is not None) and bool(getattr(self, "vx_cap_enable", True))
        use_soft_vx_cap = use_vx_cap and bool(getattr(self, "vx_cap_soft", False))
        Ns = (N + 1) if use_soft_vx_cap else 0
        Zn = Zn_base + Ns

        P = sparse.lil_matrix((Zn, Zn))
        q = np.zeros(Zn)

        q[Xn + Un: Xn + Un + m] = Jvec

        Qx = np.diag([0.0, self.q_vy, self.q_wz, self.q_epsi, 0.0, self.q_ey])
        for k in range(N):
            P[k*n_x:(k+1)*n_x, k*n_x:(k+1)*n_x] += sparse.csc_matrix(Qx)

        Ru = np.diag([self.rho_delta, self.rho_a])
        for k in range(N):
            P[Xn + k*n_u:Xn + (k+1)*n_u, Xn + k*n_u:Xn + (k+1)*n_u] += sparse.csc_matrix(Ru)

        Du = np.zeros((N, Un))
        for k in range(N):
            if k == 0:
                Du[k, 0:n_u] = 1.0
            else:
                Du[k, k*n_u:(k+1)*n_u] = 1.0
                Du[k, (k-1)*n_u:k*n_u] = -1.0

        P[Xn:Xn + Un, Xn:Xn + Un] += sparse.csc_matrix(self.rho_du * (Du.T @ Du))

        if self.soft_terminal and En > 0:
            P[Xn + Un + m:Xn + Un + m + En, Xn + Un + m:Xn + Un + m + En] = sparse.eye(En) * self.terminal_weight

        if use_soft_vx_cap:
            sigma0 = Zn_base
            P[sigma0:sigma0+Ns, sigma0:sigma0+Ns] = sparse.eye(Ns) * float(self.vx_cap_slack_weight)

        rows, cols, data = [], [], []
        lvec, uvec = [], []

        def add_row(entries, low, high):
            r = len(lvec)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                data.append(v)
            lvec.append(low)
            uvec.append(high)

        for i in range(n_x):
            add_row([(i, 1.0)], x0[i], x0[i])

        for k in range(N):
            for i in range(n_x):
                row = [((k + 1) * n_x + i, 1.0)]

                for jx in range(n_x):
                    val = -A_list[k][i, jx]
                    if val != 0.0:
                        row.append((k * n_x + jx, val))

                for ju in range(n_u):
                    val = -B_list[k][i, ju]
                    if val != 0.0:
                        row.append((Xn + k * n_u + ju, val))

                add_row(row, C_list[k][i], C_list[k][i])

        for i in range(n_x):
            row = [(N * n_x + i, 1.0)]
            for jcol in range(m):
                val = -Dmat[i, jcol]
                if val != 0.0:
                    row.append((Xn + Un + jcol, val))
            if self.soft_terminal and En > 0:
                row.append((Xn + Un + m + i, -1.0))
            add_row(row, 0.0, 0.0)

        add_row([(Xn + Un + j, 1.0) for j in range(m)], 1.0, 1.0)
        for j in range(m):
            add_row([(Xn + Un + j, 1.0)], 0.0, np.inf)

        for k in range(N + 1):
            vx_cap_k = float(vx_high_seq[k]) if use_vx_cap else self.vx_max
            vx_hi_hard = self.vx_max if use_soft_vx_cap else max(self.vx_min, vx_cap_k)

            vx_lo_hard = self.vx_min
            if k == 0 and use_vx_cap:
                vx_hi_hard = max(vx_hi_hard, float(x0[0]) + self.vx_cap_tol)
                vx_lo_hard = min(vx_lo_hard, float(x0[0]) - self.vx_cap_tol)

            add_row([(k * n_x + 0, 1.0)], vx_lo_hard, vx_hi_hard)
            add_row([(k * n_x + 1, 1.0)], -self.vy_abs_max, self.vy_abs_max)
            add_row([(k * n_x + 2, 1.0)], -self.wz_abs_max, self.wz_abs_max)
            add_row([(k * n_x + 3, 1.0)], -self.epsi_abs_max, self.epsi_abs_max)

        if use_soft_vx_cap:
            sigma0 = Zn_base
            for k in range(N + 1):
                vx_cap_k = float(vx_high_seq[k])

                # k=0 inclusion to avoid immediate contradiction with x(0)=x0
                if k == 0:
                    vx_cap_k = max(vx_cap_k, float(x0[0]) + self.vx_cap_tol)

                sigk = sigma0 + k

                # vx_k - sigma_k <= vx_cap_k
                add_row([(k * n_x + 0, 1.0), (sigk, -1.0)], -np.inf, vx_cap_k)

                # sigma_k >= 0
                add_row([(sigk, 1.0)], 0.0, np.inf)

        dmax = self.delta_rate_max * self.dt
        amax = self.a_rate_max * self.dt

        add_row([(Xn + 0, 1.0)], self.u_prev[0] - dmax, self.u_prev[0] + dmax)
        add_row([(Xn + 1, 1.0)], self.u_prev[1] - amax, self.u_prev[1] + amax)

        for k in range(1, N):
            add_row(
                [(Xn + k * n_u + 0, 1.0), (Xn + (k - 1) * n_u + 0, -1.0)],
                -dmax, dmax
            )
            add_row(
                [(Xn + k * n_u + 1, 1.0), (Xn + (k - 1) * n_u + 1, -1.0)],
                -amax, amax
            )

        Aqp = sparse.csc_matrix((data, (rows, cols)), shape=(len(lvec), Zn))
        return P.tocsc(), q, Aqp, np.array(lvec), np.array(uvec), m

    def lti_mpc_mats(self, A, B, C, x0, x_ref_seq, ey_low_seq=None, ey_high_seq=None, vx_high_seq=None):
        n_x, n_u, N = 6, 2, self.N

        Xn = (N + 1) * n_x
        Un = N * n_u
        Zn_base = Xn + Un

        use_vx_cap = (vx_high_seq is not None) and bool(getattr(self, "vx_cap_enable", True))
        use_soft_vx_cap = use_vx_cap and bool(getattr(self, "vx_cap_soft", False))
        Ns = (N + 1) if use_soft_vx_cap else 0
        Zn = Zn_base + Ns

        P = sparse.lil_matrix((Zn, Zn))
        q = np.zeros(Zn)

        Qx = np.diag([1.0, 1.0, 0.5, 8.0, 0.0, 12.0])
        QN = np.diag([1.0, 1.0, 0.5, 10.0, 0.0, 15.0])
        Ru = np.diag([2.0, 0.5])

        # state tracking cost
        for k in range(N):
            idx = k * n_x
            P[idx:idx+n_x, idx:idx+n_x] += Qx
            q[idx:idx+n_x] += -Qx @ x_ref_seq[k]

        idxN = N * n_x
        P[idxN:idxN+n_x, idxN:idxN+n_x] += QN
        q[idxN:idxN+n_x] += -QN @ x_ref_seq[N]

        # input cost
        for k in range(N):
            uid = Xn + k * n_u
            P[uid:uid+n_u, uid:uid+n_u] += Ru

        
        # soft vx-cap slack penalty (diagonal)
        if use_soft_vx_cap:
            sigma0 = Zn_base
            P[sigma0:sigma0+Ns, sigma0:sigma0+Ns] += sparse.eye(Ns) * float(self.vx_cap_slack_weight)

        rows, cols, data = [], [], []
        lvec, uvec = [], []

        def add_row(entries, low, high):
            r = len(lvec)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                data.append(v)
            lvec.append(low)
            uvec.append(high)

        # initial state
        for i in range(n_x):
            add_row([(i, 1.0)], x0[i], x0[i])

        # dynamics
        for k in range(N):
            for i in range(n_x):
                row = [((k + 1) * n_x + i, 1.0)]

                for j in range(n_x):
                    val = -A[i, j]
                    if val != 0.0:
                        row.append((k * n_x + j, val))

                for j in range(n_u):
                    val = -B[i, j]
                    if val != 0.0:
                        row.append((Xn + k * n_u + j, val))

                add_row(row, C[i], C[i])

        # state bounds
        for k in range(N + 1):
            vx_cap_k = float(vx_high_seq[k]) if use_vx_cap else self.vx_max

            # Hard vx upper bound: if soft-cap enabled, keep only vx_max hard,
            # and enforce vx_cap_k via separate soft constraints.
            vx_hi_hard = self.vx_max if use_soft_vx_cap else max(self.vx_min, vx_cap_k)

            # k=0 inclusion: always include measured vx to avoid infeasible x0 equality vs bound
            vx_lo_hard = self.vx_min
            if k == 0 and use_vx_cap:
                vx_hi_hard = max(vx_hi_hard, float(x0[0]) + self.vx_cap_tol)
                vx_lo_hard = min(vx_lo_hard, float(x0[0]) - self.vx_cap_tol)

            add_row([(k * n_x + 0, 1.0)], vx_lo_hard, vx_hi_hard)
            add_row([(k * n_x + 1, 1.0)], -self.vy_abs_max, self.vy_abs_max)
            add_row([(k * n_x + 2, 1.0)], -self.wz_abs_max, self.wz_abs_max)
            add_row([(k * n_x + 3, 1.0)], -self.epsi_abs_max, self.epsi_abs_max)

            if ey_low_seq is not None and ey_high_seq is not None:
                add_row([(k * n_x + 5, 1.0)], float(ey_low_seq[k]), float(ey_high_seq[k]))
            else:
                add_row([(k * n_x + 5, 1.0)], -self.ey_abs_max, self.ey_abs_max)


        # Soft vx-cap constraints: vx_k - sigma_k <= vx_cap_k, sigma_k >= 0
        if use_soft_vx_cap:
            sigma0 = Zn_base
            for k in range(N + 1):
                vx_cap_k = float(vx_high_seq[k])
                if k == 0:
                    vx_cap_k = max(vx_cap_k, float(x0[0]) + self.vx_cap_tol)
                sigk = sigma0 + k
                add_row([(k * n_x + 0, 1.0), (sigk, -1.0)], -np.inf, vx_cap_k)
                add_row([(sigk, 1.0)], 0.0, np.inf)

        # input bounds
        for k in range(N):
            add_row([(Xn + k * n_u + 0, 1.0)], self.delta_min, self.delta_max)
            add_row([(Xn + k * n_u + 1, 1.0)], self.a_min, self.a_max)

        # input rate bounds
        dmax = self.delta_rate_max * self.dt
        amax = self.a_rate_max * self.dt

        add_row([(Xn + 0, 1.0)], self.u_prev[0] - dmax, self.u_prev[0] + dmax)
        add_row([(Xn + 1, 1.0)], self.u_prev[1] - amax, self.u_prev[1] + amax)

        for k in range(1, N):
            add_row(
                [(Xn + k * n_u + 0, 1.0), (Xn + (k - 1) * n_u + 0, -1.0)],
                -dmax, dmax
            )
            add_row(
                [(Xn + k * n_u + 1, 1.0), (Xn + (k - 1) * n_u + 1, -1.0)],
                -amax, amax
            )

        Aqp = sparse.csc_matrix((data, (rows, cols)), shape=(len(lvec), Zn))
        return P.tocsc(), q, Aqp, np.array(lvec), np.array(uvec)

    def solve_qp(self, P, q, A, l, u, key: str = "GEN"):
        # enforce CSC and upper-triangular P for OSQP
        P = sparse.triu(P).tocsc()
        A = A.tocsc()

        if not self.osqp_reuse:
            prob = osqp.OSQP()
            prob.setup(P=P, q=q, A=A, l=l, u=u,
                       verbose=False, warm_start=True,
                       eps_abs=1e-3, eps_rel=1e-3,
                       max_iter=20000, polish=True)
            res = prob.solve()
        else:
            cache = self._osqp_cache.get(key)
            P_pat = (P.indptr.tobytes(), P.indices.tobytes())
            A_pat = (A.indptr.tobytes(), A.indices.tobytes())

            if cache is None or cache['P_pat'] != P_pat or cache['A_pat'] != A_pat:
                prob = osqp.OSQP()
                prob.setup(P=P, q=q, A=A, l=l, u=u,
                           verbose=False, warm_start=True,
                           eps_abs=1e-3, eps_rel=1e-3,
                           max_iter=20000, polish=True)
                self._osqp_cache[key] = {'prob': prob, 'P_pat': P_pat, 'A_pat': A_pat}
            else:
                prob = cache['prob']
                # update values only (pattern must match)
                prob.update(Px=P.data, Ax=A.data, q=q, l=l, u=u)
            res = prob.solve()

        if res.info.status_val not in (1, 2):
            self.qp_fail_in_lap += 1
            self.fallback_countdown = 8
            self.get_logger().warn(f"OSQP status: {res.info.status}. Using fallback control.")
            return None

        self.prev_osqp_x = res.x
        self.prev_osqp_y = res.y
        return res.x

    def extract_first_control_lti(self, sol):
        n_x, n_u = 6, 2
        Xn = (self.N + 1) * n_x
        return sol[Xn:Xn + n_u].copy()
    
    def extract_first_control(self, sol, m):
        n_x, n_u = 6, 2
        Xn = (self.N + 1) * n_x
        Un = self.N * n_u

        u0 = sol[Xn:Xn + n_u]
        lam = sol[Xn + Un:Xn + Un + m]
        return u0, lam

    def unpack_predicted_traj(self, sol: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack xPred (N+1,6) and uPred (N,2) from the LMPC solution vector."""
        n_x, n_u = 6, 2
        Xn = (self.N + 1) * n_x
        Un = self.N * n_u
        xflat = sol[:Xn]
        uflat = sol[Xn:Xn + Un]
        xPred = xflat.reshape((self.N + 1, n_x))
        uPred = uflat.reshape((self.N, n_u))
        return xPred, uPred

    def build_xbar_from_prev(self, x0: np.ndarray) -> np.ndarray:
        """Paper candidate trajectory: shift previous x* and append z_t."""
        xbar = np.zeros((self.N + 1, 6), dtype=float)
        if self.prev_x_pred is not None and self.prev_x_pred.shape == xbar.shape:
            # xbar = [x*_{t|t-1}, ..., x*_{t+N-1|t-1}, z_t], then set first to measured x0
            tail = self.prev_x_pred[1:, :]              # N rows
            last = self.z_t.reshape(1, -1) if self.z_t is not None else tail[-1:].copy()
            xbar = np.vstack((tail, last))
            xbar[0, :] = x0
            xbar[-1, 4] = float(xbar[-1, 4] % self.track.L)
            return xbar
        # fallback: use tracking reference as a benign candidate
        xbar = self.build_tracking_reference(x0)
        if self.z_t is not None:
            xbar[-1, :] = self.z_t
        return xbar

    def control_loop(self):
        if not self.have_pose:
            return

        x0 = self.x_meas.copy()

        if self.prev_s is None:
            self.prev_s = float(x0[4])

        # Initialise z_t if needed (paper eq. (10), first step of iteration)
        if self.z_t is None:
            z0 = self._init_z_from_last_lap()
            self.z_t = x0.copy() if z0 is None else z0

        kappa_here = self.track.kappa_at(x0[4])
        denom = max(1e-3, 1.0 - kappa_here * x0[5])
        fs = (x0[0] * math.cos(x0[3]) - x0[1] * math.sin(x0[3])) / denom

        s_now = float(x0[4])

        # Arm finish detection only after moving sufficiently away from the finish region
        if not self.finish_armed:
            ds_from_finish = abs(self.wrapped_s_error(s_now, self.finish_s_center))
            if ds_from_finish > self.finish_arm_s_min:
                self.finish_armed = True

        pose_ok_for_finish = (abs(x0[5]) < self.wrap_ey_tol) and (abs(x0[3]) < self.wrap_epsi_tol)

        prev_err = self.wrapped_s_error(self.prev_s, self.finish_s_center)
        now_err = self.wrapped_s_error(s_now, self.finish_s_center)

        prev_in_finish = abs(prev_err) <= self.finish_s_halfwidth
        now_in_finish = abs(now_err) <= self.finish_s_halfwidth

        if not now_in_finish:
            self.finish_latch = False

        moved_forward = (fs > self.finish_min_forward_speed)
        crossed_finish_segment = (
            self.finish_armed and
            (not self.finish_latch) and
            moved_forward and
            (prev_err < 0.0) and
            (now_err >= 0.0) and
            ((not self.finish_require_pose_ok) or pose_ok_for_finish)
        )

        if crossed_finish_segment:
            self.finish_latch = True

        if self.debug_counter % 40 == 0:
            self.publish_current_traj()
            self.publish_safe_set()

        if crossed_finish_segment:
            steps = len(self.curr_traj_u)
            lap_time = steps * self.dt if steps > 0 else 0.0

            self.get_logger().info(f"LAP FINISHED | steps={steps} | time={lap_time:.2f}s")

            if steps >= self.min_lap_steps:
                fail_ratio = self.qp_fail_in_lap / max(1, steps)
                wall_ratio = self.near_wall_in_lap / max(1, steps)

                if fail_ratio > self.fail_frac_thresh or wall_ratio > self.wall_frac_thresh:
                    self.get_logger().warn(
                        f"⚠️ LAP REJECTED | fail={fail_ratio:.2%} | wall={wall_ratio:.2%}"
                    )
                else:
                    X = np.vstack(self.curr_traj_x)
                    U = np.vstack(self.curr_traj_u)

                    Tn = min(U.shape[0], max(0, X.shape[0] - 1))
                    U = U[:Tn, :]
                    X = X[:Tn + 1, :]

                    lap_idx = len(self.laps)
                    self.laps.append({
                        'x': X,
                        'u': U,
                        'J': np.array([Tn - t for t in range(Tn + 1)], dtype=float),
                        'mode': getattr(self, "last_control_mode", "PP"),
                        'lap_idx': lap_idx
                     })

                    ds_bin = self.track.L / max(1, self.s_bins)

                    for t in range(Tn):
                        xt = X[t]
                        xnext = X[t + 1]

                        if abs(xt[5]) > 1.2 * self.ey_abs_max:
                            continue

                        bin_idx = int(math.floor((xt[4] % self.track.L) / ds_bin)) % self.s_bins

                        item = {
                            'x': xt.copy(),
                            'x_next': xnext.copy(),
                            'J': float(Tn - t),
                            's': float(xt[4] % self.track.L),
                            'lap_idx': lap_idx
                        }
                            
                        bucket = self.buckets[bin_idx]
                        bucket.append(item)
                        # keep a short list per bin; prefer lower J and smaller errors
                        bucket.sort(key=lambda it: (it['J'], abs(it['x'][5]), abs(it['x'][3])))
                        del bucket[self.per_bin_keep:]
                    total_bucket_points = sum(len(b) for b in self.buckets.values())
                    self.get_logger().info(f"[MEMDBG] laps={len(self.laps)} bucket_points={total_bucket_points}")

                    if self.best_lap_time is None or lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time
                        self.get_logger().info(f"NEW BEST LAP: {lap_time:.2f}s")
                    else:
                        self.get_logger().info(
                            f"Lap stored: {lap_time:.2f}s | best: {self.best_lap_time:.2f}s"
                        )
            else:
                self.get_logger().warn(f"LAP TOO SHORT ({steps})")

            self.curr_traj_x = []
            self.curr_traj_u = []
            self.qp_fail_in_lap = 0
            self.near_wall_in_lap = 0
            self.finish_armed = False

            # Start next iteration cleanly (paper-style iteration reset)
            self.z_t = self._init_z_from_last_lap()
            self.prev_x_pred = None
            self.prev_u_pred = None
            self.fallback_countdown = 0
            self.prev_osqp_x = None
            self.prev_osqp_y = None
            if getattr(self, "osqp_reuse", False):
                self._osqp_cache.clear()

        n_laps = len(self.laps)

        if n_laps < self.pp_laps:
            mode = "PP"
        elif n_laps < self.pp_laps + self.lti_mpc_laps:
            mode = "LTI_MPC"
        else:
            mode = "LMPC"
            # mode = "LTI_MPC"

        self.last_control_mode = mode

        if mode == "PP":
            delta, a = self.pure_pursuit_control(x0)

            self.apply_control(delta, a)
            self.append_traj(x0, np.array([delta, a]))
            self.prev_s = float(x0[4])
            self.debug_counter += 1
            return


        elif mode == "LTI_MPC":
            x_ref_seq = self.build_tracking_reference(x0)

            A_dyn, B_dyn, c_dyn = self.fit_lti_model_from_laps()
            if A_dyn is None:
                self.get_logger().warn("Not enough data for LTI fit -> fallback PP")
                delta, a = self.pure_pursuit_control(x0)
                self.apply_control(delta, a)
                self.append_traj(x0, np.array([delta, a]))
                self.prev_s = float(x0[4])
                self.debug_counter += 1
                return

            A_lti, B_lti, C_lti = self.lti_matrices(x_ref_seq[0], A_dyn, B_dyn, c_dyn)

            ey_low = ey_high = None
            if self.track.w_left is not None and self.track.w_right is not None:
                ey_low = np.zeros(self.N + 1)
                ey_high = np.zeros(self.N + 1)
                for k in range(self.N + 1):
                    wl, wr = self.track.widths_at(x_ref_seq[k, 4], margin=self.ey_margin)
                    ey_low[k] = -wr
                    ey_high[k] = wl

            vx_high = np.full(self.N + 1, self.vx_max, dtype=float)
            for k in range(self.N + 1):
                kap = abs(self.track.kappa_at(x_ref_seq[k, 4]))
                if kap < 1e-4:
                    vcap = self.vx_max
                else:
                    vcap = math.sqrt(max(0.25, self.ay_max / kap))
                vx_high[k] = min(self.vx_max, max(self.v_turn_min, vcap))

            if self.vx_cap_enable and self.vx_cap_brake_feasible:
                vx_high = self.make_vx_caps_braking_feasible(vx_high, x0)

            # Cap handling: disable / braking-feasible shaping
            vx_high_seq = None
            if self.vx_cap_enable:
                if self.vx_cap_brake_feasible:
                    vx_high = self.make_vx_caps_braking_feasible(vx_high, x0)
                # Always ensure k=0 includes measured vx (avoids immediate infeasible)
                vx_high[0] = max(vx_high[0], float(x0[0]) + self.vx_cap_tol)
                vx_high_seq = vx_high


            P, q, Aqp, lvec, uvec = self.lti_mpc_mats(
                A_lti, B_lti, C_lti, x0, x_ref_seq,
                ey_low_seq=ey_low,
                ey_high_seq=ey_high,
                vx_high_seq=vx_high_seq
            )

            sol = self.solve_qp(P, q, Aqp, lvec, uvec, key="LTI")
            steps = len(self.curr_traj_u)
            kap0 = abs(self.track.kappa_at(float(x0[4])))
            if sol is None:
                self.get_logger().info(f"[INFDBG] step={steps:03d} s={x0[4]:.2f} kappa={kappa_here:.3f} "
                                    f"vx={x0[0]:.2f} vx_cap0={vx_high_seq[0]:.2f} "
                                    f"vy={x0[1]:.2f} wz={x0[2]:.2f} epsi={x0[3]:.2f} ey={x0[5]:.2f}")
                # self.get_logger().warn(f"Step: {steps}")
                self.get_logger().warn("LTI MPC failed -> fallback PP")
                delta, a = self.pure_pursuit_control(x0)
            else:
                u0 = self.extract_first_control_lti(sol)
                delta = float(sat(u0[0], self.delta_min, self.delta_max))
                a = float(sat(u0[1], self.a_min, self.a_max))

            self.apply_control(delta, a)
            self.append_traj(x0, np.array([delta, a]))
            self.prev_s = float(x0[4])
            self.debug_counter += 1
            return

        # xbar = np.tile(x0, (self.N + 1, 1))
        xbar = self.build_xbar_from_prev(x0)
        
        A_list, B_list, C_list = self.atv_matrices(xbar)

        ey_low = ey_high = None
        if self.track.w_left is not None and self.track.w_right is not None:
            ey_low = np.zeros(self.N + 1)
            ey_high = np.zeros(self.N + 1)
            for k in range(self.N + 1):
                wl, wr = self.track.widths_at(xbar[k, 4], margin=self.ey_margin)
                ey_low[k] = -wr
                ey_high[k] = wl

        vx_high = np.full(self.N + 1, self.vx_max, dtype=float)
        for k in range(self.N + 1):
            kap = abs(self.track.kappa_at(xbar[k, 4]))
            if kap < 1e-4:
                vcap = self.vx_max
            else:
                vcap = math.sqrt(max(0.25, self.ay_max / kap))
            vx_high[k] = min(self.vx_max, max(self.v_turn_min, vcap))

        vx_high_seq = None
        if self.vx_cap_enable:
            if self.vx_cap_brake_feasible:
                vx_high = self.make_vx_caps_braking_feasible(vx_high, x0)
            vx_high[0] = max(vx_high[0], float(x0[0]) + self.vx_cap_tol)
            vx_high_seq = vx_high

        ss_query = self.z_t.copy() if self.z_t is not None else xbar[-1].copy()

        P, q, Aqp, lvec, uvec, m = self.osqp_mats(
            A_list, B_list, C_list, x0, ss_query,
            ey_low_seq=ey_low,
            ey_high_seq=ey_high,
            vx_high_seq=vx_high_seq,
            xbar_seq=xbar
        )

        state_bad = (abs(x0[5]) > 1.0) or (abs(x0[3]) > 1.0)

        if state_bad:
            self.fallback_countdown = max(self.fallback_countdown, 10)
            self.get_logger().warn("STATE TOO FAR FROM USABLE REGION -> recovery mode")
            sol = None
        elif self.fallback_countdown > 0:
            self.fallback_countdown -= 1
            sol = None
        else:
            sol = self.solve_qp(P, q, Aqp, lvec, uvec, key="LMPC")

        if sol is None:
            self.get_logger().warn("FALLBACK CONTROLLER ACTIVE")

            # Use PP recovery when we are far from nominal (more robust than small-angle fb)
            if abs(x0[5]) > 0.25 or abs(x0[3]) > 0.20:
                delta, a = self.pure_pursuit_control(x0)
                self.apply_control(delta, a)
                self.append_traj(x0, np.array([delta, a]))
                self.prev_s = float(x0[4])
                self.debug_counter += 1
                return

            delta_ff = math.atan(self.L_wb * self.track.kappa_at(x0[4]))
            delta_fb = -0.6 * x0[5] - 0.8 * x0[3]

            delta = sat(
                self.steer_sign * (delta_ff + delta_fb),
                -0.18, 0.18
            )

            recovery_speed = 2.0
            if abs(x0[5]) > 0.6 or abs(x0[3]) > 0.35:
                recovery_speed = 1.5
            if abs(x0[5]) > 1.0 or abs(x0[3]) > 0.7:
                recovery_speed = 1.0

            a = sat((recovery_speed - self.speed_cmd) / self.dt, -1.0, 1.0)

        else:
            u0, lam = self.extract_first_control(sol, m)
            delta = float(sat(u0[0], self.delta_min, self.delta_max))
            a = float(sat(u0[1], self.a_min, self.a_max))
            self.z_t = self.update_z(self.z_t, lam)
            # Store prediction to build candidate xbar at next time step
            self.prev_x_pred, self.prev_u_pred = self.unpack_predicted_traj(sol, m)
        self.apply_control(delta, a)

        if self.debug_counter % 10 == 0:
            self.get_logger().info(
                f"[STATE] vx={x0[0]:.2f} ey={x0[5]:.2f} epsi={x0[3]:.2f}"
            )
            self.get_logger().info(
                f"[CTRL ] delta={delta:.2f} a={a:.2f}"
            )
        if x0[4] < 5.0 or x0[4] > (self.track.L - 5.0):
            self.get_logger().info(
                f"[WRAPDBG] s={x0[4]:.2f} kappa={self.track.kappa_at(x0[4]):.3f} "
                f"vx={x0[0]:.2f} ey={x0[5]:.2f} epsi={x0[3]:.2f}"
            )
        self.debug_counter += 1
        self.append_traj(x0, np.array([delta, a]))
        self.prev_s = float(x0[4])

    def tracking_controller(self, s, ey, epsi, vx):
        k_e = 1.5
        k_psi = 2.5
        target_speed = 2.5
        k_v = 1.0

        kappa = self.track.kappa_at(s)

        delta_ff = kappa * self.L_wb
        delta_fb = -k_e * ey - k_psi * epsi
        delta = delta_ff + delta_fb
        delta = 0.7 * self.prev_delta + 0.3 * delta
        delta = np.clip(delta, -0.25, 0.25)

        if abs(ey) > 0.3:
            delta += -1.0 * ey
        self.prev_delta = delta

        a = k_v * (target_speed - vx)
        a = np.clip(a, -2.0, 3.0)

        return delta, a

    def apply_control(self, delta: float, a: float):
        self.speed_cmd = sat(self.speed_cmd + a * self.dt, self.vx_min, self.vx_max)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(self.speed_cmd)
        self.pub_cmd.publish(msg)

        self.u_prev = np.array([delta, a], dtype=float)

    def append_traj(self, x: np.ndarray, u: np.ndarray):
        self.curr_traj_x.append(x.copy())
        self.curr_traj_u.append(u.copy())

        if abs(x[5]) > 0.95 * self.ey_abs_max:
            self.near_wall_in_lap += 1

    def publish_safe_set(self):
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safe_set"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for lap in self.laps:
            X = lap['x']
            for i in range(0, len(X), 5):
                s = float(X[i,4])
                ey = float(X[i,5])

                gx, gy = self.frenet.frenet_to_global(s, ey)

                p = Point()
                p.x = gx
                p.y = gy
                p.z = 0.0
                marker.points.append(p)

        self.safe_set_pub.publish(marker)

    def publish_current_traj(self):
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_traj"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        X = self.curr_traj_x
        for i in range(0, len(X), 5):
            s = float(X[i][4])
            ey = float(X[i][5])

            gx, gy = self.frenet.frenet_to_global(s, ey)
            p = Point()
            p.x = gx
            p.y = gy
            p.z = 0.0
            marker.points.append(p)
        self.traj_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LMPCF1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()