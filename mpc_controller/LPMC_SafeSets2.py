#!/usr/bin/env python3
"""
LMPC for F1TENTH (ROS 2 / rclpy) — based on Rosolia & Borrelli (2019)
"Learning How to Autonomously Race a Car: a Predictive Control Approach".

State x = [v_x, v_y, w_z, e_psi, s, e_y], input u = [delta, a].
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import osqp
import pandas as pd
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from scipy import sparse

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker


# -------------------------------
# Utilities
# -------------------------------

def sat(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# -------------------------------
# Track representation
# -------------------------------

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
        """Linear interpolation on periodic s."""
        L = self.L if self.L > 0 else 1.0
        s_wrapped = s_query % L
        idx = np.searchsorted(self.s, s_wrapped) % len(self.s)
        i0 = (idx - 1) % len(self.s)
        i1 = idx
        s0, s1 = float(self.s[i0]), float(self.s[i1])
        a0, a1 = float(arr[i0]), float(arr[i1])
        if s1 == s0:
            return a0
        w = (s_wrapped - s0) / (s1 - s0)
        return (1.0 - w) * a0 + w * a1

    def kappa_at(self, s_query: float) -> float:
        return float(self._interp_pair(self.kappa, s_query))

    def widths_at(self, s_query: float, margin: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
        if self.w_left is None or self.w_right is None:
            return None, None
        wl = float(self._interp_pair(self.w_left, s_query))
        wr = float(self._interp_pair(self.w_right, s_query))
        return max(0.0, wl - margin), max(0.0, wr - margin)


# -------------------------------
# Frenet projector
# -------------------------------

class FrenetProjector:
    def __init__(self, track: Track):
        self.track = track
        self.has_xy = (track.x is not None) and (track.y is not None) and (track.theta is not None)

    def project(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        """Global pose -> (s, ey, epsi) via nearest centerline point."""
        if not self.has_xy:
            # fall back: treat x,y as s,ey (not recommended)
            return x, y, yaw

        cx, cy, cth = self.track.x, self.track.y, self.track.theta
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        i = int(np.argmin(d2))

        s_cl = float(self.track.s[i])
        th = float(cth[i])

        # normal points left of tangent
        nx, ny = -math.sin(th), math.cos(th)
        ex, ey = x - float(cx[i]), y - float(cy[i])
        e_y = ex * nx + ey * ny
        e_psi = wrap_pi(yaw - th)
        return s_cl, float(e_y), float(e_psi)

    def frenet_to_global(self, s: float, ey: float) -> Tuple[float, float]:
        """(s, ey) -> global (x,y) using centerline + normal."""
        if not self.has_xy:
            return s, ey

        L = self.track.L if self.track.L > 0 else 1.0
        s_wrapped = s % L
        idx = np.searchsorted(self.track.s, s_wrapped) % len(self.track.s)
        i0 = (idx - 1) % len(self.track.s)
        i1 = idx

        s0, s1 = float(self.track.s[i0]), float(self.track.s[i1])
        if s1 == s0:
            xc = float(self.track.x[i0])
            yc = float(self.track.y[i0])
            th = float(self.track.theta[i0])
        else:
            w = (s_wrapped - s0) / (s1 - s0)
            xc = (1.0 - w) * float(self.track.x[i0]) + w * float(self.track.x[i1])
            yc = (1.0 - w) * float(self.track.y[i0]) + w * float(self.track.y[i1])
            th = (1.0 - w) * float(self.track.theta[i0]) + w * float(self.track.theta[i1])

        nx, ny = -math.sin(th), math.cos(th)
        return float(xc + nx * ey), float(yc + ny * ey)


# -------------------------------
# LMPC Node
# -------------------------------

class LMPCF1(Node):
    def __init__(self):
        super().__init__('lmpc_f1tenth')
        qos = QoSProfile(depth=10)

        # ------------ Parameters ------------
        self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv')
        self.declare_parameter('namespace', '')

        self.declare_parameter('dt', 0.1)
        self.declare_parameter('N', 8)

        self.declare_parameter('K', 8)
        self.declare_parameter('P_neighbors', 40)
        self.declare_parameter('h_bandwidth', 8.0)

        self.declare_parameter('vx_bounds', [0.8, 4.0])
        self.declare_parameter('vy_abs_max', 3.0)
        self.declare_parameter('wz_abs_max', 6.0)
        self.declare_parameter('e_psi_abs_max', 1.2)
        self.declare_parameter('e_y_abs_max', 1.0)

        self.declare_parameter('delta_bounds', [-0.30, 0.30])
        self.declare_parameter('a_bounds', [-2.0, 3.5])
        self.declare_parameter('delta_rate_max', 1.3)   # rad/s
        self.declare_parameter('a_rate_max', 3.5)        # (m/s^2)/s

        self.declare_parameter('rho_du', 0.35)
        self.declare_parameter('trust_ey', 0.30)     # meters
        self.declare_parameter('trust_epsi', 0.25)   # rad

        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('ff_gain', 1.0)
        self.declare_parameter('ky_seed', 0.8)
        self.declare_parameter('kpsi_seed', 0.6)

        self.declare_parameter('do_seed_laps', True)
        self.declare_parameter('seed_laps', 2)
        self.declare_parameter('seed_speed', 1.5)

        self.declare_parameter('ey_margin', 0.45)

        self.declare_parameter('lap_wrap_frac', 0.5)
        self.declare_parameter('wrap_window_m', 3.0)
        self.declare_parameter('min_lap_steps', 150)

        self.declare_parameter('soft_terminal', True)
        self.declare_parameter('terminal_weight', 800.0)

        # memory
        self.declare_parameter('memory_mode', 'buckets')  # 'last2' or 'buckets'
        self.declare_parameter('s_bins', 200)
        self.declare_parameter('per_bin_keep', 3)

        # lap quality gates
        self.declare_parameter('fail_frac_thresh', 0.08)
        self.declare_parameter('wall_frac_thresh', 0.12)

        # visualization
        self.declare_parameter('viz_frame', 'map')
        self.declare_parameter('viz_ns', 'mpc')

        # ------------ Read parameters ------------
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
        self.trust_ey = float(self.get_parameter('trust_ey').value)
        self.trust_epsi = float(self.get_parameter('trust_epsi').value)

        self.L_wb = float(self.get_parameter('wheelbase').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)
        self.ff_gain = float(self.get_parameter('ff_gain').value)
        self.ky_seed = float(self.get_parameter('ky_seed').value)
        self.kpsi_seed = float(self.get_parameter('kpsi_seed').value)

        self.do_seed = bool(self.get_parameter('do_seed_laps').value)
        self.seed_laps = int(self.get_parameter('seed_laps').value)
        self.seed_speed = float(self.get_parameter('seed_speed').value)

        self.ey_margin = float(self.get_parameter('ey_margin').value)

        self.lap_wrap_frac = float(self.get_parameter('lap_wrap_frac').value)
        self.wrap_window = float(self.get_parameter('wrap_window_m').value)
        self.min_lap_steps = int(self.get_parameter('min_lap_steps').value)

        self.soft_terminal = bool(self.get_parameter('soft_terminal').value)
        self.terminal_weight = float(self.get_parameter('terminal_weight').value)

        self.memory_mode = str(self.get_parameter('memory_mode').value)
        self.s_bins = int(self.get_parameter('s_bins').value)
        self.per_bin_keep = int(self.get_parameter('per_bin_keep').value)

        self.fail_frac_thresh = float(self.get_parameter('fail_frac_thresh').value)
        self.wall_frac_thresh = float(self.get_parameter('wall_frac_thresh').value)

        self.viz_frame = str(self.get_parameter('viz_frame').value)
        self.viz_ns = str(self.get_parameter('viz_ns').value)

        # ------------ Load track ------------
        track_csv = str(self.get_parameter('track_csv').value)
        self.track = self._load_track(track_csv)
        self.frenet = FrenetProjector(self.track)

        self.get_logger().info(
            f"Track loaded: L={self.track.L:.2f} m; "
            f"kappa|min,max|=({float(np.min(self.track.kappa)):.3f}, {float(np.max(self.track.kappa)):.3f})"
        )

        # if widths exist, tighten global ey bound using min width
        if self.track.w_left is not None and self.track.w_right is not None:
            w_min = float(np.nanmin([np.nanmin(self.track.w_left), np.nanmin(self.track.w_right)]))
            bound = max(0.2, w_min - self.ey_margin)
            self.ey_abs_max = min(self.ey_abs_max, bound)
            self.get_logger().info(f"Track width detected -> |e_y| <= {self.ey_abs_max:.2f} m")

        # ------------ Internal state ------------
        self.x_meas = np.zeros(6)
        self.have_pose = False

        self.speed_cmd = 0.0
        self.u_prev = np.zeros(2)

        self.prev_s = None
        self.wrap_thresh = max(0.5, self.lap_wrap_frac * self.track.L)

        self.z_t = None  # terminal candidate

        # OSQP warm-start
        self.prev_osqp_x = None
        self.prev_osqp_y = None

        # Lap buffers
        self.laps: List[dict] = []
        self.curr_traj_x: List[np.ndarray] = []
        self.curr_traj_u: List[np.ndarray] = []

        # Bucketed memory: per s-bin keep best (x, x_next, J)
        self.buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]] = {i: [] for i in range(self.s_bins)}

        # Lap stats for quality gating
        self.qp_fail_in_lap = 0
        self.near_wall_in_lap = 0
        self.total_laps = 0
        self.best_lap_time = None

        # Metrics for KNN (includes modest weights on epsi and ey)
        self.Q_knn = np.diag([0.1, 0.5, 0.5, 0.1, 0.0, 0.4])

        # ------------ ROS I/O ------------
        topic_cmd = f'{self.ns}/drive' if self.ns else 'drive'
        topic_odom = f'{self.ns}/odom' if self.ns else 'odom'

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, topic_cmd, qos)
        self.sub_odom = self.create_subscription(Odometry, topic_odom, self.odom_cb, qos)

        self.pub_path = self.create_publisher(Path, f'{self.viz_ns}/pred_path', qos)
        self.pub_mark = self.create_publisher(Marker, f'{self.viz_ns}/pred_marker', qos)

        # Param callback (safe even if you never use live tuning)
        self.add_on_set_parameters_callback(self._on_param_change)

        # Control timer
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("LMPC F1TENTH node initialized.")
        self.get_logger().info(
            f"Seeding: {'ON' if self.do_seed else 'OFF'} | seed_laps={self.seed_laps} | "
            f"seed_speed={self.seed_speed:.2f} m/s | N={self.N} | memory_mode={self.memory_mode}"
        )

    # -------------------------------
    # Live parameter updates (optional)
    # -------------------------------
    def _on_param_change(self, params):
        try:
            for p in params:
                if p.name == 'rho_du' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.rho_du = float(p.value)
                elif p.name == 'ey_margin' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.ey_margin = float(p.value)
                elif p.name == 'delta_bounds' and p.type_ == Parameter.Type.DOUBLE_ARRAY:
                    self.delta_min, self.delta_max = float(p.value[0]), float(p.value[1])
                elif p.name == 'delta_rate_max' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.delta_rate_max = float(p.value)
                elif p.name == 'a_rate_max' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.a_rate_max = float(p.value)
                elif p.name == 'terminal_weight' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.terminal_weight = float(p.value)
                elif p.name == 'seed_speed' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.seed_speed = float(p.value)
                elif p.name == 'N' and p.type_ == Parameter.Type.INTEGER:
                    self.N = int(p.value)
                elif p.name == 'K' and p.type_ == Parameter.Type.INTEGER:
                    self.K = int(p.value)
                elif p.name == 'P_neighbors' and p.type_ == Parameter.Type.INTEGER:
                    self.Pn = int(p.value)
                elif p.name == 'h_bandwidth' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.h = float(p.value)
                elif p.name == 'memory_mode' and p.type_ == Parameter.Type.STRING:
                    self.memory_mode = str(p.value)
                elif p.name == 'per_bin_keep' and p.type_ == Parameter.Type.INTEGER:
                    self.per_bin_keep = int(p.value)
            return SetParametersResult(successful=True)
        except Exception as e:
            self.get_logger().warn(f"Param update failed: {e}")
            return SetParametersResult(successful=False)

    # -------------------------------
    # Track loader
    # -------------------------------
    def _load_track(self, path: str) -> Track:
        if not path:
            self.get_logger().warn("No track CSV provided. Using dummy straight track.")
            s = np.linspace(0.0, 20.0, 201)
            k = np.zeros_like(s)
            return Track(s=s, kappa=k)

        df = pd.read_csv(path)

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

        if kappa is None and x is not None and y is not None:
            xp = np.gradient(x)
            yp = np.gradient(y)
            xpp = np.gradient(xp)
            ypp = np.gradient(yp)
            denom = xp * xp + yp * yp
            denom[denom < 1e-6] = 1e-6
            kappa = (xp * ypp - yp * xpp) / np.power(denom, 1.5)

        if s is None or kappa is None:
            raise RuntimeError("Track CSV must have s/kappa or x/y (and optionally theta) to derive them.")

        order = np.argsort(s)
        s = s[order]
        kappa = kappa[order]
        x = x[order] if x is not None else None
        y = y[order] if y is not None else None
        theta = theta[order] if theta is not None else None
        if w_left is not None:
            w_left = w_left[order]
        if w_right is not None:
            w_right = w_right[order]

        s = s - s[0]
        return Track(
            s=s.astype(float),
            kappa=kappa.astype(float),
            x=x.astype(float) if x is not None else None,
            y=y.astype(float) if y is not None else None,
            theta=theta.astype(float) if theta is not None else None,
            w_left=w_left.astype(float) if w_left is not None else None,
            w_right=w_right.astype(float) if w_right is not None else None,
        )

    # -------------------------------
    # Odometry callback
    # -------------------------------
    def odom_cb(self, msg: Odometry):
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        s_cur, e_y, e_psi = self.frenet.project(px, py, yaw)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)

        self.x_meas = np.array([vx, vy, wz, e_psi, s_cur, e_y], dtype=float)
        self.have_pose = True

    # -------------------------------
    # Regression / ATV model
    # -------------------------------
    def epanechnikov(self, u: float) -> float:
        return 0.75 * (1 - u * u) if abs(u) < 1.0 else 0.0

    def _collect_recent_samples_for_regression(self):
        """Keep regression local/stable using only the last two stored laps."""
        if len(self.laps) < 1:
            return np.empty((0, 6)), np.empty((0, 2)), np.empty((0, 6))
        j = len(self.laps)
        l = max(0, j - 2)
        X = []
        U = []
        Xn = []
        for lap in self.laps[l:j]:
            X.append(lap['x'][:-1])
            U.append(lap['u'])
            Xn.append(lap['x'][1:])
        return np.vstack(X), np.vstack(U), np.vstack(Xn)

    def build_regression_sets(self, x_ref: np.ndarray):
        X, U, Xn = self._collect_recent_samples_for_regression()
        if X.shape[0] == 0:
            return (np.empty((0, 5)), np.empty((0,)),
                    np.empty((0, 5)), np.empty((0,)),
                    np.empty((0, 5)), np.empty((0,)),
                    np.empty((0, 6)))

        dif = X - x_ref.reshape(1, -1)
        d2 = np.einsum('ij,jk,ik->i', dif, self.Q_knn, dif)
        d = np.sqrt(np.maximum(0.0, d2))
        idx = np.argsort(d)[:min(self.Pn, len(d))]

        Xp = X[idx]
        Up = U[idx]
        Xp1 = Xn[idx]

        # vx_{k+1} = [vx,vy,wz,a,1] gamma
        Phi_vx = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 1], np.ones(len(idx))])
        y_vx = Xp1[:, 0]
        # vy_{k+1} = [vx,vy,wz,delta,1] gamma
        Phi_vy = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 0], np.ones(len(idx))])
        y_vy = Xp1[:, 1]
        # wz_{k+1} = [vx,vy,wz,delta,1] gamma
        Phi_wz = np.column_stack([Xp[:, 0], Xp[:, 1], Xp[:, 2], Up[:, 0], np.ones(len(idx))])
        y_wz = Xp1[:, 2]

        return Phi_vx, y_vx, Phi_vy, y_vy, Phi_wz, y_wz, Xp

    def weighted_ls(self, Phi: np.ndarray, y: np.ndarray, x_ref: np.ndarray, X_samples: np.ndarray) -> np.ndarray:
        if Phi.shape[0] == 0:
            return np.zeros(Phi.shape[1])

        dif = X_samples - x_ref.reshape(1, -1)
        dist2 = np.einsum('ij,jk,ik->i', dif, self.Q_knn, dif)
        dist = np.sqrt(np.maximum(0.0, dist2))
        u = dist / max(1e-6, self.h)
        w = np.array([self.epanechnikov(ui) for ui in u], dtype=float) + 1e-6

        W = np.diag(w)
        H = Phi.T @ W @ Phi + 1e-6 * np.eye(Phi.shape[1])
        b = Phi.T @ W @ y

        try:
            gamma = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            gamma = np.linalg.lstsq(H, b, rcond=None)[0]
        return gamma

    def atv_matrices(self, xbar_seq: np.ndarray):
        """Affine time-varying linear model x_{k+1} = A_k x_k + B_k u_k + C_k."""
        A_list, B_list, C_list = [], [], []
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
            dfepsi[1] = -(-sin_e / den) * kappa
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

            # learned part (vx,vy,wz)
            A[0, 0:3] = gamma_vx[0:3]
            A[1, 0:3] = gamma_vy[0:3]
            A[2, 0:3] = gamma_wz[0:3]
            B[0, 1] = gamma_vx[3]  # a
            B[1, 0] = gamma_vy[3]  # delta
            B[2, 0] = gamma_wz[3]  # delta
            C[0] = gamma_vx[4]
            C[1] = gamma_vy[4]
            C[2] = gamma_wz[4]

            # Frenet kinematics (Euler discretization)
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

    # -------------------------------
    # Safe set with bucketed memory
    # -------------------------------
    def _safe_set_last2(self, x_query: np.ndarray):
        """Fallback safe set from last 2 laps."""
        if len(self.laps) < 1:
            return np.zeros((6, 0)), np.zeros((6, 0)), np.zeros((0,)), np.zeros((0,), dtype=int)

        j = len(self.laps)
        l = max(0, j - 2)

        Xcols, Scols, Jvals, info = [], [], [], []
        for lap_idx in range(l, j):
            lap = self.laps[lap_idx]
            x_arr = lap['x']
            u_arr = lap['u']
            Tn = u_arr.shape[0]

            s_all = x_arr[:, 4]
            s_q = x_query[4]
            L = self.track.L

            ds_raw = np.abs(s_all - s_q)
            ds = np.minimum(ds_raw, L - ds_raw)

            order = np.argsort(ds)
            take = order[:min(self.K, max(0, len(order) - 1))]

            for t in take:
                if t + 1 >= len(x_arr):
                    continue
                if abs(x_arr[t, 5]) > 0.8 * self.ey_abs_max:
                    continue
                Xcols.append(x_arr[t])
                Scols.append(x_arr[t + 1])
                Jvals.append(float(Tn - t))
                info.append(lap_idx)

        if not Xcols:
            return np.zeros((6, 0)), np.zeros((6, 0)), np.zeros((0,)), np.zeros((0,), dtype=int)

        D = np.stack(Xcols, axis=1)
        S = np.stack(Scols, axis=1)
        return D, S, np.array(Jvals, dtype=float), np.array(info, dtype=int)

    def build_local_safe_set(self, x_query: np.ndarray):
        if self.memory_mode != 'buckets':
            return self._safe_set_last2(x_query)

        # bucketed memory
        if not any(self.buckets.values()):
            return self._safe_set_last2(x_query)

        ds_bin = self.track.L / max(1, self.s_bins)
        s_q = (x_query[4] % self.track.L)
        qbin = int(math.floor(s_q / ds_bin)) % self.s_bins

        window = max(3, self.K // max(1, self.per_bin_keep))
        picks: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for off in range(-window, window + 1):
            b = (qbin + off) % self.s_bins
            picks.extend(self.buckets[b])

        if not picks:
            return self._safe_set_last2(x_query)

        s_all = np.array([p[0][4] for p in picks], dtype=float)
        ds_raw = np.abs(s_all - s_q)
        ds = np.minimum(ds_raw, self.track.L - ds_raw)

        order = np.argsort(ds)[:self.K]
        Xcols = [picks[i][0] for i in order]
        Scols = [picks[i][1] for i in order]
        Jvals = [picks[i][2] for i in order]

        D = np.stack(Xcols, axis=1)
        S = np.stack(Scols, axis=1)
        return D, S, np.array(Jvals, dtype=float), np.zeros(len(order), dtype=int)

    def update_z(self, z_prev: np.ndarray, lambda_prev: np.ndarray) -> np.ndarray:
        D, S, _, _ = self.build_local_safe_set(z_prev)
        if S.shape[1] == 0:
            return z_prev
        return S @ lambda_prev

    # -------------------------------
    # QP building & solve
    # -------------------------------
    def osqp_mats(self, A_list, B_list, C_list, x0, z_t,
              ey_low_seq=None, ey_high_seq=None, xbar_seq=None):
    # def osqp_mats(self, A_list, B_list, C_list, x0, z_t, ey_low_seq=None, ey_high_seq=None):
        n_x, n_u, N = 6, 2, self.N

        Dmat, _, Jvec, _ = self.build_local_safe_set(z_t)
        m = Dmat.shape[1]
        if m == 0:
            Dmat = x0.reshape(-1, 1)
            Jvec = np.array([0.0], dtype=float)
            m = 1

        Xn = (N + 1) * n_x
        Un = N * n_u
        En = n_x if self.soft_terminal else 0
        Zn = Xn + Un + m + En

        P = sparse.csc_matrix((Zn, Zn))
        q = np.zeros(Zn)

        # terminal linear cost: J^T lambda
        q[Xn + Un: Xn + Un + m] = Jvec

        # terminal slack penalty (soft terminal)
        if self.soft_terminal and En > 0:
            start = Xn + Un + m
            P[start:start + En, start:start + En] = sparse.eye(En) * self.terminal_weight

        # input-rate penalty (Δu)
        Du = np.zeros((N, Un))
        for k in range(N):
            if k == 0:
                Du[k, 0:n_u] = 1.0
            else:
                Du[k, k * n_u:(k + 1) * n_u] = 1.0
                Du[k, (k - 1) * n_u:k * n_u] = -1.0
        Puu = self.rho_du * (Du.T @ Du)
        P[Xn:Xn + Un, Xn:Xn + Un] = sparse.csc_matrix(Puu)

        rows, cols, data = [], [], []
        l, u = [], []

        def add_row(entries, low, high):
            r = len(l)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                data.append(v)
            l.append(low)
            u.append(high)

        # x0 equality
        for i in range(n_x):
            add_row([(i, 1.0)], x0[i], x0[i])

        # dynamics
        for k in range(N):
            for i in range(n_x):
                row = [((k + 1) * n_x + i, 1.0)]
                A = A_list[k]
                B = B_list[k]
                for jx in range(n_x):
                    val = -A[i, jx]
                    if val != 0.0:
                        row.append((k * n_x + jx, val))
                for ju in range(n_u):
                    val = -B[i, ju]
                    if val != 0.0:
                        row.append((Xn + k * n_u + ju, val))
                add_row(row, C_list[k][i], C_list[k][i])

        # terminal: x_N = D lambda (+ slack)
        for i in range(n_x):
            row = [(N * n_x + i, 1.0)]
            for jcol in range(m):
                val = -Dmat[i, jcol]
                if val != 0.0:
                    row.append((Xn + Un + jcol, val))
            if self.soft_terminal and En > 0:
                row.append((Xn + Un + m + i, -1.0))
            add_row(row, 0.0, 0.0)

        # simplex lambda
        add_row([(Xn + Un + j, 1.0) for j in range(m)], 1.0, 1.0)
        for j in range(m):
            add_row([(Xn + Un + j, 1.0)], 0.0, np.inf)

        # state bounds
        for k in range(N + 1):
            add_row([(k * n_x + 0, 1.0)], self.vx_min, self.vx_max)
            add_row([(k * n_x + 1, 1.0)], -self.vy_abs_max, self.vy_abs_max)
            add_row([(k * n_x + 2, 1.0)], -self.wz_abs_max, self.wz_abs_max)
            add_row([(k * n_x + 3, 1.0)], -self.epsi_abs_max, self.epsi_abs_max)
            if ey_low_seq is not None and ey_high_seq is not None:
                add_row([(k * n_x + 5, 1.0)], float(ey_low_seq[k]), float(ey_high_seq[k]))
            else:
                add_row([(k * n_x + 5, 1.0)], -self.ey_abs_max, self.ey_abs_max)

        # input bounds
        for k in range(N):
            add_row([(Xn + k * n_u + 0, 1.0)], self.delta_min, self.delta_max)
            add_row([(Xn + k * n_u + 1, 1.0)], self.a_min, self.a_max)

        # rate limits
        dmax = self.delta_rate_max * self.dt if self.delta_rate_max > 0 else None
        amax = self.a_rate_max * self.dt if self.a_rate_max > 0 else None

        if dmax is not None:
            add_row([(Xn + 0, 1.0)], self.u_prev[0] - dmax, self.u_prev[0] + dmax)
            for k in range(1, N):
                add_row([(Xn + k * n_u + 0, 1.0), (Xn + (k - 1) * n_u + 0, -1.0)], -dmax, dmax)

        if amax is not None:
            add_row([(Xn + 1, 1.0)], self.u_prev[1] - amax, self.u_prev[1] + amax)
            for k in range(1, N):
                add_row([(Xn + k * n_u + 1, 1.0), (Xn + (k - 1) * n_u + 1, -1.0)], -amax, amax)

        Aqp = sparse.csc_matrix((data, (rows, cols)), shape=(len(l), Zn))
        return P, q, Aqp, np.array(l), np.array(u), m

    def solve_qp(self, P, q, A, l, u):
        prob = osqp.OSQP()
        prob.setup(
            P=P, q=q, A=A, l=l, u=u,
            verbose=False, warm_start=True,
            eps_abs=1e-3, eps_rel=1e-3,
            max_iter=20000, polish=True
        )
        try:
            if self.prev_osqp_x is not None:
                prob.warm_start(x=self.prev_osqp_x, y=self.prev_osqp_y)
        except Exception:
            pass

        res = prob.solve()
        if res.info.status_val not in (1, 2):
            self.qp_fail_in_lap += 1
            self.get_logger().warn(f"OSQP status: {res.info.status}. Using previous control.")
            return None

        self.prev_osqp_x = res.x
        try:
            self.prev_osqp_y = res.y
        except Exception:
            self.prev_osqp_y = None
        return res.x

    def extract_first_control(self, sol, m):
        n_x, n_u = 6, 2
        Xn = (self.N + 1) * n_x
        u0 = sol[Xn:Xn + n_u]
        lam = sol[Xn + self.N * n_u: Xn + self.N * n_u + m]
        return u0, lam

    # -------------------------------
    # Control loop
    # -------------------------------
    def control_loop(self):
        if not self.have_pose:
            return

        x0 = self.x_meas.copy()

        if self.prev_s is None:
            self.prev_s = float(x0[4])

        # robust lap wrap detection:
        # A) big s drop
        wrap_drop = (self.prev_s - x0[4]) > self.wrap_thresh
        # B) crossing start window from end->begin (in case projection doesn't drop sharply)
        kappa_here = self.track.kappa_at(x0[4])
        denom = max(1e-3, 1.0 - kappa_here * x0[5])
        fs = (x0[0] * math.cos(x0[3]) - x0[1] * math.sin(x0[3])) / denom
        wrap_window = (self.prev_s > (self.track.L - self.wrap_window)) and (x0[4] < self.wrap_window) and (fs > 0.05)

        if wrap_drop or wrap_window:
            steps = len(self.curr_traj_u)

            if steps >= self.min_lap_steps:
                fail_ratio = self.qp_fail_in_lap / max(1, steps)
                wall_ratio = self.near_wall_in_lap / max(1, steps)

                if fail_ratio > self.fail_frac_thresh or wall_ratio > self.wall_frac_thresh:
                    self.get_logger().warn(
                        f"Lap rejected (fail {fail_ratio:.1%}, near-wall {wall_ratio:.1%}). Keeping previous laps."
                    )
                else:
                    X = np.vstack(self.curr_traj_x) if self.curr_traj_x else np.empty((0, 6))
                    U = np.vstack(self.curr_traj_u) if self.curr_traj_u else np.empty((0, 2))

                    # Off-by-one safe alignment: ensure len(X)=Tn+1 and len(U)=Tn
                    Ulen = U.shape[0]
                    Xlen = X.shape[0]
                    Tn = min(Ulen, max(0, Xlen - 1))
                    U = U[:Tn, :]
                    X = X[:Tn + 1, :]

                    J = np.array([Tn - t for t in range(Tn + 1)], dtype=float)

                    self.laps.append({'x': X, 'u': U, 'J': J})
                    self.total_laps += 1

                    lap_time = Tn * self.dt
                    if self.best_lap_time is None or lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time

                    self.get_logger().info(
                        f"Lap {len(self.laps)} stored: {Tn} steps. "
                        f"Lap time: {lap_time:.2f}s (best {self.best_lap_time:.2f}s)"
                    )

                    # update buckets
                    ds_bin = self.track.L / max(1, self.s_bins)
                    for t in range(Tn):
                        xt = X[t]
                        xnext = X[t + 1]
                        if abs(xt[5]) > 0.8 * self.ey_abs_max:
                            continue
                        bin_idx = int(math.floor((xt[4] % self.track.L) / ds_bin)) % self.s_bins
                        Jt = float(Tn - t)
                        bucket = self.buckets[bin_idx]
                        bucket.append((xt, xnext, Jt))
                        bucket.sort(key=lambda tup: tup[2])  # keep best J
                        del bucket[self.per_bin_keep:]

                # seeding messages
                if self.do_seed and len(self.laps) < self.seed_laps:
                    self.get_logger().info(
                        f"Seeding lap {len(self.laps)}/{self.seed_laps} complete — continuing constant-speed path following."
                    )
                elif self.do_seed and len(self.laps) == self.seed_laps:
                    self.get_logger().info("Seeding complete — enabling LMPC from next lap.")

            else:
                self.get_logger().warn(f"Discarding lap with only {steps} steps (< {self.min_lap_steps}).")

            # reset per-lap buffers & counters
            self.curr_traj_x = []
            self.curr_traj_u = []
            self.qp_fail_in_lap = 0
            self.near_wall_in_lap = 0

            # update terminal candidate z_t
            if len(self.laps) >= 1:
                prev = self.laps[-1]['x']
                idx = min(self.N, prev.shape[0] - 1)
                self.z_t = prev[idx].copy()
            else:
                self.z_t = x0.copy()

        # seeding phase
        if self.do_seed and len(self.laps) < self.seed_laps:
            kappa = self.track.kappa_at(x0[4])
            delta_ff = self.ff_gain * math.atan(self.L_wb * kappa)
            delta = (delta_ff - self.ky_seed * x0[5] - self.kpsi_seed * x0[3]) * self.steer_sign
            delta = sat(delta, self.delta_min, self.delta_max)

            a = (self.seed_speed - self.speed_cmd) / self.dt
            a = sat(a, self.a_min, self.a_max)

            self.apply_control(delta, a)
            self.append_traj(x0, np.array([delta, a]))

            # preview horizon (straight)
            xbar = np.tile(x0, (self.N + 1, 1))
            for k in range(1, self.N + 1):
                xbar[k, 4] = (xbar[k - 1, 4] + max(0.0, x0[0]) * self.dt) % self.track.L
            self.publish_preview(xbar)

            self.prev_s = float(x0[4])
            return

        # LMPC phase
        if self.z_t is None:
            self.z_t = x0.copy()

        # candidate sequence for linearization
        xbar = np.tile(x0, (self.N + 1, 1))
        for k in range(1, self.N + 1):
            xbar[k, 4] = (xbar[k - 1, 4] + max(0.0, x0[0]) * self.dt) % self.track.L

        A_list, B_list, C_list = self.atv_matrices(xbar)

        # per-step lateral corridor
        ey_low = ey_high = None
        if self.track.w_left is not None and self.track.w_right is not None:
            ey_low = np.zeros(self.N + 1)
            ey_high = np.zeros(self.N + 1)
            for k in range(self.N + 1):
                wl, wr = self.track.widths_at(xbar[k, 4], margin=self.ey_margin)
                ey_low[k] = -wr
                ey_high[k] = wl

        P, q, Aqp, lvec, uvec, m = self.osqp_mats(
            A_list, B_list, C_list, x0, self.z_t,
            ey_low_seq=ey_low, ey_high_seq=ey_high, xbar_seq=xbar
        )

        sol = self.solve_qp(P, q, Aqp, lvec, uvec)
        use_x_seq = xbar

        if sol is None:
            # fallback PD (very mild)
            delta = sat(-0.8 * x0[5] - 0.6 * x0[3], self.delta_min, self.delta_max)
            a = 0.0
        else:
            n_x = 6
            Xn = (self.N + 1) * n_x
            Xopt = sol[:Xn].reshape(self.N + 1, n_x)
            use_x_seq = Xopt

            u0, lam = self.extract_first_control(sol, m)
            delta = float(sat(u0[0], self.delta_min, self.delta_max))
            a = float(sat(u0[1], self.a_min, self.a_max))

            self.z_t = self.update_z(self.z_t, lam)

        self.apply_control(delta, a)
        self.append_traj(x0, np.array([delta, a]))
        self.publish_preview(use_x_seq)

        self.prev_s = float(x0[4])

    # -------------------------------
    # Publishing helpers
    # -------------------------------
    def apply_control(self, delta: float, a: float):
        self.speed_cmd += a * self.dt
        self.speed_cmd = sat(self.speed_cmd, self.vx_min, self.vx_max)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(self.speed_cmd)
        self.pub_cmd.publish(msg)

        self.u_prev = np.array([delta, a], dtype=float)

    def append_traj(self, x: np.ndarray, u: np.ndarray):
        self.curr_traj_x.append(x.copy())
        self.curr_traj_u.append(u.copy())
        if abs(x[5]) > 0.8 * self.ey_abs_max:
            self.near_wall_in_lap += 1

    def publish_preview(self, Xseq: np.ndarray):
        """Publish predicted plan as Path and Marker line in RViz."""
        try:
            pts_xy = []
            for k in range(Xseq.shape[0]):
                s_k = float(Xseq[k, 4])
                ey_k = float(Xseq[k, 5])
                xg, yg = self.frenet.frenet_to_global(s_k, ey_k)
                pts_xy.append((xg, yg))

            path = Path()
            path.header.frame_id = self.viz_frame
            path.header.stamp = self.get_clock().now().to_msg()
            for xg, yg in pts_xy:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(xg)
                ps.pose.position.y = float(yg)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub_path.publish(path)

            m = Marker()
            m.header = path.header
            m.ns = self.viz_ns
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color.a = 1.0
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.1
            m.points = [Point(x=float(xg), y=float(yg), z=0.02) for xg, yg in pts_xy]
            self.pub_mark.publish(m)

        except Exception as e:
            self.get_logger().warn(f"viz failed: {e}")


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