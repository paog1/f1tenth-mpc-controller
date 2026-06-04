"""
Evaluator Centerline Tracking MPC for F1TENTH (ROS 2 / rclpy)

Includes real-time telemetry tracking, lap detection, and theoretical 
centerline lap time computation for precise Q/R matrix tuning.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import osqp
import pandas as pd
from scipy import sparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker


# ===============================
# Global Utilities
# ===============================

def sat(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ===============================
# Track & Geometry Representation
# ===============================

@dataclass
class Track:
    s: np.ndarray
    kappa: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    theta: Optional[np.ndarray] = None

    @property
    def L(self) -> float:
        return float(self.s[-1]) if len(self.s) else 1.0

    def _interp_pair(self, arr: np.ndarray, s_query: float) -> float:
        L = self.L if self.L > 0 else 1.0
        s_wrapped = s_query % L
        idx = np.searchsorted(self.s, s_wrapped)

        if idx == 0:
            i0, i1 = len(self.s) - 1, 0
            s0, s1 = float(self.s[i0]) - L, float(self.s[i1])
        elif idx >= len(self.s):
            i0, i1 = len(self.s) - 1, 0
            s0, s1 = float(self.s[i0]), float(self.s[i1]) + L
        else:
            i0, i1 = idx - 1, idx
            s0, s1 = float(self.s[i0]), float(self.s[i1])

        a0, a1 = float(arr[i0]), float(arr[i1])

        if abs(s1 - s0) < 1e-9:
            return a0

        if self.theta is not None and arr is self.theta:
            d = wrap_pi(a1 - a0)
            w = (s_wrapped - s0) / (s1 - s0)
            return wrap_pi(a0 + w * d)

        w = (s_wrapped - s0) / (s1 - s0)
        return (1.0 - w) * a0 + w * a1

    def kappa_at(self, s_query: float) -> float:
        return float(self._interp_pair(self.kappa, s_query))


class FrenetProjector:
    def __init__(self, track: Track):
        self.track = track
        self.has_xy = (track.x is not None) and (track.y is not None) and (track.theta is not None)
        self.last_idx = 0

    def project(self, x: float, y: float, yaw: float):
        if not self.has_xy: return x, y, yaw

        n = len(self.track.s)
        if n < 2: return 0.0, 0.0, 0.0

        best = None
        for off in range(-25, 26):                 
            i0   = (self.last_idx + off) % n
            i1   = (i0 + 1) % n
            cand = self._segment_project(x, y, i0, i1)
            if best is None or cand[0] < best[0]:
                best = cand

        if best is None or best[0] > 1.5**2:
            for i0 in range(n):
                i1   = (i0 + 1) % n
                cand = self._segment_project(x, y, i0, i1)
                if cand[0] < best[0]:
                    best = cand

        _, s_cl, e_y, seg_idx = best
        self.last_idx = seg_idx

        th     = float(self.track._interp_pair(self.track.theta, s_cl))
        e_psi  = wrap_pi(yaw - th)
        return float(s_cl), float(e_y), float(e_psi)

    def _segment_project(self, px: float, py: float, i0: int, i1: int):
        tx0, ty0 = float(self.track.x[i0]), float(self.track.y[i0])
        tx1, ty1 = float(self.track.x[i1]), float(self.track.y[i1])

        vx, vy = tx1 - tx0, ty1 - ty0
        seg_len2 = vx*vx + vy*vy
        if seg_len2 < 1e-9: return 1e9, 0.0, 0.0, i0

        t = sat(((px-tx0)*vx + (py-ty0)*vy)/seg_len2, 0.0, 1.0)
        xproj, yproj = tx0 + t*vx, ty0 + t*vy
        dx, dy = px - xproj, py - yproj

        s0, s1 = float(self.track.s[i0]), float(self.track.s[i1])
        if i1 == 0 and i0 == len(self.track.s)-1: s1 += self.track.L
        sproj = (s0 + t*(s1-s0)) % self.track.L

        th = float(self.track._interp_pair(self.track.theta, sproj))
        nx, ny = -math.sin(th), math.cos(th)
        ey = (px-xproj)*nx + (py-yproj)*ny

        return dx*dx + dy*dy, sproj, ey, i0

    def frenet_to_global(self, s: float, ey: float) -> Tuple[float, float]:
        if not self.has_xy: return s, ey
        xc = self.track._interp_pair(self.track.x, s)
        yc = self.track._interp_pair(self.track.y, s)
        th = self.track._interp_pair(self.track.theta, s)

        nx, ny = -math.sin(th), math.cos(th)
        return float(xc + nx * ey), float(yc + ny * ey)


# ===============================
# Main Evaluator MPC Node
# ===============================

class CenterlineEvaluatorMPC(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        qos = QoSProfile(depth=10)

        # Basic Params
        self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv')
        self.declare_parameter('namespace', '')
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('N', 20)                      # Recommended: 2.0 s preview at dt=0.1

        # Limits & Vehicle Physics
        self.declare_parameter('vx_bounds', [0.8, 8.0])
        self.declare_parameter('vy_abs_max', 4.0)
        self.declare_parameter('wz_abs_max', 10.0)            # Allow sharp turns physically
        self.declare_parameter('e_psi_abs_max', 1.5)
        self.declare_parameter('e_y_abs_max', 1.5)
        self.declare_parameter('delta_bounds', [-0.35, 0.35])
        self.declare_parameter('a_bounds', [-5.5, 3.5])
        self.declare_parameter('delta_rate_max', 5.0)
        self.declare_parameter('a_rate_max', 3.5)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('ff_gain', 1.0)

        # Performance / Evaluation Params
        self.declare_parameter('a_lat_ref_max', 1.8)          # Start stable; raise only after tracking is clean
        self.declare_parameter('curv_speed_floor', 0.03)

        # ==========================================
        # MPC MASTER TUNING KNOBS (Dimensionality Reduction)
        # ==========================================
        self.declare_parameter('tune_q_ey', 35.0)             # Lateral tracking
        self.declare_parameter('tune_q_epsi', 14.0)           # Heading tracking; do not lock this too low
        self.declare_parameter('tune_r_delta', 1.2)           # Steering effort
        self.declare_parameter('tune_rho_du', 8.0)            # Steering/throttle smoothness
        
        self.declare_parameter('w_slack', 10000.0)

        self.declare_parameter('viz_frame', 'map')
        self.declare_parameter('viz_ns', 'mpc')

        # Read logic params
        self.ns = str(self.get_parameter('namespace').value)
        self.dt = float(self.get_parameter('dt').value)
        self.N = int(self.get_parameter('N').value)
        self.vx_min, self.vx_max = self.get_parameter('vx_bounds').value
        self.vy_abs_max = float(self.get_parameter('vy_abs_max').value)
        self.wz_abs_max = float(self.get_parameter('wz_abs_max').value)
        self.epsi_abs_max = float(self.get_parameter('e_psi_abs_max').value)
        self.ey_abs_max = float(self.get_parameter('e_y_abs_max').value)
        self.delta_min, self.delta_max = self.get_parameter('delta_bounds').value
        self.a_min, self.a_max = self.get_parameter('a_bounds').value
        self.delta_rate_max = float(self.get_parameter('delta_rate_max').value)
        self.a_rate_max = float(self.get_parameter('a_rate_max').value)
        self.L_wb = float(self.get_parameter('wheelbase').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)
        self.ff_gain = float(self.get_parameter('ff_gain').value)
        self.a_lat_ref_max = float(self.get_parameter('a_lat_ref_max').value)
        self.curv_speed_floor = float(self.get_parameter('curv_speed_floor').value)

        # ==========================================
        # THE RATIO LOCKER (Internal Math)
        # ==========================================
        base_q_ey = float(self.get_parameter('tune_q_ey').value)
        base_q_epsi = float(self.get_parameter('tune_q_epsi').value)
        base_r_delta = float(self.get_parameter('tune_r_delta').value)
        self.rho_du = float(self.get_parameter('tune_rho_du').value)

        # Safe baseline weights for non-critical states (kept low so they don't interfere)
        safe_q_vx, safe_q_vy, safe_q_wz = 1.0, 0.1, 0.05
        safe_r_a = 0.1

        # Running Costs Matrix (Q)
        self.Q = np.diag([
            safe_q_vx, 
            safe_q_vy, 
            safe_q_wz, 
            base_q_epsi,       # Heading penalty must be strong enough in corners
            0.0, 
            base_q_ey         # Lateral penalty
        ])
        
        # Terminal Costs Matrix (Qf)
        self.Qf = np.diag([
            safe_q_vx * 1.5, 
            safe_q_vy * 1.5, 
            safe_q_wz * 1.5,
            base_q_epsi * 2.0, 
            0.0,
            base_q_ey * 1.3
        ])
        
        # Input Costs Matrix (R)
        self.R = np.diag([base_r_delta, safe_r_a])
        self.w_slack = float(self.get_parameter('w_slack').value)

        self.viz_frame = str(self.get_parameter('viz_frame').value)
        self.viz_ns = str(self.get_parameter('viz_ns').value)

        # Setup Track & Evaluator
        track_csv = str(self.get_parameter('track_csv').value)
        self.track = self._load_track(track_csv)
        self.frenet = FrenetProjector(self.track)
        
        self.theoretical_lap_time = self._calculate_theoretical_lap_time()
        self.get_logger().info(f"Track loaded: L = {self.track.L:.2f}m")
        self.get_logger().info(f"Theoretical Fastest Centerline Lap: {self.theoretical_lap_time:.2f} seconds")

        # Internal State & Telemetry Storage
        self.x_meas = np.zeros(6)
        self.have_pose = False
        self.speed_cmd = 0.0
        self.u_prev = np.zeros(2)
        self.prev_osqp_x = None
        self.prev_osqp_y = None

        self._reset_telemetry()
        self.lap_count = 0

        # ROS Comm
        topic_cmd = f'{self.ns}/drive' if self.ns else 'drive'
        topic_odom = f'{self.ns}/odom' if self.ns else 'odom'

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, topic_cmd, qos)
        self.sub_odom = self.create_subscription(Odometry, topic_odom, self.odom_cb, qos)
        self.pub_path = self.create_publisher(Path, f'{self.viz_ns}/pred_path', qos)
        self.pub_mark = self.create_publisher(Marker, f'{self.viz_ns}/pred_marker', qos)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("Telemetry Evaluator MPC node initialized. Drive to trigger evaluation!")

    # ==========================================
    # EVALUATION & TELEMETRY LOGIC
    # ==========================================

    def _calculate_theoretical_lap_time(self) -> float:
        """Integrates ds/v_max over the entire track to find the optimal centerline time."""
        t_total = 0.0
        for i in range(len(self.track.s) - 1):
            ds = float(self.track.s[i+1] - self.track.s[i])
            kappa = abs(self.track.kappa[i])
            kappa = max(kappa, self.curv_speed_floor)
            v = math.sqrt(max(0.1, self.a_lat_ref_max / kappa))
            v = sat(v, self.vx_min, self.vx_max)
            t_total += ds / v
        
        # Add wrap-around gap
        ds_wrap = float(self.track.L - self.track.s[-1])
        if ds_wrap > 0:
            kappa = max(abs(self.track.kappa[-1]), self.curv_speed_floor)
            v = sat(math.sqrt(max(0.1, self.a_lat_ref_max / kappa)), self.vx_min, self.vx_max)
            t_total += ds_wrap / v
            
        return t_total

    def _reset_telemetry(self):
        self.lap_start_time = None
        self.sum_sq_ey = 0.0
        self.sum_sq_epsi = 0.0
        self.sum_sq_delta_rate = 0.0
        self.step_count = 0
        self.prev_s_val = None

    def _update_telemetry(self, current_s: float, e_y: float, e_psi: float, delta: float):
        if self.prev_s_val is None:
            self.prev_s_val = current_s
            self.lap_start_time = time.time()
            return

        # Check for Lap Wrap-Around (crossed start/finish line)
        if self.prev_s_val > self.track.L * 0.8 and current_s < self.track.L * 0.2:
            self._print_tuning_report()
            self._reset_telemetry()
            self.prev_s_val = current_s
            self.lap_start_time = time.time()
            self.lap_count += 1
            return

        self.prev_s_val = current_s
        
        # Accumulate metrics for RMSE
        self.sum_sq_ey += e_y**2
        self.sum_sq_epsi += e_psi**2
        
        delta_rate = (delta - self.u_prev[0]) / self.dt
        self.sum_sq_delta_rate += delta_rate**2
        
        self.step_count += 1

    def _print_tuning_report(self):
        if self.step_count == 0 or self.lap_start_time is None: return
        
        actual_lap_time = time.time() - self.lap_start_time
        efficiency = (self.theoretical_lap_time / actual_lap_time) * 100.0
        
        rmse_ey = math.sqrt(self.sum_sq_ey / self.step_count)
        rmse_epsi = math.sqrt(self.sum_sq_epsi / self.step_count)
        rmse_chatter = math.sqrt(self.sum_sq_delta_rate / self.step_count)

        # Build the report as a list of strings so we can log it all at once cleanly
        report = []
        report.append("="*50)
        report.append(f"🏁 LAP {self.lap_count + 1} TUNING REPORT 🏁")
        report.append("="*50)
        report.append(f"⏱️ Pace Efficiency:   {efficiency:.1f}% of theoretical limit")
        report.append(f"   - Target Time:    {self.theoretical_lap_time:.2f}s")
        report.append(f"   - Actual Time:    {actual_lap_time:.2f}s")
        report.append("🎯 Tracking Accuracy (Lower is better):")
        report.append(f"   - e_y RMSE:       {rmse_ey:.4f} m   (Goal: < 0.05m)")
        report.append(f"   - e_psi RMSE:     {rmse_epsi:.4f} rad (Goal: < 0.05 rad)")
        report.append("⚙️ Control Smoothness (Lower is better):")
        report.append(f"   - Steer Chatter:  {rmse_chatter:.4f} rad/s (Goal: < 0.5 rad/s)")
        report.append("="*50)
        
        # Quick tuning advice engine
        if efficiency < 85.0:
            report.append("💡 TIP: Braking too early / losing speed. Check a_bounds and Q_vx.")
        if rmse_ey > 0.08:
            report.append(f"💡 TIP: Lateral error is high. Increase q_ey (currently {self.Q[5,5]:.1f}) or horizon N.")
        if rmse_chatter > 1.0:
            report.append("💡 TIP: Steering is jagged. Increase r_delta, rho_du, or lower q_epsi.")
        report.append("="*50)

        # Output the entire block using the native ROS 2 logger
        self.get_logger().info("\n" + "\n".join(report))

    # ==========================================
    # MPC CORE LOGIC
    # ==========================================

    def _load_track(self, path: str) -> Track:
        df = pd.read_csv(path)
        def canon(name: str) -> str: return name.strip().lstrip('#').strip().lower()
        cols = {canon(c): c for c in df.columns}

        def get_any(names):
            for n in names:
                if n in cols: return df[cols[n]].to_numpy(dtype=float)
            return None

        s = get_any(['s', 's_m', 'arc_length', 'arclength'])
        x = get_any(['x', 'x_m'])
        y = get_any(['y', 'y_m'])
        theta = get_any(['theta', 'theta_rad', 'heading'])
        kappa = get_any(['kappa', 'curvature', 'k'])

        if s is None and x is not None and y is not None:
            dx, dy = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
            s = np.cumsum(np.hypot(dx, dy))
            s -= s[0]

        if theta is None and x is not None and y is not None:
            gx, gy = np.gradient(x), np.gradient(y)
            theta = np.unwrap(np.arctan2(gy, gx))

        if kappa is None and theta is not None and s is not None:
            dth, ds = np.gradient(theta), np.gradient(s)
            ds[np.abs(ds) < 1e-6] = 1e-6
            kappa = dth / ds

        order = np.argsort(s)
        s, kappa = s[order] - s[order][0], kappa[order]
        x = x[order] if x is not None else None
        y = y[order] if y is not None else None
        theta = theta[order] if theta is not None else None

        return Track(s=s.astype(float), kappa=kappa.astype(float), x=x, y=y, theta=theta)

    def odom_cb(self, msg: Odometry):
        px, py = float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        s_cur, e_y, e_psi = self.frenet.project(px, py, yaw)

        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)

        self.x_meas = np.array([vx, vy, wz, e_psi, s_cur, e_y], dtype=float)
        self.have_pose = True

    def build_reference_horizon(self, x0: np.ndarray):
        xref = np.zeros((self.N + 1, 6), dtype=float)
        uref = np.zeros((self.N, 2), dtype=float)

        s0 = float(x0[4])
        s_ref = [s0]
        
        def target_velocity(s_val: float) -> float:
            kappa = abs(self.track.kappa_at(s_val))
            kappa = max(kappa, self.curv_speed_floor)
            vref = math.sqrt(max(0.1, self.a_lat_ref_max / kappa))
            return sat(vref, self.vx_min, self.vx_max)

        vx_ref = [target_velocity(s0)]

        for k in range(1, self.N + 1):
            s_next = s_ref[-1] + max(vx_ref[-1], self.vx_min) * self.dt
            s_ref.append(s_next)
            vx_ref.append(target_velocity(s_next))

        for k in range(self.N + 1):
            s_k = float(s_ref[k])
            xref[k, 0] = vx_ref[k]      
            xref[k, 1] = 0.0            
            delta_ref_k = sat(self.steer_sign * self.ff_gain * math.atan(self.L_wb * self.track.kappa_at(s_k)),
                              self.delta_min, self.delta_max)
            xref[k, 2] = vx_ref[k] * math.tan(delta_ref_k) / self.L_wb
            xref[k, 3] = 0.0            
            xref[k, 4] = s_k            
            xref[k, 5] = 0.0            

        for k in range(self.N):
            kappa_k = self.track.kappa_at(s_ref[k])
            delta_ff = self.steer_sign * self.ff_gain * math.atan(self.L_wb * kappa_k)
            a_ff = sat((vx_ref[k + 1] - vx_ref[k]) / self.dt, self.a_min, self.a_max)

            uref[k, 0] = sat(delta_ff, self.delta_min, self.delta_max)
            uref[k, 1] = a_ff

        return xref, uref

    def atv_matrices(self, xbar_seq: np.ndarray, ubar_seq: np.ndarray):
        A_list, B_list, C_list = [], [], []

        for k in range(self.N):
            xref, uref = xbar_seq[k], ubar_seq[k]
            vx, vy, wz, epsi, s, ey = xref
            delta_ref = uref[0]

            kappa = self.track.kappa_at(s)
            den = max(1e-3, 1.0 - kappa * ey)

            cos_e, sin_e = math.cos(epsi), math.sin(epsi)
            tan_del, cos_del = math.tan(delta_ref), math.cos(delta_ref)
            sec2_del = 1.0 / max(1e-4, cos_del * cos_del)

            wz_alg = vx * tan_del / self.L_wb
            s_dot = (vx * cos_e - vy * sin_e) / den
            f_epsi = wz_alg - s_dot * kappa
            f_s = s_dot
            f_ey = vx * sin_e + vy * cos_e

            dfepsi, dfs, dfey = np.zeros(6), np.zeros(6), np.zeros(6)
            dfepsi[0] = (tan_del / self.L_wb) - (cos_e / den) * kappa
            dfepsi[1] = (sin_e / den) * kappa
            dfepsi[3] = ((vx * sin_e + vy * cos_e) / den) * kappa
            dfepsi[5] = -((vx * cos_e - vy * sin_e) * kappa * kappa) / (den * den)

            dfs[0], dfs[1] = cos_e / den, -sin_e / den
            dfs[3] = (-vx * sin_e - vy * cos_e) / den
            dfs[5] = (vx * cos_e - vy * sin_e) * kappa / (den * den)

            dfey[0], dfey[1], dfey[3] = sin_e, cos_e, vx * cos_e - vy * sin_e

            A, B, C = np.zeros((6, 6)), np.zeros((6, 2)), np.zeros(6)
            A[0, 0], B[0, 1] = 1.0, self.dt

            A[2, 0] = tan_del / self.L_wb
            B[2, 0] = vx * sec2_del / self.L_wb
            C[2] = wz_alg - A[2, 0] * vx - B[2, 0] * delta_ref

            A[3, :] = np.eye(6)[3, :] + self.dt * dfepsi
            B[3, 0] = self.dt * vx * sec2_del / self.L_wb
            C[3] = self.dt * (f_epsi - dfepsi @ xref) - B[3, 0] * delta_ref

            A[4, :] = np.eye(6)[4, :] + self.dt * dfs
            C[4] = self.dt * (f_s - dfs @ xref)

            A[5, :] = np.eye(6)[5, :] + self.dt * dfey
            C[5] = self.dt * (f_ey - dfey @ xref)

            A_list.append(A); B_list.append(B); C_list.append(C)

        return A_list, B_list, C_list

    def osqp_mats(self, A_list, B_list, C_list, x0, xref, uref):
        n_x, n_u, N = 6, 2, self.N
        Zn = (N + 1) * n_x + N * n_u + N
        Xn, Un = (N + 1) * n_x, N * n_u

        def idx_slack(k_from_1: int) -> int: return Xn + Un + (k_from_1 - 1)

        P = sparse.lil_matrix((Zn, Zn))
        q = np.zeros(Zn)

        for k in range(N):
            idx = slice(k * n_x, (k + 1) * n_x)
            P[idx, idx] += 2.0 * self.dt * self.Q
            q[idx] += -2.0 * self.dt * (self.Q @ xref[k])

        idxN = slice(N * n_x, (N + 1) * n_x)
        P[idxN, idxN] += 2.0 * self.Qf
        q[idxN] += -2.0 * (self.Qf @ xref[N])

        for k in range(N):
            idxu = slice(Xn + k * n_u, Xn + (k + 1) * n_u)
            P[idxu, idxu] += 2.0 * self.dt * self.R
            q[idxu] += -2.0 * self.dt * (self.R @ uref[k])

        for k in range(1, N + 1): P[idx_slack(k), idx_slack(k)] += 2.0 * self.dt * self.w_slack

        if self.rho_du > 0.0:
            D, d = np.zeros((N * n_u, Un)), np.zeros(N * n_u)
            D[0:n_u, 0:n_u], d[0:n_u] = np.eye(n_u), self.u_prev.copy()
            for k in range(1, N):
                r0, c0, cp = k * n_u, k * n_u, (k - 1) * n_u
                D[r0:r0 + n_u, c0:c0 + n_u] = np.eye(n_u)
                D[r0:r0 + n_u, cp:cp + n_u] = -np.eye(n_u)
            P[Xn:Xn + Un, Xn:Xn + Un] += 2.0 * self.rho_du * (D.T @ D)
            q[Xn:Xn + Un] += -2.0 * self.rho_du * (D.T @ d)

        rows, cols, data, l, u = [], [], [], [], []
        def add_row(entries, low, high):
            r = len(l)
            for c, v in entries:
                rows.append(r); cols.append(c); data.append(v)
            l.append(low); u.append(high)

        for i in range(n_x): add_row([(i, 1.0)], x0[i], x0[i])

        for k in range(N):
            A, B, C = A_list[k], B_list[k], C_list[k]
            for i in range(n_x):
                row = [((k + 1) * n_x + i, 1.0)]
                for jx in range(n_x):
                    if A[i, jx] != 0.0: row.append((k * n_x + jx, -A[i, jx]))
                for ju in range(n_u):
                    if B[i, ju] != 0.0: row.append((Xn + k * n_u + ju, -B[i, ju]))
                add_row(row, C[i], C[i])

        for k in range(1, N + 1):
            add_row([(k * n_x + 0, 1.0)], self.vx_min, self.vx_max)
            add_row([(k * n_x + 1, 1.0)], -self.vy_abs_max, self.vy_abs_max)
            add_row([(k * n_x + 2, 1.0)], -self.wz_abs_max, self.wz_abs_max)
            add_row([(k * n_x + 3, 1.0)], -self.epsi_abs_max, self.epsi_abs_max)

            islk = idx_slack(k)
            add_row([(k * n_x + 5, 1.0), (islk, -1.0)], -np.inf, self.ey_abs_max)
            add_row([(k * n_x + 5, 1.0), (islk, 1.0)], -self.ey_abs_max, np.inf)
            add_row([(islk, 1.0)], 0.0, np.inf)

        for k in range(N):
            add_row([(Xn + k * n_u + 0, 1.0)], self.delta_min, self.delta_max)
            add_row([(Xn + k * n_u + 1, 1.0)], self.a_min, self.a_max)

        dmax = self.delta_rate_max * self.dt if self.delta_rate_max > 0 else None
        amax = self.a_rate_max * self.dt if self.a_rate_max > 0 else None

        if dmax is not None:
            add_row([(Xn + 0, 1.0)], self.u_prev[0] - dmax, self.u_prev[0] + dmax)
            for k in range(1, N): add_row([(Xn + k * n_u + 0, 1.0), (Xn + (k - 1) * n_u + 0, -1.0)], -dmax, dmax)

        if amax is not None:
            add_row([(Xn + 1, 1.0)], self.u_prev[1] - amax, self.u_prev[1] + amax)
            for k in range(1, N): add_row([(Xn + k * n_u + 1, 1.0), (Xn + (k - 1) * n_u + 1, -1.0)], -amax, amax)

        Aqp = sparse.csc_matrix((data, (rows, cols)), shape=(len(l), Zn))
        return sparse.csc_matrix(P), q, Aqp, np.array(l), np.array(u)

    def solve_qp(self, P, q, A, l, u):
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True, eps_abs=1e-3, eps_rel=1e-3)
        try:
            if self.prev_osqp_x is not None: prob.warm_start(x=self.prev_osqp_x, y=self.prev_osqp_y)
        except: pass

        res = prob.solve()
        if res.info.status_val not in (1, 2): return None

        self.prev_osqp_x = res.x
        try: self.prev_osqp_y = res.y
        except: self.prev_osqp_y = None
        return res.x

    def control_loop(self):
        if not self.have_pose: return

        x0 = self.x_meas.copy()
        xref, uref = self.build_reference_horizon(x0)
        A_list, B_list, C_list = self.atv_matrices(xref, uref)

        P, q, Aqp, lvec, uvec = self.osqp_mats(A_list, B_list, C_list, x0, xref, uref)
        sol = self.solve_qp(P, q, Aqp, lvec, uvec)

        if sol is None:
            delta = sat(uref[0, 0] - 0.9 * (x0[5] - xref[0, 5]) - 0.8 * x0[3], self.delta_min, self.delta_max)
            a = sat(0.8 * (xref[0, 0] - self.speed_cmd), self.a_min, self.a_max)
            use_x_seq = xref
        else:
            Xn = (self.N + 1) * 6
            Uopt = sol[Xn:Xn + self.N * 2].reshape(self.N, 2)
            delta, a = float(sat(Uopt[0, 0], self.delta_min, self.delta_max)), float(sat(Uopt[0, 1], self.a_min, self.a_max))
            use_x_seq = sol[:Xn].reshape(self.N + 1, 6)

        self.speed_cmd = sat(x0[0] + (a * self.dt), self.vx_min, self.vx_max)

        # Update Telemetry & print lap report if needed
        self._update_telemetry(current_s=x0[4], e_y=x0[5], e_psi=x0[3], delta=delta)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle, msg.drive.speed = delta, self.speed_cmd
        self.pub_cmd.publish(msg)

        self.u_prev = np.array([delta, a], dtype=float)
        self.publish_preview(use_x_seq)

    def publish_preview(self, Xseq: np.ndarray):
        try:
            path = Path()
            path.header.frame_id = self.viz_frame
            path.header.stamp = self.get_clock().now().to_msg()
            pts_xy = []

            for k in range(Xseq.shape[0]):
                xg, yg = self.frenet.frenet_to_global(float(Xseq[k, 4]), float(Xseq[k, 5]))
                pts_xy.append((xg, yg))
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y, ps.pose.orientation.w = float(xg), float(yg), 1.0
                path.poses.append(ps)
                
            self.pub_path.publish(path)

            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = path.header, self.viz_ns, 0, Marker.LINE_STRIP, Marker.ADD
            m.scale.x, m.color.a, m.color.r, m.color.g, m.color.b = 0.03, 1.0, 0.1, 0.9, 0.1
            m.points = [Point(x=float(xg), y=float(yg), z=0.02) for xg, yg in pts_xy]
            self.pub_mark.publish(m)
        except Exception as e:
            self.get_logger().warn(f"viz failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CenterlineEvaluatorMPC()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        try:
            stop = AckermannDriveStamped()
            stop.drive.steering_angle, stop.drive.speed = 0.0, 0.0
            node.pub_cmd.publish(stop)
        except: pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

