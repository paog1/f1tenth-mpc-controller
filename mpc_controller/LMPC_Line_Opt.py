#!/usr/bin/env python3
"""
Iterative Racing-Line Learning MPC for F1TENTH (ROS 2 / rclpy)

Main changes in this version:
- Track-centric ILC approach
- Dynamic Curvature Calculation for velocity profile generation
- Strict Discrete Kinematic Bicycle Model for MPC prediction
- Lowered lateral error weights to promote natural corner cutting
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import copy

import numpy as np
import osqp
import pandas as pd
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from scipy import sparse
import scipy.interpolate as interp

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker


# ===============================
# Utilities
# ===============================

def sat(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def circ_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='wrap')
    ker = np.ones(w, dtype=float) / float(w)
    y = np.convolve(xp, ker, mode='same')
    return y[pad:-pad]


# ===============================
# Track representation
# ===============================

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
        """Periodic interpolation on a closed track."""
        L = self.L if self.L > 0 else 1.0
        s_wrapped = s_query % L
        idx = np.searchsorted(self.s, s_wrapped)

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

        a0 = float(arr[i0])
        a1 = float(arr[i1])

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

    def theta_at(self, s_query: float) -> Optional[float]:
        if self.theta is None:
            return None
        return float(self._interp_pair(self.theta, s_query))

    def widths_at(self, s_query: float, margin: float = 0.0) -> Tuple[Optional[float], Optional[float]]:
        if self.w_left is None or self.w_right is None:
            return None, None
        wl = float(self._interp_pair(self.w_left, s_query))
        wr = float(self._interp_pair(self.w_right, s_query))
        return max(0.0, wl - margin), max(0.0, wr - margin)


# ===============================
# Frenet projector
# ===============================

class FrenetProjector:
    def __init__(self, track: Track):
        self.track = track
        self.has_xy = (track.x is not None) and (track.y is not None) and (track.theta is not None)
        self.last_idx = 0

    def _segment_project(self, px: float, py: float, i0: int, i1: int):
            tx0, ty0 = float(self.track.x[i0]), float(self.track.y[i0])
            tx1, ty1 = float(self.track.x[i1]), float(self.track.y[i1])

            vx, vy          = tx1 - tx0, ty1 - ty0
            seg_len2        = vx*vx + vy*vy
            if seg_len2 < 1e-9:                        
                return 1e9, 0.0, 0.0, i0

            t               = sat(((px-tx0)*vx + (py-ty0)*vy)/seg_len2, 0.0, 1.0)
            xproj, yproj    = tx0 + t*vx, ty0 + t*vy
            dx,    dy       = px  - xproj, py  - yproj
            dist2           = dx*dx + dy*dy

            s0, s1          = float(self.track.s[i0]), float(self.track.s[i1])
            if i1 == 0 and i0 == len(self.track.s)-1:   
                s1 += self.track.L
            sproj           = (s0 + t*(s1-s0)) % self.track.L

            th              = float(self.track._interp_pair(self.track.theta, sproj))
            nx,   ny        = -math.sin(th),  math.cos(th)
            ey              = (px-xproj)*nx + (py-yproj)*ny

            return dist2, sproj, ey, i0

    def project(self, x: float, y: float, yaw: float):
        if not self.has_xy:
            return x, y, yaw

        n = len(self.track.s)
        if n < 2:
            return 0.0, 0.0, 0.0

        best = None
        for off in range(-25, 26):                 
            i0        = (self.last_idx + off) % n
            i1        = (i0 + 1) % n
            cand      = self._segment_project(x, y, i0, i1)
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

    def frenet_to_global(self, s: float, ey: float) -> Tuple[float, float]:
        if not self.has_xy:
            return s, ey

        xc = self.track._interp_pair(self.track.x, s)
        yc = self.track._interp_pair(self.track.y, s)
        th = self.track._interp_pair(self.track.theta, s)

        nx, ny = -math.sin(th), math.cos(th)
        return float(xc + nx * ey), float(yc + ny * ey)


# ===============================
# Racing line manager
# ===============================

class RacingLineManager:
    """Lap-to-lap racing-line learner in Frenet e_y(s) with Dynamic Curvature."""

    def __init__(
        self,
        track: Track,
        frenet: FrenetProjector,  # Added Frenet Projector to convert to Global X,Y
        ey_margin: float,
        log_dir: str,
        seed_laps: int,
        exploration_amp: float,
        exploration_sectors: int,
        sectors_to_perturb: int,
        accept_margin_sec: float,
        line_blend: float,
        smooth_window: int,
        mincurv_w_curv: float,
        mincurv_w_smooth: float,
        mincurv_w_data: float,
        mincurv_w_prev: float,
        mincurv_enable: bool = True,
    ):
        self.track = track
        self.frenet = frenet
        self.ey_margin = ey_margin
        self.seed_laps = seed_laps
        self.exploration_amp = exploration_amp
        self.exploration_sectors = exploration_sectors
        self.sectors_to_perturb = sectors_to_perturb
        self.accept_margin_sec = accept_margin_sec
        self.line_blend = line_blend
        self.smooth_window = smooth_window
        self.log_dir = log_dir

        self.mincurv_enable = mincurv_enable
        self.mincurv_w_curv = mincurv_w_curv
        self.mincurv_w_smooth = mincurv_w_smooth
        self.mincurv_w_data = mincurv_w_data
        self.mincurv_w_prev = mincurv_w_prev

        self.best_line = np.zeros_like(track.s, dtype=float)
        self.candidate_line = self.best_line.copy()
        self.candidate_active = False

        self.best_lap_time = np.inf
        self.best_lap_idx = -1
        
        # Initialize dynamic curvature with standard centerline curvature
        self.dynamic_kappa = np.abs(track.kappa.copy())

        os.makedirs(self.log_dir, exist_ok=True)
        self._save_line_csv(os.path.join(self.log_dir, "best_line_init.csv"), self.best_line)

    def line_at(self, s_query: float) -> float:
        line = self.candidate_line if self.candidate_active else self.best_line
        return float(self.track._interp_pair(line, s_query))

    def active_line(self) -> np.ndarray:
        return self.candidate_line if self.candidate_active else self.best_line
        
    def dynamic_kappa_at(self, s_query: float) -> float:
        """Returns the dynamic curvature of the actively learned path."""
        return float(self.track._interp_pair(self.dynamic_kappa, s_query))

    def _line_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.track.s)
        lo = -self.ey_abs_fallback() * np.ones(n, dtype=float)
        hi = self.ey_abs_fallback() * np.ones(n, dtype=float)

        if self.track.w_left is None or self.track.w_right is None:
            return lo, hi

        for i, s in enumerate(self.track.s):
            wl, wr = self.track.widths_at(float(s), margin=self.ey_margin)
            lo[i] = -wr
            hi[i] = wl
        return lo, hi

    def ey_abs_fallback(self) -> float:
        return 1.5

    def _clip_to_walls(self, line: np.ndarray) -> np.ndarray:
        lo, hi = self._line_bounds()
        return np.minimum(np.maximum(line.copy(), lo), hi)

    def _save_line_csv(self, path: str, line: np.ndarray):
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['s', 'ey_ref'])
            for s, ey in zip(self.track.s, line):
                wr.writerow([float(s), float(ey)])
     
    def _update_dynamic_curvature(self):
        """Calculates true geometric curvature of the current racing line."""
        line = self.active_line()
        n = len(self.track.s)
        X = np.zeros(n)
        Y = np.zeros(n)
        
        # Convert Frenet to Global
        for i, s_val in enumerate(self.track.s):
            X[i], Y[i] = self.frenet.frenet_to_global(float(s_val), float(line[i]))
            
        # FIX 1: Smooth the spatial coordinates first to remove micro-wobbles
        X = circ_moving_average(X, self.smooth_window)
        Y = circ_moving_average(Y, self.smooth_window)

        ds = np.gradient(self.track.s)
        ds[np.abs(ds) < 1e-6] = 1e-6  
        
        Xp = np.gradient(X) / ds
        Yp = np.gradient(Y) / ds
        Xpp = np.gradient(Xp) / ds
        Ypp = np.gradient(Yp) / ds
        
        denom = Xp**2 + Yp**2
        denom[denom < 1e-6] = 1e-6
        kappa = (Xp * Ypp - Yp * Xpp) / np.power(denom, 1.5)
        
        # FIX 2: Heavily smooth the final curvature to eliminate brake-check spikes
        # self.dynamic_kappa = circ_moving_average(np.abs(kappa), self.smooth_window * 2)
        self.dynamic_kappa = circ_moving_average(np.abs(kappa), self.smooth_window)

    def _executed_line_from_lap(self, X: np.ndarray) -> np.ndarray:
        line = self.active_line().copy()
        if X.shape[0] < 10:
            return line

        bins = np.zeros_like(self.track.s, dtype=float)
        cnts = np.zeros_like(self.track.s, dtype=float)

        ds_nom = float(np.median(np.diff(self.track.s))) if len(self.track.s) > 2 else 0.1
        max_accept_dist = max(0.25, 4.0 * ds_nom)

        for row in X:
            s_val = float(row[4] % self.track.L)
            ey_val = float(row[5])
            idx = int(np.argmin(np.abs(self.track.s - s_val)))
            if abs(float(self.track.s[idx]) - s_val) <= max_accept_dist:
                bins[idx] += ey_val
                cnts[idx] += 1.0

        mask = cnts > 0.0
        if np.any(mask):
            line[mask] = bins[mask] / cnts[mask]

        line = circ_moving_average(line, self.smooth_window)
        line = self._clip_to_walls(line)
        return line

    def _periodic_diff_matrix(self, n: int, order: int) -> sparse.csc_matrix:
        rows, cols, data = [], [], []
        if order == 1:
            for i in range(n):
                rows += [i, i]
                cols += [i, (i + 1) % n]
                data += [-1.0, 1.0]
        elif order == 2:
            for i in range(n):
                rows += [i, i, i]
                cols += [(i - 1) % n, i, (i + 1) % n]
                data += [1.0, -2.0, 1.0]
        else:
            raise ValueError("Only first and second periodic differences are supported.")
        return sparse.csc_matrix((data, (rows, cols)), shape=(n, n))

    def _minimum_curvature_qp(self, data_line: np.ndarray, prev_line: np.ndarray) -> np.ndarray:
        if not self.mincurv_enable:
            return self._clip_to_walls(circ_moving_average(data_line, self.smooth_window))

        n = len(self.track.s)
        if n < 5:
            return self._clip_to_walls(data_line)

        D1 = self._periodic_diff_matrix(n, order=1)
        D2 = self._periodic_diff_matrix(n, order=2)
        I = sparse.eye(n, format='csc')

        # --- SPATIALLY VARYING WEIGHTS (The "Black/Red/Green Dot" Math) ---
        # 1. We heavily smooth the absolute track curvature to identify the wide 
        # "Corner Zones" (encompassing your entry, apex, and exit dots).
        macro_kappa = circ_moving_average(np.abs(self.track.kappa), int(self.smooth_window * 4))
        
        # 2. Normalize it so the sharpest point is 1.0, and straights are 0.0
        max_k = np.max(macro_kappa) + 1e-6
        norm_k = macro_kappa / max_k
        
        # 3. Create the Spatial Weight Array
        # When norm_k is 0 (Straights), the weight is HIGH (Black dots pinned).
        # When norm_k is ~1 (Corners), the weight drops to 0.0 (Red/Green dots free to float).
        w_data_array = self.mincurv_w_data * (1.0 - norm_k)
        
        # Convert array to a sparse diagonal matrix for the QP
        W_data_mat = sparse.diags(w_data_array, format='csc')

        # --- THE OPTIMIZATION MATRICES ---
        P = (
            self.mincurv_w_curv * (D2.T @ D2)
            + self.mincurv_w_smooth * (D1.T @ D1)
            + W_data_mat 
            + self.mincurv_w_prev * I
        ).tocsc()
        
        # 4. Inject the track curvature to PUSH the free dots into the Out-In-Out shape
        clean_kappa = circ_moving_average(self.track.kappa, self.smooth_window)
        q_curv = self.mincurv_w_curv * (D2.T @ clean_kappa)
        
        # The q vector also uses the spatial array now
        q = q_curv - (w_data_array * data_line) - (self.mincurv_w_prev * prev_line)

        lo, hi = self._line_bounds()
        prob = osqp.OSQP()
        try:
            prob.setup(P=P, q=q, A=I, l=lo, u=hi, verbose=False, warm_start=True,
                       eps_abs=1e-4, eps_rel=1e-4, max_iter=10000, polish=True)
            res = prob.solve()
            if res.info.status_val in (1, 2):
                out = np.asarray(res.x, dtype=float)
            else:
                out = circ_moving_average(data_line, self.smooth_window)
        except Exception:
            out = circ_moving_average(data_line, self.smooth_window)

        return self._clip_to_walls(out)
    
    def prepare_next_candidate(self, stored_laps: int):
        if stored_laps < max(2, self.seed_laps):
            self.candidate_active = True
            self.candidate_line = self.best_line.copy()
            self._update_dynamic_curvature()
            return

        base = self.best_line.copy()

        # --- 1. DECAYING EXPLORATION (Simulated Annealing) ---
        # Reduce the shift amplitude as we complete more laps to fine-tune the line.
        decay_rate = 0.85
        laps_learning = stored_laps - self.seed_laps
        current_amp = self.exploration_amp * (decay_rate ** laps_learning)
        
        # Enforce a minimum exploration amplitude so it never completely stops trying to improve
        current_amp = max(current_amp, 0.2)

        # --- 2. BOTTLENECK TARGETING (Using Dynamic Curvature) ---
        # We need the sign of the track to know left/right, but the MAGNITUDE of our 
        # current dynamically learned line to know where the actual sharpest points remain.
        signed_dyn_kappa = self.dynamic_kappa * np.sign(self.track.kappa)
        
        physical_k_limit = 0.6
        normalized_kappa = np.clip(signed_dyn_kappa / physical_k_limit, -1.0, 1.0)

        # --- 3. SECTOR-BASED RANDOMIZED MUTATION ---
        # Instead of shifting every corner on the track, let's heavily mutate only a portion of the track.
        # This isolates variables so if the lap time improves, we know exactly why.
        
        n_points = len(self.track.s)
        mutation_mask = np.ones(n_points)
        
        # Every lap, we pick a random "Sector" of the track to experiment on (e.g., 30% of the track)
        sector_start_idx = np.random.randint(0, n_points)
        sector_length_idx = int(n_points * 0.3)
        
        # Create a smoothed window mask to isolate the exploration
        for i in range(n_points):
            # Calculate shortest distance around the periodic array
            dist = min(abs(i - sector_start_idx), n_points - abs(i - sector_start_idx))
            if dist > sector_length_idx / 2:
                mutation_mask[i] = 0.0
        
        # Smooth the mask so the shift blends nicely into the rest of the track
        mutation_mask = circ_moving_average(mutation_mask, int(self.smooth_window * 2))

        # --- 4. THE OUT-IN-OUT ECHO MATH (Applied dynamically) ---
        apex_pull = current_amp * (normalized_kappa ** 3)

        ds_nom = float(np.median(np.diff(self.track.s))) if len(self.track.s) > 2 else 0.1
        shift_idx = int(5.0 / ds_nom) 

        entry_push = -0.6 * np.roll(apex_pull, -shift_idx)
        exit_push  = -0.6 * np.roll(apex_pull, shift_idx)

        raw_shift = (apex_pull + entry_push + exit_push) * mutation_mask

        # --- 5. SMEAR & SMOOTH ---
        shift = circ_moving_average(raw_shift, int(self.smooth_window * 2))
        cand = base + shift

        # Pass it through the QP Smoother to ensure kinematic feasibility
        cand = self._minimum_curvature_qp(cand, base)
        
        self.candidate_line = cand
        self.candidate_active = True
        self._update_dynamic_curvature() 
        
        self._save_line_csv(os.path.join(self.log_dir, f"candidate_line_lap_{stored_laps:03d}.csv"), cand)
        print(f"Generated Iterative candidate line for lap {stored_laps} with amp {current_amp:.2f} at sector {sector_start_idx}")

    def finish_lap(
        self,
        lap_idx: int,
        stored_laps: int,
        lap_time: float,
        valid_lap: bool,
        X_lap: np.ndarray,
    ) -> Tuple[bool, float]:
        accepted = False
        executed_line = self._executed_line_from_lap(X_lap)
        prev_best = self.best_line.copy()

        if valid_lap:
            data_target = self.line_blend * executed_line + (1.0 - self.line_blend) * self.active_line()
            learned_line = self._minimum_curvature_qp(data_target, prev_best)

            if not np.isfinite(self.best_lap_time):
                self.best_line = learned_line
                self.best_lap_time = lap_time
                self.best_lap_idx = lap_idx
                accepted = True
            elif stored_laps <= self.seed_laps:
                if lap_time < self.best_lap_time:
                    self.best_line = learned_line
                    self.best_lap_time = lap_time
                    self.best_lap_idx = lap_idx
                    accepted = True
            else:
                if lap_time <= (self.best_lap_time + self.accept_margin_sec):
                    self.best_line = learned_line
                    if lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time
                        self.best_lap_idx = lap_idx
                    accepted = True

        self.candidate_active = False
        self.candidate_line = self.best_line.copy()
        
        # Keep dynamic curvature accurately updated
        self._update_dynamic_curvature()

        self._save_line_csv(os.path.join(self.log_dir, f"best_line_after_lap_{lap_idx:03d}.csv"), self.best_line)
        return accepted, float(self.best_lap_time)


class VelocityProfileManager:
    """Velocity learning based on the Dynamic Curvature of the learned path."""

    def __init__(
        self,
        track: Track,
        line_mgr: RacingLineManager, # Passed in to access dynamic curvature
        log_dir: str,
        vx_min: float,
        vx_max: float,
        a_lat_ref_max: float,
        curv_speed_floor: float,
        speed_ref_scale: float,
        enable: bool = True,
        blend: float = 0.35,
        smooth_window: int = 21,
        accept_margin_sec: float = 0.20,
        data_margin: float = 0.15,
        max_speed_step: float = 0.25,
    ):
        self.track = track
        self.line_mgr = line_mgr
        self.log_dir = log_dir
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.a_lat_ref_max = a_lat_ref_max
        self.curv_speed_floor = curv_speed_floor
        self.speed_ref_scale = speed_ref_scale
        self.enable = enable
        self.blend = blend
        self.smooth_window = smooth_window
        self.accept_margin_sec = accept_margin_sec
        self.data_margin = data_margin
        self.max_speed_step = max_speed_step

        self.base_speed = np.array([self.curvature_speed_at(float(s)) for s in self.track.s], dtype=float)
        self.learned_speed = self.base_speed.copy()
        self.best_lap_time = np.inf
        self.num_updates = 0

        os.makedirs(self.log_dir, exist_ok=True)
        self._save_speed_csv(os.path.join(self.log_dir, "speed_profile_init.csv"), self.learned_speed)

    def curvature_speed_at(self, s_val: float) -> float:
        # Crucial fix: Use the dynamic curvature of the cut line, not the centerline!
        kappa = self.line_mgr.dynamic_kappa_at(s_val)
        kappa = max(kappa, self.curv_speed_floor)
        vref = self.speed_ref_scale * math.sqrt(max(0.1, self.a_lat_ref_max / kappa))
        return sat(vref, self.vx_min, self.vx_max)

    def speed_at(self, s_query: float) -> float:
        if not self.enable or self.num_updates == 0:
            return self.curvature_speed_at(s_query)
        learned = float(self.track._interp_pair(self.learned_speed, s_query))
        safe = self.curvature_speed_at(s_query)
        return sat(min(learned, safe + self.data_margin), self.vx_min, self.vx_max)

    def _save_speed_csv(self, path: str, speed: np.ndarray):
        with open(path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['s', 'vx_ref'])
            for s, v in zip(self.track.s, speed):
                wr.writerow([float(s), float(v)])

    def _profile_from_lap(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        profile = self.learned_speed.copy()
        counts = np.zeros_like(self.track.s, dtype=float)
        sums = np.zeros_like(self.track.s, dtype=float)

        if X.shape[0] < 10:
            return profile, counts

        for row in X:
            s_val = float(row[4] % self.track.L)
            vx_val = sat(float(row[0]), self.vx_min, self.vx_max)
            idx = int(np.argmin(np.abs(self.track.s - s_val)))
            sums[idx] += vx_val
            counts[idx] += 1.0

        mask = counts > 0.0
        if np.any(mask):
            profile[mask] = sums[mask] / counts[mask]

        profile = circ_moving_average(profile, self.smooth_window)
        for i, s in enumerate(self.track.s):
            profile[i] = sat(min(profile[i], self.curvature_speed_at(float(s)) + self.data_margin), self.vx_min, self.vx_max)
        return profile, counts

    def update_from_lap(self, lap_idx: int, lap_time: float, valid_lap: bool, X_lap: np.ndarray) -> bool:
        if not self.enable or not valid_lap:
            return False

        if np.isfinite(self.best_lap_time) and lap_time > self.best_lap_time + self.accept_margin_sec:
            return False

        measured, counts = self._profile_from_lap(X_lap)
        if np.count_nonzero(counts) < max(5, int(0.15 * len(self.track.s))):
            return False

        target = measured
        raw_update = (1.0 - self.blend) * self.learned_speed + self.blend * target

        delta = np.clip(raw_update - self.learned_speed, -self.max_speed_step, self.max_speed_step)
        self.learned_speed = self.learned_speed + delta
        self.learned_speed = circ_moving_average(self.learned_speed, self.smooth_window)

        for i, s in enumerate(self.track.s):
            self.learned_speed[i] = sat(
                min(self.learned_speed[i], self.curvature_speed_at(float(s)) + self.data_margin),
                self.vx_min,
                self.vx_max,
            )

        if lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
        self.num_updates += 1
        self._save_speed_csv(os.path.join(self.log_dir, f"speed_profile_after_lap_{lap_idx:03d}.csv"), self.learned_speed)
        return True


# ===============================
# Main node
# ===============================

class IterativeRacingLineMPC(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        qos = QoSProfile(depth=10)

        # ------------
        # Parameters
        # ------------
        # self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_centerline.csv')
        self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv')
        # self.declare_parameter('track_csv', '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/columbia_small_centerline_smooth.csv')
        self.declare_parameter('namespace', '')

        self.declare_parameter('dt', 0.1)
        self.declare_parameter('N', 15)

        self.declare_parameter('vx_bounds', [0.8, 6.0])
        self.declare_parameter('vy_abs_max', 3.0)
        self.declare_parameter('wz_abs_max', 6.0)
        self.declare_parameter('e_psi_abs_max', 1.5)
        self.declare_parameter('e_y_abs_max', 1.5)

        self.declare_parameter('delta_bounds', [-0.35, 0.35])
        self.declare_parameter('a_bounds', [-2.0, 3.5])
        self.declare_parameter('delta_rate_max', 1.0)
        self.declare_parameter('a_rate_max', 3.5)
        self.declare_parameter('rho_du', 0.35)

        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('ff_gain', 1.0)

        self.declare_parameter('do_seed_laps', True)
        self.declare_parameter('seed_laps', 1)
        # self.declare_parameter('seed_speed', 2.2)
        self.declare_parameter('seed_speed', 3.0)
        self.declare_parameter('seed_hug_speed', 1.8)
        self.declare_parameter('seed_wall_margin', 0.35)
        self.declare_parameter('seed_wall_alpha', 0.45)
        self.declare_parameter('ky_seed', 0.8)
        self.declare_parameter('kpsi_seed', 0.6)

        self.declare_parameter('ey_margin', 0.2) # 0.2

        self.declare_parameter('lap_wrap_frac', 0.5)
        self.declare_parameter('wrap_window_m', 3.0)
        self.declare_parameter('min_lap_steps', 15)

        self.declare_parameter('fail_frac_thresh', 0.20)
        self.declare_parameter('out_of_track_tol', 0.03)

        # Lowered exploration parameters (Letting MPC physics find the line naturally)
        self.declare_parameter('exploration_amp', 100)  # Reduced from 3.0 to 2.0 to prevent over-aggressive shifts
        self.declare_parameter('exploration_sectors', 8)
        self.declare_parameter('sectors_to_perturb', 10)  
        self.declare_parameter('accept_margin_sec', 2.5)
        self.declare_parameter('line_blend', 0.15)
        self.declare_parameter('line_smooth_window', 11)

        self.declare_parameter('mincurv_enable', True)
        # Increase this value for smoother trajectories, especially on high-speed tracks
        self.declare_parameter('mincurv_w_curv', 1.0)

        # Decrease if the trajectory is overly constrained and doesn't adapt well to the track.
        self.declare_parameter('mincurv_w_smooth', 10.0)

        # Decrease if the trajectory needs more freedom to optimize for curvature and smoothness
        self.declare_parameter('mincurv_w_data', 1.0)

        # Higher values ensure the new trajectory doesn't deviate significantly from the previous one
        self.declare_parameter('mincurv_w_prev', 1.0)

        # Cost weights adjusted to allow corner cutting while remaining strict on track boundaries
        self.declare_parameter('q_vx', 2.0)
        self.declare_parameter('q_vy', 0.2)
        self.declare_parameter('q_wz', 0.2)
        self.declare_parameter('q_epsi', 12.0)
        self.declare_parameter('q_s', 0.0)
        self.declare_parameter('q_ey', 10.0)  # Lowered from 10.0 to promote natural corner cutting

        self.declare_parameter('qf_vx', 4.0)
        self.declare_parameter('qf_vy', 0.5)
        self.declare_parameter('qf_wz', 0.5)
        self.declare_parameter('qf_epsi', 18.0)
        self.declare_parameter('qf_s', 0.0)
        self.declare_parameter('qf_ey', 40.0)  # Lowered from 15.0

        self.declare_parameter('r_delta', 1.0)
        self.declare_parameter('r_a', 0.3)

        self.declare_parameter('w_slack', 10000.0) # Raised drastically to prevent actual track exiting

        self.declare_parameter('a_lat_ref_max', 3.5)
        self.declare_parameter('curv_speed_floor', 0.03)
        self.declare_parameter('speed_ref_scale', 0.95)
        self.declare_parameter('velocity_learning_enable', True)
        self.declare_parameter('velocity_blend', 0.35)
        self.declare_parameter('velocity_smooth_window', 21)
        self.declare_parameter('velocity_accept_margin_sec', 2.5)
        self.declare_parameter('velocity_data_margin', 0.5)
        self.declare_parameter('velocity_max_step', 0.25)

        self.declare_parameter('log_dir', '/tmp/f1tenth_iterative_line')
        self.declare_parameter('viz_frame', 'map')
        self.declare_parameter('viz_ns', 'mpc')

        # ------------
        # Read params
        # ------------
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
        self.rho_du = float(self.get_parameter('rho_du').value)

        self.L_wb = float(self.get_parameter('wheelbase').value)
        self.steer_sign = float(self.get_parameter('steer_sign').value)
        self.ff_gain = float(self.get_parameter('ff_gain').value)

        self.do_seed = bool(self.get_parameter('do_seed_laps').value)
        self.seed_laps = int(self.get_parameter('seed_laps').value)
        self.seed_speed = float(self.get_parameter('seed_speed').value)
        self.seed_hug_speed = float(self.get_parameter('seed_hug_speed').value)
        self.seed_wall_margin = float(self.get_parameter('seed_wall_margin').value)
        self.seed_wall_alpha = float(self.get_parameter('seed_wall_alpha').value)
        self.ky_seed = float(self.get_parameter('ky_seed').value)
        self.kpsi_seed = float(self.get_parameter('kpsi_seed').value)

        self.ey_margin = float(self.get_parameter('ey_margin').value)

        self.lap_wrap_frac = float(self.get_parameter('lap_wrap_frac').value)
        self.wrap_window = float(self.get_parameter('wrap_window_m').value)
        self.min_lap_steps = int(self.get_parameter('min_lap_steps').value)

        self.fail_frac_thresh = float(self.get_parameter('fail_frac_thresh').value)
        self.out_of_track_tol = float(self.get_parameter('out_of_track_tol').value)

        self.exploration_amp = float(self.get_parameter('exploration_amp').value)
        self.exploration_sectors = int(self.get_parameter('exploration_sectors').value)
        self.sectors_to_perturb = int(self.get_parameter('sectors_to_perturb').value)
        self.accept_margin_sec = float(self.get_parameter('accept_margin_sec').value)
        self.line_blend = float(self.get_parameter('line_blend').value)
        self.line_smooth_window = int(self.get_parameter('line_smooth_window').value)

        self.mincurv_enable = bool(self.get_parameter('mincurv_enable').value)
        self.mincurv_w_curv = float(self.get_parameter('mincurv_w_curv').value)
        self.mincurv_w_smooth = float(self.get_parameter('mincurv_w_smooth').value)
        self.mincurv_w_data = float(self.get_parameter('mincurv_w_data').value)
        self.mincurv_w_prev = float(self.get_parameter('mincurv_w_prev').value)

        self.Q = np.diag([
            float(self.get_parameter('q_vx').value),
            float(self.get_parameter('q_vy').value),
            float(self.get_parameter('q_wz').value),
            float(self.get_parameter('q_epsi').value),
            float(self.get_parameter('q_s').value),
            float(self.get_parameter('q_ey').value),
        ])
        self.Qf = np.diag([
            float(self.get_parameter('qf_vx').value),
            float(self.get_parameter('qf_vy').value),
            float(self.get_parameter('qf_wz').value),
            float(self.get_parameter('qf_epsi').value),
            float(self.get_parameter('qf_s').value),
            float(self.get_parameter('qf_ey').value),
        ])
        self.R = np.diag([
            float(self.get_parameter('r_delta').value),
            float(self.get_parameter('r_a').value),
        ])
        self.w_slack = float(self.get_parameter('w_slack').value)

        self.a_lat_ref_max = float(self.get_parameter('a_lat_ref_max').value)
        self.curv_speed_floor = float(self.get_parameter('curv_speed_floor').value)
        self.speed_ref_scale = float(self.get_parameter('speed_ref_scale').value)
        self.velocity_learning_enable = bool(self.get_parameter('velocity_learning_enable').value)
        self.velocity_blend = float(self.get_parameter('velocity_blend').value)
        self.velocity_smooth_window = int(self.get_parameter('velocity_smooth_window').value)
        self.velocity_accept_margin_sec = float(self.get_parameter('velocity_accept_margin_sec').value)
        self.velocity_data_margin = float(self.get_parameter('velocity_data_margin').value)
        self.velocity_max_step = float(self.get_parameter('velocity_max_step').value)

        self.log_dir = str(self.get_parameter('log_dir').value)
        self.viz_frame = str(self.get_parameter('viz_frame').value)
        self.viz_ns = str(self.get_parameter('viz_ns').value)

        os.makedirs(self.log_dir, exist_ok=True)

        # ------------
        # Load track
        # ------------
        track_csv = str(self.get_parameter('track_csv').value)
        self.track = self._load_track(track_csv)
        self.frenet = FrenetProjector(self.track)

        self.get_logger().info(
            f"Track loaded: L={self.track.L:.2f} m | "
            f"kappa[min,max]=({float(np.min(self.track.kappa)):.3f}, {float(np.max(self.track.kappa)):.3f})"
        )

        if self.track.w_left is not None and self.track.w_right is not None:
            w_min = float(np.nanmin([np.nanmin(self.track.w_left), np.nanmin(self.track.w_right)]))
            bound = max(0.2, w_min - self.ey_margin)
            self.ey_abs_max = min(self.ey_abs_max, bound)
            self.get_logger().info(f"Track widths detected -> global |e_y| <= {self.ey_abs_max:.2f} m")

        # ------------
        # Racing line manager
        # ------------
        self.line_mgr = RacingLineManager(
            track=self.track,
            frenet=self.frenet, # Pass Frenet for Global conversions
            ey_margin=self.ey_margin,
            log_dir=self.log_dir,
            seed_laps=self.seed_laps,
            exploration_amp=self.exploration_amp,
            exploration_sectors=self.exploration_sectors,
            sectors_to_perturb=self.sectors_to_perturb,
            accept_margin_sec=self.accept_margin_sec,
            line_blend=self.line_blend,
            smooth_window=self.line_smooth_window,
            mincurv_enable=self.mincurv_enable,
            mincurv_w_curv=self.mincurv_w_curv,
            mincurv_w_smooth=self.mincurv_w_smooth,
            mincurv_w_data=self.mincurv_w_data,
            mincurv_w_prev=self.mincurv_w_prev,
        )

        self.vel_mgr = VelocityProfileManager(
            track=self.track,
            line_mgr=self.line_mgr, # Pass line manager to access dynamic curvature
            log_dir=self.log_dir,
            vx_min=self.vx_min,
            vx_max=self.vx_max,
            a_lat_ref_max=self.a_lat_ref_max,
            curv_speed_floor=self.curv_speed_floor,
            speed_ref_scale=self.speed_ref_scale,
            enable=self.velocity_learning_enable,
            blend=self.velocity_blend,
            smooth_window=self.velocity_smooth_window,
            accept_margin_sec=self.velocity_accept_margin_sec,
            data_margin=self.velocity_data_margin,
            max_speed_step=self.velocity_max_step,
        )

        # ------------
        # Internal state
        # ------------
        self.x_meas = np.zeros(6)
        self.have_pose = False

        self.speed_cmd = 0.0
        self.u_prev = np.zeros(2)

        self.prev_s = None
        self.wrap_thresh = max(0.5, self.lap_wrap_frac * self.track.L)

        self.prev_osqp_x = None
        self.prev_osqp_y = None

        self.laps: List[dict] = []
        self.curr_traj_x: List[np.ndarray] = []
        self.curr_traj_u: List[np.ndarray] = []

        self.qp_fail_in_lap = 0
        self.near_wall_in_lap = 0  
        self.out_of_track_in_lap = 0
        self.total_laps = 0
        self.current_lap_idx = 0

        self.summary_csv = os.path.join(self.log_dir, 'lap_summary.csv')
        with open(self.summary_csv, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                'lap_idx',
                'lap_time_sec',
                'stored_in_memory',
                'accepted_new_best',
                'velocity_profile_updated',
                'best_lap_time_sec',
                'fail_ratio',
                'near_wall_ratio',
                'out_of_track_ratio'
            ])

        # ------------
        # ROS I/O
        # ------------
        topic_cmd = f'{self.ns}/drive' if self.ns else 'drive'
        topic_odom = f'{self.ns}/odom' if self.ns else 'odom'

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, topic_cmd, qos)
        self.sub_odom = self.create_subscription(Odometry, topic_odom, self.odom_cb, qos)

        self.pub_path = self.create_publisher(Path, f'{self.viz_ns}/pred_path', qos)
        self.pub_mark = self.create_publisher(Marker, f'{self.viz_ns}/pred_marker', qos)
        
        # NEW: Publisher for the globally learned racing line
        self.pub_best_line = self.create_publisher(Path, f'{self.viz_ns}/best_line', qos)
        self.pub_cand_line = self.create_publisher(Path, f'{self.viz_ns}/candidate_line', qos)

        self.add_on_set_parameters_callback(self._on_param_change)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Iterative Racing-Line MPC node initialized.")
        self.get_logger().info(
            f"Seeding: {'ON' if self.do_seed else 'OFF'} | seed_laps={self.seed_laps} | "
            f"seed_speed={self.seed_speed:.2f} | seed_hug_speed={self.seed_hug_speed:.2f} | N={self.N}"
        )

    # ===============================
    # Live parameter updates
    # ===============================

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
                elif p.name == 'seed_speed' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.seed_speed = float(p.value)
                elif p.name == 'seed_hug_speed' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.seed_hug_speed = float(p.value)
                elif p.name == 'N' and p.type_ == Parameter.Type.INTEGER:
                    self.N = int(p.value)
                elif p.name == 'velocity_learning_enable':
                    self.velocity_learning_enable = bool(p.value)
                    if hasattr(self, 'vel_mgr'):
                        self.vel_mgr.enable = self.velocity_learning_enable
                elif p.name == 'speed_ref_scale' and p.type_ in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    self.speed_ref_scale = float(p.value)
                    if hasattr(self, 'vel_mgr'):
                        self.vel_mgr.speed_ref_scale = self.speed_ref_scale
                elif p.name == 'mincurv_enable':
                    self.mincurv_enable = bool(p.value)
                    if hasattr(self, 'line_mgr'):
                        self.line_mgr.mincurv_enable = self.mincurv_enable
            return SetParametersResult(successful=True)
        except Exception as e:
            self.get_logger().warn(f"Param update failed: {e}")
            return SetParametersResult(successful=False)

    # ===============================
    # Track loader
    # ===============================

    def _load_track(self, path: str) -> Track:
        df = pd.read_csv(path)

        # COLUMBIA SMALL ENABLE
        # df = df.iloc[::-1].reset_index(drop=True) 

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

    # ===============================
    # Odometry callback
    # ===============================

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

    # ===============================
    # Seeding helpers
    # ===============================

    def seed_line_offset(self, lap_idx: int, s_val: float) -> float:
        if self.track.w_left is None or self.track.w_right is None:
            return 0.0

        wl, wr = self.track.widths_at(s_val, margin=self.seed_wall_margin)
        wl = max(0.0, wl)
        wr = max(0.0, wr)

        mode = lap_idx % 5

        if mode in (0, 1, 4):
            return 0.0
        if mode == 2:
            return self.seed_wall_alpha * wl
        if mode == 3:
            return -self.seed_wall_alpha * wr
        return 0.0

    def seed_lap_label(self, lap_idx: int) -> str:
        mode = lap_idx % 5
        if mode in (0, 1, 4):
            return "centerline"
        if mode == 2:
            return "left-biased"
        if mode == 3:
            return "right-biased"
        return "centerline"

    def current_seed_speed(self) -> float:
        label = self.seed_lap_label(self.current_lap_idx)
        if label in ("left-biased", "right-biased"):
            return self.seed_hug_speed
        return self.seed_speed

    # ===============================
    # Discrete Kinematic Bicycle Model
    # ===============================

    def atv_matrices(self, xbar_seq: np.ndarray, ubar_seq: np.ndarray):
        """
        Replaces the magic number ATV matrices with a strictly linearized 
        Discrete Kinematic Bicycle Model mapped to the 6D Frenet frame.
        """
        A_list, B_list, C_list = [], [], []

        for k in range(self.N):
            xref = xbar_seq[k]
            uref = ubar_seq[k]

            vx, vy, wz, epsi, s, ey = xref
            delta_ref = uref[0]

            kappa = self.track.kappa_at(s)
            den = max(1e-3, (1.0 - kappa * ey))
            cos_e = math.cos(epsi)
            sin_e = math.sin(epsi)

            # Continuous Frenet equations
            f_epsi = wz - ((vx * cos_e - vy * sin_e) / den) * kappa
            f_s = (vx * cos_e - vy * sin_e) / den
            f_ey = vx * math.sin(epsi) + vy * math.cos(epsi)

            # Jacobians for the spatial Frenet terms
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

            # 1. Longitudinal velocity (vx_{k+1} = vx_k + a_k * dt)
            A[0, 0] = 1.0
            B[0, 1] = self.dt
            
            # 2. Lateral velocity (vy forced to 0 for pure kinematic behavior)
            A[1, :] = 0.0 
            B[1, :] = 0.0 

            # 3. Yaw rate (wz = vx * tan(delta) / L) mapped algebraically
            tan_del = math.tan(delta_ref)
            sec2_del = 1.0 / (math.cos(delta_ref)**2) if abs(math.cos(delta_ref)) > 1e-4 else 1e4
            
            A[2, 0] = tan_del / self.L_wb
            B[2, 0] = vx * sec2_del / self.L_wb
            C[2] = (vx * tan_del / self.L_wb) - (A[2, 0] * vx + B[2, 0] * delta_ref)

            # 4,5,6. Discrete Forward Euler integration for spatial terms
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

    # ===============================
    # Reference generation
    # ===============================

    def build_reference_horizon(self, x0: np.ndarray):
        xref = np.zeros((self.N + 1, 6), dtype=float)
        uref = np.zeros((self.N, 2), dtype=float)

        s0 = float(x0[4])
        s_ref = [s0]
        
        # Use velocity manager for speed lookup
        vx_ref = [self.vel_mgr.speed_at(s0)]

        for k in range(1, self.N + 1):
            s_next = s_ref[-1] + max(vx_ref[-1], self.vx_min) * self.dt
            s_ref.append(s_next)
            vx_ref.append(self.vel_mgr.speed_at(s_next))

        for k in range(self.N + 1):
            s_k = float(s_ref[k])
            ey_k = self.line_mgr.line_at(s_k)

            xref[k, 0] = vx_ref[k]
            xref[k, 1] = 0.0
            xref[k, 2] = 0.0
            xref[k, 3] = 0.0
            xref[k, 4] = s_k
            xref[k, 5] = ey_k

        for k in range(self.N):
            kappa_k = self.track.kappa_at(s_ref[k])
            delta_ff = self.steer_sign * self.ff_gain * math.atan(self.L_wb * kappa_k)
            a_ff = sat((vx_ref[k + 1] - vx_ref[k]) / self.dt, self.a_min, self.a_max)

            uref[k, 0] = sat(delta_ff, self.delta_min, self.delta_max)
            uref[k, 1] = a_ff

        return xref, uref

    # ===============================
    # QP building & solve
    # ===============================

    def osqp_mats(self, A_list, B_list, C_list, x0, xref, uref):
        n_x, n_u, N = 6, 2, self.N
        n_s = N  
        Xn = (N + 1) * n_x
        Un = N * n_u
        Sn = n_s
        Zn = Xn + Un + Sn

        def idx_slack(k_from_1: int) -> int:
            return Xn + Un + (k_from_1 - 1)

        P = sparse.lil_matrix((Zn, Zn))
        q = np.zeros(Zn)

        for k in range(N):
            idx = slice(k * n_x, (k + 1) * n_x)
            P[idx, idx] += 2.0 * self.Q
            q[idx] += -2.0 * (self.Q @ xref[k])

        idxN = slice(N * n_x, (N + 1) * n_x)
        P[idxN, idxN] += 2.0 * self.Qf
        q[idxN] += -2.0 * (self.Qf @ xref[N])

        for k in range(N):
            idxu = slice(Xn + k * n_u, Xn + (k + 1) * n_u)
            P[idxu, idxu] += 2.0 * self.R
            q[idxu] += -2.0 * (self.R @ uref[k])

        for k in range(1, N + 1):
            islk = idx_slack(k)
            P[islk, islk] += 2.0 * self.w_slack

        if self.rho_du > 0.0:
            D = np.zeros((N * n_u, Un))
            d = np.zeros(N * n_u)

            D[0:n_u, 0:n_u] = np.eye(n_u)
            d[0:n_u] = self.u_prev.copy()

            for k in range(1, N):
                r0 = k * n_u
                c0 = k * n_u
                cp = (k - 1) * n_u
                D[r0:r0 + n_u, c0:c0 + n_u] = np.eye(n_u)
                D[r0:r0 + n_u, cp:cp + n_u] = -np.eye(n_u)

            Hdu = D.T @ D
            bdu = D.T @ d

            P[Xn:Xn + Un, Xn:Xn + Un] += 2.0 * self.rho_du * Hdu
            q[Xn:Xn + Un] += -2.0 * self.rho_du * bdu

        rows, cols, data = [], [], []
        l = []
        u = []

        def add_row(entries, low, high):
            r = len(l)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                data.append(v)
            l.append(low)
            u.append(high)

        for i in range(n_x):
            add_row([(i, 1.0)], x0[i], x0[i])

        for k in range(N):
            A = A_list[k]
            B = B_list[k]
            C = C_list[k]

            for i in range(n_x):
                row = [((k + 1) * n_x + i, 1.0)]
                for jx in range(n_x):
                    val = -A[i, jx]
                    if val != 0.0:
                        row.append((k * n_x + jx, val))
                for ju in range(n_u):
                    val = -B[i, ju]
                    if val != 0.0:
                        row.append((Xn + k * n_u + ju, val))
                add_row(row, C[i], C[i])

        for k in range(1, N + 1):
            add_row([(k * n_x + 0, 1.0)], self.vx_min, self.vx_max)
            add_row([(k * n_x + 1, 1.0)], -self.vy_abs_max, self.vy_abs_max)
            add_row([(k * n_x + 2, 1.0)], -self.wz_abs_max, self.wz_abs_max)
            add_row([(k * n_x + 3, 1.0)], -self.epsi_abs_max, self.epsi_abs_max)

            s_ref_k = float(xref[k, 4])
            if self.track.w_left is not None and self.track.w_right is not None:
                wl, wr = self.track.widths_at(s_ref_k, margin=self.ey_margin)
                ey_lo = -wr
                ey_hi = wl
            else:
                ey_lo = -self.ey_abs_max
                ey_hi = self.ey_abs_max

            islk = idx_slack(k)
            add_row([(k * n_x + 5, 1.0), (islk, -1.0)], -np.inf, ey_hi)
            add_row([(k * n_x + 5, 1.0), (islk, 1.0)], ey_lo, np.inf)
            add_row([(islk, 1.0)], 0.0, np.inf)

        for k in range(N):
            add_row([(Xn + k * n_u + 0, 1.0)], self.delta_min, self.delta_max)
            add_row([(Xn + k * n_u + 1, 1.0)], self.a_min, self.a_max)

        dmax = self.delta_rate_max * self.dt if self.delta_rate_max > 0 else None
        amax = self.a_rate_max * self.dt if self.a_rate_max > 0 else None

        if dmax is not None:
            add_row([(Xn + 0, 1.0)], self.u_prev[0] - dmax, self.u_prev[0] + dmax)
            for k in range(1, N):
                add_row(
                    [(Xn + k * n_u + 0, 1.0), (Xn + (k - 1) * n_u + 0, -1.0)],
                    -dmax,
                    dmax
                )

        if amax is not None:
            add_row([(Xn + 1, 1.0)], self.u_prev[1] - amax, self.u_prev[1] + amax)
            for k in range(1, N):
                add_row(
                    [(Xn + k * n_u + 1, 1.0), (Xn + (k - 1) * n_u + 1, -1.0)],
                    -amax,
                    amax
                )

        Aqp = sparse.csc_matrix((data, (rows, cols)), shape=(len(l), Zn))
        return sparse.csc_matrix(P), q, Aqp, np.array(l), np.array(u)

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
            self.get_logger().warn(f"OSQP status: {res.info.status}. Using fallback control.")
            return None

        self.prev_osqp_x = res.x
        try:
            self.prev_osqp_y = res.y
        except Exception:
            self.prev_osqp_y = None
        return res.x

    # ===============================
    # Diagnostics / feasibility
    # ===============================

    def actual_out_of_track(self, x: np.ndarray) -> bool:
        if self.track.w_left is None or self.track.w_right is None:
            return False

        s_val = float(x[4])
        ey_val = float(x[5])

        wl, wr = self.track.widths_at(s_val, margin=0.0)
        if ey_val > wl + self.out_of_track_tol:
            return True
        if ey_val < -wr - self.out_of_track_tol:
            return True
        return False

    # ===============================
    # Lap handling
    # ===============================

    def finalize_lap_if_needed(self, x0: np.ndarray):
        if self.prev_s is None:
            self.prev_s = float(x0[4])
            return

        wrap_drop = (self.prev_s - x0[4]) > self.wrap_thresh

        kappa_here = self.track.kappa_at(x0[4])
        denom = max(1e-3, 1.0 - kappa_here * x0[5])
        fs = (x0[0] * math.cos(x0[3]) - x0[1] * math.sin(x0[3])) / denom
        wrap_window = (
            (self.prev_s > (self.track.L - self.wrap_window))
            and (x0[4] < self.wrap_window)
            and (fs > 0.05)
        )

        if not (wrap_drop or wrap_window):
            return

        steps = len(self.curr_traj_u)
        if steps < self.min_lap_steps:
            self.get_logger().warn(f"Discarding lap with only {steps} steps (< {self.min_lap_steps}).")
            self.curr_traj_x = []
            self.curr_traj_u = []
            self.qp_fail_in_lap = 0
            self.near_wall_in_lap = 0
            self.out_of_track_in_lap = 0
            self.current_lap_idx += 1
            self.u_prev = np.zeros(2, dtype=float)
            if self.do_seed and self.current_lap_idx < self.seed_laps:
                self.get_logger().info(
                    f"Next seed lap {self.current_lap_idx}: {self.seed_lap_label(self.current_lap_idx)}"
                )
            self.line_mgr.prepare_next_candidate(len(self.laps))
            return

        X = np.vstack(self.curr_traj_x) if self.curr_traj_x else np.empty((0, 6))
        U = np.vstack(self.curr_traj_u) if self.curr_traj_u else np.empty((0, 2))

        Ulen = U.shape[0]
        Xlen = X.shape[0]
        Tn = min(Ulen, max(0, Xlen - 1))
        U = U[:Tn, :]
        X = X[:Tn + 1, :]

        lap_time = Tn * self.dt
        fail_ratio = self.qp_fail_in_lap / max(1, Tn)
        near_wall_ratio = self.near_wall_in_lap / max(1, Tn)
        out_ratio = self.out_of_track_in_lap / max(1, Tn)

        stored_in_memory = False
        valid_lap = (fail_ratio <= self.fail_frac_thresh) and (out_ratio == 0.0)

        if valid_lap:
            J = np.array([Tn - t for t in range(Tn + 1)], dtype=float)
            self.laps.append({
                'x': X,
                'u': U,
                'J': J,
                'lap_time': lap_time,
                'fail_ratio': fail_ratio,
                'near_wall_ratio': near_wall_ratio,
                'out_ratio': out_ratio,
            })
            self.total_laps += 1
            stored_in_memory = True

        velocity_updated = False
        if valid_lap and hasattr(self, 'vel_mgr') and self.current_lap_idx >= self.seed_laps:
            velocity_updated = self.vel_mgr.update_from_lap(
                lap_idx=self.current_lap_idx,
                lap_time=lap_time,
                valid_lap=valid_lap,
                X_lap=X,
            )

        accepted, best_after = self.line_mgr.finish_lap(
            lap_idx=self.current_lap_idx,
            stored_laps=len(self.laps),
            lap_time=lap_time,
            valid_lap=valid_lap,
            X_lap=X
        )

        with open(self.summary_csv, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([
                self.current_lap_idx,
                lap_time,
                int(stored_in_memory),
                int(accepted),
                int(velocity_updated),
                best_after if np.isfinite(best_after) else -1.0,
                fail_ratio,
                near_wall_ratio,
                out_ratio
            ])

        if valid_lap:
            self.get_logger().info(
                f"Lap {self.current_lap_idx} stored | time={lap_time:.2f}s | "
                f"accepted={accepted} | vel_update={velocity_updated} | best={best_after:.2f}s | "
                f"fail={fail_ratio:.1%} | near-wall={near_wall_ratio:.1%} | out={out_ratio:.1%}"
            )
        else:
            self.get_logger().warn(
                f"Lap {self.current_lap_idx} not stored | time={lap_time:.2f}s | "
                f"fail={fail_ratio:.1%} | near-wall={near_wall_ratio:.1%} | out={out_ratio:.1%}"
            )

        self.curr_traj_x = []
        self.curr_traj_u = []
        self.qp_fail_in_lap = 0
        self.near_wall_in_lap = 0
        self.out_of_track_in_lap = 0
        self.current_lap_idx += 1

        self.u_prev = np.zeros(2, dtype=float)

        if self.do_seed and self.current_lap_idx < self.seed_laps:
            self.get_logger().info(
                f"Next seed lap {self.current_lap_idx}: {self.seed_lap_label(self.current_lap_idx)}"
            )

        self.line_mgr.prepare_next_candidate(len(self.laps))

        # NEW: Update the RViz display with the newly calculated line
        self.publish_learned_line()

    # ===============================
    # Control loop
    # ===============================

    def control_loop(self):
        if not self.have_pose:
            return

        x0 = self.x_meas.copy()
        self.finalize_lap_if_needed(x0)

        # --- Seeding phase ---
        if self.do_seed and self.current_lap_idx < self.seed_laps:
            s_now = float(x0[4])
            ey_ref_now = self.seed_line_offset(self.current_lap_idx, s_now)
            seed_v = self.current_seed_speed()

            kappa = self.track.kappa_at(s_now)
            delta_ff = self.ff_gain * math.atan(self.L_wb * kappa)

            ey_err = x0[5] - ey_ref_now
            delta = (delta_ff - self.ky_seed * ey_err - self.kpsi_seed * x0[3]) * self.steer_sign
            delta = sat(delta, self.delta_min, self.delta_max)

            a = (seed_v - self.speed_cmd) / self.dt
            a = sat(a, self.a_min, self.a_max)

            self.apply_control(delta, a)
            self.append_traj(x0, np.array([delta, a], dtype=float))

            xbar = np.tile(x0, (self.N + 1, 1))
            s_preview = s_now
            for k in range(self.N + 1):
                if k > 0:
                    s_preview = s_preview + max(seed_v, self.vx_min) * self.dt
                xbar[k, 4] = s_preview
                xbar[k, 5] = self.seed_line_offset(self.current_lap_idx, s_preview)
                xbar[k, 3] = 0.0
                xbar[k, 0] = seed_v

            self.publish_preview(xbar)
            self.prev_s = float(x0[4])
            return

        # --- Learning/tracking MPC phase ---
        xref, uref = self.build_reference_horizon(x0)
        
        # Now passing uref so Kinematic Bicycle Model can linearize steering!
        A_list, B_list, C_list = self.atv_matrices(xref, uref) 

        P, q, Aqp, lvec, uvec = self.osqp_mats(A_list, B_list, C_list, x0, xref, uref)
        sol = self.solve_qp(P, q, Aqp, lvec, uvec)

        if sol is None:
            ey_ref0 = float(xref[0, 5])
            delta_ff0 = float(uref[0, 0])
            delta = sat(delta_ff0 - 0.9 * (x0[5] - ey_ref0) - 0.8 * x0[3], self.delta_min, self.delta_max)
            a = sat(0.8 * (xref[0, 0] - self.speed_cmd), self.a_min, self.a_max)
            use_x_seq = xref
        else:
            n_x = 6
            Xn = (self.N + 1) * n_x
            Un = self.N * 2

            Xopt = sol[:Xn].reshape(self.N + 1, n_x)
            Uopt = sol[Xn:Xn + Un].reshape(self.N, 2)

            delta = float(sat(Uopt[0, 0], self.delta_min, self.delta_max))
            a = float(sat(Uopt[0, 1], self.a_min, self.a_max))
            use_x_seq = Xopt

        self.apply_control(delta, a)
        self.append_traj(x0, np.array([delta, a], dtype=float))
        self.publish_preview(use_x_seq)

        self.prev_s = float(x0[4])

    # ===============================
    # Publishing helpers
    # ===============================

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

        if self.actual_out_of_track(x):
            self.out_of_track_in_lap += 1

    def publish_learned_line(self):
        """Publishes both the Best Line and the Candidate Line to RViz."""
        if not self.have_pose:
            return
            
        try:
            time_now = self.get_clock().now().to_msg()

            # ---------------------------------------------------------
            # 1. Publish the BEST LINE (The confirmed fastest line)
            # ---------------------------------------------------------
            path_best = Path()
            path_best.header.frame_id = self.viz_frame
            path_best.header.stamp = time_now
            
            for i, s_val in enumerate(self.track.s):
                xg, yg = self.frenet.frenet_to_global(float(s_val), float(self.line_mgr.best_line[i]))
                ps = PoseStamped()
                ps.header = path_best.header
                ps.pose.position.x = float(xg)
                ps.pose.position.y = float(yg)
                ps.pose.orientation.w = 1.0
                path_best.poses.append(ps)
                
            self.pub_best_line.publish(path_best)

            # ---------------------------------------------------------
            # 2. Publish the CANDIDATE LINE (The experimental shift)
            # ---------------------------------------------------------
            path_cand = Path()
            path_cand.header.frame_id = self.viz_frame
            path_cand.header.stamp = time_now
            
            for i, s_val in enumerate(self.track.s):
                xg, yg = self.frenet.frenet_to_global(float(s_val), float(self.line_mgr.candidate_line[i]))
                ps = PoseStamped()
                ps.header = path_cand.header
                ps.pose.position.x = float(xg)
                ps.pose.position.y = float(yg)
                ps.pose.orientation.w = 1.0
                path_cand.poses.append(ps)
                
            self.pub_cand_line.publish(path_cand)

        except Exception as e:
            self.get_logger().warn(f"Full line viz failed: {e}")

    def publish_preview(self, Xseq: np.ndarray):
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
    node = IterativeRacingLineMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop = AckermannDriveStamped()
            stop.drive.steering_angle = 0.0
            stop.drive.speed = 0.0
            node.pub_cmd.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()