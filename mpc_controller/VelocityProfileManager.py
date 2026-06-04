# #!/usr/bin/env python3
# import pandas as pd
# import numpy as np
# import math
# import os
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# from typing import Optional, Tuple

# # ===============================
# # Utilities & Track Definition
# # ===============================
# def sat(val: float, lo: float, hi: float) -> float:
#     return max(lo, min(hi, val))

# def circ_moving_average(x: np.ndarray, w: int) -> np.ndarray:
#     if w <= 1: return x.copy()
#     pad = w // 2
#     xp = np.pad(x, (pad, pad), mode='wrap')
#     ker = np.ones(w, dtype=float) / float(w)
#     y = np.convolve(xp, ker, mode='same')
#     return y[pad:-pad]

# @dataclass
# class Track:
#     s: np.ndarray
#     kappa: np.ndarray
#     x: Optional[np.ndarray] = None
#     y: Optional[np.ndarray] = None
#     theta: Optional[np.ndarray] = None
#     w_left: Optional[np.ndarray] = None
#     w_right: Optional[np.ndarray] = None

#     @property
#     def L(self) -> float:
#         return float(self.s[-1]) if len(self.s) else 1.0

#     def _interp_pair(self, arr: np.ndarray, s_query: float) -> float:
#         L = self.L if self.L > 0 else 1.0
#         s_wrapped = s_query % L
#         idx = np.searchsorted(self.s, s_wrapped)

#         if idx == 0:
#             i0, i1 = len(self.s) - 1, 0
#             s0, s1 = float(self.s[i0]) - L, float(self.s[i1])
#         elif idx >= len(self.s):
#             i0, i1 = len(self.s) - 1, 0
#             s0, s1 = float(self.s[i0]), float(self.s[i1]) + L
#         else:
#             i0, i1 = idx - 1, idx
#             s0, s1 = float(self.s[i0]), float(self.s[i1])

#         a0, a1 = float(arr[i0]), float(arr[i1])
#         if abs(s1 - s0) < 1e-9: return a0
#         w = (s_wrapped - s0) / (s1 - s0)
#         return (1.0 - w) * a0 + w * a1

# # ===============================
# # Your CSV Loader
# # ===============================
# def load_csv_track(filepath: str) -> Track:
#     df = pd.read_csv(filepath)
#     def canon(name: str) -> str: return name.strip().lstrip('#').strip().lower()
#     cols = {canon(c): c for c in df.columns}

#     def get_any(names):
#         for n in names:
#             if n in cols: return df[cols[n]].to_numpy(dtype=float)
#         return None

#     s = get_any(['s', 's_m', 'arc_length', 'arclength'])
#     x = get_any(['x', 'x_m'])
#     y = get_any(['y', 'y_m'])
#     theta = get_any(['theta', 'theta_rad', 'heading'])
#     kappa = get_any(['kappa', 'curvature', 'k'])
#     w_left = get_any(['w_tr_left', 'w_tr_left_m', 'w_left', 'width_left'])
#     w_right = get_any(['w_tr_right', 'w_tr_right_m', 'w_right', 'width_right'])

#     if s is None and x is not None and y is not None:
#         s = np.cumsum(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
#         s -= s[0]
#     if theta is None and x is not None and y is not None:
#         theta = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
#     if kappa is None and theta is not None and s is not None:
#         ds = np.gradient(s)
#         ds[np.abs(ds) < 1e-6] = 1e-6
#         kappa = np.gradient(theta) / ds

#     order = np.argsort(s)
#     return Track(
#         s=s[order] - s[order][0], kappa=kappa[order], 
#         x=x[order] if x is not None else None, y=y[order] if y is not None else None,
#         theta=theta[order] if theta is not None else None,
#         w_left=w_left[order] if w_left is not None else None, w_right=w_right[order] if w_right is not None else None
#     )

# class MockRacingLineManager:
#     """Mocks the dynamic curvature output for offline testing."""
#     def __init__(self, track: Track):
#         self.track = track
#         # Clean the raw CSV curvature so it doesn't cause math errors
#         self.dynamic_kappa = circ_moving_average(np.abs(track.kappa.copy()), 21)

#     def dynamic_kappa_at(self, s_query: float) -> float:
#         return float(self.track._interp_pair(self.dynamic_kappa, s_query))

# # ===============================
# # Your Velocity Manager
# # ===============================
# class VelocityProfileManager:
#     def __init__(
#         self, track: Track, line_mgr, log_dir: str, vx_min: float, vx_max: float,
#         a_lat_ref_max: float, curv_speed_floor: float, speed_ref_scale: float, enable: bool = True,
#         blend: float = 0.35, smooth_window: int = 21, accept_margin_sec: float = 0.20,
#         data_margin: float = 0.15, max_speed_step: float = 0.25,
#     ):
#         self.track = track
#         self.line_mgr = line_mgr
#         self.log_dir = log_dir
#         self.vx_min = vx_min
#         self.vx_max = vx_max
#         self.a_lat_ref_max = a_lat_ref_max
#         self.curv_speed_floor = curv_speed_floor
#         self.speed_ref_scale = speed_ref_scale
#         self.enable = enable
#         self.blend = blend
#         self.smooth_window = smooth_window
#         self.accept_margin_sec = accept_margin_sec
#         self.data_margin = data_margin
#         self.max_speed_step = max_speed_step

#         self.base_speed = np.array([self.curvature_speed_at(float(s)) for s in self.track.s], dtype=float)
#         self.learned_speed = self.base_speed.copy()
#         self.best_lap_time = np.inf
#         self.num_updates = 0

#     def curvature_speed_at(self, s_val: float) -> float:
#         kappa = self.line_mgr.dynamic_kappa_at(s_val)
#         kappa = max(kappa, self.curv_speed_floor)
#         vref = self.speed_ref_scale * math.sqrt(max(0.1, self.a_lat_ref_max / kappa))
#         return sat(vref, self.vx_min, self.vx_max)

#     def speed_at(self, s_query: float) -> float:
#         if not self.enable or self.num_updates == 0:
#             return self.curvature_speed_at(s_query)
#         learned = float(self.track._interp_pair(self.learned_speed, s_query))
#         safe = self.curvature_speed_at(s_query)
#         return sat(min(learned, safe + self.data_margin), self.vx_min, self.vx_max)

#     def _apply_braking_zones(self, profile: np.ndarray) -> np.ndarray:
#         safe_decel = 6.0 
#         n = len(self.track.s)
#         for _ in range(2):
#             for i in range(n - 2, -1, -1):
#                 ds = float(self.track.s[i+1] - self.track.s[i])
#                 max_entry_v = math.sqrt(max(0.0, profile[i+1]**2 + 2.0 * safe_decel * ds))
#                 profile[i] = min(profile[i], max_entry_v)
#             ds_wrap = float(self.track.L - self.track.s[-1] + self.track.s[0])
#             max_entry_v = math.sqrt(max(0.0, profile[0]**2 + 2.0 * safe_decel * ds_wrap))
#             profile[-1] = min(profile[-1], max_entry_v)
#         return profile

#     def _profile_from_lap(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         profile = self.learned_speed.copy()
#         counts = np.zeros_like(self.track.s, dtype=float)
#         sums = np.zeros_like(self.track.s, dtype=float)

#         if X.shape[0] < 10: return profile, counts

#         for row in X:
#             s_val = float(row[4] % self.track.L)
#             vx_val = sat(float(row[0]), self.vx_min, self.vx_max)
#             idx = int(np.argmin(np.abs(self.track.s - s_val)))
#             sums[idx] += vx_val
#             counts[idx] += 1.0

#         mask = counts > 0.0
#         if np.any(mask): profile[mask] = sums[mask] / counts[mask]

#         profile = circ_moving_average(profile, self.smooth_window)
#         for i, s in enumerate(self.track.s):
#             profile[i] = sat(min(profile[i], self.curvature_speed_at(float(s)) + self.data_margin), self.vx_min, self.vx_max)
#         return profile, counts

#     def update_from_lap(self, lap_idx: int, lap_time: float, valid_lap: bool, X_lap: np.ndarray) -> bool:
#         if not self.enable or not valid_lap:
#             return False

#         if np.isfinite(self.best_lap_time) and lap_time > self.best_lap_time + self.accept_margin_sec:
#             return False

#         measured, counts = self._profile_from_lap(X_lap)
#         if np.count_nonzero(counts) < max(5, int(0.15 * len(self.track.s))):
#             return False

#         # --- THE NEW ASYMMETRIC UPDATE LAW ---
        
#         # 1. Calculate the raw error between what the car did and what it wanted to do
#         raw_error = measured - self.learned_speed
        
#         # 2. ONLY smooth the error, not the whole profile (fixes the Heat Equation diffusion)
#         smoothed_error = circ_moving_average(raw_error, self.smooth_window)
        
#         # 3. Scale the error by your learning rate (blend) and clip it for stability
#         step = self.blend * smoothed_error
#         step = np.clip(step, -self.max_speed_step, self.max_speed_step)
        
#         # 4. Generate the proposed new speed
#         proposed_speed = self.learned_speed + step
        
#         # 5. Apply the rigorous Max/Min bounding
#         for i, s in enumerate(self.track.s):
#             # Calculate the absolute physical limit for this specific point
#             safe_limit = self.curvature_speed_at(float(s)) + self.data_margin
            
#             # The speed cannot exceed the physical limit, and it cannot "forget" its previous best speed
#             self.learned_speed[i] = sat(
#                 proposed_speed[i], 
#                 self.learned_speed[i], # The MAX constraint: never go slower than you already safely proved
#                 safe_limit             # The MIN constraint: never exceed the friction circle
#             )
            
#             # Final global clamp just in case
#             self.learned_speed[i] = sat(self.learned_speed[i], self.vx_min, self.vx_max)

#         # 6. Propagate braking backwards to ensure the new speeds don't cause corner overshoots
#         self.learned_speed = self._apply_braking_zones(self.learned_speed)

#         if lap_time < self.best_lap_time:
#             self.best_lap_time = lap_time
            
#         self.num_updates += 1
        
#         # If you are in ROS, you won't use this, but it's here for the standalone script
#         if hasattr(self, '_save_speed_csv'):
#              self._save_speed_csv(os.path.join(self.log_dir, f"speed_profile_after_lap_{lap_idx:03d}.csv"), self.learned_speed)
             
#         return True

# # ===============================
# # Run Execution and Plotting
# # ===============================
# if __name__ == "__main__":
#     # --- IMPORTANT: Ensure this file is in the same folder as your script ---
#     csv_filename = '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv'
    
#     if not os.path.exists(csv_filename):
#         print(f"ERROR: Cannot find {csv_filename} in the current directory.")
#         exit(1)

#     print(f"Loading {csv_filename}...")
#     actual_track = load_csv_track(csv_filename)
#     mock_line_mgr = MockRacingLineManager(actual_track)

#     vel_mgr = VelocityProfileManager(
#         track=actual_track, line_mgr=mock_line_mgr, log_dir="./",
#         vx_min=1.0, vx_max=100.0, a_lat_ref_max=3.5, curv_speed_floor=0.03,
#         speed_ref_scale=0.95, blend=0.5
#     )

#     initial_safe_speed = vel_mgr.base_speed.copy()
#     num_laps = 1000
#     history = []

#     print("Simulating Laps on Brands Hatch...")
#     for lap in range(1, num_laps + 1):
#         X_lap = np.zeros((len(actual_track.s), 6))
#         for i, s in enumerate(actual_track.s):
#             current_target = vel_mgr.speed_at(s)
#             phys_lim = 8.0 if mock_line_mgr.dynamic_kappa[i] < 0.05 else math.sqrt(3.5 / mock_line_mgr.dynamic_kappa[i])
#             X_lap[i, 0] = current_target + 0.3 * (phys_lim - current_target) + np.random.normal(0, 0.1)
#             X_lap[i, 4] = s
            
#         vel_mgr.update_from_lap(lap_idx=lap, lap_time=30.0 - (lap*0.1), valid_lap=True, X_lap=X_lap)
#         history.append(vel_mgr.learned_speed.copy())
#         ds = np.mean(np.diff(actual_track.s))
        
#         # 2. Calculate the theoretical lap time using discrete integration
#         theoretical_lap_time = np.sum(ds / vel_mgr.learned_speed)
        
#         print(f"Lap {lap:02d} | Max Speed: {np.max(vel_mgr.learned_speed):.2f} m/s | Theoretical Lap Time: {theoretical_lap_time:.3f} sec")

#     # Plotting
#     plt.figure(figsize=(14, 7))
#     plt.plot(actual_track.s, mock_line_mgr.dynamic_kappa * 10, 'k--', alpha=0.3, label='Track Curvature (Scaled)')
#     plt.plot(actual_track.s, initial_safe_speed, 'b-', linewidth=2, label='Initial Safe Profile')
    
#     for i, prof in enumerate(history):
#         alpha = 0.3 + (0.7 * (i / num_laps))
#         if i == num_laps - 1:
#             plt.plot(actual_track.s, prof, 'r-', linewidth=2.5, label='Final Learned Profile (Lap 10)')
#         elif i % 3 == 0:
#             plt.plot(actual_track.s, prof, 'g-', alpha=alpha, label=f'Iteration {i}')
            
#     plt.title('Velocity Profile Iterative Learning - Brands Hatch')
#     plt.xlabel('Arc Length s (m)')
#     plt.ylabel('Velocity vx (m/s)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple

# ===============================
# Utilities & Track Definition
# ===============================
def sat(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def circ_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='wrap')
    ker = np.ones(w, dtype=float) / float(w)
    y = np.convolve(xp, ker, mode='same')
    return y[pad:-pad]

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
        if abs(s1 - s0) < 1e-9: return a0
        w = (s_wrapped - s0) / (s1 - s0)
        return (1.0 - w) * a0 + w * a1

# ===============================
# Your CSV Loader
# ===============================
def load_csv_track(filepath: str) -> Track:
    df = pd.read_csv(filepath)
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
    w_left = get_any(['w_tr_left', 'w_tr_left_m', 'w_left', 'width_left'])
    w_right = get_any(['w_tr_right', 'w_tr_right_m', 'w_right', 'width_right'])

    if s is None and x is not None and y is not None:
        s = np.cumsum(np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])))
        s -= s[0]
    if theta is None and x is not None and y is not None:
        theta = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    if kappa is None and theta is not None and s is not None:
        ds = np.gradient(s)
        ds[np.abs(ds) < 1e-6] = 1e-6
        kappa = np.gradient(theta) / ds

    order = np.argsort(s)
    return Track(
        s=s[order] - s[order][0], kappa=kappa[order], 
        x=x[order] if x is not None else None, y=y[order] if y is not None else None,
        theta=theta[order] if theta is not None else None,
        w_left=w_left[order] if w_left is not None else None, w_right=w_right[order] if w_right is not None else None
    )

# ===============================
# REAL Racing Line Manager
# ===============================
class RealRacingLineManager:
    """Uses the real ey_ref from your uploaded CSV to calculate true Dynamic Curvature."""
    def __init__(self, track: Track, best_line_csv: str):
        self.track = track
        self.smooth_window = 21
        
        # Load the uploaded line
        df = pd.read_csv(best_line_csv)
        self.ey_ref = df['ey_ref'].to_numpy(dtype=float)
        
        # Map Frenet (s, ey) back to Global (X, Y)
        self.X = np.zeros_like(track.s)
        self.Y = np.zeros_like(track.s)
        
        for i in range(len(track.s)):
            th = track.theta[i]
            nx, ny = -math.sin(th), math.cos(th)
            self.X[i] = track.x[i] + nx * self.ey_ref[i]
            self.Y[i] = track.y[i] + ny * self.ey_ref[i]
            
        self.X = circ_moving_average(self.X, self.smooth_window)
        self.Y = circ_moving_average(self.Y, self.smooth_window)
        
        # Calculate true dynamic curvature of the racing line
        ds = np.gradient(track.s)
        ds[np.abs(ds) < 1e-6] = 1e-6  
        
        Xp = np.gradient(self.X) / ds
        Yp = np.gradient(self.Y) / ds
        Xpp = np.gradient(Xp) / ds
        Ypp = np.gradient(Yp) / ds
        
        denom = Xp**2 + Yp**2
        denom[denom < 1e-6] = 1e-6
        kappa_dyn = (Xp * Ypp - Yp * Xpp) / np.power(denom, 1.5)
        
        self.dynamic_kappa = circ_moving_average(np.abs(kappa_dyn), self.smooth_window)

    def dynamic_kappa_at(self, s_query: float) -> float:
        return float(self.track._interp_pair(self.dynamic_kappa, s_query))

# ===============================
# Your Working Velocity Manager
# ===============================
class VelocityProfileManager:
    def __init__(
        self, track: Track, line_mgr, log_dir: str, vx_min: float, vx_max: float,
        a_lat_ref_max: float, curv_speed_floor: float, 
        speed_ref_scale: float, 
        enable: bool = True,
        blend: float = 0.8,
        smooth_window: int = 21, 
        accept_margin_sec: float = 0.20,
        data_margin: float = 0.15, 
        max_speed_step: float = 1.0,
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

    def curvature_speed_at(self, s_val: float) -> float:
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

    def _apply_braking_zones(self, profile: np.ndarray) -> np.ndarray:
        safe_decel = 5.5 # Using exact physical braking limits
        n = len(self.track.s)
        for _ in range(2):
            for i in range(n - 2, -1, -1):
                ds = float(self.track.s[i+1] - self.track.s[i])
                max_entry_v = math.sqrt(max(0.0, profile[i+1]**2 + 2.0 * safe_decel * ds))
                profile[i] = min(profile[i], max_entry_v)
            ds_wrap = float(self.track.L - self.track.s[-1] + self.track.s[0])
            max_entry_v = math.sqrt(max(0.0, profile[0]**2 + 2.0 * safe_decel * ds_wrap))
            profile[-1] = min(profile[-1], max_entry_v)
        return profile

    def _profile_from_lap(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        profile = self.learned_speed.copy()
        counts = np.zeros_like(self.track.s, dtype=float)
        sums = np.zeros_like(self.track.s, dtype=float)

        if X.shape[0] < 10: return profile, counts

        for row in X:
            s_val = float(row[4] % self.track.L)
            vx_val = sat(float(row[0]), self.vx_min, self.vx_max)
            idx = int(np.argmin(np.abs(self.track.s - s_val)))
            sums[idx] += vx_val
            counts[idx] += 1.0

        mask = counts > 0.0
        if np.any(mask): profile[mask] = sums[mask] / counts[mask]
        return profile, counts

    def update_from_lap(self, lap_idx: int, lap_time: float, valid_lap: bool, X_lap: np.ndarray) -> bool:
        if not self.enable or not valid_lap:
            return False

        if np.isfinite(self.best_lap_time) and lap_time > self.best_lap_time + self.accept_margin_sec:
            return False

        measured, counts = self._profile_from_lap(X_lap)
        if np.count_nonzero(counts) < max(5, int(0.15 * len(self.track.s))):
            return False

        # --- THE CORRECTED ASYMMETRIC MONOTONIC UPDATE LAW ---
        raw_error = measured - self.learned_speed
        smoothed_error = circ_moving_average(raw_error, self.smooth_window)
        
        step = self.blend * smoothed_error
        step = np.clip(step, -self.max_speed_step, self.max_speed_step)
        
        proposed_speed = self.learned_speed + step
        
        for i, s in enumerate(self.track.s):
            safe_limit = self.curvature_speed_at(float(s)) + self.data_margin
            self.learned_speed[i] = sat(
                proposed_speed[i], 
                self.learned_speed[i], # Prevents "forgetting" speed
                safe_limit             # Respects the physics circle
            )
            self.learned_speed[i] = sat(self.learned_speed[i], self.vx_min, self.vx_max)

        self.learned_speed = self._apply_braking_zones(self.learned_speed)

        if lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
            
        self.num_updates += 1
        return True

# ===============================
# Run Execution and Plotting
# ===============================
if __name__ == "__main__":
    # Ensure both files are in the directory
    track_csv = '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv'
    line_csv = '/home/giorgos/sim_ws/src/mpc_controller/logs/best_line_after_lap_003.csv'
    
    if not os.path.exists(track_csv):
        print(f"ERROR: Cannot find {track_csv}.")
        exit(1)
    if not os.path.exists(line_csv):
        print(f"ERROR: Cannot find {line_csv}.")
        exit(1)

    print("Loading Centerline and Real Racing Line...")
    actual_track = load_csv_track(track_csv)
    real_line_mgr = RealRacingLineManager(actual_track, line_csv)

    vel_mgr = VelocityProfileManager(
        track=actual_track, line_mgr=real_line_mgr, log_dir="./",
        vx_min=0.8, vx_max=8.0, a_lat_ref_max=3.5, curv_speed_floor=0.03,
        speed_ref_scale=0.95, blend=0.5
    )

    initial_safe_speed = vel_mgr.base_speed.copy()
    num_laps = 15 # Shorter to see convergence quickly
    history = []

    print("Simulating Laps on TRUE Racing Line...")
    
    # CRITICAL: Calculate the actual distance traveled along the racing line curve, not the centerline
    ds_array_racing_line = np.hypot(np.diff(real_line_mgr.X, prepend=real_line_mgr.X[-1]), 
                                    np.diff(real_line_mgr.Y, prepend=real_line_mgr.Y[-1]))

    for lap in range(1, num_laps + 1):
        print(lap)
        X_lap = np.zeros((len(actual_track.s), 6))
        for i, s in enumerate(actual_track.s):
            current_target = vel_mgr.speed_at(s)
            phys_lim = 8.0 if real_line_mgr.dynamic_kappa[i] < 0.05 else math.sqrt(3.5 / real_line_mgr.dynamic_kappa[i])
            # Simulated telemetry approaches physical limits iteratively
            X_lap[i, 0] = current_target + 0.3 * (phys_lim - current_target) + np.random.normal(0, 0.05)
            X_lap[i, 4] = s
            
        vel_mgr.update_from_lap(lap_idx=lap, lap_time=30.0, valid_lap=True, X_lap=X_lap)
        history.append(vel_mgr.learned_speed.copy())
        
        # Calculate theoretical lap time using TRUE racing line distance
        theoretical_lap_time = np.sum(ds_array_racing_line / vel_mgr.learned_speed)

        total_distance = np.sum(ds_array_racing_line)
        avg_speed = total_distance / theoretical_lap_time
        
        print(f"Lap {lap:02d} | Avg Speed: {avg_speed:.2f} m/s | True Theoretical Time: {theoretical_lap_time:.3f} sec")

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(actual_track.s, real_line_mgr.dynamic_kappa * 10, 'k--', alpha=0.3, label='Dynamic Curvature $\kappa(s) \\times 10$')
    plt.plot(actual_track.s, initial_safe_speed, 'b-', linewidth=2, label='Initial Safe Profile')
    
    for i, prof in enumerate(history):
        alpha = 0.3 + (0.7 * (i / num_laps))
        if i == num_laps - 1:
            plt.plot(actual_track.s, prof, 'r-', linewidth=2.5, label=f'Final Learned Profile (Lap {num_laps})')
        elif i % 3 == 0:
            plt.plot(actual_track.s, prof, 'g-', alpha=alpha, label=f'Iteration {i}')
            
    plt.title('Velocity Profile Iterative Learning - TRUE Uploaded Racing Line')
    plt.xlabel('Centerline Arc Length s (m)')
    plt.ylabel('Velocity vx (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()