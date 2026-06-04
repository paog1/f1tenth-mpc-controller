import os
import numpy as np
import pandas as pd
import math
import osqp
import matplotlib.pyplot as plt
from scipy import sparse
from dataclasses import dataclass
from typing import Optional, Tuple

# ===============================
# Utilities
# ===============================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# ===============================
# Track Geometry
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

    def widths_at(self, s_query: float, margin: float = 0.0) -> Tuple[float, float]:
        if self.w_left is None or self.w_right is None:
            return 1.5, 1.5 
        wl = float(self._interp_pair(self.w_left, s_query))
        wr = float(self._interp_pair(self.w_right, s_query))
        return max(0.0, wl - margin), max(0.0, wr - margin)

class FrenetProjector:
    def __init__(self, track: Track):
        self.track = track
        self.has_xy = (track.x is not None) and (track.y is not None) and (track.theta is not None)

    def frenet_to_global(self, s: float, ey: float) -> Tuple[float, float]:
        if not self.has_xy:
            return s, ey
        xc = self.track._interp_pair(self.track.x, s)
        yc = self.track._interp_pair(self.track.y, s)
        th = self.track._interp_pair(self.track.theta, s)
        nx, ny = -math.sin(th), math.cos(th)
        return float(xc + nx * ey), float(yc + ny * ey)

# ===============================
# Trust-Region Iterative Optimizer
# ===============================
class TrustRegionLineGenerator:
    """Iterative pure minimum curvature learning using a dynamic trust region."""
    
    def __init__(self, track: Track, frenet: FrenetProjector):
        self.track = track
        self.frenet = frenet
        
        # Physical wall margin
        self.ey_margin = 0.1
        
        # The Trust Region: How far the car is allowed to deviate from the PREVIOUS lap
        self.exploration_tube = 0.05
        
        # Initialize on the centerline
        self.best_line = np.zeros_like(track.s, dtype=float)
        self.candidate_line = self.best_line.copy()

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
        return sparse.csc_matrix((data, (rows, cols)), shape=(n, n))

    def generate_next_candidate(self, lap_index: int):
        n = len(self.track.s)
        ds = float(np.median(np.diff(self.track.s)))
        if ds < 1e-3: ds = 0.1

        # 1. Setup the Pure Curvature Objective
        # In Frenet, curvature is proportional to e_y'' + kappa_c
        D2 = self._periodic_diff_matrix(n, order=2) / (ds ** 2)
        
        # P matrix minimizes (e_y'')^2
        P = (D2.T @ D2).tocsc()
        
        # q vector offsets by the track's natural curvature (kappa_c)
        q = D2.T @ self.track.kappa

        I = sparse.eye(n, format='csc')

        # 2. Build the Trust Region Constraints
        lo, hi = np.zeros(n), np.zeros(n)
        for i, s in enumerate(self.track.s):
            # Absolute physical limits
            wl, wr = self.track.widths_at(float(s), margin=self.ey_margin)
            
            # Trust region limits (Current known line +/- exploration tube)
            trust_lo = self.best_line[i] - self.exploration_tube
            trust_hi = self.best_line[i] + self.exploration_tube
            
            # The solver is bounded by whichever is stricter: the wall or the tube
            lo[i] = max(-wr, trust_lo)
            hi[i] = min(wl, trust_hi)

        # 3. Solve for the new line
        prob = osqp.OSQP()
        try:
            prob.setup(P=P, q=q, A=I, l=lo, u=hi, verbose=False, warm_start=True, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000)
            res = prob.solve()
            if res.info.status_val in (1, 2):
                self.candidate_line = np.asarray(res.x, dtype=float)
            else:
                self.candidate_line = self.best_line.copy()
        except Exception:
            self.candidate_line = self.best_line.copy()

        # Update knowledge for the next lap
        self.best_line = self.candidate_line.copy()
        return self.candidate_line

# ===============================
# File Loading & Execution
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

def main():
    # UPDATE THIS PATH to your local track CSV file
    TRACK_CSV_PATH = '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/BrandsHatch_centerline.csv'
    # TRACK_CSV_PATH = '/home/giorgos/sim_ws/src/f1tenth_gym_ros/maps/SaoPaulo_centerline.csv'
    
    if not os.path.exists(TRACK_CSV_PATH):
        print(f"Error: Could not find {TRACK_CSV_PATH}.")
        return

    track = load_csv_track(TRACK_CSV_PATH)
    frenet = FrenetProjector(track)
    
    # Initialize the new Trust Region generator
    generator = TrustRegionLineGenerator(track, frenet)

    x_inner, y_inner, x_outer, y_outer, x_center, y_center = [], [], [], [], [], []
    for s_val in track.s:
        wl, wr = track.widths_at(s_val, margin=0.0)
        xc, yc = frenet.frenet_to_global(s_val, 0.0)
        x_center.append(xc)
        y_center.append(yc)
        xl, yl = frenet.frenet_to_global(s_val, wl)
        x_inner.append(xl)
        y_inner.append(yl)
        xr, yr = frenet.frenet_to_global(s_val, -wr)
        x_outer.append(xr)
        y_outer.append(yr)

    iterations = 10
    lap_1 = 5
    lap_2 = 8
    lap_1 = np.clip(lap_1, 0, iterations - 1)
    lap_2 = np.clip(lap_2, 0, iterations - 1)
    generated_ey_lines = []
    generated_xy_lines = []
    plt.figure(figsize=(12, 10))
    plt.plot(x_inner, y_inner, 'k--', linewidth=1.5, label='Actual Track Limits')
    plt.plot(x_outer, y_outer, 'k--', linewidth=1.5)
    plt.plot(x_center, y_center, 'gray', linestyle=':', linewidth=1, label='Centerline')

    colors = plt.cm.viridis(np.linspace(0, 1, iterations))
    for i in range(iterations):
        cand_ey = generator.generate_next_candidate(lap_index=i)
        
        cand_x, cand_y = [], []
        for s_val, ey_val in zip(track.s, cand_ey):
            cx, cy = frenet.frenet_to_global(s_val, ey_val)
            cand_x.append(cx)
            cand_y.append(cy)

        generated_ey_lines.append(np.asarray(cand_ey, dtype=float))
        generated_xy_lines.append((np.asarray(cand_x, dtype=float), np.asarray(cand_y, dtype=float)))

        if i == iterations - 1:
            plt.plot(cand_x, cand_y, color='red', linewidth=2.5, zorder=5, label=f'Final Line (Lap {i+1})')
        elif i % 10 == 0: 
            plt.plot(cand_x, cand_y, color=colors[i], linewidth=1, alpha=0.5, label=f'Lap {i+1}')
        else:
            plt.plot(cand_x, cand_y, color=colors[i], linewidth=1, alpha=0.15)

    if len(generated_ey_lines) >= 2:
        lap_last = generated_ey_lines[lap_2]
        lap_prev = generated_ey_lines[lap_1]
        ey_diff = lap_last - lap_prev
        ey_max = float(np.max(np.abs(ey_diff)))
        ey_rms = float(np.sqrt(np.mean(ey_diff ** 2)))

        x_last, y_last = generated_xy_lines[-1]
        x_prev, y_prev = generated_xy_lines[9]
        xy_diff = np.hypot(x_last - x_prev, y_last - y_prev)

        print(f"[DIFF] Lap {lap_1} vs Lap {lap_2}:")
        print(f"  max |e_y| difference = {ey_max:.6f} m")
        print(f"  RMS  |e_y| difference = {ey_rms:.6f} m")
        print(f"  max XY distance      = {float(np.max(xy_diff)):.6f} m")
        print(f"  RMS XY distance      = {float(np.sqrt(np.mean(xy_diff ** 2))):.6f} m")

    plt.title('Trust-Region Iterative Optimization (True Minimum Curvature)')
    plt.xlabel('Global X [m]')
    plt.ylabel('Global Y [m]')
    plt.axis('equal')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()