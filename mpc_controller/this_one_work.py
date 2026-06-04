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

def circ_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
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

    def circ_moving_average(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return x.copy()
        pad = w // 2
        xp = np.pad(x, (pad, pad), mode='wrap')
        ker = np.ones(w, dtype=float) / float(w)
        y = np.convolve(xp, ker, mode='same')
        return y[pad:-pad]

    # ===============================
    # Track & Geometry
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

        def kappa_at(self, s_query: float) -> float:
            return float(self._interp_pair(self.kappa, s_query))

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
    # Iterative Line Generator (From your ROS 2 file)
    # ===============================
    class CandidateLineGenerator:
        """Exact logic from RacingLineManager in MPC_LINE_OPT_SAVED.py"""
    
        def __init__(self, track: Track, frenet: FrenetProjector):
            self.track = track
            self.frenet = frenet
        
            # Hardcoding the parameters exactly as they default in your ROS node
            self.ey_margin = 0.2  
            self.exploration_amp = 5.5
            self.smooth_window = 11
        
            self.mincurv_w_curv = 100.0
            self.mincurv_w_smooth = 50.0
            self.mincurv_w_data = 1.0
            self.mincurv_w_prev = 0.00

            self.best_line = np.zeros_like(track.s, dtype=float)
            self.candidate_line = self.best_line.copy()
            self.dynamic_kappa = np.abs(track.kappa.copy())

        def _update_dynamic_curvature(self):
            n = len(self.track.s)
            X, Y = np.zeros(n), np.zeros(n)
            for i, s_val in enumerate(self.track.s):
                X[i], Y[i] = self.frenet.frenet_to_global(float(s_val), float(self.best_line[i]))
            
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
            self.dynamic_kappa = circ_moving_average(np.abs(kappa), self.smooth_window)

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

        def _minimum_curvature_qp(self, data_line: np.ndarray, prev_line: np.ndarray) -> np.ndarray:
            n = len(self.track.s)
            D1 = self._periodic_diff_matrix(n, order=1)
            D2 = self._periodic_diff_matrix(n, order=2)
            I = sparse.eye(n, format='csc')

            macro_kappa = circ_moving_average(np.abs(self.track.kappa), int(self.smooth_window * 4))
            max_k = np.max(macro_kappa) + 1e-6
            norm_k = macro_kappa / max_k
            w_data_array = self.mincurv_w_data * (1.0 - norm_k)
            W_data_mat = sparse.diags(w_data_array, format='csc')

            P = (self.mincurv_w_curv * (D2.T @ D2) + 
                 self.mincurv_w_smooth * (D1.T @ D1) + 
                 W_data_mat + 
                 self.mincurv_w_prev * I).tocsc()
        
            clean_kappa = circ_moving_average(self.track.kappa, self.smooth_window)
            q_curv = self.mincurv_w_curv * (D2.T @ clean_kappa)
            q = q_curv - (w_data_array * data_line) - (self.mincurv_w_prev * prev_line)

            lo, hi = np.zeros(n), np.zeros(n)
            # Bounding limits using the 0.2m ey_margin
            for i, s in enumerate(self.track.s):
                wl, wr = self.track.widths_at(float(s), margin=self.ey_margin)
                lo[i] = -wr
                hi[i] = wl

            prob = osqp.OSQP()
            try:
                prob.setup(P=P, q=q, A=I, l=lo, u=hi, verbose=False, warm_start=True, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)
                res = prob.solve()
                if res.info.status_val in (1, 2):
                    out = np.asarray(res.x, dtype=float)
                    out = circ_moving_average(out, int(self.smooth_window * 1.5))
                else:
                    out = circ_moving_average(data_line, self.smooth_window)
            except Exception:
                out = circ_moving_average(data_line, self.smooth_window)

            return np.minimum(np.maximum(out, lo), hi)

        def generate_next_candidate(self, lap_index: int):
            """Exact replica of your prepare_next_candidate logic"""
            base = self.best_line.copy()
            current_amp = self.exploration_amp

            signed_dyn_kappa = self.dynamic_kappa * np.sign(self.track.kappa)
            physical_k_limit = 0.6
            normalized_kappa = np.clip(signed_dyn_kappa / physical_k_limit, -1.0, 1.0)

            # Your exact sector masking math
            n_points = len(self.track.s)
            mutation_mask = np.ones(n_points)
            sector_start_idx = np.random.randint(0, n_points)
            sector_length_idx = int(n_points * 0.3) 
        
            for i in range(n_points):
                dist = min(abs(i - sector_start_idx), n_points - abs(i - sector_start_idx))
                if dist > sector_length_idx / 2:
                    mutation_mask[i] = 0.0
        
            mutation_mask = circ_moving_average(mutation_mask, int(self.smooth_window * 2))
            apex_pull = current_amp * (normalized_kappa ** 3)

            ds_nom = float(np.median(np.diff(self.track.s))) if len(self.track.s) > 2 else 0.1
            shift_idx = int(5.0 / ds_nom) 

            entry_push = -0.6 * np.roll(apex_pull, -shift_idx)
            exit_push  = -0.6 * np.roll(apex_pull, shift_idx)

            # Applying your mask
            raw_shift = (apex_pull + entry_push + exit_push) * mutation_mask
            shift = circ_moving_average(raw_shift, int(self.smooth_window * 2))
            cand = base + shift

            cand = self._minimum_curvature_qp(cand, base)
        
            self.candidate_line = cand
            # Accept the candidate line to simulate sequential laps
            self.best_line = self.candidate_line.copy()
            self._update_dynamic_curvature() 
        
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
    
        if not os.path.exists(TRACK_CSV_PATH):
            print(f"Error: Could not find {TRACK_CSV_PATH}. Please update the path in the script.")
            return

        track = load_csv_track(TRACK_CSV_PATH)
        frenet = FrenetProjector(track)
    
        # Initialize generator based on your exact file logic
        generator = CandidateLineGenerator(track, frenet)

        # Calculate actual track boundaries (0.0 margin) for plotting the physical walls
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

        # --- Generate Iterations & Plot ---
        iterations = 100
        plt.figure(figsize=(12, 10))
    
        plt.plot(x_inner, y_inner, 'k--', linewidth=1.5, label='Actual Track Limits')
        plt.plot(x_outer, y_outer, 'k--', linewidth=1.5)
        plt.plot(x_center, y_center, 'gray', linestyle=':', linewidth=1, label='Centerline')

        colors = plt.cm.viridis(np.linspace(0, 1, iterations))
    
        for i in range(iterations):
            print(i)
            cand_ey = generator.generate_next_candidate(lap_index=i)
        
            cand_x, cand_y = [], []
            for s_val, ey_val in zip(track.s, cand_ey):
                cx, cy = frenet.frenet_to_global(s_val, ey_val)
                cand_x.append(cx)
                cand_y.append(cy)
            
            if i == iterations - 1:
                plt.plot(cand_x, cand_y, color='red', linewidth=2.5, zorder=5, label=f'Final Line (Lap {i+1})')
            elif i % 10 == 0: 
                plt.plot(cand_x, cand_y, color=colors[i], linewidth=1, alpha=0.5, label=f'Lap {i+1}')
            else:
                plt.plot(cand_x, cand_y, color=colors[i], linewidth=1, alpha=0.15)

        plt.title('Racing Line Convergence (Sectors Enabled & 0.2m Wall Margin)')
        plt.xlabel('Global X [m]')
        plt.ylabel('Global Y [m]')
        plt.axis('equal')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    if __name__ == '__main__':
        main()
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()