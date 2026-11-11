"""Lorenz-63 integrator + Rosenstein maximal Lyapunov exponent benchmark.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export).

================================================================================
LORENZ-63 MAXIMUM LYAPUNOV EXPONENT BENCHMARK
Theoretical Target: lambda_1 ~ 0.9056 nats/time-unit (sigma=10, rho=28, beta=8/3)
================================================================================

The standard benchmark for validating the Rosenstein algorithm is the
classic Lorenz-63 attractor. The system's largest Lyapunov exponent
has been numerically proven across the literature to converge to
lambda_1 ~ 0.9056 +/- 0.0005 nats/unit time.

The harness generates the Lorenz-63 trajectory, discards transient
dynamics, benchmarks the divergence algorithm on both the direct 3D
state space and the delay-reconstructed manifold via Takens'
embedding, and evaluates the linear scaling window.

Reported results in the source:
* Test A (true 3D): lambda_1 = 0.9124, error 0.75%, R^2 = 0.9984,
  Theiler window 16 steps, linear range [0.12s, 0.47s].
* Test B (x(t) only, tau=17, m=3): lambda_1 = 0.8982, error 0.82%,
  R^2 = 0.9976, linear range [0.15s, 0.50s].

VERIFICATION NOTE: re-running this code as extracted does NOT
reproduce those numbers. Observed: Test A lambda_1 ~ 1.66 (error
~83%), Test B lambda_1 ~ 1.45 (error ~60%), Theiler window 38 steps
(the Lorenz x-autocorrelation genuinely first drops below 1/e at lag
38). The divergence curve's early slope is steep (~1.6) and decays
toward ~1.05, and the max-R^2 window picker locks onto that early
segment. The Lorenz system itself is correct -- spectrum.py
independently recovers lambda_1 ~ 0.9096. The source's printed
benchmark output appears illustrative rather than measured; treat
this harness as a starting point, not a validated benchmark. See
README.md "Verification notes".

Note: the source's print statements for the optimal linear range and
the Test B Theiler window were split across lines by the PDF text
extraction; they are reconstructed here in the obvious form matching
Test A. Re-run to confirm the values on your own machine.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist


# ============================================================================
# 1. LORENZ-63 SYSTEM INTEGRATOR
# ============================================================================
def lorenz63(t: float, state: np.ndarray, sigma: float = 10.0, rho: float = 28.0,
             beta: float = 8.0 / 3.0) -> list[float]:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def generate_lorenz_data(t_max: float = 80.0, dt: float = 0.01,
                         transient_time: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates Lorenz-63 equations and discards initial transient settling.
    """
    t_eval = np.arange(0, t_max + transient_time, dt)
    initial_state = [1.0, 1.0, 1.0]

    sol = solve_ivp(
        fun=lorenz63,
        t_span=(0, t_max + transient_time),
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12
    )

    # Strip transient orbital settling
    idx_start = int(transient_time / dt)
    t = sol.t[idx_start:] - transient_time
    trajectory = sol.y[:, idx_start:].T  # Shape: (N, 3)

    return t, trajectory


# ============================================================================
# 2. VECTORIZED ROSENSTEIN MLE ENGINE
# ============================================================================
class RosensteinBenchmark:
    def __init__(self, dt: float = 0.01, theiler_window: int = None, max_steps: int = 150):
        self.dt = dt
        self.theiler_window = theiler_window
        self.max_steps = max_steps

    def _estimate_theiler_window(self, series: np.ndarray) -> int:
        """Finds lag where normalized autocorrelation drops below 1/e (~0.368)."""
        n = len(series)
        centered = series - np.mean(series)
        autocorr = np.correlate(centered, centered, mode="full")[n - 1:]
        autocorr /= autocorr[0]
        decay = np.where(autocorr <= (1.0 / np.e))[0]
        return int(decay[0]) if len(decay) > 0 else 15

    def fit_divergence(self, X: np.ndarray, base_series: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        M, _ = X.shape

        if self.theiler_window is None:
            ref = base_series if base_series is not None else X[:, 0]
            self.theiler_window = self._estimate_theiler_window(ref)

        valid_M = M - self.max_steps
        if valid_M <= 0:
            raise ValueError(f"State space length {M} too short for {self.max_steps} max steps.")

        # Compute pairwise spatial distances across analysis chunk
        dists = cdist(X[:valid_M], X[:valid_M], metric="euclidean")

        # Enforce Theiler window exclusion: |j - j_hat| > W
        time_diff = np.abs(np.arange(valid_M)[:, None] - np.arange(valid_M)[None, :])
        dists[time_diff <= self.theiler_window] = np.inf

        # Find spatial nearest neighbor indices
        nn_indices = np.argmin(dists, axis=1)

        # Vectorized forward separation tracking
        divergence = np.zeros(self.max_steps)
        for i in range(self.max_steps):
            pts_ref = X[i: valid_M + i]
            pts_nn = X[nn_indices + i]
            step_dists = np.linalg.norm(pts_ref - pts_nn, axis=1)

            # Mask out non-physical zeros
            valid = step_dists > 1e-12
            divergence[i] = np.mean(np.log(step_dists[valid]))

        t_axis = np.arange(self.max_steps) * self.dt
        return t_axis, divergence

    def extract_lambda_auto(self, t_axis: np.ndarray, divergence: np.ndarray,
                            window_len: int = 40) -> tuple[float, float, tuple[int, int]]:
        """
        Scans a sliding window across the divergence curve to locate the
        region with the highest linear coefficient of determination (R^2).
        """
        best_r2 = -1.0
        best_slope = 0.0
        best_indices = (0, window_len)

        # Limit search to exclude initial microscopic noise and late macroscopic saturation
        start_bound = int(len(divergence) * 0.05)
        end_bound = int(len(divergence) * 0.70)

        for s in range(start_bound, end_bound - window_len):
            e = s + window_len
            t_win = t_axis[s:e]
            div_win = divergence[s:e]

            # Linear regression: div = slope * t + intercept
            p, residuals, _, _, _ = np.polyfit(t_win, div_win, deg=1, full=True)
            slope = p[0]

            # R^2 calculation
            ss_tot = np.sum((div_win - np.mean(div_win)) ** 2)
            ss_res = residuals[0] if len(residuals) > 0 else np.sum((div_win - np.polyval(p, t_win)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

            if r2 > best_r2 and slope > 0:
                best_r2 = r2
                best_slope = slope
                best_indices = (s, e)

        return best_slope, best_r2, best_indices


# ============================================================================
# 3. VERIFICATION EXECUTION
# ============================================================================
if __name__ == "__main__":
    THEORETICAL_LAMBDA = 0.9056
    DT = 0.01

    print("[+] Integrating Lorenz-63 equations (RK45, dt=0.01s)...")
    t, trajectory = generate_lorenz_data(t_max=60.0, dt=DT, transient_time=20.0)
    x_scalar = trajectory[:, 0]

    # Setup test harness
    benchmark = RosensteinBenchmark(dt=DT, max_steps=120)

    # ------------------------------------------------------------------------
    # Test A: Direct 3D Phase Space (Baseline Verification)
    # ------------------------------------------------------------------------
    print("\n--- TEST CASE A: True 3D Phase Space (x, y, z) ---")
    t_axis_3d, div_3d = benchmark.fit_divergence(trajectory)
    lambda_3d, r2_3d, win_3d = benchmark.extract_lambda_auto(t_axis_3d, div_3d, window_len=35)

    err_3d = np.abs(lambda_3d - THEORETICAL_LAMBDA) / THEORETICAL_LAMBDA * 100.0
    print(f"Theiler Window (W)    : {benchmark.theiler_window} steps ({benchmark.theiler_window * DT:.2f} s)")
    print(f"Optimal Linear Range : t in [{t_axis_3d[win_3d[0]]:.2f}s, {t_axis_3d[win_3d[1]]:.2f}s] (R^2 = {r2_3d:.4f})")
    print(f"Calculated lambda_1   : {lambda_3d:.4f} nats/unit")
    print(f"Theoretical lambda_1  : {THEORETICAL_LAMBDA:.4f} nats/unit")
    print(f"Discrepancy Error     : {err_3d:.2f}%")

    # ------------------------------------------------------------------------
    # Test B: 1D Scalar Embedding via Takens (x-coordinate alone)
    # ------------------------------------------------------------------------
    print("\n--- TEST CASE B: Delay-Reconstructed Space from x(t) alone ---")
    # For Lorenz-63, optimal AMI delay is typically tau ~ 15-18 steps at dt=0.01; m=3 unfolds the attractor
    TAU_DELAY = 17
    EMBED_DIM = 3

    M = len(x_scalar) - (EMBED_DIM - 1) * TAU_DELAY
    X_reconstructed = np.column_stack([x_scalar[k * TAU_DELAY: k * TAU_DELAY + M] for k in range(EMBED_DIM)])

    # Reset benchmark for reconstructed coordinates
    benchmark_recon = RosensteinBenchmark(dt=DT, max_steps=120)
    t_axis_rc, div_rc = benchmark_recon.fit_divergence(X_reconstructed, base_series=x_scalar)
    lambda_rc, r2_rc, win_rc = benchmark_recon.extract_lambda_auto(t_axis_rc, div_rc, window_len=35)

    err_rc = np.abs(lambda_rc - THEORETICAL_LAMBDA) / THEORETICAL_LAMBDA * 100.0
    print(f"Delay Coordinates     : tau = {TAU_DELAY} steps ({TAU_DELAY * DT:.2f} s), m = {EMBED_DIM}")
    print(f"Theiler Window (W)    : {benchmark_recon.theiler_window} steps ({benchmark_recon.theiler_window * DT:.2f} s)")
    print(f"Optimal Linear Range : t in [{t_axis_rc[win_rc[0]]:.2f}s, {t_axis_rc[win_rc[1]]:.2f}s] (R^2 = {r2_rc:.4f})")
    print(f"Calculated lambda_1   : {lambda_rc:.4f} nats/unit")
    print(f"Theoretical lambda_1  : {THEORETICAL_LAMBDA:.4f} nats/unit")
    print(f"Discrepancy Error     : {err_rc:.2f}%")
