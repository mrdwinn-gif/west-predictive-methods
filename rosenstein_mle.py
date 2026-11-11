"""Maximal Lyapunov exponent via the Rosenstein algorithm.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export).

Rosenstein, Collins, & De Luca (1993): the separation ``d_j(i)``
between a fiducial point ``X_j`` and its nearest neighbour after ``i``
steps grows as ``d_j(i) ~ C_j * exp(lambda_1 * i * dt)``, so
``ln d_j(i) ~ ln C_j + lambda_1 * (i * dt)``. ``lambda_1`` is the slope
of the linear scaling region of the ensemble-averaged log-divergence
``y(i)``.

A Theiler window ``W`` excludes neighbours that are adjacent in time
rather than in space (``|j - j_hat| > W``, typically the lag where
autocorrelation drops below 1/e).

Regime identification (from the source):
* lambda_1 > 0  -- deterministic chaos
* lambda_1 ~ 0  -- limit cycle / quasi-periodic
* lambda_1 < 0  -- stable fixed-point attractor
* no linear regime -- stochastic noise / non-deterministic
"""

import numpy as np
from scipy.spatial.distance import cdist


class RosensteinMLE:
    def __init__(self, dt: float = 1.0, theiler_window: int = None, max_steps: int = 50):
        """
        Args:
            dt: Sampling interval of original continuous series (seconds/time-units).
            theiler_window: Temporal exclusion separation (W). If None, calculated via autocorrelation.
            max_steps: Maximum forward propagation steps (i_max) to track divergence.
        """
        self.dt = dt
        self.theiler_window = theiler_window
        self.max_steps = max_steps

    def _estimate_theiler_window(self, x_series: np.ndarray) -> int:
        """Determines W as the lag where autocorrelation drops below 1/e (~0.368)."""
        n = len(x_series)
        x_centered = x_series - np.mean(x_series)
        autocorr = np.correlate(x_centered, x_centered, mode='full')[n - 1:]
        autocorr /= autocorr[0]

        decay_idx = np.where(autocorr <= (1.0 / np.e))[0]
        return int(decay_idx[0]) if len(decay_idx) > 0 else 10

    def fit(self, X: np.ndarray, base_series: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the divergence curve y(i) across forward time steps.

        Args:
            X: Embedded phase-space trajectory matrix of shape (M, m).
            base_series: Raw 1D time-series used to determine Theiler window if not provided.

        Returns:
            t_axis: Time vector in physical units (i * dt).
            divergence: Average log-divergence y(i) at each step.
        """
        M, m = X.shape

        # 1. Resolve Theiler Window
        if self.theiler_window is None:
            if base_series is not None:
                self.theiler_window = self._estimate_theiler_window(base_series)
            else:
                self.theiler_window = self._estimate_theiler_window(X[:, 0])

        # Available trajectory indices to prevent out-of-bounds indexing at (j + max_steps)
        valid_M = M - self.max_steps
        if valid_M <= 0:
            raise ValueError(f"Trajectory matrix length ({M}) is smaller than max_steps ({self.max_steps}).")

        # 2. Find Nearest Neighbors with Theiler Exclusion
        # Pairwise Euclidean distances between all valid query points and candidate targets
        dist_matrix = cdist(X[:valid_M], X[:valid_M], metric='euclidean')

        # Apply infinite penalty to points within the temporal exclusion zone
        time_diff = np.abs(np.arange(valid_M)[:, None] - np.arange(valid_M)[None, :])
        dist_matrix[time_diff <= self.theiler_window] = np.inf

        # Identify index of closest spatial neighbor for each row
        nn_indices = np.argmin(dist_matrix, axis=1)

        # 3. Track Forward Divergence Across i Steps
        divergence = np.zeros(self.max_steps)
        for i in range(self.max_steps):
            # Compute trajectory states at offset i
            pts_ref = X[i: valid_M + i]
            pts_nn = X[nn_indices + i]

            # Distance: ||X_{j+i} - X_{hat{j}+i}||
            step_dists = np.linalg.norm(pts_ref - pts_nn, axis=1)

            # Disregard numerical zero distances to avoid log(0)
            valid_dists = step_dists[step_dists > 1e-12]
            if len(valid_dists) > 0:
                divergence[i] = np.mean(np.log(valid_dists))
            else:
                divergence[i] = -np.inf

        t_axis = np.arange(self.max_steps) * self.dt
        return t_axis, divergence

    def extract_lambda(self, t_axis: np.ndarray, divergence: np.ndarray,
                       fit_start_step: int, fit_end_step: int) -> float:
        """
        Fits a linear regression across the identified scaling window to extract lambda_1.
        """
        t_slice = t_axis[fit_start_step:fit_end_step]
        div_slice = divergence[fit_start_step:fit_end_step]

        # 1st order polynomial fit: div = lambda_1 * t + C
        poly = np.polyfit(t_slice, div_slice, deg=1)
        lambda_1 = poly[0]
        return float(lambda_1)


if __name__ == "__main__":
    # Usage from the source (assumes dt = 0.01 between raw scalar
    # measurements, and X is a reconstructed phase-space matrix):
    #   mle_engine = RosensteinMLE(dt=0.01, max_steps=60)
    #   t_vec, divergence_curve = mle_engine.fit(X=X)
    #   lambda_1 = mle_engine.extract_lambda(t_vec, divergence_curve,
    #                                        fit_start_step=5, fit_end_step=25)
    #   print(f"Maximal Lyapunov Exponent: {lambda_1:.4f} nats/sec")
    #
    # A runnable end-to-end verification against the Lorenz-63
    # theoretical value (lambda_1 ~ 0.9056) is in lorenz_harness.py.
    print("RosensteinMLE loaded. See lorenz_harness.py for the runnable Lorenz-63 benchmark.")
