"""Theiler-window-corrected False Nearest Neighbours with PCA projection.

Extracted from a Gemini chat export titled "Telemetry Phase Space
Reconstruction Pipeline" (source: ~/workspace/user/files/
Telemetry_Phase_Space_Reconstruction_Pipeline_1_kywj.docx).

What is new here vs. what the repo already had
----------------------------------------------
* ``takens_embedding.py`` already covers scalar Takens embedding with
  AMI delay selection and Kennel et al. FNN (scalar series only).
  The scalar ``estimate_delay_ami`` / ``compute_fnn_ratio`` blocks in
  this source were therefore NOT duplicated here.
* This module adds what was missing: MULTIVARIATE zero-copy strided
  embedding (``MultiStreamPhaseSpaceReconstructor``), a
  Theiler-window-corrected FNN that excludes temporal self-neighbours
  (|i - j| <= W) before the Kennel tests, and PCA (SVD) projection of
  the C x m embedded block down to its dominant dynamic modes
  (``MultivariateTheilerPCAFNN``), plus a 3-channel Lorenz-63 test
  bench (``generate_lorenz_telemetry``).

Plain-English summary of the method
------------------------------------
Takens' theorem says a single measured series can be unfolded into a
phase space by stacking delayed copies of itself. Two things can go
wrong in practice: (1) at high sampling rates a point's nearest
"neighbour" is just the next sample on the same trajectory segment
-- the Theiler window forbids neighbours that are too close in time,
so only genuine recurrences count; (2) with many sensor channels the
embedded space explodes (C channels x m delays), so PCA keeps only
the directions that carry most of the variance before neighbours are
searched. FNN then asks, dimension by dimension, whether neighbours
stay neighbours when one more delay coordinate is added; the
embedding dimension is where the false-neighbour percentage collapses.

Status: code extracted faithfully from the source; py_compile passes.
NOT runtime-validated against the source's claimed numbers -- the
source prints no FNN table for this bench, so there is nothing to
reproduce. Run ``python3 theiler_fnn_pca.py`` for the Lorenz-63
dimension spectrum.
"""

import numpy as np
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import as_strided


class MultiStreamPhaseSpaceReconstructor:
    """
    High-performance Takens delay embedding engine for multivariate continuous streams.
    """
    def __init__(self, tau: int, m: int):
        self.tau = int(tau)
        self.m = int(m)
        
    def transform_single_channel(self, series: np.ndarray) -> np.ndarray:
        """
        Creates a zero-copy strided view of shape (N_effective, m).
        """
        series = np.ascontiguousarray(series)
        n = series.shape[0]
        n_vectors = n - (self.m - 1) * self.tau
        
        if n_vectors <= 0:
            raise ValueError(f"Telemetry stream length ({n}) is insufficient for m={self.m}, tau={self.tau}")
            
        element_size = series.strides[0]
        shape = (n_vectors, self.m)
        strides = (element_size, self.tau * element_size)
        
        return as_strided(series, shape=shape, strides=strides, writeable=False)
        
    def transform_multivariate(self, telemetry_matrix: np.ndarray) -> np.ndarray:
        """
        Builds a multivariate embedded matrix for C channels:
        Input shape: (N_samples, C_channels)
        Output shape: (N_effective, C_channels * m)
        """
        n_samples, n_channels = telemetry_matrix.shape
        embedded_channels = [
            self.transform_single_channel(telemetry_matrix[:, col])
            for col in range(n_channels)
        ]
        
        # Single contiguous consolidation pass across channels
        return np.ascontiguousarray(np.concatenate(embedded_channels, axis=1))


class MultivariateTheilerPCAFNN:
    """
    Optimized False Nearest Neighbors (FNN) evaluation pipeline for multivariate 
    telemetry featuring:
      1. Zero-copy strided delay embeddings across multi-channel inputs.
      2. Temporal autocorrelation exclusion via an explicit Theiler window.
      3. PCA dynamic mode projection to mitigate cross-channel curse of dimensionality.
      4. Kennel et al. dual-criterion false neighbor verification.
    """

    def __init__(
        self,
        tau: int,
        theiler_w: int | None = None,
        r_tol: float = 15.0,
        a_tol: float = 2.0,
        variance_explained_cutoff: float = 0.95,
        max_modes: int | None = None,
    ):
        """
        :param tau: Delay lag parameter.
        :param theiler_w: Theiler exclusion window. Defaults to 3 * tau if None.
        :param r_tol: Ratio threshold for local spatial divergence (Kennel criterion 1).
        :param a_tol: Ratio threshold against global attractor scale (Kennel criterion 2).
        :param variance_explained_cutoff: Minimum cumulative variance retained by PCA.
        :param max_modes: Hard ceiling on PCA components to retain.
        """
        self.tau = int(tau)
        self.theiler_w = int(theiler_w if theiler_w is not None else 3 * tau)
        self.r_tol = float(r_tol)
        self.a_tol = float(a_tol)
        self.variance_explained_cutoff = float(variance_explained_cutoff)
        self.max_modes = max_modes

    @staticmethod
    def _embed_channel(signal: np.ndarray, m: int, tau: int) -> np.ndarray:
        """
        Extracts an (N_eff, m) delay-embedded view without heap re-allocation.
        """
        signal = np.ascontiguousarray(signal)
        n = signal.shape[0]
        n_eff = n - (m - 1) * tau
        if n_eff <= 0:
            raise ValueError(f"Stream length ({n}) insufficient for m={m}, tau={tau}")

        itemsize = signal.strides[0]
        return as_strided(
            signal,
            shape=(n_eff, m),
            strides=(itemsize, tau * itemsize),
            writeable=False,
        )

    def _multivariate_embedding(self, telemetry: np.ndarray, m: int) -> np.ndarray:
        """
        Constructs an concatenated multivariate trajectory matrix for m delays.
        Input shape: (N_samples, C_channels) -> Output shape: (N_eff, C_channels * m)
        """
        _, n_channels = telemetry.shape
        embedded_blocks = [
            self._embed_channel(telemetry[:, c], m=m, tau=self.tau)
            for c in range(n_channels)
        ]
        # Align all blocks to common length dictated by m
        min_len = min(blk.shape[0] for blk in embedded_blocks)
        aligned = [blk[:min_len] for blk in embedded_blocks]
        return np.ascontiguousarray(np.concatenate(aligned, axis=1))

    def _apply_pca_projection(self, X: np.ndarray) -> np.ndarray:
        """
        Centers input and projects onto dominant eigenvectors retaining the target
        variance cutoff.
        """
        X_centered = X - np.mean(X, axis=0)
        # Fast thin SVD
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvalues = (s ** 2) / (X.shape[0] - 1)
        var_ratio = eigenvalues / np.sum(eigenvalues)
        cum_var = np.cumsum(var_ratio)

        # Determine cutoff count
        k = int(np.searchsorted(cum_var, self.variance_explained_cutoff) + 1)
        if self.max_modes is not None:
            k = min(k, self.max_modes)
        k = max(k, 1)

        # Coordinate projection
        return X_centered @ vt[:k].T

    def _find_theiler_nearest_neighbors(
        self, points: np.ndarray, search_k: int = 64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Queries nearest neighbors via cKDTree while skipping any indices falling 
        within |i - j| <= theiler_w. Automatically scales search_k if local clusters 
        saturate the initial search buffer.
        """
        n_points = points.shape[0]
        tree = cKDTree(points)
        
        # Guard against requested k exceeding total data size
        curr_k = min(max(search_k, self.theiler_w * 2 + 2), n_points)
        
        distances, indices = tree.query(points, k=curr_k, workers=-1)
        
        nn_indices = np.empty(n_points, dtype=np.int64)
        nn_distances = np.empty(n_points, dtype=np.float64)
        valid_mask = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            row_idx = indices[i]
            row_dist = distances[i]

            # Filter out identity (dist == 0) and points inside Theiler window
            mask = (row_idx != i) & (np.abs(row_idx - i) > self.theiler_w)
            
            valid_candidates = row_idx[mask]
            valid_dists = row_dist[mask]

            if len(valid_candidates) > 0:
                nn_indices[i] = valid_candidates[0]
                nn_distances[i] = valid_dists[0]
                valid_mask[i] = True
            else:
                # Buffer exhausted due to extreme local oversampling
                valid_mask[i] = False

        return nn_distances, nn_indices, valid_mask

    def evaluate_dimension_spectrum(
        self, telemetry: np.ndarray, max_dim: int = 8
    ) -> list[dict]:
        """
        Runs Theiler-corrected FNN across dimensions 1 to max_dim on multivariate input.
        Returns percentage false neighbors, active PCA components, and sample sizes.
        """
        n_samples, _ = telemetry.shape
        # Global attractor scale estimated across flattened telemetry variation
        attractor_scale = np.std(telemetry)
        results = []

        for m in range(1, max_dim + 1):
            n_vectors = n_samples - (m * self.tau)
            if n_vectors <= (self.theiler_w * 2):
                break

            # 1. Delay embed at dimension m
            X_m = self._multivariate_embedding(telemetry, m=m)
            # Truncate to match potential m+1 step forward projection
            X_m = X_m[:n_vectors]

            # 2. PCA projection to remove redundant cross-channel correlations
            X_m_pca = self._apply_pca_projection(X_m)
            n_modes = X_m_pca.shape[1]

            # 3. Find true topological neighbors with Theiler filtering
            r_m, nn_idx, valid_pts = self._find_theiler_nearest_neighbors(X_m_pca)
            
            # Require positive distance to prevent zero division
            stable_pts = valid_pts & (r_m > 1e-12)
            n_valid = np.sum(stable_pts)

            if n_valid == 0:
                results.append({
                    "dim_m": m,
                    "pca_modes": n_modes,
                    "fnn_percent": 0.0,
                    "valid_eval_points": 0,
                })
                continue

            # 4. Extract (m+1)-th coordinate step for original and neighbor
            # For multivariate input, measure Euclidean delta across all channels at next lag
            orig_next = telemetry[np.arange(n_vectors) + (m * self.tau)]
            neighbor_next = telemetry[nn_idx + (m * self.tau)]
            delta_next = np.linalg.norm(orig_next - neighbor_next, axis=1)

            # Criterion 1: Local divergence relative to current dimension separation
            crit1 = (delta_next[stable_pts] / r_m[stable_pts]) > self.r_tol

            # Criterion 2: Global stretching against total attractor diameter
            r_m_plus_1 = np.sqrt(r_m[stable_pts] ** 2 + delta_next[stable_pts] ** 2)
            crit2 = (r_m_plus_1 / attractor_scale) > self.a_tol

            # Total False Neighbors
            fnn_flags = crit1 | crit2
            fnn_percent = (np.sum(fnn_flags) / n_valid) * 100.0

            results.append({
                "dim_m": m,
                "pca_modes": n_modes,
                "fnn_percent": float(fnn_percent),
                "valid_eval_points": int(n_valid),
            })

        return results


def generate_lorenz_telemetry(n_steps=12000, dt=0.01):
    """Generates continuous 3-channel telemetry from the Lorenz-63 attractor."""
    xyz = np.empty((n_steps, 3), dtype=np.float64)
    state = np.array([1.0, 1.0, 1.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    for i in range(n_steps):
        xyz[i] = state
        dx = sigma * (state[1] - state[0])
        dy = state[0] * (rho - state[2]) - state[1]
        dz = state[0] * state[1] - beta * state[2]
        state = state + np.array([dx, dy, dz]) * dt

    return xyz

if __name__ == "__main__":
    # Generate multi-channel feed: 12,000 steps @ dt=0.01
    telemetry_stream = generate_lorenz_telemetry()

    # Set parameters: tau=10 steps, Theiler window=30 steps (skips high-frequency orbital drift)
    pipeline = MultivariateTheilerPCAFNN(
        tau=10,
        theiler_w=30,
        r_tol=15.0,
        a_tol=2.0,
        variance_explained_cutoff=0.95
    )

    diagnostics = pipeline.evaluate_dimension_spectrum(telemetry_stream, max_dim=6)

    print(f"{'Embedding Dim (m)':<18} | {'PCA Modes':<10} | {'FNN %':<8} | {'Evaluated Vectors'}")
    print("-" * 60)
    for entry in diagnostics:
        print(
            f"{entry['dim_m']:<18} | "
            f"{entry['pca_modes']:<10} | "
            f"{entry['fnn_percent']:>6.2f}% | "
            f"{entry['valid_eval_points']}"
        )
