"""Takens delay embedding and phase-space reconstruction.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export). The source presents Takens' Embedding
Theorem twice with two near-identical implementations; both are
preserved here:

* ``TakensEmbedding`` -- the first version in the source, with
  ``fit_transform`` as the entry point.
* ``PhaseSpaceReconstructor`` -- the second version in the source,
  with ``transform`` as the entry point (``tau_`` / ``m_`` attributes).

Theory (from the source): given a scalar series ``x = [x1, ..., xN]``
sampled at interval ``dt``, the reconstructed trajectory matrix is
``X`` with rows ``v_n = [x_n, x_{n+tau}, ..., x_{n+(m-1)*tau}]`` and
``M = N - (m - 1) * tau`` rows. If the attractor manifold has
dimension ``d_A``, ``m >= 2*d_A + 1`` guarantees the delay map is a
smooth, non-intersecting diffeomorphism into R^m.

Parameter selection:
* tau -- first local minimum of the Fraser-Swinney Average Mutual
  Information (AMI).
* m -- lowest dimension where the Kennel-Brown-Abarbanel False
  Nearest Neighbours (FNN) percentage drops below threshold
  (typically < 1-2%).
"""

import numpy as np
from scipy.spatial import cKDTree


class TakensEmbedding:
    def __init__(self, tau: int = None, m: int = None, max_tau: int = 50, max_m: int = 10):
        self.tau = tau
        self.m = m
        self.max_tau = max_tau
        self.max_m = max_m

    def average_mutual_information(self, data: np.ndarray, bins: int = 64) -> np.ndarray:
        """Calculates AMI across integer delays up to max_tau."""
        ami_vals = np.zeros(self.max_tau)
        n = len(data)

        for t in range(1, self.max_tau + 1):
            x1 = data[:-t]
            x2 = data[t:]

            p_joint, _, _ = np.histogram2d(x1, x2, bins=bins, density=True)
            p_joint = p_joint / np.sum(p_joint)

            p1 = np.sum(p_joint, axis=1, keepdims=True)
            p2 = np.sum(p_joint, axis=0, keepdims=True)

            # Mask out zero-probability cells to avoid log2(0)
            valid = (p_joint > 0) & (p1 @ p2 > 0)
            ami_vals[t - 1] = np.sum(p_joint[valid] * np.log2(p_joint[valid] / (p1 @ p2)[valid]))

        return ami_vals

    def find_optimal_tau(self, data: np.ndarray) -> int:
        """Finds first local minimum of AMI."""
        ami = self.average_mutual_information(data)
        for i in range(1, len(ami) - 1):
            if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
                self.tau = i + 1
                return self.tau
        # Fallback to absolute minimum if monotonic
        self.tau = int(np.argmin(ami) + 1)
        return self.tau

    def false_nearest_neighbors(self, data: np.ndarray, tau: int, r_tol: float = 15.0,
                                a_tol: float = 2.0) -> np.ndarray:
        """Calculates FNN percentage across dimensions 1 to max_m."""
        r_data = np.std(data)
        fnn_pct = np.zeros(self.max_m)

        for dim in range(1, self.max_m + 1):
            # Construct embedded coordinates up to dim + 1
            M = len(data) - dim * tau
            embedded_m = np.column_stack([data[i * tau: i * tau + M] for i in range(dim)])

            # KD-Tree nearest neighbor lookup in R^m
            tree = cKDTree(embedded_m)
            # k=2: index 0 is the query point itself, index 1 is the true nearest neighbor
            distances, indices = tree.query(embedded_m, k=2)

            r_m = distances[:, 1]
            nn_idx = indices[:, 1]

            # Coordinate delta at dimension m + 1.
            # NOTE: the source as transcribed slices
            # data[M + (dim - 1) * tau:] here (length tau), which cannot
            # broadcast against the M nearest-neighbour indices. The
            # (dim+1)-th coordinate of embedded row i is data[i + dim*tau],
            # i.e. data[dim*tau : dim*tau + M]; corrected to that form.
            delta_coord = np.abs(data[dim * tau: dim * tau + M] - data[nn_idx + dim * tau])
            r_m_plus_1 = np.sqrt(r_m ** 2 + delta_coord[:len(r_m)] ** 2)

            # FNN Criteria evaluation
            test1 = (delta_coord[:len(r_m)] / np.maximum(r_m, 1e-12)) > r_tol
            test2 = (r_m_plus_1 / r_data) > a_tol

            false_neighbors = test1 | test2
            fnn_pct[dim - 1] = np.mean(false_neighbors)

        return fnn_pct

    def find_optimal_m(self, data: np.ndarray, tau: int, threshold: float = 0.02) -> int:
        """Identifies minimum dimension where FNN percentage drops below threshold."""
        fnn = self.false_nearest_neighbors(data, tau)
        for dim, val in enumerate(fnn, start=1):
            if val <= threshold:
                self.m = dim
                return self.m
        self.m = int(np.argmin(fnn) + 1)
        return self.m

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Executes auto-tuning and generates the phase space trajectory matrix X."""
        if self.tau is None:
            self.find_optimal_tau(data)
        if self.m is None:
            self.find_optimal_m(data, self.tau)

        M = len(data) - (self.m - 1) * self.tau
        X = np.empty((M, self.m))
        for i in range(self.m):
            X[:, i] = data[i * self.tau: i * self.tau + M]

        return X


class PhaseSpaceReconstructor:
    """Second implementation of the same pipeline from the source.

    Same AMI / FNN methods, with ``tau_`` / ``m_`` fitted attributes
    and ``transform`` as the entry point.
    """

    def __init__(self, max_tau: int = 50, max_m: int = 10):
        self.max_tau = max_tau
        self.max_m = max_m
        self.tau_ = None
        self.m_ = None

    def compute_ami(self, series: np.ndarray, bins: int = 64) -> np.ndarray:
        """Calculates Average Mutual Information up to max_tau."""
        ami = np.zeros(self.max_tau)
        n = len(series)

        for t in range(1, self.max_tau + 1):
            x1 = series[:-t]
            x2 = series[t:]

            p_joint, _, _ = np.histogram2d(x1, x2, bins=bins, density=True)
            p_joint = p_joint / np.sum(p_joint)

            p1 = np.sum(p_joint, axis=1, keepdims=True)
            p2 = np.sum(p_joint, axis=0, keepdims=True)

            valid = (p_joint > 0) & (p1 @ p2 > 0)
            ami[t - 1] = np.sum(p_joint[valid] * np.log2(p_joint[valid] / (p1 @ p2)[valid]))

        return ami

    def find_tau(self, series: np.ndarray) -> int:
        """Finds first local minimum of AMI."""
        ami = self.compute_ami(series)
        for i in range(1, len(ami) - 1):
            if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
                self.tau_ = i + 1
                return self.tau_
        self.tau_ = int(np.argmin(ami) + 1)
        return self.tau_

    def compute_fnn(self, series: np.ndarray, tau: int, r_tol: float = 15.0,
                    a_tol: float = 2.0) -> np.ndarray:
        """Calculates False Nearest Neighbor percentage across dimensions 1 to max_m."""
        sigma = np.std(series)
        fnn_ratios = np.zeros(self.max_m)

        for dim in range(1, self.max_m + 1):
            M = len(series) - dim * tau
            embedded = np.column_stack([series[i * tau: i * tau + M] for i in range(dim)])

            tree = cKDTree(embedded)
            distances, indices = tree.query(embedded, k=2)

            r_m = distances[:, 1]
            nn_idx = indices[:, 1]

            # Step into dimension dim + 1 (same correction as in
            # TakensEmbedding: the source's data[M + (dim-1)*tau:] slice
            # cannot broadcast; row i's next coordinate is data[i + dim*tau]).
            delta = np.abs(series[dim * tau: dim * tau + M] - series[nn_idx + dim * tau])
            r_next = np.sqrt(r_m ** 2 + delta[:len(r_m)] ** 2)

            fnn_mask = ((delta[:len(r_m)] / np.maximum(r_m, 1e-12)) > r_tol) | ((r_next / sigma) > a_tol)
            fnn_ratios[dim - 1] = np.mean(fnn_mask)

        return fnn_ratios

    def find_m(self, series: np.ndarray, tau: int, threshold: float = 0.02) -> int:
        """Identifies optimal embedding dimension m."""
        fnn = self.compute_fnn(series, tau)
        for dim, val in enumerate(fnn, start=1):
            if val <= threshold:
                self.m_ = dim
                return self.m_
        self.m_ = int(np.argmin(fnn) + 1)
        return self.m_

    def transform(self, series: np.ndarray, tau: int = None, m: int = None) -> np.ndarray:
        """Generates phase space embedding matrix X in R^(M x m)."""
        tau = tau or self.tau_ or self.find_tau(series)
        m = m or self.m_ or self.find_m(series, tau)

        M = len(series) - (m - 1) * tau
        X = np.empty((M, m))
        for col in range(m):
            X[:, col] = series[col * tau: col * tau + M]

        return X


if __name__ == "__main__":
    # Small smoke demo (not part of the source): a noisy sinusoid.
    rng = np.random.default_rng(0)
    t = np.arange(0, 100, 0.05)
    series = np.sin(t) + 0.5 * np.sin(2.3 * t) + 0.05 * rng.standard_normal(len(t))
    embedder = TakensEmbedding(max_tau=30, max_m=6)
    X = embedder.fit_transform(series)
    print(f"Optimal tau: {embedder.tau}, optimal m: {embedder.m}")
    print(f"Phase-space matrix X shape: {X.shape}")
