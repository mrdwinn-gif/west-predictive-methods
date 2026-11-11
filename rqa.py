"""Recurrence Plot and Recurrence Quantification Analysis (RQA).

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export). This module was not in the original
file list for the repo, but the source contains a complete,
self-contained implementation, so it is preserved here rather than
dropped.

A Recurrence Plot maps every time pair ``(i, j)`` where a dynamical
system returns to a neighbourhood of a former state within threshold
``eps``: ``R_ij(eps) = Theta(eps - ||x_i - x_j||)``. RQA extracts
metrics from the diagonal and vertical line structures:

* RR -- recurrence rate (density of recurrence points, outside the
  Theiler window)
* DET -- determinism (fraction of recurrence points in diagonal
  lines of length >= ``l_min``)
* L / L_max -- mean / longest diagonal length (``L_max`` is inversely
  proportional to the maximal Lyapunov exponent)
* ENTR -- Shannon entropy of diagonal line lengths
* LAM -- laminarity (fraction of recurrence points in vertical lines
  of length >= ``v_min``)
* TT -- trapping time (mean vertical line length)
* V_max -- longest vertical line

Diagnostic signatures from the source: high DET + long diagonals =
periodic; high DET (0.80-0.95) + finite L_max = deterministic chaos;
low DET + point scatter = stochastic noise; high LAM + elevated TT =
intermittency / critical slowing.
"""

import numpy as np
from scipy.spatial.distance import cdist


class RecurrenceQuantificationAnalysis:
    def __init__(self, eps: float = None, target_rr: float = 0.05,
                 theiler_window: int = 1, l_min: int = 2, v_min: int = 2):
        """
        Args:
            eps: Distance cutoff threshold. If None, auto-calibrated to target_rr.
            target_rr: Target recurrence rate used to auto-tune eps (default: 5%).
            theiler_window: Diagonal exclusion corridor width |i - j| <= w.
            l_min: Minimum length for diagonal line classification.
            v_min: Minimum length for vertical line classification.
        """
        self.eps = eps
        self.target_rr = target_rr
        self.theiler_window = theiler_window
        self.l_min = l_min
        self.v_min = v_min

    def compute_recurrence_matrix(self, X: np.ndarray) -> np.ndarray:
        """Constructs binary recurrence matrix R_ij = Theta(eps - ||x_i - x_j||)."""
        dists = cdist(X, X, metric='euclidean')

        if self.eps is None:
            # Auto-calibrate eps to achieve target recurrence rate (excluding main diagonal)
            triu_dists = dists[np.triu_indices_from(dists, k=1)]
            self.eps = float(np.percentile(triu_dists, self.target_rr * 100))

        R = (dists <= self.eps).astype(np.uint8)
        return R

    def _extract_line_lengths(self, binary_1d: np.ndarray) -> list[int]:
        """Extracts lengths of contiguous True blocks in a 1D boolean array."""
        padded = np.pad(binary_1d.astype(np.int8), (1, 1), mode='constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        lengths = (ends - starts).tolist()
        return lengths

    def extract_diagonal_lengths(self, R: np.ndarray) -> list[int]:
        """Extracts diagonal line lengths strictly outside the Theiler window."""
        M = R.shape[0]
        diag_lengths = []

        # Scan upper diagonals starting outside Theiler window
        for k in range(self.theiler_window + 1, M):
            diag = np.diagonal(R, offset=k)
            if np.any(diag):
                diag_lengths.extend(self._extract_line_lengths(diag))

        return diag_lengths

    def extract_vertical_lengths(self, R: np.ndarray) -> list[int]:
        """Extracts vertical line lengths from columns, masking the Theiler band."""
        M = R.shape[0]
        R_masked = R.copy()

        # Mask out Theiler window band around the main diagonal
        for k in range(-self.theiler_window, self.theiler_window + 1):
            if k >= 0:
                np.fill_diagonal(R_masked[:, k:], 0)
            else:
                np.fill_diagonal(R_masked[-k:, :], 0)

        vert_lengths = []
        for col in range(M):
            col_data = R_masked[:, col]
            if np.any(col_data):
                vert_lengths.extend(self._extract_line_lengths(col_data))

        return vert_lengths

    def analyze(self, X: np.ndarray) -> dict:
        """
        Runs RQA on reconstructed phase space matrix X.

        Returns:
            Dictionary containing RP matrix and computed RQA metrics.
        """
        R = self.compute_recurrence_matrix(X)
        M = R.shape[0]

        # 1. Masked Recurrence Rate (RR)
        mask = np.ones((M, M), dtype=bool)
        for k in range(-self.theiler_window, self.theiler_window + 1):
            if k >= 0:
                np.fill_diagonal(mask[:, k:], False)
            else:
                np.fill_diagonal(mask[-k:, :], False)

        eval_points = np.sum(mask)
        recurrent_points = np.sum(R & mask)
        rr = float(recurrent_points / eval_points) if eval_points > 0 else 0.0

        # 2. Diagonal Line Analysis (Determinism & Stability)
        diag_lengths = np.array(self.extract_diagonal_lengths(R))
        if len(diag_lengths) > 0:
            total_diag_pts = np.sum(diag_lengths)
            det_mask = diag_lengths >= self.l_min
            det_pts = np.sum(diag_lengths[det_mask])

            det = float(det_pts / total_diag_pts) if total_diag_pts > 0 else 0.0
            l_mean = float(np.mean(diag_lengths[det_mask])) if np.any(det_mask) else 0.0
            l_max = int(np.max(diag_lengths))

            # Shannon Entropy of diagonal lines >= l_min
            valid_diags = diag_lengths[det_mask]
            if len(valid_diags) > 0:
                counts = np.bincount(valid_diags)[self.l_min:]
                probs = counts[counts > 0] / len(valid_diags)
                entr = float(-np.sum(probs * np.log(probs)))
            else:
                entr = 0.0
        else:
            det, l_mean, l_max, entr = 0.0, 0.0, 0, 0.0

        # 3. Vertical Line Analysis (Laminarity & Trapping Time)
        vert_lengths = np.array(self.extract_vertical_lengths(R))
        if len(vert_lengths) > 0:
            total_vert_pts = np.sum(vert_lengths)
            lam_mask = vert_lengths >= self.v_min
            lam_pts = np.sum(vert_lengths[lam_mask])

            lam = float(lam_pts / total_vert_pts) if total_vert_pts > 0 else 0.0
            tt = float(np.mean(vert_lengths[lam_mask])) if np.any(lam_mask) else 0.0
            v_max = int(np.max(vert_lengths))
        else:
            lam, tt, v_max = 0.0, 0.0, 0

        return {
            "R": R,
            "eps": self.eps,
            "RR": rr,
            "DET": det,
            "L_mean": l_mean,
            "L_max": l_max,
            "ENTR": entr,
            "LAM": lam,
            "TT": tt,
            "V_max": v_max
        }


if __name__ == "__main__":
    # Demonstration from the source (assumes X is a reconstructed
    # Lorenz-63 phase space matrix, e.g. from takens_embedding.py or
    # lorenz_harness.py; theiler_window=16 matches the autocorrelation
    # decay calculated there).
    #   rqa_engine = RecurrenceQuantificationAnalysis(
    #       target_rr=0.05, theiler_window=16, l_min=2, v_min=2)
    #   results = rqa_engine.analyze(X)
    #   print(f"Distance Cutoff (eps) : {results['eps']:.4f}")
    #   print(f"Recurrence Rate (RR)  : {results['RR'] * 100:.2f}%")
    #   print(f"Determinism (DET)     : {results['DET'] * 100:.2f}%")
    #   ...
    # Small runnable smoke demo on a synthetic periodic series:
    t = np.arange(0, 50, 0.05)
    series = np.sin(t) + 0.1 * np.sin(3.7 * t)
    X = np.column_stack([series[:-20], series[10:-10], series[20:]])
    results = RecurrenceQuantificationAnalysis(
        target_rr=0.05, theiler_window=5, l_min=2, v_min=2).analyze(X)
    print(f"Distance Cutoff (eps)        : {results['eps']:.4f}")
    print(f"Recurrence Rate (RR)         : {results['RR'] * 100:.2f}%")
    print(f"Determinism (DET)            : {results['DET'] * 100:.2f}%")
    print(f"Average Diag Length (L)      : {results['L_mean']:.2f} steps")
    print(f"Max Diag Length (L_max)      : {results['L_max']} steps")
    print(f"Entropy (ENTR)               : {results['ENTR']:.4f} nats")
    print(f"Laminarity (LAM)             : {results['LAM'] * 100:.2f}%")
    print(f"Trapping Time (TT)           : {results['TT']:.2f} steps")
