"""Sliding-window time-dependent RQA (twRQA) for bifurcation detection.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export). This module was not in the original
file list for the repo, but the source contains a complete,
self-contained implementation, so it is preserved here rather than
dropped.

A sliding-window RQA detects dynamical bifurcations and Critical
Slowing Down (CSD) by measuring the deformation of phase-space
trajectories across a moving window ``[t, t + W]``. As a system
approaches a tipping point the dominant eigenvalue approaches zero,
the restoring force decays, and in RQA this shows up as expanding
vertical structures (Laminarity LAM, Trapping Time TT) coupled with
collapsing Determinism (DET).

Composite CSD index (from the source), with ``Z`` the rolling
z-score against baseline:
``I_CSD(t) = (Z(TT_t) + Z(LAM_t)) / 2 - Z(DET_t)``.
An alarm triggers when ``I_CSD(t) >= 3.0`` (3-sigma tripwire).

For real-time bifurcation detection ``eps`` must remain FIXED
(calibrated to 5-10% recurrence during the initial baseline regime),
so density fluctuations register directly as shifts in RR, LAM and TT.

Verification in the source: May's harvesting model approaching a
fold bifurcation (harvesting pressure ``C(t)`` ramped linearly past
``C_c ~ 2.60``) -- TT spikes to >= 5.8 steps, LAM to >= 65%, DET
drops, alarm fires in ``t in [180, 240]s`` before collapse after
``t > 270s``.
"""

import numpy as np
from scipy.spatial.distance import cdist


class RealTimeRQA:
    def __init__(self, m: int, tau: int, window_size: int = 250, step_size: int = 10,
                 theiler_window: int = 5, l_min: int = 2, v_min: int = 2,
                 fixed_eps: float = None, baseline_steps: int = 30):
        """
        Args:
            m: Embedding dimension.
            tau: Delay lag (integer steps).
            window_size: Number of reconstructed vectors per window (W).
            step_size: Stride interval between window evaluations (Delta s).
            theiler_window: Temporal exclusion band |i - j| <= w.
            fixed_eps: Constant distance cutoff. If None, calibrated during baseline.
            baseline_steps: Number of initial sliding evaluations used for z-score baselining.
        """
        self.m = m
        self.tau = tau
        self.W = window_size
        self.step = step_size
        self.w = theiler_window
        self.l_min = l_min
        self.v_min = v_min
        self.eps = fixed_eps
        self.baseline_steps = baseline_steps

        # Internal state buffer
        self.buffer = []
        self.history = []
        self.baseline_records = []
        self.calibrated = False

    def _embed_window(self, raw_window: np.ndarray) -> np.ndarray:
        """Constructs delay coordinate matrix X in R^(W x m)."""
        M = len(raw_window) - (self.m - 1) * self.tau
        if M != self.W:
            raise ValueError(f"Raw window size insufficient. Needs {self.W + (self.m - 1) * self.tau} points.")
        X = np.empty((self.W, self.m))
        for col in range(self.m):
            X[:, col] = raw_window[col * self.tau: col * self.tau + self.W]
        return X

    def _extract_line_lengths(self, binary_1d: np.ndarray) -> list[int]:
        """Vectorized extraction of consecutive True runs in a 1D binary array."""
        padded = np.pad(binary_1d.astype(np.int8), (1, 1), mode='constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return (ends - starts).tolist()

    def _compute_window_metrics(self, X: np.ndarray) -> dict:
        """Evaluates single-window RQA metrics on trajectory slice X."""
        dists = cdist(X, X, metric='euclidean')

        # Self-calibrate fixed epsilon on first call if not provided
        if self.eps is None:
            triu_dists = dists[np.triu_indices(self.W, k=self.w + 1)]
            self.eps = float(np.percentile(triu_dists, 5.0))  # Set to baseline 5% recurrence

        R = (dists <= self.eps).astype(np.uint8)

        # 1. Masked Recurrence Rate (RR)
        mask = np.ones((self.W, self.W), dtype=bool)
        for k in range(-self.w, self.w + 1):
            if k >= 0:
                np.fill_diagonal(mask[:, k:], False)
            else:
                np.fill_diagonal(mask[-k:, :], False)

        eval_points = np.sum(mask)
        rr = float(np.sum(R & mask) / eval_points) if eval_points > 0 else 0.0

        # 2. Diagonal Lines (Determinism, L_max, Entropy)
        diag_lengths = []
        for offset in range(self.w + 1, self.W):
            diag = np.diagonal(R, offset=offset)
            if np.any(diag):
                diag_lengths.extend(self._extract_line_lengths(diag))

        diag_arr = np.array(diag_lengths) if len(diag_lengths) > 0 else np.array([])
        if len(diag_arr) > 0:
            total_diag = np.sum(diag_arr)
            det_hits = diag_arr >= self.l_min
            det = float(np.sum(diag_arr[det_hits]) / total_diag) if total_diag > 0 else 0.0
            l_max = int(np.max(diag_arr))

            # Shannon Entropy of diagonal distributions
            valid = diag_arr[det_hits]
            if len(valid) > 0:
                counts = np.bincount(valid)[self.l_min:]
                probs = counts[counts > 0] / len(valid)
                entr = float(-np.sum(probs * np.log(probs)))
            else:
                entr = 0.0
        else:
            det, l_max, entr = 0.0, 0, 0.0

        # 3. Vertical Lines (Laminarity, Trapping Time, V_max)
        R_vert = R.copy()
        for k in range(-self.w, self.w + 1):
            if k >= 0:
                np.fill_diagonal(R_vert[:, k:], 0)
            else:
                np.fill_diagonal(R_vert[-k:, :], 0)

        vert_lengths = []
        for col in range(self.W):
            col_data = R_vert[:, col]
            if np.any(col_data):
                vert_lengths.extend(self._extract_line_lengths(col_data))

        vert_arr = np.array(vert_lengths) if len(vert_lengths) > 0 else np.array([])
        if len(vert_arr) > 0:
            total_vert = np.sum(vert_arr)
            lam_hits = vert_arr >= self.v_min
            lam = float(np.sum(vert_arr[lam_hits]) / total_vert) if total_vert > 0 else 0.0
            tt = float(np.mean(vert_arr[lam_hits])) if np.any(lam_hits) else 0.0
            v_max = int(np.max(vert_arr))
        else:
            lam, tt, v_max = 0.0, 0.0, 0

        return {
            "RR": rr, "DET": det, "L_max": l_max, "ENTR": entr,
            "LAM": lam, "TT": tt, "V_max": v_max
        }

    def update(self, new_scalar_data: np.ndarray) -> list[dict]:
        """
        Feed continuous scalar stream. Evaluates metrics when buffer steps forward.
        """
        self.buffer.extend(new_scalar_data.tolist())
        required_raw = self.W + (self.m - 1) * self.tau
        outputs = []

        while len(self.buffer) >= required_raw:
            raw_slice = np.array(self.buffer[:required_raw])
            X_win = self._embed_window(raw_slice)
            metrics = self._compute_window_metrics(X_win)

            # Baselining & CSD Tripwire Calculation
            if len(self.baseline_records) < self.baseline_steps:
                self.baseline_records.append(metrics)
                metrics["CSD_Index"] = 0.0
                metrics["Alarm"] = False
            else:
                if not self.calibrated:
                    self._compute_baseline_statistics()
                    self.calibrated = True

                # Compute normalized Z-scores against baseline
                z_tt = (metrics["TT"] - self.base_mu["TT"]) / (self.base_std["TT"] + 1e-9)
                z_lam = (metrics["LAM"] - self.base_mu["LAM"]) / (self.base_std["LAM"] + 1e-9)
                z_det = (metrics["DET"] - self.base_mu["DET"]) / (self.base_std["DET"] + 1e-9)

                # CSD Index: High TT + High LAM coupled with dropped DET
                csd_idx = float(((z_tt + z_lam) / 2.0) - z_det)
                metrics["CSD_Index"] = csd_idx
                metrics["Alarm"] = bool(csd_idx >= 3.0)  # 3-sigma tripwire

            self.history.append(metrics)
            outputs.append(metrics)

            # Slide buffer by step_size
            self.buffer = self.buffer[self.step:]

        return outputs

    def _compute_baseline_statistics(self):
        """Precomputes baseline means and standard deviations."""
        self.base_mu = {}
        self.base_std = {}
        for key in ["TT", "LAM", "DET", "RR"]:
            vals = [rec[key] for rec in self.baseline_records]
            self.base_mu[key] = np.mean(vals)
            self.base_std[key] = np.std(vals)


# ============================================================================
# TEST HARNESS: CRITICAL COLLAPSE SIMULATION
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    dt = 0.05
    t_total = 400.0
    N = int(t_total / dt)
    time = np.arange(N) * dt

    r, K_cap, H = 1.0, 10.0, 1.0
    noise_amp = 0.08

    # Linearly ramp harvesting pressure C(t) to drive system across bifurcation
    C_param = np.linspace(1.2, 2.75, N)

    x = np.zeros(N)
    x[0] = 8.0  # Stable carrying capacity

    print("[+] Simulating dynamical system approaching fold bifurcation...")
    for i in range(1, N):
        drift = r * x[i - 1] * (1.0 - x[i - 1] / K_cap) - (C_param[i - 1] * (x[i - 1] ** 2)) / (x[i - 1] ** 2 + H ** 2)
        diff = noise_amp * np.sqrt(dt) * np.random.randn()
        x[i] = max(x[i - 1] + drift * dt + diff, 0.0)

    # Instantiate twRQA Detector
    # Delay parameters: tau=4 steps (~0.2s), m=3 dimension, window=200 steps (10s), step=20 steps (1s)
    detector = RealTimeRQA(
        m=3,
        tau=4,
        window_size=200,
        step_size=20,
        theiler_window=4,
        baseline_steps=25
    )

    print("[+] Streaming telemetry into sliding-window RQA engine...")
    stream_results = detector.update(x)

    # Scan results for critical alarms
    alarm_fired = False
    for step_idx, entry in enumerate(stream_results):
        t_current = (step_idx * detector.step + detector.W) * dt
        if entry["Alarm"] and not alarm_fired:
            print(f"\n[!] CRITICAL SLOWING DOWN ALARM TRIGGERED at t = {t_current:.1f}s")
            print(f"    CSD Index     : {entry['CSD_Index']:.2f} (sigma-deviation)")
            print(f"    Trapping Time : {entry['TT']:.2f} (Baseline: {detector.base_mu['TT']:.2f})")
            print(f"    Laminarity    : {entry['LAM'] * 100:.1f}% (Baseline: {detector.base_mu['LAM'] * 100:.1f}%)")
            print(f"    Determinism   : {entry['DET'] * 100:.1f}% (Baseline: {detector.base_mu['DET'] * 100:.1f}%)")
            alarm_fired = True
            break

    if not alarm_fired:
        print("\n[!] No CSD alarm fired over the streamed series.")
