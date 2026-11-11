"""Sliding-window twRQA + Welford bifurcation detector (PCA phase space).

Extracted from a Gemini chat export titled "Telemetry Phase Space
Reconstruction Pipeline" (source: ~/workspace/user/files/
Telemetry_Phase_Space_Reconstruction_Pipeline_1_kywj.docx).

What is new here vs. what the repo already had
----------------------------------------------
* ``realtime_rqa.py`` already has a scalar sliding-window twRQA with a
  composite CSD index and a 3-sigma tripwire (May harvesting model).
  This module adds the PCA-projected variant: ``RealTimeSlidingtwRQA``
  operates on PCA mode coordinates (not a scalar series), returns a
  ``RQAMetrics`` dataclass including DIV = 1/L_max, and is wrapped by
  ``RealTimeBifurcationDetector`` -- a dual-window Welford tracker
  that scores each window with a Composite Anomaly Index
  CAI = 0.50*max(0, Z_DIV) + 0.30*Z_ENTR + 0.20*max(0, Z_DET) and
  fires tri-level alerts (NOMINAL / EWS_WARNING / BIFURCATION_TRIP)
  after N consecutive triggers. Test benches use synthetic data only:
  a limit-cycle -> Lorenz-63 regime switch, and a three-phase
  limit-cycle -> critical slowing down -> Roessler-chaos tipping
  simulation. Lorenz/Roessler here are synthetic test benches, not
  real telemetry.

Plain-English summary
---------------------
Recurrence quantification watches how often a trajectory revisits
nearby states. Determinism (DET) measures how much of that revisiting
forms diagonal lines (predictable motion); divergence (DIV = 1/L_max)
measures how fast nearby trajectories peel apart. As a system nears
a tipping point, divergence surges, entropy shifts, and determinism
decays -- the detector tracks those three metrics against a rolling
Welford baseline and raises an alert when the composite score stays
out of bounds for consecutive windows.

Status: code extracted faithfully; py_compile passes. The source's
claimed detector behaviour (alerts in the transition zone) was NOT
independently reproduced here -- run the demos and read the output
before relying on any threshold. See README "Verification notes".
"""

import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum



@dataclass
class RQAMetrics:
    timestamp_idx: int
    recurrence_rate: float
    determinism: float
    avg_diagonal_length: float
    max_diagonal_length: int
    divergence: float
    entropy: float
    laminarity: float
    trapping_time: float


class RealTimeSlidingtwRQA:
    """
    Streaming Time-Dependent Recurrence Quantification Analysis (twRQA)
    operating directly on PCA-projected trajectory coordinates.
    """

    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 25,
        theiler_w: int = 15,
        eps_ratio: float = 0.20,
        fixed_eps: float | None = None,
        min_diag_len: int = 2,
        min_vert_len: int = 2,
    ):
        """
        :param window_size: Number of reconstructed phase vectors per analysis frame (L).
        :param step_size: Stride delta between sliding evaluation windows.
        :param theiler_w: Theiler exclusion window |i - j| > W to drop trivial recurrences.
        :param eps_ratio: Epsilon distance factor (fraction of phase-space standard deviation).
        :param fixed_eps: Overrides dynamic epsilon with static Euclidean distance.
        :param min_diag_len: Minimum line length l_min for determinism (DET).
        :param min_vert_len: Minimum line length v_min for laminarity (LAM).
        """
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.theiler_w = int(theiler_w)
        self.eps_ratio = float(eps_ratio)
        self.fixed_eps = fixed_eps
        self.l_min = int(min_diag_len)
        self.v_min = int(min_vert_len)

    def _extract_line_distributions(self, R: np.ndarray) -> tuple[list[int], list[int]]:
        """
        Vectorized extraction of diagonal and vertical consecutive recurrence lengths.
        Enforces upper triangle symmetry and strictly skips zero bands.
        """
        L = R.shape[0]
        diag_lengths = []
        vert_lengths = []

        # 1. Diagonal line segments (Upper triangular search outside Theiler band)
        for offset in range(self.theiler_w + 1, L):
            diag = np.diagonal(R, offset=offset)
            if not np.any(diag):
                continue
            # Pad with 0 at both boundaries to identify structural edges
            padded = np.pad(diag.astype(np.int8), (1, 1), mode="constant")
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            diag_lengths.extend(lengths.tolist())

        # 2. Vertical line segments (Trapping / Laminar states)
        # Apply Theiler masking directly onto full matrix before parsing columns
        R_masked = R.copy()
        i_idx, j_idx = np.indices((L, L))
        R_masked[np.abs(i_idx - j_idx) <= self.theiler_w] = False

        for col in range(L):
            column_vec = R_masked[:, col]
            if not np.any(column_vec):
                continue
            padded_col = np.pad(column_vec.astype(np.int8), (1, 1), mode="constant")
            diffs_c = np.diff(padded_col)
            starts_c = np.where(diffs_c == 1)[0]
            ends_c = np.where(diffs_c == -1)[0]
            col_lens = ends_c - starts_c
            vert_lengths.extend(col_lens.tolist())

        return diag_lengths, vert_lengths

    def compute_window_rqa(self, window_pts: np.ndarray, timestamp_idx: int) -> RQAMetrics:
        """
        Computes formal non-linear quantification metrics for a single phase window.
        """
        L = window_pts.shape[0]
        
        # Determine neighborhood threshold epsilon
        if self.fixed_eps is not None:
            eps = self.fixed_eps
        else:
            std_dev = np.mean(np.std(window_pts, axis=0))
            eps = self.eps_ratio * std_dev if std_dev > 1e-12 else 0.1

        # Distance matrix generation
        D = cdist(window_pts, window_pts, metric="euclidean")
        R = D <= eps

        # Number of candidate points excluding the Theiler diagonal band
        total_eval_points = (L - self.theiler_w) * (L - self.theiler_w - 1) // 2
        if total_eval_points <= 0:
            return RQAMetrics(timestamp_idx, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

        # Upper triangle mask excluding Theiler window
        upper_mask = np.triu(np.ones((L, L), dtype=bool), k=self.theiler_w + 1)
        recurrence_count = np.sum(R & upper_mask)
        rr = recurrence_count / total_eval_points if total_eval_points > 0 else 0.0

        diag_lens, vert_lens = self._extract_line_distributions(R)

        # Process Diagonal Metrics (DET, L_avg, L_max, DIV, ENTR)
        diag_arr = np.array(diag_lens, dtype=np.int32) if diag_lens else np.empty(0, dtype=np.int32)
        valid_diags = diag_arr[diag_arr >= self.l_min]

        if valid_diags.size > 0 and recurrence_count > 0:
            det = np.sum(valid_diags) / recurrence_count
            l_avg = float(np.mean(valid_diags))
            l_max = int(np.max(valid_diags))
            div = 1.0 / l_max
            
            # Shannon entropy of diagonal lines
            _, counts = np.unique(valid_diags, return_counts=True)
            p_l = counts / counts.sum()
            entr = -np.sum(p_l * np.log(p_l))
        else:
            det, l_avg, l_max, div, entr = 0.0, 0.0, 0, 0.0, 0.0

        # Process Vertical / Laminar Metrics (LAM, TT)
        vert_arr = np.array(vert_lens, dtype=np.int32) if vert_lens else np.empty(0, dtype=np.int32)
        valid_verts = vert_arr[vert_arr >= self.v_min]

        if valid_verts.size > 0 and vert_arr.size > 0:
            lam = np.sum(valid_verts) / np.sum(vert_arr)
            tt = float(np.mean(valid_verts))
        else:
            lam, tt = 0.0, 0.0

        return RQAMetrics(
            timestamp_idx=timestamp_idx,
            recurrence_rate=float(rr),
            determinism=float(det),
            avg_diagonal_length=l_avg,
            max_diagonal_length=l_max,
            divergence=float(div),
            entropy=float(entr),
            laminarity=float(lam),
            trapping_time=tt,
        )

    def process_pca_stream(self, pca_coordinates: np.ndarray) -> list[RQAMetrics]:
        """
        Executes sliding window twRQA over a full sequence of PCA coordinates (N_eff, k_modes).
        """
        n_samples = pca_coordinates.shape[0]
        results = []

        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window_slice = pca_coordinates[start_idx:end_idx]
            
            # Anchor output timestamp to window leading edge
            metrics = self.compute_window_rqa(window_slice, timestamp_idx=end_idx)
            results.append(metrics)

        return results


def generate_bifurcating_telemetry(n_samples=4000, split_point=2000):
    """
    Simulates a regime change:
      Phase 1 (0 to split): Quasi-periodic multi-channel limit cycle (high DET, high L_avg).
      Phase 2 (split to end): Chaotic Lorenz trajectory (lower DET, higher DIV).
    """
    telemetry = np.zeros((n_samples, 3), dtype=np.float64)
    t = np.linspace(0, 40 * np.pi, split_point)
    
    # 1. Torus / limit cycle
    telemetry[:split_point, 0] = np.sin(t) + 0.1 * np.cos(5 * t)
    telemetry[:split_point, 1] = np.cos(t) + 0.1 * np.sin(5 * t)
    telemetry[:split_point, 2] = np.sin(2 * t)

    # 2. Lorenz chaos
    state = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    for i in range(split_point, n_samples):
        telemetry[i] = state
        dx = 10.0 * (state[1] - state[0])
        dy = state[0] * (28.0 - state[2]) - state[1]
        dz = state[0] * state[1] - (8.0 / 3.0) * state[2]
        state = state + np.array([dx, dy, dz]) * dt

    return telemetry


def demo_twrqa_transition():
    """Limit-cycle -> Lorenz regime switch; prints the twRQA profile.

    Converted verbatim from the source's ``if __name__ == "__main__"``
    test block so the module can also run its tipping-point demo.
    """
    # 1. Ingest telemetry and project via PCA delay embedding
    raw_telemetry = generate_bifurcating_telemetry(n_samples=3000, split_point=1500)

    # Construct zero-copy strided embedding (tau=8, m=4)
    tau, m = 8, 4
    n_eff = raw_telemetry.shape[0] - (m - 1) * tau
    # NOTE (extraction fix): the source used raw_telemetry.strides[0]
    # (= 3*8 bytes for a 3-channel array), but the strided view is built
    # on a 1-D contiguous column, whose element stride is 8 bytes.
    # as_strided does no bounds checking, so the source form read
    # out-of-bounds memory (garbage values, SVD non-convergence).
    itemsize = raw_telemetry.dtype.itemsize

    embedded_channels = [
        np.lib.stride_tricks.as_strided(
            np.ascontiguousarray(raw_telemetry[:, c]),
            shape=(n_eff, m),
            strides=(itemsize, tau * itemsize)
        )
        for c in range(raw_telemetry.shape[1])
    ]
    multivariate_embedded = np.concatenate(embedded_channels, axis=1)

    # SVD projection to top 3 principal modes
    X_centered = multivariate_embedded - np.mean(multivariate_embedded, axis=0)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_phase_space = X_centered @ vt[:3].T

    # 2. Execute sliding-window twRQA
    twrqa_engine = RealTimeSlidingtwRQA(
        window_size=350,
        step_size=50,
        theiler_w=16,
        eps_ratio=0.25,
        min_diag_len=2,
        min_vert_len=2
    )

    rqa_profile = twrqa_engine.process_pca_stream(pca_phase_space)

    # 3. Output temporal dynamics log
    print(f"{'Lead Index':<12} | {'RR (%)':<8} | {'DET (%)':<8} | {'L_max':<7} | {'DIV':<7} | {'ENTR':<7} | {'LAM (%)'}")
    print("-" * 72)
    for m in rqa_profile:
        print(
            f"{m.timestamp_idx:<12} | "
            f"{m.recurrence_rate * 100:>6.2f}% | "
            f"{m.determinism * 100:>6.2f}% | "
            f"{m.max_diagonal_length:>5} | "
            f"{m.divergence:>6.3f} | "
            f"{m.entropy:>6.2f} | "
            f"{m.laminarity * 100:>6.2f}%"
        )


class AlertLevel(Enum):
    NOMINAL = "NOMINAL"
    WARNING = "EWS_WARNING"         # Early Warning Signal: Critical slowing down
    CRITICAL = "BIFURCATION_TRIP"   # Imminent or actively occurring topological transition


@dataclass
class BifurcationTelemetryAlert:
    timestamp_idx: int
    alert_level: AlertLevel
    composite_anomaly_index: float
    z_score_divergence: float
    z_score_entropy: float
    divergence_val: float
    entropy_val: float
    determinism_val: float
    details: str


class WelfordRunningStats:
    """Tracks running mean and sample standard deviation with O(1) updates."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.window = deque()
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> tuple[float, float]:
        self.window.append(x)
        if len(self.window) > self.capacity:
            # Drop oldest from Welford state
            old = self.window.popleft()
            self.count -= 1
            old_mean = self.mean
            if self.count > 0:
                self.mean = (self.mean * (self.count + 1) - old) / self.count
                self.M2 -= (old - old_mean) * (old - self.mean)
            else:
                self.mean = 0.0
                self.M2 = 0.0

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

        variance = self.M2 / (self.count - 1) if self.count > 1 else 1e-12
        return self.mean, np.sqrt(max(variance, 1e-12))


class RealTimeBifurcationDetector:
    """
    Automated non-linear tipping point and bifurcation alert processor.
    Monitors streaming twRQA telemetry metrics (Divergence, Entropy, Determinism)
    against an adaptive baseline to detect dynamical phase transitions.
    """

    def __init__(
        self,
        baseline_buffer_len: int = 40,
        z_div_threshold: float = 2.5,
        z_entr_threshold: float = 2.0,
        cai_warning_thresh: float = 2.2,
        cai_critical_thresh: float = 4.0,
        consecutive_triggers: int = 2,
    ):
        """
        :param baseline_buffer_len: Number of twRQA windows used to calibrate nominal state.
        :param z_div_threshold: Z-score floor for anomalous trajectory divergence surges.
        :param z_entr_threshold: Z-score floor for Shannon entropy shifts.
        :param cai_warning_thresh: Composite Anomaly Index threshold for Early Warning.
        :param cai_critical_thresh: Composite Anomaly Index threshold for Critical Trip.
        :param consecutive_triggers: Sustained violations required to elevate alert level.
        """
        self.baseline_len = baseline_buffer_len
        self.z_div_thresh = z_div_threshold
        self.z_entr_thresh = z_entr_threshold
        self.cai_warn = cai_warning_thresh
        self.cai_crit = cai_critical_thresh
        self.consecutive_req = consecutive_triggers

        # Statistical trackers
        self.div_stats = WelfordRunningStats(capacity=self.baseline_len)
        self.entr_stats = WelfordRunningStats(capacity=self.baseline_len)
        self.det_stats = WelfordRunningStats(capacity=self.baseline_len)

        # Alarm debouncing registers
        self._warn_counter = 0
        self._crit_counter = 0
        self.current_state = AlertLevel.NOMINAL

    def evaluate_metric_step(self, rqa_metric) -> BifurcationTelemetryAlert:
        """
        Evaluates a single twRQA output frame and returns an alert state.
        rqa_metric is expected to have attributes: 
        [timestamp_idx, divergence, entropy, determinism, recurrence_rate, max_diagonal_length]
        """
        ts = rqa_metric.timestamp_idx
        div = rqa_metric.divergence
        entr = rqa_metric.entropy
        det = rqa_metric.determinism

        # Update statistical moving profiles
        mu_div, std_div = self.div_stats.update(div)
        mu_entr, std_entr = self.entr_stats.update(entr)
        mu_det, std_det = self.det_stats.update(det)

        # Baseline warm-up check
        if self.div_stats.count < (self.baseline_len // 2):
            return BifurcationTelemetryAlert(
                timestamp_idx=ts,
                alert_level=AlertLevel.NOMINAL,
                composite_anomaly_index=0.0,
                z_score_divergence=0.0,
                z_score_entropy=0.0,
                divergence_val=div,
                entropy_val=entr,
                determinism_val=det,
                details="Calibrating nominal trajectory baseline",
            )

        # Compute directional Z-Scores
        # Divergence surge (1/L_max increases as local Lyapunov horizon collapses)
        z_div = (div - mu_div) / std_div
        # Entropy instability (both collapse to periodic or surge to disorder represent bifurcations)
        z_entr = np.abs(entr - mu_entr) / std_entr
        # Determinism degradation
        z_det = (mu_det - det) / std_det

        # Composite Anomaly Index (CAI): Weighted metric combination
        # Surging divergence and determinism decay carry higher weight
        cai = (0.50 * max(0.0, z_div)) + (0.30 * z_entr) + (0.20 * max(0.0, z_det))

        # Dynamic Condition Evaluation
        is_crit = (cai >= self.cai_crit) or (z_div >= (self.z_div_thresh * 1.5))
        is_warn = (cai >= self.cai_warn) or (z_div >= self.z_div_thresh) or (z_entr >= self.z_entr_thresh)

        if is_crit:
            self._crit_counter += 1
            self._warn_counter += 1
        elif is_warn:
            self._warn_counter += 1
            self._crit_counter = max(0, self._crit_counter - 1)
        else:
            self._warn_counter = max(0, self._warn_counter - 1)
            self._crit_counter = max(0, self._crit_counter - 1)

        # Update Alert FSM
        if self._crit_counter >= self.consecutive_req:
            self.current_state = AlertLevel.CRITICAL
            detail_msg = (
                f"Bifurcation Trip: CAI={cai:.2f} (Z_DIV={z_div:.2f}, Z_ENTR={z_entr:.2f}, "
                f"L_max={rqa_metric.max_diagonal_length})"
            )
        elif self._warn_counter >= self.consecutive_req:
            self.current_state = AlertLevel.WARNING
            detail_msg = (
                f"Critical Slowing Down Detected: CAI={cai:.2f} (Z_DIV={z_div:.2f}, Z_ENTR={z_entr:.2f})"
            )
        else:
            self.current_state = AlertLevel.NOMINAL
            detail_msg = f"Nominal: CAI={cai:.2f}"

        return BifurcationTelemetryAlert(
            timestamp_idx=ts,
            alert_level=self.current_state,
            composite_anomaly_index=float(cai),
            z_score_divergence=float(z_div),
            z_score_entropy=float(z_entr),
            divergence_val=float(div),
            entropy_val=float(entr),
            determinism_val=float(det),
            details=detail_msg,
        )



def demo_bifurcation_tipping():
    """Three-phase tipping simulation; prints frame-by-frame alerts.

    Converted verbatim from the source's ``if __name__ == "__main__"``
    test block. Note: uses unseeded ``np.random`` (as in the source),
    so the transition-zone output varies run to run.
    """
    # 1. Synthesize streaming telemetry with an explicit tipping transition
    total_samples = 3200
    pre_bifurcation = 1600
    transition_zone = 2000

    telemetry = np.zeros((total_samples, 3), dtype=np.float64)
    time_steps = np.arange(total_samples)

    # Phase 1: Robust Limit Cycle
    t1 = np.linspace(0, 30 * np.pi, pre_bifurcation)
    telemetry[:pre_bifurcation, 0] = np.sin(t1)
    telemetry[:pre_bifurcation, 1] = np.cos(t1)
    telemetry[:pre_bifurcation, 2] = np.sin(2 * t1)

    # Phase 2: Decaying Stability / Critical Slowing Down (Frequency jitter + noise amplification)
    t2 = np.linspace(30 * np.pi, 40 * np.pi, transition_zone - pre_bifurcation)
    jitter = np.cumsum(np.random.normal(0, 0.05, len(t2)))
    telemetry[pre_bifurcation:transition_zone, 0] = np.sin(t2 + jitter) + np.random.normal(0, 0.15, len(t2))
    telemetry[pre_bifurcation:transition_zone, 1] = np.cos(t2 + jitter) + np.random.normal(0, 0.15, len(t2))
    telemetry[pre_bifurcation:transition_zone, 2] = np.sin(2 * (t2 + jitter))

    # Phase 3: Post-Bifurcation Rössler Chaotic Flow
    state = np.array([0.1, 0.0, 0.0])
    dt = 0.02
    a, b, c = 0.2, 0.2, 5.7
    for i in range(transition_zone, total_samples):
        telemetry[i] = state
        dx = -state[1] - state[2]
        dy = state[0] + a * state[1]
        dz = b + state[2] * (state[0] - c)
        state = state + np.array([dx, dy, dz]) * dt

    # 2. Phase-space reconstruction & PCA mode projection
    tau, m = 6, 4
    n_eff = total_samples - (m - 1) * tau
    # NOTE (extraction fix): as above -- the source used
    # telemetry.strides[0] (= 3*8 bytes); the 1-D column stride is 8.
    itemsize = telemetry.dtype.itemsize

    embedded = np.concatenate([
        np.lib.stride_tricks.as_strided(
            np.ascontiguousarray(telemetry[:, col]),
            shape=(n_eff, m),
            strides=(itemsize, tau * itemsize)
        )
        for col in range(telemetry.shape[1])
    ], axis=1)

    # Project to top 3 principal modes
    centered = embedded - np.mean(embedded, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected_phase_space = centered @ vt[:3].T

    # 3. Instantiate twRQA engine and Bifurcation Detector
    # Note: Assumes RealTimeSlidingtwRQA class from prior module is in scope
    twrqa_engine = RealTimeSlidingtwRQA(
        window_size=300,
        step_size=25,
        theiler_w=12,
        eps_ratio=0.25,
        min_diag_len=2,
    )

    detector = RealTimeBifurcationDetector(
        baseline_buffer_len=30,
        z_div_threshold=2.2,
        z_entr_threshold=2.0,
        cai_warning_thresh=2.0,
        cai_critical_thresh=3.8,
        consecutive_triggers=2,
    )

    rqa_stream = twrqa_engine.process_pca_stream(projected_phase_space)

    print(f"{'Frame TS':<10} | {'Status':<18} | {'CAI':<6} | {'Z_DIV':<7} | {'Z_ENTR':<7} | {'DET (%)':<8} | {'Details'}")
    print("=" * 95)

    for frame in rqa_stream:
        alert = detector.evaluate_metric_step(frame)
        status_label = alert.alert_level.value
        print(
            f"{alert.timestamp_idx:<10} | "
            f"{status_label:<18} | "
            f"{alert.composite_anomaly_index:>5.2f} | "
            f"{alert.z_score_divergence:>6.2f} | "
            f"{alert.z_score_entropy:>6.2f} | "
            f"{alert.determinism_val * 100:>6.1f}% | "
            f"{alert.details}"
        )


if __name__ == "__main__":
    demo_bifurcation_tipping()
