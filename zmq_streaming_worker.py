#!/usr/bin/env python3
# Async ZeroMQ streaming worker: telemetry in, twRQA bifurcation alerts out.
#
# Extracted verbatim (self-contained) from a Gemini chat export titled
# "Telemetry Phase Space Reconstruction Pipeline" (source:
# ~/workspace/user/files/Telemetry_Phase_Space_Reconstruction_Pipeline_1_kywj.docx).
# This file re-defines the twRQA / Welford detector classes inline so it
# runs standalone; the library versions live in twrqa_bifurcation.py.
#
# Wiring (localhost only -- no remote endpoints, no credentials):
#   telemetry PUB  tcp://127.0.0.1:5555  topic "TELEMETRY"   (msgpack)
#   alerts    PUB  tcp://127.0.0.1:5556  topic "BIFURCATION_ALERT"
# Running it starts a mock telemetry publisher, the worker, and an alert
# listener concurrently; stop with Ctrl-C. Requires pyzmq + msgpack:
#   pip install pyzmq msgpack
# NOT runtime-tested here (pyzmq/msgpack not installed in this
# environment); py_compile only. See README "Verification notes".

import asyncio
import json
import logging
import signal
import sys
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum

import msgpack
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import cdist
import zmq
import zmq.asyncio

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("twRQA_Worker")


# ============================================================================
# Telemetry Data Structures & State Definitions
# ============================================================================

class AlertLevel(str, Enum):
    NOMINAL = "NOMINAL"
    WARNING = "EWS_WARNING"         # Early Warning Signal: Critical slowing down
    CRITICAL = "BIFURCATION_TRIP"   # Bifurcation breach / chaotic disintegration


@dataclass(slots=True)
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


@dataclass(slots=True)
class BifurcationAlertPayload:
    timestamp_idx: int
    status: str
    composite_anomaly_index: float
    z_div: float
    z_entr: float
    divergence: float
    entropy: float
    determinism: float
    recurrence_rate: float
    details: str


# ============================================================================
# Non-Linear Numerical Processing Engines
# ============================================================================

class WelfordRunningStats:
    """Tracks running mean and standard deviation over an sliding window."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.window = deque()
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> tuple[float, float]:
        self.window.append(x)
        if len(self.window) > self.capacity:
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
    """Evaluates dynamical anomalies using composite statistical scaling."""
    def __init__(
        self,
        baseline_buffer_len: int = 40,
        z_div_threshold: float = 2.2,
        z_entr_threshold: float = 2.0,
        cai_warning_thresh: float = 2.0,
        cai_critical_thresh: float = 3.8,
        consecutive_triggers: int = 2,
    ):
        self.baseline_len = baseline_buffer_len
        self.z_div_thresh = z_div_threshold
        self.z_entr_thresh = z_entr_threshold
        self.cai_warn = cai_warning_thresh
        self.cai_crit = cai_critical_thresh
        self.consecutive_req = consecutive_triggers

        self.div_stats = WelfordRunningStats(capacity=self.baseline_len)
        self.entr_stats = WelfordRunningStats(capacity=self.baseline_len)
        self.det_stats = WelfordRunningStats(capacity=self.baseline_len)

        self._warn_counter = 0
        self._crit_counter = 0
        self.current_state = AlertLevel.NOMINAL

    def evaluate(self, rqa: RQAMetrics) -> BifurcationAlertPayload:
        ts = rqa.timestamp_idx
        div = rqa.divergence
        entr = rqa.entropy
        det = rqa.determinism

        mu_div, std_div = self.div_stats.update(div)
        mu_entr, std_entr = self.entr_stats.update(entr)
        mu_det, std_det = self.det_stats.update(det)

        # Baseline calibration period
        if self.div_stats.count < (self.baseline_len // 2):
            return BifurcationAlertPayload(
                timestamp_idx=ts,
                status=AlertLevel.NOMINAL.value,
                composite_anomaly_index=0.0,
                z_div=0.0,
                z_entr=0.0,
                divergence=div,
                entropy=entr,
                determinism=det,
                recurrence_rate=rqa.recurrence_rate,
                details="Calibrating nominal baseline profile",
            )

        z_div = (div - mu_div) / std_div
        z_entr = np.abs(entr - mu_entr) / std_entr
        z_det = (mu_det - det) / std_det

        cai = (0.50 * max(0.0, z_div)) + (0.30 * z_entr) + (0.20 * max(0.0, z_det))

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

        if self._crit_counter >= self.consecutive_req:
            self.current_state = AlertLevel.CRITICAL
            details = f"Bifurcation Trip: CAI={cai:.2f} (Z_DIV={z_div:.2f}, Z_ENTR={z_entr:.2f})"
        elif self._warn_counter >= self.consecutive_req:
            self.current_state = AlertLevel.WARNING
            details = f"Early Warning Signal: Critical slowing down (CAI={cai:.2f})"
        else:
            self.current_state = AlertLevel.NOMINAL
            details = f"Nominal operation (CAI={cai:.2f})"

        return BifurcationAlertPayload(
            timestamp_idx=ts,
            status=self.current_state.value,
            composite_anomaly_index=float(cai),
            z_div=float(z_div),
            z_entr=float(z_entr),
            divergence=float(div),
            entropy=float(entr),
            determinism=float(det),
            recurrence_rate=float(rqa.recurrence_rate),
            details=details,
        )


class FasttwRQA:
    """Vectorized sliding-window RQA calculation on multidimensional phase arrays."""
    def __init__(self, theiler_w: int = 12, eps_ratio: float = 0.25, l_min: int = 2, v_min: int = 2):
        self.theiler_w = theiler_w
        self.eps_ratio = eps_ratio
        self.l_min = l_min
        self.v_min = v_min

    def compute(self, window_pts: np.ndarray, timestamp_idx: int) -> RQAMetrics:
        L = window_pts.shape[0]
        std_dev = np.mean(np.std(window_pts, axis=0))
        eps = self.eps_ratio * std_dev if std_dev > 1e-12 else 0.1

        # Distance matrix
        D = cdist(window_pts, window_pts, metric="euclidean")
        R = D <= eps

        total_eval = (L - self.theiler_w) * (L - self.theiler_w - 1) // 2
        if total_eval <= 0:
            return RQAMetrics(timestamp_idx, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

        # Diagonal line extraction outside Theiler exclusion band
        diag_lengths = []
        for offset in range(self.theiler_w + 1, L):
            diag = np.diagonal(R, offset=offset)
            if not np.any(diag):
                continue
            padded = np.pad(diag.astype(np.int8), (1, 1), mode="constant")
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            diag_lengths.extend((ends - starts).tolist())

        upper_mask = np.triu(np.ones((L, L), dtype=bool), k=self.theiler_w + 1)
        recurrence_count = np.sum(R & upper_mask)
        rr = recurrence_count / total_eval

        diag_arr = np.array(diag_lengths, dtype=np.int32) if diag_lengths else np.empty(0, dtype=np.int32)
        valid_diags = diag_arr[diag_arr >= self.l_min]

        if valid_diags.size > 0 and recurrence_count > 0:
            det = np.sum(valid_diags) / recurrence_count
            l_avg = float(np.mean(valid_diags))
            l_max = int(np.max(valid_diags))
            div = 1.0 / l_max
            _, counts = np.unique(valid_diags, return_counts=True)
            p_l = counts / counts.sum()
            entr = float(-np.sum(p_l * np.log(p_l)))
        else:
            det, l_avg, l_max, div, entr = 0.0, 0.0, 0, 0.0, 0.0

        return RQAMetrics(
            timestamp_idx=timestamp_idx,
            recurrence_rate=float(rr),
            determinism=float(det),
            avg_diagonal_length=l_avg,
            max_diagonal_length=l_max,
            divergence=float(div),
            entropy=entr,
            laminarity=0.0,
            trapping_time=0.0,
        )


# ============================================================================
# Core Streaming Pipeline Engine
# ============================================================================

class AsyncZMQtwRQAWorker:
    """
    Asynchronous ZeroMQ worker pipeline:
    - Ingests multi-channel telemetry over SUB socket.
    - Maintains ring buffer and executes zero-copy Takens embedding + PCA.
    - Evaluates twRQA metrics and bifurcation alerts.
    - Publishes alerts over PUB socket.
    """

    def __init__(
        self,
        sub_endpoint: str = "tcp://127.0.0.1:5555",
        pub_endpoint: str = "tcp://127.0.0.1:5556",
        sub_topic: str = "TELEMETRY",
        pub_topic: str = "BIFURCATION_ALERT",
        channels: int = 3,
        window_size: int = 300,
        step_size: int = 25,
        tau: int = 6,
        m: int = 4,
        pca_modes: int = 3,
        theiler_w: int = 12,
        use_msgpack: bool = True,
    ):
        self.sub_endpoint = sub_endpoint
        self.pub_endpoint = pub_endpoint
        self.sub_topic = sub_topic.encode("utf-8")
        self.pub_topic = pub_topic.encode("utf-8")

        self.channels = channels
        self.window_size = window_size
        self.step_size = step_size
        self.tau = tau
        self.m = m
        self.pca_modes = pca_modes
        self.use_msgpack = use_msgpack

        # Calculate buffer requirement for sliding window over delay coordinates
        # Total raw samples needed for window_size embedded vectors = window_size + (m - 1) * tau
        self.required_raw_samples = self.window_size + (self.m - 1) * self.tau
        self.buffer_capacity = self.required_raw_samples * 2
        self.ring_buffer = np.zeros((self.buffer_capacity, self.channels), dtype=np.float64)
        
        self.total_samples_received = 0
        self.buffer_write_idx = 0
        self.samples_since_last_eval = 0

        # Submodules
        self.rqa_engine = FasttwRQA(theiler_w=theiler_w, eps_ratio=0.25, l_min=2)
        self.detector = RealTimeBifurcationDetector()

        # ZeroMQ Async Context
        self.ctx = zmq.asyncio.Context()
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self._running = False

    def _embed_and_project(self, raw_window: np.ndarray) -> np.ndarray:
        """Constructs zero-copy delay coordinates and projects onto top PCA modes."""
        n_raw = raw_window.shape[0]
        n_eff = n_raw - (self.m - 1) * self.tau
        # NOTE (extraction fix): the source used raw_window.strides[0]
        # (= channels*8 bytes for the (N, C) buffer), but the strided
        # view is built on a 1-D contiguous column (element stride 8).
        # as_strided does no bounds checking, so the source form read
        # out-of-bounds memory.
        itemsize = raw_window.dtype.itemsize

        embedded_channels = [
            as_strided(
                np.ascontiguousarray(raw_window[:, c]),
                shape=(n_eff, self.m),
                strides=(itemsize, self.tau * itemsize),
                writeable=False,
            )
            for c in range(self.channels)
        ]
        X = np.concatenate(embedded_channels, axis=1)

        # Center and project via SVD
        X_centered = X - np.mean(X, axis=0)
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ vt[: self.pca_modes].T

    async def start(self):
        """Initializes sockets and enters asynchronous ingestion/evaluation loop."""
        self.sub_socket.connect(self.sub_endpoint)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, self.sub_topic)
        self.pub_socket.bind(self.pub_endpoint)
        self._running = True

        logger.info(f"Worker connected: SUB -> {self.sub_endpoint} [Topic: {self.sub_topic.decode()}]")
        logger.info(f"Worker bound:     PUB -> {self.pub_endpoint} [Topic: {self.pub_topic.decode()}]")
        logger.info(f"Configuration: L={self.window_size}, Stride={self.step_size}, tau={self.tau}, m={self.m}")

        try:
            while self._running:
                # Multi-frame receive: [topic, payload]
                frames = await self.sub_socket.recv_multipart()
                if len(frames) < 2:
                    continue

                raw_bytes = frames[1]
                data = (
                    msgpack.unpackb(raw_bytes, raw=False)
                    if self.use_msgpack
                    else json.loads(raw_bytes.decode("utf-8"))
                )

                # Expect payload format: {"ts": int, "values": [ch0, ch1, ...]}
                values = np.asarray(data["values"], dtype=np.float64)
                if values.shape[0] != self.channels:
                    logger.warning(f"Channel dimension mismatch: expected {self.channels}, got {values.shape[0]}")
                    continue

                # Ingest to ring buffer
                self.ring_buffer[self.buffer_write_idx] = values
                self.buffer_write_idx = (self.buffer_write_idx + 1) % self.buffer_capacity
                self.total_samples_received += 1
                self.samples_since_last_eval += 1

                # Evaluate twRQA when buffer holds enough history and step stride is met
                if (
                    self.total_samples_received >= self.required_raw_samples
                    and self.samples_since_last_eval >= self.step_size
                ):
                    self.samples_since_last_eval = 0
                    await self._evaluate_and_publish()

        except asyncio.CancelledError:
            logger.info("Worker execution task cancelled.")
        finally:
            await self.shutdown()

    async def _evaluate_and_publish(self):
        """Extracts current window, runs phase-space twRQA, and publishes alerts."""
        # Unroll linear slice from circular ring buffer
        end_idx = self.buffer_write_idx
        start_idx = end_idx - self.required_raw_samples
        if start_idx >= 0:
            window_data = self.ring_buffer[start_idx:end_idx]
        else:
            window_data = np.vstack((
                self.ring_buffer[start_idx % self.buffer_capacity:],
                self.ring_buffer[:end_idx],
            ))

        # 1. Project onto PCA phase space
        projected_pts = self._embed_and_project(window_data)

        # 2. Compute non-linear recurrence metrics
        rqa = self.rqa_engine.compute(projected_pts, timestamp_idx=self.total_samples_received)

        # 3. Detect early warnings / bifurcation transitions
        alert = self.detector.evaluate(rqa)

        # 4. Serialize and publish
        payload_dict = asdict(alert)
        packed_payload = (
            msgpack.packb(payload_dict)
            if self.use_msgpack
            else json.dumps(payload_dict).encode("utf-8")
        )

        await self.pub_socket.send_multipart([self.pub_topic, packed_payload])

        if alert.status != AlertLevel.NOMINAL.value:
            logger.warning(f"ALERT [{alert.status}] @ {alert.timestamp_idx} | {alert.details}")
        else:
            logger.debug(f"NOMINAL @ {alert.timestamp_idx} | CAI={alert.composite_anomaly_index:.2f}")

    async def shutdown(self):
        """Gracefully closes ZeroMQ sockets and destroys execution context."""
        logger.info("Tearing down ZMQ worker sockets...")
        self._running = False
        self.sub_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.ctx.term()
        logger.info("ZMQ worker context terminated successfully.")


# ============================================================================
# Telemetry Stream Mock Generator & Runner
# ============================================================================

async def run_telemetry_mock_publisher(endpoint: str = "tcp://127.0.0.1:5555", topic: str = "TELEMETRY"):
    """Simulates a multi-channel telemetry feed transitioning from stable to chaotic."""
    ctx = zmq.asyncio.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)
    topic_bytes = topic.encode("utf-8")

    logger.info(f"Mock telemetry publisher bound to {endpoint}")
    await asyncio.sleep(1.0)  # Allow SUB connection handshake

    try:
        t = 0.0
        sample_idx = 0
        state = np.array([1.0, 1.0, 1.0])
        dt = 0.01

        while True:
            # Phase 1: Torus / Quasi-periodic Limit Cycle
            if sample_idx < 1500:
                vals = [
                    np.sin(t) + 0.05 * np.sin(5 * t),
                    np.cos(t) + 0.05 * np.cos(5 * t),
                    np.sin(2 * t),
                ]
                t += 0.04
            # Phase 2: Decaying Stability / Noise & Dynamic Jitter
            elif sample_idx < 2200:
                jitter = np.random.normal(0, 0.08)
                vals = [
                    np.sin(t + jitter) + np.random.normal(0, 0.12),
                    np.cos(t + jitter) + np.random.normal(0, 0.12),
                    np.sin(2 * (t + jitter)),
                ]
                t += 0.04
            # Phase 3: Post-Bifurcation Rössler Chaos
            else:
                dx = -state[1] - state[2]
                dy = state[0] + 0.2 * state[1]
                dz = 0.2 + state[2] * (state[0] - 5.7)
                state = state + np.array([dx, dy, dz]) * dt
                vals = state.tolist()

            packet = msgpack.packb({"ts": sample_idx, "values": vals})
            await pub.send_multipart([topic_bytes, packet])
            sample_idx += 1

            # 200 Hz simulated sensor rate (5 ms interval)
            await asyncio.sleep(0.005)

    except asyncio.CancelledError:
        pass
    finally:
        pub.close(linger=0)
        ctx.term()


async def run_alert_listener(endpoint: str = "tcp://127.0.0.1:5556", topic: str = "BIFURCATION_ALERT"):
    """Downstream consumer printing published alerts."""
    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

    logger.info(f"Alert subscriber monitoring {endpoint} on topic [{topic}]")
    try:
        while True:
            frames = await sub.recv_multipart()
            if len(frames) >= 2:
                payload = msgpack.unpackb(frames[1], raw=False)
                status = payload["status"]
                ts = payload["timestamp_idx"]
                cai = payload["composite_anomaly_index"]
                z_div = payload["z_div"]
                det = payload["determinism"] * 100.0

                if status != "NOMINAL":
                    print(f"\033[91m>>> [{status}] TS:{ts:05d} | CAI={cai:4.2f} | Z_DIV={z_div:4.2f} | DET={det:4.1f}% | {payload['details']}\033[0m")
                else:
                    print(f"    [{status}] TS:{ts:05d} | CAI={cai:4.2f} | Z_DIV={z_div:4.2f} | DET={det:4.1f}%")
    except asyncio.CancelledError:
        pass
    finally:
        sub.close(linger=0)
        ctx.term()


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    worker = AsyncZMQtwRQAWorker(
        sub_endpoint="tcp://127.0.0.1:5555",
        pub_endpoint="tcp://127.0.0.1:5556",
        channels=3,
        window_size=250,
        step_size=20,
        tau=5,
        m=4,
        pca_modes=3,
        theiler_w=10,
        use_msgpack=True,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal caught.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows fallback

    # Run publisher, worker, and listener concurrently
    tasks = [
        asyncio.create_task(run_telemetry_mock_publisher()),
        asyncio.create_task(worker.start()),
        asyncio.create_task(run_alert_listener()),
    ]

    await stop_event.wait()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
