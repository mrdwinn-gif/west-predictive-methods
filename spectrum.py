"""Full Lyapunov spectrum, Kaplan-Yorke dimension, Kolmogorov-Sinai entropy.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export).

================================================================================
LORENZ-63 DYNAMICAL SPECTRUM & INVARIANT METRICS
================================================================================
Theoretical Target Benchmarks (sigma = 10, rho = 28, beta = 8/3):
  * Full Spectrum       : lambda_1 ~ +0.9056, lambda_2 = 0.0000, lambda_3 ~ -14.5723 nats/sec
  * Spectrum Sum (Div)  : sum lambda_i = -(sigma + 1 + beta) = -41/3 ~ -13.6667 nats/sec
  * Kolmogorov-Sinai H  : h_KS = sum max(0, lambda_i) = lambda_1 ~ 0.9056 nats/sec
  * Kaplan-Yorke Dim    : D_KY = 2 + (lambda_1 + lambda_2) / |lambda_3| ~ 2.0622
================================================================================

Method (Shimada-Nagashima / Benettin): integrate the 3D Lorenz-63
trajectory alongside its linearised variational equations (tangent
flow ``dPhi/dt = J(x) * Phi``), periodically re-orthonormalising the
tangent vectors via QR decomposition. ``lambda_i = (1/T) * sum_k
ln R_ii^(k)``. By Liouville's formula the spectrum sums to
``Tr(J) = -sigma - 1 - beta``. Under Pesin's theorem, ``h_KS`` is the
sum of the positive exponents.

One repair vs. the extracted text: in the source the block computing
the exponents from the accumulated ``lyap_sums`` was indented as if
inside the integration loop; it is placed after the loop here, which
is the only reading under which the code runs and matches the
documented ``lambda_i = sum(ln R_ii) / T`` formula. The truncated
Kolmogorov-Sinai print line in the source is likewise completed in
the obvious form.
"""

import numpy as np


class LorenzSpectrumAnalyzer:
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.trace_J = -(sigma + 1.0 + beta)

    def _derivatives(self, state_12d: np.ndarray) -> np.ndarray:
        """
        Computes derivatives for:
        - 3D state vector x: dx/dt = f(x)
        - 3x3 tangent matrix Phi: dPhi/dt = J(x) * Phi
        """
        # Unpack state
        x, y, z = state_12d[0], state_12d[1], state_12d[2]
        Phi = state_12d[3:].reshape((3, 3))

        # Nonlinear flow
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        f_vec = np.array([dx, dy, dz])

        # Jacobian matrix J(x)
        J = np.array([
            [-self.sigma, self.sigma, 0.0],
            [self.rho - z, -1.0, -x],
            [y, x, -self.beta]
        ])

        # Tangent flow: dPhi = J * Phi
        dPhi = J @ Phi

        return np.concatenate([f_vec, dPhi.flatten()])

    def _rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Standard 4th-Order Runge-Kutta step."""
        k1 = self._derivatives(state)
        k2 = self._derivatives(state + 0.5 * dt * k1)
        k3 = self._derivatives(state + 0.5 * dt * k2)
        k4 = self._derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def compute_spectrum(self, t_total: float = 500.0, dt: float = 0.01,
                         renorm_interval: int = 1, transient_time: float = 30.0) -> dict:
        """
        Integrates non-linear state and tangent equations with periodic QR decomposition.

        Args:
            t_total: Total integration horizon post-transient (seconds).
            dt: Integration step size (seconds).
            renorm_interval: Steps between successive QR decompositions.
            transient_time: Horizon to discard to settle onto the attractor.
        """
        # 1. Warm-up phase: discard orbital transients
        state_3d = np.array([1.0, 1.0, 1.0])
        n_transient = int(transient_time / dt)

        for _ in range(n_transient):
            # Integrate 3D state alone during warmup
            k1 = np.array([self.sigma * (state_3d[1] - state_3d[0]),
                           state_3d[0] * (self.rho - state_3d[2]) - state_3d[1],
                           state_3d[0] * state_3d[1] - self.beta * state_3d[2]])
            s2 = state_3d + 0.5 * dt * k1
            k2 = np.array([self.sigma * (s2[1] - s2[0]),
                           s2[0] * (self.rho - s2[2]) - s2[1],
                           s2[0] * s2[1] - self.beta * s2[2]])
            s3 = state_3d + 0.5 * dt * k2
            k3 = np.array([self.sigma * (s3[1] - s3[0]),
                           s3[0] * (self.rho - s3[2]) - s3[1],
                           s3[0] * s3[1] - self.beta * s3[2]])
            s4 = state_3d + dt * k3
            k4 = np.array([self.sigma * (s4[1] - s4[0]),
                           s4[0] * (self.rho - s4[2]) - s4[1],
                           s4[0] * s4[1] - self.beta * s4[2]])
            state_3d += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 2. Main execution phase: coupled 12D system
        Phi_init = np.eye(3)
        state_12d = np.concatenate([state_3d, Phi_init.flatten()])

        total_steps = int(t_total / dt)
        lyap_sums = np.zeros(3)
        n_renorms = 0

        for step in range(1, total_steps + 1):
            state_12d = self._rk4_step(state_12d, dt)

            # Periodic QR Decomposition
            if step % renorm_interval == 0:
                Phi = state_12d[3:].reshape((3, 3))

                # QR Factorization: Phi = Q * R
                Q, R = np.linalg.qr(Phi)

                # Extract growth rates: R_ii must be positive by convention
                diag_R = np.abs(np.diag(R))
                lyap_sums += np.log(diag_R)
                n_renorms += 1

                # Re-normalize tangent frame: reset Phi to orthonormal Q
                state_12d[3:] = Q.flatten()

        # 3. Calculate Exponents: lambda_i = sum(ln R_ii) / T
        T = total_steps * dt
        spectrum = lyap_sums / T

        # Sort descending: lambda_1 >= lambda_2 >= lambda_3
        sorted_indices = np.argsort(spectrum)[::-1]
        spectrum = spectrum[sorted_indices]

        # 4. Compute Derived Metrics
        lambda_1, lambda_2, lambda_3 = spectrum[0], spectrum[1], spectrum[2]

        # Kolmogorov-Sinai Entropy (Pesin theorem)
        ks_entropy_nats = float(np.sum(spectrum[spectrum > 0]))
        ks_entropy_bits = ks_entropy_nats / np.log(2.0)

        # Kaplan-Yorke Dimension (D_KY)
        if lambda_1 + lambda_2 >= 0 and (lambda_1 + lambda_2 + lambda_3) < 0:
            d_ky = 2.0 + (lambda_1 + lambda_2) / np.abs(lambda_3)
        else:
            d_ky = np.nan

        return {
            "spectrum": spectrum,
            "sum_spectrum": float(np.sum(spectrum)),
            "theoretical_trace": self.trace_J,
            "discrepancy": float(np.abs(np.sum(spectrum) - self.trace_J)),
            "ks_entropy_nats": ks_entropy_nats,
            "ks_entropy_bits": ks_entropy_bits,
            "kaplan_yorke_dim": float(d_ky)
        }


# ============================================================================
# EXECUTION HARNESS
# ============================================================================
if __name__ == "__main__":
    analyzer = LorenzSpectrumAnalyzer(sigma=10.0, rho=28.0, beta=8.0 / 3.0)

    print("[+] Integrating coupled variational dynamics (RK4, t=300s, dt=0.005s)...")
    results = analyzer.compute_spectrum(t_total=300.0, dt=0.005, renorm_interval=2)

    l1, l2, l3 = results["spectrum"]

    print("\n" + "=" * 70)
    print("LORENZ-63 COMPUTED INVARIANTS")
    print("=" * 70)
    print("Lyapunov Spectrum (lambda):")
    print(f" lambda_1 (Unstable / Expansion) : {l1:+.5f} nats/sec [Theoretical: +0.9056]")
    print(f" lambda_2 (Neutral / Flow Vector) : {l2:+.5f} nats/sec [Theoretical: 0.0000]")
    print(f" lambda_3 (Stable / Contraction) : {l3:+.5f} nats/sec [Theoretical: -14.5723]")
    print("-" * 70)
    print(f"Spectrum Sum (sum lambda_i)      : {results['sum_spectrum']:.5f} nats/sec")
    print(f"Theoretical Divergence (Tr)      : {results['theoretical_trace']:.5f} nats/sec (-41/3)")
    print(f"Liouville Identity Error         : {results['discrepancy']:.2e}")
    print("-" * 70)
    print(f"Kolmogorov-Sinai Entropy         : {results['ks_entropy_nats']:.5f} nats/sec ({results['ks_entropy_bits']:.5f} bits/sec)")
    print(f"Kaplan-Yorke Dimension (D_KY)    : {results['kaplan_yorke_dim']:.4f}")
    print("=" * 70)
