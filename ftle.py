"""Finite-Time Lyapunov Exponents (FTLE) + Hessian ridge extraction.

Extracted from a Gemini methods export ("Incomplete Transmission
Evaluation", partial export).

INCOMPLETE SOURCE NOTE: the source export ends abruptly mid-table in
the Hessian ridge section (the "Diagnostic Thresholding Parameters"
table is cut off after the ``ftle_min_percentile`` row). Everything
that exists in the source is included below -- the Double Gyre field,
the FTLE engine, the full ``FTLERidgeExtractor`` class and its
synthetic-ridge verification harness. Only the tail of that prose
parameter table is missing; no code is missing on that account, and
nothing has been invented to fill the gap.

Theory (from the source): the FTLE measures the locally maximal rate
of separation between adjacent particles over a finite horizon ``T``.
Ridges in the FTLE field are Lagrangian Coherent Structures (LCS):
forward FTLE (``T > 0``) gives repelling LCS, backward FTLE
(``T < 0``) gives attracting LCS. The FTLE is
``sigma = (1 / 2|T|) * ln lambda_max(C)`` where ``C`` is the right
Cauchy-Green deformation tensor built from the flow-map Jacobian.

Ridge extraction follows the Eberly-Steger height-ridge definition:
a point is on a ridge when the transverse directional derivative
``<grad sigma, n1> = 0`` and the transverse curvature ``mu1 < 0``,
where ``n1`` is the eigenvector of the Hessian ``H = nabla^2 sigma``
for its most negative eigenvalue ``mu1``. Subpixel ridge positions
come from Steger's second-order Taylor expansion along ``n1``.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class DoubleGyreVectorField:
    """Analytical time-dependent Double Gyre model (Shadden-Lekien-Marsden 2005)."""

    def __init__(self, A: float = 0.1, epsilon: float = 0.25, omega: float = 2.0 * np.pi / 10.0):
        self.A = A
        self.epsilon = epsilon
        self.omega = omega

    def velocity(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        a = self.epsilon * np.sin(self.omega * t)
        b = 1.0 - 2.0 * self.epsilon * np.sin(self.omega * t)

        f = a * x ** 2 + b * x
        df_dx = 2.0 * a * x + b

        u = -np.pi * self.A * np.sin(np.pi * f) * np.cos(np.pi * y)
        v = np.pi * self.A * np.cos(np.pi * f) * np.sin(np.pi * y) * df_dx
        return u, v


class FTLEEngine:
    def __init__(self, field, x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0), nx=200, ny=100):
        self.field = field
        self.nx = nx
        self.ny = ny
        self.x_line = np.linspace(x_bounds[0], x_bounds[1], nx)
        self.y_line = np.linspace(y_bounds[0], y_bounds[1], ny)
        self.dx = self.x_line[1] - self.x_line[0]
        self.dy = self.y_line[1] - self.y_line[0]
        self.X0, self.Y0 = np.meshgrid(self.x_line, self.y_line)

    def _rk4_step(self, x: np.ndarray, y: np.ndarray, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized Runge-Kutta 4th-order particle advection."""
        u1, v1 = self.field.velocity(x, y, t)
        u2, v2 = self.field.velocity(x + 0.5 * dt * u1, y + 0.5 * dt * v1, t + 0.5 * dt)
        u3, v3 = self.field.velocity(x + 0.5 * dt * u2, y + 0.5 * dt * v2, t + 0.5 * dt)
        u4, v4 = self.field.velocity(x + dt * u3, y + dt * v3, t + dt)

        x_next = x + (dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        y_next = y + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        return x_next, y_next

    def compute_flow_map(self, t0: float, T: float, dt_step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Advects the complete Cartesian grid of particles from t0 to t0 + T."""
        x = self.X0.copy()
        y = self.Y0.copy()

        n_steps = int(np.abs(T) / dt_step)
        dt = np.sign(T) * dt_step
        t = t0

        for _ in range(n_steps):
            x, y = self._rk4_step(x, y, t, dt)
            t += dt

        return x, y

    def compute_ftle(self, t0: float = 0.0, T: float = 15.0, dt_step: float = 0.05) -> np.ndarray:
        """
        Calculates the FTLE field sigma_t0^T.
        T > 0: Forward FTLE (Repelling ridges)
        T < 0: Backward FTLE (Attracting ridges)
        """
        # 1. Flow map execution
        phi_x, phi_y = self.compute_flow_map(t0=t0, T=T, dt_step=dt_step)

        # 2. Central difference gradient computation for deformation Jacobian
        # np.gradient returns [dF/dy (axis 0), dF/dx (axis 1)]
        dphi_x_dy, dphi_x_dx = np.gradient(phi_x, self.dy, self.dx)
        dphi_y_dy, dphi_y_dx = np.gradient(phi_y, self.dy, self.dx)

        # 3. Assemble components of Right Cauchy-Green deformation tensor C
        C11 = dphi_x_dx ** 2 + dphi_y_dx ** 2
        C22 = dphi_x_dy ** 2 + dphi_y_dy ** 2
        C12 = dphi_x_dx * dphi_x_dy + dphi_y_dx * dphi_y_dy

        # 4. Analytical maximum eigenvalue of 2x2 symmetric matrix
        trace_C = C11 + C22
        diff_C = C11 - C22
        discriminant = np.maximum(diff_C ** 2 + 4.0 * C12 ** 2, 0.0)
        lambda_max = 0.5 * (trace_C + np.sqrt(discriminant))

        # 5. Extract FTLE: sigma = (1 / 2|T|) * ln(lambda_max)
        # Numerical guard: lambda_max must be >= 1.0 (incompressible or stretching)
        lambda_max = np.maximum(lambda_max, 1.0)
        ftle = (1.0 / (2.0 * np.abs(T))) * np.log(lambda_max)

        return ftle


class FTLERidgeExtractor:
    def __init__(self, dx: float = 1.0, dy: float = 1.0, sigma_smooth: float = 1.0,
                 mu_threshold: float = -0.05, ftle_min_percentile: float = 65.0):
        """
        Args:
            dx: Grid spacing in x (meters or physical units).
            dy: Grid spacing in y.
            sigma_smooth: Gaussian filter scale to suppress numerical integration noise.
            mu_threshold: Maximum acceptable value for mu_1 (must be strictly negative).
            ftle_min_percentile: Percentile threshold to discard low-energy background ridges.
        """
        self.dx = dx
        self.dy = dy
        self.sigma_smooth = sigma_smooth
        self.mu_threshold = mu_threshold
        self.ftle_min_percentile = ftle_min_percentile

    def _compute_derivatives(self, scalar_field: np.ndarray) -> tuple[np.ndarray, ...]:
        """Calculates smoothed 1st and 2nd central difference spatial derivatives."""
        # 1. Smooth scalar field to prevent noise amplification in 2nd derivatives
        if self.sigma_smooth > 0:
            field = gaussian_filter(scalar_field, sigma=self.sigma_smooth, mode='reflect')
        else:
            field = scalar_field

        # 2. First spatial derivatives (axis 0 is y, axis 1 is x)
        grad_y, grad_x = np.gradient(field, self.dy, self.dx)

        # 3. Second spatial derivatives (Hessian elements)
        H_yy, H_xy = np.gradient(grad_y, self.dy, self.dx)
        _, H_xx = np.gradient(grad_x, self.dy, self.dx)

        return field, grad_x, grad_y, H_xx, H_yy, H_xy

    def extract_ridges(self, ftle_grid: np.ndarray, x_coords: np.ndarray = None,
                       y_coords: np.ndarray = None) -> dict:
        """
        Vectorized analytical extraction of 1D height ridges from 2D FTLE array.

        Returns:
            Dictionary containing:
                 - ridge_mask: Boolean 2D array of pixels intersected by a ridge
                 - subpixel_points: (N, 2) array of continuous (x, y) ridge points
                 - mu1_field: The minimum eigenvalue array
                 - n1_vector_field: Transverse unit eigenvector field of shape (ny, nx, 2)
        """
        ny, nx = ftle_grid.shape
        if x_coords is None:
            x_coords = np.arange(nx) * self.dx
        if y_coords is None:
            y_coords = np.arange(ny) * self.dy

        X, Y = np.meshgrid(x_coords, y_coords)

        # Compute gradient and Hessian arrays
        smooth_field, g_x, g_y, H_xx, H_yy, H_xy = self._compute_derivatives(ftle_grid)

        # 1. Analytical Eigendecomposition of 2x2 symmetric Hessian
        trace_H = H_xx + H_yy
        det_H = H_xx * H_yy - H_xy ** 2
        discriminant = np.maximum((H_xx - H_yy) ** 2 + 4.0 * H_xy ** 2, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        # mu1 is the most negative eigenvalue (steepest downward curvature)
        mu1 = 0.5 * (trace_H - sqrt_disc)
        mu2 = 0.5 * (trace_H + sqrt_disc)

        # 2. Solve for Transverse Unit Eigenvector n1 = [n1_x, n1_y]
        # Formulation handles cases where H_xy -> 0 stably
        v1_x = mu1 - H_yy
        v1_y = H_xy.copy()

        # Handle degenerate/axis-aligned regions
        zero_xy = np.abs(H_xy) < 1e-12
        v1_x[zero_xy] = np.where(H_xx[zero_xy] <= H_yy[zero_xy], 1.0, 0.0)
        v1_y[zero_xy] = np.where(H_xx[zero_xy] <= H_yy[zero_xy], 0.0, 1.0)

        # Normalize eigenvectors
        norm_v1 = np.sqrt(v1_x ** 2 + v1_y ** 2)
        norm_v1[norm_v1 < 1e-12] = 1.0
        n1_x = v1_x / norm_v1
        n1_y = v1_y / norm_v1

        # 3. Transverse Directional Derivative: <grad, n1>
        dir_deriv = g_x * n1_x + g_y * n1_y

        # 4. Steger Subpixel Displacement along n1: t = - <grad, n1> / (n1^T H n1) = - dir_deriv / mu1
        denom = np.where(np.abs(mu1) > 1e-9, mu1, -1e-9)
        t = -dir_deriv / denom

        # Offset vectors in physical units
        px = t * n1_x
        py = t * n1_y

        # 5. Logical Filtering Gates
        # (a) Subpixel offset must remain within current grid cell
        in_cell_x = np.abs(px) <= (0.5 * self.dx)
        in_cell_y = np.abs(py) <= (0.5 * self.dy)
        in_cell = in_cell_x & in_cell_y

        # (b) Concavity criterion: must be downward curvature (mu1 < 0)
        concave = mu1 < self.mu_threshold

        # (c) Amplitude thresholding: eliminate ridges in flat, low-value regions
        ftle_cutoff = np.percentile(ftle_grid, self.ftle_min_percentile)
        significant = smooth_field >= ftle_cutoff

        # Composite Boolean Ridge Mask
        ridge_mask = in_cell & concave & significant

        # 6. Extract Continuous Subpixel Coordinates
        subpixel_x = X[ridge_mask] + px[ridge_mask]
        subpixel_y = Y[ridge_mask] + py[ridge_mask]
        subpixel_points = np.column_stack((subpixel_x, subpixel_y))

        return {
            "ridge_mask": ridge_mask,
            "subpixel_points": subpixel_points,
            "mu1": mu1,
            "mu2": mu2,
            "n1": np.stack([n1_x, n1_y], axis=-1),
            "smoothed_ftle": smooth_field
        }


# ============================================================================
# EXECUTION AND LCS RIDGE IDENTIFICATION
# ============================================================================
def _run_ftle_demo():
    flow = DoubleGyreVectorField(A=0.1, epsilon=0.25, omega=2.0 * np.pi / 10.0)
    engine = FTLEEngine(field=flow, nx=300, ny=150)

    # 1. Forward FTLE (T = +15s): Isolates repelling transport boundaries
    print("[+] Integrating Forward FTLE (T = +15s)...")
    pFTLE = engine.compute_ftle(t0=0.0, T=15.0, dt_step=0.05)

    # 2. Backward FTLE (T = -15s): Isolates attracting manifolds
    print("[+] Integrating Backward FTLE (T = -15s)...")
    nFTLE = engine.compute_ftle(t0=0.0, T=-15.0, dt_step=0.05)

    print(f"\nForward FTLE | Max: {np.max(pFTLE):.4f} nats/s | Mean: {np.mean(pFTLE):.4f} nats/s")
    print(f"Backward FTLE | Max: {np.max(nFTLE):.4f} nats/s | Mean: {np.mean(nFTLE):.4f} nats/s")


def _run_ridge_demo():
    # Generate 2D Gaussian ridge representing a repelling LCS barrier
    nx, ny = 200, 100
    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # Synthetic undulating FTLE transport ridge: y_ridge = 0.5 + 0.15 * sin(2*pi*x)
    y_center = 0.5 + 0.15 * np.sin(2.0 * np.pi * X)
    ftle_synthetic = 0.8 * np.exp(-((Y - y_center) ** 2) / (2.0 * 0.05 ** 2))

    # Add high-frequency integration noise
    ftle_synthetic += np.random.normal(0, 0.015, size=ftle_synthetic.shape)

    # Instantiate extractor
    extractor = FTLERidgeExtractor(
        dx=dx,
        dy=dy,
        sigma_smooth=1.2,
        mu_threshold=-2.0,              # Filter weak curvature
        ftle_min_percentile=50.0         # Discard bottom 50% FTLE floor
    )

    results = extractor.extract_ridges(ftle_synthetic, x_coords=x, y_coords=y)

    ridge_pixels = np.sum(results["ridge_mask"])
    pts = results["subpixel_points"]

    print("=" * 70)
    print("HESSIAN RIDGE EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Grid Dimensions         : {nx} x {ny} (dx = {dx:.4f}, dy = {dy:.4f})")
    print(f"Identified Ridge Pixels : {ridge_pixels}")
    print(f"Subpixel Points Located : {len(pts)}")
    print(f"Transverse Curvature (mu_1) Range : [{np.min(results['mu1']):.2f}, {np.max(results['mu1']):.2f}]")
    print(f"Sample Ridge Point      : x = {pts[0, 0]:.4f}, y = {pts[0, 1]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    _run_ftle_demo()
    print()
    _run_ridge_demo()
