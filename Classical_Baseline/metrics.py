"""
metrics.py
----------
Benchmark metrics for the 2D Convecting Taylor-Green Vortex solver.

Provides functions to compute:
  - Relative L2 velocity error norm against the exact analytical solution
  - Domain-averaged kinetic energy from a numerical field
  - Spatial convergence order estimation (Richardson extrapolation style)
  - Runtime scaling analysis utilities

All functions are stateless and operate on numpy arrays for easy integration
with the solver and the visualization pipeline.
"""

import time
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# L2 Error Norms
# ---------------------------------------------------------------------------

def l2_error_field(f_sim: np.ndarray, f_exact: np.ndarray) -> float:
    """
    Compute the relative L2 error norm between a simulated and exact field.

    Definition:
        E_L2 = || f_sim - f_exact ||_2 / || f_exact ||_2

    where ||.||_2 = sqrt(mean(f^2)) (RMS norm, normalized by domain size).

    Parameters
    ----------
    f_sim   : np.ndarray — simulated field
    f_exact : np.ndarray — exact/reference field

    Returns
    -------
    float
        Relative L2 error (dimensionless). Returns absolute L2 if
        f_exact is near-zero to avoid division by zero.
    """
    norm_exact = np.sqrt(np.mean(f_exact**2))
    norm_error = np.sqrt(np.mean((f_sim - f_exact)**2))
    if norm_exact < 1e-14:
        return float(norm_error)
    return float(norm_error / norm_exact)


def l2_error_velocity(u_sim: np.ndarray, v_sim: np.ndarray,
                      u_exact: np.ndarray, v_exact: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute relative L2 errors for u, v, and combined velocity vector.

    Parameters
    ----------
    u_sim, v_sim     : np.ndarray — simulated velocity components
    u_exact, v_exact : np.ndarray — exact velocity components

    Returns
    -------
    (err_u, err_v, err_combined) : Tuple[float, float, float]
        err_u         : relative L2 error in x-velocity
        err_v         : relative L2 error in y-velocity
        err_combined  : relative L2 error of the full velocity vector
    """
    err_u        = l2_error_field(u_sim, u_exact)
    err_v        = l2_error_field(v_sim, v_exact)
    combined_num = np.sqrt(np.mean((u_sim - u_exact)**2 + (v_sim - v_exact)**2))
    combined_den = np.sqrt(np.mean(u_exact**2 + v_exact**2))
    err_combined = float(combined_num / combined_den) if combined_den > 1e-14 else float(combined_num)
    return err_u, err_v, err_combined


# ---------------------------------------------------------------------------
# Kinetic Energy
# ---------------------------------------------------------------------------

def kinetic_energy_numerical(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute domain-averaged kinetic energy from numerical velocity fields.

    Definition:
        E_k = (1 / |Omega|) * integral_Omega (u^2 + v^2)/2 dA
            ≈ mean((u^2 + v^2) / 2)    [on uniform grid]

    Parameters
    ----------
    u, v : np.ndarray — velocity components

    Returns
    -------
    float
        Domain-averaged kinetic energy.
    """
    return float(0.5 * np.mean(u**2 + v**2))


def kinetic_energy_error(ek_sim: float, ek_exact: float) -> float:
    """
    Relative kinetic energy error.

    Parameters
    ----------
    ek_sim   : float — simulated kinetic energy
    ek_exact : float — exact kinetic energy

    Returns
    -------
    float
        |ek_sim - ek_exact| / ek_exact
    """
    if abs(ek_exact) < 1e-14:
        return abs(ek_sim - ek_exact)
    return abs(ek_sim - ek_exact) / abs(ek_exact)


# ---------------------------------------------------------------------------
# Divergence Residual (mass conservation)
# ---------------------------------------------------------------------------

def max_divergence(u: np.ndarray, v: np.ndarray,
                   dx: float, dy: float,
                   order: int = 4) -> float:
    """
    Compute the maximum absolute divergence of the velocity field using
    finite difference approximation.

    For a divergence-free flow (∇·u = 0), this should remain at or near
    machine precision (~1e-14) for the projection-corrected velocity field.

    Parameters
    ----------
    u, v    : np.ndarray — velocity components
    dx, dy  : float      — grid spacings
    order   : int        — FD order (4 supported)

    Returns
    -------
    float
        Max |∇·u| over the domain.
    """
    from fd_operators import divergence_4
    if order == 4:
        div = divergence_4(u, v, dx, dy)
    else:
        raise ValueError(f"Only 4th-order divergence is currently supported, got order={order}")
    return float(np.max(np.abs(div)))


# ---------------------------------------------------------------------------
# Spatial Convergence Analysis
# ---------------------------------------------------------------------------

def estimate_convergence_order(errors: List[float],
                               grid_sizes: List[int]) -> List[float]:
    """
    Estimate the spatial convergence order from a set of grid-refinement errors
    using Richardson extrapolation formula:

        p_i = log(E_{i} / E_{i+1}) / log(N_{i+1} / N_{i})

    Parameters
    ----------
    errors     : List[float] — L2 errors at each grid resolution
    grid_sizes : List[int]   — corresponding grid sizes N (ascending)

    Returns
    -------
    List[float]
        Estimated convergence orders between consecutive grid refinements.
        Length = len(errors) - 1.
    """
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] < 1e-15 or errors[i + 1] < 1e-15:
            orders.append(float("nan"))
        else:
            ratio = errors[i] / errors[i + 1]
            h_ratio = grid_sizes[i + 1] / grid_sizes[i]
            orders.append(np.log(ratio) / np.log(h_ratio))
    return orders


# ---------------------------------------------------------------------------
# Runtime Scaling Utilities
# ---------------------------------------------------------------------------

class RuntimeTracker:
    """
    Lightweight timer utility for measuring solver wall-clock time.

    Usage
    -----
    >>> tracker = RuntimeTracker()
    >>> tracker.start()
    >>> # ... run simulation ...
    >>> elapsed = tracker.stop()
    """

    def __init__(self):
        self._t0: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self):
        """Start or restart the timer."""
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._t0 is None:
            raise RuntimeError("Timer was not started. Call start() first.")
        self.elapsed = time.perf_counter() - self._t0
        self._t0 = None
        return self.elapsed

    def __repr__(self):
        return f"RuntimeTracker(elapsed={self.elapsed:.4f}s)"


def scaling_table(grid_sizes: List[int],
                  runtimes:   List[float],
                  l2_errors:  List[float]) -> str:
    """
    Format a grid-refinement scaling table as a printable string.

    Parameters
    ----------
    grid_sizes : List[int]   — list of N values (grid sizes)
    runtimes   : List[float] — wall-clock runtimes in seconds
    l2_errors  : List[float] — L2 combined velocity errors

    Returns
    -------
    str
        Formatted table string.
    """
    orders = estimate_convergence_order(l2_errors, grid_sizes)
    orders = [float("nan")] + orders  # pad first entry

    header = f"{'N':>6}  {'Runtime (s)':>14}  {'L2 Error':>14}  {'Conv. Order':>14}"
    separator = "-" * len(header)
    rows = [header, separator]
    for N, rt, err, ord_ in zip(grid_sizes, runtimes, l2_errors, orders):
        ord_str = f"{ord_:.2f}" if not np.isnan(ord_) else "  ---"
        rows.append(f"{N:>6}  {rt:>14.4f}  {err:>14.3e}  {ord_str:>14}")
    return "\n".join(rows)


if __name__ == "__main__":
    # --- Self-test using known values ---

    # Dummy fields
    N = 64
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")

    u_exact = np.sin(X) * np.cos(Y)
    v_exact = -np.cos(X) * np.sin(Y)
    # Perturb slightly
    u_sim = u_exact + 1e-4 * np.random.randn(*u_exact.shape)
    v_sim = v_exact + 1e-4 * np.random.randn(*v_exact.shape)

    err_u, err_v, err_comb = l2_error_velocity(u_sim, v_sim, u_exact, v_exact)
    ek_sim  = kinetic_energy_numerical(u_sim, v_sim)
    ek_ex   = kinetic_energy_numerical(u_exact, v_exact)
    ek_err  = kinetic_energy_error(ek_sim, ek_ex)

    print("=== metrics.py self-test ===")
    print(f"  L2 error u:       {err_u:.4e}")
    print(f"  L2 error v:       {err_v:.4e}")
    print(f"  L2 error combined:{err_comb:.4e}")
    print(f"  Ek_sim:           {ek_sim:.6f}")
    print(f"  Ek_exact:         {ek_ex:.6f}")
    print(f"  Ek rel error:     {ek_err:.4e}")

    # Convergence order test
    errs = [1.0e-2, 1.0e-3, 1.0e-4]
    Ns   = [16, 32, 64]
    orders = estimate_convergence_order(errs, Ns)
    print(f"\n  Convergence orders: {[f'{o:.2f}' for o in orders]}")
    print(f"\n{scaling_table(Ns, [0.1, 0.4, 1.8], errs)}")
