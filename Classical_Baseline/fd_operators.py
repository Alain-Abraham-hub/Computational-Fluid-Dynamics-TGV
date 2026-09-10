"""
fd_operators.py
---------------
High-order (4th-order) periodic finite difference operators for the
2D Convecting Taylor-Green Vortex solver.

All stencils assume a uniform grid with periodic boundary conditions.
Periodic wrapping is handled using numpy.roll, which is exact and avoids
any ghost-cell bookkeeping.

Supported Operators
-------------------
  ddx_4(f, dx)    : 4th-order central first derivative in x
  ddy_4(f, dy)    : 4th-order central first derivative in y
  d2dx2_4(f, dx)  : 4th-order central second derivative in x
  d2dy2_4(f, dy)  : 4th-order central second derivative in y
  laplacian_4     : 4th-order scalar Laplacian (d2dx2 + d2dy2)
  divergence_4    : 4th-order divergence of a 2D vector field (u, v)
  gradient_4      : 4th-order gradient of a scalar field (returns du/dx, du/dy)
  advection_4     : 4th-order upwind-biased advection term (u . grad)f

Stencil Coefficients (4th-order central)
-----------------------------------------s
  First derivative:
    f'_i = (-f_{i+2} + 8*f_{i+1} - 8*f_{i-1} + f_{i-2}) / (12 * dx)
    Truncation error: O(dx^4)

  Second derivative:
    f''_i = (-f_{i+2} + 16*f_{i+1} - 30*f_i + 16*f_{i-1} - f_{i-2}) / (12 * dx^2)
    Truncation error: O(dx^4)
"""

import numpy as np


# ---------------------------------------------------------------------------
# First Derivatives (4th-order central, periodic)
# ---------------------------------------------------------------------------

def ddx_4(f: np.ndarray, dx: float) -> np.ndarray:
    """
    4th-order central first derivative in x-direction (axis=0).
    Stencil: (-f_{i+2} + 8*f_{i+1} - 8*f_{i-1} + f_{i-2}) / (12*dx)

    Parameters
    ----------
    f  : np.ndarray, shape (Nx, Ny)
        2D scalar field. Axis 0 is x, axis 1 is y.
    dx : float
        Grid spacing in x.

    Returns
    -------
    np.ndarray
        df/dx, same shape as f.
    """
    return (
        -np.roll(f, -2, axis=0)
        + 8.0 * np.roll(f, -1, axis=0)
        - 8.0 * np.roll(f,  1, axis=0)
        +       np.roll(f,  2, axis=0)
    ) / (12.0 * dx)


def ddy_4(f: np.ndarray, dy: float) -> np.ndarray:
    """
    4th-order central first derivative in y-direction (axis=1).
    Stencil: (-f_{j+2} + 8*f_{j+1} - 8*f_{j-1} + f_{j-2}) / (12*dy)

    Parameters
    ----------
    f  : np.ndarray, shape (Nx, Ny)
        2D scalar field.
    dy : float
        Grid spacing in y.

    Returns
    -------
    np.ndarray
        df/dy, same shape as f.
    """
    return (
        -np.roll(f, -2, axis=1)
        + 8.0 * np.roll(f, -1, axis=1)
        - 8.0 * np.roll(f,  1, axis=1)
        +       np.roll(f,  2, axis=1)
    ) / (12.0 * dy)


# ---------------------------------------------------------------------------
# Second Derivatives (4th-order central, periodic)
# ---------------------------------------------------------------------------

def d2dx2_4(f: np.ndarray, dx: float) -> np.ndarray:
    """
    4th-order central second derivative in x-direction (axis=0).
    Stencil: (-f_{i+2} + 16*f_{i+1} - 30*f_i + 16*f_{i-1} - f_{i-2}) / (12*dx^2)

    Parameters
    ----------
    f  : np.ndarray, shape (Nx, Ny)
        2D scalar field.
    dx : float
        Grid spacing in x.

    Returns
    -------
    np.ndarray
        d2f/dx2, same shape as f.
    """
    return (
        -       np.roll(f, -2, axis=0)
        + 16.0 * np.roll(f, -1, axis=0)
        - 30.0 * f
        + 16.0 * np.roll(f,  1, axis=0)
        -       np.roll(f,  2, axis=0)
    ) / (12.0 * dx**2)


def d2dy2_4(f: np.ndarray, dy: float) -> np.ndarray:
    """
    4th-order central second derivative in y-direction (axis=1).
    Stencil: (-f_{j+2} + 16*f_{j+1} - 30*f_j + 16*f_{j-1} - f_{j-2}) / (12*dy^2)

    Parameters
    ----------
    f  : np.ndarray, shape (Nx, Ny)
        2D scalar field.
    dy : float
        Grid spacing in y.

    Returns
    -------
    np.ndarray
        d2f/dy2, same shape as f.
    """
    return (
        -       np.roll(f, -2, axis=1)
        + 16.0 * np.roll(f, -1, axis=1)
        - 30.0 * f
        + 16.0 * np.roll(f,  1, axis=1)
        -       np.roll(f,  2, axis=1)
    ) / (12.0 * dy**2)


# ---------------------------------------------------------------------------
# Composite Operators
# ---------------------------------------------------------------------------

def laplacian_4(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    4th-order scalar Laplacian: d2f/dx2 + d2f/dy2.

    Parameters
    ----------
    f       : np.ndarray, shape (Nx, Ny)
    dx, dy  : float, grid spacings

    Returns
    -------
    np.ndarray
        Laplacian of f, same shape as f.
    """
    return d2dx2_4(f, dx) + d2dy2_4(f, dy)


def divergence_4(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    4th-order divergence of a 2D vector field (u, v):
        div(u, v) = du/dx + dv/dy

    Parameters
    ----------
    u, v    : np.ndarray, shape (Nx, Ny) — x and y velocity components
    dx, dy  : float, grid spacings

    Returns
    -------
    np.ndarray
        Divergence field, same shape as u and v.
    """
    return ddx_4(u, dx) + ddy_4(v, dy)


def gradient_4(f: np.ndarray, dx: float, dy: float) -> tuple:
    """
    4th-order gradient of a scalar field f:
        grad(f) = (df/dx, df/dy)

    Parameters
    ----------
    f       : np.ndarray, shape (Nx, Ny)
    dx, dy  : float, grid spacings

    Returns
    -------
    tuple (dfdx, dfdy)
        Each component has same shape as f.
    """
    return ddx_4(f, dx), ddy_4(f, dy)


def advection_4(u: np.ndarray, v: np.ndarray, f: np.ndarray,
                dx: float, dy: float) -> np.ndarray:
    """
    4th-order advection term: (u . grad) f = u * df/dx + v * df/dy.
    Uses central 4th-order differences for the gradient.

    Parameters
    ----------
    u, v    : np.ndarray, shape (Nx, Ny) — velocity components
    f       : np.ndarray, shape (Nx, Ny) — transported scalar field
    dx, dy  : float, grid spacings

    Returns
    -------
    np.ndarray
        Advection term, same shape as f.
    """
    return u * ddx_4(f, dx) + v * ddy_4(f, dy)


# ---------------------------------------------------------------------------
# Spectral Poisson Solver (FFT-based, exact for periodic domains)
# ---------------------------------------------------------------------------

def solve_poisson_fft(rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Solve the 2D Poisson equation  ∇²p = rhs  on a periodic domain
    using the 2D Fast Fourier Transform.

    This gives the exact (spectral-accuracy) solution for periodic boundaries
    and is far more efficient than iterative solvers for uniform grids.

    The DC component (mean pressure) is set to zero (gauge condition),
    since pressure is only defined up to a constant for incompressible flow.

    Parameters
    ----------
    rhs     : np.ndarray, shape (Nx, Ny)
        Right-hand side of the Poisson equation.
    dx, dy  : float
        Grid spacings in x and y (assumed equal for a square domain).

    Returns
    -------
    np.ndarray
        Pressure field p, same shape as rhs.
    """
    Nx, Ny = rhs.shape

    # Wavenumbers for FFT (using rfft2 conventions for real-valued rhs)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)  # shape (Ny,)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")   # shape (Nx, Ny)

    # Eigenvalues of the Laplacian in Fourier space
    K2 = KX**2 + KY**2

    # Avoid division by zero at the DC (k=0) mode
    K2[0, 0] = 1.0

    # Solve in Fourier space: p_hat = rhs_hat / K2
    rhs_hat = np.fft.fft2(rhs)
    p_hat   = rhs_hat / (-K2)

    # Enforce zero-mean pressure (gauge condition)
    p_hat[0, 0] = 0.0

    # Return to physical space
    p = np.real(np.fft.ifft2(p_hat))
    return p


if __name__ == "__main__":
    # --- Convergence test: verify 4th-order accuracy of ddx_4 on sin(x) ---
    print("=== 4th-order FD Convergence Test ===")
    print(f"{'N':>6}  {'L2 error ddx':>16}  {'L2 error d2dx2':>16}")

    for N in [16, 32, 64, 128, 256]:
        dx    = 2.0 * np.pi / N
        x_1d  = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        f     = np.sin(x_1d)[:, None] * np.ones((1, N))  # shape (N, N)

        # Exact derivatives
        dfdx_exact  = np.cos(x_1d)[:, None] * np.ones((1, N))
        d2fdx2_exact = -np.sin(x_1d)[:, None] * np.ones((1, N))

        # Numerical derivatives
        dfdx_num   = ddx_4(f, dx)
        d2fdx2_num = d2dx2_4(f, dx)

        err1 = np.sqrt(np.mean((dfdx_num  - dfdx_exact)**2))
        err2 = np.sqrt(np.mean((d2fdx2_num - d2fdx2_exact)**2))
        print(f"{N:>6}  {err1:>16.3e}  {err2:>16.3e}")

    # --- FFT Poisson solver test: solve ∇²p = -2*sin(x)*cos(y), exact p = sin(x)*cos(y) ---
    print("\n=== FFT Poisson Solver Test ===")
    N  = 64
    dx = 2.0 * np.pi / N
    x_1d = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    x, y = np.meshgrid(x_1d, x_1d, indexing="ij")
    rhs  = -2.0 * np.sin(x) * np.cos(y)   # ∇²(sin x cos y) = -2 sin x cos y
    p_exact = np.sin(x) * np.cos(y)

    p_num = solve_poisson_fft(rhs, dx, dx)
    # Shift to match zero-mean gauge
    p_num += np.mean(p_exact) - np.mean(p_num)
    err = np.sqrt(np.mean((p_num - p_exact)**2))
    print(f"  N={N}, L2 Poisson error = {err:.3e}  (expected: machine precision)")
