"""
exact_solution.py
-----------------
Analytical closed-form solution for the 2D Convecting Taylor-Green Vortex (TGV).

The convecting TGV has an exact solution for all time t >= 0, which makes it
an ideal benchmark problem — numerical errors can be evaluated precisely against
this reference.

Problem Parameters (from Airbus Specification):
    Domain     : [0, 2*pi] x [0, 2*pi], periodic in both x and y
    L          : characteristic length scale (domain length / (2*pi))
    V0         : characteristic vortex velocity amplitude
    Uc, Vc     : convection velocities in x and y
    rho        : fluid density
    Re         : Reynolds number
    nu         : kinematic viscosity = V0 * L / Re
    p0         : background reference pressure

Exact Solution:
    u(x, y, t) = Uc + V0 * sin((x - Uc*t)/L) * cos((y - Vc*t)/L) * exp(-2*nu*t / L^2)
    v(x, y, t) = Vc - V0 * cos((x - Uc*t)/L) * sin((y - Vc*t)/L) * exp(-2*nu*t / L^2)
    p(x, y, t) = p0 + (rho * V0^2 / 4) * [cos(2*(x-Uc*t)/L) + cos(2*(y-Vc*t)/L)] * exp(-4*nu*t / L^2)
"""

import numpy as np


class TGVParameters:
    """
    Encapsulates all physical parameters for the 2D Convecting Taylor-Green Vortex.

    Attributes
    ----------
    L_domain : float
        Full domain length (default: 2*pi).
    L : float
        Characteristic length scale = L_domain / (2*pi). For L_domain = 2*pi, L = 1.0.
    V0 : float
        Characteristic vortex velocity amplitude.
    Uc : float
        Convection velocity in x-direction.
    Vc : float
        Convection velocity in y-direction.
    rho : float
        Fluid density.
    Re : float
        Reynolds number.
    nu : float
        Kinematic viscosity, computed as V0 * L / Re.
    p0 : float
        Background reference pressure.
    """

    def __init__(
        self,
        Re: float = 100.0,
        V0: float = 1.0,
        Uc: float = 1.0,
        Vc: float = 0.0,
        rho: float = 1.0,
        p0: float = 0.0,
        L_domain: float = 2.0 * np.pi,
    ):
        self.L_domain = L_domain
        self.L = L_domain / (2.0 * np.pi)  # = 1.0 for default domain
        self.V0 = V0
        self.Uc = Uc
        self.Vc = Vc
        self.rho = rho
        self.Re = Re
        self.p0 = p0
        self.nu = V0 * self.L / Re  # kinematic viscosity

    def __repr__(self) -> str:
        return (
            f"TGVParameters(Re={self.Re}, V0={self.V0}, Uc={self.Uc}, "
            f"Vc={self.Vc}, rho={self.rho}, L={self.L:.4f}, nu={self.nu:.6e})"
        )


class ExactSolution:
    """
    Computes the exact closed-form solution of the 2D Convecting Taylor-Green Vortex
    at any given time t on a 2D spatial grid (x, y).

    Parameters
    ----------
    params : TGVParameters
        Physical parameters of the TGV problem.
    """

    def __init__(self, params: TGVParameters):
        self.p = params

    def _decay_factor(self, t: float) -> float:
        """Exponential viscous decay factor: exp(-2 * nu * t / L^2)."""
        return np.exp(-2.0 * self.p.nu * t / self.p.L**2)

    def _pressure_decay_factor(self, t: float) -> float:
        """Pressure decay factor: exp(-4 * nu * t / L^2)."""
        return np.exp(-4.0 * self.p.nu * t / self.p.L**2)

    def u(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Exact x-velocity field.

        Parameters
        ----------
        x, y : np.ndarray
            2D meshgrid arrays of spatial coordinates.
        t : float
            Current simulation time.

        Returns
        -------
        np.ndarray
            x-velocity field of same shape as x and y.
        """
        p = self.p
        xi  = (x - p.Uc * t) / p.L
        eta = (y - p.Vc * t) / p.L
        return p.Uc + p.V0 * np.sin(xi) * np.cos(eta) * self._decay_factor(t)

    def v(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Exact y-velocity field.

        Parameters
        ----------
        x, y : np.ndarray
            2D meshgrid arrays of spatial coordinates.
        t : float
            Current simulation time.

        Returns
        -------
        np.ndarray
            y-velocity field of same shape as x and y.
        """
        p = self.p
        xi  = (x - p.Uc * t) / p.L
        eta = (y - p.Vc * t) / p.L
        return p.Vc - p.V0 * np.cos(xi) * np.sin(eta) * self._decay_factor(t)

    def pressure(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Exact pressure field.

        Parameters
        ----------
        x, y : np.ndarray
            2D meshgrid arrays of spatial coordinates.
        t : float
            Current simulation time.

        Returns
        -------
        np.ndarray
            Pressure field of same shape as x and y.
        """
        p = self.p
        xi  = (x - p.Uc * t) / p.L
        eta = (y - p.Vc * t) / p.L
        return (
            p.p0
            + (p.rho * p.V0**2 / 4.0)
            * (np.cos(2.0 * xi) + np.cos(2.0 * eta))
            * self._pressure_decay_factor(t)
        )

    def kinetic_energy(self, t: float) -> float:
        """
        Exact domain-averaged kinetic energy at time t.
        Computed analytically:
            E_k(t) = Uc^2/2 + Vc^2/2 + (V0^2/4) * exp(-4*nu*t/L^2)

        Parameters
        ----------
        t : float
            Current simulation time.

        Returns
        -------
        float
            Domain-averaged kinetic energy.
        """
        p = self.p
        vortex_energy    = (p.V0**2 / 4.0) * self._pressure_decay_factor(t)
        background_energy = 0.5 * (p.Uc**2 + p.Vc**2)
        return background_energy + vortex_energy

    def all_fields(self, x: np.ndarray, y: np.ndarray, t: float) -> dict:
        """
        Compute all exact fields (u, v, pressure) at time t.

        Parameters
        ----------
        x, y : np.ndarray
            2D meshgrid arrays of spatial coordinates.
        t : float
            Current simulation time.

        Returns
        -------
        dict
            Dictionary with keys 'u', 'v', 'p' containing the exact fields.
        """
        return {
            "u": self.u(x, y, t),
            "v": self.v(x, y, t),
            "p": self.pressure(x, y, t),
        }


def make_grid(N: int, L_domain: float = 2.0 * np.pi):
    """
    Construct a uniform 2D periodic grid over [0, L_domain) x [0, L_domain).
    The endpoint is excluded to maintain periodicity (x[0] == x[-1] + dx).

    Parameters
    ----------
    N : int
        Number of grid points in each direction.
    L_domain : float
        Length of the domain in both x and y directions.

    Returns
    -------
    x, y : np.ndarray
        2D meshgrid arrays of shape (N, N).
    dx : float
        Grid spacing.
    """
    dx   = L_domain / N
    x_1d = np.linspace(0.0, L_domain, N, endpoint=False)
    y_1d = np.linspace(0.0, L_domain, N, endpoint=False)
    x, y = np.meshgrid(x_1d, y_1d, indexing="ij")
    return x, y, dx


if __name__ == "__main__":
    # --- Quick sanity check: evaluate exact solution at t=0 and t=1 ---
    params = TGVParameters(Re=100)
    exact  = ExactSolution(params)

    N = 64
    x, y, dx = make_grid(N)

    u0       = exact.u(x, y, t=0.0)
    v0       = exact.v(x, y, t=0.0)
    p0_field = exact.pressure(x, y, t=0.0)

    print(f"Parameters: {params}")
    print(f"Grid: {N}x{N}, dx = {dx:.6f}")
    print(f"\nt=0:")
    print(f"  u: min={u0.min():.6f}, max={u0.max():.6f}  (expected: {params.Uc - params.V0:.4f} to {params.Uc + params.V0:.4f})")
    print(f"  v: min={v0.min():.6f}, max={v0.max():.6f}  (expected: {-params.V0:.4f} to {params.V0:.4f})")
    print(f"  p: min={p0_field.min():.6f}, max={p0_field.max():.6f}")
    print(f"  Exact kinetic energy at t=0: {exact.kinetic_energy(0.0):.6f}")

    t_eval = 1.0
    print(f"\nt={t_eval}:")
    print(f"  Exact kinetic energy: {exact.kinetic_energy(t_eval):.6f}")
    print(f"  Decay factor:         {exact._decay_factor(t_eval):.6f}")
