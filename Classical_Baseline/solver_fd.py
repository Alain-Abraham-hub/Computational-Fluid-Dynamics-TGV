"""
solver_fd.py
------------
2D incompressible Navier-Stokes solver for the Convecting Taylor-Green Vortex
using a high-order (4th-order) finite difference spatial discretization and
a fractional-step (Chorin projection) method with 4th-order Runge-Kutta (RK4)
time integration.

Algorithm Overview (each timestep)
------------------------------------
1. Compute the RHS of the momentum equations (advection + diffusion) at u^n, v^n.
2. Advance with RK4 to obtain an intermediate velocity field (u*, v*) — no pressure.
3. Solve the Pressure Poisson Equation (PPE) via FFT for p^{n+1}:
       ∇²p^{n+1} = (ρ/Δt) * ∇·u*
4. Correct velocities to enforce divergence-free constraint:
       u^{n+1} = u* - (Δt/ρ) * ∂p/∂x
       v^{n+1} = v* - (Δt/ρ) * ∂p/∂y
5. Record diagnostics (L2 error, kinetic energy, divergence residual).

CFL Condition
--------------
    Δt ≤ CFL * dx / max(|u|, |v|)
The solver auto-selects Δt each step based on the current velocity field.

Dependencies
------------
    exact_solution.TGVParameters, ExactSolution, make_grid
    fd_operators.*
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

from exact_solution import TGVParameters, ExactSolution, make_grid
from fd_operators import (
    ddx_4, ddy_4,
    laplacian_4,
    divergence_4,
    solve_poisson_fft,
)


# ---------------------------------------------------------------------------
# Solver Configuration
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """
    Configuration for the high-order FD Navier-Stokes solver.

    Attributes
    ----------
    N : int
        Grid resolution (N x N uniform grid).
    t_end : float
        End time of the simulation.
    CFL : float
        Courant-Friedrichs-Lewy number for adaptive time-stepping.
    dt_max : float
        Maximum allowed timestep (safety cap).
    dt_fixed : float, optional
        If > 0, use a fixed timestep instead of adaptive CFL-based stepping.
    record_every : int
        Record diagnostics every this many timesteps.
    verbose : bool
        Print progress during simulation.
    """
    N          : int   = 64
    t_end      : float = 1.0
    CFL        : float = 0.5
    dt_max     : float = 1e-2
    dt_fixed   : float = 0.0      # set > 0 to use fixed timestep
    record_every: int  = 10
    verbose    : bool  = True


# ---------------------------------------------------------------------------
# Diagnostics Data Container
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsRecord:
    """
    Stores time-history of solver diagnostics.

    Attributes
    ----------
    times : List[float]
        Simulation times at which diagnostics were recorded.
    l2_error_u : List[float]
        Relative L2 error of u-velocity vs. exact solution.
    l2_error_v : List[float]
        Relative L2 error of v-velocity vs. exact solution.
    l2_error_combined : List[float]
        Combined relative L2 velocity error norm.
    kinetic_energy_sim : List[float]
        Domain-averaged kinetic energy from the simulation.
    kinetic_energy_exact : List[float]
        Domain-averaged kinetic energy from the exact solution.
    divergence_max : List[float]
        Maximum absolute divergence of the velocity field (should remain ~0).
    dt_history : List[float]
        Timestep size at each recorded step.
    wall_clock : List[float]
        Elapsed wall-clock time at each recorded step (seconds).
    """
    times               : List[float] = field(default_factory=list)
    l2_error_u          : List[float] = field(default_factory=list)
    l2_error_v          : List[float] = field(default_factory=list)
    l2_error_combined   : List[float] = field(default_factory=list)
    kinetic_energy_sim  : List[float] = field(default_factory=list)
    kinetic_energy_exact: List[float] = field(default_factory=list)
    divergence_max      : List[float] = field(default_factory=list)
    dt_history          : List[float] = field(default_factory=list)
    wall_clock          : List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Navier-Stokes RHS
# ---------------------------------------------------------------------------

def _ns_rhs(u: np.ndarray, v: np.ndarray, nu: float,
            dx: float, dy: float) -> tuple:
    """
    Compute the right-hand side of the momentum equations (without pressure):
        RHS_u = -(u*du/dx + v*du/dy) + nu*(d²u/dx² + d²u/dy²)
        RHS_v = -(u*dv/dx + v*dv/dy) + nu*(d²v/dx² + d²v/dy²)

    Parameters
    ----------
    u, v    : np.ndarray — current velocity fields (Nx, Ny)
    nu      : float      — kinematic viscosity
    dx, dy  : float      — grid spacings

    Returns
    -------
    (rhs_u, rhs_v) : tuple of np.ndarray
    """
    # Advection terms (4th-order)
    adv_u = u * ddx_4(u, dx) + v * ddy_4(u, dy)
    adv_v = u * ddx_4(v, dx) + v * ddy_4(v, dy)

    # Viscous diffusion terms (4th-order)
    diff_u = nu * laplacian_4(u, dx, dy)
    diff_v = nu * laplacian_4(v, dx, dy)

    return (-adv_u + diff_u), (-adv_v + diff_v)


# ---------------------------------------------------------------------------
# RK4 Integrator
# ---------------------------------------------------------------------------

def _rk4_step(u: np.ndarray, v: np.ndarray, dt: float,
              nu: float, dx: float, dy: float) -> tuple:
    """
    Advance the intermediate velocity (u*, v*) by one RK4 step (no pressure).

    Parameters
    ----------
    u, v : np.ndarray — velocity at current time step n
    dt   : float      — timestep
    nu   : float      — kinematic viscosity
    dx, dy : float    — grid spacings

    Returns
    -------
    (u_star, v_star) : tuple of np.ndarray
        Intermediate velocity field after RK4 advance (not yet divergence-free).
    """
    k1u, k1v = _ns_rhs(u,                         v,                         nu, dx, dy)
    k2u, k2v = _ns_rhs(u + 0.5 * dt * k1u,        v + 0.5 * dt * k1v,        nu, dx, dy)
    k3u, k3v = _ns_rhs(u + 0.5 * dt * k2u,        v + 0.5 * dt * k2v,        nu, dx, dy)
    k4u, k4v = _ns_rhs(u +       dt * k3u,        v +       dt * k3v,        nu, dx, dy)

    u_star = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    v_star = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return u_star, v_star


# ---------------------------------------------------------------------------
# Main Solver Class
# ---------------------------------------------------------------------------

class NavierStokesFDSolver:
    """
    High-order (4th-order) finite difference solver for the 2D incompressible
    Navier-Stokes equations using fractional-step projection and RK4 time integration.

    Parameters
    ----------
    params : TGVParameters
        Physical parameters of the TGV problem.
    config : SolverConfig
        Solver configuration (grid size, time stepping, etc.).
    """

    def __init__(self, params: TGVParameters, config: SolverConfig):
        self.params = params
        self.config = config
        self.exact  = ExactSolution(params)

        # Build spatial grid
        self.x, self.y, self.dx = make_grid(config.N, params.L_domain)
        self.dy = self.dx  # uniform square grid

        # Initialize velocity and pressure fields from exact IC (t=0)
        self.u = self.exact.u(self.x, self.y, t=0.0)
        self.v = self.exact.v(self.x, self.y, t=0.0)
        self.p = self.exact.pressure(self.x, self.y, t=0.0)

        # Time tracking
        self.t      = 0.0
        self.step   = 0
        self.diag   = DiagnosticsRecord()
        self._t_wall_start = time.perf_counter()

        if config.verbose:
            print(f"NavierStokesFDSolver initialized")
            print(f"  {params}")
            print(f"  Grid: {config.N}x{config.N}, dx={self.dx:.5f}")
            print(f"  t_end={config.t_end}, CFL={config.CFL}")

    def _compute_dt(self) -> float:
        """Compute CFL-limited adaptive timestep."""
        cfg = self.config
        if cfg.dt_fixed > 0.0:
            return cfg.dt_fixed

        u_max = max(np.max(np.abs(self.u)), 1e-10)
        v_max = max(np.max(np.abs(self.v)), 1e-10)
        dt_cfl = cfg.CFL * min(self.dx / u_max, self.dy / v_max)
        return min(dt_cfl, cfg.dt_max)

    def _record_diagnostics(self, dt: float):
        """Compute and store diagnostic quantities at the current time."""
        diag  = self.diag
        exact = self.exact
        x, y, t = self.x, self.y, self.t

        # Exact fields at current time
        u_ex = exact.u(x, y, t)
        v_ex = exact.v(x, y, t)

        # L2 error norms (relative)
        norm_u = np.sqrt(np.mean(u_ex**2))
        norm_v = np.sqrt(np.mean(v_ex**2))
        err_u  = np.sqrt(np.mean((self.u - u_ex)**2)) / max(norm_u, 1e-14)
        err_v  = np.sqrt(np.mean((self.v - v_ex)**2)) / max(norm_v, 1e-14)
        err_combined = np.sqrt(np.mean((self.u - u_ex)**2 + (self.v - v_ex)**2)) / \
                       max(np.sqrt(np.mean(u_ex**2 + v_ex**2)), 1e-14)

        # Domain-averaged kinetic energy
        ek_sim  = 0.5 * np.mean(self.u**2 + self.v**2)
        ek_exact = exact.kinetic_energy(t)

        # Divergence residual (should stay ~0 for incompressible flow)
        div_max = np.max(np.abs(divergence_4(self.u, self.v, self.dx, self.dy)))

        # Wall clock
        wall_t = time.perf_counter() - self._t_wall_start

        diag.times.append(t)
        diag.l2_error_u.append(err_u)
        diag.l2_error_v.append(err_v)
        diag.l2_error_combined.append(err_combined)
        diag.kinetic_energy_sim.append(ek_sim)
        diag.kinetic_energy_exact.append(ek_exact)
        diag.divergence_max.append(div_max)
        diag.dt_history.append(dt)
        diag.wall_clock.append(wall_t)

    def _advance_one_step(self):
        """Advance the solution by one timestep using fractional-step projection."""
        nu = self.params.nu
        dx, dy = self.dx, self.dy
        rho    = self.params.rho

        dt = self._compute_dt()
        # Clip dt so we don't overshoot t_end
        dt = min(dt, self.config.t_end - self.t)

        # --- Step 1: RK4 advance (intermediate velocity, no pressure) ---
        u_star, v_star = _rk4_step(self.u, self.v, dt, nu, dx, dy)

        # --- Step 2: Solve Pressure Poisson Equation ---
        # PPE: ∇²p = (rho/dt) * ∇·u*
        div_star = divergence_4(u_star, v_star, dx, dy)
        rhs_ppe  = (rho / dt) * div_star
        p_new    = solve_poisson_fft(rhs_ppe, dx, dy)

        # --- Step 3: Velocity correction (projection onto divergence-free space) ---
        dpdx, dpdy = ddx_4(p_new, dx), ddy_4(p_new, dy)
        u_new = u_star - (dt / rho) * dpdx
        v_new = v_star - (dt / rho) * dpdy

        self.u  = u_new
        self.v  = v_new
        self.p  = p_new
        self.t += dt
        self.step += 1

        return dt

    def run(self) -> DiagnosticsRecord:
        """
        Run the simulation from t=0 to t=t_end.

        Returns
        -------
        DiagnosticsRecord
            All recorded diagnostics.
        """
        cfg = self.config

        # Record initial state (t=0)
        self._record_diagnostics(dt=0.0)

        if cfg.verbose:
            print(f"\n{'Step':>6}  {'t':>10}  {'dt':>10}  {'L2_err':>12}  {'Ek_sim':>12}  {'div_max':>12}")
            print("-" * 70)

        while self.t < cfg.t_end - 1e-14:
            dt = self._advance_one_step()

            if self.step % cfg.record_every == 0 or self.t >= cfg.t_end - 1e-14:
                self._record_diagnostics(dt)
                if cfg.verbose:
                    d = self.diag
                    print(
                        f"{self.step:>6}  {self.t:>10.5f}  {dt:>10.3e}  "
                        f"{d.l2_error_combined[-1]:>12.3e}  "
                        f"{d.kinetic_energy_sim[-1]:>12.6f}  "
                        f"{d.divergence_max[-1]:>12.3e}"
                    )

        total_time = time.perf_counter() - self._t_wall_start
        if cfg.verbose:
            print(f"\nSimulation complete. Wall-clock time: {total_time:.3f}s")
            print(f"Final L2 error (combined): {self.diag.l2_error_combined[-1]:.3e}")

        return self.diag


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = TGVParameters(Re=100)
    config = SolverConfig(N=64, t_end=1.0, CFL=0.5, record_every=20, verbose=True)

    solver = NavierStokesFDSolver(params, config)
    diag   = solver.run()

    print(f"\nFinal diagnostics:")
    print(f"  L2 error (u)       : {diag.l2_error_u[-1]:.4e}")
    print(f"  L2 error (v)       : {diag.l2_error_v[-1]:.4e}")
    print(f"  L2 error (combined): {diag.l2_error_combined[-1]:.4e}")
    print(f"  Ek_sim             : {diag.kinetic_energy_sim[-1]:.6f}")
    print(f"  Ek_exact           : {diag.kinetic_energy_exact[-1]:.6f}")
    print(f"  Max divergence     : {diag.divergence_max[-1]:.4e}")
