"""
visualize.py
------------
Visualization utilities for the 2D Convecting Taylor-Green Vortex solver.

Provides:
  - Contour plots of velocity (u, v) and pressure fields (sim vs. exact side-by-side)
  - Velocity vector (quiver) plots
  - Time-series plots: L2 error decay, kinetic energy decay, divergence history
  - Grid refinement convergence plots (L2 error vs. N)
  - Runtime scaling plots

All plotting functions return matplotlib Figure objects so they can be
saved to file or displayed interactively without side effects.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Optional

# Use non-interactive backend by default (safe for headless environments)
# Override by calling matplotlib.use("TkAgg") or similar before importing
matplotlib.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 12,
    "figure.dpi"     : 120,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Field Contour Plots
# ---------------------------------------------------------------------------

def plot_velocity_fields(
    x: np.ndarray, y: np.ndarray,
    u_sim: np.ndarray, v_sim: np.ndarray,
    u_exact: np.ndarray, v_exact: np.ndarray,
    t: float,
    title_prefix: str = "",
) -> plt.Figure:
    """
    Plot simulated vs. exact u and v velocity fields side-by-side as filled contours.

    Parameters
    ----------
    x, y         : np.ndarray — 2D meshgrid arrays
    u_sim, v_sim : np.ndarray — simulated velocity components
    u_exact, v_exact : np.ndarray — exact velocity components
    t            : float — simulation time (for title)
    title_prefix : str   — optional prefix for the figure title

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{title_prefix}Velocity Fields at t = {t:.4f}", fontsize=15, y=1.01)

    fields = [
        (u_sim,         u_exact,         "u (x-velocity)"),
        (v_sim,         v_exact,         "v (y-velocity)"),
        (u_sim - u_exact, v_sim - v_exact, "Error"),
    ]

    for col, (sim_f, ex_f, label) in enumerate(fields):
        if col < 2:
            # Simulated row
            im0 = axes[0, col].contourf(x, y, sim_f, levels=40, cmap="RdBu_r")
            plt.colorbar(im0, ax=axes[0, col])
            axes[0, col].set_title(f"{label}\n(Simulated)")
            axes[0, col].set_xlabel("x")
            axes[0, col].set_ylabel("y")

            # Exact row
            im1 = axes[1, col].contourf(x, y, ex_f, levels=40, cmap="RdBu_r")
            plt.colorbar(im1, ax=axes[1, col])
            axes[1, col].set_title(f"{label}\n(Exact)")
            axes[1, col].set_xlabel("x")
            axes[1, col].set_ylabel("y")
        else:
            # Error plots (column 2, both rows)
            err_u = u_sim - u_exact
            err_v = v_sim - v_exact
            lim_u = max(np.abs(err_u).max(), 1e-16)
            lim_v = max(np.abs(err_v).max(), 1e-16)

            im_eu = axes[0, 2].contourf(x, y, err_u, levels=40,
                                         cmap="coolwarm", vmin=-lim_u, vmax=lim_u)
            plt.colorbar(im_eu, ax=axes[0, 2])
            axes[0, 2].set_title("u Error\n(Sim - Exact)")
            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("y")

            im_ev = axes[1, 2].contourf(x, y, err_v, levels=40,
                                         cmap="coolwarm", vmin=-lim_v, vmax=lim_v)
            plt.colorbar(im_ev, ax=axes[1, 2])
            axes[1, 2].set_title("v Error\n(Sim - Exact)")
            axes[1, 2].set_xlabel("x")
            axes[1, 2].set_ylabel("y")
            break  # col 2 handled above, stop loop

    fig.tight_layout()
    return fig


def plot_pressure_field(
    x: np.ndarray, y: np.ndarray,
    p_sim: np.ndarray,
    p_exact: Optional[np.ndarray] = None,
    t: float = 0.0,
) -> plt.Figure:
    """
    Plot simulated (and optionally exact) pressure fields.

    Parameters
    ----------
    x, y    : np.ndarray — 2D meshgrid arrays
    p_sim   : np.ndarray — simulated pressure field
    p_exact : np.ndarray, optional — exact pressure field
    t       : float — simulation time

    Returns
    -------
    plt.Figure
    """
    ncols = 3 if p_exact is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"Pressure Field at t = {t:.4f}", fontsize=14)

    im0 = axes[0].contourf(x, y, p_sim, levels=40, cmap="viridis")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title("Simulated Pressure")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    if p_exact is not None:
        im1 = axes[1].contourf(x, y, p_exact, levels=40, cmap="viridis")
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title("Exact Pressure")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

        err_p = p_sim - p_exact
        lim   = max(np.abs(err_p).max(), 1e-16)
        im2   = axes[2].contourf(x, y, err_p, levels=40, cmap="coolwarm",
                                  vmin=-lim, vmax=lim)
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title("Pressure Error")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

    fig.tight_layout()
    return fig


def plot_quiver(
    x: np.ndarray, y: np.ndarray,
    u: np.ndarray, v: np.ndarray,
    t: float = 0.0,
    skip: int = 4,
    title: str = "Velocity Field",
) -> plt.Figure:
    """
    Plot velocity vector field using quiver arrows.

    Parameters
    ----------
    x, y    : np.ndarray — 2D meshgrid arrays
    u, v    : np.ndarray — velocity components
    t       : float      — simulation time
    skip    : int        — plot every `skip`-th arrow to avoid clutter
    title   : str        — figure title

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    # Speed as background color
    speed = np.sqrt(u**2 + v**2)
    cf = ax.contourf(x, y, speed, levels=30, cmap="plasma", alpha=0.5)
    plt.colorbar(cf, ax=ax, label="|u| (speed)")

    # Subsample for quiver
    xs = x[::skip, ::skip]
    ys = y[::skip, ::skip]
    us = u[::skip, ::skip]
    vs = v[::skip, ::skip]
    ax.quiver(xs, ys, us, vs, color="white", alpha=0.8, scale=15)

    ax.set_title(f"{title} at t = {t:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Time-Series Diagnostic Plots
# ---------------------------------------------------------------------------

def plot_l2_error(times: List[float],
                  l2_errors: List[float],
                  label: str = "FD Solver",
                  title: str = "L2 Velocity Error vs. Time") -> plt.Figure:
    """
    Plot the relative L2 velocity error norm vs. simulation time (log scale).

    Parameters
    ----------
    times     : List[float] — simulation times
    l2_errors : List[float] — combined L2 error at each time
    label     : str — legend label
    title     : str — figure title

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(times, l2_errors, "b-o", ms=4, label=label)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"Relative $L_2$ Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_kinetic_energy(times: List[float],
                        ek_sim: List[float],
                        ek_exact: List[float],
                        title: str = "Kinetic Energy Decay") -> plt.Figure:
    """
    Plot domain-averaged kinetic energy (simulated vs. exact) vs. time.

    Parameters
    ----------
    times    : List[float] — simulation times
    ek_sim   : List[float] — simulated kinetic energy at each time
    ek_exact : List[float] — exact kinetic energy at each time
    title    : str — figure title

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14)

    # Absolute values
    axes[0].plot(times, ek_exact, "k-",  lw=2,   label="Exact")
    axes[0].plot(times, ek_sim,   "r--", lw=1.5, label="FD Solver")
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel(r"$E_k$ (domain-averaged)")
    axes[0].set_title("Kinetic Energy")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)

    # Relative error
    ek_exact_arr = np.array(ek_exact)
    ek_sim_arr   = np.array(ek_sim)
    rel_err = np.abs(ek_sim_arr - ek_exact_arr) / np.maximum(np.abs(ek_exact_arr), 1e-14)
    axes[1].semilogy(times, rel_err, "g-o", ms=4, label=r"$|E_{k,sim} - E_{k,exact}| / E_{k,exact}$")
    axes[1].set_xlabel("Time (t)")
    axes[1].set_ylabel("Relative Energy Error")
    axes[1].set_title("Kinetic Energy Relative Error")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_divergence(times: List[float],
                    div_max: List[float],
                    title: str = "Max Velocity Divergence vs. Time") -> plt.Figure:
    """
    Plot the maximum absolute divergence ∇·u vs. time (should be ~machine epsilon).

    Parameters
    ----------
    times   : List[float] — simulation times
    div_max : List[float] — max |∇·u| at each time

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(times, div_max, "m-o", ms=4, label=r"$\max|\nabla \cdot \mathbf{u}|$")
    ax.axhline(1e-12, color="gray", ls="--", alpha=0.6, label="Machine precision ref.")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel(r"$\max|\nabla \cdot \mathbf{u}|$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convergence and Scaling Plots
# ---------------------------------------------------------------------------

def plot_convergence(grid_sizes: List[int],
                     l2_errors: List[float],
                     labels: Optional[List[str]] = None,
                     title: str = "Spatial Convergence (L2 Error vs. Grid Size)") -> plt.Figure:
    """
    Log-log plot of L2 error vs. grid size N with slope reference lines.

    Parameters
    ----------
    grid_sizes : List[int]         — grid sizes N
    l2_errors  : List[float]       — corresponding L2 errors
                 OR List[List[float]] for multiple datasets
    labels     : List[str], optional — legend labels
    title      : str

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Support single or multiple error datasets
    if not isinstance(l2_errors[0], (list, np.ndarray)):
        l2_errors = [l2_errors]
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(l2_errors))]

    colors = plt.cm.tab10(np.linspace(0, 1, len(l2_errors)))

    for errs, label, color in zip(l2_errors, labels, colors):
        Ns   = np.array(grid_sizes)
        errs = np.array(errs)
        ax.loglog(Ns, errs, "o-", color=color, label=label, lw=2, ms=6)

    # Reference slope lines (2nd, 4th order)
    Ns_ref = np.array([grid_sizes[0], grid_sizes[-1]], dtype=float)
    for order, ls, color_ref in [(2, "--", "gray"), (4, "-.", "darkorange")]:
        slope = l2_errors[0][0] * (Ns_ref[0] / Ns_ref)**order
        ax.loglog(Ns_ref, slope, ls=ls, color=color_ref, lw=1.5,
                  label=f"O(N^{{{-order}}}) reference")

    ax.set_xlabel("Grid size N")
    ax.set_ylabel(r"Relative $L_2$ Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_runtime_scaling(grid_sizes: List[int],
                         runtimes: List[float],
                         title: str = "Runtime Scaling vs. Grid Size") -> plt.Figure:
    """
    Log-log plot of wall-clock runtime vs. grid size N.

    Parameters
    ----------
    grid_sizes : List[int]   — grid sizes N
    runtimes   : List[float] — wall-clock times in seconds
    title      : str

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = np.array(grid_sizes)
    rt = np.array(runtimes)
    ax.loglog(Ns, rt, "bs-", lw=2, ms=7, label="FD Solver")

    # O(N^2) reference for 2D solver
    slope_n2 = rt[0] * (Ns / Ns[0])**2
    ax.loglog(Ns, slope_n2, "k--", lw=1.5, label=r"$O(N^2)$ reference")

    ax.set_xlabel("Grid size N")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # --- Quick demo: generate and save all plot types ---
    print("visualize.py: running demo plots (saving to /tmp)...")

    N   = 64
    L   = 2.0 * np.pi
    x1d = np.linspace(0, L, N, endpoint=False)
    x, y = np.meshgrid(x1d, x1d, indexing="ij")

    # Dummy fields
    u_exact = np.sin(x) * np.cos(y)
    v_exact = -np.cos(x) * np.sin(y)
    u_sim   = u_exact + 1e-3 * np.random.randn(N, N)
    v_sim   = v_exact + 1e-3 * np.random.randn(N, N)
    p_exact = 0.25 * (np.cos(2 * x) + np.cos(2 * y))

    # Time-series dummy data
    times    = np.linspace(0, 1, 20).tolist()
    l2_err   = (1e-3 * np.exp(-np.linspace(0, 3, 20))).tolist()
    ek_exact = (0.5 + 0.25 * np.exp(-np.linspace(0, 1, 20))).tolist()
    ek_sim   = (np.array(ek_exact) + 1e-4 * np.random.randn(20)).tolist()
    div_max  = (1e-12 * np.ones(20)).tolist()

    figs = {
        "velocity": plot_velocity_fields(x, y, u_sim, v_sim, u_exact, v_exact, t=0.5),
        "pressure": plot_pressure_field(x, y, p_exact * 1.001, p_exact, t=0.5),
        "quiver"  : plot_quiver(x, y, u_sim, v_sim, t=0.5),
        "l2_error": plot_l2_error(times, l2_err),
        "energy"  : plot_kinetic_energy(times, ek_sim, ek_exact),
        "divergence": plot_divergence(times, div_max),
    }

    for name, fig in figs.items():
        path = f"/tmp/tgv_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    print("Demo complete.")
