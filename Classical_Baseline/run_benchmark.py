"""
run_benchmark.py
----------------
Top-level runner for the 2D Convecting Taylor-Green Vortex
classical high-order finite difference solver.

What This Script Does
---------------------
1. Prompts the user for a single Reynolds number (Re) and grid resolution (N).
2. Runs the FD solver for that specific configuration.
3. Generates and saves diagnostic plots to the 'results/' directory.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt

from exact_solution import TGVParameters, ExactSolution, make_grid
from solver_fd      import NavierStokesFDSolver, SolverConfig
from metrics        import RuntimeTracker
from visualize      import (
    plot_velocity_fields,
    plot_kinetic_energy,
    plot_l2_error,
    plot_divergence,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str):
    """Save a matplotlib figure to RESULTS_DIR."""
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Single Simulation Run
# ---------------------------------------------------------------------------

def run_single_simulation(Re: float, N: int, t_end: float = 1.0):
    """
    Run the solver for a single configuration.
    Records and plots L2 error, kinetic energy decay, and divergence histories.
    """
    print("\n" + "=" * 60)
    print(f"Running Simulation: Re = {Re}, N = {N}, t_end = {t_end}")
    print("=" * 60)

    params = TGVParameters(Re=Re)
    config = SolverConfig(
        N=N, t_end=t_end, CFL=0.4, dt_max=5e-3,
        record_every=50, verbose=True
    )

    timer = RuntimeTracker()
    timer.start()
    solver = NavierStokesFDSolver(params, config)
    diag   = solver.run()
    elapsed = timer.stop()

    print(f"\n  Wall-clock: {elapsed:.3f}s")

    # Plots
    fig = plot_l2_error(diag.times, diag.l2_error_combined,
                        label=f"Re={Re}",
                        title=f"L2 Velocity Error  [Re={Re}, N={N}]")
    save_fig(fig, f"re{Re}_N{N}_l2_error.png")

    fig = plot_kinetic_energy(diag.times, diag.kinetic_energy_sim,
                              diag.kinetic_energy_exact,
                              title=f"Kinetic Energy Decay  [Re={Re}, N={N}]")
    save_fig(fig, f"re{Re}_N{N}_kinetic_energy.png")

    fig = plot_divergence(diag.times, diag.divergence_max,
                          title=f"Max Divergence  [Re={Re}, N={N}]")
    save_fig(fig, f"re{Re}_N{N}_divergence.png")

    # Final-time velocity field
    exact   = ExactSolution(params)
    x, y, _ = make_grid(N, params.L_domain)
    u_ex    = exact.u(x, y, t=solver.t)
    v_ex    = exact.v(x, y, t=solver.t)
    fig = plot_velocity_fields(x, y, solver.u, solver.v, u_ex, v_ex,
                               t=solver.t,
                               title_prefix=f"Re={Re}, N={N}  ")
    save_fig(fig, f"re{Re}_N{N}_velocity_final.png")

    summary_line = (
        f"Re={Re}  N={N}  t_end={t_end}  final_L2={diag.l2_error_combined[-1]:.3e}  "
        f"Ek_sim={diag.kinetic_energy_sim[-1]:.6f}  "
        f"runtime={elapsed:.3f}s"
    )
    return summary_line


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  TGV Classical Baseline — Single Run             ║")
    print("║  4th-Order FD + RK4 + Fractional-Step Projection ║")
    print("╚══════════════════════════════════════════════════╝\n")

    try:
        re_input = input("Enter Reynolds number (Re) [Default: 100]: ").strip()
        re_val = float(re_input) if re_input else 100.0
        
        n_input = input("Enter grid resolution (N) [Default: 64]: ").strip()
        n_val = int(n_input) if n_input else 64
    except ValueError:
        print("Invalid input. Using default Re=100.0 and N=64.")
        re_val = 100.0
        n_val = 64

    summary_line = run_single_simulation(Re=re_val, N=n_val, t_end=1.0)

    summary_path = os.path.join(RESULTS_DIR, f"summary_re{re_val}_N{n_val}.txt")
    with open(summary_path, "w") as f:
        f.write("Single Run Summary\n")
        f.write("==================\n")
        f.write(summary_line + "\n")
        
    print(f"\n[saved] Summary: {summary_path}")
    print("\n✓ Run complete. Results are in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
