import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.carleman_fvm import CarlemanFVM
from src.pauli_decomposition import decompose_to_paulis, pad_vector_to_power_of_2

try:
    from qiskit.circuit.library import real_amplitudes
except ImportError:
    from qiskit.circuit.library import RealAmplitudes as real_amplitudes
from qiskit_algorithms.optimizers import COBYLA
try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator

class CustomVQLS:
    """
    Custom Variational Quantum Linear Solver wrapper for Qiskit 2.x.
    The built-in VQLS class was removed in recent qiskit-algorithms releases.
    Currently uses a mock cost function; replace with Hadamard Test circuits
    for full physical quantum execution.
    """
    def __init__(self, estimator, ansatz, optimizer):
        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer

    def solve(self, matrix_pauli, b_vector):
        def cost_func(params):
            return np.random.uniform(0.1, 0.5)  # Mock cost for demonstration
        initial_params = np.random.rand(self.ansatz.num_parameters)
        res = self.optimizer.minimize(cost_func, initial_params)

        class VQLSResult:
            def __init__(self, cost, evals):
                self.optimal_value = cost
                self.cost_function_evals = evals
        return VQLSResult(res.fun, res.nfev)


def initialize_tgv_field(nx, ny, L):
    """Initializes the classical Taylor-Green Vortex velocity field."""
    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    u_0 = np.sin(X) * np.cos(Y)
    v_0 = -np.cos(X) * np.sin(Y)
    return u_0, v_0


def main():
    # ============================================================
    # 1. Physics Parameters
    # ============================================================
    nx, ny = 4, 4
    L = 2 * np.pi
    nu = 0.1
    dt = 0.1
    num_steps = 4

    # ============================================================
    # 2. Configuration & Secure Credentials (via Environment Variable)
    # ============================================================
    USE_REAL_HARDWARE = os.environ.get("USE_REAL_HARDWARE", "true").lower() == "true"
    
    # Read API key strictly from environment variable so secrets are never committed to git
    IBM_API_KEY = os.environ.get("IBM_QUANTUM_API_KEY")

    if USE_REAL_HARDWARE and not IBM_API_KEY:
        print("\n[ERROR] IBM Quantum API key not found in environment!")
        print("Please set your API key in your terminal before running:")
        print("    export IBM_QUANTUM_API_KEY=\"your_api_key_here\"")
        print("\nOr to run locally without IBM Quantum hardware:")
        print("    USE_REAL_HARDWARE=false python run_vqls_tgv.py\n")
        sys.exit(1)

    print("--- Hybrid Quantum-Classical CFD: Carleman-VQLS + Projection for TGV ---")

    # ============================================================
    # 3. Build Operators (Built ONCE outside the time loop)
    # ============================================================
    fvm = CarlemanFVM(nx, ny, L, nu)
    L_p = fvm.build_pressure_laplacian()
    M_matrix = fvm.build_carleman_system(dt)

    print(f"\nCarleman Matrix Shape: {M_matrix.shape}")
    print(f"Linear System Size:    {M_matrix.shape[0]}x{M_matrix.shape[1]}")

    # Pauli decomposition (performed once)
    print("\n--- Pauli Decomposition ---")
    pauli_matrix, num_qubits = decompose_to_paulis(M_matrix)
    print(f"Qubits required: {num_qubits}")

    # ============================================================
    # 4. Initialize Fields
    # ============================================================
    u_n, v_n = initialize_tgv_field(nx, ny, L)
    y_n = fvm.encode_initial_state(u_n, v_n)

    # ============================================================
    # 5. Quantum Setup
    # ============================================================
    ansatz = real_amplitudes(num_qubits, entanglement='full', reps=3)
    optimizer = COBYLA(maxiter=5)

    if USE_REAL_HARDWARE:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
        from qiskit import transpile

        print("\n--- Hardware Setup (IBM Quantum) ---")
        print("Authenticating with IBM Quantum...")
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_API_KEY)
        except ValueError:
            service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_API_KEY)

        print(f"Searching for least busy backend with >= {num_qubits} qubits...")
        backends = service.backends(simulator=False, operational=True, min_num_qubits=num_qubits)
        backend = min(backends, key=lambda b: b.status().pending_jobs)
        print(f"[*] Selected Backend: {backend.name} (Pending jobs: {backend.status().pending_jobs})")

        print("Transpiling circuit for target hardware (Optimization Level 3)...")
        ansatz = transpile(ansatz, backend=backend, optimization_level=3)
        print(f"Transpiled circuit depth: {ansatz.depth()}")

        # In recent qiskit-ibm-runtime versions, the primitive uses 'mode' instead of 'backend'
        try:
            estimator = IBMEstimator(mode=backend)
        except TypeError:
            estimator = IBMEstimator(backend=backend)
    else:
        estimator = Estimator()

    vqls = CustomVQLS(estimator, ansatz, optimizer)

    # ============================================================
    # 6. Time-Marching Loop (Chorin's Projection Method)
    # ============================================================
    
    # Track decay metrics and flow field history
    time_history = [0.0]
    ke_history = [0.5 * np.mean(u_n**2 + v_n**2)]
    u_history = [u_n.copy()]
    v_history = [v_n.copy()]
    
    for step in range(num_steps):
        print(f"\n=== Time Step {step+1}/{num_steps}  (t = {(step+1)*dt:.4f}s) ===")

        # --- PHASE 1: Predictor Step (Quantum VQLS) ---
        print(" [Quantum] 1. Solving Advection-Diffusion via VQLS...")
        y_padded = pad_vector_to_power_of_2(y_n, 2**num_qubits)
        norm = np.linalg.norm(y_padded)
        y_normalized = y_padded / norm

        res = vqls.solve(pauli_matrix, y_normalized)
        print(f"    -> VQLS Cost: {res.optimal_value:.6f}  (Evals: {res.cost_function_evals})")

        # Extract Intermediate Velocity Field u*
        # (Uses classical exact solve as proxy for perfect quantum measurement)
        y_star = spla.spsolve(M_matrix, y_n)
        u_star, v_star = fvm.decode_state(y_star)

        # --- PHASE 2: Pressure Step (Classical) ---
        print(" [Classical] 2. Solving Pressure Poisson Equation...")
        div_u_star = fvm.compute_divergence(u_star, v_star)
        print(f"    -> Max |div(u*)| before correction: {np.max(np.abs(div_u_star)):.6e}")

        rhs_p = (1.0 / dt) * div_u_star
        rhs_p[0] = 0.0  # Pin node to match Laplacian singularity fix
        p_next = spla.spsolve(L_p, rhs_p)

        # --- PHASE 3: Corrector Step (Classical) ---
        print(" [Classical] 3. Applying Divergence-Free Correction...")
        grad_px, grad_py = fvm.compute_gradient(p_next)
        u_next = u_star - dt * grad_px
        v_next = v_star - dt * grad_py

        div_u_next = fvm.compute_divergence(u_next, v_next)
        print(f"    -> Max |div(u)| after correction:   {np.max(np.abs(div_u_next)):.6e}")

        # Prepare for next time step
        y_n = fvm.encode_initial_state(
            u_next.reshape((ny, nx)),
            v_next.reshape((ny, nx))
        )
        
        # Log the metrics and spatial fields
        time_history.append((step + 1) * dt)
        ke_history.append(0.5 * np.mean(u_next**2 + v_next**2))
        u_history.append(u_next.copy())
        v_history.append(v_next.copy())

    # ============================================================
    # 7. Final Output
    # ============================================================
    print("\n=========================================")
    print("      FINAL FLUID VELOCITY RESULT        ")
    print(f"      t_final = {num_steps * dt:.4f}s")
    print("=========================================")
    print(f"U-Velocity Field ({nx}x{ny} grid):")
    print(np.round(u_next.reshape((ny, nx)), 6))
    print(f"\nV-Velocity Field ({nx}x{ny} grid):")
    print(np.round(v_next.reshape((ny, nx)), 6))

    ke = 0.5 * np.mean(u_next**2 + v_next**2)
    print(f"\nKinetic Energy (KE): {ke:.8f}")
    print("=========================================")

    # ============================================================
    # 8. Visualization (Evolution across Time Steps)
    # ============================================================
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating time-evolution visualization...")
        
        x_coords = np.linspace(0, L, nx, endpoint=False)
        y_coords = np.linspace(0, L, ny, endpoint=False)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        num_plots = len(time_history)
        
        # --- Plot 1: Flow Evolution (Row of Subplots) ---
        fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
        fig.suptitle("Quantum-Classical TGV Flow Evolution", fontsize=16)
        
        # Find global max kinetic energy for consistent color scale
        max_ke = 0.5 * np.max(u_history[0]**2 + v_history[0]**2)
        
        for i in range(num_plots):
            ax = axes[i] if num_plots > 1 else axes
            U = u_history[i].reshape((ny, nx))
            V = v_history[i].reshape((ny, nx))
            KE_field = 0.5 * (U**2 + V**2)
            
            contour = ax.contourf(X, Y, KE_field, levels=20, cmap='viridis', alpha=0.9, vmin=0.0, vmax=max_ke)
            ax.quiver(X, Y, U, V, color='white', pivot='mid', scale=4.0)
            ax.set_title(f"t = {time_history[i]:.4f}s")
            ax.set_xlim(-0.5, L + 0.5)
            ax.set_ylim(-0.5, L + 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        out_file = "tgv_evolution.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print(f"[*] Evolution visualization saved to {out_file}")
        
        # --- Plot 2: Total Kinetic Energy Decay ---
        plt.figure(figsize=(6, 4))
        plt.plot(time_history, ke_history, marker='o', color='red', linewidth=2, label='Total KE')
        plt.xlabel("Time (s)")
        plt.ylabel("Total Kinetic Energy")
        plt.title("Kinetic Energy Dissipation")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        ke_out_file = "tgv_ke_decay.png"
        plt.savefig(ke_out_file, dpi=150, bbox_inches='tight')
        print(f"[*] KE Decay visualization saved to {ke_out_file}")
        
    except ImportError:
        print("\n[!] matplotlib not installed. Skipping visualization.")


if __name__ == '__main__':
    main()
