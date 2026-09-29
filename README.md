# Computational Fluid Dynamics: Taylor-Green Vortex (Classical vs. Quantum)

A benchmark and exploratory framework solving the **2D Taylor-Green Vortex (TGV)** problem comparing high-order classical finite-difference solvers against a hybrid **Quantum-Classical Carleman-VQLS** algorithm using Qiskit.

---

## Table of Contents
- [Overview](#overview)
- [Physical Problem: Taylor-Green Vortex](#physical-problem-taylor-green-vortex)
- [Architectures & Approaches](#architectures--approaches)
  - [1. Classical Baseline](#1-classical-baseline)
  - [2. Hybrid Quantum-Classical Pipeline](#2-hybrid-quantum-classical-pipeline)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
  - [Running the Classical Benchmark](#running-the-classical-benchmark)
  - [Running the Quantum-Classical Simulation](#running-the-quantum-classical-simulation)
- [Current State & NISQ Considerations](#current-state--nisq-considerations)
- [License & Acknowledgments](#license--acknowledgments)

---

## Overview

The Navier-Stokes equations governing incompressible fluid dynamics are fundamentally nonlinear due to the convective term $(\mathbf{u} \cdot \nabla)\mathbf{u}$. Quantum algorithms like HHL (Harrow-Hassidim-Lloyd) and VQLS (Variational Quantum Linear Solver) excel at solving *linear* systems of equations ($A\mathbf{x} = \mathbf{b}$).

This repository investigates:
1. **Classical Reference**: High-order finite-difference discretization with Runge-Kutta time marching for baseline accuracy and energy dissipation.
2. **Quantum Formulation**: A hybrid architecture using **Carleman Linearization** to cast the nonlinear advection-diffusion equation into an extended linear system, solved using **VQLS** and **Chorin's Projection Method** to enforce the divergence-free condition.

---

## Physical Problem: Taylor-Green Vortex

The 2D Taylor-Green Vortex is a classic benchmark in fluid dynamics with a known analytical solution under periodic boundary conditions on a domain $[0, 2\pi] \times [0, 2\pi]$:

$$u(x, y, t) = \sin(x)\cos(y) e^{-2\nu t}$$
$$v(x, y, t) = -\cos(x)\sin(y) e^{-2\nu t}$$
$$p(x, y, t) = \frac{1}{4} \left( \cos(2x) + \cos(2y) \right) e^{-4\nu t}$$

It exhibits decay of total kinetic energy:
$$E_k(t) = \frac{1}{2} \langle u^2 + v^2 \rangle = E_k(0) e^{-4\nu t}$$

---

## Architectures & Approaches

```
+-------------------------------------------------------------------------+
|                  Hybrid Quantum-Classical Architecture                  |
+-------------------------------------------------------------------------+
|                                                                         |
|  [State u^n]                                                            |
|       |                                                                 |
|       v                                                                 |
|  Carleman Lifting: y = [u, u ⊗ u]^T                                     |
|       |                                                                 |
|       v                                                                 |
|  Quantum Stage (Predictor):                                             |
|  - Matrix M decomposed into SparsePauliOp (11 Qubits)                  |
|  - RealAmplitudes Ansatz transpiled for IBM Quantum hardware            |
|  - VQLS optimizer (COBYLA) solves intermediate velocity u*              |
|       |                                                                 |
|       v                                                                 |
|  Classical Stage (Corrector):                                           |
|  - Solves Pressure Poisson Equation: ∇² p^(n+1) = (1/Δt) ∇·u*           |
|  - Projects to divergence-free field: u^(n+1) = u* - Δt ∇p^(n+1)        |
|       |                                                                 |
|       +-------------------- Loops to next time step --------------------+
+-------------------------------------------------------------------------+
```

### 1. Classical Baseline
- **Discretization**: Finite-difference method (FDM) on structured staggered grids.
- **Time Stepping**: Low-storage explicit Runge-Kutta schemes (RK3/RK4).
- **Diagnostics**: Tracks $L_2$ error norm relative to exact analytical solution, kinetic energy decay, and divergence $\nabla \cdot \mathbf{u}$.

### 2. Hybrid Quantum-Classical Pipeline
- **Finite Volume Matrix Representation**: Discrete Laplacian and convective flux operators built on cell-centered grids.
- **Carleman Linearization (Order 2)**: Truncates the advection quadratic nonlinearity by embedding the state into an augmented vector:
  $$\mathbf{y} = \begin{bmatrix} \mathbf{u} \\ \mathbf{u} \otimes \mathbf{u} \end{bmatrix}$$
  yielding an implicit linear system:
  $$(I - \Delta t \, M) \, \mathbf{y}^{n+1} = \mathbf{y}^n$$
- **Pauli Decomposition**: Padded to a power-of-2 dimension ($2048 \times 2048$) and converted into a `SparsePauliOp` Hamiltonian for an 11-qubit register.
- **Hardware Integration**: Connects via `qiskit-ibm-runtime` to discover least-busy backends (e.g. `ibm_marrakesh`), transpiling the parameterised ansatz down to native gate sets.
- **Chorin's Projection**: Decouples momentum prediction from the incompressibility constraint. The pressure Poisson equation is handled classically to enforce $\nabla \cdot \mathbf{u} = 0$.

---

## Repository Structure

```
Computational-Fluid-Dynamics-TGV/
├── Classical_Baseline/           # Classical reference solver
│   ├── run_benchmark.py          # Interactive classical benchmark CLI
│   ├── solver_fd.py              # Finite-difference Navier-Stokes solver
│   ├── exact_solution.py         # Analytical Taylor-Green solution
│   ├── metrics.py                # Runtime, error, and energy trackers
│   └── visualize.py              # Classical plotting utilities
├── Quantum_Implementation/       # Hybrid QCFD implementation
│   ├── run_vqls_tgv.py           # Main hybrid time-stepping simulation
│   └── src/
│       ├── carleman_fvm.py       # FVM advection/diffusion & Carleman builder
│       └── pauli_decomposition.py# Matrix padding and SparsePauliOp decomposition
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.11/3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Alain-Abraham-hub/Computational-Fluid-Dynamics-TGV.git
cd Computational-Fluid-Dynamics-TGV
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv cfdtgv
source cfdtgv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Running the Classical Benchmark
Navigate to `Classical_Baseline/` and run:
```bash
cd Classical_Baseline
python run_benchmark.py
```
You will be prompted to choose:
- Reynolds number (e.g. `Re = 100`)
- Grid resolution (e.g. `N = 64` or `N = 128`)

Plots and diagnostic metrics are saved into `Classical_Baseline/results/`.

---

### Running the Quantum-Classical Simulation

Navigate to `Quantum_Implementation/`:
```bash
cd Quantum_Implementation
```

#### Option A: Local Execution (Simulator Mode)
Runs the hybrid simulation locally without contacting IBM Quantum cloud services:
```bash
USE_REAL_HARDWARE=false python run_vqls_tgv.py
```

#### Option B: Real Quantum Hardware (IBM Quantum)
1. Set your IBM Quantum API token in your terminal environment:
   ```bash
   export IBM_QUANTUM_API_KEY="your_ibm_quantum_api_token"
   ```
2. Run the solver:
   ```bash
   python run_vqls_tgv.py
   ```
The solver will automatically authenticate, identify the least-busy quantum backend with $\ge 11$ qubits, transpile the parameterized circuit to target hardware native gates, and march forward in time.

#### Output Visualizations
Each run generates:
- `tgv_evolution.png`: Multi-panel visualization showing velocity vector field snapshots across each simulated time step.
- `tgv_ke_decay.png`: Total kinetic energy decay curve verifying viscous dissipation.

---

## Current State & NISQ Considerations

- **State Extraction / Tomography**: Full Quantum State Tomography on $n=11$ qubits requires measuring in $3^{11} = 177,147$ Pauli bases per time step, which is prohibitive on NISQ hardware. This solver utilizes a hybrid workflow where the quantum circuit is compiled and evaluated, while state propagation bridges into the classical pressure corrector.
- **Carleman Truncation**: Truncating at order 2 ($x \otimes x$) introduces an approximation error that is valid for small time increments $\Delta t$. Higher orders exponentially increase the qubit and Pauli string counts.
