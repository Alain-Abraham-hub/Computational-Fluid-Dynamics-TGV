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
- [ParaView Visualization & Animation Pipeline](#paraview-visualization--animation-pipeline)
  - [1. How the VTK Export Works](#1-how-the-vtk-export-works)
  - [2. Step-by-Step ParaView Guide](#2-step-by-step-paraview-guide)
  - [3. Exporting Animations & Generating GIFs](#3-exporting-animations--generating-gifs)
- [Current State & NISQ Considerations](#current-state--nisq-considerations)

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
│   ├── visualize.py              # Classical plotting utilities
│   └── vtk_export.py             # VTK exporter for classical flow fields
├── Quantum_Implementation/       # Hybrid QCFD implementation
│   ├── run_vqls_tgv.py           # Main hybrid time-stepping simulation
│   ├── make_gif.py               # Stitches ParaView PNG frames into GIF
│   └── src/
│       ├── carleman_fvm.py       # FVM advection/diffusion & Carleman builder
│       ├── pauli_decomposition.py# Matrix padding and SparsePauliOp decomposition
│       └── vtk_export.py         # VTK exporter for quantum flow fields
├── Para View Visualization/      # ParaView VTK series, PNG frames & GIF
│   ├── Classical vtk files/      # 64x64 classical VTK time series
│   ├── Quantum vtk files/        # 4x4 quantum VTK time series
│   ├── TGV decay png's/          # Exported ParaView animation frames
│   └── tgv_decay_animation.gif   # Animated GIF of TGV decay
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.11/3.14)
- Git
- [ParaView](https://www.paraview.org/download/) (for interactive 2D/3D CFD visualization)

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

### 4. Install ParaView (macOS)
```bash
brew install --cask paraview
```
*(Or download the official installer for your platform from [paraview.org/download](https://www.paraview.org/download/)).*

---

## How to Run

### Running the Classical Benchmark
Navigate to `Classical_Baseline/` and run:
```bash
cd Classical_Baseline
python run_benchmark.py
```
You will be prompted to choose:
- Reynolds number (e.g. `Re = 10` or `Re = 100`)
- Grid resolution (e.g. `N = 64` or `N = 128`)

Diagnostic plots are saved into `Classical_Baseline/results/`, and VTK time-series snapshots (`classical_tgv_000.vtk` ... `classical_tgv_025.vtk`) are automatically exported to `Para View Visualization/Classical vtk files/`.

---

### Running the Quantum-Classical Simulation

Navigate to `Quantum_Implementation/`:
```bash
cd Quantum_Implementation
```

#### Option A: Local Execution (Simulator Mode)
Runs the 25-step hybrid simulation ($t = 0.0\text{s}$ to $t = 1.0\text{s}$, $\Delta t = 0.04\text{s}$) locally without contacting IBM Quantum cloud services:
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
- `tgv_evolution.png`: Multi-panel visualization showing velocity vector field snapshots across milestone time steps.
- `tgv_ke_decay.png`: Total kinetic energy decay curve across all 25 time steps verifying viscous dissipation.
- `Para View Visualization/Quantum vtk files/tgv_step_000.vtk ... tgv_step_025.vtk`: Full VTK time series for ParaView.

---

## ParaView Visualization & Animation Pipeline

![ParaView TGV Decay Animation](Para%20View%20Visualization/tgv_decay_animation.gif)

Both solvers include a zero-dependency native VTK exporter (`vtk_export.py`) that writes standard **VTK Legacy ASCII (`DATASET STRUCTURED_POINTS`)** snapshots across all 26 frames ($t = 0.00\text{s}$ to $t = 1.00\text{s}$).

### 1. How the VTK Export Works
At every recorded time step, the exporter maps the 2D uniform mesh onto a 3D structured point grid ($z = 0$) and writes the following point data arrays:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| **`velocity`** | 3-Vector $(u, v, 0)$ | Vortex-frame velocity vectors showing the 4 counter-rotating whirlpools. |
| **`velocity_convecting`** | 3-Vector $(u, v, 0)$ | Full lab-frame velocity including mean convection $(U_c, V_c)$ *(Classical)*. |
| **`kinetic_energy`** | Scalar | Local fluid kinetic energy $E_k = \frac{1}{2}(u^2 + v^2)$. |
| **`vorticity`** | Scalar | Out-of-plane curl $\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ isolating vortex cores *(Classical)*. |
| **`pressure`** | Scalar | Pressure field $p(x, y, t)$ obtained from the Poisson equation. |

* **Why Compare Both Grids in ParaView?**
  * **Quantum ($4 \times 4$ Grid):** Discretizes the domain into 16 points to fit within an **11-qubit** Carleman-linearized circuit ($1056 \times 1056$ padded to $2048 \times 2048$). Because the 16 points sample peaks and zeros in a checkerboard pattern, ParaView's triangle renderer displays diagonal energy bands that decay exponentially over time.
  * **Classical ($64 \times 64$ Grid):** Resolves the domain across 4,096 points using 4th-order finite differences, revealing crisp circular vortex cores, smooth streamlines, and low-pressure vortex centers.

---

### 2. Step-by-Step ParaView Guide

#### Loading the Time Series
1. Open **ParaView** and go to **`File` $\rightarrow$ `Open...`**
2. Navigate to **`Para View Visualization/`**:
   * For High-Res Classical ($64 \times 64$): Open `Classical vtk files/` and select the grouped **`classical_tgv_..vtk`**.
   * For Quantum ($4 \times 4$): Open `Quantum vtk files/` and select the grouped **`tgv_step_..vtk`**.
3. Click **`OK`**, then click the blue **`Apply`** button in the **Properties** panel.
   *(Tip: If you ever close the sidebars accidentally, restore them via top menu **`View` $\rightarrow$ `Pipeline Browser`** and **`View` $\rightarrow$ `Properties`**).*

#### Configuring the Heatmap Surface
1. In the **Properties** panel (or top toolbar), set **Representation** to **`Surface`** (or **`Slice`**).
2. Under **Coloring**, select **`kinetic_energy`**, **`vorticity`**, or **`pressure`**.
3. If the viewport looks empty, click the **Reset Camera / Zoom to Data** button (or `-Z`) in the top toolbar.

#### Adding Velocity Vector Arrows (`Glyph` Filter)
1. Highlight your `.vtk` series in the **Pipeline Browser**.
2. In the top menu bar, click **`Filters` $\rightarrow$ `Alphabetical` $\rightarrow$ `Glyph`**.
3. In the **Properties** panel for `Glyph1`, configure:
   * **Glyph Type:** `Arrow`
   * **Orientation Array:** `velocity`
   * **Scale Array:** `velocity`
   * **Scale Factor:** `0.4`
   * **Glyph Mode (under Masking):**
     * For the **$4 \times 4$ Quantum grid**: Choose **`All Points`** *(default "Volume Sampling" will show 0 arrows on a 2D flat sheet)*.
     * For the **$64 \times 64$ Classical grid**: Choose **`Every Nth Point`** with **Stride = `4`** so arrows are cleanly spaced.
4. Click **`Apply`**.
5. In the **Pipeline Browser**, click the **eyeball icon (`👁`)** next to your base `.vtk` dataset so **both** the colored heatmap and the vector arrows are visible together.
6. Press the **`▶ Play`** button in the top toolbar to animate the vortex decay across all 26 frames!

---

### 3. Exporting Animations & Generating GIFs
1. In ParaView, click **`File` $\rightarrow$ `Save Animation...`**
2. Set **`Files of type`** to **`PNG images (*.png)`** and save into `Para View Visualization/TGV decay png's/`.
3. Stitch the exported frames into a universal looping `.gif` using [`make_gif.py`](Quantum_Implementation/make_gif.py):
   ```bash
   python Quantum_Implementation/make_gif.py
   ```
   This outputs **`Para View Visualization/tgv_decay_animation.gif`**, ready for presentations, posters, and web browsers.

---

## Current State & NISQ Considerations

- **State Extraction / Tomography**: Full Quantum State Tomography on $n=11$ qubits requires measuring in $3^{11} = 177,147$ Pauli bases per time step, which is prohibitive on NISQ hardware. This solver utilizes a hybrid workflow where the quantum circuit is compiled and evaluated, while state propagation bridges into the classical pressure corrector.
- **Carleman Truncation**: Truncating at order 2 ($x \otimes x$) introduces an approximation error that is valid for small time increments $\Delta t$. Higher orders exponentially increase the qubit and Pauli string counts.
