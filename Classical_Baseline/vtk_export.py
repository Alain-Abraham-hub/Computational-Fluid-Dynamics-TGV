"""
vtk_export.py
-------------
Pure-Python zero-dependency VTK exporter for the Classical Baseline 2D Taylor-Green Vortex solver.
Generates VTK Legacy ASCII Structured Points and a master ParaView (.pvd) time collection.
"""

import os
import numpy as np


def export_classical_to_vtk(
    output_dir: str,
    frame_idx: int,
    time_val: float,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    dx: float,
    dy: float,
    Uc: float = 0.0,
    Vc: float = 0.0,
    vorticity: np.ndarray = None,
):
    """
    Exports a single 2D snapshot (indexed as [ix, iy]) to a VTK file for ParaView.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"classical_tgv_{frame_idx:03d}.vtk"
    filepath = os.path.join(output_dir, filename)

    nx, ny = u.shape
    num_points = nx * ny

    # Convert from 'ij' indexing (x=row, y=col) so that x varies fastest for VTK
    u_vortex = (u - Uc).T.ravel()
    v_vortex = (v - Vc).T.ravel()
    u_total = u.T.ravel()
    v_total = v.T.ravel()
    p_flat = p.T.ravel()

    # Kinetic energy in the vortex frame (matches stationary TGV benchmark)
    ke_flat = 0.5 * (u_vortex**2 + v_vortex**2)

    with open(filepath, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Classical TGV step={frame_idx} time={time_val:.6f}s\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {dx:.6e} {dy:.6e} 1.0\n")
        f.write(f"POINT_DATA {num_points}\n")

        # Primary velocity (vortex-frame for clean 4-vortex visualization)
        f.write("VECTORS velocity double\n")
        for ui, vi in zip(u_vortex, v_vortex):
            f.write(f"{ui:.6e} {vi:.6e} 0.000000e+00\n")

        # Convecting lab-frame velocity
        f.write("VECTORS velocity_convecting double\n")
        for ui, vi in zip(u_total, v_total):
            f.write(f"{ui:.6e} {vi:.6e} 0.000000e+00\n")

        # Kinetic Energy scalar
        f.write("SCALARS kinetic_energy double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for kei in ke_flat:
            f.write(f"{kei:.6e}\n")

        # Pressure scalar
        f.write("SCALARS pressure double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for pi in p_flat:
            f.write(f"{pi:.6e}\n")

        # Vorticity scalar (if provided)
        if vorticity is not None:
            vort_flat = vorticity.T.ravel()
            f.write("SCALARS vorticity double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for wi in vort_flat:
                f.write(f"{wi:.6e}\n")

    return filename


def write_classical_pvd(output_dir: str, times: list, pvd_filename: str = "classical_tgv_series.pvd"):
    """
    Writes a master ParaView (.pvd) collection file linking all VTK frames with physical timestamps.
    """
    os.makedirs(output_dir, exist_ok=True)
    pvd_path = os.path.join(output_dir, pvd_filename)

    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for idx, t in enumerate(times):
            vtk_name = f"classical_tgv_{idx:03d}.vtk"
            f.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vtk_name}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    return pvd_path
