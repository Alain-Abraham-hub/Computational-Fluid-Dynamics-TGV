"""
vtk_export.py
-------------
Pure-Python zero-dependency VTK exporter for Taylor-Green Vortex simulations.
Generates VTK Legacy ASCII Structured Points and a master ParaView (.pvd) time collection.
"""

import os
import numpy as np


def export_tgv_to_vtk(output_dir, step, time_val, u, v, nx, ny, L, p=None):
    """
    Exports a single time-step flow field to a standard VTK file for ParaView.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"tgv_step_{step:03d}.vtk"
    filepath = os.path.join(output_dir, filename)
    
    dx = L / nx
    dy = L / ny
    
    u_flat = np.asarray(u).ravel()
    v_flat = np.asarray(v).ravel()
    ke_flat = 0.5 * (u_flat**2 + v_flat**2)
    
    num_points = nx * ny
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Taylor-Green Vortex step={step} time={time_val:.6f}s\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {dx:.6e} {dy:.6e} 1.0\n")
        f.write(f"POINT_DATA {num_points}\n")
        
        # Velocity vectors: (u, v, w=0)
        f.write("VECTORS velocity double\n")
        for ui, vi in zip(u_flat, v_flat):
            f.write(f"{ui:.6e} {vi:.6e} 0.000000e+00\n")
            
        # Kinetic Energy scalar
        f.write("SCALARS kinetic_energy double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for kei in ke_flat:
            f.write(f"{kei:.6e}\n")
            
        # Pressure scalar (if available)
        if p is not None:
            p_flat = np.asarray(p).ravel()
            f.write("SCALARS pressure double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for pi in p_flat:
                f.write(f"{pi:.6e}\n")
                
    return filename


def write_pvd_collection(output_dir, time_history, pvd_filename="tgv_series.pvd"):
    """
    Writes a master ParaView (.pvd) collection file linking all VTK files with physical times.
    """
    os.makedirs(output_dir, exist_ok=True)
    pvd_path = os.path.join(output_dir, pvd_filename)
    
    with open(pvd_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for step, t in enumerate(time_history):
            vtk_name = f"tgv_step_{step:03d}.vtk"
            f.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vtk_name}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
        
    return pvd_path

