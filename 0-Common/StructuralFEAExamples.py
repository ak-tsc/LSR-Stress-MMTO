import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
from PyTOImports import *
import scipy.sparse as spy_sprs
from scipy.sparse import lil_matrix

def getStructuralFEAProblem(
        nDOFDesired: int = 30000,
        **kwargs):
    """
    Main entry point.
    Returns:
        mesh
        mat_prop
        bc
        elem_body_force
    """
    return createStructuralProblem(nDOFDesired=nDOFDesired, **kwargs)

# =====================================================
# USER INPUTS
# =====================================================

# STL geometry
# Support locations
# Load locations
# Non-design regions
# Material properties

# Modify these sections to define a new problem.

def createStructuralProblem(nDOFDesired: int = 30000, total_load: float = 1000.0):

    stl_file = os.path.join(script_dir, './Models/RectangularBeam/RectangularBeam.STL')

    nElemsDesired = nDOFDesired / 3

    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)

    mesh.createEdofMatStructural()

    p = mesh.node_xyz

    x = p[:, 0]
    y = p[:, 1]
#    z = p[:, 2]

    xmin_nodes = x.min()
    xmax_nodes = x.max()
    ymax_nodes = y.max()

    tol = 0.5 * np.min(mesh.elem_size)

    # ---------------------------------------
    # Fixed left face
    # ---------------------------------------

    left_nodes = np.where(np.abs(x - xmin_nodes) < tol)[0]

    # ---------------------------------------
    # Fixed right face
    # ---------------------------------------

    right_nodes = np.where(np.abs(x - xmax_nodes) < tol)[0]

    fixed_nodes = np.union1d(left_nodes, right_nodes)

    fixed_dofs = np.array([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]).flatten().astype(int)

    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)

    # ---------------------------------------
    # Non-Design Elements (Do not remove these elements during optimization)
    # ---------------------------------------
    ec = np.asarray(mesh.elem_centers)

    xec = ec[:,0]
    yec = ec[:,1]

    xmin_elems = xec.min()
    xmax_elems = xec.max()

    attachment_length = 100.0   # mm

    left_nd = np.where(xec < xmin_elems + attachment_length)[0]
    right_nd = np.where(xec > xmax_elems - attachment_length)[0]

    mesh.nonDesignElems = np.unique(np.concatenate([left_nd, right_nd]) ).astype(int)
    print(
        f"Non-design elements: "
        f"{len(mesh.nonDesignElems)} / {mesh.num_elems}"
    )
    
    # plotting
    mesh.node_indices[fixed_nodes, 3] = 1

    # ---------------------------------------
    # Uniform downward load on top surface
    # ---------------------------------------

    xmid = 0.5*(xmin_nodes+xmax_nodes)

    load_nodes = np.where((np.abs(y-ymax_nodes) < tol) & (np.abs(x-xmid) < 250.0) )[0]


    force = np.zeros(3 * mesh.num_nodes)

    load_dofs = 3 * load_nodes + 1
    
    if len(load_nodes) == 0:
        raise ValueError(
            "No load nodes found. Check load region definition."
        )

    force[load_dofs] = (-total_load / len(load_nodes))

    mesh.node_indices[load_nodes, 3] = 2

    # ---------------------------------------
    # Boundary conditions
    # ---------------------------------------

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    # Placeholder material
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)

    elem_body_force = None

    print(
        f"[RectangularBeam] elems={mesh.num_elems} "
        f"elem_size={np.round(mesh.elem_size,3)} "
        f"fixed_nodes={len(fixed_nodes)} "
        f"load_nodes={len(load_nodes)}"
    )

    return mesh, mat_prop, bc, elem_body_force