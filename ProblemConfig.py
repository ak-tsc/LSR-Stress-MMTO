
# This file is intended to be the primary user interface
# for defining a topology optimization problem.
# No changes to the optimization source code should be
# required for standard problems.
#
# ============================================================
# USER WORKFLOW
# ============================================================
#
# 1. Choose an STL file, DOF Required
# 2. Define supports
# 3. Define loads
# 4. Define non-design regions
# 5. Choose objective
# 6. Choose constraints
# 7. Select material database
# 8. Select filter and optimization settings (Advanced)
# 9. Select VAE parameters (Advanced)
# 10. Run Tutorial.ipynb


import os
import numpy as np

from PyTOImports import *
script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# FEA PROBLEM DEFINITION
# ============================================================
STL_FILE = os.path.join(script_dir, './0-Common/Models/RectangularBeam.STL')
NDOF_DESIRED = 12000

COMMENT = "Rectangular Beam Optimization Problem"

# ============================================================
# FEA MODEL CREATION
# ============================================================

def createStructuralProblem():

    # ==========CREATE MESH ===================================

    nElemsDesired = NDOF_DESIRED / 3
    mesh = hex_mesher.HexMesher()
    mesh.createMeshFromSTLFile(STL_FILE, nElemsDesired=nElemsDesired) # STL-based mesh generation

    mesh.createEdofMatStructural() # Structural connectivity matrix (EDOF)

    # ==========EXTRACT NODE COORDINATES ===================================

    p = mesh.node_xyz 
    x = p[:, 0]
    y = p[:, 1]
#    z = p[:, 2] # Uncomment for 3D selection operations

    xmin_nodes = x.min() # Mesh dimensions
    xmax_nodes = x.max()

    ymax_nodes = y.max()
#    ymin_nodes = y.min()

#    zmin_nodes = z.min()
#    zmax_nodes = z.max()

    # ==========EXTRACT ELEMENT DATA =================================

    ec = np.asarray(mesh.elem_centers) # Element center coordinates

    xec = ec[:,0]
#    yec = ec[:,1]
#    zec = ec[:,2]

    xmin_elems = xec.min()
    xmax_elems = xec.max()
#    ymin_elems = yec.min()
#    ymax_elems = yec.max()

#    dx, dy, dz = mesh.elem_size     # Element dimensions

    tol = 0.5 * np.min(mesh.elem_size) # Node selection tolerance

    # ========== NODE SELECTION EXAMPLES===========================

    left_nodes = np.where(np.abs(x - xmin_nodes) < tol)[0]  # Left face
    right_nodes = np.where(np.abs(x - xmax_nodes) < tol)[0] # Right face
    
#    front_nodes = np.where(np.abs(z - zmax_nodes) < tol)[0] # Front face
#    back_nodes = np.where(np.abs(z - zmin_nodes) < tol)[0] # Back face

    
#    nodes = np.where((x > x1) & (x < x2) & (y > y1) & (y < y2))[0] # Box region Node Selection

    
#    r = np.sqrt((x-x0)**2 + (y-y0)**2) # Cylindrical region Node Selection  
#    nodes = np.where(r < radius)[0]

    
#    r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2) # Spherical region Node Selection
#    nodes = np.where(r < radius)[0]


    # ==========DEFINE SUPPORTS ===================================
   
    # DOF numbering (IMPORTANT): The dofs for each node are numbered consecutively as follows: ux_0, uy0, uz0, ux_1, uy_1, uz_1, ... 
    # Therefore fixing 30 and 60 corresponds to fixing the ux for node 10 and 20 respectively. Similar logic applies for uy and uz.
    # fixed_dofs is a list of DOFs that are fixed (constrained) in the analysis. 
    # fixed_dofs =[0,1,2,          # node 0 fully fixed
    #              3,4,5,          # node 1 fully fixed
    #              30,45,60,       # x rollers at node 10, 15 and 20
    #              106]            # uy of node 35 fixed
    #
    # ux = 3*node
    # uy = 3*node + 1
    # uz = 3*node + 2
   
    # Combine support regions
    # support_nodes = np.union1d(region1, region2)

    fixed_nodes = np.union1d(left_nodes, right_nodes) # # Combine node regions

    fixed_dofs = np.array([3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]).flatten().astype(int) # All three displacements fixed

    # X Roller
#    fixed_dofs = np.array([ 3*nodes]) # X Roller ux=0

    # Y Roller uy=0
#    fixed_dofs = np.array([ 3*nodes + 1]) # Y Roller uy=0

    # Single node support
#    support_node = np.argmin((x-x0)**2 +(y-y0)**2)
#    fixed_dofs = np.array([ 3*support_node, 3*support_node + 1, 3*support_node + 2])

    dirichlet_values = np.zeros_like(fixed_dofs, dtype=float) # Imposed Displacement


    # ==========DEFINE NON-DESIGN ELEMENTS ===================================
    # Elements listed in mesh.nonDesignElems will be
    # preserved during topology optimization and cannot
    # be removed.

    attachment_length = 100.0   # mm

    left_nd = np.where(xec < xmin_elems + attachment_length)[0]
    right_nd = np.where(xec > xmax_elems - attachment_length)[0]

    mesh.nonDesignElems = np.unique(np.concatenate([left_nd, right_nd]) ).astype(int)
    
    # Elements near xmin
    # elems = np.where(
    #     xec < xmin_elems + distance
    # )[0]

    # Elements inside box
    # elems = np.where(
    #     (xec > x1) &
    #     (xec < x2) &
    #     (yec > y1) &
    #     (yec < y2)
    # )[0]

    # Cylindrical region
    # r = np.sqrt(
    #     (xec-x0)**2 +
    #     (yec-y0)**2
    # )
    #
    # elems = np.where(r < radius)[0]

    # Multiple regions
    # elems = np.union1d(
    #     region1,
    #     region2
    # )

    # ==========DEFINE LOADS ================================================

    xmid = 0.5*(xmin_nodes+xmax_nodes)
    load_nodes = np.where((np.abs(y-ymax_nodes) < tol) & (np.abs(x-xmid) < 250.0) )[0]

    # Global nodal force vector
    #
    # Fx = force[3*node]
    # Fy = force[3*node + 1]
    # Fz = force[3*node + 2]

    force = np.zeros(3 * mesh.num_nodes)
    
#    load_dofs = 3 * load_nodes # Distributed X load
    load_dofs = 3 * load_nodes + 1 # Distributed Y load
#    load_dofs = 3 * load_nodes + 2 # Distributed Z load

#    force[3*node_id + 1] = -1000 # Point Y Load

    TOTAL_LOAD = 1000.0

    if len(load_nodes) == 0:
        raise ValueError(
            "No load nodes found. Check load region definition."
        )

    force[load_dofs] = (-TOTAL_LOAD / len(load_nodes)) # Distributed Y load


    # ==========VISUALIZATION MARKERS ===================================

    mesh.node_indices[fixed_nodes, 3] = 1 # Support nodes
    mesh.node_indices[load_nodes, 3] = 2 # Load nodes


    # ==========CREATE BOUNDARY CONDITION GROUP ===================================

    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)

    # Placeholder material - Will be overwritten by the VAE
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0, yield_strength=1.0)

    elem_body_force = None

    print(
        f"[RectangularBeam] elems={mesh.num_elems} "
        f"elem_size={np.round(mesh.elem_size,3)} "
        f"fixed_nodes={len(fixed_nodes)} "
        f"load_nodes={len(load_nodes)}"
    )

    return mesh, mat_prop, bc, elem_body_force



# ============================================================
# OBJECTIVES
# ============================================================


# ========== COMPLIANCE MINIMIZATION =======================================================
# Minimize structural compliance (equivalent to maximizing stiffness)


OBJECTIVE = (TO_QOI.COMPLIANCE, None)


# ========== P-NORM VON MISES STRESS MINIMIZATION ==========================================
# Minimize p-norm aggregated von Mises stress. Requires: PNORM_EXPONENT


#OBJECTIVE = (TO_QOI.PNORM_STRESS, None)
#PNORM_EXPONENT = 6

# ========== P-NORM FAILURE FACTOR MINIMIZATION =============================================
# Minimize p-norm aggregated failure factor: FailureFactor = Stress / YieldStrength
# FailureFactor < 1 is safe. Requires: Yield_Strength material property and PNORM_EXPONENT


#OBJECTIVE = ( TO_QOI.PNORM_STRESS, "FAILURE_FACTOR")
#PNORM_EXPONENT = 6


# ========== MASS MINIMIZATION ===============================================================
# Minimize total mass. Requires: Density material property


# OBJECTIVE = ( TO_QOI.MASS, None)


# ========== AUGMENTED-LAGRANGIAN LOCAL STRESS OBJECTIVE ======================================
# Mentioned in SensitivitiesPureStructural.py: 
# Uses: weighted local failure factor,  Powell-Hestenes-Rockafellar AL formulation. TO_QOI.MASS_AL_LOCAL_STRESS
# Check implementation before use.




# ============================================================
# CONSTRAINTS
# ============================================================


# ========== MASS CONSTRAINT =======================================================
# Total mass must remain below the specified limit.
# Requires: Density material property
# Constraint: TotalMass / MASS_LIMIT - 1 <= 0


CONSTRAINTS = [(TO_QOI.MASS, None, 1.75e6)]


# ========== COMPLIANCE CONSTRAINT ================================================
# Structural compliance must remain below the specified limit.
# Constraint: Compliance / COMPLIANCE_LIMIT - 1 <= 0


#CONSTRAINTS = [(TO_QOI.COMPLIANCE, None, COMPLIANCE_LIMIT)]


# ========== GLOBAL VOLUME FRACTION CONSTRAINT ====================================
# Limits the average density variable x over all elements.
# Constraint: mean(x) <= VOLFRAC_LIMIT


#CONSTRAINTS = [(TO_QOI.VOLUME_FRACTION, None, 0.30)]


# ========== MATERIAL-SPECIFIC VOLUME FRACTION CONSTRAINT =========================
# Limits the volume fraction of a particular material.
# Uses softmax latent-space material assignment.
# Requires: matEncoder.training_latents
# Parameters:
#     material_id : material index
#     tau         : softmax temperature
# Example: Material 0 <= 12% volume fraction


#CONSTRAINTS = [( TO_QOI.VOLUME_FRACTION, {"material_id": 0, "tau": 0.5}, 0.12)]


# ========== COST CONSTRAINT ======================================================
# Limits total material cost.
# Requires:
#     Density material property
#     Cost material property
# Constraint: TotalCost / COST_LIMIT - 1 <= 0


#CONSTRAINTS = [(TO_QOI.COST, None, COST_LIMIT)]


# ========== MAX CRITICALITY CONSTRAINT ===========================================
# Limits p-norm aggregated material criticality.
# Requires:
#     Criticality material property
#     PNORM_EXPONENT


#CONSTRAINTS = [(TO_QOI.MAX_CRITICALITY, None, CRITICALITY_LIMIT)]



# ========== MEAN CRITICALITY CONSTRAINT ==========================================
# Limits average material criticality.
# Requires: Criticality material property


#CONSTRAINTS = [(TO_QOI.MEAN_CRITICALITY, None, CRITICALITY_LIMIT)]


# ========== STRESS FAILURE FACTOR CONSTRAINT =====================================
# Limits p-norm aggregated failure factor: FailureFactor = Stress / YieldStrength
# FailureFactor < 1 is safe.
# Requires:
#     Yield_Strength material property
#     PNORM_EXPONENT


#CONSTRAINTS = [(TO_QOI.STRESS_FAILURE_FACTOR, None, 1.0)]
#PNORM_EXPONENT = 6


# ============================================================
# MATERIAL DATABASE
# ============================================================

MATERIALS_EXCEL_FILE = (
        './1-PureStructural/'
        'MaterialDataPureStructural/'
        '6Materials_AF_3Attr.xlsx'
    )


# ============================================================
# FILTER & OPTIMIZATION SETTINGS
# ============================================================

RELATIVE_FILTER_RADIUS = 5.0
MATERIAL_FILTER_RADIUS = 5.0
MAX_ITERATIONS = 300

BETA_SCHEME = "PAPER" # "PAPER" or "PAPER_BM1"

PNORM_EXPONENT = 6


# ============================================================
# VAE SETTINGS
# ============================================================

VAE_LATENT_DIM = 2
VAE_LEARNING_RATE = 1e-4
VAE_HIDDEN_DIM = 50
VAE_MAX_ATTRIBUTE_ERROR_PERCENT = 0.1
VAE_KL_FACTOR = 1e-4
VAE_NUM_EPOCHS = 100000
