import sys
import os
import numpy as np
# Add the 0-Common folder to Python path
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))
from materialColors import material_colors # type: ignore[reportMissingImports]
# Now you can import normally
from ProblemConfig import *

from PyTOImports import  TO_QOI, find_elements_with_fixedDOF # type: ignore
from mmto_common import MMTOParams # type: ignore
from dataclasses import dataclass, field
import topopt_material_model  # to override the shared p-norm exponent per problem
@dataclass(slots=True) # avoid accidental modification

class VAEParams: #default VAE parameters
    klFactor: float = 5e-6
    learningRate: float = 2e-6
    numEpochs: int = 200000 # max epochs
    vae_hiddenDim: int = 500
    latentDim: int = 2 # default latent dimension
    maxAttributeErrorPercent: float = 1 # termination criteria for VAE training

def getMMTOProblemPureStructural(nDOFDesired = None, **kwargs):

    to_params = MMTOParams()
    vae_params = VAEParams()
    topopt_material_model._PNORM_EXPONENT = PNORM_EXPONENT

    to_params.Comment = COMMENT
    to_params.RelativeFilterRadius = RELATIVE_FILTER_RADIUS
  
    to_params.nDOFDesired = (
        NDOF_DESIRED if nDOFDesired is None else nDOFDesired
    )
    to_params.MaterialsExcelFile = MATERIALS_EXCEL_FILE
    to_params.Objective = OBJECTIVE
    to_params.Constraints = CONSTRAINTS
    to_params.ExtrudeZ = False

    vae_params.latentDim = VAE_LATENT_DIM
    vae_params.learningRate = VAE_LEARNING_RATE
    vae_params.vae_hiddenDim = VAE_HIDDEN_DIM
    vae_params.maxAttributeErrorPercent = VAE_MAX_ATTRIBUTE_ERROR_PERCENT
    vae_params.klFactor = VAE_KL_FACTOR
    vae_params.numEpochs = VAE_NUM_EPOCHS

    mesh, mat_prop, bc, elem_body_force = createStructuralProblem()

    to_params.ElemsToKeep  = None # default value

    # Here we add additional parameters specific to the optimization problem
    if (to_params.KeepFixedElems):
        to_params.ElemsToKeep = find_elements_with_fixedDOF(mesh, bc,nDOFPerNode=3)
    # GE-style non-design: union the NonDesign collar into ElemsToKeep.
    # This only biases the DENSITY block toward solid (MainPureStructural.py:608);
    # the latent/material block is left free. Fires only for problems that set
    # mesh.nonDesignElems, so other examples are unaffected.
    if getattr(mesh, "nonDesignElems", None) is not None and len(mesh.nonDesignElems) > 0:
        if to_params.ElemsToKeep is None:
            to_params.ElemsToKeep = mesh.nonDesignElems
        else:
            to_params.ElemsToKeep = np.union1d(to_params.ElemsToKeep, mesh.nonDesignElems)

    return mesh,mat_prop, bc, elem_body_force, to_params, vae_params