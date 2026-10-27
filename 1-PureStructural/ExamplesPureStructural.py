import sys
import os
import numpy as np
# Add the 0-Common folder to Python path
common_path = os.path.join(os.path.dirname(__file__), '..', '0-Common')
sys.path.insert(0, os.path.abspath(common_path))
from materialColors import material_colors # type: ignore[reportMissingImports]
# Now you can import normally
from StructuralFEAExamples import getStructuralFEAProblem
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
    topopt_material_model._PNORM_EXPONENT = 6   # default; Xie branches override to 10
    vae_params = VAEParams()
    kwargs['total_load'] = 1000.0
    
    to_params.Comment = (
    "User-defined STL optimization problem"
    )

    to_params.ExtrudeZ = False

    to_params.RelativeFilterRadius = 5.0

    to_params.nDOFDesired = (
        12000 if nDOFDesired is None else nDOFDesired
    )

    to_params.MaterialsExcelFile = (
        './1-PureStructural/'
        'MaterialDataPureStructural/'
        '6Materials_AF_3Attr.xlsx'
    )

    #
    # Objective
    #
    to_params.Objective = (
        TO_QOI.COMPLIANCE,
        None
    )
    # to_params.Objective = (
    #     TO_QOI.PNORM_STRESS,
    #     None
    # )

    #
    # Mass constraint
    #
    to_params.Constraints = [
        (
            TO_QOI.MASS,
            None,
            1.75e6
        )
    ]

    vae_params.latentDim = 2
    vae_params.learningRate = 1e-4
    vae_params.vae_hiddenDim = 50
    vae_params.maxAttributeErrorPercent = 0.1
    vae_params.klFactor = 1e-4
    vae_params.numEpochs = 100000


    mesh, mat_prop, bc, elem_body_force = getStructuralFEAProblem(
        nDOFDesired=to_params.nDOFDesired,
        **kwargs
    )


    # Add  elements to keep
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