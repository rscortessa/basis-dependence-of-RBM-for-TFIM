import netket as nk
import numpy as np
import jax.numpy as jnp


# Assuming these are properly defined in your Methods folder
from hams import rotated_IsingModel
from optuna_setup import objective

def create_system(params):
    """
    Factory function to instantiate the Hilbert space, Hamiltonian, 
    variational model, and cost function based on the provided parameters.
    """
    # 1. Define Hilbert Space
    hi = nk.hilbert.Spin(s=1/2, N=params["L"] * params["W"], inverted_ordering=True)
    
    # 2. Define Hamiltonian, Cost Function, and string identifier
    models_name = params["model"]
    
    if params["model"] == "QIM":
        ham = rotated_IsingModel(
            params["angle"] * (2 * np.pi /360),
            params["g"] ,
            params["L"],
            hi,
            pbc=params["pbc"]
        )
        
        cost_func = objective
        
    else:
        raise ValueError(f"The Hamiltonian model {params['model']} does not exist.")

    # 3. Define the Neural Network Architecture
    if params["architecture"] == "RBM_COMPLEX":
        model = nk.models.RBM(alpha=params["NN"], param_dtype=complex)
    else:
        raise ValueError(f"Architecture {params['architecture']} not recognized.")

    return hi, ham, model, cost_func, models_name
