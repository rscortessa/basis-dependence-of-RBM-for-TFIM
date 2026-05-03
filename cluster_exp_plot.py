import os
import numpy as np
from config import setup_workspace
from define_models import create_system
import argparse
from functools import partial
import optuna
import netket as nk
import jax

import optax
import flax
from optuna.trial import TrialState
from NQS_utils import create_checkpoint_callback,create_optimizer,load_checkpoint,load_resume_logger,setup_trial_dir,save_checkpoint
# Import the refactored factory functions

from config import get_params,get_optuna_args,ALL_NN_PARAMS
from cluster_exp import Fast_Hadamard
from hams import rotated_IsingModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Verify setup
    # 1. Get global parameters
    params = get_params()
    optuna_args=get_optuna_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--NNs",nargs='+',type=float,default=[1.0])
    trial_spec, _ = parser.parse_known_args()
    NNs=trial_spec.NNs
    print("Plotting for hidden unit densities ",NNs)

    ntrial=0
    fontsize=15
    eps=1e-16
    
   
    
    hi = nk.hilbert.Spin(s=1/2, N=params["L"] * params["W"], inverted_ordering=True)
    H=rotated_IsingModel(params["angle"]*np.pi/180,params["g"],params["L"],hi,pbc=params["pbc"])
    eigvals,eigvecs = np.linalg.eigh(H.to_dense())

    Psi = eigvecs[:,0]
    log_Psi = np.log(Psi+eps+1j*eps)
    log_Psi_ht = Fast_Hadamard(log_Psi)
    sorting_coeffs = np.argsort(np.abs(log_Psi_ht))[::-1]

    inf_exact = np.ones(2**params['L'])
    for n_terms in range(2**params['L']):
        aux_log = log_Psi_ht.copy()
        aux_log[sorting_coeffs[n_terms+1:-1]] = 0.0
        aux_psi = np.exp(Fast_Hadamard(aux_log)*2**params["L"])
        aux_psi = aux_psi / np.linalg.norm(aux_psi)        
        inf_exact[n_terms] = 1 -np.linalg.norm(Psi@np.conjugate(aux_psi))**2


    
    inf_trunc_wfs = np.ones((len(NNs),2**params['L']))

    # 3. Setup the workspace (MPI-safe directory and DB creation)
    NN_params = ALL_NN_PARAMS[params["architecture"]]
    
    for idx_nn,nn in enumerate(NNs):

        params['NN'] = nn
        model = nk.models.RBM(alpha=params["NN"],param_dtype=complex)
        key = jax.random.PRNGKey(seed=0)
        all_states = hi.all_states()
        v_dict = model.init(key,all_states)

        
        working_dir, study_name, storage = setup_workspace(params,optuna_args,NN_params, params["ROOT_DIR"], params["model"])
            
        # 4. Load the Optuna study
        print(storage)
        study = optuna.load_study(study_name=study_name, storage=storage)
        study = optuna.load_study(study_name=optuna.get_all_study_names(storage=storage)[0], storage=storage)
        n_trials_done = len(study.trials)
        n_trials_completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        if ntrial> len(n_trials_completed):
            print("OPERATION CANNOT BE DONE")
            exit()
        else:
            trial_idx = np.argsort(np.array([x.value for x in n_trials_completed]))[ntrial]
            trial_number = n_trials_completed[trial_idx].number 
            print("TRIAL involved #",trial_number)
          
            trial_dir = setup_trial_dir(working_dir,trial_number)
            v_path = os.path.join(trial_dir, f"checkpoint_variables_{params["NR"]}.mpack")
    
            with open(v_path, "rb") as f:
                v_dict = flax.serialization.from_bytes(v_dict, f.read())

            log_Psi_RBM = np.array(model.apply(v_dict,all_states))
            log_Psi_RBM_ht = Fast_Hadamard(log_Psi_RBM)
            
            for n_terms in range(2**params['L']):
                aux_log = log_Psi_RBM_ht.copy()
                aux_log[sorting_coeffs[n_terms+1:]] = 0.0
                aux_psi = np.exp(Fast_Hadamard(aux_log)*2**params["L"])
                aux_psi = aux_psi / np.linalg.norm(aux_psi)
                
                inf_trunc_wfs[idx_nn,n_terms] = 1 -np.linalg.norm(Psi@np.conjugate(aux_psi))**2
            print('Finished math for alpha=',nn)
            
            
    angle=params["angle"]*np.pi/180
    x = np.arange(2**params['L'])

    color=["orange","magenta","blue","darksalmon",'green','brown','red','peru']
    
    plt.figure()
    
    plt.title(rf"Cluster Expansion  $L={params["L"]}\;g={params["g"]}\;\theta={np.round(params["angle"]/180,2)}\pi \;\alpha={params["NN"]}$",fontsize=fontsize)
    

    for idx_nn,nn in enumerate(NNs):
        
        plt.scatter(x,inf_trunc_wfs[idx_nn],color=color[idx_nn],marker="*",label=r"$"+str(nn)+"$")    
        plt.axvline(x= nn*params["L"]**2+params["L"]*(1+nn),color=color[idx_nn])

    plt.plot(x,inf_exact,linestyle="dashed",c='black',label=r"exact")

    plt.xlabel(r" $N_{trunc}$",fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([10**(-10),1])
    plt.legend()
    plt.savefig(f"ce_g{params["g"]}L{params["L"]}angle{params["angle"]}.pdf")

if __name__ == "__main__":
    main()
