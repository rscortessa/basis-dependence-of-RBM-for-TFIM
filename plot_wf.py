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
from hams import rotated_IsingModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Verify setup
    # 1. Get global parameters
    params = get_params()
    optuna_args=get_optuna_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--psiplus", action="store_true")
    parser.add_argument("--psiminus", action="store_true")
    parser.add_argument("--angle", type=float,default=0.0)
    trial_spec, _ = parser.parse_known_args()
    psiplus=trial_spec.psiplus
    psiminus=trial_spec.psiminus
    params["angle"]=trial_spec.angle
    ntrial=0
    fontsize=15

    eps=1e-16
    
    model = nk.models.RBM(alpha=params["NN"],param_dtype=complex)
    key = jax.random.PRNGKey(seed=0)
    
    hi = nk.hilbert.Spin(s=1/2, N=params["L"] * params["W"], inverted_ordering=True)
    all_states = hi.all_states()
    v_dict = model.init(key,all_states)
    
    H=rotated_IsingModel(params["angle"]*np.pi/180,params["g"],params["L"],hi,pbc=params["pbc"])
    eigvals,eigvecs = np.linalg.eigh(H.to_dense())
    Psi0_WF = eigvecs[:,0]
    Psi1_WF = eigvecs[:,1]

    Psi0_WF[np.abs(Psi0_WF)<eps]=0.0
    Psi1_WF[np.abs(Psi1_WF)<eps]=0.0

    Psi0_WF = Psi0_WF * ( np.sign(Psi0_WF[0]) + (1-np.sign(Psi0_WF[0])) * (1+np.sign(Psi0_WF[0])) ) * np.sign(Psi0_WF[1]) )
    Psi1_WF = Psi1_WF * ( np.sign(Psi1_WF[0]) + (1-np.sign(Psi1_WF[0])) * (1+np.sign(Psi1_WF[0])) ) * np.sign(Psi1_WF[1]) )

    if psiplus:
        Psiplus_WF= (Psi0_WF+Psi1_WF)/np.sqrt(2)
    if psiminus:
        Psiminus_WF= (Psi0_WF-Psi1_WF)/np.sqrt(2)

    sorting = np.argsort(Psi0_WF**2)[::-1]
    print(Psi0_WF)
    # 3. Setup the workspace (MPI-safe directory and DB creation)
    NN_params = ALL_NN_PARAMS[params["architecture"]]
    working_dir, study_name, storage = setup_workspace(params,optuna_args,NN_params, params["ROOT_DIR"], params["model"])
            
    # 4. Load the Optuna study

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
        log_psi = model.apply(v_dict,all_states)
        psi = np.exp(log_psi)
        psi = psi/np.linalg.norm(psi)
        psi = psi*np.exp(-1j*np.angle(psi[0]))
        
            
            
    angle=params["angle"]*np.pi/180
    x = np.arange(1,1025,1)
    color=["orange","magenta","blue","darksalmon",'green','brown','red','peru']
    
    plt.figure()
    plt.title(rf"$L={params["L"]}\;g={params["g"]}\;\alpha={params["NN"]}$",fontsize=fontsize)
    
  
    color_psi0 = [color[int(psi_amp<=0)] for psi_amp in Psi0_WF[sorting]]
    color_psi = [color[int(psi_amp<0)+2] for psi_amp in psi[sorting]]

    if psiplus:
        color_psiplus = [color[int(psi_amp<0)+4] for psi_amp in Psiplus_WF[sorting]]
        plt.scatter(x,(Psiplus_WF*np.conjugate(Psiplus_WF))[sorting],c=color_psiplus,marker="^",label=r"$|\Psi_+(s)|^2$")
    if psiminus:
        color_psiminus = [color[int(psi_amp<0)+6] for psi_amp in Psiminus_WF[sorting]]
        plt.scatter(x,(Psiminus_WF*np.conjugate(Psiminus_WF))[sorting],c=color_psiminus,marker="^",label=r"$|\Psi_-(s)|^2$")

    plt.scatter(x,(Psi0_WF*np.conjugate(Psi0_WF))[sorting],c=color_psi0,marker="*",label=r"$|\Psi_0(s)|^2$")    

    plt.plot(x,(psi*np.conjugate(psi))[sorting],linestyle="dashed",c=color[2],label=r"$|\Psi_{RBM}(s)|^2$")
    
    plt.xlabel(r"label index $s$",fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([10**(-10),1])
    plt.legend()
    plt.savefig(f"square_amplitudes_g{params["g"]}L{params["L"]}angle{params["angle"]}.pdf")
    plt.show()
if __name__ == "__main__":
    main()
