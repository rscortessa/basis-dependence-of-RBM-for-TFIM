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
    trial_spec, _ = parser.parse_known_args()
    inf_psiplus=trial_spec.psiplus
    inf_psiminus=trial_spec.psiminus
    
    ntrials = 10
    fontsize=15
    model = nk.models.RBM(alpha=params["NN"],param_dtype=complex)
    key = jax.random.PRNGKey(seed=0)
    
    hi = nk.hilbert.Spin(s=1/2, N=params["L"] * params["W"], inverted_ordering=True)
    all_states = hi.all_states()
    v_dict = model.init(key,all_states)
    
    angles = np.arange(0,97.5,7.5)

    mean_sign_0 = np.zeros(len(angles))
    mean_sign_plus = np.zeros(len(angles))
    mean_sign_minus = np.zeros(len(angles))

    for angle_idx,angle in enumerate(angles):
        H=rotated_IsingModel(angle*np.pi/180,params["g"],params["L"],hi,pbc=params["pbc"])
        eigvals,eigvecs = np.linalg.eigh(H.to_dense())
        Psi0_WFs = eigvecs[:,0]
        Psi1_WFs = eigvecs[:,1]

        Psi0_WFs = Psi0_WFs*np.exp(-1j*np.angle(Psi0_WFs[0]))
        Psi1_WFs = Psi1_WFs*np.exp(-1j*np.angle(Psi1_WFs[0]))

        wf_plus = (Psi0_WFs+Psi1_WFs)/np.sqrt(2)
        wf_minus = (Psi0_WFs-Psi1_WFs)/np.sqrt(2)
        
        mean_sign_0[angle_idx] = np.abs((np.sign(Psi0_WFs)*Psi0_WFs)@np.conjugate(Psi0_WFs))
        mean_sign_plus[angle_idx] = np.abs((np.sign(wf_plus)*wf_plus)@np.conjugate(wf_plus))
        mean_sign_minus[angle_idx] = np.abs((np.sign(wf_minus)*wf_minus)@np.conjugate(wf_minus))
  
        
    # 3. Setup the workspace (MPI-safe directory and DB creation)
    NN_params = ALL_NN_PARAMS[params["architecture"]]

    RBM_mean_sign = np.zeros((len(angles),ntrials))

    for angle_idx,angle in enumerate(angles):
        params["angle"]=angle
        working_dir, study_name, storage = setup_workspace(params,optuna_args,NN_params, params["ROOT_DIR"], params["model"])
        
        
        # 4. Load the Optuna study

        study = optuna.load_study(study_name=study_name, storage=storage)

        study = optuna.load_study(study_name=optuna.get_all_study_names(storage=storage)[0], storage=storage)
        n_trials_done = len(study.trials)
        n_trials_completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        if ntrials> len(n_trials_completed):
            print("OPERATION CANNOT BE DONE")
            exit()
        else:
            trial_idxs = np.argsort(np.array([x.value for x in n_trials_completed]))[:ntrials]
            trial_numbers = [ n_trials_completed[trial_idx].number for trial_idx in trial_idxs ]
            print("TRIALS involved #",trial_numbers)
        for trial,trial_number in enumerate(trial_numbers):
            
            trial_dir = setup_trial_dir(working_dir,trial_number)
            v_path = os.path.join(trial_dir, f"checkpoint_variables_{params["NR"]}.mpack")
    
            with open(v_path, "rb") as f:
                v_dict = flax.serialization.from_bytes(v_dict, f.read())
            log_psi = model.apply(v_dict,all_states)
            psi = np.exp(log_psi)
            psi = psi/np.linalg.norm(psi)
            psi = psi*np.exp(-1j*np.angle(psi[0]))
            RBM_mean_sign[angle_idx,trial] = (np.sign(psi)*psi)@np.conjugate(psi)
            
            
    angles=angles*np.pi/180
    label_angles=[str(round(angle/np.pi,2)) for angle in angles]
    plt.figure()
    plt.title(rf"$L={params["L"]}\;g={params["g"]}\;\alpha={params["NN"]}$",fontsize=fontsize)
    color=["blue","red","orange","green"]
    for trial,trial_number in enumerate(trial_numbers):
        if trial==0:
            plt.plot(angles,RBM_mean_sign[:,trial],marker="*",color=color[3],label=r"$\Psi_{RBM}$")       
        else:
            plt.plot(angles,RBM_mean_sign[:,trial],marker="*",color=color[3])


    plt.plot(angles,mean_sign_0,label=r"$\Psi_0$",color=color[0])
    plt.plot(angles,mean_sign_plus,label=r"$\Psi_+$",color=color[1])
    plt.plot(angles,mean_sign_minus,label=r"$\Psi_-$",color=color[2])
  
    
    plt.xlabel(r"$\theta[\pi]$",fontsize=fontsize)
    plt.xticks(angles,label_angles)
    plt.legend()
    plt.savefig(f"mean_sign_g{params["g"]}L{params["L"]}.pdf")

if __name__ == "__main__":
    main()
