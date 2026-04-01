import os
import numpy as np
from config import setup_workspace
from define_models import create_system
import argparse
from functools import partial
import optuna
import netket as nk

import optax
import flax
from optuna.trial import TrialState
from NQS_utils import create_checkpoint_callback,create_optimizer,load_checkpoint,load_resume_logger,setup_trial_dir,save_checkpoint
# Import the refactored factory functions

from config import get_params,get_optuna_args,ALL_NN_PARAMS

import matplotlib.pyplot as plt
import numpy as np

def plot_vmc_results(logger,plotname):
    """
    Plots the Energy Mean and Variance from a NetKet RuntimeLog.
    """
    # 1. Extract the history object for Energy
    energy_data = logger.data["Energy"]
    
    # 2. Extract iterations and convert complex values to real numbers
    iters = np.array(energy_data.iters)
    energy_mean = np.real(energy_data.Mean)
    energy_variance = np.real(energy_data.Variance)
    
    # 3. Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Energy
    ax1.plot(iters, energy_mean, color='#1f77b4', linewidth=1.5, label='Energy Mean')
    ax1.set_ylabel('Energy (Real Part)')
    ax1.set_title('VMC Optimization: Stitched Resume Data')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot Variance
    ax2.plot(iters, energy_variance, color='#d62728', linewidth=1.5, label='Energy Variance')
    ax2.set_xlabel('Iterations (Steps)')
    ax2.set_ylabel('Variance')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plotname)
    plt.show()

# Add this to the very end of your script:
# plot_vmc_results(logger)

def main():
    # Verify setup
    # 1. Get global parameters
    params = get_params()
    optuna_args=get_optuna_args()
    # 2. Get the parameter of the file

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrial", type=int, default=1)
    trial_spec, _ = parser.parse_known_args()
    ntrial=trial_spec.ntrial

    
    # 2. Create the physical system and models dynamically
    hi, ham, model, cost_func, models_name = create_system(params)
    
    # 3. Setup the workspace (MPI-safe directory and DB creation)
    NN_params = ALL_NN_PARAMS[params["architecture"]]
    working_dir, study_name, storage = setup_workspace(
        params,optuna_args,NN_params, params["ROOT_DIR"], models_name
    )
    
    # 4. Load the Optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)
   
    n_trials_done = len(study.trials)
    n_trials_completed = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    if ntrial> len(n_trials_completed):
        print("OPERATION CANNOT BE DONE")
        exit()
    else:
        trial_idx = np.argsort(np.array([x.value for x in n_trials_completed]))[ntrial]
        trial_number = n_trials_completed[trial_idx].number
        print("RESUMMING TRIAL #",trial_number)

    trial_dir = setup_trial_dir(working_dir,trial_number)
    # 5. Initialize variational state
    best_params=n_trials_completed[trial_idx].params
    print(best_params)
    optimizer,diag_shift = create_optimizer(optuna_args.niter,optuna_args.warmup, optuna_args.clipping, optuna_args.dynamic_hyp,best_params)
    
    # 3. Setup Preconditioner (Stochastic Reconfiguration)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=params["holomorphic"])
    
    # 4. Setup Native Variational State and Driver
    vstate = nk.vqs.FullSumState(hi, model)
    driver = nk.driver.VMC(
        ham, 
        optimizer, 
        variational_state=vstate, 
        preconditioner=sr
    )
    
    # SAFELY find files without glob
    

    ckpt_files = []

    
    
    if os.path.exists(trial_dir):
        ckpt_files = [
            os.path.join(trial_dir, f) for f in os.listdir(trial_dir) 
                if f.startswith("checkpoint_variables_") and f.endswith(".mpack")
            ]

    if ckpt_files:
        
        latest_ckpt = max(ckpt_files, key=lambda f: int(os.path.basename(f).split('_')[2].split('.')[0]))
        start_step = int(os.path.basename(latest_ckpt).split('_')[2].split('.')[0])

        driver._step_count = start_step
        v_dict,o_dict = load_checkpoint(trial_dir, driver, start_step) 

        vstate.variables = v_dict
        driver._optimizer_state = o_dict
        
        print(f"🔄 Resumed safely from {os.path.basename(latest_ckpt)}")


    # Replace this line:
    # logger = nk.logging.RuntimeLog.deserialize(trial_dir+f"/run_resume_{n_iter}")

    # With this line:

    log_path = trial_dir + f"/run_resume_{start_step}.json"  # Make sure the extension matches!
    logger = load_resume_logger(log_path)
    log = nk.logging.RuntimeLog()
    log = logger
    #log_file = trial_dir+f"/run_resume_{NR}"
    #log_file = os.path.join(ckpt_dir,log_file)
    #logger = nk.logging.JsonLog(log_file, mode="w")

    # 12. Run the optimization

    
    checkpoints = np.unique(np.geomspace(start_step,params["NR"], num=optuna_args.reports, dtype=int))
    steps_to_run = np.diff(checkpoints, prepend=0)

    callback= create_checkpoint_callback(trial_dir,checkpoints,optuna_args.reports)
    
    n_run = params["NR"]  - start_step

    if n_run > 0:
        print(f"🚀 Starting VMC optimization for {n_run} steps...")
            
        driver.run(
            n_iter=n_run,
            out=logger,
            callback=[callback]
        )
    else:
        print("Number of steps already taken")

    save_checkpoint(trial_dir, driver,params["NR"])

    logger.serialize(trial_dir+f'/run_resume_{params["NR"]}')

    plot_vmc_results(logger,trial_dir+f'/run_resume_{params["NR"]}.pdf')
    
if __name__ == "__main__":
    main()
