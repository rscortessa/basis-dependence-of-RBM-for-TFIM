import math
import numpy as np
import netket as nk
import optax
import optuna
import netket_fidelity as nkf
from netket_fidelity.infidelity import InfidelityOperator
import os
import flax
from NQS_utils import create_checkpoint_callback,setup_trial_dir,create_optimizer,save_checkpoint,generate_hyperparameters
import shutil

def setup_workspace(params,optuna_args,NN_params, ROOT_DIR, models_name):
    """
    Sets up the working directory and Optuna study, also it is 
    created the directories and initialized the database.
    """
    # 1. Build the identifier dynamically
    identifier = params["architecture"]
    for name in NN_params:
        identifier += name + str(params[name])

    # Construct the working directory string
    system_directory = (f"FULLSUM_{models_name}{identifier}"
                         f"L{params['L']}G{params['g']}"
                         f"ANGLE{params['angle']}{params['add']}")



    
    optuna_directory =   (f"CLIPPING{optuna_args.clipping*1}"
                         f"DYNAMIC{optuna_args.dynamic_hyp*1}"
                          f"NITER{optuna_args.niter}")

    working_directory = os.path.join(system_directory, optuna_directory)
    
    full_path = os.path.join(ROOT_DIR, working_directory)
    study_name =os.path.join(full_path,optuna_directory)
    
    storage = f"sqlite:///{study_name}.db"

    # 2. MPI SAFE I/O: Only the root node creates folders and DB!
    os.makedirs(full_path, exist_ok=True) # exist_ok=True is safer than try/except
        
    try:
        optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        print(f"✅ Created Optuna study: {study_name}")
    except optuna.exceptions.DuplicatedStudyError:
        print(f"ℹ️ Study {study_name} already exists.")
           
    print(f"✅ Workspace ready. Storage: {storage}")

    return full_path, study_name, storage


def objective(trial, model, L, hi, H, n_iter, holomorphic,working_dir,bounds,init_state=None,clipping=False,dynamic_hyp=False,WARMUP=None,num_reports=12,erase=True):
    
    """
    Optuna objective for finding the Ground State (Energy Minimization)

    * model is the N.N
    * H the operator to minimize
    * n_iter number of steps on which decaying of hyperparameters is active
    * WARMUP maximum number of steps where pruning is active
    * NR Total number of steps
    * init_state if we start from a given configuration
    
    """
    
    # 1. Directory if the trial    
    trial_dir = setup_trial_dir(working_dir,trial.number)
    
    
    # 2. Suggest Hyperparameters
    optimizer, diag_shift = generate_hyperparameters(trial, n_iter, WARMUP, clipping, dynamic_hyp,bounds)

    # 3. Setup Preconditioner (Stochastic Reconfiguration)
    sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=holomorphic)
    
    # 4. Setup Native Variational State and Driver
    vstate = nk.vqs.FullSumState(hi, model)
    driver = nk.driver.VMC(
        H, 
        optimizer, 
        variational_state=vstate, 
        preconditioner=sr
    )

    # 5. Choose the non linear grid of points
    checkpoints = np.unique(np.geomspace(5,n_iter, num=num_reports, dtype=int))
    steps_to_run = np.diff(checkpoints, prepend=0)
    callback= create_checkpoint_callback(trial_dir,checkpoints,num_reports)
    log = nk.logging.RuntimeLog()
    
    # 6. Run the optimization in chunks
    for chunk_size, current_step in zip(steps_to_run, checkpoints):
        
        # Run the chunk. NetKet handles this entirely internally, very fast!
        driver.run(obs={}, n_iter=int(chunk_size), out=log,callback=[callback])
        
        # Extract the intermediate score
        intermediate_score = log.data["Energy"]["Mean"][-1].real
        
        # Report the score to Optuna AT the specific step
        trial.report(intermediate_score, int(current_step))
        
        # Handle pruning
        if trial.should_prune():
            shutil.rmtree(trial_dir)
            raise optuna.TrialPruned()

    # 7. missing iterations:
    

    # 8. Saved the data
    log.serialize(trial_dir+f"/run_resume_{n_iter}")
    
    final_energy = log.data["Energy"]["Mean"][-1].real
    # 9. Safely handle exploding gradients so Optuna doesn't crash
    if math.isnan(final_energy) or math.isinf(final_energy):
        raise optuna.TrialPruned("Trial diverged resulting in NaN energy.")

    # 10. Save the last iteration:
    save_checkpoint(trial_dir, driver,n_iter)
    

    return final_energy
    
