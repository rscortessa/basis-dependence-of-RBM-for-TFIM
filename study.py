
from optuna_setup import setup_workspace
from define_models import create_system
import argparse
from functools import partial
import optuna
import netket as nk
import os
# Import the refactored factory functions

from config import get_params,get_optuna_args,ALL_NN_PARAMS


def main():
    # Verify setup
    # 1. Get global parameters
    params = get_params()
    optuna_args=get_optuna_args()
    # 2. Create the physical system and models dynamically
    hi, ham, model, cost_func, models_name = create_system(params)
    
    # 3. Setup the workspace (MPI-safe directory and DB creation)
    NN_params = ALL_NN_PARAMS[params["architecture"]]
    working_dir, study_name, storage = setup_workspace(
        params,optuna_args,NN_params, params["ROOT_DIR"], models_name
    )
    
    # 4. Load the Optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)
    ckpt_dir = os.path.abspath(os.path.join(working_dir, "optuna_trials"))
    os.makedirs(ckpt_dir, exist_ok=True)

    n_trials_done = len(study.trials)

    bounds={}
    bounds["learning_rate"] = [1e-5,1e-2]
    bounds["diag_shift"] = [1e-5,1e-1]
    bounds["max_step_size"] = [1e-5,1e-1]
    
    
    if n_trials_done < optuna_args.trials:
        # Prepare the objective function
        objective_final = partial(
            cost_func,
            model=model,
            L=params["L"] * params["W"],
            hi=hi,
            H=ham,
            n_iter=optuna_args.niter,
            WARMUP=optuna_args.warmup,
            holomorphic=params["holomorphic"],
            working_dir=working_dir,
            bounds=bounds,
            clipping=optuna_args.clipping,
            dynamic_hyp=optuna_args.dynamic_hyp,
            num_reports=optuna_args.reports
        )
        

        print(f"🚀 Running Optuna trials for parameters:\n{params}")
        print(f"⏳ Remaining trials to run: {optuna_args.trials - n_trials_done}")

        # Run optimization
        study.optimize(
            objective_final, 
            n_trials=optuna_args.trials - n_trials_done, 
            n_jobs=optuna_args.runners
        )
        
        print("✅ Finished trials for these parameters.")
        
    else:
        print(f"✅ Target number of trials ({optuna_args.trials}) already reached!")
        
    
    
if __name__ == "__main__":
    main()
