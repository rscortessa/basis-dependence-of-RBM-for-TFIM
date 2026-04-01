import argparse
import optuna
import os
# Dictionary mapping architectures to their specific hyperparameters
ALL_NN_PARAMS = {"RBM_COMPLEX": ["NN"]}

def get_optuna_args(args_list=None):
    """
    This function initializes all the parameters required for the optuna study:

    - niter indicates the number of iterations of the optuna search
    - trials indicates the number of random searches for the optuna study
    - clipping if set true, gives an upper bound to the gradient norm of the energy
    - runners is the number of searches run in parallel
    - dynamic_hyp if set true implements a cosine decay for the hyperparameters
    - warmup if dynamic_hyp is set true indicates the number of steps on which the hyperaparameters decay
    - reports is the number of times on which the state of the driver is saved.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--clipping", action="store_true")
    parser.add_argument("--runners", type=int,default=1)
    parser.add_argument("--dynamic_hyp", action="store_true")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--reports", type=int, default=12)
    optuna_args, _ = parser.parse_known_args()
    return optuna_args


def get_params(args_list=None):
    """
    This function initializes all the physical variables of the system:
    - L 
    - NR number of time steps in total
    - angle indicates the rotation angle (measured in degrees).
    - g transversal field
    - model indicates what type of statistical model we will minimize
    - pbc if true periodic boundary conditions are set
    - NN indicates the hidden unit density
    """
    # Initialize the parser (Pass 1: Core arguments)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--W", type=int, default=1)
    parser.add_argument("--NR", type=int, default=1000)
    parser.add_argument("--angle", type=float, default=0)
    parser.add_argument("--g", type=float, default=150)
    parser.add_argument("--model", type=str, default="QIM")
    parser.add_argument("--pbc", action="store_true")
    parser.add_argument("--NN", type=float, default=1.0)
    
    # Parse known args to check the architecture and model
    args, _ = parser.parse_known_args(args_list)
    params = vars(args)
    # Derive fixed/background parameters
    params['architecture'] = 'RBM_COMPLEX'
    params["add"] = "PBC" if params["pbc"] else ""

    # Determine holomorphicity dynamically to avoid JAX gradient crashes
    params["holomorphic"] =True
    # Define the Root Directory strictly as a parameter string 
    # (Directory creation logic is moved to ground_state_search.py)
    params["ROOT_DIR"] = f"PLAYING_WITH_{params['architecture']}"

    return params

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

# Optional: You can test the config easily if you run this script directly
if __name__ == "__main__":
    test_params = get_params()
    print("Parsed Parameters:")
    for k, v in test_params.items():
        print(f"  {k}: {v}")
