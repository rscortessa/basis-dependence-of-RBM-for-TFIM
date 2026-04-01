# utils.py
import os
import flax
import math
import numpy as np
import netket as nk
import optax
import optuna
import netket_fidelity as nkf
from netket_fidelity.infidelity import InfidelityOperator
from config import setup_workspace
import os
import flax
import shutil
import json
import numpy as np
import netket as nk


def setup_trial_dir(working_dir, trial_number):
    ckpt_dir = os.path.abspath(os.path.join(working_dir, "optuna_trials"))
    trial_dir = os.path.join(ckpt_dir, f"TRIAL{trial_number}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def create_optimizer(n_iter, warmup, clipping, dynamic_hyp,hyperparameters):
    
    if dynamic_hyp:
        warmup_steps = warmup if warmup is not None else n_iter
        learning_rate = optax.cosine_decay_schedule(init_value=hyperparameters["learning_rate"], decay_steps=n_iter, alpha=0.01)
        diag_shift = optax.cosine_decay_schedule(init_value=hyperparameters["diag_shift"], decay_steps=n_iter, alpha=1e-5/hyperparameters["diag_shift"])
    else:
        learning_rate, diag_shift = hyperparameters["learning_rate"], hyperparameters["diag_shift"]

    if clipping:
        optimizer = optax.chain(optax.clip_by_global_norm(hyperparameters["max_step_size"]), optax.sgd(hyperparameters["learning_rate"]))
    else:
        optimizer = optax.sgd(hyperparameters["learning_rate"])
        
    return optimizer, diag_shift


def generate_hyperparameters(trial, n_iter, warmup, clipping, dynamic_hyp,bounds):
    
    hyperparameters = {}
    hyperparameters["learning_rate"] = trial.suggest_float("learning_rate", bounds["learning_rate"][0],bounds["learning_rate"][1], log=True)
    hyperparameters["diag_shift"] = trial.suggest_float("diag_shift",bounds["diag_shift"][0],bounds["diag_shift"][1], log=True)

    
    if clipping:
        hyperparameters["max_step_size"] = trial.suggest_float("max_step_size",bounds["max_step_size"][0],bounds["max_step_size"][1], log=True)
    
    return create_optimizer(n_iter, warmup, clipping, dynamic_hyp,hyperparameters)



def save_checkpoint(trial_dir, driver, step):
    v_path = os.path.join(trial_dir, f"checkpoint_variables_{step}.mpack")
    o_path = os.path.join(trial_dir, f"checkpoint_optimizer_{step}.mpack")
    
    with open(v_path, "wb") as f:
        f.write(flax.serialization.to_bytes(driver.state.variables))
    with open(o_path, "wb") as f:
        f.write(flax.serialization.to_bytes(driver._optimizer_state))

    return True

def load_checkpoint(trial_dir, driver, step):

    v_path = os.path.join(trial_dir, f"checkpoint_variables_{step}.mpack")
    o_path = os.path.join(trial_dir, f"checkpoint_optimizer_{step}.mpack")

    v_dict = driver.state.variables
    o_dict = driver._optimizer_state
    
    with open(v_path, "rb") as f:
        v_dict = flax.serialization.from_bytes(v_dict, f.read())
    with open(o_path, "rb") as f:
        o_dict = flax.serialization.from_bytes(o_dict, f.read())

    return v_dict,o_dict

def create_checkpoint_callback(ckpt_dir, save_at,N_reports):
    """
    Factory function to create a NetKet callback for checkpointing.
    """
    def checkpoint_callback(step, log_data, driver):
        if step in save_at:
            variables_dict = driver.state.variables
            optimizer_dict = driver._optimizer_state
            
            variables_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_variables_{step}.mpack")
            optimizer_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_optimizer_{step}.mpack")

            with open(variables_ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(variables_dict))
                
            with open(optimizer_ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(optimizer_dict))

        return True        
    return checkpoint_callback

def create_checkpoint_callback_TDVP(ckpt_dir,order_nt,N_reports,obs=None):
    """
    Factory function to create a NetKet callback for checkpointing.
    """

    def checkpoint_callback_TDVP(step,log_data, driver):
        variables_dict = driver.state.variables
        str_step=f"{step:.{order_nt}f}"
        variables_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_variables_{str_step}.mpack")
        with open(variables_ckpt_path, "wb") as f:
            f.write(flax.serialization.to_bytes(variables_dict))

        if obs is not None:
            name_ops=obs.keys()
            time=driver.t
            for name in name_ops:
                op_t=obs[name](time)
                expect_val=driver.state.expect(op_t)
                log_data[name]=expect_val
        
        return True
    
    return checkpoint_callback_TDVP


# A clever duck-typing class to mimic NetKet's internal arrays
class ResizableList(list):
    def resize(self, new_size):
        if new_size > len(self):
            self.extend([0] * (new_size - len(self)))
        else:
            del self[new_size:]

def load_resume_logger(filepath: str) -> nk.logging.RuntimeLog:
    """
    Loads a NetKet JSON log file, handles complex numbers, and dynamically 
    reconstructs History objects with duck-typed arrays to perfectly mimic 
    NetKet's internal append behaviors.
    """
    # 1. Load the raw JSON data
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    # 2. Fix complex numbers
    def _convert_complex(obj):
        if isinstance(obj, dict):
            if 'real' in obj and 'imag' in obj and len(obj) == 2:
                comp_val = np.array(obj['real']) + 1j * np.array(obj['imag'])
                return complex(comp_val) if comp_val.ndim == 0 else comp_val.tolist()
            return {k: _convert_complex(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_complex(item) for item in obj]
        else:
            return obj

    fixed_data = _convert_complex(raw_data)

    # 3. Dynamically rebuild History objects
    def _build_history_tree(data_dict):
        tree = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Leaf node check (History object)
                if not any(isinstance(v, dict) for v in value.values()):
                    try:
                        h = nk.utils.history.History()
                    except TypeError:
                        h = nk.utils.history.History([], [])
                        
                    # Wrap EVERYTHING (including 'iters') in our custom ResizableList
                    # This prevents both the KeyError and the upcoming AttributeError
                    data_vals = {k: ResizableList(v) for k, v in value.items()}
                    
                    h._value_dict = data_vals
                        
                    # Keep surface attributes so nothing else breaks
                    for k, v in data_vals.items():
                        try:
                            setattr(h, k, v)
                        except AttributeError:
                            h.__dict__[k] = v
                            
                    tree[key] = h
                else:
                    # Keep digging for nested dicts
                    tree[key] = _build_history_tree(value)
            else:
                tree[key] = value
        return tree

    history_tree = _build_history_tree(fixed_data)

    # 4. Create empty logger and inject
    logger = nk.logging.RuntimeLog()
    
    for key, value in history_tree.items():
        try:
            logger.data[key] = value
        except Exception:
            if hasattr(logger.data, '_dict'):
                logger.data._dict[key] = value
            else:
                raise RuntimeError(f"Could not inject key '{key}'.")
        
    return logger
