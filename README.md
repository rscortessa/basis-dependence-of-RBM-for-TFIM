# basis-dependence-of-RBM-for-TFIM
This repository contains the code and the data concerning the manuscript found in https://arxiv.org/abs/2512.11632

## Overview

This repository contains the code and data supporting the study on basis dependence of Restricted Boltzmann Machines (RBMs) for the Transverse-Field Ising Model (TFIM), as presented in the manuscript at https://arxiv.org/abs/2512.11632.

## Contents

- **Core utilities**: `NQS_utils.py` - Neural quantum state utilities
- **Configuration & models**: `config.py`, `define_models.py`, `hams.py` - Setup and model definitions for RBM and Hamiltonians
- **Data generation**: `generate_data.sh`, `cluster_expansion_data.sh` - Scripts for generating computational data
- **Optimization**: `optuna_setup.py`, `study.py`, `resume_study.py` - Hyperparameter optimization using Optuna
- **Analysis & visualization**: `energy_and_infidelity_plot.py`, `mean_sign_plot.py`, `plot_wf.py` - Plotting and analysis scripts

The code implements RBM-based ansatze for TFIM systems and explores how the choice of basis affects model performance and quantum state representation.