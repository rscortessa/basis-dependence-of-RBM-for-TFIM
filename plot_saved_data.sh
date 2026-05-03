#!bin/bash

# mean sign plot for the ferromagnetic case
python mean_sign_plot.py --g 0.5
# wavefunction amplitudes for the ferromagnetic case
python plot_wf.py --g 0.5
# cluster expansion plots for the paramagnetic case 
python cluster_exp_plot.py --g 1.5 --NNs 1.0 2.0 3.0 --angle 7.5
python cluster_exp_plot.py --g 1.5 --NNs 1.0 2.0 3.0 --angle 30.0
python cluster_exp_plot.py --g 1.5 --NNs 1.0 2.0 3.0 --angle 52.5
python cluster_exp_plot.py --g 1.5 --NNs 1.0 2.0 3.0 --angle 75.0
