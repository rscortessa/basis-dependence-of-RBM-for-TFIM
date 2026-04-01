import netket as nk                                                                                                                   
import numpy as np
from netket.operator.spin import sigmax,sigmaz,sigmay,identity,sigmam,sigmap                                                                               

def rotated_sigmax(angle):
    r_sigmax = np.array([[0, 1], [1, 0]])
    r_sigmaz = np.array([[1, 0], [0, -1]])
    return np.cos(angle)*r_sigmax+np.sin(angle)*r_sigmaz


def rotated_sigmaz(angle):
    r_sigmax = np.array([[0, 1], [1, 0]])
    r_sigmaz = np.array([[1, 0], [0, -1]])
    return np.cos(angle)*r_sigmaz-np.sin(angle)*r_sigmax

def isigmay():
    return np.array([[0,1],[-1,0]])

def rotated_IsingModel(angle,Gamma,L,hi,pbc=False):
     # Initialize Hamiltonian as a LocalOperator
    pseudo_sigma_x=rotated_sigmax(angle)
    pseudo_sigma_z=rotated_sigmaz(angle)
    H = nk.operator.LocalOperator(hi)

    # Add 2 body- interactions
    for i in range(L - 1):
        H -= nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [i, i+1])
    # Add single body term
    for i in range(L):
        H -= Gamma * nk.operator.LocalOperator(hi,pseudo_sigma_x,[i])
    if pbc:
        H-=nk.operator.LocalOperator(hi, np.kron(pseudo_sigma_z,pseudo_sigma_z), [L-1,0])
    return H



