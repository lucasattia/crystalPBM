from deathrate import a
import numpy as np
def crystal_birth_nucleation(S, alpha, k_N): 
    """
    Description: Function that defines the birth rate for nucleation 

    Inputs: 
        S: supersaturation [M]
        alpha: empirical constant [dimensionless]
        k_N: primary nucleation constant [1/m^3 *s]
    Outputs: 
        B: birth rate
    """
    B = k_N * S**alpha
    return B

def crystal_birth_breakage(n, L, L_list): 
    """
    Description: Function that defines the birth rate due to breakage 

    Inputs: 

    Outputs: 
        B: birth rate
    """
    B = np.trapz(b(L)*a(L)*n, L_list)
    return B

def b(L):
    return 1

