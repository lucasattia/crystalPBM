from deathrate import a
import numpy as np

def crystal_birth_nucleation(x, params): 
    """
    Description: Function that defines the birth rate for nucleation 

    Inputs: 
        S: supersaturation [M]
        alpha: empirical constant [dimensionless]
        k_N: primary nucleation constant [1/m^3 *s]
    Outputs: 
        B: birth rate
    """
    k_N = params['k_N']
    alpha = params['alpha']
    S = x[0]
    n = x[1:]
    B = np.zeros_like(n)
    B[0] = 1
    
    return B* k_N * S**alpha

# def crystal_birth_nucleation(S, alpha, k_N): 
#     """
#     Description: Function that defines the birth rate for nucleation 

#     Inputs: 
#         S: supersaturation [M]
#         alpha: empirical constant [dimensionless]
#         k_N: primary nucleation constant [1/m^3 *s]
#     Outputs: 
#         B: birth rate
#     """
#     B = k_N * S**alpha
#     return B

def crystal_birth_breakage(n, params): 
    """
    Description: Function that defines the birth rate due to breakage 

    Inputs: 

    Outputs: 
        B: birth rate
    """
    
    #TODO: speed this up, slowest part I think
    L_list = params['L_list']
    B = np.zeros_like(L_list)
    for i in range(len(L_list)):
        B[i] = np.trapz(b(L_list[i:])*a(L_list[i:])*n[i:], L_list[i:])
        
    return B

# def crystal_birth_breakage(n, L, L_list): 
#     """
#     Description: Function that defines the birth rate due to breakage 

#     Inputs: 

#     Outputs: 
#         B: birth rate
#     """
#     B = np.trapz(b(L)*a(L)*n, L_list)
#     return B

def b(L):
    return 1

