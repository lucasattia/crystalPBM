import numpy as np
# import jax.numpy as np

from growthrate import crystal_growth
from deathrate import crystal_death
from birthrate import crystal_birth_breakage, crystal_birth_nucleation
from WENO5 import WENO5_calc
def calc_dndt(x, params):
    """
    Calculates dn/dt
    
    Inputs:
        x: state vector. Contains supersaturation concentration followed by 
        the population density discretized over a dimension L[S, n(L_0), n(L_1), n(L_2),...]
        
        params: dictionary of parameters, e.g. evaporation rate, primary nucleation constant,...
    
    Outputs:
        dn/dt, array discretized over L
    """
    S = x[0] #unpack x vector
    V = x[1]
    n = x[2:]
    print("Type for n", type(n))
    L_list = params['L_list']
    dL = L_list[1]-L_list[0] #assume L evenly spaced
        
    #I'm curious how the accuracy of this compares to using an
    #analytical form for dG_dL and computing n*dG_dL + G*dn_dL
    G = crystal_growth(S, params)
    # dGn_dL = np.gradient(G*n, edge_order=2)/dL 
    k_m =1
    delta_x=dL
    # if params['weno']:
    dGn_dL =WENO5_calc(k_m, G*n, delta_x, eps = 1.0e-40, power=2)
    
    # else:
    # dGn_dL = np.gradient(G*n)/dL 
    print("Type for dGn_dL", type(dGn_dL))
    
    B = crystal_birth_nucleation(x, params) + crystal_birth_breakage(n,params)
    D = crystal_death(n, params)
    dlogV_dt = - params['E']/V #assuming constant evaporation
    print(" We are in dndt")
    print("size of B", np.shape(B))
    print("size of D", np.shape(D))
    print("size of n", np.shape(n))
    print("size of dlogV_dt", np.shape(dlogV_dt))
    print("size of dGn_dL", np.shape(dGn_dL))
    val = B - D - n*dlogV_dt - dGn_dL
    return val