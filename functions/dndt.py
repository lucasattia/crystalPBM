import numpy as np

from growthrate import crystal_growth
from deathrate import crystal_death
from birthrate import crystal_birth_breakage, crystal_birth_nucleation
from WENO5 import WENO5_calc
from findiff import FinDiff

def diff(x, p):
    npad = 30
    x = np.concatenate([[x[0]]*npad, x])
    L_list = p['L_list']
    dL = L_list[1]-L_list[0]
    
    if 'order' not in p.keys() or p['order'] == 'np':
        dx_dL = np.gradient(x)/dL
    else:
        dx_dL = FinDiff(0,dL,1, acc=int(p['order']))(x)

    return dx_dL[npad:]

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

    L_list = params['L_list']
    dL = L_list[1]-L_list[0] #assume L evenly spaced
        
    #I'm curious how the accuracy of this compares to using an
    #analytical form for dG_dL and computing n*dG_dL + G*dn_dL
    G = crystal_growth(S, params)
    # dGn_dL = np.gradient(G*n, edge_order=2)/dL 
    k_m = 1
    delta_x = dL
    if params['weno']:
        dGn_dL = WENO5_calc(k_m, G*n, delta_x, eps = 1.0e-6, power=2)
    else: 
        dGn_dL = diff(G*n, params)
        
    # wenodiff = WENO5_calc(k_m, G*n, delta_x, eps = 1.0e-6, power=2)
    # npdiff = diff(G*n, params)
    # print(np.linalg.norm(wenodiff-npdiff))
    # print('n:', G*n[:10]/dL)
    # print('wenodiff:', wenodiff[:10])
    # print('npdiff: ', npdiff[:10])

        
        
    B = crystal_birth_nucleation(x, params) + crystal_birth_breakage(n,params)
    D = crystal_death(n, params)
    dlogV_dt = - params['E']/V #assuming constant evaporation
    
    return B - D - n*dlogV_dt - dGn_dL