from dndt import calc_dndt
from dS_dt import calc_dS_dt
import numpy as np
from deathrate import a

def expand_params(p):
    L_list = p["L_list"]
    L_matrix = np.tile(L_list, (len(L_list),1))
    for i in range(len(L_list)):
        L_matrix[i,:i] = 0
            
    a_L_list = a(L_list)
    a_L_matrix = a(L_matrix)

    B0 = np.zeros_like(L_list)  # nucleation birth matrix
    B0[0] = 1
    expanded_p = p | {
     "L_matrix" : L_matrix,
     "a_L_list" : a_L_list,
     "a_L_matrix" : a_L_matrix,
     "dL" : L_list[1]-L_list[0],
     "B0" : B0,
    }
    return expanded_p


def evalf(x, p):
    """
    forward pass of the time evolution
    
    Inputs:
        x: state vector. Contains supersaturation concentration followed by 
        the population density discretized over a dimension L[S, n(L_0), n(L_1), n(L_2),...]
        
        t: time. Not used, but expected by scipy odeint
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        u: loads/sources??? do we have any???? perhaps evaporation rate?
        
    outputs:
        dxdt
    """
    p = expand_params(p)
    dndt = calc_dndt(x, p)
    dS_dt = calc_dS_dt(x, p)
    dV_dt = -p['E']
    return np.hstack([dS_dt, dV_dt, dndt])