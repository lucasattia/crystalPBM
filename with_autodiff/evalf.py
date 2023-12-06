from dndt import calc_dndt
from dS_dt import calc_dS_dt
# import numpy as np
import jax.numpy as np



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
    
    dndt = calc_dndt(x, p)
    dS_dt = calc_dS_dt(x, p)
    dV_dt = -p['E']
    return np.hstack([dS_dt, dV_dt, dndt])