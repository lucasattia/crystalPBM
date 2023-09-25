from dndt import calc_dndt
from placeholders import calc_ddelC_dt
import numpy as np


def evalf(x, t, p, u):
    """
    forward pass of the time evolution
    
    Inputs:
        x: state vector. Contains supersaturation concentration followed by 
        the population density discretized over a dimension L[delC, n(L_0), n(L_1), n(L_2),...]
        
        t: time. Not used, but expected by scipy odeint
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        u: loads/sources??? do we have any???? perhaps evaporation rate?
        
    outputs:
        dxdt
    """
    dndt = calc_dndt(x, p)
    ddelC_dt = calc_ddelC_dt(x, p )
    return np.hstack([ddelC_dt, dndt])