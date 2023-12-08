from dndt import calc_dndt
from dS_dt import calc_dS_dt
import numpy as np
from evalf import evalf
from tqdm.notebook import tqdm 
def euler(x0, t_vec, p):
    """
    Euler time integration
    
    Inputs:
        x0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        
        t_vec: time vector
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        
    outputs:
        x_vec: state vector computed at each time step
    """
    x_vec = np.zeros((len(t_vec), len(x0)))
    x_prev = x0

    # assuming constant delta_t
    dt = (t_vec[1] - t_vec[0])/2
    for i in range(len(t_vec)):
        f = evalf(x_prev, p = p)
        x_vec[i] = x_prev + dt*evalf(x_prev, p = p)
        x_prev = x_vec[i]
    return x_vec