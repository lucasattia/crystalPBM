from dndt import calc_dndt
from dS_dt import calc_dS_dt
import numpy as np
from evalf import evalf
from tqdm.notebook import tqdm 
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
    p = expand_params(p)
    x_vec = np.zeros((len(t_vec), len(x0)))
    x_prev = x0

    # assuming constant delta_t
    dt = (t_vec[1] - t_vec[0])/2
    for i in range(len(t_vec)):
        f = evalf(x_prev, p = p)
        x_vec[i] = x_prev + dt*evalf(x_prev, p = p)
        x_prev = x_vec[i]
    return x_vec