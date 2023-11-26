import numpy as np
import math
from newton import newtonNd

def trapezoidal(f, x0, t_start, t_end, alpha, jf, e_f, e_delta_x, e_x_rel, maxiter):
    """
    trapezoidal time integration
    
    Inputs:
        x0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        
        t_start: start time

        t_end: end time 
        
        p: dictionary of parameters
        
        eps_dyn: 
        
        V: parameter, V = 10
    outputs:
        x_vec: state vector computed at each time step
    """
    x_vec = []
    x_vec.append(x0)
    x_prev = x0

    I = np.identity(len(x0))

    time = t_start
    t_list =[]
    t_list.append(t_start)
    while  time < t_end:
        fval = f(x_prev)
        dt = alpha/np.linalg.norm(fval)
        time += dt
        t_list.append(time)
        trap = lambda x_new: x_new - x_prev - (dt/2)*(fval + f(x_new))
        j_trap = lambda x_new: I - (dt/2)*jf(x_new)
        x_guess = x_prev
        x_list, _, _ = newtonNd(trap, j_trap, x_guess, e_f, e_delta_x, e_x_rel, maxiter)
        x_vec.append(x_prev)
        x_prev = x_list[-1]

    return x_vec, t_list