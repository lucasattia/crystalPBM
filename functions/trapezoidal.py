from dndt import calc_dndt
from dS_dt import calc_dS_dt
import numpy as np
from evalf import evalf
from tqdm.notebook import tqdm 

def rk_loop(x, dt, p):
    """
    This function runs the internal loop of the RK4 method
    """
    x_k1= evalf(x,p)
    x_k2 = evalf(x+dt*x_k1/2, p)
    x_k3 = evalf(x+dt*x_k2/2, p)
    x_k4,= evalf(x+dt*x_k3, p)
    return x + (dt/6.0)*(x_k1 + 2*x_k2 + 2*x_k3 + x_k4),


def trapezoidal(x0, t_vec, p, eps_dyn):
    """
    trapezoidal time integration
    
    Inputs:
        x0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        
        t_vec: time vector
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        eps_dyn: 
        
    outputs:
        x_vec: state vector computed at each time step
    """
    x_vec = np.zeros((len(t_vec), len(x0)))
    x_prev = x0

    # assuming constant delta_t
    dt = (t_vec[1] - t_vec[0])/2
    for i in tqdm(range(len(t_vec))):
        f = evalf(x_prev, t = None, p = p, u = None)
        x_vec[i] = rk_loop(x_prev, dt, p)
        x_prev = x_vec[i]
    return x_vec