from dndt import calc_dndt
from dS_dt import calc_dS_dt
# import numpy as np
import jax.numpy as np
from evalf import evalf
from tqdm.notebook import tqdm 
from jax import jit
from deathrate import a

@jit
def rk_loop(x, dt, p):
    """ 
    This function runs the internal loop of the RK4 method
    """
    x_k1= evalf(x,p)
    x_k2 = evalf(x+dt*x_k1/2, p)
    x_k3 = evalf(x+dt*x_k2/2, p)
    x_k4 = evalf(x+dt*x_k3, p) 
    return x + (dt/6.0)*(x_k1 + 2*x_k2 + 2*x_k3 + x_k4)


def expand_params(p):
    L_list = p["L_list"]
    L_matrix = np.tile(L_list, (len(L_list),1))
    for i in range(len(L_list)):
        L_matrix = L_matrix.at[i,:i].set(0)
            
    a_L_list = a(L_list)
    a_L_matrix = a(L_matrix)

    B0 = np.zeros_like(L_list)  # nucleation birth matrix
    B0 = B0.at[0].set(1)
    expanded_p = p | {
     "L_matrix" : L_matrix,
     "a_L_list" : a_L_list,
     "a_L_matrix" : a_L_matrix,
     "dL" : L_list[1]-L_list[0],
     "B0" : B0,
    }
    return expanded_p

def rk_integrate(x0, t_vec, p):
    """
   
    
    Inputs:
        x0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        
        t_vec: time vector
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        
    outputs:
        x_vec: state vector computed at each time step
    """
    
    p = expand_params(p)  # precomupte some things to save time
    xlist = [x0]

    # assuming constant delta_t
    dt = (t_vec[1] - t_vec[0])/2
    # for i in tqdm(range(len(t_vec))):
    for _ in range(len(t_vec)-1):
        # x_vec[i] = rk_loop(x_prev, dt, p)
        # x_prev = x_vec[i]
        xlist.append(rk_loop(xlist[-1], dt, p))
    return np.array(xlist)