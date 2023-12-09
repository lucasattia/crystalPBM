

import numpy as np

from newtonGCR import NewtonGCR
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



def trapezoidalMatrixFree( evalf,x0, deltat, t0, T, p, errf, errDeltax, relDeltax, MaxIter,tolrGCR,epsMF):
    """ 
    TRAPEZOIDAL Trapezoidal time integration.
      :param str eval_f: f(x, p, u)
      :param str eval_Jf: Jf(x, p, u)
      :param Array x0:
      :param float deltat:
      :parma func u: Function u(t)
    """
    # print("x0 in trap", x0)
    p = expand_params(p)
    x_t =[]
    x_t.append(x0) 
    x_prev = x0 
    t = t0 
    
    t = t0
    t_list =[]
    t_list.append(t0)
    count =0
    while t < T:
        x_guess =x_prev
        print("t",t)
        x_next,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X = NewtonGCR(x_guess,evalf,p,errf,errDeltax,relDeltax,MaxIter,tolrGCR,epsMF)
        print("x_next.shape",x_next.shape)
        
        x_t.append(x_prev)
        x_prev =x_next[:,-1]
        print("x_prev.shape",x_prev.shape)
        # print("size of x_next",x_next[-1].shape)
       
        t += deltat
        t_list.append(t)
        count = count+1
        
    print("count",count)
        
     
   

    return x_t,t_list

