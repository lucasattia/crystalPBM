

import numpy as np

from newtonGCR import NewtonGCR
from deathrate import a
from trapezoidal import rk_loop

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



def trapezoidalMatrixFree(evalf,x0, alpha, t0, T, p, errf, errDeltax, relDeltax, MaxIter,tolrGCR,epsMF):
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
    # count =0
    
    # dt = 0.1  # idk try some guess
    while t < T:
        fval = evalf(x_prev, p)
        
        # pick dt. Feels like there are lots of different interesting things we can try here.
        # fval_weighted = np.copy(fval)
        # fval_weighted[0] *= 5*np.linalg.norm(x_prev)  # supersaturation rate of change is really important, give it more weight
        # dt = alpha*np.linalg.norm(x_prev)/np.linalg.norm(fval_weighted)
        
        # let's look at the maximum normalized rate of change of any component
        normalized_derivative = np.abs(fval/(x_prev + alpha[0]))
        max_derivative = np.nanmax(normalized_derivative)
        dt = alpha[1]/max_derivative
        dt = np.min([dt, alpha[2]])
        # dt = alpha/np.nanmax(np.abs(fval/(x_prev + 0.1)))
        
        # dt = 0.5
        if t + dt > T:
            dt = T-t
            

        f_trap = lambda x_new, _: x_new - x_prev - (dt/2)*(fval + evalf(x_new, p))
        x_guess = x_prev + dt*fval
        # x_guess = rk_loop(x_prev, dt, p)
        x_next, converged,errf_k,errDeltax_k,relDeltax_k,iterations,X = NewtonGCR(x_guess, f_trap, p, errf, errDeltax, relDeltax, MaxIter, tolrGCR, epsMF)
        
        
        x_t.append(x_prev)
        x_prev =x_next[:,-1]
       
        t += dt
        t_list.append(t)
        # print('t:',t)
        # count = count+1
        
    # print("count",count)
        
     
   

    return x_t,t_list

