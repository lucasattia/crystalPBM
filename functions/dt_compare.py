import numpy as np
import time 


def dt_compare(x0, dt, p, f, t_final):

    t_ref = np.arange(0,t_final,dt[-1])

    if (f.__name__ == 'euler' or f.__name__ == 'rk_integrator'):
        x_ref = f(x0, t_ref, p)

    elif (f.__name__ == 'simple_matrix_free_trap' or f.__name__ == 'simple_trap'):
        x_ref,_ = f(x0, t_ref, p)
    
    error = []
    t = []

    for step in dt[:-1]:
        t_vec = np.arange(0,t_final,step)

        t0 = time.time()

        if (f.__name__ == 'euler' or f.__name__ == 'rk_integrator'):
            x = f(x0, t_ref, p)

        elif (f.__name__ == 'simple_matrix_free_trap' or f.__name__ == 'simple_trap'):
            x,_ = f(x0, t_ref, p)
        
        t1 = time.time()
        t.append(t1-t0)
        indices = np.linspace(0, len(x_ref) - 1, len(x)).astype(int)
        error.append(np.linalg.norm(x-x_ref[indices])/np.linalg.norm(x_ref[indices]))

    print('Errors for different time discretizations are:',error)
    print('Time taken for different time discretizations are:',t)

    return error, t