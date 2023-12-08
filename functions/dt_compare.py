import numpy as np
import time 


def dt_compare(x0, dt, p, f, t_final):

    t_ref = np.arange(0,t_final,dt[-1])

    x_ref = f(x0, t_ref, p)

    error = []
    t = []

    for step in dt[:-1]:
        t_vec = np.arange(0,t_final,step)

        t0 = time.time()
        x = f(x0, t_vec, p)
        t1 = time.time()

        t.append(t1-t0)

        error.append(np.linalg.norm(x-x_ref)/np.linalg.norm(x_ref))

    print('Errors for different time discretizations are:',error)
    print('Time taken for different time discretizations are:',time)

    return error, t