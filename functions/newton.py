import numpy as np 

def newtonNd(f, jf, x0, e_f, e_delta_x, e_x_rel, maxiter):
    """Return the root of `f` within `tol` by using Newton's method.

    Parameters
    f : the function of which the root should be found.
    p: param structure
    V: voltage
    jf : the jacobian of `f`.
    x_0 : the initial guess for the algorithm.
    e_f : the tolerance of the function value.
    e_delta_x: the tolerance of the x values
    e_x_rel: the tolerance of the relative x value
    Returns
    x:  The root of `f` within a tolerance of `tol`.

    """
    k  = 0 # iteration counter
    x = [x0]
    delta_x = np.array(x0)
    delta_x_list = []
    delta_x_list.append(delta_x)
    x_rel = np.linalg.norm(delta_x, np.inf)

    f_list = []

    while k < maxiter and (np.linalg.norm(f(x[k]), np.inf) > e_f or np.linalg.norm(delta_x, np.inf) > e_delta_x or x_rel > e_x_rel):
        delta_x = np.linalg.solve(jf(x[k]), f(x[k]))
        f_list.append(f(x[k]))
        delta_x_list.append(delta_x)

        x_prev = x[k]
        x.append(x_prev - delta_x)
        k = k +1
        x_rel = np.linalg.norm(delta_x, np.inf)/np.linalg.norm(x[k], np.inf)

    if k >= maxiter:
        print("Did not converge, max iterations reached")
    if np.linalg.norm(f(x[k]), np.inf) < e_f and np.linalg.norm(delta_x, np.inf) < e_delta_x and x_rel > e_x_rel:
        print("Converged in " + f'{k}' + " iterations")
    return x, delta_x_list, f_list