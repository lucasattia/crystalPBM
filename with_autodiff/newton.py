# import numpy as np 
import jax.numpy as np

def newtonNd_to_1d(f, jf, x0, e_f, e_delta_x, e_x_rel, maxiter):
    """Return the root of `f` within `tol` by using Newton's method.

    version only for when f returns a scalar
    
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
    fx = f(x0)
    while k < maxiter and (np.abs(fx) > e_f or np.linalg.norm(delta_x, ord=np.inf) > e_delta_x or x_rel > e_x_rel):
        # delta_x = np.linalg.solve(jf(x[k]), f(x[k]))
        # delta_x = -f(x[k]) / jf(x[k])  # simplified because scalar output
        gradient = jf(x[k])
        print(gradient)
        delta_x = -fx*gradient/np.linalg.norm(gradient)
        f_list.append(fx)
        delta_x_list.append(delta_x)

        x_prev = x[k]
        x.append(x_prev + delta_x)
        k = k +1
        fx = f(x[k])
        x_rel = np.linalg.norm(delta_x, np.inf)/np.linalg.norm(x[k], np.inf)

    if k >= maxiter:
        print("Did not converge, max iterations reached")
    if np.abs(f(x[k]))< e_f and np.linalg.norm(delta_x, np.inf) < e_delta_x and x_rel > e_x_rel:
        print("Converged in " + f'{k}' + " iterations")
    return x, delta_x_list, f_list