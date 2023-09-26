import numpy as np
from evalf import evalf

def calc_jac(x0, params, eps=1e-10):
    jac = np.zeros((x0.shape[0], x0.shape[0]))
    f0 = evalf(x0, None, params, None)
    
    for i in range(x0.shape[0]):
        dx = np.zeros_like(x0)
        dx[i] = eps
        jac[:, i] = (evalf(x0+dx, None, params, None) - f0)/eps
        
    return jac
    
    