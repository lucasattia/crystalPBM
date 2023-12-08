import numpy as np
# from evalf import evalf
def evalf(x):
    """
    dummy evalf to test NewtonGCR
    Inputs:
        x: x-values
        
        
    outputs:
        nonlinear output
    """
    
    out = -np.arctan(x**2)*np.cos(x**2)-0.3
    return out
def calc_jac(x0):
    """dummy calc_jac to work with NewtonNd
    Inputs: 
        x0: initial input
    Outputs:
        jac: output jacobian
    
    
    """
    
    eps=1e-8
    jac = np.zeros((x0.shape[0], x0.shape[0]))
    f0 = evalf(x0,)
    
    for i in range(x0.shape[0]):
        dx = np.zeros_like(x0)
        dx[i] = eps
        jac[:, i] = (evalf(x0+dx) - f0)/eps
        
    return jac
    
    