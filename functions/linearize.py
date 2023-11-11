import numpy as np
from evalf import evalf
from jacobian import calc_jac
from copy import copy


def eval_linearized_f(x, p, x0, p0, eps=1e-8):
    """
    x is state
    p is parameters
    
    """
    jac = calc_jac(x0, p0, eps)
    f0 =  evalf(x0, None, p0, None)

    df_dp = {}
    for k in p0.keys():
        if k in ['L_list', 'Breakage']:
            continue
        perturbed_p = copy(p0)
        perturbed_p[k] += eps
        df_dp[k] = (evalf(x0, None, perturbed_p, None) -f0)/eps
    
    dxdt = f0 + jac@(x-x0)
    for k in p0.keys():
        if k in ['L_list', 'Breakage']:
            continue
        dxdt += df_dp[k]*(p[k] - p0[k])
        
    return dxdt, df_dp
    
    