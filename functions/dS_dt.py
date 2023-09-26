from growthrate import crystal_growth
import numpy as np

def calc_dS_dt(x, params):
    S = x[0]
    E = params['E']
    V = params['V']

    return  (E/V)*S - V*Nc(x, params)

def Nc(x, params):
    rho = params['rho']
    k_v = params['k_v']
    L_list = params['L_list']

    S = x[0]
    n = x[1:]

    g = crystal_growth(S, params)  # Call G function developed separately

    result = np.trapz(3*L_list**2*g*n,L_list)

    result = result*rho*k_v
    
    return result