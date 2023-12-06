import numpy as np


def deathrate(delC, L_list):
    return 0.1*np.ones_like(L_list)

def growthrate(delC, L_list):
    return delC*np.ones_like(L_list)
    # return np.ones_like(L_list)

def birthrate(delC, L_list):
    """exponentially decays for larger size"""
    # return 0.3*delC*np.ones_like(L_list)*np.exp(-L_list**2)
    return delC

def calc_ddelC_dt(x, params):
    # return -0.1*delC #just drops over time lol
    delC = x[0]
    n = x[1:]
    E = params['E']
    V = params['V']
    rho = params['rho']
    k = params['k']
    L_list = params['L_list']
    
    L_bound = L_list[-1]
    dL = L_list[1]-L_list[0]
    
    return  (E/V)*delC - V*Nc(x, params)

def Nc(x, params):
    # print('hi!')
    rho = params['rho']
    k = params['k']
    L_list = params['L_list']

    delC = x[0]
    n = x[1:]

    g = growthrate(delC, L_list)  # Call growthrate function

    result = np.trapz(3*L_list**2*g*n,L_list)

    result = result*rho*k
    
    return result