import numpy as np

from placeholders import growthrate, deathrate, birthrate

#TODO: isnt death rate supposed to be D*n? otherwise n can be negative...
def calc_dndt(x, params):
    """
    Calculates dn/dt
    
    Inputs:
        x: state vector. Contains supersaturation concentration followed by 
        the population density discretized over a dimension L[delC, n(L_0), n(L_1), n(L_2),...]
        
        params: dictionary of parameters, e.g. evaporation rate, primary nucleation constant,...
    
    Outputs:
        dn/dt, array discretized over L
    """
    delC = x[0] #unpack x vector
    n = x[1:]

    L_list = params['L_list']
    dL = L_list[1]-L_list[0] #assume L evenly spaced
        
    #I'm curious how the accuracy of this compares to using an
    #analytical form for dG_dL and computing n*dG_dL + G*dn_dL
    G = growthrate(delC, L_list)
    dGn_dL = np.gradient(G*n, edge_order=2)/dL 
    
    bin1 = np.zeros_like(n)
    bin1[0] = 1
    B = birthrate(delC, L_list)*bin1
    D = deathrate(delC, L_list)
    # dlogV_dt = params['dlogV_dt'] #assuming constant evaporation
    dlogV_dt = -np.log(params['E'])
    
    return B - D*n*L_list - n*dlogV_dt - dGn_dL