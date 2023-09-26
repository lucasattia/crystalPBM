import numpy as np

def crystal_death(n, params): 
    """
    Description: Function that defines the death rate due to breakage

    Inputs: 
        n: list of n values
        params['L_list']: array of size discretization values
    Outputs: 
        D: death rate
    """
    L_list = params['L_list']
    return a(L_list)*n


# def crystal_death(n, L): 
#     """
#     Description: Function that defines the death rate due to breakage

#     Inputs: 
#         n: list of n values
#     Outputs: 
#         D: death rate
#     """
#     return a(L)*n


def a(L_list): 
    """
    Description: Selection function that defines the probability of breakage at a given crystal size

    Inputs: 
        L: size [m]
    Outputs: 
        a: probability
    """
    a = (6/5)*((L_list**3)/3 + (L_list**2)/2)
    a[L_list < 0] = 0
    a[L_list > 1] = 0
    
    return (6/5)*((L_list**3)/3 + (L_list**2)/2)


# def a(L): 
#     """
#     Description: Selection function that defines the probability of breakage at a given crystal size

#     Inputs: 
#         L: size [m]
#     Outputs: 
#         a: probability
#     """
#     if L < 0:
#         return 0
#     elif 0 <= L <= 1:
#         return (6/5)*((L^3)/3 + (L^2)/2)
#     else: 
#         return 0