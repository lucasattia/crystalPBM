import numpy as np
def crystal_death(n, L): 
    """
    Description: Function that defines the death rate due to breakage

    Inputs: 
        n: list of n values
    Outputs: 
        D: death rate
    """
    return a(L)*n


def a(L): 
    """
    Description: Selection function that defines the probability of breakage at a given crystal size

    Inputs: 
        L: size [m]
    Outputs: 
        a: probability
    """
    if L < 0:
        return 0
    elif 0 <= L <= 1:
        return (6/5)*((L^3)/3 + (L^2)/2)
    else: 
        return 0