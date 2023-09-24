import numpy as np


def deathrate(delC, L_list):
    return 0.1*np.ones_like(L_list)

def growthrate(delC, L_list):
    return delC*np.ones_like(L_list)

def birthrate(delC, L_list):
    """exponentially decays for larger size"""
    return 0.3*delC*np.ones_like(L_list)*np.exp(-L_list**2)

def calc_ddelC_dt(delC, params):
    return -0.1*delC #just drops over time lol