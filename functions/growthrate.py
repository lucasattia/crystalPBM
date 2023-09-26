def crystal_growth(S, g, k_g): 
    """
    Description: Function that defines the crystal growth rate 

    Inputs: 
        S: supersaturation [M]
        g: empirical power constant [dimensionless]
        k_g: crystal growth constant [1/m^3 *s]
    Outputs: 
        B: birth rate [m/s]
    """
    G = k_g * S**g
    return G