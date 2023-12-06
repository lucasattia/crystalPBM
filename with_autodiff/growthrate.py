def crystal_growth(S, params):
    """
    Description: Function that defines the crystal growth rate 

    Inputs: 
        S: supersaturation [M]
        g: empirical power constant [dimensionless]
        k_g: crystal growth constant [1/m^3 *s]
    Outputs: 
        B: birth rate [m/s]
    """
    k_g = params['k_g']
    g = params['g']
    G = k_g * S**g
    return G