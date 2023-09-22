def crystal_birth(S, alpha, k_N): 
    """
    Description: Function that defines the birth rate for nucleation 

    Inputs: 
        S: supersaturation [M]
        alpha: empirical constant [dimensionless]
        k_N: primary nucleation constant [1/m^3 *s]
    Outputs: 
        B: birth rate
    """
    B = k_N * S**alpha
    return B