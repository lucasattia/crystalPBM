import numpy as np
L_list = np.linspace(0,1, 100)

p = {"L_list" : L_list,
     'E' : 1e-7,
     'V' : 1e-3, 
     'rho' : 1200,
     'k_g' : 1e10, #growth rate constant m/s
     'k_N' :  1e5, #nucleation rate constant 
     'k_V' :  1, #goes in N_C, volumetric shape factor
     'g'   : 3, #power constant for growth
     'a'   : 5 #power constant for nucleation
     }