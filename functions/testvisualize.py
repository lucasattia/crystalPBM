
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from evalf import evalf
from visualize import visualize

L_list = np.linspace(0,1, 100)
tlist = np.linspace(0,1e-5,100)

S0 = 1 #initial supersaturation
V0 = 1
n0 = np.zeros_like(L_list) #initial population density distribution

#parameters
p = {"L_list" : L_list, #discretization bins
     'E' :  1e-7, #evaporation rate
     'V' : 1e-3, #solvent volume
     'rho' : 1200, # density of the crystal
     'k_v' : 1, #goes in N_C, volumetric shape factor
     'k_g' : 1e2, #growth rate constant m/s
     'g' : 3, #power constant for growth
     'k_N' : 1e5, #nucleation rate constant 
     'alpha' :5, #power constant for nucleation
     'Breakage': True #toggle breakage for debug
     }


#integrate the equations
x = np.hstack([S0, V0, n0])
x_t = odeint(evalf, y0=x, t=tlist, args=(p,None))


#plot results
n_t = x_t[:,2:]

t_ind=50
visualize(n_t,L_list,t_ind)

