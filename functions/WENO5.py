

import numpy as np
# import jax.numpy as np
from tqdm.notebook import tqdm 
# from jax import jit

def six_pt_stensil(x0):
    x_m3 = x0[0:-5]
    x_m2 = x0[1:-4]
    x_m1 = x0[2:-3]
    x_0  = x0[3:-2]
    x_p1 = x0[4:-1]
    x_p2 = x0[5:]
    
    return x_m3, x_m2, x_m1, x_0, x_p1, x_p2
# @jit
def WENO5_calc(k_m, x0, delta_x, eps = 1.0e-40, power=2):
    """
    WENO 5th order integration scheme
    
    Inpoweruts:
        u0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        
        t_vec: time vector
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        eps_dyn: 
        
    outputs:
        du/dspatial: finite difference spatial derivative
    """
    # x_vec = np.zeros((len(t_vec), len(x0)))
    # x_prev = x0
   
    # 6-pt stencil biased in upwind direction = info from +2 nodes forward, -3 nodes backward
    len_x0=len(x0)
    # print("This is the size of x0", len_x0)
    first_num = x0[0]
    last_num = x0[len_x0-1]
    
    
    # first_nums= np.interp(np.array([0,1,2]),np.array([3,4,5]),x0[0:3])
    # last_nums= np.interp(np.array([3,4]),np.array([0,1,2]),x0[-3:])
    # poly_this = np.polyfit(np.array([8,4,6,8,9]), np.ones(5), deg=3)
    # print("Data type x0[0:8]",type(x0[0:8]))
    # print(x0[0:8])
    # print("Data type np.array([0,1,2,3,4,5,6,7])",type(np.array([0,1,2,3,4,5,6,7])))
    # poly_first_nums = np.polyfit(np.ones(15)*np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), np.ones(15)*x0[0:15], deg=8)
    # first_nums = np.polyval(poly_first_nums, np.ones(3)*np.array([-3,-2,-1]))
    
    
    # poly_last_nums = np.polyfit(np.ones(15)*np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), np.ones(15)*x0[-15:], deg=8)
    # last_nums = np.polyval(poly_last_nums,np.ones(2)* np.array([15,16]))
    
    
    # first_nums = np.array(x0[0],x0[1],x0[2])
    # last_nums = np.array(x0[len_x0],x0[-1:])
    
    # x0 = np.insert(x0,0,first_nums )
    # x0 = np.insert(x0,len_x0,last_nums)
    x0 = np.insert(x0,0,np.array([first_num,first_num,first_num]) )
    len_x0=len(x0)
    x0 = np.insert(x0,len_x0,np.array([last_num,last_num]))
    x_m3, x_m2, x_m1, x_0, x_p1, x_p2 = six_pt_stensil(x0)
   
    x_m3 = k_m * x_m3
    x_m2 = k_m * x_m2
    x_m1 = k_m * x_m1
    x_0 = k_m * x_0
    x_p1 = k_m * x_p1
    x_p2 = k_m * x_p2
   
    # convex combination of stencil options, also written as ̄ω (bar over omega) in Henrick work, but γ elsewhere
    # expressed as "ideal weights" (converge to them when smooth, rebalance when some stencils see discontinuities)
    gamma0 = 1 / 10
    gamma1 = 3 / 5
    gamma2 = 3 / 10
   
    # indicators for smoothness (IS) for f_{i-1/2} postfixed with m for minus, for f_{i+1/2} with p for positive
    beta0m = 13/12 * (x_m3 - 2*x_m2 + x_m1)**2 + 1/4 * (x_m3 - 4*x_m2 + 3*x_m1)**2
    beta1m = 13/12 * (x_m2 - 2*x_m1 + x_0)**2 + 1/4 * (x_0 - x_m2)**2
    beta2m = 13/12 * (x_m1 - 2*x_0 + x_p1)**2 + 1/4 * (3*x_m1 - 4*x_0 + x_p1)**2
   
    beta0p = 13/12 * (x_m2 - 2*x_m1 + x_0)**2 + 1/4 * (x_m2 - 4*x_m1 + 3*x_0)**2
    beta1p = 13/12 * (x_m1 - 2*x_0 + x_p1)**2 + 1/4 * (x_p1 - x_m1)**2
    beta2p = 13/12 * (x_0 - 2*x_p1 + x_p2)**2 + 1/4 * (3*x_0 - 4*x_p1 + x_p2)**2
 
    # Jiang and Shu constraint satisfaction by rescaling of ideal weights on stencils, this is the non-oscillatory part of WENO!
    # (1) in smooth regions, weights converge converge to "ideal weights" γ_k as Δx → 0 in non-smooth regions
    # (2) in non-smooth regions (near discontinuities), weights remove contribution of stencils which contain the discontinuity
    # post-fixed with JS
    alpha0m = gamma0 / (beta0m + eps)**power
    alpha1m = gamma1 / (beta1m + eps)**power
    alpha2m = gamma2 / (beta2m + eps)**power
    alpha_m = alpha0m + alpha1m + alpha2m
    alpha0m_JS = alpha0m / alpha_m
    alpha1m_JS = alpha1m / alpha_m
    alpha2m_JS = alpha2m / alpha_m
   
    alpha0p = gamma0 / (beta0p + eps)**power
    alpha1p = gamma1 / (beta1p + eps)**power
    alpha2p = gamma2 / (beta2p + eps)**power
    alpha_p = alpha0p + alpha1p + alpha2p
    alpha0p_JS = alpha0p / alpha_p
    alpha1p_JS = alpha1p / alpha_p
    alpha2p_JS = alpha2p / alpha_p
   
    # compute all the 3-pt stencil weighted options of numerical fluxes f_{i+1/2} (p) and f_{i-1/2} (m)
    # at the three different stencils possible, each, for a total of 6 approximations (see eqn 15 in Henrick 2005)
    h_m0 = (2*x_m3 - 7*x_m2 + 11*x_m1)/6
    h_m1 = (-x_m2 + 5*x_m1 + 2*x_0)/6
    h_m2 = (2*x_m1 + 5*x_0 - x_p1)/6
   
    h_p0 = (2*x_m2 - 7*x_m1 + 11*x_0)/6
    h_p1 = (-x_m1 + 5*x_0 + 2*x_p1)/6
    h_p2 = (2*x_0 + 5*x_p1 - x_p2)/6
   
    # output H mapped weighted flux term
    # if henrick_flag
       
        # Henrick mappings (post-fixed with H), bound the error in rescaling of weights near
        # non-smooth regions to still be 5th order total, and to not degrade
    g0_m = alpha0m_JS * (gamma0 + gamma0**2 - 3*gamma0*alpha0m_JS + alpha0m_JS**2) / (gamma0**2 + alpha0m_JS * (1 - 2*gamma0))
    g1_m = alpha1m_JS * (gamma1 + gamma1**2 - 3*gamma1*alpha1m_JS + alpha1m_JS**2) / (gamma1**2 + alpha1m_JS * (1 - 2*gamma1))
    g2_m = alpha2m_JS * (gamma2 + gamma2**2 - 3*gamma2*alpha2m_JS + alpha2m_JS**2) / (gamma2**2 + alpha2m_JS * (1 - 2*gamma2))
    g_m = g0_m + g1_m + g2_m
    alpha0m_H = g0_m / g_m
    alpha1m_H = g1_m / g_m
    alpha2m_H = g2_m / g_m
    
    g0_p = alpha0p_JS * (gamma0 + gamma0**2 - 3*gamma0*alpha0p_JS + alpha0p_JS**2) / (gamma0**2 + alpha0p_JS * (1 - 2*gamma0))
    g1_p = alpha1p_JS * (gamma1 + gamma1**2 - 3*gamma1*alpha1p_JS + alpha1p_JS**2) / (gamma1**2 + alpha1p_JS * (1 - 2*gamma1))
    g2_p = alpha2p_JS * (gamma2 + gamma2**2 - 3*gamma2*alpha2p_JS + alpha2p_JS**2) / (gamma2**2 + alpha2p_JS * (1 - 2*gamma2))
    g_p = g0_p + g1_p + g2_p
    alpha0p_H = g0_p / g_p
    alpha1p_H = g1_p / g_p
    alpha2p_H = g2_p / g_p
    
    # convex linear combination of Henrick weighted stencil points to get (f_{i+1/2} - f_{i-1/2}) / Δx
    hp_M = alpha0p_H*h_p0 + alpha1p_H*h_p1 + alpha2p_H*h_p2
    hm_M = alpha0m_H*h_m0 + alpha1m_H*h_m1 + alpha2m_H*h_m2

    # print("hp_M", hp_M)
    # print("hm_M", hm_M)
    return (hp_M - hm_M) / delta_x
   
    # less computationally expensive, output JS weighted flux term
    # however, convergence weakens to 3rd order for any ϵ outside of [1e-7,1e-5]
    # else
       
    #     # convex linear combination of JS weighted stencil points to get (f_{i+1/2} - f_{i-1/2}) / Δx
    #     hp_JS = alpha0p_JS*h_p0 + alpha1p_JS*h_p1 + alpha2p_JS*h_p2
    #     hm_JS = alpha0m_JS*h_m0 + alpha1m_JS*h_m1 + alpha2m_JS*h_m2
 
    #     return (hp_JS - hm_JS) / Δx
    # end



def Runge_kutta(k_m,x,delta_t,delta_x):
    """
    3rd order Runge Kutta method
        Inputs:
            k_m: 
            x: the state vector 
            delta_t: time step
            delta_x: discrete spatial step
            
        Outputs:
            x_new: new state vector computed at each time step
            
    """
   
    
    WENO_soln0 = WENO5_calc(k_m, x, delta_x, eps = 1.0e-40, power=2)
    x_star = x + delta_t *WENO_soln0
    
    WENO_soln1 = WENO5_calc(k_m, x_star, delta_x, eps = 1.0e-40, power=2)
    x_star_star = (3/4)*x +(1/4)*x_star +(1/4)*delta_t *WENO_soln1
    
    WENO_soln2 = WENO5_calc(k_m, x_star_star, delta_x, eps = 1.0e-40, power=2)
    x_new = (1/3)*x +(2/3)*x_star +(2/3)*delta_t *WENO_soln2
    
    return x_new
def time_int_main_calc (x0,k_m,delta_x, t_vec, p):
    
    """
    Time integration method with WENO fifth order integration and 3rd order Runge Kutta method
    
    Inputs:
        x0: initial state vector. Contains supersaturation concentration, then volume, followed by 
        the population density discretized over a dimension L[S, V, n(L_0), n(L_1), n(L_2),...]
        k_m:
        delta_x: discrete spatial step
        t_vec: time vector
        
        p: dictionary of parameters, e.g. L_List ( L values we discretized over -- assume evenly spaced),
        evaporation rate, primary nucleation constant,...
        
        
    outputs:
        x_vec: state vector computed at each time step
    
    
    """
    x_vec = np.zeros((len(t_vec), len(x0)))
    x_prev = x0
    
    delta_t = (t_vec[1] - t_vec[0])/2
    for i in tqdm(range(len(t_vec))):
        # f = evalf(x_prev, t = None, p = p, u = None)
        x_vec[i] = Runge_kutta(k_m,x_prev,delta_t,delta_x)
        x_prev = x_vec[i]
    
    
    
    return x_vec