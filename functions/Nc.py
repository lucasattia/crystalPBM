import numpy as np
from scipy.integrate import trapz
from scipy.integrate import solve_ivp



def Nc(n, rho, k, dL, L_bound):

    x = np.arange(0,L_bound+dL,dL) 

    g = G(x)  # Call G function developed separately
        
    result = trapz(3*x**2*g*n,dL)

    result = result*rho*k
    
    return result


def ode_function(c, V, E, n, rho, k, dL, L_bound):

    dcdt = (E/V)*c - V*Nc(n, rho, k, dL, L_bound)

    return dcdt

# Set the Initial Condition and Time Span
initial_condition = [0]  # c(0) = 0
t_span = (0, 5)  # Solve the ODE from t=0 to t=5

def ode_solve ( t_span, initial_condition, E, V, n, rho, k, dL, L_bound):

    # Solve the ODE with the given parameters
    solution = solve_ivp(ode_function, t_span, initial_condition, args=(E, V, n, rho, k, dL, L_bound), t_eval=np.linspace(t_span[0], t_span[1], 100))

    return solution


