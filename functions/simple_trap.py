import numpy as np
from evalf import evalf
from implicit_trap import trapezoidal
from trapezoidalMatrixFree import trapezoidalMatrixFree
from jacobian import calc_jac

def simple_trap(x0,t_vec,p):
    f= lambda x: evalf(x,p)
   
    return trapezoidal(f, x0, t_start=t_vec[0], t_end=t_vec[-1], alpha=5e-3, jf= lambda x: calc_jac(x, p, eps=1e-8), e_f =1e-8, e_delta_x=1e-8, e_x_rel=1e-8, maxiter=100)

def simple_matrix_free_trap(x0,t_vec,p):
    alpha= [1e-1, 5e-1, 2]
    return trapezoidalMatrixFree(evalf,x0, alpha, t_vec[0], T=t_vec[-1], p=p, errf=1e-3, errDeltax=1e-3, relDeltax=1e-3, MaxIter=40,tolrGCR=1e-2,epsMF=1e-6)