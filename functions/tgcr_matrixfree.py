
#%%

# import jax.numpy as np
import numpy as np



def tgcr_MatrixFree(evalf, xf, pf, b, tolrGCR, MaxItersGCR, epsMF):
    """
    using a matrix-free (i.e. matrix-implicit) technique
    INPUTS
    eval_f     : name of the function that evaluates f(xf,pf,uf)
    xf         : state vector where to evaluate the Jacobian [df/dx]
    pf         : structure containing parameters used by eval_f
    b          : right hand side of the linear system to be solved
    tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
    MaxItersGCR: maximum number of iterations before giving up
    epsMF      : finite difference perturbation for Matrix Free directional derivative

    """
    x = np.zeros_like(b)
    # Set the initial residual to b - Ax^0 = b
    r = b
    r_norms = [np.linalg.norm(r)]
    
    k = 0
    p = []
    Ap = []
    while (r_norms[k]/r_norms[0] > tolrGCR) & (k < MaxItersGCR):
        # Use the residual as the first guess for the ne search direction 
        # and computer its image
        p.append(r)
        
        #  The following three lines are an approximation for Ap(:, k) = A * p(:,k);
        epsilon = 2*epsMF*np.sqrt(1+np.linalg.norm(xf,np.inf))/np.linalg.norm(p[k],np.inf)
        fepsMF  = evalf(xf+epsilon*p[k],pf)
        f       = evalf(xf,pf)
        Ap.append((fepsMF - f ) / epsilon)
        
        # Make the new Ap vector orthogonal to the previous Ap vectors,
        # and the p vectors A^TA orthogonal to the previous p vectors.
        # Notice that if you know A is symmetric
        # you can save computation by limiting the for loop to just j=k-1
        # however if you need relative accuracy better than  1e-10
        # it might be safer to keep full orthogonalization even for symmetric A
        if k >0:
            for j in range(0,k):
                beta = Ap[k].T @ Ap[j]
                p[k]  =  p[k] - beta * p[j]
                Ap[k] = Ap[k] - beta * Ap[j]

            
        # Make the orthogonal Ap vector of unit length, and scale the
        # p vector so that A * p  is of unit length
        norm_Ap = np.linalg.norm(Ap[k])
        Ap[k] = Ap[k]/norm_Ap
        p[k] =  p[k]/norm_Ap
        
        # Determine the optimal amount to change x in the p direction
        # by projecting r onto Ap
        alpha = r.T @ Ap[k]
        
        # Update x and r
        x = x + alpha *  p[k]
        r = r - alpha * Ap[k]

        # Save the norm of r
        r_norms.append(np.linalg.norm(r))
        
        k = k + 1

    if r_norms[k] > (tolrGCR * r_norms[0]):
        print('GCR did NOT converge! Maximum Number of Iterations reached\n')
    # else:
    #     print("GCR converged in", k, "iterations")
    return x, r_norms/r_norms[0]
    
    
    