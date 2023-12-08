
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
    uf         : input needed by eval_f
    b          : right hand side of the linear system to be solved
    tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
    MaxItersGCR: maximum number of iterations before giving up
    epsMF      : finite difference perturbation for Matrix Free directional derivative

    """
    x = np.zeros_like(b)
    # Set the initial residual to b - Ax^0 = b
    r = b
    r_norms = [np.linalg.norm(r)]
    
    k = -1
    p = []
    Ap = []
    while (r_norms[k]/r_norms[0] > tolrGCR) & (k <= MaxItersGCR):
        k = k + 1
        # Use the residual as the first guess for the ne search direction 
        # and computer its image
        p.append(r)
        
        #  The following three lines are an approximation for Ap(:, k) = A * p(:,k);
        epsilon = 1e-8
        fepsMF  = evalf(xf+epsilon*p[k],pf)
        f       = evalf(xf,pf)
        Ap.append((fepsMF - f ) / epsilon)
        
        # Make the new Ap vector orthogonal to the previous Ap vectors,
        # and the p vectors A^TA orthogonal to the previous p vectors.
        # Notice that if you know A is symmetric
        # you can save computation by limiting the for loop to just j=k-1
        # however if you need relative accuracy better than  1e-10
        # it might be safer to keep full orthogonalization even for symmetric A
        if k >1:
            for j in range(1,k):
                beta = Ap[k].T @ Ap[j]
                p[k]  =  p[k] - beta @ p[j]
                Ap[k] = Ap[k] - beta @ Ap[j]

            
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

        # Print the norm during the iteration
        # fprintf('||r||=%g i=%d\n', norms(k+1), k+1);
        
        return x, r_norms/r_norms[0]
    

# %%
