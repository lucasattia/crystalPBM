#%%

import numpy as np
from tgcr_matrixfree import tgcr_MatrixFree




def NewtonGCR(x0,evalf,p,errf,errDeltax,relDeltax,MaxIter,tolrGCR,epsMF):
    """
        uses Newton Method to solve the VECTOR nonlinear system f(x)=0
        INPUTS
        x0         is the initial guess for Newton iteration
        p          is a structure containing all parameters needed to evaluate f( )
        u          contains values of inputs 
        eval_f     is a text string with name of function evaluating f for a given x 
        eval_Jf    is a text string with name of function evaluating Jacobian of f at x (i.e. derivative in 1D)
        FiniteDifference = 1 forces the use of Finite Difference Jacobian instead of given eval_Jf
        errF       = absolute equation error: how close do you want f to zero?
        errDeltax  = absolute output error:   how close do you want x?
        relDeltax  = relative output error:   how close do you want x in perentage?
        note: 		 declares convergence if ALL three criteria are satisfied 
        MaxItersGCR= maximum number of iterations allowed
        visualize  = 1 shows intermediate results
        tolrGCR   = residual tolerance target for GCR
        epsMF     (OPTIONAL) perturbation for directional derivative for matrix-free Newton-GCR
        

    """

    inf = np.inf


    N           = len(x0) 
    #MaxItersGCR = max(N,round(N*0.2))      
    MaxItersGCR = N*1.1  

    k           = 0                          # Newton iteration index
    X = np.zeros((len(x0),MaxIter+5))
    # print("type(x0)",type(x0))
    X[:,k]           = x0
   
    # print("x0 in newtonGCR",x0)
    f           = evalf(x0,p) 
    errf_k 	   = np.linalg.norm(f,inf) 


    errDeltax_k = inf 
    relDeltax_k = inf 

    # print("shape of x0 in newton", x0.shape)

    while k<=MaxIter and (errf_k>errf or errDeltax_k>errDeltax or relDeltax_k>relDeltax):
    
       
        Deltax, _ = tgcr_MatrixFree(evalf,X[:,k],p,-f,tolrGCR,MaxItersGCR,epsMF)  # uses matrix-free
        # print("Deltax",Deltax)
        # print("size of X[:,k]",X[:,k].shape)
        # print("type(X[:,k])",type(X[:,k]))
        X[:,k+1]=X[:,k]+ np.array(Deltax )
        # print("X[:,k]",X[:,k])
        # print("X[:,k]",X[:,k][0])
        
       

        k           = k+1 
        # print("X",X)
        # print("X[k]",X[k])
        f           = evalf(X[:,k],p)
        
        # print("size of f in newton",f.shape)
        errf_k      = np.linalg.norm(f,inf) 
        # print("errf_k",errf_k)
        errDeltax_k = np.linalg.norm(Deltax,inf)
        # print("errDeltax_k",errDeltax_k) 
        relDeltax_k = np.linalg.norm(Deltax,inf)/max(abs(X[:,k])) 
        # print("relDeltax_k",relDeltax_k)
 

    x = X[:,0:k]   
    # print('final x  size in newton', x.shape)
    # print("type(x), output",type(x))

    # returning the number of iterations with ACTUAL computation
    # i.e. exclusing the given initial guess
    iterations = k-1  


    if errf_k<=errf and errDeltax_k<=errDeltax and relDeltax_k<=relDeltax :
        converged = 1 
        # print('Newton converged in iterations\n', iterations) 
    else:
        converged=0 
        print("errf_k",errf_k)
        print("errDeltax_k",errDeltax_k)
        print("relDeltax_k",relDeltax_k)
        print('Newton did NOT converge! Maximum Number of Iterations reached\n') 
   

    return x,converged,errf_k,errDeltax_k,relDeltax_k,iterations,X





   
  
