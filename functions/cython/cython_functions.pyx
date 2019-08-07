"""
Created on Mon Jul 29 11:13:45 2019

@author: cantaro86
"""

import numpy as np
from scipy.linalg import norm
cimport numpy as np
cimport cython

cdef np.float64_t distance2(np.float64_t[:] a, np.float64_t[:] b, unsigned int N):
    cdef np.float64_t dist = 0
    cdef unsigned int i    
    for i in range(N):
        dist += (a[i] - b[i]) * (a[i] - b[i])
    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
def SOR(np.float64_t aa, 
        np.float64_t bb, np.float64_t cc, 
        np.float64_t[:] b, 
        np.float64_t w=1, np.float64_t eps=1e-10, unsigned int N_max = 500):
    
    cdef unsigned int N = b.size
    
    cdef np.float64_t[:] x0 = np.ones(N, dtype=np.float64)          # initial guess
    cdef np.float64_t[:] x_new = np.ones(N, dtype=np.float64)      # new solution

    
    cdef unsigned int i, k
    cdef np.float64_t S
    
    for k in range(1,N_max+1):
        for i in range(N):
            if (i==0):
                S = cc * x_new[1]
            elif (i==N-1):
                S = aa * x_new[N-2]
            else:
                S = aa * x_new[i-1] + cc * x_new[i+1]
            x_new[i] = (1-w)*x_new[i] + (w/bb) * (b[i] - S)  
        if distance2(x_new, x0, N) < eps*eps:
            return x_new
        x0[:] = x_new
        if k==N_max:
            print("Fail to converge in {} iterations".format(k))
            return x_new
