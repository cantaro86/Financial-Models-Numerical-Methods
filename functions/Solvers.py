#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:13:45 2019

@author: cantaro86
"""

import numpy as np
from scipy import sparse
from scipy.linalg import norm, solve_triangular
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError


def Thomas(A, b):
    """
    Solver for the linear equation Ax=b using the Thomas algorithm.
    It is a wrapper of the LAPACK function dgtsv.
    """
    
    D = A.diagonal(0)
    L = A.diagonal(-1)
    U = A.diagonal(1)
    
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    if A.shape[0] != b.shape[0]:
        raise ValueError('incompatible dimensions')
    
    dgtsv, = get_lapack_funcs(('gtsv',)) 
    du2,d,du,x,info = dgtsv(L, D, U, b)

    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" %
                          (info-1))    


def SOR(A, b, w=1, eps=1e-10, N_max = 100):
    """
    Solver for the linear equation Ax=b using the SOR algorithm.
          A = L + D + U 
    Arguments:
        L = Strict Lower triangular matrix
        D = Diagonal
        U = Strict Upper triangular matrix
        w = Relaxation coefficient
        eps = tollerance
        N_max = Max number of iterations
    """
    
    x0 = b.copy()            # initial guess
    
    if sparse.issparse(A):
        D = sparse.diags(A.diagonal())          # diagonal
        U = sparse.triu(A, k=1)                 # Strict U       
        L = sparse.tril(A, k=-1)                # Strict L
        DD = (w*L + D).toarray()
    else:
        D = np.eye(A.shape[0]) * np.diag(A)       # diagonal
        U = np.triu(A, k=1)                       # Strict U       
        L = np.tril(A, k=-1)                      # Strict L
        DD = (w*L + D)
    
    for i in range(1,N_max+1):
        x_new = solve_triangular( DD, (w*b - w*U@x0 - (w-1)*D@x0), lower=True)
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new
        if i==N_max:
            raise ValueError("Fail to converge in {} iterations".format(i))


def SOR2(A, b, w=1, eps=1e-10, N_max = 100):
    """
    Solver for the linear equation Ax=b using the SOR algorithm.
    It uses the coefficients and not the matrix multiplication. 
    """
    N = len(b)
    x0 = np.ones_like(b, dtype=np.float64) # initial guess
    x_new = np.ones_like(x0)               # new solution
    
    for k in range(1,N_max+1):
        for i in range(N):
            S = 0
            for j in range(N):
                if j != i:
                    S += A[i,j]*x_new[j]
            x_new[i] = (1-w)*x_new[i] + (w/A[i,i]) * (b[i] - S)  
                   
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new.copy()
        if k==N_max:
            print("Fail to converge in {} iterations".format(k))

