#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:43:12 2019

@author: cantaro86
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd

def Kalman_beta(X, Y, alpha, beta0, var_eta, var_eps, P0 = 10, likelihood=False, Rsq=False):
    """ Kalman Filter algorithm for the linear regression beta estimation. Alpha is assumed constant. 
    
    INPUT:
    X = predictor variable. ndarray, Series or DataFrame. 
    Y = response variable.
    alpha = constant alpha. The regression intercept.
    beta0 = initial beta. 
    var_eta = variance of process error
    var_eps = variance of measurement error
    P0 = initial covariance of beta
    likelihood = boolean
    Rsq = boolean   
    
    OUTPUT:
    If likelihood is false, it returns a list of betas and a list of variances P. 
    If likelihood is true, it returns the log-likelihood and the last values of beta and P.
    If Rsq is True, it returns the R squared
    """
    
    # it checks only X. 
    if  ( (not isinstance(X, np.ndarray)) and (not isinstance(X, pd.core.frame.DataFrame))
                        and (not isinstance(X, pd.core.series.Series)) ):
        raise ValueError("invalid type.")
 
    if ( isinstance(X, pd.core.series.Series) or isinstance(X, pd.core.frame.DataFrame) ):
        X = X.values
        Y = Y.values
    
    N = len(X)
    assert len(Y) == N
    
    betas = np.zeros_like(X)
    Ps = np.zeros_like(X)
        
    Y = Y - alpha         # re-define Y
    P = P0
    beta = beta0
    log_2pi = np.log(2 * np.pi)
    loglikelihood = 0
    
    
    for k in range(N):
        # Prediction
        beta_p = beta                  # predicted beta 
        P_p = P + var_eta              # predicted P

        # ausiliary variables
        r = Y[k] - beta_p * X[k]
        S = P_p * X[k]**2 + var_eps
        KG = X[k] * P_p / S            # Kalman gain
        
        # Update
        beta = beta_p + KG * r
        P = P_p * (1 - KG * X[k])

        loglikelihood += 0.5 * ( -log_2pi - np.log(S) - (r**2/S) )      
        
        if likelihood == False:
            betas[k] = beta
            Ps[k] = P
    
    beta_last = beta
    P_last = P
    
    if likelihood == False and Rsq == False:
        return betas, Ps
    elif (likelihood == True and Rsq == False):
        return loglikelihood, beta_last, P_last
    else:
        res = Y - X * betas   # post fit residuals 
        sqr_err = Y - np.mean(Y)                             
        R2 = 1 - ( res @ res )/(sqr_err @ sqr_err)
        return R2



def calibrate_Kalman_MLE(X, Y, alpha_tr, beta_tr, var_eps_ols):
    """ Returns the result of the MLE calibration for the Beta Kalman filter, using the L-BFGS-B method. 
    The calibrated parameters are var_eta and var_eps. 
    X, Y          = Series, array, or DataFrame for the regression 
    alpha_tr      = initial alpha 
    beta_tr       = initial beta 
    var_eps_ols   = initial guess for the errors
    """

    def minus_likelihood(c):
        """ Function to minimize in order to calibrate the kalman parameters: var_eta and var_eps. """
        loglik, _, _ = Kalman_beta(X, Y, alpha_tr, beta_tr, c[0], c[1], 1, True)
        return -loglik
        
    result = minimize(minus_likelihood, x0=[var_eps_ols,var_eps_ols], 
                      method='L-BFGS-B', bounds=[[1e-15,None],[1e-15,None]], tol=1e-8)
    return result



def calibrate_Kalman_R2(X, Y, alpha_tr, beta_tr, var_eps_ols):
    """ Returns the result of the R2 calibration for the Beta Kalman filter, using the L-BFGS-B method. 
    The calibrated parameters is var_eta
    """

    def minus_R2(c):
        """ Function to minimize in order to calibrate the kalman parameters: var_eta and var_eps. """
        R2 = Kalman_beta(X, Y, alpha_tr, beta_tr, c, var_eps_ols, 1, False, True)
        return -R2
        
    result = minimize(minus_R2, x0=[var_eps_ols], 
                      method='L-BFGS-B', bounds=[[1e-15,1]], tol=1e-8)
    return result

