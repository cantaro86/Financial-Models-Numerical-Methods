#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:43:12 2019

@author: cantaro86
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt


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


def rolling_regression_test(X, Y, rolling_window, training_size):
    """ Rolling regression in the test set """
    
    rolling_beta = []
    for i in range( len(X)-training_size ):
        beta_temp,_ ,_ ,_ ,_  = ss.linregress(X[1+i+training_size-rolling_window : 1+i+training_size],
                                          Y[1+i+training_size-rolling_window : 1+i+training_size])
        rolling_beta.append(beta_temp)
    return rolling_beta



def RTS_smoother(X, Y, alpha, beta0, var_eta, var_eps, P0):
    """
    Kalman smoother for the beta estimation. It uses the Rauch–Tung–Striebel (RTS) algorithm. 
    """
    betas, Ps = Kalman_beta(X, Y, alpha, beta0, var_eta, var_eps, P0 = 10, likelihood=False, Rsq=False)
    
    betas_smooth = np.zeros_like(betas)
    Ps_smooth = np.zeros_like(Ps)
    betas_smooth[-1] = betas[-1] 
    Ps_smooth[-1] = Ps[-1]
    
    for k in range( len(X)-2,-1,-1):
        C = Ps[k]/(Ps[k]+var_eta)  
        betas_smooth[k] = betas[k] + C*( betas_smooth[k+1] - betas[k] )
        Ps_smooth[k] = Ps[k] + C**2 *( Ps_smooth[k+1] - (Ps[k]+var_eta) )

    return betas_smooth, Ps_smooth 



def plot_betas(X, Y, true_rho, rho_err, var_eta=None, training_size = 250, rolling_window = 50):
    """
    This function performs all the calculations necessary for the plot of:
        - Kalman beta
        - Rolling beta
        - Smoothed beta
    Input:
        X, Y:  predictor and response variables
        true_rho: (an array) the true value of the autocorrelation coefficient
        rho_err: (an array) rho with model error
        var_eta: If None, MLE estimator is used
        training_size: size of the training set
        rolling window: for the computation of the rolling regression
    """
    
    X_train = X[:training_size] 
    X_test = X[training_size:] 
    Y_train = Y[:training_size] 
    Y_test = Y[training_size:] 
    beta_tr, alpha_tr, _ ,_ ,_  = ss.linregress(X_train, Y_train)
    resid_tr = Y_train - beta_tr * X_train - alpha_tr
    var_eps = resid_tr.var(ddof=2)
    
    if var_eta is None:
        var_eta, var_eps = calibrate_Kalman_MLE(X_train, Y_train, alpha_tr, beta_tr, 10).x
        if var_eta < 1e-8:
            print(" MLE FAILED.  var_eta set equal to var_eps")
            var_eta = var_eps
        else:
            print("MLE parameters")
        
    print("var_eta = ", var_eta)
    print("var_eps = ", var_eps)
    
    # last values of beta and P in the training set. Are used as initial values in the test set 
    _, beta_last, P_last = Kalman_beta(X_train, Y_train, 0, beta_tr, var_eta, var_eps, 10, True)
    #   Kalman
    betas_KF, Ps_KF = Kalman_beta(X_test, Y_test, 0, beta_last, var_eta, var_eps, P_last)
    # Rolling betas
    rolling_beta = rolling_regression_test(X, Y, rolling_window, training_size)
    # Smoother
    betas_smooth, Ps_smooth = RTS_smoother(X_test, Y_test, 0, beta_last, var_eta, var_eps, P_last)
 
    plt.figure(figsize=(16,6))
    plt.plot(betas_KF, color="royalblue", label="Kalman filter betas")
    plt.plot(rolling_beta, color="orange", label="Rolling beta, window={}".format(rolling_window))
    plt.plot( betas_smooth, label="RTS smoother", color="maroon" )
    plt.plot(rho_err[training_size+1:], color="springgreen", marker='o', linestyle="None", label="rho with model error")
#    x = np.array(range( len(X)-1 )) x[:(len(X)-1-training_size)],
    plt.plot( true_rho[training_size+1:], color="black", alpha=2, label="True rho")
    plt.fill_between(x=range(len(betas_KF)) ,y1=betas_KF + np.sqrt(Ps_KF), y2=betas_KF - np.sqrt(Ps_KF), 
                     alpha=0.5, linewidth=2, color='seagreen', label="Kalman Std Dev: $\pm 1 \sigma$")
    plt.legend(); plt.title("Kalman results")
    
    print("MSE Rolling regression: ", np.mean((np.array(rolling_beta) - true_rho[training_size+1:]  )**2) )
    print("MSE Kalman Filter: ", np.mean((betas_KF - true_rho[training_size+1:])**2) )
    print("MSE RTS Smoother: ", np.mean((betas_smooth - true_rho[training_size+1:] )**2) )
