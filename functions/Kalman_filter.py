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


class Kalman_regression():
    """ Kalman Filter algorithm for the linear regression beta estimation. Alpha is assumed constant. 
    
    INPUT:
    X = predictor variable. ndarray, Series or DataFrame. 
    Y = response variable.
    alpha0 = constant alpha. The regression intercept.
    beta0 = initial beta. 
    var_eta = variance of process error
    var_eps = variance of measurement error
    P0 = initial covariance of beta
    """
    
    def __init__(self, X, Y, alpha0=None, beta0=None, var_eta=None, var_eps=None, P0 = 10):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.var_eta = var_eta
        self.var_eps = var_eps
        self.P0 = P0
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.loglikelihood = None
        self.R2_pre_fit = None
        self.R2_post_fit = None
        
        self.betas = None
        self.Ps = None
        
        if self.alpha0==None or self.beta0==None or self.var_eps==None:
            self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()
            print("alpha0, beta0 and var_eps initialized by OLS")
            
####################  enforce X and Y to be numpy arrays ######################    
#    @property
#    def X(self):
#        return self._X
#    @X.setter
#    def X(self, value):
#        if not isinstance(value, np.ndarray):
#            raise TypeError('X must be a numpy array')
#        self._X = value
#    
#    @property
#    def Y(self):
#        return self._Y
#    @Y.setter
#    def Y(self, value):
#        if not isinstance(value, np.ndarray):
#            raise TypeError('Y must be a numpy array')
#        self._Y = value
###############################################################################    
    
    def get_OLS_params(self):
        """ Returns the OLS alpha, beta and sigma^2 (variance of epsilon)
            Y = alpha + beta * X + epsilon
        """
        beta, alpha, _ ,_ ,_  = ss.linregress(self.X, self.Y)
        resid = self.Y - beta * self.X - alpha
        sig2 = resid.var(ddof=2)
        return alpha, beta, sig2
    
    def set_OLS_params(self):
        self.alpha0, self.beta0, self.var_eps = self.get_OLS_params()
        
    
    def run(self, X=None, Y=None, var_eta=None, var_eps=None):
        """ 
            Run the Kalman Filter
        """
        
        if (X is None) and (Y is None):
            X = self.X
            Y = self.Y
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        N = len(X)
        if len(Y) != N:
            raise ValueError("Y and X must have same length") 
    
        if var_eta is not None:
            self.var_eta = var_eta
        if var_eps is not None:
            self.var_eps = var_eps
        if self.var_eta==None:
            raise ValueError("var_eta is None")
        
        betas = np.zeros_like(X)
        Ps = np.zeros_like(X)
        res_pre = np.zeros_like(X)       # pre-fit residuals
        
        Y = Y - self.alpha0              # re-define Y
        P = self.P0
        beta = self.beta0
        
        log_2pi = np.log(2 * np.pi)
        loglikelihood = 0
    
        for k in range(N):
            # Prediction
            beta_p = beta                  # predicted beta 
            P_p = P + self.var_eta              # predicted P

            # ausiliary variables
            r = Y[k] - beta_p * X[k]
            S = P_p * X[k]**2 + self.var_eps
            KG = X[k] * P_p / S            # Kalman gain
        
            # Update
            beta = beta_p + KG * r
            P = P_p * (1 - KG * X[k])

            loglikelihood += 0.5 * ( -log_2pi - np.log(S) - (r**2/S) )      
        
            betas[k] = beta
            Ps[k] = P
            res_pre[k] = r
            
        res_post = Y - X * betas   # post fit residuals 
        sqr_err = Y - np.mean(Y)                             
        R2_pre = 1 - ( res_pre @ res_pre )/(sqr_err @ sqr_err)
        R2_post = 1 - ( res_post @ res_post )/(sqr_err @ sqr_err)
        
        self.loglikelihood = loglikelihood
        self.R2_post_fit = R2_post
        self.R2_pre_fit = R2_pre
        
        self.betas = betas 
        self.Ps = Ps
        


    def calibrate_MLE(self):
        """ Returns the result of the MLE calibration for the Beta Kalman filter, using the L-BFGS-B method. 
        The calibrated parameters are var_eta and var_eps. 
        X, Y          = Series, array, or DataFrame for the regression 
        alpha_tr      = initial alpha 
        beta_tr       = initial beta 
        var_eps_ols   = initial guess for the errors
        """

        def minus_likelihood(c):
            """ Function to minimize in order to calibrate the kalman parameters: var_eta and var_eps. """
            self.var_eps = c[0]
            self.var_eta = c[1]
            self.run()
            return -1 * self.loglikelihood
        
        result = minimize(minus_likelihood, x0=[self.var_eps, self.var_eps], 
                          method='L-BFGS-B', bounds=[[1e-15,None],[1e-15,None]], tol=1e-6)
        
        if result.success == True:
            self.beta0 = self.betas[-1]
            self.P0 = self.Ps[-1]
            self.var_eps = result.x[0] 
            self.var_eta = result.x[1]
            print("Optimization converged successfully")
            print( "var_eps = {}, var_eta = {}".format(result.x[0], result.x[1]) )


    def calibrate_R2(self, mode="pre-fit"):
        """ Returns the result of the R2 calibration for the Beta Kalman filter, using the L-BFGS-B method. 
        The calibrated parameters is var_eta
        """

        def minus_R2(c):
            """ Function to minimize in order to calibrate the kalman parameters: var_eta and var_eps. """
            self.var_eta = c
            self.run()
            if mode == "pre-fit":
                return -1 * self.R2_pre_fit
            elif mode == "post-fit":
                return -1 * self.R2_post_fit
        
        result = minimize(minus_R2, x0=[self.var_eps], 
                      method='L-BFGS-B', bounds=[[1e-15,1]], tol=1e-6)
        
        if result.success == True:
            self.beta0 = self.betas[-1]
            self.P0 = self.Ps[-1]
            self.var_eta = result.x[0]
            print("Optimization converged successfully")
            print( "var_eta = {}".format(result.x[0]) )


    def RTS_smoother(self, X, Y):
        """
        Kalman smoother for the beta estimation. It uses the Rauch–Tung–Striebel (RTS) algorithm. 
        """
        self.run(X,Y)
        betas, Ps = self.betas, self.Ps 
        
        betas_smooth = np.zeros_like(betas)
        Ps_smooth = np.zeros_like(Ps)
        betas_smooth[-1] = betas[-1] 
        Ps_smooth[-1] = Ps[-1]
        
        for k in range( len(X)-2,-1,-1):
            C = Ps[k]/(Ps[k]+self.var_eta)  
            betas_smooth[k] = betas[k] + C*( betas_smooth[k+1] - betas[k] )
            Ps_smooth[k] = Ps[k] + C**2 *( Ps_smooth[k+1] - (Ps[k]+self.var_eta) )
            
        return betas_smooth, Ps_smooth 



def rolling_regression_test(X, Y, rolling_window, training_size):
    """ Rolling regression in the test set """
    
    rolling_beta = []
    for i in range( len(X)-training_size ):
        beta_temp,_ ,_ ,_ ,_  = ss.linregress(X[1+i+training_size-rolling_window : 1+i+training_size],
                                          Y[1+i+training_size-rolling_window : 1+i+training_size])
        rolling_beta.append(beta_temp)
    return rolling_beta






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
    #beta_tr, alpha_tr, _ ,_ ,_  = ss.linregress(X_train, Y_train)
    #resid_tr = Y_train - beta_tr * X_train - alpha_tr
    #var_eps = resid_tr.var(ddof=2)
    
    KR = Kalman_regression(X_train, Y_train)
    var_eps = KR.var_eps
    
    if var_eta is None:
        KR.calibrate_MLE()
        var_eta, var_eps = KR.var_eta, KR.var_eps
        if var_eta < 1e-8:
            print(" MLE FAILED.  var_eta set equal to var_eps")
            var_eta = var_eps
        else:
            print("MLE parameters")
        
    print("var_eta = ", var_eta)
    print("var_eps = ", var_eps)
    
    KR.run(X_train, Y_train, var_eps=var_eps, var_eta=var_eta)
    KR.beta0, KR.P0 = KR.betas[-1], KR.Ps[-1]
    KR.run(X_test, Y_test)
    #   Kalman
    betas_KF, Ps_KF = KR.betas, KR.Ps
    # Rolling betas
    rolling_beta = rolling_regression_test(X, Y, rolling_window, training_size)
    # Smoother
    betas_smooth, Ps_smooth = KR.RTS_smoother(X_test, Y_test)
 
    plt.figure(figsize=(16,6))
    plt.plot(betas_KF, color="royalblue", label="Kalman filter betas")
    plt.plot(rolling_beta, color="orange", label="Rolling beta, window={}".format(rolling_window))
    plt.plot( betas_smooth, label="RTS smoother", color="maroon" )
    plt.plot(rho_err[training_size+1:], color="springgreen", marker='o', linestyle="None", label="rho with model error")
    plt.plot( true_rho[training_size+1:], color="black", alpha=2, label="True rho")
    plt.fill_between(x=range(len(betas_KF)) ,y1=betas_KF + np.sqrt(Ps_KF), y2=betas_KF - np.sqrt(Ps_KF), 
                     alpha=0.5, linewidth=2, color='seagreen', label="Kalman Std Dev: $\pm 1 \sigma$")
    plt.legend(); plt.title("Kalman results")
    
    print("MSE Rolling regression: ", np.mean((np.array(rolling_beta) - true_rho[training_size+1:]  )**2) )
    print("MSE Kalman Filter: ", np.mean((betas_KF - true_rho[training_size+1:])**2) )
    print("MSE RTS Smoother: ", np.mean((betas_smooth - true_rho[training_size+1:] )**2) )
