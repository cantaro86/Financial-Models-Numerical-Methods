#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:06:01 2019

@author: cantaro86
"""

import numpy as np
import scipy.stats as ss
from functions.probabilities import VG_pdf
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess
import pandas as pd


class Diffusion_process():
    """
    Class for the diffusion process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    mu = constant drift 
    """
    def __init__(self, r=0.1, sig=0.2, mu=0.1):
        self.r = r
        self.mu = mu
        if (sig<=0):
            raise ValueError("sig must be positive")
        else:
            self.sig = sig

    def exp_RV(self, S0, T, N):
        W = ss.norm.rvs( (self.r-0.5*self.sig**2)*T , np.sqrt(T)*self.sig, N )
        S_T = S0 * np.exp(W)
        return S_T.reshape((N,1))



class Merton_process():
    """
    Class for the Merton process:
    r = risk free constant rate
    sig = constant diffusion coefficient
    lam = jump activity
    muJ = jump mean
    sigJ = jump standard deviation
    """
    def __init__(self, r=0.1, sig=0.2, lam = 0.8, muJ = 0, sigJ = 0.5):
        self.r = r
        self.lam = lam
        self.muJ = muJ
        if (sig<0 or sigJ<0):
            raise ValueError("sig and sigJ must be positive")
        else:
            self.sig = sig
            self.sigJ = sigJ
        
        # moments
        self.var = self.sig**2 + self.lam * self.sigJ**2 + self.lam * self.muJ**2
        self.skew = self.lam * (3* self.sigJ**2 * self.muJ + self.muJ**3) / self.var**(1.5)
        self.kurt = self.lam * (3* self.sigJ**3 + 6 * self.sigJ**2 * self.muJ**2 + self.muJ**4) / self.var**2
     
    def exp_RV(self, S0, T, N):
        m = self.lam * (np.exp(self.muJ + (self.sigJ**2)/2) -1)    # coefficient m
        W = ss.norm.rvs(0, 1, N)              # The normal RV vector  
        P = ss.poisson.rvs(self.lam*T, size=N)    # Poisson random vector (number of jumps)
        Jumps = np.asarray([ss.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P ]) # Jumps vector
        S_T = S0 * np.exp( (self.r - 0.5*self.sig**2 -m )*T + np.sqrt(T)*self.sig*W + Jumps )     # Martingale exponential Merton
        return S_T.reshape((N,1))
 

       
class VG_process():
    """
    Class for the Variance Gamma process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are: 
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process 
    """
    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.c = self.r
        self.theta = theta
        self.kappa = kappa
        if (sigma<0):
            raise ValueError("sigma must be positive")
        else:
            self.sigma = sigma
            
        # moments
        self.mean = self.c + self.theta
        self.var = self.sigma**2 + self.theta**2 * self.kappa 
        self.skew = (2 * self.theta**3 * self.kappa**2 + 3*self.sigma**2 * self.theta * self.kappa) / (self.var**(1.5)) 
        self.kurt = ( 3*self.sigma**4 * self.kappa +12*self.sigma**2 * self.theta**2 \
                     * self.kappa**2 + 6*self.theta**4 * self.kappa**3 ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        w = -np.log(1 - self.theta * self.kappa - self.kappa/2 * self.sigma**2 ) /self.kappa    # coefficient w
        rho = 1 / self.kappa
        G = ss.gamma(rho * T).rvs(N) / rho     # The gamma RV
        Norm = ss.norm.rvs(0,1,N)              # The normal RV  
        VG = self.theta * G + self.sigma * np.sqrt(G) * Norm     # VG process at final time G
        S_T = S0 * np.exp( (self.r-w)*T + VG )                 # Martingale exponential VG       
        return S_T.reshape((N,1))
    
    def path(self, T=1, N=10000, paths=1):
        """
        Creates Variance Gamma paths    
        N = number of time points (time steps are N-1)
        paths = number of generated paths
        """
        dt = T/(N-1)          # time interval        
        X0 = np.zeros((paths,1))
        G = ss.gamma( dt/self.kappa, scale=self.kappa).rvs( size=(paths,N-1) )     # The gamma RV
        Norm = ss.norm.rvs(loc=0, scale=1, size=(paths,N-1))                       # The normal RV  
        increments = self.c*dt + self.theta * G + self.sigma * np.sqrt(G) * Norm
        X = np.concatenate((X0,increments), axis=1).cumsum(1)
        return X
 
    
    def fit_from_data(self, data, dt=1, method="Nelder-Mead"):
        """
        Fit the 4 parameters of the VG process using MM (method of moments), Nelder-Mead, L-BFGS-B.
        data (array): datapoints
        dt (float):     is the increment time
        Returns (c,theta,sigma,kappa)
        """
        X = data
        sigma_mm =  np.std(X) / np.sqrt(dt)                                
        kappa_mm = dt * ss.kurtosis(X)/3                              
        theta_mm = np.sqrt(dt) * ss.skew(X) * sigma_mm / (3*kappa_mm)   
        c_mm     = np.mean(X)/dt - theta_mm
        
        def log_likely(x, data, T):
            return (-1) * np.sum( np.log( VG_pdf(data, T, x[0], x[1], x[2], x[3]) ))
        
        if method=="L-BFGS-B":
            if theta_mm<0:
                result = minimize(log_likely, x0=[c_mm,theta_mm,sigma_mm,kappa_mm], method='L-BFGS-B', args=(X,dt), tol=1e-8,
                     bounds=[[-0.5,0.5],[-0.6,-1e-15],[1e-15,1],[1e-15,2]])
            else:
                result = minimize(log_likely, x0=[c_mm,theta_mm,sigma_mm,kappa_mm], method='L-BFGS-B', args=(X,dt), tol=1e-8,
                     bounds=[[-0.5,0.5],[1e-15,0.6],[1e-15,1],[1e-15,2]])
            print(result.message)
        elif method=="Nelder-Mead":
            result = minimize(log_likely, x0=[c_mm,theta_mm,sigma_mm,kappa_mm], method='Nelder-Mead', args=(X,dt),
                              options={'disp':False, 'maxfev':3000}, tol=1e-8)
            print(result.message)
        elif "MM":
            self.c, self.theta, self.sigma, self.kappa = c_mm, theta_mm, sigma_mm, kappa_mm
            return
        self.c, self.theta, self.sigma, self.kappa = result.x
         
        
    
class Heston_process():
    """
    Class for the Heston process:
    r = risk free constant rate
    rho = correlation between stock noise and variance noise
    theta = long term mean of the variance process
    sigma = volatility coefficient of the variance process
    kappa = mean reversion coefficient for the variance process
    """
    def __init__(self, mu=0.1, rho=0, sigma=0.2, theta=-0.1, kappa=0.1):
        self.mu = mu
        if (np.abs(rho)>1):
            raise ValueError("|rho| must be <=1")
        self.rho = rho
        if (theta<0 or sigma<0 or kappa<0):
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.theta = theta
            self.sigma = sigma
            self.kappa = kappa            
    
    def path(self, S0, v0, N, T=1):
        """
        Produces one path of the Heston process.
        N = number of time steps
        T = Time in years
        Returns two arrays S (price) and v (variance). 
        """
        
        MU = np.array([0, 0])
        COV = np.matrix([[1, self.rho], [self.rho, 1]])
        W = ss.multivariate_normal.rvs( mean=MU, cov=COV, size=N-1 )
        W_S = W[:,0]                   # Stock Brownian motion:     W_1
        W_v = W[:,1]                   # Variance Brownian motion:  W_2

        # Initialize vectors
        T_vec, dt = np.linspace(0,T,N, retstep=True )
        dt_sq = np.sqrt(dt)
        
        X0 = np.log(S0)
        v = np.zeros(N)
        v[0] = v0
        X = np.zeros(N)
        X[0] = X0

        # Generate paths
        for t in range(0,N-1):
            v_sq = np.sqrt(v[t])
            v[t+1] = np.abs( v[t] + self.kappa*(self.theta - v[t])*dt + self.sigma * v_sq * dt_sq * W_v[t] )   
            X[t+1] = X[t] + (self.mu - 0.5*v[t])*dt + v_sq * dt_sq * W_S[t]
        
        return np.exp(X), v
    
    

class NIG_process():
    """
    Class for the Normal Inverse Gaussian process:
    r = risk free constant rate
    Using the representation of Brownian subordination, the parameters are: 
        theta = drift of the Brownian motion
        sigma = standard deviation of the Brownian motion
        kappa = variance of the of the Gamma process 
    """
    def __init__(self, r=0.1, sigma=0.2, theta=-0.1, kappa=0.1):
        self.r = r
        self.theta = theta
        if (sigma<0 or kappa<0):
            raise ValueError("sigma and kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa
            
        # moments
        self.var = self.sigma**2 + self.theta**2 * self.kappa 
        self.skew = (3 * self.theta**3 * self.kappa**2 + 3*self.sigma**2 * self.theta * self.kappa) / (self.var**(1.5)) 
        self.kurt = ( 3*self.sigma**4 * self.kappa +18*self.sigma**2 * self.theta**2 \
                     * self.kappa**2 + 15*self.theta**4 * self.kappa**3 ) / (self.var**2)

    def exp_RV(self, S0, T, N):
        lam = T**2 / self.kappa     # scale for the IG process
        mu_s = T / lam              # scaled mean
        w = ( 1 - np.sqrt( 1 - 2*self.theta*self.kappa -self.kappa*self.sigma**2) )/self.kappa 
        IG = ss.invgauss.rvs(mu=mu_s, scale=lam, size=N)         # The IG RV
        Norm = ss.norm.rvs(0,1,N)                                # The normal RV  
        X = self.theta * IG + self.sigma * np.sqrt(IG) * Norm    # NIG random vector
        S_T = S0 * np.exp( (self.r-w)*T + X )                    # exponential dynamics        
        return S_T.reshape((N,1))



class GARCH():
    """
    Class for the GARCH(1,1) process. Variance process:
        
        V(t) = omega + alpha R^2(t-1) + beta V(t-1) 
        
        VL:  Unconditional variance >=0  
        alpha: coefficient > 0   
        beta:  coefficient > 0
        gamma = 1 - alpha - beta
        omega = gamma*VL
    """
    def __init__(self, VL=0.04, alpha=0.08, beta=0.9):
        if (VL<0 or alpha<=0 or beta<=0):
            raise ValueError("VL>=0, alpha>0 and beta>0")
        else:
            self.VL = VL
            self.alpha = alpha
            self.beta = beta
        self.gamma = 1 - self.alpha - self.beta
        self.omega = self.gamma * self.VL
        
    def path(self, N=1000):
        """
        Generates a path with N points. 
        Returns the return process R and the variance process var
        """
        eps = ss.norm.rvs(loc=0, scale=1, size=N)
        R = np.zeros_like(eps)
        var = np.zeros_like(eps)
        for i in range(N):
            var[i] = self.omega + self.alpha*R[i-1]**2 + self.beta*var[i-1]
            R[i] =  np.sqrt(var[i]) * eps[i]
        return R, var
   
    
    def fit_from_data(self, data, disp=True):
        """
        MLE estimator for the GARCH
        """
        # Automatic re-scaling:  1. the solver has problems with positive derivative in linesearch. 
        #                        2. the log has overflows using small values 
        n = np.floor(np.log10( np.abs(data.mean()) ))
        R = data / 10**n
    
        # initial guesses
        a0 = 0.05
        b0 = 0.9
        g0 = 1-a0-b0
        w0 = g0*np.var(R)

        # bounds and constraint
        bounds = ((0, None), (0, 1), (0, 1))
        def sum_small_1(x):
            return 1-x[1]-x[2]
        cons = ({"fun": sum_small_1, "type": "ineq"})  
        
        def log_likely(x):
            var = R[0]**2             # initial variance
            N = len(R)
            log_lik=0
            for i in range(1,N):
                var = x[0] + x[1]*R[i-1]**2 + x[2]*var    # variance update 
                log_lik += -np.log(var) - ( R[i]**2 / var )  
            return (-1)*log_lik
    
        result = minimize(log_likely, x0=[w0,a0,b0], method='SLSQP', bounds=bounds,
                          constraints=cons, tol=1e-8, options={"maxiter":150})
        print(result.message)
        self.omega = result.x[0] * 10**(2*n)  
        self.alpha, self.beta = result.x[1:]
        self.gamma = 1-self.alpha-self.beta
        self.VL = self.omega / self.gamma

        if disp==True:            
            hess = approx_hess(result.x, log_likely )    # hessian by finite differences
            se = np.sqrt(np.diag(np.linalg.inv(hess) ) ) # standard error
            cv = ss.norm.ppf(1.0 - 0.05 / 2.0)           # alpha=0.05 
            p_val = ss.norm.sf(np.abs( result.x / se))   # survival function
            
            df = pd.DataFrame(index=["omega", "alpha", "beta"])
            df["Params"] = result.x
            df["SE"] = se 
            df["P-val"] = p_val 
            df["95% CI lower"] = result.x - cv * se
            df["95% CI upper"] = result.x + cv * se
            df.loc["omega", ["Params", "SE", "95% CI lower", "95% CI upper"]] *= 10**(2*n)
            print(df)
    
    
    def log_likelihood(self, R, last_var=True):
        """
        Computes the log-likelihood and optionally returns the last value of the variance
        """
        var = R[0]**2             # initial variance
        N = len(R)
        log_lik=0
        log_2pi = np.log(2 * np.pi)
        for i in range(1,N):
            var = self.omega + self.alpha*R[i-1]**2 + self.beta*var    # variance update 
            log_lik += 0.5 * ( -log_2pi -np.log(var) - ( R[i]**2 / var )  )
        if last_var==True:  
            return log_lik, var
        else:
            return log_lik
        
    def generate_var(self, R, R0, var0):
        """
        generate the variance process.
        R (array): return array
        R0: initial value of the returns
        var0: initial value of the variance
        """
        N = len(R)
        var = np.zeros(N)
        var[0] = self.omega + self.alpha*(R0**2) + self.beta*var0
        for i in range(1,N):
            var[i] = self.omega + self.alpha*R[i-1]**2 + self.beta*var[i-1]
        return var
    
    

class OU_process():
    """
    Class for the OU process:
    theta = long term mean
    sigma = diffusion coefficient
    kappa = mean reversion coefficient
    """
    def __init__(self, sigma=0.2, theta=-0.1, kappa=0.1):
        self.theta = theta
        if (sigma<0 or kappa<0):
            raise ValueError("sigma,theta,kappa must be positive")
        else:
            self.sigma = sigma
            self.kappa = kappa            
    
    def path(self, X0=0, T=1, N=10000, paths=1):
        """
        Produces a matrix of OU process:  X[paths,N]
        X0 = starting point
        N = number of time steps
        T = Time in years
        paths = number of paths
        """
        
        T_vec, dt = np.linspace(0, T, N, retstep=True ) 
        X = np.zeros((paths,N))
        X[:,0] = X0
        W = ss.norm.rvs( loc=0, scale=1, size=(paths,N-1) )

        std_dt = np.sqrt( self.sigma**2 /(2*self.kappa) * (1-np.exp(-2*self.kappa*dt)) )
        for t in range(0,N-1):
            X[:,t+1] = self.theta + np.exp(-self.kappa*dt)*(X[:,t]-self.theta) + std_dt * W[:,t]        
                
        return X
