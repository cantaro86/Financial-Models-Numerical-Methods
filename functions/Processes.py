#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:06:01 2019

@author: cantaro86
"""

import numpy as np
import scipy.stats as ss


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
        W = ss.norm.rvs( (self.r-0.5*self.sig**2)*T , np.sqrt(T)*self.sig, N)
        S_T = S0 * np.exp(W)
        return S_T



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
        return S_T
 

       
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
        self.theta = theta
        self.kappa = kappa
        if (sigma<0):
            raise ValueError("sigma must be positive")
        else:
            self.sigma = sigma
            
        # moments
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
        return S_T
 
    
    
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
        return S_T



 