#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:18:39 2019

@author: cantaro86
"""

import numpy as np
import scipy as scp
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from time import time
import scipy.stats as ss
from functions.Solvers import Thomas
from functions.cython.cython_functions import SOR
from functions.CF import cf_normal
from functions.probabilities import Q1, Q2
from functools import partial



class BS_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info:  of type Diffusion_process. It contains (r,mu, sig) i.e.  interest rate, drift coefficient, diffusion coefficient
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sig = Process_info.sig       # diffusion coefficient
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        self.exp_RV = Process_info.exp_RV # function to generate solution of GBM
        
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff
        
        
    def payoff_f(self, S):
        if self.payoff == "call":
            Payoff = np.maximum( S - self.K, 0 )
        elif self.payoff == "put":    
            Payoff = np.maximum( self.K - S, 0 )  
        return Payoff
        
    
    @staticmethod
    def BlackScholes(payoff='call', S0=100., K=100., T=1., r=0.1, sigma=0.2 ):
        """ Black Scholes closed formula:
            payoff: call or put.
            S0: float.    initial stock/index level.
            K: float strike price.
            T: float maturity (in year fractions).  
            r: float constant risk-free short rate.
            sigma: volatility factor in diffusion term. """
   
        d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

        if payoff=="call":
            return S0 * ss.norm.cdf( d1 ) - K * np.exp(-r * T) * ss.norm.cdf( d2 )
        elif payoff=="put":
            return K * np.exp(-r * T) * ss.norm.cdf( -d2 ) - S0 * ss.norm.cdf( -d1 )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
    
    
    def closed_formula(self):
        """ 
        Black Scholes closed formula:
        """
        d1 = (np.log(self.S0/self.K) + (self.r + self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        d2 = (np.log(self.S0/self.K) + (self.r - self.sig**2 / 2) * self.T) / (self.sig * np.sqrt(self.T))

        if self.payoff=="call":
            return self.S0 * ss.norm.cdf( d1 ) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf( d2 )
        elif self.payoff=="put":
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf( -d2 ) - self.S0 * ss.norm.cdf( -d1 )
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
    
    
    
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K/self.S0)
        cf_GBM = partial(cf_normal, mu=( self.r - 0.5 * self.sig**2 )*self.T, sig=self.sig*np.sqrt(self.T))  # function binding
        
        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_GBM, np.inf) - self.K * np.exp(-self.r*self.T) * Q2(k, cf_GBM, np.inf)   # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r*self.T) * (1 - Q2(k, cf_GBM, np.inf)) - self.S0 * (1-Q1(k, cf_GBM, np.inf))  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")
            
            
    
    def MC(self, N, Err=False, Time=False):
        """
        BS Monte Carlo
        Err = return Standard Error if True
        Time = return execution time if True
        """
        t_init = time()
             
        S_T = self.exp_RV( self.S0, self.T, N )
        V = scp.mean( np.exp(-self.r*self.T) * self.payoff_f(S_T) )
        
        if (Err == True):
            if (Time == True):
                elapsed = time()-t_init
                return V, ss.sem(np.exp(-self.r*self.T) * self.payoff_f(S_T)), elapsed
            else:
                return V, ss.sem(np.exp(-self.r*self.T) * self.payoff_f(S_T))
        else:
            if (Time == True):
                elapsed = time()-t_init
                return V, elapsed
            else:
                return V
    
    
    
    def PDE_price(self, steps, Time=False, solver="splu"):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        Solver = spsolve or splu or Thomas or SOR
        """
        t_init = time()
        
        Nspace = steps[0]   
        Ntime = steps[1]
        
        S_max = 6*float(self.K)                
        S_min = float(self.K)/6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x0 = np.log(self.S0)                            # current log-price
        
        x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)  
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)
        
        self.S_vec = np.exp(x)        # vector of S
        Payoff = self.payoff_f(self.S_vec)

        V = np.zeros((Nspace,Ntime))
        if self.payoff == "call":
            V[:,-1] = Payoff
            V[-1,:] = np.exp(x_max) - self.K * np.exp(-self.r* t[::-1] )
            V[0,:] = 0
        else:    
            V[:,-1] = Payoff
            V[-1,:] = 0
            V[0,:] = self.K * np.exp(-self.r* t[::-1] )
        
        sig2 = self.sig**2 
        dxx = dx**2
        a = ( (dt/2) * ( (self.r-0.5*sig2)/dx - sig2/dxx ) )
        b = ( 1 + dt * ( sig2/dxx + self.r ) )
        c = (-(dt/2) * ( (self.r-0.5*sig2)/dx + sig2/dxx ) )
        
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()
            
        offset = np.zeros(Nspace-2)
        
        
        if solver == "spsolve":        
            if self.exercise=="European":        
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = spsolve( D, (V[1:-1,i+1] - offset) )
            elif self.exercise=="American":
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = np.maximum( spsolve( D, (V[1:-1,i+1] - offset) ), Payoff[1:-1])
        elif solver == "Thomas":        
            if self.exercise=="European":        
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = Thomas( D, (V[1:-1,i+1] - offset) )
            elif self.exercise=="American":
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = np.maximum( Thomas( D, (V[1:-1,i+1] - offset) ), Payoff[1:-1]) 
        elif solver == "SOR":        
            if self.exercise=="European":        
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = SOR( a,b,c, (V[1:-1,i+1] - offset), w=1.68, eps=1e-10, N_max=600 )
            elif self.exercise=="American":
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = np.maximum( SOR( a,b,c, (V[1:-1,i+1] - offset), w=1.68, eps=1e-10, N_max=600 ), Payoff[1:-1]) 
        elif solver == "splu":
            DD = splu(D)
            if self.exercise=="European":        
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = DD.solve( V[1:-1,i+1] - offset )
            elif self.exercise=="American":
                for i in range(Ntime-2,-1,-1):
                    offset[0] = a * V[0,i]
                    offset[-1] = c * V[-1,i]
                    V[1:-1,i] = np.maximum( DD.solve( V[1:-1,i+1] - offset ), Payoff[1:-1])
        else:
            raise ValueError("Solver is splu, spsolve, SOR or Thomas")    
        
        self.price = np.interp(x0, x, V[:,0])
        self.price_vec = V[:,0]
        self.mesh = V
        
        if (Time == True):
            elapsed = time()-t_init
            return self.price, elapsed
        else:
            return self.price
    
    
       
    def plot(self, axis=None):
        if (type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray):
            self.PDE_price((7000,5000))
            #print("run the PDE_price method")
            #return
        
        plt.plot(self.S_vec, self.payoff_f(self.S_vec) , color='blue',label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color='red',label="BS curve")
        if (type(axis) == list):
            plt.axis(axis)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title("Black Scholes price")
        plt.legend(loc='upper left')
        plt.show()
        
        
    def mesh_plt(self):
        if (type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray):
            self.PDE_price((7000,5000))
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid( np.linspace(0, self.T, self.mesh.shape[1]) , self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("BS price surface")
        ax.set_xlabel("S"); ax.set_ylabel("t"); ax.set_zlabel("V")
        ax.view_init(30, -100) # this function rotates the 3d plot
        plt.show()
        
        
    def LSM(self, N=10000, paths=10000, order=2):
        """
        Longstaff-Schwartz Method for pricing American options
        
        N = number of time steps
        paths = number of generated paths
        order = order of the polynomial for the regression 
        """
        
        if self.payoff!="put":
            raise ValueError("invalid type. Set 'call' or 'put'")
        
        dt = self.T/(N-1)          # time interval
        df = np.exp(-self.r * dt)  # discount factor per time time interval
        
        X0 = np.zeros((paths,1))
        increments = ss.norm.rvs(loc=(self.r-self.sig**2/2)*dt, scale=np.sqrt(dt)*self.sig, size=(paths,N-1))
        X = np.concatenate((X0,increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)
        
        H = np.maximum(self.K - S, 0)   # intrinsic values for put option
        V = np.zeros_like(H)            # value matrix
        V[:,-1] = H[:,-1]

        # Valuation by LS Method
        for t in range(N-2, 0, -1):
            good_paths = H[:,t] > 0    
            rg = np.polyfit( S[good_paths, t], V[good_paths, t+1] * df, 2)    # polynomial regression
            C = np.polyval( rg, S[good_paths,t] )                             # evaluation of regression  
    
            exercise = np.zeros( len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths,t] > C
    
            V[exercise,t] = H[exercise,t]
            V[exercise,t+1:] = 0
            discount_path = (V[:,t] == 0)
            V[discount_path,t] = V[discount_path,t+1] * df
    
        V0 = np.mean(V[:,1]) * df  # 
        return V0
        