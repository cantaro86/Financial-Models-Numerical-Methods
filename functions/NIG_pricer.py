#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1 12:47:00 2019

@author: cantaro86
"""

from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
import numpy as np
import scipy as scp
from scipy import signal
from scipy.integrate import quad
import scipy.stats as ss
import scipy.special as scps

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from functions.CF import cf_NIG
from functions.probabilities import Q1, Q2
from functools import partial


class NIG_pricer():
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation         
    
        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \int[ V(x+y) nu(dy) ] -(r+lam)V  
    """
    def __init__(self, Option_info, Process_info ):
        """
        Process_info:  of type NIG_process. It contains the interest rate r and the NIG parameters (sigma, theta, kappa) 
    
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        """
        self.r = Process_info.r           # interest rate
        self.sigma = Process_info.sigma       # NIG parameter
        self.theta = Process_info.theta       # NIG parameter
        self.kappa = Process_info.kappa       # NIG parameter
        self.exp_RV = Process_info.exp_RV     # function to generate exponential NIG Random Variables
        
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        
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
    
             
    
    def Fourier_inversion(self):
        """
        Price obtained by inversion of the characteristic function
        """
        k = np.log(self.K/self.S0)                # log moneyness
        w = ( 1 - np.sqrt( 1 - 2*self.theta*self.kappa -self.kappa*self.sigma**2) )/self.kappa # martingale correction
        
        cf_NIG_b = partial(cf_NIG, t=self.T, mu=(self.r-w), theta=self.theta, sigma=self.sigma, kappa=self.kappa )
        
        if self.payoff == "call":
            call = self.S0 * Q1(k, cf_NIG_b, np.inf) - self.K * np.exp(-self.r*self.T) * Q2(k, cf_NIG_b, np.inf)   # pricing function
            return call
        elif self.payoff == "put":
            put = self.K * np.exp(-self.r*self.T) * (1 - Q2(k, cf_NIG_b, np.inf)) - self.S0 * (1-Q1(k, cf_NIG_b, np.inf))  # pricing function
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")


    
    def MC(self, N, Err=False, Time=False):
        """
        NIG Monte Carlo
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
    
    
    
    def NIG_measure(self, x):
        A = self.theta/(self.sigma**2)
        B = np.sqrt( self.theta**2 + self.sigma**2/self.kappa ) / self.sigma**2
        C = np.sqrt( self.theta**2 + self.sigma**2/self.kappa) /(np.pi*self.sigma * np.sqrt(self.kappa))
        return C/np.abs(x) * np.exp(A*(x)) * scps.kv(1, B*np.abs(x) )
    
    
    
    def PIDE_price(self, steps, Time=False):
        """
        steps = tuple with number of space steps and time steps
        payoff = "call" or "put"
        exercise = "European" or "American"
        Time = Boolean. Execution time.
        """
        t_init = time()
        
        Nspace = steps[0]   
        Ntime = steps[1]
        
        S_max = 2000*float(self.K)                
        S_min = float(self.K)/2000
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        
        dev_X = np.sqrt(self.sigma**2 + self.theta**2 * self.kappa)     # std dev NIG process
        
        dx = (x_max - x_min)/(Nspace-1)
        extraP = int(np.floor(7*dev_X/dx))            # extra points beyond the B.C.
        x = np.linspace(x_min-extraP*dx, x_max+extraP*dx, Nspace + 2*extraP)   # space discretization
        t, dt = np.linspace(0, self.T, Ntime, retstep=True)       # time discretization
        
        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace-2)
        V = np.zeros((Nspace + 2*extraP, Ntime))       # grid initialization
        
        if self.payoff == "call":
            V[:,-1] = Payoff                   # terminal conditions 
            V[-extraP-1:,:] = np.exp(x[-extraP-1:]).reshape(extraP+1,1) * np.ones((extraP+1,Ntime)) - \
                 self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))  # boundary condition
            V[:extraP+1,:] = 0
        else:    
            V[:,-1] = Payoff
            V[-extraP-1:,:] = 0
            V[:extraP+1,:] = self.K * np.exp(-self.r* t[::-1] ) * np.ones((extraP+1,Ntime))
        

        eps = 1.5*dx    # the cutoff near 0
        lam = quad(self.NIG_measure,-(extraP+1.5)*dx,-eps)[0] + quad(self.NIG_measure,eps,(extraP+1.5)*dx)[0] # approximated intensity

        int_w = lambda y: (np.exp(y)-1) * self.NIG_measure(y)
        int_s = lambda y: y**2 * self.NIG_measure(y)

        w = quad(int_w, -(extraP+1.5)*dx, -eps)[0] + quad(int_w, eps, (extraP+1.5)*dx)[0]   # is the approx of w
        sig2 = quad(int_s, -eps, eps, points=0)[0]         # the small jumps variance
        
        
        dxx = dx * dx
        a = ( (dt/2) * ( (self.r - w - 0.5*sig2)/dx - sig2/dxx ) )
        b = ( 1 + dt * ( sig2/dxx + self.r + lam) )
        c = (-(dt/2) * ( (self.r - w - 0.5*sig2)/dx + sig2/dxx ) )
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()
        DD = splu(D)
       
        nu = np.zeros(2*extraP+3)        # Lévy measure vector
        x_med = extraP+1                 # middle point in nu vector
        x_nu = np.linspace(-(extraP+1+0.5)*dx, (extraP+1+0.5)*dx, 2*(extraP+2) )    # integration domain
        for i in range(len(nu)):
            if (i==x_med) or (i==x_med-1) or (i==x_med+1):
                continue
            nu[i] = quad(self.NIG_measure, x_nu[i], x_nu[i+1])[0]


        if self.exercise=="European":        
            # Backward iteration
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]
                offset[-1] = c * V[-1-extraP,i]
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="auto")
                V[extraP+1 : -extraP-1, i] = DD.solve( V_jump - offset ) 
        elif self.exercise=="American":
            for i in range(Ntime-2,-1,-1):
                offset[0] = a * V[extraP,i]
                offset[-1] = c * V[-1-extraP,i]
                V_jump = V[extraP+1 : -extraP-1, i+1] + dt * signal.convolve(V[:,i+1],nu[::-1],mode="valid",method="auto")
                V[extraP+1 : -extraP-1, i] = np.maximum( DD.solve( V_jump - offset ), Payoff[extraP+1 : -extraP-1] )
                
        X0 = np.log(self.S0)                            # current log-price
        self.S_vec = np.exp(x[extraP+1 : -extraP-1])        # vector of S
        self.price = np.interp(X0, x, V[:,0])
        self.price_vec = V[extraP+1 : -extraP-1,0]
        self.mesh = V[extraP+1 : -extraP-1, :]
        
        if (Time == True):
            elapsed = time()-t_init
            return self.price, elapsed
        else:
            return self.price


    def plot(self, axis=None):
        if (type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray):
            self.PIDE_price((5000,4000))
            
        plt.plot(self.S_vec, self.payoff_f(self.S_vec) , color='blue',label="Payoff")
        plt.plot(self.S_vec, self.price_vec, color='red',label="NIG curve")
        if (type(axis) == list):
            plt.axis(axis)
        plt.xlabel("S"); plt.ylabel("price"); plt.title("NIG price")
        plt.legend(loc='best')
        plt.show()
        
    def mesh_plt(self):
        if (type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray):
            self.PDE_price((7000,5000))
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid( np.linspace(0, self.T, self.mesh.shape[1]) , self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title("NIG price surface")
        ax.set_xlabel("S"); ax.set_ylabel("t"); ax.set_zlabel("V")
        ax.view_init(30, -100) # this function rotates the 3d plot
        plt.show()    
   


     
