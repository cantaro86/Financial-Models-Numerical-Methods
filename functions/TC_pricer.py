#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:56:25 2019

@author: cantaro86
"""

from time import time
import numpy as np
import numpy.matlib
import functions.cost_utils as cost


class TC_pricer():
    """
    Solver for the option pricing model of Davis-Panas-Zariphopoulou.
    """
    
    def __init__(self, Option_info, Process_info, cost_b=0, cost_s=0, gamma=0.001):
        """
        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price, strike, maturity in years
        Process_info:  of type Diffusion_process. It contains (r,mu, sig) i.e.  interest rate, drift coefficient, diffusion coefficient
        cost_b:  (lambda in the paper) BUY cost  
        cost_s: (mu in the paper)  SELL cost
        gamma: risk avversion coefficient
        """
        
        if Option_info.payoff == "put":
            raise ValueError("Not implemented for Put Options") 
        
        
        self.r = Process_info.r           # interest rate
        self.mu = Process_info.mu         # drift coefficient
        self.sig = Process_info.sig       # diffusion coefficient
        self.S0 = Option_info.S0          # current price
        self.K = Option_info.K            # strike
        self.T = Option_info.T            # maturity in years
        self.cost_b = cost_b              # (lambda in the paper) BUY cost  
        self.cost_s = cost_s              # (mu in the paper)  SELL cost
        self.gamma = gamma                # risk avversion coefficient
        

    def price(self, N=500, TYPE="writer", Time=False ):
        """
        N =  number of time steps
        TYPE writer or buyer
        Time: Boolean
        """
        t=time()                                               # measures run time  
        np.seterr(all='ignore')                                # ignore Warning for overflows                       

        x0 = np.log(self.S0)                                   # current log-price
        T_vec, dt = np.linspace(0, self.T, N+1, retstep=True)  # vector of time steps and time steps
        delta = np.exp(- self.r * (self.T - T_vec) )           # discount factor
        dx = self.sig * np.sqrt(dt)                            # space step1 
        dy = dx                                                # space step2 
        M = int(np.floor(N/2))                          
        y = np.linspace(-M*dy,M*dy,2*M+1)               
        N_y = len(y)                                           # dim of vector y 
        med = np.where(y == 0)[0].item()                       # point where y==0 

        F = lambda x,l,n: np.exp(  self.gamma * (1+self.cost_b) * np.exp(x)*l / delta[n] )  
        G = lambda x,m,n: np.exp( -self.gamma * (1-self.cost_s) * np.exp(x)*m / delta[n] )

        for portfolio in ["no_opt", TYPE]:     # interates on the zero option and writer/buyer portfolios
            
            # Tree nodes at time N
            x = np.array( [x0 + (self.mu - 0.5 * self.sig**2)*dt*N + (2*i-N)*dx for i in range(N+1) ] ) 

            # Terminal conditions
            if portfolio == "no_opt":
                Q = np.exp( -self.gamma * cost.no_opt(x, y, self.cost_b, self.cost_s) ) 
            elif portfolio == "writer":
                Q = np.exp( -self.gamma * cost.writer(x, y, self.cost_b, self.cost_s, self.K) )
            elif portfolio == "buyer":
                Q = np.exp( -self.gamma * cost.buyer(x, y, self.cost_b, self.cost_s, self.K) )
            else:
                raise ValueError("TYPE can be only writer or buyer")


            for k in range(N-1,-1,-1):
                #  expectation term
                Q_new = ( Q[:-1,:] + Q[1:,:] ) / 2
                    
                # create the logprice vector at time k
                x = np.array( [x0 + (self.mu - 0.5 * self.sig**2)*dt*k + (2*i-k)*dx for i in range(k+1) ] )

                # buy term
                Buy = np.copy(Q_new)  
                Buy[:,:-1] = np.matlib.repmat(F(x,dy,k),N_y-1,1).T * Q_new[:,1:]  

                # sell term
                Sell = np.copy(Q_new) 
                Sell[:,1:] = np.matlib.repmat(G(x,dy,k),N_y-1,1).T * Q_new[:,:-1] 

                # update the Q(:,:,k) 
                Q = np.minimum( np.minimum(Buy,Sell) , Q_new )

            if (portfolio == "no_opt"):
                Q_no = Q[0,med]
            else:
                Q_yes = Q[0,med]

        if (TYPE == "writer" ):
            price = (delta[0] / self.gamma) * np.log( Q_yes / Q_no )
        else:
            price = (delta[0] / self.gamma) * np.log( Q_no / Q_yes )

        if (Time == True):
            elapsed = time()-t
            return price, elapsed
        else:
            return price

