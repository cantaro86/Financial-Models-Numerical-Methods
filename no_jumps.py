#! /usr/bin/python3 

import time
#import sys

import numpy as np
import numpy.matlib
from copy import deepcopy

t=time.time()

#%%
def cost_f(x,y,cost_b,cost_s):
    
    cost = np.zeros( (len(x), len(y) ) )

    for i in range(len(y)):
        if y[i] <= 0 :
            cost[:,i] = (1+cost_b) * y[i] * np.exp(x)
        else:
            cost[:,i] = (1-cost_s) * y[i] * np.exp(x)

    return cost     # np.longfloat(np.transpose(cost))



def cost_opt(x,y,cost_b,cost_s,K):

    cost = np.zeros( (len(x),len(y)) )
    
    for i in range(len(x)):
        for j in range(len(y)):
    
            if y[j] < 0 and np.exp(x[i]) <= K :
                cost[i][j] = (1+cost_b) * y[j] * np.exp(x[i])
            
            elif y[j] >= 0 and np.exp(x[i]) <= K : 
                cost[i][j] = (1-cost_s) * y[j] * np.exp(x[i])
        
            elif y[j]-1 >= 0 and np.exp(x[i]) > K :
                cost[i][j] = ( (1-cost_s) * (y[j]-1) * np.exp(x[i]) ) + K
        
            elif y[j]-1 < 0 and np.exp(x[i]) > K :
                cost[i][j] = ( (1+cost_b) * (y[j]-1) * np.exp(x[i]) ) + K

    return np.longfloat(cost)
    
def cost_opt2(x,y,cost_b,cost_s,K):

    cost = np.zeros( (len(x),len(y)) )
    
    for i in range(len(x)):
        for j in range(len(y)):
            cost[i,j] = cost_f([x[i]],[y[j]],cost_b,cost_s) if (np.exp(x[i]) <= K) else cost_f([x[i]],[y[j]-1],cost_b,cost_s) + K
    return cost
    
    
#%%    


r = 0.1                   # interest rate
mu = 0.1                  # drift coefficient
sig = 0.25                # diffusion coefficient


S0 = 15.0                  # current price
x0 = np.log(S0)           # current log-price

K = 15.0                   # strike
T = 1.0                  # year

cost_b = 0.0             # (lambda in the paper) BUY cost  
cost_s = 0.0             # (mu in the paper)  SELL cost
gamma = 0.01              # risk avversion coefficient

#%%
#def trans_cost_NJ( T=1, S0=15, K=15, r=0.1, mu=0.1, sig=0.25, cost=0.005, gamma=0.1, N=800  ):

N = 2000          # time steps even!!!
#cost1 = cost2 = cost
        
dt = T/N
T_vec = np.linspace(0,T,N+1)
delta = np.exp(-r*(T-T_vec))        # discount factor

dx = sig * np.sqrt(dt)


M = int(np.floor(N/2))
dy = dx
y = np.longfloat(np.linspace(-M*dy,M*dy,2*M+1))       
N_y = len(y)
med = np.where(y == 0)[0].item()



F = lambda x,l,N: np.exp( gamma * (1+cost_b)*np.exp(x)*l / delta[N] )  

G = lambda x,m,N: np.exp( -gamma * (1-cost_s)*np.exp(x)*m / delta[N] )

#%%

for TYPE in range(2):

    x = np.array( [x0 + (mu-0.5*sig**2)*dt*N + (2*i-N)*dx for i in range(N+1) ] , dtype=np.float128 )
    
    ############# Boundary conditions ##############

    # Terminal
    if TYPE == 0:
        Q = np.exp( np.longfloat( -gamma * np.longfloat( cost_f(x,y,cost_b,cost_s)) ))
    else:
        Q = np.exp( np.longfloat( -gamma * np.longfloat( cost_opt(x,y,cost_b,cost_s,K)) ))


    for k in range(N-1,-1,-1):

        ###  expectation term
        Q_new = ( Q[:-1,:] + Q[1:,:] ) / 2

        ### create the logprice vector at time k
        x = np.array( [x0 + (mu-0.5*sig**2)*dt*k + (2*i-k)*dx for i in range(k+1) ] , dtype=np.float128)


        ### buy term
        Buy = np.longfloat(deepcopy(Q_new))
        Buy[:,:-1] = np.longfloat( np.matlib.repmat(F(x,dy,k),N_y-1,1).transpose() ) * np.longfloat( Q_new[:,1:] ) 

        ### sell term
        Sell = np.longfloat(deepcopy(Q_new))
        Sell[:,1:] = np.matlib.repmat(G(x,dy,k),N_y-1,1).transpose() * np.longfloat( Q_new[:,:-1] )

        ### update the Q(:,:,k) 
        Q = np.minimum( np.minimum(Buy,Sell) , Q_new )

    if (TYPE == 0):
        Q_no = Q[0,med]
    else:
        Q_yes = Q[0,med]


price = (delta[0] / gamma) * np.log( Q_yes / Q_no )
#return price
elapsed=time.time()-t

print("price: %.15f" %price )
print("time: %.15f" %elapsed )