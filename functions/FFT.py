#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:13:17 2020

@author: cantaro86
"""

import numpy as np
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import fsolve


def fft_Lewis(K, S0, r, T, cf, interp="cubic"):
    """ 
    K = vector of strike
    S = spot price scalar
    cf = characteristic function
    interp can be cubic or linear
    """
    N=2**15                          # FFT more efficient for N power of 2
    B = 500                          # integration limit 
    dx = B/N
    x = np.arange(N) * dx            # the final value B is excluded

    weight = np.arange(N)            # Simpson weights
    weight = 3 + (-1)**(weight+1)
    weight[0] = 1; weight[N-1]=1

    dk = 2*np.pi/B
    b = N * dk /2
    ks = -b + dk * np.arange(N)

    integrand = np.exp(- 1j * b * np.arange(N)*dx) * cf(x - 0.5j) * 1/(x**2 + 0.25) * weight * dx/3
    integral_value = np.real( ifft(integrand)*N )
    
    if interp == "linear":
        spline_lin = interp1d(ks, integral_value, kind='linear')
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r*T)/np.pi * spline_lin( np.log(S0/K) )
    elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind='cubic')
        prices = S0 - np.sqrt(S0 * K) * np.exp(-r*T)/np.pi * spline_cub( np.log(S0/K) )
    return prices



def IV_from_Lewis(K, S0, T, r, cf, disp=False):
    """ Implied Volatility from the Lewis formula 
    K = strike; S0 = spot stock; T = time to maturity; r = interest rate
    cf = characteristic function """
    k = np.log(S0/K)
    def obj_fun(sig):
        integrand = lambda u: np.real( np.exp(u*k*1j)\
                            * (cf(u - 0.5j) - np.exp(1j*u*r*T+0.5*r*T) * np.exp(-0.5*T*(u**2+0.25)*sig**2) ) )\
                            * 1/(u**2 + 0.25)
        int_value = quad(integrand, 1e-15, 2000, limit=2000, full_output=1 )[0]
        return int_value
    
    X0 = [0.2, 1, 2, 4, 0.0001]   # set of initial guess points
    for x0 in X0:
        x, _, solved, msg = fsolve( obj_fun, [x0,], full_output=True, xtol=1e-4)
        if solved == 1:
            return x[0]    
    if disp == True:
        print("Strike", K, msg)
    return -1
