#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:57:19 2019

@author: cantaro86
"""

import numpy as np


def cf_normal(u, mu=1, sig=2):
    """
    Characteristic function of a Normal random variable
    """
    return np.exp( 1j * u * mu - 0.5 * u**2 * sig**2 )

def cf_gamma(u, a=1, b=2):
    """
    Characteristic function of a Gamma random variable
    - shape: a
    - scale: b
    """
    return (1 - b * u * 1j)**(-a)

def cf_poisson(u, lam=1):
    """
    Characteristic function of a Poisson random variable
    - rate: lam
    """    
    return np.exp( lam * (np.exp(1j * u) -1) )


def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    """
    Characteristic function of a Merton random variable at time t
    mu: drift
    sig: diffusion coefficient
    lam: jump activity
    muJ: jump mean size
    sigJ: jump size standard deviation 
    """    
    return np.exp( t * ( 1j * u * mu - 0.5 * u**2 * sig**2 \
                  + lam*( np.exp(1j*u*muJ - 0.5 * u**2 * sigJ**2) -1 ) ) )


def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    """
    Characteristic function of a Variance Gamma random variable at time t
    mu: additional drift
    theta: Brownian motion drift 
    sigma: Brownian motion diffusion
    kappa: Gamma process variance
    """    
    return np.exp( t * ( 1j*mu*u - np.log(1 - 1j*theta*kappa*u + 0.5*kappa*sigma**2 * u**2 ) /kappa  ) )
