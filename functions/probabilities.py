#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:33:39 2019

@author: cantaro86
"""

import numpy as np
from scipy.integrate import quad
from functools import partial
from functions.CF import cf_Heston_good


def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( (np.exp(-u*k*1j) / (u*1j)) * 
                                  cf(u-1j) / cf(-1.0000000000001j) )  
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=500 )[0]


def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( np.exp(-u*k*1j) /(u*1j) * cf(u) )
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=500 )[0]


def Gil_Pelaez_pdf(x, cf, right_lim):
    """
    Gil Pelaez formula for the inversion of the characteristic function
    INPUT
    - x: is a number
    - right_lim: is the right extreme of integration
    - cf: is the characteristic function
    OUTPUT
    - the value of the density at x.
    """
    integrand = lambda u: np.real( np.exp(-u*x*1j) * cf(u) )
    return 1/np.pi * quad(integrand, 1e-15, right_lim )[0]


def Heston_pdf(i, t, v0, mu, theta, sigma, kappa, rho):
    """
    Heston density by Fourier inversion.
    """
    cf_H_b_good = partial(cf_Heston_good, t=t, v0=v0, mu=mu, theta=theta, sigma=sigma, kappa=kappa, rho=rho )
    return Gil_Pelaez_pdf(i, cf_H_b_good, np.inf)




