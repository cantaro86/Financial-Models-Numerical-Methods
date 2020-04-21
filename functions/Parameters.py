#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:56:25 2019

@author: cantaro86
"""

class Option_param():
    """
    Option class wants the option parameters:
    S0 = current stock price
    K = Strike price
    T = time to maturity
    v0 = (optional) spot variance 
    exercise = European or American
    """
    def __init__(self, S0=15, K=15, T=1, v0=0.04, payoff="call", exercise="European"):
        self.S0 = S0
        self.v0 = v0
        self.K = K
        self.T = T
        
        if (exercise=="European" or exercise=="American"):
            self.exercise = exercise
        else: 
            raise ValueError("invalid type. Set 'European' or 'American'")
        
        if (payoff=="call" or payoff=="put"):
            self.payoff = payoff
        else: 
            raise ValueError("invalid type. Set 'call' or 'put'")


