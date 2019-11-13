#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:49:22 2019

@author: cantaro86
"""

# Compile with:
# python setup.py build_ext --inplace



from distutils.core import setup
from Cython.Build import cythonize
import numpy



setup(
    ext_modules = cythonize("cython_functions.pyx", language_level='3'),
    include_dirs = [numpy.get_include()]
)


setup(
    ext_modules = cythonize("cython_Heston.pyx", language_level='3'),
    include_dirs = [numpy.get_include()]
)
