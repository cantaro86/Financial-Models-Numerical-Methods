#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:49:22 2023

@author: cantaro86
"""


from setuptools import Extension, setup
from Cython.Build import build_ext, cythonize
import numpy

extensions = [
    Extension(
        "src/FMNM/cython/*",
        ["src/FMNM/cython/*.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


setup(
    name="fmnm_cython",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level="3"),
)
