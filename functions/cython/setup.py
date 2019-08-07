# Compile with:
# python setup.py build_ext --inplace

from distutils.core import setup#, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cython_functions.pyx", language_level='3'),
    include_dirs=[numpy.get_include()]
)

