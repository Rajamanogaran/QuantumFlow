#!/usr/bin/env python
"""Setup script for QuantumFlow."""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Cython extensions for performance-critical operations
extensions = [
    Extension(
        "quantumflow.core._fast_gates",
        ["quantumflow/core/_fast_gates.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
    Extension(
        "quantumflow.simulation._fast_simulator",
        ["quantumflow/simulation/_fast_simulator.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
    Extension(
        "quantumflow.utils._fast_math",
        ["quantumflow/utils/_fast_math.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
