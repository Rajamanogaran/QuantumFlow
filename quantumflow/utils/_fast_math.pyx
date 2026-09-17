# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast math kernels for QuantumFlow
==================================

Small, hot numerical helpers:

* ``abs2`` — element-wise ``|z|^2`` of a complex array
* ``trace_abs2`` — sum of ``|z|^2`` (norm squared)
* ``kron2`` — Kronecker product of two complex matrices

The compiled module is optional; QuantumFlow falls back to equivalent
numpy code when it is unavailable.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def abs2(cnp.ndarray[double complex, ndim=1] z):
    """Element-wise ``|z_i|^2`` as float64."""
    cdef Py_ssize_t i, n = z.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double complex v
    with nogil:
        for i in range(n):
            v = z[i]
            out[i] = v.real * v.real + v.imag * v.imag
    return out


def norm_squared(cnp.ndarray[double complex, ndim=1] z):
    """Total ``sum |z_i|^2``."""
    cdef Py_ssize_t i, n = z.shape[0]
    cdef double acc = 0.0
    cdef double complex v
    with nogil:
        for i in range(n):
            v = z[i]
            acc += v.real * v.real + v.imag * v.imag
    return acc


def kron2(cnp.ndarray[double complex, ndim=2] a,
          cnp.ndarray[double complex, ndim=2] b):
    """Kronecker product of two complex128 matrices."""
    cdef Py_ssize_t ra = a.shape[0], ca = a.shape[1]
    cdef Py_ssize_t rb = b.shape[0], cb = b.shape[1]
    cdef Py_ssize_t i, j, k, l
    cdef double complex av
    cdef cnp.ndarray[double complex, ndim=2] out = np.empty((ra * rb, ca * cb), dtype=np.complex128)

    with nogil:
        for i in range(ra):
            for k in range(rb):
                av = a[i, k]
                for j in range(ca):
                    for l in range(cb):
                        out[i * rb + k, j * cb + l] = av * b[k, l]
    return out
