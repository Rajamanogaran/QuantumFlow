# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast simulation kernels for QuantumFlow
========================================

Numerically-critical routines used by the statevector simulator:

* ``normalise`` — renormalise a statevector in place
* ``probabilities`` — ``|amplitude|^2`` as a real array
* ``sample_cumulative`` — inverse-CDF sampling from a probability array

All operate on contiguous float64/complex128 buffers for speed. The
compiled module is optional; QuantumFlow falls back to equivalent numpy
code when it is unavailable.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


def normalise(cnp.ndarray[double complex, ndim=1] state, double tol=1e-12):
    """Renormalise *state* in place; returns the norm before scaling."""
    cdef Py_ssize_t i, n = state.shape[0]
    cdef double acc = 0.0
    cdef double complex v

    with nogil:
        for i in range(n):
            v = state[i]
            acc += (v.real * v.real + v.imag * v.imag)

    norm = sqrt(acc)
    if norm > tol:
        inv = 1.0 / norm
        with nogil:
            for i in range(n):
                state[i] = state[i] * inv
    return norm


def probabilities(cnp.ndarray[double complex, ndim=1] state):
    """Return ``|amplitude|^2`` as a float64 array."""
    cdef Py_ssize_t i, n = state.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double complex v

    with nogil:
        for i in range(n):
            v = state[i]
            out[i] = v.real * v.real + v.imag * v.imag
    return out


def sample_cumulative(cnp.ndarray[cnp.float64_t, ndim=1] probs,
                      double u):
    """Inverse-CDF sample: index *i* such that ``cdf[i-1] <= u < cdf[i]``.

    ``u`` must be drawn uniformly from ``[0, 1)`` by the caller.
    """
    cdef Py_ssize_t i, n = probs.shape[0]
    cdef Py_ssize_t found = n - 1
    cdef double acc = 0.0

    with nogil:
        for i in range(n):
            acc += probs[i]
            if u < acc:
                found = i
                break
    return found
