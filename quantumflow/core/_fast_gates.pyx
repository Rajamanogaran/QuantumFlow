# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast gate-application kernels for QuantumFlow
==============================================

Cython kernels that apply small (1- and 2-qubit) gate matrices directly to
a flat statevector buffer. Numerically identical to the pure-Python einsum
path, but without tensor reshaping overhead.

The compiled module is optional: when it is unavailable QuantumFlow falls
back to the pure-Python implementation transparently.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def apply_gate_1(cnp.ndarray[double complex, ndim=1] state,
                 cnp.ndarray[double complex, ndim=2] gate,
                 int qubit, int n):
    """Apply a 1-qubit gate in place.

    ``qubit`` follows the MSB-first convention (qubit 0 is the most
    significant bit of the amplitude index).
    """
    cdef Py_ssize_t dim = state.shape[0]
    cdef Py_ssize_t block_start = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t i0, i1
    cdef int shift = n - 1 - qubit
    cdef Py_ssize_t stride = <Py_ssize_t>1 << shift
    cdef Py_ssize_t block = stride << 1
    cdef double complex a0, a1, g00, g01, g10, g11

    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]

    with nogil:
        while block_start < dim:
            j = 0
            while j < stride:
                i0 = block_start + j
                i1 = i0 + stride
                a0 = state[i0]
                a1 = state[i1]
                state[i0] = g00 * a0 + g01 * a1
                state[i1] = g10 * a0 + g11 * a1
                j += 1
            block_start += block


def apply_gate_2(cnp.ndarray[double complex, ndim=1] state,
                 cnp.ndarray[double complex, ndim=2] gate,
                 int qubit0, int qubit1, int n):
    """Apply a 2-qubit gate in place.

    ``gate`` is in standard kron order (qubit0 = most significant of the
    pair, matching :class:`~quantumflow.core.gate.Gate` matrices).
    """
    cdef Py_ssize_t dim = state.shape[0]
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i00, i01, i10, i11
    cdef Py_ssize_t r = 0
    cdef int shift0 = n - 1 - qubit0
    cdef int shift1 = n - 1 - qubit1
    cdef Py_ssize_t s0 = <Py_ssize_t>1 << shift0
    cdef Py_ssize_t s1 = <Py_ssize_t>1 << shift1
    cdef double complex amps[4]
    cdef double complex out[4]
    cdef int bit0, bit1

    with nogil:
        while idx < dim:
            bit0 = (idx >> shift0) & 1
            bit1 = (idx >> shift1) & 1
            if bit0 == 0 and bit1 == 0:
                i00 = idx
                i01 = idx | s1
                i10 = idx | s0
                i11 = idx | s0 | s1
                amps[0] = state[i00]
                amps[1] = state[i01]
                amps[2] = state[i10]
                amps[3] = state[i11]
                r = 0
                while r < 4:
                    out[r] = (
                        gate[r, 0] * amps[0] + gate[r, 1] * amps[1] +
                        gate[r, 2] * amps[2] + gate[r, 3] * amps[3]
                    )
                    r += 1
                state[i00] = out[0]
                state[i01] = out[1]
                state[i10] = out[2]
                state[i11] = out[3]
            idx += 1
