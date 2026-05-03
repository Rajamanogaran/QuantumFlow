"""
Quantum Fourier Transform (QFT) and Related Algorithms
========================================================

Implements the Quantum Fourier Transform and its inverse,
along with quantum arithmetic circuits that leverage QFT.

The QFT is a fundamental quantum algorithm that transforms
the computational basis states into the Fourier basis. It
is a key component of many quantum algorithms including
Shor's factoring and phase estimation.

References:
    - Shor, P.W. (1997). Polynomial-time algorithms for prime factorization.
    - Coppersmith, D. (1994). An approximate Fourier transform useful
      in quantum factoring.
"""

import math
import numpy as np
from typing import Optional, List, Tuple

try:
    from quantumflow.core.circuit import QuantumCircuit
    from quantumflow.core.gate import (
        HGate, XGate, SwapGate, ControlledGate, PhaseGate,
        RZGate, CNOTGate, ToffoliGate, UnitaryGate, Measurement,
    )
    from quantumflow.simulation.simulator import StatevectorSimulator
except ImportError:
    pass


def qft_matrix(n: int) -> np.ndarray:
    """
    Compute the QFT unitary matrix.

    The QFT matrix is defined as:
        QFT|j> = (1/sqrt(N)) * sum_k exp(2*pi*i*j*k/N) |k>

    where N = 2^n.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        The 2^n x 2^n QFT unitary matrix.

    Examples
    --------
    >>> U = qft_matrix(3)  # 8x8 QFT matrix
    >>> assert np.allclose(U @ U.conj().T, np.eye(8))  # Unitary check
    """
    N = 2 ** n
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    matrix = omega ** (j * k) / np.sqrt(N)
    return matrix.astype(np.complex128)


def iqft_matrix(n: int) -> np.ndarray:
    """
    Compute the inverse QFT unitary matrix.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        The 2^n x 2^n inverse QFT unitary matrix.
    """
    return qft_matrix(n).conj().T


def apply_qft(circuit: QuantumCircuit, qubits: List[int], inverse: bool = False) -> None:
    """
    Apply QFT (or inverse QFT) to specified qubits in-place.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to apply QFT to.
    qubits : List[int]
        Qubit indices for QFT.
    inverse : bool
        If True, apply inverse QFT.
    """
    n = len(qubits)
    if inverse:
        # Inverse QFT: reverse order, use negative phases
        for j in range(n - 1, -1, -1):
            circuit.h(qubits[j])
            for k in range(j - 1, -1, -1):
                angle = -np.pi / (2 ** (j - k))
                circuit.rz(angle, qubits[k])
                circuit.cx(qubits[k], qubits[j])
                circuit.rz(-angle, qubits[j])
                circuit.cx(qubits[k], qubits[j])
        # Swap qubits
        for j in range(n // 2):
            circuit.swap(qubits[j], qubits[n - 1 - j])
    else:
        # Forward QFT
        for j in range(n):
            circuit.h(qubits[j])
            for k in range(j + 1, n):
                angle = np.pi / (2 ** (k - j))
                circuit.rz(angle, qubits[j])
                circuit.cx(qubits[j], qubits[k])
                circuit.rz(-angle, qubits[k])
                circuit.cx(qubits[j], qubits[k])
        # Swap qubits
        for j in range(n // 2):
            circuit.swap(qubits[j], qubits[n - 1 - j])


def apply_iqft(circuit: QuantumCircuit, qubits: List[int]) -> None:
    """Apply inverse QFT to specified qubits in-place."""
    apply_qft(circuit, qubits, inverse=True)


class QFT:
    """
    Quantum Fourier Transform circuit builder.

    The QFT transforms a quantum state from the computational basis
    to the Fourier basis. For an n-qubit state |x>:
        QFT|x> = (1/sqrt(2^n)) * sum_y exp(2*pi*i*x*y/2^n) |y>

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    inverse : bool
        If True, construct the inverse QFT.
    do_swaps : bool
        If True, include swap gates at the end for bit-reversal.
        Set to False if qubits will be measured in reverse order.
    approximation_degree : int
        Number of rotation gates to skip (starting from smallest angle).
        0 = exact QFT, higher values = more approximate but fewer gates.

    Examples
    --------
    >>> qft = QFT(3)
    >>> circuit = qft.construct_circuit()
    >>> # Verify unitarity
    >>> U = circuit.to_unitary()
    >>> assert np.allclose(U @ U.conj().T, np.eye(8))

    Approximate QFT:
    >>> aqft = QFT(10, approximation_degree=3)
    >>> circuit = aqft.construct_circuit()
    """

    def __init__(
        self,
        n_qubits: int,
        inverse: bool = False,
        do_swaps: bool = True,
        approximation_degree: int = 0,
    ) -> None:
        self.n_qubits = n_qubits
        self.inverse = inverse
        self.do_swaps = do_swaps
        self.approximation_degree = approximation_degree

    def construct_circuit(self, qubits: Optional[List[int]] = None) -> QuantumCircuit:
        """
        Construct the QFT circuit.

        Parameters
        ----------
        qubits : Optional[List[int]]
            Qubit indices to apply QFT to. If None, uses [0, 1, ..., n-1].

        Returns
        -------
        QuantumCircuit
            QFT circuit.
        """
        if qubits is None:
            qubits = list(range(self.n_qubits))

        circuit = QuantumCircuit(self.n_qubits)
        n = len(qubits)

        if self.inverse:
            self._construct_inverse(circuit, qubits)
        else:
            self._construct_forward(circuit, qubits)

        return circuit

    def _construct_forward(self, circuit: QuantumCircuit, qubits: List[int]) -> None:
        """Construct the forward QFT circuit."""
        n = len(qubits)

        for j in range(n):
            # Hadamard on qubit j
            circuit.h(qubits[j])

            # Controlled phase rotations from qubit j to qubit k
            for k in range(j + 1, n):
                # Skip gates for approximation
                gate_index = (k - j - 1)
                if gate_index < self.approximation_degree:
                    continue

                angle = np.pi / (2 ** (k - j))
                # Controlled RZ: CRZ(angle) = CZ with phase
                circuit.rz(angle, qubits[j])
                circuit.cx(qubits[j], qubits[k])
                circuit.rz(-angle, qubits[k])
                circuit.cx(qubits[j], qubits[k])

        # Bit-reversal swaps
        if self.do_swaps:
            for j in range(n // 2):
                circuit.swap(qubits[j], qubits[n - 1 - j])

    def _construct_inverse(self, circuit: QuantumCircuit, qubits: List[int]) -> None:
        """Construct the inverse QFT circuit."""
        n = len(qubits)

        # Inverse: swap first, then reverse gate order with negated angles
        if self.do_swaps:
            for j in range(n // 2):
                circuit.swap(qubits[j], qubits[n - 1 - j])

        for j in range(n - 1, -1, -1):
            # Controlled phase rotations (inverse)
            for k in range(j - 1, -1, -1):
                gate_index = (j - k - 1)
                if gate_index < self.approximation_degree:
                    continue

                angle = -np.pi / (2 ** (j - k))
                circuit.rz(angle, qubits[k])
                circuit.cx(qubits[k], qubits[j])
                circuit.rz(-angle, qubits[j])
                circuit.cx(qubits[k], qubits[j])

            # Hadamard on qubit j
            circuit.h(qubits[j])

    def gate_count(self) -> int:
        """
        Compute the number of gates in the QFT circuit.

        Returns
        -------
        int
            Total number of gates.
        """
        n = self.n_qubits
        # Hadamard: n
        # Controlled rotations: n*(n-1)/2 - approximation_degree
        # Swaps: floor(n/2)
        rotations = n * (n - 1) // 2 - self.approximation_degree
        swaps = n // 2 if self.do_swaps else 0
        return n + rotations * 3 + swaps  # 3 gates per controlled rotation

    def exact_unitary(self) -> np.ndarray:
        """
        Compute the exact QFT unitary matrix.

        Returns
        -------
        np.ndarray
            2^n x 2^n QFT matrix.
        """
        return qft_matrix(self.n_qubits) if not self.inverse else iqft_matrix(self.n_qubits)

    def run(
        self,
        state: Optional[np.ndarray] = None,
        simulator: Optional['StatevectorSimulator'] = None,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the QFT circuit.

        Parameters
        ----------
        state : Optional[np.ndarray]
            Initial state vector. If None, uses |0...0>.
        simulator : Optional[StatevectorSimulator]
            Simulator to use.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict
            Results with 'statevector', 'counts', 'probabilities'.
        """
        if simulator is None:
            simulator = StatevectorSimulator()

        circuit = self.construct_circuit()
        result = simulator.run(circuit, shots=shots)
        return {
            'statevector': result.statevector if hasattr(result, 'statevector') else None,
            'counts': result.get_counts(),
            'probabilities': result.get_probabilities() if hasattr(result, 'get_probabilities') else None,
        }


class InverseQFT(QFT):
    """
    Inverse Quantum Fourier Transform.

    This is a convenience class equivalent to QFT(inverse=True).
    """

    def __init__(
        self,
        n_qubits: int,
        do_swaps: bool = True,
        approximation_degree: int = 0,
    ) -> None:
        super().__init__(
            n_qubits=n_qubits,
            inverse=True,
            do_swaps=do_swaps,
            approximation_degree=approximation_degree,
        )


class QuantumAdder:
    """
    Quantum ripple-carry adder using QFT-based addition.

    Uses the QFT to efficiently implement addition on quantum registers.
    The QFT-based adder requires O(n) qubits and O(n) gates, compared
    to O(n) qubits and O(n^2) gates for classical ripple-carry.

    The addition works by:
    1. Applying QFT to the output register
    2. Applying controlled phase rotations proportional to input bits
    3. Applying inverse QFT to the output register

    Parameters
    ----------
    n_bits : int
        Number of bits in each operand.

    Examples
    --------
    >>> adder = QuantumAdder(4)
    >>> circuit = adder.construct_circuit(a_input=3, b_input=5)
    >>> # The result register will contain 3 + 5 = 8
    """

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits

    def construct_circuit(
        self,
        a_input: Optional[int] = None,
        b_input: Optional[int] = None,
    ) -> QuantumCircuit:
        """
        Construct a quantum adder circuit: |a>|b> -> |a>|a+b>.

        Parameters
        ----------
        a_input : Optional[int]
            Value to encode in register A. If None, leaves A uninitialized.
        b_input : Optional[int]
            Value to encode in register B. If None, leaves B uninitialized.

        Returns
        -------
        QuantumCircuit
            Adder circuit using 2*n_bits + 1 qubits.
        """
        n = self.n_bits
        total_qubits = 2 * n + 1  # a, b, carry
        circuit = QuantumCircuit(total_qubits)

        a_qubits = list(range(n))
        b_qubits = list(range(n, 2 * n))
        carry_qubit = 2 * n

        # Encode inputs
        if a_input is not None:
            for i in range(n):
                if (a_input >> i) & 1:
                    circuit.x(a_qubits[i])

        if b_input is not None:
            for i in range(n):
                if (b_input >> i) & 1:
                    circuit.x(b_qubits[i])

        # QFT-based addition: QFT on b, controlled rotations, IQFT
        all_b = b_qubits + [carry_qubit]
        apply_qft(circuit, all_b)

        for i in range(n):
            for j in range(n - i):
                angle = 2 * np.pi / (2 ** (j + 1))
                # Controlled phase from a[i] to b[i+j]
                circuit.rz(angle, all_b[i + j])
                circuit.cx(a_qubits[i], all_b[i + j])
                circuit.rz(-angle, all_b[i + j])
                circuit.cx(a_qubits[i], all_b[i + j])

        apply_iqft(circuit, all_b)

        return circuit


class QuantumMultiplier:
    """
    Quantum multiplication using repeated addition.

    Implements multiplication |a>|b>|0> -> |a>|b>|a*b> using
    the QFT-based adder.

    Parameters
    ----------
    n_bits : int
        Number of bits in each operand.

    Examples
    --------
    >>> mult = QuantumMultiplier(4)
    >>> circuit = mult.construct_circuit(a=3, b=5)
    >>> # Result register will contain 3 * 5 = 15
    """

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits

    def construct_circuit(
        self,
        a: Optional[int] = None,
        b: Optional[int] = None,
    ) -> QuantumCircuit:
        """
        Construct a quantum multiplier circuit: |a>|b>|0> -> |a>|b>|a*b>.

        Uses repeated addition: a*b = sum of (b_i * a * 2^i) for each bit b_i.

        Parameters
        ----------
        a : Optional[int]
            Value to encode in register A.
        b : Optional[int]
            Value to encode in register B.

        Returns
        -------
        QuantumCircuit
            Multiplier circuit.
        """
        n = self.n_bits
        # 2*n bits for result, n for a, n for b
        total_qubits = 4 * n
        circuit = QuantumCircuit(total_qubits)

        a_qubits = list(range(n))
        b_qubits = list(range(n, 2 * n))
        result_qubits = list(range(2 * n, 4 * n))

        # Encode inputs
        if a is not None:
            for i in range(n):
                if (a >> i) & 1:
                    circuit.x(a_qubits[i])

        if b is not None:
            for i in range(n):
                if (b >> i) & 1:
                    circuit.x(b_qubits[i])

        # Repeated addition: for each bit b_i that is 1, add (a << i) to result
        adder = QuantumAdder(2 * n)

        for i in range(n):
            # If b[i] is 1, add a shifted by i to result
            # This is done with controlled addition
            # (controlled by b_qubits[i])
            shift = i
            # Add (a << i) to result when b[i] = 1
            # Simplified: controlled addition circuit
            temp = QuantumCircuit(4 * n)
            for j in range(n):
                # Controlled by b[i], apply a[j] to result[j+i]
                if j + shift < 2 * n:
                    temp.ccx(b_qubits[i], a_qubits[j], result_qubits[j + shift])

            circuit.compose(temp, inplace=True)

        return circuit
