"""
Quantum Error Channels
======================

Mathematical representations of quantum noise as completely positive
trace-preserving (CPTP) maps via Kraus operators.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class ErrorChannel(ABC):
    """
    Abstract base class for quantum error channels.

    An error channel is a completely positive trace-preserving (CPTP) map
    described by a set of Kraus operators {E_k} satisfying:
        sum_k E_k^dagger E_k = I

    The channel acts on a density matrix as:
        rho' = sum_k E_k * rho * E_k^dagger
    """

    def __init__(self, n_qubits: int = 1) -> None:
        self.n_qubits = n_qubits
        self._dim = 2 ** n_qubits

    @abstractmethod
    def kraus_operators(self) -> List[np.ndarray]:
        """Return the Kraus operators for this channel."""
        pass

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply this error channel to a density matrix."""
        kraus = self.kraus_operators()
        result = np.zeros_like(rho)
        for K in kraus:
            result += K @ rho @ K.conj().T
        return result

    def apply_to_statevector(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply error channel to a pure state.

        Converts to density matrix, applies channel, returns
        the resulting density matrix.
        """
        rho = np.outer(psi, psi.conj())
        return self.apply(rho)

    def superoperator(self) -> np.ndarray:
        """
        Compute the superoperator (Liouville) representation.

        Returns a matrix S such that vec(rho') = S @ vec(rho).
        """
        d = self._dim
        S = np.zeros((d*d, d*d), dtype=np.complex128)
        for K in self.kraus_operators():
            S += np.kron(K.conj(), K)
        return S

    def choi_matrix(self) -> np.ndarray:
        """
        Compute the Choi matrix representation.

        C = sum_k |vec(E_k)><vec(E_k)|
        """
        d = self._dim
        C = np.zeros((d*d, d*d), dtype=np.complex128)
        for K in self.kraus_operators():
            vec_K = K.reshape(-1, 1)
            C += vec_K @ vec_K.conj().T
        return C

    def is_cptp(self, atol: float = 1e-10) -> bool:
        """Verify that this channel is CPTP."""
        kraus = self.kraus_operators()
        # Trace-preserving: sum K^dagger K = I
        sum_kdk = sum(K.conj().T @ K for K in kraus)
        identity = np.eye(self._dim, dtype=np.complex128)
        return np.allclose(sum_kdk, identity, atol=atol)

    def fidelity(self, rho_in: np.ndarray, rho_out: np.ndarray) -> float:
        """Compute the fidelity between input and output states."""
        return float(np.abs(np.sqrt(rho_in.conj().T @ rho_out @ rho_in.conj()).trace()))


class DepolarizingChannel(ErrorChannel):
    """
    Depolarizing noise channel.

    With probability p, replaces the state with the maximally mixed state.
    With probability 1-p, leaves the state unchanged.

    For a single qubit:
        rho -> (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)

    Parameters
    ----------
    error_probability : float
        Depolarizing probability p in [0, 1].
    """

    def __init__(self, error_probability: float, n_qubits: int = 1) -> None:
        super().__init__(n_qubits)
        self.error_probability = error_probability

    def kraus_operators(self) -> List[np.ndarray]:
        d = self._dim
        p = self.error_probability
        sqrt_1mp = np.sqrt(1 - p)
        K0 = sqrt_1mp * np.eye(d, dtype=np.complex128)

        kraus = [K0]
        paulis = [
            np.array([[0, 1], [1, 0]], dtype=np.complex128),   # X
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # Y
            np.array([[1, 0], [0, -1]], dtype=np.complex128),   # Z
        ]

        sqrt_p = np.sqrt(p / 3.0)
        if self.n_qubits == 1:
            for P in paulis:
                kraus.append(sqrt_p * P)
        else:
            # Multi-qubit: depolarize each qubit independently
            for q in range(self.n_qubits):
                for P in paulis:
                    op = np.eye(d, dtype=np.complex128)
                    idx = [[1, 0], [0, 1]]
                    op_p = np.kron(np.kron(np.eye(2**q, dtype=np.complex128), P),
                                   np.eye(2**(self.n_qubits - q - 1), dtype=np.complex128))
                    kraus.append(sqrt_p * op_p)

        return kraus


class AmplitudeDampingChannel(ErrorChannel):
    """
    Amplitude damping channel (T1 relaxation).

    Models energy dissipation: |1> -> |0> with probability gamma.
    This models spontaneous emission / T1 decay.

    Kraus operators:
        K0 = [[1, 0], [0, sqrt(1-gamma)]]
        K1 = [[0, sqrt(gamma)], [0, 0]]

    Parameters
    ----------
    gamma : float
        Damping probability in [0, 1].
    """

    def __init__(self, gamma: float, n_qubits: int = 1) -> None:
        super().__init__(n_qubits)
        self.gamma = gamma

    def kraus_operators(self) -> List[np.ndarray]:
        d = self._dim
        g = self.gamma
        sqrt_1mg = np.sqrt(1 - g)
        sqrt_g = np.sqrt(g)

        if self.n_qubits == 1:
            K0 = np.array([[1, 0], [0, sqrt_1mg]], dtype=np.complex128)
            K1 = np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128)
            return [K0, K1]
        else:
            kraus = [np.eye(d, dtype=np.complex128)]
            for q in range(self.n_qubits):
                new_kraus = []
                K0 = np.array([[1, 0], [0, sqrt_1mg]], dtype=np.complex128)
                K1 = np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128)
                for existing in kraus:
                    left = np.eye(2**q, dtype=np.complex128)
                    right = np.eye(2**(self.n_qubits - q - 1), dtype=np.complex128)
                    new_kraus.append(np.kron(np.kron(left, K0), right) @ existing)
                    new_kraus.append(np.kron(np.kron(left, K1), right) @ existing)
                kraus = new_kraus
            return kraus

    @property
    def t1_from_gamma(self, gate_time: float) -> float:
        """Estimate T1 from gamma and gate time: T1 = -gate_time / ln(1-gamma)."""
        if self.gamma >= 1 or self.gamma <= 0:
            return float('inf')
        return -gate_time / np.log(1 - self.gamma)


class PhaseDampingChannel(ErrorChannel):
    """
    Phase damping (dephasing) channel (T2 process).

    Models loss of phase coherence without energy loss.
    |+> -> |-> with probability lambda.

    Kraus operators:
        K0 = [[1, 0], [0, sqrt(1-lambda)]]
        K1 = [[0, 0], [0, sqrt(lambda)]]

    Parameters
    ----------
    lambda_param : float
        Phase damping probability in [0, 1].
    """

    def __init__(self, lambda_param: float, n_qubits: int = 1) -> None:
        super().__init__(n_qubits)
        self.lambda_param = lambda_param

    def kraus_operators(self) -> List[np.ndarray]:
        d = self._dim
        lam = self.lambda_param
        sqrt_1ml = np.sqrt(1 - lam)
        sqrt_l = np.sqrt(lam)

        if self.n_qubits == 1:
            K0 = np.array([[1, 0], [0, sqrt_1ml]], dtype=np.complex128)
            K1 = np.array([[0, 0], [0, sqrt_l]], dtype=np.complex128)
            return [K0, K1]
        else:
            kraus = [np.eye(d, dtype=np.complex128)]
            for q in range(self.n_qubits):
                new_kraus = []
                K0 = np.array([[1, 0], [0, sqrt_1ml]], dtype=np.complex128)
                K1 = np.array([[0, 0], [0, sqrt_l]], dtype=np.complex128)
                for existing in kraus:
                    left = np.eye(2**q, dtype=np.complex128)
                    right = np.eye(2**(self.n_qubits - q - 1), dtype=np.complex128)
                    new_kraus.append(np.kron(np.kron(left, K0), right) @ existing)
                    new_kraus.append(np.kron(np.kron(left, K1), right) @ existing)
                kraus = new_kraus
            return kraus


class BitFlipChannel(ErrorChannel):
    """
    Bit flip error channel.

    Flips the qubit state |0><->|1> with probability p.

    Kraus operators:
        K0 = sqrt(1-p) * I
        K1 = sqrt(p) * X

    Parameters
    ----------
    p : float
        Bit flip probability in [0, 1].
    """

    def __init__(self, p: float, n_qubits: int = 1) -> None:
        super().__init__(n_qubits)
        self.p = p

    def kraus_operators(self) -> List[np.ndarray]:
        d = self._dim
        K0 = np.sqrt(1 - self.p) * np.eye(d, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        K1 = np.sqrt(self.p) * X if self.n_qubits == 1 else np.sqrt(self.p) * X
        return [K0, K1]


class PhaseFlipChannel(ErrorChannel):
    """
    Phase flip error channel.

    Applies a Z gate with probability p.

    Kraus operators:
        K0 = sqrt(1-p) * I
        K1 = sqrt(p) * Z

    Parameters
    ----------
    p : float
        Phase flip probability in [0, 1].
    """

    def __init__(self, p: float, n_qubits: int = 1) -> None:
        super().__init__(n_qubits)
        self.p = p

    def kraus_operators(self) -> List[np.ndarray]:
        d = self._dim
        K0 = np.sqrt(1 - self.p) * np.eye(d, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        K1 = np.sqrt(self.p) * Z
        return [K0, K1]


class PauliErrorChannel(ErrorChannel):
    """
    General Pauli error channel.

    Applies Pauli X, Y, Z errors with independent probabilities.

    rho -> (1-px-py-pz)*rho + px*X*rho*X + py*Y*rho*Y + pz*Z*rho*Z

    Parameters
    ----------
    px : float
        X error probability.
    py : float
        Y error probability.
    pz : float
        Z error probability.
    """

    def __init__(self, px: float = 0.0, py: float = 0.0, pz: float = 0.0) -> None:
        super().__init__(n_qubits=1)
        self.px = px
        self.py = py
        self.pz = pz

    def kraus_operators(self) -> List[np.ndarray]:
        p_identity = 1 - self.px - self.py - self.pz
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        kraus = []
        if p_identity > 0:
            kraus.append(np.sqrt(p_identity) * I)
        if self.px > 0:
            kraus.append(np.sqrt(self.px) * X)
        if self.py > 0:
            kraus.append(np.sqrt(self.py) * Y)
        if self.pz > 0:
            kraus.append(np.sqrt(self.pz) * Z)
        return kraus


class ThermalRelaxationChannel(ErrorChannel):
    """
    Thermal relaxation channel combining T1 and T2 processes.

    Combines amplitude damping (T1) and phase damping (T2) into
    a single physically motivated channel.

    The effective dephasing rate accounts for both pure dephasing
    and relaxation-induced dephasing:
        1/T2* = 1/T2 + 1/(2*T1)

    Parameters
    ----------
    t1 : float
        T1 relaxation time in seconds.
    t2 : float
        T2 dephasing time in seconds.
    gate_time : float
        Gate operation time in seconds.
    temperature : float
        Temperature in Kelvin (default: 0 K, no thermal excitation).
    """

    def __init__(
        self,
        t1: float = 100e-6,
        t2: float = 50e-6,
        gate_time: float = 100e-9,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(n_qubits=1)
        self.t1 = t1
        self.t2 = t2
        self.gate_time = gate_time
        self.temperature = temperature

        # Compute effective rates
        self.gamma = 1 - np.exp(-gate_time / t1) if t1 > 0 else 0.0
        rate_t2_star = 1/t2 if t2 > 0 else 0.0
        rate_t1_half = 1/(2*t1) if t1 > 0 else 0.0
        lambda_param = 1 - np.exp(-gate_time * (rate_t2_star - rate_t1_half))
        self.lambda_param = max(0, min(1, lambda_param))

        # Thermal excitation probability (Boltzmann distribution)
        if temperature > 0:
            freq = 5e9  # ~5 GHz qubit frequency
            kb = 1.381e-23  # Boltzmann constant
            h = 6.626e-34  # Planck constant
            energy = h * freq
            self.thermal_prob = np.exp(-energy / (kb * temperature))
            self.thermal_prob = min(0.5, self.thermal_prob / (1 + self.thermal_prob))
        else:
            self.thermal_prob = 0.0

    def kraus_operators(self) -> List[np.ndarray]:
        g = self.gamma
        lam = self.lambda_param
        p_th = self.thermal_prob
        sqrt_1mg = np.sqrt(1 - g)
        sqrt_g = np.sqrt(g)
        sqrt_lam = np.sqrt(lam)

        # Combined amplitude + phase damping
        K0 = np.array([[sqrt_1mg, 0], [0, sqrt_1mg * np.sqrt(1 - lam)]], dtype=np.complex128)
        K1 = np.array([[0, sqrt_g], [0, 0]], dtype=np.complex128)
        K2 = np.array([[0, 0], [0, sqrt_1mg * sqrt_lam]], dtype=np.complex128)

        kraus = [K0, K1, K2]

        # Add thermal excitation
        if p_th > 0:
            K3 = np.array([[0, 0], [np.sqrt(p_th), 0]], dtype=np.complex128)
            kraus.append(K3)

        return kraus
