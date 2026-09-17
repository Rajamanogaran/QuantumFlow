"""
Quantum Noise Models
====================

Models for simulating noise in quantum circuits, including depolarizing
noise, thermal relaxation, and configurable per-gate noise settings.
"""

import itertools

import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class NoiseType(Enum):
    """Types of quantum noise."""
    DEPOLARIZING = "depolarizing"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    PAULI_ERROR = "pauli_error"
    THERMAL_RELAXATION = "thermal_relaxation"
    CUSTOM = "custom"


@dataclass
class GateNoise:
    """Noise configuration for a specific gate type."""
    gate_name: str
    noise_type: NoiseType
    error_probability: float = 0.0
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1], got {self.error_probability}")


@dataclass
class QubitNoise:
    """Per-qubit noise configuration."""
    qubit: int
    t1: float = 100e-6       # Relaxation time (seconds)
    t2: float = 50e-6        # Dephasing time (seconds)
    readout_error: float = 0.01  # Readout assignment error probability
    single_gate_error: float = 0.001  # Single-qubit gate error
    two_gate_error: float = 0.01     # Two-qubit gate error


@dataclass
class NoiseConfig:
    """
    Configuration for circuit-level noise.

    Parameters
    ----------
    single_gate_error : float
        Default error probability for single-qubit gates.
    two_gate_error : float
        Default error probability for two-qubit gates.
    measurement_error : float
        Readout error probability.
    noise_type : str
        Default noise type: 'depolarizing', 'amplitude_damping', etc.
    gate_noise : Dict[str, GateNoise]
        Per-gate noise overrides.
    qubit_noise : Dict[int, QubitNoise]
        Per-qubit noise configurations.
    thermal : bool
        Whether to include thermal relaxation noise.
    temperature : float
        Temperature in milliKelvin (for thermal noise).
    """

    def __init__(
        self,
        single_gate_error: float = 0.001,
        two_gate_error: float = 0.01,
        measurement_error: float = 0.01,
        noise_type: str = 'depolarizing',
        gate_noise: Optional[Dict[str, GateNoise]] = None,
        qubit_noise: Optional[Dict[int, QubitNoise]] = None,
        thermal: bool = False,
        temperature: float = 15.0,
    ) -> None:
        self.single_gate_error = single_gate_error
        self.two_gate_error = two_gate_error
        self.measurement_error = measurement_error
        self.noise_type = noise_type
        self.gate_noise = gate_noise or {}
        self.qubit_noise = qubit_noise or {}
        self.thermal = thermal
        self.temperature = temperature

    def get_error_probability(self, gate_name: str, n_qubits: int) -> float:
        """Get error probability for a specific gate."""
        if gate_name in self.gate_noise:
            return self.gate_noise[gate_name].error_probability
        if n_qubits >= 2:
            return self.two_gate_error
        return self.single_gate_error

    def get_noise_type(self, gate_name: str) -> NoiseType:
        """Get noise type for a specific gate."""
        if gate_name in self.gate_noise:
            return self.gate_noise[gate_name].noise_type
        return NoiseType(self.noise_type)


class NoiseModel:
    """
    Quantum noise model for circuit simulation.

    Applies noise channels after each gate in a circuit to simulate
    realistic quantum hardware noise.

    Parameters
    ----------
    config : NoiseConfig
        Noise configuration.

    Examples
    --------
    >>> config = NoiseConfig(single_gate_error=0.01, two_gate_error=0.05)
    >>> noise = NoiseModel(config)
    >>> noisy_circuit = noise.apply_noise(quantum_circuit)
    """

    def __init__(self, config: Optional[NoiseConfig] = None) -> None:
        self.config = config or NoiseConfig()
        self._noise_cache: Dict[str, np.ndarray] = {}

    def apply_noise(self, circuit: Any, noise_scale: float = 1.0) -> Any:
        """
        Apply noise model to a quantum circuit.

        Creates a new circuit with noise channels inserted after each gate.

        Parameters
        ----------
        circuit : QuantumCircuit
            Input circuit.
        noise_scale : float
            Scale factor for noise probabilities. Used in error mitigation.

        Returns
        -------
        QuantumCircuit
            Noisy circuit.
        """
        noisy = type(circuit)(circuit.width if hasattr(circuit, 'width') else 0)

        for op in circuit._data if hasattr(circuit, '_data') else (circuit._operations if hasattr(circuit, '_operations') else []):
            # Add the original operation - extract gate, qubits, params from Operation
            if hasattr(noisy, 'append'):
                gate = getattr(op, 'gate', None)
                qubits = getattr(op, 'qubits', None)
                params = getattr(op, 'params', None)
                label = getattr(op, 'label', None)
                if gate is not None and hasattr(noisy, '_data'):
                    # QuantumCircuit: append Gate with qubits and params
                    from quantumflow.core.operation import Barrier, Reset
                    if isinstance(op, (Barrier, Reset)):
                        noisy._data.append(op)
                    else:
                        noisy.append(gate, qubits, params, label=label)
                elif hasattr(op, 'apply_to'):
                    op.apply_to(noisy)
                else:
                    noisy.append(op)
            elif hasattr(op, 'apply_to'):
                op.apply_to(noisy)

            # Add noise after the operation
            gate_name = getattr(op, 'name', type(op).__name__)
            n_qubits = getattr(op, 'num_qubits', 1)
            qubits = getattr(op, 'qubits', list(range(n_qubits)))

            error_prob = self.config.get_error_probability(gate_name, n_qubits) * noise_scale
            if error_prob > 0:
                self._add_noise_channel(noisy, qubits, error_prob, gate_name)

        return noisy

    def after_gate(
        self,
        rho: np.ndarray,
        gate: Any,
        qubits: Any,
        num_qubits: int,
    ) -> np.ndarray:
        """Apply this model's channel for ``gate`` acting on ``qubits``.

        Called by the density-matrix backend after every gate when a
        noise model is attached to
        :class:`~quantumflow.simulation.simulator.DensityMatrixSimulator`.
        ``qubits`` may be a tuple (Operation) or list; identity / no-op
        gates are skipped.

        Parameters
        ----------
        rho : numpy.ndarray
            Current density matrix of all ``num_qubits`` qubits.
        gate : Gate
            The gate that was just applied.
        qubits : sequence of int
        num_qubits : int

        Returns
        -------
        numpy.ndarray
            The (possibly) noise-updated density matrix.
        """
        qubits = list(qubits)
        if not qubits:
            return rho
        gate_name = getattr(gate, "name", type(gate).__name__)
        error_prob = self.config.get_error_probability(gate_name, len(qubits))
        if error_prob <= 0:
            return rho

        noise_type = self.config.get_noise_type(gate_name)
        if noise_type == NoiseType.AMPLITUDE_DAMPING:
            kraus = self._amplitude_damping_kraus(error_prob)
        elif noise_type == NoiseType.PHASE_DAMPING:
            kraus = self._phase_damping_kraus(error_prob)
        elif noise_type == NoiseType.BIT_FLIP:
            kraus = self._bit_flip_kraus(error_prob)
        elif noise_type == NoiseType.PHASE_FLIP:
            kraus = self._phase_flip_kraus(error_prob)
        else:
            kraus = self._depolarizing_kraus(error_prob, len(qubits))

        embedded = [self._embed(k, qubits, num_qubits) for k in kraus]
        out = np.zeros_like(rho)
        for e in embedded:
            out += e @ rho @ e.conj().T
        return out

    @classmethod
    def _embed(cls, k: np.ndarray, qubits: List[int], n: int) -> np.ndarray:
        """Embed a ``2**len(qubits)`` operator into the full ``2**n`` space.

        Qubit 0 of the operator maps to ``qubits[0]`` (MSB-first
        convention, matching the simulators).
        """
        n_gate = len(qubits)
        dim_gate = 2 ** n_gate
        if k.shape != (dim_gate, dim_gate):
            raise ValueError(
                f"Kraus operator shape {k.shape} does not match "
                f"{n_gate} qubit(s)"
            )
        if n_gate == n and qubits == list(range(n)):
            return k
        full = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
        for col in range(2 ** n):
            col_bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
            col_local = sum(
                col_bits[q] << (n_gate - 1 - i) for i, q in enumerate(qubits)
            )
            for row_local in range(dim_gate):
                coef = k[row_local, col_local]
                if coef == 0:
                    continue
                row_bits = list(col_bits)
                for i, q in enumerate(qubits):
                    row_bits[q] = (row_local >> (n_gate - 1 - i)) & 1
                row = sum(b << (n - 1 - q) for q, b in enumerate(row_bits))
                full[row, col] += coef
        return full

    def _add_noise_channel(
        self,
        circuit: Any,
        qubits: List[int],
        error_prob: float,
        gate_name: str,
    ) -> None:
        """Add a noise channel after a gate."""
        noise_type = self.config.get_noise_type(gate_name)

        if noise_type == NoiseType.DEPOLARIZING:
            kraus = self._depolarizing_kraus(error_prob, len(qubits))
        elif noise_type == NoiseType.AMPLITUDE_DAMPING:
            kraus = self._amplitude_damping_kraus(error_prob)
        elif noise_type == NoiseType.PHASE_DAMPING:
            kraus = self._phase_damping_kraus(error_prob)
        elif noise_type == NoiseType.BIT_FLIP:
            kraus = self._bit_flip_kraus(error_prob)
        elif noise_type == NoiseType.PHASE_FLIP:
            kraus = self._phase_flip_kraus(error_prob)
        else:
            kraus = self._depolarizing_kraus(error_prob, len(qubits))

        if hasattr(circuit, 'append_kraus'):
            # QuantumCircuit: attach the whole channel as one instruction
            # (executed by the density-matrix simulator).
            circuit.append_kraus(kraus, qubits)

    @staticmethod
    def _depolarizing_kraus(p: float, n_qubits: int) -> List[np.ndarray]:
        """Kraus operators for depolarizing noise on n qubits.

        With probability ``1 - p`` the state is unchanged; the
        non-identity part applies each generalised Pauli equally:
        ``rho -> (1-p) rho + p/(4^n - 1) sum_i W_i rho W_i^dag``.
        Kraus operators: ``K_0 = sqrt(1-p) I`` and
        ``K_i = sqrt(p/(4^n - 1)) W_i`` over the ``4^n - 1``
        non-identity Weyl (generalised Pauli) operators, which satisfies
        the completeness relation exactly.
        (A previous version returned two operators that did not satisfy
        the completeness relation, so the channel was not physical.)
        """
        d = 2 ** n_qubits
        paulis = [
            np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
        ]
        # All non-identity generalised Paulis: the 4^n - 1 tensor
        # products of I/X/Y/Z over the n qubits, excluding the identity.
        weyls = []
        for combo in itertools.product(range(4), repeat=n_qubits):
            if all(c == 0 for c in combo):
                continue
            mat = paulis[combo[0]]
            for c in combo[1:]:
                mat = np.kron(mat, paulis[c])
            weyls.append(mat)
        kraus = [np.sqrt(1.0 - p) * np.eye(d, dtype=np.complex128)]
        for w in weyls:
            kraus.append(np.sqrt(p / (d * d - 1)) * w)
        return kraus

    @staticmethod
    def _amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
        """Kraus operators for amplitude damping."""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
        return [K0, K1]

    @staticmethod
    def _phase_damping_kraus(lam: float) -> List[np.ndarray]:
        """Kraus operators for phase damping."""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128)
        K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128)
        return [K0, K1]

    @staticmethod
    def _bit_flip_kraus(p: float) -> List[np.ndarray]:
        """Kraus operators for bit flip."""
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=np.complex128)
        K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
        return [K0, K1]

    @staticmethod
    def _phase_flip_kraus(p: float) -> List[np.ndarray]:
        """Kraus operators for phase flip."""
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=np.complex128)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=np.complex128)
        return [K0, K1]

    def get_noise_matrix(self, noise_type: str, p: float, n_qubits: int = 1) -> np.ndarray:
        """Get the noise superoperator (Chi matrix) for visualization."""
        key = f"{noise_type}_{p}_{n_qubits}"
        if key not in self._noise_cache:
            if noise_type == 'depolarizing':
                d = 2 ** n_qubits
                chi = np.zeros((d*d, d*d), dtype=np.complex128)
                chi[0, 0] = 1 - p
                for i in range(1, d*d):
                    chi[i, i] = p / (d*d - 1)
                self._noise_cache[key] = chi
            elif noise_type == 'bit_flip':
                chi = np.diag([1-p, p])
                self._noise_cache[key] = chi
            elif noise_type == 'phase_flip':
                chi = np.diag([1-p, p])
                self._noise_cache[key] = chi
            elif noise_type == 'amplitude_damping':
                chi = np.array([
                    [1, 0, 0, p],
                    [0, 1-p, 0, 0],
                    [0, 0, np.sqrt(p*(1-p)), 0],
                    [p, 0, 0, 1-p],
                ], dtype=np.complex128)
                self._noise_cache[key] = chi
            else:
                self._noise_cache[key] = np.eye(4, dtype=np.complex128)

        return self._noise_cache[key]
