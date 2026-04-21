"""
Quantum Noise Models
====================

Models for simulating noise in quantum circuits, including depolarizing
noise, thermal relaxation, and configurable per-gate noise settings.
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple, Union
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
        noisy = type(circuit)(circuit.width() if hasattr(circuit, 'width') else 0)

        for op in circuit._operations if hasattr(circuit, '_operations') else []:
            # Add the original operation
            if hasattr(noisy, 'append'):
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
            for k in kraus:
                circuit.append_kraus(k, qubits)

    @staticmethod
    def _depolarizing_kraus(p: float, n_qubits: int) -> List[np.ndarray]:
        """Kraus operators for depolarizing noise on n qubits."""
        d = 2 ** n_qubits
        identity = np.eye(d, dtype=np.complex128)

        # Depolarizing channel: rho -> (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z) for 1 qubit
        # General: rho -> (1-p)*rho + p * (I*d) / d
        return [np.sqrt(1 - p) * identity, np.sqrt(p / (d * d - 1)) * (identity - np.eye(1, d*d).reshape(d, d) + identity)]

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
