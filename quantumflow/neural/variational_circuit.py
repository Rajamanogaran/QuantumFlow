"""
Variational Quantum Circuits
==============================

Provides parameterized quantum circuits for Variational Quantum Algorithms
(VQA), including Variational Quantum Eigensolver (VQE), Quantum Approximate
Optimization Algorithm (QAOA), and general hybrid quantum-classical
optimization.

Classes
-------
* :class:`VariationalCircuit` — full parameterized quantum circuit for VQA.
* :class:`AngleEncoder` — encode classical data via angle encoding with
  configurable feature maps.
* :class:`AmplitudeEncoder` — encode classical data via amplitude encoding
  with automatic padding and normalisation.
* :class:`HardwareEfficientAnsatz` — hardware-efficient ansatz circuit
  with configurable entanglement and rotation gates.
* :class:`StronglyEntanglingAnsatz` — strongly entangling layers with
  parameterized 3-rotation blocks and CNOT entanglement.

The circuits are designed to be differentiable, returning numpy arrays that
can be used with autograd frameworks (JAX, TensorFlow, PyTorch).

Examples
--------
>>> vc = VariationalCircuit(n_qubits=4, n_layers=3, entanglement='full')
>>> vc.count_parameters()
36
>>> output = vc.forward([0.1, 0.2, 0.3, 0.4])
>>> grads = vc.gradients(list(range(vc.count_parameters())))

>>> hea = HardwareEfficientAnsatz(n_qubits=3, n_layers=2, rotations='rycz')
>>> circuit = hea.circuit([0.5] * hea.count_parameters())

>>> sea = StronglyEntanglingAnsatz(n_qubits=3, n_layers=2)
>>> circuit = sea.circuit(sea.random_params())
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import (
    CNOTGate,
    CZGate,
    HGate,
    RXXGate,
    RXGate,
    RYYGate,
    RYGate,
    RZZGate,
    RZGate,
)

__all__ = [
    "VariationalCircuit",
    "AngleEncoder",
    "AmplitudeEncoder",
    "HardwareEfficientAnsatz",
    "StronglyEntanglingAnsatz",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_VALID_ENTANGLEMENTS = frozenset({
    "linear", "circular", "full", "pairwise", "star", "alternating",
})

_VALID_ROTATION_SETS = frozenset({
    "rx", "ry", "rz", "rycz", "rxyz", "czrx", "xyz",
})

_VALID_FEATURE_MAPS = frozenset({
    "zx", "zz", "zxyz",
})


# ---------------------------------------------------------------------------
# Entanglement utilities
# ---------------------------------------------------------------------------

def _get_entanglement_edges(
    n_qubits: int,
    pattern: str,
    layer_index: int = 0,
) -> List[Tuple[int, int]]:
    """Return list of (control, target) pairs for an entanglement pattern.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    pattern : str
        Entanglement pattern name.
    layer_index : int
        Current layer index (used for alternating patterns).

    Returns
    -------
    list of (int, int)
    """
    edges: List[Tuple[int, int]] = []

    if pattern == "linear":
        for i in range(n_qubits - 1):
            edges.append((i, i + 1))

    elif pattern == "circular":
        for i in range(n_qubits):
            edges.append((i, (i + 1) % n_qubits))

    elif pattern == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                edges.append((i, j))

    elif pattern == "pairwise":
        for i in range(0, n_qubits - 1, 2):
            edges.append((i, min(i + 1, n_qubits - 1)))

    elif pattern == "star":
        center = layer_index % n_qubits
        for i in range(n_qubits):
            if i != center:
                edges.append((center, i))

    elif pattern == "alternating":
        # Even layers: linear forward. Odd layers: linear backward.
        if layer_index % 2 == 0:
            for i in range(0, n_qubits - 1, 2):
                edges.append((i, min(i + 1, n_qubits - 1)))
            for i in range(1, n_qubits - 1, 2):
                edges.append((i, min(i + 1, n_qubits - 1)))
        else:
            for i in range(n_qubits - 2, -1, -2):
                edges.append((min(i + 1, n_qubits - 1), i))
            for i in range(n_qubits - 3, 0, -2):
                edges.append((min(i + 1, n_qubits - 1), i))

    return edges


def _apply_rotation(
    circuit: QuantumCircuit,
    gate_name: str,
    theta: float,
    qubit: int,
) -> None:
    """Apply a single-qubit rotation gate.

    Parameters
    ----------
    circuit : QuantumCircuit
        Target circuit.
    gate_name : str
        ``'rx'``, ``'ry'``, or ``'rz'``.
    theta : float
        Rotation angle.
    qubit : int
        Target qubit.
    """
    if gate_name == "rx":
        circuit.rx(theta, qubit)
    elif gate_name == "ry":
        circuit.ry(theta, qubit)
    elif gate_name == "rz":
        circuit.rz(theta, qubit)
    else:
        raise ValueError(f"Unsupported rotation gate: {gate_name}")


def _parse_rotation_set(rotations: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Parse rotation specification into a tuple of gate names.

    Parameters
    ----------
    rotations : str or tuple of str
        Rotation specification.

    Returns
    -------
    tuple of str
        Gate names.
    """
    if isinstance(rotations, str):
        preset_map = {
            "rx": ("rx",),
            "ry": ("ry",),
            "rz": ("rz",),
            "rycz": ("ry", "rz"),
            "rxyz": ("rx", "ry", "rz"),
            "czrx": ("rz", "rx"),
            "xyz": ("rx", "ry", "rz"),
        }
        if rotations in preset_map:
            return preset_map[rotations]
        raise ValueError(
            f"Unknown rotation preset '{rotations}'. "
            f"Choose from {sorted(preset_map.keys())}"
        )
    return tuple(rotations)


# ---------------------------------------------------------------------------
# AngleEncoder
# ---------------------------------------------------------------------------

class AngleEncoder:
    """Encode classical data via angle encoding with configurable feature maps.

    Supports multiple feature maps for different encoding expressiveness:

    * ``'zx'`` — Z-X feature map: ``H → RZ(x_i)`` per qubit, then
      entangle, then ``RX(x_j)``.
    * ``'zz'`` — Z-Z feature map: ``RZ(x_i) → ZZ entanglement → RZ(x_j)``.
    * ``'zxyz'`` — Z-XY-Z feature map: full rotation chain with
      entangling layers.

    Data re-uploading can be enabled to repeat the encoding across
    multiple layers for increased expressiveness.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    feature_map : str, optional
        Feature map type: ``'zx'``, ``'zz'``, ``'zxyz'``.
    n_reuploads : int, optional
        Number of data re-uploading rounds. Default ``1``.
    entanglement : str, optional
        Entanglement pattern for feature map entangling layers.

    Examples
    --------
    >>> enc = AngleEncoder(4, feature_map='zx', n_reuploads=2)
    >>> qc = QuantumCircuit(4)
    >>> enc.encode(qc, [0.1, 0.2, 0.3, 0.4])
    >>> enc.input_dim
    4
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map: str = "zx",
        n_reuploads: int = 1,
        entanglement: str = "linear",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if feature_map not in _VALID_FEATURE_MAPS:
            raise ValueError(
                f"Unknown feature map '{feature_map}'. "
                f"Choose from {sorted(_VALID_FEATURE_MAPS)}"
            )
        if n_reuploads < 1:
            raise ValueError(f"n_reuploads must be >= 1, got {n_reuploads}")

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._n_reuploads = n_reuploads
        self._entanglement = entanglement

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def feature_map(self) -> str:
        """str: Feature map type."""
        return self._feature_map

    @property
    def input_dim(self) -> int:
        """int: Expected input dimensionality."""
        return self._n_qubits

    def encode(
        self,
        circuit: QuantumCircuit,
        data: Sequence[float],
    ) -> None:
        """Encode classical data onto *circuit* (in-place).

        Parameters
        ----------
        circuit : QuantumCircuit
            Target quantum circuit.
        data : sequence of float
            Input data of length ``n_qubits``.
        """
        data = np.asarray(data, dtype=np.float64)
        if len(data) != self._n_qubits:
            raise ValueError(
                f"Expected {self._n_qubits} features, got {len(data)}"
            )

        for r in range(self._n_reuploads):
            self._apply_feature_map(circuit, data, layer=r)

    def _apply_feature_map(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
        layer: int,
    ) -> None:
        """Apply a single round of the feature map."""
        if self._feature_map == "zx":
            self._feature_map_zx(circuit, data, layer)
        elif self._feature_map == "zz":
            self._feature_map_zz(circuit, data, layer)
        elif self._feature_map == "zxyz":
            self._feature_map_zxyz(circuit, data, layer)

    def _feature_map_zx(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
        layer: int,
    ) -> None:
        """Z-X feature map.

        For each round:
            1. H on each qubit, then RZ(x_i).
            2. Entangling CNOT layer.
            3. RX(x_i) on each qubit.
        """
        # H + RZ layer
        for i, val in enumerate(data):
            circuit.h(i)
            circuit.rz(float(val) + layer * _PI / self._n_reuploads, i)

        # Entangling layer
        edges = _get_entanglement_edges(
            self._n_qubits, self._entanglement, layer
        )
        for ctrl, tgt in edges:
            circuit.cx(ctrl, tgt)

        # RX layer
        for i, val in enumerate(data):
            circuit.rx(float(val) + layer * _PI / self._n_reuploads, i)

    def _feature_map_zz(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
        layer: int,
    ) -> None:
        """Z-Z feature map.

        For each round:
            1. RZ(x_i) on each qubit.
            2. ZZ entangling layer.
            3. RZ(x_i) on each qubit again.
        """
        phase = layer * _PI / max(1, self._n_reuploads)

        for i, val in enumerate(data):
            circuit.rz(float(val) * (1 + phase), i)

        edges = _get_entanglement_edges(
            self._n_qubits, self._entanglement, layer
        )
        for ctrl, tgt in edges:
            circuit.rzz(float(np.pi * (data[ctrl] + data[tgt]) / (2.0 + phase)),
                        ctrl, tgt)

        for i, val in enumerate(data):
            circuit.rz(float(val) * (1 + phase), i)

    def _feature_map_zxyz(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
        layer: int,
    ) -> None:
        """Z-XY-Z feature map.

        Full rotation chain:
            1. RZ(x_i)
            2. Entangling layer
            3. RX(x_i)
            4. Entangling layer
            5. RY(x_i)
            6. Entangling layer
            7. RZ(x_i)
        """
        phase = layer * _PI / max(1, self._n_reuploads)
        n = self._n_qubits

        # RZ
        for i, val in enumerate(data):
            circuit.rz(float(val) + phase, i)

        # Entangle + RX
        edges = _get_entanglement_edges(n, self._entanglement, 3 * layer)
        for ctrl, tgt in edges:
            circuit.cx(ctrl, tgt)
        for i, val in enumerate(data):
            circuit.rx(float(val) + phase, i)

        # Entangle + RY
        edges = _get_entanglement_edges(n, self._entanglement, 3 * layer + 1)
        for ctrl, tgt in edges:
            circuit.cx(ctrl, tgt)
        for i, val in enumerate(data):
            circuit.ry(float(val) + phase, i)

        # Entangle + RZ
        edges = _get_entanglement_edges(n, self._entanglement, 3 * layer + 2)
        for ctrl, tgt in edges:
            circuit.cx(ctrl, tgt)
        for i, val in enumerate(data):
            circuit.rz(float(val) + phase, i)

    def get_encoding_circuit(
        self,
        data: Sequence[float],
    ) -> QuantumCircuit:
        """Build a standalone encoding circuit.

        Parameters
        ----------
        data : sequence of float

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        self.encode(qc, data)
        return qc

    def __repr__(self) -> str:
        return (
            f"AngleEncoder(n_qubits={self._n_qubits}, "
            f"feature_map={self._feature_map!r}, "
            f"n_reuploads={self._n_reuploads})"
        )


# ---------------------------------------------------------------------------
# AmplitudeEncoder
# ---------------------------------------------------------------------------

class AmplitudeEncoder:
    """Encode classical data via amplitude encoding.

    Normalizes the data vector and encodes it as the amplitude vector
    of a quantum state. Automatically pads non-power-of-2 dimension
    vectors with zeros.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines maximum encoding capacity
        of ``2^n_qubits`` features).
    normalize : bool, optional
        Whether to L2-normalize the data. Default ``True``.
    pad_value : float, optional
        Value to use for padding. Default ``0.0``.

    Examples
    --------
    >>> enc = AmplitudeEncoder(3)  # can encode up to 8 features
    >>> enc.input_dim
    8
    >>> qc = QuantumCircuit(3)
    >>> enc.encode(qc, [1.0, 2.0, 3.0, 4.0])
    """

    def __init__(
        self,
        n_qubits: int,
        normalize: bool = True,
        pad_value: float = 0.0,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self._n_qubits = n_qubits
        self._normalize = normalize
        self._pad_value = pad_value

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def input_dim(self) -> int:
        """int: Maximum encodable features (2^n_qubits)."""
        return 1 << self._n_qubits

    def prepare_data(
        self,
        data: Sequence[float],
    ) -> np.ndarray:
        """Prepare data for amplitude encoding.

        Pads to power of 2 and optionally normalises.

        Parameters
        ----------
        data : sequence of float
            Raw input data.

        Returns
        -------
        numpy.ndarray
            Prepared data of length ``2^n_qubits``.
        """
        data = np.asarray(data, dtype=np.complex128)
        max_dim = self.input_dim

        if len(data) > max_dim:
            raise ValueError(
                f"Data length {len(data)} exceeds maximum encodable "
                f"dimension {max_dim} for {self._n_qubits} qubits"
            )

        # Pad if necessary
        if len(data) < max_dim:
            padded = np.full(max_dim, self._pad_value, dtype=np.complex128)
            padded[:len(data)] = data
            data = padded

        # Normalize
        if self._normalize:
            norm = np.linalg.norm(data)
            if norm < 1e-12:
                raise ValueError(
                    "Cannot normalize data with near-zero norm. "
                    "Check input values."
                )
            data = data / norm

        return data

    def encode(
        self,
        circuit: QuantumCircuit,
        data: Sequence[float],
    ) -> None:
        """Encode classical data onto *circuit* using amplitude encoding.

        Uses a recursive rotation-based approach to prepare the
        amplitude-encoded state from ``|0...0⟩``.

        Parameters
        ----------
        circuit : QuantumCircuit
            Target quantum circuit.
        data : sequence of float
            Input data (up to ``2^n_qubits`` features).
        """
        amplitudes = self.prepare_data(data)
        self._encode_recursive(circuit, amplitudes, start_qubit=0)

    def _encode_recursive(
        self,
        circuit: QuantumCircuit,
        amplitudes: np.ndarray,
        start_qubit: int,
    ) -> None:
        """Recursively encode amplitudes using rotations.

        Splits the amplitude vector in half, computes the angle needed
        to achieve the correct measurement probability, applies the
        rotation, and recurses on both halves.
        """
        n = len(amplitudes)
        if n <= 1:
            return

        mid = n // 2
        left = amplitudes[:mid]
        right = amplitudes[mid:]

        # Probability of measuring 0 on this qubit
        prob_0 = float(np.sum(np.abs(left) ** 2))
        prob_0 = np.clip(prob_0, 0.0, 1.0)

        # Rotation angle: probability = cos²(θ/2)
        theta = 2.0 * math.acos(math.sqrt(prob_0))
        circuit.ry(theta, start_qubit)

        # Phase correction between left and right halves
        if n >= 2:
            phase_l = np.angle(left[0]) if np.abs(left[0]) > 1e-12 else 0.0
            phase_r = np.angle(right[0]) if np.abs(right[0]) > 1e-12 else 0.0
            delta = phase_r - phase_l
            if abs(delta) > 1e-12:
                circuit.rz(delta, start_qubit)

        # Recurse on left half
        if mid > 1:
            self._encode_recursive(circuit, left, start_qubit + 1)

        # Recurse on right half (controlled by X)
        if mid > 1:
            circuit.x(start_qubit)
            self._encode_recursive(circuit, right, start_qubit + 1)
            circuit.x(start_qubit)

    def get_encoding_circuit(
        self,
        data: Sequence[float],
    ) -> QuantumCircuit:
        """Build a standalone encoding circuit.

        Parameters
        ----------
        data : sequence of float

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        self.encode(qc, data)
        return qc

    def decode(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Decode quantum probabilities back to the amplitude magnitudes.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Measurement probabilities ``|α_i|²``.

        Returns
        -------
        numpy.ndarray
            Amplitude magnitudes ``|α_i|``.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        return np.sqrt(np.clip(probs, 0.0, 1.0))

    def __repr__(self) -> str:
        return (
            f"AmplitudeEncoder(n_qubits={self._n_qubits}, "
            f"input_dim={self.input_dim}, "
            f"normalize={self._normalize})"
        )


# ---------------------------------------------------------------------------
# HardwareEfficientAnsatz
# ---------------------------------------------------------------------------

class HardwareEfficientAnsatz:
    """Hardware-efficient ansatz circuit.

    A parameterized circuit designed to be efficient on near-term quantum
    hardware. Uses single-qubit rotations followed by entangling gates,
    repeated for a configurable number of layers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers (repetitions of the rotation-entangle
        block).
    rotations : str or tuple of str, optional
        Single-qubit rotation gates per qubit per layer. Default ``'rycz'``
        (RY then RZ).
    entanglement : str, optional
        Entanglement connectivity pattern. Default ``'linear'``.
    entangling_gate : str, optional
        Entangling gate: ``'cnot'``, ``'cz'``, ``'xx'``, ``'yy'``, ``'zz'``.
    reps : int, optional
        Number of repetitions of the full ansatz. Default ``1``.

    Examples
    --------
    >>> hea = HardwareEfficientAnsatz(4, 3, rotations='rycz')
    >>> params = hea.random_params()
    >>> circuit = hea.circuit(params)
    >>> hea.count_parameters()
    24
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        rotations: Union[str, Tuple[str, ...]] = "rycz",
        entanglement: str = "linear",
        entangling_gate: str = "cnot",
        reps: int = 1,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._rotation_gates = _parse_rotation_set(rotations)
        self._entanglement = entanglement
        self._entangling_gate = entangling_gate
        self._reps = reps

        # Validate entangling gate
        _valid_ent = frozenset({"cnot", "cz", "xx", "yy", "zz"})
        if entangling_gate not in _valid_ent:
            raise ValueError(
                f"Unknown entangling gate '{entangling_gate}'. "
                f"Choose from {sorted(_valid_ent)}"
            )

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of variational layers."""
        return self._n_layers

    @property
    def rotation_gates(self) -> Tuple[str, ...]:
        """tuple of str: Rotation gate names."""
        return self._rotation_gates

    @property
    def entanglement(self) -> str:
        """str: Entanglement pattern."""
        return self._entanglement

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
        """
        n_rot_per_layer = len(self._rotation_gates) * self._n_qubits
        n_ent_per_layer = len(
            _get_entanglement_edges(self._n_qubits, self._entanglement)
        )
        if self._entangling_gate not in ("cnot", "cz"):
            ent_params = n_ent_per_layer
        else:
            ent_params = 0

        return (n_rot_per_layer + ent_params) * self._n_layers * self._reps

    def random_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate random parameter values.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        numpy.ndarray
            Parameter vector of shape ``(count_parameters(),)``.
        """
        rng = np.random.default_rng(seed)
        n = self.count_parameters()
        # Initialise in [0, 2π) for rotation parameters
        return rng.uniform(0, 2 * _PI, size=n).astype(np.float64)

    def get_parameter_shapes(self) -> List[Tuple[int, ...]]:
        """Return shapes of all parameter tensors.

        Returns
        -------
        list of tuple of int
        """
        return [()] * self.count_parameters()

    def circuit(
        self,
        params: Union[Sequence[float], np.ndarray],
    ) -> QuantumCircuit:
        """Build the hardware-efficient ansatz circuit.

        Parameters
        ----------
        params : sequence of float or numpy.ndarray
            Trainable parameters. Length must equal ``count_parameters()``.

        Returns
        -------
        QuantumCircuit
            The parameterized quantum circuit.

        Raises
        ------
        ValueError
            If parameter count is incorrect.
        """
        params = np.asarray(params, dtype=np.float64)
        expected = self.count_parameters()
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(params)}"
            )

        qc = QuantumCircuit(self._n_qubits)
        self._apply_to_circuit(qc, params)
        return qc

    def _apply_to_circuit(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
    ) -> None:
        """Apply the ansatz to *circuit* (in-place).

        Parameters
        ----------
        circuit : QuantumCircuit
            Target quantum circuit.
        params : numpy.ndarray
            Parameter array.
        """
        n_rot_per_layer = len(self._rotation_gates) * self._n_qubits
        edges = _get_entanglement_edges(self._n_qubits, self._entanglement)
        has_ent_params = self._entangling_gate not in ("cnot", "cz")

        param_offset = 0

        for rep in range(self._reps):
            for layer in range(self._n_layers):
                # Rotation layer
                for q in range(self._n_qubits):
                    for gate_name in self._rotation_gates:
                        theta = float(params[param_offset])
                        param_offset += 1
                        _apply_rotation(circuit, gate_name, theta, q)

                # Entangling layer
                layer_edges = _get_entanglement_edges(
                    self._n_qubits, self._entanglement, layer
                )
                for ctrl, tgt in layer_edges:
                    if self._entangling_gate == "cnot":
                        circuit.cx(ctrl, tgt)
                    elif self._entangling_gate == "cz":
                        circuit.cz(ctrl, tgt)
                    elif has_ent_params:
                        theta = float(params[param_offset])
                        param_offset += 1
                        if self._entangling_gate == "xx":
                            circuit.rxx(theta, ctrl, tgt)
                        elif self._entangling_gate == "yy":
                            circuit.ryy(theta, ctrl, tgt)
                        elif self._entangling_gate == "zz":
                            circuit.rzz(theta, ctrl, tgt)

    def __repr__(self) -> str:
        return (
            f"HardwareEfficientAnsatz("
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"rotations={self._rotation_gates}, "
            f"entanglement={self._entanglement!r}, "
            f"entangling_gate={self._entangling_gate!r}, "
            f"reps={self._reps}, "
            f"n_params={self.count_parameters()})"
        )


# ---------------------------------------------------------------------------
# StronglyEntanglingAnsatz
# ---------------------------------------------------------------------------

class StronglyEntanglingAnsatz:
    """Strongly entangling variational ansatz.

    Uses 3-parameter rotation blocks on each qubit followed by CNOT
    entanglement, repeated for multiple layers. This ansatz is known
    for its strong expressive power.

    Each layer consists of:
        1. For each qubit: ``RZ(φ_i) → RY(θ_i) → RZ(ω_i)``
        2. CNOT entanglement layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of entangling layers.
    entanglement : str, optional
        Entanglement pattern. Default ``'circular'``.

    Examples
    --------
    >>> sea = StronglyEntanglingAnsatz(3, 2)
    >>> params = sea.random_params()
    >>> circuit = sea.circuit(params)
    >>> sea.count_parameters()
    18
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entanglement: str = "circular",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._entanglement = entanglement

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of layers."""
        return self._n_layers

    @property
    def entanglement(self) -> str:
        """str: Entanglement pattern."""
        return self._entanglement

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Each qubit per layer gets 3 rotation parameters (φ, θ, ω).

        Returns
        -------
        int
        """
        return 3 * self._n_qubits * self._n_layers

    def random_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate random parameter values.

        Parameters
        ----------
        seed : int, optional

        Returns
        -------
        numpy.ndarray
        """
        rng = np.random.default_rng(seed)
        n = self.count_parameters()
        return rng.uniform(0, 2 * _PI, size=n).astype(np.float64)

    def get_parameter_shapes(self) -> List[Tuple[int, ...]]:
        """Return shapes of parameter tensors.

        Returns
        -------
        list of tuple of int
        """
        # 3 params per qubit per layer: shape (3, n_qubits) per layer
        shapes = []
        for _ in range(self._n_layers):
            shapes.append((3, self._n_qubits))
        return shapes

    def circuit(
        self,
        params: Union[Sequence[float], np.ndarray],
    ) -> QuantumCircuit:
        """Build the strongly entangling ansatz circuit.

        Parameters
        ----------
        params : sequence of float or numpy.ndarray
            Parameters. Must have length ``3 * n_qubits * n_layers``.
            Can be flat or reshaped to ``(n_layers, 3, n_qubits)``.

        Returns
        -------
        QuantumCircuit
        """
        params = np.asarray(params, dtype=np.float64)
        expected = self.count_parameters()

        # Allow (n_layers, 3, n_qubits) shaped input
        if params.ndim == 3:
            if params.shape != (self._n_layers, 3, self._n_qubits):
                raise ValueError(
                    f"Expected shape ({self._n_layers}, 3, {self._n_qubits}), "
                    f"got {params.shape}"
                )
            params = params.reshape(-1)

        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(params)}"
            )

        qc = QuantumCircuit(self._n_qubits)
        self._apply_to_circuit(qc, params)
        return qc

    def _apply_to_circuit(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
    ) -> None:
        """Apply the strongly entangling ansatz.

        Parameters
        ----------
        circuit : QuantumCircuit
            Target circuit.
        params : numpy.ndarray
            Flat parameter array.
        """
        offset = 0
        for layer in range(self._n_layers):
            # 3-rotation block on each qubit
            for q in range(self._n_qubits):
                phi = float(params[offset])
                theta = float(params[offset + 1])
                omega = float(params[offset + 2])
                offset += 3
                # Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ)
                circuit.rz(phi, q)
                circuit.ry(theta, q)
                circuit.rz(omega, q)

            # Entangling layer
            edges = _get_entanglement_edges(
                self._n_qubits, self._entanglement, layer
            )
            for ctrl, tgt in edges:
                circuit.cx(ctrl, tgt)

    def get_layer_params(
        self,
        params: Union[Sequence[float], np.ndarray],
        layer_index: int,
    ) -> np.ndarray:
        """Extract parameters for a specific layer.

        Parameters
        ----------
        params : sequence of float or numpy.ndarray
            Full parameter array.
        layer_index : int
            Layer index (0-indexed).

        Returns
        -------
        numpy.ndarray
            Parameters for the specified layer, shape ``(3, n_qubits)``.
        """
        params = np.asarray(params, dtype=np.float64)
        if layer_index < 0 or layer_index >= self._n_layers:
            raise ValueError(
                f"Layer index {layer_index} out of range "
                f"[0, {self._n_layers})"
            )
        start = layer_index * 3 * self._n_qubits
        end = start + 3 * self._n_qubits
        layer_params = params[start:end]
        return layer_params.reshape(3, self._n_qubits)

    def set_layer_params(
        self,
        params: np.ndarray,
        layer_index: int,
        layer_params: Union[Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Set parameters for a specific layer.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter array (will be copied).
        layer_index : int
            Layer index.
        layer_params : sequence of float or numpy.ndarray
            New parameters for the layer, shape ``(3, n_qubits)`` or flat
            of length ``3 * n_qubits``.

        Returns
        -------
        numpy.ndarray
            Updated full parameter array.
        """
        params = params.copy()
        layer_params = np.asarray(layer_params, dtype=np.float64).reshape(-1)
        expected = 3 * self._n_qubits
        if len(layer_params) != expected:
            raise ValueError(
                f"Expected {expected} layer params, got {len(layer_params)}"
            )
        start = layer_index * expected
        params[start:start + expected] = layer_params
        return params

    def __repr__(self) -> str:
        return (
            f"StronglyEntanglingAnsatz("
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"entanglement={self._entanglement!r}, "
            f"n_params={self.count_parameters()})"
        )


# ---------------------------------------------------------------------------
# VariationalCircuit
# ---------------------------------------------------------------------------

class VariationalCircuit:
    """Full parameterized quantum circuit for Variational Quantum Algorithms.

    Combines an encoder (classical-to-quantum data mapping) with a
    variational ansatz (trainable quantum circuit). Provides forward
    pass execution, gradient computation, and circuit inspection.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    entanglement : str, optional
        Entanglement connectivity. Default ``'full'``.
    rotations : str or tuple of str, optional
        Rotation gates. Default ``'rycz'``.
    reps : int, optional
        Number of repetitions. Default ``1``.
    encoder : str or None, optional
        Encoder type (``'angle'``, ``'amplitude'``, or ``None``).
        If ``None``, no encoding is applied (raw parameters only).

    Examples
    --------
    >>> vc = VariationalCircuit(4, 3)
    >>> vc.count_parameters()
    36
    >>> params = vc.random_params()
    >>> output = vc.forward([0.1, 0.2, 0.3, 0.4])
    >>> circuit = vc.circuit()
    >>> print(circuit.depth())
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entanglement: str = "full",
        rotations: Union[str, Tuple[str, ...]] = "rycz",
        reps: int = 1,
        encoder: Optional[str] = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._entanglement = entanglement
        self._rotations = _parse_rotation_set(rotations)
        self._reps = reps
        self._encoder_type = encoder

        # Build encoder
        self._angle_encoder: Optional[AngleEncoder] = None
        self._amplitude_encoder: Optional[AmplitudeEncoder] = None
        if encoder == "angle":
            self._angle_encoder = AngleEncoder(n_qubits, feature_map="zx")
        elif encoder == "amplitude":
            self._amplitude_encoder = AmplitudeEncoder(n_qubits)

        # Initialise trainable parameters
        self._parameters = self.random_params()

    @property
    def parameters(self) -> np.ndarray:
        """numpy.ndarray: Trainable parameters (writable).

        Setting this property updates all parameters at once.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Union[Sequence[float], np.ndarray]) -> None:
        self.assign_parameters(value)

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of layers."""
        return self._n_layers

    @property
    def entanglement(self) -> str:
        """str: Entanglement pattern."""
        return self._entanglement

    @property
    def rotation_gates(self) -> Tuple[str, ...]:
        """tuple of str: Rotation gate names."""
        return self._rotations

    @property
    def input_dim(self) -> int:
        """int: Expected input dimensionality."""
        if self._encoder_type == "amplitude":
            return 1 << self._n_qubits
        return self._n_qubits

    # -- Parameter management -------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
        """
        n_rot_per_layer = len(self._rotations) * self._n_qubits
        edges = _get_entanglement_edges(self._n_qubits, self._entanglement)
        return n_rot_per_layer * self._n_layers * self._reps

    def random_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate random parameters.

        Parameters
        ----------
        seed : int, optional

        Returns
        -------
        numpy.ndarray
        """
        rng = np.random.default_rng(seed)
        n = self.count_parameters()
        return rng.uniform(0, 2 * _PI, size=n).astype(np.float64)

    def assign_parameters(
        self,
        values: Union[Sequence[float], np.ndarray],
    ) -> None:
        """Set the trainable parameters.

        Parameters
        ----------
        values : sequence of float or numpy.ndarray
            New parameter values. Length must equal ``count_parameters()``.

        Raises
        ------
        ValueError
            If the parameter count is incorrect.
        """
        values = np.asarray(values, dtype=np.float64)
        expected = self.count_parameters()
        if len(values) != expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(values)}"
            )
        self._parameters = values.copy()

    def get_parameter_shapes(self) -> List[Tuple[int, ...]]:
        """Return shapes of parameter tensors.

        Returns
        -------
        list of tuple of int
        """
        n_rot = len(self._rotations) * self._n_qubits
        shapes: List[Tuple[int, ...]] = []
        for _ in range(self._n_layers * self._reps):
            shapes.append((n_rot,))
        return shapes

    # -- Circuit construction -------------------------------------------------

    def forward(
        self,
        x: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Encode input data and execute the variational circuit.

        If an encoder is configured, the input data is first encoded.
        Then the variational ansatz is applied with the current parameters.

        Parameters
        ----------
        x : sequence of float, optional
            Classical input data. Required if an encoder is set.

        Returns
        -------
        numpy.ndarray
            Measurement probabilities, shape ``(2^n_qubits,)``.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        qc = self.circuit(x)
        simulator = StatevectorSimulator()
        return simulator.probabilities(qc)

    def circuit(
        self,
        x: Optional[Sequence[float]] = None,
        params: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> QuantumCircuit:
        """Build the full parameterized quantum circuit.

        Parameters
        ----------
        x : sequence of float, optional
            Input data (required if encoder is set).
        params : sequence of float or numpy.ndarray, optional
            Trainable parameters. If ``None``, uses current parameters.

        Returns
        -------
        QuantumCircuit
            The full circuit (encoding + variational).

        Raises
        ------
        ValueError
            If encoder requires input but none provided.
        """
        if params is None:
            params = self._parameters
        else:
            params = np.asarray(params, dtype=np.float64)

        expected = self.count_parameters()
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(params)}"
            )

        qc = QuantumCircuit(self._n_qubits)

        # Encode input data
        if self._angle_encoder is not None and x is not None:
            self._angle_encoder.encode(qc, x)
        elif self._amplitude_encoder is not None and x is not None:
            self._amplitude_encoder.encode(qc, x)
        elif x is not None and self._encoder_type is not None:
            raise ValueError(
                f"Encoder type '{self._encoder_type}' not recognized"
            )

        # Apply variational layers
        self._apply_variational(qc, params)

        return qc

    def _apply_variational(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
    ) -> None:
        """Apply the variational ansatz to the circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
            Target circuit.
        params : numpy.ndarray
            Flat parameter array.
        """
        n_rot_per_layer = len(self._rotations) * self._n_qubits
        offset = 0

        for rep in range(self._reps):
            for layer in range(self._n_layers):
                # Rotation layer
                for q in range(self._n_qubits):
                    for gate_name in self._rotations:
                        theta = float(params[offset])
                        offset += 1
                        _apply_rotation(circuit, gate_name, theta, q)

                # Entangling layer
                edges = _get_entanglement_edges(
                    self._n_qubits, self._entanglement, layer
                )
                for ctrl, tgt in edges:
                    circuit.cx(ctrl, tgt)

    # -- Gradient computation -------------------------------------------------

    def gradients(
        self,
        param_indices: Optional[Sequence[int]] = None,
        x: Optional[Sequence[float]] = None,
        observable: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute parameter gradients using finite differences.

        Parameters
        ----------
        param_indices : sequence of int, optional
            Indices of parameters to differentiate. If ``None``,
            computes gradients for all parameters.
        x : sequence of float, optional
            Input data.
        observable : numpy.ndarray, optional
            Observable matrix. If ``None``, uses Pauli-Z on qubit 0.
        eps : float
            Finite difference step size.

        Returns
        -------
        numpy.ndarray
            Gradient values. Length equals ``len(param_indices)`` or
            ``count_parameters()``.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        simulator = StatevectorSimulator()
        n_params = self.count_parameters()

        if param_indices is None:
            param_indices = list(range(n_params))

        # Default observable: Z on qubit 0
        if observable is None:
            observable = np.kron(
                np.array([[1, 0], [0, -1]], dtype=np.complex128),
                np.eye(1 << (self._n_qubits - 1), dtype=np.complex128),
            )

        grads = np.zeros(len(param_indices), dtype=np.float64)
        base_circuit = self.circuit(x)
        base_val = simulator.expectation(base_circuit, observable)

        for out_idx, pidx in enumerate(param_indices):
            # Forward shift
            params_plus = self._parameters.copy()
            params_plus[pidx] += eps
            circ_plus = self.circuit(x, params_plus)
            val_plus = simulator.expectation(circ_plus, observable)

            # Backward shift
            params_minus = self._parameters.copy()
            params_minus[pidx] -= eps
            circ_minus = self.circuit(x, params_minus)
            val_minus = simulator.expectation(circ_minus, observable)

            grads[out_idx] = (val_plus - val_minus) / (2.0 * eps)

        return grads

    # -- Inspection helpers ---------------------------------------------------

    def circuit_depth(self) -> int:
        """Return the depth of the variational circuit (no encoding).

        Returns
        -------
        int
        """
        qc = QuantumCircuit(self._n_qubits)
        self._apply_variational(qc, self._parameters)
        return qc.depth()

    def circuit_size(self) -> int:
        """Return the total number of gates.

        Returns
        -------
        int
        """
        qc = QuantumCircuit(self._n_qubits)
        self._apply_variational(qc, self._parameters)
        return qc.size()

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary.

        Returns
        -------
        dict
        """
        return {
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "entanglement": self._entanglement,
            "rotations": list(self._rotations),
            "reps": self._reps,
            "encoder": self._encoder_type,
        }

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VariationalCircuit("
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"entanglement={self._entanglement!r}, "
            f"rotations={self._rotations}, "
            f"reps={self._reps}, "
            f"encoder={self._encoder_type!r}, "
            f"n_params={self.count_parameters()})"
        )

    def __call__(
        self,
        x: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Shortcut for ``self.forward(x)``."""
        return self.forward(x)
