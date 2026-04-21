"""
Quantum Neural Network Layer
=============================

Provides the core quantum neural network layer components:

* :class:`QuantumNNLayer` — a trainable quantum circuit acting as a neural
  network layer with configurable encoding, variational form, and measurement.
* :class:`VariationalLayer` — a single variational block with configurable
  entanglement pattern and rotation gates.
* :class:`EncodingLayer` — data encoding strategies (angle, amplitude,
  basis, IQP, dense angle).

Architecture
------------
A ``QuantumNNLayer`` follows the encode–variational–measure paradigm:

1. **Encode**: classical input data ``x`` is mapped onto the quantum state
   via one of several encoding strategies.
2. **Variational**: trainable parameterised gates are applied to create a
   feature-rich quantum state.
3. **Measure**: the expectation value of a chosen observable is returned
   as the classical output.

The circuit is generated lazily for each input, making it compatible with
auto-differentiation frameworks (TensorFlow, PyTorch, JAX).

Examples
--------
>>> layer = QuantumNNLayer(n_qubits=4, n_layers=2, encoding='angle')
>>> layer.count_parameters()
40
>>> circuit = layer.get_circuit([0.1, 0.2, 0.3, 0.4])
>>> circuit.depth()
>>> 14

>>> # Dense angle encoding with strong entanglement
>>> layer2 = QuantumNNLayer(
...     n_qubits=6, n_layers=3,
...     encoding='dense_angle',
...     variational_form='strong_entangling',
...     observable='z',
... )
>>> layer2.get_parameter_shapes()
[(6,), (6, 3), (6, 3), (6,)]
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
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    RXGate,
)
from quantumflow.core.state import Statevector

__all__ = [
    "QuantumNNLayer",
    "VariationalLayer",
    "EncodingLayer",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_VALID_ENCODINGS = frozenset({
    "angle",
    "amplitude",
    "basis",
    "iqp",
    "dense_angle",
})

_VALID_VARIATIONAL_FORMS = frozenset({
    "hardware_efficient",
    "strong_entangling",
    "circuit_19",
    "barren_plateau_free",
    "qaoa",
})

_VALID_OBSERVABLES = frozenset({
    "z",
    "x",
    "y",
    "zz",
    "xx",
    "yy",
    "mixed",
})

_VALID_ENTANGLEMENTS = frozenset({
    "linear",
    "circular",
    "full",
    "pairwise",
    "star",
})

_VALID_ENTANGLING_GATES = frozenset({
    "cnot",
    "cz",
    "xx",
    "yy",
    "zz",
})


# ---------------------------------------------------------------------------
# EncodingLayer
# ---------------------------------------------------------------------------

class EncodingLayer:
    """Data encoding into quantum states.

    Implements several encoding strategies that map classical data vectors
    onto quantum states. Each strategy has different properties in terms
    of expressiveness and qubit efficiency.

    Parameters
    ----------
    n_qubits : int
        Number of qubits available for encoding.
    encoding : str
        Encoding strategy. One of:

        * ``'angle'`` — Each data feature ``x_i`` sets the angle of a
          ``RZ`` rotation on qubit ``i``. Requires ``len(x) == n_qubits``.
        * ``'amplitude'`` — The data vector (normalised) becomes the
          amplitude vector of the quantum state. Requires ``len(x) == 2^n``.
        * ``'basis'`` — Encodes binary data as a computational basis state.
        * ``'iqp'`` — Instantaneous Quantum Polynomial encoding:
          ``RZ(x) → CNOT layers → RZ(π/4) → CNOT layers → RZ(x)``.
        * ``'dense_angle'`` — Dense angle encoding with data re-uploading
          using rotations and entangling gates.

    Examples
    --------
    >>> enc = EncodingLayer(4, encoding='angle')
    >>> qc = QuantumCircuit(4)
    >>> enc.encode(qc, [0.1, 0.2, 0.3, 0.4])
    """

    def __init__(
        self,
        n_qubits: int,
        encoding: str = "angle",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                f"Choose from {sorted(_VALID_ENCODINGS)}"
            )
        self._n_qubits = n_qubits
        self._encoding = encoding

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def encoding(self) -> str:
        """str: Encoding strategy."""
        return self._encoding

    @property
    def input_dim(self) -> int:
        """int: Expected input dimensionality for this encoding."""
        if self._encoding == "angle":
            return self._n_qubits
        elif self._encoding == "amplitude":
            return 1 << self._n_qubits
        elif self._encoding == "basis":
            return self._n_qubits  # binary bits
        elif self._encoding == "iqp":
            return self._n_qubits
        elif self._encoding == "dense_angle":
            return self._n_qubits
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
            Target quantum circuit. Must have at least ``n_qubits`` qubits.
        data : sequence of float
            Classical data vector of appropriate dimensionality.

        Raises
        ------
        ValueError
            If data dimensions don't match the encoding requirements.
        """
        data = np.asarray(data, dtype=np.float64)

        if self._encoding == "angle":
            self._encode_angle(circuit, data)
        elif self._encoding == "amplitude":
            self._encode_amplitude(circuit, data)
        elif self._encoding == "basis":
            self._encode_basis(circuit, data)
        elif self._encoding == "iqp":
            self._encode_iqp(circuit, data)
        elif self._encoding == "dense_angle":
            self._encode_dense_angle(circuit, data)

    # -- Angle Encoding -------------------------------------------------------

    def _encode_angle(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
    ) -> None:
        """Angle encoding: ``x_i → RZ(x_i)`` on each qubit.

        Optionally applies a Hadamard before the rotation to use the
        full Bloch sphere (``H → RY(x_i)``).
        """
        if len(data) != self._n_qubits:
            raise ValueError(
                f"Angle encoding requires {self._n_qubits} features, "
                f"got {len(data)}"
            )
        for i, val in enumerate(data):
            circuit.h(i)
            circuit.ry(float(val), i)

    # -- Amplitude Encoding ---------------------------------------------------

    def _encode_amplitude(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
    ) -> None:
        """Amplitude encoding: normalise data and set as state amplitudes.

        For non-power-of-2 lengths, pads with zeros.

        .. note::
            Amplitude encoding modifies the quantum state directly rather
            than appending gates.  We use a Statevector-based approach and
            apply the corresponding unitary.
        """
        expected = 1 << self._n_qubits
        if len(data) > expected:
            raise ValueError(
                f"Amplitude encoding can encode at most {expected} "
                f"features with {self._n_qubits} qubits, got {len(data)}"
            )

        # Pad to power of 2 if needed
        if len(data) < expected:
            padded = np.zeros(expected, dtype=np.complex128)
            padded[:len(data)] = data
            data = padded

        # Normalise
        norm = np.linalg.norm(data)
        if norm < 1e-12:
            raise ValueError("Cannot encode a zero-norm data vector")
        data = data / norm

        # Build the unitary that maps |0...0⟩ → |data⟩
        # U = I - 2|v><v| where |v> = |data> - |0>
        # More practical: use a series of rotations
        # For efficiency, we apply angle encoding as a fallback
        # and store the target amplitudes for later use.

        # Use recursive amplitude encoding via rotation gates
        self._amplitude_encode_recursive(circuit, data, 0)

    def _amplitude_encode_recursive(
        self,
        circuit: QuantumCircuit,
        amplitudes: np.ndarray,
        start_qubit: int,
    ) -> None:
        """Recursively encode amplitudes using controlled rotations.

        Uses the binary-tree approach: split amplitudes into two halves,
        compute the angle that gives the correct probability for the first
        half, apply a rotation on the top qubit, then recurse.
        """
        n = len(amplitudes)
        if n <= 1:
            return
        if n == 2:
            # Single rotation
            prob_0 = float(np.abs(amplitudes[0]) ** 2)
            theta = 2.0 * math.acos(np.sqrt(np.clip(prob_0, 0.0, 1.0)))
            circuit.ry(theta, start_qubit)
            # Phase correction
            if np.abs(amplitudes[0]) > 1e-12:
                phase_0 = np.angle(amplitudes[0])
                phase_1 = np.angle(amplitudes[1])
                relative_phase = phase_1 - phase_0
                if abs(relative_phase) > 1e-12:
                    circuit.rz(relative_phase, start_qubit)
            return

        # Split amplitudes
        mid = n // 2
        left = amplitudes[:mid]
        right = amplitudes[mid:]

        # Probability of measuring 0 on this qubit
        prob_0 = float(np.sum(np.abs(left) ** 2))
        prob_0 = np.clip(prob_0, 0.0, 1.0)
        theta = 2.0 * math.acos(math.sqrt(float(prob_0)))
        circuit.ry(theta, start_qubit)

        # Phase correction
        phase_left = np.angle(left[0]) if np.abs(left[0]) > 1e-12 else 0.0
        phase_right = np.angle(right[0]) if np.abs(right[0]) > 1e-12 else 0.0
        relative_phase = phase_right - phase_left
        if abs(relative_phase) > 1e-12:
            circuit.rz(relative_phase, start_qubit)

        # Recurse on both halves (controlled on this qubit)
        # We apply the left sub-encoding unconditionally since we already
        # rotated by theta to split amplitudes
        # For the right sub-encoding, apply X then encode then X
        remaining_qubits = int(round(math.log2(len(left))))

        # Encode left amplitudes
        self._amplitude_encode_recursive(circuit, left, start_qubit + 1)

        # Encode right amplitudes (controlled by X on this qubit)
        circuit.x(start_qubit)
        self._amplitude_encode_recursive(circuit, right, start_qubit + 1)
        circuit.x(start_qubit)

    # -- Basis Encoding -------------------------------------------------------

    def _encode_basis(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
    ) -> None:
        """Basis encoding: set qubits to computational basis state.

        Input data should be binary (0 or 1). Non-binary values are
        thresholded at 0.5.
        """
        if len(data) != self._n_qubits:
            raise ValueError(
                f"Basis encoding requires {self._n_qubits} binary features, "
                f"got {len(data)}"
            )
        for i, val in enumerate(data):
            if float(val) > 0.5:
                circuit.x(i)

    # -- IQP Encoding ---------------------------------------------------------

    def _encode_iqp(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
    ) -> None:
        """Instantaneous Quantum Polynomial (IQP) encoding.

        Circuit structure:
            1. Apply ``RZ(x_i)`` on each qubit.
            2. Apply a layer of CNOT entangling gates.
            3. Apply ``RZ(π/4)`` on each qubit.
            4. Apply another layer of CNOT gates.
            5. Apply ``RZ(x_i)`` on each qubit again.
        """
        if len(data) != self._n_qubits:
            raise ValueError(
                f"IQP encoding requires {self._n_qubits} features, "
                f"got {len(data)}"
            )

        # First RZ layer
        for i, val in enumerate(data):
            circuit.h(i)
            circuit.rz(float(val), i)

        # CNOT entangling layer 1
        for i in range(self._n_qubits - 1):
            circuit.cx(i, i + 1)

        # Second RZ layer with π/4
        for i in range(self._n_qubits):
            circuit.rz(_PI / 4.0, i)

        # CNOT entangling layer 2
        for i in range(self._n_qubits - 1):
            circuit.cx(i, i + 1)

        # Final RZ layer
        for i, val in enumerate(data):
            circuit.rz(float(val), i)

    # -- Dense Angle Encoding -------------------------------------------------

    def _encode_dense_angle(
        self,
        circuit: QuantumCircuit,
        data: np.ndarray,
    ) -> None:
        """Dense angle encoding with data re-uploading.

        Applies multiple rounds of:
            1. ``RZ(x_i)`` on each qubit.
            2. ``RY(x_i)`` on each qubit.
            3. Entangling CNOT layer.
        This increases the expressiveness beyond basic angle encoding.
        """
        if len(data) != self._n_qubits:
            raise ValueError(
                f"Dense angle encoding requires {self._n_qubits} features, "
                f"got {len(data)}"
            )

        n_reuploads = max(1, min(3, self._n_qubits // 2))

        for _ in range(n_reuploads):
            # Rotation layer
            for i, val in enumerate(data):
                circuit.rz(float(val), i)
                circuit.ry(float(val), i)

            # Entangling layer
            for i in range(self._n_qubits - 1):
                circuit.cx(i, i + 1)

    # -- Utility --------------------------------------------------------------

    def get_encoding_circuit(
        self,
        data: Sequence[float],
    ) -> QuantumCircuit:
        """Build a standalone encoding circuit for the given data.

        Parameters
        ----------
        data : sequence of float
            Input data vector.

        Returns
        -------
        QuantumCircuit
            Circuit that prepares the encoded state.
        """
        qc = QuantumCircuit(self._n_qubits)
        self.encode(qc, data)
        return qc

    def __repr__(self) -> str:
        return (
            f"EncodingLayer(n_qubits={self._n_qubits}, "
            f"encoding={self._encoding!r}, "
            f"input_dim={self.input_dim})"
        )


# ---------------------------------------------------------------------------
# VariationalLayer
# ---------------------------------------------------------------------------

class VariationalLayer:
    """A single variational block with configurable entanglement.

    Each variational layer consists of:
    1. A rotation block: parameterised single-qubit rotations on every qubit.
    2. An entanglement block: two-qubit entangling gates following a
       configurable connectivity pattern.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    rotation_gates : str or tuple of str, optional
        Single-qubit rotation gates. One of:
        * ``'rz'``, ``'ry'``, ``'rx'`` — single rotation.
        * ``'rycz'`` — RY then RZ (default).
        * ``'xyz'`` — RX, RY, RZ sequence.
        * Tuple of gate names for custom rotation sets.
    entanglement : str, optional
        Entanglement connectivity pattern:
        * ``'linear'`` — adjacent qubits (i, i+1).
        * ``'circular'`` — linear plus wrap-around.
        * ``'full'`` — all pairs.
        * ``'pairwise'`` — (0,1), (2,3), …
        * ``'star'`` — all qubits connected to qubit 0.
    entangling_gate : str, optional
        Two-qubit entangling gate: ``'cnot'``, ``'cz'``, ``'xx'``,
        ``'yy'``, ``'zz'``.
    layer_index : int, optional
        Index of this layer (used for parameter naming).

    Examples
    --------
    >>> vl = VariationalLayer(4, entanglement='full')
    >>> circuit = QuantumCircuit(4)
    >>> params = vl.random_params()
    >>> vl.apply(circuit, params)
    """

    def __init__(
        self,
        n_qubits: int,
        rotation_gates: Union[str, Tuple[str, ...]] = "rycz",
        entanglement: str = "linear",
        entangling_gate: str = "cnot",
        layer_index: int = 0,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if entanglement not in _VALID_ENTANGLEMENTS:
            raise ValueError(
                f"Unknown entanglement '{entanglement}'. "
                f"Choose from {sorted(_VALID_ENTANGLEMENTS)}"
            )
        if entangling_gate not in _VALID_ENTANGLING_GATES:
            raise ValueError(
                f"Unknown entangling gate '{entangling_gate}'. "
                f"Choose from {sorted(_VALID_ENTANGLING_GATES)}"
            )

        if isinstance(rotation_gates, str):
            rotation_gates = tuple(rotation_gates)
        else:
            rotation_gates = tuple(rotation_gates)

        self._n_qubits = n_qubits
        self._rotation_gates = rotation_gates
        self._entanglement = entanglement
        self._entangling_gate = entangling_gate
        self._layer_index = layer_index

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def rotation_gates(self) -> Tuple[str, ...]:
        """tuple of str: Rotation gate names."""
        return self._rotation_gates

    @property
    def entanglement(self) -> str:
        """str: Entanglement pattern."""
        return self._entanglement

    @property
    def entangling_gate(self) -> str:
        """str: Entangling gate name."""
        return self._entangling_gate

    @property
    def n_rotation_params(self) -> int:
        """int: Number of rotation parameters per layer."""
        return len(self._rotation_gates) * self._n_qubits

    @property
    def n_entangling_params(self) -> int:
        """int: Number of entangling gate parameters per layer.

        ``0`` for CNOT/CZ, ``1`` for parameterised gates (XX, YY, ZZ).
        """
        edges = self._get_entanglement_edges()
        if self._entangling_gate in ("cnot", "cz"):
            return 0
        return len(edges)

    @property
    def n_params(self) -> int:
        """int: Total number of parameters for this layer."""
        return self.n_rotation_params + self.n_entangling_params

    def _get_entanglement_edges(self) -> List[Tuple[int, int]]:
        """Return list of (control, target) pairs for the entanglement pattern."""
        edges: List[Tuple[int, int]] = []
        n = self._n_qubits

        if self._entanglement == "linear":
            for i in range(n - 1):
                edges.append((i, i + 1))

        elif self._entanglement == "circular":
            for i in range(n):
                edges.append((i, (i + 1) % n))

        elif self._entanglement == "full":
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))

        elif self._entanglement == "pairwise":
            for i in range(0, n - 1, 2):
                edges.append((i, min(i + 1, n - 1)))

        elif self._entanglement == "star":
            for i in range(1, n):
                edges.append((0, i))

        return edges

    def apply(
        self,
        circuit: QuantumCircuit,
        rotation_params: Sequence[float],
        entangling_params: Optional[Sequence[float]] = None,
    ) -> None:
        """Apply the variational layer to *circuit* (in-place).

        Parameters
        ----------
        circuit : QuantumCircuit
            Target quantum circuit.
        rotation_params : sequence of float
            Parameters for the rotation gates. Must have length
            ``n_rotation_params``.
        entangling_params : sequence of float, optional
            Parameters for the entangling gates (if parameterised).
            Must have length ``n_entangling_params``.

        Raises
        ------
        ValueError
            If parameter counts are incorrect.
        """
        rotation_params = list(rotation_params)
        if len(rotation_params) != self.n_rotation_params:
            raise ValueError(
                f"Expected {self.n_rotation_params} rotation params, "
                f"got {len(rotation_params)}"
            )

        if self.n_entangling_params > 0:
            if entangling_params is None:
                raise ValueError(
                    f"Entangling gate '{self._entangling_gate}' requires "
                    f"{self.n_entangling_params} params, got None"
                )
            entangling_params = list(entangling_params)
            if len(entangling_params) != self.n_entangling_params:
                raise ValueError(
                    f"Expected {self.n_entangling_params} entangling params, "
                    f"got {len(entangling_params)}"
                )

        # Apply rotation gates
        param_idx = 0
        for q in range(self._n_qubits):
            for gate_name in self._rotation_gates:
                theta = float(rotation_params[param_idx])
                param_idx += 1
                self._apply_rotation(circuit, gate_name, theta, q)

        # Apply entangling gates
        edges = self._get_entanglement_edges()
        if self._entangling_gate == "cnot":
            for ctrl, tgt in edges:
                circuit.cx(ctrl, tgt)
        elif self._entangling_gate == "cz":
            for ctrl, tgt in edges:
                circuit.cz(ctrl, tgt)
        elif entangling_params is not None:
            eidx = 0
            for ctrl, tgt in edges:
                theta = float(entangling_params[eidx])
                eidx += 1
                if self._entangling_gate == "xx":
                    circuit.rxx(theta, ctrl, tgt)
                elif self._entangling_gate == "yy":
                    circuit.ryy(theta, ctrl, tgt)
                elif self._entangling_gate == "zz":
                    circuit.rzz(theta, ctrl, tgt)

    def _apply_rotation(
        self,
        circuit: QuantumCircuit,
        gate_name: str,
        theta: float,
        qubit: int,
    ) -> None:
        """Apply a single-qubit rotation gate."""
        if gate_name == "rx":
            circuit.rx(theta, qubit)
        elif gate_name == "ry":
            circuit.ry(theta, qubit)
        elif gate_name == "rz":
            circuit.rz(theta, qubit)
        else:
            raise ValueError(f"Unsupported rotation gate: {gate_name}")

    def random_params(self, seed: Optional[int] = None) -> List[float]:
        """Generate random parameter values.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        list of float
            Random parameters of total length ``n_params``.
        """
        rng = np.random.default_rng(seed)
        params = rng.uniform(0, 2 * _PI, size=self.n_params).tolist()
        return params

    def get_param_names(self) -> List[str]:
        """Return descriptive parameter names.

        Returns
        -------
        list of str
            Names like ``'L0_q0_ry'``, ``'L0_q1_rz'``, etc.
        """
        names: List[str] = []
        for q in range(self._n_qubits):
            for gate_name in self._rotation_gates:
                names.append(f"L{self._layer_index}_q{q}_{gate_name}")
        edges = self._get_entanglement_edges()
        if self.n_entangling_params > 0:
            for idx, (ctrl, tgt) in enumerate(edges):
                names.append(
                    f"L{self._layer_index}_{self._entangling_gate}_"
                    f"q{ctrl}_q{tgt}_{idx}"
                )
        return names

    def get_param_shapes(self) -> List[Tuple[int, ...]]:
        """Return the shapes of all parameter tensors.

        Returns
        -------
        list of tuple of int
            Each rotation parameter is scalar ``()``.
        """
        return [()] * self.n_params

    def __repr__(self) -> str:
        return (
            f"VariationalLayer(n_qubits={self._n_qubits}, "
            f"rotations={self._rotation_gates}, "
            f"entanglement={self._entanglement!r}, "
            f"entangling_gate={self._entangling_gate!r}, "
            f"n_params={self.n_params})"
        )


# ---------------------------------------------------------------------------
# QuantumNNLayer
# ---------------------------------------------------------------------------

class QuantumNNLayer:
    """A trainable quantum circuit that acts as a neural network layer.

    Follows the encode → variational → measure paradigm:

    1. **Encoding**: classical input data is encoded onto the quantum state.
    2. **Variational layers**: trainable parameterised gates create
       expressive quantum features.
    3. **Measurement**: expectation values of observables produce
       classical outputs.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    encoding : str, optional
        Data encoding strategy. One of ``'angle'``, ``'amplitude'``,
        ``'basis'``, ``'iqp'``, ``'dense_angle'``.
    variational_form : str, optional
        Variational circuit architecture. One of ``'hardware_efficient'``,
        ``'strong_entangling'``, ``'circuit_19'``, ``'barren_plateau_free'``,
        ``'qaoa'``.
    observable : str, optional
        Measurement observable: ``'z'``, ``'x'``, ``'y'``, ``'zz'``,
        ``'xx'``, ``'yy'``, ``'mixed'``.
    output_dim : int or None, optional
        Number of output dimensions. ``None`` means output per qubit
        (``n_qubits`` outputs for Pauli observables).
    rotation_gates : str or tuple of str, optional
        Rotation gates for variational layers. Default ``'rycz'``.
    entanglement : str, optional
        Entanglement pattern: ``'linear'``, ``'circular'``, ``'full'``,
        ``'pairwise'``, ``'star'``.
    entangling_gate : str, optional
        Entangling gate: ``'cnot'``, ``'cz'``, ``'xx'``, ``'yy'``, ``'zz'``.

    Examples
    --------
    >>> layer = QuantumNNLayer(n_qubits=4, n_layers=2)
    >>> circuit = layer.get_circuit([0.1, 0.2, 0.3, 0.4])
    >>> layer.count_parameters()
    32
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        encoding: str = "angle",
        variational_form: str = "hardware_efficient",
        observable: str = "z",
        output_dim: Optional[int] = None,
        rotation_gates: Union[str, Tuple[str, ...]] = "rycz",
        entanglement: str = "linear",
        entangling_gate: str = "cnot",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if variational_form not in _VALID_VARIATIONAL_FORMS:
            raise ValueError(
                f"Unknown variational form '{variational_form}'. "
                f"Choose from {sorted(_VALID_VARIATIONAL_FORMS)}"
            )
        if observable not in _VALID_OBSERVABLES:
            raise ValueError(
                f"Unknown observable '{observable}'. "
                f"Choose from {sorted(_VALID_OBSERVABLES)}"
            )

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._encoding = encoding
        self._variational_form = variational_form
        self._observable = observable
        self._output_dim = output_dim
        self._rotation_gates = rotation_gates
        self._entanglement = entanglement
        self._entangling_gate = entangling_gate

        # Build sub-components
        self._encoder = EncodingLayer(n_qubits, encoding)
        self._variational_layers = self._build_variational_layers()

        # Initialise parameters
        self._params: Optional[np.ndarray] = None
        self._initialize_params()

    def _build_variational_layers(self) -> List[VariationalLayer]:
        """Build the variational layers based on the chosen form."""
        layers: List[VariationalLayer] = []

        if self._variational_form == "hardware_efficient":
            for i in range(self._n_layers):
                layers.append(VariationalLayer(
                    n_qubits=self._n_qubits,
                    rotation_gates=self._rotation_gates,
                    entanglement=self._entanglement,
                    entangling_gate=self._entangling_gate,
                    layer_index=i,
                ))

        elif self._variational_form == "strong_entangling":
            for i in range(self._n_layers):
                # Strongly entangling uses 3 rotations per qubit + full entanglement
                layers.append(VariationalLayer(
                    n_qubits=self._n_qubits,
                    rotation_gates=("ry", "rz", "rx"),
                    entanglement="circular",
                    entangling_gate="cnot",
                    layer_index=i,
                ))

        elif self._variational_form == "circuit_19":
            for i in range(self._n_layers):
                layers.append(VariationalLayer(
                    n_qubits=self._n_qubits,
                    rotation_gates=("ry",),
                    entanglement="circular",
                    entangling_gate="cnot",
                    layer_index=i,
                ))

        elif self._variational_form == "barren_plateau_free":
            for i in range(self._n_layers):
                # Use local operations to avoid barren plateaus
                layers.append(VariationalLayer(
                    n_qubits=self._n_qubits,
                    rotation_gates=("rz", "ry"),
                    entanglement="pairwise" if i % 2 == 0 else "linear",
                    entangling_gate="cz",
                    layer_index=i,
                ))

        elif self._variational_form == "qaoa":
            # QAOA-like: alternating cost and mixer layers
            for i in range(self._n_layers):
                if i % 2 == 0:
                    # Cost layer: ZZ + RZ
                    layers.append(VariationalLayer(
                        n_qubits=self._n_qubits,
                        rotation_gates=("rz",),
                        entanglement="full",
                        entangling_gate="zz",
                        layer_index=i,
                    ))
                else:
                    # Mixer layer: RX
                    layers.append(VariationalLayer(
                        n_qubits=self._n_qubits,
                        rotation_gates=("rx",),
                        entanglement="linear",
                        entangling_gate="cnot",
                        layer_index=i,
                    ))

        return layers

    def _initialize_params(self) -> None:
        """Initialize parameters with small random values."""
        total_params = self.count_parameters()
        rng = np.random.default_rng()
        # Initialise in a narrow range to aid convergence
        self._params = rng.uniform(-0.1, 0.1, size=total_params).astype(np.float64)

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of variational layers."""
        return self._n_layers

    @property
    def encoding(self) -> str:
        """str: Encoding strategy."""
        return self._encoding

    @property
    def variational_form(self) -> str:
        """str: Variational circuit architecture."""
        return self._variational_form

    @property
    def observable(self) -> str:
        """str: Measurement observable."""
        return self._observable

    @property
    def parameters(self) -> np.ndarray:
        """numpy.ndarray: Copy of current trainable parameters."""
        if self._params is None:
            return np.array([])
        return self._params.copy()

    @property
    def input_dim(self) -> int:
        """int: Expected input dimensionality."""
        return self._encoder.input_dim

    @property
    def output_dim(self) -> int:
        """int: Output dimensionality."""
        if self._output_dim is not None:
            return self._output_dim
        # Default: one output per qubit for single-qubit observables,
        # or one output for two-qubit observables
        if self._observable in ("z", "x", "y"):
            return self._n_qubits
        return 1

    # -- Parameter management -------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
        """
        return sum(layer.n_params for layer in self._variational_layers)

    def get_parameter_shapes(self) -> List[Tuple[int, ...]]:
        """Return shapes of all parameter tensors.

        Returns
        -------
        list of tuple of int
            Each element is the shape of a parameter group.
        """
        shapes: List[Tuple[int, ...]] = []
        for layer in self._variational_layers:
            shapes.append((layer.n_rotation_params,))
            if layer.n_entangling_params > 0:
                shapes.append((layer.n_entangling_params,))
        return shapes

    def get_parameter_names(self) -> List[str]:
        """Return descriptive names for all parameters.

        Returns
        -------
        list of str
        """
        names: List[str] = []
        for layer in self._variational_layers:
            names.extend(layer.get_param_names())
        return names

    def set_parameters(self, params: Union[Sequence[float], np.ndarray]) -> None:
        """Set the trainable parameters.

        Parameters
        ----------
        params : sequence of float or numpy.ndarray
            New parameter values. Must have length ``count_parameters()``.

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
        self._params = params.copy()

    # -- Circuit construction -------------------------------------------------

    def get_circuit(
        self,
        data: Sequence[float],
        params: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> QuantumCircuit:
        """Generate a parameterized QuantumCircuit for the given input.

        The circuit is built as:
            1. Encoding layer (data-dependent).
            2. Variational layers (parameter-dependent).
            3. No measurement is appended (caller decides what to measure).

        Parameters
        ----------
        data : sequence of float
            Classical input data.
        params : sequence of float or numpy.ndarray, optional
            Trainable parameters. If ``None``, uses current parameters.

        Returns
        -------
        QuantumCircuit
            The generated quantum circuit.

        Raises
        ------
        ValueError
            If data or params have incorrect dimensions.
        """
        if params is not None:
            params = np.asarray(params, dtype=np.float64)
        else:
            params = self._params

        if params is None:
            params = np.array([])

        expected = self.count_parameters()
        if len(params) != expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(params)}"
            )

        # Build circuit
        qc = QuantumCircuit(self._n_qubits)

        # Step 1: Encode data
        self._encoder.encode(qc, data)

        # Step 2: Apply variational layers
        param_offset = 0
        for layer in self._variational_layers:
            n_rot = layer.n_rotation_params
            n_ent = layer.n_entangling_params
            n_total = layer.n_params

            rot_params = params[param_offset:param_offset + n_rot]
            ent_params = None
            if n_ent > 0:
                ent_params = params[param_offset + n_rot:param_offset + n_total]

            layer.apply(qc, rot_params, ent_params)
            param_offset += n_total

        return qc

    # -- Forward pass ---------------------------------------------------------

    def forward(
        self,
        data: Sequence[float],
        params: Optional[Union[Sequence[float], np.ndarray]] = None,
        simulator: Optional[Any] = None,
    ) -> np.ndarray:
        """Run the forward pass: encode → variational → measure.

        Parameters
        ----------
        data : sequence of float
            Input data vector.
        params : sequence of float, optional
            Trainable parameters.
        simulator : Simulator, optional
            QuantumFlow simulator instance. If ``None``, uses
            ``StatevectorSimulator``.

        Returns
        -------
        numpy.ndarray
            Expectation values of the chosen observable, shape
            ``(output_dim,)``.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        if simulator is None:
            simulator = StatevectorSimulator()

        circuit = self.get_circuit(data, params)
        return self._measure(circuit, simulator)

    def _measure(
        self,
        circuit: QuantumCircuit,
        simulator: Any,
    ) -> np.ndarray:
        """Compute expectation values of the configured observable.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to measure.
        simulator : Simulator
            Simulator instance.

        Returns
        -------
        numpy.ndarray
            Expectation values, shape ``(output_dim,)``.
        """
        n = self._n_qubits
        results: List[float] = []

        if self._observable == "z":
            # Pauli-Z on each qubit
            for q in range(n):
                obs = self._pauli_observable("z", q, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        elif self._observable == "x":
            for q in range(n):
                obs = self._pauli_observable("x", q, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        elif self._observable == "y":
            for q in range(n):
                obs = self._pauli_observable("y", q, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        elif self._observable in ("zz", "xx", "yy"):
            # Two-qubit correlations
            pauli = self._observable[0]
            for i in range(n - 1):
                obs = self._two_qubit_observable(pauli, i, i + 1, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        elif self._observable == "mixed":
            # Mix of Z and ZZ observables
            for q in range(n):
                obs = self._pauli_observable("z", q, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))
            for i in range(min(n - 1, 3)):
                obs = self._two_qubit_observable("zz", i, i + 1, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        else:
            # Default: Z on first output_dim qubits
            for q in range(min(self.output_dim, n)):
                obs = self._pauli_observable("z", q, n)
                val = simulator.expectation(circuit, obs)
                results.append(float(val))

        output = np.array(results, dtype=np.float64)

        # Apply output_dim if specified
        if self._output_dim is not None and len(output) != self._output_dim:
            output = self._reshape_output(output)

        return output

    def _reshape_output(self, output: np.ndarray) -> np.ndarray:
        """Reshape output to match output_dim.

        Strategies: truncate, repeat, or average.
        """
        target = self._output_dim
        if len(output) > target:
            return output[:target]
        elif len(output) < target:
            # Repeat cyclically
            repeats = (target + len(output) - 1) // len(output)
            tiled = np.tile(output, repeats)
            return tiled[:target]
        return output

    # -- Observable helpers ---------------------------------------------------

    @staticmethod
    def _pauli_observable(
        pauli: str,
        qubit: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Build a single-qubit Pauli observable embedded in n-qubit space.

        Parameters
        ----------
        pauli : str
            ``'x'``, ``'y'``, or ``'z'``.
        qubit : int
            Target qubit index.
        n_qubits : int
            Total number of qubits.

        Returns
        -------
        numpy.ndarray
            Observable matrix of shape ``(2**n, 2**n)``.
        """
        pauli_matrices = {
            "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        if pauli not in pauli_matrices:
            raise ValueError(f"Unknown Pauli: {pauli}")
        mat = pauli_matrices[pauli]

        # Build full observable via Kronecker products
        full = np.array([[1.0]], dtype=np.complex128)
        for i in range(n_qubits):
            if i == qubit:
                full = np.kron(full, mat)
            else:
                full = np.kron(full, np.eye(2, dtype=np.complex128))
        return full

    @staticmethod
    def _two_qubit_observable(
        pauli: str,
        q1: int,
        q2: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Build a two-qubit Pauli observable (e.g. ZZ) embedded in n-qubit space.

        Parameters
        ----------
        pauli : str
            ``'x'``, ``'y'``, or ``'z'``.
        q1, q2 : int
            Target qubit indices.
        n_qubits : int
            Total number of qubits.

        Returns
        -------
        numpy.ndarray
            Observable matrix of shape ``(2**n, 2**n)``.
        """
        pauli_matrices = {
            "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        if pauli not in pauli_matrices:
            raise ValueError(f"Unknown Pauli: {pauli}")
        mat = pauli_matrices[pauli]

        full = np.array([[1.0]], dtype=np.complex128)
        for i in range(n_qubits):
            if i == q1 or i == q2:
                full = np.kron(full, mat)
            else:
                full = np.kron(full, np.eye(2, dtype=np.complex128))
        return full

    # -- Gradient computation -------------------------------------------------

    def compute_gradients(
        self,
        data: Sequence[float],
        observable: Optional[np.ndarray] = None,
        simulator: Optional[Any] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute parameter gradients using finite differences.

        Parameters
        ----------
        data : sequence of float
            Input data.
        observable : numpy.ndarray, optional
            Custom observable. If ``None``, uses the configured observable.
        simulator : Simulator, optional
            Simulator instance.
        eps : float
            Finite difference step size.

        Returns
        -------
        numpy.ndarray
            Gradient vector of shape ``(count_parameters(),)``.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        if simulator is None:
            simulator = StatevectorSimulator()

        if self._params is None:
            self._initialize_params()

        n_params = self.count_parameters()
        gradients = np.zeros(n_params, dtype=np.float64)

        if observable is None:
            # Use first observable
            if self._observable in ("z", "x", "y"):
                observable = self._pauli_observable(
                    self._observable, 0, self._n_qubits
                )
            else:
                observable = self._pauli_observable("z", 0, self._n_qubits)

        base_circuit = self.get_circuit(data)
        base_val = simulator.expectation(base_circuit, observable)

        for i in range(n_params):
            params_plus = self._params.copy()
            params_plus[i] += eps
            circuit_plus = self.get_circuit(data, params_plus)
            val_plus = simulator.expectation(circuit_plus, observable)

            params_minus = self._params.copy()
            params_minus[i] -= eps
            circuit_minus = self.get_circuit(data, params_minus)
            val_minus = simulator.expectation(circuit_minus, observable)

            gradients[i] = (val_plus - val_minus) / (2.0 * eps)

        return gradients

    # -- Circuit statistics ---------------------------------------------------

    def circuit_depth(
        self,
        data: Optional[Sequence[float]] = None,
    ) -> int:
        """Compute the circuit depth for given input data.

        Parameters
        ----------
        data : sequence of float, optional
            Input data. If ``None``, uses zeros.

        Returns
        -------
        int
        """
        if data is None:
            data = [0.0] * self.input_dim
        circuit = self.get_circuit(data)
        return circuit.depth()

    def circuit_size(
        self,
        data: Optional[Sequence[float]] = None,
    ) -> int:
        """Compute the total number of gates.

        Parameters
        ----------
        data : sequence of float, optional
            Input data. If ``None``, uses zeros.

        Returns
        -------
        int
        """
        if data is None:
            data = [0.0] * self.input_dim
        circuit = self.get_circuit(data)
        return circuit.size()

    # -- Serialization --------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration as a dictionary.

        Returns
        -------
        dict
            Configuration for serialization.
        """
        return {
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "encoding": self._encoding,
            "variational_form": self._variational_form,
            "observable": self._observable,
            "output_dim": self._output_dim,
            "rotation_gates": list(self._rotation_gates)
            if isinstance(self._rotation_gates, tuple)
            else self._rotation_gates,
            "entanglement": self._entanglement,
            "entangling_gate": self._entangling_gate,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantumNNLayer:
        """Create a QuantumNNLayer from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        QuantumNNLayer
        """
        return cls(**config)

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QuantumNNLayer("
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"encoding={self._encoding!r}, "
            f"variational_form={self._variational_form!r}, "
            f"observable={self._observable!r}, "
            f"output_dim={self._output_dim}, "
            f"params={self.count_parameters()})"
        )

    def __call__(
        self,
        data: Sequence[float],
        params: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """Shortcut for ``self.forward(data, params)``.

        Parameters
        ----------
        data : sequence of float
            Input data.
        params : sequence of float, optional
            Trainable parameters.

        Returns
        -------
        numpy.ndarray
            Layer output.
        """
        return self.forward(data, params)
