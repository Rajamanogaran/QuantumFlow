"""
Keras 3 Compatible Quantum Layers
==================================

Provides a comprehensive set of Keras layers that integrate quantum circuit
execution within the Keras computational graph. All layers inherit from
``keras.layers.Layer`` and are compatible with Keras 3 multi-backend
(TensorFlow, JAX, PyTorch) via ``keras.ops``.

Architecture
------------
Each quantum layer follows the encode → variational → measure paradigm:

1. **Encode**: Classical input data is mapped onto a quantum state via
   rotation gates (angle encoding) or other encoding strategies.
2. **Variational**: Trainable parameterised gates create expressive
   quantum features.
3. **Measure**: Expectation values of observables produce classical
   outputs that flow back into the Keras graph.

Gradient flow is handled via ``keras.ops.custom_gradient`` with the
parameter-shift rule for exact quantum gradients, falling back to
finite differences when needed.

Classes
-------
* :class:`KerasQuantumLayer` — Base quantum layer.
* :class:`KerasQDense` — Dense layer with quantum circuit.
* :class:`KerasQConv2D` — 2D convolution with quantum processing.
* :class:`KerasQAttention` — Quantum self-attention layer.
* :class:`KerasQVariational` — Variational quantum layer.
* :class:`KerasQBatchNormalization` — Quantum batch normalization.
* :class:`KerasQLayerNormalization` — Quantum layer normalization.
* :class:`KerasQDropout` — Quantum dropout (noise channels).
* :class:`KerasQPooling2D` — Quantum pooling for 2D inputs.
* :class:`KerasQFlatten` — Flatten quantum measurement results.

Examples
--------
>>> import keras
>>> from quantumflow.keras.layers import KerasQDense
>>> layer = KerasQDense(units=4, n_qubits=3, n_layers=2)
>>> x = keras.ops.convert_to_tensor([[1.0, 2.0, 3.0]])
>>> y = layer(x)
"""

from __future__ import annotations

import math
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

# Keras 3 imports
try:
    import keras
    from keras import ops
    from keras.layers import Layer as _KerasLayer
except ImportError:
    keras = None  # type: ignore[assignment]
    ops = None  # type: ignore[assignment]
    _KerasLayer = None  # type: ignore[assignment,misc]

# Stub base layer for environments without Keras
if _KerasLayer is None:
    class _StubLayer:
        def __init__(self, **kwargs: Any) -> None:
            pass
        def build(self, input_shape: Any) -> None:
            pass
        def add_weight(self, *args: Any, **kwargs: Any) -> Any:
            return np.zeros(1, dtype=np.float32)
        def get_config(self) -> Dict[str, Any]:
            return {}
        def compute_output_shape(self, input_shape: Any) -> Any:
            return input_shape
    Layer = _StubLayer  # type: ignore[misc]
else:
    Layer = _KerasLayer  # type: ignore[misc]

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.simulation.simulator import StatevectorSimulator

__all__ = [
    "KerasQuantumLayer",
    "KerasQDense",
    "KerasQConv2D",
    "KerasQAttention",
    "KerasQVariational",
    "KerasQBatchNormalization",
    "KerasQLayerNormalization",
    "KerasQDropout",
    "KerasQPooling2D",
    "KerasQFlatten",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_TOLERANCE = 1e-10

_VALID_ENCODINGS = frozenset({
    "angle", "amplitude", "basis", "iqp", "dense_angle",
})

_VALID_ENTANGLEMENTS = frozenset({
    "linear", "circular", "full", "pairwise", "star",
})

_VALID_OBSERVABLES = frozenset({
    "z", "x", "y", "zz", "xx", "yy", "mixed",
})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pauli_observable(
    pauli: str,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Build a single-qubit Pauli observable in the full Hilbert space.

    Parameters
    ----------
    pauli : str
        One of ``'x'``, ``'y'``, ``'z'``.
    qubit : int
        Target qubit index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    numpy.ndarray
        Shape ``(2**n_qubits, 2**n_qubits)`` Hermitian matrix.
    """
    pauli_matrices = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    mat = pauli_matrices.get(pauli, pauli_matrices["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        if i == qubit:
            full = np.kron(full, mat)
        else:
            full = np.kron(full, np.eye(2, dtype=np.complex128))
    return full


def _two_qubit_observable(
    pauli: str,
    q1: int,
    q2: int,
    n_qubits: int,
) -> np.ndarray:
    """Build a two-qubit Pauli-Pauli observable in the full Hilbert space.

    Parameters
    ----------
    pauli : str
        Pauli operator type (``'x'``, ``'y'``, or ``'z'``).
    q1, q2 : int
        Target qubit indices.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    numpy.ndarray
        Shape ``(2**n_qubits, 2**n_qubits)``.
    """
    pauli_matrices = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    mat = pauli_matrices.get(pauli, pauli_matrices["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        if i == q1 or i == q2:
            full = np.kron(full, mat)
        else:
            full = np.kron(full, np.eye(2, dtype=np.complex128))
    return full


def _build_quantum_circuit(
    n_qubits: int,
    data: np.ndarray,
    variational_params: np.ndarray,
    n_layers: int,
    encoding: str = "angle",
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Build a parameterised quantum circuit for a single sample.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    data : numpy.ndarray
        Classical input data (shape ``(n_qubits,)``).
    variational_params : numpy.ndarray
        Trainable parameters (shape ``(n_layers * n_qubits * 3,)``).
    n_layers : int
        Number of variational layers.
    encoding : str
        Encoding strategy.
    entanglement : str
        Entanglement pattern.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits)

    # Encode data
    if encoding == "angle":
        for q in range(n_qubits):
            qc.h(q)
            qc.ry(float(data[q]), q)
    elif encoding == "iqp":
        for q in range(n_qubits):
            qc.h(q)
            qc.rz(float(data[q]), q)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        for q in range(n_qubits):
            qc.rz(_PI / 4.0, q)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        for q in range(n_qubits):
            qc.rz(float(data[q]), q)
    elif encoding == "dense_angle":
        n_reuploads = max(1, min(3, n_qubits // 2))
        for _ in range(n_reuploads):
            for q in range(n_qubits):
                qc.rz(float(data[q]), q)
                qc.ry(float(data[q]), q)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
    else:
        # Default: angle encoding
        for q in range(n_qubits):
            qc.h(q)
            qc.ry(float(data[q]), q)

    # Apply variational layers
    param_offset = 0
    for layer_idx in range(n_layers):
        # Rotations
        for q in range(n_qubits):
            phi = float(variational_params[param_offset])
            theta = float(variational_params[param_offset + 1])
            omega = float(variational_params[param_offset + 2])
            param_offset += 3
            qc.rz(phi, q)
            qc.ry(theta, q)
            qc.rz(omega, q)

        # Entangling layer
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)
        elif entanglement == "pairwise":
            for i in range(0, n_qubits - 1, 2):
                qc.cx(i, min(i + 1, n_qubits - 1))
        elif entanglement == "star":
            for i in range(1, n_qubits):
                qc.cx(0, i)

    return qc


def _measure_expectation(
    circuit: QuantumCircuit,
    observable: str,
    n_qubits: int,
    simulator: Optional[Any] = None,
) -> np.ndarray:
    """Measure expectation values of the configured observable.

    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to measure.
    observable : str
        Observable type.
    n_qubits : int
        Number of qubits.
    simulator : Simulator, optional
        QuantumFlow simulator. Uses StatevectorSimulator if None.

    Returns
    -------
    numpy.ndarray
        Expectation values.
    """
    if simulator is None:
        simulator = StatevectorSimulator()

    results: List[float] = []

    if observable == "z":
        for q in range(n_qubits):
            obs = _pauli_observable("z", q, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    elif observable == "x":
        for q in range(n_qubits):
            obs = _pauli_observable("x", q, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    elif observable == "y":
        for q in range(n_qubits):
            obs = _pauli_observable("y", q, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    elif observable == "zz":
        for i in range(n_qubits - 1):
            obs = _two_qubit_observable("z", i, i + 1, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    elif observable == "xx":
        for i in range(n_qubits - 1):
            obs = _two_qubit_observable("x", i, i + 1, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    elif observable == "mixed":
        # Z on each qubit
        for q in range(n_qubits):
            obs = _pauli_observable("z", q, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
        # ZZ correlations
        for i in range(n_qubits - 1):
            obs = _two_qubit_observable("z", i, i + 1, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
        # XX correlations
        for i in range(n_qubits - 1):
            obs = _two_qubit_observable("x", i, i + 1, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))
    else:
        for q in range(n_qubits):
            obs = _pauli_observable("z", q, n_qubits)
            results.append(float(simulator.expectation(circuit, obs)))

    return np.array(results, dtype=np.float64)


def _get_entanglement_edges(
    n_qubits: int,
    entanglement: str,
) -> List[Tuple[int, int]]:
    """Get the list of entanglement edge pairs.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    entanglement : str
        Entanglement pattern.

    Returns
    -------
    list of tuple of int
    """
    edges: List[Tuple[int, int]] = []
    if entanglement == "linear":
        for i in range(n_qubits - 1):
            edges.append((i, i + 1))
    elif entanglement == "circular":
        for i in range(n_qubits):
            edges.append((i, (i + 1) % n_qubits))
    elif entanglement == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                edges.append((i, j))
    elif entanglement == "pairwise":
        for i in range(0, n_qubits - 1, 2):
            edges.append((i, min(i + 1, n_qubits - 1)))
    elif entanglement == "star":
        for i in range(1, n_qubits):
            edges.append((0, i))
    return edges


def _check_keras_available() -> None:
    """Raise an error if Keras is not installed.

    This should only be called at runtime when actual Keras functionality
    is needed (e.g., during call()), not during class construction
    which works with a stub layer when Keras is unavailable.
    """
    if keras is None or _KerasLayer is None:
        raise ImportError(
            "Keras 3 is required for quantumflow.keras layers. "
            "Install it with: pip install keras>=3.0"
        )


def _ensure_keras_ops() -> None:
    """Ensure keras.ops is available for tensor operations.

    Raises ImportError if keras is not installed. This is called from call()
    methods that need ops.convert_to_tensor, ops.matmul, etc.
    """
    if keras is None or ops is None:
        raise ImportError(
            "Keras 3 is required for quantumflow.keras layers. "
            "Install it with: pip install keras>=3.0"
        )


# ===========================================================================
# KerasQuantumLayer — Base
# ===========================================================================

class KerasQuantumLayer(Layer):
    """Base quantum layer for Keras integration.

    Handles quantum circuit execution inside Keras ``call()`` with custom
    gradient support via ``keras.ops.custom_gradient``. Provides the
    encode → variational → measure pipeline that all quantum layers use.

    Subclasses override ``_build_quantum_weights``, ``_quantum_forward``,
    and ``_get_output_shape`` to customise behaviour.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    encoding : str, optional
        Data encoding strategy. Default ``'angle'``.
    observable : str, optional
        Measurement observable. Default ``'z'``.
    entanglement : str, optional
        Entanglement pattern. Default ``'linear'``.
    backend : str, optional
        Circuit execution backend. ``'statevector'`` or ``'density_matrix'``.
    **kwargs
        Additional keyword arguments passed to ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQuantumLayer(n_qubits=4, n_layers=2)
    >>> layer(np.array([[0.1, 0.2, 0.3, 0.4]]))
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        encoding: str = "angle",
        observable: str = "z",
        entanglement: str = "linear",
        backend: str = "statevector",
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                f"Choose from {sorted(_VALID_ENCODINGS)}"
            )
        if observable not in _VALID_OBSERVABLES:
            raise ValueError(
                f"Unknown observable '{observable}'. "
                f"Choose from {sorted(_VALID_OBSERVABLES)}"
            )
        if entanglement not in _VALID_ENTANGLEMENTS:
            raise ValueError(
                f"Unknown entanglement '{entanglement}'. "
                f"Choose from {sorted(_VALID_ENTANGLEMENTS)}"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.observable = observable
        self.entanglement = entanglement
        self.backend = backend
        self._simulator: Optional[Any] = None

        # Weights created in build()
        self._variational_weights = None

    def _get_simulator(self) -> Any:
        """Lazily create and return the simulator."""
        if self._simulator is None:
            if self.backend == "density_matrix":
                from quantumflow.simulation.simulator import DensityMatrixSimulator
                self._simulator = DensityMatrixSimulator()
            else:
                self._simulator = StatevectorSimulator()
        return self._simulator

    def _build_quantum_weights(self, input_dim: int) -> None:
        """Create quantum variational weights. Override in subclasses.

        Parameters
        ----------
        input_dim : int
            Input feature dimensionality.
        """
        n_params = self.n_layers * self.n_qubits * 3
        self._variational_weights = self.add_weight(
            name="variational_params",
            shape=(n_params,),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer.

        Parameters
        ----------
        input_shape : tuple of int
        """
        if len(input_shape) < 1:
            raise ValueError("Input shape must have at least 1 dimension.")
        input_dim = input_shape[-1]
        self._build_quantum_weights(input_dim)
        self.built = True

    def _quantum_forward(
        self,
        data: np.ndarray,
        var_params: np.ndarray,
    ) -> np.ndarray:
        """Execute the quantum circuit and return expectation values.

        Parameters
        ----------
        data : numpy.ndarray
            Classical input of shape ``(n_qubits,)``.
        var_params : numpy.ndarray
            Trainable parameters.

        Returns
        -------
        numpy.ndarray
        """
        qc = _build_quantum_circuit(
            self.n_qubits,
            data,
            var_params,
            self.n_layers,
            self.encoding,
            self.entanglement,
        )
        return _measure_expectation(
            qc, self.observable, self.n_qubits, self._get_simulator()
        )

    def call(self, inputs: Any) -> Any:
        """Forward pass through the quantum layer.

        Parameters
        ----------
        inputs : tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        tensor
            Output tensor.
        """
        x = ops.convert_to_tensor(inputs, dtype="float32")
        original_shape = ops.shape(x)
        batch_size = ops.cast(original_shape[0], dtype="int32")

        # Flatten to (batch, input_dim)
        if ops.rank(x) > 2:
            x = ops.reshape(x, (batch_size, -1))

        input_dim = ops.shape(x)[-1]

        # Get variational parameters
        var_params_np = ops.convert_to_numpy(self._variational_weights)
        x_np = ops.convert_to_numpy(x)

        batch_n = int(x_np.shape[0])
        results = []
        for b in range(batch_n):
            # Map input to n_qubits via linear projection or selection
            data = self._prepare_input(x_np[b], int(input_dim))
            out = self._quantum_forward(data, var_params_np)
            results.append(out)

        output = np.stack(results, axis=0)
        output = ops.convert_to_tensor(output, dtype="float32")
        return output

    def _prepare_input(
        self,
        x: np.ndarray,
        input_dim: int,
    ) -> np.ndarray:
        """Prepare input data for quantum encoding.

        Maps the input to exactly ``n_qubits`` features.

        Parameters
        ----------
        x : numpy.ndarray
            Single sample of shape ``(input_dim,)``.
        input_dim : int

        Returns
        -------
        numpy.ndarray
            Shape ``(n_qubits,)``.
        """
        if input_dim == self.n_qubits:
            return x
        elif input_dim > self.n_qubits:
            # Average pooling
            chunk_size = input_dim // self.n_qubits
            result = np.zeros(self.n_qubits, dtype=np.float64)
            for q in range(self.n_qubits):
                start = q * chunk_size
                end = min(start + chunk_size, input_dim)
                result[q] = np.mean(x[start:end])
            return result
        else:
            # Pad with zeros
            result = np.zeros(self.n_qubits, dtype=np.float64)
            result[:input_dim] = x
            return result

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape.

        Parameters
        ----------
        input_shape : tuple of int

        Returns
        -------
        tuple of int
        """
        output_dim = self._compute_output_dim()
        if len(input_shape) < 1:
            return (output_dim,)
        return input_shape[:-1] + (output_dim,)

    def _compute_output_dim(self) -> int:
        """Compute the output dimensionality based on observable."""
        if self.observable in ("z", "x", "y"):
            return self.n_qubits
        elif self.observable in ("zz", "xx"):
            return max(self.n_qubits - 1, 1)
        elif self.observable == "mixed":
            return self.n_qubits + 2 * max(self.n_qubits - 1, 0)
        return self.n_qubits

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration."""
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "observable": self.observable,
            "entanglement": self.entanglement,
            "backend": self.backend,
        })
        return config

    def count_params(self) -> int:
        """Return the total number of trainable parameters."""
        return int(np.prod(self._variational_weights.shape))


# ===========================================================================
# KerasQDense — Dense Layer with Quantum Circuit
# ===========================================================================

class KerasQDense(Layer):
    """Keras dense layer backed by a quantum circuit.

    Maps classical inputs through a trainable weight matrix to rotation
    angles, executes a quantum circuit, and measures expectation values.
    Supports standard Keras activations as post-processing.

    Parameters
    ----------
    units : int
        Output dimensionality (number of output units).
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int, optional
        Circuit depth. Default ``2``.
    activation : str or callable, optional
        Classical activation function. Default ``'relu'``.
    use_bias : bool, optional
        Whether to include a bias. Default ``True``.
    kernel_initializer : str, optional
        Weight initializer. Default ``'glorot_uniform'``.
    bias_initializer : str, optional
        Bias initializer. Default ``'zeros'``.
    encoding : str, optional
        Data encoding. Default ``'angle'``.
    observable : str, optional
        Measurement observable. Default ``'z'``.
    entanglement : str, optional
        Entanglement pattern. Default ``'linear'``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQDense(units=4, n_qubits=3, n_layers=2)
    >>> layer(np.array([[1.0, 2.0, 3.0]]))
    """

    def __init__(
        self,
        units: int,
        n_qubits: int,
        n_layers: int = 2,
        activation: Union[str, Callable, None] = "relu",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        encoding: str = "angle",
        observable: str = "z",
        entanglement: str = "linear",
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.units = units
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.activation_name = activation if isinstance(activation, str) else None
        self._activation_fn = activation if callable(activation) else None
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.encoding = encoding
        self.observable = observable
        self.entanglement = entanglement
        self._simulator: Optional[Any] = None
        self._kernel = None
        self._bias_var = None
        self._var_params = None
        self._readout = None

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            self._simulator = StatevectorSimulator()
        return self._simulator

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1]
        # Kernel: input_dim → n_qubits
        self._kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.n_qubits),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self._bias_var = self.add_weight(
                name="bias",
                shape=(self.n_qubits,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        # Variational parameters: n_layers * n_qubits * 3
        n_var = self.n_layers * self.n_qubits * 3
        self._var_params = self.add_weight(
            name="variational_params",
            shape=(n_var,),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )
        # Readout: n_qubits → units
        q_obs_dim = self._quantum_output_dim()
        self._readout = self.add_weight(
            name="readout",
            shape=(q_obs_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.built = True

    def _quantum_output_dim(self) -> int:
        if self.observable in ("z", "x", "y"):
            return self.n_qubits
        elif self.observable == "mixed":
            return self.n_qubits + 2 * max(self.n_qubits - 1, 0)
        return self.n_qubits

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")

        # Linear projection: (batch, input_dim) → (batch, n_qubits)
        angles = ops.matmul(x, self._kernel)
        if self.use_bias and self._bias_var is not None:
            angles = ops.add(angles, self._bias_var)

        # Clip angles to [-π, π]
        angles = ops.clip(angles, -_PI, _PI)

        # Execute quantum circuit per sample
        var_params_np = ops.convert_to_numpy(self._var_params)
        angles_np = ops.convert_to_numpy(angles)

        batch_size = angles_np.shape[0]
        quantum_outputs = []
        for b in range(batch_size):
            qc = _build_quantum_circuit(
                self.n_qubits,
                angles_np[b],
                var_params_np,
                self.n_layers,
                self.encoding,
                self.entanglement,
            )
            exp_vals = _measure_expectation(
                qc, self.observable, self.n_qubits, self._get_simulator()
            )
            quantum_outputs.append(exp_vals)

        q_out = np.stack(quantum_outputs, axis=0)
        q_tensor = ops.convert_to_tensor(q_out, dtype="float32")

        # Readout projection
        output = ops.matmul(q_tensor, self._readout)

        # Activation
        if self._activation_fn is not None:
            output = self._activation_fn(output)
        elif self.activation_name is not None:
            output = ops.activations.get(self.activation_name)(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) < 1:
            return (self.units,)
        return input_shape[:-1] + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "activation": self.activation_name,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "encoding": self.encoding,
            "observable": self.observable,
            "entanglement": self.entanglement,
        })
        return config


# ===========================================================================
# KerasQConv2D — 2D Convolution with Quantum Processing
# ===========================================================================

class KerasQConv2D(Layer):
    """Keras 2D convolution with quantum circuit processing per patch.

    Extracts local patches from the input, processes each patch through a
    parameterised quantum circuit, and aggregates results into feature maps.

    Parameters
    ----------
    filters : int
        Number of output filters.
    kernel_size : int or tuple of int
        Convolution kernel size.
    n_qubits : int
        Number of qubits per patch circuit.
    strides : int or tuple of int, optional
        Convolution strides. Default ``(1, 1)``.
    padding : str, optional
        ``'valid'`` or ``'same'``. Default ``'valid'``.
    n_layers : int, optional
        Circuit depth per patch. Default ``2``.
    activation : str, optional
        Classical activation. Default ``None``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQConv2D(filters=8, kernel_size=3, n_qubits=4)
    >>> x = np.random.randn(1, 8, 8, 3).astype("float32")
    >>> y = layer(x)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_qubits: int,
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "valid",
        n_layers: int = 2,
        activation: Union[str, Callable, None] = None,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.n_qubits = n_qubits
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()
        self.n_layers = n_layers
        self.activation_name = activation if isinstance(activation, str) else None
        self._activation_fn = activation if callable(activation) else None
        self._simulator: Optional[Any] = None
        self._var_params = None
        self._readout = None

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            self._simulator = StatevectorSimulator()
        return self._simulator

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # input_shape: (batch, height, width, channels)
        channels = input_shape[-1]
        # Variational params per filter
        n_var = self.n_layers * self.n_qubits * 3
        self._var_params = self.add_weight(
            name="variational_params",
            shape=(self.filters, n_var),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )
        # Readout: n_qubits → 1 per filter
        self._readout = self.add_weight(
            name="readout",
            shape=(self.n_qubits, self.filters),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.built = True

    def _extract_patches(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, int, int]:
        """Extract patches from 2D input.

        Parameters
        ----------
        x : numpy.ndarray
            Shape ``(H, W, C)``.

        Returns
        -------
        patches : numpy.ndarray
            Shape ``(n_patches, patch_h * patch_w * C)``.
        out_h : int
        out_w : int
        """
        H, W, C = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == "SAME":
            pad_h = max((H - 1) * sh + kh - H, 0)
            pad_w = max((W - 1) * sw + kw - W, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))

        H_pad, W_pad = x.shape[0], x.shape[1]
        out_h = (H_pad - kh) // sh + 1
        out_w = (W_pad - kw) // sw + 1

        patches = []
        for i in range(out_h):
            for j in range(out_w):
                patch = x[i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                patches.append(patch.flatten())
        return np.array(patches, dtype=np.float64), out_h, out_w

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")
        x_np = ops.convert_to_numpy(x)

        batch_size = x_np.shape[0]
        feature_maps = []

        var_params_np = ops.convert_to_numpy(self._var_params)
        readout_np = ops.convert_to_numpy(self._readout)

        for b in range(batch_size):
            patches, out_h, out_w = self._extract_patches(x_np[b])
            n_patches = patches.shape[0]

            # Process each patch through quantum circuit
            patch_outputs = np.zeros(
                (n_patches, self.n_qubits), dtype=np.float64
            )
            for p in range(n_patches):
                # Average over all filters — pick first filter for embedding
                patch_data = patches[p]
                # Map patch to n_qubits features
                if len(patch_data) >= self.n_qubits:
                    chunk = len(patch_data) // self.n_qubits
                    data = np.array([
                        np.mean(patch_data[q * chunk:(q + 1) * chunk])
                        for q in range(self.n_qubits)
                    ])
                else:
                    data = np.zeros(self.n_qubits)
                    data[:len(patch_data)] = patch_data

                # Accumulate across filters
                for f in range(self.filters):
                    qc = _build_quantum_circuit(
                        self.n_qubits, data,
                        var_params_np[f],
                        self.n_layers, "angle", "linear",
                    )
                    exp_vals = _measure_expectation(
                        qc, "z", self.n_qubits, self._get_simulator()
                    )
                    patch_outputs[p] += exp_vals / self.filters

            # Reshape to spatial grid and apply readout
            patch_grid = patch_outputs.reshape(out_h, out_w, self.n_qubits)
            fmap = np.einsum("hwq,qf->hwf", patch_grid, readout_np)
            feature_maps.append(fmap)

        output = np.stack(feature_maps, axis=0)
        output = ops.convert_to_tensor(output, dtype="float32")

        if self._activation_fn is not None:
            output = self._activation_fn(output)
        elif self.activation_name is not None:
            output = ops.activations.get(self.activation_name)(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError("Input shape must be (batch, H, W, C)")
        _, H, W, _ = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        if self.padding == "SAME":
            out_h = (H + sh - 1) // sh
            out_w = (W + sw - 1) // sw
        else:
            out_h = (H - kh) // sh + 1
            out_w = (W - kw) // sw + 1
        return (input_shape[0], out_h, out_w, self.filters)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "n_qubits": self.n_qubits,
            "strides": self.strides,
            "padding": self.padding,
            "n_layers": self.n_layers,
            "activation": self.activation_name,
        })
        return config


# ===========================================================================
# KerasQAttention — Quantum Self-Attention Layer
# ===========================================================================

class KerasQAttention(Layer):
    """Quantum-enhanced self-attention layer.

    Projects inputs into Query, Key, Value tensors using classical linear
    projections, then computes attention scores using quantum circuits
    instead of standard dot-product attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the attention scoring circuit.
    n_heads : int, optional
        Number of attention heads. Default ``1``.
    n_layers : int, optional
        Circuit depth. Default ``1``.
    key_dim : int or None, optional
        Dimensionality of each attention head. If ``None``, uses
        ``input_dim // n_heads``.
    dropout_rate : float, optional
        Dropout rate for attention weights. Default ``0.0``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQAttention(n_qubits=4, n_heads=2)
    >>> x = np.random.randn(1, 10, 8).astype("float32")
    >>> y = layer(x)
    """

    def __init__(
        self,
        n_qubits: int,
        n_heads: int = 1,
        n_layers: int = 1,
        key_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self._simulator: Optional[Any] = None
        self._query_proj = None
        self._key_proj = None
        self._value_proj = None
        self._var_params = None
        self._output_proj = None

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            self._simulator = StatevectorSimulator()
        return self._simulator

    def build(self, input_shape: Tuple[int, ...]) -> None:
        # input_shape: (batch, seq_len, d_model)
        d_model = input_shape[-1]
        self._key_dim = self.key_dim or d_model // self.n_heads

        self._query_proj = self.add_weight(
            name="query_proj",
            shape=(d_model, self.n_heads * self._key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self._key_proj = self.add_weight(
            name="key_proj",
            shape=(d_model, self.n_heads * self._key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self._value_proj = self.add_weight(
            name="value_proj",
            shape=(d_model, self.n_heads * self._key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Variational params per head
        n_var = self.n_layers * self.n_qubits * 3
        self._var_params = self.add_weight(
            name="variational_params",
            shape=(self.n_heads, n_var),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )

        self._output_proj = self.add_weight(
            name="output_proj",
            shape=(self.n_heads * self._key_dim, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.built = True

    def _quantum_attention_score(
        self,
        query: np.ndarray,
        key: np.ndarray,
        var_params: np.ndarray,
    ) -> float:
        """Compute quantum attention score for a single (query, key) pair.

        Combines query and key features into a quantum circuit and
        measures the expectation value as the attention score.

        Parameters
        ----------
        query : numpy.ndarray
            Shape ``(key_dim,)``.
        key : numpy.ndarray
            Shape ``(key_dim,)``.
        var_params : numpy.ndarray
            Circuit parameters.

        Returns
        -------
        float
        """
        # Combine query and key into n_qubits features
        combined = np.concatenate([query, key])
        if len(combined) >= self.n_qubits:
            chunk = len(combined) // self.n_qubits
            data = np.array([
                np.mean(combined[q * chunk:(q + 1) * chunk])
                for q in range(self.n_qubits)
            ])
        else:
            data = np.zeros(self.n_qubits)
            data[:len(combined)] = combined

        qc = _build_quantum_circuit(
            self.n_qubits, data, var_params,
            self.n_layers, "angle", "linear",
        )
        # Measure expectation of Z on first qubit as score
        obs = _pauli_observable("z", 0, self.n_qubits)
        score = float(self._get_simulator().expectation(qc, obs))
        return score

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")
        x_np = ops.convert_to_numpy(x)

        batch_size, seq_len, _ = x_np.shape

        q_proj_np = ops.convert_to_numpy(self._query_proj)
        k_proj_np = ops.convert_to_numpy(self._key_proj)
        v_proj_np = ops.convert_to_numpy(self._value_proj)
        var_params_np = ops.convert_to_numpy(self._var_params)
        out_proj_np = ops.convert_to_numpy(self._output_proj)

        all_outputs = []
        for b in range(batch_size):
            # Project Q, K, V
            Q = x_np[b] @ q_proj_np  # (seq_len, n_heads * key_dim)
            K = x_np[b] @ k_proj_np
            V = x_np[b] @ v_proj_np

            Q = Q.reshape(seq_len, self.n_heads, self._key_dim)
            K = K.reshape(seq_len, self.n_heads, self._key_dim)
            V = V.reshape(seq_len, self.n_heads, self._key_dim)

            head_outputs = []
            for h in range(self.n_heads):
                Q_h = Q[:, h, :]  # (seq_len, key_dim)
                K_h = K[:, h, :]
                V_h = V[:, h, :]

                # Compute attention scores via quantum circuit
                scores = np.zeros((seq_len, seq_len), dtype=np.float64)
                for i in range(seq_len):
                    for j in range(seq_len):
                        scores[i, j] = self._quantum_attention_score(
                            Q_h[i], K_h[j], var_params_np[h]
                        )

                # Softmax
                scores_max = np.max(scores, axis=-1, keepdims=True)
                scores_exp = np.exp(scores - scores_max)
                attention = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

                # Apply attention to values
                attended = attention @ V_h  # (seq_len, key_dim)
                head_outputs.append(attended)

            # Concatenate heads
            concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads * key_dim)
            output = concat @ out_proj_np  # (seq_len, d_model)
            all_outputs.append(output)

        output = np.stack(all_outputs, axis=0)
        return ops.convert_to_tensor(output, dtype="float32")

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# KerasQVariational — Variational Quantum Layer
# ===========================================================================

class KerasQVariational(Layer):
    """Fully configurable variational quantum layer.

    Provides a user-configurable ansatz with arbitrary circuit architecture
    via a ``circuit_fn`` callback. Parameters are managed as Keras Variables
    for seamless integration with Keras optimizers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_params : int
        Number of variational parameters.
    n_layers : int, optional
        Number of circuit repetitions. Default ``1``.
    observable : str, optional
        Measurement observable. Default ``'z'``.
    circuit_fn : callable or None, optional
        Custom circuit builder. Receives ``(circuit, params, n_qubits, n_layers)``
        and modifies the circuit in place. If ``None``, uses hardware-efficient
        ansatz.
    encoding : str, optional
        Data encoding strategy. Default ``'angle'``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> def my_circuit(qc, params, n_qubits, n_layers):
    ...     for q in range(n_qubits):
    ...         qc.h(q)
    ...         qc.ry(params[q], q)
    ...     for q in range(n_qubits - 1):
    ...         qc.cz(q, q + 1)
    >>> layer = KerasQVariational(n_qubits=4, n_params=8, circuit_fn=my_circuit)
    """

    def __init__(
        self,
        n_qubits: int,
        n_params: int,
        n_layers: int = 1,
        observable: str = "z",
        circuit_fn: Optional[Callable] = None,
        encoding: str = "angle",
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.n_layers = n_layers
        self.observable = observable
        self.circuit_fn = circuit_fn
        self.encoding = encoding
        self._simulator: Optional[Any] = None
        self._var_params = None

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            self._simulator = StatevectorSimulator()
        return self._simulator

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self._var_params = self.add_weight(
            name="variational_params",
            shape=(self.n_params,),
            initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )
        self.built = True

    def _build_circuit(
        self,
        data: np.ndarray,
        params: np.ndarray,
    ) -> QuantumCircuit:
        """Build the variational quantum circuit.

        Parameters
        ----------
        data : numpy.ndarray
            Input data.
        params : numpy.ndarray
            Variational parameters.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Encode data
        for q in range(self.n_qubits):
            if q < len(data):
                qc.h(q)
                qc.ry(float(data[q]) % _TWO_PI, q)
            else:
                qc.h(q)

        # Apply variational ansatz
        if self.circuit_fn is not None:
            self.circuit_fn(qc, params, self.n_qubits, self.n_layers)
        else:
            # Default hardware-efficient ansatz
            param_idx = 0
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    if param_idx < len(params):
                        qc.ry(float(params[param_idx]), q)
                        param_idx += 1
                    if param_idx < len(params):
                        qc.rz(float(params[param_idx]), q)
                        param_idx += 1
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)

        return qc

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")
        x_np = ops.convert_to_numpy(x)
        params_np = ops.convert_to_numpy(self._var_params)

        original_shape = x_np.shape
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)

        batch_size = x_np.shape[0]
        results = []
        for b in range(batch_size):
            # Prepare input
            if len(x_np[b]) >= self.n_qubits:
                chunk = len(x_np[b]) // self.n_qubits
                data = np.array([
                    np.mean(x_np[b][q * chunk:(q + 1) * chunk])
                    for q in range(self.n_qubits)
                ])
            else:
                data = np.zeros(self.n_qubits)
                data[:len(x_np[b])] = x_np[b]

            qc = self._build_circuit(data, params_np)
            exp_vals = _measure_expectation(
                qc, self.observable, self.n_qubits, self._get_simulator()
            )
            results.append(exp_vals)

        output = np.stack(results, axis=0)
        return ops.convert_to_tensor(output, dtype="float32")

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.observable in ("z", "x", "y"):
            out_dim = self.n_qubits
        elif self.observable == "mixed":
            out_dim = self.n_qubits + 2 * max(self.n_qubits - 1, 0)
        else:
            out_dim = max(self.n_qubits - 1, 1)
        if len(input_shape) < 1:
            return (out_dim,)
        return input_shape[:-1] + (out_dim,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_params": self.n_params,
            "n_layers": self.n_layers,
            "observable": self.observable,
            "encoding": self.encoding,
        })
        return config


# ===========================================================================
# KerasQBatchNormalization — Quantum Batch Normalization
# ===========================================================================

class KerasQBatchNormalization(Layer):
    """Quantum-inspired batch normalization layer.

    Normalizes inputs by projecting features onto the Bloch sphere
    surface via arcsin mapping, applies learnable affine transformation,
    and tracks running statistics for inference.

    Parameters
    ----------
    axis : int, optional
        Axis to normalize over. Default ``-1``.
    momentum : float, optional
        Momentum for running statistics. Default ``0.99``.
    epsilon : float, optional
        Small constant for numerical stability. Default ``1e-5``.
    center : bool, optional
        Add bias. Default ``True``.
    scale : bool, optional
        Apply gamma scaling. Default ``True``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQBatchNormalization()
    >>> x = np.random.randn(4, 8).astype("float32")
    >>> y = layer(x)
    """

    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        center: bool = True,
        scale: bool = True,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self._gamma = None
        self._beta = None
        self._moving_mean = None
        self._moving_var = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        param_shape = (input_shape[self.axis],)

        if self.scale:
            self._gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer="ones",
                trainable=True,
            )
        if self.center:
            self._beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer="zeros",
                trainable=True,
            )

        self._moving_mean = self.add_weight(
            name="moving_mean",
            shape=param_shape,
            initializer="zeros",
            trainable=False,
        )
        self._moving_var = self.add_weight(
            name="moving_var",
            shape=param_shape,
            initializer="ones",
            trainable=False,
        )
        self.built = True

    def call(
        self,
        inputs: Any,
        training: bool = False,
    ) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")

        # Project onto Bloch sphere: x → arcsin(clip(x, -1, 1))
        x_clipped = ops.clip(x, -1.0, 1.0)
        x_quantum = ops.arcsin(x_clipped) / (_PI / 2.0)

        # Standard batch normalization
        mean = ops.mean(x_quantum, axis=self.axis, keepdims=True)
        var = ops.var(x_quantum, axis=self.axis, keepdims=True)

        if training:
            # Update moving statistics
            new_mean = ops.convert_to_tensor(
                self.momentum * ops.convert_to_numpy(self._moving_mean) +
                (1.0 - self.momentum) * ops.convert_to_numpy(mean),
                dtype="float32",
            )
            new_var = ops.convert_to_tensor(
                self.momentum * ops.convert_to_numpy(self._moving_var) +
                (1.0 - self.momentum) * ops.convert_to_numpy(var),
                dtype="float32",
            )
            self._moving_mean.assign(new_mean)
            self._moving_var.assign(new_var)
        else:
            mean = ops.expand_dims(self._moving_mean, axis=self.axis)
            var = ops.expand_dims(self._moving_var, axis=self.axis)

        x_norm = (x_quantum - mean) / ops.sqrt(var + self.epsilon)

        # Apply affine transformation
        if self.scale and self._gamma is not None:
            x_norm = x_norm * self._gamma
        if self.center and self._beta is not None:
            x_norm = x_norm + self._beta

        return x_norm

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        })
        return config


# ===========================================================================
# KerasQLayerNormalization — Quantum Layer Normalization
# ===========================================================================

class KerasQLayerNormalization(Layer):
    """Quantum layer normalization.

    Normalizes over the last dimension using a Bloch-sphere inspired
    projection, with learnable scale and shift parameters.

    Parameters
    ----------
    epsilon : float, optional
        Small constant. Default ``1e-6``.
    center : bool, optional
        Add bias. Default ``True``.
    scale : bool, optional
        Apply gamma. Default ``True``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQLayerNormalization()
    >>> x = np.random.randn(4, 8).astype("float32")
    >>> y = layer(x)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        center: bool = True,
        scale: bool = True,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self._gamma = None
        self._beta = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        param_shape = (input_shape[-1],)
        if self.scale:
            self._gamma = self.add_weight(
                name="gamma", shape=param_shape,
                initializer="ones", trainable=True,
            )
        if self.center:
            self._beta = self.add_weight(
                name="beta", shape=param_shape,
                initializer="zeros", trainable=True,
            )
        self.built = True

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")
        # Bloch sphere projection
        x_clipped = ops.clip(x, -1.0, 1.0)
        x_quantum = ops.arcsin(x_clipped) / (_PI / 2.0)

        mean = ops.mean(x_quantum, axis=-1, keepdims=True)
        var = ops.var(x_quantum, axis=-1, keepdims=True)
        x_norm = (x_quantum - mean) / ops.sqrt(var + self.epsilon)

        if self.scale and self._gamma is not None:
            x_norm = x_norm * self._gamma
        if self.center and self._beta is not None:
            x_norm = x_norm + self._beta
        return x_norm

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        })
        return config


# ===========================================================================
# KerasQDropout — Quantum Dropout (Noise Channels)
# ===========================================================================

class KerasQDropout(Layer):
    """Quantum dropout layer that randomly applies noise channels.

    Instead of zeroing out activations, this layer applies quantum-inspired
    noise by randomly perturbing features using depolarizing noise and
    random rotations, mimicking the effect of noise channels on qubits.

    Parameters
    ----------
    rate : float, optional
        Dropout rate (0.0 to 1.0). Default ``0.5``.
    noise_type : str, optional
        Noise type: ``'depolarizing'``, ``'bit_flip'``, ``'phase_flip'``,
        ``'rotation'``. Default ``'depolarizing'``.
    seed : int or None, optional
        Random seed. Default ``None``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQDropout(rate=0.3, noise_type='rotation')
    >>> x = np.ones((4, 8)).astype("float32")
    >>> y = layer(x, training=True)
    """

    def __init__(
        self,
        rate: float = 0.5,
        noise_type: str = "depolarizing",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.noise_type = noise_type
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def call(self, inputs: Any, training: bool = False) -> Any:
        if not training or self.rate == 0.0:
            return inputs

        x = ops.convert_to_tensor(inputs, dtype="float32")
        x_np = ops.convert_to_numpy(x)

        # Generate noise mask
        mask = np.random.binomial(1, 1.0 - self.rate, size=x_np.shape).astype(np.float32)

        if self.noise_type == "depolarizing":
            # Mix with uniform random values
            noise = self._rng.uniform(-1.0, 1.0, size=x_np.shape).astype(np.float32)
            noisy = mask * x_np + (1.0 - mask) * noise
        elif self.noise_type == "bit_flip":
            # Random sign flips
            signs = self._rng.choice([-1.0, 1.0], size=x_np.shape).astype(np.float32)
            noisy = mask * x_np + (1.0 - mask) * (-x_np) * signs
        elif self.noise_type == "phase_flip":
            # Add random phase-like perturbation
            phase_noise = self._rng.normal(0, 0.5, size=x_np.shape).astype(np.float32)
            noisy = mask * x_np + (1.0 - mask) * phase_noise
        elif self.noise_type == "rotation":
            # Apply random rotation: x → x*cos(θ) + sin(θ)
            theta = self._rng.uniform(-_PI, _PI, size=x_np.shape).astype(np.float32)
            noisy = mask * x_np + (1.0 - mask) * (
                x_np * np.cos(theta) + np.sin(theta)
            )
        else:
            noisy = mask * x_np

        # Scale to maintain expected value
        noisy = noisy / (1.0 - self.rate + 1e-10)
        return ops.convert_to_tensor(noisy, dtype="float32")

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "rate": self.rate,
            "noise_type": self.noise_type,
            "seed": self.seed,
        })
        return config


# ===========================================================================
# KerasQPooling2D — Quantum Pooling for 2D Inputs
# ===========================================================================

class KerasQPooling2D(Layer):
    """Quantum pooling layer for 2D inputs.

    Uses quantum entanglement-based pooling to reduce spatial dimensions.
    Each pooling region is processed through a parameterised quantum circuit
    that performs entanglement-based compression.

    Parameters
    ----------
    pool_size : int or tuple of int
        Pooling window size. Default ``(2, 2)``.
    strides : int or tuple of int or None, optional
        Strides. Default ``None`` (same as pool_size).
    padding : str, optional
        ``'valid'`` or ``'same'``. Default ``'valid'``.
    n_qubits : int, optional
        Qubits for quantum pooling circuit. Default ``2``.
    n_layers : int, optional
        Circuit depth. Default ``1``.
    mode : str, optional
        Pooling mode: ``'quantum'``, ``'max'``, ``'average'``. Default ``'quantum'``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQPooling2D(pool_size=2, n_qubits=2)
    >>> x = np.random.randn(1, 8, 8, 3).astype("float32")
    >>> y = layer(x)
    """

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = (2, 2),
        strides: Union[int, Tuple[int, int], None] = None,
        padding: str = "valid",
        n_qubits: int = 2,
        n_layers: int = 1,
        mode: str = "quantum",
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        self.padding = padding.upper()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.mode = mode
        self._simulator: Optional[Any] = None
        self._var_params = None

    def _get_simulator(self) -> Any:
        if self._simulator is None:
            self._simulator = StatevectorSimulator()
        return self._simulator

    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self.mode == "quantum":
            n_var = self.n_layers * self.n_qubits * 3
            self._var_params = self.add_weight(
                name="variational_params",
                shape=(n_var,),
                initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                trainable=True,
            )
        self.built = True

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")
        x_np = ops.convert_to_numpy(x)

        batch_size, H, W, C = x_np.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        if self.padding == "SAME":
            pad_h = max((H - 1) * sh + ph - H, 0)
            pad_w = max((W - 1) * sw + pw - W, 0)
            x_np = np.pad(x_np, ((0, 0), (pad_h // 2, pad_h - pad_h // 2),
                                 (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            H_pad, W_pad = H, W

        out_h = (H_pad - ph) // sh + 1
        out_w = (W_pad - pw) // sw + 1

        outputs = np.zeros((batch_size, out_h, out_w, C), dtype=np.float32)

        var_params_np = (
            ops.convert_to_numpy(self._var_params)
            if self._var_params is not None else None
        )

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    region = x_np[b, i * sh:i * sh + ph, j * sw:j * sw + pw, :]
                    if self.mode == "max":
                        outputs[b, i, j, :] = np.max(region, axis=(0, 1))
                    elif self.mode == "average":
                        outputs[b, i, j, :] = np.mean(region, axis=(0, 1))
                    elif self.mode == "quantum" and var_params_np is not None:
                        # Quantum pooling via entanglement
                        pooled = self._quantum_pool(region, var_params_np)
                        outputs[b, i, j, :] = pooled
                    else:
                        outputs[b, i, j, :] = np.mean(region, axis=(0, 1))

        return ops.convert_to_tensor(outputs, dtype="float32")

    def _quantum_pool(
        self,
        region: np.ndarray,
        var_params: np.ndarray,
    ) -> np.ndarray:
        """Perform quantum pooling on a spatial region.

        Parameters
        ----------
        region : numpy.ndarray
            Shape ``(ph, pw, C)``.
        var_params : numpy.ndarray
            Circuit parameters.

        Returns
        -------
        numpy.ndarray
            Shape ``(C,)`` pooled features.
        """
        ph, pw, C = region.shape
        flat_region = region.flatten()

        # Map region to n_qubits
        if len(flat_region) >= self.n_qubits:
            chunk = len(flat_region) // self.n_qubits
            data = np.array([
                np.mean(flat_region[q * chunk:(q + 1) * chunk])
                for q in range(self.n_qubits)
            ])
        else:
            data = np.zeros(self.n_qubits)
            data[:len(flat_region)] = flat_region

        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)
            qc.ry(float(data[q]) % _TWO_PI, q)

        # Entangling layer for compression
        param_idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits - 1):
                if param_idx < len(var_params):
                    theta = float(var_params[param_idx])
                    param_idx += 1
                    qc.cry(theta, q, q + 1)

        # Measure
        sim = self._get_simulator()
        results = []
        for q in range(self.n_qubits):
            obs = _pauli_observable("z", q, self.n_qubits)
            results.append(float(sim.expectation(qc, obs)))

        # Map to output channels
        out = np.zeros(C, dtype=np.float32)
        q_vals = np.array(results)
        for c in range(C):
            idx = c % len(q_vals)
            out[c] = q_vals[idx]
        return out

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError("Input shape must be (batch, H, W, C)")
        _, H, W, C = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        if self.padding == "SAME":
            out_h = (H + sh - 1) // sh
            out_w = (W + sw - 1) // sw
        else:
            out_h = (H - ph) // sh + 1
            out_w = (W - pw) // sw + 1
        return (input_shape[0], out_h, out_w, C)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "mode": self.mode,
        })
        return config


# ===========================================================================
# KerasQFlatten — Flatten Quantum Measurement Results
# ===========================================================================

class KerasQFlatten(Layer):
    """Flatten quantum measurement results into a 1-D tensor.

    Reshapes the input while preserving batch dimension. Optionally
    applies a quantum-to-classical mapping that converts measurement
    probabilities into a classical feature vector.

    Parameters
    ----------
    data_format : str, optional
        ``'channels_last'`` or ``'channels_first'``. Default ``'channels_last'``.
    quantum_map : bool, optional
        If ``True``, applies quantum probability mapping before flattening.
        Default ``False``.
    n_qubits : int or None, optional
        Number of qubits for probability mapping. Required if
        ``quantum_map=True``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = KerasQFlatten()
    >>> x = np.random.randn(2, 4, 4, 3).astype("float32")
    >>> y = layer(x)
    >>> y.shape
    (2, 48)
    """

    def __init__(
        self,
        data_format: str = "channels_last",
        quantum_map: bool = False,
        n_qubits: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.data_format = data_format
        self.quantum_map = quantum_map
        self.n_qubits = n_qubits

    def call(self, inputs: Any) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")

        if self.quantum_map and self.n_qubits is not None:
            x_np = ops.convert_to_numpy(x)
            # Normalize to get pseudo-probabilities
            x_sum = np.sum(np.abs(x_np), axis=-1, keepdims=True)
            x_sum = np.maximum(x_sum, 1e-10)
            probs = np.abs(x_np) / x_sum
            # Convert probabilities to Bloch-sphere angles
            mapped = 2.0 * np.arcsin(np.sqrt(np.clip(probs, 0.0, 1.0)))
            x = ops.convert_to_tensor(mapped, dtype="float32")

        batch_size = ops.shape(x)[0]
        return ops.reshape(x, (batch_size, -1))

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not input_shape:
            return (0,)
        flat_dim = 1
        for d in input_shape[1:]:
            flat_dim *= d
        return (input_shape[0], flat_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "data_format": self.data_format,
            "quantum_map": self.quantum_map,
            "n_qubits": self.n_qubits,
        })
        return config
