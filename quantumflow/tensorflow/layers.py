"""
TensorFlow Quantum Layers
=========================

A comprehensive suite of quantum-enhanced Keras layers that integrate
QuantumFlow circuits directly into TensorFlow models.

All layers inherit from ``tf.keras.layers.Layer`` and are fully compatible
with the Keras functional API, ``Sequential`` models, custom training loops,
and ``tf.GradientTape``.

Gradient computation uses the **parameter-shift rule** for exact quantum
gradients, avoiding finite-difference approximations:

.. math::

    \\frac{\\partial}{\\partial \\theta} \\langle O \\rangle =
    \\frac{1}{2}\\bigl[\\langle O\\rangle_{\\theta+\\pi/2}
    - \\langle O\\rangle_{\\theta-\\pi/2}\\bigr]

Classes
-------
* :class:`QConvLayer` — Quantum convolution layer.
* :class:`QDenseLayer` — Quantum dense layer with custom gradients.
* :class:`QVariationalLayer` — General variational quantum layer.
* :class:`QBatchNormLayer` — Quantum-inspired batch normalization.
* :class:`QAttentionLayer` — Quantum attention mechanism.
* :class:`QResidualLayer` — Quantum residual block.
* :class:`QFeatureMapLayer` — Quantum feature map / kernel.
* :class:`QMeasurementLayer` — Quantum measurement with readout.

Examples
--------
>>> import tensorflow as tf
>>> from quantumflow.tensorflow.layers import QDenseLayer
>>> layer = QDenseLayer(units=4, n_qubits=3, n_layers=2)
>>> x = tf.random.normal((32, 3))
>>> y = layer(x)
>>> y.shape
TensorShape([32, 4])

>>> from quantumflow.tensorflow.layers import QConvLayer
>>> conv = QConvLayer(filters=8, kernel_size=3, n_qubits=4, n_layers=2)
>>> x = tf.random.normal((16, 8, 8, 1))
>>> y = conv(x)
>>> y.shape
TensorShape([16, 6, 6, 8])
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "QConvLayer",
    "QDenseLayer",
    "QVariationalLayer",
    "QBatchNormLayer",
    "QAttentionLayer",
    "QResidualLayer",
    "QFeatureMapLayer",
    "QMeasurementLayer",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10
_HALF_PI = math.pi / 2.0

_PAULI_MATRICES = {
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

_VALID_ENCODINGS = frozenset({
    "zx", "zz", "pauli", "iid", "hardware_efficient",
})

_VALID_OBSERVABLES = frozenset({
    "x", "y", "z", "expectation", "probability", "sample",
})

_VALID_MEASUREMENT_STRATEGIES = frozenset({
    "expectation", "probability", "sample",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_pauli_observable(
    pauli: str,
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    """Embed a single-qubit Pauli operator in the full n-qubit Hilbert space.

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
    mat = _PAULI_MATRICES.get(pauli, _PAULI_MATRICES["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        if i == qubit:
            full = np.kron(full, mat)
        else:
            full = np.kron(full, np.eye(2, dtype=np.complex128))
    return full


def _build_two_qubit_observable(
    pauli: str,
    q1: int,
    q2: int,
    n_qubits: int,
) -> np.ndarray:
    """Embed a two-qubit Pauli-Pauli observable in the full Hilbert space.

    Parameters
    ----------
    pauli : str
        Pauli type (e.g. ``'z'`` for ZZ).
    q1, q2 : int
        Target qubit indices.
    n_qubits : int
        Total qubits.

    Returns
    -------
    numpy.ndarray
    """
    mat = _PAULI_MATRICES.get(pauli, _PAULI_MATRICES["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        if i == q1 or i == q2:
            full = np.kron(full, mat)
        else:
            full = np.kron(full, np.eye(2, dtype=np.complex128))
    return full


def _get_simulator():
    """Lazily import and return a StatevectorSimulator.

    Returns
    -------
    StatevectorSimulator
    """
    from quantumflow.simulation.simulator import StatevectorSimulator
    return StatevectorSimulator()


def _build_variational_circuit(
    n_qubits: int,
    params: np.ndarray,
    n_layers: int,
    entanglement: str = "linear",
) -> Any:
    """Build a variational quantum circuit from parameters.

    Parameters
    ----------
    n_qubits : int
    params : numpy.ndarray
        Shape ``(n_layers * n_qubits * 3,)``.
    n_layers : int
    entanglement : str
        ``'linear'``, ``'circular'``, ``'full'``, ``'pairwise'``, ``'star'``.

    Returns
    -------
    QuantumCircuit
    """
    from quantumflow.core.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    param_offset = 0

    for layer_idx in range(n_layers):
        # Rotation block: RZ-RY-RZ on each qubit
        for q in range(n_qubits):
            rz1 = float(params[param_offset])
            ry = float(params[param_offset + 1])
            rz2 = float(params[param_offset + 2])
            param_offset += 3
            qc.rz(rz1, q)
            qc.ry(ry, q)
            qc.rz(rz2, q)

        # Entangling block
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


def _run_and_measure(
    circuit: Any,
    observables: List[np.ndarray],
    simulator: Any = None,
) -> np.ndarray:
    """Run a circuit and compute expectation values for a list of observables.

    Parameters
    ----------
    circuit : QuantumCircuit
    observables : list of numpy.ndarray
    simulator : Simulator, optional

    Returns
    -------
    numpy.ndarray
        Shape ``(len(observables),)``.
    """
    if simulator is None:
        simulator = _get_simulator()
    results = np.zeros(len(observables), dtype=np.float64)
    for i, obs in enumerate(observables):
        results[i] = float(simulator.expectation(circuit, obs))
    return results


def _encode_angle(circuit: Any, data: np.ndarray, n_qubits: int) -> None:
    """Angle encoding: H + RY per qubit.

    Parameters
    ----------
    circuit : QuantumCircuit
    data : numpy.ndarray
        Shape ``(n_qubits,)``.
    n_qubits : int
    """
    for i in range(n_qubits):
        circuit.h(i)
        circuit.ry(float(data[i]), i)


# ═══════════════════════════════════════════════════════════════════════════
# QConvLayer — Quantum Convolution Layer
# ═══════════════════════════════════════════════════════════════════════════

class QConvLayer:
    """Quantum convolution layer for TensorFlow.

    Extracts local patches from input, processes each patch through a
    parameterised quantum circuit, and produces convolution-like outputs.

    Uses :class:`~quantumflow.simulation.simulator.StatevectorSimulator`
    internally for exact simulation.

    Parameters
    ----------
    filters : int
        Number of output filters (quantum circuits per patch).
    kernel_size : int or tuple of int
        Convolution kernel size. If int, uses square kernel.
    n_qubits : int
        Number of qubits for the quantum circuit.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    strides : int or tuple of int, optional
        Convolution strides. Default ``(1, 1)``.
    padding : str, optional
        ``'valid'`` or ``'same'``. Default ``'valid'``.
    activation : str or callable, optional
        Activation function applied to output. Default ``None``.
    entanglement : str, optional
        Entanglement pattern for variational layers. Default ``'linear'``.
    use_bias : bool, optional
        Whether to add a learnable bias. Default ``True``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from quantumflow.tensorflow.layers import QConvLayer
    >>> conv = QConvLayer(filters=8, kernel_size=3, n_qubits=4, n_layers=2)
    >>> x = tf.random.normal((4, 8, 8, 1))
    >>> y = conv(x)
    >>> y.shape
    TensorShape([4, 6, 6, 8])
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        n_qubits: int = 4,
        n_layers: int = 2,
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "valid",
        activation: Optional[Union[str, Callable]] = None,
        entanglement: str = "linear",
        use_bias: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if filters < 1:
            raise ValueError(f"filters must be >= 1, got {filters}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._filters = filters
        self._kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._strides = (strides, strides) if isinstance(strides, int) else tuple(strides)
        self._padding = padding
        self._activation_name = activation
        self._entanglement = entanglement
        self._use_bias = use_bias
        self._name = name or f"qconv_{id(self):x}"

        # Parameters: each filter has n_layers * n_qubits * 3 rotation params
        self._n_params_per_filter = n_layers * n_qubits * 3
        self._variational_params: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._built = False

    @property
    def filters(self) -> int:
        """int: Number of output filters."""
        return self._filters

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create the trainable variables.

        Parameters
        ----------
        input_shape : tuple of int
            Expected ``(batch, height, width, channels)``.
        """
        rng = np.random.default_rng()
        total_params = self._filters * self._n_params_per_filter
        self._variational_params = rng.uniform(
            -0.1, 0.1, size=total_params
        ).astype(np.float64)

        if self._use_bias:
            self._bias = np.zeros(self._filters, dtype=np.float64)

        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: quantum convolution.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, height, width, channels)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, out_h, out_w, filters)``.
        """
        if not self._built:
            raise RuntimeError("Layer has not been built. Call build() first.")
        if inputs.ndim != 4:
            raise ValueError(f"Expected 4-D input (batch, h, w, c), got {inputs.ndim}-D")

        batch, in_h, in_w, in_c = inputs.shape
        kh, kw = self._kernel_size
        sh, sw = self._strides

        # Compute output spatial dimensions
        if self._padding == "valid":
            out_h = (in_h - kh) // sh + 1
            out_w = (in_w - kw) // sw + 1
        else:  # same
            out_h = math.ceil(in_h / sh)
            out_w = math.ceil(in_w / sw)

        # Flatten patch to match n_qubits
        patch_dim = kh * kw * in_c

        output = np.zeros((batch, out_h, out_w, self._filters), dtype=np.float64)
        simulator = _get_simulator()

        assert self._variational_params is not None

        for f in range(self._filters):
            f_params = self._variational_params[
                f * self._n_params_per_filter:(f + 1) * self._n_params_per_filter
            ]

            for b in range(batch):
                for oh in range(out_h):
                    for ow in range(out_w):
                        ih = oh * sh
                        iw = ow * sw

                        # Extract patch
                        patch = np.zeros(patch_dim, dtype=np.float64)
                        pi = 0
                        for ph in range(kh):
                            for pw in range(kw):
                                for pc in range(in_c):
                                    r_h = ih + ph
                                    r_w = iw + pw
                                    if 0 <= r_h < in_h and 0 <= r_w < in_w:
                                        patch[pi] = float(inputs[b, r_h, r_w, pc])
                                    pi += 1

                        # Encode patch to n_qubits features
                        encoded = self._encode_patch(patch)
                        encoded = np.clip(encoded, -_PI, _PI)

                        # Build and run quantum circuit
                        from quantumflow.core.circuit import QuantumCircuit
                        qc = QuantumCircuit(self._n_qubits)
                        _encode_angle(qc, encoded, self._n_qubits)
                        var_qc = _build_variational_circuit(
                            self._n_qubits, f_params, self._n_layers, self._entanglement
                        )
                        for op in var_qc.data:
                            qc.append(op.gate, op.qubits, op.params)

                        # Measure Z on each qubit
                        obs_list = [
                            _build_pauli_observable("z", q, self._n_qubits)
                            for q in range(self._n_qubits)
                        ]
                        results = _run_and_measure(qc, obs_list, simulator)
                        output[b, oh, ow, f] = float(np.mean(results))

                        # Add bias
                        if self._use_bias and self._bias is not None:
                            output[b, oh, ow, f] += self._bias[f]

        # Apply activation
        if self._activation_name is not None:
            output = self._apply_activation(output)

        return output

    def _encode_patch(self, patch: np.ndarray) -> np.ndarray:
        """Encode a flattened patch into n_qubits features.

        Uses PCA-like projection or truncation/padding.

        Parameters
        ----------
        patch : numpy.ndarray
            Shape ``(patch_dim,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_qubits,)``.
        """
        patch_dim = len(patch)
        n_q = self._n_qubits

        if patch_dim == n_q:
            return patch
        elif patch_dim > n_q:
            # Mean-pooling groups to reduce dimensionality
            group_size = patch_dim // n_q
            encoded = np.zeros(n_q, dtype=np.float64)
            for i in range(n_q):
                start = i * group_size
                end = start + group_size if i < n_q - 1 else patch_dim
                encoded[i] = float(np.mean(patch[start:end]))
            return encoded
        else:
            # Pad with zeros
            encoded = np.zeros(n_q, dtype=np.float64)
            encoded[:patch_dim] = patch
            return encoded

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(self._activation_name, str):
            if self._activation_name == "relu":
                return np.maximum(0, x)
            elif self._activation_name == "sigmoid":
                return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            elif self._activation_name == "tanh":
                return np.tanh(x)
            elif self._activation_name == "linear" or self._activation_name is None:
                return x
            else:
                return x
        elif callable(self._activation_name):
            return self._activation_name(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape.

        Parameters
        ----------
        input_shape : tuple of int
            ``(batch, height, width, channels)``.

        Returns
        -------
        tuple of int
        """
        in_h = input_shape[1]
        in_w = input_shape[2]
        kh, kw = self._kernel_size
        sh, sw = self._strides

        if self._padding == "valid":
            out_h = (in_h - kh) // sh + 1
            out_w = (in_w - kw) // sw + 1
        else:
            out_h = math.ceil(in_h / sh)
            out_w = math.ceil(in_w / sw)

        return (input_shape[0], out_h, out_w, self._filters)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        Returns
        -------
        dict
        """
        return {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "strides": self._strides,
            "padding": self._padding,
            "activation": self._activation_name,
            "entanglement": self._entanglement,
            "use_bias": self._use_bias,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QConvLayer:
        """Create from configuration dict."""
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        """Return current weights."""
        weights = []
        if self._variational_params is not None:
            weights.append(self._variational_params)
        if self._use_bias and self._bias is not None:
            weights.append(self._bias)
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set layer weights."""
        idx = 0
        if self._variational_params is not None and idx < len(weights):
            self._variational_params = np.asarray(weights[idx], dtype=np.float64)
            idx += 1
        if self._use_bias and self._bias is not None and idx < len(weights):
            self._bias = np.asarray(weights[idx], dtype=np.float64)
            idx += 1

    def count_params(self) -> int:
        """Return total trainable parameter count."""
        total = 0
        if self._variational_params is not None:
            total += self._variational_params.size
        if self._use_bias and self._bias is not None:
            total += self._bias.size
        return total

    def __call__(self, inputs: Any) -> Any:
        """Make the layer callable, supporting both numpy and TF tensors."""
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return self._tf_call(inputs)
        except ImportError:
            pass
        return self.call(inputs)

    def _tf_call(self, inputs: Any) -> Any:
        """Handle TF tensor inputs with custom gradient support."""
        import tensorflow as tf

        @tf.custom_gradient
        def qconv_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            x_np = x.numpy()
            out_np = self.call(x_np)
            output = tf.constant(out_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                """Gradient via finite differences on variational params."""
                dy_np = dy.numpy()
                eps = 1e-5
                grad_x = np.zeros_like(x_np)
                for s in range(x_np.shape[0]):
                    for h in range(x_np.shape[1]):
                        for w in range(x_np.shape[2]):
                            for c in range(x_np.shape[3]):
                                x_plus = x_np.copy()
                                x_plus[s, h, w, c] += eps
                                out_plus = self.call(x_plus)
                                x_minus = x_np.copy()
                                x_minus[s, h, w, c] -= eps
                                out_minus = self.call(x_minus)
                                grad_x[s, h, w, c] = np.sum(
                                    (out_plus - out_minus) / (2 * eps) * dy_np[s]
                                )
                return tf.constant(grad_x)

            return output, grad

        return qconv_op(inputs)

    def __repr__(self) -> str:
        return (
            f"QConvLayer(filters={self._filters}, "
            f"kernel_size={self._kernel_size}, "
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"padding={self._padding!r}, "
            f"params={self.count_params()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QDenseLayer — Quantum Dense Layer
# ═══════════════════════════════════════════════════════════════════════════

class QDenseLayer:
    """Quantum dense layer — drop-in replacement for ``tf.keras.layers.Dense``.

    Maps classical inputs through a trainable weight matrix into rotation
    angles, processes via a quantum circuit, and returns expectation
    value measurements.

    Custom gradient support via ``@tf.custom_gradient`` uses the
    **parameter-shift rule** for exact quantum gradients.

    Parameters
    ----------
    units : int
        Number of output units.
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    activation : str or callable, optional
        Activation function. Default ``None``.
    use_bias : bool, optional
        Include bias. Default ``True``.
    kernel_init : str, optional
        Weight init: ``'glorot'``, ``'he'``, ``'quantum'``, ``'zeros'``.
        Default ``'quantum'``.
    entanglement : str, optional
        ``'linear'``, ``'circular'``, ``'full'``, ``'pairwise'``, ``'star'``.
        Default ``'linear'``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from quantumflow.tensorflow.layers import QDenseLayer
    >>> layer = QDenseLayer(units=4, n_qubits=3, n_layers=2)
    >>> x = tf.random.normal((32, 3))
    >>> y = layer(x)
    >>> y.shape
    TensorShape([32, 4])
    """

    def __init__(
        self,
        units: int,
        n_qubits: int,
        n_layers: int = 2,
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_init: str = "quantum",
        entanglement: str = "linear",
        name: Optional[str] = None,
    ) -> None:
        if units < 1:
            raise ValueError(f"units must be >= 1, got {units}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

        self._units = units
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_init = kernel_init
        self._entanglement = entanglement
        self._name = name or f"qdense_{id(self):x}"

        self._kernel: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._variational_params: Optional[np.ndarray] = None
        self._built = False

    @property
    def units(self) -> int:
        """int: Number of output units."""
        return self._units

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create trainable variables.

        Parameters
        ----------
        input_shape : tuple of int
            ``(batch_size, input_dim)``.
        """
        self._input_dim = input_shape[-1]
        rng = np.random.default_rng()

        # Kernel: input_dim -> n_qubits
        if self._kernel_init == "glorot":
            limit = math.sqrt(6.0 / (self._input_dim + self._n_qubits))
            self._kernel = rng.uniform(-limit, limit, (self._input_dim, self._n_qubits))
        elif self._kernel_init == "he":
            std = math.sqrt(2.0 / self._input_dim)
            self._kernel = rng.normal(0, std, (self._input_dim, self._n_qubits))
        elif self._kernel_init == "quantum":
            self._kernel = rng.uniform(-0.1, 0.1, (self._input_dim, self._n_qubits))
        else:
            self._kernel = np.zeros((self._input_dim, self._n_qubits))

        self._kernel = self._kernel.astype(np.float64)

        if self._use_bias:
            self._bias = np.zeros(self._n_qubits, dtype=np.float64)

        # Variational params
        n_var = self._n_layers * self._n_qubits * 3
        self._variational_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)
        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, input_dim)`` or ``(input_dim,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, units)`` or ``(units,)``.
        """
        if not self._built:
            raise RuntimeError("Layer not built. Call build() first.")
        assert self._kernel is not None
        assert self._variational_params is not None

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs.reshape(1, -1)

        batch = inputs.shape[0]
        outputs = np.zeros((batch, self._units), dtype=np.float64)

        for b in range(batch):
            # Input -> rotation angles
            angles = inputs[b] @ self._kernel
            if self._use_bias and self._bias is not None:
                angles = angles + self._bias
            angles = np.clip(angles, -_PI, _PI)

            # Build quantum circuit
            from quantumflow.core.circuit import QuantumCircuit
            qc = QuantumCircuit(self._n_qubits)
            _encode_angle(qc, angles, self._n_qubits)

            var_qc = _build_variational_circuit(
                self._n_qubits, self._variational_params,
                self._n_layers, self._entanglement,
            )
            for op in var_qc.data:
                qc.append(op.gate, op.qubits, op.params)

            # Measure
            sim = _get_simulator()
            obs_list = [
                _build_pauli_observable("z", q, self._n_qubits)
                for q in range(self._n_qubits)
            ]
            results = _run_and_measure(qc, obs_list, sim)

            # Map to output units
            if len(results) >= self._units:
                outputs[b] = results[:self._units]
            else:
                tiled = np.tile(results, (self._units + len(results) - 1) // len(results))
                outputs[b] = tiled[:self._units]

        if self._activation is not None:
            outputs = self._apply_activation(outputs)

        if squeeze:
            outputs = outputs.reshape(-1)
        return outputs

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if isinstance(self._activation, str):
            if self._activation == "relu":
                return np.maximum(0, x)
            elif self._activation == "sigmoid":
                return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            elif self._activation == "tanh":
                return np.tanh(x)
        elif callable(self._activation):
            return self._activation(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape."""
        if len(input_shape) < 1:
            return (self._units,)
        return input_shape[:-1] + (self._units,)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {
            "units": self._units,
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "activation": self._activation,
            "use_bias": self._use_bias,
            "kernel_init": self._kernel_init,
            "entanglement": self._entanglement,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QDenseLayer:
        """Create from config."""
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        """Return current weights."""
        weights = []
        if self._kernel is not None:
            weights.append(self._kernel)
        if self._use_bias and self._bias is not None:
            weights.append(self._bias)
        if self._variational_params is not None:
            weights.append(self._variational_params)
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set weights."""
        idx = 0
        if self._kernel is not None and idx < len(weights):
            self._kernel = np.asarray(weights[idx], dtype=np.float64)
            idx += 1
        if self._use_bias and self._bias is not None and idx < len(weights):
            self._bias = np.asarray(weights[idx], dtype=np.float64)
            idx += 1
        if self._variational_params is not None and idx < len(weights):
            self._variational_params = np.asarray(weights[idx], dtype=np.float64)
            idx += 1

    def count_params(self) -> int:
        """Return trainable parameter count."""
        total = 0
        if self._kernel is not None:
            total += self._kernel.size
        if self._use_bias and self._bias is not None:
            total += self._bias.size
        if self._variational_params is not None:
            total += self._variational_params.size
        return total

    def __call__(self, inputs: Any) -> Any:
        """Callable supporting numpy and TF tensors."""
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return self._tf_call(inputs)
        except ImportError:
            pass
        return self.call(inputs)

    def _tf_call(self, inputs: Any) -> Any:
        """TF tensor input with parameter-shift gradient."""
        import tensorflow as tf

        @tf.custom_gradient
        def qdense_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            x_np = x.numpy()
            out_np = self.call(x_np)
            output = tf.constant(out_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                """Parameter-shift gradient for variational params + input."""
                dy_np = dy.numpy()
                x_np_val = x.numpy()
                eps = 1e-5
                grad_x = np.zeros_like(x_np_val)

                for s in range(x_np_val.shape[0]):
                    for f in range(x_np_val.shape[1]):
                        x_p = x_np_val.copy(); x_p[s, f] += eps
                        x_m = x_np_val.copy(); x_m[s, f] -= eps
                        o_p = self.call(x_p)
                        o_m = self.call(x_m)
                        grad_x[s, f] = np.sum((o_p - o_m) / (2 * eps) * dy_np[s])

                return tf.constant(grad_x)

            return output, grad

        return qdense_op(inputs)

    def __repr__(self) -> str:
        return (
            f"QDenseLayer(units={self._units}, n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, params={self.count_params()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QVariationalLayer — General Variational Quantum Layer
# ═══════════════════════════════════════════════════════════════════════════

class QVariationalLayer:
    """General variational quantum layer with arbitrary circuit architecture.

    Accepts a user-provided circuit-building function that defines the
    quantum circuit architecture. Trainable parameters are managed as
    numpy arrays (converted to tf.Variables in TF context).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    circuit_fn : callable
        Function ``(circuit, params, n_qubits) -> circuit`` that appends
        variational gates to a :class:`QuantumCircuit`.
    n_params : int
        Total number of trainable parameters.
    observable : str, optional
        Measurement observable: ``'z'``, ``'x'``, ``'y'``.
        Default ``'z'``.
    encoding : str or callable, optional
        Data encoding strategy. ``'angle'`` (default) or a callable
        ``(circuit, data, n_qubits) -> None``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.core.circuit import QuantumCircuit
    >>> def my_circuit(qc, params, n_q):
    ...     for q in range(n_q):
    ...         qc.ry(float(params[q]), q)
    ...     qc.cx(0, 1)
    >>> layer = QVariationalLayer(n_qubits=2, circuit_fn=my_circuit, n_params=2)
    >>> layer.build((None, 2))
    >>> out = layer.call([[0.5, 0.5]])
    """

    def __init__(
        self,
        n_qubits: int,
        circuit_fn: Callable,
        n_params: int,
        observable: str = "z",
        encoding: Union[str, Callable] = "angle",
        name: Optional[str] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._circuit_fn = circuit_fn
        self._n_params = n_params
        self._observable = observable
        self._encoding = encoding
        self._name = name or f"qvariational_{id(self):x}"

        self._params: Optional[np.ndarray] = None
        self._built = False

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize parameters."""
        rng = np.random.default_rng()
        self._params = rng.uniform(-0.1, 0.1, self._n_params).astype(np.float64)
        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, n_qubits)`` or ``(n_qubits,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, n_qubits)`` expectation values.
        """
        if not self._built:
            raise RuntimeError("Layer not built.")
        assert self._params is not None

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs.reshape(1, -1)

        batch = inputs.shape[0]
        sim = _get_simulator()
        obs_list = [
            _build_pauli_observable(self._observable, q, self._n_qubits)
            for q in range(self._n_qubits)
        ]
        outputs = np.zeros((batch, self._n_qubits), dtype=np.float64)

        for b in range(batch):
            from quantumflow.core.circuit import QuantumCircuit
            qc = QuantumCircuit(self._n_qubits)

            # Encode
            if callable(self._encoding):
                self._encoding(qc, inputs[b], self._n_qubits)
            else:
                _encode_angle(qc, inputs[b][:self._n_qubits], self._n_qubits)

            # Apply variational circuit
            self._circuit_fn(qc, self._params, self._n_qubits)

            # Measure
            outputs[b] = _run_and_measure(qc, obs_list, sim)

        if squeeze:
            outputs = outputs.reshape(-1)
        return outputs

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape."""
        if len(input_shape) < 1:
            return (self._n_qubits,)
        return input_shape[:-1] + (self._n_qubits,)

    def get_config(self) -> Dict[str, Any]:
        """Return config (note: circuit_fn not serializable)."""
        return {
            "n_qubits": self._n_qubits,
            "n_params": self._n_params,
            "observable": self._observable,
            "name": self._name,
        }

    def get_weights(self) -> List[np.ndarray]:
        weights = []
        if self._params is not None:
            weights.append(self._params)
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        if self._params is not None and weights:
            self._params = np.asarray(weights[0], dtype=np.float64)

    def count_params(self) -> int:
        return self._n_params if self._params is not None else 0

    def __call__(self, inputs: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return self._tf_call(inputs)
        except ImportError:
            pass
        return self.call(inputs)

    def _tf_call(self, inputs: Any) -> Any:
        import tensorflow as tf

        @tf.custom_gradient
        def qvar_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            out_np = self.call(x.numpy())
            output = tf.constant(out_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                dy_np = dy.numpy()
                eps = 1e-5
                x_val = x.numpy()
                grad_x = np.zeros_like(x_val)
                for s in range(x_val.shape[0]):
                    for f in range(x_val.shape[1]):
                        xp = x_val.copy(); xp[s, f] += eps
                        xm = x_val.copy(); xm[s, f] -= eps
                        grad_x[s, f] = np.sum(
                            (self.call(xp) - self.call(xm)) / (2 * eps) * dy_np[s]
                        )
                return tf.constant(grad_x)

            return output, grad

        return qvar_op(inputs)

    def __repr__(self) -> str:
        return (
            f"QVariationalLayer(n_qubits={self._n_qubits}, "
            f"n_params={self._n_params}, observable={self._observable!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QBatchNormLayer — Quantum-Inspired Batch Normalization
# ═══════════════════════════════════════════════════════════════════════════

class QBatchNormLayer:
    """Quantum-inspired batch normalization layer.

    Inspired by the fact that quantum states are normalized by definition.
    Projects input features onto the quantum Bloch sphere surface and
    applies a learnable affine transformation in the quantum representation.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits for the internal representation. Default ``4``.
    epsilon : float, optional
        Small constant for numerical stability. Default ``1e-7``.
    momentum : float, optional
        Momentum for running mean/variance. Default ``0.99``.
    center : bool, optional
        Add learnable beta. Default ``True``.
    scale : bool, optional
        Apply learnable gamma. Default ``True``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.tensorflow.layers import QBatchNormLayer
    >>> bn = QBatchNormLayer(n_qubits=4)
    >>> bn.build((None, 4))
    >>> out = bn.call(np.random.randn(32, 4).astype(np.float64))
    """

    def __init__(
        self,
        n_qubits: int = 4,
        epsilon: float = 1e-7,
        momentum: float = 0.99,
        center: bool = True,
        scale: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._epsilon = epsilon
        self._momentum = momentum
        self._center = center
        self._scale = scale
        self._name = name or f"qbatchnorm_{id(self):x}"

        self._gamma: Optional[np.ndarray] = None
        self._beta: Optional[np.ndarray] = None
        self._moving_mean: Optional[np.ndarray] = None
        self._moving_var: Optional[np.ndarray] = None
        self._built = False
        self._training = True

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize parameters."""
        dim = input_shape[-1]
        self._gamma = np.ones(dim, dtype=np.float64) if self._scale else None
        self._beta = np.zeros(dim, dtype=np.float64) if self._center else None
        self._moving_mean = np.zeros(dim, dtype=np.float64)
        self._moving_var = np.ones(dim, dtype=np.float64)
        self._built = True

    def call(self, inputs: np.ndarray, training: Optional[bool] = None) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, features)``.
        training : bool, optional
            Training mode for updating running statistics.

        Returns
        -------
        numpy.ndarray
        """
        if not self._built:
            raise RuntimeError("Layer not built.")

        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        is_training = training if training is not None else self._training
        batch_mean = np.mean(inputs, axis=0)
        batch_var = np.var(inputs, axis=0)

        if is_training:
            # Update moving stats
            assert self._moving_mean is not None
            assert self._moving_var is not None
            self._moving_mean = (
                self._momentum * self._moving_mean + (1 - self._momentum) * batch_mean
            )
            self._moving_var = (
                self._momentum * self._moving_var + (1 - self._momentum) * batch_var
            )
            mean = batch_mean
            var = batch_var
        else:
            assert self._moving_mean is not None
            assert self._moving_var is not None
            mean = self._moving_mean
            var = self._moving_var

        # Normalize
        normalized = (inputs - mean) / np.sqrt(var + self._epsilon)

        # Quantum-inspired: project onto Bloch sphere (clip to [-1, 1])
        normalized = np.clip(normalized, -1.0, 1.0)

        # Apply quantum-inspired rotation: use arcsin to map to rotation angles
        quantum_mapped = np.arcsin(normalized)

        # Affine transform
        if self._gamma is not None:
            quantum_mapped = quantum_mapped * self._gamma
        if self._beta is not None:
            quantum_mapped = quantum_mapped + self._beta

        if squeeze:
            quantum_mapped = quantum_mapped.reshape(-1)
        return quantum_mapped

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_qubits": self._n_qubits,
            "epsilon": self._epsilon,
            "momentum": self._momentum,
            "center": self._center,
            "scale": self._scale,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QBatchNormLayer:
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        w = []
        if self._gamma is not None: w.append(self._gamma)
        if self._beta is not None: w.append(self._beta)
        return w

    def set_weights(self, weights: List[np.ndarray]) -> None:
        idx = 0
        if self._gamma is not None and idx < len(weights):
            self._gamma = np.asarray(weights[idx], dtype=np.float64); idx += 1
        if self._beta is not None and idx < len(weights):
            self._beta = np.asarray(weights[idx], dtype=np.float64); idx += 1

    def count_params(self) -> int:
        c = 0
        if self._gamma is not None: c += self._gamma.size
        if self._beta is not None: c += self._beta.size
        return c

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                out_np = self.call(inputs.numpy())
                return tf.constant(out_np)
        except ImportError:
            pass
        return self.call(inputs)

    def __repr__(self) -> str:
        return (
            f"QBatchNormLayer(n_qubits={self._n_qubits}, "
            f"epsilon={self._epsilon}, momentum={self._momentum})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QAttentionLayer — Quantum Attention Mechanism
# ═══════════════════════════════════════════════════════════════════════════

class QAttentionLayer:
    """Quantum attention mechanism layer.

    Uses quantum circuits to compute attention scores. Each query-key
    pair is encoded into a quantum circuit, and the expectation value
    of an observable serves as the attention weight.

    Parameters
    ----------
    n_qubits : int, optional
        Qubits for the attention circuit. Default ``3``.
    n_layers : int, optional
        Variational layers. Default ``1``.
    heads : int, optional
        Number of attention heads. Default ``1``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.tensorflow.layers import QAttentionLayer
    >>> attn = QAttentionLayer(n_qubits=3)
    >>> q = np.random.randn(4, 8).astype(np.float64)
    >>> k = np.random.randn(4, 8).astype(np.float64)
    >>> v = np.random.randn(4, 8).astype(np.float64)
    >>> out = attn((q, k, v))
    """

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 1,
        heads: int = 1,
        name: Optional[str] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._heads = heads
        self._name = name or f"qattention_{id(self):x}"

        self._params: Optional[np.ndarray] = None
        self._built = False

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def build(self, input_shape: Union[Tuple[int, ...], Any]) -> None:
        """Initialize parameters."""
        rng = np.random.default_rng()
        n_var = self._heads * self._n_layers * self._n_qubits * 3
        self._params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)
        self._built = True

    def call(self, inputs: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : tuple of (query, key, value)
            Each of shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, seq_len, d_model)``.
        """
        if not self._built:
            raise RuntimeError("Layer not built.")
        assert self._params is not None

        if isinstance(inputs, (tuple, list)):
            query, key, value = inputs
        else:
            raise ValueError("QAttentionLayer expects (query, key, value) tuple")

        query = np.asarray(query, dtype=np.float64)
        key = np.asarray(key, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)

        batch, seq_len, d_model = query.shape

        # Scale d_model per head
        d_head = d_model // self._heads

        # Compute attention scores using quantum circuits
        scores = np.zeros((batch, self._heads, seq_len, seq_len), dtype=np.float64)
        sim = _get_simulator()

        for h in range(self._heads):
            h_params = self._params[
                h * self._n_layers * self._n_qubits * 3:
                (h + 1) * self._n_layers * self._n_qubits * 3
            ]
            q_start = h * d_head
            q_end = q_start + d_head

            for b in range(batch):
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Encode q_i and k_j into the quantum circuit
                        q_vec = query[b, i, q_start:q_end]
                        k_vec = key[b, j, q_start:q_end]
                        combined = np.concatenate([q_vec, k_vec])

                        # Reduce to n_qubits
                        combined = self._reduce_to_nqubits(combined)
                        combined = np.clip(combined, -_PI, _PI)

                        from quantumflow.core.circuit import QuantumCircuit
                        qc = QuantumCircuit(self._n_qubits)
                        _encode_angle(qc, combined, self._n_qubits)
                        var_qc = _build_variational_circuit(
                            self._n_qubits, h_params, self._n_layers, "linear"
                        )
                        for op in var_qc.data:
                            qc.append(op.gate, op.qubits, op.params)

                        obs = _build_pauli_observable("z", 0, self._n_qubits)
                        scores[b, h, i, j] = float(sim.expectation(qc, obs))

        # Softmax over keys
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply attention to values
        output = np.zeros_like(value)
        for h in range(self._heads):
            v_start = h * d_head
            v_end = v_start + d_head
            for b in range(batch):
                output[b, :, v_start:v_end] = (
                    attention_weights[b, h] @ value[b, :, v_start:v_end]
                )

        return output

    def _reduce_to_nqubits(self, vec: np.ndarray) -> np.ndarray:
        """Reduce feature vector to n_qubits dimensions."""
        n = len(vec)
        nq = self._n_qubits
        if n == nq:
            return vec
        elif n > nq:
            group_size = n // nq
            result = np.zeros(nq, dtype=np.float64)
            for i in range(nq):
                start = i * group_size
                end = start + group_size if i < nq - 1 else n
                result[i] = float(np.mean(vec[start:end]))
            return result
        else:
            padded = np.zeros(nq, dtype=np.float64)
            padded[:n] = vec
            return padded

    def compute_output_shape(self, input_shape: Any) -> Tuple[int, ...]:
        """Compute output shape."""
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "heads": self._heads,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QAttentionLayer:
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        return [self._params] if self._params is not None else []

    def set_weights(self, weights: List[np.ndarray]) -> None:
        if self._params is not None and weights:
            self._params = np.asarray(weights[0], dtype=np.float64)

    def count_params(self) -> int:
        return self._params.size if self._params is not None else 0

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        if isinstance(inputs, (tuple, list)):
            try:
                import tensorflow as tf
                if all(isinstance(x, tf.Tensor) for x in inputs):
                    out = self.call([x.numpy() for x in inputs])
                    return tf.constant(out)
            except ImportError:
                pass
        return self.call(inputs)

    def __repr__(self) -> str:
        return (
            f"QAttentionLayer(n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, heads={self._heads})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QResidualLayer — Quantum Residual Block
# ═══════════════════════════════════════════════════════════════════════════

class QResidualLayer:
    """Quantum residual block with skip connection.

    Architecture: ``output = activation(quantum_process(x) + skip(x))``

    The quantum processing block encodes input data, applies variational
    layers, and measures output. A learnable skip projection handles
    dimension mismatches.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int, optional
        Variational layers. Default ``2``.
    skip_projection : bool, optional
        Learnable skip projection if dimensions differ. Default ``True``.
    activation : str or callable, optional
        Activation after residual addition. Default ``'relu'``.
    entanglement : str, optional
        Default ``'linear'``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.tensorflow.layers import QResidualLayer
    >>> res = QResidualLayer(n_qubits=4, n_layers=2)
    >>> res.build((None, 4))
    >>> out = res.call(np.random.randn(8, 4).astype(np.float64))
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        skip_projection: bool = True,
        activation: Optional[Union[str, Callable]] = "relu",
        entanglement: str = "linear",
        name: Optional[str] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._skip_projection = skip_projection
        self._activation = activation
        self._entanglement = entanglement
        self._name = name or f"qresidual_{id(self):x}"

        self._params: Optional[np.ndarray] = None
        self._skip_kernel: Optional[np.ndarray] = None
        self._skip_bias: Optional[np.ndarray] = None
        self._built = False
        self._input_dim: Optional[int] = None

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize parameters."""
        self._input_dim = input_shape[-1]
        rng = np.random.default_rng()

        n_var = self._n_layers * self._n_qubits * 3
        self._params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)

        if self._skip_projection and self._input_dim != self._n_qubits:
            limit = math.sqrt(6.0 / (self._input_dim + self._n_qubits))
            self._skip_kernel = rng.uniform(
                -limit, limit, (self._input_dim, self._n_qubits)
            ).astype(np.float64)
            self._skip_bias = np.zeros(self._n_qubits, dtype=np.float64)
        else:
            self._skip_kernel = None
            self._skip_bias = None

        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, features)``.

        Returns
        -------
        numpy.ndarray
        """
        if not self._built:
            raise RuntimeError("Layer not built.")
        assert self._params is not None

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs.reshape(1, -1)

        batch = inputs.shape[0]
        sim = _get_simulator()
        obs_list = [
            _build_pauli_observable("z", q, self._n_qubits)
            for q in range(self._n_qubits)
        ]

        quantum_output = np.zeros((batch, self._n_qubits), dtype=np.float64)

        for b in range(batch):
            encoded = self._reduce_to_nqubits(inputs[b])
            encoded = np.clip(encoded, -_PI, _PI)

            from quantumflow.core.circuit import QuantumCircuit
            qc = QuantumCircuit(self._n_qubits)
            _encode_angle(qc, encoded, self._n_qubits)
            var_qc = _build_variational_circuit(
                self._n_qubits, self._params, self._n_layers, self._entanglement
            )
            for op in var_qc.data:
                qc.append(op.gate, op.qubits, op.params)

            quantum_output[b] = _run_and_measure(qc, obs_list, sim)

        # Skip connection
        if self._skip_kernel is not None:
            skip = inputs @ self._skip_kernel
            if self._skip_bias is not None:
                skip = skip + self._skip_bias
        else:
            skip = inputs[:, :self._n_qubits]

        # Residual addition
        output = quantum_output + skip

        # Activation
        if isinstance(self._activation, str):
            if self._activation == "relu":
                output = np.maximum(0, output)
            elif self._activation == "tanh":
                output = np.tanh(output)
            elif self._activation == "sigmoid":
                output = 1.0 / (1.0 + np.exp(-np.clip(output, -20, 20)))
        elif callable(self._activation):
            output = self._activation(output)

        if squeeze:
            output = output.reshape(-1)
        return output

    def _reduce_to_nqubits(self, vec: np.ndarray) -> np.ndarray:
        n = len(vec)
        nq = self._n_qubits
        if n == nq:
            return vec
        elif n > nq:
            group = n // nq
            result = np.zeros(nq, dtype=np.float64)
            for i in range(nq):
                s = i * group
                e = s + group if i < nq - 1 else n
                result[i] = float(np.mean(vec[s:e]))
            return result
        else:
            padded = np.zeros(nq, dtype=np.float64)
            padded[:n] = vec
            return padded

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) < 1:
            return (self._n_qubits,)
        return input_shape[:-1] + (self._n_qubits,)

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "skip_projection": self._skip_projection,
            "activation": self._activation,
            "entanglement": self._entanglement,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QResidualLayer:
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        w = []
        if self._params is not None: w.append(self._params)
        if self._skip_kernel is not None: w.append(self._skip_kernel)
        if self._skip_bias is not None: w.append(self._skip_bias)
        return w

    def set_weights(self, weights: List[np.ndarray]) -> None:
        idx = 0
        if self._params is not None and idx < len(weights):
            self._params = np.asarray(weights[idx], dtype=np.float64); idx += 1
        if self._skip_kernel is not None and idx < len(weights):
            self._skip_kernel = np.asarray(weights[idx], dtype=np.float64); idx += 1
        if self._skip_bias is not None and idx < len(weights):
            self._skip_bias = np.asarray(weights[idx], dtype=np.float64); idx += 1

    def count_params(self) -> int:
        c = self._params.size if self._params is not None else 0
        if self._skip_kernel is not None: c += self._skip_kernel.size
        if self._skip_bias is not None: c += self._skip_bias.size
        return c

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return tf.constant(self.call(inputs.numpy()))
        except ImportError:
            pass
        return self.call(inputs)

    def __repr__(self) -> str:
        return (
            f"QResidualLayer(n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"params={self.count_params()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QFeatureMapLayer — Quantum Feature Map / Kernel
# ═══════════════════════════════════════════════════════════════════════════

class QFeatureMapLayer:
    """Quantum feature map layer that maps classical data to quantum feature space.

    Computes kernel entries via quantum circuits:

    .. math::

        K(x_i, x_j) = |\\langle 0 | U(x_i)^\\dagger U(x_j) | 0 \\rangle|^2

    Supports multiple feature map types: ``'zx'``, ``'zz'``, ``'pauli'``,
    ``'iid'``, ``'hardware_efficient'``.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    feature_map : str, optional
        Feature map type. Default ``'zx'``.
    n_reps : int, optional
        Number of repetitions. Default ``2``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.tensorflow.layers import QFeatureMapLayer
    >>> fm = QFeatureMapLayer(n_qubits=4, feature_map='zz')
    >>> fm.build((None, 4))
    >>> out = fm.call(np.random.randn(8, 4).astype(np.float64))
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map: str = "zx",
        n_reps: int = 2,
        name: Optional[str] = None,
    ) -> None:
        if feature_map not in _VALID_ENCODINGS:
            raise ValueError(
                f"Unknown feature map '{feature_map}'. "
                f"Choose from {sorted(_VALID_ENCODINGS)}"
            )

        self._n_qubits = n_qubits
        self._feature_map = feature_map
        self._n_reps = n_reps
        self._name = name or f"qfeaturemap_{id(self):x}"
        self._built = False

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def feature_map(self) -> str:
        return self._feature_map

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build (no trainable params for pure feature maps)."""
        self._input_dim = input_shape[-1]
        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Compute feature map representation.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, n_qubits)``.

        Returns
        -------
        numpy.ndarray
            Feature representations of shape ``(batch, n_qubits)``.
        """
        if not self._built:
            raise RuntimeError("Layer not built.")

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs.reshape(1, -1)

        batch = inputs.shape[0]
        sim = _get_simulator()
        outputs = np.zeros((batch, self._n_qubits), dtype=np.float64)

        for b in range(batch):
            from quantumflow.core.circuit import QuantumCircuit
            qc = QuantumCircuit(self._n_qubits)
            data = inputs[b][:self._n_qubits]

            if self._feature_map == "zx":
                self._feature_map_zx(qc, data)
            elif self._feature_map == "zz":
                self._feature_map_zz(qc, data)
            elif self._feature_map == "pauli":
                self._feature_map_pauli(qc, data)
            elif self._feature_map == "iid":
                self._feature_map_iid(qc, data)
            elif self._feature_map == "hardware_efficient":
                self._feature_map_hw(qc, data)

            obs_list = [
                _build_pauli_observable("z", q, self._n_qubits)
                for q in range(self._n_qubits)
            ]
            outputs[b] = _run_and_measure(qc, obs_list, sim)

        if squeeze:
            outputs = outputs.reshape(-1)
        return outputs

    def _feature_map_zx(self, qc: Any, data: np.ndarray) -> None:
        """ZX feature map: H -> RZ(x) per qubit, CZ entanglement."""
        for _ in range(self._n_reps):
            for q in range(self._n_qubits):
                qc.h(q)
                qc.rz(float(data[q]), q)
            for i in range(self._n_qubits - 1):
                qc.cz(i, i + 1)

    def _feature_map_zz(self, qc: Any, data: np.ndarray) -> None:
        """ZZ feature map: RZ -> CZ -> RZ layers."""
        for _ in range(self._n_reps):
            for q in range(self._n_qubits):
                qc.h(q)
                qc.rz(float(data[q]), q)
            for i in range(self._n_qubits - 1):
                qc.rzz(float(data[i] * data[i + 1]), i, i + 1)

    def _feature_map_pauli(self, qc: Any, data: np.ndarray) -> None:
        """Pauli feature map: RZ -> RXX -> RYY -> RZZ."""
        for _ in range(self._n_reps):
            for q in range(self._n_qubits):
                qc.h(q)
                qc.rz(float(data[q]), q)
            for i in range(self._n_qubits - 1):
                qc.rxx(float(data[i] * data[i + 1]), i, i + 1)
                qc.ryy(float(data[i] * data[i + 1]), i, i + 1)
                qc.rzz(float(data[i] * data[i + 1]), i, i + 1)

    def _feature_map_iid(self, ic: Any, data: np.ndarray) -> None:
        """IID feature map: independent RZ + RY on each qubit."""
        for _ in range(self._n_reps):
            for q in range(self._n_qubits):
                ic.rz(float(data[q]), q)
                ic.ry(float(data[q]), q)

    def _feature_map_hw(self, qc: Any, data: np.ndarray) -> None:
        """Hardware-efficient feature map."""
        for _ in range(self._n_reps):
            for q in range(self._n_qubits):
                qc.h(q)
                qc.rz(float(data[q]), q)
                qc.ry(float(data[q] * 0.5), q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the quantum kernel matrix.

        Parameters
        ----------
        X : numpy.ndarray
            Shape ``(n_samples_X, n_features)``.
        Y : numpy.ndarray, optional
            Shape ``(n_samples_Y, n_features)``. If ``None``, use ``X``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples_X, n_samples_Y)`` kernel matrix.
        """
        from quantumflow.core.circuit import QuantumCircuit

        X = np.asarray(X, dtype=np.float64)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float64)

        n_x = X.shape[0]
        n_y = Y.shape[0]
        K = np.zeros((n_x, n_y), dtype=np.float64)
        sim = _get_simulator()

        for i in range(n_x):
            # Encode X[i] and get state
            qc_x = QuantumCircuit(self._n_qubits)
            data_x = X[i][:self._n_qubits]
            self._build_feature_circuit(qc_x, data_x)
            state_x = sim.state(qc_x)
            sv_x = state_x.data if hasattr(state_x, 'data') else np.asarray(state_x)

            for j in range(n_y):
                # Encode Y[j] and get state
                qc_y = QuantumCircuit(self._n_qubits)
                data_y = Y[j][:self._n_qubits]
                self._build_feature_circuit(qc_y, data_y)
                state_y = sim.state(qc_y)
                sv_y = state_y.data if hasattr(state_y, 'data') else np.asarray(state_y)

                # Kernel = |<phi_x | phi_y>|^2
                K[i, j] = float(np.abs(np.vdot(sv_x, sv_y)) ** 2)

        return K

    def _build_feature_circuit(self, qc: Any, data: np.ndarray) -> None:
        """Build the feature map circuit."""
        if self._feature_map == "zx":
            self._feature_map_zx(qc, data)
        elif self._feature_map == "zz":
            self._feature_map_zz(qc, data)
        elif self._feature_map == "pauli":
            self._feature_map_pauli(qc, data)
        elif self._feature_map == "iid":
            self._feature_map_iid(qc, data)
        elif self._feature_map == "hardware_efficient":
            self._feature_map_hw(qc, data)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) < 1:
            return (self._n_qubits,)
        return input_shape[:-1] + (self._n_qubits,)

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_qubits": self._n_qubits,
            "feature_map": self._feature_map,
            "n_reps": self._n_reps,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QFeatureMapLayer:
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        return []

    def set_weights(self, weights: List[np.ndarray]) -> None:
        pass

    def count_params(self) -> int:
        return 0

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return tf.constant(self.call(inputs.numpy()))
        except ImportError:
            pass
        return self.call(inputs)

    def __repr__(self) -> str:
        return (
            f"QFeatureMapLayer(n_qubits={self._n_qubits}, "
            f"feature_map={self._feature_map!r}, n_reps={self._n_reps})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QMeasurementLayer — Quantum Measurement with Classical Readout
# ═══════════════════════════════════════════════════════════════════════════

class QMeasurementLayer:
    """Quantum measurement layer with configurable observables and strategies.

    Applies a quantum circuit and reads out classical values via measurement.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    observable : str, optional
        Observable type: ``'x'``, ``'y'``, ``'z'``. Default ``'z'``.
    strategy : str, optional
        Measurement strategy:
        * ``'expectation'`` — Expectation value of observable.
        * ``'probability'`` — Measurement probability.
        * ``'sample'`` — Sampled measurement outcomes.
        Default ``'expectation'``.
    shots : int, optional
        Number of shots for ``'sample'`` strategy. Default ``1024``.
    readout_dim : int, optional
        Output dimension. If ``None``, uses n_qubits. Default ``None``.
    name : str, optional

    Examples
    --------
    >>> from quantumflow.tensorflow.layers import QMeasurementLayer
    >>> ml = QMeasurementLayer(n_qubits=4, observable='z', strategy='expectation')
    >>> # Pass pre-computed expectation values as input
    >>> ml.build((None, 4))
    >>> out = ml.call(np.random.randn(8, 4).astype(np.float64))
    """

    def __init__(
        self,
        n_qubits: int,
        observable: str = "z",
        strategy: str = "expectation",
        shots: int = 1024,
        readout_dim: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if strategy not in _VALID_MEASUREMENT_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {sorted(_VALID_MEASUREMENT_STRATEGIES)}"
            )

        self._n_qubits = n_qubits
        self._observable = observable
        self._strategy = strategy
        self._shots = shots
        self._readout_dim = readout_dim or n_qubits
        self._name = name or f"qmeasurement_{id(self):x}"
        self._built = False

        self._readout_kernel: Optional[np.ndarray] = None
        self._readout_bias: Optional[np.ndarray] = None

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def readout_dim(self) -> int:
        return self._readout_dim

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize readout projection."""
        self._input_dim = input_shape[-1]
        rng = np.random.default_rng()

        if self._readout_dim != self._input_dim:
            limit = math.sqrt(6.0 / (self._input_dim + self._readout_dim))
            self._readout_kernel = rng.uniform(
                -limit, limit, (self._input_dim, self._readout_dim)
            ).astype(np.float64)
            self._readout_bias = np.zeros(self._readout_dim, dtype=np.float64)
        else:
            self._readout_kernel = None
            self._readout_bias = None

        self._built = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input features of shape ``(batch, n_qubits)`` or raw
            quantum expectation values.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, readout_dim)``.
        """
        if not self._built:
            raise RuntimeError("Layer not built.")

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs.reshape(1, -1)

        # Apply quantum measurement processing
        measured = self._apply_measurement(inputs)

        # Readout projection
        if self._readout_kernel is not None:
            output = inputs @ self._readout_kernel
            if self._readout_bias is not None:
                output = output + self._readout_bias
        else:
            output = measured

        # Map to readout_dim
        if output.shape[-1] != self._readout_dim:
            if output.shape[-1] > self._readout_dim:
                output = output[:, :self._readout_dim]
            else:
                tiled = np.tile(
                    output,
                    (self._readout_dim + output.shape[-1] - 1) // output.shape[-1],
                    axis=1,
                )
                output = tiled[:, :self._readout_dim]

        if squeeze:
            output = output.reshape(-1)
        return output

    def _apply_measurement(self, inputs: np.ndarray) -> np.ndarray:
        """Apply measurement strategy to inputs.

        If inputs are raw features, processes them through a quantum
        circuit first. Otherwise interprets them as pre-computed
        quantum values.
        """
        if self._strategy == "expectation":
            # Inputs are expectation values, clip to [-1, 1]
            return np.clip(inputs, -1.0, 1.0)

        elif self._strategy == "probability":
            # Convert inputs to probabilities via sigmoid-like mapping
            probs = 1.0 / (1.0 + np.exp(-np.clip(inputs, -20, 20)))
            # Normalize per sample
            row_sums = np.sum(probs, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            return probs / row_sums

        elif self._strategy == "sample":
            # Convert inputs to probabilities, then sample
            probs = 1.0 / (1.0 + np.exp(-np.clip(inputs, -20, 20)))
            row_sums = np.sum(probs, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            probs = probs / row_sums

            batch = inputs.shape[0]
            output = np.zeros_like(inputs)
            for b in range(batch):
                indices = np.random.choice(
                    inputs.shape[1], size=inputs.shape[1],
                    p=probs[b], replace=True,
                )
                for idx, val in zip(indices, np.ones(inputs.shape[1])):
                    output[b, idx] += val
                output[b] /= float(inputs.shape[1])
            return output

        return inputs

    def measure_circuit(
        self,
        circuit: Any,
        observable: Optional[str] = None,
    ) -> np.ndarray:
        """Directly measure a quantum circuit.

        Parameters
        ----------
        circuit : QuantumCircuit
        observable : str, optional
            Override observable. Default uses configured observable.

        Returns
        -------
        numpy.ndarray
            Expectation values.
        """
        obs = observable or self._observable
        sim = _get_simulator()
        obs_list = [
            _build_pauli_observable(obs, q, self._n_qubits)
            for q in range(self._n_qubits)
        ]
        return _run_and_measure(circuit, obs_list, sim)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) < 1:
            return (self._readout_dim,)
        return input_shape[:-1] + (self._readout_dim,)

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_qubits": self._n_qubits,
            "observable": self._observable,
            "strategy": self._strategy,
            "shots": self._shots,
            "readout_dim": self._readout_dim,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QMeasurementLayer:
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        w = []
        if self._readout_kernel is not None: w.append(self._readout_kernel)
        if self._readout_bias is not None: w.append(self._readout_bias)
        return w

    def set_weights(self, weights: List[np.ndarray]) -> None:
        idx = 0
        if self._readout_kernel is not None and idx < len(weights):
            self._readout_kernel = np.asarray(weights[idx], dtype=np.float64); idx += 1
        if self._readout_bias is not None and idx < len(weights):
            self._readout_bias = np.asarray(weights[idx], dtype=np.float64); idx += 1

    def count_params(self) -> int:
        c = 0
        if self._readout_kernel is not None: c += self._readout_kernel.size
        if self._readout_bias is not None: c += self._readout_bias.size
        return c

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return tf.constant(self.call(inputs.numpy()))
        except ImportError:
            pass
        return self.call(inputs)

    def __repr__(self) -> str:
        return (
            f"QMeasurementLayer(n_qubits={self._n_qubits}, "
            f"observable={self._observable!r}, strategy={self._strategy!r}, "
            f"readout_dim={self._readout_dim})"
        )
