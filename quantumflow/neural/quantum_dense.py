"""
Quantum Dense Layer
===================

Provides quantum implementations of fully-connected (dense) layers that
can be used as drop-in replacements for ``tf.keras.layers.Dense`` in
TensorFlow/Keras models.

Classes
-------
* :class:`QuantumDense` — dense layer implemented with a quantum circuit,
  compatible with Keras ``Layer`` API.
* :class:`QuantumDenseWithMeasurement` — dense layer with configurable
  measurement basis (Pauli-Z, Pauli-X, Pauli-Y, custom observables, or
  multi-observable measurement).

Architecture
------------
Each quantum dense layer:
1. Maps the classical input through a trainable weight matrix to generate
   rotation angles for the quantum circuit.
2. Encodes the angles onto qubits via rotation gates.
3. Applies variational layers for feature extraction.
4. Measures expectation values to produce the classical output.

The layer supports multiple weight initialization strategies:
* ``'glorot'`` — Xavier/Glorot uniform initialization.
* ``'he'`` — He normal initialization.
* ``'quantum'`` — Initialization optimized for quantum circuits
  (small angles near zero).

Examples
--------
Using with TensorFlow:

    >>> import tensorflow as tf
    >>> from quantumflow.neural import QuantumDense
    >>> layer = QuantumDense(output_dim=4, n_qubits=3, n_layers=2)
    >>> x = tf.constant([[1.0, 2.0, 3.0]])
    >>> output = layer(x)
    >>> output.shape
    TensorShape([1, 4])

Using standalone (without TensorFlow):

    >>> layer = QuantumDense(output_dim=4, n_qubits=3, n_layers=2)
    >>> layer.build((None, 3))
    >>> output = layer.call([[1.0, 2.0, 3.0]])
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

from quantumflow.core.circuit import QuantumCircuit

__all__ = [
    "QuantumDense",
    "QuantumDenseWithMeasurement",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10

_VALID_ACTIVATIONS = frozenset({
    "quantum_relu",
    "quantum_sigmoid",
    "quantum_tanh",
    "quantum_swish",
    "relu",
    "sigmoid",
    "tanh",
    "linear",
    None,
})

_VALID_INITS = frozenset({
    "glorot",
    "he",
    "quantum",
    "zeros",
    "ones",
    "random",
})


# ---------------------------------------------------------------------------
# Weight initializers
# ---------------------------------------------------------------------------

def _glorot_uniform(
    shape: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Xavier/Glorot uniform initialization.

    Parameters
    ----------
    shape : tuple of int
        Weight shape ``(fan_in, fan_out)``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
    """
    fan_in, fan_out = shape[0], shape[-1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float64)


def _he_normal(
    shape: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """He normal initialization.

    Parameters
    ----------
    shape : tuple of int
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
    """
    fan_in = shape[0]
    std = math.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=shape).astype(np.float64)


def _quantum_init(
    shape: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Quantum-optimized initialization.

    Initializes weights with small values near zero, which is
    optimal for quantum rotation angles (avoids barren plateaus).

    Parameters
    ----------
    shape : tuple of int
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
    """
    limit = 0.1
    return rng.uniform(-limit, limit, size=shape).astype(np.float64)


_INITIALIZER_MAP: Dict[str, Callable] = {
    "glorot": _glorot_uniform,
    "he": _he_normal,
    "quantum": _quantum_init,
    "zeros": lambda shape, rng: np.zeros(shape, dtype=np.float64),
    "ones": lambda shape, rng: np.ones(shape, dtype=np.float64),
    "random": lambda shape, rng: rng.uniform(-1, 1, size=shape).astype(np.float64),
}


# ---------------------------------------------------------------------------
# Classical activation functions (used when quantum activation not available)
# ---------------------------------------------------------------------------

def _apply_classical_activation(
    x: np.ndarray,
    activation: str,
) -> np.ndarray:
    """Apply a classical activation function.

    Parameters
    ----------
    x : numpy.ndarray
    activation : str

    Returns
    -------
    numpy.ndarray
    """
    if activation == "relu" or activation is None:
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")


# ---------------------------------------------------------------------------
# QuantumDense
# ---------------------------------------------------------------------------

class QuantumDense:
    """Dense layer implemented with a quantum circuit.

    Can be used as a drop-in replacement for ``tf.keras.layers.Dense``.
    Maps input features to a quantum circuit, applies variational
    transformations, and measures the output.

    Parameters
    ----------
    output_dim : int
        Number of output units.
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int, optional
        Number of variational layers. Default ``3``.
    activation : str, optional
        Activation function. Options:
        * ``'quantum_relu'``, ``'quantum_sigmoid'``, ``'quantum_tanh'``,
          ``'quantum_swish'`` — quantum activations.
        * ``'relu'``, ``'sigmoid'``, ``'tanh'``, ``'linear'`` — classical
          activations.
        * ``None`` — no activation.
    use_bias : bool, optional
        Whether to include a bias term. Default ``True``.
    kernel_init : str, optional
        Weight initialization strategy. Default ``'quantum'``.
        Options: ``'glorot'``, ``'he'``, ``'quantum'``, ``'zeros'``,
        ``'ones'``, ``'random'``.
    bias_init : str, optional
        Bias initialization. Default ``'zeros'``.
    encoding : str, optional
        Data encoding strategy. Default ``'angle'``.
    observable : str, optional
        Measurement observable. Default ``'z'``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> layer = QuantumDense(4, n_qubits=3, n_layers=2)
    >>> layer.build((None, 3))
    >>> output = layer.call([[1.0, 2.0, 3.0]])
    """

    def __init__(
        self,
        output_dim: int,
        n_qubits: int,
        n_layers: int = 3,
        activation: Optional[str] = "quantum_relu",
        use_bias: bool = True,
        kernel_init: str = "quantum",
        bias_init: str = "zeros",
        encoding: str = "angle",
        observable: str = "z",
        name: Optional[str] = None,
    ) -> None:
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if kernel_init not in _VALID_INITS:
            raise ValueError(
                f"Unknown kernel_init '{kernel_init}'. "
                f"Choose from {sorted(_VALID_INITS)}"
            )
        if bias_init not in _VALID_INITS:
            raise ValueError(
                f"Unknown bias_init '{bias_init}'. "
                f"Choose from {sorted(_VALID_INITS)}"
            )

        self._output_dim = output_dim
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._activation_name = activation
        self._use_bias = use_bias
        self._kernel_init_name = kernel_init
        self._bias_init_name = bias_init
        self._encoding = encoding
        self._observable = observable
        self._name = name or f"quantum_dense_{id(self):x}"

        # Weights (created in build)
        self._kernel: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._variational_params: Optional[np.ndarray] = None

        # Input/output info
        self._input_dim: Optional[int] = None
        self._built = False

        # Activation function reference
        self._activation: Optional[Any] = None

    @property
    def output_dim(self) -> int:
        """int: Number of output units."""
        return self._output_dim

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of variational layers."""
        return self._n_layers

    @property
    def built(self) -> bool:
        """bool: Whether the layer has been built."""
        return self._built

    @property
    def kernel(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Weight matrix."""
        return self._kernel

    @property
    def bias(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Bias vector."""
        return self._bias

    @property
    def variational_params(self) -> Optional[np.ndarray]:
        """numpy.ndarray or None: Variational circuit parameters."""
        return self._variational_params

    # -- Build ----------------------------------------------------------------

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create the weight variables.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input. Expected ``(batch_size, input_dim)`` or
            ``(input_dim,)``.
        """
        if len(input_shape) < 1:
            raise ValueError(f"input_shape must have at least 1 dimension, got {input_shape}")

        self._input_dim = input_shape[-1]

        rng = np.random.default_rng()

        # Kernel: maps input_dim → n_qubits rotation angles
        kernel_shape = (self._input_dim, self._n_qubits)
        init_fn = _INITIALIZER_MAP[self._kernel_init_name]
        self._kernel = init_fn(kernel_shape, rng)

        # Bias
        if self._use_bias:
            bias_shape = (self._n_qubits,)
            bias_init_fn = _INITIALIZER_MAP[self._bias_init_name]
            self._bias = bias_init_fn(bias_shape, rng)

        # Variational parameters: n_layers * n_qubits * 3 rotations
        n_var_params = self._n_layers * self._n_qubits * 3
        self._variational_params = rng.uniform(
            -0.1, 0.1, size=n_var_params
        ).astype(np.float64)

        # Setup activation
        self._setup_activation()

        self._built = True

    def _setup_activation(self) -> None:
        """Initialize the activation function."""
        if self._activation_name is None or self._activation_name in (
            "relu", "sigmoid", "tanh", "linear"
        ):
            self._activation = self._activation_name
        else:
            try:
                # Try quantum activation
                if self._activation_name == "quantum_relu":
                    from quantumflow.neural.quantum_activation import QuantumReLU
                    self._activation = QuantumReLU(
                        n_qubits=min(self._n_qubits, 2),
                        n_layers=min(self._n_layers, 2),
                    )
                elif self._activation_name == "quantum_sigmoid":
                    from quantumflow.neural.quantum_activation import QuantumSigmoid
                    self._activation = QuantumSigmoid(
                        n_qubits=min(self._n_qubits, 2),
                        n_layers=min(self._n_layers, 2),
                    )
                elif self._activation_name == "quantum_tanh":
                    from quantumflow.neural.quantum_activation import QuantumTanh
                    self._activation = QuantumTanh(
                        n_qubits=min(self._n_qubits, 2),
                        n_layers=min(self._n_layers, 2),
                    )
                elif self._activation_name == "quantum_swish":
                    from quantumflow.neural.quantum_activation import QuantumSwish
                    self._activation = QuantumSwish(
                        n_qubits=min(self._n_qubits, 2),
                        n_layers=min(self._n_layers, 2),
                    )
                else:
                    self._activation = self._activation_name
            except (ImportError, AttributeError):
                # Fallback to classical
                classical_map = {
                    "quantum_relu": "relu",
                    "quantum_sigmoid": "sigmoid",
                    "quantum_tanh": "tanh",
                    "quantum_swish": "linear",
                }
                self._activation = classical_map.get(
                    self._activation_name, self._activation_name
                )

    # -- Forward pass ---------------------------------------------------------

    def call(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[float]]],
    ) -> np.ndarray:
        """Compute the quantum dense layer output.

        Parameters
        ----------
        inputs : numpy.ndarray or sequence of sequence of float
            Input tensor of shape ``(batch_size, input_dim)`` or
            ``(input_dim,)``.

        Returns
        -------
        numpy.ndarray
            Output tensor of shape ``(batch_size, output_dim)`` or
            ``(output_dim,)``.
        """
        if not self._built:
            raise RuntimeError(
                "Layer has not been built. Call build(input_shape) first."
            )

        inputs = np.asarray(inputs, dtype=np.float64)
        original_shape = inputs.shape

        # Handle single sample (1-D input)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        batch_size = inputs.shape[0]
        outputs = np.zeros((batch_size, self._output_dim), dtype=np.float64)

        for b in range(batch_size):
            outputs[b] = self._forward_single(inputs[b])

        if single_sample:
            outputs = outputs.reshape(-1)

        return outputs

    def _forward_single(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a single sample.

        Parameters
        ----------
        x : numpy.ndarray
            Input vector of shape ``(input_dim,)``.

        Returns
        -------
        numpy.ndarray
            Output vector of shape ``(output_dim,)``.
        """
        assert self._kernel is not None
        assert self._variational_params is not None

        # Step 1: Compute rotation angles via weight matrix
        angles = x @ self._kernel  # shape: (n_qubits,)

        # Add bias
        if self._use_bias and self._bias is not None:
            angles = angles + self._bias

        # Normalize angles to [-π, π]
        angles = np.clip(angles, -_PI, _PI)

        # Step 2: Build and execute quantum circuit
        quantum_output = self._execute_quantum_circuit(angles)

        # Step 3: Map quantum output to output_dim
        output = self._map_quantum_to_output(quantum_output)

        # Step 4: Apply activation
        output = self._apply_activation(output)

        return output

    def _execute_quantum_circuit(
        self,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Execute the quantum circuit for the given rotation angles.

        Parameters
        ----------
        angles : numpy.ndarray
            Rotation angles of shape ``(n_qubits,)``.

        Returns
        -------
        numpy.ndarray
            Expectation values from the quantum circuit.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        qc = QuantumCircuit(self._n_qubits)

        # Encode angles
        for q, angle in enumerate(angles):
            qc.h(q)
            qc.ry(float(angle), q)

        # Apply variational layers
        assert self._variational_params is not None
        param_offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._variational_params[param_offset])
                theta = float(self._variational_params[param_offset + 1])
                omega = float(self._variational_params[param_offset + 2])
                param_offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)

            # Entangling layer
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        # Measure
        simulator = StatevectorSimulator()
        results = []
        for q in range(self._n_qubits):
            obs = self._pauli_observable(self._observable, q, self._n_qubits)
            val = simulator.expectation(qc, obs)
            results.append(float(val))

        return np.array(results, dtype=np.float64)

    def _map_quantum_to_output(
        self,
        quantum_output: np.ndarray,
    ) -> np.ndarray:
        """Map quantum measurement results to output_dim.

        If output_dim > n_qubits, tiles cyclically.
        If output_dim < n_qubits, truncates.

        Parameters
        ----------
        quantum_output : numpy.ndarray
            Shape ``(n_qubits,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(output_dim,)``.
        """
        if len(quantum_output) == self._output_dim:
            return quantum_output
        elif len(quantum_output) > self._output_dim:
            return quantum_output[:self._output_dim]
        else:
            # Tile cyclically
            repeats = (self._output_dim + len(quantum_output) - 1) // len(quantum_output)
            tiled = np.tile(quantum_output, repeats)
            return tiled[:self._output_dim]

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if self._activation is None:
            return x
        elif isinstance(self._activation, str):
            return _apply_classical_activation(x, self._activation)
        else:
            # Quantum activation object
            return self._activation.forward(x)

    # -- Observable helper ----------------------------------------------------

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
        n_qubits : int

        Returns
        -------
        numpy.ndarray
        """
        pauli_matrices = {
            "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        mat = pauli_matrices.get(pauli)
        if mat is None:
            mat = pauli_matrices["z"]

        full = np.array([[1.0]], dtype=np.complex128)
        for i in range(n_qubits):
            if i == qubit:
                full = np.kron(full, mat)
            else:
                full = np.kron(full, np.eye(2, dtype=np.complex128))
        return full

    # -- Output shape ---------------------------------------------------------

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape.

        Parameters
        ----------
        input_shape : tuple of int

        Returns
        -------
        tuple of int
        """
        if len(input_shape) < 1:
            return (self._output_dim,)
        return input_shape[:-1] + (self._output_dim,)

    # -- Serialization --------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        Returns
        -------
        dict
        """
        config = {
            "output_dim": self._output_dim,
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "activation": self._activation_name,
            "use_bias": self._use_bias,
            "kernel_init": self._kernel_init_name,
            "bias_init": self._bias_init_name,
            "encoding": self._encoding,
            "observable": self._observable,
            "name": self._name,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantumDense:
        """Create a layer from configuration.

        Parameters
        ----------
        config : dict

        Returns
        -------
        QuantumDense
        """
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        """Return current weights.

        Returns
        -------
        list of numpy.ndarray
        """
        weights = []
        if self._kernel is not None:
            weights.append(self._kernel)
        if self._use_bias and self._bias is not None:
            weights.append(self._bias)
        if self._variational_params is not None:
            weights.append(self._variational_params)
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set weights.

        Parameters
        ----------
        weights : list of numpy.ndarray
        """
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

    # -- TensorFlow/Keras compatibility ----------------------------------------

    def __call__(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[float]], Any],
    ) -> np.ndarray:
        """Make the layer callable.

        Supports both numpy arrays and TensorFlow tensors.

        Parameters
        ----------
        inputs : array_like or tf.Tensor

        Returns
        -------
        numpy.ndarray
        """
        # Check if this is a TensorFlow tensor
        try:
            import tensorflow as tf
            if isinstance(inputs, tf.Tensor):
                return self._tf_call(inputs)
        except ImportError:
            pass

        return self.call(inputs)

    def _tf_call(self, inputs: Any) -> Any:
        """Handle TensorFlow tensor inputs.

        Parameters
        ----------
        inputs : tf.Tensor

        Returns
        -------
        tf.Tensor
        """
        import tensorflow as tf

        @tf.custom_gradient
        def quantum_dense_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            """Custom TensorFlow operation wrapping the quantum dense layer."""
            # Forward pass
            x_np = x.numpy()
            output_np = self.call(x_np)
            output = tf.constant(output_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                """Gradient computation via finite differences."""
                dy_np = dy.numpy()
                x_np_val = x.numpy()
                eps = 1e-5
                n_samples, n_features = x_np_val.shape

                # Jacobian via finite differences
                jacobian = np.zeros_like(x_np_val)
                for s in range(n_samples):
                    for f in range(n_features):
                        x_plus = x_np_val.copy()
                        x_plus[s, f] += eps
                        out_plus = self.call(x_plus)

                        x_minus = x_np_val.copy()
                        x_minus[s, f] -= eps
                        out_minus = self.call(x_minus)

                        jacobian[s, f] = np.sum((out_plus - out_minus) / (2 * eps) * dy_np[s])

                return tf.constant(jacobian)

            return output, grad

        return quantum_dense_op(inputs)

    # -- Count parameters -----------------------------------------------------

    def count_params(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
        """
        total = 0
        if self._kernel is not None:
            total += self._kernel.size
        if self._use_bias and self._bias is not None:
            total += self._bias.size
        if self._variational_params is not None:
            total += self._variational_params.size
        return total

    # -- Dunder methods -------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QuantumDense("
            f"output_dim={self._output_dim}, "
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"activation={self._activation_name!r}, "
            f"use_bias={self._use_bias}, "
            f"kernel_init={self._kernel_init_name!r}, "
            f"params={self.count_params()})"
        )


# ---------------------------------------------------------------------------
# QuantumDenseWithMeasurement
# ---------------------------------------------------------------------------

class QuantumDenseWithMeasurement(QuantumDense):
    """Dense layer with configurable measurement basis.

    Extends :class:`QuantumDense` to support:
    * Pauli-Z, Pauli-X, Pauli-Y single-qubit measurements.
    * Custom observables (user-provided Hermitian matrices).
    * Multi-observable measurement for richer feature extraction.

    Parameters
    ----------
    output_dim : int
        Number of output units.
    n_qubits : int
        Number of qubits.
    n_layers : int, optional
        Number of variational layers. Default ``3``.
    measurement_basis : str or list of str, optional
        Measurement configuration:
        * ``'z'`` — Pauli-Z on all qubits.
        * ``'x'`` — Pauli-X on all qubits.
        * ``'y'`` — Pauli-Y on all qubits.
        * ``'mixed'`` — Z + ZZ correlations.
        * ``['z', 'x']`` — Multiple observables stacked.
        * ``'custom'`` — Use custom_observables.
    custom_observables : list of numpy.ndarray, optional
        Custom observable matrices. Each must be Hermitian of shape
        ``(2**n_qubits, 2**n_qubits)``.
    activation : str, optional
        Activation function name. Default ``'quantum_relu'``.
    use_bias : bool, optional
        Default ``True``.
    kernel_init : str, optional
        Default ``'quantum'``.
    bias_init : str, optional
        Default ``'zeros'``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> layer = QuantumDenseWithMeasurement(
    ...     output_dim=6, n_qubits=3,
    ...     measurement_basis='mixed',
    ... )
    >>> layer.build((None, 3))
    >>> output = layer.call([[1.0, 2.0, 3.0]])
    """

    def __init__(
        self,
        output_dim: int,
        n_qubits: int,
        n_layers: int = 3,
        measurement_basis: Union[str, List[str]] = "z",
        custom_observables: Optional[List[np.ndarray]] = None,
        activation: Optional[str] = "quantum_relu",
        use_bias: bool = True,
        kernel_init: str = "quantum",
        bias_init: str = "zeros",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            output_dim=output_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            activation=activation,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            name=name,
        )

        self._measurement_basis = measurement_basis
        self._custom_observables = custom_observables

        # Resolve observables
        self._resolved_observables: Optional[List[Tuple[str, Any]]] = None
        self._resolve_observables()

    def _resolve_observables(self) -> None:
        """Resolve the measurement configuration into a list of observables."""
        self._resolved_observables = []
        n = self._n_qubits

        if isinstance(self._measurement_basis, str):
            basis = self._measurement_basis
            if basis == "z":
                for q in range(n):
                    obs = self._pauli_observable("z", q, n)
                    self._resolved_observables.append(("z", q, obs))
            elif basis == "x":
                for q in range(n):
                    obs = self._pauli_observable("x", q, n)
                    self._resolved_observables.append(("x", q, obs))
            elif basis == "y":
                for q in range(n):
                    obs = self._pauli_observable("y", q, n)
                    self._resolved_observables.append(("y", q, obs))
            elif basis == "mixed":
                # Z on all qubits
                for q in range(n):
                    obs = self._pauli_observable("z", q, n)
                    self._resolved_observables.append(("z", q, obs))
                # ZZ correlations
                for i in range(n - 1):
                    obs = self._two_qubit_observable("z", i, i + 1, n)
                    self._resolved_observables.append(("zz", (i, i + 1), obs))
                # XX correlations
                for i in range(n - 1):
                    obs = self._two_qubit_observable("x", i, i + 1, n)
                    self._resolved_observables.append(("xx", (i, i + 1), obs))
            elif basis == "custom":
                if self._custom_observables is not None:
                    for idx, obs in enumerate(self._custom_observables):
                        obs = np.asarray(obs, dtype=np.complex128)
                        self._resolved_observables.append(("custom", idx, obs))
            else:
                # Default to Z
                for q in range(n):
                    obs = self._pauli_observable("z", q, n)
                    self._resolved_observables.append(("z", q, obs))

        elif isinstance(self._measurement_basis, list):
            for basis in self._measurement_basis:
                if basis in ("z", "x", "y"):
                    for q in range(n):
                        obs = self._pauli_observable(basis, q, n)
                        self._resolved_observables.append((basis, q, obs))

    @staticmethod
    def _two_qubit_observable(
        pauli: str,
        q1: int,
        q2: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Build a two-qubit Pauli observable embedded in n-qubit space.

        Parameters
        ----------
        pauli : str
            ``'x'``, ``'y'``, or ``'z'``.
        q1, q2 : int
            Target qubit indices.
        n_qubits : int

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

    # -- Override quantum measurement ------------------------------------------

    def _execute_quantum_circuit(
        self,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Execute the quantum circuit with multi-observable measurement.

        Parameters
        ----------
        angles : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        qc = QuantumCircuit(self._n_qubits)

        # Encode angles
        for q, angle in enumerate(angles):
            qc.h(q)
            qc.ry(float(angle), q)

        # Variational layers
        assert self._variational_params is not None
        param_offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._variational_params[param_offset])
                theta = float(self._variational_params[param_offset + 1])
                omega = float(self._variational_params[param_offset + 2])
                param_offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        # Measure all configured observables
        simulator = StatevectorSimulator()
        results = []

        if self._resolved_observables:
            for obs_info in self._resolved_observables:
                obs = obs_info[-1]  # Last element is the observable matrix
                val = simulator.expectation(qc, obs)
                results.append(float(val))
        else:
            # Fallback: Z on each qubit
            for q in range(self._n_qubits):
                obs = self._pauli_observable("z", q, self._n_qubits)
                val = simulator.expectation(qc, obs)
                results.append(float(val))

        return np.array(results, dtype=np.float64)

    # -- Additional configuration ---------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        Returns
        -------
        dict
        """
        config = super().get_config()
        config["measurement_basis"] = self._measurement_basis
        config["custom_observables"] = (
            [obs.tolist() for obs in self._custom_observables]
            if self._custom_observables else None
        )
        return config

    @property
    def measurement_basis(self) -> Union[str, List[str]]:
        """str or list of str: Configured measurement basis."""
        return self._measurement_basis

    @property
    def n_observables(self) -> int:
        """int: Number of configured observables."""
        if self._resolved_observables:
            return len(self._resolved_observables)
        return self._n_qubits

    def __repr__(self) -> str:
        return (
            f"QuantumDenseWithMeasurement("
            f"output_dim={self._output_dim}, "
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"measurement_basis={self._measurement_basis!r}, "
            f"n_observables={self.n_observables}, "
            f"activation={self._activation_name!r}, "
            f"params={self.count_params()})"
        )
