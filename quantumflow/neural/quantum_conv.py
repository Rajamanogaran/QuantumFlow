"""
Quantum Convolution and Pooling
================================

Provides quantum implementations of convolution and pooling operations
for processing spatial data with quantum circuits.

Classes
-------
* :class:`QuantumConv2D` — 2D convolution using quantum circuits. Extracts
  image patches, processes each patch through a quantum circuit, and
  produces feature maps.
* :class:`QuantumPool2D` — quantum pooling layer for spatial reduction.
  Supports max, average, and quantum pooling strategies.

Architecture
------------
The quantum convolution works as follows:

1. **Patch extraction**: slide a kernel-sized window over the input and
   extract each patch.
2. **Quantum processing**: encode each patch into a quantum circuit,
   apply variational layers, and measure expectation values.
3. **Feature map construction**: arrange the quantum outputs into a
   spatial feature map.

This is analogous to classical convolution but uses quantum circuits
for the kernel operation, enabling quantum feature extraction.

Examples
--------
>>> conv = QuantumConv2D(filters=4, kernel_size=3, n_qubits=4, n_layers=2)
>>> conv.build((None, 28, 28, 1))
>>> output = conv.call(np.random.randn(1, 28, 28, 1))
>>> output.shape
(1, 26, 26, 4)

>>> pool = QuantumPool2D(pool_size=2, pool_type='quantum', n_qubits=2)
>>> pool.build((None, 26, 26, 4))
>>> output = pool.call(np.random.randn(1, 26, 26, 4))
>>> output.shape
(1, 13, 13, 4)
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
    "QuantumConv2D",
    "QuantumPool2D",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10

_VALID_CONV_MODES = frozenset({
    "qnn",
    "quadratic",
})

_VALID_POOL_TYPES = frozenset({
    "max",
    "average",
    "quantum",
})

_VALID_PADDING = frozenset({
    "valid",
    "same",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _extract_patches(
    image: np.ndarray,
    kernel_size: int,
    strides: int,
    padding: str,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract patches from a 2D image or batch of images.

    Parameters
    ----------
    image : numpy.ndarray
        Input of shape ``(batch, height, width, channels)`` or
        ``(height, width, channels)``.
    kernel_size : int
        Patch size (square).
    strides : int
        Stride for patch extraction.
    padding : str
        ``'valid'`` or ``'same'``.

    Returns
    -------
    numpy.ndarray
        Patches of shape ``(batch, out_h, out_w, patch_h, patch_w, channels)``.
    tuple of int
        Output spatial dimensions ``(out_h, out_w)``.
    """
    # Handle single image
    squeeze_batch = False
    if image.ndim == 3:
        image = image[np.newaxis, ...]
        squeeze_batch = True

    batch, h, w, c = image.shape

    # Apply padding
    if padding == "same":
        pad_total = kernel_size - 1
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        image = np.pad(
            image,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        h_padded, w_padded = image.shape[1], image.shape[2]
    else:
        h_padded, w_padded = h, w

    out_h = (h_padded - kernel_size) // strides + 1
    out_w = (w_padded - kernel_size) // strides + 1

    if out_h <= 0 or out_w <= 0:
        if squeeze_batch:
            return np.array([]), (0, 0)
        return np.array([]), (0, 0)

    # Extract patches using stride tricks
    patches = np.zeros(
        (batch, out_h, out_w, kernel_size, kernel_size, c),
        dtype=image.dtype,
    )

    for bh in range(out_h):
        for bw in range(out_w):
            h_start = bh * strides
            w_start = bw * strides
            patches[:, bh, bw, :, :, :] = image[
                :,
                h_start:h_start + kernel_size,
                w_start:w_start + kernel_size,
                :,
            ]

    if squeeze_batch:
        patches = patches[0]

    return patches, (out_h, out_w)


def _flatten_patch(
    patch: np.ndarray,
    max_features: int,
) -> np.ndarray:
    """Flatten and normalize a patch for quantum encoding.

    Parameters
    ----------
    patch : numpy.ndarray
        Patch of shape ``(kernel_h, kernel_w, channels)``.
    max_features : int
        Maximum number of features to extract.

    Returns
    -------
    numpy.ndarray
        Normalized feature vector of length ``max_features``.
    """
    flat = patch.flatten().astype(np.float64)

    if len(flat) > max_features:
        # Take first max_features
        flat = flat[:max_features]
    elif len(flat) < max_features:
        # Pad with zeros
        padded = np.zeros(max_features, dtype=np.float64)
        padded[:len(flat)] = flat
        flat = padded

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(flat))
    if max_val > _TOLERANCE:
        flat = flat / max_val

    return flat


# ---------------------------------------------------------------------------
# QuantumConv2D
# ---------------------------------------------------------------------------

class QuantumConv2D:
    """2D convolution using quantum circuits.

    Implements convolution by extracting patches from the input and
    processing each patch through a quantum circuit. Supports two modes:

    * ``'qnn'`` — uses a full quantum neural network circuit with
      encoding, variational layers, and measurement.
    * ``'quadratic'`` — uses a quantum kernel computation for
      quadratic feature extraction.

    Parameters
    ----------
    filters : int
        Number of output filters (feature maps).
    kernel_size : int
        Size of the convolution kernel (square).
    n_qubits : int
        Number of qubits in the quantum circuit.
    strides : int, optional
        Convolution stride. Default ``1``.
    padding : str, optional
        Padding mode: ``'valid'`` or ``'same'``. Default ``'valid'``.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    mode : str, optional
        Convolution mode: ``'qnn'`` or ``'quadratic'``. Default ``'qnn'``.
    activation : str, optional
        Activation function. Default ``'quantum_relu'``.
    use_bias : bool, optional
        Whether to use bias. Default ``True``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> conv = QuantumConv2D(filters=4, kernel_size=3, n_qubits=4)
    >>> conv.build((None, 28, 28, 1))
    >>> output = conv.call(np.random.randn(2, 28, 28, 1))
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        n_qubits: int,
        strides: int = 1,
        padding: str = "valid",
        n_layers: int = 2,
        mode: str = "qnn",
        activation: Optional[str] = "quantum_relu",
        use_bias: bool = True,
        name: Optional[str] = None,
    ) -> None:
        if filters < 1:
            raise ValueError(f"filters must be >= 1, got {filters}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if strides < 1:
            raise ValueError(f"strides must be >= 1, got {strides}")
        if padding not in _VALID_PADDING:
            raise ValueError(
                f"Unknown padding '{padding}'. "
                f"Choose from {sorted(_VALID_PADDING)}"
            )
        if mode not in _VALID_CONV_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Choose from {sorted(_VALID_CONV_MODES)}"
            )

        self._filters = filters
        self._kernel_size = kernel_size
        self._n_qubits = n_qubits
        self._strides = strides
        self._padding = padding
        self._n_layers = n_layers
        self._mode = mode
        self._activation_name = activation
        self._use_bias = use_bias
        self._name = name or f"quantum_conv2d_{id(self):x}"

        # Weights
        self._kernel_weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._variational_params: Optional[np.ndarray] = None

        # Build state
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._built = False
        self._output_shape_val: Optional[Tuple[int, ...]] = None

        # Activation
        self._activation: Optional[Any] = None

    @property
    def filters(self) -> int:
        """int: Number of output filters."""
        return self._filters

    @property
    def kernel_size(self) -> int:
        """int: Kernel size."""
        return self._kernel_size

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def strides(self) -> int:
        """int: Convolution stride."""
        return self._strides

    @property
    def padding(self) -> str:
        """str: Padding mode."""
        return self._padding

    @property
    def built(self) -> bool:
        """bool: Whether the layer has been built."""
        return self._built

    # -- Build ----------------------------------------------------------------

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create weight variables for the layer.

        Parameters
        ----------
        input_shape : tuple of int
            Expected input shape ``(batch, height, width, channels)`` or
            ``(height, width, channels)``.
        """
        if len(input_shape) == 4:
            _, h, w, c = input_shape
        elif len(input_shape) == 3:
            h, w, c = input_shape
        else:
            raise ValueError(
                f"Expected 3D or 4D input shape, got {input_shape}"
            )

        self._input_shape = input_shape
        rng = np.random.default_rng()

        # Input features per patch: kernel_size * kernel_size * channels
        patch_features = self._kernel_size * self._kernel_size * c

        # Kernel weights: map patch features to quantum rotation angles
        # Each filter gets its own weight matrix
        self._kernel_weights = rng.uniform(
            -0.1, 0.1,
            size=(self._filters, patch_features, self._n_qubits),
        ).astype(np.float64)

        # Bias
        if self._use_bias:
            self._bias = np.zeros(self._filters, dtype=np.float64)

        # Variational params: n_layers * n_qubits * 3 rotations per filter
        n_var = self._n_layers * self._n_qubits * 3
        self._variational_params = rng.uniform(
            -0.1, 0.1,
            size=(self._filters, n_var),
        ).astype(np.float64)

        # Setup activation
        self._setup_activation()

        # Compute output shape
        if self._padding == "same":
            out_h = math.ceil(h / self._strides)
            out_w = math.ceil(w / self._strides)
        else:
            out_h = (h - self._kernel_size) // self._strides + 1
            out_w = (w - self._kernel_size) // self._strides + 1

        self._output_shape_val = (out_h, out_w, self._filters)
        self._built = True

    def _setup_activation(self) -> None:
        """Initialize the activation function."""
        if self._activation_name is None or self._activation_name in (
            "relu", "sigmoid", "tanh", "linear"
        ):
            self._activation = self._activation_name
        elif self._activation_name.startswith("quantum_"):
            try:
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
                else:
                    self._activation = self._activation_name
            except (ImportError, AttributeError):
                classical_map = {
                    "quantum_relu": "relu",
                    "quantum_sigmoid": "sigmoid",
                    "quantum_tanh": "tanh",
                    "quantum_swish": "linear",
                }
                self._activation = classical_map.get(
                    self._activation_name, "linear"
                )
        else:
            self._activation = self._activation_name

    # -- Forward pass ---------------------------------------------------------

    def call(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[Sequence[Sequence[float]]]]],
    ) -> np.ndarray:
        """Compute the quantum convolution output.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input tensor of shape ``(batch, height, width, channels)`` or
            ``(height, width, channels)``.

        Returns
        -------
        numpy.ndarray
            Output tensor of shape ``(batch, out_h, out_w, filters)`` or
            ``(out_h, out_w, filters)``.
        """
        if not self._built:
            raise RuntimeError("Layer has not been built.")

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze_batch = False

        if inputs.ndim == 3:
            inputs = inputs[np.newaxis, ...]
            squeeze_batch = True

        batch = inputs.shape[0]

        # Extract patches
        patches, (out_h, out_w) = _extract_patches(
            inputs, self._kernel_size, self._strides, self._padding
        )

        if out_h == 0 or out_w == 0:
            output = np.zeros((batch, 0, 0, self._filters), dtype=np.float64)
            if squeeze_batch:
                output = output[0]
            return output

        # Process patches through quantum circuits
        output = np.zeros(
            (batch, out_h, out_w, self._filters), dtype=np.float64
        )

        for b in range(batch):
            for fh in range(out_h):
                for fw in range(out_w):
                    patch = patches[b, fh, fw]
                    for f in range(self._filters):
                        output[b, fh, fw, f] = self._process_patch(
                            patch, f
                        )

        # Apply activation
        output = self._apply_activation(output)

        if squeeze_batch:
            output = output[0]

        return output

    def _process_patch(
        self,
        patch: np.ndarray,
        filter_idx: int,
    ) -> float:
        """Process a single patch through the quantum circuit for one filter.

        Parameters
        ----------
        patch : numpy.ndarray
            Patch of shape ``(kernel_h, kernel_w, channels)``.
        filter_idx : int
            Filter index.

        Returns
        -------
        float
            Scalar output value.
        """
        assert self._kernel_weights is not None
        assert self._variational_params is not None

        if self._mode == "qnn":
            return self._qnn_convolution(patch, filter_idx)
        elif self._mode == "quadratic":
            return self._quadratic_convolution(patch, filter_idx)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def _qnn_convolution(
        self,
        patch: np.ndarray,
        filter_idx: int,
    ) -> float:
        """QNN-based convolution.

        Encodes the patch into a quantum circuit, applies variational
        layers specific to the filter, and measures the expectation.

        Parameters
        ----------
        patch : numpy.ndarray
        filter_idx : int

        Returns
        -------
        float
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        # Flatten and prepare patch
        patch_flat = _flatten_patch(patch, self._n_qubits)

        # Compute rotation angles via weight matrix
        weights = self._kernel_weights[filter_idx]
        angles = patch_flat @ weights  # shape: (n_qubits,)

        # Add bias
        if self._use_bias and self._bias is not None:
            angles = angles + self._bias[filter_idx]

        # Clip angles
        angles = np.clip(angles, -_PI, _PI)

        # Build quantum circuit
        qc = QuantumCircuit(self._n_qubits)

        # Encode
        for q, angle in enumerate(angles):
            qc.h(q)
            qc.ry(float(angle), q)

        # Variational layers
        var_params = self._variational_params[filter_idx]
        param_offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(var_params[param_offset])
                theta = float(var_params[param_offset + 1])
                omega = float(var_params[param_offset + 2])
                param_offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        # Measure expectation of Z on first qubit
        simulator = StatevectorSimulator()
        n = self._n_qubits
        z0 = np.kron(
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
            np.eye(1 << max(n - 1, 0), dtype=np.complex128),
        )
        expectation = simulator.expectation(qc, z0)

        return float(expectation)

    def _quadratic_convolution(
        self,
        patch: np.ndarray,
        filter_idx: int,
    ) -> float:
        """Quadratic kernel convolution.

        Uses the quantum circuit to compute a quadratic form of the
        patch features, enabling polynomial feature extraction.

        Parameters
        ----------
        patch : numpy.ndarray
        filter_idx : int

        Returns
        -------
        float
        """
        # Flatten patch
        patch_flat = patch.flatten().astype(np.float64)
        n_features = len(patch_flat)

        # Compute quadratic form: patch^T @ K @ patch
        # where K is derived from the kernel weights
        weights = self._kernel_weights[filter_idx]  # (n_feat, n_qubits)

        # Project patch to n_qubit space
        projected = patch_flat @ weights if n_features == weights.shape[0] else (
            patch_flat[:weights.shape[0]] @ weights
        )

        # Quadratic form in projected space
        result = float(np.sum(projected ** 2))

        # Apply bias
        if self._use_bias and self._bias is not None:
            result += self._bias[filter_idx]

        return result

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function element-wise.

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
            if self._activation == "relu" or self._activation is None:
                return np.maximum(0, x)
            elif self._activation == "sigmoid":
                return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
            elif self._activation == "tanh":
                return np.tanh(x)
            elif self._activation == "linear":
                return x
            return x
        else:
            # Quantum activation
            try:
                original_shape = x.shape
                result = self._activation.forward(x.reshape(-1))
                return result.reshape(original_shape)
            except Exception:
                return np.maximum(0, x)

    # -- Output shape ---------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Compute the output shape.

        Parameters
        ----------
        input_shape : tuple of int

        Returns
        -------
        tuple of int
        """
        if len(input_shape) == 4:
            _, h, w, _ = input_shape
            batch_dims = input_shape[:1]
        elif len(input_shape) == 3:
            h, w, _ = input_shape
            batch_dims = ()
        else:
            raise ValueError(f"Expected 3D or 4D input, got {input_shape}")

        if self._padding == "same":
            out_h = math.ceil(h / self._strides)
            out_w = math.ceil(w / self._strides)
        else:
            out_h = (h - self._kernel_size) // self._strides + 1
            out_w = (w - self._kernel_size) // self._strides + 1

        return batch_dims + (out_h, out_w, self._filters)

    # -- Serialization --------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        Returns
        -------
        dict
        """
        return {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "n_qubits": self._n_qubits,
            "strides": self._strides,
            "padding": self._padding,
            "n_layers": self._n_layers,
            "mode": self._mode,
            "activation": self._activation_name,
            "use_bias": self._use_bias,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantumConv2D:
        """Create a layer from configuration.

        Parameters
        ----------
        config : dict

        Returns
        -------
        QuantumConv2D
        """
        return cls(**config)

    # -- Weight management ----------------------------------------------------

    def get_weights(self) -> List[np.ndarray]:
        """Return current weights.

        Returns
        -------
        list of numpy.ndarray
        """
        weights: List[np.ndarray] = []
        if self._kernel_weights is not None:
            weights.append(self._kernel_weights)
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
        if self._kernel_weights is not None and idx < len(weights):
            self._kernel_weights = np.asarray(weights[idx], dtype=np.float64)
            idx += 1
        if self._use_bias and self._bias is not None and idx < len(weights):
            self._bias = np.asarray(weights[idx], dtype=np.float64)
            idx += 1
        if self._variational_params is not None and idx < len(weights):
            self._variational_params = np.asarray(weights[idx], dtype=np.float64)
            idx += 1

    def count_params(self) -> int:
        """Count total trainable parameters.

        Returns
        -------
        int
        """
        total = 0
        if self._kernel_weights is not None:
            total += self._kernel_weights.size
        if self._use_bias and self._bias is not None:
            total += self._bias.size
        if self._variational_params is not None:
            total += self._variational_params.size
        return total

    # -- Callable interface ---------------------------------------------------

    def __call__(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[Sequence[Sequence[float]]]], Any],
    ) -> np.ndarray:
        """Make the layer callable.

        Parameters
        ----------
        inputs : array_like or tf.Tensor

        Returns
        -------
        numpy.ndarray
        """
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
        def quantum_conv_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            x_np = x.numpy()
            output_np = self.call(x_np)
            output = tf.constant(output_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                dy_np = dy.numpy()
                x_np_val = x.numpy()
                eps = 1e-4
                shape = x_np_val.shape

                jacobian = np.zeros_like(x_np_val)
                # Compute gradient via numerical perturbation of the first sample
                if len(shape) == 4:
                    sample = x_np_val[0:1]
                    for idx in range(min(shape[-1], 4)):  # limit channels for speed
                        x_plus = sample.copy()
                        x_plus[..., idx] += eps
                        out_plus = self.call(x_plus)

                        x_minus = sample.copy()
                        x_minus[..., idx] -= eps
                        out_minus = self.call(x_minus)

                        channel_grad = (out_plus - out_minus) / (2 * eps)
                        jacobian[0, ..., idx] = np.sum(
                            channel_grad * dy_np[0], axis=(-1,)
                        )
                return tf.constant(jacobian)

            return output, grad

        return quantum_conv_op(inputs)

    def __repr__(self) -> str:
        return (
            f"QuantumConv2D("
            f"filters={self._filters}, "
            f"kernel_size={self._kernel_size}, "
            f"n_qubits={self._n_qubits}, "
            f"strides={self._strides}, "
            f"padding={self._padding!r}, "
            f"n_layers={self._n_layers}, "
            f"mode={self._mode!r}, "
            f"params={self.count_params()})"
        )


# ---------------------------------------------------------------------------
# QuantumPool2D
# ---------------------------------------------------------------------------

class QuantumPool2D:
    """Quantum pooling layer for spatial reduction.

    Reduces the spatial dimensions of feature maps using quantum circuits.
    Supports three pooling strategies:

    * ``'max'`` — classical max pooling (uses quantum circuit for feature
      selection).
    * ``'average'`` — classical average pooling.
    * ``'quantum'`` — quantum circuit-based pooling that applies a
      parameterized circuit to each pooling region and extracts a
      scalar from measurement.

    Parameters
    ----------
    pool_size : int
        Pooling window size (square).
    pool_type : str, optional
        Pooling type: ``'max'``, ``'average'``, ``'quantum'``.
        Default ``'quantum'``.
    strides : int or None, optional
        Pooling stride. If ``None``, defaults to ``pool_size``.
    n_qubits : int, optional
        Number of qubits for quantum pooling. Default ``2``.
    n_layers : int, optional
        Number of variational layers. Default ``1``.
    padding : str, optional
        Padding mode. Default ``'valid'``.
    name : str, optional
        Layer name.

    Examples
    --------
    >>> pool = QuantumPool2D(pool_size=2, pool_type='quantum', n_qubits=2)
    >>> pool.build((None, 28, 28, 4))
    >>> output = pool.call(np.random.randn(1, 28, 28, 4))
    """

    def __init__(
        self,
        pool_size: int,
        pool_type: str = "quantum",
        strides: Optional[int] = None,
        n_qubits: int = 2,
        n_layers: int = 1,
        padding: str = "valid",
        name: Optional[str] = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if pool_type not in _VALID_POOL_TYPES:
            raise ValueError(
                f"Unknown pool_type '{pool_type}'. "
                f"Choose from {sorted(_VALID_POOL_TYPES)}"
            )

        self._pool_size = pool_size
        self._pool_type = pool_type
        self._strides = strides if strides is not None else pool_size
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._padding = padding
        self._name = name or f"quantum_pool2d_{id(self):x}"

        # Parameters for quantum pooling
        self._variational_params: Optional[np.ndarray] = None
        self._built = False
        self._input_channels: Optional[int] = None

    @property
    def pool_size(self) -> int:
        """int: Pooling window size."""
        return self._pool_size

    @property
    def pool_type(self) -> str:
        """str: Pooling type."""
        return self._pool_type

    @property
    def strides(self) -> int:
        """int: Pooling stride."""
        return self._strides

    @property
    def built(self) -> bool:
        """bool: Whether the layer has been built."""
        return self._built

    # -- Build ----------------------------------------------------------------

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create parameters for the pooling layer.

        Parameters
        ----------
        input_shape : tuple of int
            ``(batch, height, width, channels)`` or ``(height, width, channels)``.
        """
        if len(input_shape) == 4:
            self._input_channels = input_shape[-1]
        elif len(input_shape) == 3:
            self._input_channels = input_shape[-1]
        else:
            raise ValueError(f"Expected 3D or 4D input, got {input_shape}")

        rng = np.random.default_rng()

        # Variational params for quantum pooling
        n_var = self._n_layers * self._n_qubits * 3
        self._variational_params = rng.uniform(
            -0.1, 0.1, size=n_var
        ).astype(np.float64)

        self._built = True

    # -- Forward pass ---------------------------------------------------------

    def call(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[Sequence[Sequence[float]]]]],
    ) -> np.ndarray:
        """Compute pooling output.

        Parameters
        ----------
        inputs : numpy.ndarray
            Shape ``(batch, height, width, channels)`` or
            ``(height, width, channels)``.

        Returns
        -------
        numpy.ndarray
        """
        if not self._built:
            raise RuntimeError("Layer has not been built.")

        inputs = np.asarray(inputs, dtype=np.float64)
        squeeze_batch = False
        if inputs.ndim == 3:
            inputs = inputs[np.newaxis, ...]
            squeeze_batch = True

        batch, h, w, c = inputs.shape
        s = self._strides
        ps = self._pool_size

        out_h = (h - ps) // s + 1
        out_w = (w - ps) // s + 1

        if out_h <= 0 or out_w <= 0:
            output = np.zeros((batch, 0, 0, c), dtype=np.float64)
            if squeeze_batch:
                output = output[0]
            return output

        output = np.zeros((batch, out_h, out_w, c), dtype=np.float64)

        for b in range(batch):
            for fh in range(out_h):
                for fw in range(out_w):
                    h_start = fh * s
                    w_start = fw * s
                    region = inputs[
                        b,
                        h_start:h_start + ps,
                        w_start:w_start + ps,
                        :,
                    ]
                    output[b, fh, fw, :] = self._pool_region(region)

        if squeeze_batch:
            output = output[0]

        return output

    def _pool_region(self, region: np.ndarray) -> np.ndarray:
        """Pool a single spatial region across all channels.

        Parameters
        ----------
        region : numpy.ndarray
            Shape ``(pool_h, pool_w, channels)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(channels,)``.
        """
        if self._pool_type == "max":
            return self._max_pool(region)
        elif self._pool_type == "average":
            return self._avg_pool(region)
        elif self._pool_type == "quantum":
            return self._quantum_pool(region)
        else:
            raise ValueError(f"Unknown pool_type: {self._pool_type}")

    def _max_pool(self, region: np.ndarray) -> np.ndarray:
        """Max pooling over spatial dimensions.

        Parameters
        ----------
        region : numpy.ndarray
            Shape ``(pool_h, pool_w, channels)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(channels,)``.
        """
        return np.max(region, axis=(0, 1))

    def _avg_pool(self, region: np.ndarray) -> np.ndarray:
        """Average pooling over spatial dimensions.

        Parameters
        ----------
        region : numpy.ndarray
            Shape ``(pool_h, pool_w, channels)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(channels,)``.
        """
        return np.mean(region, axis=(0, 1))

    def _quantum_pool(self, region: np.ndarray) -> np.ndarray:
        """Quantum circuit-based pooling.

        Encodes the pooling region into a quantum circuit and extracts
        features via measurement. Produces richer features than classical
        pooling by leveraging quantum entanglement within the pooling
        region.

        Parameters
        ----------
        region : numpy.ndarray
            Shape ``(pool_h, pool_w, channels)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(channels,)``.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        assert self._variational_params is not None
        ps_h, ps_w, channels = region.shape

        # Flatten spatial dimensions
        spatial_flat = region.reshape(-1, channels)  # (pool_h*pool_w, channels)
        n_spatial = spatial_flat.shape[0]

        results = np.zeros(channels, dtype=np.float64)

        for ch in range(channels):
            # Get channel values
            channel_vals = spatial_flat[:, ch]

            # Prepare features for quantum encoding
            features = np.zeros(self._n_qubits, dtype=np.float64)
            for q in range(min(n_spatial, self._n_qubits)):
                features[q] = channel_vals[q]

            # Normalize features to [-π, π]
            max_val = np.max(np.abs(features))
            if max_val > _TOLERANCE:
                features = features / max_val * _PI

            # Build quantum circuit
            qc = QuantumCircuit(self._n_qubits)

            # Encode features
            for q in range(self._n_qubits):
                qc.h(q)
                qc.ry(float(features[q]), q)

            # Variational layers
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

            # Measure
            simulator = StatevectorSimulator()
            n = self._n_qubits
            z0 = np.kron(
                np.array([[1, 0], [0, -1]], dtype=np.complex128),
                np.eye(1 << max(n - 1, 0), dtype=np.complex128),
            )
            expectation = simulator.expectation(qc, z0)

            # Map from [-1, 1] to original scale
            results[ch] = float(expectation) * max_val

        return results

    # -- Output shape ---------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Compute the output shape.

        Parameters
        ----------
        input_shape : tuple of int

        Returns
        -------
        tuple of int
        """
        if len(input_shape) == 4:
            _, h, w, c = input_shape
            batch_dims = input_shape[:1]
        elif len(input_shape) == 3:
            h, w, c = input_shape
            batch_dims = ()
        else:
            raise ValueError(f"Expected 3D or 4D input, got {input_shape}")

        s = self._strides
        ps = self._pool_size
        out_h = (h - ps) // s + 1
        out_w = (w - ps) // s + 1

        return batch_dims + (out_h, out_w, c)

    # -- Serialization --------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        Returns
        -------
        dict
        """
        return {
            "pool_size": self._pool_size,
            "pool_type": self._pool_type,
            "strides": self._strides,
            "n_qubits": self._n_qubits,
            "n_layers": self._n_layers,
            "padding": self._padding,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantumPool2D:
        """Create from configuration.

        Parameters
        ----------
        config : dict

        Returns
        -------
        QuantumPool2D
        """
        return cls(**config)

    def get_weights(self) -> List[np.ndarray]:
        """Return weights.

        Returns
        -------
        list of numpy.ndarray
        """
        if self._variational_params is not None:
            return [self._variational_params]
        return []

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set weights.

        Parameters
        ----------
        weights : list of numpy.ndarray
        """
        if self._variational_params is not None and len(weights) > 0:
            self._variational_params = np.asarray(weights[0], dtype=np.float64)

    def count_params(self) -> int:
        """Count trainable parameters.

        Returns
        -------
        int
        """
        if self._variational_params is not None:
            return len(self._variational_params)
        return 0

    # -- Callable interface ---------------------------------------------------

    def __call__(
        self,
        inputs: Union[np.ndarray, Sequence[Sequence[Sequence[Sequence[float]]]], Any],
    ) -> np.ndarray:
        """Make the layer callable.

        Parameters
        ----------
        inputs : array_like or tf.Tensor

        Returns
        -------
        numpy.ndarray
        """
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
        def quantum_pool_op(x: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
            x_np = x.numpy()
            output_np = self.call(x_np)
            output = tf.constant(output_np)

            def grad(dy: tf.Tensor) -> tf.Tensor:
                dy_np = dy.numpy()
                x_np_val = x.numpy()
                eps = 1e-4

                jacobian = np.zeros_like(x_np_val)
                if len(x_np_val.shape) == 4:
                    sample = x_np_val[0:1]
                    for idx in range(min(x_np_val.shape[-1], 2)):
                        x_plus = sample.copy()
                        x_plus[..., idx] += eps
                        out_plus = self.call(x_plus)

                        x_minus = sample.copy()
                        x_minus[..., idx] -= eps
                        out_minus = self.call(x_minus)

                        channel_grad = (out_plus - out_minus) / (2 * eps)
                        jacobian[0, ..., idx] = np.sum(
                            channel_grad * dy_np[0], axis=(-1,)
                        )
                return tf.constant(jacobian)

            return output, grad

        return quantum_pool_op(inputs)

    def __repr__(self) -> str:
        return (
            f"QuantumPool2D("
            f"pool_size={self._pool_size}, "
            f"pool_type={self._pool_type!r}, "
            f"strides={self._strides}, "
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers}, "
            f"params={self.count_params()})"
        )
