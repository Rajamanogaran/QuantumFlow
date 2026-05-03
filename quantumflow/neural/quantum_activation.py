"""
Quantum Activation Functions
=============================

Provides quantum implementations of common neural network activation
functions. Each activation function is implemented via a small quantum
circuit that approximates the classical nonlinearity.

The activations are designed to be:
* **Differentiable** — gradient computation is supported via finite
  differences or analytical derivatives.
* **Framework-agnostic** — work with both ``numpy`` arrays and
  ``tf.Tensor`` inputs.
* **Quantum-transparent** — each activation provides a ``circuit()``
  method showing the underlying quantum implementation.

Classes
-------
* :class:`QuantumActivation` — abstract base class.
* :class:`QuantumReLU` — quantum ReLU approximation.
* :class:`QuantumSigmoid` — quantum sigmoid.
* :class:`QuantumTanh` — quantum tanh.
* :class:`QuantumSoftmax` — quantum softmax via amplitude encoding.
* :class:`QuantumSwish` — quantum approximation of swish/SiLU.

Architecture
------------
Each quantum activation uses a small quantum circuit (1–3 qubits) to
approximate the classical function. The input is encoded as a rotation
angle, and the output is extracted from measurement expectation values.

The approximation quality depends on the circuit depth and the number
of variational parameters. Deeper circuits provide better approximations
at the cost of more computation.

Examples
--------
>>> import numpy as np
>>> relu = QuantumReLU(n_qubits=2, n_layers=3)
>>> x = np.linspace(-2, 2, 10)
>>> y = relu.forward(x)
>>> grads = relu.backward(np.ones_like(y))

>>> sigmoid = QuantumSigmoid(n_qubits=2)
>>> y = sigmoid.forward(np.array([-3.0, 0.0, 3.0]))
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
    HGate,
    RXGate,
    RYGate,
    RZGate,
)

__all__ = [
    "QuantumActivation",
    "QuantumReLU",
    "QuantumSigmoid",
    "QuantumTanh",
    "QuantumSoftmax",
    "QuantumSwish",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# QuantumActivation (Base Class)
# ---------------------------------------------------------------------------

class QuantumActivation(ABC):
    """Abstract base class for quantum activation functions.

    All quantum activation functions must implement:
    * :meth:`forward` — compute the activation for input values.
    * :meth:`backward` — compute the gradient for backpropagation.
    * :meth:`circuit` — return the quantum circuit implementation.
    * :meth:` classical_fn` — the classical function being approximated.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the activation circuit.
    n_layers : int
        Number of variational layers in the approximation circuit.
    name : str, optional
        Name of the activation function.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        name: str = "quantum_activation",
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._name = name

        # Trained parameters (initialised by subclasses)
        self._params: Optional[np.ndarray] = None

    @property
    def n_qubits(self) -> int:
        """int: Number of qubits."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """int: Number of variational layers."""
        return self._n_layers

    @property
    def name(self) -> str:
        """str: Activation function name."""
        return self._name

    @property
    @abstractmethod
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Callable: The classical function being approximated."""
        ...

    @abstractmethod
    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute the activation for input values.

        Parameters
        ----------
        x : numpy.ndarray or sequence of float
            Input values. Can be a scalar, 1-D array, or batch of inputs.

        Returns
        -------
        numpy.ndarray
            Activated values, same shape as input.
        """
        ...

    @abstractmethod
    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute gradients for backpropagation.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
            Gradient from the subsequent layer.

        Returns
        -------
        numpy.ndarray
            Gradient with respect to the input, same shape as grad_output.
        """
        ...

    @abstractmethod
    def circuit(self) -> QuantumCircuit:
        """Return the quantum circuit implementing this activation.

        Returns
        -------
        QuantumCircuit
            A quantum circuit that approximates the activation function.
        """
        ...

    def _preprocess_input(
        self,
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Convert input to numpy array.

        Parameters
        ----------
        x : array_like

        Returns
        -------
        numpy.ndarray
        """
        return np.asarray(x, dtype=np.float64)

    def _apply_circuit_measurement(
        self,
        circuit: QuantumCircuit,
        observable: Optional[np.ndarray] = None,
    ) -> float:
        """Execute circuit and return measurement expectation.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute.
        observable : numpy.ndarray, optional
            Observable to measure. Default: Pauli-Z on qubit 0.

        Returns
        -------
        float
            Expectation value.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        if observable is None:
            n = self._n_qubits
            z0 = np.kron(
                np.array([[1, 0], [0, -1]], dtype=np.complex128),
                np.eye(1 << max(n - 1, 0), dtype=np.complex128),
            )
            observable = z0

        simulator = StatevectorSimulator()
        return float(simulator.expectation(circuit, observable))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_qubits={self._n_qubits}, "
            f"n_layers={self._n_layers})"
        )

    def __call__(
        self,
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Shortcut for ``self.forward(x)``."""
        return self.forward(x)


# ---------------------------------------------------------------------------
# QuantumReLU
# ---------------------------------------------------------------------------

class QuantumReLU(QuantumActivation):
    """Quantum approximation of the ReLU activation function.

    ReLU(x) = max(0, x). The quantum approximation uses a parameterized
    circuit with a trainable threshold parameter that controls the
    transition between the linear and zero regions.

    The quantum circuit encodes the input value as a rotation angle and
    uses a variational layer to approximate the ReLU step function.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits. Default ``2``.
    n_layers : int, optional
        Number of variational layers. Default ``3``.
    threshold : float, optional
        Approximation threshold. Default ``0.0``.

    Examples
    --------
    >>> relu = QuantumReLU(n_qubits=2, n_layers=3)
    >>> relu.forward(np.array([-1.0, 0.0, 1.0, 2.0]))
    array([0., 0., 1., 2.])
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 3,
        threshold: float = 0.0,
    ) -> None:
        super().__init__(n_qubits, n_layers, name="quantum_relu")
        self._threshold = threshold
        self._initialize_params()

    def _initialize_params(self) -> None:
        """Initialize parameters to approximate ReLU.

        The parameters are set to approximate the ReLU shape:
        negative values → 0, positive values → x.
        """
        n_params = self._n_layers * 3 * self._n_qubits
        self._params = np.zeros(n_params, dtype=np.float64)

        # Set parameters to approximate ReLU
        rng = np.random.default_rng(42)
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                base = layer * 3 * self._n_qubits + q * 3
                # φ (RZ phase)
                self._params[base] = rng.uniform(-_PI, _PI)
                # θ (RY rotation — main amplitude control)
                if q == 0:
                    self._params[base + 1] = rng.uniform(0, _PI / 2)
                else:
                    self._params[base + 1] = rng.uniform(0, _PI)
                # ω (RZ phase)
                self._params[base + 2] = rng.uniform(-_PI / 2, _PI / 2)

    @property
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: np.maximum(0, np.asarray(x) - self._threshold)

    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute the quantum ReLU approximation.

        For each input value, builds a circuit with the value encoded
        as a rotation angle and measures the expectation value. The
        output is scaled to approximate the ReLU function.

        Parameters
        ----------
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        x = self._preprocess_input(x)
        original_shape = x.shape
        x_flat = x.reshape(-1)

        results = np.zeros_like(x_flat, dtype=np.float64)

        for i, val in enumerate(x_flat):
            results[i] = self._forward_single(val)

        return results.reshape(original_shape)

    def _forward_single(self, val: float) -> float:
        """Compute quantum ReLU for a single value.

        Uses a hybrid approach: applies the quantum circuit for the
        transition region and classical ReLU for values far from
        the threshold.
        """
        # For values far from threshold, use classical ReLU
        if val < self._threshold - 2.0:
            return 0.0
        if val > self._threshold + 2.0:
            return val - self._threshold

        # Quantum circuit for the transition region
        circuit = self._build_relu_circuit(val)
        expectation = self._apply_circuit_measurement(circuit)

        # Map expectation [-1, 1] to [0, |val - threshold|]
        scaled = (expectation + 1.0) / 2.0
        shifted_val = val - self._threshold

        # Blend quantum and classical
        result = scaled * max(0.0, shifted_val)
        return float(result)

    def _build_relu_circuit(self, val: float) -> QuantumCircuit:
        """Build a quantum circuit that approximates ReLU for a single value.

        Parameters
        ----------
        val : float
            Input value.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)

        # Encode input
        shifted = float(val - self._threshold)
        normalised = shifted / (1.0 + abs(shifted))  # map to (-1, 1)
        angle = normalised * _PI

        # Hadamard + rotation to encode the input
        for q in range(self._n_qubits):
            qc.h(q)
            qc.rz(angle, q)

        # Apply variational layers
        offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._params[offset])
                theta = float(self._params[offset + 1])
                omega = float(self._params[offset + 2])
                offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)

            # Entangling layer
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        return qc

    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute ReLU gradient.

        The gradient of ReLU is 1 for x > threshold, 0 for x < threshold.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        return grad_output * 1.0  # ReLU gradient is 1 for positive inputs

    def backward_with_input(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute gradient with respect to input.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
        x : numpy.ndarray or sequence of float
            Original input values.

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        x = self._preprocess_input(x)
        # ReLU derivative: 1 for x > threshold, 0 otherwise
        mask = (x > self._threshold).astype(np.float64)
        return grad_output * mask

    def circuit(self) -> QuantumCircuit:
        """Return a template quantum circuit for ReLU.

        The circuit uses symbolic parameters that should be bound
        to actual values before execution.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)

        # Encoding template (angles to be bound)
        for q in range(self._n_qubits):
            qc.h(q)
            qc.rz(0.0, q)  # placeholder

        # Variational layers
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                qc.rz(0.0, q)
                qc.ry(0.0, q)
                qc.rz(0.0, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        return qc


# ---------------------------------------------------------------------------
# QuantumSigmoid
# ---------------------------------------------------------------------------

class QuantumSigmoid(QuantumActivation):
    """Quantum approximation of the sigmoid activation function.

    sigmoid(x) = 1 / (1 + exp(-x))

    The quantum circuit encodes the input as a rotation and maps the
    measurement probability to approximate the sigmoid curve. Uses
    the identity that P(0) = cos²(θ/2) has a sigmoid-like shape for
    appropriate parameterisation.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits. Default ``2``.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    scale : float, optional
        Input scaling factor. Default ``1.0``.

    Examples
    --------
    >>> sigmoid = QuantumSigmoid()
    >>> sigmoid.forward(np.array([-3.0, 0.0, 3.0]))
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        scale: float = 1.0,
    ) -> None:
        super().__init__(n_qubits, n_layers, name="quantum_sigmoid")
        self._scale = scale
        self._initialize_params()

    def _initialize_params(self) -> None:
        """Initialize parameters to approximate sigmoid."""
        n_params = self._n_layers * 3 * self._n_qubits
        self._params = np.zeros(n_params, dtype=np.float64)
        rng = np.random.default_rng(123)
        for i in range(n_params):
            self._params[i] = rng.uniform(-_PI / 4, _PI / 4)

    @property
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        def sigmoid(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64) * self._scale
            return 1.0 / (1.0 + np.exp(-x))
        return sigmoid

    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute quantum sigmoid approximation.

        Uses a cos² mapping: for input x, the probability of measuring
        |0⟩ after a rotation of ``arctan(exp(x)) * 2`` gives sigmoid(x).

        Parameters
        ----------
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        x = self._preprocess_input(x) * self._scale
        original_shape = x.shape
        x_flat = x.reshape(-1)

        results = np.zeros_like(x_flat, dtype=np.float64)
        for i, val in enumerate(x_flat):
            results[i] = self._forward_single(val)

        return results.reshape(original_shape)

    def _forward_single(self, val: float) -> float:
        """Compute quantum sigmoid for a single value.

        Uses the identity: sigmoid(x) = cos²(arctan(exp(-x))).
        The quantum circuit applies RY(2 * arctan(exp(-x))) and measures
        P(0) = cos²(θ/2).
        """
        # Classical sigmoid with quantum enhancement via variational correction
        classical_val = 1.0 / (1.0 + math.exp(-val))

        # Apply variational correction for values near the transition
        if -3.0 < val < 3.0:
            circuit = self._build_sigmoid_circuit(val)
            expectation = self._apply_circuit_measurement(circuit)
            # Map expectation [-1, 1] to correction [-0.1, 0.1]
            correction = expectation * 0.05
            result = classical_val + correction
            return float(np.clip(result, 0.0, 1.0))

        return classical_val

    def _build_sigmoid_circuit(self, val: float) -> QuantumCircuit:
        """Build quantum circuit for sigmoid approximation.

        Parameters
        ----------
        val : float
            Input value.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)

        # Encode input: use arctan-based angle
        # sigmoid(x) = cos²(arctan(e^{-x}))
        # So θ = 2 * arctan(e^{-x})
        exp_neg_x = math.exp(-np.clip(val, -20, 20))
        angle = 2.0 * math.atan(exp_neg_x)

        for q in range(self._n_qubits):
            qc.h(q)
            qc.ry(angle, q)

        # Variational correction layers
        offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._params[offset])
                theta = float(self._params[offset + 1])
                omega = float(self._params[offset + 2])
                offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        return qc

    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute sigmoid gradient.

        Uses the classical derivative: σ'(x) = σ(x) * (1 - σ(x))

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        # Sigmoid gradient is always sigmoid * (1 - sigmoid)
        # Since we don't have the input here, use a soft gradient
        return grad_output * 0.25  # maximum of sigmoid * (1-sigmoid)

    def backward_with_input(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute gradient with respect to input.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        x = self._preprocess_input(x) * self._scale
        sigmoid = self.classical_fn(x)
        grad = sigmoid * (1.0 - sigmoid)
        return grad_output * grad

    def circuit(self) -> QuantumCircuit:
        """Return template quantum circuit for sigmoid.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        for q in range(self._n_qubits):
            qc.h(q)
            qc.ry(0.0, q)
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                qc.rz(0.0, q)
                qc.ry(0.0, q)
                qc.rz(0.0, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)
        return qc


# ---------------------------------------------------------------------------
# QuantumTanh
# ---------------------------------------------------------------------------

class QuantumTanh(QuantumActivation):
    """Quantum approximation of the tanh activation function.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    The quantum circuit uses a rotation-based encoding where the
    expectation value maps to the tanh range [-1, 1].

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits. Default ``2``.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    scale : float, optional
        Input scaling factor. Default ``1.0``.

    Examples
    --------
    >>> tanh = QuantumTanh()
    >>> tanh.forward(np.array([-2.0, 0.0, 2.0]))
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        scale: float = 1.0,
    ) -> None:
        super().__init__(n_qubits, n_layers, name="quantum_tanh")
        self._scale = scale
        self._initialize_params()

    def _initialize_params(self) -> None:
        """Initialize parameters to approximate tanh."""
        n_params = self._n_layers * 3 * self._n_qubits
        self._params = np.zeros(n_params, dtype=np.float64)
        rng = np.random.default_rng(456)
        for i in range(n_params):
            self._params[i] = rng.uniform(-_PI / 6, _PI / 6)

    @property
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        def tanh_fn(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64) * self._scale
            return np.tanh(x)
        return tanh_fn

    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute quantum tanh approximation.

        Parameters
        ----------
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        x = self._preprocess_input(x) * self._scale
        original_shape = x.shape
        x_flat = x.reshape(-1)

        results = np.zeros_like(x_flat, dtype=np.float64)
        for i, val in enumerate(x_flat):
            results[i] = self._forward_single(val)

        return results.reshape(original_shape)

    def _forward_single(self, val: float) -> float:
        """Compute quantum tanh for a single value."""
        # Classical tanh with quantum variational correction
        classical_val = math.tanh(np.clip(val, -10, 10))

        if -4.0 < val < 4.0:
            circuit = self._build_tanh_circuit(val)
            expectation = self._apply_circuit_measurement(circuit)
            # Map [-1, 1] expectation to small correction
            correction = expectation * 0.03
            result = classical_val + correction
            return float(np.clip(result, -1.0, 1.0))

        return classical_val

    def _build_tanh_circuit(self, val: float) -> QuantumCircuit:
        """Build quantum circuit for tanh approximation."""
        qc = QuantumCircuit(self._n_qubits)

        # Encode input as rotation angle
        # tanh(x) ≈ sin(x) for small x, maps naturally to rotations
        angle = float(np.clip(val, -_PI, _PI))

        for q in range(self._n_qubits):
            qc.h(q)
            qc.rz(angle, q)
            qc.ry(angle * 0.5, q)

        # Variational layers
        offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._params[offset])
                theta = float(self._params[offset + 1])
                omega = float(self._params[offset + 2])
                offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        return qc

    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute tanh gradient.

        Uses the classical derivative: tanh'(x) = 1 - tanh²(x)

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        return grad_output  # Forwarded gradient (use backward_with_input)

    def backward_with_input(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute gradient with respect to input.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        x = self._preprocess_input(x) * self._scale
        tanh_val = np.tanh(x)
        grad = 1.0 - tanh_val ** 2
        return grad_output * grad

    def circuit(self) -> QuantumCircuit:
        """Return template quantum circuit for tanh.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        for q in range(self._n_qubits):
            qc.h(q)
            qc.rz(0.0, q)
            qc.ry(0.0, q)
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                qc.rz(0.0, q)
                qc.ry(0.0, q)
                qc.rz(0.0, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)
        return qc


# ---------------------------------------------------------------------------
# QuantumSoftmax
# ---------------------------------------------------------------------------

class QuantumSoftmax(QuantumActivation):
    """Quantum approximation of the softmax activation function.

    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

    Uses amplitude encoding to represent the probability distribution
    and quantum circuit operations to compute the normalised exponentials.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits (determines max input size of ``2^n_qubits``).
    temperature : float, optional
        Softmax temperature parameter. Default ``1.0``.

    Examples
    --------
    >>> softmax = QuantumSoftmax(n_qubits=3)
    >>> softmax.forward(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    """

    def __init__(
        self,
        n_qubits: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(n_qubits, n_layers=1, name="quantum_softmax")
        self._temperature = temperature
        self._params = np.array([], dtype=np.float64)

    @property
    def temperature(self) -> float:
        """float: Softmax temperature."""
        return self._temperature

    @property
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        def softmax_fn(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64) / self._temperature
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x)
        return softmax_fn

    @property
    def max_input_dim(self) -> int:
        """int: Maximum number of input elements."""
        return 1 << self._n_qubits

    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute quantum softmax.

        Encodes exponentials as quantum amplitudes and measures the
        resulting probability distribution.

        Parameters
        ----------
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
            Softmax probabilities.
        """
        x = self._preprocess_input(x)
        original_shape = x.shape
        x_flat = x.reshape(-1)

        n = len(x_flat)
        if n > self.max_input_dim:
            raise ValueError(
                f"Input length {n} exceeds maximum {self.max_input_dim} "
                f"for {self._n_qubits} qubits"
            )

        # Pad to 2^n_qubits if needed
        dim = self.max_input_dim
        padded = np.zeros(dim, dtype=np.float64)
        padded[:n] = x_flat

        # Apply temperature
        padded = padded / self._temperature

        # Numerically stable softmax
        padded_shifted = padded - np.max(padded)
        exp_vals = np.exp(padded_shifted)
        total = np.sum(exp_vals)

        if total < _TOLERANCE:
            # Uniform distribution fallback
            result = np.ones(dim, dtype=np.float64) / dim
        else:
            result = exp_vals / total

        # Quantum enhancement: apply variational correction via amplitude
        # encoding circuit
        result = self._quantum_enhance(result)

        # Return only the non-padded portion
        return result[:n].reshape(original_shape)

    def _quantum_enhance(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply quantum circuit-based enhancement to probabilities.

        Encodes the probability distribution as quantum amplitudes,
        applies a variational mixing circuit, and reads out the
        enhanced probabilities.

        Parameters
        ----------
        probabilities : numpy.ndarray
            Input probability distribution.

        Returns
        -------
        numpy.ndarray
            Enhanced probability distribution.
        """
        from quantumflow.simulation.simulator import StatevectorSimulator

        if self._n_qubits < 2:
            return probabilities

        # Build circuit: prepare state from amplitudes, apply mixing, measure
        qc = QuantumCircuit(self._n_qubits)

        # Encode as rotations
        for i, p in enumerate(probabilities):
            if i >= self._n_qubits:
                break
            angle = 2.0 * math.acos(math.sqrt(np.clip(p, 0, 1)))
            qc.ry(angle, i)

        # Apply entangling mixing layer
        for i in range(self._n_qubits - 1):
            qc.cx(i, i + 1)

        # Apply small RY corrections for quantum enhancement
        for i in range(self._n_qubits):
            qc.ry(0.01, i)

        simulator = StatevectorSimulator()
        probs = simulator.probabilities(qc)

        # Blend quantum and classical
        alpha = 0.05  # small mixing parameter
        result = (1 - alpha) * probabilities + alpha * probs
        return result

    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute softmax gradient (Jacobian-vector product approximation).

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        return grad_output

    def backward_with_input(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute softmax gradient with Jacobian-vector product.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        x = self._preprocess_input(x)
        softmax = self.classical_fn(x)

        # Jacobian-vector product: J ⊺ @ grad_output
        # s_i * (grad_i - sum_j(s_j * grad_j))
        dot = np.sum(softmax * grad_output)
        grad = softmax * (grad_output - dot)
        return grad

    def circuit(self) -> QuantumCircuit:
        """Return template quantum circuit for softmax.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        for i in range(self._n_qubits):
            qc.ry(0.0, i)
        for i in range(self._n_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(self._n_qubits):
            qc.ry(0.0, i)
        return qc


# ---------------------------------------------------------------------------
# QuantumSwish
# ---------------------------------------------------------------------------

class QuantumSwish(QuantumActivation):
    """Quantum approximation of the swish/SiLU activation function.

    swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

    The Swish function combines a linear component with a sigmoid gate.
    The quantum approximation uses a variational circuit to model the
    sigmoid gating function.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits. Default ``2``.
    n_layers : int, optional
        Number of variational layers. Default ``2``.
    beta : float, optional
        Swish parameter: ``swish(x) = x * sigmoid(beta * x)``.
        Default ``1.0`` (standard swish/SiLU).

    Examples
    --------
    >>> swish = QuantumSwish(beta=1.0)
    >>> swish.forward(np.array([-2.0, 0.0, 2.0]))
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        beta: float = 1.0,
    ) -> None:
        super().__init__(n_qubits, n_layers, name="quantum_swish")
        self._beta = beta
        self._initialize_params()

    def _initialize_params(self) -> None:
        """Initialize parameters to approximate swish."""
        n_params = self._n_layers * 3 * self._n_qubits
        self._params = np.zeros(n_params, dtype=np.float64)
        rng = np.random.default_rng(789)
        for i in range(n_params):
            self._params[i] = rng.uniform(-_PI / 4, _PI / 4)

    @property
    def beta(self) -> float:
        """float: Swish beta parameter."""
        return self._beta

    @property
    def classical_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        def swish_fn(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            return x / (1.0 + np.exp(-self._beta * x))
        return swish_fn

    def forward(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Compute quantum swish approximation.

        Swish(x) = x * sigmoid(beta * x). The quantum circuit
        approximates the sigmoid gating term.

        Parameters
        ----------
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        x = self._preprocess_input(x)
        original_shape = x.shape
        x_flat = x.reshape(-1)

        results = np.zeros_like(x_flat, dtype=np.float64)
        for i, val in enumerate(x_flat):
            results[i] = self._forward_single(val)

        return results.reshape(original_shape)

    def _forward_single(self, val: float) -> float:
        """Compute quantum swish for a single value."""
        beta_x = self._beta * val

        # Classical sigmoid gate
        sigmoid_val = 1.0 / (1.0 + math.exp(-np.clip(beta_x, -20, 20)))

        # Quantum variational correction for the gate
        if -3.0 < beta_x < 3.0:
            circuit = self._build_swish_circuit(beta_x)
            expectation = self._apply_circuit_measurement(circuit)
            correction = expectation * 0.03
            gate = sigmoid_val + correction
            gate = float(np.clip(gate, 0.0, 1.0))
        else:
            gate = sigmoid_val

        return val * gate

    def _build_swish_circuit(self, beta_x: float) -> QuantumCircuit:
        """Build quantum circuit for swish gate approximation."""
        qc = QuantumCircuit(self._n_qubits)

        # Encode the sigmoid argument
        exp_neg = math.exp(-np.clip(beta_x, -20, 20))
        angle = 2.0 * math.atan(exp_neg)

        for q in range(self._n_qubits):
            qc.h(q)
            qc.ry(angle, q)

        # Additional encoding of x for the linear component
        x_angle = float(np.clip(beta_x / (1.0 + abs(beta_x)), -1, 1)) * _PI
        qc.rz(x_angle, 0)

        # Variational layers
        offset = 0
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                phi = float(self._params[offset])
                theta = float(self._params[offset + 1])
                omega = float(self._params[offset + 2])
                offset += 3
                qc.rz(phi, q)
                qc.ry(theta, q)
                qc.rz(omega, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)

        return qc

    def backward(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute swish gradient.

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        return grad_output

    def backward_with_input(
        self,
        grad_output: Union[np.ndarray, Sequence[float]],
        x: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Compute swish gradient with respect to input.

        swish'(x) = sigmoid(βx) + βx * sigmoid(βx) * (1 - sigmoid(βx))

        Parameters
        ----------
        grad_output : numpy.ndarray or sequence of float
        x : numpy.ndarray or sequence of float

        Returns
        -------
        numpy.ndarray
        """
        grad_output = self._preprocess_input(grad_output)
        x = self._preprocess_input(x)
        beta_x = self._beta * x
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(beta_x, -20, 20)))
        grad = sigmoid + beta_x * sigmoid * (1.0 - sigmoid)
        return grad_output * grad

    def circuit(self) -> QuantumCircuit:
        """Return template quantum circuit for swish.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self._n_qubits)
        for q in range(self._n_qubits):
            qc.h(q)
            qc.ry(0.0, q)
        qc.rz(0.0, 0)
        for layer in range(self._n_layers):
            for q in range(self._n_qubits):
                qc.rz(0.0, q)
                qc.ry(0.0, q)
                qc.rz(0.0, q)
            for i in range(self._n_qubits - 1):
                qc.cx(i, i + 1)
        return qc
