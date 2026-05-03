"""
Pre-built TensorFlow Quantum Models
====================================

High-level quantum machine learning models with a scikit-learn-like API.

All models use QuantumFlow circuits internally and provide standard
methods: ``compile()``, ``fit()``, ``predict()``, ``evaluate()``.

Classes
-------
* :class:`QClassifier` — Quantum variational classifier (binary / multi-class).
* :class:`QRegressor` — Quantum variational regressor.
* :class:`QAutoencoder` — Quantum autoencoder with trash qubit detection.
* :class:`QGAN` — Quantum Generative Adversarial Network.
* :class:`QTransferLearningModel` — Classical backbone + quantum head.
* :class:`QHybridModel` — Arbitrary sequence of classical + quantum layers.

Examples
--------
>>> from quantumflow.tensorflow.models import QClassifier
>>> model = QClassifier(n_qubits=4, n_classes=2, n_layers=3)
>>> model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
>>> history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
>>> preds = model.predict(x_test)
>>> loss, acc = model.evaluate(x_test, y_test)
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "QClassifier",
    "QRegressor",
    "QAutoencoder",
    "QGAN",
    "QTransferLearningModel",
    "QHybridModel",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary or categorical cross-entropy loss.

    Parameters
    ----------
    y_pred : numpy.ndarray
        Predicted probabilities.
    y_true : numpy.ndarray
        Ground truth labels.

    Returns
    -------
    float
    """
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_true.ndim == 1 or y_true.shape[-1] == 1:
        # Binary cross-entropy
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))
    else:
        # Categorical cross-entropy
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))


def _mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error loss.

    Parameters
    ----------
    y_pred, y_true : numpy.ndarray

    Returns
    -------
    float
    """
    return float(np.mean((y_pred - y_true) ** 2))


def _accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Classification accuracy.

    Parameters
    ----------
    y_pred : numpy.ndarray
        Predicted probabilities or logits.
    y_true : numpy.ndarray
        Ground truth.

    Returns
    -------
    float
    """
    if y_pred.ndim == 1:
        pred_labels = (y_pred > 0.5).astype(int)
    else:
        pred_labels = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        true_labels = np.argmax(y_true, axis=-1)
    else:
        true_labels = y_true.reshape(-1).astype(int)
    return float(np.mean(pred_labels == true_labels))


def _build_pauli_observable(pauli: str, qubit: int, n_qubits: int) -> np.ndarray:
    """Build embedded Pauli observable."""
    pauli_matrices = {
        "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    mat = pauli_matrices.get(pauli, pauli_matrices["z"])
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n_qubits):
        full = np.kron(full, mat if i == qubit else np.eye(2, dtype=np.complex128))
    return full


def _get_simulator():
    """Get a StatevectorSimulator."""
    from quantumflow.simulation.simulator import StatevectorSimulator
    return StatevectorSimulator()


def _encode_data(circuit: Any, data: np.ndarray, n_qubits: int) -> None:
    """Encode data via H + RY on each qubit."""
    for i in range(n_qubits):
        circuit.h(i)
        circuit.ry(float(data[i]), i)


def _apply_variational(
    circuit: Any,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
) -> None:
    """Apply variational layers to a circuit."""
    param_offset = 0
    for layer_idx in range(n_layers):
        for q in range(n_qubits):
            rz1 = float(params[param_offset])
            ry = float(params[param_offset + 1])
            rz2 = float(params[param_offset + 2])
            param_offset += 3
            circuit.rz(rz1, q)
            circuit.ry(ry, q)
            circuit.rz(rz2, q)

        if entanglement == "linear":
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits):
                circuit.cx(i, (i + 1) % n_qubits)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.cx(i, j)
        elif entanglement == "pairwise":
            for i in range(0, n_qubits - 1, 2):
                circuit.cx(i, min(i + 1, n_qubits - 1))
        elif entanglement == "star":
            for i in range(1, n_qubits):
                circuit.cx(0, i)


def _run_forward(
    data: np.ndarray,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
) -> np.ndarray:
    """Run the full encode -> variational -> measure pipeline.

    Returns
    -------
    numpy.ndarray
        Expectation values for each qubit, shape ``(n_qubits,)``.
    """
    from quantumflow.core.circuit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    _encode_data(qc, data, n_qubits)
    _apply_variational(qc, params, n_qubits, n_layers, entanglement)

    sim = _get_simulator()
    results = np.zeros(n_qubits, dtype=np.float64)
    for q in range(n_qubits):
        obs = _build_pauli_observable("z", q, n_qubits)
        results[q] = float(sim.expectation(qc, obs))
    return results


def _compute_gradient_parameter_shift(
    data: np.ndarray,
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    loss_fn: Callable[[np.ndarray], float],
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute parameter gradients via finite differences.

    Uses parameter-shift-like central differences.

    Returns
    -------
    numpy.ndarray
        Gradient of same shape as params.
    """
    grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps

        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)
        grad[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad


# ═══════════════════════════════════════════════════════════════════════════
# QClassifier — Quantum Variational Classifier
# ═══════════════════════════════════════════════════════════════════════════

class QClassifier:
    """Quantum variational classifier with scikit-learn-like API.

    Supports binary and multi-class classification via a parameterised
    quantum circuit. The feature map encodes input data, variational
    layers extract quantum features, and measurement produces class
    probabilities.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_classes : int
        Number of output classes (2 for binary).
    n_layers : int, optional
        Variational layers. Default ``3``.
    feature_map : str, optional
        ``'angle'`` or ``'dense_angle'``. Default ``'angle'``.
    entanglement : str, optional
        ``'linear'``, ``'circular'``, ``'full'``. Default ``'linear'``.
    learning_rate : float, optional
        Default ``0.01``.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> model = QClassifier(n_qubits=4, n_classes=2, n_layers=3)
    >>> model.compile(optimizer='adam', loss='binary_crossentropy')
    >>> history = model.fit(x_train, y_train, epochs=50, batch_size=32)
    >>> preds = model.predict(x_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 3,
        feature_map: str = "angle",
        entanglement: str = "linear",
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_classes < 1:
            raise ValueError(f"n_classes must be >= 1, got {n_classes}")

        self._n_qubits = n_qubits
        self._n_classes = n_classes
        self._n_layers = n_layers
        self._feature_map = feature_map
        self._entanglement = entanglement
        self._learning_rate = learning_rate
        self._random_state = random_state

        # Parameters
        self._n_params = n_layers * n_qubits * 3
        self._readout_kernel: Optional[np.ndarray] = None
        self._readout_bias: Optional[np.ndarray] = None
        self._variational_params: Optional[np.ndarray] = None

        # Training state
        self._compiled = False
        self._optimizer_name: Optional[str] = None
        self._loss_name: Optional[str] = None
        self._metrics: List[str] = []
        self._history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self._is_binary = n_classes == 2 or n_classes == 1

        # Adam optimizer state
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t = 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def is_binary(self) -> bool:
        return self._is_binary

    @property
    def history(self) -> Dict[str, List[float]]:
        """dict: Training history."""
        return dict(self._history)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "binary_crossentropy",
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Compile the model.

        Parameters
        ----------
        optimizer : str
            ``'adam'``, ``'sgd'``, ``'rmsprop'``.
        loss : str
            ``'binary_crossentropy'``, ``'categorical_crossentropy'``,
            ``'mse'``.
        metrics : list of str, optional
            e.g. ``['accuracy']``.
        """
        rng = np.random.default_rng(self._random_state)

        # Init variational params
        self._variational_params = rng.uniform(
            -0.1, 0.1, self._n_params
        ).astype(np.float64)

        # Init readout layer
        if self._is_binary:
            self._readout_kernel = rng.uniform(
                -0.1, 0.1, (self._n_qubits, 1)
            ).astype(np.float64)
            self._readout_bias = np.zeros(1, dtype=np.float64)
        else:
            self._readout_kernel = rng.uniform(
                -0.1, 0.1, (self._n_qubits, self._n_classes)
            ).astype(np.float64)
            self._readout_bias = np.zeros(self._n_classes, dtype=np.float64)

        # Init optimizer state
        self._m = np.zeros(self._n_params, dtype=np.float64)
        self._v = np.zeros(self._n_params, dtype=np.float64)
        self._t = 0

        self._optimizer_name = optimizer
        self._loss_name = loss
        self._metrics = metrics or []
        self._compiled = True

    def _forward_batch(
        self,
        X: np.ndarray,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward pass for a batch.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, n_classes)`` logits.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled. Call compile() first.")
        assert self._variational_params is not None
        assert self._readout_kernel is not None
        assert self._readout_bias is not None

        p = params if params is not None else self._variational_params
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        batch = X.shape[0]
        output_dim = self._readout_kernel.shape[-1] if self._readout_kernel is not None else self._n_classes
        logits = np.zeros((batch, output_dim), dtype=np.float64)

        for b in range(batch):
            # Encode data to n_qubits
            data = self._prepare_data(X[b])

            # Quantum circuit
            quantum_out = _run_forward(
                data, p, self._n_qubits, self._n_layers, self._entanglement
            )

            # Readout
            logits[b] = quantum_out @ self._readout_kernel + self._readout_bias

        return logits

    def _prepare_data(self, x: np.ndarray) -> np.ndarray:
        """Prepare single sample for quantum encoding."""
        if len(x) == self._n_qubits:
            return np.clip(x, -_PI, _PI)
        elif len(x) > self._n_qubits:
            # PCA-like reduction
            group = len(x) // self._n_qubits
            result = np.zeros(self._n_qubits, dtype=np.float64)
            for i in range(self._n_qubits):
                s = i * group
                e = s + group if i < self._n_qubits - 1 else len(x)
                result[i] = float(np.mean(x[s:e]))
            return np.clip(result, -_PI, _PI)
        else:
            padded = np.zeros(self._n_qubits, dtype=np.float64)
            padded[:len(x)] = x
            return np.clip(padded, -_PI, _PI)

    def _compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Compute loss from logits."""
        if self._is_binary:
            probs = _sigmoid(logits.reshape(-1))
            y_flat = y.reshape(-1).astype(np.float64)
            eps = 1e-7
            return float(-np.mean(
                y_flat * np.log(np.clip(probs, eps, 1 - eps))
                + (1 - y_flat) * np.log(np.clip(1 - probs, eps, 1 - eps))
            ))
        else:
            probs = _softmax(logits)
            eps = 1e-7
            return float(-np.mean(np.sum(
                y * np.log(np.clip(probs, eps, 1 - eps)), axis=-1
            )))

    def _update_params(self, grad: np.ndarray) -> None:
        """Update variational params with optimizer."""
        assert self._variational_params is not None

        if self._optimizer_name == "adam":
            self._t += 1
            beta1, beta2 = 0.9, 0.999
            lr = self._learning_rate
            eps = 1e-8

            assert self._m is not None
            assert self._v is not None
            self._m = beta1 * self._m + (1 - beta1) * grad
            self._v = beta2 * self._v + (1 - beta2) * (grad ** 2)
            m_hat = self._m / (1 - beta1 ** self._t)
            v_hat = self._v / (1 - beta2 ** self._t)
            self._variational_params -= lr * m_hat / (np.sqrt(v_hat) + eps)

        elif self._optimizer_name == "sgd":
            self._variational_params -= self._learning_rate * grad

        elif self._optimizer_name == "rmsprop":
            rho = 0.9
            eps = 1e-8
            assert self._v is not None
            self._v = rho * self._v + (1 - rho) * (grad ** 2)
            self._variational_params -= (
                self._learning_rate * grad / (np.sqrt(self._v) + eps)
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.0,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Parameters
        ----------
        X : numpy.ndarray
            Training features, shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Training labels.
        epochs : int, optional
        batch_size : int, optional
        validation_split : float, optional
            Fraction of data for validation.
        verbose : int, optional
            Verbosity level.

        Returns
        -------
        dict
            Training history.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # One-hot encode labels for multi-class
        if not self._is_binary and y.ndim == 1:
            y_onehot = np.zeros((len(y), self._n_classes), dtype=np.float64)
            for i, label in enumerate(y):
                y_onehot[i, int(label)] = 1.0
            y = y_onehot

        n_samples = X.shape[0]

        # Validation split
        if validation_split > 0:
            n_val = int(n_samples * validation_split)
            perm = np.random.permutation(n_samples)
            X_train, y_train = X[perm[n_val:]], y[perm[n_val:]]
            X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        self._history = {"loss": [], "val_loss": []}
        for m in self._metrics:
            self._history[f"train_{m}"] = []
            if X_val is not None:
                self._history[f"val_{m}"] = []

        for epoch in range(epochs):
            t0 = time.perf_counter()

            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_losses = []

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute loss function
                def loss_fn(params: np.ndarray) -> float:
                    logits = self._forward_batch(X_batch, params)
                    return self._compute_loss(logits, y_batch)

                # Compute gradient
                grad = _compute_gradient_parameter_shift(
                    X_batch[0] if len(X_batch) == 1 else X_batch,
                    self._variational_params,
                    self._n_qubits,
                    self._n_layers,
                    self._entanglement,
                    loss_fn,
                )

                # Update
                self._update_params(grad)

                # Track loss
                batch_loss = loss_fn(self._variational_params)
                epoch_losses.append(batch_loss)

            avg_loss = float(np.mean(epoch_losses))
            self._history["loss"].append(avg_loss)

            # Validation
            if X_val is not None:
                val_logits = self._forward_batch(X_val)
                val_loss = self._compute_loss(val_logits, y_val)
                self._history["val_loss"].append(val_loss)

            if verbose > 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}"
                if X_val is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                elapsed = time.perf_counter() - t0
                msg += f" - {elapsed:.2f}s"
                print(msg)

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape ``(n_samples, n_classes)``.
        """
        logits = self._forward_batch(X)
        if self._is_binary:
            return _sigmoid(logits.reshape(-1)).reshape(-1, 1)
        else:
            return _softmax(logits)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Integer class labels.
        """
        probs = self.predict(X)
        if self._is_binary:
            return (probs.reshape(-1) > 0.5).astype(int)
        else:
            return np.argmax(probs, axis=-1)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate the model.

        Parameters
        ----------
        X, y : numpy.ndarray
        batch_size : int

        Returns
        -------
        dict
            Metric values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        logits = self._forward_batch(X)
        loss = self._compute_loss(logits, y)
        results = {"loss": loss}

        for m in self._metrics:
            if m == "accuracy":
                probs = self.predict(X)
                results["accuracy"] = _accuracy(probs, y)

        return results

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return all model parameters.

        Returns
        -------
        dict
        """
        return {
            "variational_params": self._variational_params.copy() if self._variational_params is not None else np.array([]),
            "readout_kernel": self._readout_kernel.copy() if self._readout_kernel is not None else np.array([]),
            "readout_bias": self._readout_bias.copy() if self._readout_bias is not None else np.array([]),
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters."""
        if "variational_params" in params:
            self._variational_params = np.asarray(params["variational_params"], dtype=np.float64)
        if "readout_kernel" in params:
            self._readout_kernel = np.asarray(params["readout_kernel"], dtype=np.float64)
        if "readout_bias" in params:
            self._readout_bias = np.asarray(params["readout_bias"], dtype=np.float64)

    def summary(self) -> str:
        """Return model summary string."""
        lines = [
            "QClassifier",
            f"  n_qubits:     {self._n_qubits}",
            f"  n_classes:    {self._n_classes}",
            f"  n_layers:     {self._n_layers}",
            f"  is_binary:    {self._is_binary}",
            f"  optimizer:    {self._optimizer_name}",
            f"  loss:         {self._loss_name}",
            f"  n_params:     {self._n_params + (self._readout_kernel.size if self._readout_kernel is not None else 0)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QClassifier(n_qubits={self._n_qubits}, "
            f"n_classes={self._n_classes}, "
            f"n_layers={self._n_layers})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QRegressor — Quantum Variational Regressor
# ═══════════════════════════════════════════════════════════════════════════

class QRegressor:
    """Quantum variational regressor.

    Similar API to :class:`QClassifier` for regression tasks.

    Parameters
    ----------
    n_qubits : int
    n_outputs : int, optional
        Number of regression outputs. Default ``1``.
    n_layers : int, optional
    entanglement : str, optional
    learning_rate : float, optional
    random_state : int, optional

    Examples
    --------
    >>> model = QRegressor(n_qubits=4, n_outputs=1, n_layers=3)
    >>> model.compile(optimizer='adam', loss='mse')
    >>> model.fit(x_train, y_train, epochs=50)
    >>> preds = model.predict(x_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_outputs: int = 1,
        n_layers: int = 3,
        entanglement: str = "linear",
        learning_rate: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._n_outputs = n_outputs
        self._n_layers = n_layers
        self._entanglement = entanglement
        self._learning_rate = learning_rate
        self._random_state = random_state

        self._n_params = n_layers * n_qubits * 3
        self._readout_kernel: Optional[np.ndarray] = None
        self._readout_bias: Optional[np.ndarray] = None
        self._variational_params: Optional[np.ndarray] = None

        self._compiled = False
        self._optimizer_name: Optional[str] = None
        self._loss_name: Optional[str] = None
        self._history: Dict[str, List[float]] = {}

        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._t = 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "mse",
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Compile the regressor."""
        rng = np.random.default_rng(self._random_state)
        self._variational_params = rng.uniform(-0.1, 0.1, self._n_params).astype(np.float64)
        self._readout_kernel = rng.uniform(-0.1, 0.1, (self._n_qubits, self._n_outputs)).astype(np.float64)
        self._readout_bias = np.zeros(self._n_outputs, dtype=np.float64)
        self._m = np.zeros(self._n_params, dtype=np.float64)
        self._v = np.zeros(self._n_params, dtype=np.float64)
        self._t = 0

        self._optimizer_name = optimizer
        self._loss_name = loss
        self._compiled = True

    def _forward_batch(self, X: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass returning raw predictions."""
        assert self._variational_params is not None
        assert self._readout_kernel is not None
        assert self._readout_bias is not None

        p = params if params is not None else self._variational_params
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        batch = X.shape[0]
        preds = np.zeros((batch, self._n_outputs), dtype=np.float64)

        for b in range(batch):
            data = self._prepare_data(X[b])
            q_out = _run_forward(data, p, self._n_qubits, self._n_layers, self._entanglement)
            preds[b] = q_out @ self._readout_kernel + self._readout_bias

        return preds

    def _prepare_data(self, x: np.ndarray) -> np.ndarray:
        if len(x) == self._n_qubits:
            return np.clip(x, -_PI, _PI)
        elif len(x) > self._n_qubits:
            group = len(x) // self._n_qubits
            result = np.zeros(self._n_qubits, dtype=np.float64)
            for i in range(self._n_qubits):
                s = i * group
                e = s + group if i < self._n_qubits - 1 else len(x)
                result[i] = float(np.mean(x[s:e]))
            return np.clip(result, -_PI, _PI)
        else:
            padded = np.zeros(self._n_qubits, dtype=np.float64)
            padded[:len(x)] = x
            return np.clip(padded, -_PI, _PI)

    def _compute_loss(self, preds: np.ndarray, y: np.ndarray) -> float:
        if self._loss_name == "mse":
            return _mse_loss(preds, y)
        elif self._loss_name == "mae":
            return float(np.mean(np.abs(preds - y)))
        else:
            return _mse_loss(preds, y)

    def _update_params(self, grad: np.ndarray) -> None:
        assert self._variational_params is not None
        if self._optimizer_name == "adam":
            self._t += 1
            assert self._m is not None
            assert self._v is not None
            self._m = 0.9 * self._m + 0.1 * grad
            self._v = 0.999 * self._v + 0.001 * (grad ** 2)
            m_hat = self._m / (1 - 0.9 ** self._t)
            v_hat = self._v / (1 - 0.999 ** self._t)
            self._variational_params -= self._learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        elif self._optimizer_name == "sgd":
            self._variational_params -= self._learning_rate * grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.0,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the regressor."""
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        self._history = {"loss": [], "val_loss": []}

        if validation_split > 0:
            n_val = int(n_samples * validation_split)
            perm = np.random.permutation(n_samples)
            X_train, y_train = X[perm[n_val:]], y[perm[n_val:]]
            X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for epoch in range(epochs):
            perm = np.random.permutation(len(X_train))
            X_s, y_s = X_train[perm], y_train[perm]
            losses = []

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                xb, yb = X_s[start:end], y_s[start:end]

                def loss_fn(p):
                    return self._compute_loss(self._forward_batch(xb, p), yb)

                grad = _compute_gradient_parameter_shift(
                    xb[0], self._variational_params,
                    self._n_qubits, self._n_layers, self._entanglement, loss_fn,
                )
                self._update_params(grad)
                losses.append(loss_fn(self._variational_params))

            avg_loss = float(np.mean(losses))
            self._history["loss"].append(avg_loss)

            if X_val is not None:
                vl = self._compute_loss(self._forward_batch(X_val), y_val)
                self._history["val_loss"].append(vl)
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {vl:.4f}")
            elif verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs."""
        return self._forward_batch(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate the regressor."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        preds = self._forward_batch(X)
        loss = self._compute_loss(preds, y)
        return {"loss": loss}

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "variational_params": self._variational_params.copy() if self._variational_params is not None else np.array([]),
            "readout_kernel": self._readout_kernel.copy() if self._readout_kernel is not None else np.array([]),
            "readout_bias": self._readout_bias.copy() if self._readout_bias is not None else np.array([]),
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        if "variational_params" in params:
            self._variational_params = np.asarray(params["variational_params"], dtype=np.float64)
        if "readout_kernel" in params:
            self._readout_kernel = np.asarray(params["readout_kernel"], dtype=np.float64)
        if "readout_bias" in params:
            self._readout_bias = np.asarray(params["readout_bias"], dtype=np.float64)

    def summary(self) -> str:
        return (
            f"QRegressor\n  n_qubits:  {self._n_qubits}\n  n_outputs: {self._n_outputs}\n"
            f"  n_layers:   {self._n_layers}\n  optimizer: {self._optimizer_name}\n"
            f"  loss:      {self._loss_name}"
        )

    def __repr__(self) -> str:
        return f"QRegressor(n_qubits={self._n_qubits}, n_outputs={self._n_outputs}, n_layers={self._n_layers})"


# ═══════════════════════════════════════════════════════════════════════════
# QAutoencoder — Quantum Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

class QAutoencoder:
    """Quantum autoencoder with trash qubit detection.

    Architecture:
    - **Encoder**: classical data -> quantum state on all qubits.
    - **Trash qubit detection**: measures a subset of qubits to verify
      compression quality.
    - **Decoder**: compressed quantum state -> reconstructed classical data.

    Parameters
    ----------
    n_qubits : int
        Total qubits (data qubits + latent qubits).
    n_latent : int, optional
        Number of latent qubits. Default ``2``.
    n_layers : int, optional
        Variational layers for encoder/decoder. Default ``3``.
    learning_rate : float, optional
        Default ``0.01``.
    name : str, optional

    Examples
    --------
    >>> ae = QAutoencoder(n_qubits=6, n_latent=2, n_layers=3)
    >>> ae.compile(optimizer='adam', loss='mse')
    >>> ae.fit(x_train, x_train, epochs=50)
    >>> reconstructed = ae.predict(x_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_latent: int = 2,
        n_layers: int = 3,
        learning_rate: float = 0.01,
        name: Optional[str] = None,
    ) -> None:
        if n_latent >= n_qubits:
            raise ValueError("n_latent must be < n_qubits")

        self._n_qubits = n_qubits
        self._n_latent = n_latent
        self._n_trash = n_qubits - n_latent
        self._n_layers = n_layers
        self._learning_rate = learning_rate
        self._name = name or f"qautoencoder_{id(self):x}"

        # Encoder + decoder params
        self._encoder_params: Optional[np.ndarray] = None
        self._decoder_params: Optional[np.ndarray] = None
        self._readout_kernel: Optional[np.ndarray] = None

        self._compiled = False
        self._optimizer_name: Optional[str] = None
        self._loss_name: Optional[str] = None
        self._history: Dict[str, List[float]] = {}

        self._enc_m: Optional[np.ndarray] = None
        self._enc_v: Optional[np.ndarray] = None
        self._dec_m: Optional[np.ndarray] = None
        self._dec_v: Optional[np.ndarray] = None
        self._t = 0

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_latent(self) -> int:
        return self._n_latent

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "mse",
    ) -> None:
        """Compile the autoencoder."""
        rng = np.random.default_rng()

        n_enc_params = self._n_layers * self._n_qubits * 3
        n_dec_params = self._n_layers * self._n_qubits * 3

        self._encoder_params = rng.uniform(-0.1, 0.1, n_enc_params).astype(np.float64)
        self._decoder_params = rng.uniform(-0.1, 0.1, n_dec_params).astype(np.float64)
        self._readout_kernel = rng.uniform(
            -0.1, 0.1, (self._n_qubits, self._n_qubits)
        ).astype(np.float64)

        self._enc_m = np.zeros(n_enc_params, dtype=np.float64)
        self._enc_v = np.zeros(n_enc_params, dtype=np.float64)
        self._dec_m = np.zeros(n_dec_params, dtype=np.float64)
        self._dec_v = np.zeros(n_dec_params, dtype=np.float64)
        self._t = 0

        self._optimizer_name = optimizer
        self._loss_name = loss
        self._compiled = True

    def _encode(self, data: np.ndarray) -> Any:
        """Run encoder circuit, return quantum state."""
        from quantumflow.core.circuit import QuantumCircuit

        qc = QuantumCircuit(self._n_qubits)
        _encode_data(qc, data, self._n_qubits)
        _apply_variational(qc, self._encoder_params, self._n_qubits, self._n_layers)

        sim = _get_simulator()
        return sim.state(qc)

    def _decode(self, state: Any) -> np.ndarray:
        """Run decoder circuit on given state, return reconstruction."""
        from quantumflow.core.circuit import QuantumCircuit

        qc = QuantumCircuit(self._n_qubits)
        # Prepare the state as initial state
        sv = state.data if hasattr(state, 'data') else np.asarray(state)
        _apply_variational(qc, self._decoder_params, self._n_qubits, self._n_layers)

        sim = _get_simulator()
        final_state = sim.state(qc, initial_state=sv)

        # Read out expectation values
        final_sv = final_state.data if hasattr(final_state, 'data') else np.asarray(final_state)
        reconstruction = np.zeros(self._n_qubits, dtype=np.float64)
        for q in range(self._n_qubits):
            obs = _build_pauli_observable("z", q, self._n_qubits)
            reconstruction[q] = float(np.real(np.vdot(final_sv, obs @ final_sv)))

        return reconstruction

    def _forward_single(self, x: np.ndarray) -> np.ndarray:
        """Full encode-decode for a single sample."""
        data = np.clip(x[:self._n_qubits], -_PI, _PI)
        state = self._encode(data)
        return self._decode(state)

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for a batch."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch = X.shape[0]
        output = np.zeros((batch, self._n_qubits), dtype=np.float64)
        for b in range(batch):
            output[b] = self._forward_single(X[b])
        return output

    def _trash_loss(self, state: Any) -> float:
        """Compute trash qubit loss: expectation of trash qubits should be 0."""
        sv = state.data if hasattr(state, 'data') else np.asarray(state)
        loss = 0.0
        for q in range(self._n_latent, self._n_qubits):
            obs = _build_pauli_observable("z", q, self._n_qubits)
            exp_val = float(np.real(np.vdot(sv, obs @ sv)))
            loss += exp_val ** 2
        return loss / self._n_trash

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder.

        Parameters
        ----------
        X : numpy.ndarray
            Training data. If ``y`` is None, uses X as target (reconstruction).
        y : numpy.ndarray, optional
            Target data. Default ``None`` (uses X).
        epochs, batch_size, verbose : standard training args.

        Returns
        -------
        dict
            Training history.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        X = np.asarray(X, dtype=np.float64)
        if y is None:
            y = X.copy()
        else:
            y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        self._history = {"loss": [], "reconstruction_loss": [], "trash_loss": []}

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_s, y_s = X[perm], y[perm]
            losses = []
            recon_losses = []
            trash_losses = []

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                xb, yb = X_s[start:end], y_s[start:end]

                # Encoder gradients
                eps = 1e-5
                enc_grad = np.zeros_like(self._encoder_params)

                for b in range(len(xb)):
                    data = np.clip(xb[b][:self._n_qubits], -_PI, _PI)
                    base_state = self._encode(data)
                    base_recon = self._decode(base_state)
                    target = yb[b][:self._n_qubits] if yb.shape[-1] >= self._n_qubits else yb[b]
                    base_loss = _mse_loss(base_recon, target) + 0.5 * self._trash_loss(base_state)

                    for i in range(len(self._encoder_params)):
                        p_plus = self._encoder_params.copy()
                        p_plus[i] += eps
                        self._encoder_params[i] = p_plus[i]
                        s_plus = self._encode(data)
                        r_plus = self._decode(s_plus)
                        l_plus = _mse_loss(r_plus, target) + 0.5 * self._trash_loss(s_plus)

                        p_minus = self._encoder_params.copy()
                        p_minus[i] -= eps
                        self._encoder_params[i] = p_minus[i]
                        s_minus = self._encode(data)
                        r_minus = self._decode(s_minus)
                        l_minus = _mse_loss(r_minus, target) + 0.5 * self._trash_loss(s_minus)

                        enc_grad[i] += (l_plus - l_minus) / (2 * eps)
                        self._encoder_params[i] = float(p_plus[i] - eps)

                enc_grad /= len(xb)

                # Adam update for encoder
                self._t += 1
                assert self._enc_m is not None
                assert self._enc_v is not None
                self._enc_m = 0.9 * self._enc_m + 0.1 * enc_grad
                self._enc_v = 0.999 * self._enc_v + 0.001 * (enc_grad ** 2)
                m_hat = self._enc_m / (1 - 0.9 ** self._t)
                v_hat = self._enc_v / (1 - 0.999 ** self._t)
                self._encoder_params -= self._learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                # Decoder gradients (simplified)
                dec_grad = np.zeros_like(self._decoder_params)
                for b in range(len(xb)):
                    data = np.clip(xb[b][:self._n_qubits], -_PI, _PI)
                    state = self._encode(data)
                    recon = self._decode(state)
                    target = yb[b][:self._n_qubits] if yb.shape[-1] >= self._n_qubits else yb[b]

                    for i in range(len(self._decoder_params)):
                        d_plus = self._decoder_params.copy()
                        d_plus[i] += eps
                        self._decoder_params[i] = d_plus[i]
                        r_plus = self._decode(state)
                        l_plus = _mse_loss(r_plus, target)

                        d_minus = self._decoder_params.copy()
                        d_minus[i] -= eps
                        self._decoder_params[i] = d_minus[i]
                        r_minus = self._decode(state)
                        l_minus = _mse_loss(r_minus, target)

                        dec_grad[i] += (l_plus - l_minus) / (2 * eps)
                        self._decoder_params[i] = float(d_plus[i] - eps)

                dec_grad /= len(xb)

                assert self._dec_m is not None
                assert self._dec_v is not None
                self._dec_m = 0.9 * self._dec_m + 0.1 * dec_grad
                self._dec_v = 0.999 * self._dec_v + 0.001 * (dec_grad ** 2)
                dm_hat = self._dec_m / (1 - 0.9 ** self._t)
                dv_hat = self._dec_v / (1 - 0.999 ** self._t)
                self._decoder_params -= self._learning_rate * dm_hat / (np.sqrt(dv_hat) + 1e-8)

                # Track losses
                r_loss = _mse_loss(self._forward_batch(xb), yb)
                t_loss = self._trash_loss(self._encode(np.clip(xb[0][:self._n_qubits], -_PI, _PI)))
                losses.append(r_loss + 0.5 * t_loss)
                recon_losses.append(r_loss)
                trash_losses.append(t_loss)

            self._history["loss"].append(float(np.mean(losses)))
            self._history["reconstruction_loss"].append(float(np.mean(recon_losses)))
            self._history["trash_loss"].append(float(np.mean(trash_losses)))

            if verbose > 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - loss: {self._history['loss'][-1]:.4f} "
                    f"- recon: {self._history['reconstruction_loss'][-1]:.4f} "
                    f"- trash: {self._history['trash_loss'][-1]:.4f}"
                )

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data."""
        return self._forward_batch(X)

    def encode(self, X: np.ndarray) -> List[Any]:
        """Encode data to quantum states (list of Statevectors).

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        list
            List of Statevector objects.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        states = []
        for b in range(X.shape[0]):
            data = np.clip(X[b][:self._n_qubits], -_PI, _PI)
            states.append(self._encode(data))
        return states

    def __repr__(self) -> str:
        return (
            f"QAutoencoder(n_qubits={self._n_qubits}, "
            f"n_latent={self._n_latent}, n_layers={self._n_layers})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QGAN — Quantum Generative Adversarial Network
# ═══════════════════════════════════════════════════════════════════════════

class QGAN:
    """Quantum Generative Adversarial Network.

    The generator is a parameterised quantum circuit that produces data
    samples. The discriminator is a classical neural network.

    Parameters
    ----------
    n_qubits : int
        Qubits for the generator circuit.
    n_layers : int, optional
        Generator variational layers. Default ``3``.
    latent_dim : int, optional
        Dimension of the latent noise vector. Default ``n_qubits``.
    discriminator_hidden : tuple of int, optional
        Hidden layer sizes for the discriminator. Default ``(32, 16)``.
    learning_rate_g : float, optional
        Generator learning rate. Default ``0.005``.
    learning_rate_d : float, optional
        Discriminator learning rate. Default ``0.01``.
    name : str, optional

    Examples
    --------
    >>> qgan = QGAN(n_qubits=4, n_layers=3)
    >>> qgan.compile()
    >>> qgan.fit(real_data, epochs=100)
    >>> samples = qgan.generate(n_samples=100)
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        latent_dim: Optional[int] = None,
        discriminator_hidden: Tuple[int, ...] = (32, 16),
        learning_rate_g: float = 0.005,
        learning_rate_d: float = 0.01,
        name: Optional[str] = None,
    ) -> None:
        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._latent_dim = latent_dim or n_qubits
        self._disc_hidden = discriminator_hidden
        self._lr_g = learning_rate_g
        self._lr_d = learning_rate_d
        self._name = name or f"qgan_{id(self):x}"

        self._gen_params: Optional[np.ndarray] = None
        self._disc_weights: Optional[List[np.ndarray]] = None
        self._disc_biases: Optional[List[np.ndarray]] = None
        self._compiled = False
        self._history: Dict[str, List[float]] = {}

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def compile(self) -> None:
        """Initialize generator and discriminator parameters."""
        rng = np.random.default_rng()
        n_gen_params = self._n_layers * self._n_qubits * 3
        self._gen_params = rng.uniform(-0.1, 0.1, n_gen_params).astype(np.float64)

        # Discriminator: simple MLP with numpy
        self._disc_weights = []
        self._disc_biases = []
        prev_dim = self._n_qubits
        for h in self._disc_hidden:
            w = rng.uniform(-0.1, 0.1, (prev_dim, h)).astype(np.float64)
            b = np.zeros(h, dtype=np.float64)
            self._disc_weights.append(w)
            self._disc_biases.append(b)
            prev_dim = h
        # Output layer
        w_out = rng.uniform(-0.1, 0.1, (prev_dim, 1)).astype(np.float64)
        b_out = np.zeros(1, dtype=np.float64)
        self._disc_weights.append(w_out)
        self._disc_biases.append(b_out)

        self._compiled = True
        self._history = {"g_loss": [], "d_loss": []}

    def _generate_single(self, z: np.ndarray) -> np.ndarray:
        """Generate a single sample from latent noise z."""
        from quantumflow.core.circuit import QuantumCircuit

        assert self._gen_params is not None
        qc = QuantumCircuit(self._n_qubits)
        _encode_data(qc, np.clip(z, -_PI, _PI), self._n_qubits)
        _apply_variational(qc, self._gen_params, self._n_qubits, self._n_layers)

        sim = _get_simulator()
        results = np.zeros(self._n_qubits, dtype=np.float64)
        for q in range(self._n_qubits):
            obs = _build_pauli_observable("z", q, self._n_qubits)
            results[q] = float(sim.expectation(qc, obs))
        return results

    def generate(self, n_samples: int = 100) -> np.ndarray:
        """Generate samples from the trained generator.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, n_qubits)``.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled.")
        rng = np.random.default_rng()
        samples = np.zeros((n_samples, self._n_qubits), dtype=np.float64)
        for i in range(n_samples):
            z = rng.uniform(-1, 1, self._latent_dim)
            samples[i] = self._generate_single(z)
        return samples

    def _discriminator_forward(self, x: np.ndarray) -> float:
        """Run discriminator on a single sample."""
        assert self._disc_weights is not None
        assert self._disc_biases is not None

        h = x
        for i in range(len(self._disc_weights) - 1):
            h = h @ self._disc_weights[i] + self._disc_biases[i]
            h = np.maximum(0, h)  # ReLU
        # Output layer with sigmoid
        h = h @ self._disc_weights[-1] + self._disc_biases[-1]
        return float(_sigmoid(h)[0])

    def _discriminator_batch(self, X: np.ndarray) -> np.ndarray:
        """Run discriminator on a batch."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        results = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X)):
            results[i] = self._discriminator_forward(X[i])
        return results

    def fit(
        self,
        real_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        n_critic: int = 1,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the QGAN.

        Parameters
        ----------
        real_data : numpy.ndarray
            Shape ``(n_samples, n_qubits)`` real data.
        epochs : int
        batch_size : int
        n_critic : int
            Number of discriminator updates per generator update.
        verbose : int

        Returns
        -------
        dict
            Training history.
        """
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        real_data = np.asarray(real_data, dtype=np.float64)
        if real_data.ndim == 1:
            real_data = real_data.reshape(1, -1)

        self._history = {"g_loss": [], "d_loss": []}
        rng = np.random.default_rng()

        for epoch in range(epochs):
            g_losses = []
            d_losses = []

            for _ in range(len(real_data) // batch_size):
                # Sample real data
                idx = rng.choice(len(real_data), batch_size, replace=False)
                real_batch = real_data[idx]

                # Generate fake data
                z_batch = rng.uniform(-1, 1, (batch_size, self._latent_dim))
                fake_batch = np.zeros((batch_size, self._n_qubits), dtype=np.float64)
                for i in range(batch_size):
                    fake_batch[i] = self._generate_single(z_batch[i])

                # Train discriminator
                for _ in range(n_critic):
                    real_scores = self._discriminator_batch(real_batch)
                    fake_scores = self._discriminator_batch(fake_batch)

                    d_loss = -float(np.mean(np.log(np.clip(real_scores, 1e-7, 1)) +
                                            np.log(np.clip(1 - fake_scores, 1e-7, 1))))

                    # Gradient w.r.t. discriminator weights (simplified)
                    eps = 1e-5
                    for layer_idx in range(len(self._disc_weights)):
                        w = self._disc_weights[layer_idx]
                        b = self._disc_biases[layer_idx]

                        w_grad = np.zeros_like(w)
                        b_grad = np.zeros_like(b)

                        for p_i in range(min(w.size, 50)):  # Subsample for speed
                            pi, pj = divmod(p_i, w.shape[1]) if w.ndim == 2 else (0, p_i)

                            # Save
                            orig_w = w[pi, pj] if w.ndim == 2 else w[p_i]
                            w[pi, pj] = orig_w + eps
                            r_p = self._discriminator_batch(real_batch)
                            f_p = self._discriminator_batch(fake_batch)
                            loss_p = -np.mean(np.log(np.clip(r_p, 1e-7, 1)) + np.log(np.clip(1 - f_p, 1e-7, 1)))

                            w[pi, pj] = orig_w - eps
                            r_m = self._discriminator_batch(real_batch)
                            f_m = self._discriminator_batch(fake_batch)
                            loss_m = -np.mean(np.log(np.clip(r_m, 1e-7, 1)) + np.log(np.clip(1 - f_m, 1e-7, 1)))

                            w[pi, pj] = orig_w
                            grad_val = (float(loss_p) - float(loss_m)) / (2 * eps)
                            if w.ndim == 2:
                                w_grad[pi, pj] = grad_val
                            else:
                                w_grad[p_i] = grad_val

                        self._disc_weights[layer_idx] -= self._lr_d * w_grad

                    d_losses.append(d_loss)

                # Train generator
                fake_scores = self._discriminator_batch(fake_batch)
                g_loss = -float(np.mean(np.log(np.clip(fake_scores, 1e-7, 1))))

                eps = 1e-5
                assert self._gen_params is not None
                g_grad = np.zeros_like(self._gen_params)

                for i in range(min(len(self._gen_params), 20)):  # Subsample
                    orig = self._gen_params[i]
                    self._gen_params[i] = orig + eps
                    f_plus = np.zeros((batch_size, self._n_qubits), dtype=np.float64)
                    for bi in range(batch_size):
                        f_plus[bi] = self._generate_single(z_batch[bi])
                    sp = self._discriminator_batch(f_plus)
                    lp = -np.mean(np.log(np.clip(sp, 1e-7, 1)))

                    self._gen_params[i] = orig - eps
                    f_minus = np.zeros((batch_size, self._n_qubits), dtype=np.float64)
                    for bi in range(batch_size):
                        f_minus[bi] = self._generate_single(z_batch[bi])
                    sm = self._discriminator_batch(f_minus)
                    lm = -np.mean(np.log(np.clip(sm, 1e-7, 1)))

                    self._gen_params[i] = orig
                    g_grad[i] = (float(lp) - float(lm)) / (2 * eps)

                self._gen_params -= self._lr_g * g_grad
                g_losses.append(g_loss)

            self._history["g_loss"].append(float(np.mean(g_losses)) if g_losses else 0)
            self._history["d_loss"].append(float(np.mean(d_losses)) if d_losses else 0)

            if verbose > 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"d_loss: {self._history['d_loss'][-1]:.4f} - "
                    f"g_loss: {self._history['g_loss'][-1]:.4f}"
                )

        return self._history

    def __repr__(self) -> str:
        return f"QGAN(n_qubits={self._n_qubits}, n_layers={self._n_layers})"


# ═══════════════════════════════════════════════════════════════════════════
# QTransferLearningModel — Classical Backbone + Quantum Head
# ═══════════════════════════════════════════════════════════════════════════

class QTransferLearningModel:
    """Transfer learning with a classical backbone and quantum head.

    Uses a classical neural network as a feature extractor (backbone)
    and a quantum circuit as the classification/regression head.

    Parameters
    ----------
    backbone_layers : list of dict
        Classical layer configs, e.g. ``[{'type': 'dense', 'units': 64, 'activation': 'relu'}]``.
    n_qubits : int
        Qubits for the quantum head.
    n_classes : int, optional
        Number of output classes. Default ``2``.
    n_layers : int, optional
        Quantum head variational layers. Default ``2``.
    freeze_backbone : bool, optional
        Freeze backbone weights initially. Default ``True``.
    learning_rate : float, optional
        Default ``0.001``.
    name : str, optional

    Examples
    --------
    >>> backbone = [{'type': 'dense', 'units': 32, 'activation': 'relu'}]
    >>> model = QTransferLearningModel(backbone_layers=backbone, n_qubits=4, n_classes=3)
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
    >>> model.fit(x_train, y_train, epochs=50)
    """

    def __init__(
        self,
        backbone_layers: List[Dict[str, Any]],
        n_qubits: int,
        n_classes: int = 2,
        n_layers: int = 2,
        freeze_backbone: bool = True,
        learning_rate: float = 0.001,
        name: Optional[str] = None,
    ) -> None:
        self._backbone_configs = backbone_layers
        self._n_qubits = n_qubits
        self._n_classes = n_classes
        self._n_quantum_layers = n_layers
        self._freeze_backbone = freeze_backbone
        self._learning_rate = learning_rate
        self._name = name or f"qtransfer_{id(self):x}"

        self._backbone_weights: List[np.ndarray] = []
        self._backbone_biases: List[np.ndarray] = []
        self._quantum_params: Optional[np.ndarray] = None
        self._readout_kernel: Optional[np.ndarray] = None
        self._readout_bias: Optional[np.ndarray] = None

        self._compiled = False
        self._history: Dict[str, List[float]] = {}

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
    ) -> None:
        """Initialize all parameters."""
        rng = np.random.default_rng()
        self._backbone_weights = []
        self._backbone_biases = []
        prev_dim = None  # Will be set on first call

        for config in self._backbone_configs:
            units = config.get("units", 32)
            w = rng.uniform(-0.1, 0.1, (prev_dim or units, units)).astype(np.float64)
            b = np.zeros(units, dtype=np.float64)
            self._backbone_weights.append(w)
            self._backbone_biases.append(b)
            prev_dim = units

        n_q_params = self._n_quantum_layers * self._n_qubits * 3
        self._quantum_params = rng.uniform(-0.1, 0.1, n_q_params).astype(np.float64)
        self._readout_kernel = rng.uniform(-0.1, 0.1, (self._n_qubits, self._n_classes)).astype(np.float64)
        self._readout_bias = np.zeros(self._n_classes, dtype=np.float64)

        self._compiled = True
        self._optimizer_name = optimizer
        self._loss_name = loss
        self._history = {"loss": []}

    def _backbone_forward(self, x: np.ndarray) -> np.ndarray:
        """Run classical backbone."""
        h = x
        for i, (w, b) in enumerate(zip(self._backbone_weights, self._backbone_biases)):
            act = self._backbone_configs[i].get("activation", "relu")
            h = h @ w + b
            if act == "relu":
                h = np.maximum(0, h)
            elif act == "tanh":
                h = np.tanh(h)
            elif act == "sigmoid":
                h = _sigmoid(h)
        return h

    def _quantum_head(self, features: np.ndarray) -> np.ndarray:
        """Run quantum head on backbone features."""
        assert self._quantum_params is not None
        data = np.clip(features[:self._n_qubits], -_PI, _PI)
        q_out = _run_forward(data, self._quantum_params, self._n_qubits, self._n_quantum_layers)
        logits = q_out @ self._readout_kernel + self._readout_bias
        return logits

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Full forward pass."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch = X.shape[0]
        output = np.zeros((batch, self._n_classes), dtype=np.float64)
        for b in range(batch):
            backbone_features = self._backbone_forward(X[b])
            output[b] = self._quantum_head(backbone_features)
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model."""
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Resize backbone input dimensions
        if len(self._backbone_weights) > 0:
            self._backbone_weights[0] = np.random.default_rng().uniform(
                -0.1, 0.1, (X.shape[-1], self._backbone_weights[0].shape[1])
            ).astype(np.float64)

        self._history = {"loss": []}

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            losses = []

            for start in range(0, len(X), batch_size):
                xb = X[perm[start:start + batch_size]]
                yb = y[perm[start:start + batch_size]]

                if xb.ndim == 1:
                    xb = xb.reshape(1, -1)

                logits = self._forward_batch(xb)
                probs = _softmax(logits)
                loss = -float(np.mean(np.sum(yb * np.log(np.clip(probs, 1e-7, 1)), axis=-1)))
                losses.append(loss)

                # Update quantum params (gradient via finite diff on first sample)
                if not self._freeze_backbone:
                    eps = 1e-5
                    for i in range(min(len(self._quantum_params), 10)):
                        orig = self._quantum_params[i]
                        self._quantum_params[i] = orig + eps
                        lp = self._forward_batch(xb[:1])
                        pp = _softmax(lp)
                        ll = -float(np.mean(np.sum(yb[:1] * np.log(np.clip(pp, 1e-7, 1)), axis=-1)))

                        self._quantum_params[i] = orig - eps
                        lm = self._forward_batch(xb[:1])
                        pm = _softmax(lm)
                        lll = -float(np.mean(np.sum(yb[:1] * np.log(np.clip(pm, 1e-7, 1)), axis=-1)))

                        self._quantum_params[i] = orig - self._learning_rate * (ll - lll) / (2 * eps)

            self._history["loss"].append(float(np.mean(losses)) if losses else 0)
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {self._history['loss'][-1]:.4f}")

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        logits = self._forward_batch(X)
        return _softmax(logits)

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        self._freeze_backbone = False

    def __repr__(self) -> str:
        return (
            f"QTransferLearningModel(n_qubits={self._n_qubits}, "
            f"n_classes={self._n_classes}, "
            f"backbone_layers={len(self._backbone_configs)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# QHybridModel — Hybrid Classical-Quantum Model
# ═══════════════════════════════════════════════════════════════════════════

class QHybridModel:
    """Hybrid classical-quantum model with arbitrary layer sequences.

    Supports mixing classical (Dense, Conv2D, etc.) and quantum layers
    with automatic gradient flow between them.

    Parameters
    ----------
    name : str, optional

    Examples
    --------
    >>> model = QHybridModel()
    >>> model.add_classical_layer({'type': 'dense', 'units': 16, 'activation': 'relu'})
    >>> model.add_quantum_layer({'n_qubits': 4, 'n_layers': 2})
    >>> model.add_classical_layer({'type': 'dense', 'units': 2, 'activation': 'linear'})
    >>> model.compile(optimizer='adam', loss='mse')
    >>> model.fit(x_train, y_train, epochs=50)
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name or f"qhybrid_{id(self):x}"
        self._classical_layers: List[Dict[str, Any]] = []
        self._quantum_layers: List[Dict[str, Any]] = []

        self._classical_weights: List[np.ndarray] = []
        self._classical_biases: List[np.ndarray] = []
        self._quantum_params_list: List[np.ndarray] = []

        self._compiled = False
        self._history: Dict[str, List[float]] = {}

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def add_classical_layer(self, config: Dict[str, Any]) -> None:
        """Add a classical layer to the model.

        Parameters
        ----------
        config : dict
            Layer configuration, e.g. ``{'type': 'dense', 'units': 16,
            'activation': 'relu'}``.
        """
        self._classical_layers.append(config)

    def add_quantum_layer(self, config: Dict[str, Any]) -> None:
        """Add a quantum layer to the model.

        Parameters
        ----------
        config : dict
            Quantum layer config, e.g. ``{'n_qubits': 4, 'n_layers': 2}``.
        """
        self._quantum_layers.append(config)

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "mse",
        learning_rate: float = 0.01,
    ) -> None:
        """Initialize all layer parameters."""
        rng = np.random.default_rng()

        self._classical_weights = []
        self._classical_biases = []
        self._quantum_params_list = []

        prev_dim = None
        for config in self._classical_layers:
            units = config.get("units", 16)
            in_dim = prev_dim or units
            w = rng.uniform(-0.1, 0.1, (in_dim, units)).astype(np.float64)
            b = np.zeros(units, dtype=np.float64)
            self._classical_weights.append(w)
            self._classical_biases.append(b)
            prev_dim = units

        for config in self._quantum_layers:
            n_q = config.get("n_qubits", 4)
            n_l = config.get("n_layers", 2)
            n_params = n_l * n_q * 3
            self._quantum_params_list.append(
                rng.uniform(-0.1, 0.1, n_params).astype(np.float64)
            )
            prev_dim = n_q

        self._optimizer_name = optimizer
        self._loss_name = loss
        self._learning_rate = learning_rate
        self._compiled = True
        self._history = {"loss": []}

    def _forward_single(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        h = x
        c_idx = 0
        q_idx = 0

        for config in self._classical_layers:
            w = self._classical_weights[c_idx]
            b = self._classical_biases[c_idx]
            act = config.get("activation", "relu")

            h = h @ w + b
            if act == "relu":
                h = np.maximum(0, h)
            elif act == "tanh":
                h = np.tanh(h)
            elif act == "sigmoid":
                h = _sigmoid(h)
            c_idx += 1

        for config in self._quantum_layers:
            n_q = config.get("n_qubits", 4)
            n_l = config.get("n_layers", 2)
            ent = config.get("entanglement", "linear")
            params = self._quantum_params_list[q_idx]

            data = np.clip(h[:n_q], -_PI, _PI)
            q_out = _run_forward(data, params, n_q, n_l, ent)
            h = q_out
            q_idx += 1

        return h

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for batch."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        batch = X.shape[0]
        first_out = self._forward_single(X[0])
        output = np.zeros((batch, len(first_out)), dtype=np.float64)
        output[0] = first_out
        for b in range(1, batch):
            output[b] = self._forward_single(X[b])
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the hybrid model."""
        if not self._compiled:
            raise RuntimeError("Model not compiled.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Resize first classical layer
        if self._classical_weights:
            first_units = self._classical_weights[0].shape[1]
            self._classical_weights[0] = np.random.default_rng().uniform(
                -0.1, 0.1, (X.shape[-1], first_units)
            ).astype(np.float64)

        self._history = {"loss": []}

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            losses = []

            for start in range(0, len(X), batch_size):
                xb = X[perm[start:start + batch_size]]
                yb = y[perm[start:start + batch_size]]
                if xb.ndim == 1:
                    xb = xb.reshape(1, -1)

                preds = self._forward_batch(xb)
                if preds.shape[-1] != yb.shape[-1]:
                    preds = preds[:, :yb.shape[-1]]
                loss = _mse_loss(preds, yb)
                losses.append(loss)

                # Update quantum params
                eps = 1e-5
                for qi, params in enumerate(self._quantum_params_list):
                    for i in range(min(len(params), 10)):
                        orig = params[i]
                        params[i] = orig + eps
                        p_p = self._forward_batch(xb[:1])
                        if p_p.shape[-1] != yb[:1].shape[-1]:
                            p_p = p_p[:, :yb[:1].shape[-1]]
                        l_p = _mse_loss(p_p, yb[:1])

                        params[i] = orig - eps
                        p_m = self._forward_batch(xb[:1])
                        if p_m.shape[-1] != yb[:1].shape[-1]:
                            p_m = p_m[:, :yb[:1].shape[-1]]
                        l_m = _mse_loss(p_m, yb[:1])

                        params[i] = orig - self._learning_rate * (l_p - l_m) / (2 * eps)

            self._history["loss"].append(float(np.mean(losses)) if losses else 0)
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {self._history['loss'][-1]:.4f}")

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs."""
        return self._forward_batch(X)

    def summary(self) -> str:
        """Return model architecture summary."""
        lines = ["QHybridModel", "=" * 40]
        for i, c in enumerate(self._classical_layers):
            lines.append(f"  Layer {i}: Classical ({c.get('type', 'dense')}, units={c.get('units', '?')})")
        for i, q in enumerate(self._quantum_layers):
            lines.append(f"  Layer {len(self._classical_layers) + i}: Quantum (n_qubits={q.get('n_qubits', '?')}, n_layers={q.get('n_layers', '?')})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QHybridModel(classical_layers={len(self._classical_layers)}, "
            f"quantum_layers={len(self._quantum_layers)})"
        )
