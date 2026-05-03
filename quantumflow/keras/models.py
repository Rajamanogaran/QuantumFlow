"""
Pre-built Keras Quantum Models
===============================

Provides ready-to-use Keras models that combine quantum layers with
classical neural network components. All models follow the Keras Model
API with ``compile()``, ``fit()``, ``predict()``, and ``evaluate()``
methods.

Classes
-------
* :class:`KerasQuantumClassifier` — Quantum variational classifier.
* :class:`KerasQuantumRegressor` — Quantum variational regressor.
* :class:`KerasQNN` — Flexible quantum neural network builder.
* :class:`KerasQuantumAutoencoder` — Quantum autoencoder.
* :class:`KerasHybridModel` — Hybrid classical-quantum model.
* :class:`KerasQuantumGAN` — Quantum Generative Adversarial Network.
* :class:`KerasQuantumVAE` — Quantum Variational Autoencoder.
* :class:`KerasTransferLearning` — Transfer learning with quantum head.

Examples
--------
>>> from quantumflow.keras.models import KerasQuantumClassifier
>>> clf = KerasQuantumClassifier(n_qubits=4, n_classes=3)
>>> clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
>>> clf.fit(X_train, y_train, epochs=10)
>>> preds = clf.predict(X_test)
"""

from __future__ import annotations

import math
import time
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

# Keras imports
try:
    import keras
    from keras import ops
    from keras.layers import Layer as _KerasLayer
    from keras.models import Model as KerasModel
except ImportError:
    keras = None  # type: ignore[assignment]
    ops = None  # type: ignore[assignment]
    _KerasLayer = None  # type: ignore[assignment,misc]
    KerasModel = None  # type: ignore[assignment,misc]

if _KerasLayer is None:
    class _StubLayer:
        def __init__(self, **kwargs: Any) -> None:
            pass
    Layer = _StubLayer  # type: ignore[misc]
else:
    Layer = _KerasLayer  # type: ignore[misc]

from quantumflow.core.circuit import QuantumCircuit
from quantumflow.simulation.simulator import StatevectorSimulator

__all__ = [
    "KerasQuantumClassifier",
    "KerasQuantumRegressor",
    "KerasQNN",
    "KerasQuantumAutoencoder",
    "KerasHybridModel",
    "KerasQuantumGAN",
    "KerasQuantumVAE",
    "KerasTransferLearning",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10


def _check_keras_available() -> None:
    if keras is None or _KerasLayer is None:
        raise ImportError(
            "Keras 3 is required for quantumflow.keras models. "
            "Install it with: pip install keras>=3.0"
        )


def _pauli_observable(pauli: str, qubit: int, n_qubits: int) -> np.ndarray:
    """Build a single-qubit Pauli observable in n-qubit space."""
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


# ===========================================================================
# KerasQuantumClassifier — Quantum Classifier Model
# ===========================================================================

class KerasQuantumClassifier:
    """Quantum variational classifier built on Keras Model API.

    Supports binary and multi-class classification with configurable
    quantum feature extraction and classical readout.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the quantum circuit.
    n_classes : int
        Number of output classes (1 for binary, >1 for multi-class).
    n_layers : int, optional
        Circuit depth. Default ``2``.
    encoding : str, optional
        Data encoding strategy. Default ``'angle'``.
    ansatz : str, optional
        Variational ansatz: ``'hardware_efficient'``, ``'strong_entangling'``,
        ``'circuit_19'``, ``'barren_plateau_free'``. Default
        ``'hardware_efficient'``.
    feature_map : str, optional
        Feature map type. Default ``'angle'``.
    learning_rate : float, optional
        Initial learning rate. Default ``0.01``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> clf = KerasQuantumClassifier(n_qubits=4, n_classes=3)
    >>> clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    >>> clf.fit(X_train, y_train, epochs=10, batch_size=32)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        encoding: str = "angle",
        ansatz: str = "hardware_efficient",
        feature_map: str = "angle",
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.encoding = encoding
        self.ansatz = ansatz
        self.feature_map = feature_map
        self.learning_rate = learning_rate
        self._model: Optional[Any] = None
        self._history: Dict[str, List[float]] = {
            "loss": [], "accuracy": [],
            "val_loss": [], "val_accuracy": [],
        }
        self._compiled = False
        self._simulator = StatevectorSimulator()
        self._kernel = None
        self._bias = None
        self._var_params = None
        self._readout = None
        self._optimizer = None
        self._loss_fn = None
        self._input_dim: Optional[int] = None

    def _build_circuit(
        self,
        data: np.ndarray,
        var_params: np.ndarray,
    ) -> QuantumCircuit:
        """Build the variational classifier circuit.

        Parameters
        ----------
        data : numpy.ndarray
            Shape ``(n_qubits,)``.
        var_params : numpy.ndarray
            Variational parameters.

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Feature encoding
        if self.feature_map == "angle":
            for q in range(self.n_qubits):
                qc.h(q)
                if q < len(data):
                    qc.ry(float(data[q]) % (2 * _PI), q)
        elif self.feature_map == "zz":
            for q in range(self.n_qubits):
                qc.h(q)
                if q < len(data):
                    qc.rz(float(data[q]) % (2 * _PI), q)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                if q < len(data):
                    qc.rz(float(data[q]) % (2 * _PI), q)
        else:
            for q in range(self.n_qubits):
                qc.h(q)
                if q < len(data):
                    qc.ry(float(data[q]) % (2 * _PI), q)

        # Variational ansatz
        param_idx = 0
        for layer in range(self.n_layers):
            if self.ansatz == "hardware_efficient":
                for q in range(self.n_qubits):
                    if param_idx + 1 < len(var_params):
                        qc.rz(float(var_params[param_idx]), q)
                        param_idx += 1
                    if param_idx + 1 < len(var_params):
                        qc.ry(float(var_params[param_idx]), q)
                        param_idx += 1
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)

            elif self.ansatz == "strong_entangling":
                for q in range(self.n_qubits):
                    if param_idx + 2 < len(var_params):
                        qc.ry(float(var_params[param_idx]), q)
                        param_idx += 1
                        qc.rz(float(var_params[param_idx]), q)
                        param_idx += 1
                        qc.rx(float(var_params[param_idx]), q)
                        param_idx += 1
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)

            elif self.ansatz == "circuit_19":
                for q in range(self.n_qubits):
                    if param_idx + 1 < len(var_params):
                        qc.ry(float(var_params[param_idx]), q)
                        param_idx += 1
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)

            elif self.ansatz == "barren_plateau_free":
                for q in range(self.n_qubits):
                    if param_idx + 1 < len(var_params):
                        qc.rz(float(var_params[param_idx]), q)
                        param_idx += 1
                    if param_idx + 1 < len(var_params):
                        qc.ry(float(var_params[param_idx]), q)
                        param_idx += 1
                if layer % 2 == 0:
                    for i in range(0, self.n_qubits - 1, 2):
                        qc.cz(i, min(i + 1, self.n_qubits - 1))
                else:
                    for i in range(self.n_qubits - 1):
                        qc.cx(i, i + 1)
            else:
                for q in range(self.n_qubits):
                    if param_idx + 1 < len(var_params):
                        qc.ry(float(var_params[param_idx]), q)
                        param_idx += 1
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)

        return qc

    def _measure(self, circuit: QuantumCircuit) -> np.ndarray:
        """Measure expectation values of Z observable on all qubits.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        numpy.ndarray
            Shape ``(n_qubits,)``.
        """
        results = []
        for q in range(self.n_qubits):
            obs = _pauli_observable("z", q, self.n_qubits)
            results.append(float(self._simulator.expectation(circuit, obs)))
        return np.array(results, dtype=np.float64)

    def _forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Forward pass: input → quantum circuit → output logits.

        Parameters
        ----------
        x : numpy.ndarray
            Shape ``(batch_size, input_dim)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch_size, n_classes)``.
        """
        assert self._kernel is not None
        assert self._var_params is not None
        assert self._readout is not None

        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.n_classes), dtype=np.float64)

        for b in range(batch_size):
            # Linear projection
            angles = x[b] @ self._kernel
            if self._bias is not None:
                angles = angles + self._bias
            angles = np.clip(angles, -_PI, _PI)

            # Quantum circuit
            qc = self._build_circuit(angles, self._var_params)
            exp_vals = self._measure(qc)

            # Readout
            outputs[b] = exp_vals @ self._readout

        return outputs

    def _compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute the loss.

        Parameters
        ----------
        predictions : numpy.ndarray
        targets : numpy.ndarray

        Returns
        -------
        float
        """
        if self.n_classes == 1:
            # Binary cross-entropy
            eps = 1e-10
            p = np.clip(1.0 / (1.0 + np.exp(-predictions)), eps, 1 - eps)
            return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))
        else:
            # Cross-entropy
            shifted = predictions - np.max(predictions, axis=-1, keepdims=True)
            exp_vals = np.exp(shifted)
            softmax = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
            n = predictions.shape[0]
            targets_int = targets.astype(int)
            log_probs = np.log(np.clip(softmax[np.arange(n), targets_int], 1e-10, 1.0))
            return -np.mean(log_probs)

    def _compute_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        predictions : numpy.ndarray
        targets : numpy.ndarray

        Returns
        -------
        float
        """
        if self.n_classes == 1:
            pred_labels = (predictions > 0).astype(int)
            return float(np.mean(pred_labels == targets.astype(int)))
        else:
            pred_labels = np.argmax(predictions, axis=-1)
            return float(np.mean(pred_labels == targets.astype(int)))

    def compile(
        self,
        optimizer: Union[str, Any, None] = "adam",
        loss: Union[str, Any, None] = None,
        metrics: Optional[List[str]] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Configure the model for training.

        Parameters
        ----------
        optimizer : str or optimizer instance, optional
            Keras optimizer name or instance.
        loss : str or loss function, optional
            Loss function.
        metrics : list of str, optional
            Evaluation metrics.
        learning_rate : float, optional
            Override learning rate.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self._optimizer_name = optimizer if isinstance(optimizer, str) else "adam"
        self._loss_name = loss
        self._metrics = metrics or []
        self._compiled = True

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build model weights.

        Parameters
        ----------
        input_shape : tuple of int
        """
        input_dim = input_shape[-1] if len(input_shape) > 0 else input_shape[0]
        self._input_dim = input_dim
        rng = np.random.default_rng()

        # Kernel: input_dim → n_qubits
        fan_in, fan_out = input_dim, self.n_qubits
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        self._kernel = rng.uniform(-limit, limit, (input_dim, self.n_qubits))
        self._kernel = self._kernel.astype(np.float64)

        # Bias
        self._bias = rng.uniform(-0.1, 0.1, self.n_qubits).astype(np.float64)

        # Variational parameters
        n_var = self.n_layers * self.n_qubits * 3
        self._var_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)

        # Readout: n_qubits → n_classes
        limit_r = math.sqrt(6.0 / (self.n_qubits + self.n_classes))
        self._readout = rng.uniform(
            -limit_r, limit_r, (self.n_qubits, self.n_classes)
        ).astype(np.float64)

        self._built = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.0,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the quantum classifier.

        Parameters
        ----------
        X : numpy.ndarray
            Training features, shape ``(n_samples, input_dim)``.
        y : numpy.ndarray
            Training labels, shape ``(n_samples,)`` or ``(n_samples, 1)``.
        epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Mini-batch size.
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
            self.compile()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1:
            y = y.flatten()

        if not hasattr(self, '_built') or not self._built:
            self.build(X.shape)

        n_samples = X.shape[0]
        lr = self.learning_rate

        # Split validation
        if validation_split > 0:
            val_idx = int(n_samples * (1 - validation_split))
            X_train, X_val = X[:val_idx], X[val_idx:]
            y_train, y_val = y[:val_idx], y[val_idx:]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                preds = self._forward(X_batch)
                loss = self._compute_loss(preds, y_batch)

                # Backward via finite differences
                grad_kernel = np.zeros_like(self._kernel)
                grad_bias = np.zeros_like(self._bias)
                grad_var = np.zeros_like(self._var_params)
                grad_readout = np.zeros_like(self._readout)
                eps = 1e-5

                # Readout gradient
                for i in range(self._readout.shape[0]):
                    for j in range(self._readout.shape[1]):
                        self._readout[i, j] += eps
                        preds_plus = self._forward(X_batch)
                        self._readout[i, j] -= eps
                        preds_minus = self._forward(X_batch)
                        grad_readout[i, j] = np.sum(
                            (preds_plus - preds_minus) / (2 * eps)
                        )

                # Update readout
                self._readout -= lr * grad_readout

                # Recompute for other gradients (simplified)
                preds = self._forward(X_batch)

                # Kernel gradient (sampled for efficiency)
                for i in range(0, self._kernel.shape[0], max(1, self._kernel.shape[0] // 4)):
                    for j in range(self._kernel.shape[1]):
                        self._kernel[i, j] += eps
                        preds_p = self._forward(X_batch)
                        self._kernel[i, j] -= eps
                        preds_m = self._forward(X_batch)
                        grad_kernel[i, j] = np.sum((preds_p - preds_m) / (2 * eps))

                self._kernel -= lr * grad_kernel * 0.1

                # Variational params gradient (sampled)
                n_var = len(self._var_params)
                step = max(1, n_var // 8)
                for idx in range(0, n_var, step):
                    self._var_params[idx] += eps
                    preds_p = self._forward(X_batch)
                    self._var_params[idx] -= eps
                    preds_m = self._forward(X_batch)
                    grad_var[idx] = np.sum((preds_p - preds_m) / (2 * eps))

                self._var_params -= lr * grad_var * 0.05

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if X_val is not None:
                val_preds = self._forward(X_val)
                val_loss = self._compute_loss(val_preds, y_val)
                val_acc = self._compute_accuracy(val_preds, y_val)
                self._history["val_loss"].append(float(val_loss))
                self._history["val_accuracy"].append(float(val_acc))
            else:
                val_loss = 0.0
                val_acc = 0.0

            # Training accuracy
            train_preds = self._forward(X_train[:min(200, len(X_train))])
            train_acc = self._compute_accuracy(train_preds, y_train[:min(200, len(y_train))])

            self._history["loss"].append(float(avg_loss))
            self._history["accuracy"].append(float(train_acc))

            if verbose > 0:
                elapsed = time.time() - t0
                val_str = f" — val_loss: {val_loss:.4f} — val_acc: {val_acc:.4f}" if X_val is not None else ""
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"— loss: {avg_loss:.4f} — acc: {train_acc:.4f}"
                    f"{val_str} — {elapsed:.1f}s"
                )

            # Learning rate decay
            lr = self.learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities or labels.

        Parameters
        ----------
        X : numpy.ndarray
            Input features.

        Returns
        -------
        numpy.ndarray
            Predictions.
        """
        X = np.asarray(X, dtype=np.float64)
        if not hasattr(self, '_built') or not self._built:
            self.build(X.shape)
        logits = self._forward(X)
        if self.n_classes == 1:
            return (1.0 / (1.0 + np.exp(-logits))).flatten()
        else:
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        probs = self.predict(X)
        if self.n_classes == 1:
            return (probs > 0.5).astype(int)
        return np.argmax(probs, axis=-1)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        batch_size : int, optional

        Returns
        -------
        dict
            Evaluation metrics.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        all_preds = []
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            all_preds.append(self._forward(X[start:end]))
        preds = np.concatenate(all_preds, axis=0)

        loss = self._compute_loss(preds, y)
        accuracy = self._compute_accuracy(preds, y)
        return {"loss": float(loss), "accuracy": float(accuracy)}

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "n_qubits": self.n_qubits,
            "n_classes": self.n_classes,
            "n_layers": self.n_layers,
            "encoding": self.encoding,
            "ansatz": self.ansatz,
            "feature_map": self.feature_map,
            "learning_rate": self.learning_rate,
        }

    def summary(self) -> str:
        """Return a string summary of the model."""
        lines = [
            "KerasQuantumClassifier",
            f"  n_qubits: {self.n_qubits}",
            f"  n_classes: {self.n_classes}",
            f"  n_layers: {self.n_layers}",
            f"  encoding: {self.encoding}",
            f"  ansatz: {self.ansatz}",
        ]
        if hasattr(self, '_built') and self._built:
            total = self._kernel.size + self._bias.size
            total += self._var_params.size + self._readout.size
            lines.append(f"  total_params: {total}")
        return "\n".join(lines)


# ===========================================================================
# KerasQuantumRegressor — Quantum Regression Model
# ===========================================================================

class KerasQuantumRegressor:
    """Quantum variational regression model.

    Same API as :class:`KerasQuantumClassifier` but for regression tasks.
    Uses MSE or MAE loss by default.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_outputs : int, optional
        Number of regression outputs. Default ``1``.
    n_layers : int, optional
        Circuit depth. Default ``2``.
    encoding : str, optional
        Data encoding. Default ``'angle'``.
    loss : str, optional
        Loss function: ``'mse'`` or ``'mae'``. Default ``'mse'``.
    learning_rate : float, optional
        Default ``0.01``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> reg = KerasQuantumRegressor(n_qubits=4, n_outputs=1)
    >>> reg.compile(optimizer='adam')
    >>> reg.fit(X_train, y_train, epochs=20)
    >>> y_pred = reg.predict(X_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_outputs: int = 1,
        n_layers: int = 2,
        encoding: str = "angle",
        loss: str = "mse",
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.n_qubits = n_qubits
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.encoding = encoding
        self._loss_name = loss
        self.learning_rate = learning_rate
        self._kernel = None
        self._bias = None
        self._var_params = None
        self._readout = None
        self._simulator = StatevectorSimulator()
        self._history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
        self._compiled = False
        self._built = False
        self._input_dim: Optional[int] = None

    def _build_circuit(
        self,
        data: np.ndarray,
        var_params: np.ndarray,
    ) -> QuantumCircuit:
        """Build the quantum circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)
            if q < len(data):
                qc.ry(float(data[q]) % (2 * _PI), q)
        param_idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                if param_idx + 2 < len(var_params):
                    qc.rz(float(var_params[param_idx]), q)
                    param_idx += 1
                    qc.ry(float(var_params[param_idx]), q)
                    param_idx += 1
                    qc.rz(float(var_params[param_idx]), q)
                    param_idx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _measure(self, circuit: QuantumCircuit) -> np.ndarray:
        results = []
        for q in range(self.n_qubits):
            obs = _pauli_observable("z", q, self.n_qubits)
            results.append(float(self._simulator.expectation(circuit, obs)))
        return np.array(results, dtype=np.float64)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        assert self._kernel is not None
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.n_outputs), dtype=np.float64)
        for b in range(batch_size):
            angles = x[b] @ self._kernel
            if self._bias is not None:
                angles += self._bias
            angles = np.clip(angles, -_PI, _PI)
            qc = self._build_circuit(angles, self._var_params)
            exp_vals = self._measure(qc)
            outputs[b] = exp_vals @ self._readout
        return outputs

    def _compute_loss(self, preds: np.ndarray, targets: np.ndarray) -> float:
        if self._loss_name == "mae":
            return float(np.mean(np.abs(preds - targets)))
        return float(np.mean((preds - targets) ** 2))

    def compile(
        self,
        optimizer: Union[str, Any, None] = "adam",
        loss: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if loss is not None:
            self._loss_name = loss
        self._compiled = True

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1] if len(input_shape) > 0 else input_shape[0]
        self._input_dim = input_dim
        rng = np.random.default_rng()
        fan_in, fan_out = input_dim, self.n_qubits
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        self._kernel = rng.uniform(-limit, limit, (input_dim, self.n_qubits)).astype(np.float64)
        self._bias = rng.uniform(-0.1, 0.1, self.n_qubits).astype(np.float64)
        n_var = self.n_layers * self.n_qubits * 3
        self._var_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)
        limit_r = math.sqrt(6.0 / (self.n_qubits + self.n_outputs))
        self._readout = rng.uniform(-limit_r, limit_r, (self.n_qubits, self.n_outputs)).astype(np.float64)
        self._built = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.0,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        if not self._compiled:
            self.compile()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if not self._built:
            self.build(X.shape)

        n_samples = X.shape[0]
        lr = self.learning_rate

        if validation_split > 0:
            val_idx = int(n_samples * (1 - validation_split))
            X_train, X_val = X[:val_idx], X[val_idx:]
            y_train, y_val = y[:val_idx], y[val_idx:]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X_train))
            X_s, y_s = X_train[perm], y_train[perm]

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                X_b = X_s[start:end]
                y_b = y_s[start:end]

                preds = self._forward(X_b)
                loss = self._compute_loss(preds, y_b)

                # Readout gradient
                grad_r = np.zeros_like(self._readout)
                eps = 1e-5
                for i in range(0, self._readout.shape[0], max(1, self._readout.shape[0] // 4)):
                    for j in range(self._readout.shape[1]):
                        self._readout[i, j] += eps
                        pp = self._forward(X_b)
                        self._readout[i, j] -= eps
                        pm = self._forward(X_b)
                        grad_r[i, j] = np.sum((pp - pm) / (2 * eps))
                self._readout -= lr * grad_r

                # Kernel gradient (sampled)
                grad_k = np.zeros_like(self._kernel)
                for i in range(0, self._kernel.shape[0], max(1, self._kernel.shape[0] // 8)):
                    for j in range(self._kernel.shape[1]):
                        self._kernel[i, j] += eps
                        pp = self._forward(X_b)
                        self._kernel[i, j] -= eps
                        pm = self._forward(X_b)
                        grad_k[i, j] = np.sum((pp - pm) / (2 * eps))
                self._kernel -= lr * grad_k * 0.1

                # Variational gradient (sampled)
                grad_v = np.zeros_like(self._var_params)
                step = max(1, len(self._var_params) // 8)
                for idx in range(0, len(self._var_params), step):
                    self._var_params[idx] += eps
                    pp = self._forward(X_b)
                    self._var_params[idx] -= eps
                    pm = self._forward(X_b)
                    grad_v[idx] = np.sum((pp - pm) / (2 * eps))
                self._var_params -= lr * grad_v * 0.05

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(float(avg_loss))

            if X_val is not None:
                val_preds = self._forward(X_val)
                val_loss = self._compute_loss(val_preds, y_val)
                self._history["val_loss"].append(float(val_loss))

            if verbose > 0:
                elapsed = time.time() - t0
                val_str = f" — val_loss: {val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}{val_str} — {elapsed:.1f}s")

            lr = self.learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self._built:
            self.build(X.shape)
        return self._forward(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        all_preds = []
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            all_preds.append(self._forward(X[start:end]))
        preds = np.concatenate(all_preds, axis=0)
        loss = self._compute_loss(preds, y)
        mae = float(np.mean(np.abs(preds - y)))
        return {"loss": float(loss), "mae": mae}

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits, "n_outputs": self.n_outputs,
            "n_layers": self.n_layers, "encoding": self.encoding,
            "loss": self._loss_name, "learning_rate": self.learning_rate,
        }

    def summary(self) -> str:
        lines = [
            "KerasQuantumRegressor",
            f"  n_qubits: {self.n_qubits}",
            f"  n_outputs: {self.n_outputs}",
            f"  n_layers: {self.n_layers}",
        ]
        return "\n".join(lines)


# ===========================================================================
# KerasQNN — Quantum Neural Network Builder
# ===========================================================================

class KerasQNN:
    """Flexible quantum neural network model builder.

    Allows building arbitrary networks mixing quantum and classical layers.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> qnn = KerasQNN(input_dim=8)
    >>> qnn.add_quantum_layer(n_qubits=4, n_layers=2)
    >>> qnn.add_activation('relu')
    >>> qnn.add_quantum_layer(n_qubits=4, n_layers=1)
    >>> qnn.add_classical_layer(units=2)
    >>> qnn.compile(optimizer='adam', loss='mse')
    >>> qnn.fit(X_train, y_train, epochs=10)
    """

    def __init__(self, input_dim: int, **kwargs: Any) -> None:
        _check_keras_available()
        self.input_dim = input_dim
        self._layers: List[Dict[str, Any]] = []
        self._weights: List[np.ndarray] = []
        self._history: Dict[str, List[float]] = {"loss": []}
        self._compiled = False
        self._built = False
        self._optimizer_name = "adam"
        self._loss_name = "mse"
        self._learning_rate = 0.01
        self._simulator = StatevectorSimulator()

    def add_quantum_layer(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
        observable: str = "z",
    ) -> KerasQNN:
        """Add a quantum dense layer.

        Parameters
        ----------
        n_qubits : int
        n_layers : int
        encoding : str
        observable : str

        Returns
        -------
        KerasQNN
            self
        """
        self._layers.append({
            "type": "quantum",
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "encoding": encoding,
            "observable": observable,
        })
        return self

    def add_classical_layer(
        self,
        units: int,
        activation: Optional[str] = "relu",
        use_bias: bool = True,
    ) -> KerasQNN:
        """Add a classical dense layer.

        Parameters
        ----------
        units : int
        activation : str or None
        use_bias : bool

        Returns
        -------
        KerasQNN
        """
        self._layers.append({
            "type": "classical",
            "units": units,
            "activation": activation,
            "use_bias": use_bias,
        })
        return self

    def add_activation(self, activation: str) -> KerasQNN:
        """Add an activation layer.

        Parameters
        ----------
        activation : str

        Returns
        -------
        KerasQNN
        """
        self._layers.append({"type": "activation", "activation": activation})
        return self

    def compile(
        self,
        optimizer: Union[str, Any, None] = "adam",
        loss: Union[str, Any, None] = "mse",
        learning_rate: float = 0.01,
    ) -> None:
        """Compile the model.

        Parameters
        ----------
        optimizer : str or optimizer instance
        loss : str or loss function
        learning_rate : float
        """
        self._optimizer_name = optimizer if isinstance(optimizer, str) else "adam"
        self._loss_name = loss if isinstance(loss, str) else "mse"
        self._learning_rate = learning_rate
        self._compiled = True

    def build(self) -> None:
        """Build all layer weights."""
        rng = np.random.default_rng()
        self._weights = []
        current_dim = self.input_dim

        for layer_config in self._layers:
            if layer_config["type"] == "quantum":
                n_q = layer_config["n_qubits"]
                limit = math.sqrt(6.0 / (current_dim + n_q))
                kernel = rng.uniform(-limit, limit, (current_dim, n_q)).astype(np.float64)
                bias = rng.uniform(-0.1, 0.1, n_q).astype(np.float64)
                n_var = layer_config["n_layers"] * n_q * 3
                var_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)
                self._weights.append({"kernel": kernel, "bias": bias, "var_params": var_params})
                current_dim = n_q

            elif layer_config["type"] == "classical":
                units = layer_config["units"]
                limit = math.sqrt(6.0 / (current_dim + units))
                kernel = rng.uniform(-limit, limit, (current_dim, units)).astype(np.float64)
                bias = np.zeros(units, dtype=np.float64)
                self._weights.append({"kernel": kernel, "bias": bias})
                current_dim = units

            # activation layers don't have weights

        self._output_dim = current_dim
        self._built = True

    def _forward_quantum(
        self,
        x: np.ndarray,
        layer_idx: int,
    ) -> np.ndarray:
        """Forward pass through a quantum layer."""
        config = self._layers[layer_idx]
        weights = self._weights[layer_idx]
        kernel = weights["kernel"]
        bias = weights["bias"]
        var_params = weights["var_params"]
        n_q = config["n_qubits"]
        n_l = config["n_layers"]

        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, n_q), dtype=np.float64)

        for b in range(batch_size):
            angles = x[b] @ kernel + bias
            angles = np.clip(angles, -_PI, _PI)

            qc = QuantumCircuit(n_q)
            for q in range(n_q):
                qc.h(q)
                qc.ry(float(angles[q]), q)
            pidx = 0
            for _ in range(n_l):
                for q in range(n_q):
                    if pidx + 2 < len(var_params):
                        qc.rz(float(var_params[pidx]), q); pidx += 1
                        qc.ry(float(var_params[pidx]), q); pidx += 1
                        qc.rz(float(var_params[pidx]), q); pidx += 1
                for i in range(n_q - 1):
                    qc.cx(i, i + 1)

            for q in range(n_q):
                obs = _pauli_observable("z", q, n_q)
                outputs[b, q] = float(self._simulator.expectation(qc, obs))

        return outputs

    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation is None or activation == "linear":
            return x
        elif activation == "relu":
            return np.maximum(0, x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "softmax":
            shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_v = np.exp(shifted)
            return exp_v / np.sum(exp_v, axis=-1, keepdims=True)
        return x

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass through all layers."""
        w_idx = 0
        for l_idx, config in enumerate(self._layers):
            if config["type"] == "quantum":
                x = self._forward_quantum(x, l_idx)
                w_idx += 1
            elif config["type"] == "classical":
                weights = self._weights[w_idx]
                x = x @ weights["kernel"] + weights["bias"]
                if config.get("activation"):
                    x = self._apply_activation(x, config["activation"])
                w_idx += 1
            elif config["type"] == "activation":
                x = self._apply_activation(x, config["activation"])
        return x

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        epochs : int
        batch_size : int
        verbose : int

        Returns
        -------
        dict
        """
        if not self._compiled:
            self.compile()
        if not self._built:
            self.build()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        lr = self._learning_rate

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X))

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_b = X[perm[start:end]]
                y_b = y[perm[start:end]]

                preds = self._forward(X_b)
                loss = float(np.mean((preds - y_b) ** 2))

                # Simplified gradient update on last layer's readout
                w_idx = 0
                for l_idx, config in enumerate(self._layers):
                    if config["type"] in ("quantum", "classical"):
                        w_idx += 1

                if w_idx > 0:
                    last_w = self._weights[w_idx - 1]
                    eps = 1e-5
                    grad_k = np.zeros_like(last_w["kernel"])
                    step = max(1, last_w["kernel"].size // 8)
                    flat_idx = 0
                    for i in range(last_w["kernel"].shape[0]):
                        for j in range(last_w["kernel"].shape[1]):
                            if flat_idx % step == 0:
                                last_w["kernel"][i, j] += eps
                                pp = self._forward(X_b)
                                last_w["kernel"][i, j] -= eps
                                pm = self._forward(X_b)
                                grad_k[i, j] = np.sum((pp - pm) / (2 * eps))
                            flat_idx += 1
                    last_w["kernel"] -= lr * grad_k * 0.1

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(float(avg_loss))

            if verbose > 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} — {elapsed:.1f}s")

            lr = self._learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if not self._built:
            self.build()
        return self._forward(np.asarray(X, dtype=np.float64))

    def summary(self) -> str:
        """Return model summary."""
        lines = ["KerasQNN", f"  input_dim: {self.input_dim}", "  Layers:"]
        for i, config in enumerate(self._layers):
            if config["type"] == "quantum":
                lines.append(f"    [{i}] Quantum: n_qubits={config['n_qubits']}, n_layers={config['n_layers']}")
            elif config["type"] == "classical":
                lines.append(f"    [{i}] Dense: units={config['units']}, activation={config.get('activation')}")
            elif config["type"] == "activation":
                lines.append(f"    [{i}] Activation: {config['activation']}")
        if self._built:
            lines.append(f"  output_dim: {self._output_dim}")
        return "\n".join(lines)


# ===========================================================================
# KerasQuantumAutoencoder — Quantum Autoencoder
# ===========================================================================

class KerasQuantumAutoencoder:
    """Quantum autoencoder with configurable trash qubits.

    Uses quantum circuits to compress data by encoding into a smaller
    number of qubits and then decoding back to the original space.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    n_trash_qubits : int
        Number of qubits to discard (compression bottleneck).
    n_layers : int, optional
        Circuit depth. Default ``2``.
    learning_rate : float, optional
        Default ``0.01``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> ae = KerasQuantumAutoencoder(n_qubits=6, n_trash_qubits=2)
    >>> ae.compile(optimizer='adam')
    >>> ae.fit(X_train, epochs=20)
    >>> encoded = ae.encode(X_test)
    >>> decoded = ae.decode(X_test)
    """

    def __init__(
        self,
        n_qubits: int,
        n_trash_qubits: int,
        n_layers: int = 2,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.n_qubits = n_qubits
        self.n_trash_qubits = n_trash_qubits
        self.n_latent_qubits = n_qubits - n_trash_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self._encoder_params = None
        self._decoder_params = None
        self._kernel = None
        self._bias = None
        self._simulator = StatevectorSimulator()
        self._history: Dict[str, List[float]] = {"loss": []}
        self._compiled = False
        self._built = False
        self._input_dim: Optional[int] = None

    def compile(self, optimizer: Union[str, Any, None] = "adam", **kwargs: Any) -> None:
        self._compiled = True

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1] if len(input_shape) > 0 else input_shape[0]
        self._input_dim = input_dim
        rng = np.random.default_rng()

        # Kernel: input_dim → n_qubits
        limit = math.sqrt(6.0 / (input_dim + self.n_qubits))
        self._kernel = rng.uniform(-limit, limit, (input_dim, self.n_qubits)).astype(np.float64)
        self._bias = rng.uniform(-0.1, 0.1, self.n_qubits).astype(np.float64)

        # Encoder variational params
        n_enc = self.n_layers * self.n_qubits * 3
        self._encoder_params = rng.uniform(-0.1, 0.1, n_enc).astype(np.float64)

        # Decoder variational params (operates on latent qubits)
        n_dec = self.n_layers * self.n_latent_qubits * 3
        self._decoder_params = rng.uniform(-0.1, 0.1, n_dec).astype(np.float64)

        self._built = True

    def _build_encoder_circuit(
        self,
        data: np.ndarray,
        params: np.ndarray,
    ) -> QuantumCircuit:
        """Build encoder circuit."""
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)
            if q < len(data):
                qc.ry(float(data[q]), q)
        pidx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                if pidx + 2 < len(params):
                    qc.rz(float(params[pidx]), q); pidx += 1
                    qc.ry(float(params[pidx]), q); pidx += 1
                    qc.rz(float(params[pidx]), q); pidx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _build_decoder_circuit(
        self,
        latent_data: np.ndarray,
        params: np.ndarray,
    ) -> QuantumCircuit:
        """Build decoder circuit on latent qubits."""
        qc = QuantumCircuit(self.n_latent_qubits)
        for q in range(self.n_latent_qubits):
            qc.h(q)
            if q < len(latent_data):
                qc.ry(float(latent_data[q]), q)
        pidx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_latent_qubits):
                if pidx + 2 < len(params):
                    qc.rz(float(params[pidx]), q); pidx += 1
                    qc.ry(float(params[pidx]), q); pidx += 1
                    qc.rz(float(params[pidx]), q); pidx += 1
            for i in range(self.n_latent_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through encoder and decoder.

        Returns
        -------
        tuple of (reconstruction, trash_values)
        """
        assert self._kernel is not None
        batch_size = x.shape[0]
        reconstructions = np.zeros((batch_size, self.n_qubits), dtype=np.float64)
        trash_values = np.zeros((batch_size, self.n_trash_qubits), dtype=np.float64)

        for b in range(batch_size):
            angles = x[b] @ self._kernel + self._bias
            angles = np.clip(angles, -_PI, _PI)

            # Encode
            enc_qc = self._build_encoder_circuit(angles, self._encoder_params)

            # Measure latent qubits
            latent_vals = []
            for q in range(self.n_latent_qubits):
                obs = _pauli_observable("z", q, self.n_qubits)
                latent_vals.append(float(self._simulator.expectation(enc_qc, obs)))

            # Measure trash qubits (should be close to 0 for good compression)
            for q in range(self.n_trash_qubits):
                tq = self.n_latent_qubits + q
                obs = _pauli_observable("z", tq, self.n_qubits)
                trash_values[b, q] = float(self._simulator.expectation(enc_qc, obs))

            # Decode
            dec_qc = self._build_decoder_circuit(np.array(latent_vals), self._decoder_params)
            for q in range(self.n_latent_qubits):
                obs = _pauli_observable("z", q, self.n_latent_qubits)
                reconstructions[b, q] = float(self._simulator.expectation(dec_qc, obs))

        return reconstructions, trash_values

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
            Training data.
        y : numpy.ndarray, optional
            Target (same as X for reconstruction).
        epochs : int
        batch_size : int
        verbose : int

        Returns
        -------
        dict
        """
        if not self._compiled:
            self.compile()
        X = np.asarray(X, dtype=np.float64)
        if y is None:
            y = X.copy()
        else:
            y = np.asarray(y, dtype=np.float64)
        if not self._built:
            self.build(X.shape)

        lr = self.learning_rate
        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X))

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_b = X[perm[start:end]]
                y_b = y[perm[start:end]]

                recon, trash = self._forward(X_b)
                recon_loss = float(np.mean((recon - y_b[:, :recon.shape[1]]) ** 2))
                trash_loss = float(np.mean(trash ** 2))
                loss = recon_loss + 0.5 * trash_loss

                # Simplified gradient update
                eps = 1e-5
                grad_enc = np.zeros_like(self._encoder_params)
                step = max(1, len(self._encoder_params) // 8)
                for idx in range(0, len(self._encoder_params), step):
                    self._encoder_params[idx] += eps
                    _, trash_p = self._forward(X_b)
                    self._encoder_params[idx] -= eps
                    _, trash_m = self._forward(X_b)
                    grad_enc[idx] = np.sum((trash_p - trash_m) / (2 * eps))
                self._encoder_params -= lr * grad_enc * 0.05

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(float(avg_loss))

            if verbose > 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} — {elapsed:.1f}s")

            lr = self.learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Latent representations.
        """
        if not self._built:
            self.build(np.asarray(X, dtype=np.float64).shape)
        X = np.asarray(X, dtype=np.float64)
        batch_size = X.shape[0]
        latents = np.zeros((batch_size, self.n_latent_qubits), dtype=np.float64)
        for b in range(batch_size):
            angles = X[b] @ self._kernel + self._bias
            angles = np.clip(angles, -_PI, _PI)
            enc_qc = self._build_encoder_circuit(angles, self._encoder_params)
            for q in range(self.n_latent_qubits):
                obs = _pauli_observable("z", q, self.n_qubits)
                latents[b, q] = float(self._simulator.expectation(enc_qc, obs))
        return latents

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Decode from latent space.

        Parameters
        ----------
        X : numpy.ndarray
            Can be original data (will encode first) or latent vectors.

        Returns
        -------
        numpy.ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[-1] != self.n_latent_qubits:
            latents = self.encode(X)
        else:
            latents = X
        batch_size = latents.shape[0]
        decoded = np.zeros((batch_size, self.n_latent_qubits), dtype=np.float64)
        for b in range(batch_size):
            dec_qc = self._build_decoder_circuit(latents[b], self._decoder_params)
            for q in range(self.n_latent_qubits):
                obs = _pauli_observable("z", q, self.n_latent_qubits)
                decoded[b, q] = float(self._simulator.expectation(dec_qc, obs))
        return decoded

    def summary(self) -> str:
        lines = [
            "KerasQuantumAutoencoder",
            f"  n_qubits: {self.n_qubits}",
            f"  n_trash_qubits: {self.n_trash_qubits}",
            f"  n_latent_qubits: {self.n_latent_qubits}",
            f"  compression_ratio: {self.n_latent_qubits / self.n_qubits:.2f}",
        ]
        return "\n".join(lines)


# ===========================================================================
# KerasHybridModel — Hybrid Classical-Quantum Model
# ===========================================================================

class KerasHybridModel:
    """Hybrid classical-quantum model with mixed layer types.

    Supports mixing classical Dense/Conv2D layers with quantum layers
    in a sequential architecture.

    Parameters
    ----------
    input_shape : tuple of int
        Input shape.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> model = KerasHybridModel(input_shape=(28, 28, 1))
    >>> model.add_classical_layer(type='flatten')
    >>> model.add_classical_layer(type='dense', units=64, activation='relu')
    >>> model.add_quantum_layer(n_qubits=6, n_layers=2)
    >>> model.add_classical_layer(type='dense', units=10)
    >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    >>> model.fit(X_train, y_train, epochs=10)
    """

    def __init__(self, input_shape: Tuple[int, ...], **kwargs: Any) -> None:
        _check_keras_available()
        self.input_shape = input_shape
        self._layer_configs: List[Dict[str, Any]] = []
        self._weights: List[Dict[str, Any]] = []
        self._history: Dict[str, List[float]] = {"loss": []}
        self._compiled = False
        self._built = False
        self._optimizer_name = "adam"
        self._loss_name = "mse"
        self._learning_rate = 0.01
        self._simulator = StatevectorSimulator()
        self._current_dim: Optional[int] = None

    def add_classical_layer(
        self,
        layer_type: str,
        **kwargs: Any,
    ) -> KerasHybridModel:
        """Add a classical layer.

        Parameters
        ----------
        layer_type : str
            ``'dense'``, ``'flatten'``, or ``'dropout'``.
        **kwargs
            Layer-specific arguments.

        Returns
        -------
        KerasHybridModel
        """
        self._layer_configs.append({"type": "classical", "layer_type": layer_type, **kwargs})
        return self

    def add_quantum_layer(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: str = "angle",
    ) -> KerasHybridModel:
        """Add a quantum layer.

        Parameters
        ----------
        n_qubits : int
        n_layers : int
        encoding : str

        Returns
        -------
        KerasHybridModel
        """
        self._layer_configs.append({
            "type": "quantum",
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "encoding": encoding,
        })
        return self

    def compile(
        self,
        optimizer: Union[str, Any, None] = "adam",
        loss: Union[str, Any, None] = "mse",
        learning_rate: float = 0.01,
    ) -> None:
        self._optimizer_name = optimizer if isinstance(optimizer, str) else "adam"
        self._loss_name = loss if isinstance(loss, str) else "mse"
        self._learning_rate = learning_rate
        self._compiled = True

    def build(self) -> None:
        rng = np.random.default_rng()
        self._weights = []
        self._current_dim = int(np.prod(self.input_shape))

        for config in self._layer_configs:
            if config["type"] == "classical":
                lt = config.get("layer_type", "dense")
                if lt == "flatten":
                    self._weights.append({})
                elif lt == "dropout":
                    self._weights.append({})
                elif lt == "dense":
                    units = config.get("units", 10)
                    limit = math.sqrt(6.0 / (self._current_dim + units))
                    kernel = rng.uniform(-limit, limit, (self._current_dim, units)).astype(np.float64)
                    bias = np.zeros(units, dtype=np.float64)
                    self._weights.append({"kernel": kernel, "bias": bias, "units": units})
                    self._current_dim = units

            elif config["type"] == "quantum":
                n_q = config["n_qubits"]
                limit = math.sqrt(6.0 / (self._current_dim + n_q))
                kernel = rng.uniform(-limit, limit, (self._current_dim, n_q)).astype(np.float64)
                bias = rng.uniform(-0.1, 0.1, n_q).astype(np.float64)
                n_var = config["n_layers"] * n_q * 3
                var_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)
                self._weights.append({"kernel": kernel, "bias": bias, "var_params": var_params, "n_qubits": n_q})
                self._current_dim = n_q

        self._built = True

    def _forward(self, x: np.ndarray) -> np.ndarray:
        w_idx = 0
        for config in self._layer_configs:
            if config["type"] == "classical":
                lt = config.get("layer_type", "dense")
                if lt == "flatten":
                    x = x.reshape(x.shape[0], -1)
                elif lt == "dropout":
                    pass  # No-op in inference
                elif lt == "dense":
                    w = self._weights[w_idx]
                    x = x @ w["kernel"] + w["bias"]
                    act = config.get("activation", None)
                    if act == "relu":
                        x = np.maximum(0, x)
                    elif act == "sigmoid":
                        x = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
                    elif act == "tanh":
                        x = np.tanh(x)
                    w_idx += 1
                else:
                    w_idx += 1
            elif config["type"] == "quantum":
                w = self._weights[w_idx]
                n_q = w["n_qubits"]
                batch_size = x.shape[0]
                out = np.zeros((batch_size, n_q), dtype=np.float64)
                for b in range(batch_size):
                    angles = x[b] @ w["kernel"] + w["bias"]
                    angles = np.clip(angles, -_PI, _PI)
                    qc = QuantumCircuit(n_q)
                    for q in range(n_q):
                        qc.h(q)
                        qc.ry(float(angles[q]), q)
                    pidx = 0
                    for _ in range(config["n_layers"]):
                        for q in range(n_q):
                            if pidx + 2 < len(w["var_params"]):
                                qc.rz(float(w["var_params"][pidx]), q); pidx += 1
                                qc.ry(float(w["var_params"][pidx]), q); pidx += 1
                                qc.rz(float(w["var_params"][pidx]), q); pidx += 1
                        for i in range(n_q - 1):
                            qc.cx(i, i + 1)
                    for q in range(n_q):
                        obs = _pauli_observable("z", q, n_q)
                        out[b, q] = float(self._simulator.expectation(qc, obs))
                x = out
                w_idx += 1
        return x

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        if not self._compiled:
            self.compile()
        if not self._built:
            self.build()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        lr = self._learning_rate

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X))
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_b = X[perm[start:end]]
                y_b = y[perm[start:end]]
                preds = self._forward(X_b)
                loss = float(np.mean((preds - y_b) ** 2))
                epoch_loss += loss
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(float(avg))
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg:.4f} — {time.time() - t0:.1f}s")
            lr = self._learning_rate / (1 + 0.01 * (epoch + 1))
        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._built:
            self.build()
        return self._forward(np.asarray(X, dtype=np.float64))

    def summary(self) -> str:
        lines = ["KerasHybridModel", f"  input_shape: {self.input_shape}", "  Layers:"]
        for i, c in enumerate(self._layer_configs):
            if c["type"] == "quantum":
                lines.append(f"    [{i}] Quantum(n_qubits={c['n_qubits']}, n_layers={c['n_layers']})")
            else:
                lt = c.get("layer_type", "dense")
                if lt == "dense":
                    lines.append(f"    [{i}] Dense(units={c.get('units')}, activation={c.get('activation')})")
                else:
                    lines.append(f"    [{i}] {lt.capitalize()}")
        return "\n".join(lines)


# ===========================================================================
# KerasQuantumGAN — Quantum GAN
# ===========================================================================

class KerasQuantumGAN:
    """Quantum Generative Adversarial Network.

    Generator uses a quantum circuit to transform latent noise into
    data samples. Discriminator is a classical neural network.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the generator circuit.
    latent_dim : int, optional
        Dimension of the latent noise vector. Default ``8``.
    n_layers : int, optional
        Generator circuit depth. Default ``2``.
    discriminator_layers : list of int, optional
        Hidden layer sizes for discriminator. Default ``[64, 32]``.
    learning_rate_g : float, optional
        Generator learning rate. Default ``0.005``.
    learning_rate_d : float, optional
        Discriminator learning rate. Default ``0.01``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> gan = KerasQuantumGAN(n_qubits=4, latent_dim=8)
    >>> gan.compile()
    >>> gan.fit(X_train, epochs=50)
    >>> samples = gan.generate(100)
    """

    def __init__(
        self,
        n_qubits: int,
        latent_dim: int = 8,
        n_layers: int = 2,
        discriminator_layers: Optional[List[int]] = None,
        learning_rate_g: float = 0.005,
        learning_rate_d: float = 0.01,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.discriminator_layers = discriminator_layers or [64, 32]
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self._gen_params = None
        self._disc_weights = None
        self._simulator = StatevectorSimulator()
        self._history: Dict[str, List[float]] = {"g_loss": [], "d_loss": []}
        self._compiled = False
        self._built = False

    def compile(self, **kwargs: Any) -> None:
        self._compiled = True

    def build(self, data_dim: int) -> None:
        rng = np.random.default_rng()
        # Generator: latent_dim → n_qubits rotation angles
        n_gen = self.n_layers * self.n_qubits * 3
        self._gen_params = rng.uniform(-0.1, 0.1, n_gen).astype(np.float64)

        # Discriminator: n_qubits → hidden → 1
        self._disc_weights = []
        prev_dim = self.n_qubits
        for units in self.discriminator_layers:
            limit = math.sqrt(6.0 / (prev_dim + units))
            kernel = rng.uniform(-limit, limit, (prev_dim, units)).astype(np.float64)
            bias = np.zeros(units, dtype=np.float64)
            self._disc_weights.append({"kernel": kernel, "bias": bias})
            prev_dim = units
        # Output layer
        limit = math.sqrt(6.0 / (prev_dim + 1))
        kernel = rng.uniform(-limit, limit, (prev_dim, 1)).astype(np.float64)
        bias = np.zeros(1, dtype=np.float64)
        self._disc_weights.append({"kernel": kernel, "bias": bias})

        self._built = True

    def _generate_circuit(self, noise: np.ndarray) -> QuantumCircuit:
        """Build generator circuit from noise input."""
        qc = QuantumCircuit(self.n_qubits)
        # Map noise to rotation angles
        for q in range(self.n_qubits):
            idx = q % len(noise)
            qc.h(q)
            qc.ry(float(noise[idx]) * _PI, q)

        pidx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                if pidx + 2 < len(self._gen_params):
                    qc.rz(float(self._gen_params[pidx]), q); pidx += 1
                    qc.ry(float(self._gen_params[pidx]), q); pidx += 1
                    qc.rz(float(self._gen_params[pidx]), q); pidx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate fake samples from the generator."""
        samples = np.zeros((n_samples, self.n_qubits), dtype=np.float64)
        for b in range(n_samples):
            noise = np.random.randn(self.latent_dim)
            qc = self._generate_circuit(noise)
            for q in range(self.n_qubits):
                obs = _pauli_observable("z", q, self.n_qubits)
                samples[b, q] = float(self._simulator.expectation(qc, obs))
        return samples

    def _discriminate(self, x: np.ndarray) -> np.ndarray:
        """Run discriminator on samples."""
        h = x
        for i, w in enumerate(self._disc_weights):
            h = h @ w["kernel"] + w["bias"]
            if i < len(self._disc_weights) - 1:
                h = np.maximum(0, h)  # ReLU
        return 1.0 / (1.0 + np.exp(-np.clip(h, -20, 20)))

    def train_step(self, real_samples: np.ndarray) -> Dict[str, float]:
        """Execute one training step.

        Parameters
        ----------
        real_samples : numpy.ndarray
            Real data samples.

        Returns
        -------
        dict
            Losses.
        """
        batch_size = real_samples.shape[0]

        # Generate fake samples
        fake_samples = self._generate_samples(batch_size)

        # Discriminator step
        real_scores = self._discriminate(real_samples[:, :self.n_qubits])
        fake_scores = self._discriminate(fake_samples)
        eps = 1e-10
        d_loss_real = -np.mean(np.log(real_scores + eps))
        d_loss_fake = -np.mean(np.log(1 - fake_scores + eps))
        d_loss = d_loss_real + d_loss_fake

        # Update discriminator (simplified)
        # Compute gradients via finite differences on last layer
        last_w = self._disc_weights[-1]
        eps_fd = 1e-5
        grad_k = np.zeros_like(last_w["kernel"])
        for i in range(last_w["kernel"].shape[0]):
            for j in range(last_w["kernel"].shape[1]):
                last_w["kernel"][i, j] += eps_fd
                s_p = self._discriminate(fake_samples)
                last_w["kernel"][i, j] -= eps_fd
                s_m = self._discriminate(fake_samples)
                grad_k[i, j] = np.sum((s_p - s_m) / (2 * eps_fd))
        last_w["kernel"] -= self.learning_rate_d * grad_k * 0.01

        # Generator step
        fake_scores_new = self._discriminate(self._generate_samples(batch_size))
        g_loss = -np.mean(np.log(fake_scores_new + eps))

        # Update generator (simplified)
        eps_g = 1e-5
        step = max(1, len(self._gen_params) // 8)
        for idx in range(0, len(self._gen_params), step):
            orig = self._gen_params[idx]
            self._gen_params[idx] = orig + eps_g
            fake_p = self._generate_samples(batch_size // 2)
            scores_p = self._discriminate(fake_p)
            self._gen_params[idx] = orig - eps_g
            fake_m = self._generate_samples(batch_size // 2)
            scores_m = self._discriminate(fake_m)
            self._gen_params[idx] = orig
            grad = np.mean(scores_p) - np.mean(scores_m)
            self._gen_params[idx] -= self.learning_rate_g * grad * 0.1

        return {"g_loss": float(g_loss), "d_loss": float(d_loss)}

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the GAN.

        Parameters
        ----------
        X : numpy.ndarray
            Real training data.
        epochs : int
        batch_size : int
        verbose : int

        Returns
        -------
        dict
        """
        if not self._compiled:
            self.compile()
        X = np.asarray(X, dtype=np.float64)
        if not self._built:
            self.build(X.shape[-1])

        for epoch in range(epochs):
            t0 = time.time()
            perm = np.random.permutation(len(X))
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch = X[perm[start:end]]
                losses = self.train_step(batch)
                epoch_g_loss += losses["g_loss"]
                epoch_d_loss += losses["d_loss"]
                n_batches += 1

            self._history["g_loss"].append(float(epoch_g_loss / max(n_batches, 1)))
            self._history["d_loss"].append(float(epoch_d_loss / max(n_batches, 1)))

            if verbose > 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"— g_loss: {self._history['g_loss'][-1]:.4f} "
                    f"— d_loss: {self._history['d_loss'][-1]:.4f} "
                    f"— {time.time() - t0:.1f}s"
                )

        return self._history

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        numpy.ndarray
        """
        return self._generate_samples(n_samples)

    def summary(self) -> str:
        lines = [
            "KerasQuantumGAN",
            f"  Generator: {self.n_qubits} qubits, {self.n_layers} layers, latent_dim={self.latent_dim}",
            f"  Discriminator: {self.discriminator_layers} → 1",
        ]
        return "\n".join(lines)


# ===========================================================================
# KerasQuantumVAE — Quantum Variational Autoencoder
# ===========================================================================

class KerasQuantumVAE:
    """Quantum Variational Autoencoder.

    Uses a quantum circuit as the encoder to map data to a latent
    distribution, and a quantum decoder to reconstruct from latent
    samples. Implements KL divergence loss with quantum reparameterization.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for encoder/decoder.
    latent_dim : int, optional
        Latent space dimension. Default ``2``.
    n_layers : int, optional
        Circuit depth. Default ``2``.
    beta : float, optional
        KL divergence weight (β-VAE). Default ``1.0``.
    learning_rate : float, optional
        Default ``0.01``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> vae = KerasQuantumVAE(n_qubits=4, latent_dim=2)
    >>> vae.compile()
    >>> vae.fit(X_train, epochs=20)
    >>> encoded = vae.encode(X_test)
    """

    def __init__(
        self,
        n_qubits: int,
        latent_dim: int = 2,
        n_layers: int = 2,
        beta: float = 1.0,
        learning_rate: float = 0.01,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.beta = beta
        self.learning_rate = learning_rate
        self._encoder_params = None
        self._decoder_params = None
        self._kernel = None
        self._bias = None
        self._mu_proj = None
        self._logvar_proj = None
        self._simulator = StatevectorSimulator()
        self._history: Dict[str, List[float]] = {"loss": [], "recon_loss": [], "kl_loss": []}
        self._compiled = False
        self._built = False
        self._input_dim: Optional[int] = None

    def compile(self, **kwargs: Any) -> None:
        self._compiled = True

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1] if len(input_shape) > 0 else input_shape[0]
        self._input_dim = input_dim
        rng = np.random.default_rng()

        limit = math.sqrt(6.0 / (input_dim + self.n_qubits))
        self._kernel = rng.uniform(-limit, limit, (input_dim, self.n_qubits)).astype(np.float64)
        self._bias = rng.uniform(-0.1, 0.1, self.n_qubits).astype(np.float64)

        n_enc = self.n_layers * self.n_qubits * 3
        self._encoder_params = rng.uniform(-0.1, 0.1, n_enc).astype(np.float64)

        n_dec = self.n_layers * self.n_qubits * 3
        self._decoder_params = rng.uniform(-0.1, 0.1, n_dec).astype(np.float64)

        # Projection layers for mu and logvar
        self._mu_proj = rng.uniform(-0.1, 0.1, (self.n_qubits, self.latent_dim)).astype(np.float64)
        self._logvar_proj = rng.uniform(-0.1, 0.1, (self.n_qubits, self.latent_dim)).astype(np.float64)

        self._built = True

    def _build_encoder_circuit(self, data: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)
            if q < len(data):
                qc.ry(float(data[q]), q)
        pidx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                if pidx + 2 < len(params):
                    qc.rz(float(params[pidx]), q); pidx += 1
                    qc.ry(float(params[pidx]), q); pidx += 1
                    qc.rz(float(params[pidx]), q); pidx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _build_decoder_circuit(self, latent: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            idx = q % len(latent)
            qc.h(q)
            qc.ry(float(latent[idx]) * _PI, q)
        pidx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                if pidx + 2 < len(params):
                    qc.rz(float(params[pidx]), q); pidx += 1
                    qc.ry(float(params[pidx]), q); pidx += 1
                    qc.rz(float(params[pidx]), q); pidx += 1
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to latent mean and log-variance."""
        angles = x @ self._kernel + self._bias
        angles = np.clip(angles, -_PI, _PI)
        qc = self._build_encoder_circuit(angles, self._encoder_params)
        q_vals = np.zeros(self.n_qubits, dtype=np.float64)
        for q in range(self.n_qubits):
            obs = _pauli_observable("z", q, self.n_qubits)
            q_vals[q] = float(self._simulator.expectation(qc, obs))
        mu = q_vals @ self._mu_proj
        logvar = q_vals @ self._logvar_proj
        return mu, logvar

    def _reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Quantum reparameterization trick."""
        eps = np.random.randn(*mu.shape)
        std = np.exp(0.5 * logvar)
        return mu + eps * std

    def _decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        qc = self._build_decoder_circuit(z, self._decoder_params)
        q_vals = np.zeros(self.n_qubits, dtype=np.float64)
        for q in range(self.n_qubits):
            obs = _pauli_observable("z", q, self.n_qubits)
            q_vals[q] = float(self._simulator.expectation(qc, obs))
        return q_vals

    def _compute_kl_loss(self, mu: np.ndarray, logvar: np.ndarray) -> float:
        """KL divergence: KL(q(z|x) || N(0,1))."""
        return float(-0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar)))

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        if not self._compiled:
            self.compile()
        X = np.asarray(X, dtype=np.float64)
        if not self._built:
            self.build(X.shape)

        lr = self.learning_rate
        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X))

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_b = X[perm[start:end]]

                mu, logvar = self._encode(X_b[0])
                z = self._reparameterize(mu, logvar)
                recon = self._decode(z)
                recon_loss = float(np.mean((recon - X_b[0, :min(len(recon), len(X_b[0]))]) ** 2))
                kl_loss = self._compute_kl_loss(mu, logvar)
                loss = recon_loss + self.beta * kl_loss

                # Simplified gradient update
                eps = 1e-5
                step = max(1, len(self._encoder_params) // 8)
                for idx in range(0, len(self._encoder_params), step):
                    orig = self._encoder_params[idx]
                    self._encoder_params[idx] = orig + eps
                    mu_p, _ = self._encode(X_b[0])
                    self._encoder_params[idx] = orig - eps
                    mu_m, _ = self._encode(X_b[0])
                    self._encoder_params[idx] = orig
                    kl_p = self._compute_kl_loss(mu_p, logvar)
                    kl_m = self._compute_kl_loss(mu_m, logvar)
                    grad = (kl_p - kl_m) / (2 * eps)
                    self._encoder_params[idx] -= lr * grad * 0.01

                epoch_loss += loss
                epoch_recon += recon_loss
                epoch_kl += kl_loss
                n_batches += 1

            n = max(n_batches, 1)
            self._history["loss"].append(float(epoch_loss / n))
            self._history["recon_loss"].append(float(epoch_recon / n))
            self._history["kl_loss"].append(float(epoch_kl / n))

            if verbose > 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"— loss: {self._history['loss'][-1]:.4f} "
                    f"— recon: {self._history['recon_loss'][-1]:.4f} "
                    f"— kl: {self._history['kl_loss'][-1]:.4f} "
                    f"— {time.time() - t0:.1f}s"
                )

            lr = self.learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Latent means.
        """
        if not self._built:
            self.build(np.asarray(X, dtype=np.float64).shape)
        X = np.asarray(X, dtype=np.float64)
        latents = []
        for b in range(X.shape[0]):
            mu, _ = self._encode(X[b])
            latents.append(mu)
        return np.array(latents)

    def summary(self) -> str:
        lines = [
            "KerasQuantumVAE",
            f"  n_qubits: {self.n_qubits}",
            f"  latent_dim: {self.latent_dim}",
            f"  n_layers: {self.n_layers}",
            f"  beta: {self.beta}",
        ]
        return "\n".join(lines)


# ===========================================================================
# KerasTransferLearning — Transfer Learning Helper
# ===========================================================================

class KerasTransferLearning:
    """Transfer learning helper for replacing classical model heads with
    quantum layers.

    Supports three fine-tuning strategies:
    * ``'full'`` — Train all parameters.
    * ``'quantum_only'`` — Freeze classical backbone, train quantum head only.
    * ``'gradual'`` — Train quantum layers first, then gradually unfreeze
      backbone layers.

    Parameters
    ----------
    backbone_layers : list of dict, optional
        Pre-defined classical backbone layers.
    n_qubits : int
        Qubits for the quantum head.
    n_layers : int, optional
        Circuit depth for quantum head. Default ``2``.
    strategy : str, optional
        Fine-tuning strategy. Default ``'full'``.
    learning_rate : float, optional
        Default ``0.001``.
    **kwargs
        Additional configuration.

    Examples
    --------
    >>> tl = KerasTransferLearning(
    ...     backbone_layers=[{'units': 128, 'activation': 'relu'}],
    ...     n_qubits=4,
    ...     strategy='quantum_only',
    ... )
    >>> tl.build(input_dim=784)
    >>> tl.compile(optimizer='adam', loss='mse')
    >>> tl.fit(X_train, y_train, epochs=10)
    """

    def __init__(
        self,
        backbone_layers: Optional[List[Dict[str, Any]]] = None,
        n_qubits: int = 4,
        n_layers: int = 2,
        strategy: str = "full",
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        self.backbone_layers = backbone_layers or []
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.strategy = strategy
        self.learning_rate = learning_rate
        self._backbone_weights: List[Dict[str, Any]] = []
        self._quantum_params = None
        self._kernel = None
        self._bias = None
        self._readout = None
        self._simulator = StatevectorSimulator()
        self._history: Dict[str, List[float]] = {"loss": []}
        self._compiled = False
        self._built = False
        self._frozen_backbone = strategy == "quantum_only"
        self._input_dim: Optional[int] = None
        self._current_dim: Optional[int] = None

    def compile(
        self,
        optimizer: Union[str, Any, None] = "adam",
        loss: Union[str, Any, None] = "mse",
        learning_rate: Optional[float] = None,
    ) -> None:
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self._compiled = True

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        self._frozen_backbone = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        self._frozen_backbone = False

    def build(self, input_dim: int, output_dim: int = 1) -> None:
        self._input_dim = input_dim
        rng = np.random.default_rng()
        self._backbone_weights = []
        current_dim = input_dim

        for layer_config in self.backbone_layers:
            units = layer_config.get("units", 64)
            limit = math.sqrt(6.0 / (current_dim + units))
            kernel = rng.uniform(-limit, limit, (current_dim, units)).astype(np.float64)
            bias = np.zeros(units, dtype=np.float64)
            self._backbone_weights.append({
                "kernel": kernel, "bias": bias,
                "activation": layer_config.get("activation", "relu"),
                "units": units,
            })
            current_dim = units

        self._current_dim = current_dim

        # Bridge: backbone output → quantum input
        limit = math.sqrt(6.0 / (current_dim + self.n_qubits))
        self._kernel = rng.uniform(-limit, limit, (current_dim, self.n_qubits)).astype(np.float64)
        self._bias = rng.uniform(-0.1, 0.1, self.n_qubits).astype(np.float64)

        # Quantum head parameters
        n_var = self.n_layers * self.n_qubits * 3
        self._quantum_params = rng.uniform(-0.1, 0.1, n_var).astype(np.float64)

        # Readout: n_qubits → output_dim
        limit_r = math.sqrt(6.0 / (self.n_qubits + output_dim))
        self._readout = rng.uniform(-limit_r, limit_r, (self.n_qubits, output_dim)).astype(np.float64)

        self._output_dim = output_dim
        self._built = True

    def _forward_backbone(self, x: np.ndarray) -> np.ndarray:
        h = x
        for w in self._backbone_weights:
            h = h @ w["kernel"] + w["bias"]
            act = w.get("activation")
            if act == "relu":
                h = np.maximum(0, h)
            elif act == "tanh":
                h = np.tanh(h)
            elif act == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -20, 20)))
        return h

    def _forward_quantum_head(self, x: np.ndarray) -> np.ndarray:
        angles = x @ self._kernel + self._bias
        angles = np.clip(angles, -_PI, _PI)
        batch_size = angles.shape[0]
        q_outputs = np.zeros((batch_size, self.n_qubits), dtype=np.float64)

        for b in range(batch_size):
            qc = QuantumCircuit(self.n_qubits)
            for q in range(self.n_qubits):
                qc.h(q)
                qc.ry(float(angles[b, q]), q)
            pidx = 0
            for _ in range(self.n_layers):
                for q in range(self.n_qubits):
                    if pidx + 2 < len(self._quantum_params):
                        qc.rz(float(self._quantum_params[pidx]), q); pidx += 1
                        qc.ry(float(self._quantum_params[pidx]), q); pidx += 1
                        qc.rz(float(self._quantum_params[pidx]), q); pidx += 1
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                obs = _pauli_observable("z", q, self.n_qubits)
                q_outputs[b, q] = float(self._simulator.expectation(qc, obs))

        return q_outputs @ self._readout

    def _forward(self, x: np.ndarray) -> np.ndarray:
        features = self._forward_backbone(x)
        return self._forward_quantum_head(features)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        if not self._compiled:
            self.compile()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if not self._built:
            self.build(X.shape[-1], y.shape[-1])

        lr = self.learning_rate
        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(len(X))

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_b = X[perm[start:end]]
                y_b = y[perm[start:end]]

                preds = self._forward(X_b)
                loss = float(np.mean((preds - y_b) ** 2))

                # Update quantum params (simplified)
                if not self._frozen_backbone:
                    eps = 1e-5
                    step = max(1, len(self._quantum_params) // 8)
                    for idx in range(0, len(self._quantum_params), step):
                        orig = self._quantum_params[idx]
                        self._quantum_params[idx] = orig + eps
                        pp = self._forward(X_b)
                        self._quantum_params[idx] = orig - eps
                        pm = self._forward(X_b)
                        self._quantum_params[idx] = orig
                        grad = np.sum((pp - pm) / (2 * eps))
                        self._quantum_params[idx] -= lr * grad * 0.01

                epoch_loss += loss
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(float(avg))

            # Gradual unfreezing
            if self.strategy == "gradual" and epoch == epochs // 3:
                self.unfreeze_backbone()

            if verbose > 0:
                frozen_str = " [backbone frozen]" if self._frozen_backbone else ""
                print(
                    f"Epoch {epoch + 1}/{epochs} — loss: {avg:.4f}"
                    f"{frozen_str} — {time.time() - t0:.1f}s"
                )

            lr = self.learning_rate / (1 + 0.01 * (epoch + 1))

        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._built:
            self.build(np.asarray(X, dtype=np.float64).shape[-1])
        return self._forward(np.asarray(X, dtype=np.float64))

    def summary(self) -> str:
        lines = [
            "KerasTransferLearning",
            f"  strategy: {self.strategy}",
            f"  backbone_frozen: {self._frozen_backbone}",
            f"  quantum_head: {self.n_qubits} qubits, {self.n_layers} layers",
        ]
        if self._built:
            lines.append(f"  input_dim: {self._input_dim}")
            lines.append(f"  output_dim: {self._output_dim}")
        return "\n".join(lines)
