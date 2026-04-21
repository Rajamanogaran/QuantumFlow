"""
Quantum Data Preprocessing
============================

Provides quantum-aware data preprocessing utilities for preparing classical
data for quantum layers and models. Includes encoders, augmenters, normalizers,
and feature scalers.

Classes
-------
* :class:`QuantumDataEncoder` — Encode classical data for quantum layers.
* :class:`QuantumDataAugmenter` — Augment training data with quantum-inspired
  transformations.
* :class:`QuantumNormalizer` — Keras preprocessing layer for quantum normalization.
* :class:`QuantumFeatureScaler` — Scale features to quantum-appropriate ranges.

Examples
--------
>>> from quantumflow.keras.preprocessing import QuantumDataEncoder
>>> encoder = QuantumDataEncoder(n_qubits=4, encoding='angle')
>>> encoded = encoder.angle_encode(X_train)

>>> scaler = QuantumFeatureScaler()
>>> X_scaled = scaler.fit_transform(X_train)
>>> X_test_scaled = scaler.transform(X_test)
"""

from __future__ import annotations

import math
from typing import (
    Any,
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
except ImportError:
    keras = None  # type: ignore[assignment]
    ops = None  # type: ignore[assignment]
    _KerasLayer = None  # type: ignore[assignment,misc]

if _KerasLayer is None:
    class _StubLayer:
        def __init__(self, **kwargs: Any) -> None:
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

__all__ = [
    "QuantumDataEncoder",
    "QuantumDataAugmenter",
    "QuantumNormalizer",
    "QuantumFeatureScaler",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_TOLERANCE = 1e-10


# ===========================================================================
# QuantumDataEncoder — Encode Classical Data for Quantum Layers
# ===========================================================================

class QuantumDataEncoder:
    """Encode classical data for quantum layers.

    Provides multiple encoding strategies for transforming classical data
    into quantum-compatible representations, including normalization, PCA
    reduction, amplitude encoding, and angle encoding.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the target quantum circuit.
    encoding : str, optional
        Default encoding strategy: ``'angle'``, ``'amplitude'``,
        ``'basis'``, ``'iqp'``, ``'dense_angle'``. Default ``'angle'``.
    normalize : bool, optional
        Whether to automatically normalize data. Default ``True``.
    clip_values : bool, optional
        Clip values to valid quantum parameter ranges. Default ``True``.

    Examples
    --------
    >>> encoder = QuantumDataEncoder(n_qubits=4, encoding='angle')
    >>> X_encoded = encoder.encode(X_train)
    >>> X_test_encoded = encoder.encode(X_test)

    >>> # PCA reduction followed by angle encoding
    >>> encoder = QuantumDataEncoder(n_qubits=4)
    >>> X_reduced = encoder.pca_reduce(X_train, n_components=4)
    >>> X_encoded = encoder.angle_encode(X_reduced)
    """

    def __init__(
        self,
        n_qubits: int,
        encoding: str = "angle",
        normalize: bool = True,
        clip_values: bool = True,
    ) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        valid_encodings = {"angle", "amplitude", "basis", "iqp", "dense_angle"}
        if encoding not in valid_encodings:
            raise ValueError(
                f"Unknown encoding '{encoding}'. Choose from {sorted(valid_encodings)}"
            )
        self.n_qubits = n_qubits
        self.encoding = encoding
        self._should_normalize = normalize
        self.clip_values = clip_values

        # Fitted statistics
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._fitted: bool = False

    @property
    def input_dim(self) -> int:
        """int: Expected input dimensionality for the current encoding."""
        if self.encoding == "amplitude":
            return 1 << self.n_qubits
        return self.n_qubits

    def fit(self, X: np.ndarray) -> QuantumDataEncoder:
        """Fit the encoder statistics on training data.

        Computes mean, std, min, max for normalization.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape ``(n_samples, n_features)``.

        Returns
        -------
        QuantumDataEncoder
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + _TOLERANCE
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)
        self._fitted = True
        return self

    def normalize(
        self,
        X: np.ndarray,
        method: str = "standard",
        range_val: Tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Normalize data to a specified range.

        Parameters
        ----------
        X : numpy.ndarray
            Data to normalize, shape ``(n_samples, n_features)``.
        method : str, optional
            Normalization method:
            * ``'standard'`` — zero mean, unit variance (z-score).
            * ``'minmax'`` — scale to ``[range_val[0], range_val[1]]``.
            * ``'quantum'`` — scale to ``[-π, π]`` for rotation angles.
            * ``'l2'`` — L2 normalization per sample.
        range_val : tuple of float, optional
            Target range for minmax normalization. Default ``(0, 1)``.

        Returns
        -------
        numpy.ndarray
            Normalized data.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if method == "standard":
            if not self._fitted:
                self.fit(X)
            return (X - self._mean) / self._std

        elif method == "minmax":
            if not self._fitted:
                self.fit(X)
            data_range = self._max - self._min + _TOLERANCE
            X_norm = (X - self._min) / data_range
            X_norm = X_norm * (range_val[1] - range_val[0]) + range_val[0]
            return X_norm

        elif method == "quantum":
            """Scale to [-π, π] suitable for quantum rotation angles."""
            if not self._fitted:
                self.fit(X)
            data_range = self._max - self._min + _TOLERANCE
            X_norm = (X - self._min) / data_range  # [0, 1]
            X_norm = X_norm * 2 * _PI - _PI  # [-π, π]
            return X_norm

        elif method == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, _TOLERANCE)
            return X / norms

        else:
            raise ValueError(
                f"Unknown normalization method '{method}'. "
                f"Choose from 'standard', 'minmax', 'quantum', 'l2'"
            )

    def pca_reduce(
        self,
        X: np.ndarray,
        n_components: int,
        method: str = "simple",
    ) -> np.ndarray:
        """PCA dimension reduction.

        Reduces the feature dimensionality to ``n_components`` using
        Principal Component Analysis.

        Parameters
        ----------
        X : numpy.ndarray
            Data of shape ``(n_samples, n_features)``.
        n_components : int
            Target number of components.
        method : str, optional
            ``'simple'`` (SVD-based) or ``'power'`` (power iteration).
            Default ``'simple'``.

        Returns
        -------
        numpy.ndarray
            Reduced data of shape ``(n_samples, n_components)``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_components = min(n_components, n_features, n_samples)

        # Center the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        if method == "simple":
            # SVD-based PCA
            if n_samples > n_features:
                # Use covariance matrix (faster for n_features << n_samples)
                cov = (X_centered.T @ X_centered) / (n_samples - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Sort by eigenvalue descending
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx[:n_components]]
            else:
                # Direct SVD
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                eigenvectors = Vt[:n_components].T

            # Store for transform
            self._pca_components = eigenvectors
            self._pca_mean = mean

        elif method == "power":
            # Power iteration for top eigenvectors
            eigenvectors = np.zeros((n_features, n_components), dtype=np.float64)
            for k in range(n_components):
                v = np.random.randn(n_features)
                v = v / np.linalg.norm(v)
                for _ in range(100):
                    v_new = X_centered.T @ (X_centered @ v)
                    v_norm = np.linalg.norm(v_new)
                    if v_norm < _TOLERANCE:
                        break
                    v = v_new / v_norm
                eigenvectors[:, k] = v
            self._pca_components = eigenvectors
            self._pca_mean = mean

        else:
            raise ValueError(f"Unknown PCA method '{method}'")

        return (X_centered @ self._pca_components)

    def pca_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA components.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if self._pca_components is None:
            raise RuntimeError("PCA has not been fitted. Call pca_reduce() first.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self._pca_mean) @ self._pca_components

    def amplitude_encode(
        self,
        X: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode classical data as quantum amplitudes.

        Normalizes data to form a valid quantum state vector and returns
        the amplitude values. Input dimension must be a power of 2, or
        will be padded.

        Parameters
        ----------
        X : numpy.ndarray
            Data of shape ``(n_samples, n_features)``.
        normalize : bool, optional
            L2-normalize each sample. Default ``True``.

        Returns
        -------
        numpy.ndarray
            Amplitude-encoded data of shape ``(n_samples, 2**n_qubits)``.
            Each row is a valid quantum state (L2-normalized).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        target_dim = 1 << self.n_qubits
        n_samples = X.shape[0]

        # Pad or truncate to target dimension
        encoded = np.zeros((n_samples, target_dim), dtype=np.float64)
        for i in range(n_samples):
            data = X[i].flatten()
            if len(data) > target_dim:
                encoded[i] = data[:target_dim].real
            else:
                encoded[i, :len(data)] = data.real

        if normalize:
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            norms = np.maximum(norms, _TOLERANCE)
            encoded = encoded / norms

        return encoded

    def angle_encode(
        self,
        X: np.ndarray,
        method: str = "rotation",
    ) -> np.ndarray:
        """Encode classical data as rotation angles.

        Maps each feature to a rotation angle suitable for quantum
        rotation gates (RZ, RY, RX).

        Parameters
        ----------
        X : numpy.ndarray
            Data of shape ``(n_samples, n_features)``.
        method : str, optional
            Angle mapping method:
            * ``'rotation'`` — ``x → x * π`` (maps [-1,1] → [-π, π]).
            * ``'arcsin'`` — ``x → arcsin(clip(x))`` (maps [-1,1] → [-π/2, π/2]).
            * ``'quantum_scale'`` — ``x → x * 2π`` (full circle).

        Returns
        -------
        numpy.ndarray
            Angle-encoded data of shape ``(n_samples, n_qubits)``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Map to n_qubits features
        if X.shape[-1] != self.n_qubits:
            X = self._map_features(X, self.n_qubits)

        if method == "rotation":
            angles = X * _PI
        elif method == "arcsin":
            angles = np.arcsin(np.clip(X, -1.0, 1.0))
        elif method == "quantum_scale":
            angles = X * 2 * _PI
        else:
            raise ValueError(f"Unknown angle encoding method '{method}'")

        if self.clip_values:
            angles = np.clip(angles, -_PI, _PI)

        return angles

    def _map_features(
        self,
        X: np.ndarray,
        target_dim: int,
    ) -> np.ndarray:
        """Map features from input_dim to target_dim.

        Parameters
        ----------
        X : numpy.ndarray
            Shape ``(n_samples, n_features)``.
        target_dim : int

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, target_dim)``.
        """
        n_features = X.shape[-1]
        if n_features == target_dim:
            return X
        elif n_features > target_dim:
            # Average pooling
            chunk_size = n_features // target_dim
            result = np.zeros((X.shape[0], target_dim), dtype=np.float64)
            for q in range(target_dim):
                start = q * chunk_size
                end = min(start + chunk_size, n_features)
                result[:, q] = np.mean(X[:, start:end], axis=1)
            return result
        else:
            # Pad with zeros
            result = np.zeros((X.shape[0], target_dim), dtype=np.float64)
            result[:, :n_features] = X
            return result

    def encode(
        self,
        X: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """Full encoding pipeline: normalize → encode.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        fit : bool, optional
            Fit statistics on this data. Default ``False``.

        Returns
        -------
        numpy.ndarray
            Encoded data.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit or not self._fitted:
            self.fit(X)

        # Normalize
        if self._should_normalize:
            X = self.normalize(X, method="quantum")

        # Encode
        if self.encoding == "angle":
            return self.angle_encode(X)
        elif self.encoding == "amplitude":
            return self.amplitude_encode(X)
        elif self.encoding == "basis":
            return self._basis_encode(X)
        elif self.encoding == "iqp":
            return self.angle_encode(X, method="quantum_scale")
        elif self.encoding == "dense_angle":
            return self.angle_encode(X, method="rotation")
        return X

    def _basis_encode(self, X: np.ndarray) -> np.ndarray:
        """Basis encoding: map to binary {0, 1} representation.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = self._map_features(X, self.n_qubits)
        return (X > 0.5).astype(np.float64)

    def get_encoding_circuit(
        self,
        data: np.ndarray,
    ) -> QuantumCircuit:
        """Build a quantum circuit that encodes the given data.

        Parameters
        ----------
        data : numpy.ndarray
            Single data sample.

        Returns
        -------
        QuantumCircuit
        """
        data = np.asarray(data, dtype=np.float64).flatten()
        data = self._map_features(data.reshape(1, -1), self.n_qubits).flatten()

        qc = QuantumCircuit(self.n_qubits)

        if self.encoding == "angle":
            for q in range(self.n_qubits):
                qc.h(q)
                qc.ry(float(data[q]), q)
        elif self.encoding == "amplitude":
            # Use recursive rotation-based amplitude encoding
            self._amplitude_encode_circuit(qc, data, 0)
        elif self.encoding == "basis":
            for q in range(self.n_qubits):
                if float(data[q]) > 0.5:
                    qc.x(q)
        elif self.encoding == "iqp":
            for q in range(self.n_qubits):
                qc.h(q)
                qc.rz(float(data[q]), q)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                qc.rz(_PI / 4.0, q)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            for q in range(self.n_qubits):
                qc.rz(float(data[q]), q)
        elif self.encoding == "dense_angle":
            n_reuploads = max(1, min(3, self.n_qubits // 2))
            for _ in range(n_reuploads):
                for q in range(self.n_qubits):
                    qc.rz(float(data[q]), q)
                    qc.ry(float(data[q]), q)
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)

        return qc

    def _amplitude_encode_circuit(
        self,
        circuit: QuantumCircuit,
        amplitudes: np.ndarray,
        start_qubit: int,
    ) -> None:
        """Recursively encode amplitudes using rotation gates.

        Parameters
        ----------
        circuit : QuantumCircuit
        amplitudes : numpy.ndarray
        start_qubit : int
        """
        n = len(amplitudes)
        if n <= 1:
            return
        if n == 2:
            prob_0 = float(np.abs(amplitudes[0]) ** 2)
            theta = 2.0 * math.acos(np.sqrt(np.clip(prob_0, 0.0, 1.0)))
            circuit.ry(theta, start_qubit)
            if np.abs(amplitudes[0]) > 1e-12:
                phase_0 = np.angle(amplitudes[0])
                phase_1 = np.angle(amplitudes[1])
                relative_phase = phase_1 - phase_0
                if abs(relative_phase) > 1e-12:
                    circuit.rz(relative_phase, start_qubit)
            return

        mid = n // 2
        left = amplitudes[:mid]
        right = amplitudes[mid:]

        prob_0 = float(np.sum(np.abs(left) ** 2))
        prob_0 = np.clip(prob_0, 0.0, 1.0)
        theta = 2.0 * math.acos(math.sqrt(float(prob_0)))
        circuit.ry(theta, start_qubit)

        phase_left = np.angle(left[0]) if np.abs(left[0]) > 1e-12 else 0.0
        phase_right = np.angle(right[0]) if np.abs(right[0]) > 1e-12 else 0.0
        relative_phase = phase_right - phase_left
        if abs(relative_phase) > 1e-12:
            circuit.rz(relative_phase, start_qubit)

        self._amplitude_encode_circuit(circuit, left, start_qubit + 1)
        circuit.x(start_qubit)
        self._amplitude_encode_circuit(circuit, right, start_qubit + 1)
        circuit.x(start_qubit)

    def get_config(self) -> Dict[str, Any]:
        """Return encoder configuration."""
        return {
            "n_qubits": self.n_qubits,
            "encoding": self.encoding,
            "normalize": self._should_normalize,
            "clip_values": self.clip_values,
            "fitted": self._fitted,
            "input_dim": self.input_dim,
        }

    def __repr__(self) -> str:
        return (
            f"QuantumDataEncoder("
            f"n_qubits={self.n_qubits}, "
            f"encoding={self.encoding!r}, "
            f"input_dim={self.input_dim}, "
            f"fitted={self._fitted})"
        )


# ===========================================================================
# QuantumDataAugmenter — Augment Training Data
# ===========================================================================

class QuantumDataAugmenter:
    """Augment training data with quantum-inspired transformations.

    Applies random rotations, flips, noise injection, and other
    transformations to increase the effective size of the training set.

    Parameters
    ----------
    rotation_range : float, optional
        Maximum rotation angle in radians. Default ``0.3``.
    flip_horizontal : bool, optional
            Randomly flip features horizontally. Default ``True``.
    flip_vertical : bool, optional
        Randomly reverse feature order. Default ``True``.
    noise_std : float, optional
        Standard deviation of Gaussian noise. Default ``0.05``.
    noise_type : str, optional
        Noise distribution: ``'gaussian'``, ``'uniform'``, ``'quantum_depolarizing'``.
        Default ``'gaussian'``.
    scale_range : tuple of float, optional
        Random scaling range. Default ``(0.9, 1.1)``.
    shift_range : float, optional
        Maximum random shift. Default ``0.1``.
    seed : int or None, optional
        Random seed for reproducibility. Default ``None``.

    Examples
    --------
    >>> augmenter = QuantumDataAugmenter(rotation_range=0.2, noise_std=0.03)
    >>> X_augmented = augmenter.augment(X_train, augment_factor=2)
    """

    def __init__(
        self,
        rotation_range: float = 0.3,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
        noise_std: float = 0.05,
        noise_type: str = "gaussian",
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.scale_range = scale_range
        self.shift_range = shift_range
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def _apply_rotation(
        self,
        x: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Apply a random rotation to the feature vector.

        Rotation is applied as a 2D rotation in random feature planes.

        Parameters
        ----------
        x : numpy.ndarray
            Shape ``(n_features,)``.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        numpy.ndarray
        """
        if len(x) < 2:
            return x
        result = x.copy()
        # Rotate in random 2D planes
        n_planes = max(1, len(x) // 4)
        for _ in range(n_planes):
            i = self._rng.integers(0, len(x) - 1)
            j = i + 1
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            new_i = cos_a * result[i] - sin_a * result[j]
            new_j = sin_a * result[i] + cos_a * result[j]
            result[i] = new_i
            result[j] = new_j
        return result

    def _apply_noise(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply noise to the feature vector.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if self.noise_type == "gaussian":
            noise = self._rng.normal(0, self.noise_std, size=x.shape)
        elif self.noise_type == "uniform":
            noise = self._rng.uniform(
                -self.noise_std, self.noise_std, size=x.shape
            )
        elif self.noise_type == "quantum_depolarizing":
            # Depolarizing-like noise: mix with uniform random values
            noise = self._rng.normal(0, self.noise_std, size=x.shape)
            mask = self._rng.binomial(1, 0.1, size=x.shape).astype(np.float64)
            noise = mask * self._rng.uniform(-1, 1, size=x.shape)
        else:
            noise = np.zeros_like(x)
        return x + noise

    def _apply_flip(
        self,
        x: np.ndarray,
        horizontal: bool = True,
    ) -> np.ndarray:
        """Flip features.

        Parameters
        ----------
        x : numpy.ndarray
        horizontal : bool

        Returns
        -------
        numpy.ndarray
        """
        if horizontal:
            return x[::-1].copy()
        return x.copy()

    def _apply_scale(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply random scaling.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        scale = self._rng.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale

    def _apply_shift(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply random shift.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        shift = self._rng.uniform(-self.shift_range, self.shift_range)
        return x + shift

    def _apply_phase_shift(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply quantum-inspired global phase shift.

        Simulates the effect of a global phase gate on amplitudes
        by applying a small complex rotation to features.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        phase = self._rng.uniform(0, 2 * _PI)
        # Mix features via global rotation
        return x * math.cos(phase) + math.sin(phase) * 0.01 * self._rng.normal(0, 1, size=x.shape)

    def augment_single(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply a random augmentation to a single sample.

        Parameters
        ----------
        x : numpy.ndarray
            Shape ``(n_features,)``.

        Returns
        -------
        numpy.ndarray
            Augmented sample.
        """
        x = x.copy()

        # Random rotation
        angle = self._rng.uniform(-self.rotation_range, self.rotation_range)
        x = self._apply_rotation(x, angle)

        # Random flips
        if self.flip_horizontal and self._rng.random() > 0.5:
            x = self._apply_flip(x, horizontal=True)
        if self.flip_vertical and self._rng.random() > 0.5:
            # Vertical flip: random feature subset sign flip
            mask = self._rng.binomial(1, 0.5, size=len(x)).astype(np.float64)
            sign_flip = 2 * mask - 1
            x = x * sign_flip

        # Noise injection
        if self.noise_std > 0:
            x = self._apply_noise(x)

        # Scale
        x = self._apply_scale(x)

        # Shift
        x = self._apply_shift(x)

        # Phase shift (quantum-inspired)
        if self._rng.random() > 0.5:
            x = self._apply_phase_shift(x)

        return x

    def augment(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augment_factor: int = 2,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Augment a dataset by generating transformed copies.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape ``(n_samples, n_features)``.
        y : numpy.ndarray, optional
            Labels. If provided, returns augmented (X, y) pair.
        augment_factor : int, optional
            Number of augmented copies per original sample. Default ``2``.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            Augmented data (and labels if y was provided).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        augmented = [X]
        for _ in range(augment_factor):
            aug_batch = np.array([
                self.augment_single(X[i]) for i in range(len(X))
            ])
            augmented.append(aug_batch)

        X_aug = np.concatenate(augmented, axis=0)

        if y is not None:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y_aug = np.concatenate([y] * (augment_factor + 1), axis=0)
            return X_aug, y_aug

        return X_aug

    def __repr__(self) -> str:
        return (
            f"QuantumDataAugmenter("
            f"rotation_range={self.rotation_range}, "
            f"noise_std={self.noise_std}, "
            f"noise_type={self.noise_type!r})"
        )


# ===========================================================================
# QuantumNormalizer — Keras Preprocessing Layer
# ===========================================================================

class QuantumNormalizer(Layer):
    """Keras preprocessing layer that normalizes inputs for quantum feature maps.

    Projects features onto the Bloch sphere surface using an arcsin mapping,
    then optionally applies standard normalization. This ensures that inputs
    are in a valid range for quantum rotation gates.

    Parameters
    ----------
    method : str, optional
        Normalization method:
        * ``'bloch'`` — Arcsin projection to Bloch sphere surface.
        * ``'quantum_scale'`` — Scale to [-π, π].
        * ``'standard'`` — Z-score normalization.
        * ``'minmax'`` — Scale to [0, 1].
        Default ``'bloch'``.
    epsilon : float, optional
        Small constant for numerical stability. Default ``1e-7``.
    clip : bool, optional
        Clip outputs to valid range. Default ``True``.
    **kwargs
        Additional arguments for ``keras.layers.Layer``.

    Examples
    --------
    >>> layer = QuantumNormalizer(method='bloch')
    >>> x = np.random.randn(4, 8).astype("float32")
    >>> y = layer(x, training=True)
    """

    def __init__(
        self,
        method: str = "bloch",
        epsilon: float = 1e-7,
        clip: bool = True,
        **kwargs: Any,
    ) -> None:
        _check_keras_available()
        super().__init__(**kwargs)
        self.method = method
        self.epsilon = epsilon
        self.clip = clip
        self._mean = None
        self._var = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        param_shape = (input_shape[-1],)
        self._mean = self.add_weight(
            name="mean", shape=param_shape,
            initializer="zeros", trainable=False,
        )
        self._var = self.add_weight(
            name="var", shape=param_shape,
            initializer="ones", trainable=False,
        )
        self.built = True

    def call(self, inputs: Any, training: bool = False) -> Any:
        x = ops.convert_to_tensor(inputs, dtype="float32")

        if training:
            # Update statistics
            batch_mean = ops.mean(x, axis=0)
            batch_var = ops.var(x, axis=0)
            new_mean = ops.convert_to_tensor(
                0.9 * ops.convert_to_numpy(self._mean) +
                0.1 * ops.convert_to_numpy(batch_mean),
                dtype="float32",
            )
            new_var = ops.convert_to_tensor(
                0.9 * ops.convert_to_numpy(self._var) +
                0.1 * ops.convert_to_numpy(batch_var),
                dtype="float32",
            )
            self._mean.assign(new_mean)
            self._var.assign(new_var)

        if self.method == "bloch":
            # Standardize
            x_norm = (x - self._mean) / ops.sqrt(self._var + self.epsilon)
            # Clip to [-1, 1] for arcsin
            x_clipped = ops.clip(x_norm, -1.0, 1.0)
            # Arcsin projection to Bloch sphere
            output = ops.arcsin(x_clipped) / (_PI / 2.0)

        elif self.method == "quantum_scale":
            x_norm = (x - self._mean) / ops.sqrt(self._var + self.epsilon)
            output = x_norm * _PI

        elif self.method == "standard":
            output = (x - self._mean) / ops.sqrt(self._var + self.epsilon)

        elif self.method == "minmax":
            output = ops.clip((x - self._mean) / (ops.sqrt(self._var) + self.epsilon), 0.0, 1.0)

        else:
            output = x

        if self.clip:
            output = ops.clip(output, -_PI, _PI)

        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "method": self.method,
            "epsilon": self.epsilon,
            "clip": self.clip,
        })
        return config


# ===========================================================================
# QuantumFeatureScaler — Scale Features for Quantum Layers
# ===========================================================================

class QuantumFeatureScaler:
    """Scale features to appropriate ranges for quantum layers.

    Provides fit/transform interface similar to scikit-learn scalers,
    with quantum-specific scaling modes.

    Parameters
    ----------
    method : str, optional
        Scaling method:
        * ``'quantum'`` — Scale to [-π, π] for rotation gates.
        * ``'amplitude'`` — L2-normalize for amplitude encoding.
        * ``'bloch'`` — Scale to [-1, 1] for Bloch sphere projection.
        * ``'standard'`` — Z-score normalization.
        * ``'minmax'`` — Scale to [0, 1].
        Default ``'quantum'``.
    clip : bool, optional
        Clip scaled values. Default ``True``.
    epsilon : float, optional
        Numerical stability constant. Default ``1e-10``.

    Examples
    --------
    >>> scaler = QuantumFeatureScaler(method='quantum')
    >>> X_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    >>> X_original = scaler.inverse_transform(X_test_scaled)
    """

    def __init__(
        self,
        method: str = "quantum",
        clip: bool = True,
        epsilon: float = 1e-10,
    ) -> None:
        self.method = method
        self.clip = clip
        self.epsilon = epsilon
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> QuantumFeatureScaler:
        """Compute scaling statistics from training data.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape ``(n_samples, n_features)``.

        Returns
        -------
        QuantumFeatureScaler
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + self.epsilon
        self._min = np.min(X, axis=0)
        self._max = np.max(X, axis=0)
        self._norms = np.linalg.norm(X, axis=1) + self.epsilon
        self._fitted = True
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return self.fit(X).transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale data using fitted statistics.

        Parameters
        ----------
        X : numpy.ndarray
            Data to scale.

        Returns
        -------
        numpy.ndarray
            Scaled data.
        """
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "quantum":
            # Scale to [-π, π]
            data_range = self._max - self._min + self.epsilon
            X_scaled = (X - self._min) / data_range  # [0, 1]
            X_scaled = X_scaled * 2 * _PI - _PI  # [-π, π]

        elif self.method == "amplitude":
            # L2 normalize per sample
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, self.epsilon)
            X_scaled = X / norms

        elif self.method == "bloch":
            # Scale to [-1, 1]
            data_range = self._max - self._min + self.epsilon
            X_scaled = (X - self._min) / data_range
            X_scaled = X_scaled * 2 - 1  # [-1, 1]

        elif self.method == "standard":
            X_scaled = (X - self._mean) / self._std

        elif self.method == "minmax":
            data_range = self._max - self._min + self.epsilon
            X_scaled = (X - self._min) / data_range

        else:
            raise ValueError(f"Unknown scaling method '{self.method}'")

        if self.clip:
            if self.method == "quantum":
                X_scaled = np.clip(X_scaled, -_PI, _PI)
            elif self.method == "bloch":
                X_scaled = np.clip(X_scaled, -1.0, 1.0)
            elif self.method == "minmax":
                X_scaled = np.clip(X_scaled, 0.0, 1.0)

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation.

        Parameters
        ----------
        X : numpy.ndarray
            Scaled data.

        Returns
        -------
        numpy.ndarray
            Original-scale data.
        """
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "quantum":
            # [-π, π] → original
            X_inv = (X + _PI) / (2 * _PI)
            data_range = self._max - self._min + self.epsilon
            X_inv = X_inv * data_range + self._min

        elif self.method == "amplitude":
            # Cannot fully invert L2 normalization without original norms
            X_inv = X * self._norms.reshape(-1, 1) / len(self._norms)
            X_inv = X_inv * np.mean(self._norms)

        elif self.method == "bloch":
            X_inv = (X + 1) / 2  # [-1, 1] → [0, 1]
            data_range = self._max - self._min + self.epsilon
            X_inv = X_inv * data_range + self._min

        elif self.method == "standard":
            X_inv = X * self._std + self._mean

        elif self.method == "minmax":
            data_range = self._max - self._min + self.epsilon
            X_inv = X * data_range + self._min

        else:
            X_inv = X.copy()

        return X_inv

    def get_params(self) -> Dict[str, Any]:
        """Return scaler parameters.

        Returns
        -------
        dict
        """
        return {
            "method": self.method,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "fitted": self._fitted,
            "n_features": int(self._mean.shape[0]) if self._mean is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"QuantumFeatureScaler("
            f"method={self.method!r}, "
            f"clip={self.clip}, "
            f"fitted={self._fitted})"
        )
