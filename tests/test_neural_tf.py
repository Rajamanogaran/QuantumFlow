"""
Neural / TensorFlow / Keras integration tests
==============================================

Exercised only when TensorFlow is installed (the full test suite must
stay green without TF). These lock in the lazy-build fixes (layers are
usable without an explicit ``build()``), Keras 3 compatibility
(``keras.activations.get``), and the rotation-preset parsing fix in
:class:`~quantumflow.neural.qnn_layer.QuantumNNLayer`.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from quantumflow.keras.layers import KerasQDense  # noqa: E402
from quantumflow.keras.models import (  # noqa: E402
    KerasQNN,
    KerasQuantumClassifier,
    KerasQuantumRegressor,
)
from quantumflow.keras.preprocessing import QuantumDataEncoder  # noqa: E402
from quantumflow.neural.qnn_layer import (  # noqa: E402
    QuantumNNLayer,
    VariationalLayer,
)
from quantumflow.tensorflow.layers import (  # noqa: E402
    QDenseLayer,
    QMeasurementLayer,
)


def _tiny_data(n=12, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Plain-TF quantum layers: lazy auto-build
# ---------------------------------------------------------------------------


class TestTensorFlowLayers:
    def test_qdense_layer_forward_shape(self):
        layer = QDenseLayer(units=4, n_qubits=3, n_layers=1)
        x = tf.random.normal((8, 3))
        y = layer(x)
        assert y.shape == (8, 4)
        assert not np.any(np.isnan(y.numpy()))

    def test_qdense_layer_gradient_flows(self):
        """Regression: 'Layer not built' RuntimeError on first call."""
        layer = QDenseLayer(units=1, n_qubits=2, n_layers=1)
        x = tf.constant(np.random.default_rng(1).normal(size=(4, 2)), tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = layer(x)
            loss = tf.reduce_sum(y**2)
        grads = tape.gradient(loss, x)
        assert grads is not None

    def test_qmeasurement_layer_expectation(self):
        layer = QMeasurementLayer(n_qubits=2, observable="z", strategy="expectation")
        x = tf.constant(
            np.random.default_rng(2).normal(size=(6, 2)), dtype=tf.float32
        )
        y = layer(x)
        assert y.shape == (6, 2)
        assert not np.any(np.isnan(y.numpy()))


# ---------------------------------------------------------------------------
# Keras 3 layers inside a real model
# ---------------------------------------------------------------------------


class TestKerasLayers:
    def test_keras_qdense_in_sequential_trains(self):
        model = keras.Sequential(
            [keras.Input(shape=(2,)), KerasQDense(units=1, n_qubits=2, n_layers=1)]
        )
        model.compile(optimizer="adam", loss="mse")
        X, y = _tiny_data()
        hist = model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        assert "loss" in hist.history
        preds = model.predict(X, verbose=0)
        assert preds.shape == (12, 1)
        assert not np.any(np.isnan(preds))

    def test_keras_qdense_gradient_through_quantum_layer(self):
        """End-to-end: gradients must flow through the quantum circuit."""
        model = keras.Sequential(
            [keras.Input(shape=(2,)), KerasQDense(units=1, n_qubits=2, n_layers=1)]
        )
        model.compile(optimizer="adam", loss="mse")
        X, y = _tiny_data(n=4)
        w_before = model.get_weights()[0].copy()
        model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        w_after = model.get_weights()[0]
        assert not np.allclose(w_before, w_after)


# ---------------------------------------------------------------------------
# High-level models
# ---------------------------------------------------------------------------


class TestKerasModels:
    def test_keras_qnn_build_fit_predict(self):
        qnn = KerasQNN(input_dim=2)
        qnn.add_quantum_layer(n_qubits=2, n_layers=1)
        qnn.add_classical_layer(units=1, activation=None)
        qnn.compile(optimizer="adam", loss="mse")
        X, y = _tiny_data()
        hist = qnn.fit(X, y, epochs=1, batch_size=4, verbose=0)
        assert "loss" in hist
        preds = qnn.predict(X)
        assert preds.shape[0] == 12

    def test_keras_quantum_classifier(self):
        clf = KerasQuantumClassifier(n_qubits=2, n_classes=2, n_layers=1)
        clf.compile()
        X, y = _tiny_data(n=8)
        clf.fit(X, y, epochs=1, batch_size=4, verbose=0)
        preds = clf.predict_classes(X)
        assert set(np.unique(preds)) <= {0, 1}

    def test_keras_quantum_regressor(self):
        reg = KerasQuantumRegressor(n_qubits=2, n_layers=1)
        reg.compile()
        X, y = _tiny_data(n=8)
        reg.fit(X, y, epochs=1, batch_size=4, verbose=0)
        preds = reg.predict(X)
        assert preds.shape[0] == 8


# ---------------------------------------------------------------------------
# Data encoding
# ---------------------------------------------------------------------------


class TestQuantumDataEncoder:
    def test_angle_encode_shape_and_range(self):
        enc = QuantumDataEncoder(n_qubits=3, encoding="angle")
        X = np.random.default_rng(3).normal(size=(10, 3))
        out = enc.encode(X, fit=True)
        assert out.shape == (10, 3)
        assert np.all(np.abs(out) <= np.pi + 1e-9)

    def test_fit_then_encode_uses_fitted_stats(self):
        enc = QuantumDataEncoder(n_qubits=2)
        Xtr = np.random.default_rng(4).normal(size=(20, 2))
        enc.fit(Xtr)
        out = enc.encode(Xtr)
        assert out.shape == (20, 2)

    def test_invalid_encoding_rejected(self):
        with pytest.raises(ValueError):
            QuantumDataEncoder(n_qubits=2, encoding="nonexistent")

    def test_amplitude_encode_normalized(self):
        enc = QuantumDataEncoder(n_qubits=2, encoding="amplitude")
        X = np.random.default_rng(5).normal(size=(4, 4))
        out = enc.encode(X, fit=True)
        assert out.shape == (4, 4)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# QuantumNNLayer preset parsing
# ---------------------------------------------------------------------------


class TestQuantumNNLayer:
    def test_invalid_rotation_preset_raises(self):
        with pytest.raises(ValueError):
            QuantumNNLayer(n_qubits=2, n_layers=1, rotation_gates="rxyzq")

    def test_valid_preset_parsed_correctly(self):
        """Regression: presets were exploded into single characters."""
        assert VariationalLayer(n_qubits=2, rotation_gates="rycz").rotation_gates == (
            "ry",
            "rz",
        )
        assert VariationalLayer(n_qubits=2, rotation_gates="rxyz").rotation_gates == (
            "rx",
            "ry",
            "rz",
        )

    def test_qnn_layer_builds_with_preset(self):
        layer = QuantumNNLayer(n_qubits=2, n_layers=1, rotation_gates="rycz")
        assert layer.get_circuit([0.1, -0.2]) is not None

    def test_forward_output_finite(self):
        layer = QuantumNNLayer(n_qubits=2, n_layers=1, rotation_gates="ry")
        out = layer.forward([0.1, -0.2])
        assert np.all(np.isfinite(out))

