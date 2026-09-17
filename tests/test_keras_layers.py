"""Keras 3 layer tests: graph-mode (tf.function) compatibility.

Regression tests for KerasQBatchNormalization: under Keras 3, ``call`` is
traced under tf.function during ``fit``, so any eager-only operation such as
``ops.convert_to_numpy`` on a variable raises
``numpy() is only available when eager execution is enabled``.
"""
import numpy as np
import pytest

keras = pytest.importorskip("keras")

import quantumflow.keras as qf_keras  # noqa: E402


def _tiny_model(n_layers=1):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            qf_keras.KerasQDense(units=8, n_qubits=4, n_layers=n_layers),
            qf_keras.KerasQBatchNormalization(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


class TestKerasQBatchNormalization:
    def test_fit_graph_mode(self):
        """fit() traces call() under tf.function — must not raise."""
        X = np.random.randn(64, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        model = _tiny_model()
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)

    def test_moving_stats_update(self):
        X = np.random.randn(64, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        model = _tiny_model()
        bn = model.layers[1]
        before = np.array(bn._moving_mean.numpy())
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        after = np.array(bn._moving_mean.numpy())
        assert not np.allclose(before, after), "moving stats did not update"

    def test_inference_uses_moving_stats(self):
        X = np.random.randn(64, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        model = _tiny_model()
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        p1 = model.predict(X[:8], verbose=0)
        p2 = model.predict(X[:8], verbose=0)
        assert p1.shape == (8, 1)
        assert np.allclose(p1, p2), "inference should be deterministic"

    def test_finite_outputs(self):
        X = np.random.randn(32, 4).astype(np.float32)
        y = (X[:, 1] > 0).astype(np.float32)
        model = _tiny_model()
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        preds = model.predict(X, verbose=0)
        assert np.all(np.isfinite(preds))
