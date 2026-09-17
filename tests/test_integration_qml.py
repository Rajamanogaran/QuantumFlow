"""
Quantum ML integration tests
============================

Functional checks for the full quantum-ML stack on a text-classification
workload: plain-TF quantum layers, Keras 3 hybrid training, QClassifier /
QRegressor joint (circuit + readout) training, data encodings, quantum
optimizers, quantum conv/pool on character glyphs, quantum activations,
and the Keras model zoo.  Heavy full-length training runs live in
``examples/qml_text_classifier.py``.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import re

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from quantumflow.keras.layers import KerasQDense  # noqa: E402
from quantumflow.neural.qnn_layer import QuantumNNLayer  # noqa: E402
from quantumflow.neural.quantum_activation import (  # noqa: E402
    QuantumReLU,
    QuantumSigmoid,
    QuantumSoftmax,
    QuantumSwish,
    QuantumTanh,
)
from quantumflow.neural.quantum_conv import QuantumConv2D, QuantumPool2D  # noqa: E402
from quantumflow.tensorflow.layers import (  # noqa: E402
    QAttentionLayer,
    QBatchNormLayer,
    QDenseLayer,
    QFeatureMapLayer,
    QMeasurementLayer,
    QResidualLayer,
)
from quantumflow.tensorflow.models import QClassifier, QRegressor  # noqa: E402
from quantumflow.tensorflow.optimizers import (  # noqa: E402
    NaturalGradientOptimizer,
    ParameterShiftOptimizer,
    QuantumAdam,
    QuantumLAMB,
    QuantumSGD,
    SpsaOptimizer,
)

# ---------------------------------------------------------------------------
# Text data: tiny sentiment corpus -> char-trigram hashing features
# ---------------------------------------------------------------------------

POS = [
    "i loved this film", "what a great movie", "absolutely wonderful acting",
    "the story was beautiful and moving", "best film of the year",
    "brilliant and heartwarming", "a delightful experience",
    "superb direction and cast", "i enjoyed every minute",
    "fantastic plot and pacing", "this comedy made my day",
    "an instant classic", "the characters felt alive",
    "wonderful soundtrack and visuals", "i would watch it again",
    "excellent from start to finish",
]
NEG = [
    "i hated this film", "what a terrible movie", "awful acting throughout",
    "the story was dull and lifeless", "worst film of the year",
    "boring and predictable", "a painful experience",
    "poor direction and miscast actors", "i checked my watch constantly",
    "the plot made no sense", "this comedy fell completely flat",
    "a total waste of time", "the characters felt cardboard",
    "grating soundtrack and ugly visuals", "i walked out halfway",
    "bad from start to finish",
]
LABELS = np.array([1] * len(POS) + [0] * len(NEG), dtype=np.float64)
TEXTS = POS + NEG


def text_features(texts, dim):
    """Char 3-gram hashing -> signed counts -> L2 norm -> scale to [-pi, pi]."""
    rows = []
    for t in texts:
        t = re.sub(r"\s+", " ", t.strip().lower())
        v = np.zeros(dim)
        for i in range(max(len(t) - 2, 0)):
            g = t[i:i + 3].encode()
            h = int.from_bytes(g, "little") % dim
            sign = 1.0 if (int.from_bytes(g, "little") // dim) % 2 == 0 else -1.0
            v[h] += sign
        n = np.linalg.norm(v)
        rows.append(v / n if n > 0 else v)
    X = np.array(rows)
    return X * (np.pi / (2.0 * np.abs(X).max()))


@pytest.fixture(scope="module")
def data():
    X = text_features(TEXTS, dim=12)
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], LABELS[perm]
    return X[:-8], X[-8:], y[:-8], y[-8:]


# ---------------------------------------------------------------------------
# A. deep plain-TF quantum stack
# ---------------------------------------------------------------------------


def test_deep_plain_tf_quantum_stack(data):
    Xtr = data[0]
    attn = QAttentionLayer(n_qubits=6, n_layers=1)
    h = Xtr[:4]
    for layer in [
        QFeatureMapLayer(n_qubits=6, feature_map="zx", n_reps=1),
        QDenseLayer(units=6, n_qubits=6, n_layers=1),
        QBatchNormLayer(n_qubits=6),
    ]:
        h = np.asarray(layer(h))
        assert np.all(np.isfinite(h))
    # attention consumes a (query, key, value) tuple of (batch, seq, d)
    h3 = h.reshape(4, 2, 3)
    h = np.asarray(attn((h3, h3, h3))).reshape(4, 6)
    assert np.all(np.isfinite(h))
    for layer in [
        QResidualLayer(n_qubits=6, n_layers=1),
        QMeasurementLayer(n_qubits=6, observable="z", strategy="expectation"),
    ]:
        h = np.asarray(layer(h))
        assert np.all(np.isfinite(h))
    assert h.shape == (4, 6)


# ---------------------------------------------------------------------------
# B. Keras 3 hybrid text classifier learns
# ---------------------------------------------------------------------------


def test_keras_hybrid_text_classifier_learns(data):
    Xtr, Xte, ytr, yte = data
    model = keras.Sequential([
        keras.Input(shape=(12,)),
        keras.layers.Dense(8, activation="relu"),
        KerasQDense(units=1, n_qubits=4, n_layers=1),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05),
                  loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                     epochs=4, batch_size=8, verbose=0)
    losses = hist.history["loss"]
    assert losses[-1] < losses[0] - 0.03
    preds = model.predict(Xtr, verbose=0)
    assert preds.shape == (len(Xtr), 1)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# C. QClassifier / QRegressor joint circuit+readout training
# ---------------------------------------------------------------------------
# Regression: the readout previously stayed frozen at its random init, so
# neither model could learn; these checks guard the fix.


def test_qclassifier_learns():
    Xq = text_features(TEXTS, dim=4)
    perm = np.random.default_rng(0).permutation(len(Xq))
    Xq, yq = Xq[perm], LABELS[perm]
    clf = QClassifier(n_qubits=4, n_classes=2, n_layers=1,
                      feature_map="angle", random_state=1, learning_rate=0.1)
    clf.compile()
    hist = clf.fit(Xq[:-6], yq[:-6], epochs=6, batch_size=6, verbose=0)
    losses = hist["loss"]
    assert len(losses) == 6
    assert min(losses) < losses[0]  # training moves the loss down
    assert np.all(np.isfinite(clf.predict(Xq[:4])))


def test_qregressor_learns():
    rng = np.random.default_rng(5)
    Xr = rng.uniform(-np.pi, np.pi, (16, 4))
    yr = np.sin(Xr[:, 0]).reshape(-1, 1)
    reg = QRegressor(n_qubits=4, n_outputs=1, n_layers=1,
                     random_state=2, learning_rate=0.3)
    reg.compile()
    hist = reg.fit(Xr, yr, epochs=10, batch_size=4, verbose=0)
    losses = hist["loss"]
    assert losses[-1] < losses[0] * 0.6  # substantial convergence


# ---------------------------------------------------------------------------
# D. QuantumNNLayer encodings sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enc,dim", [("angle", 4), ("amplitude", 16),
                                     ("basis", 4), ("iqp", 4),
                                     ("dense_angle", 4)])
def test_qnn_layer_encodings(enc, dim, data):
    Xtr = data[0]
    x = Xtr[0]
    if dim <= len(x):
        x = x[:dim]
    else:  # amplitude needs 2**n_qubits features: tile and renormalise
        x = np.resize(x, dim)
        x = x / np.linalg.norm(x) * np.pi
    layer = QuantumNNLayer(n_qubits=4, n_layers=1, encoding=enc)
    out = np.asarray(layer.forward(x))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# E. quantum optimizer sweep
# ---------------------------------------------------------------------------


def _quadratic_objective(p):
    return float(np.sum((p - np.array([0.3, -0.7, 0.5])) ** 2))


@pytest.mark.parametrize(
    "Opt",
    [QuantumAdam, QuantumSGD, QuantumLAMB, SpsaOptimizer,
     ParameterShiftOptimizer, NaturalGradientOptimizer],
)
def test_quantum_optimizers_converge(Opt):
    # QuantumLAMB regression: with zero-initialised parameters the trust
    # ratio was 0/|u| = 0, so the optimizer never moved off the origin.
    opt = Opt(learning_rate=0.05)
    p0 = np.zeros(3)
    params = opt.minimize(_quadratic_objective, p0, n_iterations=30, verbose=0)
    assert _quadratic_objective(params) < _quadratic_objective(p0)


# ---------------------------------------------------------------------------
# F. quantum conv/pool on character glyphs
# ---------------------------------------------------------------------------


def test_quantum_conv_pool_on_glyphs():
    GLYPHS = {
        "X": ["1 0 0 0 1", "0 1 0 1 0", "0 0 1 0 0", "0 1 0 1 0", "1 0 0 0 1"],
        "O": ["0 1 1 1 0", "1 0 0 0 1", "1 0 0 0 1", "1 0 0 0 1", "0 1 1 1 0"],
        "L": ["1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 1 1 1 1"],
    }
    batch = []
    for letter in "XOL":
        im = np.zeros((8, 8))
        g = np.array([[float(c) for c in row.split()] for row in GLYPHS[letter]])
        im[1:6, 1:6] = g * (np.pi / 2)
        batch.append(im)
    batch = np.stack(batch)[..., None]

    conv = QuantumConv2D(filters=2, kernel_size=3, n_qubits=4, n_layers=1)
    feat = np.asarray(conv(batch))
    assert feat.shape[0] == 3 and np.all(np.isfinite(feat))

    pool = QuantumPool2D(pool_size=2, n_qubits=2)
    pooled = np.asarray(pool(feat))  # lazy auto-build (previously raised)
    assert pooled.shape[0] == 3 and np.all(np.isfinite(pooled))


# ---------------------------------------------------------------------------
# G. quantum activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Act",
    [QuantumReLU, QuantumSigmoid, QuantumTanh, QuantumSoftmax, QuantumSwish],
)
def test_quantum_activations(Act):
    act = Act(n_qubits=2) if Act is QuantumSoftmax else Act(n_qubits=2, n_layers=2)
    out = np.asarray(act.forward([0.3, -0.8]))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# H. Keras model zoo smoke
# ---------------------------------------------------------------------------


def test_keras_model_zoo(data):
    from quantumflow.keras.models import (
        KerasHybridModel,
        KerasQuantumAutoencoder,
        KerasQuantumGAN,
        KerasQuantumVAE,
        KerasTransferLearning,
    )

    Xtr, _, ytr, _ = data
    Xa = Xtr[:, :4]

    ae = KerasQuantumAutoencoder(n_qubits=4, n_trash_qubits=2)
    ae.compile()
    ae.fit(Xa, epochs=1, batch_size=4, verbose=0)
    lat = np.asarray(ae.encode(Xa[:3]))
    rec = np.asarray(ae.decode(lat))
    assert np.all(np.isfinite(lat)) and np.all(np.isfinite(rec))

    vae = KerasQuantumVAE(n_qubits=4, latent_dim=2)
    vae.compile()
    vae.fit(Xa, epochs=1, batch_size=4, verbose=0)

    gan = KerasQuantumGAN(n_qubits=4, latent_dim=4)
    gan.compile()
    gan.fit(Xa, epochs=1, batch_size=4, verbose=0)

    hyb = KerasHybridModel(input_shape=(12,))
    hyb.add_classical_layer("dense", units=8, activation="relu")
    hyb.add_quantum_layer(n_qubits=4, n_layers=1)
    hyb.add_classical_layer("dense", units=1, activation="sigmoid")
    hyb.compile(optimizer="adam", loss="binary_crossentropy")
    hyb.fit(Xtr, ytr, epochs=1, batch_size=4, verbose=0)

    tl = KerasTransferLearning(
        backbone_layers=[{"units": 8, "activation": "relu"}],
        n_qubits=4, strategy="quantum_only",
    )
    tl.compile()
    tl.fit(Xtr, ytr, epochs=1, batch_size=4, verbose=0)
