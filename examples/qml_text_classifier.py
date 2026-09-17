"""End-to-end quantum ML integration check on a text-classification workload.

Builds a sentiment classifier from a tiny corpus and pushes every major
QuantumFlow subsystem through it: the plain-TF quantum stack (feature map,
dense, batch-norm, attention, residual, measurement), a Keras 3 hybrid
model trained with fit(), QClassifier/QRegressor, all five data encodings,
all six quantum optimizers, quantum conv/pool on character glyphs, quantum
activations, and the autoencoder/VAE/GAN/hybrid/transfer model zoo.

Run with:  python examples/qml_text_classifier.py   (~10-15 min; TF required)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re

import numpy as np

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


def text_features(texts, dim=12):
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


X = text_features(TEXTS)
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
X, y = X[idx], LABELS[idx]
Xtr, Xte, ytr, yte = X[:-8], X[-8:], y[:-8], y[-8:]
print(f"data: X{Xtr.shape} y{ytr.shape}, val {Xte.shape[0]}")

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True

# ---------------------------------------------------------------------------
# A. plain-TF deep quantum stack (forward)
# ---------------------------------------------------------------------------
from quantumflow.tensorflow.layers import (
    QFeatureMapLayer, QDenseLayer, QBatchNormLayer,
    QAttentionLayer, QResidualLayer, QMeasurementLayer,
)

print("\n[A] deep plain-TF quantum stack")
try:
    attn = QAttentionLayer(n_qubits=6, n_layers=1)
    stack = [
        QFeatureMapLayer(n_qubits=6, feature_map="zx", n_reps=1),
        QDenseLayer(units=6, n_qubits=6, n_layers=1),
        QBatchNormLayer(n_qubits=6),
    ]
    h = Xtr[:4]
    for layer in stack:
        h = layer(h)
        assert np.all(np.isfinite(np.asarray(h))), type(layer).__name__
    # attention stage: (query, key, value) each (batch, seq=2, d=3)
    h3 = np.asarray(h).reshape(4, 2, 3)
    h = attn((h3, h3, h3)).reshape(4, 6)
    assert np.all(np.isfinite(np.asarray(h))), "attention"
    for layer in [QResidualLayer(n_qubits=6, n_layers=1),
                  QMeasurementLayer(n_qubits=6, observable="z", strategy="expectation")]:
        h = layer(h)
        assert np.all(np.isfinite(np.asarray(h))), type(layer).__name__
    assert np.asarray(h).shape == (4, 6), np.asarray(h).shape
    ok &= check("stack forward (4,6) finite incl. attention", True)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"stack forward: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# B. Keras 3 hybrid text classifier
# ---------------------------------------------------------------------------
print("\n[B] Keras 3 hybrid text classifier")
import keras
from quantumflow.keras.layers import KerasQDense, KerasQVariational

try:
    model = keras.Sequential([
        keras.Input(shape=(12,)),
        keras.layers.Dense(8, activation="relu"),
        KerasQDense(units=4, n_qubits=4, n_layers=2),
        KerasQVariational(n_qubits=4, n_params=12, n_layers=1, observable="z"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),
                  loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, ytr, validation_data=(Xte, yte),
                     epochs=30, batch_size=8, verbose=0)
    tr0, tr1 = hist.history["loss"][0], hist.history["loss"][-1]
    acc = hist.history["val_accuracy"][-1]
    preds = model.predict(X, verbose=0)
    ok &= check(f"fit loss {tr0:.3f}->{tr1:.3f}, val_acc {acc:.2f}, finite preds",
                tr1 < tr0 - 0.05 and np.all(np.isfinite(preds)) and acc >= 0.25)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"keras classifier: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# C. QClassifier (TF models, own optimizer stack)
# ---------------------------------------------------------------------------
print("\n[C] QClassifier (tensorflow/models)")
from quantumflow.tensorflow.models import QClassifier

try:
    # native 4-qubit features (the 12-dim hashed set would be chunk-averaged
    # to 4 inputs, destroying most of the signal for a 4-qubit PQC)
    qclf = QClassifier(n_qubits=4, n_classes=2, n_layers=2,
                       feature_map="angle", random_state=1, learning_rate=0.1)
    qclf.compile(optimizer="adam", loss="binary_crossentropy")
    Xq = text_features(TEXTS, dim=4)
    perm = np.random.default_rng(0).permutation(len(Xq))
    Xq, yq = Xq[perm], LABELS[perm]
    hist = qclf.fit(Xq[:-6], yq[:-6], epochs=25, batch_size=6, verbose=0)
    losses = hist["loss"]
    ok &= check(f"QClassifier learns: loss {losses[0]:.3f}->{losses[-1]:.3f}",
                losses[-1] < losses[0] - 0.05)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"QClassifier: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# D. QuantumNNLayer with all five encodings
# ---------------------------------------------------------------------------
print("\n[D] QuantumNNLayer encodings sweep")
from quantumflow.neural.qnn_layer import QuantumNNLayer

enc_dims = {"angle": 4, "amplitude": 16, "basis": 4, "iqp": 4, "dense_angle": 4}
for enc, d in enc_dims.items():
    try:
        Xenc = Xtr[:, :d]
        layer = QuantumNNLayer(n_qubits=4, n_layers=1, encoding=enc)
        out = np.asarray(layer.forward(Xenc[0]))
        ok &= check(f"encoding={enc} forward finite {out.shape}",
                    np.all(np.isfinite(out)))
    except Exception as e:
        ok &= check(f"encoding={enc}: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# E. Quantum optimizer sweep on a small objective
# ---------------------------------------------------------------------------
print("\n[E] optimizer sweep")
from quantumflow.tensorflow.optimizers import (
    NaturalGradientOptimizer, ParameterShiftOptimizer,
    QuantumAdam, QuantumLAMB, QuantumSGD, SpsaOptimizer,
)


def objective(p):
    return float(np.sum((p - np.array([0.3, -0.7, 0.5])) ** 2))


for Opt in [QuantumAdam, QuantumSGD, QuantumLAMB, SpsaOptimizer,
            ParameterShiftOptimizer, NaturalGradientOptimizer]:
    try:
        opt = Opt(learning_rate=0.05)
        p0 = np.zeros(3)
        params = opt.minimize(objective, p0, n_iterations=30, verbose=0)
        f0, f1 = objective(p0), objective(params)
        ok &= check(f"{Opt.__name__:26s} {f0:.3f} -> {f1:.3f}", f1 < f0)
    except Exception as e:
        import traceback; traceback.print_exc()
        ok &= check(f"{Opt.__name__}: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# F. text glyph images -> quantum conv/pool
# ---------------------------------------------------------------------------
print("\n[F] QuantumConv2D/Pool2D on character glyphs")
from quantumflow.neural.quantum_conv import QuantumConv2D, QuantumPool2D

try:
    GLYPHS = {
        "X": ["1 0 0 0 1", "0 1 0 1 0", "0 0 1 0 0", "0 1 0 1 0", "1 0 0 0 1"],
        "O": ["0 1 1 1 0", "1 0 0 0 1", "1 0 0 0 1", "1 0 0 0 1", "0 1 1 1 0"],
        "L": ["1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 0 0 0 0", "1 1 1 1 1"],
    }
    imgs = []
    for letter in "XOL":
        im = np.zeros((8, 8))
        g = np.array([[float(c) for c in row.split()] for row in GLYPHS[letter]])
        im[1:6, 1:6] = g * (np.pi / 2)
        imgs.append(im)
    batch = np.stack(imgs)[..., None]  # (3, 8, 8, 1)

    conv = QuantumConv2D(filters=2, kernel_size=3, n_qubits=4, n_layers=1)
    feat = np.asarray(conv(batch))
    pool = QuantumPool2D(pool_size=2, n_qubits=2)
    pooled = np.asarray(pool(feat))
    ok &= check(f"conv {feat.shape} -> pool {pooled.shape} finite",
                feat.shape[0] == 3 and np.all(np.isfinite(feat))
                and np.all(np.isfinite(pooled)))
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"conv/pool: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# G. quantum activations
# ---------------------------------------------------------------------------
print("\n[G] quantum activations")
from quantumflow.neural.quantum_activation import (
    QuantumReLU, QuantumSigmoid, QuantumSoftmax, QuantumSwish, QuantumTanh,
)

for Act, kw in [(QuantumReLU, {"n_layers": 2}), (QuantumSigmoid, {"n_layers": 2}),
                (QuantumTanh, {"n_layers": 2}), (QuantumSoftmax, {}),
                (QuantumSwish, {"n_layers": 2})]:
    try:
        act = Act(n_qubits=2, **kw)
        out = np.asarray(act.forward([0.3, -0.8]))
        ok &= check(f"{Act.__name__:16s} finite {out.shape}",
                    np.all(np.isfinite(out)))
    except Exception as e:
        ok &= check(f"{Act.__name__}: {type(e).__name__}: {e}", False)

# ---------------------------------------------------------------------------
# H. generative / unsupervised Keras models (smoke)
# ---------------------------------------------------------------------------
print("\n[H] autoencoder / VAE / GAN / hybrid / transfer (Keras)")
from quantumflow.keras.models import (
    KerasHybridModel, KerasQuantumAutoencoder, KerasQuantumGAN,
    KerasQuantumVAE, KerasTransferLearning,
)

try:
    Xa = Xtr[:, :4]
    ae = KerasQuantumAutoencoder(n_qubits=4, n_trash_qubits=2)
    ae.compile()
    ae.fit(Xa, epochs=1, batch_size=4, verbose=0)
    lat = ae.encode(Xa[:3])
    rec = ae.decode(lat)
    ok &= check(f"KerasQuantumAutoencoder enc {np.shape(lat)} -> rec {np.shape(rec)}",
                np.all(np.isfinite(rec)))
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"KerasQuantumAutoencoder: {type(e).__name__}: {e}", False)

try:
    vae = KerasQuantumVAE(n_qubits=4, latent_dim=2)
    vae.compile()
    vae.fit(Xa, epochs=1, batch_size=4, verbose=0)
    ok &= check("KerasQuantumVAE fit", True)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"KerasQuantumVAE: {type(e).__name__}: {e}", False)

try:
    gan = KerasQuantumGAN(n_qubits=4, latent_dim=4)
    gan.compile()
    gan.fit(Xa, epochs=1, batch_size=4, verbose=0)
    ok &= check("KerasQuantumGAN fit", True)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"KerasQuantumGAN: {type(e).__name__}: {e}", False)

try:
    hyb = KerasHybridModel(input_shape=(12,))
    hyb.add_classical_layer("dense", units=8, activation="relu")
    hyb.add_quantum_layer(n_qubits=4, n_layers=1)
    hyb.add_classical_layer("dense", units=1, activation="sigmoid")
    hyb.compile(optimizer="adam", loss="binary_crossentropy")
    hyb.fit(Xtr, ytr, epochs=1, batch_size=4, verbose=0)
    ok &= check("KerasHybridModel fit", True)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"KerasHybridModel: {type(e).__name__}: {e}", False)

try:
    tl = KerasTransferLearning(backbone_layers=[{"units": 8, "activation": "relu"}],
                               n_qubits=4, strategy="quantum_only")
    tl.compile()
    tl.fit(Xtr, ytr, epochs=1, batch_size=4, verbose=0)
    ok &= check("KerasTransferLearning fit", True)
except Exception as e:
    import traceback; traceback.print_exc()
    ok &= check(f"KerasTransferLearning: {type(e).__name__}: {e}", False)

n_pass = sum(1 for _, p in results if p)
print(f"\n===== RESULT: {n_pass}/{len(results)} PASS =====")
for name, p in results:
    if not p:
        print("  FAILED:", name)
