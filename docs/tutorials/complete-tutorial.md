# The Complete QuantumFlow Tutorial

*From your first qubit to trained quantum neural networks — every code
block in this tutorial is executed as part of the documentation test
harness, so you can follow along line by line.*

**Prerequisites:** Python 3.9+, `pip install quantumflow` (or
`pip install .` from a source checkout). Chapters 8–9 additionally need
the ML extra: `pip install "quantumflow[tf]"` (TensorFlow ≥ 2.14 with
Keras 3).

**Conventions used everywhere (memorise these three):**

1. **Qubit 0 is the most significant bit.** Bitstring results read
   `q0 q1 q2 …`.
2. **`shots=0` returns exact quantum states** (no sampling). Any positive
   shot count returns measurement statistics.
3. **Randomness is seeded through `BackendConfig`**, not the simulator
   constructor: `StatevectorSimulator(config=BackendConfig(seed=42))`.

---

## Table of contents

1. [Quantum states](#1-quantum-states)
2. [Circuits and gates](#2-circuits-and-gates)
3. [Simulation](#3-simulation)
4. [Noise and error mitigation](#4-noise-and-error-mitigation)
5. [Quantum algorithms](#5-quantum-algorithms)
6. [Variational algorithms: VQE and QAOA](#6-variational-algorithms-vqe-and-qaoa)
7. [Quantum machine learning with Keras 3](#7-quantum-machine-learning-with-keras-3)
8. [The wider ML toolbox](#8-the-wider-ml-toolbox)
9. [Visualization](#9-visualization)
10. [Performance, testing and FAQ](#10-performance-testing-and-faq)

---

## 1. Quantum states

A quantum state of *n* qubits is a vector of `2**n` complex amplitudes.
`Statevector` wraps one; `DensityMatrix` wraps the mixed-state general.

```python
import numpy as np
import quantumflow as qf

# A Bell state: (|00> + |11>) / sqrt(2)
bell = qf.Statevector([1, 0, 0, 1]) / np.sqrt(2)
print("amplitudes:", np.round(bell.data, 4))
print("qubits:", bell.num_qubits)
```

`Statevector` arithmetic keeps results physical (states stay
normalised):

```python
scaled = 2 * bell          # still a Statevector, re-normalised
print(type(scaled).__name__, "norm:", np.linalg.norm(scaled.data))

plus = qf.Statevector.from_label("++")   # |+> on each qubit
probs = plus.probabilities()             # marginal probabilities
print("P(qubit-1 = 1):", probs)
```

Pure states are a special case of density matrices; conversion and the
usual quantities live in `quantumflow.utils.math`:

```python
from quantumflow.utils.math import fidelity, purity, von_neumann_entropy

rho = bell.to_density_matrix()
print("purity:", round(purity(rho.data), 6))                 # 1.0 -> pure
print("entropy:", round(von_neumann_entropy(rho.data), 6))   # 1 bit (entangled)

product = qf.Statevector([1, 0, 0, 0]).to_density_matrix()
print("fidelity(bell, |00>):", round(fidelity(rho.data, product.data), 4))
```

Individual qubits of a mixed state can be inspected by partial trace:

```python
from quantumflow.utils.math import partial_trace

# trace out qubit 0 -> the reduced state of qubit 1 is maximally mixed
red = partial_trace(rho.data, qubits_to_keep=[1], n_qubits=2)
print("reduced state diagonal:", np.round(np.diag(red).real, 4))
print("mixed? purity:", round(purity(red), 4))
```

---

## 2. Circuits and gates

`QuantumCircuit` is the centre of the framework. Gates are appended with
fluent methods; 50+ gates are available.

```python
qc = qf.QuantumCircuit(3)

# Single-qubit gates
qc.h(0)              # Hadamard
qc.x(1)              # Pauli-X (NOT)
qc.ry(np.pi / 2, 2)  # rotation about Y

# Two- and three-qubit gates
qc.cx(0, 1)          # CNET: control=0, target=1
qc.swap(0, 2)        # SWAP
qc.ccx(0, 1, 2)      # Toffoli
qc.rzz(0.5, 0, 1)    # ZZ rotation (used heavily in QAOA)

print("depth():", qc.depth(), "| operations:", len(qc.data))
```

Circuits are unitary operators; `to_unitary()` materialises the full
`2**n x 2**n` matrix (feasible for small n):

```python
bell_circ = qf.QuantumCircuit(2)
bell_circ.h(0)
bell_circ.cx(0, 1)

U = bell_circ.to_unitary()
expected = np.array([[1, 0, 0, 1],
                     [0, 1, 1, 0],
                     [0, 1, 1, 0],
                     [1, 0, 0, 1]]) / np.sqrt(2)
print("circuit == Bell unitary:", np.allclose(U, expected))
```

Circuits compose, invert, and copy:

```python
inv = bell_circ.inverse()                      # undo the Bell prep
roundtrip = inv.compose(bell_circ)             # other runs AFTER self
identity = roundtrip.to_unitary()
print("inverse composes to identity:", np.allclose(identity, np.eye(4)))
```

`compose` can also splice small circuits onto specific wires of a larger
one (used internally by the algorithm library):

```python
big = qf.QuantumCircuit(3)
small = qf.QuantumCircuit(1)
small.x(0)
big.compose(small, qubits=[2], inplace=True)   # X on qubit 2 only
print("composed ops:", len(big.data))
```

Measurements map qubits to classical bits; the classical register is
created automatically if the circuit does not have one:

```python
meas = qf.QuantumCircuit(3)
meas.h(0)
meas.measure([0, 1, 2], [0, 1, 2])
print("classical bits:", meas.num_clbits)
```

Gates can also be attached explicitly — useful for building custom
gates:

```python
from quantumflow.core import UnitaryGate

any_unitary = qf.utils.math.random_unitary(4)          # random 2-qubit unitary
custom = qf.QuantumCircuit(2)
custom.append(UnitaryGate(any_unitary, name="my_gate"), [0, 1])
print("custom gate applied:", np.allclose(custom.to_unitary(), any_unitary))
```

---

## 3. Simulation

Three simulators share one interface:

| Simulator | State | Use for |
|---|---|---|
| `StatevectorSimulator` | pure state vector | fast, default choice |
| `DensityMatrixSimulator` | density matrix | noise, mixed states, channels |
| `MPSimulator` | matrix product state | shallow circuits on many qubits |

```python
qc = qf.QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)                       # GHZ state
qc.measure([0, 1, 2], [0, 1, 2])

result = qf.StatevectorSimulator().run(qc, shots=1024)
print("counts:", result.get_counts())
print("most frequent:", result.most_frequent(2))
```

With `shots=0` you get the exact final state instead of samples:

```python
exact = qf.StatevectorSimulator().run(qc, shots=0).statevector
print("amplitudes:", np.round(exact.data, 4))
```

Sampling statistics are seeded via `BackendConfig`; the same seed
reproduces the same counts:

```python
from quantumflow.simulation.simulator import BackendConfig

seeded = qf.StatevectorSimulator(config=BackendConfig(seed=7))
coin = qf.QuantumCircuit(1)
coin.h(0)
coin.measure([0], [0])
runs = [seeded.run(coin, shots=100).get_counts().get("1", 0) for _ in range(1)]
print("seeded sample of |1>:", runs)
```

**Mid-circuit measurements are handled correctly**: every shot is a
fresh execution (fresh collapse). Compare a mid-circuit measurement —
the second H undoes the first one only when the first outcome was 0:

```python
mid = qf.QuantumCircuit(2)
mid.h(0)
mid.measure([0], [0])     # collapse qubit 0 halfway through
mid.h(0)                  # H H = I, but only for the |0> branch
mid.measure([0, 1], [0, 1])

counts = qf.StatevectorSimulator().run(mid, shots=2000).get_counts()
print("mid-circuit counts:", counts)   # '00' more likely than '10'
```

Expectation values are first-class:

```python
Z = np.diag([1, -1])
sim = qf.StatevectorSimulator()
print("<Z0> of GHZ:", sim.expectation(qc, np.kron(Z, np.eye(4))))
```

The other simulators drop in seamlessly — same `run`, same result type:

```python
dm_result = qf.DensityMatrixSimulator().run(qc, shots=512)
print("DM counts:", dm_result.get_counts())

mps_result = qf.MPSimulator().run(qc, shots=512)
print("MPS counts:", mps_result.get_counts())
```

Custom input states are passed as vectors:

```python
one = np.array([0, 1], dtype=complex)                       # |1>
out = qf.StatevectorSimulator().run(
    qf.QuantumCircuit(1), shots=0, initial_state=np.kron(one, one)
)
print("started from |11>, stayed:", np.round(out.statevector.data, 3))
```

Any statevector can be turned back into a preparation circuit:

```python
prep = exact.to_circuit()                     # exact state-prep circuit
out2 = qf.StatevectorSimulator().run(prep, shots=0).statevector
fid = abs(np.vdot(exact.data, out2.data)) ** 2
print("prep fidelity:", round(fid, 10))
```

---

## 4. Noise and error mitigation

Real hardware is noisy. QuantumFlow models noise as **Kraus channels** —
general (non-unitary) operations that only the density-matrix simulator
can execute.

### 4.1 Noise models

`NoiseModel` describes per-gate error probabilities. Attaching it to the
density-matrix simulator applies a channel after every gate during
simulation:

```python
from quantumflow.noise.noise_model import NoiseConfig, NoiseModel

config = NoiseConfig(single_gate_error=0.05, two_gate_error=0.10)
noise = NoiseModel(config)

noisy_sim = qf.DensityMatrixSimulator(noise_model=noise)

ideal_circ = qf.QuantumCircuit(2)
ideal_circ.x(0)
ideal_circ.measure([0, 1], [0, 1])

clean = qf.StatevectorSimulator().run(ideal_circ, shots=4000).get_counts()
noisy = noisy_sim.run(ideal_circ, shots=4000).get_counts()
print("ideal P(10):", clean.get("10", 0) / 4000)
print("noisy P(10):", round(noisy.get("10", 0) / 4000, 4))
```

Alternatively, bake the noise into the circuit itself. Each gate gets a
Kraus channel appended; `noise_scale` amplifies the error probabilities
(the basis of zero-noise extrapolation below):

```python
noisy_circ = noise.apply_noise(ideal_circ, noise_scale=1.0)
print("ops before:", len(ideal_circ.data), "| after:", len(noisy_circ.data))

for scale in (1.0, 2.0, 4.0):
    res = qf.DensityMatrixSimulator().run(
        noise.apply_noise(ideal_circ, noise_scale=scale), shots=4000
    )
    print(f"scale {scale}: P(10) = {res.get_counts().get('10', 0) / 4000:.4f}")
```

You can attach channels manually with `append_kraus` — any physical
channel works as long as the Kraus operators satisfy the completeness
relation:

```python
manual = qf.QuantumCircuit(1)
manual.x(0)
manual.append_kraus(
    [np.sqrt(0.7) * np.eye(2), np.sqrt(0.3) * np.array([[0, 1], [1, 0]])],
    [0],
)
manual.measure([0], [0])
counts = qf.DensityMatrixSimulator().run(manual, shots=2000).get_counts()
print("30% bit-flip channel:", counts)
```

### 4.2 Zero-noise extrapolation

Run the circuit at amplified noise levels and extrapolate back to the
zero-noise limit:

```python
from quantumflow.noise.error_mitigation import ZeroNoiseExtrapolation

def p10_at(scale):
    circ = noise.apply_noise(ideal_circ, noise_scale=scale)
    counts = qf.DensityMatrixSimulator().run(circ, shots=4000).get_counts()
    return counts.get("10", 0) / 4000

values = [p10_at(s) for s in (1.0, 2.0, 3.0)]
zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], method="linear")
mitigated = zne.mitigate({}, noisy_expectations=values)
print("noisy values:", [round(v, 4) for v in values])
print("mitigated:", round(mitigated["mitigated_value"], 4), "(ideal: 1.0)")
```

---

## 5. Quantum algorithms

### 5.1 Quantum Fourier transform

```python
from quantumflow.algorithms.qft import QFT, InverseQFT, qft_matrix

U = QFT(3).construct_circuit().to_unitary()
print("QFT(3) == textbook matrix:", np.allclose(U, qft_matrix(3)))

iqft = InverseQFT(3).construct_circuit().to_unitary()
print("QFT @ IQFT == I:", np.allclose(U @ iqft, np.eye(8)))
```

`apply_qft` / `apply_iqft` append the transform to an existing circuit
on any contiguous set of wires:

```python
from quantumflow.algorithms.qft import apply_qft, apply_iqft

circ = qf.QuantumCircuit(3)
circ.x(1)                    # |010>
apply_qft(circ, [0, 2])      # QFT on qubits 0 and 2 only
apply_iqft(circ, [0, 2])
back = qf.StatevectorSimulator().run(circ, shots=0).statevector
print("roundtrip |010>:", np.allclose(back.data, [0, 0, 1, 0, 0, 0, 0, 0]))
```

### 5.2 Phase estimation

QPE reads out the phase φ of `U|ψ⟩ = e^{2πiφ}|ψ⟩` exactly when φ is a
dyadic fraction:

```python
phi = 0.25
U = np.diag([1, np.exp(1j * 2 * np.pi * phi)])   # eigenvalue e^{2πiφ} on |1>

qpe = qf.PhaseEstimation(U, n_evaluation_qubits=4, n_state_qubits=1)
res = qpe.run(shots=4096)
print("estimated phase:", res["phase"], "| exact:", phi)
```

The *iterative* variant re-uses a single evaluation qubit with a
classical feedback loop — the same result with minimal hardware:

```python
from quantumflow.algorithms.qpe import IterativePhaseEstimation

ipe = IterativePhaseEstimation(
    U, n_state_qubits=1, n_iterations=8,
    eigenstate=np.array([0, 1], dtype=complex),
)
res = ipe.run(shots_per_iteration=256)
print("IPE phase:", round(res["phase"], 5), "| bits (MSB-first):", res["bit_estimates"])
```

### 5.3 Grover search

```python
grover = qf.GroverSearch(n_qubits=4, marked_states=["1010"])
res = grover.run(shots=512)
print("found:", res["most_frequent"])
print("theoretical success probability:", round(res["success_probability"], 4))
```

Multiple marked states work too:

```python
g2 = qf.GroverSearch(n_qubits=4, marked_states=["0110", "1001"])
print("found one of the two:", g2.run(shots=512)["most_frequent"] in ("0110", "1001"))
```

### 5.4 Shor factoring

Shor reduces factoring to quantum order finding (exact controlled
modular multiplication + QPE + continued fractions):

```python
shor = qf.ShorAlgorithm(N=15, a=7)
res = shor.factor(shots=1024)
print("N=15 ->", res["factors"], "| success:", res["success"])
```

---

## 6. Variational algorithms: VQE and QAOA

### 6.1 VQE

The variational quantum eigensolver finds the ground state of a
Hamiltonian with a parameterised circuit plus a classical optimiser.

```python
from quantumflow.algorithms.vqe import Hamiltonian, HWEAnsatz, PauliTerm, VQE

H = Hamiltonian(2, [PauliTerm(1.0, "ZZ"), PauliTerm(0.5, "XX")])
exact = float(np.min(np.linalg.eigvalsh(H.matrix())))
print("exact ground energy:", exact)

vqe = VQE(H, ansatz=HWEAnsatz(2, n_layers=1))
result = vqe.run(max_iterations=200)
print("VQE energy:", round(result.optimal_energy, 6))
print("within 1e-2 of exact:", abs(result.optimal_energy - exact) < 1e-2)
```

### 6.2 QAOA

```python
from quantumflow.algorithms.qaoa import MaxCutQAOA

# Path graph 0-1-2-3: the maximum cut has size 3
maxcut = MaxCutQAOA([(0, 1), (1, 2), (2, 3)], n_nodes=4, p=1)
solution = maxcut.solve()
best = solution.best_bitstring
print("best bitstring:", best)
print("cut value:", maxcut.cut_value(best), "(optimal is 3)")
print("partition:", maxcut.get_cut(best))
```

Maximum independent set follows the same pattern:

```python
from quantumflow.algorithms.qaoa import MISQAOA

mis = MISQAOA([(0, 1), (1, 2)], n_nodes=3, p=1)
mis_solution = mis.solve()
print("independent set:", mis.get_independent_set(mis_solution.best_bitstring))
```

---

## 7. Quantum machine learning with Keras 3

The `quantumflow.keras` layers are real Keras 3 layers: they drop into
`Sequential`, train with `model.fit()`, and gradients flow *through* the
quantum circuit evaluation.

```python
import keras
from quantumflow.keras.layers import KerasQDense

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 4)).astype("float32")
y = (X[:, 0] + X[:, 1] > 0).astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation="relu"),
    KerasQDense(units=1, n_qubits=4, n_layers=2),   # the quantum layer
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02),
              loss="binary_crossentropy")
hist = model.fit(X, y, epochs=5, batch_size=8, verbose=0)
print("loss:", [round(v, 3) for v in hist.history["loss"]])
```

High-level models wrap the workflow — compile, fit, predict:

```python
from quantumflow.keras.models import KerasQuantumClassifier

clf = KerasQuantumClassifier(n_qubits=4, n_classes=2, n_layers=2)
clf.compile()
clf.fit(X, y, epochs=3, batch_size=8, verbose=0)
print("predicted labels:", clf.predict_classes(X[:6]))
```

Classical data must be encoded into quantum parameters.
`QuantumDataEncoder` offers five strategies:

```python
from quantumflow.keras.preprocessing import QuantumDataEncoder

encoder = QuantumDataEncoder(n_qubits=4, encoding="angle")
Xq = encoder.encode(X, fit=True)          # normalise + scale to [-pi, pi]
print("encoded shape:", Xq.shape, "| range:", float(Xq.min()), float(Xq.max()))
```

---

## 8. The wider ML toolbox

The `quantumflow.neural` module is framework-agnostic (pure NumPy), and
`quantumflow.tensorflow` provides plain-TF layers with parameter-shift
gradients for custom training loops.

```python
from quantumflow.neural.qnn_layer import QuantumNNLayer

layer = QuantumNNLayer(n_qubits=4, n_layers=2, encoding="angle",
                       rotation_gates="rycz")
out = layer.forward([0.2, -0.4, 0.6, 0.1])     # encode -> variational -> measure
print("expectations:", np.round(out, 4))
```

Quantum convolution and pooling operate on image-like batches:

```python
from quantumflow.neural.quantum_conv import QuantumConv2D, QuantumPool2D

batch = rng.uniform(0, np.pi / 2, size=(2, 8, 8, 1)).astype(np.float64)
conv = QuantumConv2D(filters=2, kernel_size=3, n_qubits=4, n_layers=1)
feat = conv(batch)
pool = QuantumPool2D(pool_size=2, n_qubits=2)
pooled = pool(feat)
print("conv:", feat.shape, "-> pool:", pooled.shape)
```

Quantum activations map classical activations onto expectation values:

```python
from quantumflow.neural.quantum_activation import QuantumReLU, QuantumSigmoid

print("qReLU:", np.round(QuantumReLU(n_qubits=2, n_layers=2).forward([1.0, -1.0]), 4))
print("qSigmoid:", np.round(QuantumSigmoid(n_qubits=2, n_layers=2).forward([0.5, -0.5]), 4))
```

The quantum optimizers combine parameter-shift gradients with classical
update rules — usable standalone:

```python
from quantumflow.tensorflow.optimizers import QuantumAdam, QuantumLAMB

def objective(p):
    return float(np.sum((p - np.array([0.3, -0.7, 0.5])) ** 2))

opt = QuantumAdam(learning_rate=0.05)
best = opt.minimize(objective, np.zeros(3), n_iterations=40, verbose=0)
print("start:", round(objective(np.zeros(3)), 3),
      "-> optimum:", round(objective(best), 4))

lamb = QuantumLAMB(learning_rate=0.05)          # trust-ratio optimizer
best2 = lamb.minimize(objective, np.zeros(3), n_iterations=40, verbose=0)
print("LAMB:", round(objective(best2), 4))
```

Plain-TF layers (not Keras layers — use them directly or inside a
custom `tf.keras.Model`; for `Sequential` use the
`quantumflow.keras` equivalents):

```python
import tensorflow as tf
from quantumflow.tensorflow.layers import QDenseLayer

qlayer = QDenseLayer(units=2, n_qubits=4, n_layers=2)
y_tf = qlayer(tf.constant(X[:4]))              # TF tensors give TF gradients
print("QDenseLayer output:", y_tf.shape)
```

---

## 9. Visualization

ASCII circuit drawing works everywhere; matplotlib variants for
notebooks.

```python
drawer = qf.CircuitDrawer(bell_circ)
print(drawer.draw_text())
```

Bloch-sphere visualisation of single-qubit states:

```python
import matplotlib

matplotlib.use("Agg")                       # headless-friendly; drop in notebooks

bloch = qf.BlochSphere()
bloch.add_state(qf.Statevector([1, 0]), label="|0>")
bloch.add_state(qf.Statevector([0, 1]), label="|1>")
bloch.show(filename="/tmp/qf_bloch.png")    # also renders inline in notebooks
```

Measurement histograms:

```python
ax = qf.StatevectorSimulator().run(qc, shots=512).plot_histogram(
    title="GHZ outcomes"
)
```

---

## 10. Performance, testing and FAQ

### Performance

* Terminal measurements (all measurements after all gates) are sampled
  from a single execution — statistically identical to per-shot runs and
  orders of magnitude faster on wide registers.
* Mid-circuit measurements re-execute per shot for correct statistics.
* Optional Cython kernels (`pip install .` with a C compiler + Cython)
  accelerate gate application and sampling; pure-Python fallbacks are
  always available and the build never fails without a compiler.

### Testing

The suite runs with or without TensorFlow (ML tests auto-skip):

```python
# Run outside this tutorial:
#   pytest tests/ -q                 # everything TF has to offer included
#   ruff check quantumflow/ tests/   # lint gate
```

### FAQ / gotchas

**Why does my measurement string read `q0 q1 q2` left to right?**
Qubit 0 is the most significant bit throughout the library.

**My seeded run differs between runs?** Pass the seed to
`BackendConfig(seed=...)`, not the simulator constructor.

**`TypeError` when putting `QDenseLayer` in `keras.Sequential`?**
Plain-TF layers are not Keras layers. Use `quantumflow.keras.KerasQDense`
inside Keras models; use `quantumflow.tensorflow.*` layers directly or
in subclassed models.

**Noisy circuit on the statevector simulator raises?** Channels are
non-unitary — noise requires `DensityMatrixSimulator` (Chapter 4).

**VQE/QAOA results vary run-to-run?** The classical optimiser starts
from random parameters. Pass `initial_params=` to `VQE` for
reproducibility.

---

*Continue with `docs/tutorials/advanced-tutorials.md` for the
MNIST/CIFAR-scale hybrid models, and with `docs/api-reference.md` for
the full API surface.*
