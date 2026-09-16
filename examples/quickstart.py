"""QuantumFlow quickstart — every snippet below runs as-is.

Run with:  python examples/quickstart.py
Requires:  pip install .            (core, pure Python)
           pip install ".[tf]"      (only for the Keras section)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logs

import numpy as np

import quantumflow as qf

print("=" * 60)
print("1. Build a circuit and simulate it")
print("=" * 60)

qc = qf.QuantumCircuit(3)          # 3 qubits, qubit 0 = most significant
qc.h(0)                            # Hadamard
qc.cx(0, 1)                        # CNOT (control, target)
qc.cx(1, 2)                        # -> GHZ state
qc.measure([0, 1, 2], [0, 1, 2])   # measure qubits into classical bits

result = qf.StatevectorSimulator().run(qc, shots=1024)
print("GHZ counts:", result.get_counts())

print()
print("=" * 60)
print("2. Statevectors, expectation values, seeded shots")
print("=" * 60)

bell = qf.QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)

sim = qf.StatevectorSimulator()
sv = sim.run(bell, shots=0).statevector       # shots=0 -> exact state
print("Bell amplitudes:", np.round(sv.data, 3))

Z = np.diag([1, -1])
print("<Z on qubit 0>:", sim.expectation(bell, np.kron(Z, np.eye(2))))

seeded = qf.StatevectorSimulator(config=qf.BackendConfig(seed=42))
fair = qf.QuantumCircuit(1)
fair.h(0)
fair.measure([0], [0])
print("seeded coin flips:", seeded.run(fair, shots=100).get_counts())

# Density-matrix and MPS simulators share the same interface
dm = qf.DensityMatrixSimulator().run(bell, shots=0).density_matrix
print("density matrix trace:", np.trace(dm.data).real)
mps_counts = qf.MPSimulator().run(qc, shots=256).get_counts()
print("MPS GHZ counts:", mps_counts)

print()
print("=" * 60)
print("3. Algorithms")
print("=" * 60)

# Grover search
grover = qf.GroverSearch(n_qubits=4, marked_states=["1010"])
print("Grover found:", grover.run(shots=256)["most_frequent"])

# Quantum phase estimation: U|1> = e^{2*pi*i*0.25}|1>
phi = 0.25
U = np.diag([1, np.exp(1j * 2 * np.pi * phi)])
qpe = qf.PhaseEstimation(U, n_evaluation_qubits=4, n_state_qubits=1)
print("QPE phase:", qpe.run(shots=4096)["phase"])

# Shor factoring
shor = qf.ShorAlgorithm(N=15, a=7)
print("Shor 15 =", shor.factor(shots=1024)["factors"])

# VQE ground state of H = ZZ + 0.5 XX   (exact minimum: -1.5)
from quantumflow.algorithms.vqe import HWEAnsatz, Hamiltonian, PauliTerm, VQE

H = Hamiltonian(2, [PauliTerm(1.0, "ZZ"), PauliTerm(0.5, "XX")])
vqe = VQE(H, ansatz=HWEAnsatz(2, n_layers=1))
print("VQE energy:", round(vqe.run(max_iterations=200).optimal_energy, 6))

# MaxCut QAOA (path graph P4, optimal cut = 3)
from quantumflow.algorithms.qaoa import MaxCutQAOA

maxcut = MaxCutQAOA([(0, 1), (1, 2), (2, 3)], n_nodes=4, p=1)
cut_res = maxcut.solve()
print("QAOA best:", cut_res.best_bitstring, "cut =", maxcut.cut_value(cut_res.best_bitstring))

print()
print("=" * 60)
print("4. Quantum machine learning (Keras 3)")
print("=" * 60)

rng = np.random.default_rng(0)
X = rng.normal(size=(16, 2)).astype("float32")
y = (X[:, 0] > 0).astype("float32")

# A quantum layer inside a plain Keras model — fit() just works
import keras
from quantumflow.keras.layers import KerasQDense

model = keras.Sequential(
    [keras.Input(shape=(2,)), KerasQDense(units=1, n_qubits=2, n_layers=1)]
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=3, batch_size=4, verbose=0)
print("KerasQDense predictions:", model.predict(X[:3], verbose=0).ravel().round(3))

# High-level quantum classifier
from quantumflow.keras.models import KerasQuantumClassifier

clf = KerasQuantumClassifier(n_qubits=2, n_classes=2, n_layers=1)
clf.compile()
clf.fit(X, y, epochs=3, batch_size=4, verbose=0)
print("classifier labels:", clf.predict_classes(X[:5]))

# Classical data -> quantum features
from quantumflow.keras.preprocessing import QuantumDataEncoder

encoder = QuantumDataEncoder(n_qubits=2, encoding="angle")
Xq = encoder.encode(X, fit=True)
print("encoded features shape:", Xq.shape, "range: [-pi, pi]")

print()
print("=" * 60)
print("5. Visualization")
print("=" * 60)

print(qf.CircuitDrawer(bell).draw_text())
