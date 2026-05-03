# QuantumFlow Tutorials

Advanced tutorials for specific use cases.

## Tutorial 1: Quantum Convolutional Neural Network

This tutorial demonstrates building a Quantum CNN for image classification.

```python
import tensorflow as tf
import quantumflow.tensorflow as qf_tf
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Binary classification: T-shirt/top (0) vs Trouser (1)
train_mask = np.isin(y_train, [0, 1])
test_mask = np.isin(y_test, [0, 1])
x_train, y_train = x_train[train_mask] / 255.0, (y_train[train_mask] == 1).astype(float)
x_test, y_test = x_test[test_mask] / 255.0, (y_test[test_mask] == 1).astype(float)

# Reshape and downscale
x_train = x_train[..., np.newaxis]
x_train = tf.image.resize(x_train, [8, 8]).numpy()
x_test = x_test[..., np.newaxis]
x_test = tf.image.resize(x_test, [8, 8]).numpy()

# Build Quantum CNN
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(8, 8, 1)),

    # Classical convolutional layers
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),

    # Quantum layers
    qf_tf.QDenseLayer(16, n_qubits=4, n_layers=3, activation='quantum_relu'),
    tf.keras.layers.Dropout(0.3),

    # Classical output
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(x_train, y_train, epochs=15, batch_size=32,
                    validation_split=0.2, verbose=1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")
```

---

## Tutorial 2: Molecular Simulation with VQE

Simulate the Lithium Hydride (LiH) molecule.

```python
import numpy as np
from quantumflow.algorithms.vqe import VQE, Hamiltonian, HWEAnsatz, UCCSDAnsatz

# LiH Hamiltonian (simplified, 4 qubits after active space reduction)
lih_terms = [
    (-3.9782, "IIII"),
    (0.3907, "IIIZ"),
    (0.3907, "IIZI"),
    (-0.4798, "IZII"),
    (0.0140, "IIZZ"),
    (0.0140, "IZIZ"),
    (-0.5225, "ZIII"),
    (0.0927, "ZIIZ"),
    (0.0927, "ZZII"),
    (0.1809, "IZZZ"),
    (0.1809, "ZZIZ"),
    (-0.2257, "IIIX"),
    (-0.2257, "IIXI"),
    (0.0141, "IIXZ"),
    (-0.0141, "IIZX"),
    (0.0141, "IXIZ"),
    (-0.0141, "IZIX"),
    (-0.2257, "IXII"),
    (-0.2257, "XIII"),
    (-0.2257, "XIIY"),
    (-0.2257, "YIIY"),
    (-0.2257, "XIZY"),
    (0.2257, "YIZX"),
    (-0.2257, "XIYZ"),
    (0.2257, "YIXZ"),
    (-0.2257, "IXZY"),
    (0.2257, "YXZI"),
    (0.0141, "XXYY"),
    (0.0141, "XYYX"),
    (-0.0141, "XYXY"),
    (-0.0141, "YXXY"),
    (-0.0141, "YYYY"),
    (0.0141, "YYXX"),
]

H = Hamiltonian.from_terms(lih_terms)

# Exact energy
eigenvalues = np.linalg.eigvalsh(H.matrix())
print(f"Exact ground state energy: {eigenvalues[0]:.6f} Hartree")

# Try multiple ansätze and optimizers
best_energy = float('inf')

for name, ansatz in [
    ("HWE (3 layers)", HWEAnsatz(4, n_layers=3)),
    ("HWE (5 layers)", HWEAnsatz(4, n_layers=5)),
    ("HWE full entangle", HWEAnsatz(4, n_layers=3, entanglement='full')),
]:
    for opt in ['COBYLA', 'L-BFGS-B']:
        vqe = VQE(H, ansatz, optimizer=opt)
        result = vqe.run(max_iterations=300)
        energy = result.optimal_energy
        if energy < best_energy:
            best_energy = energy
        print(f"  {name} + {opt}: E = {energy:.6f} Hartree (error: {abs(energy - eigenvalues[0]):.6f})")

print(f"\nBest VQE energy: {best_energy:.6f} Hartree")
print(f"Exact energy: {eigenvalues[0]:.6f} Hartree")
print(f"Chemical accuracy: {'YES' if abs(best_energy - eigenvalues[0]) < 0.0016 else 'NO'}")
```

---

## Tutorial 3: Hybrid Quantum-Classical Autoencoder

```python
import tensorflow as tf
import quantumflow.tensorflow as qf_tf
import numpy as np

# Generate data (2D circles)
n_samples = 1000
theta = np.random.uniform(0, 2 * np.pi, n_samples)
r = np.random.uniform(0.8, 1.2, n_samples)
X = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float32)

# Build quantum autoencoder
class QuantumAutoencoder(tf.keras.Model):
    def __init__(self, n_qubits=4, n_latent=2, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(n_latent),
        ])
        self.quantum = qf_tf.QDenseLayer(n_latent, n_qubits=n_qubits, n_layers=n_layers)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2),
        ])

    def call(self, x):
        z = self.encoder(x)
        q = self.quantum(z)
        return self.decoder(q)

model = QuantumAutoencoder(n_qubits=4, n_latent=2, n_layers=2)
model.compile(optimizer='adam', loss='mse')
model.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

# Encode and decode
encoded = model.encoder(X[:10])
decoded = model(X[:10])
print(f"Input shape: {X[:10].shape}")
print(f"Latent shape: {encoded.shape}")
print(f"Reconstruction error: {np.mean((X[:10] - decoded.numpy())**2):.6f}")
```

---

## Tutorial 4: QAOA for Portfolio Optimization

```python
import numpy as np
from quantumflow.algorithms.qaoa import QAOA
from quantumflow.noise.error_channels import DepolarizingChannel

# Portfolio optimization as MaxCut
# Nodes = stocks, edges = correlation between stocks
# We want to find the most uncorrelated subset

# Correlation matrix
returns = np.random.randn(5, 100)  # 5 stocks, 100 days
corr = np.corrcoef(returns)

# Build graph: edges for highly correlated stocks (threshold > 0.5)
threshold = 0.5
edges = []
for i in range(5):
    for j in range(i + 1, 5):
        if abs(corr[i, j]) > threshold:
            edges.append((i, j))

print(f"Portfolio optimization with {len(edges)} correlations")

# Build QAOA cost Hamiltonian
from quantumflow.algorithms.qaoa import CostHamiltonian
n_stocks = 5
terms = []
for (i, j) in edges:
    pauli = ['I'] * n_stocks
    pauli[i] = 'Z'
    pauli[j] = 'Z'
    terms.append((-0.5 * abs(corr[i, j]), ''.join(pauli)))

cost = CostHamiltonian(n_stocks, terms)
qaoa = QAOA(cost, p=3, mixer='x')

result = qaoa.run(optimizer='COBYLA', max_iterations=200)
print(f"Optimal cost: {result.optimal_cost:.4f}")
print(f"Best allocation: {result.best_bitstring}")
print(f"Approximation ratio: {result.approximation_ratio:.4f}")

# Decode portfolio
selected = [i for i, b in enumerate(result.best_bitstring) if b == '1']
not_selected = [i for i, b in enumerate(result.best_bitstring) if b == '0']
print(f"Selected stocks: {selected}")
print(f"Excluded stocks: {not_selected}")
```

---

## Tutorial 5: Quantum Error Mitigation

```python
import numpy as np
from quantumflow import QuantumCircuit, StatevectorSimulator
from quantumflow.noise.noise_model import NoiseModel, NoiseConfig
from quantumflow.noise.error_mitigation import (
    ZeroNoiseExtrapolation, MeasurementErrorMitigation, VirtualDistillation,
)

# Create a simple circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Ideal simulation
sim = StatevectorSimulator()
ideal_result = sim.run(qc, shots=10000)

# Add noise
config = NoiseConfig(single_gate_error=0.02, two_gate_error=0.05)
noise = NoiseModel(config)
noisy_qc = noise.apply_noise(qc)
noisy_result = sim.run(noisy_qc, shots=10000)

# === Zero Noise Extrapolation ===
print("=== Zero Noise Extrapolation ===")
zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], method='richardson')

# Run at different noise levels
noisy_values = []
for scale in [1.0, 2.0, 3.0]:
    noisy = noise.apply_noise(qc, noise_scale=scale)
    r = sim.run(noisy, shots=5000)
    counts = r.get_counts()
    # Bell state fidelity proxy: P(|00>) + P(|11>)
    fidelity_proxy = (counts.get('00', 0) + counts.get('11', 0)) / sum(counts.values())
    noisy_values.append(fidelity_proxy)
    print(f"  Scale {scale:.1f}: fidelity_proxy = {fidelity_proxy:.4f}")

zne_result = zne.mitigate(None, noisy_expectations=noisy_values)
print(f"  Extrapolated (zero noise): {zne_result['mitigated_value']:.4f}")

# === Measurement Error Mitigation ===
print("\n=== Measurement Error Mitigation ===")
mem = MeasurementErrorMitigation(n_qubits=2)
cm = mem.create_confusion_matrix(2, {0: 0.03, 1: 0.02})
mem.calibrate(confusion_matrix=cm)
mitigated = mem.mitigate(noisy_result.get_counts())
print(f"  Noisy counts: {noisy_result.get_counts()}")
print(f"  Mitigated counts: {mitigated['mitigated_counts']}")
print(f"  Improvement: {mitigated['improvement']:.4f}")

# === Virtual Distillation ===
print("\n=== Virtual Distillation ===")
rho = np.outer(ideal_result.statevector, ideal_result.statevector.conj())
# Add some noise to the state
noise_ch = DepolarizingChannel(0.1)
rho_noisy = noise_ch.apply(rho)

vd = VirtualDistillation(power=2)
vd_result = vd.mitigate(rho=rho_noisy)
print(f"  Noisy purity: {np.trace(rho_noisy @ rho_noisy).real:.4f}")
print(f"  Distilled purity: {vd_result['purity']:.4f}")
```

---

## Tutorial 6: Transfer Learning with Quantum Layers

```python
import tensorflow as tf
import quantumflow.keras as qf_keras
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[:5000] / 255.0  # Subset for demo
y_train = y_train[:5000].flatten()
x_test = x_test[:1000] / 255.0
y_test = y_test[:1000].flatten()

# Load pre-trained backbone
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg',
)

# Freeze backbone
backbone.trainable = False

# Replace head with quantum layers
transfer = qf_keras.KerasTransferLearning(
    backbone=backbone,
    n_qubits=4,
    n_layers=2,
    n_classes=10,
    fine_tune_strategy='quantum_only',
)

transfer.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

transfer.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Fine-tune all layers
transfer.set_fine_tuning(strategy='full', lr=1e-5)
transfer.fit(x_train, y_train, epochs=5, batch_size=32)

loss, acc = transfer.evaluate(x_test, y_test)
print(f"Transfer learning accuracy: {acc:.4f}")
```
