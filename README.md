# QuantumFlow

**Advanced Quantum Computing Framework with TensorFlow/Keras Integration for Quantum Neural Networks**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](pyproject.toml)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Guide](#module-guide)
  - [Core Quantum Computing](#1-core-quantum-computing)
  - [Simulation Engine](#2-simulation-engine)
  - [Quantum Neural Networks](#3-quantum-neural-networks)
  - [TensorFlow Integration](#4-tensorflow-integration)
  - [Keras Integration](#5-keras-integration)
  - [Quantum Algorithms](#6-quantum-algorithms)
  - [Noise & Error Mitigation](#7-noise--error-mitigation)
  - [Visualization](#8-visualization)
  - [Utilities](#9-utilities)
- [Tutorials](#tutorials)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

QuantumFlow is a comprehensive, production-ready Python framework for quantum computing that seamlessly integrates with **TensorFlow** and **Keras** to enable **Quantum Machine Learning (QML)** and **Quantum Neural Networks (QNNs)**. Inspired by Qiskit but purpose-built for deep learning integration, QuantumFlow provides everything you need to design, simulate, and optimize quantum circuits and hybrid quantum-classical algorithms.

The framework is structured around 10 tightly integrated modules, totaling over **36,000 lines** of production-quality Python code with comprehensive type hints, docstrings, and examples.

### Why QuantumFlow?

| Feature | QuantumFlow | Qiskit | PennyLane | Cirq |
|---------|------------|--------|-----------|------|
| Native TensorFlow layers | Yes | Plugin | Yes | Limited |
| Native Keras 3 layers | Yes | No | Partial | No |
| Quantum Conv2D | Yes | No | No | No |
| Quantum GAN/VAE | Yes | Plugin | Yes | No |
| VQE with UCCSD | Yes | Yes | Yes | Yes |
| QAOA (MaxCut, MIS, TSP) | Yes | Yes | Yes | Yes |
| MPS Simulator | Yes | Yes | Limited | No |
| 7 Error channels + 5 mitigation | Yes | Partial | Limited | Partial |
| Custom quantum optimizers | Yes (8) | No | Yes (3) | No |

---

## Key Features

### Quantum Computing Core
- **50+ quantum gates**: Pauli (X, Y, Z), Clifford (H, S, T), rotation (RX, RY, RZ, U3), controlled (CNOT, CZ, Toffoli, Fredkin), multi-qubit (MCX, MCZ), parameterized gates
- **QuantumCircuit**: Full circuit construction with compose, tensor, inverse, QASM export, parameter binding
- **State management**: Statevector and DensityMatrix with measurement, sampling, partial trace, fidelity

### Simulation
- **StatevectorSimulator**: Ideal statevector simulation with einsum-based gate application
- **DensityMatrixSimulator**: Mixed-state simulation with Kraus operator evolution
- **MPSimulator**: Matrix Product State for efficient shallow-circuit simulation (configurable bond dimension)
- **Batch simulation**: Run circuits on multiple initial states simultaneously
- **Parameter gradients**: Parameter-shift rule and finite-difference gradient computation

### Quantum Neural Networks
- **6 QNN layers**: QuantumDense, QuantumConv2D, QuantumNNLayer, VariationalLayer, QuantumActivation, QuantumPool2D
- **5 data encodings**: Angle, amplitude, basis, IQP, dense-angle encoding
- **5 variational forms**: Hardware-efficient, strongly entangling, Circuit-19, barren-plateau-free, QAOA
- **5 quantum activations**: ReLU, Sigmoid, Tanh, Softmax, Swish approximations

### TensorFlow Integration
- **8 TF quantum layers**: QDenseLayer, QConvLayer, QVariationalLayer, QBatchNormLayer, QAttentionLayer, QResidualLayer, QFeatureMapLayer, QMeasurementLayer
- **6 pre-built models**: QClassifier, QRegressor, QAutoencoder, QGAN, QTransferLearningModel, QHybridModel
- **8 quantum optimizers**: ParameterShiftOptimizer, NaturalGradientOptimizer, QuantumAdam, QuantumLAMB, QuantumSGD, SpsaOptimizer

### Keras Integration
- **10 Keras layers**: Fully Keras 3 compatible with `build()`, `call()`, `get_config()` methods
- **8 Keras models**: Including QGAN with `train_step()`, QVAE with KL divergence, transfer learning
- **4 preprocessing utilities**: QuantumDataEncoder, QuantumDataAugmenter, QuantumNormalizer, QuantumFeatureScaler

### Algorithms
- **Grover's Search**: Oracle construction, diffusion operator, optimal iteration count, fixed-point variant
- **Shor's Factoring**: Order finding, modular exponentiation, continued fractions
- **QFT**: Exact and approximate, quantum adder and multiplier
- **Phase Estimation**: Standard, iterative, and Bayesian approaches
- **VQE**: COBYLA, SPSA, L-BFGS-B, Adam optimizers; HWE and UCCSD ansatze; H2 molecule Hamiltonian
- **QAOA**: MaxCut, Maximum Independent Set, Traveling Salesman Problem

### Noise & Error Mitigation
- **7 error channels**: Depolarizing, amplitude damping, phase damping, bit flip, phase flip, Pauli, thermal relaxation
- **NoiseModel**: Per-gate and per-qubit noise configuration
- **5 mitigation techniques**: Zero Noise Extrapolation (Richardson/exponential/linear), Probabilistic Error Cancellation, Measurement Error Mitigation, Virtual Distillation, Symmetry Verification

### Visualization
- **Circuit drawing**: ASCII art, matplotlib, and LaTeX (Qcircuit) output
- **Bloch sphere**: Interactive 3D visualization with state labels and trails
- **Histograms**: Built into SimulationResult for measurement outcome plotting

---

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip (Python package manager)

### Install from Source

```bash
# Clone or download the package
cd quantumflow/

# Install with ALL dependencies (core + TensorFlow/Keras)
pip install .

# Install in development mode (recommended when hacking on the source)
pip install -e .

# Optional extras
pip install -e ".[gpu]"    # GPU support (requires CUDA)
pip install -e ".[dev]"    # development tools (pytest, ruff, mypy, ...)
```

TensorFlow and Keras (>= 3) are installed **automatically** on Python
3.9–3.12 — no extra needed. On Python >= 3.13 TensorFlow wheels are not
published yet, so QuantumFlow installs with the pure-Python stack and
the quantum-ML modules raise a clear "install tensorflow" message when
used.

### Install Dependencies Separately

Only needed if you want to manage the stack by hand:

```bash
# Core dependencies
pip install numpy scipy matplotlib networkx sympy tqdm joblib opt-einsum

# Quantum-ML stack (installed automatically by pip install quantumflow)
pip install tensorflow keras

# Development tools
pip install pytest pytest-cov mypy ruff black
```

### Verify Installation

```python
import quantumflow as qf
print(f"QuantumFlow v{qf.__version__}")

# Quick test
qc = qf.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Bell circuit created successfully!")
```

---

## Quick Start

### 1. Create Your First Quantum Circuit

```python
import quantumflow as qf

# Create a 3-qubit circuit
qc = qf.QuantumCircuit(3)

# Build a GHZ state: (|000> + |111>) / sqrt(2)
qc.h(0)          # Hadamard on qubit 0
qc.cx(0, 1)      # CNOT: 0 -> 1
qc.cx(1, 2)      # CNOT: 1 -> 2

# Simulate
sim = qf.StatevectorSimulator()
result = sim.run(qc, shots=1024)

print("Measurement counts:", result.get_counts())
# Expected: {'000': ~512, '111': ~512}
```

### 2. Build a Quantum Neural Network with TensorFlow

```python
import tensorflow as tf
import numpy as np
from quantumflow.keras.layers import KerasQDense

# Generate synthetic data (quantum layers are simulator-backed:
# keep demos small — a few hundred samples train in ~a minute)
X = np.random.randn(200, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Build a hybrid quantum-classical model (Keras 3)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    KerasQDense(units=8, n_qubits=4, n_layers=2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=2, batch_size=32, validation_split=0.2, verbose=0)
print("training complete")
```

### 3. Solve a Quantum Chemistry Problem with VQE

```python
from quantumflow.algorithms.vqe import VQE, Hamiltonian, HWEAnsatz

# Define Hamiltonian (transverse-field Ising model)
H = Hamiltonian.transverse_field_ising(n_qubits=4, j=1.0, h=-1.0)

# Set up variational ansatz
ansatz = HWEAnsatz(n_qubits=4, n_layers=3, rotation_set=['ry', 'rz'])

# Run VQE
vqe = VQE(H, ansatz, optimizer='COBYLA')
result = vqe.run(max_iterations=200)

print(f"Ground state energy: {result.optimal_energy:.6f}")
```

### 4. Use QAOA for Combinatorial Optimization

```python
from quantumflow.algorithms.qaoa import MaxCutQAOA

# Define a graph (triangle)
edges = [(0, 1), (1, 2), (2, 0)]

# Solve MaxCut
maxcut = MaxCutQAOA(edges, n_nodes=3, p=3)
result = maxcut.solve(optimizer='COBYLA')

partition = maxcut.get_cut(result.best_bitstring)
print(f"Max cut value: {result.optimal_cost}")
print(f"Partition: {partition}")
```

---

## Module Guide

### 1. Core Quantum Computing

**Package**: `quantumflow.core`

The core module provides the fundamental building blocks: gates, circuits, states, and registers.

#### Gates (50+)

```python
import numpy as np
from quantumflow.core.gate import (
    HGate, XGate, RXGate, U3Gate, CNOTGate, SwapGate, ToffoliGate, UnitaryGate,
)

# Single-qubit gates (parameterized gates take a sequence of parameters)
h = HGate()                     # Hadamard
x = XGate()                     # Pauli X (NOT)
rx = RXGate([0.5])              # Rotation about X by 0.5
u3 = U3Gate([0.1, 0.2, 0.3])    # General single-qubit unitary

# Multi-qubit gates
cnot = CNOTGate()               # Controlled-NOT
swap = SwapGate()               # SWAP
toffoli = ToffoliGate()         # Controlled-controlled-NOT

# Custom unitary
my_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
custom = UnitaryGate(my_matrix, name="MyGate")

# Gate operations
matrix = h.matrix               # 2x2 unitary as a numpy array
adjoint = h.matrix.conj().T     # Adjoint (inverse)
ctrl = h.controlled(2)          # Double-controlled H
print("RX(0.5):", np.round(rx.matrix, 4))
```

#### Quantum Circuit

```python
import numpy as np
from quantumflow import QuantumCircuit

qc = QuantumCircuit(4)

# Gate operations
qc.h(0)
qc.x(1)
qc.y(2)
qc.z(3)
qc.rx(np.pi / 4, 0)
qc.ry(np.pi / 3, 1)
qc.rz(np.pi / 6, 2)

# Multi-qubit gates
qc.cx(0, 1)       # CNOT
qc.cz(1, 2)       # Controlled-Z
qc.swap(2, 3)     # SWAP
qc.ccx(0, 1, 2)   # Toffoli

# Properties (width is a property; the rest are methods)
print(f"Depth: {qc.depth()}")
print(f"Width: {qc.width}")
print(f"Size: {qc.size()}")
print(f"Gate counts: {qc.count_gates()}")

# Advanced operations (before measurement: measurements are irreversible)
inv_qc = qc.inverse()
combined = qc.compose(inv_qc)
unitary = qc.to_unitary()
qasm = qc.qasm()  # OpenQASM 2.0
print(qasm.splitlines()[0])

# Circuit operations
qc.barrier()
qc.measure(0, 0)  # Measure qubit 0 to classical bit 0
```

#### Quantum States

```python
import numpy as np
from quantumflow import Statevector, DensityMatrix

# Create states
sv = Statevector.from_label("00+")                     # |00+>
sv = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))  # (|00> + |11>)/sqrt(2)

# Operations
probs = sv.probabilities()
outcome, new_sv = sv.measure()   # Single measurement -> (outcome, post-state)
counts = sv.sample(shots=1000)   # Multiple shots

# Expectation value of an observable (Z tensor Z)
Z = np.diag([1, -1])
z_obs = np.kron(Z, Z)
expect = sv.expectation(z_obs)
print(f"<ZZ> = {expect:.4f}")

# Density matrices
dm = DensityMatrix.from_statevector(sv)
print(f"Purity: {dm.purity()}")
print(f"Entropy: {dm.von_neumann_entropy()}")
print(f"Is pure: {dm.is_pure()}")

# Partial trace
reduced = dm.partial_trace(qubits_to_keep=[0])
print("reduced purity:", reduced.purity())
```

### 2. Simulation Engine

**Package**: `quantumflow.simulation`

Three simulators for different needs and scale.

#### Statevector Simulator

```python
from quantumflow import QuantumCircuit, StatevectorSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sim = StatevectorSimulator()
result = sim.run(qc, shots=4096)

# Results
print("Counts:", result.get_counts())
print("Probabilities:", result.get_probabilities())
print("Statevector:", result.statevector)
result.plot_histogram()
```

#### Density Matrix Simulator (with noise)

```python
from quantumflow import QuantumCircuit, DensityMatrixSimulator
from quantumflow.noise.noise_model import NoiseConfig, NoiseModel

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

config = NoiseConfig(single_gate_error=0.01, two_gate_error=0.05)
sim = DensityMatrixSimulator(noise_model=NoiseModel(config))
result = sim.run(qc, shots=4096)
print("noisy counts:", result.get_counts())
```

#### MPS Simulator (large systems)

```python
from quantumflow.simulation.simulator import MPSimulator, BackendConfig

config = BackendConfig(max_qubits=30, optimization_level=2)
sim = MPSimulator(config=config)

result = sim.run(qc, shots=1024)
```

### 3. Quantum Neural Networks

**Package**: `quantumflow.neural`

Build quantum-enhanced neural network components.

#### Quantum Dense Layer

```python
import numpy as np
from quantumflow.neural.quantum_dense import QuantumDense

# Quantum dense layer — a plain layer with parameter-shift gradients.
# (For tf.keras.Sequential models use quantumflow.keras.KerasQDense instead.)
q_dense = QuantumDense(
    10,                      # output_dim
    n_qubits=5,
    n_layers=3,
    activation='quantum_relu',
)

# Forward pass on a batch of 5-dimensional inputs
batch = np.random.randn(4, 5).astype(np.float32)
out = q_dense(batch)
print("QuantumDense output shape:", out.shape)
```

#### Variational Circuit

```python
from quantumflow.neural.variational_circuit import VariationalCircuit

vc = VariationalCircuit(
    n_qubits=4,
    n_layers=3,
    entanglement='full',
    rotations='rycz',
)

# Forward pass with data
output = vc.forward(x=np.array([0.1, 0.2, 0.3, 0.4]))

# Access parameters
print(f"Parameters: {vc.parameters.shape}")
print(f"Circuit:\n{vc.circuit()}")
```

#### Quantum Activation Functions

```python
from quantumflow.neural.quantum_activation import (
    QuantumReLU, QuantumSigmoid, QuantumTanh
)

qrelu = QuantumReLU()
output = qrelu.forward(np.array([-1.0, 0.5, 2.0]))
gradient = qrelu.backward(grad_output=np.ones(3))
```

#### Quantum Convolution

```python
from quantumflow.neural.quantum_conv import QuantumConv2D, QuantumPool2D

q_conv = QuantumConv2D(
    filters=16,
    kernel_size=3,
    n_qubits=4,
    n_layers=2,
)
q_pool = QuantumPool2D(pool_size=2, pool_type='quantum', n_qubits=4)
print("conv/pool layers ready")
```

### 4. TensorFlow Integration

**Package**: `quantumflow.tensorflow`

Native TensorFlow quantum layers with custom gradients.

#### Quantum Layers

```python
import quantumflow.tensorflow as qf_tf

# Quantum dense layer with parameter-shift gradients
q_layer = qf_tf.QDenseLayer(
    units=8,
    n_qubits=4,
    n_layers=3,
)

# Quantum attention layer
q_attn = qf_tf.QAttentionLayer(
    n_qubits=4,
    n_layers=2,
    heads=4,
)

# Quantum feature map
q_fm = qf_tf.QFeatureMapLayer(
    n_qubits=4,
    feature_map='zx',
)

# Quantum residual block
q_res = qf_tf.QResidualLayer(n_qubits=4, n_layers=2)
print("quantum layers ready")
```

#### Pre-built Models

```python
import numpy as np
import quantumflow.tensorflow as qf_tf

# Synthetic binary-classification data
X = np.random.randn(200, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]

# Quantum Classifier
classifier = qf_tf.QClassifier(
    n_qubits=4,
    n_layers=3,
    n_classes=2,
)
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
classifier.fit(X_train, y_train, epochs=2, verbose=0)
predictions = classifier.predict(X_test)
print("predictions:", predictions[:5])

# Quantum Autoencoder
autoencoder = qf_tf.QAutoencoder(
    n_qubits=6,
    n_latent=2,
    n_layers=3,
)

# Quantum GAN
qgan = qf_tf.QGAN(
    n_qubits=4,
    n_layers=3,
    latent_dim=4,
)
print("models ready")
```

#### Quantum Optimizers

```python
from quantumflow.tensorflow.optimizers import (
    ParameterShiftOptimizer, QuantumAdam, SpsaOptimizer
)

# Exact quantum gradients via parameter-shift rule
opt = ParameterShiftOptimizer(learning_rate=0.01)

# Adam adapted for quantum landscapes (barren plateau detection)
opt = QuantumAdam(learning_rate=0.001)

# Gradient-free optimization for noisy hardware
opt = SpsaOptimizer(learning_rate=0.01, perturbation=0.1)
print("optimizers ready")
```

### 5. Keras Integration

**Package**: `quantumflow.keras`

Keras 3-compatible quantum layers with multi-backend support.

```python
import numpy as np
import quantumflow.keras as qf_keras
import keras

# Synthetic data
X_train = np.random.randn(200, 4).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.float32)

# Keras quantum dense
model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    qf_keras.KerasQDense(units=8, n_qubits=4, n_layers=2),
    qf_keras.KerasQBatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=2, verbose=0)

# Pre-built classifier
classifier = qf_keras.KerasQuantumClassifier(
    n_qubits=4,
    n_layers=2,
    n_classes=2,
)
print("keras models ready")
```

### 6. Quantum Algorithms

**Package**: `quantumflow.algorithms`

#### Grover's Search

```python
from quantumflow.algorithms.grover import GroverSearch, AmplitudeAmplification

# Search for |101> in 3-qubit space
grover = GroverSearch(n_qubits=3, marked_states=['101'])
result = grover.run(shots=1024)
print(f"Found: {result['most_frequent']}")
print(f"Success probability: {result['success_probability']:.2%}")
```

#### Shor's Factoring

```python
from quantumflow.algorithms.shor import ShorAlgorithm

shor = ShorAlgorithm(N=15)
result = shor.factor()
print(f"Factors of 15: {result['factors']}")
```

#### Quantum Fourier Transform

```python
from quantumflow.algorithms.qft import QFT, InverseQFT, qft_matrix

# Exact QFT
qft = QFT(n_qubits=8)
circuit = qft.construct_circuit()

# Approximate QFT (fewer gates)
aqft = QFT(n_qubits=8, approximation_degree=4)

# QFT matrix
U = qft_matrix(3)
```

#### Phase Estimation

```python
from quantumflow.algorithms.qpe import PhaseEstimation, IterativePhaseEstimation

# Standard QPE
U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
qpe = PhaseEstimation(U, n_evaluation_qubits=6, n_state_qubits=1)
result = qpe.run()
print(f"Phase: {result['phase']:.6f}")

# Iterative QPE (qubit-efficient)
iqpe = IterativePhaseEstimation(U, n_state_qubits=1, n_iterations=8)
result = iqpe.run()
```

#### Variational Quantum Eigensolver (VQE)

```python
from quantumflow.algorithms.vqe import VQE, Hamiltonian, HWEAnsatz, UCCSDAnsatz

# H2 molecule
H = Hamiltonian.hydrogen_molecule()
ansatz = HWEAnsatz(H.n_qubits, n_layers=3)
vqe = VQE(H, ansatz, optimizer='COBYLA')
result = vqe.run(max_iterations=200)
print(f"H2 ground state: {result.optimal_energy:.6f} Hartree")

# Custom Hamiltonian
H = Hamiltonian.from_terms([
    (1.0, "ZZ"),
    (0.5, "XX"),
    (-0.5, "YY"),
])
```

#### QAOA

```python
from quantumflow.algorithms.qaoa import QAOA, MaxCutQAOA, MISQAOA, TSPQAOA

# MaxCut
maxcut = MaxCutQAOA(edges=[(0,1),(1,2),(2,3),(3,0)], n_nodes=4, p=3)
result = maxcut.solve(optimizer='COBYLA')

# Maximum Independent Set
mis = MISQAOA(edges=[(0,1),(1,2),(2,0)], n_nodes=3, p=2)
result = mis.solve()

# Traveling Salesman Problem
import numpy as np
dist = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
tsp = TSPQAOA(dist, p=2)
result = tsp.solve()
tour = tsp.decode_tour(result.best_bitstring)
```

### 7. Noise & Error Mitigation

**Package**: `quantumflow.noise`

#### Error Channels

```python
from quantumflow.noise.error_channels import (
    DepolarizingChannel, AmplitudeDampingChannel,
    ThermalRelaxationChannel,
)

# Depolarizing noise (p=0.01)
ch = DepolarizingChannel(error_probability=0.01)
kraus = ch.kraus_operators()
print(f"CPTP: {ch.is_cptp()}")

# Amplitude damping (T1 relaxation)
ch = AmplitudeDampingChannel(gamma=0.1)

# Thermal relaxation
ch = ThermalRelaxationChannel(
    t1=100e-6,    # 100 us T1
    t2=50e-6,     # 50 us T2
    gate_time=100e-9,  # 100 ns gate
    temperature=15.0,  # 15 mK
)
```

#### Noise Model

```python
from quantumflow import QuantumCircuit
from quantumflow.noise.noise_model import NoiseModel, NoiseConfig, GateNoise, NoiseType

quantum_circuit = QuantumCircuit(2)
quantum_circuit.h(0)
quantum_circuit.cx(0, 1)

config = NoiseConfig(
    single_gate_error=0.001,
    two_gate_error=0.01,
    measurement_error=0.01,
    noise_type='depolarizing',
)
config.gate_noise['cx'] = GateNoise('cx', NoiseType.DEPOLARIZING, 0.02)

noise = NoiseModel(config)
noisy_circuit = noise.apply_noise(quantum_circuit)
print("noisy ops:", len(noisy_circuit.data), "vs clean:", len(quantum_circuit.data))
```

#### Error Mitigation

```python
import numpy as np
from quantumflow.noise.error_mitigation import (
    ZeroNoiseExtrapolation, MeasurementErrorMitigation, VirtualDistillation,
)

# Zero Noise Extrapolation
zne = ZeroNoiseExtrapolation(
    noise_factors=[1.0, 2.0, 3.0],
    method='linear',  # or 'richardson', 'exponential'
)
counts = {'00': 480, '11': 470, '01': 25, '10': 25}
expectations = [0.95, 0.86, 0.79]   # one observable value per noise factor
result = zne.mitigate(counts, noisy_expectations=expectations)
print("mitigated value:", round(result['mitigated_value'], 4))

# Measurement Error Mitigation
mem = MeasurementErrorMitigation(n_qubits=4)
cm = mem.create_confusion_matrix(4, {0: 0.02, 1: 0.02, 2: 0.01, 3: 0.01})
mem.calibrate(confusion_matrix=cm)
mitigated = mem.mitigate(counts)
print("mitigated counts:", mitigated['mitigated_counts'])

# Virtual Distillation
rho = np.diag([0.9, 0.03, 0.03, 0.04]).astype(complex)
vd = VirtualDistillation(power=2)
print("distilled purity:", round(vd.mitigate(rho=rho)['purity'], 4))
```

### 8. Visualization

**Package**: `quantumflow.visualization`

#### Circuit Drawing

```python
from quantumflow import QuantumCircuit
from quantumflow.visualization import CircuitDrawer

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.rz(np.pi/4, 2)

drawer = CircuitDrawer(qc)

# ASCII art
print(drawer.draw_text())

# Matplotlib figure
fig = drawer.draw_matplotlib(title="My Circuit", filename="circuit.png")

# LaTeX
latex = drawer.draw_latex()
```

#### Bloch Sphere

```python
from quantumflow.visualization import BlochSphere
import numpy as np

bloch = BlochSphere(title="Single-Qubit States")

# Add states
bloch.add_state(np.array([1, 0]), label="|0>")
bloch.add_state(np.array([0, 1]), label="|1>")
bloch.add_state(np.array([1, 1])/np.sqrt(2), label="|+>")
bloch.add_state(np.array([1, 1j])/np.sqrt(2), label="|+i>")

# Show
bloch.show(filename="bloch.png")
```

### 9. Utilities

**Package**: `quantumflow.utils`

```python
import numpy as np
from quantumflow import HGate, XGate, Statevector
from quantumflow.utils.math import (
    kron, partial_trace, fidelity, trace_distance,
    purity, von_neumann_entropy, expectation_value,
    random_unitary, random_density_matrix,
    state_to_bloch, bloch_to_state,
)

# Tensor products
H_X = kron(HGate().matrix, XGate().matrix)
print("H(X) shape:", H_X.shape)

# State comparison (fidelity/trace_distance take density matrices)
state1 = Statevector.from_label("00").data
state2 = Statevector.from_label("11").data
rho1 = np.outer(state1, state1.conj())
rho2 = np.outer(state2, state2.conj())
f = fidelity(rho1, rho1)
d = trace_distance(rho1, rho2)
print(f"fidelity={f:.4f}  trace_distance={d:.4f}")

# Entropy and purity
density_matrix = rho1
p = purity(density_matrix)
s = von_neumann_entropy(density_matrix)

# Partial trace (keep qubit 0 of a 2-qubit state)
reduced = partial_trace(rho1, [0], n_qubits=2)
print("reduced shape:", reduced.shape)

# Random objects
u = random_unitary(3)
rho_r = random_density_matrix(2)

# Bloch vector round-trip
bloch = state_to_bloch(Statevector.from_label("0").data)
back = bloch_to_state(bloch)
print("bloch vector:", np.round(bloch, 4))
```

---

## Tutorials

> **New:** a complete, verified walkthrough of the whole framework —
> from qubits to QAOA and quantum machine learning — lives in
> [`docs/tutorials/complete-tutorial.md`](docs/tutorials/complete-tutorial.md).
> More advanced material (custom layers, error mitigation in practice,
> transfer learning) is covered in
> [`docs/tutorials/advanced-tutorials.md`](docs/tutorials/advanced-tutorials.md).

### Tutorial 1: Quantum-Classical Hybrid MNIST Classifier

```python
import tensorflow as tf
import numpy as np
from quantumflow.keras.layers import KerasQDense

# Synthetic 16-feature binary dataset (stand-in for downsampled images)
n = 200
imgs = np.random.rand(n, 4, 4).astype(np.float32)      # "4x4 images"
X = imgs.reshape(-1, 16)                                # flatten to 16 features
y = (imgs.mean(axis=(1, 2)) > 0.5).astype(np.float32)   # bright vs dark

# Build hybrid model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    KerasQDense(units=8, n_qubits=4, n_layers=2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=2, batch_size=32, validation_split=0.2, verbose=0)
print("hybrid model trained")
```

### Tutorial 2: Quantum Chemistry - Water Molecule with VQE

```python
from quantumflow.algorithms.vqe import VQE, Hamiltonian, UCCSDAnsatz

# Define molecular Hamiltonian (simplified)
H = Hamiltonian(4)
H.add_term(-0.8105, "IIII")
H.add_term(0.1721, "IIIZ")
H.add_term(-0.2257, "IIZI")
# ... (full molecular Hamiltonian terms)

# UCCSD ansatz (gold standard for chemistry)
ansatz = UCCSDAnsatz(n_qubits=4, n_electrons=2)

# Run VQE with COBYLA optimizer
vqe = VQE(H, ansatz, optimizer='COBYLA')
result = vqe.run(max_iterations=500, convergence_threshold=1e-8)

print(f"Ground state energy: {result.optimal_energy:.8f} Hartree")
print(f"Converged in {result.iteration_count} iterations")
```

### Tutorial 3: Quantum Generative Adversarial Network

```python
import numpy as np
import quantumflow.tensorflow as qf_tf

# 4-D Gaussian data (matches n_qubits=4).  A small sample keeps this
# demo fast — parameter-shift gradients make QGAN training expensive.
data = np.random.randn(200, 4).astype(np.float64)

# Build QGAN
qgan = qf_tf.QGAN(
    n_qubits=4,
    n_layers=3,
    latent_dim=4,
)
qgan.compile()

# Train
qgan.fit(data, epochs=1, batch_size=50, verbose=0)

# Generate samples
generated = qgan.generate(n_samples=100)
print("generated shape:", generated.shape)
```

---

## API Reference

### Core Module (`quantumflow.core`)

| Class | Description |
|-------|-------------|
| `QuantumCircuit` | Main circuit class with gate operations, compose, tensor, inverse, QASM export |
| `Gate` | Base class for all quantum gates |
| `UnitaryGate` | Custom gate from arbitrary unitary matrix |
| `ControlledGate` | N-controlled version of any gate |
| `ParameterizedGate` | Gate with trainable parameters |
| `QuantumRegister` | Container for qubits |
| `ClassicalRegister` | Container for classical bits |
| `Statevector` | Pure quantum state (2^n complex vector) |
| `DensityMatrix` | Mixed quantum state (2^n x 2^n matrix) |
| `Operator` | General linear operator |
| `Observable` | Hermitian operator for measurements |
| `Measurement` | Measurement operation |

### Simulation Module (`quantumflow.simulation`)

| Class | Description |
|-------|-------------|
| `StatevectorSimulator` | Ideal pure-state simulator |
| `DensityMatrixSimulator` | Mixed-state simulator with noise support |
| `MPSimulator` | Matrix Product State simulator for large systems |
| `SimulationResult` | Container for simulation results |
| `BackendConfig` | Simulator configuration |
| `SimulatorFactory` | Factory for creating simulators |

### Neural Module (`quantumflow.neural`)

| Class | Description |
|-------|-------------|
| `QuantumNNLayer` | General QNN layer with configurable encoding and ansatz |
| `VariationalLayer` | Single variational block |
| `VariationalCircuit` | Full parameterized circuit |
| `AngleEncoder` | Angle-based data encoding |
| `AmplitudeEncoder` | Amplitude-based data encoding |
| `QuantumDense` | Dense layer with quantum circuit |
| `QuantumConv2D` | 2D convolution with quantum processing |
| `QuantumActivation` | Base class for quantum activations |
| `QuantumReLU/Sigmoid/Tanh/Softmax/Swish` | Quantum activation functions |

### TensorFlow Module (`quantumflow.tensorflow`)

| Class | Description |
|-------|-------------|
| `QDenseLayer` | tf.keras.layers.Dense replacement with quantum circuit |
| `QConvLayer` | Quantum convolution layer |
| `QVariationalLayer` | General variational quantum layer |
| `QAttentionLayer` | Quantum multi-head attention |
| `QFeatureMapLayer` | Quantum kernel/feature map |
| `QClassifier` | Pre-built quantum classifier |
| `QRegressor` | Pre-built quantum regressor |
| `QAutoencoder` | Quantum autoencoder |
| `QGAN` | Quantum generative adversarial network |
| `QHybridModel` | Hybrid classical-quantum model |
| `ParameterShiftOptimizer` | Exact quantum gradients |
| `NaturalGradientOptimizer` | Fubini-Study metric optimizer |
| `QuantumAdam` | Adam with barren plateau detection |
| `SpsaOptimizer` | Gradient-free noisy optimizer |

### Algorithms Module (`quantumflow.algorithms`)

| Class | Description |
|-------|-------------|
| `GroverSearch` | Grover's search algorithm |
| `AmplitudeAmplification` | Generalized amplitude amplification |
| `QFT` | Quantum Fourier Transform |
| `InverseQFT` | Inverse QFT |
| `ShorAlgorithm` | Shor's factoring algorithm |
| `PhaseEstimation` | Standard QPE |
| `IterativePhaseEstimation` | Iterative QPE |
| `VQE` | Variational Quantum Eigensolver |
| `Hamiltonian` | Hamiltonian as Pauli terms |
| `HWEAnsatz` | Hardware-Efficient Ansatz |
| `UCCSDAnsatz` | UCCSD chemistry ansatz |
| `QAOA` | Quantum Approximate Optimization Algorithm |
| `MaxCutQAOA` | QAOA for MaxCut problem |
| `MISQAOA` | QAOA for Maximum Independent Set |
| `TSPQAOA` | QAOA for Traveling Salesman Problem |

### Noise Module (`quantumflow.noise`)

| Class | Description |
|-------|-------------|
| `NoiseModel` | Circuit-level noise configuration |
| `NoiseConfig` | Noise parameters |
| `DepolarizingChannel` | Depolarizing error |
| `AmplitudeDampingChannel` | T1 relaxation |
| `PhaseDampingChannel` | T2 dephasing |
| `ThermalRelaxationChannel` | Combined T1+T2+thermal |
| `ZeroNoiseExtrapolation` | ZNE error mitigation |
| `ProbabilisticErrorCancellation` | PEC error mitigation |
| `MeasurementErrorMitigation` | Readout error correction |
| `VirtualDistillation` | State purification |
| `SymmetryVerification` | Symmetry-based post-selection |

---

## Architecture

```
quantumflow/
├── __init__.py              # Top-level exports (149 lines)
├── core/                    # Core quantum computing primitives
│   ├── qubit.py             # Qubit, QubitState, MultiQubitState
│   ├── gate.py              # 50+ quantum gates, GateLibrary
│   ├── circuit.py           # QuantumCircuit (main API)
│   ├── register.py          # QuantumRegister, ClassicalRegister
│   ├── state.py             # Statevector, DensityMatrix, Operator
│   └── operation.py         # Operation, CompositeOperation, Barrier
├── simulation/              # Quantum simulators
│   ├── statevector.py       # StatevectorBackend (einsum-based)
│   ├── density_matrix.py    # DensityMatrixBackend (Kraus evolution)
│   └── simulator.py         # Simulator interface + MPSimulator
├── neural/                  # Quantum neural network components
│   ├── qnn_layer.py         # QuantumNNLayer, VariationalLayer, EncodingLayer
│   ├── variational_circuit.py # VariationalCircuit, Encoders, Ansätze
│   ├── quantum_activation.py  # QuantumReLU/Sigmoid/Tanh/Softmax/Swish
│   ├── quantum_dense.py     # QuantumDense, QuantumDenseWithMeasurement
│   └── quantum_conv.py      # QuantumConv2D, QuantumPool2D
├── tensorflow/              # TensorFlow integration
│   ├── layers.py            # 8 quantum TF layers
│   ├── models.py            # 6 pre-built models
│   └── optimizers.py        # 8 quantum-aware optimizers
├── keras/                   # Keras 3 integration
│   ├── layers.py            # 10 quantum Keras layers
│   ├── models.py            # 8 pre-built Keras models
│   └── preprocessing.py     # 4 preprocessing utilities
├── algorithms/              # Quantum algorithms
│   ├── grover.py            # Grover's search, amplitude amplification
│   ├── shor.py              # Shor's factoring, modular exponentiation
│   ├── qft.py               # QFT, quantum adder/multiplier
│   ├── qpe.py               # Phase estimation (standard/iterative/Bayesian)
│   ├── vqe.py               # VQE, Hamiltonian, HWE/UCCSD ansätze
│   └── qaoa.py              # QAOA, MaxCut, MIS, TSP
├── noise/                   # Noise and error mitigation
│   ├── noise_model.py       # NoiseModel, NoiseConfig
│   ├── error_channels.py    # 7 error channels
│   └── error_mitigation.py  # 5 mitigation techniques
├── visualization/           # Visualization tools
│   ├── circuit_drawer.py    # Text/matplotlib/LaTeX drawing
│   └── bloch_sphere.py      # 3D Bloch sphere
└── utils/                   # Mathematical utilities
    └── math.py              # 25+ quantum math functions
```

---

## Performance

### Simulator Benchmarks (approximate, CPU)

| Simulators | 10 qubits | 15 qubits | 20 qubits | 25 qubits |
|-----------|-----------|-----------|-----------|-----------|
| Statevector | < 1 ms | ~10 ms | ~500 ms | ~30 s |
| Density Matrix | ~5 ms | ~100 ms | ~5 s | ~5 min |
| MPS (bond=64) | < 1 ms | ~5 ms | ~50 ms | ~500 ms |

### Gate Application

- Single-qubit gate: O(2^n) via numpy vectorized operations
- Two-qubit gate: O(2^n) via efficient einsum contractions
- Multi-qubit gate: O(2^n) with matrix precomputation and caching

### Optimization

- All gate matrices are lazily computed and cached via `@functools.cached_property`
- Batch simulation uses vectorized einsum for multiple initial states
- MPS simulator uses SVD-based bond dimension truncation
- GPU support via optional CuPy integration

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Run linting (`ruff check quantumflow/`)
5. Run type checking (`mypy quantumflow/`)
6. Commit your changes
7. Push to the branch
8. Open a Pull Request

### Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v --cov=quantumflow
```

---

## License

QuantumFlow is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Citation

If you use QuantumFlow in your research, please cite:

```bibtex
@software{quantumflow2026,
  title = {QuantumFlow: Advanced Quantum Computing Framework with TensorFlow/Keras Integration},
  author = {Mr M Rajamanogaran,manogaran248@gmail.com},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/Rajamanogaran/QuantumFlow.git}
}
```
