# Getting Started with QuantumFlow

Welcome to QuantumFlow! This guide will walk you through installing the framework and running your first quantum programs.

## Installation

### Step 1: Ensure Python 3.10+

```bash
python --version
# Should show Python 3.10, 3.11, or 3.12
```

### Step 2: Install QuantumFlow

```bash
cd quantumflow/
pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "import quantumflow as qf; print(f'QuantumFlow v{qf.__version__} installed!')"
```

### Step 4: (Optional) Install with Full Dependencies

```bash
# For TensorFlow/Keras integration
pip install tensorflow keras

# For development (testing, linting, type checking)
pip install -e ".[dev]"

# For GPU acceleration (requires CUDA toolkit)
pip install -e ".[gpu]"
```

## Your First Quantum Circuit

Create a file called `hello_quantum.py`:

```python
import numpy as np
import quantumflow as qf

# === 1. Create a Bell State ===
print("=== Bell State (EPR Pair) ===")

qc = qf.QuantumCircuit(2)
qc.h(0)        # Apply Hadamard to qubit 0
qc.cx(0, 1)    # Apply CNOT with control=0, target=1

# Print the circuit
from quantumflow.visualization import CircuitDrawer
drawer = CircuitDrawer(qc)
print(drawer.draw_text())

# Simulate
sim = qf.StatevectorSimulator()
result = sim.run(qc, shots=1000)

print(f"Counts: {result.get_counts()}")
print(f"Probabilities: {result.get_probabilities()}")

# === 2. Verify Entanglement ===
sv = result.statevector
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
print(f"Fidelity with |Bell>: {qf.fidelity(sv, bell):.6f}")

# === 3. Partial Trace ===
dm = qf.DensityMatrix.from_statevector(sv)
reduced = dm.partial_trace(keep_qubits=[0])
print(f"Reduced state purity: {reduced.purity():.4f} (should be 0.5 for maximally entangled)")
```

Run it:
```bash
python hello_quantum.py
```

## Your First Quantum Neural Network

Create a file called `hello_qnn.py`:

```python
import numpy as np
import tensorflow as tf
import quantumflow.tensorflow as qf_tf

# === 1. Generate Synthetic Data ===
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 4).astype(np.float32)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(np.float32)

# === 2. Build Quantum-Classical Model ===
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),
    qf_tf.QDenseLayer(8, n_qubits=4, n_layers=2, activation='quantum_relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# === 3. Train ===
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# === 4. Evaluate ===
loss, accuracy = model.evaluate(X[-100:], y[-100:])
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```

## Your First Quantum Algorithm: Grover's Search

```python
from quantumflow.algorithms.grover import GroverSearch

# Search for |101> in a database of 8 items (3 qubits)
grover = GroverSearch(n_qubits=3, marked_states=['101'])

print(f"Database size: {2**3} items")
print(f"Marked states: {grover.marked_states}")
print(f"Optimal iterations: {grover.optimal_iterations()}")
print(f"Success probability: {grover.success_probability():.2%}")

# Run the algorithm
result = grover.run(shots=1024)
print(f"\nMost frequent outcome: {result['most_frequent']}")
print(f"Counts: {result['counts']}")
```

## Your First VQE: Finding Ground State Energy

```python
import numpy as np
from quantumflow.algorithms.vqe import VQE, Hamiltonian, HWEAnsatz

# Define a Hamiltonian (Heisenberg model)
H = Hamiltonian.heisenberg_hamiltonian(n_qubits=3, jx=1.0, jy=1.0, jz=1.0)

# Exact solution for comparison
eigenvalues = np.linalg.eigvalsh(H.matrix())
print(f"Exact ground state energy: {eigenvalues[0]:.6f}")

# Set up VQE
ansatz = HWEAnsatz(n_qubits=3, n_layers=3, rotation_set=['ry', 'rz'], entanglement='full')
vqe = VQE(H, ansatz, optimizer='COBYLA')

# Run optimization
result = vqe.run(max_iterations=200, convergence_threshold=1e-6)

print(f"\nVQE ground state energy: {result.optimal_energy:.6f}")
print(f"Error: {abs(result.optimal_energy - eigenvalues[0]):.6f}")
print(f"Iterations: {result.iteration_count}")
```

## Next Steps

- Read the full [README.md](../README.md) for complete API reference
- Explore the [tutorials](./tutorials/) for advanced examples
- Check the [API Reference](./api-reference.md) for detailed documentation
- Run `examples/demo.py` for 8 comprehensive examples
- Run `pytest tests/ -v` to verify everything works
