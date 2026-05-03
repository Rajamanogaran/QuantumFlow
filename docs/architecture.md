# QuantumFlow Architecture

This document describes the internal architecture and design decisions of QuantumFlow.

## Design Philosophy

QuantumFlow follows these design principles:

1. **Modularity**: Each module is self-contained with clear interfaces
2. **Type Safety**: Full type hints throughout the codebase
3. **Performance**: Lazy evaluation, matrix caching, and efficient tensor contractions
4. **Extensibility**: Easy to add new gates, layers, algorithms, and noise models
5. **Integration**: Seamless TensorFlow/Keras compatibility as first-class citizens

## Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                      │
├─────────────────────────────────────────────────────────┤
│  keras/          │  tensorflow/      │     neural/       │
│  (Keras 3 API)   │  (TF layers,      │  (QNN primitives) │
│                   │   models, opts)  │                   │
├─────────────────────────────────────────────────────────┤
│                   algorithms/                            │
│  (Grover, Shor, QFT, QPE, VQE, QAOA)                  │
├─────────────────────────────────────────────────────────┤
│  simulation/      │  noise/                              │
│  (Statevector,    │  (Error channels,                   │
│   DensityMatrix,  │   NoiseModel,                        │
│   MPS)            │   Error mitigation)                 │
├─────────────────────────────────────────────────────────┤
│                      core/                               │
│  (Gates, Circuit, Statevector, DensityMatrix, Register) │
├─────────────────────────────────────────────────────────┤
│              utils/              │  visualization/       │
│  (Math, helpers)  │  (Circuit drawer, Bloch sphere)      │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Quantum Circuit Execution

```
QuantumCircuit
    │
    ├── Operations: [Gate, QubitIndices, Parameters]
    │
    ▼
Simulator.run()
    │
    ├── StatevectorSimulator
    │   ├── Initialize |0...0>
    │   ├── For each gate:
    │   │   ├── Get gate matrix (cached)
    │   │   ├── Apply via einsum: U ⊗ I ⊗ ... ⊗ I
    │   │   └── Normalize
    │   ├── Optional: compute gradients (parameter-shift)
    │   └── Measure / sample
    │
    ├── DensityMatrixSimulator
    │   ├── Initialize rho = |0><0|
    │   ├── For each gate:
    │   │   ├── rho → U @ rho @ U†
    │   │   └── Optional: apply noise (Kraus)
    │   └── Measure (POVM)
    │
    └── MPSimulator
        ├── Initialize MPS tensors
        ├── For each gate:
        │   ├── If adjacent: SVD-based application
        │   └── Else: contract → apply → decompose
        └── Contract to statevector for measurement
    │
    ▼
SimulationResult
    ├── statevector
    ├── counts
    ├── probabilities
    └── metadata
```

### Quantum Neural Network Training

```
Input Data (x)
    │
    ▼
QuantumDense / QDenseLayer
    │
    ├── build(input_shape)
    │   └── Create tf.Variable for parameters
    │
    ├── call(inputs)
    │   ├── Encode: x → quantum feature map
    │   ├── Variational: apply parameterized circuit
    │   ├── Measure: compute expectation values
    │   └── Return classical output
    │
    ├── @tf.custom_gradient (if using parameter-shift)
    │   ├── Forward: normal execution
    │   └── Backward: parameter-shift rule
    │       ∂f/∂θ = [f(θ+π/2) - f(θ-π/2)] / 2
    │
    ▼
Output (predictions)
    │
    ▼
Loss Function (cross-entropy, MSE, etc.)
    │
    ▼
Optimizer (QuantumAdam, SPSA, etc.)
    │
    └── Update quantum parameters + classical weights
```

## Key Design Patterns

### 1. Lazy Gate Matrix Computation

Gate matrices are computed only when needed and cached:

```python
class Gate:
    @functools.cached_property
    def matrix(self) -> np.ndarray:
        """Compute and cache the unitary matrix."""
        matrix = self._compute_matrix()
        # Validate unitarity
        assert np.allclose(matrix @ matrix.conj().T, np.eye(2**self.num_qubits))
        return matrix
```

### 2. Backend Abstraction

Simulators share a common interface:

```python
class Simulator(ABC):
    @abstractmethod
    def run(self, circuit, shots, initial_state) -> SimulationResult: ...

    @abstractmethod
    def expectation(self, circuit, observable) -> float: ...

    @abstractmethod
    def state(self, circuit) -> Statevector: ...
```

### 3. Noise as Kraus Operators

All noise channels implement CPTP maps via Kraus operators:

```python
class ErrorChannel(ABC):
    @abstractmethod
    def kraus_operators(self) -> List[np.ndarray]: ...

    def apply(self, rho):
        return sum(K @ rho @ K.conj().T for K in self.kraus_operators())
```

### 4. Hybrid Layer Architecture

Quantum layers follow the Keras layer protocol:

```python
class QDenseLayer(tf.keras.layers.Layer):
    def build(self, input_shape):     # Create variables
    def call(self, inputs):           # Forward pass
    def compute_output_shape(...):    # Shape inference
    def get_config(self):             # Serialization
```

## Memory Management

- **Statevector**: Single complex128 array of size 2^n
- **Density Matrix**: Complex128 array of size 2^n × 2^n
- **MPS**: Array of tensors, each (bond_left, 2, bond_right)
- **Gate matrices**: Cached per-gate, shared across circuits

## Numerical Precision

- All computations use `numpy.complex128` (double precision)
- State vectors are normalized after each operation
- Gate matrices are validated for unitarity on construction
- Tolerance thresholds: `atol=1e-10` for comparisons

## Extension Points

Users can extend QuantumFlow by:

1. **Custom Gates**: Subclass `Gate` and implement `matrix` property
2. **Custom Layers**: Subclass `tf.keras.layers.Layer` or `keras.layers.Layer`
3. **Custom Algorithms**: Use the `Simulator` and `QuantumCircuit` APIs
4. **Custom Noise**: Subclass `ErrorChannel` and provide Kraus operators
5. **Custom Optimizers**: Subclass `tf.keras.optimizers.Optimizer`
