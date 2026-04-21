# QuantumFlow API Reference

This document provides detailed API documentation for all QuantumFlow modules.

---

## Module: `quantumflow.core`

### Classes

#### `QuantumCircuit(n_qubits, name=None)`

The main circuit construction class. Supports 50+ gate types and circuit operations.

**Methods:**

| Method | Description |
|--------|-------------|
| `h(q)` | Hadamard gate on qubit q |
| `x(q)` | Pauli-X (NOT) gate |
| `y(q)` | Pauli-Y gate |
| `z(q)` | Pauli-Z gate |
| `s(q)` | S gate (sqrt of Z) |
| `sdg(q)` | S-dagger gate |
| `t(q)` | T gate (4th root of Z) |
| `tdg(q)` | T-dagger gate |
| `sx(q)` | Square root of X gate |
| `rx(theta, q)` | Rotation about X axis |
| `ry(theta, q)` | Rotation about Y axis |
| `rz(theta, q)` | Rotation about Z axis |
| `p(theta, q)` | Phase gate |
| `u(theta, phi, lam, q)` | General single-qubit rotation |
| `u3(theta, phi, lam, q)` | U3 gate |
| `cx(control, target)` | Controlled-X (CNOT) |
| `cy(control, target)` | Controlled-Y |
| `cz(control, target)` | Controlled-Z |
| `swap(a, b)` | SWAP two qubits |
| `ccx(a, b, c)` | Toffoli (CCX) |
| `rxx(theta, a, b)` | XX rotation |
| `ryy(theta, a, b)` | YY rotation |
| `rzz(theta, a, b)` | ZZ rotation |
| `crx(theta, control, target)` | Controlled RX |
| `cry(theta, control, target)` | Controlled RY |
| `crz(theta, control, target)` | Controlled RZ |
| `measure(qubit, cbit)` | Measure qubit to classical bit |
| `barrier()` | Add a barrier |
| `reset(qubit)` | Reset qubit to |0> |
| `append(gate, qubits, params)` | Append a custom gate |
| `compose(other)` | Compose with another circuit |
| `tensor(other)` | Tensor product with another circuit |
| `inverse()` | Return inverse circuit |
| `reverse()` | Reverse gate order |
| `copy()` | Deep copy |
| `to_unitary()` | Compute full unitary matrix |
| `qasm()` | Export to OpenQASM 2.0 string |
| `depth()` | Circuit depth |
| `width()` | Number of qubits |
| `size()` | Total gate count |
| `count_gates()` | Dict of gate counts |
| `bind_parameters(params)` | Bind parameter values |

**Properties:**
- `n_qubits`: Number of qubits
- `_operations`: List of operations

---

#### `Gate` (base class)

All gates inherit from this class.

**Properties:**
- `name` (str): Gate name
- `num_qubits` (int): Number of qubits the gate acts on
- `num_params` (int): Number of parameters
- `matrix` (np.ndarray): Unitary matrix (cached)

**Methods:**
- `dagger()`: Return the adjoint (inverse) gate
- `controlled(n_controls=1)`: Return a controlled version
- `power(exponent)`: Return gate raised to a power

---

#### `Statevector(data)`

Pure quantum state representation.

**Constructor Parameters:**
- `data` (np.ndarray): Complex array of shape (2^n,)

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `probabilities()` | np.ndarray | Measurement probabilities |
| `measure()` | (str, Statevector) | Single measurement |
| `sample(shots)` | Dict[str, int] | Multiple measurements |
| `expectation(observable)` | complex | Expectation value |
| `evolve(unitary)` | None | Apply unitary in-place |
| `purity()` | float | State purity (always 1.0) |
| `entropy()` | float | Von Neumann entropy (always 0.0) |
| `bloch_vectors()` | List[np.ndarray] | Per-qubit Bloch vectors |
| `to_density_matrix()` | DensityMatrix | Convert to density matrix |
| `tensor(other)` | Statevector | Tensor product |
| `fidelity(other)` | float | Fidelity with another state |

**Class Methods:**
- `from_label(label)`: Create from computational basis label (e.g., "01+", "10-")
- `random(n_qubits)`: Generate random statevector

---

#### `DensityMatrix(data)`

Mixed quantum state representation.

**Constructor Parameters:**
- `data` (np.ndarray): Complex array of shape (2^n, 2^n)

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `purity()` | float | Tr(rho^2) |
| `von_neumann_entropy()` | float | -Tr(rho log rho) |
| `is_pure()` | bool | Check if state is pure |
| `is_mixed()` | bool | Check if state is mixed |
| `fidelity(other)` | float | Uhlmann fidelity |
| `trace_distance(other)` | float | 0.5 * Tr|rho - sigma| |
| `evolve(unitary)` | None | rho -> U rho U^dagger |
| `evolve_kraus(ops)` | None | rho -> sum E_k rho E_k^dagger |
| `partial_trace(keep)` | DensityMatrix | Trace out subsystems |
| `tensor_product(other)` | DensityMatrix | Tensor product |

**Class Methods:**
- `from_statevector(sv)`: Create from pure state
- `maximally_mixed(n)`: Create maximally mixed state

---

## Module: `quantumflow.simulation`

### `StatevectorSimulator(config=None)`

Ideal statevector simulator.

```python
sim = StatevectorSimulator()
result = sim.run(circuit, shots=1024, initial_state=None)
state = sim.state(circuit)
probs = sim.probabilities(circuit)
expt = sim.expectation(circuit, observable)
```

### `DensityMatrixSimulator(config=None, noise_config=None)`

Density matrix simulator with noise support.

```python
from quantumflow.noise.noise_model import NoiseConfig
config = NoiseConfig(single_gate_error=0.01)
sim = DensityMatrixSimulator(noise_config=config)
result = sim.run(circuit, shots=1024)
```

### `MPSimulator(config=None)`

Matrix Product State simulator for larger systems.

```python
from quantumflow.simulation.simulator import MPSimulator, BackendConfig
config = BackendConfig(max_qubits=30, optimization_level=2)
sim = MPSimulator(config=config)
result = sim.run(circuit, shots=1024)
```

### `SimulationResult`

Result container with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `statevector` | np.ndarray | Final quantum state |
| `density_matrix` | np.ndarray | Final density matrix (if applicable) |
| `counts` | Dict[str, int] | Measurement outcome counts |
| `probabilities` | Dict[str, float] | Outcome probabilities |
| `memory` | List[str] | Individual shot results |
| `metadata` | Dict | Execution info |

| Method | Returns | Description |
|--------|---------|-------------|
| `get_counts()` | Dict[str, int] | Bitstring -> count mapping |
| `get_probabilities()` | Dict[str, float] | Bitstring -> probability mapping |
| `plot_histogram()` | matplotlib.figure.Figure | Bar chart of results |

---

## Module: `quantumflow.neural`

### `QuantumNNLayer(n_qubits, n_layers, encoding, variational_form, observable)`

Full quantum neural network layer.

**Parameters:**
- `n_qubits` (int): Number of qubits
- `n_layers` (int): Number of variational layers
- `encoding` (str): 'angle', 'amplitude', 'basis', 'iqp', 'dense_angle'
- `variational_form` (str): 'hardware_efficient', 'strong_entangling', 'circuit_19', 'barren_plateau_free', 'qaoa'
- `observable` (str): 'z', 'x', 'y', 'zz', 'xx', 'mixed', 'computational_basis'

### `VariationalCircuit(n_qubits, n_layers, entanglement, rotations)`

Parameterized quantum circuit for VQA.

**Entanglement patterns:** 'linear', 'circular', 'full', 'pairwise', 'star'
**Rotation sets:** 'rx', 'ry', 'rz', 'rycz', 'full'

### `QuantumDense(output_dim, n_qubits, n_layers, activation, use_bias)`

Drop-in replacement for Dense layers using quantum circuits.

### `QuantumConv2D(filters, kernel_size, n_qubits, strides, padding, n_layers)`

2D convolution with quantum circuit processing.

---

## Module: `quantumflow.tensorflow`

### Layers (all inherit from `tf.keras.layers.Layer`)

| Layer | Key Parameters |
|-------|---------------|
| `QDenseLayer(units, n_qubits, n_layers)` | Output units, quantum circuit size |
| `QConvLayer(filters, kernel_size, n_qubits)` | Convolution with quantum processing |
| `QVariationalLayer(n_qubits, circuit_fn)` | User-defined quantum circuit |
| `QBatchNormLayer(n_qubits)` | Quantum-inspired batch normalization |
| `QAttentionLayer(n_qubits, num_heads)` | Quantum self-attention |
| `QResidualLayer(n_qubits, n_layers)` | Quantum skip connection |
| `QFeatureMapLayer(n_qubits, feature_map)` | Quantum kernel map |
| `QMeasurementLayer(n_qubits, observable)` | Configurable measurement |

### Models

| Model | Methods |
|-------|---------|
| `QClassifier` | `compile()`, `fit()`, `predict()`, `evaluate()` |
| `QRegressor` | `compile()`, `fit()`, `predict()`, `evaluate()` |
| `QAutoencoder` | `compile()`, `fit()`, `encode()`, `decode()` |
| `QGAN` | `compile()`, `fit()`, `generate()` |
| `QTransferLearningModel` | `build()`, `fine_tune()` |
| `QHybridModel` | `add_classical_layer()`, `add_quantum_layer()` |

### Optimizers (all extend `tf.keras.optimizers.Optimizer`)

| Optimizer | Key Feature |
|-----------|-------------|
| `ParameterShiftOptimizer` | Exact quantum gradients |
| `NaturalGradientOptimizer` | Fubini-Study metric |
| `QuantumAdam` | Barren plateau detection |
| `QuantumLAMB` | Large-batch training |
| `QuantumSGD` | Shot-based gradients |
| `SpsaOptimizer` | Gradient-free (noisy hardware) |

---

## Module: `quantumflow.keras`

### Layers (all inherit from `keras.layers.Layer` with Keras 3 support)

| Layer | Description |
|-------|-------------|
| `KerasQuantumLayer` | Base quantum layer with custom gradients |
| `KerasQDense(units, n_qubits, n_layers)` | Dense with quantum circuit |
| `KerasQConv2D(filters, kernel_size, n_qubits)` | Quantum 2D convolution |
| `KerasQAttention(num_heads)` | Quantum self-attention |
| `KerasQVariational` | General variational layer |
| `KerasQBatchNormalization` | Quantum batch norm |
| `KerasQLayerNormalization` | Quantum layer norm |
| `KerasQDropout(rate)` | Noise-channel dropout |
| `KerasQPooling2D` | Quantum pooling |
| `KerasQFlatten` | Flatten quantum results |

### Models

| Model | Description |
|-------|-------------|
| `KerasQuantumClassifier` | Binary + multi-class classification |
| `KerasQuantumRegressor` | Regression with MSE/MAE |
| `KerasQNN` | Flexible model builder |
| `KerasQuantumAutoencoder` | Quantum compression |
| `KerasHybridModel` | Mixed classical-quantum |
| `KerasQuantumGAN` | Quantum GAN with train_step() |
| `KerasQuantumVAE` | Quantum VAE with beta-VAE |
| `KerasTransferLearning` | Transfer learning with 3 strategies |

---

## Module: `quantumflow.algorithms`

### `GroverSearch(n_qubits, oracle, marked_states, num_iterations)`

```python
grover = GroverSearch(n_qubits=3, marked_states=['101'])
result = grover.run(shots=1024)
# result: {'counts', 'marked_found', 'most_frequent', 'success_probability', ...}
```

### `QFT(n_qubits, inverse, do_swaps, approximation_degree)`

```python
qft = QFT(8, approximation_degree=3)  # Approximate QFT
circuit = qft.construct_circuit()
U = qft.exact_unitary()  # Full QFT matrix
```

### `ShorAlgorithm(N, a)`

```python
shor = ShorAlgorithm(N=15)
result = shor.factor()
# result: {'factors', 'N', 'attempts', 'success', ...}
```

### `PhaseEstimation(unitary, n_evaluation_qubits, n_state_qubits)`

```python
qpe = PhaseEstimation(U, 8, 1)
result = qpe.run(shots=4096)
# result: {'phase', 'phase_bits', 'precision', ...}
```

### `VQE(hamiltonian, ansatz, optimizer, initial_params)`

```python
vqe = VQE(H, ansatz, optimizer='COBYLA')
result = vqe.run(max_iterations=200)
# VQEResult: optimal_energy, optimal_params, convergence_history, ...
```

### `QAOA(cost_hamiltonian, p, mixer)`

```python
qaoa = QAOA(cost, p=3, mixer='x')
result = qaoa.run(optimizer='COBYLA')
# QAOAResult: optimal_cost, optimal_params, best_bitstring, approximation_ratio
```

### `MaxCutQAOA(edges, n_nodes, p)`

```python
maxcut = MaxCutQAOA(edges=[(0,1),(1,2),(2,3)], n_nodes=4, p=3)
result = maxcut.solve()
partition = maxcut.get_cut(result.best_bitstring)
cut_value = maxcut.cut_value(result.best_bitstring)
```

---

## Module: `quantumflow.noise`

### Error Channels (all subclasses of `ErrorChannel`)

| Channel | Parameter | Description |
|---------|-----------|-------------|
| `DepolarizingChannel(p)` | p: error probability | Replace state with maximally mixed |
| `AmplitudeDampingChannel(gamma)` | gamma: damping rate | Energy dissipation (T1) |
| `PhaseDampingChannel(lambda)` | lambda: dephasing rate | Phase decoherence (T2) |
| `BitFlipChannel(p)` | p: flip probability | Apply X with probability p |
| `PhaseFlipChannel(p)` | p: flip probability | Apply Z with probability p |
| `PauliErrorChannel(px, py, pz)` | Pauli error rates | General Pauli noise |
| `ThermalRelaxationChannel(t1, t2, gate_time)` | Physical parameters | Combined T1+T2+thermal |

**Common Methods:**
- `kraus_operators()`: List of Kraus operators
- `apply(rho)`: Apply channel to density matrix
- `superoperator()`: Liouville representation
- `choi_matrix()`: Choi matrix representation
- `is_cptp()`: Verify CPTP property

### Error Mitigation

| Technique | Class | Method |
|-----------|-------|--------|
| Zero Noise Extrapolation | `ZeroNoiseExtrapolation` | Richardson/exponential/linear |
| Probabilistic Error Cancellation | `ProbabilisticErrorCancellation` | Pseudo-inverse noise |
| Measurement Error Mitigation | `MeasurementErrorMitigation` | Confusion matrix inversion |
| Virtual Distillation | `VirtualDistillation` | State purification (rho^k) |
| Symmetry Verification | `SymmetryVerification` | Post-selection on symmetries |

---

## Module: `quantumflow.visualization`

### `CircuitDrawer(circuit, scale, wire_order)`

```python
drawer = CircuitDrawer(qc)
drawer.draw_text(label="My Circuit")        # ASCII art
drawer.draw_matplotlib(title="Circuit")      # matplotlib figure
drawer.draw_latex()                          # LaTeX Qcircuit code
```

### `BlochSphere(figsize, title)`

```python
bloch = BlochSphere(title="States")
bloch.add_state(state, label="|psi>")
bloch.add_trail(state_list, color='green')
bloch.show(filename="bloch.png")
bloch.clear()
```

---

## Module: `quantumflow.utils`

### Mathematical Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `kron` | `kron(*matrices)` | Kronecker product |
| `tensor_product` | `tensor_product(*states)` | Tensor product of states |
| `partial_trace` | `partial_trace(rho, keep, n)` | Trace out subsystems |
| `density_from_statevector` | `density_from_statevector(psi)` | Create density matrix |
| `fidelity` | `fidelity(state1, state2)` | Uhlmann fidelity |
| `trace_distance` | `trace_distance(rho, sigma)` | Trace distance |
| `purity` | `purity(rho)` | Tr(rho^2) |
| `von_neumann_entropy` | `von_neumann_entropy(rho)` | -Tr(rho log rho) |
| `expectation_value` | `expectation_value(state, op)` | Expectation value |
| `state_to_bloch` | `state_to_bloch(psi)` | State to Bloch vector |
| `bloch_to_state` | `bloch_to_state(vec)` | Bloch vector to state |
| `random_unitary` | `random_unitary(n)` | Random Haar unitary |
| `random_density_matrix` | `random_density_matrix(n, rank)` | Random density matrix |
| `is_hermitian` | `is_hermitian(M)` | Hermiticity check |
| `is_unitary` | `is_unitary(M)` | Unitarity check |
| `is_positive_semidefinite` | `is_positive_semidefinite(M)` | PSD check |
| `normalize_state` | `normalize_state(psi)` | Normalize state vector |
| `pauli_matrices` | `pauli_matrices()` | Dict of Pauli matrices |
| `commutator` | `commutator(A, B)` | [A, B] |
| `anticommutator` | `anticommutator(A, B)` | {A, B} |
