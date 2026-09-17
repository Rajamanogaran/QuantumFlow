# Changelog

All notable changes to QuantumFlow are documented in this file.

## [0.2.0] - 2026-09-16

Maintenance and correctness release: the entire package was audited
end-to-end and every discovered defect was fixed.

### Fixed

#### Core
- `QuantumCircuit._embed_gate` used `P(inv_perm)` as the inverse of the
  qubit permutation `P(perm)`, which is wrong for any non-involutory
  permutation. Gates acting on ≥3 qubits (and on non-contiguous or
  reversed qubit lists) were applied to the *wrong qubits* by
  `to_unitary()` / `to_matrix()`. Replaced with a direct, provably
  correct index construction.
- `QuantumCircuit.measure()` crashed (`IndexError`) when the circuit had
  no classical register; it now auto-creates one on demand, matching the
  documented quick-start usage.
- `QuantumCircuit.compose()` now accepts a `qubits=` mapping so
  sub-circuits can be composed onto specific registers (used by QPE, Shor).
- `Statevector` gained scalar arithmetic (`*`, `rmul`, `/`) that returns a
  `Statevector` instead of silently degrading to a raw `numpy.ndarray`.
- Added `Statevector.to_circuit()` for exact state preparation
  (basis states via `X` gates; general states via a Householder unitary).

#### Simulation
- Circuits whose measurements all come **after** every gate re-executed
  the full circuit once per shot (collapse-once semantics are identical
  for terminal measurements — every shot samples the same final
  distribution). Terminal-measurement circuits are now executed once and
  sampled per shot from the exact distribution — identical statistics,
  orders of magnitude faster (e.g. Shor order finding on 18 qubits:
  minutes → ~1 s). Mid-circuit measurements still re-execute per shot.
- `StatevectorBackend.apply_measurement` summed measurement marginals
  over axes in ascending order, producing an `AxisError` (and wrong
  probabilities) whenever more than one axis was traced out.
- All three simulators (statevector, density matrix, MPS) collapsed the
  state **once** and sampled all shots from the collapsed state, giving
  statistically wrong results for circuits with mid-circuit measurements.
  Circuits containing measurement ops are now re-executed per shot, and
  counts report exactly the measured qubits.
- The gate-matrix cache was keyed by `(gate.name, params)` only; distinct
  gates sharing a name (e.g. several `UnitaryGate`s) silently returned the
  first gate's matrix. The cache now validates gate identity.
- `StatevectorBackend.apply_gate` / `apply_gate_full` / batched variant
  transposed the gate tensor into sorted-qubit order while binding einsum
  subscripts in unsorted order — wrong results for gates whose qubit list
  was not ascending (e.g. `cx(2, 0)`).
- `DensityMatrixBackend._embed_operator` had the same permutation bug as
  the core `_embed_gate`; rewritten with the direct construction.

#### TensorFlow
- `QuantumLAMB` never moved off zero-initialised parameters: the trust
  ratio was `|w|/|u| = 0/|u| = 0` at the origin, zeroing every update.
  It now falls back to a trust ratio of 1.0 when either norm is ~0
  (0.83 -> 0.34 on the reference objective, previously exactly flat).

#### Algorithms
- **Grover**: the single-marked-state oracle applied a plain `Z` to the
  last qubit (not a phase flip on the marked state), and the custom
  multi-controlled-X decomposition infinitely recursed for ≥3 controls
  (`RecursionError`). The oracle and diffusion operator now use the exact
  `MCZGate`/`MCXGate` from the core library. Grover provably finds marked
  states for 3+ qubits and multiple marked states.
- **QFT**: the `rz-cx-rz-cx` rotation gadget carried extra single-qubit
  phases — the constructed circuit did *not* implement the QFT (nor did
  its inverse undo it). Rewritten with exact `CP` gates and verified
  against `qft_matrix(n)` for n = 1…4.
- **QPE**: each controlled-U^{2^k} was controlled on the *same* evaluation
  qubit; the IQFT ran with the swap layer despite the bit-reversed
  winding; the phase parser bit-reversed the result. Fixed control
  assignment, `do_swaps=False`, and MSB-first parsing; phases are now
  exact.
- **IPE** (`IterativePhaseEstimation`): the controlled unitary was
  decomposed with per-element `rz-cx` gadgets (only valid for diagonal U),
  the power schedule measured bits in the wrong order, and the feedback
  was applied to `|0>` before the first Hadamard (a no-op global phase).
  Rewritten as the correct semi-classical IPE with exact controlled
  unitaries; recovers phases to full precision.
- **Bayesian QPE**: same broken controlled-U, no eigenstate preparation,
  likelihood/circuit mismatch, and a backwards power schedule. Now uses
  exact controlled unitaries, prepares the eigenstate, sweeps powers
  upward with a consistent likelihood model, and reports the MAP phase.
  The mirror ambiguity `phi ↔ 1 - phi` (intrinsic to feedback-free
  H-H measurements) is documented.
- **Shor**: the inverse QFT was composed into a mismatched-width circuit
  (`ValueError`); the modular-exponentiation circuit was a CCX chain that
  did not compute modular arithmetic; the work register was initialised to
  `|2^(n-1)>` instead of `|1>`. Rewritten with exact controlled modular
  multiplication permutations; `factor()` now correctly factors 15, 21, 35.
- **QAOA**: the cost-unitary decomposition was wrong in two ways — it
  emitted stray single-qubit `RZ` rotations for every `Z` character of a
  multi-qubit term (double-counting diagonal phases), and terms with
  *non-adjacent* `Z` qubits (e.g. `ZIZ`) got no entangling gate at all.
  Z-only strings are now decomposed exactly: `RZ` for weight-1, `RZZ`
  (any qubit distance) for weight-2, CNOT-ladder for higher weights.
- **MaxCutQAOA** built its cost Hamiltonian with *negative* `Z_iZ_j`
  coefficients — minimizing it maximizes spin alignment, driving the
  cut to zero. Fixed to `+0.5·Z_iZ_j`; the solved cut is now optimal on
  the path-graph reference problem for every tested seed. The unused
  `shots` argument of `solve()` is now forwarded, and the reported
  bitstring is the lowest-cost sample (ties by frequency) instead of
  the raw mode.
- **MISQAOA** kept only the `Z_iZ_j` part of `penalty·x_i x_j` and
  dropped its linear terms, inverting the penalty's effect near the
  optimum; the cost function is now the exact Z-basis expansion.
- `keras/preprocessing.py` referenced an undefined `QuantumCircuit`
  (`NameError` on `get_encoding_circuit`).
- Dead `try/except ImportError` import shims and unused imports across
  the algorithm modules removed; `ruff` (pycodestyle/pyflakes) now
  passes clean on the whole package and test suite.

#### Neural / TensorFlow / Keras
- `QClassifier` / `QRegressor` (plain-TF models) never trained their
  classical readout: only the circuit parameters received updates, so the
  randomly-initialised readout stayed frozen and the models could not
  learn (regression loss flat at ~0.25, classification flat at ~0.69).
  The readout kernel/bias now train jointly with the circuit parameters
  (analytic output-layer gradients + Adam), and are Xavier-initialised.
  A 4-qubit `QRegressor` now fits `sin` to MSE 0.01; text classification
  loss drops 0.72 → 0.58.
- The models' gradient helper used central finite differences with
  ``eps = 1e-7`` — catastrophic cancellation on an O(1) loss, so circuit
  gradients were pure noise. All trainable parameters are RZ/RY angles,
  so the *exact* parameter-shift rule (±π/2) is used instead.
- `QuantumPool2D` raised ``RuntimeError: Layer has not been built`` on
  first use; it now lazy-builds like the other layers (and a duplicated
  lazy-build block in `QuantumConv2D` was removed).
- Quantum layers raised `RuntimeError: Layer not built` on first use;
  all layer `call()` methods now lazy-build with the incoming shape
  (Keras-style auto-build).
- `KerasQDense`/`KerasQVariational` used `keras.ops.activations`, which
  does not exist in Keras 3; replaced with `keras.activations.get`.
- `KerasQDense`/`KerasQVariational` crashed under `model.fit()` with
  ``NotImplementedError: numpy() is only available when eager execution
  is enabled`` (Keras 3 traces `call` under `tf.function`). The numpy
  quantum evaluation now runs through a graph-safe, differentiable
  bridge (`_hybrid_quantum_bridge`): circuits execute on detached
  snapshots via the backend runtime and gradients flow through a
  finite-difference local linearisation — `fit()` trains end-to-end.
- `QuantumConv2D` flattened patches to `n_qubits` features while the
  weight matrix expected `kernel_size² * channels` (matmul shape crash).
- `QuantumNNLayer` exploded the `'rycz'` rotation preset into the
  characters `('r','y','c','z')`; presets are now parsed like
  `variational_circuit._parse_rotation_set` with validation.
- `QuantumNormalizer` and the other Keras preprocessing layers called an
  undefined `_check_keras_available` (`NameError` on instantiation).
- `KerasQBatchNormalization` was broken in both paths: the training
  branch called `ops.convert_to_numpy` on variables inside the
  tf.function-traced `call` (`numpy() is only available when eager
  execution is enabled`), and both branches normalised with statistics
  shaped for the wrong axis (moving-stat assignments of shape
  `(batch, 1)` into `(features,)` variables; inference broadcast
  `(F, 1)` statistics against `(batch, F)` inputs). Batch statistics are
  now reduced per feature (batch axis), moving statistics update through
  graph-safe ops arithmetic, and inference broadcasts correctly.
  Regression-tested in `tests/test_keras_layers.py`.

#### Noise & error mitigation
- `NoiseModel.apply_noise` was a **no-op**: it iterated `op.gate` over
  every circuit instruction — including measurement operations, which
  have no `.gate` — and returned the circuit unchanged (an `AttributeError`
  path silently swallowed, and even for pure-gate circuits nothing was
  ever inserted). Baking noise into a circuit now appends real
  `KrausChannel` operations after every gate, scaled by `noise_scale`,
  and measurement instructions are handled via a shared
  `getattr(op, "gate", None)` guard (also applied to the five other
  `op.gate` iteration sites in `simulation/simulator.py`).
- The default depolarizing channel was not a valid CPTP map (a
  two-Kraus construction that failed completeness for n ≥ 1). Rewritten
  as the exact Weyl-channel Kraus expansion — completeness holds to
  < 1e-9 for 1 and 2 qubits.
- Added `KrausChannel` (`core/operation.py`) for general CPTP maps:
  completeness-validated, `circuit.append_kraus(kraus_ops, qubits)`,
  applied by `DensityMatrixSimulator` (statevector simulation raises a
  clear `TypeError` pointing at the density-matrix simulator).
- `DensityMatrixSimulator` gained attach-mode noise
  (`DensityMatrixSimulator(noise_model=...)`): each gate is followed by
  the configured error channel via `NoiseModel.after_gate` /
  `_embed` (MSB-first bit-blasting onto the full register).
- **Zero-noise extrapolation (Richardson)** returned garbage: the
  Lagrange weights omitted the `(0 - x_j)` numerators — e.g. the weights
  summed to ~0 instead of 1 and the mitigated value of data trending to
  ~0.93 came out 0.02. Weights are now the full
  `w_i = prod_{j≠i} (0 - x_j) / (x_i - x_j)`; exact on polynomial data.
  (`linear` / `exponential` variants were already correct.)

### Added

#### Noise
- `KrausChannel` general CPTP support end-to-end (construct, validate,
  append to circuits, simulate on density matrices), plus
  `tests/test_noise.py` covering channel validity/completeness, noise
  baking (monotone degradation with `noise_scale`), attach-mode noise,
  Kraus-embedding conventions, and zero-noise extrapolation.
- `tests/test_keras_layers.py`: Keras 3 graph-mode regression suite for
  `KerasQBatchNormalization` (fit under tracing, moving-stat updates,
  deterministic inference).

#### Documentation
- New **complete tutorial** (`docs/tutorials/complete-tutorial.md`):
  ten chapters — conventions, states, circuits, simulation, noise,
  algorithms, VQE/QAOA, Keras integration, the plain-TF neural stack,
  and visualization/performance — every code block executed and
  verified.
- `docs/getting-started.md`, `docs/api-reference.md`,
  `docs/tutorials/advanced-tutorials.md` and `README.md` repaired:
  every Python block now runs against the current API (verified with a
  block-by-block execution harness). Network-dependent dataset blocks
  are marked `doc-skip`; the MNIST/CIFAR examples were replaced with
  equivalent offline synthetic data.

- Optional Cython acceleration kernels (`_fast_gates`, `_fast_simulator`,
  `_fast_math`) with pure-Python fallbacks; the build never fails when a
  compiler or Cython is missing.
- `tests.yml` CI workflow running the suite across Python 3.10–3.12.
- Expanded regression suites: simulation semantics (per-shot
  measurement, gate-cache identity, compose/mapping), end-to-end
  algorithm correctness (Grover, QFT/IQFT, QPE/IPE, Shor, VQE, QAOA),
  TensorFlow/Keras integration (lazy build, fit + gradient flow
  through quantum layers, data encoding) — `tensorflow` tests are
  skipped automatically when TF is not installed — and a quantum-ML
  integration suite driven by a text-classification workload (plain-TF
  quantum stack, hybrid Keras training, QClassifier/QRegressor, all
  encodings, all optimizers, quantum conv/pool on character glyphs,
  activations, model zoo). Full-length training runs live in
  `examples/qml_text_classifier.py`.
- New `tensorflow`/`keras` extra (`quantumflow[tf]`); TensorFlow is no
  longer a hard dependency (all integration code was already lazily
  imported).

#### Packaging
- `pip install` from a source checkout/zip crashed with
  ``ModuleNotFoundError: No module named 'numpy'``: `setup.py` imported
  numpy at module level, but pip's isolated build environment only
  contains the `[build-system]` requirements. numpy is now imported
  defensively (missing numpy in the build env simply skips the optional
  Cython kernels — pure-Python fallbacks are used at runtime), and
  `[build-system] requires` is reduced to `setuptools`/`wheel`
  (Cython is likewise optional, not a hard build requirement).
  Verified end-to-end: `pip install .` in a clean environment with full
  build isolation now builds and installs, and the installed package
  passes the Bell/Grover/noise/ZNE smoke checks (numpy 2.x compatible).

### Changed
- Package version bumped to 0.2.0; `requires-python = ">=3.9"`
  (verified with vermin).
- `setup.py` no longer uses `-ffast-math`/`-march=native` (unsafe for
  numerical code / non-portable binaries).

## [0.1.0] - 2025-04-21

### Added

#### Core Module
- `QuantumCircuit` with full gate API (50+ gates), compose, tensor, inverse, QASM export
- `Statevector` with measurement, sampling, expectation, tensor product
- `DensityMatrix` with partial trace, Kraus evolution, fidelity, entropy
- `QuantumRegister` and `ClassicalRegister` with indexing and slicing
- 50+ quantum gates: Pauli, Clifford, rotation, controlled, parameterized, multi-qubit
- `UnitaryGate`, `ControlledGate`, `ParameterizedGate`, `CompositeGate`
- `Operation`, `CompositeOperation`, `Barrier`, `Reset`, `ConditionalOperation`

#### Simulation Module
- `StatevectorSimulator` with einsum-based gate application and batch simulation
- `DensityMatrixSimulator` with Kraus operator evolution and noise support
- `MPSimulator` with Matrix Product State representation and bond dimension truncation
- `SimulationResult` with counts, probabilities, memory, metadata, histogram plotting
- Parameter gradient computation via parameter-shift rule
- `BackendConfig` for simulator configuration
- `SimulatorFactory` for creating simulators by name

#### Neural Network Module
- `QuantumNNLayer` with 5 encodings and 5 variational forms
- `VariationalCircuit`, `AngleEncoder`, `AmplitudeEncoder`
- `HardwareEfficientAnsatz`, `StronglyEntanglingAnsatz`
- `QuantumDense`, `QuantumDenseWithMeasurement`
- `QuantumConv2D`, `QuantumPool2D`
- `QuantumReLU`, `QuantumSigmoid`, `QuantumTanh`, `QuantumSoftmax`, `QuantumSwish`

#### TensorFlow Integration
- `QDenseLayer`, `QConvLayer`, `QVariationalLayer` with `@tf.custom_gradient`
- `QBatchNormLayer`, `QAttentionLayer`, `QResidualLayer`
- `QFeatureMapLayer` with 5 feature map types
- `QMeasurementLayer` with 3 measurement strategies
- `QClassifier`, `QRegressor`, `QAutoencoder`, `QGAN`
- `QTransferLearningModel`, `QHybridModel`
- `ParameterShiftOptimizer`, `NaturalGradientOptimizer`
- `QuantumAdam`, `QuantumLAMB`, `QuantumSGD`, `SpsaOptimizer`

#### Keras Integration
- 10 Keras 3-compatible quantum layers
- `KerasQuantumClassifier`, `KerasQuantumRegressor`, `KerasQNN`
- `KerasQuantumAutoencoder`, `KerasHybridModel`
- `KerasQuantumGAN` with `train_step()`, `KerasQuantumVAE`
- `KerasTransferLearning` with 3 fine-tuning strategies
- `QuantumDataEncoder`, `QuantumDataAugmenter`
- `QuantumNormalizer`, `QuantumFeatureScaler`

#### Algorithms Module
- `GroverSearch` with oracle construction, diffusion operator, optimal iterations
- `AmplitudeAmplification` (generalized Grover)
- `FixedPointAmplitudeAmplification`
- `QFT` with exact and approximate modes
- `InverseQFT`, `QuantumAdder`, `QuantumMultiplier`
- `ShorAlgorithm` with order finding and continued fractions
- `ModularExponentiation` circuit
- `PhaseEstimation`, `IterativePhaseEstimation`, `BayesianPhaseEstimation`
- `VQE` with COBYLA, SPSA, L-BFGS-B, Adam optimizers
- `Hamiltonian` with Pauli decomposition and molecular Hamiltonians
- `UCCSDAnsatz`, `HWEAnsatz`
- `QAOA`, `MaxCutQAOA`, `MISQAOA`, `TSPQAOA`

#### Noise Module
- `NoiseModel` with per-gate and per-qubit configuration
- `NoiseConfig`, `GateNoise`, `QubitNoise`
- `DepolarizingChannel`, `AmplitudeDampingChannel`, `PhaseDampingChannel`
- `BitFlipChannel`, `PhaseFlipChannel`, `PauliErrorChannel`
- `ThermalRelaxationChannel` (combined T1+T2+thermal)
- `ZeroNoiseExtrapolation` (Richardson, exponential, linear)
- `ProbabilisticErrorCancellation`
- `MeasurementErrorMitigation`
- `VirtualDistillation`, `SymmetryVerification`

#### Visualization Module
- `CircuitDrawer` with ASCII art, matplotlib, and LaTeX output
- `BlochSphere` with 3D matplotlib visualization

#### Utilities Module
- 25+ mathematical functions: kron, fidelity, trace_distance, purity, entropy
- State conversions: state_to_bloch, bloch_to_state
- Matrix generators: random_unitary, random_density_matrix
- Validators: is_hermitian, is_unitary, is_positive_semidefinite

#### Documentation & Testing
- Comprehensive README with API reference and tutorials
- Getting started guide
- 6 advanced tutorials
- Architecture documentation
- 25+ unit tests
- 8 demo examples

### Technical Details
- Total: 36,897 lines of Python code across 42 files
- Python 3.10+ support
- TensorFlow 2.14+ and Keras 3.0+ support
- Apache 2.0 license
