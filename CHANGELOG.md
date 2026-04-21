# Changelog

All notable changes to QuantumFlow are documented in this file.

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
