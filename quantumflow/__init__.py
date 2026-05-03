"""
QuantumFlow — Advanced Quantum Computing Framework
===================================================

A comprehensive Python library for quantum computing with seamless
TensorFlow/Keras integration for Quantum Neural Networks (QNNs).

Features:
    - Full quantum circuit construction and simulation
    - 50+ built-in quantum gates (unitary, controlled, parameterized)
    - Statevector and density matrix simulators with GPU acceleration
    - Quantum Neural Network layers compatible with TensorFlow/Keras
    - Variational algorithms (VQE, QAOA, variational classifiers)
    - Famous quantum algorithms (Shor, Grover, QFT, QPE)
    - Noise models and quantum error mitigation
    - Circuit visualization (text, matplotlib, LaTeX)
    - TensorFlow quantum-aware optimizers

Quick Start:
    >>> import quantumflow as qf
    >>> qc = qf.QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> result = qf.StatevectorSimulator().run(qc)
    >>> result.statevector

TensorFlow Integration:
    >>> import quantumflow.tensorflow as qf_tf
    >>> import tensorflow as tf
    >>> qnn = qf_tf.QuantumDenseLayer(4, n_qubits=2, n_layers=3)
    >>> model = tf.keras.Sequential([qnn, tf.keras.layers.Dense(1)])
"""

__version__ = "0.1.0"
__author__ = "QuantumFlow Team"
__license__ = "Apache-2.0"

# Core imports
from quantumflow.core.circuit import QuantumCircuit
from quantumflow.core.gate import (
    Gate, UnitaryGate, ControlledGate, ParameterizedGate,
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, CNOTGate, CZGate, SwapGate,
    ToffoliGate, FredkinGate, PhaseGate, UGate, U3Gate,
    RXXGate, RYYGate, RZZGate, RZXGate, CRXGate, CRYGate, CRZGate,
    Measurement,
)
from quantumflow.core.qubit import Qubit, QubitState
from quantumflow.core.register import QuantumRegister, ClassicalRegister
from quantumflow.core.state import QuantumState, Statevector, DensityMatrix
from quantumflow.core.operation import Operation, CompositeOperation

# Optional imports — modules not yet available will be silently skipped.
# This allows the core module to be used independently.

def _safe_import(module_path: str, names: list, target_globals: dict) -> None:
    """Try to import *names* from *module_path* into *target_globals*.

    Silently skips on :class:`ImportError` (module not yet available).
    """
    try:
        mod = __import__(module_path, fromlist=names)
        for name in names:
            target_globals[name] = getattr(mod, name)
    except ImportError:
        pass

# Simulation imports
_simulation_names = [
    "Simulator", "StatevectorSimulator", "DensityMatrixSimulator",
    "MPSimulator", "StatevectorBackend", "DensityMatrixBackend",
]
_safe_import("quantumflow.simulation.simulator",
             ["Simulator", "StatevectorSimulator", "DensityMatrixSimulator", "MPSimulator"],
             globals())
_safe_import("quantumflow.simulation.statevector", ["StatevectorBackend"], globals())
_safe_import("quantumflow.simulation.density_matrix", ["DensityMatrixBackend"], globals())

# Neural network imports
_neural_names = [
    "QuantumNNLayer", "VariationalLayer", "VariationalCircuit",
    "AngleEncoder", "AmplitudeEncoder", "QuantumActivation",
    "QuantumDense", "QuantumConv2D",
]
_safe_import("quantumflow.neural.qnn_layer",
             ["QuantumNNLayer", "VariationalLayer"], globals())
_safe_import("quantumflow.neural.variational_circuit",
             ["VariationalCircuit", "AngleEncoder", "AmplitudeEncoder"], globals())
_safe_import("quantumflow.neural.quantum_activation", ["QuantumActivation"], globals())
_safe_import("quantumflow.neural.quantum_dense", ["QuantumDense"], globals())
_safe_import("quantumflow.neural.quantum_conv", ["QuantumConv2D"], globals())

# Algorithm imports
_safe_import("quantumflow.algorithms.grover", ["GroverSearch"], globals())
_safe_import("quantumflow.algorithms.qft", ["QFT", "InverseQFT"], globals())
_safe_import("quantumflow.algorithms.shor", ["ShorAlgorithm"], globals())
_safe_import("quantumflow.algorithms.qpe", ["PhaseEstimation"], globals())
_safe_import("quantumflow.algorithms.vqe", ["VQE"], globals())
_safe_import("quantumflow.algorithms.qaoa", ["QAOA"], globals())

# Noise imports
_safe_import("quantumflow.noise.noise_model", ["NoiseModel"], globals())
_safe_import("quantumflow.noise.error_channels",
             ["DepolarizingChannel", "AmplitudeDampingChannel",
              "PhaseDampingChannel", "BitFlipChannel",
              "PhaseFlipChannel", "PauliErrorChannel"], globals())
_safe_import("quantumflow.noise.error_mitigation",
             ["ZeroNoiseExtrapolation", "ProbabilisticErrorCancellation",
              "MeasurementErrorMitigation"], globals())

# Visualization imports
_safe_import("quantumflow.visualization.circuit_drawer", ["CircuitDrawer"], globals())
_safe_import("quantumflow.visualization.bloch_sphere", ["BlochSphere"], globals())

# Utility imports
_safe_import("quantumflow.utils.math",
             ["kron", "tensor_product", "partial_trace", "density_from_statevector",
              "fidelity", "trace_distance", "purity", "von_neumann_entropy",
              "expectation_value"], globals())

# Build __all__ dynamically from what was successfully imported
_all_core = [
    # Version info
    "__version__", "__author__", "__license__",
    # Core
    "QuantumCircuit", "QuantumRegister", "ClassicalRegister",
    "Qubit", "QubitState", "QuantumState", "Statevector", "DensityMatrix",
    "Gate", "UnitaryGate", "ControlledGate", "ParameterizedGate",
    "HGate", "XGate", "YGate", "ZGate", "SGate", "TGate",
    "RXGate", "RYGate", "RZGate", "CNOTGate", "CZGate", "SwapGate",
    "ToffoliGate", "FredkinGate", "PhaseGate", "UGate", "U3Gate",
    "RXXGate", "RYYGate", "RZZGate", "RZXGate", "CRXGate", "CRYGate", "CRZGate",
    "Measurement", "Operation", "CompositeOperation",
]

_all_optional = _simulation_names + _neural_names + [
    "GroverSearch", "QFT", "InverseQFT", "ShorAlgorithm",
    "PhaseEstimation", "VQE", "QAOA",
    "NoiseModel", "DepolarizingChannel", "AmplitudeDampingChannel",
    "PhaseDampingChannel", "BitFlipChannel", "PhaseFlipChannel",
    "PauliErrorChannel", "ZeroNoiseExtrapolation",
    "ProbabilisticErrorCancellation", "MeasurementErrorMitigation",
    "CircuitDrawer", "BlochSphere",
    "kron", "tensor_product", "partial_trace", "density_from_statevector",
    "fidelity", "trace_distance", "purity", "von_neumann_entropy",
    "expectation_value",
]

__all__ = [name for name in _all_core + _all_optional if name in globals()]
