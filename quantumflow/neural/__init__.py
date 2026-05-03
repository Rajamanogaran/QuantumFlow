"""
Quantum Neural Network Module
==============================

Provides quantum-enhanced neural network components that integrate with
classical deep learning frameworks (NumPy, TensorFlow/Keras).

Classes
-------
Quantum Neural Network Layers:
    :class:`~quantumflow.neural.qnn_layer.QuantumNNLayer`
        Trainable quantum circuit acting as a neural network layer.
    :class:`~quantumflow.neural.qnn_layer.VariationalLayer`
        Single variational block with configurable entanglement.
    :class:`~quantumflow.neural.qnn_layer.EncodingLayer`
        Data encoding into quantum states.

Variational Circuits:
    :class:`~quantumflow.neural.variational_circuit.VariationalCircuit`
        Full parameterized quantum circuit for variational algorithms.
    :class:`~quantumflow.neural.variational_circuit.AngleEncoder`
        Angle-based classical data encoder.
    :class:`~quantumflow.neural.variational_circuit.AmplitudeEncoder`
        Amplitude-based classical data encoder.
    :class:`~quantumflow.neural.variational_circuit.HardwareEfficientAnsatz`
        Hardware-efficient variational ansatz.
    :class:`~quantumflow.neural.variational_circuit.StronglyEntanglingAnsatz`
        Strongly entangling variational ansatz.

Quantum Activation Functions:
    :class:`~quantumflow.neural.quantum_activation.QuantumActivation`
        Base class for quantum activation functions.
    :class:`~quantumflow.neural.quantum_activation.QuantumReLU`
        Quantum ReLU approximation.
    :class:`~quantumflow.neural.quantum_activation.QuantumSigmoid`
        Quantum sigmoid.
    :class:`~quantumflow.neural.quantum_activation.QuantumTanh`
        Quantum tanh.
    :class:`~quantumflow.neural.quantum_activation.QuantumSoftmax`
        Quantum softmax via amplitude encoding.
    :class:`~quantumflow.neural.quantum_activation.QuantumSwish`
        Quantum approximation of swish/SiLU.

Quantum Dense Layers:
    :class:`~quantumflow.neural.quantum_dense.QuantumDense`
        Dense layer implemented with a quantum circuit (Keras-compatible).
    :class:`~quantumflow.neural.quantum_dense.QuantumDenseWithMeasurement`
        Dense layer with configurable measurement basis.

Quantum Convolution:
    :class:`~quantumflow.neural.quantum_conv.QuantumConv2D`
        2D convolution using quantum circuits.
    :class:`~quantumflow.neural.quantum_conv.QuantumPool2D`
        Quantum pooling layer for spatial reduction.

Typical Usage
-------------
    >>> from quantumflow.neural import QuantumNNLayer
    >>> layer = QuantumNNLayer(n_qubits=4, n_layers=3, encoding='angle')
    >>> circuit = layer.get_circuit([0.1, 0.2, 0.3, 0.4])

    >>> from quantumflow.neural import QuantumDense
    >>> import tensorflow as tf
    >>> qdense = QuantumDense(output_dim=4, n_qubits=3, n_layers=2)
    >>> output = qdense(tf.constant([[1.0, 2.0, 3.0]]))
"""

from quantumflow.neural.qnn_layer import (
    QuantumNNLayer,
    VariationalLayer,
    EncodingLayer,
)
from quantumflow.neural.variational_circuit import (
    VariationalCircuit,
    AngleEncoder,
    AmplitudeEncoder,
    HardwareEfficientAnsatz,
    StronglyEntanglingAnsatz,
)
from quantumflow.neural.quantum_activation import (
    QuantumActivation,
    QuantumReLU,
    QuantumSigmoid,
    QuantumTanh,
    QuantumSoftmax,
    QuantumSwish,
)
from quantumflow.neural.quantum_dense import (
    QuantumDense,
    QuantumDenseWithMeasurement,
)
from quantumflow.neural.quantum_conv import (
    QuantumConv2D,
    QuantumPool2D,
)

__all__ = [
    # QNN Layer
    "QuantumNNLayer",
    "VariationalLayer",
    "EncodingLayer",
    # Variational Circuits
    "VariationalCircuit",
    "AngleEncoder",
    "AmplitudeEncoder",
    "HardwareEfficientAnsatz",
    "StronglyEntanglingAnsatz",
    # Quantum Activations
    "QuantumActivation",
    "QuantumReLU",
    "QuantumSigmoid",
    "QuantumTanh",
    "QuantumSoftmax",
    "QuantumSwish",
    # Quantum Dense
    "QuantumDense",
    "QuantumDenseWithMeasurement",
    # Quantum Convolution
    "QuantumConv2D",
    "QuantumPool2D",
]
