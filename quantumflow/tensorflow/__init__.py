"""
TensorFlow Integration Module
==============================

Seamless integration of QuantumFlow quantum circuits with TensorFlow/Keras.

Provides quantum-enhanced Keras layers, pre-built quantum models, and
quantum-aware optimizers that can be used directly in TensorFlow pipelines.

Classes
-------
Quantum Layers (layers.py):
    :class:`~quantumflow.tensorflow.layers.QConvLayer`
        Quantum convolution layer for TensorFlow.
    :class:`~quantumflow.tensorflow.layers.QDenseLayer`
        Quantum dense layer with parameter-shift gradients.
    :class:`~quantumflow.tensorflow.layers.QVariationalLayer`
        General variational quantum layer.
    :class:`~quantumflow.tensorflow.layers.QBatchNormLayer`
        Quantum-inspired batch normalization.
    :class:`~quantumflow.tensorflow.layers.QAttentionLayer`
        Quantum attention mechanism.
    :class:`~quantumflow.tensorflow.layers.QResidualLayer`
        Quantum residual block with skip connections.
    :class:`~quantumflow.tensorflow.layers.QFeatureMapLayer`
        Quantum feature map / kernel layer.
    :class:`~quantumflow.tensorflow.layers.QMeasurementLayer`
        Quantum measurement with classical readout.

Pre-built Models (models.py):
    :class:`~quantumflow.tensorflow.models.QClassifier`
        Quantum variational classifier (binary / multi-class).
    :class:`~quantumflow.tensorflow.models.QRegressor`
        Quantum variational regressor.
    :class:`~quantumflow.tensorflow.models.QAutoencoder`
        Quantum autoencoder with trash qubit detection.
    :class:`~quantumflow.tensorflow.models.QGAN`
        Quantum Generative Adversarial Network.
    :class:`~quantumflow.tensorflow.models.QTransferLearningModel`
        Transfer learning with classical backbone + quantum head.
    :class:`~quantumflow.tensorflow.models.QHybridModel`
        Hybrid classical-quantum model with arbitrary layer sequences.

Quantum-Aware Optimizers (optimizers.py):
    :class:`~quantumflow.tensorflow.optimizers.QuantumOptimizer`
        Base quantum optimizer class.
    :class:`~quantumflow.tensorflow.optimizers.ParameterShiftOptimizer`
        Exact gradient computation via parameter-shift rule.
    :class:`~quantumflow.tensorflow.optimizers.NaturalGradientOptimizer`
        Natural gradient descent with Fubini-Study metric.
    :class:`~quantumflow.tensorflow.optimizers.QuantumAdam`
        Adam adapted for quantum parameter landscapes.
    :class:`~quantumflow.tensorflow.optimizers.QuantumLAMB`
        LAMB optimizer for quantum batch training.
    :class:`~quantumflow.tensorflow.optimizers.QuantumSGD`
        SGD with shot-based quantum gradient estimation.
    :class:`~quantumflow.tensorflow.optimizers.SpsaOptimizer`
        Gradient-free SPSA optimizer for noisy settings.
    :class:`~quantumflow.tensorflow.optimizers.GradientFactory`
        Factory for creating gradient estimators.

Typical Usage
-------------
    >>> import tensorflow as tf
    >>> from quantumflow.tensorflow import QDenseLayer, QClassifier
    >>>
    >>> # Individual layer usage
    >>> qdense = QDenseLayer(units=4, n_qubits=3, n_layers=2)
    >>> x = tf.random.normal((32, 3))
    >>> y = qdense(x)
    >>>
    >>> # Pre-built model
    >>> model = QClassifier(n_qubits=4, n_classes=2, n_layers=3)
    >>> model.compile(optimizer='adam', loss='binary_crossentropy')
    >>> model.fit(x_train, y_train, epochs=10)

Notes
-----
All layers are fully compatible with the Keras ``Layer`` API and can be
combined with any standard TensorFlow/Keras layers in a ``Sequential`` or
functional model.

Gradient computation uses the parameter-shift rule for exact quantum
gradients (no finite-difference approximation), ensuring numerically exact
backpropagation through quantum circuits.
"""

# Layer exports
from quantumflow.tensorflow.layers import (
    QConvLayer,
    QDenseLayer,
    QVariationalLayer,
    QBatchNormLayer,
    QAttentionLayer,
    QResidualLayer,
    QFeatureMapLayer,
    QMeasurementLayer,
)

# Model exports
from quantumflow.tensorflow.models import (
    QClassifier,
    QRegressor,
    QAutoencoder,
    QGAN,
    QTransferLearningModel,
    QHybridModel,
)

# Optimizer exports
from quantumflow.tensorflow.optimizers import (
    QuantumOptimizer,
    ParameterShiftOptimizer,
    NaturalGradientOptimizer,
    QuantumAdam,
    QuantumLAMB,
    QuantumSGD,
    SpsaOptimizer,
    GradientFactory,
)

__all__ = [
    # Layers
    "QConvLayer",
    "QDenseLayer",
    "QVariationalLayer",
    "QBatchNormLayer",
    "QAttentionLayer",
    "QResidualLayer",
    "QFeatureMapLayer",
    "QMeasurementLayer",
    # Models
    "QClassifier",
    "QRegressor",
    "QAutoencoder",
    "QGAN",
    "QTransferLearningModel",
    "QHybridModel",
    # Optimizers
    "QuantumOptimizer",
    "ParameterShiftOptimizer",
    "NaturalGradientOptimizer",
    "QuantumAdam",
    "QuantumLAMB",
    "QuantumSGD",
    "SpsaOptimizer",
    "GradientFactory",
]
