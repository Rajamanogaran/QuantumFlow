"""
Keras Integration Module
========================

Provides Keras 3 / tf.keras compatible quantum layers, pre-built models,
and quantum-aware data preprocessing utilities for the QuantumFlow
framework.

Layers
------
* :class:`KerasQuantumLayer` — Base quantum layer for Keras.
* :class:`KerasQDense` — Dense layer with quantum circuit.
* :class:`KerasQConv2D` — 2D convolution with quantum processing.
* :class:`KerasQAttention` — Quantum self-attention layer.
* :class:`KerasQVariational` — Variational quantum layer.
* :class:`KerasQBatchNormalization` — Quantum batch normalization.
* :class:`KerasQLayerNormalization` — Quantum layer normalization.
* :class:`KerasQDropout` — Quantum dropout (noise channels).
* :class:`KerasQPooling2D` — Quantum pooling for 2D inputs.
* :class:`KerasQFlatten` — Flatten quantum measurement results.

Models
------
* :class:`KerasQuantumClassifier` — Quantum classifier model.
* :class:`KerasQuantumRegressor` — Quantum regression model.
* :class:`KerasQNN` — Quantum Neural Network model builder.
* :class:`KerasQuantumAutoencoder` — Quantum autoencoder.
* :class:`KerasHybridModel` — Hybrid classical-quantum model.
* :class:`KerasQuantumGAN` — Quantum Generative Adversarial Network.
* :class:`KerasQuantumVAE` — Quantum Variational Autoencoder.
* :class:`KerasTransferLearning` — Transfer learning helper.

Preprocessing
-------------
* :class:`QuantumDataEncoder` — Encode classical data for quantum layers.
* :class:`QuantumDataAugmenter` — Augment training data.
* :class:`QuantumNormalizer` — Keras preprocessing layer for normalization.
* :class:`QuantumFeatureScaler` — Scale features to appropriate ranges.
"""

from quantumflow.keras.layers import (
    KerasQuantumLayer,
    KerasQDense,
    KerasQConv2D,
    KerasQAttention,
    KerasQVariational,
    KerasQBatchNormalization,
    KerasQLayerNormalization,
    KerasQDropout,
    KerasQPooling2D,
    KerasQFlatten,
)
from quantumflow.keras.models import (
    KerasQuantumClassifier,
    KerasQuantumRegressor,
    KerasQNN,
    KerasQuantumAutoencoder,
    KerasHybridModel,
    KerasQuantumGAN,
    KerasQuantumVAE,
    KerasTransferLearning,
)
from quantumflow.keras.preprocessing import (
    QuantumDataEncoder,
    QuantumDataAugmenter,
    QuantumNormalizer,
    QuantumFeatureScaler,
)

__all__ = [
    # Layers
    "KerasQuantumLayer",
    "KerasQDense",
    "KerasQConv2D",
    "KerasQAttention",
    "KerasQVariational",
    "KerasQBatchNormalization",
    "KerasQLayerNormalization",
    "KerasQDropout",
    "KerasQPooling2D",
    "KerasQFlatten",
    # Models
    "KerasQuantumClassifier",
    "KerasQuantumRegressor",
    "KerasQNN",
    "KerasQuantumAutoencoder",
    "KerasHybridModel",
    "KerasQuantumGAN",
    "KerasQuantumVAE",
    "KerasTransferLearning",
    # Preprocessing
    "QuantumDataEncoder",
    "QuantumDataAugmenter",
    "QuantumNormalizer",
    "QuantumFeatureScaler",
]
