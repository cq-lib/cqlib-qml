# cqlib_qml/layer/__init__.py
"""
Classical neural network layers for hybrid quantum-classical models.

This module provides classical neural network layers that can be combined
with quantum circuits (ansätze) to build hybrid quantum-classical models.
These layers support automatic differentiation and are designed to be
compatible with the quantum components.

Available Layers:
    - Layer: Base class for all neural network layers.
    - Linear: Fully connected (dense) layer with optional activation.
    - Activation: Activation functions for neural networks.

The layers support:
    - Automatic differentiation (backpropagation)
    - Parameter optimization
    - Checkpoint saving and loading
    - Freezing/unfreezing of trainable parameters

Examples:
    >>> from cqlib_qml.layer import Linear, ActivationInitializer
    >>>
    >>> # Create a linear layer with sigmoid activation
    >>> layer = Linear(in_dim=10, out_dim=5, act_fn="sigmoid")
    >>>
    >>> # Forward pass
    >>> X = np.random.randn(32, 10)
    >>> output = layer.forward(X)
    >>>
    >>> # Backward pass
    >>> dLdy = np.random.randn(32, 5)
    >>> dX = layer.backward(dLdy)
"""

from .layer import Layer
from .linear import Linear
from .activation import (
    ActivationBase,
    ActivationInitializer,
    Sigmoid,
    ReLU,
    Tanh,
    SoftPlus,
)

__all__ = [
    "Layer",
    "Linear",
    "ActivationBase",
    "ActivationInitializer",
    "Sigmoid",
    "ReLU",
    "Tanh",
    "SoftPlus",
]
