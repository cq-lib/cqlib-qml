# cqlib_qml/models/__init__.py
"""
Quantum neural network models for hybrid quantum-classical learning.

This module provides quantum and hybrid quantum-classical neural network
models that combine parameterized quantum circuits with classical layers.
These models support automatic differentiation and are designed for
various machine learning tasks.

Available Models:
    - Module: Base class for composing quantum and classical components.
    - QNN: Pure Quantum Neural Network using only quantum circuits.
    - HQNN: Hybrid Quantum-Classical Neural Network with a classical output layer.

The models support:
    - Automatic differentiation (backpropagation)
    - Parameter optimization
    - Batch processing of quantum circuits
    - Checkpoint saving and loading

Examples:
    >>> from cqlib_qml.models import QNN, HQNN
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.encoder import FRQI
    >>>
    >>> # Pure quantum neural network
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> ansatz.set_measurement(readouts=[0])
    >>> qnn = QNN(ansatz=ansatz)
    >>>
    >>> # Hybrid quantum-classical neural network
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> hqnn = HQNN(ansatz=ansatz, out_dim=3)
"""

from .module import Module
from .QNN import QNN
from .HQNN import HQNN

__all__ = [
    "Module",
    "QNN",
    "HQNN",
]
