# cqlib_qml/__init__.py
"""
Cqlib-QML: Quantum machine learning package based on Cqlib.

This package provides quantum machine learning building blocks on top of
the `cqlib <https://github.com/cq-lib/cqlib>`_ quantum computing library,
including data encoders, parameterized quantum circuits (ansatz),
differentiators, neural network models, and quantum algorithms.

Available Subpackages:
    - algorithms: Quantum machine learning algorithms (QKM, QSVM, VQC).
    - ansatz: Parameterized quantum circuits (HEAnsatz, BasicQNN, CRADL, CRAML).
    - data: Classical data loading and preprocessing utilities.
    - differentiator: Gradient computation strategies (adjoint, parameter-shift).
    - encoder: Quantum data encoders (Amplitude, Angle, Basis, FRQI, NEQR).
    - layer: Hybrid quantum-classical layers (Linear, activations).
    - models: Quantum neural network models (QNN, HQNN).
    - utils: Gradient matrix computation and other utilities.

Examples:
    >>> import numpy as np
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.models import QNN
    >>>
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> ansatz.set_measurement(readouts=[0])
    >>> model = QNN(ansatz=ansatz)
"""

from . import algorithms, ansatz, data, differentiator, encoder, layer, models, utils
from .loss import BCELoss, CrossEntropy, HingeLoss, LossFun, MSELoss, SoftmaxCrossEntropy
from .optimizer import AdaGrad, Adam, OptimizerBase, OptimizerInitializer, RMSProp, SGD

__version__ = "0.1.0-beta.3"

__all__ = [
    "algorithms",
    "ansatz",
    "data",
    "differentiator",
    "encoder",
    "layer",
    "models",
    "utils",
    "LossFun",
    "HingeLoss",
    "MSELoss",
    "BCELoss",
    "CrossEntropy",
    "SoftmaxCrossEntropy",
    "OptimizerBase",
    "OptimizerInitializer",
    "SGD",
    "AdaGrad",
    "RMSProp",
    "Adam",
    "__version__",
]
