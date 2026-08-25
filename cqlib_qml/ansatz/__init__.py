# cqlib_qml/ansatz/__init__.py
"""
Parameterized quantum circuits (ansatz) for variational quantum algorithms.

This module provides a collection of parameterized quantum circuit architectures
(ansatz) for use in variational quantum algorithms, quantum neural networks,
and other quantum machine learning models.

The ansatz is a key component that defines the structure of the parameterized
quantum circuit. Different ansatz architectures offer different trade-offs
between expressivity, trainability, and hardware efficiency.

Available Ansatz:
    - Ansatz: Base class for all ansatz implementations.
    - BasicQNN: Basic QNN ansatz with alternating 2-qubit gates.
    - CRADL: Color-Readout-Alternating-Double-Layer architecture.
    - CRAML: Color-Readout-Alternating-Mixed-Layer architecture.
    - HEAnsatz: Hardware-Efficient ansatz with alternating single and 2-qubit gates.

Examples:
    >>> from cqlib_qml.ansatz import HEAnsatz, BasicQNN, CRADL
    >>>
    >>> # Hardware-Efficient ansatz
    >>> he = HEAnsatz(n_qubits=4, d=3, layers=["RY", "CX"])
    >>>
    >>> # Basic QNN ansatz
    >>> basic = BasicQNN(n_qubits=4, layers=["XX", "ZZ"])
    >>>
    >>> # CRADL ansatz for image data
    >>> cradl = CRADL(n_qubits=5, layers=2)
"""

from .ansatz import Ansatz
from .BasicQNN import BasicQNN
from .CRADL import CRADL
from .CRAML import CRAML
from .HE_ansatz import HEAnsatz

__all__ = [
    "Ansatz",
    "BasicQNN",
    "CRADL",
    "CRAML",
    "HEAnsatz",
]
