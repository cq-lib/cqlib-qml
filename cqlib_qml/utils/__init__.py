# cqlib_qml/utils/__init__.py
"""
Utility functions for quantum machine learning.

This module provides utility functions and tools used across the
cqlib-qml library. Currently, it includes gradient computation utilities
for parameterized quantum circuits.

The utilities are designed to be used internally by the library components
but may also be useful for advanced users extending the library.

Available Utilities:
    - grad_matrix: Gradient matrix computation for quantum gates
    - Operation extensions: Monkey-patched methods for gradient calculation

Examples:
    >>> from cqlib_qml.utils import grad_matrix
    >>> from cqlib.circuit import Circuit, Parameter
    >>>
    >>> # Create a parameterized circuit
    >>> circuit = Circuit(1)
    >>> theta = Parameter("theta")
    >>> circuit.ry(0, theta)
    >>>
    >>> # Get gradient matrix for an operation
    >>> op = circuit.operations[0]
    >>> grad = op.grad_matrix()
"""

from .grad_matrix import grad_matrix

__all__ = [
    "grad_matrix",
]
