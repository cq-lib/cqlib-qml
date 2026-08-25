# cqlib_qml/differentiator/__init__.py
"""
Gradient computation methods for parameterized quantum circuits.

This module provides different differentiation methods for computing
gradients of parameterized quantum circuits. The gradients are essential
for training variational quantum algorithms and quantum neural networks.

Available Differentiators:
    - AdjointDifferentiator: Uses the adjoint method for efficient gradient
      computation. Suitable for classical simulation.
    - ParameterShiftDifferentiator: Uses the parameter shift rule for
      gradient computation. More suitable for hardware execution.

The choice of differentiator affects both computational efficiency and
accuracy. The adjoint method is generally faster for simulation, while
the parameter shift method is more hardware-compatible.

References:
    - Parameter Shift Rule: Mitarai et al. (2018). "Quantum circuit learning"
    - Adjoint Method: Jones (2020). "Efficient quantum gradient computation"

Examples:
    >>> from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator
    >>> from cqlib.circuit import Circuit, Parameter
    >>>
    >>> # Create a parameterized circuit
    >>> circuit = Circuit(1)
    >>> theta = Parameter("theta")
    >>> circuit.ry(0, theta)
    >>>
    >>> # Adjoint differentiator (faster for simulation)
    >>> adjoint = AdjointDifferentiator()
    >>> grads = adjoint.run(circuit, {"theta": 0.5}, readouts=[0])
    >>>
    >>> # Parameter shift differentiator (hardware-friendly)
    >>> ps = ParameterShiftDifferentiator(shift=np.pi/2)
    >>> grads = ps.run(circuit, {"theta": 0.5}, readouts=[0])
"""

from .adjoint import AdjointDifferentiator
from .parameter_shift import ParameterShiftDifferentiator

__all__ = [
    "AdjointDifferentiator",
    "ParameterShiftDifferentiator",
]
