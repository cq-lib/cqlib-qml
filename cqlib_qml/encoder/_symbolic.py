"""Owned symbolic encoding composition; numerical encoders use their old path."""
from __future__ import annotations

import re

import numpy as np

from cqlib_qml._state import clone_state
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator


def positive_integer(value, name):
    """Accept Python/NumPy integers, excluding their boolean subclasses."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def compose_encoding(ansatz, *, num_features, num_qubits, input_prefix, build):
    """Build a fresh base Ansatz without copying training or subclass state.

    Circuit operands are mapped by source qubit position to canonical IDs. This
    also keeps execution and differentiation consistent for qubit permutations.
    """
    if not isinstance(ansatz, Ansatz):
        raise TypeError("ansatz must be an Ansatz")
    if not isinstance(input_prefix, str):
        raise TypeError("input_prefix must be a string")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", input_prefix) is None:
        raise ValueError("input_prefix must match [A-Za-z_][A-Za-z0-9_]*")
    if ansatz.num_qubits != num_qubits:
        raise ValueError("Encoding width must match ansatz.num_qubits")
    if {q.id for q in ansatz.qubits} != set(range(num_qubits)):
        raise ValueError("Qubit IDs must be a permutation of 0..Q-1")
    if ansatz._encoder is not None:
        raise ValueError("Remove the attached numerical encoder before composition")
    if ansatz.input_params:
        raise ValueError("The source Ansatz must not already have input parameters")
    weights = ansatz.weight_params
    if len(set(weights)) != len(weights) or set(weights) != set(ansatz.symbols):
        raise ValueError("Source weight roles must cover every circuit symbol")
    names = [f"{input_prefix}_{i}" for i in range(num_features)]
    if set(names) & set(ansatz.symbols):
        raise ValueError("Input parameter names conflict with source circuit symbols")

    result = Ansatz(num_qubits, random_state=ansatz._rng)
    result._circuit.compose(build(names))
    result._circuit.compose(ansatz._circuit, qubits=result.qubits)
    result.set_parameter_roles(input_params=names, weight_params=weights)
    result._weights = clone_state({name: ansatz._weights[name]
                                  for name in weights if name in ansatz._weights})
    if ansatz.readouts is not None:
        result.set_measurement(readouts=list(ansatz.readouts))
    elif ansatz.hams is not None:
        result.set_measurement(hams=clone_state(ansatz.hams))
    if isinstance(ansatz._differentiator, ParameterShiftDifferentiator):
        result.set_differentiator("parameter_shift", shift=ansatz._differentiator._shift)
    elif isinstance(ansatz._differentiator, AdjointDifferentiator):
        result.set_differentiator("adjoint")
    elif ansatz._differentiator is not None:
        raise TypeError("Unsupported source differentiator")
    return result
