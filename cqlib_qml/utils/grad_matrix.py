"""Exact derivatives of cqlib gate matrices used by adjoint differentiation.

Each parameter of the supported standard gates has matrix frequencies 0,
1/2 and/or 1. The symmetric shifts below differentiate all these frequencies
exactly, including the constant inactive block of controlled gates. Native
matrices preserve cqlib's parameter and qubit ordering.
"""
import numpy as np
from cqlib.circuit import ValueOperation


def grad_matrix(self):
    if self.num_params == 0:
        return None
    if not (self.instruction.is_standard or self.instruction.is_mcgate):
        raise ValueError("Parameterized custom instructions do not support adjoint differentiation")
    gradients = []
    for index in range(self.num_params):
        derivative = np.zeros_like(self.matrix(), dtype=np.complex128)
        for shift, coefficient in ((np.pi / 2, 0.5), (np.pi, (1 - np.sqrt(2)) / 4)):
            positive = list(self.params)
            negative = list(self.params)
            positive[index] += shift
            negative[index] -= shift
            plus = ValueOperation(self.instruction, self.qubits, positive).matrix()
            minus = ValueOperation(self.instruction, self.qubits, negative).matrix()
            derivative += coefficient * (plus - minus)
        gradients.append(derivative)
    return gradients
