"""Regressions discovered by checking QML against cqlib 2.0.0b3."""
from copy import deepcopy

import numpy as np
import pytest
from cqlib.circuit import Circuit, Instruction, Parameter, StandardGate, UnitaryGate
from cqlib.circuit.gates import Directive
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.differentiator import ParameterShiftDifferentiator
from cqlib_qml.encoder import AmplitudeEncoder, ZZFeatureEncoder
from cqlib_qml.models import Module
from cqlib_qml.optimizer import SGD


def ry_ansatz():
    ansatz = Ansatz(1)
    ansatz.ry(0, Parameter('t'))
    ansatz.set_measurement(readouts=[0])
    ansatz.assign_parameters({'t': 0.3})
    return ansatz


def test_parameter_shift_public_initial_state():
    ansatz = ry_ansatz()
    initial = np.array([np.cos(0.35), np.sin(0.35)])
    gradients = ParameterShiftDifferentiator().run(ansatz._circuit, {'t': 0.3},
                                                   readouts=[0], initial_state=initial)
    np.testing.assert_allclose(gradients['t'], [-np.sin(1.0)], atol=1e-12)
