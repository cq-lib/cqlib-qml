"""Regressions for checkpoint schemas, numerical scale, and repeated use."""
from copy import deepcopy

import numpy as np
import pytest
from cqlib.circuit import Parameter
from cqlib.qis.state import Statevector

from cqlib_qml.ansatz import Ansatz, CRAML
from cqlib_qml.algorithms import QSVM
from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder


@pytest.mark.parametrize('gate,occurrences,expected', [('ry', 1, 2), ('ry', 2, 4), ('crx', 1, 8)])
def test_parameter_shift_evaluation_cost(gate, occurrences, expected):
    from unittest.mock import patch
    from cqlib.circuit import Circuit
    from cqlib_qml.differentiator import ParameterShiftDifferentiator
    circuit = Circuit(2)
    circuit.h(0)
    for _ in range(occurrences):
        if gate == 'ry':
            circuit.ry(1, Parameter('theta'))
        else:
            circuit.crx(0, 1, Parameter('theta'))
    with patch('cqlib_qml.differentiator.parameter_shift.Statevector', wraps=Statevector) as factory:
        ParameterShiftDifferentiator().run(circuit, {'theta': .3}, readouts=[1])
        assert factory.call_count == expected
