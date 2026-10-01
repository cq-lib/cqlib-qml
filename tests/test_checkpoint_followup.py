"""Regressions for checkpoint schemas, numerical scale, and repeated use."""
from copy import deepcopy

import numpy as np
import pytest
from cqlib.circuit import Parameter
from cqlib.qis.state import Statevector

from cqlib_qml.ansatz import Ansatz, CRAML
from cqlib_qml.algorithms import QSVM
from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder


@pytest.mark.parametrize('scale', [1e200, 1e-200, 1e308, 1e-308])
@pytest.mark.parametrize('base', [np.array([1., 1.]), np.array([1., -1.]), np.array([1.+1j, -1.+1j])])
def test_amplitude_large_and_small_finite_inputs(scale, base):
    circuit = AmplitudeEncoder()(scale * base)[0]
    state = Statevector(circuit.num_qubits)
    state.apply_circuit(circuit)
    expected = base / np.linalg.norm(base)
    np.testing.assert_allclose(abs(np.vdot(expected, state.data)), 1., atol=1e-12)


@pytest.mark.parametrize('differentiator', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('batch', [None, np.array([[.2], [.4], [.7]])])
def test_repeated_backward_reuses_jacobian(differentiator, batch):
    a = Ansatz(1)
    a.ry(0, Parameter('t'))
    a.set_measurement(readouts=[0])
    a.set_differentiator(differentiator)
    a.assign_parameters({'t': .3})
    output = a.forward(batch)
    first = deepcopy(a.backward(np.ones_like(output)))
    second = deepcopy(a.backward(2 * np.ones_like(output)))
    if batch is None:
        assert second['t'] == pytest.approx(2 * first['t'])
        assert second['t'] == pytest.approx(-2 * np.sin(.3))
    else:
        np.testing.assert_allclose(second, 2 * first)
        np.testing.assert_allclose(second[:, 0], -2 * np.sin(batch[:, 0]))
    a.zero_grad()
    with pytest.raises(ValueError, match='No gradients'):
        a.backward(np.ones_like(output))


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
