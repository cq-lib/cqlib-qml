"""Numerical, ownership and failure-state regressions from the final audit."""
import importlib

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString, Phase
from cqlib.qis.state import Statevector

from cqlib_qml.algorithms import QKM, QSVM, VQC
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.data import DataLoader, Dataset
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator
from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder, FRQI, NEQR
from cqlib_qml.loss import MSELoss, SoftmaxCrossEntropy
from cqlib_qml.models import QNN, HQNN
from cqlib_qml.optimizer import SGD
from cqlib_qml.scheduler import KingScheduler


def differentiate(diff, circuit, bindings, ham):
    kwargs = {'hamiltonians': ham}
    if isinstance(diff, AdjointDifferentiator):
        state = Statevector(circuit.num_qubits)
        state.apply_circuit(circuit.assign_parameters(bindings))
        kwargs['state_vector'] = state.data
    return diff.run(circuit, bindings, **kwargs)['t'][0]


def test_adjoint_preserves_near_zero_direction_with_large_chain_factor():
    circuit = Circuit(1)
    circuit.ry(0, 1e-13 + 1e13 * Parameter('t'))
    ham = Hamiltonian.from_list([(PauliString.from_str('I'), .5),
                                 (PauliString.from_str('Z'), -.5)])
    actual = differentiate(AdjointDifferentiator(), circuit, {'t': 0.}, ham)
    np.testing.assert_allclose(actual, .5, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('scale', [1e-200, 1e-13, 1., 1e200])
def test_adjoint_normalization_is_scale_invariant(scale):
    circuit = Circuit(1)
    circuit.ry(0, Parameter('t'))
    ham = Hamiltonian.from_list([(PauliString.from_str('X'), scale)])
    actual = differentiate(AdjointDifferentiator(), circuit, {'t': .3}, ham)
    np.testing.assert_allclose(actual / scale, np.cos(.3), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('factory', [AdjointDifferentiator, ParameterShiftDifferentiator])
@pytest.mark.parametrize('terms,expected', [
    ([('X', 1e-10)], 1e-10 * np.cos(.3)),
    ([('X', 2e-11), ('X', 3e-11)], 5e-11 * np.cos(.3)),
    ([('X', 1.), ('X', -1.)], 0.),
    ([], 0.),
])
def test_observable_terms_are_preserved(factory, terms, expected):
    circuit = Circuit(1)
    circuit.ry(0, Parameter('t'))
    ham = Hamiltonian(1)
    for pauli, coefficient in terms:
        ham.add_term(PauliString.from_str(pauli), coefficient)
    before = repr(ham.terms)
    actual = differentiate(factory(), circuit, {'t': .3}, ham)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-25)
    assert repr(ham.terms) == before


@pytest.mark.parametrize('factory', [AdjointDifferentiator, ParameterShiftDifferentiator])
@pytest.mark.parametrize('exponent', [0, 1, 2, 3])
def test_observable_pauli_phase_is_preserved(factory, exponent):
    pauli = PauliString.from_str('X')
    pauli.phase = Phase(exponent)
    ham = Hamiltonian.from_list([(pauli, 1 / pauli.phase.to_complex())])
    before = repr(ham.terms)
    circuit = Circuit(1)
    circuit.ry(0, Parameter('t'))
    np.testing.assert_allclose(differentiate(factory(), circuit, {'t': .3}, ham),
                               np.cos(.3), rtol=1e-12, atol=1e-14)
    assert repr(ham.terms) == before


def training_data():
    return np.array([[.1, .2], [.3, .4], [.9, 1.], [1.2, 1.3]]), np.array([0, 0, 1, 1])




def vqc():
    return VQC(HEAnsatz(2, 1, layers=['RY']), AngleEncoder(), readouts=[0, 1],
               loss='CrossEntropy', epochs=1, verbose=False)
















@pytest.mark.parametrize('factory,grayscale,value', [
    (FRQI, 2, 0), (FRQI, 2, 1), (FRQI, 4, 2),
    (NEQR, 2, 0), (NEQR, 2, 1), (NEQR, 4, 2), (NEQR, 8, 7),
])
def test_single_pixel_qic_matches_normal_encoding(factory, grayscale, value):
    states = []
    for qic in [False, True]:
        encoder = factory(1, grayscale=grayscale)
        circuit = encoder(np.array([[value]]), use_qic=qic)
        state = Statevector(circuit.num_qubits)
        state.apply_circuit(circuit)
        states.append(state.data)
    np.testing.assert_allclose(abs(np.vdot(*states)) ** 2, 1., atol=1e-12)
