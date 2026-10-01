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


def circuit_state(circuit):
    sv = Statevector(circuit.num_qubits)
    sv.apply_circuit(circuit)
    return sv.data


def test_parameter_shift_public_initial_state():
    ansatz = ry_ansatz()
    initial = np.array([np.cos(0.35), np.sin(0.35)])
    gradients = ParameterShiftDifferentiator().run(ansatz._circuit, {'t': 0.3},
                                                   readouts=[0], initial_state=initial)
    np.testing.assert_allclose(gradients['t'], [-np.sin(1.0)], atol=1e-12)


@pytest.mark.parametrize('length', [1, 2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize('kind', ['positive', 'signed', 'complex', 'sparse'])
def test_amplitude_encoding_matches_every_amplitude(length, kind):
    rng = np.random.default_rng(41)
    vector = np.arange(1, length + 1, dtype=float)
    if kind == 'signed':
        vector[::2] *= -1
    elif kind == 'complex':
        vector = rng.normal(size=length) + 1j * rng.normal(size=length)
    elif kind == 'sparse':
        vector = np.zeros(length, dtype=complex)
        vector[-1] = -1j if length > 1 else 1
    circuit = AmplitudeEncoder()(vector)[0]
    expected = np.zeros(1 << circuit.num_qubits, dtype=complex)
    expected[:length] = vector / np.linalg.norm(vector)
    actual = circuit_state(circuit)
    # Zero qubits have only global phase; all other cases match amplitudes,
    # including relative signs and phases, in native little-endian order.
    if length == 1:
        assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(1)
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_amplitude_batch_preserves_relative_phases():
    data = np.array([[1, 1j, -2], [-1j, 2, 1]])
    for vector, circuit in zip(data, AmplitudeEncoder()(data)):
        expected = np.pad(vector / np.linalg.norm(vector), (0, 1))
        np.testing.assert_allclose(circuit_state(circuit), expected, atol=1e-12)


@pytest.mark.parametrize('data', [np.array([]), np.zeros(4), np.array([np.nan, 1]),
                                 np.array([np.inf, 1]), np.zeros((1, 2, 2))])
def test_amplitude_rejects_invalid_data(data):
    with pytest.raises(ValueError):
        AmplitudeEncoder()(data)


def test_single_feature_circular_encoder_matches_linear():
    data = np.array([[0.3], [-0.8]])
    circular = ZZFeatureEncoder(entanglement='circular')(data)
    linear = ZZFeatureEncoder(entanglement='linear')(data)
    for actual, expected in zip(circular, linear):
        np.testing.assert_allclose(circuit_state(actual), circuit_state(expected))
