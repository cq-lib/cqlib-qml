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


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('input_batch', [False, True])
@pytest.mark.parametrize('with_encoder', [False, True])
@pytest.mark.parametrize('with_state', [False, True])
def test_model_gradients_include_encoder_initial_state_and_batch(method, input_batch, with_encoder, with_state):
    ansatz = ry_ansatz()
    ansatz.set_differentiator(method)
    values = np.array([0.3, -0.4]) if input_batch else np.array([0.3, 0.3])
    encoder_angles = np.array([0.7, -0.2]) if with_encoder else np.zeros(2)
    initial_angles = np.array([0.2, 0.8]) if with_state else np.zeros(2)
    if with_encoder:
        encoders = []
        for angle in encoder_angles:
            encoder = Circuit(1)
            encoder.ry(0, angle)
            encoders.append(encoder)
        ansatz.add_encoder(encoders)
    quantum_state = np.array([np.array([np.cos(a / 2), np.sin(a / 2)]) for a in initial_angles])
    # A single circuit broadcasts over two initial states; encoders and X can
    # also supply the batch. Use a single state when no source creates a batch.
    count = 2 if input_batch or with_encoder or with_state else 1
    actual = ansatz.forward(values[:, None] if input_batch else None,
                            quantum_state if with_state else None)
    totals = (values + encoder_angles + initial_angles)[:count]
    np.testing.assert_allclose(actual[:, 0], np.cos(totals), atol=1e-12)
    np.testing.assert_allclose(np.asarray(ansatz.jacobian['t']).reshape(-1), -np.sin(totals), atol=1e-10)
    result = ansatz.backward(np.ones((count, 1)))
    if input_batch:
        np.testing.assert_allclose(np.asarray(result).reshape(-1), -np.sin(totals), atol=1e-10)
    else:
        assert result.shape == (count, 0)
        assert ansatz.gradients['t'] == pytest.approx(-np.sin(totals).sum())


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


def test_parameter_object_bindings_copy_and_update():
    ansatz = ry_ansatz()
    bindings = {Parameter('t'): 0.4}
    ansatz.assign_parameters(bindings)
    bindings[Parameter('t')] = 2.0
    ansatz.set_optimizer(SGD(lr=0.1))
    np.testing.assert_allclose(ansatz.forward(), [[np.cos(0.4)]])
    ansatz.backward(np.ones((1, 1)))
    ansatz.update()
    assert ansatz._bindings == {'t': pytest.approx(0.4 + 0.1 * np.sin(0.4))}


@pytest.mark.parametrize('mutation', ['ry', 'append', 'multi_control', 'unitary', 'add_qubits'])
def test_mutation_invalidates_bound_circuit_without_losing_bindings(mutation):
    ansatz = ry_ansatz()
    ansatz.forward()
    if mutation == 'ry':
        ansatz.ry(0, 0.7)
    elif mutation == 'append':
        ansatz.append(StandardGate.RY, [0], [0.7])
    elif mutation == 'unitary':
        rotation = Circuit(1)
        rotation.ry(0, 0.7)
        ansatz.unitary(UnitaryGate('rotation', 1).with_matrix(rotation.to_matrix()), [0])
    elif mutation == 'multi_control':
        ansatz.add_qubits([1])
        ansatz.x(1)
        ansatz.multi_control(StandardGate.RY, [1], [0], [0.7])
    else:
        ansatz.add_qubits([1])
        ansatz.x(1)
    expected = np.cos(0.3 if mutation == 'add_qubits' else 1.0)
    np.testing.assert_allclose(ansatz.forward(), [[expected]], atol=1e-12)
    assert ansatz._bindings == {'t': 0.3}
    assert ansatz._assigned_cir.num_qubits == ansatz.num_qubits


def test_new_symbol_keeps_existing_value_after_mutation():
    ansatz = ry_ansatz()
    ansatz.ry(0, Parameter('new'))
    ansatz.forward()
    assert ansatz._bindings['t'] == 0.3
    assert set(ansatz._bindings) == {'t', 'new'}


def test_forward_replaces_jacobian_and_restores_update_mode():
    ansatz = ry_ansatz()
    ansatz.forward(np.array([[0.2], [0.6]]))
    assert not ansatz.updatable
    ansatz.forward()
    assert ansatz.updatable
    np.testing.assert_allclose(ansatz.jacobian['t'], [[-np.sin(0.3)]])
    ansatz.zero_grad()
    assert ansatz.backward(np.ones((1, 1))).shape == (1, 0)
    assert ansatz.gradients['t'] == pytest.approx(-np.sin(.3))


def test_measurement_modes_replace_each_other():
    ansatz = ry_ansatz()
    ham = Hamiltonian.from_pauli(PauliString.from_str('X'))
    ansatz.set_measurement(hams=ham)
    assert ansatz.readouts is None
    np.testing.assert_allclose(ansatz.forward(), [[np.sin(0.3)]])
    np.testing.assert_allclose(ansatz.jacobian['t'], [[np.cos(0.3)]])
    ansatz.set_measurement(readouts=[0])
    assert ansatz.hams is None
    np.testing.assert_allclose(ansatz.forward(), [[np.cos(0.3)]])
    np.testing.assert_allclose(ansatz.jacobian['t'], [[-np.sin(0.3)]])


def test_hamiltonian_checkpoint_round_trip_and_next_step(tmp_path):
    original = ry_ansatz()
    ham = Hamiltonian(1)
    ham.add_term(PauliString.from_str('-X'), -0.7)
    ham.add_term(PauliString.from_str('Z'), 0.2)
    original.set_measurement(hams=[ham, Hamiltonian(1)])
    original.set_optimizer('adam')
    original.forward()
    original.backward(np.ones((1, 2)))
    original.update()
    original.zero_grad()
    Module(original).save_checkpoint(str(tmp_path), 2, 7)
    restored = ry_ansatz()
    restored.set_measurement(hams=[ham, Hamiltonian(1)])
    Module(restored).load_checkpoint(str(tmp_path))
    np.testing.assert_allclose(restored.forward(), original.forward())
    assert restored.hams[0] == ham
    assert restored.hams[1].num_terms == 0
    for model in (original, restored):
        model.backward(np.ones((1, 2)))
        model.update()
    assert restored._bindings == pytest.approx(original._bindings)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_nested_circuit_and_barrier_checkpoint_restore_matrix_and_gradients(tmp_path, method):
    original = ry_ansatz()
    nested = Circuit(1)
    nested.ry(0, 2 * Parameter('p'))
    original.append(Instruction.from_circuit_gate(nested.to_gate('nested')), [0], [Parameter('t')])
    original.append(Instruction.from_directive(Directive.barrier()), [0])
    # Exercise barrier together with a custom matrix in adjoint's inverse path.
    original.unitary(UnitaryGate('identity', 1).with_matrix(np.eye(2)), [0])
    original.assign_parameters({'t': 0.3})
    original.set_differentiator(method)
    Module(original).save_checkpoint(str(tmp_path), 1, 3)
    restored = Ansatz(1)
    restored.set_measurement(readouts=[0])
    Module(restored).load_checkpoint(str(tmp_path))
    restored.set_differentiator(method)
    np.testing.assert_allclose(restored._assigned_cir.to_matrix(), original._assigned_cir.to_matrix())
    for model in (original, restored):
        np.testing.assert_allclose(model.forward(), [[np.cos(0.9)]], atol=1e-12)
        np.testing.assert_allclose(model.jacobian['t'], [[-3 * np.sin(0.9)]], atol=1e-10)


@pytest.mark.parametrize('difference', ['gate', 'qubit', 'expression', 'order', 'matrix'])
def test_checkpoint_rejects_different_circuit_structure(difference):
    source = Ansatz(2)
    source.ry(0, Parameter('t'))
    source.rx(1, 0.2)
    source.unitary(UnitaryGate('custom', 1).with_matrix(np.eye(2)), [0])
    source.set_measurement(readouts=[0])
    source.assign_parameters({'t': 0.3})
    target = Ansatz(2)
    if difference == 'order':
        target.rx(1, 0.2)
    getattr(target, 'rz' if difference == 'gate' else 'ry')(
        1 if difference == 'qubit' else 0,
        2 * Parameter('t') if difference == 'expression' else Parameter('t'))
    if difference != 'order':
        target.rx(1, 0.2)
    matrix = np.array([[0, 1], [1, 0]]) if difference == 'matrix' else np.eye(2)
    target.unitary(UnitaryGate('custom', 1).with_matrix(matrix), [0])
    target.set_measurement(readouts=[0])
    target.assign_parameters({'t': 0.8})
    with pytest.raises(ValueError, match='structure'):
        target.load_params(deepcopy(source.summary))
    assert target._bindings == {'t': 0.8}


def test_legacy_in_memory_hamiltonian_summary():
    source = ry_ansatz()
    ham = Hamiltonian.from_pauli(PauliString.from_str('X'))
    source.set_measurement(hams=[ham])
    summary = source.summary
    summary['hamiltonians'] = [ham]
    restored = Ansatz(1)
    restored.load_params(summary)
    np.testing.assert_allclose(restored.forward(), [[np.sin(0.3)]])


def test_single_feature_circular_encoder_matches_linear():
    data = np.array([[0.3], [-0.8]])
    circular = ZZFeatureEncoder(entanglement='circular')(data)
    linear = ZZFeatureEncoder(entanglement='linear')(data)
    for actual, expected in zip(circular, linear):
        np.testing.assert_allclose(circuit_state(actual), circuit_state(expected))
