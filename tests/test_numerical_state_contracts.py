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


def test_qsvm_owns_training_snapshot():
    X, y = training_data()
    query = X.copy()
    model = QSVM(AngleEncoder()).fit(X, y)
    predictions = model.predict(query)
    decisions = model.decision_function(query)
    assert not np.shares_memory(model._X_fit, X)
    X[:] = 0
    np.testing.assert_array_equal(model.predict(query), predictions)
    np.testing.assert_allclose(model.decision_function(query), decisions)


def vqc():
    return VQC(HEAnsatz(2, 1, layers=['RY']), AngleEncoder(), readouts=[0, 1],
               loss='CrossEntropy', epochs=1, verbose=False)


def test_vqc_invalid_loss_does_not_destroy_fitted_state():
    X, y = training_data()
    model = vqc().fit(X, y)
    predictions = model.predict(X)
    qnn = model._qnn
    with pytest.raises(ValueError, match='Unsupported loss'):
        model.set_params(loss='typo', epochs=2)
    assert model.loss == 'CrossEntropy' and model.epochs == 1
    assert model._qnn is qnn
    np.testing.assert_array_equal(model.predict(X), predictions)


def test_vqc_direct_invalid_loss_is_rejected_before_fit():
    X, y = training_data()
    model = vqc().fit(X, y)
    qnn = model._qnn
    model.loss = 'typo'
    with pytest.raises(ValueError, match='Unsupported loss'):
        model.fit(X, y)
    assert model._qnn is qnn


def test_qkm_failed_first_encoding_does_not_lock_dimensions():
    model = QKM(AmplitudeEncoder())
    with pytest.raises(ValueError):
        model.kernel(np.zeros((1, 2)))
    assert model._feature_dim is None
    assert model._circuit_cache == {}
    result = model.kernel([[1., 2., 3.]])
    np.testing.assert_allclose(result, [[1 + 1e-8]], atol=1e-12)


@pytest.mark.parametrize('data', [np.array(1), np.empty((0, 2)), np.empty((2, 0)),
                                  np.ones((1, 2, 2)), [[np.nan, 1]], [[np.inf, 1]]])
@pytest.mark.parametrize('side', ['X', 'Y'])
def test_qkm_invalid_inputs_preserve_state(data, side):
    model = QKM(AngleEncoder())
    kwargs = {'X': data} if side == 'X' else {'X': [[.1, .2]], 'Y': data}
    with pytest.raises(ValueError):
        model.kernel(**kwargs)
    assert model._feature_dim is None and model._n_qubits is None
    assert model._circuit_cache == {}


@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('failure', ['encoding', 'fidelity'])
def test_qkm_partial_failure_is_atomic(monkeypatch, fitted, failure):
    model = QKM(AmplitudeEncoder())
    if fitted:
        model.kernel([[1., 0.]])
    snapshot = (model._feature_dim, model._n_qubits, dict(model._circuit_cache))
    if failure == 'fidelity':
        def fail(*args):
            raise ValueError('simulator failed')
        monkeypatch.setattr(model, '_fidelity', fail)
        args = ([[.3, .4]], [[.6, .8]])
    else:
        args = ([[.3, .4]], [[0., 0.]])
    with pytest.raises(ValueError):
        model.kernel(*args)
    assert (model._feature_dim, model._n_qubits, model._circuit_cache) == snapshot


def test_qkm_accepts_complex_amplitudes_and_single_samples():
    result = QKM(AmplitudeEncoder()).kernel([1, 1j], [1, -1j])
    np.testing.assert_allclose(result, [[0.]], atol=1e-12)


@pytest.mark.parametrize('hybrid', [False, True])
def test_training_examples_pass_loss_to_king_scheduler(tmp_path, hybrid):
    X, y = training_data()
    circuits = AngleEncoder()(X)
    scheduler = KingScheduler(patience=2)
    optimizer = SGD(lr_scheduler=scheduler)
    ansatz = HEAnsatz(2, 1, layers=['RY'])
    params = np.full(len(ansatz.symbols), .2)
    model = HQNN(ansatz, out_dim=2, params=params, optimizer=optimizer) if hybrid else QNN(
        ansatz, readouts=[0], params=params, optimizer=optimizer)
    module = importlib.import_module('cqlib_qml.algorithms.' + (
        'HQNN_classification' if hybrid else 'QNN_classification'))
    original = dict(ansatz._bindings)
    losses = []
    update = model.update
    def record_update(*args, **kwargs):
        losses.append(kwargs.get('cur_loss'))
        return update(*args, **kwargs)
    model.update = record_update
    module.train(0, 0, model, [(circuits[:2], y[:2]), (circuits[2:], y[2:])],
                 SoftmaxCrossEntropy() if hybrid else MSELoss(), str(tmp_path), 1)
    assert len(losses) == 2 and all(np.isfinite(loss) for loss in losses)
    assert ansatz._bindings != original


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


@pytest.mark.parametrize('batch_size', [-1, 0, 1.5, True, False, None])
def test_dataloader_invalid_batch_size_fails_at_construction(batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        DataLoader(Dataset(np.arange(4)), batch_size=batch_size)


@pytest.mark.parametrize('batch_size', [1, 3, np.int64(3)])
@pytest.mark.parametrize('drop_last', [False, True])
def test_dataloader_valid_batch_sizes_preserve_tail(batch_size, drop_last):
    loader = DataLoader(Dataset(np.arange(4)), batch_size=batch_size,
                        shuffle=False, drop_last=drop_last)
    batches = [batch[0] for batch in loader]
    assert len(batches) == len(loader)
    expected = 4 // batch_size * batch_size if drop_last else 4
    np.testing.assert_array_equal(np.concatenate(batches), np.arange(expected))
