"""Regressions for classifier validation, ownership and native serialization."""
import pickle
import copyreg
from copy import copy

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_validate

from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.gates import UnitaryGate
from cqlib.qis import Hamiltonian, PauliString
from cqlib_qml.algorithms import QSVM, VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.encoder import AngleEncoder
from cqlib_qml._state import clone_state

try:
    import cloudpickle
except ImportError:  # Older joblib versions vendor this dependency.
    from joblib.externals import cloudpickle


X = np.array([[.1, .2], [.2, .1], [.3, .2], [.8, .7], [.7, .8], [.9, .8]])
Y = np.array(['left', 'left', 'left', 'right', 'right', 'right'])
NEW_X = np.array([[.15, .25], [.75, .85]])


def classifier(kind):
    if kind == 'qsvm':
        return QSVM(AngleEncoder(), random_state=7)
    return VQC(HEAnsatz(2, d=1, layers=['RY', 'CX']), AngleEncoder(),
               loss='BCE', epochs=2, batch_size=2, verbose=False, random_state=7)


@pytest.mark.parametrize('fitted', [False, True])
def test_vqc_rejects_continuous_targets_without_changing_state(fitted):
    model = classifier('vqc')
    if fitted:
        model.fit(X, Y)
        qnn = model._qnn
        before = model.predict_proba(NEW_X)
    with pytest.raises(ValueError, match='continuous|Unknown label type'):
        model.fit(X, np.array([.1, .1, .1, .8, .8, .8]))
    if fitted:
        assert model._qnn is qnn
        np.testing.assert_array_equal(model.predict_proba(NEW_X), before)
    else:
        assert not model.__sklearn_is_fitted__()


@pytest.mark.parametrize('method', ['predict', 'predict_proba', 'decision_function'])
def test_qsvm_validates_raw_feature_count_before_encoding(method, monkeypatch):
    model = QSVM(AngleEncoder(), probability=True, random_state=7).fit(X, Y)
    assert model.n_features_in_ == 2  # The internal kernel has six columns.

    def unexpected_kernel(*args):
        pytest.fail('Feature validation must happen before quantum execution')

    monkeypatch.setattr(model._qkm, 'kernel', unexpected_kernel)
    with pytest.raises(ValueError, match='2 features|Expected 2'):
        getattr(model, method)(np.ones((2, 3)))


@pytest.mark.parametrize('clear', ['reset', 'set_params'])
def test_qsvm_clears_fitted_metadata(clear):
    model = classifier('qsvm').fit(X, Y)
    assert model.n_features_in_ == 2
    if clear == 'reset':
        model.reset()
    else:
        model.set_params(C=2.)
    assert not hasattr(model, 'classes_')
    assert not hasattr(model, 'n_features_in_')
    with pytest.raises(NotFittedError):
        model.predict(NEW_X)


def test_qsvm_refit_updates_feature_count_and_failed_fit_keeps_it():
    model = classifier('qsvm').fit(X, Y)
    assert model.n_features_in_ == 2
    wider = np.column_stack([X, X[:, 0]])
    model.fit(wider, Y)
    assert model.n_features_in_ == 3
    before = model.predict(wider)
    with pytest.raises(ValueError):
        model.fit(X, np.full(len(X), 'only-class'))
    assert model.n_features_in_ == 3
    np.testing.assert_array_equal(model.predict(wider), before)


@pytest.mark.parametrize('method', ['predict', 'decision_function', 'predict_proba'])
@pytest.mark.parametrize('clear_cache', [False, True])
def test_qsvm_owns_fitted_encoder(method, clear_cache):
    encoder = AngleEncoder()
    model = QSVM(encoder, probability=True, random_state=7).fit(X, Y)
    reference = QSVM(AngleEncoder(), probability=True, random_state=7).fit(X, Y)
    assert model.encoder is encoder
    encoder._mode = 'dense'
    if clear_cache:
        model.clear_cache()
    compare = np.testing.assert_array_equal if method == 'predict' else np.testing.assert_allclose
    compare(getattr(model, method)(NEW_X), getattr(reference, method)(NEW_X))


@pytest.mark.parametrize('kind', ['vqc', 'qsvm'])
@pytest.mark.parametrize('fitted', [False, True])
@pytest.mark.parametrize('serialization', ['pickle', 'joblib'])
def test_estimator_serialization_preserves_state(kind, fitted, serialization, tmp_path):
    model = classifier(kind)
    if fitted:
        model.fit(X, Y)
        expected = model.predict_proba(NEW_X) if kind == 'vqc' else model.decision_function(NEW_X)
    if serialization == 'pickle':
        restored = pickle.loads(pickle.dumps(model))
    else:
        path = tmp_path / 'classifier.joblib'
        joblib.dump(model, path)
        restored = joblib.load(path)
    assert restored.__sklearn_is_fitted__() == fitted
    assert restored.encoder is not model.encoder
    assert not clone(restored).__sklearn_is_fitted__()
    if fitted:
        actual = restored.predict_proba(NEW_X) if kind == 'vqc' else restored.decision_function(NEW_X)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(restored.classes_, model.classes_)
        assert restored.n_features_in_ == 2
        if kind == 'vqc':
            assert restored._qnn._ansatz is restored.ansatz_
            assert restored._qnn._nets[0] is restored.ansatz_
            for estimator in (model, restored):
                estimator.resume_fit(X, Y, additional_epochs=1)
            np.testing.assert_array_equal(restored.predict_proba(NEW_X), model.predict_proba(NEW_X))
    else:
        for estimator in (model, restored):
            estimator.fit(X, Y)
        np.testing.assert_array_equal(restored.predict(NEW_X), model.predict(NEW_X))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_vqc_serialization_preserves_custom_circuit_and_paused_training(method):
    ansatz = Ansatz(2)
    ansatz.ry(0, Parameter('theta') * 2 + .1)
    ansatz.rx(1, Parameter('phi'))
    ansatz.cx(0, 1)
    ansatz.append(UnitaryGate('phase', 1).with_matrix(np.diag([1., 1.j])), [0])
    ansatz._circuit.set_global_phase(.27)
    ansatz.set_differentiator(method)
    # Native observables in construction templates also need serialization.
    ham = Hamiltonian(2)
    ham.add_term(PauliString.from_str('ZI'), .7)
    ansatz.set_measurement(hams=[ham])
    model = VQC(ansatz, AngleEncoder(), loss='BCE', epochs=2, batch_size=2,
                random_state=7, verbose=False).fit(X, Y, max_steps=1)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.ansatz_._assigned_cir.to_matrix(),
                                  model.ansatz_._assigned_cir.to_matrix())
    assert str(restored.ansatz._circuit.global_phase) == str(model.ansatz._circuit.global_phase)
    np.testing.assert_array_equal(restored.ansatz.hams[0].to_matrix(), model.ansatz.hams[0].to_matrix())
    for estimator in (model, restored):
        estimator.resume_fit(X, Y)
    np.testing.assert_array_equal(restored.predict_proba(NEW_X), model.predict_proba(NEW_X))
    assert restored._progress == model._progress


def test_vqc_serialization_preserves_native_template_metadata():
    model = classifier('vqc')
    parameter = Parameter('phi') * 2 + .1
    empty = Circuit([2, 5])
    empty.set_global_phase(Parameter('phi') / 3)
    model.ansatz.metadata = {'parameter': parameter, 'shared': parameter,
                             'empty': empty, 'qubits': empty.qubits,
                             'pauli': PauliString.from_str('-iY')}
    restored = pickle.loads(pickle.dumps(model)).ansatz.metadata
    assert restored['parameter'] is restored['shared']
    assert str(restored['parameter']) == str(parameter)
    assert [q.index for q in restored['qubits']] == [2, 5]
    assert [q.index for q in restored['empty'].qubits] == [2, 5]
    assert len(restored['empty']) == 0
    assert str(restored['empty'].global_phase) == str(empty.global_phase)
    assert str(restored['pauli']) == '-iY'


def test_vqc_copy_does_not_require_picklable_python_attributes():
    model = classifier('vqc')
    model.callback = lambda value: value + 1
    shallow = copy(model)
    assert shallow is not model
    assert shallow.ansatz is model.ansatz
    owned = clone_state(model)
    assert owned.ansatz is not model.ansatz
    assert owned.callback(2) == 3


def test_qsvm_serialization_keeps_live_cache_and_probability_predictions():
    model = QSVM(AngleEncoder(), probability=True, random_state=7).fit(X, Y)
    expected = model.predict_proba(NEW_X)
    cache = model._qkm._circuit_cache.copy()
    restored = pickle.loads(pickle.dumps(model))
    assert model._qkm._circuit_cache == cache
    assert not restored._qkm._circuit_cache
    assert restored._qkm._encoder is not restored.encoder
    restored.encoder._mode = 'dense'
    np.testing.assert_array_equal(restored.predict_proba(NEW_X), expected)


@pytest.mark.parametrize('kind', ['vqc', 'qsvm'])
def test_process_cross_validation_can_return_fitted_estimators(kind):
    result = cross_validate(classifier(kind), X, Y, cv=2, n_jobs=2,
                            return_estimator=True, error_score='raise')
    assert np.isfinite(result['test_score']).all()
    for model in result['estimator']:
        assert model.n_features_in_ == 2
        assert model.predict(NEW_X).shape == (2,)


def test_process_cross_validation_with_local_ansatz_and_callback():
    class CustomAnsatz(HEAnsatz):
        def __init__(self):
            super().__init__(2, d=1, layers=['RY', 'CX'])
            self.callback = lambda value: value + 1

    model = classifier('vqc')
    model.ansatz = CustomAnsatz()
    serial = cross_validate(model, X, Y, cv=2, n_jobs=1, error_score='raise')
    parallel = cross_validate(model, X, Y, cv=2, n_jobs=2,
                              return_estimator=True, error_score='raise')
    np.testing.assert_array_equal(parallel['test_score'], serial['test_score'])
    for restored in parallel['estimator']:
        assert isinstance(restored.ansatz, CustomAnsatz)
        assert restored.ansatz.callback(2) == 3
        assert restored.predict(NEW_X).shape == (2,)


@pytest.mark.parametrize('symbolic', [False, True])
def test_vqc_serialization_with_operations_on_noncontiguous_qubits(symbolic):
    model = classifier('vqc')
    circuit = Circuit([2, 5])
    angle = Parameter('theta') * 2 + .1 if symbolic else .3
    circuit.ry(2, angle)
    circuit.cx(2, 5)
    circuit.rx(5, -.2)
    circuit.set_global_phase(.27)
    model.ansatz.metadata = circuit
    restored = pickle.loads(pickle.dumps(model)).ansatz.metadata
    assert [q.index for q in restored.qubits] == [2, 5]
    assert [[q.index for q in op.qubits] for op in restored.operations] == [[2], [2, 5], [5]]
    assert str(restored.global_phase) == str(circuit.global_phase)
    if symbolic:
        circuit = circuit.assign_parameters({'theta': .4})
        restored = restored.assign_parameters({'theta': .4})
    np.testing.assert_array_equal(restored.to_matrix(), circuit.to_matrix())


@pytest.mark.parametrize('serialization', ['pickle', 'joblib', 'cloudpickle'])
@pytest.mark.parametrize('self_reference', [False, True])
@pytest.mark.parametrize('native_first', [False, True])
def test_vqc_serialization_preserves_outer_object_graph(
        serialization, self_reference, native_first, tmp_path):
    model = classifier('vqc').fit(X, Y)
    ham = Hamiltonian(2)
    ham.add_term(PauliString.from_str('ZI'), .7)
    model.ansatz.metadata = ham
    if self_reference:
        model.self_ref = model
    graph = (model, model.encoder, model.ansatz, model.ansatz._circuit, ham)
    if native_first:
        graph = graph[::-1]
    if serialization == 'joblib':
        path = tmp_path / 'graph.joblib'
        joblib.dump(graph, path)
        loaded = joblib.load(path)
    else:
        serializer = pickle if serialization == 'pickle' else cloudpickle
        loaded = serializer.loads(serializer.dumps(graph))
    restored, encoder, ansatz, circuit, ham = loaded[::-1] if native_first else loaded
    assert restored.encoder is encoder
    assert restored.ansatz is ansatz
    assert restored.ansatz._circuit is circuit
    assert restored.ansatz.metadata is ham
    if self_reference:
        assert restored.self_ref is restored
    np.testing.assert_array_equal(restored.predict_proba(NEW_X), model.predict_proba(NEW_X))


def test_native_registration_keeps_application_reducers(monkeypatch):
    from cqlib_qml._serialization import register_native_reducers

    def application_reducer(value):
        return Parameter, ('application',)

    monkeypatch.setitem(copyreg.dispatch_table, Parameter, application_reducer)
    register_native_reducers()
    assert copyreg.dispatch_table[Parameter] is application_reducer
