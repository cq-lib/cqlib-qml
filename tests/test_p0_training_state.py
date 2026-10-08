"""Acceptance tests for reproducible retraining and exact batch-boundary resume."""
from copy import deepcopy
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from cqlib.circuit import Parameter

from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.data import DataLoader, Dataset
from cqlib_qml.encoder import AngleEncoder
from cqlib_qml.optimizer import SGD, Adam, OptimizerBase
from cqlib_qml.scheduler import ExponentialScheduler, KingScheduler
from cqlib_qml._state import clone_state
from cqlib_qml.layer import Layer, Linear
from cqlib_qml.models import Module


def quantum(method='adjoint'):
    q = Ansatz(2, random_state=11)
    x, z, t, u = [Parameter(name) for name in ('x', 'z', 't', 'u')]
    q.ry(0, x + 2 * t)
    q.rx(1, z)
    q.cx(0, 1)
    q.rz(0, u)
    q.ry(1, t)
    q.set_measurement(readouts=[0, 1])
    q.set_parameter_roles(input_params=[z, x], weight_params=[u, t])
    q.assign_weights([.31, -.24])
    q.set_differentiator(method, shift=np.pi / 4)
    q.set_optimizer('sgd(lr=.02)')
    return q


def network(method='adjoint'):
    a, b = Linear(3, 2, act_fn='tanh', random_state=5), Linear(2, 2, random_state=7)
    a.init_params()
    b.init_params()
    model = Module(a, quantum(method), b, random_state=10)
    model.set_optimizer('sgd(lr=.02)')
    return model


X = np.array([[.1, .3], [.3, -.1], [.4, .2], [.7, .5], [.8, -.2], [.9, .2], [.6, .7]])
Y = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'b'])


def estimator(optimizer='adam', **kwargs):
    return VQC(HEAnsatz(2, 1, layers=['RY', 'CX', 'RY']), AngleEncoder(),
               epochs=3, batch_size=2, verbose=False, random_state=17,
               optimizer=optimizer, **kwargs)


def state_bytes(model):
    if isinstance(model, Ansatz):
        value = (model.summary, model.gradients, model.jacobian, model._forward_valid,
                 model._gradient_valid, model._bindings)
    elif isinstance(model, VQC):
        value = (state_bytes(model.ansatz), state_bytes(model.ansatz_), model._progress,
                 model._rng.bit_generator.state, model._initial_rng_state, model._classes,
                 model._X_fit, model.encoder_.__dict__, model._qnn._forward_valid)
    elif hasattr(model, '_nets'):
        value = ([state_bytes(net) for net in model._nets], model._rng.bit_generator.state,
                 model.training, model._forward_valid)
    elif hasattr(model, 'summary'):
        value = (model.summary(), model.gradients, model._forward_valid, model._derived_variables)
    else:
        value = model
    return pickle.dumps(value, protocol=5)


def assert_exact_state(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact_state(a, b)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected



@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_module_updates_pending_legacy_quantum_gradients_after_input_inference(method):
    q = Ansatz(1)
    q.ry(0, Parameter('theta'))
    q.set_measurement(readouts=[0])
    q.assign_weights([.3])
    q.set_differentiator(method)
    model = Module(q)
    model.set_optimizer('sgd(lr=.1)')
    model.forward()
    model.backward(np.ones((1, 1)))
    expected = clone_state(q)
    expected.update()
    gradients = deepcopy(q.gradients)
    model.forward(np.array([[.8]]), retain_derived=False)
    assert not q.updatable
    assert_exact_state(q.gradients, gradients)
    model.update()
    np.testing.assert_array_equal(q.weights, expected.weights)
    assert q._optimizer.cur_step == 1

def test_owned_randomness_does_not_consume_global_or_caller_generator():
    global_state = pickle.dumps(np.random.get_state())
    generator = np.random.default_rng(5)
    owned_state = deepcopy(generator.bit_generator.state)
    model = estimator()
    model.set_params(random_state=generator)
    model.fit(X, Y)
    network().random_init()
    loader = DataLoader(Dataset(X, Y), random_state=generator)
    list(loader)
    assert generator.bit_generator.state == owned_state
    assert pickle.dumps(np.random.get_state()) == global_state

def test_qnn_and_hqnn_owned_random_initialization():
    from cqlib_qml.models import QNN, HQNN
    circuits = AngleEncoder()(X)
    for cls, kwargs in [(QNN, {}), (HQNN, {'out_dim': 2})]:
        first = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        second = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        np.testing.assert_array_equal(first.forward(circuits), second.forward(circuits))

def test_fresh_repeated_fit_clone_and_initial_point():
    model, other = estimator(), estimator()
    model.fit(X, Y)
    other.fit(X, Y)
    np.testing.assert_array_equal(model.ansatz_.weights, other.ansatz_.weights)
    first = model.ansatz_.weights.copy()
    model.fit(X, Y)
    np.testing.assert_array_equal(model.ansatz_.weights, first)
    assert model.ansatz._weights == {}  # configuration never gains learning state
    assert model.n_classes == 2
    cloned = clone(model)
    assert not cloned.__sklearn_is_fitted__()
    assert cloned.ansatz._weights == {}
    cloned.fit(X, Y)
    np.testing.assert_array_equal(cloned.ansatz_.weights, first)
    point = np.linspace(.1, .5, model.ansatz.num_weights)
    initialized = estimator(initial_point=point, optimizer='sgd(lr=0)').fit(X, Y)
    np.testing.assert_array_equal(initialized.ansatz_.weights, point)
    np.testing.assert_array_equal(point, initialized.initial_point)
