"""Numerical acceptance tests for independent input/weight roles and accumulation."""
from copy import deepcopy

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.layer import Linear
from cqlib_qml.models import Module
from cqlib_qml._state import clone_state


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


def finite_difference(function, values, eps=1e-6):
    result = np.empty_like(values, dtype=float)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += eps
        minus[index] -= eps
        result[index] = (function(plus) - function(minus)) / (2 * eps)
    return result


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_input_and_weight_gradients_and_jacobians(method):
    q = quantum(method)
    X = np.array([[.3, -.7], [.6, .2], [-.2, .5]])
    upstream = np.array([[.4, -.2], [1.2, .3], [-.5, .8]])
    y = q.forward(X)
    jx, jw = q.input_jacobian.copy(), q.weight_jacobian.copy()
    assert y.shape == (3, 2) and jx.shape == (3, 2, 2) and jw.shape == (3, 2, 2)
    dx = q.backward(upstream)
    dw = q.weight_gradients.copy()
    numeric_x = finite_difference(lambda v: np.sum(q.forward(v, retain_derived=False) * upstream), X)
    base_weights = q.weights.copy()
    def value(weights):
        q.assign_weights(weights)
        return np.sum(q.forward(X, retain_derived=False) * upstream)
    numeric_w = finite_difference(value, base_weights)
    np.testing.assert_allclose(dx, numeric_x, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dw, numeric_w, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dx, np.einsum('bo,boi->bi', upstream, jx), atol=1e-12)
    np.testing.assert_allclose(dw, np.einsum('bo,bow->w', upstream, jw), atol=1e-12)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_complete_hybrid_network_gradients_and_training(method):
    model = network(method)
    X = np.array([[.3, -.7, .1], [.6, .2, -.4]])
    target = np.array([[.2, -.1], [-.3, .4]])
    y = model.forward(X)
    dx = model.backward(2 * (y - target) / y.size)
    gradients = [deepcopy(net.gradients) for net in model._nets]
    reference = clone_state(model)
    def loss(v):
        return np.mean((reference.forward(v, retain_derived=False) - target) ** 2)
    np.testing.assert_allclose(dx, finite_difference(loss, X), rtol=1e-5, atol=1e-7)
    for i, net in enumerate(model._nets):
        if isinstance(net, Ansatz):
            values = net.weights.copy()
            def function(v):
                reference._nets[i].assign_weights(v)
                return loss(X)
            numerical = finite_difference(function, values)
            np.testing.assert_allclose(net.weight_gradients, numerical, rtol=1e-5, atol=1e-7)
            reference._nets[i].assign_weights(values)
        else:
            for name, values in net.parameters.items():
                def function(v):
                    reference._nets[i]._parameters[name] = v
                    return loss(X)
                numerical = finite_difference(function, values)
                np.testing.assert_allclose(gradients[i][name], numerical, rtol=1e-5, atol=1e-7)
                reference._nets[i]._parameters[name] = values.copy()
    before = [net.weights.copy() if isinstance(net, Ansatz) else net.parameters['W'].copy() for net in model._nets]
    model.update()
    for old, net in zip(before, model._nets):
        new = net.weights if isinstance(net, Ansatz) else net.parameters['W']
        assert np.max(abs(old - new)) > 1e-8
    assert np.mean((model.forward(X) - target) ** 2) < np.mean((y - target) ** 2)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_unequal_microbatches_match_full_gradient_and_update(method):
    full = network(method)
    micro = clone_state(full)
    X = np.array([[.3, -.7, .1], [.6, .2, -.4], [-.1, .9, .3], [.1, .2, .3], [.8, -.2, .5]])
    target = np.linspace(-.3, .4, 10).reshape(5, 2)
    y = full.forward(X)
    full.backward(2 * (y - target) / y.size)
    for start, end in [(0, 2), (2, 3), (3, 5)]:
        output = micro.forward(X[start:end])
        micro.backward(2 * (output - target[start:end]) / output.size * ((end - start) / len(X)))
    for a, b in zip(full._nets, micro._nets):
        for key in a.gradients:
            np.testing.assert_allclose(a.gradients[key], b.gradients[key], rtol=1e-12, atol=1e-12)
    full.update()
    micro.update()
    np.testing.assert_allclose(full.forward(X), micro.forward(X), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_freeze_keeps_input_gradient_and_zero_grad_retains_cache(method):
    model = network(method)
    X = np.array([[.3, -.7, .1]])
    model._nets[1].freeze()
    weights = model._nets[1].weights.copy()
    out = model.forward(X)
    dx = model.backward(np.ones_like(out))
    first = model._nets[0].gradients['W'].copy()
    assert dx.shape == (1, 3) and np.linalg.norm(first) > 0
    assert model._nets[1].gradients == {}
    model.backward(np.ones_like(out))
    np.testing.assert_allclose(model._nets[0].gradients['W'], 2 * first)
    model.zero_grad()
    np.testing.assert_allclose(model.backward(np.ones_like(out)), dx)
    np.testing.assert_allclose(model._nets[0].gradients['W'], first)
    model.update()
    np.testing.assert_array_equal(model._nets[1].weights, weights)
    with pytest.raises(ValueError):
        model.backward(np.ones_like(out))
    model.zero_grad()
    model.eval()
    model.forward(X)  # eval is independent of recording
    model.backward(np.ones_like(out))
    saved = model._nets[0].gradients['W'].copy()
    model.forward(X, retain_derived=False)
    np.testing.assert_array_equal(model._nets[0].gradients['W'], saved)
    with pytest.raises(ValueError):
        model.backward(np.ones_like(out))


@pytest.mark.parametrize('inputs,weights', [(['x', 'x'], ['z', 't', 'u']), (['x'], ['x', 'z', 't', 'u']),
                                            (['missing'], ['z', 't', 'u']), (['x'], ['t'])])
def test_invalid_roles_are_atomic(inputs, weights):
    q = quantum()
    roles = (q.input_params, q.weight_params)
    with pytest.raises(ValueError):
        q.set_parameter_roles(input_params=inputs, weight_params=weights)
    assert (q.input_params, q.weight_params) == roles


def test_binding_shapes_and_legacy_parameter_modes():
    q = quantum()
    for values in (None, np.zeros((2, 3)), np.zeros((0, 2)), np.array([[np.nan, 0]])):
        with pytest.raises(ValueError):
            q.forward(values)
    assert q.forward(np.array([.2, .3])).shape == (1, 2)
    for values in ([.1], [.1, np.inf], {'t': .1}, {'x': .1, 't': .2}):
        with pytest.raises(ValueError):
            q.assign_weights(values)
    q.ry(0, Parameter('new'))
    with pytest.raises(ValueError, match='redeclare'):
        q.forward(np.ones((1, 2)))
    old = Ansatz(1)
    old.ry(0, Parameter('t'))
    old.set_measurement(readouts=[0])
    old.assign_parameters({'t': .2})
    output = old.forward()
    assert old.backward().shape == (1, 0)
    assert old.gradients['t'] == pytest.approx(-np.sin(.2))
    np.testing.assert_allclose(old.jacobian['t'], [[-np.sin(.2)]])
    old.zero_grad()
    assert old.forward(np.array([.4])).shape == (1, 1)
    np.testing.assert_allclose(old.backward(), [[-np.sin(.4)]])


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_roles_with_encoder_and_initial_state(method):
    q = quantum(method)
    encoder = Circuit(2)
    encoder.h(0)
    q.add_encoder([encoder, encoder])
    X = np.array([[.2, .3], [.4, -.5]])
    initial = np.array([0., 1., 0., 0.])
    q.forward(X, quantum_state=initial)
    dx = q.backward()
    numerical = finite_difference(lambda v: q.forward(v, quantum_state=initial, retain_derived=False).sum(), X)
    np.testing.assert_allclose(dx, numerical, rtol=1e-5, atol=1e-7)
