"""Torch adapter contracts, numerical gradients and state ownership."""
from copy import deepcopy
import io

import numpy as np
import pytest
import torch
from cqlib.circuit import Circuit, Parameter, UnitaryGate
from cqlib.qis import Hamiltonian, PauliString, Phase

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.torch import QuantumLayer


def quantum(method='adjoint', *, hamiltonian=False):
    q = Ansatz(2, random_state=11)
    x, z, t, u = [Parameter(name) for name in ('x', 'z', 't', 'u')]
    q.ry(0, x + 2 * t)
    q.rx(1, z)
    q.cx(0, 1)
    q.rz(0, u)
    q.ry(1, t)
    q.set_parameter_roles(input_params=[z, x], weight_params=[u, t])
    q.assign_weights([.31, -.24])
    q.set_differentiator(method, shift=np.pi / 4)
    if hamiltonian:
        q.set_measurement(hams=[Hamiltonian.from_list([
            (PauliString.from_str('ZI'), .7), (PauliString.from_str('XX'), -.2)])])
    else:
        q.set_measurement(readouts=[0, 1])
    return q


def inputs(*, batch=True, dtype=torch.float64, requires_grad=True):
    values = [[.3, -.7], [.6, .2]] if batch else [.3, -.7]
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('hamiltonian', [False, True])
def test_shapes_values_and_gradcheck(method, batch, hamiltonian):
    q = quantum(method, hamiltonian=hamiltonian)
    layer = QuantumLayer(q, dtype=torch.float64)
    x = inputs(batch=batch)
    expected = q.forward(x.detach().numpy())
    if not batch:
        expected = expected[0]
    torch.testing.assert_close(layer(x), torch.tensor(expected))
    assert (layer.num_inputs, layer.num_weights, layer.num_outputs) == (2, 2, q.out_dim)
    assert torch.autograd.gradcheck(
        lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
        (x, layer.weight))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_complete_hybrid_network_gradcheck_and_training(method):
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh(),
                                QuantumLayer(quantum(method)), torch.nn.Linear(2, 1)).double()
    x = torch.tensor([[.3, -.7, .1], [.6, .2, -.4]], dtype=torch.float64,
                     requires_grad=True)
    parameters = dict(model.named_parameters())
    def evaluate(a, *weights):
        return torch.func.functional_call(model, dict(zip(parameters, weights)), (a,))
    assert torch.autograd.gradcheck(evaluate, (x, *parameters.values()))
    target = torch.tensor([[.2], [-.3]], dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = torch.nn.functional.mse_loss(model(x), target).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(x), target).item() < initial


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_multiple_forwards_repeated_backward_and_unequal_microbatches(method):
    layer = QuantumLayer(quantum(method), dtype=torch.float64)
    x = torch.tensor([[.3, -.7], [.6, .2], [-.2, .5]], dtype=torch.float64,
                     requires_grad=True)
    target = torch.tensor([[.2, -.1], [-.3, .4], [.1, .6]], dtype=torch.float64)
    loss = torch.nn.functional.mse_loss(layer(x), target)
    loss.backward(retain_graph=True)
    dx, dw = x.grad.clone(), layer.weight.grad.clone()
    loss.backward()
    torch.testing.assert_close(x.grad, 2 * dx)
    torch.testing.assert_close(layer.weight.grad, 2 * dw)
    layer.zero_grad()
    x.grad = None
    first, second = layer(x[:1]), layer(x[1:])
    micro_loss = (torch.nn.functional.mse_loss(first, target[:1]) / 3 +
                  torch.nn.functional.mse_loss(second, target[1:]) * 2 / 3)
    micro_loss.backward()
    torch.testing.assert_close(x.grad, dx)
    torch.testing.assert_close(layer.weight.grad, dw)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_freeze_eval_no_grad_and_dtype(dtype, monkeypatch):
    layer = QuantumLayer(quantum(), dtype=dtype).eval()
    layer.weight.requires_grad_(False)
    x = inputs(dtype=dtype)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is None
    def forbid_jacobian(*args, **kwargs):
        raise AssertionError('Inference must skip quantum differentiation')
    with monkeypatch.context() as patch:
        patch.setattr(Ansatz, '_bwd', forbid_jacobian)
        with torch.no_grad():
            result = layer(x)
            assert not result.requires_grad and result.dtype == dtype
        assert not layer(x.detach()).requires_grad
    converted = layer.double() if dtype == torch.float32 else layer.float()
    target_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    converted(inputs(dtype=target_dtype)).sum().backward()
    assert converted.weight.dtype == target_dtype


@pytest.mark.parametrize('has_inputs,has_weights', [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 3), (2, 1, 3)])
def test_empty_parameter_dimensions(has_inputs, has_weights, batch_shape):
    q = Ansatz(1)
    q.ry(0, Parameter('x') if has_inputs else .2)
    if has_weights:
        q.rz(0, Parameter('t'))
        q.ry(0, .4)
    q.set_parameter_roles(input_params=['x'] if has_inputs else [],
                          weight_params=['t'] if has_weights else [])
    q.set_measurement(readouts=[0])
    layer = QuantumLayer(q, dtype=torch.float64)
    shape = (*batch_shape, int(has_inputs))
    x = torch.full(shape, .3, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    assert y.shape == (*batch_shape, 1)
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == (int(has_weights),)
    assert torch.autograd.gradcheck(
        lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
        (x, layer.weight))


@pytest.mark.parametrize('bad', [torch.ones(2, dtype=torch.int64),
                                torch.ones(2, dtype=torch.complex128),
                                torch.ones(2, dtype=torch.float32),
                                torch.ones(2, device='meta'),
                                torch.tensor(.3, dtype=torch.float64),
                                torch.empty(0, 2, dtype=torch.float64),
                                torch.empty(2, 0, 2, dtype=torch.float64),
                                torch.empty(0, 2, 3, 2, dtype=torch.float64),
                                torch.ones(2, 3, 1, dtype=torch.float64),
                                torch.ones(3, dtype=torch.float64),
                                torch.tensor([float('nan'), 0.], dtype=torch.float64),
                                torch.tensor([float('inf'), 0.], dtype=torch.float64),
                                [.3, -.7]])
def test_invalid_input_rejected(bad):
    with pytest.raises((TypeError, ValueError), match='Tensor|CPU|dtype|shape|finite'):
        QuantumLayer(quantum(), dtype=torch.float64)(bad)


@pytest.mark.parametrize('bad', [[1.], [float('nan'), 0.], [complex(1, 1), 0.],
                                torch.ones(2, device='meta')])
def test_invalid_initial_weights_rejected(bad):
    with pytest.raises((TypeError, ValueError), match='weights|CPU|real|finite|shape'):
        QuantumLayer(quantum(), initial_weights=bad)


def test_initialization_and_constructor_validation():
    q = quantum()
    layer = QuantumLayer(q, initial_weights=[.1, .2], dtype=torch.float64)
    torch.testing.assert_close(layer.weight, torch.tensor([.1, .2], dtype=torch.float64),
                               rtol=0, atol=0)
    torch.testing.assert_close(QuantumLayer(q).weight, torch.tensor([.31, -.24]))
    q._weights = {'u': .9}
    torch.manual_seed(13)
    first = QuantumLayer(q).weight.detach().clone()
    torch.manual_seed(13)
    torch.testing.assert_close(QuantumLayer(q).weight, first)
    assert torch.all(first.abs() <= torch.pi)
    q._roles = None
    with pytest.raises(ValueError, match='roles'):
        QuantumLayer(q)
    q = quantum()
    q.add_encoder([Circuit(2)])
    with pytest.raises(ValueError, match='encoder'):
        QuantumLayer(q)
    q = quantum()
    q._out_dim = 0
    with pytest.raises(ValueError, match='measurement'):
        QuantumLayer(q)
    with pytest.raises(TypeError, match='Ansatz'):
        QuantumLayer(object())
    with pytest.raises((TypeError, ValueError), match='dtype'):
        QuantumLayer(quantum(), dtype=torch.float16)
    q = quantum()
    q.ry(0, Parameter('new'))
    with pytest.raises(ValueError, match='roles'):
        QuantumLayer(q)


def test_ownership_and_nonfinite_weights():
    q = quantum(hamiltonian=True)
    q.set_optimizer('adam(lr=.02)')
    q.forward(inputs().detach().numpy())
    q.backward(np.ones((2, 1)))
    before = deepcopy(q.summary)
    jacobian = q.jacobian
    gradients = dict(q.gradients)
    layer = QuantumLayer(q, dtype=torch.float64)
    y = layer(inputs())
    assert q.summary == before
    for name in jacobian:
        np.testing.assert_array_equal(q.jacobian[name], jacobian[name])
    assert q.gradients == gradients
    q.assign_weights([1., 2.])
    q.set_measurement(readouts=[1])
    torch.testing.assert_close(layer(inputs()), y)
    assert not layer._template._forward_valid
    assert layer._template._optimizer is None and layer._template.gradients == {}
    with torch.no_grad():
        layer.weight[0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        layer(inputs())


def test_saved_versions_and_second_derivative():
    layer = QuantumLayer(quantum(), dtype=torch.float64)
    x = inputs()
    y = layer(x)
    with torch.no_grad():
        layer.weight.add_(.1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        y.sum().backward()
    y = layer(x)
    dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dx.sum(), x)


def test_weight_only_input_only_and_inference_modes():
    layer = QuantumLayer(quantum(), dtype=torch.float64)
    layer(inputs(requires_grad=False)).sum().backward()
    assert torch.isfinite(layer.weight.grad).all()
    with torch.inference_mode():
        assert not layer(inputs()).requires_grad
    with torch.autocast('cpu'):
        with pytest.raises(ValueError, match='autocast'):
            layer(inputs())
    layer.half()
    with pytest.raises(ValueError, match='dtype'):
        layer(inputs(dtype=torch.float16))
    layer = QuantumLayer(quantum(), dtype=torch.float64)
    x = inputs()
    y = layer(x)
    with torch.no_grad():
        x.add_(.1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        y.sum().backward()


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('hamiltonian', [False, True])
def test_state_dict_roundtrip_and_training(nested, hamiltonian):
    def model():
        layer = QuantumLayer(quantum(hamiltonian=hamiltonian), dtype=torch.float64)
        return torch.nn.Sequential(layer) if nested else layer
    source, restored = model(), model()
    optimizer = torch.optim.Adam(source.parameters(), lr=.02)
    source(inputs()).square().sum().backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({'model': source.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored.load_state_dict(state['model'])
    other_optimizer = torch.optim.Adam(restored.parameters(), lr=.02)
    other_optimizer.load_state_dict(state['optimizer'])
    for current, opt in [(source, optimizer), (restored, other_optimizer)]:
        opt.zero_grad()
        current(inputs()).square().sum().backward()
        opt.step()
    torch.testing.assert_close(restored(inputs()), source(inputs()))
    for a, b in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize('difference', ['inputs', 'weights', 'circuit', 'expression',
                                      'measurement', 'method', 'shift', 'version', 'missing'])
@pytest.mark.parametrize('strict', [False, True])
def test_incompatible_state_rejected_before_weight_copy(difference, strict):
    source = QuantumLayer(quantum(), dtype=torch.float64)
    q = quantum()
    if difference == 'inputs':
        q.set_parameter_roles(input_params=['x', 'z'], weight_params=['u', 't'])
    elif difference == 'weights':
        q.set_parameter_roles(input_params=['z', 'x'], weight_params=['t', 'u'])
    elif difference == 'circuit':
        q.x(0)
    elif difference == 'expression':
        q = quantum()
        q._circuit = Circuit(2)
        q.ry(0, Parameter('x') + Parameter('t'))
        q.rx(1, Parameter('z'))
        q.rz(0, Parameter('u'))
    elif difference == 'measurement':
        q.set_measurement(readouts=[1, 0])
    elif difference in ('method', 'shift'):
        q.set_differentiator('parameter_shift', shift=.3)
        if difference == 'shift':
            source = QuantumLayer(quantum('parameter_shift'), dtype=torch.float64)
    target = QuantumLayer(q, initial_weights=[1., 2.], dtype=torch.float64)
    before = target.weight.detach().clone()
    state = source.state_dict()
    if difference == 'version':
        state['_extra_state']['format_version'] = 99
    elif difference == 'missing':
        del state['_extra_state']
    with pytest.raises(RuntimeError, match='structure|metadata'):
        target.load_state_dict(state, strict=strict)
    torch.testing.assert_close(target.weight, before)


def test_nested_mismatch_and_metadata_ownership():
    source = torch.nn.Sequential(QuantumLayer(quantum()))
    target = torch.nn.Sequential(QuantumLayer(quantum(), initial_weights=[1., 2.]))
    state = source.state_dict()
    state['0._extra_state']['input_params'].reverse()
    with pytest.raises(RuntimeError, match='structure'):
        target.load_state_dict(state)
    torch.testing.assert_close(target[0].weight, torch.tensor([1., 2.]))
    assert source[0].get_extra_state()['input_params'] == ['z', 'x']


@pytest.mark.parametrize('bad', [torch.tensor([float('nan'), 0.]), torch.ones(3),
                                torch.ones(2, dtype=torch.complex64),
                                torch.ones(2, device='meta')])
def test_bad_checkpoint_weights_rejected_before_copy(bad):
    layer = QuantumLayer(quantum())
    before = layer.weight.detach().clone()
    state = layer.state_dict()
    state['weight'] = bad
    with pytest.raises(RuntimeError, match='weight'):
        layer.load_state_dict(state)
    torch.testing.assert_close(layer.weight, before)


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('value', [1e40, -1e40])
def test_checkpoint_dtype_overflow_rejected_before_copy(nested, strict, value):
    source = QuantumLayer(quantum(), initial_weights=[value, .2], dtype=torch.float64)
    target = QuantumLayer(quantum(), initial_weights=[.3, -.4], dtype=torch.float32)
    before = target.weight.detach().clone()
    original_parameter = target.weight
    source_model = torch.nn.Sequential(source) if nested else source
    target_model = torch.nn.Sequential(target) if nested else target
    with pytest.raises(RuntimeError, match='finite|overflow|dtype'):
        target_model.load_state_dict(source_model.state_dict(), strict=strict)
    assert target.weight is original_parameter
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('assign', [False, True])
@pytest.mark.parametrize('source_dtype,target_dtype', [
    (torch.float64, torch.float32), (torch.float32, torch.float64)])
def test_checkpoint_cross_dtype_loading(nested, assign, source_dtype, target_dtype):
    source = QuantumLayer(quantum(), initial_weights=[.1, -.2], dtype=source_dtype)
    target = QuantumLayer(quantum(), initial_weights=[.3, -.4], dtype=target_dtype)
    target.weight.requires_grad_(False)
    source_model = torch.nn.Sequential(source) if nested else source
    target_model = torch.nn.Sequential(target) if nested else target
    result = target_model.load_state_dict(source_model.state_dict(), assign=assign)
    assert not result.missing_keys and not result.unexpected_keys
    expected_dtype = source_dtype if assign else target_dtype
    assert target.weight.dtype == expected_dtype
    assert not target.weight.requires_grad
    torch.testing.assert_close(target.weight, source.weight.to(expected_dtype), rtol=0, atol=0)
    x = inputs(dtype=expected_dtype)
    target(x).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('nested', [False, True])
def test_checkpoint_assign_preserves_finite_source_dtype(nested):
    source = QuantumLayer(quantum(), initial_weights=[1e40, -1e40], dtype=torch.float64)
    target = QuantumLayer(quantum(), dtype=torch.float32)
    source_model = torch.nn.Sequential(source) if nested else source
    target_model = torch.nn.Sequential(target) if nested else target
    target_model.load_state_dict(source_model.state_dict(), assign=True)
    assert target.weight.dtype == torch.float64
    assert torch.isfinite(target.weight).all()
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_complex_unitary_and_observable_metadata(method):
    def model(matrix, coefficient=1.):
        q = Ansatz(1)
        q.unitary(UnitaryGate('custom', 1).with_matrix(matrix), [0])
        q.ry(0, Parameter('x') + Parameter('t'))
        q.set_parameter_roles(input_params=['x'], weight_params=['t'])
        pauli = PauliString.from_str('X')
        pauli.phase = Phase(1)
        q.set_measurement(hams=[Hamiltonian.from_list([(pauli, -1j * coefficient)])])
        q.set_differentiator(method)
        return QuantumLayer(q, initial_weights=[.3], dtype=torch.float64)
    matrix = np.array([[1, 0], [0, 1j]])
    source, target = model(matrix), model(matrix)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    target.load_state_dict(state)
    x = torch.tensor([[.2]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(target(x), source(x))
    assert torch.autograd.gradcheck(lambda a: target(a), (x,))
    for incompatible in (model(np.eye(2)), model(matrix, coefficient=.5)):
        with pytest.raises(RuntimeError, match='structure'):
            incompatible.load_state_dict(state)


def test_default_differentiator_metadata_and_tensor_initialization():
    q = quantum()
    q._differentiator = None
    tensor = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
    layer = QuantumLayer(q, initial_weights=tensor, dtype=torch.float64)
    other = QuantumLayer(quantum(), dtype=torch.float64)
    other.load_state_dict(layer.state_dict())
    with torch.no_grad():
        tensor.add_(1.)
    torch.testing.assert_close(layer.weight, torch.tensor([.1, .2], dtype=torch.float64),
                               rtol=0, atol=0)
    torch.testing.assert_close(other(inputs()), layer(inputs()))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('batch_shape', [(2, 3), (2, 1, 3)])
@pytest.mark.parametrize('hamiltonian', [False, True])
def test_multidimensional_batch_values_and_full_gradients(method, batch_shape, hamiltonian):
    layer = QuantumLayer(quantum(method, hamiltonian=hamiltonian), dtype=torch.float64)
    count = int(np.prod(batch_shape))
    x = torch.linspace(-.7, .8, count * 2, dtype=torch.float64).reshape(
        *batch_shape, 2).requires_grad_()
    output = layer(x)
    assert output.shape == (*batch_shape, layer.num_outputs)
    upstream = torch.linspace(-.4, 1.2, output.numel(), dtype=torch.float64).reshape(output.shape)
    dx, dw = torch.autograd.grad(output, (x, layer.weight), grad_outputs=upstream)
    # Independent sample graphs check every element, including all leading axes.
    separate = torch.stack([layer(row) for row in x.reshape(count, 2)]).reshape(output.shape)
    expected_dx, expected_dw = torch.autograd.grad(separate, (x, layer.weight), grad_outputs=upstream)
    torch.testing.assert_close(output, separate)
    torch.testing.assert_close(dx, expected_dx)
    torch.testing.assert_close(dw, expected_dw)
    assert torch.autograd.gradcheck(
        lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
        (x, layer.weight))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('layout', ['transpose', 'permute', 'slice', 'expand', 'negative'])
def test_tensor_views_values_and_full_gradients(method, layout):
    layer = QuantumLayer(quantum(method), dtype=torch.float64)
    shape = {'transpose': (2, 3), 'permute': (3, 2, 2), 'slice': (3, 4),
             'expand': (1, 2), 'negative': (3, 2)}[layout]
    base = torch.linspace(-.6, .8, int(np.prod(shape)), dtype=torch.float64).reshape(shape).requires_grad_()
    reference = base.detach().clone().requires_grad_()
    def view(value, *, materialize=False):
        if layout == 'transpose':
            value = value.T
        elif layout == 'permute':
            value = value.transpose(0, 1)
        elif layout == 'slice':
            value = value[:, ::2]
        elif layout == 'expand':
            value = value.expand(3, -1)
        else:
            value = -value if materialize else torch._neg_view(value)
        return value.clone() if materialize else value
    output = layer(view(base))
    expected = layer(view(reference, materialize=True))
    upstream = torch.linspace(-.4, .8, output.numel(), dtype=torch.float64).reshape(
        *reversed(output.shape)).permute(*reversed(range(output.ndim)))
    actual_gradients = torch.autograd.grad(output, (base, layer.weight), grad_outputs=upstream)
    expected_gradients = torch.autograd.grad(expected, (reference, layer.weight), grad_outputs=upstream)
    torch.testing.assert_close(output, expected)
    for actual, wanted in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, wanted)


def test_negative_weight_view_and_multidimensional_freeze():
    layer = QuantumLayer(quantum(), dtype=torch.float64)
    layer.weight = torch.nn.Parameter(torch._neg_view(layer.weight.detach().clone()))
    assert layer.weight.is_neg()
    x = torch.tensor([[[.2, .3], [.5, -.1]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
        (x, layer.weight))
    layer.weight.requires_grad_(False)
    layer(x).square().sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    assert layer.weight.grad is None
    with torch.no_grad():
        assert layer(x).shape == (1, 2, 2)
