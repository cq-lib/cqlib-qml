"""Encoded input gradients against closed forms and independent matrix algebra."""
from copy import deepcopy
import io
from pathlib import Path
import re

import numpy as np
import pytest
import torch
from cqlib.circuit import Parameter

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.encoder import AngleEncoder, ZZFeatureEncoder
from cqlib_qml.torch import QuantumLayer


def angle_ansatz(method='adjoint', mode='classical'):
    q = Ansatz(1, random_state=23)
    q.ry(0, Parameter('theta'))
    q.assign_weights([.31])
    q.set_measurement(readouts=[0])
    q.set_differentiator(method)
    return AngleEncoder(mode).to_ansatz(q, num_features=1 if mode == 'classical' else 2)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch', [False, True])
def test_angle_closed_form_including_factor_two(method, dtype, batch):
    q = angle_ansatz(method)
    layer = QuantumLayer(q, dtype=dtype)
    x = torch.tensor([[.2], [-.4]] if batch else [.2], dtype=dtype, requires_grad=True)
    reference = torch.cos(2 * x + layer.weight)
    torch.testing.assert_close(layer(x), reference)
    dx, dw = torch.autograd.grad(layer(x).sum(), (x, layer.weight))
    phase = 2 * x.detach() + layer.weight.detach()
    torch.testing.assert_close(dx, -2 * torch.sin(phase))
    torch.testing.assert_close(dw, -torch.sin(phase).sum().reshape(1))
    assert q._weights == {'theta': .31} and q._optimizer is None


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dense_phase_sensitive_closed_form_and_gradcheck(method, dtype):
    layer = QuantumLayer(angle_ansatz(method, 'dense'), dtype=dtype)
    x = torch.tensor([[.27, .62], [-.4, -.8]], dtype=dtype, requires_grad=True)
    a, phase, t = 2 * x[:, :1], x[:, 1:], layer.weight
    reference = torch.cos(t) * torch.cos(a) - torch.sin(t) * torch.sin(a) * torch.cos(phase)
    torch.testing.assert_close(layer(x), reference)
    expected = torch.autograd.grad(reference.sum(), (x, layer.weight))
    actual = torch.autograd.grad(layer(x).sum(), (x, layer.weight))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    assert actual[0][:, 1].abs().min() > .01
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(
            lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
            (x, layer.weight))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_dense_odd_features_elementwise_gradients_against_closed_form(method, dtype):
    q = Ansatz(2)
    q.ry(0, Parameter('theta'))
    q.ry(1, Parameter('phi'))
    q.set_parameter_roles(input_params=[], weight_params=['theta', 'phi'])
    q.assign_weights([.31, -.42])
    q.set_measurement(readouts=[0, 1])
    q.set_differentiator(method)
    layer = QuantumLayer(AngleEncoder('dense').to_ansatz(q, num_features=3), dtype=dtype)
    x = torch.tensor([[[.27, .62, -.31], [-.4, -.8, .15]],
                      [[.12, -.43, .58], [.39, .24, -.22]]], dtype=dtype, requires_grad=True)
    theta, phi = layer.weight
    first = (torch.cos(theta) * torch.cos(2 * x[..., 0])
             - torch.sin(theta) * torch.sin(2 * x[..., 0]) * torch.cos(x[..., 1]))
    # The odd final feature has an RY encoding and zero phase, hence RY addition.
    reference = torch.stack([first, torch.cos(2 * x[..., 2] + phi)], dim=-1)
    actual = layer(x)
    assert actual.shape == (2, 2, 2) and actual.dtype == dtype
    torch.testing.assert_close(actual, reference)
    for output in range(2):
        upstream = torch.tensor([[.7, -.4], [1.1, -.2]], dtype=dtype)
        expected = torch.autograd.grad((reference[..., output] * upstream).sum(),
                                       (x, layer.weight), retain_graph=True)
        gradients = torch.autograd.grad((actual[..., output] * upstream).sum(),
                                        (x, layer.weight), retain_graph=True)
        for left, right in zip(gradients, expected):
            torch.testing.assert_close(left, right)
        active_inputs = [0, 1] if output == 0 else [2]
        assert gradients[0][..., active_inputs].abs().min() > .001
        inactive_inputs = [2] if output == 0 else [0, 1]
        torch.testing.assert_close(gradients[0][..., inactive_inputs],
                                   torch.zeros_like(gradients[0][..., inactive_inputs]), atol=1e-7, rtol=0)


def kron_all(matrices):
    value = matrices[0]
    for matrix in matrices[1:]:
        value = torch.kron(value, matrix)
    return value


def zz_reference(x, weights, repeats, topology):
    """Small state vector reference using Torch algebra, without circuit APIs."""
    n = len(x)
    identity = torch.eye(2, dtype=torch.complex128)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
    bits = torch.arange(2 ** n)
    signs = [1 - 2 * ((bits >> (n - 1 - i)) & 1) for i in range(n)]
    pairs = [(i, i + 1) for i in range(n - 1)]
    if topology == 'circular' and n > 1:
        pairs.append((n - 1, 0))
    if topology == 'full':
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    state = torch.zeros(2 ** n, dtype=torch.complex128)
    state[0] = 1
    for _ in range(repeats):
        state = kron_all([hadamard] * n) @ state
        for i in range(n):
            state = torch.exp(-1j * np.pi * x[i] * signs[i]) * state
        for i, j in pairs:
            angle = (np.pi - x[i]) * (np.pi - x[j])
            state = torch.exp(-.5j * angle * signs[i] * signs[j]) * state
    t, u = weights
    ry = torch.stack([torch.stack([torch.cos(t / 2), -torch.sin(t / 2)]),
                      torch.stack([torch.sin(t / 2), torch.cos(t / 2)])]).to(torch.complex128)
    rx = torch.stack([torch.stack([torch.cos(u / 2), -1j * torch.sin(u / 2)]),
                      torch.stack([-1j * torch.sin(u / 2), torch.cos(u / 2)])])
    state = kron_all([ry] + [identity] * (n - 2) + [rx]) @ state
    probabilities = state.abs().square()
    return torch.stack([(probabilities * signs[i]).sum() for i in (0, n - 1)])


def zz_ansatz(n, repeats=2, topology='linear', method='adjoint'):
    q = Ansatz(n)
    q.ry(0, Parameter('theta'))
    q.rx(n - 1, Parameter('phi'))
    q.set_parameter_roles(input_params=[], weight_params=['theta', 'phi'])
    q.assign_weights([.31, -.42])
    q.set_measurement(readouts=[0, n - 1])
    q.set_differentiator(method)
    return ZZFeatureEncoder(repeats, topology).to_ansatz(q, num_features=n)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 2)], ids=['sample', 'batch', 'multidim_batch'])
@pytest.mark.parametrize('n,repeats,topology', [(2, 1, 'linear'), (2, 2, 'circular'),
                                              (3, 2, 'full')])
def test_zz_values_and_repeated_symbol_gradients_against_independent_algebra(
        method, dtype, batch_shape, n, repeats, topology):
    layer = QuantumLayer(zz_ansatz(n, repeats, topology, method), dtype=dtype)
    count = int(np.prod(batch_shape)) * n
    x = torch.linspace(-.32, .46, count, dtype=dtype).reshape(*batch_shape, n).requires_grad_()
    # Double reference algebra also avoids float32 roundoff in repeated phases.
    reference = torch.stack([zz_reference(row.double(), layer.weight.double(), repeats, topology)
                             for row in x.reshape(-1, n)]).reshape(*batch_shape, 2)
    actual = layer(x)
    assert actual.shape == batch_shape + (2,) and actual.dtype == dtype
    value_atol, value_rtol = (2e-7, 2e-6) if dtype == torch.float32 else (1e-12, 1e-12)
    torch.testing.assert_close(actual, reference.to(dtype), atol=value_atol, rtol=value_rtol)
    # Nonuniform upstream weights catch elementwise errors and extra batch averaging.
    upstream = torch.linspace(.7, -.4, actual.numel(), dtype=dtype).reshape(actual.shape)
    expected = torch.autograd.grad((reference * upstream.double()).sum(), (x, layer.weight))
    gradients = torch.autograd.grad((actual * upstream).sum(), (x, layer.weight))
    grad_atol, grad_rtol = (2e-6, 2e-5) if dtype == torch.float32 else (2e-10, 2e-9)
    for left, right in zip(gradients, expected):
        assert left.dtype == dtype
        torch.testing.assert_close(left, right, atol=grad_atol, rtol=grad_rtol)
    if dtype == torch.float64 and not batch_shape:
        assert torch.autograd.gradcheck(
            lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)),
            (x, layer.weight))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_encoded_hybrid_network_all_parameters_gradcheck(method):
    torch.manual_seed(31)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh(),
                                QuantumLayer(zz_ansatz(2, method=method)),
                                torch.nn.Linear(2, 1)).double()
    x = torch.tensor([[.3, -.2, .5]], dtype=torch.float64, requires_grad=True)
    parameters = dict(model.named_parameters())
    assert torch.autograd.gradcheck(
        lambda a, *values: torch.func.functional_call(model, dict(zip(parameters, values)), (a,)),
        (x, *parameters.values()))


def test_composed_shared_calls_microbatches_freeze_and_no_grad(monkeypatch):
    layer = QuantumLayer(angle_ansatz(), dtype=torch.float64)
    x = torch.tensor([[[.1], [.4], [-.3]]], dtype=torch.float64, requires_grad=True)
    layer(x).square().mean().backward()
    dx, dw = x.grad.clone(), layer.weight.grad.clone()
    layer.zero_grad()
    x.grad = None
    first, rest = layer(x[:, :1]), layer(x[:, 1:])
    (first.square().mean() / 3 + 2 * rest.square().mean() / 3).backward()
    torch.testing.assert_close(x.grad, dx)
    torch.testing.assert_close(layer.weight.grad, dw)
    layer.zero_grad()
    layer.eval().weight.requires_grad_(False)
    x.grad = None
    layer(x).sum().backward()
    assert x.grad is not None and layer.weight.grad is None
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not compute a Jacobian')
    monkeypatch.setattr(Ansatz, '_bwd', forbidden)
    with torch.no_grad():
        assert layer(x).shape == (1, 3, 1)
        assert not layer(x).requires_grad


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('ids', [[1, 0], [2, 0, 1]])
def test_normalized_qubit_positions_have_correct_gradients(method, ids):
    q = Ansatz(ids)
    q.ry(ids[0], Parameter('theta'))
    q.cx(ids[0], ids[1])
    q.set_measurement(readouts=list(range(len(ids))))
    q.set_differentiator(method)
    layer = QuantumLayer(AngleEncoder().to_ansatz(q, num_features=len(ids)),
                         initial_weights=[.31], dtype=torch.float64)
    x = torch.linspace(-.2, .4, len(ids), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, w: torch.func.functional_call(layer, {'weight': w}, (a,)), (x, layer.weight))


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_encoded_model_and_optimizer_checkpoint_then_equal_update(method):
    torch.manual_seed(29)
    def build():
        return torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float64),
                                   QuantumLayer(zz_ansatz(2, method=method), dtype=torch.float64),
                                   torch.nn.Linear(2, 1, dtype=torch.float64))
    model = build()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    x = torch.tensor([[.2, -.3], [.4, .6]], dtype=torch.float64)
    def step(network, opt):
        opt.zero_grad()
        network(x).square().mean().backward()
        opt.step()
    step(model, optimizer)
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = build()
    restored.load_state_dict(saved['model'])
    other_optimizer = torch.optim.Adam(restored.parameters(), lr=.02)
    other_optimizer.load_state_dict(saved['optimizer'])
    assert_tree_equal(model.state_dict(), restored.state_dict())
    assert_tree_equal(optimizer.state_dict(), other_optimizer.state_dict())
    step(model, optimizer)
    step(restored, other_optimizer)
    assert_tree_equal(model.state_dict(), restored.state_dict())
    assert_tree_equal(optimizer.state_dict(), other_optimizer.state_dict())


def test_checkpoint_matches_actual_structure_not_encoder_identity():
    linear = QuantumLayer(zz_ansatz(2, topology='linear'), dtype=torch.float64)
    full = QuantumLayer(zz_ansatz(2, topology='full'), dtype=torch.float64)
    full.load_state_dict(linear.state_dict())
    assert_tree_equal(linear.state_dict(), full.state_dict())
    circular = QuantumLayer(zz_ansatz(2, topology='circular'), dtype=torch.float64)
    before = deepcopy(circular.state_dict())
    with pytest.raises((RuntimeError, ValueError), match='structure|metadata|incompatible'):
        circular.load_state_dict(linear.state_dict())
    assert_tree_equal(before, circular.state_dict())


def test_native_restore_restores_measurement_and_differentiator():
    q = zz_ansatz(2, method='parameter_shift')
    summary = q.summary
    other = zz_ansatz(2)
    other.set_measurement(readouts=[1, 0])
    other.load_params(summary)
    assert other.readouts == q.readouts
    assert type(other._differentiator) is type(q._differentiator)
    np.testing.assert_allclose(other.forward([.1, .2]), q.forward([.1, .2]))


def test_partial_weights_native_and_torch_initialization_boundaries():
    q = Ansatz(1, random_state=17)
    q.ry(0, Parameter('a'))
    q.assign_weights([.37])
    q.rz(0, Parameter('b'))
    q.set_measurement(readouts=[0])
    combined = AngleEncoder().to_ansatz(q, num_features=1)
    torch.manual_seed(5)
    expected = torch.empty(2, dtype=torch.float64).uniform_(-np.pi, np.pi)
    torch.manual_seed(5)
    layer = QuantumLayer(combined, dtype=torch.float64)
    torch.testing.assert_close(layer.weight, expected)
    explicit = QuantumLayer(combined, initial_weights=[.37, -.2], dtype=torch.float64)
    torch.testing.assert_close(explicit.weight, torch.tensor([.37, -.2], dtype=torch.float64))
    assert combined._weights == q._weights == {'a': .37}
    combined.forward([.2])
    assert combined._weights['a'] == .37 and 'b' in combined._weights
    inherited = QuantumLayer(combined, dtype=torch.float64)
    torch.testing.assert_close(inherited.weight, torch.tensor(combined.weights))


def test_empty_body_still_has_input_gradients():
    q = AngleEncoder().to_ansatz(Ansatz(1), num_features=1)
    q.set_measurement(readouts=[0])
    layer = QuantumLayer(q, dtype=torch.float64)
    x = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    torch.testing.assert_close(x.grad, -2 * torch.sin(2 * x.detach()))
    assert layer.weight.shape == layer.weight.grad.shape == (0,)


@pytest.mark.parametrize('tutorial,heading', [
    ('encoder.md', '## 保留输入梯度的符号编码'),
    ('torch.md', '## 使用编码器自动声明输入'),
])
def test_symbolic_tutorial_code_runs(tutorial, heading):
    text = (Path(__file__).resolve().parents[1] / 'docs/tutorials' / tutorial).read_text(encoding='utf-8')
    section = text.split(heading, 1)[1]
    code = re.findall(r'```python\n(.*?)```', section, re.S)[0]
    namespace = {}
    with torch.random.fork_rng():
        torch.manual_seed(7)
        exec(compile(code, tutorial, 'exec'), namespace)
    if tutorial == 'encoder.md':
        assert namespace['values'].shape == (2, 2)
        assert namespace['encoded'].input_jacobian.shape == (2, 2, 2)
        assert np.isfinite(namespace['encoded'].input_jacobian).all()
    else:
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   and parameter.grad.norm() > 0 for parameter in namespace['model'].parameters())
