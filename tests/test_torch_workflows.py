"""Native Torch optimizers, composed quantum layers and classification training."""
import pytest
import torch
from cqlib.circuit import Parameter

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.torch import QuantumLayer


def quantum_layer(method):
    q = Ansatz(2)
    q.ry(0, Parameter('x0') + Parameter('t0'))
    q.rx(1, Parameter('x1') + Parameter('t1'))
    q.cx(0, 1)
    q.rz(0, .3)
    q.ry(0, .4)
    q.ry(1, -.2)
    q.set_parameter_roles(input_params=['x0', 'x1'], weight_params=['t0', 't1'])
    q.set_measurement(readouts=[0, 1])
    q.set_differentiator(method)
    return QuantumLayer(q, initial_weights=[.3, -.2], dtype=torch.float64)


def hybrid(method, output_size=1):
    return torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float64),
                                torch.nn.Tanh(), quantum_layer(method),
                                torch.nn.Linear(2, output_size, dtype=torch.float64))


def check_gradients(model):
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('optimizer_name', ['SGD', 'AdamW', 'LBFGS'])
def test_native_optimizers_and_repeated_closure(method, optimizer_name):
    torch.manual_seed(5)
    model = hybrid(method)
    samples = torch.tensor([[.2, .7], [.4, -.3], [-.2, .9]], dtype=torch.float64)
    target = torch.tensor([[.2], [-.4], [.5]], dtype=torch.float64)
    options = {'lr': .03}
    if optimizer_name == 'SGD':
        options['momentum'] = .5
    elif optimizer_name == 'LBFGS':
        options = {'lr': .5, 'max_iter': 6, 'line_search_fn': 'strong_wolfe'}
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **options)
    initial = torch.nn.functional.mse_loss(model(samples), target).item()
    initial_weights = model[2].weight.detach().clone()
    closures = 0
    def closure():
        nonlocal closures
        closures += 1
        optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in model.parameters())
        loss = torch.nn.functional.mse_loss(model(samples), target)
        loss.backward()
        check_gradients(model)
        return loss
    for _ in range(7):
        if optimizer_name == 'LBFGS':
            optimizer.step(closure)
        else:
            closure()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    assert torch.nn.functional.mse_loss(model(samples), target).item() < initial * .8
    assert not torch.allclose(model[2].weight, initial_weights)
    if optimizer_name == 'LBFGS':
        assert closures > 7


class SharedQuantumNetwork(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        self.before = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.quantum = quantum_layer(method)
        self.after = torch.nn.Linear(4, 1, dtype=torch.float64)

    def forward(self, samples):
        features = torch.tanh(self.before(samples))
        return self.after(torch.cat((self.quantum(features), self.quantum(features + .2)), dim=-1))


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('shared', [False, True])
def test_composed_network_numeric_and_per_sample_gradients(method, shared):
    torch.manual_seed(9)
    if shared:
        model = SharedQuantumNetwork(method)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float64),
                                    quantum_layer(method), torch.nn.Tanh(), quantum_layer(method),
                                    torch.nn.Linear(2, 1, dtype=torch.float64))
    samples = torch.tensor([[[.2, .3], [-.1, .4], [.7, -.5]]], dtype=torch.float64,
                           requires_grad=True)
    parameters = dict(model.named_parameters())
    def evaluate(x, *weights):
        return torch.func.functional_call(model, dict(zip(parameters, weights)), (x,))
    assert torch.autograd.gradcheck(evaluate, (samples, *parameters.values()))
    upstream = torch.tensor([[[.2], [-.7], [1.3]]], dtype=torch.float64)
    batched = model(samples)
    gradients = torch.autograd.grad(batched, (samples, *parameters.values()), grad_outputs=upstream)
    individual = torch.stack([model(row) for row in samples.reshape(-1, 2)]).reshape(batched.shape)
    individual_gradients = torch.autograd.grad(
        individual, (samples, *parameters.values()), grad_outputs=upstream)
    torch.testing.assert_close(batched, individual)
    for actual, expected in zip(gradients, individual_gradients):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
@pytest.mark.parametrize('task', ['binary', 'multiclass', 'cnn'])
def test_classification_training_receives_all_gradients(method, task):
    torch.manual_seed(11)
    classes = 2 if task == 'binary' else 3
    labels = torch.arange(classes).repeat_interleave(4)
    if task == 'cnn':
        prototypes = torch.zeros(3, 1, 4, 4, dtype=torch.float64)
        prototypes[0, :, :, :2] = 1.
        prototypes[1, :, :, 2:] = 1.
        prototypes[2, :, :2, :] = 1.
        samples = prototypes[labels] + .05 * torch.randn(len(labels), 1, 4, 4, dtype=torch.float64)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 2, dtype=torch.float64), torch.nn.Tanh(), torch.nn.Flatten(),
            torch.nn.Linear(18, 2, dtype=torch.float64), torch.nn.Tanh(), quantum_layer(method),
            torch.nn.Linear(2, classes, dtype=torch.float64))
    else:
        centers = torch.tensor([[-1., -.3], [1., -.3], [0., 1.]], dtype=torch.float64)
        samples = centers[labels] + .05 * torch.randn(len(labels), 2, dtype=torch.float64)
        model = hybrid(method, output_size=1 if task == 'binary' else classes)
    samples.requires_grad_()
    targets = labels.to(torch.float64).unsqueeze(-1) if task == 'binary' else labels
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == 'binary' else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    with torch.no_grad():
        initial = loss_fn(model(samples), targets).item()
    quantum_weights = [p.detach().clone() for net in model.modules()
                       if isinstance(net, QuantumLayer) for p in net.parameters()]
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        samples.grad = None
        loss_fn(model(samples), targets).backward()
        check_gradients(model)
        assert samples.grad is not None and torch.isfinite(samples.grad).all()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(samples)
        final = loss_fn(logits, targets).item()
        predicted = (logits[:, 0] > 0).long() if task == 'binary' else logits.argmax(dim=-1)
        assert final < initial * .7
        assert (predicted == labels).to(torch.float64).mean().item() >= .8
    current_weights = [p for net in model.modules() if isinstance(net, QuantumLayer) for p in net.parameters()]
    for initial_weight, current_weight in zip(quantum_weights, current_weights):
        assert not torch.allclose(initial_weight, current_weight)
