"""Numerical, state-restoration and public API regressions from the audit."""
from copy import deepcopy
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from cqlib.circuit import Parameter
from cqlib.qis.state import Statevector
from cqlib_qml.algorithms import VQC, QSVM
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.encoder import AngleEncoder, QubitLattice, FRQI, NEQR
from cqlib_qml.layer import Linear
from cqlib_qml.layer.activation import SoftPlus
from cqlib_qml.models import Module
from cqlib_qml.loss import CrossEntropy, SoftmaxCrossEntropy
from cqlib_qml.optimizer import SGD, Adam, AdaGrad, RMSProp, OptimizerInitializer
from cqlib_qml.scheduler import KingScheduler, SchedulerInitializer, ConstantScheduler, NoamScheduler, ExponentialScheduler
from cqlib_qml.data.data_preprocess import downscale, remove_conflict, change_grayscale


def ansatz():
    a = Ansatz(1)
    a.ry(0, Parameter('t'))
    a.set_measurement(readouts=[0])
    a.assign_parameters({'t': 0.3})
    return a


def linear(in_dim=1, out_dim=1, weight=1., act_fn=None):
    layer = Linear(in_dim, out_dim, bias=False, act_fn=act_fn)
    layer.init_params()
    layer.parameters['W'][:] = weight
    return layer


@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_single_parameter_quantum_input_gradient_batch(frozen, method):
    first = linear(2, 1, weight=0.2)
    a = ansatz()
    a.set_differentiator(method)
    if frozen:
        a.freeze()
    model = Module(first, a)
    X = np.array([[1., 2.], [2., -1.], [3., 1.]])
    model.forward(X)
    dx = model.backward(np.ones((3, 1)))
    expected = -np.sin(X @ first.parameters['W'].T)
    np.testing.assert_allclose(dx, expected @ first.parameters['W'], atol=1e-10)
    np.testing.assert_allclose(first.gradients['W'], expected.T @ X, atol=1e-10)


def test_frozen_classical_middle_layer_propagates_gradient_without_training():
    first, middle, last = linear(2, 3), linear(3, 2, 2), linear(2, 1, 3)
    middle.freeze()
    model = Module(first, middle, last)
    X = np.array([[1., 2.], [2., 3.]])
    model.forward(X)
    dx = model.backward(np.ones((2, 1)))
    upstream = np.ones((2, 1)) @ last.parameters['W'] @ middle.parameters['W']
    np.testing.assert_allclose(first.gradients['W'], upstream.T @ X)
    np.testing.assert_allclose(dx, upstream @ first.parameters['W'])
    np.testing.assert_array_equal(middle.gradients['W'], 0)


def test_linear_backward_uses_forward_weights_and_activation():
    layer = linear(act_fn='sigmoid')
    layer.forward(np.ones((2, 1)))
    layer.parameters['W'][:] = 2
    dx = layer.backward(np.ones((2, 1)))
    derivative = np.exp(-1) / (1 + np.exp(-1)) ** 2
    np.testing.assert_allclose(dx, derivative)
    np.testing.assert_allclose(layer.gradients['W'], 2 * derivative)
    layer.zero_grad()
    with pytest.raises(ValueError, match='forward'):
        layer.backward(np.ones((2, 1)))


@pytest.mark.parametrize('loss_cls', [CrossEntropy, SoftmaxCrossEntropy])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_cross_entropy_reduction_and_numeric_gradient(loss_cls, reduction):
    pred = np.array([[.2, .8], [.6, .4]])
    target = np.array([[1., 0.], [0., 1.]])
    loss = loss_cls(reduction=reduction)
    value = loss(pred, target)
    gradient = loss.grads().copy()
    duplicated = loss_cls(reduction=reduction)(np.tile(pred, (2, 1)), np.tile(target, (2, 1)))
    assert duplicated == pytest.approx(value * (2 if reduction == 'sum' else 1))
    for index in np.ndindex(pred.shape):
        plus, minus = pred.copy(), pred.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical = (loss(plus, target) - loss(minus, target)) / 2e-6
        assert gradient[index] == pytest.approx(numerical, abs=1e-8)


def test_cross_entropy_large_batch_is_not_clipped():
    loss = CrossEntropy()
    value = loss(np.full((200, 2), .5), np.tile([1., 0.], (200, 1)))
    assert value == pytest.approx(200 * np.log(2))
    np.testing.assert_allclose(loss.grads()[:, 0], -2)


def test_softplus_extreme_values_and_derivatives():
    activation = SoftPlus()
    x = np.array([-1000., -1., 0., 1., 1000.])
    with np.errstate(over='raise', invalid='raise'):
        output, grad, grad2 = activation.act(x), activation.grad(x), activation.grad2(x)
    assert np.isfinite(output).all() and np.isfinite(grad).all() and np.isfinite(grad2).all()
    np.testing.assert_allclose(output[[0, 2, 4]], [0, np.log(2), 1000])
    np.testing.assert_allclose(grad[[0, 2, 4]], [0, .5, 1])
    np.testing.assert_allclose(grad2[[0, 2, 4]], [0, .25, 0])


@pytest.mark.parametrize('patience', [1, 2, 3])
@pytest.mark.parametrize('optimizer_cls', [SGD, Adam, AdaGrad, RMSProp])
def test_adaptive_scheduler_advances_once_per_optimizer_step(patience, optimizer_cls):
    opt = optimizer_cls(lr_scheduler=KingScheduler(initial_lr=.1, patience=patience, decay=.5))
    for step in range(6):
        history = len(opt.lr_scheduler.loss_history)
        opt.step()
        values = [opt.update(np.zeros(1), np.ones(1), str(i), cur_loss=1.) for i in range(8)]
        for value in values:
            np.testing.assert_allclose(value, values[0])
        assert len(opt.lr_scheduler.loss_history) == min(history + 1, opt.lr_scheduler.max_history)
        assert np.isfinite(opt.lr_scheduler.current_lr)
    restored = OptimizerInitializer(opt.state_dict())()
    for instance in [opt, restored]:
        instance.step()
    np.testing.assert_allclose(opt.update(np.zeros(1), np.ones(1), '0', cur_loss=1.),
                               restored.update(np.zeros(1), np.ones(1), '0', cur_loss=1.))
    assert restored.lr_scheduler.loss_history == opt.lr_scheduler.loss_history


@pytest.mark.parametrize('patience', [0, -1, 1.5])
def test_king_invalid_patience(patience):
    with pytest.raises(ValueError):
        KingScheduler(patience=patience)
    with pytest.raises(ValueError):
        KingScheduler().set_params({'patience': patience})


@pytest.mark.parametrize('cls', [ConstantScheduler, NoamScheduler, ExponentialScheduler, KingScheduler])
def test_documented_scheduler_restore_preserves_next_step(cls):
    scheduler = cls()
    for step in range(5):
        scheduler(step, cur_loss=1.)
    restored = SchedulerInitializer(scheduler.state_dict())()
    assert restored(5, cur_loss=1.) == pytest.approx(scheduler(5, cur_loss=1.))


@pytest.mark.parametrize('pixel', [0, 1])
def test_qubit_lattice_uniform_binary_state(pixel):
    circuit = QubitLattice(4)(np.full((2, 2), pixel))
    state = Statevector(4)
    state.apply_circuit(circuit)
    expected = np.zeros(16)
    expected[15 if pixel else 0] = 1
    np.testing.assert_allclose(state.data, expected)
    with pytest.raises(ValueError):
        QubitLattice(4)(np.array([[2, 3], [2, 3]]))


@pytest.mark.parametrize('cls', [FRQI, NEQR])
@pytest.mark.parametrize('image', [np.zeros((4, 4)), np.array([[np.inf, 0], [0, 0]]), np.zeros(4)])
def test_image_encoder_rejects_wrong_size_or_nonfinite(cls, image):
    with pytest.raises(ValueError):
        cls(n_pixels=4)(image)


@pytest.mark.parametrize('tensor', [False, True])
def test_preprocessing_accepts_numpy_and_torch(tensor):
    X = np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[255., 255.], [255., 255.]]])
    y = np.array([0, 1, 1])
    source = torch.as_tensor(X) if tensor else X
    labels = torch.as_tensor(y) if tensor else y
    resized = downscale(source, (1, 1))
    assert isinstance(resized, torch.Tensor if tensor else np.ndarray)
    assert resized.shape == (3, 1, 1)
    clean, targets = remove_conflict(source, labels, (2, 2))
    np.testing.assert_array_equal(clean, np.ones((1, 2, 2)))
    np.testing.assert_array_equal(targets, [1])
    empty, _ = remove_conflict(source[:2], labels[:2], (2, 2))
    assert empty.shape == (0, 2, 2)


def test_grayscale_does_not_mutate_input():
    original = np.array([.26, .51, .76])
    result = change_grayscale(original, 4)
    np.testing.assert_array_equal(original, [.26, .51, .76])
    np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1])
    assert not np.shares_memory(original, result)


def test_tutorial_custom_layer_and_model_composition():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/tutorials/layer.md').read_text()
    code = text.split('### 使用示例', 1)[1].split('\n---', 1)[0]
    from textwrap import dedent
    namespace = {}
    exec(dedent(code), namespace)
    layer = namespace['MyLayer'](2, 3)
    X = np.array([[1., 2.], [3., 4.]])
    layer.forward(X)
    layer.backward(np.ones((2, 3)))
    np.testing.assert_array_equal(layer.gradients['W'], X.T @ np.ones((2, 3)))
    text = (root / 'docs/tutorials/models.md').read_text()
    code = text.split('### 使用示例', 1)[1].split('### 核心方法', 1)[0]
    namespace = {}
    exec(dedent(code), namespace)
    # Recreate example 3 using its actual documented dimensions.
    a = namespace['ansatz']
    model = Module(Linear(10, a.in_dim), a)
    assert model.forward(np.ones((2, 10))).shape == (2, a.out_dim)


def test_tutorial_softmax_loss_value():
    loss = SoftmaxCrossEntropy()
    assert loss(np.array([[2., 1., .1]]), np.array([[1., 0., 0.]])) == pytest.approx(.4170300163)


def test_optimizer_mid_step_restore_does_not_repeat_scheduler_observation():
    opt = SGD(lr_scheduler=KingScheduler(patience=2))
    opt.step()
    opt.update(np.zeros(1), np.ones(1), 'first', cur_loss=1)
    restored = OptimizerInitializer(opt.state_dict())()
    expected = opt.update(np.zeros(1), np.ones(1), 'second', cur_loss=1)
    actual = restored.update(np.zeros(1), np.ones(1), 'second', cur_loss=1)
    np.testing.assert_array_equal(actual, expected)
    assert restored.lr_scheduler.loss_history == [1]


@pytest.mark.parametrize('patience', [1, 2])
def test_king_short_patience_eventually_decays_on_constant_loss(patience):
    scheduler = KingScheduler(initial_lr=.1, patience=patience, decay=.5)
    for step in range(10):
        scheduler(step, cur_loss=1)
    assert 0 < scheduler.current_lr < .1


def test_linear_retained_input_is_a_snapshot():
    layer = linear()
    X = np.array([[1.], [2.]])
    layer.forward(X)
    X[:] = 100
    layer.backward(np.ones((2, 1)))
    np.testing.assert_array_equal(layer.gradients['W'], [[3.]])
    layer.forward(X, retain_derived=False)
    with pytest.raises(ValueError, match='forward'):
        layer.backward(np.ones((2, 1)))


def test_tutorial_ansatz_training_loop_runs(capsys):
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/tutorials/ansatz.md').read_text()
    codes = re.findall(r'```python\n(.*?)```', text, flags=re.S)
    namespace = {}
    declaration = next(code for code in codes if 'class RotationAnsatz' in code)
    training = next(code for code in codes if 'for epoch in range(100)' in code)
    exec(declaration, namespace)
    exec(training, namespace)
    assert len(re.findall(r'Epoch \d+: loss', capsys.readouterr().out)) == 10
    assert namespace['ansatz']._gradients == {}


def test_tutorial_neqr_example_encodes_integer_color_indices():
    from textwrap import dedent
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/tutorials/encoder.md').read_text().split('## NEQR:', 1)[1]
    code = text.split('### 使用示例', 1)[1].split('**输出：**', 1)[0]
    namespace = {}
    exec(dedent(code), namespace)
    assert namespace['circuit'].num_qubits == 6
    assert np.isin(namespace['img_quantized'], [0, 1, 2, 3]).all()


def test_mnist_preprocessing_to_neqr_uses_integer_color_indices(monkeypatch):
    from types import SimpleNamespace
    from cqlib_qml.data import data_preprocess as prep
    images = torch.tensor([[[0, 85], [170, 255]], [[85, 0], [255, 170]]], dtype=torch.uint8)
    labels = torch.tensor([0, 1])
    monkeypatch.setattr(prep.datasets, 'MNIST', lambda **kwargs: SimpleNamespace(data=images, targets=labels))
    encoder = NEQR(4, grayscale=4)
    train, test = prep.get_mnist_dataloader(classes=[0, 1], resize=(2, 2),
                                           encoding=encoder, batch_size=1, grayscale=4)
    for loader in [train, test]:
        assert len(loader) == 2
        for circuits, target in loader:
            circuit = circuits[0]
            state = Statevector(circuit.num_qubits)
            state.apply_circuit(circuit)
            assert np.count_nonzero(np.abs(state.data) > 1e-10) == 4
    with pytest.raises(ValueError, match='grayscale'):
        prep.get_mnist_dataloader(classes=[0, 1], resize=(2, 2), encoding=encoder, grayscale=2)
