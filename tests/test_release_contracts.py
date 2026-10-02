"""Behavior regressions for the release audit, independent of downloads."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from cqlib_qml.algorithms import QSVM, VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.data import DataLoader, Dataset
from cqlib_qml.data import data_preprocess as prep
from cqlib_qml.encoder import AngleEncoder, BasisEncoder, FRQI, NEQR
from cqlib_qml.loss import BCELoss, MSELoss, SoftmaxCrossEntropy
from cqlib_qml.models import HQNN, QNN
from cqlib_qml.optimizer import OptimizerInitializer, SGD
from cqlib_qml.scheduler import SchedulerInitializer, ExponentialScheduler
from cqlib.qis.state import Statevector


def ansatz():
    circuit = HEAnsatz(2, 1, layers=['RY'])
    circuit.set_measurement(readouts=[0])
    return circuit


@pytest.mark.parametrize('labels,classes,expected', [
    ([0, 1, 0, 1], [1, 0], [1, 0, 1, 0]),
    ([10, 30, 20], [20, 10], [1, 0]),
    (['cat', 'dog'], ['dog', 'cat'], [1, 0]),
])
def test_class_mapping_is_simultaneous(labels, classes, expected):
    labels = np.array(labels)
    original = labels.copy()
    _, result = prep.filter_targets(np.arange(len(labels)), labels, classes)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype.kind in 'iu'
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize('classes', [[], [0, 0]])
def test_invalid_classes_rejected(classes):
    with pytest.raises(ValueError):
        prep.filter_targets(np.arange(2), np.array([0, 1]), classes)


def test_configuration_values_affect_behavior():
    opt = OptimizerInitializer('SGD(lr=0.2, clip_norm=1.0)')()
    np.testing.assert_allclose(opt.update(np.zeros(2), np.array([3., 4.]), 'w'), [-.12, -.16])
    adam = OptimizerInitializer('adam(decay1=0.5, decay2=0.8)')()
    assert adam.hyperparameters['decay1'] == .5
    assert adam.hyperparameters['decay2'] == .8
    scheduler = SchedulerInitializer('exponential(initial_lr=.2, stage_length=5, decay=.5, staircase=True)')()
    assert scheduler(4) == .2
    assert scheduler(5) == .1
    noam = SchedulerInitializer('noam(model_dim=16, scale_factor=2, warmup_steps=10)')()
    assert noam.hyperparameters['model_dim'] == 16
    assert noam.hyperparameters['scale_factor'] == 2
    assert noam.hyperparameters['warmup_steps'] == 10


@pytest.mark.parametrize('value', [
    'sgd(lr_scheduler=exponential(initial_lr=.2, stage_length=5, decay=.5))',
    'sgd(lr_scheduler="ExponentialScheduler(initial_lr=.2, stage_length=5, decay=.5, staircase=False)")',
])
def test_nested_scheduler(value):
    opt = OptimizerInitializer(value)()
    assert isinstance(opt.lr_scheduler, ExponentialScheduler)
    assert opt.lr_scheduler(5) == .1


@pytest.mark.parametrize('factory,value', [
    (OptimizerInitializer, 'notadam(lr=.2)'),
    (OptimizerInitializer, 'sgd(clip_nrom=1)'),
    (OptimizerInitializer, 'sgd(lr=.1) trailing'),
    (SchedulerInitializer, 'exponential(stage_lenght=5)'),
    (SchedulerInitializer, 'noam(model_dim=__import__("os"))'),
])
def test_bad_config_rejected(factory, value):
    with pytest.raises((ValueError, TypeError)):
        factory(value)()


def test_direct_constructor_rejects_unknown_keyword():
    with pytest.raises(TypeError):
        SGD(clip_nrom=1)


def test_qsvm_sample_weights_match_svc():
    X = np.array([[.1, .2], [.3, .4], [.31, .4], [.9, 1.]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([1., 1., 20., 1.])
    model = QSVM(AngleEncoder()).fit(X, y, sample_weight=weights)
    expected = SVC(kernel='precomputed').fit(model._qkm.kernel(X), y, sample_weight=weights)
    np.testing.assert_allclose(model.dual_coef_, expected.dual_coef_)
    with pytest.raises(TypeError):
        model.fit(X, y, ignored_parameter=True)


@pytest.mark.parametrize('encoder', [FRQI, NEQR])
@pytest.mark.parametrize('batch', [False, True])
def test_image_tensor_matches_numpy(encoder, batch):
    image = np.array([[0., 1.], [1., 0.]])
    if batch:
        image = np.stack([image, 1-image])
    tensor = torch.tensor(image, requires_grad=True)
    first, second = encoder(4)(image), encoder(4)(tensor)
    first = first if isinstance(first, list) else [first]
    second = second if isinstance(second, list) else [second]
    for a, b in zip(first, second):
        left, right = Statevector(a.num_qubits), Statevector(b.num_qubits)
        left.apply_circuit(a)
        right.apply_circuit(b)
        np.testing.assert_allclose(left.data, right.data)


@pytest.mark.parametrize('model_type', [QNN, HQNN])
@pytest.mark.parametrize('values', [[.1], [.1, .2, .3], [[.1, .2]], [np.nan, .2], [np.inf, .2], [1j, .2]])
def test_initial_params_validated_atomically(model_type, values):
    a = ansatz()
    before = dict(a._bindings or {})
    with pytest.raises((ValueError, TypeError)):
        model_type(a, params=np.array(values), **({'out_dim': 2} if model_type is HQNN else {}))
    assert (a._bindings or {}) == before


@pytest.mark.parametrize('model_type', [QNN, HQNN])
def test_freeze_survives_forward(model_type):
    model = model_type(ansatz(), **({'out_dim': 2} if model_type is HQNN else {}))
    model.freeze()
    model.forward(AngleEncoder()(np.array([[.1, .2]])))
    assert all(not net.trainable for net in model._nets)


@pytest.mark.parametrize('model_type', [QNN, HQNN])
def test_inference_invalidates_old_gradients(model_type):
    model = model_type(ansatz(), **({'out_dim': 2} if model_type is HQNN else {}))
    circuits = AngleEncoder()(np.array([[.1, .2], [.3, .4]]))
    output = model.forward(circuits)
    model.backward(np.ones_like(output))
    bindings = dict(model._ansatz._bindings)
    weights = model._linear.parameters['W'].copy() if model_type is HQNN else None
    steps = [net._optimizer.cur_step for net in model._nets]
    model.forward(circuits, trainable=False)
    model.update()
    assert dict(model._ansatz._bindings) == bindings
    assert [net._optimizer.cur_step for net in model._nets] == steps
    if weights is not None:
        np.testing.assert_array_equal(model._linear.parameters['W'], weights)
    with pytest.raises(ValueError, match='forward|training'):
        model.backward(np.ones_like(output))


def test_basis_integer_floats():
    circuits = BasisEncoder()(np.array([0., 1., 2.]))
    for value, circuit in enumerate(circuits):
        state = Statevector(circuit.num_qubits)
        state.apply_circuit(circuit)
        assert np.argmax(abs(state.data)) == value


@pytest.mark.parametrize('values', [[], [1], [None]])
def test_encoding_list_contract(values):
    with pytest.raises((ValueError, TypeError)):
        ansatz().add_encoder(values)


def test_fixed_ansatz_forward():
    a = Ansatz(1)
    a.h(0)
    a.set_measurement(readouts=[0])
    np.testing.assert_allclose(a.forward(), [[0.]], atol=1e-12)
    assert a.backward() == {}


@pytest.mark.parametrize('image,levels', [([np.nan], 3), ([np.inf], 3), ([-.1], 3), ([1.1], 3), ([.5], 2.5), ([.5], True)])
def test_invalid_grayscale_input(image, levels):
    with pytest.raises((ValueError, TypeError)):
        prep.change_grayscale(np.array(image), levels)


@pytest.mark.parametrize('algorithm', ['QNN', 'HQNN'])
def test_partial_validation_batch(algorithm):
    module = __import__(f'cqlib_qml.algorithms.{algorithm}_classification', fromlist=['validate'])
    class Perfect:
        def forward(self, x, trainable=False):
            if algorithm == 'QNN':
                return -(2*x-1).reshape(-1, 1)
            return np.eye(2)[x] * 2
    y = np.array([0, 1, 0])
    loader = DataLoader(Dataset(y, y), batch_size=2, shuffle=False, drop_last=False)
    loss = MSELoss() if algorithm == 'QNN' else SoftmaxCrossEntropy()
    average, accuracy = module.validate(0, Perfect(), loader, loss, 2)
    assert accuracy == 1.
    if algorithm == 'HQNN':
        assert average == pytest.approx(np.log1p(np.exp(-2)))
    with pytest.raises(ValueError):
        module.validate(0, Perfect(), DataLoader(Dataset(y[:0], y[:0]), batch_size=2), loss, 2)


@pytest.mark.parametrize('model_type', [QSVM, VQC])
def test_classifier_change_invalidates_fit(model_type):
    model = QSVM(AngleEncoder()) if model_type is QSVM else VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    model.fit(X, np.array([0, 1]))
    with pytest.raises(ValueError):
        model.set_params(typo_parameter=1)
    model.set_params(encoder=AngleEncoder(mode='dense'))
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_vqc_wrong_feature_count():
    model = VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    model.fit(np.array([[.1, .2], [.3, .4]]), [0, 1])
    for method in [model.predict, model.predict_proba]:
        with pytest.raises(ValueError, match='feature'):
            method(np.array([[.1]]))


def test_preprocess_validation_loader_keeps_all_samples(monkeypatch):
    fake = type('MNIST', (), {'data': torch.tensor([[[0, 255], [255, 0]], [[255, 0], [0, 255]], [[255, 255], [0, 0]]], dtype=torch.uint8), 'targets': torch.tensor([0, 1, 0])})
    monkeypatch.setattr(prep.datasets, 'MNIST', lambda **kwargs: fake())
    _, loader = prep.get_mnist_dataloader([0, 1], (2, 2), FRQI(4), batch_size=2)
    assert not loader._shuffle
    assert sum(len(y) for _, y in loader) == 3


@pytest.mark.parametrize('algorithm', ['QNN', 'HQNN'])
def test_mnist_example_starts_without_tensorboard(algorithm, tmp_path, monkeypatch):
    import cqlib_qml.models as models
    class FakeModel:
        def __init__(self, **kwargs):
            pass
        def forward(self, x, **kwargs):
            raise RuntimeError('training reached')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(models, algorithm, FakeModel)
    monkeypatch.setattr(prep, 'get_mnist_dataloader', lambda *args: ([([None], np.array([0]))], []))
    with patch.dict('sys.modules', {'torch.utils.tensorboard': None}):
        with pytest.raises(RuntimeError, match='training reached'):
            import importlib.util
            source = Path(importlib.util.find_spec(f"cqlib_qml.algorithms.{algorithm}_classification").origin)
            exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), {'__name__': '__main__'})
    assert len(list(tmp_path.glob('*/config.yaml'))) == 1


def test_tutorial_gradient_expressions():
    root = Path(__file__).resolve().parents[1]
    models = (root / 'docs/tutorials/models.md').read_text(encoding='utf-8')
    layer = (root / 'docs/tutorials/layer.md').read_text(encoding='utf-8')
    assert 'qnn.backward(loss_fn.grads(-0.5))' in models
    assert 'dLdy = 2 * (output - target) / output.size' in layer
    expectation, target = np.array([[.2], [-.4]]), np.array([[0.], [1.]])
    loss = BCELoss()
    loss((1-expectation)/2, target)
    gradient = loss.grads(-.5)
    for index in range(2):
        delta = np.zeros_like(expectation)
        delta[index] = 1e-6
        numerical = (BCELoss()((1-expectation-delta)/2, target) - BCELoss()((1-expectation+delta)/2, target)) / 2e-6
        assert gradient[index, 0] == pytest.approx(numerical)


def test_import_does_not_patch_cqlib():
    import subprocess
    import sys
    code = '''from cqlib.circuit import ValueOperation
before = set(vars(ValueOperation))
import cqlib_qml
assert set(vars(ValueOperation)) == before
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_sdist_manifest_includes_release_sources():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / 'MANIFEST.in').read_text(encoding='utf-8')
    assert 'recursive-include docs/tutorials *.md' in manifest
    assert 'recursive-include scripts *.py *.sh' in manifest


def test_release_gate_requires_pytest(tmp_path):
    import os
    import shutil
    import subprocess
    if os.name != 'posix':
        pytest.skip('release wrapper requires POSIX shell execution; wheel validation is portable')
    if shutil.which('bash') is None:
        pytest.skip('bash release wrapper requires bash; wheel validation is portable')
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy(root / 'scripts/release_check.sh', scripts / 'release_check.sh')
    python = tmp_path / 'missing-pytest'
    python.write_text('#!/bin/sh\nif [ "$2" = "pytest" ]; then exit 1; fi\ntouch build-was-attempted\nexit 1\n')
    python.chmod(0o755)
    result = subprocess.run(['bash', str(scripts / 'release_check.sh')], env={**os.environ, 'PYTHON': str(python)}, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'pytest is required' in result.stdout + result.stderr
    assert not (tmp_path / 'build-was-attempted').exists()


def test_linear_tutorial_gradient_matches_finite_difference():
    from textwrap import dedent
    from cqlib_qml.layer import Linear
    text = (Path(__file__).resolve().parents[1] / 'docs/tutorials/layer.md').read_text(encoding='utf-8')
    code = text.split('### 完整训练步骤', 1)[1].split('    # 5. 更新参数', 1)[0]
    namespace = {'np': np, 'Linear': Linear}
    exec(dedent(code), namespace)
    layer, X, target = namespace['layer'], namespace['X'], namespace['target']
    analytic = layer.gradients['W'].copy()
    for index in [(0, 0), (2, 4), (4, 9)]:
        original = layer.parameters['W'][index]
        layer.parameters['W'][index] = original + 1e-6
        positive = np.mean((layer.forward(X, retain_derived=False) - target)**2)
        layer.parameters['W'][index] = original - 1e-6
        negative = np.mean((layer.forward(X, retain_derived=False) - target)**2)
        layer.parameters['W'][index] = original
        assert analytic[index] == pytest.approx((positive-negative)/2e-6, abs=1e-9)


def test_module_inference_invalidates_frozen_intermediate_state():
    from cqlib_qml.models import Module
    from cqlib_qml.layer import Linear
    middle = Ansatz(1)
    from cqlib.circuit import Parameter
    middle.ry(0, Parameter('t'))
    middle.set_measurement(readouts=[0])
    model = Module(Linear(2, 1), middle, Linear(1, 1))
    middle.freeze()
    output = model.forward(np.array([[.1, .2], [.3, .4]]))
    model.backward(np.ones_like(output))
    model.forward(np.array([[.1, .2], [.3, .4]]), retain_derived=False)
    model.update()
    with pytest.raises(ValueError, match='training forward'):
        model.backward(np.ones_like(output))
    assert not middle.trainable


@pytest.mark.parametrize('value', ['SGD(lr=1e-3)', 'Adam(lr=.01, clip_norm=None)', 'SGD(lr_scheduler="constant(lr=.03)")'])
def test_optimizer_string_round_trip(value):
    original = OptimizerInitializer(value)()
    restored = OptimizerInitializer(str(original))()
    assert restored.hyperparameters == original.hyperparameters


def test_legacy_custom_layer_still_updates_after_zero_grad():
    from cqlib_qml.layer import Layer
    class Custom(Layer):
        @property
        def hyperparameters(self):
            return {'layer': 'custom', 'in_dim': 1, 'out_dim': 1}
        def init_params(self):
            self._parameters = {'W': np.array([1.])}
        def forward(self, x):
            self._X = x
            return x * self._parameters['W']
        def backward(self, out):
            self._gradients['W'] = np.array([np.sum(self._X * out)])
            return out * self._parameters['W']
    layer = Custom()
    layer.init_params()
    layer.set_optimizer('sgd(lr=.1)')
    layer.zero_grad()
    layer.forward(np.array([2.]))
    layer.backward(np.array([1.]))
    layer.update()
    np.testing.assert_allclose(layer.parameters['W'], [.8])


@pytest.mark.parametrize('model_type', [QSVM, VQC])
def test_equal_classifier_config_keeps_fit(model_type):
    model = QSVM(AngleEncoder(), C=.7) if model_type is QSVM else VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    model.fit(X, [0, 1])
    before = model.predict(X)
    if model_type is QSVM:
        model.set_params(C=float('0.7'))
    else:
        model.set_params(optimizer=''.join(['a', 'd', 'a', 'm']))
    np.testing.assert_array_equal(model.predict(X), before)


@pytest.mark.parametrize('installed', [True, False])
def test_wheel_verifier_distinguishes_checkout_from_local_venv(tmp_path, monkeypatch, installed):
    import runpy
    import sys
    from types import SimpleNamespace

    source = tmp_path / 'checkout'
    for directory in ['tests', 'docs/tutorials', 'scripts']:
        (source / directory).mkdir(parents=True)
    (source / 'MANIFEST.in').write_text('')
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/verify_wheel.py'))
    module_path = source / ('.wheel-venv/lib/site-packages/cqlib_qml/__init__.py'
                            if installed else 'cqlib_qml/__init__.py')

    def run(command, **kwargs):
        if command[1] == '-c':
            with monkeypatch.context() as context:
                context.setitem(sys.modules, 'cqlib_qml', SimpleNamespace(__file__=str(module_path)))
                exec(command[2], {})

    monkeypatch.setattr(namespace['subprocess'], 'run', run)
    monkeypatch.setattr(sys, 'argv', ['verify_wheel.py', '--tests', str(source / 'tests'),
                                    '--tutorials', str(source / 'docs/tutorials')])
    if installed:
        namespace['main']()
    else:
        with pytest.raises(AssertionError, match='source checkout'):
            namespace['main']()


@pytest.mark.parametrize('name,value', [
    (name, value)
    for name in ['epochs', 'batch_size']
    for value in [-1, 0, 1.5, '2', True, np.bool_(False), np.nan, np.inf]
] + [('epochs', None)])
def test_vqc_invalid_training_config_does_not_fit(name, value):
    model = VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    setattr(model, name, value)
    with patch.object(model, '_create_qnn') as create:
        with pytest.raises(ValueError, match=name):
            model.fit([[.1, .2], [.3, .4]], [0, 1])
        create.assert_not_called()
    assert model._classes is None
    assert model._X_fit is None
    with pytest.raises(NotFittedError):
        model.predict([[.1, .2]])


@pytest.mark.parametrize('name', ['epochs', 'batch_size'])
def test_vqc_invalid_refit_keeps_existing_training_state(name):
    model = VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    model.fit(X, [0, 1])
    before = model.predict(X)
    qnn, classes, training_data = model._qnn, model._classes, model._X_fit
    bindings = dict(model.ansatz._bindings)
    steps = model.ansatz._optimizer.cur_step
    setattr(model, name, -1)
    with pytest.raises(ValueError, match=name):
        model.fit(X, [10, 20])
    assert model._qnn is qnn
    assert model._classes is classes
    assert model._X_fit is training_data
    assert model.ansatz._bindings == bindings
    assert model.ansatz._optimizer.cur_step == steps
    np.testing.assert_array_equal(model.predict(X), before)


@pytest.mark.parametrize('name', ['epochs', 'batch_size'])
def test_vqc_invalid_set_params_is_atomic(name):
    model = VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    model.fit([[.1, .2], [.3, .4]], [0, 1])
    qnn = model._qnn
    old = getattr(model, name)
    with pytest.raises(ValueError, match=name):
        model.set_params(**{name: -1, 'verbose': True})
    assert getattr(model, name) == old
    assert not model.verbose
    assert model._qnn is qnn


@pytest.mark.parametrize('batch_size,expected_steps', [(None, 1), (np.int64(1), 2), (5, 1)])
def test_vqc_valid_training_config_performs_updates(batch_size, expected_steps):
    model = VQC(ansatz(), AngleEncoder(), epochs=np.int64(1), batch_size=batch_size, verbose=False)
    assert model.fit([[.1, .2], [.3, .4]], [0, 1]) is model
    assert model.ansatz._optimizer.cur_step == expected_steps
    assert model.predict([[.1, .2]]).shape == (1,)


@pytest.mark.parametrize('via_checkpoint', [False, True])
@pytest.mark.parametrize('with_backward', [False, True])
def test_restore_discards_old_quantum_training_state(tmp_path, via_checkpoint, with_backward):
    from copy import deepcopy
    from cqlib.circuit import Parameter
    from cqlib_qml.models import Module
    circuit = Ansatz(1)
    circuit.ry(0, Parameter('t'))
    circuit.set_measurement(readouts=[0])
    circuit.set_optimizer('sgd(lr=.1)')
    circuit.assign_parameters({'t': 1.})
    summary = deepcopy(circuit.summary)
    model = Module(circuit)
    model.save_checkpoint(str(tmp_path), 1, 0)
    circuit.assign_parameters({'t': .3})
    model.forward()
    if with_backward:
        model.backward(np.ones((1, 1)))
    if via_checkpoint:
        model.load_checkpoint(str(tmp_path))
        with pytest.raises(ValueError, match='training forward'):
            model.backward(np.ones((1, 1)))
        model.update()
    else:
        circuit.load_params(summary)
    with pytest.raises(ValueError, match='forward'):
        circuit.backward(np.ones((1, 1)))
    circuit.update()
    assert circuit._bindings == {'t': 1.}
    assert circuit._optimizer.cur_step == 0
    assert circuit._jacobian == {}
    assert circuit._gradients == {}
    circuit.forward()
    circuit.backward(np.ones((1, 1)))
    assert circuit._gradients['t'] == pytest.approx(-np.sin(1.))


def test_restore_discards_old_classical_training_state():
    from copy import deepcopy
    from cqlib_qml.layer import Linear
    layer = Linear(2, 1)
    layer.set_optimizer('sgd(lr=.1)')
    X = np.array([[.1, .2]])
    output = layer.forward(X)
    summary = deepcopy(layer.summary())
    layer.backward(np.ones_like(output))
    layer.load_params(summary)
    with pytest.raises(ValueError, match='forward'):
        layer.backward(np.ones_like(output))
    layer.update()
    assert layer._optimizer.cur_step == 0
    for name, value in summary['parameters'].items():
        np.testing.assert_array_equal(layer.parameters[name], value)


@pytest.mark.parametrize('previous_fit', [False, True])
@pytest.mark.parametrize('failure_phase', ['encoding', 'construction', 'training'])
def test_failed_vqc_fit_keeps_previous_state(previous_fit, failure_phase):
    from copy import deepcopy
    from cqlib_qml.models import QNN
    model = VQC(ansatz(), AngleEncoder(), epochs=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    if previous_fit:
        model.fit(X, [0, 1])
        before = model.predict(X)
    qnn, classes, training_data = model._qnn, model._classes, model._X_fit
    circuit = model.ansatz
    summary = deepcopy(circuit.summary)
    if failure_phase == 'encoding':
        target, method = AngleEncoder, '__call__'
    elif failure_phase == 'construction':
        target, method = VQC, '_create_qnn'
    else:
        target, method = QNN, 'update'
    with patch.object(target, method, side_effect=ValueError('fit failed')):
        with pytest.raises(ValueError, match='fit failed'):
            model.fit(X, ['cat', 'dog'])
    assert model.ansatz is circuit
    assert model._qnn is qnn
    assert model._classes is classes
    assert model._X_fit is training_data
    assert model.ansatz._bindings == summary['circuit']['parameters']
    if previous_fit:
        np.testing.assert_array_equal(model.predict(X), before)
    else:
        with pytest.raises(NotFittedError):
            model.predict(X)


def test_successful_vqc_refit_preserves_ansatz_reference():
    circuit = ansatz()
    model = VQC(circuit, AngleEncoder(), epochs=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    model.fit(X, [0, 1])
    model.fit(X, ['cat', 'dog'])
    assert model.ansatz is circuit
    assert model._qnn._ansatz is circuit
    assert model._qnn._nets[0] is circuit
    assert set(model.predict(X)) <= {'cat', 'dog'}


def test_vqc_late_training_failure_preserves_supplied_optimizer():
    import pickle
    from cqlib_qml.models import QNN
    optimizer = SGD(lr=.1, momentum=.3)
    circuit = ansatz()
    encoder = AngleEncoder()
    model = VQC(circuit, encoder, optimizer=optimizer, epochs=1, batch_size=1, verbose=False)
    X = np.array([[.1, .2], [.3, .4]])
    model.fit(X, [0, 1])
    assert circuit._optimizer is optimizer
    assert optimizer.cur_step == 2
    qnn = model._qnn
    before = model.predict(X)
    bindings = dict(circuit._bindings)
    state = pickle.dumps(optimizer.state_dict())
    update = QNN.update
    calls = []

    def fail_after_second_update(candidate, *args, **kwargs):
        update(candidate, *args, **kwargs)
        calls.append(candidate._ansatz._optimizer.cur_step)
        if len(calls) == 2:
            raise ValueError('second batch failed')

    with patch.object(QNN, 'update', fail_after_second_update):
        with pytest.raises(ValueError, match='second batch failed'):
            model.fit(X, ['cat', 'dog'])
    assert len(calls) == 2
    assert model._qnn is qnn
    assert model.ansatz is circuit
    assert model.encoder is encoder
    assert circuit._optimizer is optimizer
    assert circuit._bindings == bindings
    assert pickle.dumps(optimizer.state_dict()) == state
    np.testing.assert_array_equal(model.predict(X), before)


@pytest.mark.parametrize('encoding', ['amplitude', 'angle', 'zz'])
def test_staged_vqc_fit_supports_all_public_encoders(encoding):
    from cqlib_qml.encoder import AmplitudeEncoder, ZZFeatureEncoder
    encoders = {'amplitude': AmplitudeEncoder, 'angle': AngleEncoder, 'zz': ZZFeatureEncoder}
    encoder = encoders[encoding]()
    X = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]]) if encoding == 'amplitude' else np.array([[.1, .2], [.3, .4]])
    circuit = ansatz()
    model = VQC(circuit, encoder, epochs=1, verbose=False)
    model.fit(X, [0, 1])
    model.fit(X, ['cat', 'dog'])
    assert model.ansatz is circuit
    assert model.encoder is encoder
    assert model._qnn._ansatz is circuit
    assert set(model.predict(X)) <= {'cat', 'dog'}
