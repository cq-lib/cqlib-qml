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


def test_warm_start_uses_weights_but_resets_optimizer_and_scheduler():
    optimizer = SGD(lr=.02, momentum=.8,
                    lr_scheduler=KingScheduler(initial_lr=.02, patience=2, decay=.8))
    model = estimator(optimizer, warm_start=True).fit(X, Y)
    old = model.ansatz_.weights.copy()
    model.fit(X, Y, max_steps=1)
    assert model.ansatz_._optimizer.cur_step == 1
    assert len(model.ansatz_._optimizer.lr_scheduler.loss_history) == 1
    fresh = estimator(optimizer, initial_point=old)
    fresh._initial_rng_state = deepcopy(model._initial_rng_state)
    # Compare the exact first update using the retained shuffle RNG state.
    order = model._progress['permutation'][:2]
    from cqlib_qml.models import QNN
    a = clone_state(model.ansatz_)
    a.assign_weights(old)
    qnn = QNN(a, optimizer=fresh._fresh_optimizer())
    exps = qnn.forward([model._encode(X)[index] for index in order])
    labels = np.unique(Y, return_inverse=True)[1][order]
    pred, target = fresh._prepare_for_loss(exps, labels)
    loss = fresh._get_loss_fn()
    value = loss(pred, target)
    qnn.backward(loss.grads(-1))
    qnn.update(value)
    np.testing.assert_array_equal(a.weights, model.ansatz_.weights)
    assert optimizer.cur_step == 0 and optimizer.cache == {}


def test_custom_optimizer_preserves_type_and_resets_additional_history():
    class CustomScheduler(KingScheduler):
        def __init__(self):
            super().__init__(initial_lr=.02, patience=2)
            self.calls = 0

        def learning_rate(self, step=None, cur_loss=None):
            self.calls += 1
            return super().learning_rate(step, cur_loss)

        def reset_state(self):
            super().reset_state()
            self.calls = 0
            return self

    class CustomOptimizer(OptimizerBase):
        def __init__(self):
            super().__init__(.02, CustomScheduler())
            self.hyperparameters = {'id': 'CustomOptimizer', 'lr': .02}
            self.calls = 0

        def reset_state(self):
            super().reset_state()
            self.calls = 0
            return self

        def update(self, param, param_grad, param_name, cur_loss=None):
            self.calls += 1
            return param - self._learning_rate(cur_loss) * param_grad

    optimizer = CustomOptimizer()
    optimizer.calls = 91
    optimizer.cur_step = 8
    optimizer.cache = {'old': np.ones(2)}
    optimizer.lr_scheduler.current_lr = .003
    optimizer.lr_scheduler.loss_history = [3., 2.]
    optimizer.lr_scheduler.calls = 8
    def snapshot():
        return deepcopy({**optimizer.__dict__, 'lr_scheduler': optimizer.lr_scheduler.__dict__})
    before = snapshot()
    model = estimator(optimizer).fit(X, Y)
    first = model.ansatz_.weights.copy()
    trained = model.ansatz_._optimizer
    assert type(trained) is CustomOptimizer
    assert trained.cur_step == 12
    assert trained.calls == 12 * model.ansatz_.num_weights
    model.fit(X, Y)
    np.testing.assert_array_equal(model.ansatz_.weights, first)
    model = estimator(optimizer, warm_start=True).fit(X, Y)
    model.fit(X, Y, max_steps=1)
    trained = model.ansatz_._optimizer
    assert type(trained) is CustomOptimizer
    assert trained.cur_step == 1 and trained.calls == model.ansatz_.num_weights
    assert trained.cache == {}
    assert type(trained.lr_scheduler) is CustomScheduler
    assert trained.lr_scheduler.calls == 1
    assert trained.lr_scheduler.loss_history == [model._progress['epoch_loss'] / 2]
    assert_exact_state(snapshot(), before)


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


@pytest.mark.parametrize('optimizer_cls', [SGD, Adam])
@pytest.mark.parametrize('scheduler_cls', [ExponentialScheduler, KingScheduler])
@pytest.mark.parametrize('pause', [1, 4, 5])
def test_checkpoint_restores_next_batch_update_and_final_training(tmp_path, optimizer_cls, scheduler_cls, pause):
    scheduler = (ExponentialScheduler(initial_lr=.02, stage_length=2, decay=.8)
                 if scheduler_cls is ExponentialScheduler else KingScheduler(initial_lr=.02, patience=2, decay=.8))
    opt = optimizer_cls(lr=.02, **({'momentum': .8} if optimizer_cls is SGD else {}), lr_scheduler=scheduler)
    continuous = estimator(opt).fit(X, Y, max_steps=pause)
    uninterrupted = estimator(opt).fit(X, Y)
    continuous.save_checkpoint(tmp_path)
    restored = estimator().load_checkpoint(tmp_path)
    assert restored._progress['next_batch'] == continuous._progress['next_batch']
    assert restored._progress['global_step'] == continuous._progress['global_step']
    np.testing.assert_array_equal(restored._progress['permutation'], continuous._progress['permutation'])
    continuous.resume_fit(X, Y, max_steps=1)
    restored.resume_fit(X, Y, max_steps=1)
    np.testing.assert_array_equal(restored.ansatz_.weights, continuous.ansatz_.weights)
    assert_exact_state(restored.ansatz_._optimizer.state_dict(), continuous.ansatz_._optimizer.state_dict())
    restored.resume_fit(X, Y)
    np.testing.assert_array_equal(restored.ansatz_.weights, uninterrupted.ansatz_.weights)
    assert restored._progress['epoch'] == 3
    extended = estimator(opt).fit(X, Y)
    extended.resume_fit(X, Y, additional_epochs=1)
    restored.resume_fit(X, Y, additional_epochs=1)
    np.testing.assert_array_equal(restored.ansatz_.weights, extended.ansatz_.weights)


def test_fit_resume_and_load_failures_are_atomic(tmp_path, monkeypatch):
    model = estimator().fit(X, Y, max_steps=2)
    before = state_bytes(model)
    with pytest.raises(ValueError):
        model.resume_fit(X[::-1], Y[::-1])
    assert state_bytes(model) == before
    def fail(*args, **kwargs):
        raise RuntimeError('training failure after RNG consumption')
    with monkeypatch.context() as patch:
        patch.setattr(VQC, '_run_training', fail)
        with pytest.raises(RuntimeError):
            model.fit(X, Y)
    assert state_bytes(model) == before
    model.save_checkpoint(tmp_path)
    state = np.load(tmp_path / 'model.npy', allow_pickle=True).item()
    for change in ('version', 'rng', 'roles', 'optimizer', 'progress'):
        bad = deepcopy(state)
        if change == 'version':
            bad['format_version'] = 99
        elif change == 'rng':
            bad['rng_state']['bit_generator'] = 'bad'
        elif change == 'roles':
            bad['model']['parameter_roles'] = [['missing'], []]
        elif change == 'optimizer':
            bad['model']['optimizer']['hyperparameters']['id'] = 'invalid'
        else:
            bad['progress']['next_batch'] = 999
        np.save(tmp_path / 'bad.npy', bad)
        with pytest.raises(ValueError):
            model.load_checkpoint(tmp_path / 'bad.npy')
        assert state_bytes(model) == before


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


@pytest.mark.parametrize('method', ['adjoint', 'parameter_shift'])
def test_component_restore_roles_freeze_differentiator_and_input_bound_state(method):
    source = quantum(method)
    source.freeze()
    source.forward(np.array([[.2, .3]]))
    saved = deepcopy(source.summary)
    source.load_params(saved)
    assert not source.trainable
    assert source.input_params == ['z', 'x'] and source.weight_params == ['u', 't']
    assert source.summary['differentiator'] == saved['differentiator']
    source.forward(np.array([[.2, .3]]))
    assert source.backward().shape == (1, 2)
    assert source.gradients == {}
    source.zero_grad()
    legacy = Ansatz(1)
    legacy.ry(0, Parameter('t'))
    legacy.set_measurement(readouts=[0])
    legacy.forward(np.array([[.2], [.4]]))
    legacy.load_params(deepcopy(legacy.summary))
    assert legacy.gradients == {} and not legacy._forward_valid


def test_module_loader_next_step_restore_and_atomic_load(tmp_path):
    model = network('parameter_shift')
    dataset = Dataset(np.tile(X[:, :1], (1, 3)), np.zeros((len(X), 2)))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False, random_state=13)
    iterator = iter(loader)
    batch, target = next(iterator)
    output = model.forward(batch)
    model.backward(2 * (output - target) / output.size)
    with pytest.raises(ValueError, match='zero_grad'):
        model.save_checkpoint(str(tmp_path), 0, 0)
    model.update()
    model.zero_grad()
    model._nets[1].freeze()
    model.save_checkpoint(str(tmp_path), 0, 0, data_loader=loader)
    restored = network()
    restored_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
    restored.load_checkpoint(str(tmp_path), data_loader=restored_loader)
    actual = next(iterator)
    recovered = next(iter(restored_loader))
    np.testing.assert_array_equal(actual[0], recovered[0])
    assert not restored._nets[1].trainable
    for net in (model, restored):
        net.forward(actual[0])
        net.backward(np.ones((len(actual[0]), 2)))
        net.update()
        net.zero_grad()
    np.testing.assert_array_equal(model.forward(actual[0]), restored.forward(actual[0]))
    state = np.load(tmp_path / 'model.npy', allow_pickle=True).item()
    state['module']['layer2']['parameters']['W'] = np.ones((9, 9))
    np.save(tmp_path / 'bad.npy', state)
    before = state_bytes(restored)
    with pytest.raises(ValueError):
        restored.load_checkpoint(str(tmp_path / 'bad.npy'))
    assert state_bytes(restored) == before


def test_module_loader_restore_preserves_shared_dataset(tmp_path):
    class SharedDataset(Dataset):
        def __deepcopy__(self, memo):
            raise AssertionError("Checkpoint restoration must not copy the dataset")

    values = np.arange(4., dtype=float)
    dataset = SharedDataset(values)
    loader = DataLoader(dataset, shuffle=False, drop_last=False)
    next(iter(loader))
    model = network()
    model.save_checkpoint(tmp_path, 0, 0, data_loader=loader)
    restored_loader = DataLoader(dataset, shuffle=False, drop_last=False)
    restored = network()
    assert restored.load_checkpoint(tmp_path, data_loader=restored_loader) == (0, 1)
    assert restored_loader.dataset is dataset
    assert restored_loader.dataset._datas[0] is values
    assert_exact_state(restored_loader.state_dict(), loader.state_dict())
    dataset[1] = (99.,)
    np.testing.assert_array_equal(next(iter(restored_loader))[0], [99.])


def test_module_loader_mismatch_preserves_model_and_loader(tmp_path):
    dataset = Dataset(np.arange(4.))
    loader = DataLoader(dataset, shuffle=False, drop_last=False)
    next(iter(loader))
    network().save_checkpoint(tmp_path, 0, 0, data_loader=loader)
    restored = network()
    restored._nets[1].assign_weights([.8, .9])
    other_dataset = Dataset(np.arange(4.) + 1)
    other_loader = DataLoader(other_dataset, shuffle=False, drop_last=False)
    before_model = state_bytes(restored)
    before_loader = deepcopy(other_loader.state_dict())
    before_rng = other_loader._rng
    with pytest.raises(ValueError, match='dataset or configuration mismatch'):
        restored.load_checkpoint(tmp_path, data_loader=other_loader)
    assert state_bytes(restored) == before_model
    assert other_loader.dataset is other_dataset
    assert other_loader._rng is before_rng
    assert_exact_state(other_loader.state_dict(), before_loader)


def test_checkpoint_requires_clean_legacy_custom_layer_gradients(tmp_path):
    class LegacyLayer(Layer):
        def __init__(self):
            super().__init__()
            self._in_dim = self._out_dim = 1
            self._parameters = {'W': np.ones((1, 1))}
            self._gradients = {'W': None}

        @property
        def hyperparameters(self):
            return {'layer': 'LegacyLayer', 'in_dim': 1, 'out_dim': 1}

        def init_params(self):
            self.zero_grad()

        def forward(self, x):
            self._X = x
            return x @ self._parameters['W']

        def backward(self, out, **kwargs):
            self._gradients['W'] += self._X.T @ out
            return out @ self._parameters['W'].T

    layer = LegacyLayer()
    model = Module(layer)
    model.set_optimizer('sgd(lr=.1)')
    model.save_checkpoint(tmp_path, 0, 0)  # None gradients are clean.
    original_file = (tmp_path / 'model.npy').read_bytes()
    model.zero_grad()
    model.backward(np.ones_like(model.forward(np.ones((1, 1)))))
    assert not layer._gradient_valid and not layer._tracks_gradient_validity
    for updated in (False, True):
        if updated:
            model.update()
            np.testing.assert_allclose(layer.parameters['W'], [[.9]])
        with pytest.raises(ValueError, match='zero_grad'):
            model.save_checkpoint(tmp_path, 0, 1)
        assert (tmp_path / 'model.npy').read_bytes() == original_file
        assert not (tmp_path / '0_1.npy').exists()
    model.zero_grad()
    model.save_checkpoint(tmp_path, 0, 1)
    restored = Module(LegacyLayer())
    restored.load_checkpoint(tmp_path)
    np.testing.assert_array_equal(restored._nets[0].parameters['W'], layer.parameters['W'])
    np.testing.assert_array_equal(restored._nets[0].gradients['W'], [[0.]])


@pytest.mark.parametrize('field', ['format_version', 'weights', 'parameter_roles',
                                 'trainable', 'training', 'rng_state', 'differentiator'])
def test_component_restore_rejects_missing_metadata_atomically(field):
    source = quantum('parameter_shift')
    saved = deepcopy(source.summary)
    saved.pop(field)
    source.freeze()
    before = state_bytes(source)
    with pytest.raises(ValueError):
        source.load_params(saved)
    assert state_bytes(source) == before


def test_loader_state_validation_preserves_live_state():
    loader = DataLoader(Dataset(X, Y), batch_size=2, drop_last=False, random_state=2)
    next(iter(loader))
    before = state_bytes(loader)
    bad = loader.state_dict()
    bad['next_batch'] = 999
    with pytest.raises(ValueError):
        loader.load_state_dict(bad)
    assert state_bytes(loader) == before


def test_encoded_loader_restore_across_epoch_boundary(tmp_path):
    from cqlib_qml.models import QNN
    circuits = AngleEncoder()(X)
    dataset = Dataset(circuits, Y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False, random_state=21)
    model = QNN(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=22)
    for _ in loader:
        pass
    model.forward(circuits[:1], trainable=False)
    model.save_checkpoint(str(tmp_path), 0, len(loader)-1, data_loader=loader)
    restored = QNN(HEAnsatz(2, 1, layers=['RY']), readouts=[0])
    second = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)
    assert restored.load_checkpoint(str(tmp_path), data_loader=second) == (1, 0)
    expected = next(iter(loader))[1]
    actual = next(iter(second))[1]
    np.testing.assert_array_equal(actual, expected)


def test_summary_snapshot_and_component_failed_load_are_atomic():
    q = quantum('parameter_shift')
    q.forward(np.array([[.1, .2]]))
    saved = q.summary
    old_weights = deepcopy(saved['weights'])
    q.assign_weights([.7, .8])
    assert saved['weights'] == old_weights
    before = state_bytes(q)
    for change in ('missing', 'flag', 'cache'):
        bad = q.summary
        if change == 'missing':
            del bad['rng_state']
        elif change == 'flag':
            bad['trainable'] = 'false'
        else:
            bad['optimizer']['cache']['u'] = np.ones(2)
        with pytest.raises(ValueError):
            q.load_params(bad)
        assert state_bytes(q) == before
    layer = Linear(2, 2, random_state=2)
    layer.forward(np.ones((1, 2)))
    layer.backward(np.ones((1, 2)))
    before = state_bytes(layer)
    saved = layer.summary()
    saved['hyperparameters']['optimizer'] = {'hyperparameters': {'id': 'typo'}}
    with pytest.raises(ValueError):
        layer.load_params(saved)
    assert state_bytes(layer) == before


def test_atomic_checkpoint_write_preserves_existing_file(tmp_path, monkeypatch):
    model = estimator().fit(X, Y)
    model.save_checkpoint(tmp_path)
    previous = (tmp_path / 'model.npy').read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk failure')
    monkeypatch.setattr(np, 'save', fail)
    with pytest.raises(OSError):
        model.save_checkpoint(tmp_path)
    assert (tmp_path / 'model.npy').read_bytes() == previous
    assert list(tmp_path.iterdir()) == [tmp_path / 'model.npy']


@pytest.mark.parametrize('version', [None, 0, 2, True, '1'])
def test_module_rejects_unsupported_checkpoint_versions_atomically(tmp_path, version):
    from cqlib_qml.models import QNN
    model = QNN(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=4)
    model.forward(AngleEncoder()(X), trainable=False)
    model.save_checkpoint(str(tmp_path), 0, 0)
    saved = np.load(tmp_path / 'model.npy', allow_pickle=True).item()
    if version is None:
        saved.pop('format_version')
    else:
        saved['format_version'] = version
    np.save(tmp_path / 'invalid.npy', saved)
    before = state_bytes(model)
    with pytest.raises(ValueError, match='format version'):
        model.load_checkpoint(tmp_path / 'invalid.npy')
    assert state_bytes(model) == before


def test_initial_checkpoint_format_and_custom_generators_round_trip(tmp_path):
    from cqlib_qml._state import FORMAT_VERSION
    assert FORMAT_VERSION == 1
    generator = np.random.Generator(np.random.Philox(4))
    source = estimator()
    source.set_params(random_state=generator)
    source.fit(X, Y, max_steps=1).save_checkpoint(tmp_path)
    saved = np.load(tmp_path / 'model.npy', allow_pickle=True).item()
    assert saved['format_version'] == 1
    assert saved['template']['format_version'] == saved['model']['format_version'] == 1
    layer = Linear(2, 1)
    assert layer.summary()['format_version'] == 1
    loader = DataLoader(Dataset(X, Y))
    assert loader.state_dict()['format_version'] == 1
    restored = estimator().load_checkpoint(tmp_path)
    source.resume_fit(X, Y)
    restored.resume_fit(X, Y)
    np.testing.assert_array_equal(source.ansatz_.weights, restored.ansatz_.weights)


def test_qnn_and_hqnn_owned_random_initialization():
    from cqlib_qml.models import QNN, HQNN
    circuits = AngleEncoder()(X)
    for cls, kwargs in [(QNN, {}), (HQNN, {'out_dim': 2})]:
        first = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        second = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        np.testing.assert_array_equal(first.forward(circuits), second.forward(circuits))


def test_malformed_loader_metadata_never_partially_loads_module(tmp_path):
    source = network()
    loader = DataLoader(Dataset(np.ones((4, 3))), batch_size=2, random_state=3)
    next(iter(loader))
    source.save_checkpoint(str(tmp_path), 0, 0, data_loader=loader)
    saved = np.load(tmp_path / 'model.npy', allow_pickle=True).item()
    target = network()
    target._nets[1].assign_weights([.9, -.8])
    before = state_bytes(target)
    for corruption in ('zero_batch', 'bad_order', 'missing_rng'):
        bad = deepcopy(saved)
        if corruption == 'zero_batch':
            bad['data_loader']['batch_size'] = 0
        elif corruption == 'bad_order':
            bad['data_loader']['permutation'] = np.zeros(4, dtype=int)
        else:
            del bad['data_loader']['rng_state']
        np.save(tmp_path / 'bad.npy', bad)
        with pytest.raises(ValueError):
            target.load_checkpoint(str(tmp_path / 'bad.npy'))  # no live loader supplied
        assert state_bytes(target) == before


def test_vqc_failure_after_real_update_preserves_rng_and_optimizer(monkeypatch):
    from cqlib_qml.models import QNN
    model = estimator(SGD(lr=.02, momentum=.8)).fit(X, Y, max_steps=2)
    before = state_bytes(model)
    rng_state = deepcopy(model._rng.bit_generator.state)
    original = QNN.update
    calls = []
    def fail_after_update(candidate, *args, **kwargs):
        original(candidate, *args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('late update failure')
    monkeypatch.setattr(QNN, 'update', fail_after_update)
    with pytest.raises(RuntimeError):
        model.resume_fit(X, Y)
    assert len(calls) == 2 and state_bytes(model) == before
    assert model._rng.bit_generator.state == rng_state


def test_custom_ansatz_hamiltonian_restore_and_copy_are_independent(tmp_path):
    from cqlib.qis import Hamiltonian, PauliString

    class CustomAnsatz(Ansatz):
        pass

    class CustomModule(Module):
        pass

    observable = Hamiltonian(1)
    observable.add_term(PauliString.from_str('Z'), 1.)
    ansatz = CustomAnsatz(1)
    ansatz.ry(0, Parameter('theta'))
    ansatz.assign_weights([.3])
    ansatz.set_measurement(hams=[observable])
    expected = ansatz.forward(retain_derived=False)
    model = CustomModule(ansatz)
    duplicate = clone_state({'model': model, 'observable': observable})
    copied = duplicate['model']._nets[0].hams[0]
    assert copied is duplicate['observable']
    assert copied is not observable
    copied.scale(2.)
    np.testing.assert_allclose(ansatz.forward(retain_derived=False), expected)
    np.testing.assert_allclose(duplicate['model'].forward(retain_derived=False), 2 * expected)

    summary = ansatz.summary
    ansatz.assign_weights([.8])
    ansatz.load_params(summary)
    np.testing.assert_allclose(ansatz.forward(retain_derived=False), expected)
    model.save_checkpoint(tmp_path, 0, 0)
    ansatz.assign_weights([.9])
    model.load_checkpoint(tmp_path)
    np.testing.assert_allclose(model.forward(retain_derived=False), expected)


@pytest.mark.parametrize('drop_last', [False, True])
@pytest.mark.parametrize('finished', [False, True])
def test_loader_restore_resave_preserves_cursor_and_next_epoch(drop_last, finished):
    dataset = Dataset(np.arange(11))
    source = DataLoader(dataset, batch_size=2, random_state=7, drop_last=drop_last)
    iterator = iter(source)
    for _ in range(len(source) if finished else 2):
        next(iterator)
    saved = source.state_dict()
    first = DataLoader(dataset, batch_size=2, drop_last=drop_last)
    first.load_state_dict(saved)
    assert_exact_state(first.state_dict(), saved)
    second = DataLoader(dataset, batch_size=2, drop_last=drop_last)
    second.load_state_dict(first.state_dict())
    expected = list(source) if finished else []
    if not finished:
        while True:
            try:
                expected.append(next(iterator))
            except StopIteration:
                break
    actual = list(second)
    assert_exact_state(actual, expected)
    assert_exact_state(second.state_dict(), source.state_dict())


def test_vqc_checkpoint_restores_overridden_template_readouts(tmp_path):
    source = estimator(readouts=[0])
    source.ansatz.set_measurement(readouts=[0, 1])
    source.fit(X, Y, max_steps=1)
    source.save_checkpoint(tmp_path)

    restored = estimator().load_checkpoint(tmp_path)
    assert restored.ansatz.readouts == source.ansatz.readouts == [0, 1]
    assert restored.ansatz_.readouts == source.ansatz_.readouts == [0]
    np.testing.assert_array_equal(restored.predict_proba(X), source.predict_proba(X))
    source.resume_fit(X, Y)
    restored.resume_fit(X, Y)
    np.testing.assert_array_equal(restored.ansatz_.weights, source.ansatz_.weights)
    assert_exact_state(restored.ansatz_._optimizer.state_dict(), source.ansatz_._optimizer.state_dict())


@pytest.mark.parametrize('started', [False, True])
@pytest.mark.parametrize('via_module', [False, True])
def test_loader_restore_before_first_batch_preserves_shuffle(tmp_path, started, via_module):
    dataset = Dataset(np.arange(10))
    source = DataLoader(dataset, batch_size=2, drop_last=False, random_state=7)
    if started:
        iter(source)
    saved = source.state_dict()
    restored = DataLoader(dataset, batch_size=2, drop_last=False, random_state=99)
    if not started:
        iter(restored)
    if via_module:
        network().save_checkpoint(tmp_path, 0, 0, data_loader=source)
        network().load_checkpoint(tmp_path, data_loader=restored)
    else:
        restored.load_state_dict(saved)
    assert_exact_state(restored.state_dict(), saved)

    # Direct next() preserves an already-created permutation; fresh iter() shuffles.
    expected = [next(source) for _ in range(len(source))] if started else list(source)
    assert_exact_state(list(restored), expected)
    assert_exact_state(restored.state_dict(), source.state_dict())


def test_loader_restores_format_one_without_iteration_started():
    dataset = Dataset(np.arange(10))
    source = DataLoader(dataset, batch_size=2, drop_last=False, random_state=7)
    next(iter(source))
    saved = source.state_dict()
    saved.pop('iteration_started')
    restored = DataLoader(dataset, batch_size=2, drop_last=False)
    restored.load_state_dict(saved)
    expected = [next(source) for _ in range(len(source) - 1)]
    assert_exact_state(list(restored), expected)
    assert_exact_state(restored.state_dict(), source.state_dict())


@pytest.mark.parametrize('drop_last', [False, True])
def test_loader_restore_after_dataset_replacement_preserves_shuffle(drop_last):
    source = DataLoader(Dataset(np.arange(8)), batch_size=2, random_state=7,
                        drop_last=drop_last)
    next(iter(source))
    rng_state = deepcopy(source._rng.bit_generator.state)
    dataset = Dataset(np.arange(11))
    source.dataset = dataset
    assert_exact_state(source._rng.bit_generator.state, rng_state)
    saved = source.state_dict()
    restored = DataLoader(dataset, batch_size=2, random_state=99, drop_last=drop_last)
    restored.load_state_dict(saved)
    for _ in range(2):
        assert_exact_state(list(restored), list(source))
        assert_exact_state(restored.state_dict(), source.state_dict())
    assert saved['iteration_started'] is False


@pytest.mark.parametrize('bias', [False, True])
def test_linear_init_params_invalidates_recorded_forward(bias):
    layer = Linear(1, 1, bias=bias, random_state=7)
    X = np.array([[2.]])
    layer.forward(X)
    layer.backward(np.ones((1, 1)))
    layer.init_params()
    with pytest.raises(ValueError, match='forward'):
        layer.backward(np.ones((1, 1)))
    assert not layer._gradient_valid
    for gradient in layer.gradients.values():
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
    layer.forward(X)
    np.testing.assert_allclose(layer.backward(np.ones((1, 1))), layer._parameters['W'])


def test_linear_init_params_prevents_update_with_old_adam_momentum():
    layer = Linear(1, 1, random_state=7)
    layer.set_optimizer('adam(lr=.1)')
    X = np.array([[2.]])
    layer.forward(X)
    layer.backward(np.ones((1, 1)))
    layer.update()
    optimizer_state = deepcopy(layer._optimizer.state_dict())
    layer.init_params()
    parameters = deepcopy(layer._parameters)
    layer.update()
    assert_exact_state(layer._parameters, parameters)
    assert_exact_state(layer._optimizer.state_dict(), optimizer_state)
    layer.forward(X)
    layer.backward(np.ones((1, 1)))
    layer.update()
    assert not np.array_equal(layer._parameters['W'], parameters['W'])


def test_linear_set_activation_invalidates_recorded_forward():
    layer = Linear(1, 1, bias=False, random_state=7)
    X = np.array([[-2.]])
    layer.forward(X)
    layer.backward(np.ones((1, 1)))
    gradients = deepcopy(layer.gradients)
    layer.set_activation('relu')
    with pytest.raises(ValueError, match='forward'):
        layer.backward(np.ones((1, 1)))
    assert_exact_state(layer.gradients, gradients)
    layer.zero_grad()
    np.testing.assert_array_equal(layer.forward(X), [[0.]])
    np.testing.assert_array_equal(layer.backward(np.ones((1, 1))), [[0.]])


@pytest.mark.parametrize('hybrid', [False, True])
def test_random_init_discards_old_gradients_and_forward_cache(hybrid):
    q = Ansatz(1)
    q.ry(0, Parameter('theta'))
    q.set_measurement(readouts=[0])
    q.assign_weights([.3])
    components = [q, Linear(1, 1)] if hybrid else [q]
    model = Module(*components, random_state=2)
    model.set_optimizer('sgd(lr=.1)')
    model.backward(np.ones_like(model.forward()))
    assert q.gradients['theta'] != 0

    model.random_init()
    with pytest.raises(ValueError):
        model.backward(np.ones((1, 1)))
    with pytest.raises(ValueError):
        q.backward()
    if hybrid:
        with pytest.raises(ValueError):
            components[1].backward(np.ones((1, 1)))

    weights = q.weights.copy()
    model.update()
    np.testing.assert_array_equal(q.weights, weights)
    for net in components:
        assert net._optimizer.cur_step == 0
        assert not any(np.any(value) for value in net.gradients.values())

    model.backward(np.ones_like(model.forward()))
    model.update()
    assert not np.array_equal(q.weights, weights)
    assert all(net._optimizer.cur_step == 1 for net in components)
