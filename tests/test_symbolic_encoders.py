"""Symbolic composition parity, validation and ownership contracts."""
from copy import deepcopy
from contextlib import contextmanager

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.encoder import AngleEncoder, ZZFeatureEncoder


def body(width=1):
    q = Ansatz(width, random_state=17)
    q.ry(q.qubits[0], Parameter('theta'))
    q.assign_weights([.37])
    q.set_measurement(readouts=list(range(width)))
    return q


def operation_sequence(circuit):
    return [(op.name, tuple(q.id for q in op.qubits)) for op in circuit.operations]


@pytest.mark.parametrize('mode,n', [('classical', 3), ('dense', 3), ('dense', 4)])
def test_angle_operation_order_and_body_boundary(mode, n):
    width = n if mode == 'classical' else (n + 1) // 2
    q = body(width)
    q.cx(0, 1)
    result = AngleEncoder(mode).to_ansatz(q, num_features=n)
    gates = ['RY'] if mode == 'classical' else ['RY', 'RZ']
    expected = [(gate, (i,)) for i in range(width) for gate in gates]
    assert operation_sequence(result) == expected + [('RY', (0,)), ('CX', (0, 1))]


@pytest.mark.parametrize('n,topology,pairs', [
    (1, 'circular', []),
    (4, 'linear', [(0, 1), (1, 2), (2, 3)]),
    (4, 'circular', [(0, 1), (1, 2), (2, 3), (3, 0)]),
    (4, 'full', [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    (2, 'circular', [(0, 1), (1, 0)]),
])
@pytest.mark.parametrize('repeats', [1, 3])
def test_zz_operation_order_repetition_boundaries_and_body(n, topology, pairs, repeats):
    # Explicit edges protect order even though same-layer RZZ matrices commute.
    result = ZZFeatureEncoder(repeats, topology).to_ansatz(body(n), num_features=n)
    layer = ([('H', (i,)) for i in range(n)]
             + [('RZ', (i,)) for i in range(n)]
             + [('RZZ', edge) for edge in pairs])
    assert operation_sequence(result) == layer * repeats + [('RY', (0,))]


def snapshot_value(value):
    """Capture owned state without relying on native object equality/copy hooks."""
    if isinstance(value, Circuit):
        return {'qubits': [q.id for q in value.qubits],
                'operations': [(op.name, [q.id for q in op.qubits],
                                [str(p) for p in op.params]) for op in value.operations]}
    if isinstance(value, Hamiltonian):
        return value.to_matrix().copy()
    if isinstance(value, np.random.Generator):
        return deepcopy(value.bit_generator.state)
    if isinstance(value, dict):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot_value(item) for item in value)
    if hasattr(value, 'state_dict'):
        return (type(value), snapshot_value(value.state_dict()))
    if hasattr(value, '__dict__'):
        return (type(value), snapshot_value(vars(value)))
    return deepcopy(value)


def assert_snapshot_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_snapshot_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_snapshot_equal(left, right)
    else:
        assert actual == expected


@contextmanager
def source_unchanged(q):
    original_objects = dict(vars(q))
    before = snapshot_value(vars(q))
    try:
        yield
    finally:
        assert_snapshot_equal(snapshot_value(vars(q)), before)
        assert all(vars(q)[key] is value for key, value in original_objects.items())


@pytest.mark.parametrize('mode,n', [('classical', 1), ('classical', 3),
                                   ('dense', 1), ('dense', 3), ('dense', 4)])
def test_angle_numeric_circuit_parity(mode, n):
    encoder = AngleEncoder(mode)
    q = body(n if mode == 'classical' else (n + 1) // 2)
    check_numeric_parity(encoder, q, n)


@pytest.mark.parametrize('topology', ['linear', 'circular', 'full'])
@pytest.mark.parametrize('n,repeats', [(1, 1), (2, 2), (3, 2)])
def test_zz_numeric_circuit_parity(topology, n, repeats):
    check_numeric_parity(ZZFeatureEncoder(repeats, topology), body(n), n)


def check_numeric_parity(encoder, q, n):
    combined = encoder.to_ansatz(q, num_features=n)
    for x in (np.linspace(-.3, .7, n), np.linspace(.2, -.5, n)):
        numeric = encoder(x)[0]
        numeric.compose(q._circuit.assign_parameters({'theta': .37}))
        bindings = dict(zip(combined.input_params, x)) | {'theta': .37}
        symbolic = combined._circuit.assign_parameters(bindings)
        np.testing.assert_allclose(symbolic.to_matrix(), numeric.to_matrix(), atol=1e-12, rtol=1e-12)
    assert q.input_params == [] and q._weights == {'theta': .37}


@pytest.mark.parametrize('encoder,width', [(AngleEncoder(), 12),
                                          (AngleEncoder('dense'), 6),
                                          (ZZFeatureEncoder(1), 12)])
def test_numeric_feature_order_without_large_matrix(encoder, width):
    combined = encoder.to_ansatz(Ansatz(width), num_features=np.int64(12), input_prefix='feature')
    assert combined.input_params == [f'feature_{i}' for i in range(12)]
    x = np.linspace(-.4, .8, 12)
    _, bindings = combined._get_fwd_circuits(x)
    assert list(bindings[0]) == combined.input_params
    assert list(bindings[0].values()) == list(x)


@pytest.mark.parametrize('encoder', [AngleEncoder(), ZZFeatureEncoder()])
@pytest.mark.parametrize('bad', [True, np.bool_(True), 1.0, '1', None, 0, -1])
def test_feature_count_validation(encoder, bad):
    q = body()
    with source_unchanged(q), pytest.raises((TypeError, ValueError), match='num_features'):
        encoder.to_ansatz(q, num_features=bad)


@pytest.mark.parametrize('bad', [True, np.bool_(True), 1.5, '2', None, 0, -1])
def test_repeat_validation_only_new_entry(bad):
    encoder = ZZFeatureEncoder(bad)
    q = body()
    with source_unchanged(q), pytest.raises((TypeError, ValueError), match='n_repeats'):
        encoder.to_ansatz(q, num_features=1)


def test_numpy_repeat_integer():
    assert ZZFeatureEncoder(np.int64(2)).to_ansatz(body(), num_features=1).num_inputs == 1
    assert len(ZZFeatureEncoder(0)(np.array([.2]))[0]) == 0


@pytest.mark.parametrize('bad', ['', '0x', 'x-y', 'x y', 'é', 'x\n', 2, None])
def test_prefix_validation(bad):
    q = body()
    with source_unchanged(q), pytest.raises((TypeError, ValueError), match='input_prefix'):
        AngleEncoder().to_ansatz(q, num_features=1, input_prefix=bad)


def test_invalid_source_contracts():
    encoder = AngleEncoder()
    with pytest.raises(TypeError, match='Ansatz'):
        encoder.to_ansatz(Circuit(1), num_features=1)
    q = body(2)
    with source_unchanged(q), pytest.raises(ValueError, match='width'):
        encoder.to_ansatz(q, num_features=1)
    q = body()
    q.set_parameter_roles(input_params=['theta'], weight_params=[])
    with source_unchanged(q), pytest.raises(ValueError, match='input parameters'):
        encoder.to_ansatz(q, num_features=1)
    q = body()
    q.add_encoder(Circuit(1))
    with source_unchanged(q), pytest.raises(ValueError, match='numerical encoder'):
        encoder.to_ansatz(q, num_features=1)
    q = body()
    q.set_parameter_roles(input_params=[], weight_params=['theta'])
    q.rz(0, Parameter('extra'))
    with source_unchanged(q), pytest.raises(ValueError, match='cover every'):
        encoder.to_ansatz(q, num_features=1)
    q = Ansatz(1)
    q.ry(0, Parameter('x_0'))
    with source_unchanged(q), pytest.raises(ValueError, match='conflict'):
        encoder.to_ansatz(q, num_features=1)
    assert encoder.to_ansatz(q, num_features=1, input_prefix='_input').input_params == ['_input_0']


@pytest.mark.parametrize('encoder', [AngleEncoder(), ZZFeatureEncoder()])
@pytest.mark.parametrize('measurement', ['readouts', 'hams'])
@pytest.mark.parametrize('failure,error,message', [
    ('feature_type', TypeError, 'num_features'),
    ('feature_value', ValueError, 'num_features'),
    ('prefix_type', TypeError, 'input_prefix'),
    ('prefix_syntax', ValueError, 'input_prefix'),
    ('width', ValueError, 'width'),
    ('attached', ValueError, 'numerical encoder'),
    ('input_roles', ValueError, 'input parameters'),
    ('stale_roles', ValueError, 'cover every'),
    ('duplicate_roles', ValueError, 'cover every'),
    ('collision', ValueError, 'conflict'),
    ('unsupported_diff', TypeError, 'Unsupported source differentiator'),
])
def test_failed_composition_preserves_all_source_state(encoder, measurement, failure, error, message):
    q = Ansatz(2, random_state=np.random.Generator(np.random.MT19937(41)))
    name = 'x_0' if failure == 'collision' else 'theta'
    q.ry(0, Parameter(name))
    q.rx(1, Parameter('phi'))
    inputs = [name] if failure == 'input_roles' else []
    weights = ['phi'] if inputs else [name, 'phi']
    q.set_parameter_roles(input_params=inputs, weight_params=weights)
    q.assign_weights([-.24] if inputs else [.37, -.24])
    if measurement == 'hams':
        q.set_measurement(hams=[Hamiltonian.from_list([(PauliString.from_str('ZZ'), .7)])])
    else:
        q.set_measurement(readouts=[1, 0])
    q.set_differentiator('parameter_shift', shift=np.pi / 4)
    q.set_optimizer('adam')
    x = [[.19], [.41]] if inputs else None
    q.forward(x)
    q.backward()
    q.update()  # Populate optimizer momentum as well as execution caches.
    q.forward(x)
    q.backward()
    q.backward()
    q.eval()
    assert q._forward_valid and q._gradient_valid and q._jacobian
    assert any(abs(g) > .01 for g in q.gradients.values())
    assert q._optimizer.cache

    options = {'num_features': 2}
    if failure == 'feature_type':
        options['num_features'] = np.bool_(True)
    elif failure == 'feature_value':
        options['num_features'] = 0
    elif failure == 'prefix_type':
        options['input_prefix'] = None
    elif failure == 'prefix_syntax':
        options['input_prefix'] = 'x-y'
    elif failure == 'width':
        options['num_features'] = 1
    elif failure == 'attached':
        q.add_encoder(Circuit(2))
    elif failure in ('stale_roles', 'duplicate_roles'):
        # Inject invalid roles after recording caches so failure must preserve both.
        q._roles[1].append('missing' if failure == 'stale_roles' else name)
    elif failure == 'unsupported_diff':
        class UnsupportedDifferentiator:
            def __init__(self):
                self.config = {'steps': np.array([2, 3])}
        q._differentiator = UnsupportedDifferentiator()

    with source_unchanged(q), pytest.raises(error, match=message):
        encoder.to_ansatz(q, **options)


@pytest.mark.parametrize('bad', [np.bool_(True), 0])
def test_invalid_repeats_preserve_pending_gradients_and_caches(bad):
    q = body(2)
    q.set_differentiator('parameter_shift', shift=np.pi / 4)
    q.forward()
    q.backward()
    q.update()
    q.forward()
    q.backward()
    q.eval()
    assert q._jacobian and q._gradient_valid and q._optimizer.cache
    with source_unchanged(q), pytest.raises((TypeError, ValueError), match='n_repeats'):
        ZZFeatureEncoder(bad).to_ansatz(q, num_features=2)


@pytest.mark.parametrize('explicit', [False, True])
def test_weight_order_partial_bindings_rng_and_native_initialization(explicit):
    q = Ansatz(1, random_state=19)
    q.ry(0, Parameter('z'))
    q.assign_weights([.37])
    q.rz(0, Parameter('a'))
    if explicit:
        q.set_parameter_roles(input_params=[], weight_params=['z', 'a'])
    q.set_measurement(readouts=[0])
    rng_state = deepcopy(q._rng.bit_generator.state)
    combined = AngleEncoder().to_ansatz(q, num_features=1)
    assert combined.weight_params == (['z', 'a'] if explicit else q.symbols)
    assert combined._weights == q._weights == {'z': .37}
    assert q._rng.bit_generator.state == rng_state == combined._rng.bit_generator.state
    with pytest.raises(ValueError, match='incomplete'):
        combined.load_params(combined.summary)
    expected_missing = np.random.default_rng(19).normal()
    assert combined.forward([.2]).shape == (1, 1)
    assert combined._weights == {'z': .37, 'a': expected_missing}
    assert q._weights == {'z': .37} and q._rng.bit_generator.state == rng_state


def test_composition_discards_training_and_subclass_state():
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError('Unrelated subclass state must not be copied')

    class Custom(Ansatz):
        pass

    q = Custom(1, random_state=3)
    q.extra = CannotCopy()
    q.ry(0, Parameter('theta'))
    q.assign_weights([.4])
    q.set_measurement(readouts=[0])
    q.set_differentiator('parameter_shift', shift=np.pi / 4)
    q.forward()
    q.backward()
    q.freeze()
    q.eval()
    gradients = deepcopy(q.gradients)
    jacobian = deepcopy(q._jacobian)
    optimizer = q._optimizer
    result = AngleEncoder().to_ansatz(q, num_features=1)
    assert type(result) is Ansatz and not hasattr(result, 'extra')
    assert result.training and result.trainable and result.updatable
    assert result._optimizer is None and result._assigned_cir is None and result._bindings is None
    assert result._jacobian == {} and not result._forward_valid and not result._gradient_valid
    assert all(value == 0 for value in result.gradients.values())
    assert result._differentiator is not q._differentiator
    assert result._differentiator._shift == np.pi / 4
    before = result.forward([.2]).copy()
    assert q.gradients == gradients and q._optimizer is optimizer
    for name in jacobian:
        np.testing.assert_array_equal(q._jacobian[name], jacobian[name])
    q.assign_weights([2.])
    q.x(0)
    q.readouts.clear()
    q._differentiator._shift = .2
    np.testing.assert_array_equal(result.forward([.2]), before)
    result.assign_weights([-.8])
    assert q._weights == {'theta': 2.}


def test_hamiltonian_isolation():
    q = body(2)
    ham = Hamiltonian.from_list([(PauliString.from_str('ZX'), .7)])
    q.set_measurement(hams=[ham])
    result = AngleEncoder().to_ansatz(q, num_features=2)
    before = result.forward([.2, -.3]).copy()
    q.hams[0].add_term(PauliString.from_str('II'), .9)
    np.testing.assert_array_equal(result.forward([.2, -.3]), before)
    assert not np.allclose(q.hams[0].to_matrix(), result.hams[0].to_matrix())


@pytest.mark.parametrize('ids', [[1, 0], [2, 0, 1]])
def test_qubit_positions_normalized(ids):
    q = Ansatz(ids)
    q.ry(ids[0], Parameter('theta'))
    q.cx(ids[0], ids[1])
    q.assign_weights([.37])
    q.set_measurement(readouts=list(range(len(ids))))
    result = AngleEncoder().to_ansatz(q, num_features=len(ids))
    assert [qubit.id for qubit in result.qubits] == list(range(len(ids)))
    reference = body(len(ids))
    reference.cx(0, 1)
    canonical = AngleEncoder().to_ansatz(reference, num_features=len(ids))
    x = np.linspace(.1, .5, len(ids))
    np.testing.assert_allclose(result.forward(x), canonical.forward(x), atol=1e-12, rtol=1e-12)
    assert [qubit.id for qubit in q.qubits] == ids


def test_sparse_ids_rejected_and_empty_body_supported():
    q = Ansatz([5, 2])
    with source_unchanged(q), pytest.raises(ValueError, match='Qubit IDs'):
        AngleEncoder().to_ansatz(q, num_features=2)
    result = AngleEncoder().to_ansatz(Ansatz(1), num_features=1)
    assert result.training and result.trainable and not result.updatable
    assert result.weight_params == [] and result.readouts is None
    result.set_measurement(readouts=[0])
    np.testing.assert_allclose(result.forward([.2]), [[np.cos(.4)]])
