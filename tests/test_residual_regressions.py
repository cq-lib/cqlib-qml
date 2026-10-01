"""Public input errors and VQC epoch metric regression coverage."""
from unittest.mock import Mock

import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.qis.state import Statevector

from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.differentiator import AdjointDifferentiator
from cqlib_qml.encoder import AngleEncoder


def parameterized_circuit():
    circuit = Circuit(1)
    circuit.ry(0, Parameter('theta'))
    return circuit


@pytest.mark.parametrize('readouts', [[0], [0, 0]])
def test_adjoint_requires_final_state(readouts):
    with pytest.raises(ValueError, match='state_vector is required'):
        AdjointDifferentiator().run(parameterized_circuit(), {'theta': .3}, readouts=readouts)


@pytest.mark.parametrize('state,message', [
    ([1], 'dimension'),
    ([[1, 0]], 'dimension'),
    ([np.nan, 0], 'finite'),
    ([np.inf, 0], 'finite'),
])
def test_adjoint_invalid_final_state(state, message):
    with pytest.raises(ValueError, match=message):
        AdjointDifferentiator().run(parameterized_circuit(), {'theta': .3},
                                   state_vector=state, readouts=[0])


@pytest.mark.parametrize('operation', ['measure', 'reset'])
def test_adjoint_rejects_irreversible_operations(operation):
    circuit = parameterized_circuit()
    getattr(circuit, operation)(0)
    with pytest.raises(ValueError, match='requires unitary evolution'):
        AdjointDifferentiator().run(circuit, {'theta': .3},
                                   state_vector=[1, 0], readouts=[0])


def test_adjoint_accepts_barrier_and_final_state_list():
    circuit = parameterized_circuit()
    circuit.barrier([0])
    state = Statevector(1)
    state.apply_circuit(circuit.assign_parameters({'theta': .3}))
    grads = AdjointDifferentiator().run(circuit, {'theta': .3},
                                      state_vector=state.data.tolist(), readouts=[0, 0])
    np.testing.assert_allclose(grads['theta'], [-np.sin(.3)] * 2, atol=1e-10)


def test_legacy_parameter_registration_does_not_create_unused_parameters():
    ansatz = Ansatz(1)
    theta = Parameter('theta')
    assert ansatz.add_parameter(theta) == (0, True)
    assert ansatz.add_parameter(theta) == (0, False)
    assert len(ansatz.parameters) == 0
    ansatz.ry(0, theta)
    assert list(map(str, ansatz.parameters)) == ['theta']


@pytest.mark.parametrize('batch_size', [2, 5])
def test_multiclass_mse_epoch_log_is_element_mean(monkeypatch, capsys, batch_size):
    X = np.arange(5, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 2, 0, 1])
    predictions = np.array([[.1, .2, .3], [.4, .1, .2], [.3, .5, .7],
                            [.9, .2, .1], [.2, .8, .4]])
    qnn = Mock()
    qnn.forward.side_effect = lambda circuits, trainable: predictions[np.asarray(circuits, dtype=int)]
    model = VQC(HEAnsatz(3, d=1, layers=['RY', 'CX']), AngleEncoder(), readouts=[0, 1, 2], loss='MSE',
                n_classes=3, epochs=1, batch_size=batch_size, verbose=True)
    monkeypatch.setattr(model, '_create_qnn', lambda: qnn)
    monkeypatch.setattr(model, '_encode', lambda data: data[:, 0].astype(int).tolist())
    model.fit(X, y)
    expected = np.mean((predictions - np.eye(3)[y]) ** 2)
    assert f'loss: {expected:.4f}' in capsys.readouterr().out
    assert qnn.backward.call_count == (len(X) + batch_size - 1) // batch_size
