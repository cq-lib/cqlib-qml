"""Scale invariance and failed-inference behavior at floating-point boundaries."""
import numpy as np
import pytest
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator
from cqlib_qml.encoder import FRQI, AmplitudeEncoder, AngleEncoder
from cqlib_qml.models import QNN, HQNN


def state(circuit):
    vector = Statevector(circuit.num_qubits)
    vector.apply_circuit(circuit)
    return vector.data


@pytest.mark.parametrize('qic', [False, True])
@pytest.mark.parametrize('image', [np.ones((1, 1)), np.ones((2, 2)),
                                  np.array([[0., 1.], [1., 0.]])])
@pytest.mark.parametrize('scale', [1e-13, 1e-310])
def test_frqi_preserves_relative_brightness(image, scale, qic):
    encoder = FRQI(image.size)
    expected = state(encoder(image, use_qic=qic))
    actual = state(encoder(image * scale, use_qic=qic))
    np.testing.assert_allclose(abs(np.vdot(expected, actual)) ** 2, 1., atol=1e-14)


@pytest.mark.parametrize('qic', [False, True])
@pytest.mark.parametrize('size', [1, 2])
def test_frqi_zero_image_has_finite_state(size, qic):
    actual = state(FRQI(size * size)(np.zeros((size, size)), use_qic=qic))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.linalg.norm(actual), 1.)


@pytest.mark.parametrize('scale', [1e-200, 1e-310, np.nextafter(0., 1.), 1e200, 1e308])
@pytest.mark.parametrize('values', [np.array([1., 1.]), np.array([1+1j, 1-1j])])
def test_amplitude_normalizes_extreme_real_and_complex_inputs(scale, values):
    actual = state(AmplitudeEncoder()(scale * values)[0])
    expected = values / np.linalg.norm(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('scale', [1e-200, 1e-310])
@pytest.mark.parametrize('factory', [AdjointDifferentiator, ParameterShiftDifferentiator])
def test_gradient_preserves_subnormal_observable_scale(scale, factory):
    circuit = Circuit(1)
    circuit.ry(0, Parameter('t'))
    bindings = {'t': .3}
    ham = Hamiltonian.from_list([(PauliString.from_str('X'), scale)])
    kwargs = {'hamiltonians': ham}
    diff = factory()
    if isinstance(diff, AdjointDifferentiator):
        kwargs['state_vector'] = state(circuit.assign_parameters(bindings))
    actual = diff.run(circuit, bindings, **kwargs)['t'][0]
    np.testing.assert_allclose(actual / scale, np.cos(.3), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('small', [1e-170, 1e-200])
def test_recursive_amplitude_preserves_representable_gate_angles(small):
    circuit = AmplitudeEncoder()(np.array([1., small]))[0]
    angle = list(circuit)[0].params[0]
    np.testing.assert_allclose(angle / small, 2., rtol=1e-12, atol=0.)
