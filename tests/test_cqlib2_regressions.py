"""Numerical and end-to-end regressions for the cqlib 2 migration."""
import itertools
from copy import deepcopy

import numpy as np
import pytest
from cqlib.circuit import Circuit, MCGate, Parameter, StandardGate, UnitaryGate
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import Ansatz, CRAML, HEAnsatz
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator
from cqlib_qml.encoder import AngleEncoder, FRQI, NEQR
from cqlib_qml.loss import SoftmaxCrossEntropy
from cqlib_qml.models import HQNN, QNN
from cqlib_qml.optimizer import Adam, OptimizerInitializer, SGD
from cqlib_qml.scheduler import KingScheduler, SchedulerInitializer


def state(circuit, bindings=None):
    result = Statevector(circuit.num_qubits)
    result.apply_circuit(circuit if bindings is None else circuit.assign_parameters(bindings))
    return result


def expectations(circuit, bindings, hams):
    sv = state(circuit, bindings)
    return np.array([ham.expectation_statevector(sv) for ham in hams])


def finite_difference(fun, bindings):
    result = {}
    for symbol in bindings:
        plus, minus = dict(bindings), dict(bindings)
        plus[symbol] += 1e-6
        minus[symbol] -= 1e-6
        result[symbol] = (fun(plus) - fun(minus)) / 2e-6
    return result


GATES = ["rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx", "crx", "cry", "crz",
         "u", "rxy", "xy", "xy2p", "xy2m", "phase", "fsim", "mcry"]


@pytest.mark.parametrize("gate", GATES)
def test_gate_gradients_against_finite_difference(gate):
    circuit = Circuit(3)
    for qubit in range(3):
        circuit.ry(qubit, 0.3 + qubit * 0.4)
        circuit.rz(qubit, 0.2 + qubit * 0.3)
    circuit.cx(0, 1)
    t, p, q = Parameter("t"), Parameter("p"), Parameter("q")
    if gate == "u":
        circuit.u(1, t, p, q)
    elif gate in ("rxy", "fsim"):
        getattr(circuit, gate)(*( [1, 2] if gate == "fsim" else [1]), t, p)
    elif gate == "mcry":
        circuit.append_mc_gate(MCGate(2, StandardGate.RY(t)), [0, 1, 2])
    elif gate in ("rxx", "ryy", "rzz", "rzx", "crx", "cry", "crz"):
        getattr(circuit, gate)(1, 2, t)
    else:
        getattr(circuit, gate)(1, t)
    circuit.rx(1, 0.6)
    circuit.ry(2, 0.8)
    hams = []
    for pauli in ("ZXI", "XYZ", "YZY"):
        ham = Hamiltonian(3)
        ham.add_term(PauliString.from_str(pauli), 1.0)
        hams.append(ham)
    bindings = {s: {"t": 0.37, "p": 0.71, "q": -0.26}[s] for s in circuit.symbols}
    expected = finite_difference(lambda b: expectations(circuit, b, hams), bindings)
    adjoint = AdjointDifferentiator().run(circuit, bindings, state(circuit, bindings).data, hamiltonians=hams)
    shifted = ParameterShiftDifferentiator().run(circuit, bindings, hamiltonians=hams)
    for symbol in bindings:
        np.testing.assert_allclose(adjoint[symbol], expected[symbol], atol=2e-8)
        np.testing.assert_allclose(shifted[symbol], expected[symbol], atol=2e-8)


@pytest.mark.parametrize("shift", [np.pi / 2, np.pi / 4, -np.pi / 3])
def test_shared_expression_chain_rule(shift):
    circuit = Circuit(1)
    t = Parameter("t")
    circuit.ry(0, 2 * t)
    circuit.ry(0, t * t)
    bindings = {"t": 0.3}
    expected = -(2 + 0.6) * np.sin(0.6 + 0.09)
    shifted = ParameterShiftDifferentiator(shift).run(circuit, {t: 0.3}, readouts=[0])
    adjoint = AdjointDifferentiator().run(circuit, {t: 0.3}, state(circuit, bindings).data, readouts=[0])
    np.testing.assert_allclose(shifted["t"], [expected], atol=1e-10)
    np.testing.assert_allclose(adjoint["t"], [expected], atol=1e-10)


def test_fixed_controlled_gate_in_adjoint():
    circuit = Circuit(2)
    circuit.ry(0, Parameter("t"))
    circuit.append_mc_gate(MCGate(1, StandardGate.X), [0, 1])
    bindings = {"t": 0.3}
    result = AdjointDifferentiator().run(circuit, bindings, state(circuit, bindings).data, readouts=[1])
    np.testing.assert_allclose(result["t"], [-np.sin(0.3)])


@pytest.mark.parametrize("cls,levels", [(FRQI, 2), (NEQR, 4)])
def test_qic_state_equivalence_exhaustive(cls, levels):
    # Includes empty/full groups, overlap of Boolean clauses, and zero images.
    for pixels in itertools.product(range(levels), repeat=4):
        image = np.array(pixels, dtype=np.float32).reshape(2, 2)
        plain = state(cls(4, grayscale=levels)(image)).data
        compressed = state(cls(4, grayscale=levels)(image, use_qic=True)).data
        np.testing.assert_allclose(plain, compressed, atol=1e-12, err_msg=str(pixels))


def test_neqr_basis_encodes_actual_levels():
    sv = state(NEQR(4, grayscale=4)(np.array([[0, 1], [2, 3]]))).data
    expected = np.zeros(16)
    # Color bits appear in MSB-to-LSB order on qubits 2 and 3.
    for position, color in enumerate(range(4)):
        index = (position >> 1) | ((position & 1) << 1) | ((color >> 1) << 2) | ((color & 1) << 3)
        expected[index] = 0.5
    np.testing.assert_allclose(sv, expected, atol=1e-12)


@pytest.mark.parametrize("pixel", [0.5, 4.0, np.inf])
def test_neqr_rejects_unrepresentable_pixels(pixel):
    with pytest.raises(ValueError):
        NEQR(4, grayscale=4)(np.full((2, 2), pixel))


@pytest.mark.parametrize("logits,target", [
    ([[1000., -1000.]], [[0., 1.]]),
    ([[-1000., -1001.]], [[0.25, 0.75]]),
    ([[0.2, 0.8]], [[0.4, 0.6]]),
])
def test_softmax_loss_gradient_consistency(logits, target):
    logits, target = np.array(logits), np.array(target)
    loss = SoftmaxCrossEntropy()
    value = loss(logits, target)
    gradient = loss.grads().copy()
    assert np.isfinite(value)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-4
        minus[index] -= 1e-4
        numeric = (loss(plus, target) - loss(minus, target)) / 2e-4
        assert gradient[index] == pytest.approx(numeric, abs=2e-8)
    assert loss(logits + 1e4, target) == pytest.approx(value)


def test_adaptive_scheduler_and_optimizer_state_roundtrip():
    scheduler = KingScheduler(initial_lr=0.1)
    scheduler.current_lr = 0.025
    scheduler.loss_history = [2., 1., 1.]
    optimizer = Adam(lr=0.1, lr_scheduler=scheduler)
    optimizer.step()
    optimizer(np.array([0.4]), np.array([0.7]), "theta", 1.0)
    saved = optimizer.state_dict()
    loaded = OptimizerInitializer(saved)()
    assert loaded.cur_step == optimizer.cur_step
    assert loaded.lr_scheduler.current_lr == scheduler.current_lr
    assert loaded.lr_scheduler.loss_history == scheduler.loss_history
    loaded.lr_scheduler.loss_history.append(99)
    assert loaded.lr_scheduler.loss_history != scheduler.loss_history
    reconstructed = SchedulerInitializer(scheduler.state_dict())()
    assert reconstructed.current_lr == scheduler.current_lr
    assert reconstructed.loss_history == scheduler.loss_history


def test_legacy_optimizer_cache_keys_migrate():
    saved = Adam().state_dict()
    saved["cache"] = {"12345_params0_0": {"mean": np.array([0.4])}}
    loaded = OptimizerInitializer(saved)()
    assert list(loaded.cache) == ["params0_0"]


@pytest.mark.parametrize("reverse", [False, True])
def test_optimizer_dictionary_lr_order_independent(reverse):
    items = [("lr", 0.5), ("lr_scheduler", "ConstantScheduler(lr=0.01)")]
    optimizer = SGD().set_params(dict(reversed(items) if reverse else items))
    assert optimizer(np.array([1.]), np.array([1.]), "weight") == pytest.approx([0.5])
