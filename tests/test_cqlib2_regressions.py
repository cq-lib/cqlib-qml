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


def test_expression_parameter_updates():
    ansatz = Ansatz(1)
    ansatz.ry(0, 2 * Parameter("t"))
    ansatz.set_measurement(readouts=[0])
    ansatz.assign_parameters({"t": 0.3})
    ansatz.set_optimizer(SGD(lr=0.1))
    ansatz.forward()
    ansatz.backward(np.ones((1, 1)))
    ansatz.update()
    assert ansatz._bindings["t"] == pytest.approx(0.3 + 0.2 * np.sin(0.6))
    assert ansatz._assigned_cir.symbols == []


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


@pytest.mark.parametrize("loss_name,readouts", [
    ("CrossEntropy", [0, 1]), ("BCE", None), ("MSE", [0]), ("MSE", [0, 1, 2]),
])
def test_vqc_training_step_matches_loss_gradient(loss_name, readouts):
    n_qubits = max(2, len(readouts or [0]))
    ansatz = HEAnsatz(n_qubits, 1, ["RY", "CX"])
    bindings = dict(zip(ansatz.symbols, np.linspace(0.3, 0.7, n_qubits)))
    ansatz.assign_parameters(bindings)
    vqc = VQC(ansatz, AngleEncoder(mode="classical"), readouts=readouts, loss=loss_name,
              n_classes=3 if n_qubits == 3 else 2,
              epochs=1, batch_size=3, optimizer="sgd(lr=0.001)", verbose=False)
    if n_qubits == 3:
        x, y = np.array([[0.1, 0.3, 0.4], [0.5, 0.8, 0.2], [0.7, 0.3, 0.9]]), np.array([0, 1, 2])
    else:
        x, y = np.array([[0.1, 0.3], [0.5, 0.8]]), np.array([0, 1])
    encoded = vqc._encode(x)
    actual_readouts = [0] if readouts is None else readouts
    hams = []
    for qubit in actual_readouts:
        ham = Hamiltonian(n_qubits)
        pauli = "".join("Z" if index == n_qubits - 1 - qubit else "I" for index in range(n_qubits))
        ham.add_term(PauliString.from_str(pauli), 1.0)
        hams.append(ham)
    def value(b):
        rows = []
        for encoder in encoded:
            circuit = Circuit(n_qubits)
            circuit.compose(encoder)
            circuit.compose(ansatz._circuit)
            rows.append(expectations(circuit, b, hams))
        pred, target = vqc._prepare_for_loss(np.array(rows), y)
        return vqc._get_loss_fn()(pred, target)
    gradient = finite_difference(value, bindings)
    before = value(bindings)
    vqc.fit(x, y)
    for symbol in bindings:
        assert (bindings[symbol] - vqc.ansatz_._bindings[symbol]) / 0.001 == pytest.approx(gradient[symbol], abs=1e-7)
    assert value(vqc.ansatz_._bindings) < before


def train_step(model, data):
    model.zero_grad()
    output = model.forward(data)
    model.backward(2 * output)
    model.update(cur_loss=float(np.sum(output**2)))
    model.zero_grad()


@pytest.mark.parametrize("hybrid", [False, True])
def test_checkpoint_restores_predictions_and_next_adam_step(tmp_path, hybrid):
    def make_model():
        ansatz = HEAnsatz(2, 1, ["RY", "CX"])
        if hybrid:
            return HQNN(ansatz, 1, np.array([0.2, 0.4]), "adam", readouts=[0])
        return QNN(ansatz, [0], np.array([0.2, 0.4]), "adam")
    model = make_model()
    data = Circuit(2)
    data.ry(1, 0.3)
    for _ in range(3):
        train_step(model, data)
    expected = model.forward(data, False)
    model.save_checkpoint(str(tmp_path), ep=2, it=7)
    restored = make_model()
    assert restored.load_checkpoint(str(tmp_path)) == (2, 8)
    np.testing.assert_allclose(restored.forward(data, False), expected, atol=1e-14)
    for original, loaded in zip(model._nets, restored._nets):
        assert loaded._optimizer.cur_step == original._optimizer.cur_step == 3
        assert loaded._optimizer.cache.keys() == original._optimizer.cache.keys()
    train_step(model, data)
    train_step(restored, data)
    np.testing.assert_allclose(restored.forward(data, False), model.forward(data, False), atol=1e-14)


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


def test_legacy_ansatz_gate_wrappers():
    ansatz = Ansatz(2)
    theta = Parameter("theta")
    assert ansatz.add_parameter(theta) == (0, True)
    assert ansatz.add_parameter(theta) == (0, False)
    ansatz.multi_control(StandardGate.RY, [0], [1], [theta])
    ansatz.unitary(UnitaryGate("X", 1).with_matrix(np.array([[0, 1], [1, 0]])), [0])
    reference = Circuit(2)
    reference.append_mc_gate(MCGate(1, StandardGate.RY(theta)), [0, 1])
    reference.x(0)
    np.testing.assert_allclose(ansatz._circuit.assign_parameters({"theta": 0.3}).to_matrix(),
                               reference.assign_parameters({"theta": 0.3}).to_matrix())


def test_qnn_parameters_learn_over_multiple_epochs():
    model = QNN(HEAnsatz(2, 1, ["RY", "CX"]), [0], np.array([0.2, 0.4]), "sgd(lr=0.1)")
    circuit = Circuit(2)
    initial = model.forward(circuit, False)[0, 0] ** 2
    for _ in range(50):
        train_step(model, circuit)
    final = model.forward(circuit, False)[0, 0] ** 2
    assert final < initial * 1e-5
    assert abs(model._ansatz._bindings["params0_0"] - 0.2) > 0.1


def test_craml_checkpoint_legacy_symbols():
    ansatz = CRAML(4, 3)
    expected = [f"params{layer}_{index}" for layer, count in [(0, 8), (1, 4)] for index in range(count)]
    assert ansatz.symbols == expected
    bindings = dict(zip(expected, np.linspace(0.1, 0.8, len(expected))))
    ansatz.assign_parameters(bindings)
    restored = CRAML(4, 3)
    restored.load_params(deepcopy(ansatz.summary))
    np.testing.assert_allclose(restored.forward(), ansatz.forward())


def test_optimizer_instances_are_independent_between_components():
    from cqlib_qml.layer import Linear
    from cqlib_qml.models import Module
    configuration = Adam(lr=0.01)
    model = Module(Linear(2, 2), Linear(2, 1))
    model.set_optimizer(configuration)
    first, second = model._nets
    assert first._optimizer is not second._optimizer
    train_step(model, np.array([[0.3, 0.8]]))
    assert first._optimizer.cur_step == second._optimizer.cur_step == 1
    assert configuration.cur_step == 0
    assert first._optimizer.cache["W"]["mean"].shape != second._optimizer.cache["W"]["mean"].shape


def test_checkpoint_rebuilds_native_gate_payloads():
    ansatz = Ansatz(2)
    ansatz.h(0)
    ansatz.phase(0, Parameter("theta"))
    ansatz.multi_control(StandardGate.RY, [0], [1], [Parameter("phi")])
    ansatz.unitary(UnitaryGate("X", 1).with_matrix(np.array([[0, 1], [1, 0]])), [1])
    ansatz.set_measurement(readouts=[0, 1])
    ansatz.assign_parameters({"theta": 0.3, "phi": 0.7})
    restored = Ansatz(2)
    restored.load_params(deepcopy(ansatz.summary))
    np.testing.assert_allclose(restored._assigned_cir.to_matrix(), ansatz._assigned_cir.to_matrix())
    np.testing.assert_allclose(restored.forward(), ansatz.forward())
    ham = Hamiltonian(2)
    ham.add_term(PauliString.from_str("XX"), 1.0)
    expected = finite_difference(lambda b: expectations(ansatz._circuit, b, [ham]), ansatz._bindings)
    gradients = AdjointDifferentiator().run(ansatz._circuit, ansatz._bindings,
                                            state(ansatz._assigned_cir).data, hamiltonians=[ham])
    for symbol in expected:
        np.testing.assert_allclose(gradients[symbol], expected[symbol], atol=1e-8)


def test_craml_intermediate_parameter_names_migrate_with_optimizer():
    ansatz = CRAML(4, 2)
    bindings = dict(zip(ansatz.symbols, np.linspace(0.1, 0.8, 8)))
    ansatz.assign_parameters(bindings)
    ansatz.set_optimizer("adam")
    ansatz.forward()
    ansatz.backward(np.ones((1, 1)))
    ansatz.update()
    summary = deepcopy(ansatz.summary)
    modern = [f"params{layer}_{index}" for layer in range(2) for index in range(4)]
    mapping = dict(zip([str(p) for p in ansatz._init_parameters(4)], modern))
    summary["circuit"]["parameters"] = {mapping[key]: val for key, val in ansatz._bindings.items()}
    for gate in summary["circuit"]["gates"]:
        gate["params"] = [mapping.get(value, value) if isinstance(value, str) else value
                          for value in gate["params"]]
    summary["optimizer"]["cache"] = {mapping[key]: val for key, val in summary["optimizer"]["cache"].items()}
    restored = CRAML(4, 2)
    restored.load_params(summary)
    np.testing.assert_allclose(restored.forward(), ansatz.forward())
    ansatz.zero_grad()
    restored.zero_grad()
    for model in (ansatz, restored):
        model.forward()
        model.backward(np.ones((1, 1)))
        model.update()
    np.testing.assert_allclose(list(restored._bindings.values()), list(ansatz._bindings.values()))
