import numpy as np
import pytest

from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector

from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator


def _simulate(circuit: Circuit) -> np.ndarray:
    """Simulate a circuit from |0...0> and return the final state vector."""
    sv = Statevector(circuit.num_qubits)
    sv.apply_circuit(circuit)
    return sv.data


class TestAdjointDifferentiator:
    def test_init(self):
        diff = AdjointDifferentiator()
        assert diff is not None

    def test_run_single_qubit(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        bindings = {"theta": 0.5}
        hams = [Hamiltonian(1)]
        hams[0].add_term(PauliString.from_str("Z"), 1)

        state_vector = np.array([1.0, 0.0], dtype=np.complex128)
        grads = diff.run(circuit, bindings, state_vector=state_vector, hamiltonians=hams)
        assert "theta" in grads
        assert grads["theta"].shape == (1,)

    def test_run_with_readouts(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(2)
        theta = Parameter("theta")
        circuit.ry(0, theta)
        circuit.cx(0, 1)

        bindings = {"theta": 0.5}
        state_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        grads = diff.run(circuit, bindings, state_vector=state_vector, readouts=[0])
        assert "theta" in grads

    def test_run_with_state_vector(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        bindings = {"theta": 0.5}
        state_vector = np.array([1.0, 0.0], dtype=np.complex128)
        hams = [Hamiltonian(1)]
        hams[0].add_term(PauliString.from_str("Z"), 1)

        grads = diff.run(circuit, bindings, state_vector=state_vector, hamiltonians=hams)
        assert "theta" in grads

    def test_multiple_parameters(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(2)
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        circuit.ry(0, theta1)
        circuit.ry(1, theta2)

        bindings = {"theta1": 0.5, "theta2": 0.3}
        state_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        hams = [Hamiltonian(2)]
        hams[0].add_term(PauliString.from_str("ZI"), 1)

        grads = diff.run(circuit, bindings, state_vector=state_vector, hamiltonians=hams)
        assert "theta1" in grads
        assert "theta2" in grads

    def test_run_gradient_values(self):
        # RY(theta) on |0> with Z measurement: <Z> = cos(theta),
        # so d<Z>/d(theta) = -sin(theta).
        diff = AdjointDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        bindings = {"theta": 0.5}
        hams = [Hamiltonian(1)]
        hams[0].add_term(PauliString.from_str("Z"), 1)

        # state_vector is the final state after the forward pass
        # (the assigned circuit simulated from |0>).
        state_vector = _simulate(circuit.assign_parameters(bindings))
        grads = diff.run(circuit, bindings, state_vector=state_vector, hamiltonians=hams)
        assert np.allclose(grads["theta"], [-np.sin(0.5)], atol=1e-8)

    def test_run_gradient_matches_parameter_shift(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(2)
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        circuit.ry(0, theta1)
        circuit.cx(0, 1)
        circuit.rx(1, theta2)

        bindings = {"theta1": 0.5, "theta2": 0.3}
        state_vector = _simulate(circuit.assign_parameters(bindings))
        adjoint_grads = diff.run(circuit, bindings, state_vector=state_vector, readouts=[0])
        shift_grads = ParameterShiftDifferentiator().run(circuit, bindings, readouts=[0])
        for key in shift_grads:
            assert np.allclose(adjoint_grads[key], shift_grads[key], atol=1e-8)

    def test_run_gradient_matches_parameter_shift_hamiltonian(self):
        # RY(0, t1); CX(0, 1); RY(1, t2) with <ZZ>: <ZZ> = cos(t2),
        # so d/dt1 = 0 and d/dt2 = -sin(t2).
        diff = AdjointDifferentiator()
        circuit = Circuit(2)
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        circuit.ry(0, theta1)
        circuit.cx(0, 1)
        circuit.ry(1, theta2)

        bindings = {"theta1": 0.5, "theta2": 0.3}
        hams = [Hamiltonian(2)]
        hams[0].add_term(PauliString.from_str("ZZ"), 1)

        state_vector = _simulate(circuit.assign_parameters(bindings))
        adjoint_grads = diff.run(circuit, bindings, state_vector=state_vector, hamiltonians=hams)
        assert np.allclose(adjoint_grads["theta1"], [0.0], atol=1e-8)
        assert np.allclose(adjoint_grads["theta2"], [-np.sin(0.3)], atol=1e-8)
        shift_grads = ParameterShiftDifferentiator().run(circuit, bindings, hamiltonians=hams)
        for key in shift_grads:
            assert np.allclose(adjoint_grads[key], shift_grads[key], atol=1e-8)

    def test_invalid_circuit(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(1)
        circuit.h(0)

        with pytest.raises(ValueError, match="The input circuit must be a parameterized quantum circuit"):
            diff.run(circuit, {}, readouts=[0])

    def test_unassigned_parameters_raises(self):
        diff = AdjointDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        with pytest.raises(ValueError, match="All parameters in the circuit must be assigned"):
            diff.run(circuit, {}, readouts=[0])


class TestParameterShiftDifferentiator:
    def test_init_default(self):
        diff = ParameterShiftDifferentiator()
        assert diff is not None

    def test_init_custom_shift(self):
        diff = ParameterShiftDifferentiator(shift=np.pi / 4)
        assert diff is not None

    def test_invalid_shift_raises(self):
        with pytest.raises(ValueError, match="shift = 0 rad .* is not allowed"):
            ParameterShiftDifferentiator(shift=0)

    def test_run_single_qubit(self):
        diff = ParameterShiftDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        bindings = {"theta": 0.5}
        hams = [Hamiltonian(1)]
        hams[0].add_term(PauliString.from_str("Z"), 1)

        grads = diff.run(circuit, bindings, hamiltonians=hams)
        assert "theta" in grads
        assert grads["theta"].shape == (1,)

    def test_run_with_readouts(self):
        diff = ParameterShiftDifferentiator()
        circuit = Circuit(2)
        theta = Parameter("theta")
        circuit.ry(0, theta)
        circuit.cx(0, 1)

        bindings = {"theta": 0.5}
        grads = diff.run(circuit, bindings, readouts=[0])
        assert "theta" in grads

    def test_multiple_parameters(self):
        diff = ParameterShiftDifferentiator()
        circuit = Circuit(2)
        theta1 = Parameter("theta1")
        theta2 = Parameter("theta2")
        circuit.ry(0, theta1)
        circuit.ry(1, theta2)

        bindings = {"theta1": 0.5, "theta2": 0.3}
        hams = [Hamiltonian(2)]
        hams[0].add_term(PauliString.from_str("ZI"), 1)

        grads = diff.run(circuit, bindings, hamiltonians=hams)
        assert "theta1" in grads
        assert "theta2" in grads

    def test_invalid_circuit(self):
        diff = ParameterShiftDifferentiator()
        circuit = Circuit(1)
        circuit.h(0)

        with pytest.raises(ValueError, match="The input circuit must be a parameterized quantum circuit"):
            diff.run(circuit, {}, readouts=[0])

    def test_unassigned_parameters_raises(self):
        diff = ParameterShiftDifferentiator()
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.ry(0, theta)

        with pytest.raises(ValueError, match="All parameters in the circuit must be assigned"):
            diff.run(circuit, {}, readouts=[0])