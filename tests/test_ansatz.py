import numpy as np
import pytest

from cqlib_qml.ansatz import Ansatz, BasicQNN, CRADL, CRAML, HEAnsatz


class TestBasicQNN:
    def test_init(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX", "YY"])
        assert ansatz.num_qubits == 3
        assert ansatz.in_dim == 4

    def test_str(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        assert "BasicQNN" in str(ansatz)

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="The ansatz only supports the following gates"):
            BasicQNN(n_qubits=3, layers=["INVALID"])

    def test_too_few_qubits(self):
        with pytest.raises(ValueError, match="n_qubits should be >= 2"):
            BasicQNN(n_qubits=1, layers=["XX"])

    def test_forward(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ansatz.set_measurement(readouts=[0])
        result = ansatz.forward()
        assert result.shape == (1, 1)


class TestCRADL:
    def test_init(self):
        ansatz = CRADL(n_qubits=4, layers=2)
        assert ansatz.num_qubits == 4
        assert ansatz.in_dim == 8

    def test_too_few_qubits(self):
        with pytest.raises(ValueError, match="n_qubits should be >= 3"):
            CRADL(n_qubits=2, layers=1)

    def test_forward(self):
        ansatz = CRADL(n_qubits=4, layers=1)
        ansatz.set_measurement(readouts=[3])
        result = ansatz.forward()
        assert result.shape == (1, 1)


class TestCRAML:
    def test_init(self):
        ansatz = CRAML(n_qubits=4, layers=2)
        assert ansatz.num_qubits == 4

    def test_num_parameters(self):
        # Each layer uses 2 * (n_qubits - 2) trainable parameters:
        # one shared XX parameter and one shared ZZ parameter per position qubit.
        assert CRAML(n_qubits=3, layers=2).in_dim == 4
        assert CRAML(n_qubits=4, layers=2).in_dim == 8
        assert CRAML(n_qubits=5, layers=3).in_dim == 18

    def test_no_unused_parameters_created(self):
        # _init_parameters must create exactly the parameters the circuit uses.
        ansatz = CRAML(n_qubits=4, layers=2)
        assert len(ansatz._init_parameters(4)) == ansatz.in_dim

    def test_layer_parameter_names(self):
        # Preserve the symbols actually used in pre-migration checkpoints.
        ansatz = CRAML(n_qubits=4, layers=2)
        assert ansatz.symbols == [
            "params0_0", "params0_1", "params0_2", "params0_3",
            "params0_4", "params0_5", "params0_6", "params0_7",
        ]

    def test_too_few_qubits(self):
        with pytest.raises(ValueError, match="n_qubits should be >= 3"):
            CRAML(n_qubits=2, layers=1)

    def test_forward(self):
        ansatz = CRAML(n_qubits=4, layers=1)
        ansatz.set_measurement(readouts=[3])
        result = ansatz.forward()
        assert result.shape == (1, 1)


class TestHEAnsatz:
    def test_init_with_single_qubit_gates(self):
        """Test HEAnsatz with only single-qubit gates."""
        ansatz = HEAnsatz(n_qubits=3, d=2, layers=["RY"])
        # RY: n_qubits = 3 params per layer, d=2 => 6 params
        assert ansatz.num_qubits == 3
        assert ansatz.in_dim == 6

    def test_init_with_single_qubit_gates_only(self):
        ansatz = HEAnsatz(n_qubits=3, d=2, layers=["RY", "RZ"])
        # RY: 3 params, RZ: 3 params = 6 params per layer
        # 2 layers = 12 params
        assert ansatz.in_dim == 12

    def test_init_with_mixed_gates(self):
        """Test HEAnsatz with mixed single and two-qubit gates."""
        ansatz = HEAnsatz(n_qubits=3, d=2, layers=["RY", "CX"])
        # RY: n_qubits = 3 params per layer, d=2 => 6 params
        assert ansatz.num_qubits == 3
        assert ansatz.in_dim == 6

    def test_downstairs_entangler(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CX"], entangler="downstairs")
        assert ansatz.num_qubits == 3
        assert ansatz.in_dim == 0

    def test_downstairs_entangler_with_cry(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CRY"], entangler="downstairs")
        assert ansatz.num_qubits == 3
        # CRY downstairs: n_qubits-1 = 2 params
        assert ansatz.in_dim == 2

    def test_full_entangler(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CRY"], entangler="full")
        assert ansatz.num_qubits == 3
        # CRY full: C(3,2) = 3 params
        assert ansatz.in_dim == 3

    def test_last_target_entangler_with_cry(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CRY"], entangler="last_target")
        assert ansatz.num_qubits == 3
        # CRY last_target: n_qubits-1 = 2 params
        assert ansatz.in_dim == 2

    def test_last_control_entangler_with_cry(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CRY"], entangler="last_control")
        assert ansatz.num_qubits == 3
        # CRY last_control: n_qubits-1 = 2 params
        assert ansatz.in_dim == 2

    def test_cx_no_params(self):
        """CX gate has no parameters, so it doesn't contribute to in_dim."""
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CX"], entangler="downstairs")
        assert ansatz.in_dim == 0

    def test_cz_no_params(self):
        """CZ gate has no parameters, so it doesn't contribute to in_dim."""
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CZ"], entangler="downstairs")
        assert ansatz.in_dim == 0

    def test_cry_has_params(self):
        """CRY gate has parameters."""
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["CRY"], entangler="downstairs")
        assert ansatz.in_dim == 2

    def test_invalid_layers(self):
        with pytest.raises(ValueError, match="Invalid layer"):
            HEAnsatz(n_qubits=3, d=1, layers=["INVALID"])

    def test_invalid_entangler(self):
        with pytest.raises(ValueError, match="Invalid entangler"):
            HEAnsatz(n_qubits=3, d=1, layers=["CX"], entangler="invalid")

    def test_too_few_qubits(self):
        with pytest.raises(ValueError, match="n_qubits should be >= 2"):
            HEAnsatz(n_qubits=1, d=1, layers=["RY"])

    def test_forward(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        result = ansatz.forward()
        assert result.shape == (1, 1)

    def test_assign_parameters(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["RY", "CX"])
        bindings = {sym: 0.5 for sym in ansatz.symbols}
        ansatz.assign_parameters(bindings)
        assert ansatz._bindings == bindings

    def test_with_multiple_single_qubit_gates(self):
        ansatz = HEAnsatz(n_qubits=3, d=2, layers=["RY", "RZ", "CX"])
        # RY: 3 params, RZ: 3 params, CX: 0 params = 6 params per layer
        # 2 layers = 12 params
        assert ansatz.in_dim == 12

    def test_with_cry_full(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["CRY"], entangler="full")
        # CRY full: C(4,2) = 6 params
        assert ansatz.in_dim == 6

    def test_with_cry_downstairs(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["CRY"], entangler="downstairs")
        # CRY downstairs: n_qubits - 1 = 3 params
        assert ansatz.in_dim == 3

    def test_with_cx_and_rz_mixed(self):
        ansatz = HEAnsatz(n_qubits=4, d=3, layers=["RZ", "CX", "RY"])
        # RZ: 4 params, CX: 0 params, RY: 4 params = 8 params per layer
        # 3 layers = 24 params
        assert ansatz.in_dim == 24

    def test_str(self):
        ansatz = HEAnsatz(n_qubits=3, d=1, layers=["RY", "CX"])
        assert "HE-ansatz" in str(ansatz)
        assert "n_qubits=3" in str(ansatz)
        assert "d=1" in str(ansatz)


class TestAnsatzBase:
    def test_instantiate_ansatz(self):
        """Ansatz base class can be instantiated."""
        ansatz = Ansatz(2)
        assert ansatz.num_qubits == 2
        assert ansatz.in_dim == 0

    def test_set_measurement_readouts(self):
        """Test setting measurement with readouts."""
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ansatz.set_measurement(readouts=[0, 1])
        assert ansatz.readouts == [0, 1]
        assert ansatz.out_dim == 2

    def test_set_measurement_hamiltonians(self):
        """Test setting measurement with Hamiltonians."""
        from cqlib.qis import Hamiltonian, PauliString

        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ham = Hamiltonian(3)
        ham.add_term(PauliString.from_str("ZII"), 1.0)
        ansatz.set_measurement(hams=[ham])
        assert ansatz.hams == [ham]
        assert ansatz.out_dim == 1

    def test_freeze_and_unfreeze(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        assert ansatz.trainable is True
        ansatz.freeze()
        assert ansatz.trainable is False
        ansatz.unfreeze()
        assert ansatz.trainable is True

    def test_zero_grad(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ansatz.set_measurement(readouts=[0])
        ansatz.forward()
        ansatz.backward()
        assert len(ansatz.gradients) > 0
        ansatz.zero_grad()
        assert len(ansatz.gradients) == 0

    def test_zero_grad_frozen_allowed(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ansatz.freeze()
        ansatz.zero_grad()
        assert ansatz.gradients == {}

    def test_update_frozen_is_noop(self):
        ansatz = BasicQNN(n_qubits=3, layers=["XX"])
        ansatz.freeze()
        ansatz.update()
        assert not ansatz.trainable
