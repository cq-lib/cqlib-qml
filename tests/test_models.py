import numpy as np
import pytest

from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import FRQI
from cqlib_qml.models import HQNN, QNN
from cqlib_qml.optimizer import SGD


class TestHQNN:
    def test_init(self):
        ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3)
        assert model is not None

    def test_init_with_params(self):
        ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
        params = np.random.randn(len(ansatz.symbols))
        model = HQNN(ansatz=ansatz, out_dim=3, params=params)
        assert model is not None

    def test_forward(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3)
        encoder = FRQI(n_pixels=4, grayscale=2)
        data = np.array([[0, 1], [1, 0]], dtype=np.float32)
        data_circuits = encoder(data)
        output = model.forward(data_circuits)
        assert output.shape == (1, 3)

    def test_forward_batch(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3)
        encoder = FRQI(n_pixels=4, grayscale=2)
        imgs = np.array([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ], dtype=np.float32)
        data_circuits = encoder(imgs)
        output = model.forward(data_circuits)
        assert output.shape == (2, 3)

    def test_set_optimizer(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3)
        model.set_optimizer("adam(lr=0.001)")
        assert model._ansatz._optimizer is not None

    def test_init_with_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3, readouts=[0, 2])
        assert model._ansatz.readouts == [0, 2]
        assert model._linear.in_dim == 2

    def test_init_keeps_ansatz_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[1, 3])
        model = HQNN(ansatz=ansatz, out_dim=2)
        assert model._ansatz.readouts == [1, 3]
        assert model._linear.in_dim == 2

    def test_forward_with_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3, readouts=[0, 1])
        encoder = FRQI(n_pixels=4, grayscale=2)
        data = np.array([[0, 1], [1, 0]], dtype=np.float32)
        data_circuits = encoder(data)
        output = model.forward(data_circuits)
        assert output.shape == (1, 3)

    def test_trainable_mode(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = HQNN(ansatz=ansatz, out_dim=3)
        encoder = FRQI(n_pixels=4, grayscale=2)
        data = np.array([[0, 1], [1, 0]], dtype=np.float32)
        data_circuits = encoder(data)
        # Trainable mode
        output = model.forward(data_circuits, trainable=True)
        assert output.shape == (1, 3)
        # Non-trainable mode
        output = model.forward(data_circuits, trainable=False)
        assert output.shape == (1, 3)


class TestQNN:
    def test_init(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        model = QNN(ansatz=ansatz)
        assert model is not None

    def test_init_with_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        model = QNN(ansatz=ansatz, readouts=[0, 1])
        assert model is not None

    def test_init_readouts_override_ansatz_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        model = QNN(ansatz=ansatz, readouts=[1, 2, 3])
        assert model._ansatz.readouts == [1, 2, 3]

    def test_forward_readouts_override_ansatz_readouts(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        model = QNN(ansatz=ansatz, readouts=[0, 1, 2])
        encoder = FRQI(n_pixels=4, grayscale=2)
        data = np.array([[0, 1], [1, 0]], dtype=np.float32)
        data_circuits = encoder(data)
        output = model.forward(data_circuits)
        assert output.shape == (1, 3)

    def test_init_with_params(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        params = np.random.randn(len(ansatz.symbols))
        model = QNN(ansatz=ansatz, readouts=[0], params=params)
        assert model is not None

    def test_forward(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        model = QNN(ansatz=ansatz)
        encoder = FRQI(n_pixels=4, grayscale=2)
        data = np.array([[0, 1], [1, 0]], dtype=np.float32)
        data_circuits = encoder(data)
        output = model.forward(data_circuits)
        assert output.shape == (1, 1)

    def test_forward_batch(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        model = QNN(ansatz=ansatz)
        encoder = FRQI(n_pixels=4, grayscale=2)
        imgs = np.array([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ], dtype=np.float32)
        data_circuits = encoder(imgs)
        output = model.forward(data_circuits)
        assert output.shape == (2, 1)

    def test_no_readouts_raises(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        with pytest.raises(ValueError):
            QNN(ansatz=ansatz)