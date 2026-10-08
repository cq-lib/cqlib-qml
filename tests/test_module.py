import numpy as np
import pytest

from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.layer import Linear
from cqlib_qml.models import Module


class TestModule:
    def test_init_single_layer(self):
        layer = Linear(in_dim=10, out_dim=5)
        module = Module(layer)
        assert len(module._nets) == 1

    def test_init_multiple_layers(self):
        layer1 = Linear(in_dim=10, out_dim=5)
        layer2 = Linear(in_dim=5, out_dim=2)
        module = Module(layer1, layer2)
        assert len(module._nets) == 2

    def test_init_with_ansatz(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        module = Module(ansatz)
        assert len(module._nets) == 1

    def test_init_mixed(self):
        ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        layer = Linear(in_dim=4, out_dim=2)
        module = Module(ansatz, layer)
        assert len(module._nets) == 2

    def test_invalid_net_raises(self):
        with pytest.raises(TypeError, match="Expected Layer or Ansatz"):
            Module("invalid")

    def test_forward_layers(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        X = np.array([[1.0, 2.0, 3.0]])
        output = module.forward(X)
        assert output.shape == (1, 1)

    def test_forward_ansatz(self):
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        module = Module(ansatz)
        output = module.forward()
        assert output.shape == (1, 1)

    def test_backward(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        X = np.array([[1.0, 2.0, 3.0]])
        module.forward(X)
        dLdout = np.array([[0.5]])
        grad = module.backward(dLdout)
        assert grad.shape == (1, 3)

    def test_zero_grad(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        module.set_optimizer("sgd(lr=0.1)")
        X = np.array([[1.0, 2.0, 3.0]])
        module.forward(X)
        module.backward(np.array([[0.5]]))
        assert np.any(layer1._gradients["W"] != 0)
        module.zero_grad()
        assert np.all(layer1._gradients["W"] == 0)

    def test_update(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        module.set_optimizer("sgd(lr=0.1)")
        X = np.array([[1.0, 2.0, 3.0]])
        module.forward(X)
        module.backward(np.array([[0.5]]))
        old_W1 = layer1._parameters["W"].copy()
        module.update()
        new_W1 = layer1._parameters["W"]
        assert not np.array_equal(old_W1, new_W1)

    def test_freeze(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        assert layer1.trainable
        module.freeze()
        assert not layer1.trainable

    def test_unfreeze(self):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        module.freeze()
        assert not layer1.trainable
        module.unfreeze()
        assert layer1.trainable

    def test_save_load_checkpoint_layers(self, tmp_path):
        layer1 = Linear(in_dim=3, out_dim=2)
        layer2 = Linear(in_dim=2, out_dim=1)
        module = Module(layer1, layer2)
        module.set_optimizer("adam")
        module.forward(np.array([[1.0, 2.0, 3.0]]))  # trigger parameter init
        module.save_checkpoint(str(tmp_path), ep=2, it=50)

        new_module = Module(Linear(in_dim=3, out_dim=2), Linear(in_dim=2, out_dim=1))
        ep, it = new_module.load_checkpoint(str(tmp_path))
        assert (ep, it) == (2, 51)
        for net, new_net in zip(module._nets, new_module._nets):
            for key, val in net._parameters.items():
                assert np.array_equal(new_net._parameters[key], val)

    def test_save_load_checkpoint_mixed(self, tmp_path):
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY", "CX"])
        ansatz.set_measurement(readouts=[0])
        layer = Linear(in_dim=1, out_dim=1)
        module = Module(ansatz, layer)
        module.set_optimizer("adam")
        module.forward()  # random ansatz init + linear parameter init
        module.save_checkpoint(str(tmp_path), ep=2, it=50)

        new_ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY", "CX"])
        new_ansatz.set_measurement(readouts=[0])
        new_module = Module(new_ansatz, Linear(in_dim=1, out_dim=1))
        ep, it = new_module.load_checkpoint(str(tmp_path))
        assert (ep, it) == (2, 51)
        for key, val in layer._parameters.items():
            assert np.array_equal(new_module._nets[1]._parameters[key], val)

    def test_load_checkpoint_resume_position(self, tmp_path):
        """save_checkpoint is called after iteration `it` completes (see
        QNN_classification.train / HQNN_classification.train), so the restored
        position must be (ep, it + 1): the training loop passes it as
        `it_start` and skips iterations with `it < it_start`."""
        module = Module(Linear(in_dim=2, out_dim=1))
        module.set_optimizer("adam")
        module.forward(np.array([[1.0, 2.0]]))  # trigger parameter init
        module.save_checkpoint(str(tmp_path), ep=2, it=50)

        new_module = Module(Linear(in_dim=2, out_dim=1))
        ep, it_start = new_module.load_checkpoint(str(tmp_path))
        assert (ep, it_start) == (2, 51)
        resumed = [it for it in range(100) if it >= it_start]
        assert resumed[0] == 51
