import numpy as np
import pytest

from cqlib_qml.layer import Linear


class TestLinear:
    def test_init(self):
        layer = Linear(in_dim=10, out_dim=5)
        assert layer.in_dim == 10
        assert layer.out_dim == 5
        assert not layer._init

    def test_init_params(self):
        layer = Linear(in_dim=10, out_dim=5)
        layer.init_params()
        assert layer._init
        assert "W" in layer._parameters
        assert layer._parameters["W"].shape == (5, 10)

    def test_init_with_bias(self):
        layer = Linear(in_dim=10, out_dim=5, bias=True)
        layer.init_params()
        assert "b" in layer._parameters
        assert layer._parameters["b"].shape == (1, 5)

    def test_init_without_bias(self):
        layer = Linear(in_dim=10, out_dim=5, bias=False)
        layer.init_params()
        assert "b" not in layer._parameters

    def test_forward(self):
        layer = Linear(in_dim=3, out_dim=2)
        X = np.array([[1.0, 2.0, 3.0]])
        output = layer.forward(X)
        assert output.shape == (1, 2)

    def test_forward_with_activation(self):
        layer = Linear(in_dim=3, out_dim=2, act_fn="sigmoid")
        X = np.array([[1.0, 2.0, 3.0]])
        output = layer.forward(X)
        assert output.shape == (1, 2)
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_backward(self):
        layer = Linear(in_dim=3, out_dim=2)
        X = np.array([[1.0, 2.0, 3.0]])
        layer.forward(X)
        dLdy = np.array([[0.5, 0.5]])
        dX = layer.backward(dLdy)
        assert dX.shape == (3,)

    def test_backward_batch(self):
        layer = Linear(in_dim=3, out_dim=2)
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        layer.forward(X)
        dLdy = np.array([[0.5, 0.5], [0.1, 0.1]])
        dX = layer.backward(dLdy)
        assert dX.shape == (2, 3)

    def test_update(self):
        layer = Linear(in_dim=3, out_dim=2)
        layer.set_optimizer("sgd")
        X = np.array([[1.0, 2.0, 3.0]])
        layer.forward(X)
        dLdy = np.array([[0.5, 0.5]])
        layer.backward(dLdy)
        old_W = layer._parameters["W"].copy()
        layer.update()
        new_W = layer._parameters["W"]
        assert not np.array_equal(old_W, new_W)

    def test_zero_grad(self):
        layer = Linear(in_dim=3, out_dim=2)
        layer.set_optimizer("sgd")
        X = np.array([[1.0, 2.0, 3.0]])
        layer.forward(X)
        dLdy = np.array([[0.5, 0.5]])
        layer.backward(dLdy)
        assert np.any(layer._gradients["W"] != 0)
        layer.zero_grad()
        assert np.all(layer._gradients["W"] == 0)

    def test_freeze(self):
        layer = Linear(in_dim=3, out_dim=2)
        assert layer.trainable
        layer.freeze()
        assert not layer.trainable

    def test_unfreeze(self):
        layer = Linear(in_dim=3, out_dim=2)
        layer.freeze()
        assert not layer.trainable
        layer.unfreeze()
        assert layer.trainable

    def test_str(self):
        layer = Linear(in_dim=10, out_dim=5)
        assert "Linear" in str(layer)
        assert "in_dim=10" in str(layer)
        assert "out_dim=5" in str(layer)
