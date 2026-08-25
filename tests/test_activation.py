import numpy as np
import pytest

from cqlib_qml.layer.activation import Sigmoid, ReLU, Tanh, SoftPlus, ActivationInitializer


class TestSigmoid:
    def test_act(self):
        act = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])
        y = act.act(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(y, expected)

    def test_grad(self):
        act = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])
        grad = act.grad(x)
        if grad.ndim == 2 and grad.shape[0] == 1:
            grad = grad.flatten()
        y = act.act(x)
        expected = y * (1 - y)
        np.testing.assert_array_almost_equal(grad, expected)

    def test_grad2(self):
        act = Sigmoid()
        x = np.array([0.0, 1.0, -1.0])
        grad2 = act.grad2(x)
        if grad2.ndim == 2 and grad2.shape[0] == 1:
            grad2 = grad2.flatten()
        y = act.act(x)
        expected = y * (1 - y) * (1 - 2 * y)
        np.testing.assert_array_almost_equal(grad2, expected)

    def test_str(self):
        act = Sigmoid()
        assert str(act) == "sigmoid"

    def test_call(self):
        act = Sigmoid()
        x = np.array([[0.0, 1.0], [-1.0, 0.5]])
        y = act(x)
        assert y.shape == (2, 2)


class TestReLU:
    def test_act(self):
        act = ReLU()
        x = np.array([-1.0, 0.0, 2.0])
        y = act.act(x)
        expected = np.array([0.0, 0.0, 2.0])
        np.testing.assert_array_equal(y, expected)

    def test_grad(self):
        act = ReLU()
        x = np.array([-1.0, 0.0, 2.0])
        grad = act.grad(x)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(grad, expected)

    def test_grad2(self):
        act = ReLU()
        x = np.array([-1.0, 0.0, 2.0])
        grad2 = act.grad2(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(grad2, expected)

    def test_str(self):
        act = ReLU()
        assert str(act) == "relu"


class TestTanh:
    def test_act(self):
        act = Tanh()
        x = np.array([0.0, 1.0, -1.0])
        y = act.act(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(y, expected)

    def test_grad(self):
        act = Tanh()
        x = np.array([0.0, 1.0, -1.0])
        grad = act.grad(x)
        expected = 1 - np.tanh(x) ** 2
        np.testing.assert_array_almost_equal(grad, expected)

    def test_str(self):
        act = Tanh()
        assert str(act) == "tanh"


class TestSoftPlus:
    def test_act(self):
        act = SoftPlus()
        x = np.array([0.0, 1.0, -1.0])
        y = act.act(x)
        expected = np.log(np.exp(x) + 1)
        np.testing.assert_array_almost_equal(y, expected)

    def test_grad(self):
        act = SoftPlus()
        x = np.array([0.0, 1.0, -1.0])
        grad = act.grad(x)
        exp_x = np.exp(x)
        expected = exp_x / (exp_x + 1)
        np.testing.assert_array_almost_equal(grad, expected)

    def test_str(self):
        act = SoftPlus()
        assert str(act) == "softplus"


class TestActivationInitializer:
    def test_sigmoid(self):
        init = ActivationInitializer("sigmoid")
        act = init()
        assert isinstance(act, Sigmoid)

    def test_relu(self):
        init = ActivationInitializer("relu")
        act = init()
        assert isinstance(act, ReLU)

    def test_tanh(self):
        init = ActivationInitializer("tanh")
        act = init()
        assert isinstance(act, Tanh)

    def test_softplus(self):
        init = ActivationInitializer("softplus")
        act = init()
        assert isinstance(act, SoftPlus)

    def test_none(self):
        init = ActivationInitializer(None)
        act = init()
        assert act is None

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unsupported activation function"):
            ActivationInitializer("invalid")
