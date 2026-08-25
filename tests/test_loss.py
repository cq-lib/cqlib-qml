import numpy as np
import pytest

from cqlib_qml.loss import (
    HingeLoss,
    MSELoss,
    BCELoss,
    CrossEntropy,
    SoftmaxCrossEntropy,
)


class TestHingeLoss:
    def test_forward(self):
        loss_fn = HingeLoss()
        pred = np.array([[0.8, -0.2]])
        target = np.array([[1, -1]])
        loss = loss_fn(pred, target)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gradients(self):
        loss_fn = HingeLoss()
        pred = np.array([[0.8, -0.2]])
        target = np.array([[1, -1]])
        loss_fn(pred, target)
        grads = loss_fn.grads()
        assert grads.shape == pred.shape


class TestMSELoss:
    def test_forward(self):
        loss_fn = MSELoss()
        pred = np.array([[0.5, 0.3]])
        target = np.array([[0.5, 0.3]])
        loss = loss_fn(pred, target)
        assert loss == 0.0

    def test_gradients(self):
        loss_fn = MSELoss()
        pred = np.array([[0.5, 0.3]])
        target = np.array([[0.0, 0.0]])
        loss_fn(pred, target)
        grads = loss_fn.grads()
        assert grads.shape == pred.shape

    def test_str(self):
        loss_fn = MSELoss()
        assert str(loss_fn) == "MSELoss"


class TestBCELoss:
    def test_forward(self):
        loss_fn = BCELoss()
        pred = np.array([[0.9, 0.1]])
        target = np.array([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gradients(self):
        loss_fn = BCELoss()
        pred = np.array([[0.9, 0.1]])
        target = np.array([[1.0, 0.0]])
        loss_fn(pred, target)
        grads = loss_fn.grads()
        assert grads.shape == pred.shape


class TestCrossEntropy:
    def test_forward(self):
        loss_fn = CrossEntropy()
        pred = np.array([[0.9, 0.1]])
        target = np.array([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gradients(self):
        loss_fn = CrossEntropy()
        pred = np.array([[0.9, 0.1]])
        target = np.array([[1.0, 0.0]])
        loss_fn(pred, target)
        grads = loss_fn.grads()
        assert grads.shape == pred.shape


class TestSoftmaxCrossEntropy:
    def test_forward(self):
        loss_fn = SoftmaxCrossEntropy()
        pred = np.array([[1.0, 0.5]])
        target = np.array([[1.0, 0.0]])
        loss = loss_fn(pred, target)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_gradients(self):
        loss_fn = SoftmaxCrossEntropy()
        pred = np.array([[1.0, 0.5]])
        target = np.array([[1.0, 0.0]])
        loss_fn(pred, target)
        grads = loss_fn.grads()
        assert grads.shape == pred.shape

    def test_with_dpred(self):
        loss_fn = SoftmaxCrossEntropy()
        pred = np.array([[1.0, 0.5]])
        target = np.array([[1.0, 0.0]])
        loss_fn(pred, target)
        grads = loss_fn.grads(dpred=2.0)
        assert grads.shape == pred.shape