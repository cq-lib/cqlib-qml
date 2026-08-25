import numpy as np
import pytest

from cqlib_qml.optimizer import SGD, AdaGrad, RMSProp, Adam, OptimizerInitializer


class TestSGD:
    def test_init(self):
        opt = SGD(lr=0.01, momentum=0.9)
        assert opt.hyperparameters["lr"] == 0.01
        assert opt.hyperparameters["momentum"] == 0.9

    def test_update(self):
        opt = SGD(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        new_param = opt.update(param, grad, "weight")
        expected = param - 0.1 * grad
        np.testing.assert_array_almost_equal(new_param, expected)

    def test_with_momentum(self):
        opt = SGD(lr=0.1, momentum=0.5)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        # First update
        new_param = opt.update(param, grad, "weight")
        # Cache should be updated
        assert "weight" in opt.cache
        # Second update should use momentum
        grad2 = np.array([0.05, 0.1])
        new_param2 = opt.update(new_param, grad2, "weight")
        assert new_param2 is not None

    def test_clip_norm(self):
        opt = SGD(lr=0.1, clip_norm=0.5)
        param = np.array([1.0, 2.0])
        grad = np.array([10.0, 10.0])
        new_param = opt.update(param, grad, "weight")
        assert new_param is not None


class TestAdaGrad:
    def test_init(self):
        opt = AdaGrad(lr=0.01, eps=1e-8)
        assert opt.hyperparameters["lr"] == 0.01
        assert opt.hyperparameters["eps"] == 1e-8

    def test_update(self):
        opt = AdaGrad(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        new_param = opt.update(param, grad, "weight")
        assert "weight" in opt.cache
        assert new_param is not None

    def test_cache_accumulates(self):
        opt = AdaGrad(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        opt.update(param, grad, "weight")
        cache1 = opt.cache["weight"].copy()
        opt.update(param, grad, "weight")
        cache2 = opt.cache["weight"]
        assert np.all(cache2 > cache1)


class TestRMSProp:
    def test_init(self):
        opt = RMSProp(lr=0.001, decay=0.9)
        assert opt.hyperparameters["lr"] == 0.001
        assert opt.hyperparameters["decay"] == 0.9

    def test_update(self):
        opt = RMSProp(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        new_param = opt.update(param, grad, "weight")
        assert "weight" in opt.cache
        assert new_param is not None


class TestAdam:
    def test_init(self):
        opt = Adam(lr=0.001, decay1=0.9, decay2=0.999)
        assert opt.hyperparameters["lr"] == 0.001
        assert opt.hyperparameters["decay1"] == 0.9
        assert opt.hyperparameters["decay2"] == 0.999

    def test_update(self):
        opt = Adam(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        new_param = opt.update(param, grad, "weight")
        assert "weight" in opt.cache
        assert opt.cache["weight"]["t"] == 1
        assert new_param is not None

    def test_multiple_updates(self):
        opt = Adam(lr=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        opt.update(param, grad, "weight")
        opt.update(param, grad, "weight")
        assert opt.cache["weight"]["t"] == 2


class TestOptimizerInitializer:
    def test_from_none(self):
        init = OptimizerInitializer(None)
        opt = init()
        assert isinstance(opt, SGD)

    def test_from_str(self):
        init = OptimizerInitializer("adam(lr=0.01)")
        opt = init()
        assert isinstance(opt, Adam)
        assert opt.hyperparameters["lr"] == 0.01

    def test_from_dict(self):
        init = OptimizerInitializer({
            "hyperparameters": {"id": "SGD", "lr": 0.01, "momentum": 0.9, "clip_norm": None},
            "cache": {}
        })
        opt = init()
        assert isinstance(opt, SGD)
        assert opt.hyperparameters["lr"] == 0.01

    def test_from_optimizer(self):
        sgd = SGD(lr=0.01)
        init = OptimizerInitializer(sgd)
        opt = init()
        assert isinstance(opt, SGD)
        assert opt.hyperparameters["lr"] == 0.01