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

    @pytest.mark.parametrize('target', [-0.1, 2.0, np.nan, np.inf, 1j])
    def test_invalid_target_is_rejected_without_changing_stored_batch(self, target):
        loss_fn = BCELoss()
        pred = np.array([[0.9]])
        valid_target = np.array([[1.0]])
        loss_fn(pred, valid_target)
        with pytest.raises(ValueError, match='BCE targets'):
            loss_fn(pred, np.array([[target]]))
        assert loss_fn._pred is pred
        assert loss_fn._target is valid_target

    def test_soft_targets_in_unit_interval_remain_supported(self):
        loss_fn = BCELoss()
        loss = loss_fn(np.array([[0.9]]), np.array([[0.25]]))
        assert np.isfinite(loss)
        assert np.isfinite(loss_fn.grads()).all()

    @pytest.mark.parametrize('target', [0.0, 1.0])
    @pytest.mark.parametrize('pred', [-1e-15, 1 + 1e-15])
    def test_roundoff_outside_probability_range_is_finite(self, pred, target):
        loss_fn = BCELoss()
        loss = loss_fn(np.array([[pred]]), np.array([[target]]))
        assert np.isfinite(loss)
        assert np.all(np.isfinite(loss_fn.grads()))

    @pytest.mark.parametrize('pred', [-0.01, 1.01, np.nan, np.inf])
    def test_invalid_probability_is_rejected(self, pred):
        loss_fn = BCELoss()
        with pytest.raises(ValueError, match='BCE predictions'):
            loss_fn(np.array([[pred]]), np.array([[0.0]]))

    @pytest.mark.parametrize('pred,target,expected_sign', [
        (0.0, 1.0, -1.0),
        (1.0, 0.0, 1.0),
    ])
    def test_exact_endpoint_keeps_nonzero_gradient(self, pred, target, expected_sign):
        loss_fn = BCELoss()
        loss_fn(np.array([[pred]]), np.array([[target]]))
        assert loss_fn.grads()[0, 0] == pytest.approx(
            expected_sign / np.finfo(float).eps
        )

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @pytest.mark.parametrize('ulps', [1, 64])
    @pytest.mark.parametrize('side,target,expected_sign', [
        ('lower', 1.0, -1.0),
        ('upper', 0.0, 1.0),
    ])
    def test_wrong_label_roundoff_keeps_endpoint_gradient(self, dtype, ulps, side, target, expected_sign):
        eps = np.finfo(dtype).eps
        value = -ulps * eps if side == 'lower' else 1 + ulps * eps
        prediction = np.array([[value]], dtype=dtype)
        loss_fn = BCELoss()
        loss = loss_fn(prediction, np.array([[target]], dtype=dtype))
        assert np.isfinite(loss)
        assert loss_fn.grads()[0, 0] == pytest.approx(expected_sign / eps, rel=1e-5)

    @pytest.mark.parametrize('pred,target', [(-1e-15, 0.0), (1 + 1e-15, 1.0)])
    def test_correct_label_roundoff_has_zero_gradient(self, pred, target):
        loss_fn = BCELoss()
        loss_fn(np.array([[pred]]), np.array([[target]]))
        assert loss_fn.grads()[0, 0] == pytest.approx(0.0)


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

    def test_large_logits_stable(self):
        # exp(1000) overflows to inf -> inf/inf = nan without max-subtraction
        loss_fn = SoftmaxCrossEntropy()
        logits = np.array([[1000.0, -1000.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0]])
        loss = loss_fn(logits, target)
        grads = loss_fn.grads()
        assert np.isfinite(loss)
        assert np.all(np.isfinite(grads))
        # Log-sum-exp retains the correct loss even when softmax underflows.
        assert loss == pytest.approx(2000.0)
        assert np.allclose(grads, np.array([[1.0, -1.0, 0.0]]))

    def test_large_logits_correct_label_stable(self):
        loss_fn = SoftmaxCrossEntropy()
        logits = np.array([[1000.0, -1000.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0]])
        loss = loss_fn(logits, target)
        grads = loss_fn.grads()
        assert np.isfinite(loss)
        assert loss == pytest.approx(0.0)
        assert np.allclose(grads, np.zeros((1, 3)), atol=1e-9)

    def test_all_negative_large_logits_stable(self):
        # exp(-1000) underflows to 0 -> 0/0 = nan without max-subtraction
        loss_fn = SoftmaxCrossEntropy()
        logits = np.array([[-1000.0, -1000.0]])
        target = np.array([[1.0, 0.0]])
        loss = loss_fn(logits, target)
        grads = loss_fn.grads()
        assert np.isfinite(loss)
        assert loss == pytest.approx(np.log(2.0))
        assert np.allclose(grads, np.array([[-0.5, 0.5]]))


class TestLossGradsNearTarget:
    """LossFun.grads must return the exact autograd gradient even where
    |pred - target| < 1e-12: the analytic gradient at (near-)equal
    predictions is generally nonzero."""

    def test_cross_entropy_exact_match(self):
        # d/dpred -sum(target*log(pred)) = -target/pred = [-1, -1], not 0
        loss_fn = CrossEntropy()
        pred = np.array([[0.9, 0.1]])
        target = pred.copy()
        loss_fn(pred, target)
        assert np.allclose(loss_fn.grads(), -target / pred, atol=1e-6)

    def test_hinge_near_margin(self):
        # pred within 1e-13 of target is still inside the margin:
        # d/dpred clip(1 - pred*target, 0) = -target = -1
        loss_fn = HingeLoss()
        pred = np.array([[1.0 - 1e-13]])
        target = np.array([[1.0]])
        loss_fn(pred, target)
        assert np.allclose(loss_fn.grads(), np.array([[-1.0]]))

    def test_bce_near_boundary(self):
        # d/dpred -log(pred) at pred = 1 - 1e-13 is ~-1, not 0
        loss_fn = BCELoss()
        pred = np.array([[1.0 - 1e-13]])
        target = np.array([[1.0]])
        loss_fn(pred, target)
        assert np.allclose(loss_fn.grads(), np.array([[-1.0]]), atol=1e-6)

    def test_bce_exact_match_is_zero(self):
        # control: at pred == target == 1 the loss clips to [0, 100]
        # (raw value ~ -eps < 0), so the exact gradient is genuinely 0
        loss_fn = BCELoss()
        pred = np.array([[1.0]])
        target = np.array([[1.0]])
        loss_fn(pred, target)
        assert np.all(loss_fn.grads() == 0.0)

    def test_mse_matches_analytic(self):
        # control: 2*(pred - target)/N already vanishes at pred == target
        loss_fn = MSELoss()
        pred = np.array([[0.5, 0.3]])
        target = np.array([[0.5, 0.0]])
        loss_fn(pred, target)
        assert np.allclose(loss_fn.grads(), 2 * (pred - target) / pred.size)
