import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from cqlib_qml.algorithms.QKM import QKM
from cqlib_qml.algorithms.QSVM import QSVM
from cqlib_qml.algorithms.VQC import VQC
from cqlib_qml.encoder import AngleEncoder, ZZFeatureEncoder, AmplitudeEncoder
from cqlib_qml.ansatz import HEAnsatz

# ============================================================================
# QKM Tests
# ============================================================================


class TestQKM:
    """QKM class unit tests."""

    def test_init_with_valid_encoder(self):
        """Test initialization with a valid encoder."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)
        assert qkm._encoder is encoder
        assert qkm._swap_test is False
        assert qkm._circuit_cache == {}

    def test_init_with_invalid_encoder(self):
        """Test initialization with an invalid encoder raises ValueError."""
        with pytest.raises(ValueError, match="QKM only supports"):
            QKM(encoder="invalid")  # type: ignore

    def test_init_with_swap_test_true(self):
        """Test initialization with swap_test enabled."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=True)
        assert qkm._swap_test is True

    def test_kernel_self_symmetric(self):
        """Test that the self-kernel matrix is symmetric."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3], [0.8, 0.1], [0.2, 0.9]])
        K = qkm.kernel(X)

        np.testing.assert_array_almost_equal(K, K.T)

    def test_kernel_diagonal_ones(self):
        """Test that the kernel matrix diagonal entries are 1."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3], [0.8, 0.1]])
        K = qkm.kernel(X)

        np.testing.assert_array_almost_equal(np.diag(K), np.array([1.0, 1.0]))

    def test_kernel_values_in_range(self):
        """Test that all kernel matrix values are in the range [0, 1]."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3], [0.8, 0.1], [0.2, 0.9]])
        K = qkm.kernel(X)

        tolerance = 1e-7
        assert np.all(K >= -tolerance)
        assert np.all(K <= 1 + tolerance)

    def test_kernel_shape(self):
        """Test that the kernel matrix has the correct shape."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3], [0.8, 0.1], [0.2, 0.9]])
        K = qkm.kernel(X)
        assert K.shape == (3, 3)

    def test_kernel_cross_shape(self):
        """Test that the cross-kernel matrix has the correct shape."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3], [0.8, 0.1]])
        Y = np.array([[0.2, 0.9]])
        K = qkm.kernel(X, Y)
        assert K.shape == (2, 1)

    def test_kernel_callable(self):
        """Test that QKM instance is callable."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3]])
        K1 = qkm.kernel(X)
        K2 = qkm(X)
        np.testing.assert_array_almost_equal(K1, K2)

    def test_clear_cache(self):
        """Test clearing the circuit cache."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3]])
        qkm.kernel(X)
        assert len(qkm._circuit_cache) == 1

        qkm.clear_cache()
        assert qkm._circuit_cache == {}

    def test_kernel_1d_input(self):
        """Test kernel computation with 1D input."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([0.5, 0.3])
        K = qkm.kernel(X)
        assert K.shape == (1, 1)

    def test_feature_dim_mismatch_raises(self):
        """Test that feature dimension mismatch raises ValueError."""
        encoder = AngleEncoder(mode="classical")
        qkm = QKM(encoder=encoder, swap_test=False)

        X = np.array([[0.5, 0.3]])
        Y = np.array([[0.2, 0.9, 0.1]])
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            qkm.kernel(X, Y)

    @pytest.mark.parametrize("encoder_class", [AngleEncoder, ZZFeatureEncoder, AmplitudeEncoder])
    def test_all_encoder_types(self, encoder_class):
        """Test that all supported encoder types work."""
        if encoder_class == AngleEncoder:
            encoder = encoder_class(mode="classical")
        elif encoder_class == ZZFeatureEncoder:
            encoder = encoder_class(n_repeats=1, entanglement="linear")
        else:
            encoder = encoder_class()

        qkm = QKM(encoder=encoder, swap_test=False)
        X = np.array([[0.5, 0.3]])
        K = qkm.kernel(X)
        assert K.shape == (1, 1)


# ============================================================================
# QSVM Tests
# ============================================================================


class TestQSVM:
    """QSVM class unit tests."""

    def test_init_with_valid_encoder(self):
        """Test initialization with a valid encoder."""
        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)
        assert qsvm.encoder is encoder
        assert qsvm.C == 1.0
        assert qsvm.swap_test is False
        assert qsvm.probability is False

    def test_init_with_invalid_encoder(self):
        """Test initialization with an invalid encoder raises ValueError."""
        with pytest.raises(ValueError, match="QSVM only supports"):
            QSVM(encoder="invalid")  # type: ignore

    def test_init_with_custom_C(self):
        """Test initialization with a custom C parameter."""
        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=0.5)
        assert qsvm.C == 0.5

    def test_init_with_probability(self):
        """Test initialization with probability estimation enabled."""
        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, probability=True)
        assert qsvm.probability is True

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X_train, y_train)
        y_pred = qsvm.predict(X_test)

        assert len(y_pred) == len(y_test)
        assert set(y_pred) <= set(qsvm.classes_)

    def test_score(self):
        """Test accuracy score computation."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X_train, y_train)
        score = qsvm.score(X_test, y_test)

        assert 0.0 <= score <= 1.0

    def test_predict_proba_without_probability_raises(self):
        """Test that predict_proba raises RuntimeError when probability is disabled."""
        X, y = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, probability=False)

        qsvm.fit(X, y)
        with pytest.raises(RuntimeError, match="Probability estimates not available"):
            qsvm.predict_proba(X)

    def test_predict_proba_with_probability(self):
        """Test probability prediction with probability estimation enabled."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, probability=True)

        qsvm.fit(X_train, y_train)
        proba = qsvm.predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        assert np.all(proba >= 0)

    def test_decision_function(self):
        """Test the decision function."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X_train, y_train)
        scores = qsvm.decision_function(X_test)

        assert len(scores) == len(X_test)

    def test_attributes_after_fit(self):
        """Test that attributes are set after fitting."""
        X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X, y)

        assert hasattr(qsvm, "classes_")
        assert hasattr(qsvm, "_svm")
        assert hasattr(qsvm, "_X_fit")
        np.testing.assert_array_equal(qsvm._X_fit, X)
        assert not np.shares_memory(qsvm._X_fit, X)

    def test_clear_cache(self):
        """Test clearing the circuit cache."""
        X, y = make_classification(n_samples=10, n_features=4, n_classes=2, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X, y)
        qsvm.clear_cache()
        assert qsvm._qkm is None or qsvm._qkm._circuit_cache == {}

    def test_reset(self):
        """Test resetting the model state."""
        X, y = make_classification(n_samples=10, n_features=4, n_classes=2, random_state=42)

        encoder = AngleEncoder(mode="classical")
        qsvm = QSVM(encoder=encoder, C=1.0)

        qsvm.fit(X, y)
        qsvm.reset()

        assert qsvm._qkm is None
        assert qsvm._svm is None
        assert qsvm._X_fit is None


# ============================================================================
# VQC Tests
# ============================================================================


class TestVQC:
    """VQC class unit tests."""

    def test_init_default(self):
        """Test default initialization."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0])
        assert vqc.ansatz is ansatz
        assert vqc.encoder is encoder
        assert vqc.readouts == [0]
        assert vqc.loss == "MSE"
        assert vqc.epochs == 100

    def test_init_bce_loss_single_readout(self):
        """Test BCE loss with a single readout."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="BCE")
        assert vqc.loss == "BCE"

    def test_init_bce_loss_default_readouts(self):
        """Test BCE loss with readouts omitted falls back to default [0]."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        vqc = VQC(ansatz=ansatz, encoder=encoder, loss="BCE")
        assert vqc.loss == "BCE"
        assert vqc.readouts is None  # Constructor arguments remain cloneable; effective readout is [0].

    def test_init_cross_entropy_default_readouts_raises(self):
        """Test that CrossEntropy with readouts omitted raises ValueError (not TypeError)."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        with pytest.raises(ValueError, match="CrossEntropy loss requires 2 readouts"):
            VQC(ansatz=ansatz, encoder=encoder, loss="CrossEntropy", n_classes=2)

    def test_init_bce_loss_multiple_readouts_raises(self):
        """Test that BCE loss with multiple readouts raises ValueError."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        with pytest.raises(ValueError, match="BCE loss requires exactly 1 readout"):
            VQC(ansatz=ansatz, encoder=encoder, readouts=[0, 1], loss="BCE")

    def test_init_cross_entropy_with_matching_readouts(self):
        """Test CrossEntropy loss with matching readouts."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0, 1], loss="CrossEntropy", n_classes=2)
        assert vqc.loss == "CrossEntropy"
        assert vqc.n_classes == 2

    def test_init_cross_entropy_readouts_mismatch_raises(self):
        """Test that CrossEntropy readouts mismatch raises ValueError."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        with pytest.raises(ValueError, match="CrossEntropy loss requires 2 readouts"):
            VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="CrossEntropy", n_classes=2)

    def test_init_invalid_loss_raises(self):
        """Test that an invalid loss type raises ValueError."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")

        with pytest.raises(ValueError, match="Unsupported loss"):
            VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="INVALID")  # type: ignore

    def test_get_loss_fn_mse(self):
        """Test creation of MSE loss function."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="MSE")

        loss_fn = vqc._get_loss_fn()
        from cqlib_qml.loss import MSELoss

        assert isinstance(loss_fn, MSELoss)

    def test_get_loss_fn_bce(self):
        """Test creation of BCE loss function."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="BCE")

        loss_fn = vqc._get_loss_fn()
        from cqlib_qml.loss import BCELoss

        assert isinstance(loss_fn, BCELoss)

    def test_get_loss_fn_cross_entropy(self):
        """Test creation of CrossEntropy loss function."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0, 1], loss="CrossEntropy", n_classes=2)

        loss_fn = vqc._get_loss_fn()
        from cqlib_qml.loss import SoftmaxCrossEntropy

        assert isinstance(loss_fn, SoftmaxCrossEntropy)

    def test_encode(self):
        """Test data encoding."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0])

        X = np.array([[0.5, 0.3], [0.8, 0.1]])
        circuits = vqc._encode(X)

        assert len(circuits) == 2
        from cqlib.circuit import Circuit

        assert isinstance(circuits[0], Circuit)

    def test_predict_before_fit_raises(self):
        """Test that predict raises an exception when called before fit."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0])

        X = np.array([[0.5, 0.3]])

        # check_is_fitted checks attribute existence, not value
        # _qnn is initialized to None, so check_is_fitted passes,
        # but calling _qnn.forward raises AttributeError
        with pytest.raises((NotFittedError, AttributeError)):
            vqc.predict(X)

    def test_predict_proba_before_fit_raises(self):
        """Test that predict_proba raises an exception when called before fit."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0])

        X = np.array([[0.5, 0.3]])

        with pytest.raises((NotFittedError, AttributeError)):
            vqc.predict_proba(X)

    def test_get_params(self):
        """Test getting parameters."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="BCE", epochs=50)

        params = vqc.get_params()
        assert params["ansatz"] is ansatz
        assert params["encoder"] is encoder
        assert params["readouts"] == [0]
        assert params["loss"] == "BCE"
        assert params["epochs"] == 50

    def test_set_params(self):
        """Test setting parameters."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0])

        vqc.set_params(epochs=200, batch_size=32)
        assert vqc.epochs == 200
        assert vqc.batch_size == 32

    def test_mse_loss_prepare(self):
        """Test data preparation for MSE loss."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="MSE")

        expectations = np.array([[0.5]])
        y = np.array([0])

        y_pred, y_true = vqc._prepare_for_loss(expectations, y)
        assert y_pred.shape == y_true.shape

    def test_bce_loss_prepare(self):
        """Test data preparation for BCE loss."""
        ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY"])
        encoder = AngleEncoder(mode="classical")
        vqc = VQC(ansatz=ansatz, encoder=encoder, readouts=[0], loss="BCE")

        expectations = np.array([[0.5]])
        y = np.array([1])

        y_pred, y_true = vqc._prepare_for_loss(expectations, y)
        assert y_pred.shape == y_true.shape

    def test_fit_updates_quantum_parameters(self):
        ansatz = HEAnsatz(2, 1, ["RY", "CX"])
        initial = dict(zip(ansatz.symbols, [0.3, 0.7]))
        ansatz.assign_parameters(initial)
        vqc = VQC(ansatz, AngleEncoder(mode="classical"), loss="BCE",
                  epochs=2, batch_size=2, optimizer="sgd(lr=0.01)", verbose=False)
        vqc.fit(np.array([[0.1, 0.3], [0.5, 0.8]]), np.array([0, 1]))
        assert abs(vqc.ansatz_._bindings["params0_0"] - initial["params0_0"]) > 1e-5


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlgorithmsIntegration:
    """Integration tests for multiple algorithms working together."""

    def test_qkm_qsvm_workflow(self):
        """Test QKM + QSVM workflow."""
        X, y = make_classification(n_samples=30, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        encoder = AngleEncoder(mode="classical")

        qkm = QKM(encoder=encoder, swap_test=False)
        K_train = qkm.kernel(X_train)

        assert K_train.shape == (len(X_train), len(X_train))

        qsvm = QSVM(encoder=encoder, C=1.0)
        qsvm.fit(X_train, y_train)

        y_pred = qsvm.predict(X_test)
        assert len(y_pred) == len(y_test)

    def test_vqc_train_predict_workflow(self):
        ansatz = HEAnsatz(2, 1, ["RY", "CX"])
        ansatz.assign_parameters(dict(zip(ansatz.symbols, [0.3, 0.7])))
        vqc = VQC(ansatz, AngleEncoder(mode="classical"), readouts=[0, 1],
                  loss="CrossEntropy", epochs=2, verbose=False)
        x, y = np.array([[0.1, 0.3], [0.5, 0.8]]), np.array([0, 1])
        vqc.fit(x, y)
        probabilities = vqc.predict_proba(x)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert set(vqc.predict(x)).issubset({0, 1})


# ============================================================================
# HQNN_classification / QNN_classification Tests
# ============================================================================


class TestHQNNClassification:
    """HQNN_classification module tests."""

    def test_train_function_exists(self):
        """Test that the train function exists and is callable."""
        from cqlib_qml.algorithms import HQNN_classification

        assert hasattr(HQNN_classification, "train")
        assert callable(HQNN_classification.train)

    def test_validate_function_exists(self):
        """Test that the validate function exists and is callable."""
        from cqlib_qml.algorithms import HQNN_classification

        assert hasattr(HQNN_classification, "validate")
        assert callable(HQNN_classification.validate)


class TestQNNClassification:
    """QNN_classification module tests."""

    def test_train_function_exists(self):
        """Test that the train function exists and is callable."""
        from cqlib_qml.algorithms import QNN_classification

        assert hasattr(QNN_classification, "train")
        assert callable(QNN_classification.train)

    def test_validate_function_exists(self):
        """Test that the validate function exists and is callable."""
        from cqlib_qml.algorithms import QNN_classification

        assert hasattr(QNN_classification, "validate")
        assert callable(QNN_classification.validate)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
