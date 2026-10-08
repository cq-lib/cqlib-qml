# cqlib_qml/algorithms/VQC.py
"""
Variational Quantum Classifier (VQC).

This module provides a scikit-learn compatible implementation of the
Variational Quantum Classifier. VQC uses a parameterized quantum circuit
(an ansatz) to classify data, with trainable parameters optimized via
gradient-based methods.

The VQC class wraps the QNN model with a scikit-learn API, making it
easy to integrate with standard machine learning pipelines.

References:
    - Farhi, E., & Neven, H. (2018). "Classification with quantum neural
      networks on near term processors." arXiv:1802.06002.

Examples:
    >>> from cqlib_qml.algorithms.VQC import VQC
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.encoder import AngleEncoder
    >>>
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> encoder = AngleEncoder(mode="dense")
    >>>
    >>> vqc = VQC(
    ...     ansatz=ansatz,
    ...     encoder=encoder,
    ...     readouts=[0, 1],
    ...     loss="CrossEntropy",
    ...     epochs=100,
    ...     batch_size=32
    ... )
    >>>
    >>> vqc.fit(X_train, y_train)
    >>> accuracy = vqc.score(X_test, y_test)
"""

from cqlib_qml._configuration import same_parameter_value

import numpy as np
from numbers import Integral
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

from cqlib.circuit import Circuit
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder, ZZFeatureEncoder
from cqlib_qml.models import QNN
from cqlib_qml.loss import BCELoss, MSELoss, SoftmaxCrossEntropy
from cqlib_qml.optimizer import OptimizerBase, OptimizerInitializer
from cqlib_qml._state import clone_state, make_rng, rng_from_state


class VQC(ClassifierMixin, BaseEstimator):
    """
    Variational Quantum Classifier (VQC) with sklearn compatibility.

    The VQC is a quantum machine learning model that uses a parameterized
    quantum circuit for classification. It supports multiple loss functions
    and provides a scikit-learn compatible interface for easy integration.

    Args:
        ansatz (Ansatz): The parameterized quantum circuit (ansatz).
        encoder: Encoding strategy for data-to-quantum state mapping.
            Must be one of AmplitudeEncoder, AngleEncoder, or ZZFeatureEncoder.
        readouts (list, optional): The readout qubits for measurement.
            Defaults to [0].
        loss (str): Loss function type. Options: 'MSE', 'BCE', or 'CrossEntropy'.
            Defaults to 'MSE'.
        optimizer (Union[str, dict, OptimizerBase]): Optimizer for training.
            Defaults to "adam".
        n_classes (int): Number of output classes. Defaults to 2.
        epochs (int): Positive number of training epochs. Defaults to 100.
        batch_size (int, optional): Positive batch size. If None, uses full batch.
            Defaults to None.
        verbose (bool): Whether to print training progress. Defaults to True.

    Attributes:
        classes_: Unique class labels from training data.
        n_features_in_: Number of features in the training data.

    Raises:
        ValueError: If loss type doesn't match readouts configuration:
            - 'BCE' requires exactly 1 readout
            - 'CrossEntropy' requires n_classes readouts

    Examples:
        >>> # Binary classification with BCE loss
        >>> vqc = VQC(
        ...     ansatz=HEAnsatz(n_qubits=3, d=2, layers=["RY", "CX"]),
        ...     encoder=AngleEncoder(),
        ...     readouts=[0],
        ...     loss="BCE",
        ...     epochs=50
        ... )
        >>> vqc.fit(X_train, y_train)
        >>>
        >>> # Multi-class classification with CrossEntropy
        >>> vqc = VQC(
        ...     ansatz=HEAnsatz(n_qubits=4, d=3, layers=["RY", "CX"]),
        ...     encoder=ZZFeatureEncoder(n_repeats=2),
        ...     readouts=[0, 1, 2],
        ...     loss="CrossEntropy",
        ...     n_classes=3,
        ...     epochs=100
        ... )
        >>> vqc.fit(X_train, y_train)
        >>> y_pred = vqc.predict(X_test)
    """

    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels from training data."""
        check_is_fitted(self)
        return self._classes

    @property
    def n_features_in_(self) -> int:
        """Number of features in the training data."""
        check_is_fitted(self)
        return self._X_fit.shape[1]

    def __init__(
        self,
        ansatz: Ansatz,
        encoder: Union[AmplitudeEncoder, AngleEncoder, ZZFeatureEncoder],
        readouts: Optional[List[int]] = None,
        loss: str = "MSE",
        optimizer: Union[str, dict, OptimizerBase] = "adam",
        n_classes: int = 2,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        warm_start: bool = False,
        initial_point=None,
        random_state=None,
    ):
        """Initialize the VQC model.

        Args:
            ansatz: Parameterized quantum circuit.
            encoder: Data encoding strategy.
            readouts: List of qubit indices to measure. Defaults to [0].
            loss: Loss function type. Defaults to "MSE".
            optimizer: Optimizer for training. Defaults to "adam".
            n_classes: Number of output classes. Defaults to 2.
            epochs: Number of training epochs. Defaults to 100.
            batch_size: Batch size for mini-batch training. Defaults to None.
            verbose: Whether to print progress during training. Defaults to True.
            warm_start: Reuse learned weights on fit, resetting optimizer and scheduler.
            initial_point: Complete initial weight vector or mapping. Takes precedence
                over declared template bindings on a fresh fit.
            random_state: Integer, Generator (copied) or None. Owns initialization
                and shuffling; does not change the caller's random state.

        Raises:
            ValueError: If loss type is incompatible with readouts.
        """
        self._validate_loss_config(loss, readouts, n_classes)

        self.ansatz = ansatz
        self.encoder = encoder
        self.readouts = readouts
        self.loss = loss
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.warm_start = warm_start
        self.initial_point = initial_point
        self.random_state = random_state
        self._rng = make_rng(random_state)
        self._initial_rng_state = deepcopy(self._rng.bit_generator.state)
        self._progress = None
        self._resume_complete = False

        self._qnn = None
        self._loss_fn = None
        self._classes = None
        self._X_fit = None

    def __sklearn_clone__(self):
        """Clone construction templates only, including native observables."""
        return type(self)(**clone_state(self.get_params(deep=False)))

    def __sklearn_is_fitted__(self):
        return self._qnn is not None and self._X_fit is not None

    @staticmethod
    def _validate_loss_config(loss, readouts, n_classes):
        """Validate the complete proposed configuration before publishing it."""
        if loss not in ('MSE', 'BCE', 'CrossEntropy'):
            raise ValueError(f"Unsupported loss: {loss}. Supported losses: 'MSE', 'BCE', 'CrossEntropy'.")
        effective_readouts = readouts if readouts is not None else [0]
        if loss == 'BCE' and len(effective_readouts) != 1:
            raise ValueError(f"BCE loss requires exactly 1 readout, got {len(effective_readouts)}.")
        if loss == 'CrossEntropy' and len(effective_readouts) != n_classes:
            raise ValueError(f"CrossEntropy loss requires {n_classes} readouts, got {len(effective_readouts)}.")

    @staticmethod
    def _validate_training_config(epochs, batch_size):
        """Reject invalid loop bounds before changing any fitted state."""
        for name, value in [('epochs', epochs), ('batch_size', batch_size)]:
            if name == 'batch_size' and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer"
                                 + (" or None" if name == 'batch_size' else ""))

    def _get_loss_fn(self):
        """Get the appropriate loss function instance based on loss type."""
        if self.loss == "MSE":
            return MSELoss()
        elif self.loss == "BCE":
            return BCELoss()
        elif self.loss == 'CrossEntropy':
            return SoftmaxCrossEntropy()
        raise ValueError(f"Unsupported loss: {self.loss}")

    def _create_qnn(self) -> QNN:
        """
        Create and configure the QNN model.

        Returns:
            QNN: Configured Quantum Neural Network model.
        """
        self.ansatz_.set_measurement(readouts=self.readouts if self.readouts is not None else [0])
        return QNN(
            ansatz=self.ansatz_,
            readouts=self.readouts,
            optimizer=self._fresh_optimizer(),
        )

    def _encode(self, X: np.ndarray) -> List[Circuit]:
        """
        Encode data into quantum circuits.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            List[Circuit]: List of quantum circuits, one per sample.

        Note:
            The encoder may return a single Circuit or a list. This method
            handles both cases.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        circuits = []
        for x in X:
            result = getattr(self, "encoder_", self.encoder)(x)
            circuits.append(result[0] if isinstance(result, list) else result)
        return circuits

    def _prepare_for_loss(self, expectations: np.ndarray, y: np.ndarray):
        """
        Prepare predictions and labels according to the loss function.

        This method transforms the raw quantum circuit expectations and
        ground truth labels into the format expected by the loss function.

        Args:
            expectations (np.ndarray): Output from QNN forward pass.
            y (np.ndarray): Ground truth labels.

        Returns:
            tuple: (y_pred, y_true) formatted for the loss function.
        """
        y = np.asarray(y)
        if self.loss == "MSE":
            if (len(self._classes) if self._classes is not None else self.n_classes) > 2:
                # Multi-class: one-hot encoding
                n_samples = len(y)
                y_true = np.zeros(expectations.shape)
                y_true[np.arange(n_samples), y.astype(int)] = 1.0
                y_pred = expectations
            else:
                # Binary: convert to {-1, 1}
                y_true = (2 * y - 1.0).reshape(expectations.shape)
                y_pred = -expectations
        elif self.loss == "BCE":
            # Binary: convert to {0, 1}
            y_true = y.reshape(expectations.shape)
            y_pred = (1 - expectations) / 2
        else:  # CrossEntropy
            # Multi-class: one-hot encoding
            n_samples = len(y)
            y_true = np.zeros(expectations.shape)
            y_true[np.arange(n_samples), y.astype(int)] = 1.0
            y_pred = expectations

        return y_pred, y_true

    def _compute_accuracy(self, expectations: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy based on the loss function type.

        Args:
            expectations (np.ndarray): Output from QNN forward pass.
            y (np.ndarray): Ground truth labels.

        Returns:
            float: Accuracy in [0, 1].
        """
        y = np.asarray(y)
        if self.loss == "MSE":
            if (len(self._classes) if self._classes is not None else self.n_classes) > 2:
                pred_labels = np.argmax(expectations, axis=1)
                correct = np.sum(pred_labels == y)
            else:
                y_true = (2 * y - 1.0).reshape(expectations.shape)
                y_pred = -expectations
                correct = np.where(y_true * y_pred > 0)[0].shape[0]
        elif self.loss == "BCE":
            probabilities = (1 - expectations) / 2
            y_pred_labels = (probabilities.flatten() > 0.5).astype(int)
            correct = np.sum(y_pred_labels == y)
        else:  # CrossEntropy
            correct = np.sum(expectations.argmax(axis=1) == y)

        return correct / len(y)

    def _fresh_optimizer(self):
        """Warm start deliberately retains weights only, never optimizer history."""
        optimizer = OptimizerInitializer(clone_state(self.optimizer))()
        optimizer.reset_state()
        return optimizer



    def fit(self, X, y):
        """Fit from declared initialization, or retain weights with warm_start.

        The construction objects are templates. Learned components are exposed
        as ansatz_ and encoder_. Warm starts reset optimizer/scheduler history.
        All operations are staged: failures leave the fitted estimator intact.
        """
        if type(self.warm_start) is not bool:
            raise ValueError("warm_start must be a boolean")
        self._validate_training_config(self.epochs, self.batch_size)
        self._validate_loss_config(self.loss, self.readouts, self.n_classes)
        candidate = clone_state(self)
        candidate._fit_in_place(X, y)
        candidate.__dict__.update(self.get_params(deep=False))
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
        return self

    def _fit_in_place(self, X, y):
        X, raw_y = check_X_y(X, y)
        classes, y = np.unique(raw_y, return_inverse=True)
        count = len(classes)
        readouts = self.readouts if self.readouts is not None else [0]
        binary = self.loss == "BCE" or (self.loss == "MSE" and count == 2)
        expected = 1 if binary else count
        if count < 2 or (self.loss == "BCE" and count != 2) or len(readouts) != expected:
            raise ValueError(f"Loss {self.loss} with {count} classes requires {expected} compatible readouts")
        warm = self.warm_start and self.__sklearn_is_fitted__()
        if warm:
            if not np.array_equal(self._classes, classes) or X.shape[1] != self.n_features_in_:
                raise ValueError("Warm start requires the same classes and feature dimension")
        else:
            self._rng = rng_from_state(self._initial_rng_state)
            self.ansatz_ = clone_state(self.ansatz)
            self.encoder_ = clone_state(self.encoder)
            self.ansatz_._encoder = None
            self.ansatz_.zero_grad()
            self.ansatz_._invalidate_gradients()
            if self.initial_point is not None:
                self.ansatz_.assign_weights(self.initial_point)
            elif set(self.ansatz_._weights) != set(self.ansatz_.weight_params):
                self.ansatz_.assign_weights(self._rng.normal(size=self.ansatz_.num_weights))
        self._classes = classes
        self._qnn = self._create_qnn()
        self._loss_fn = self._get_loss_fn()
        self._X_fit = X.copy()
        circuits = self._encode(X)
        n_samples = len(X)
        batch_size = n_samples if self.batch_size is None else self.batch_size

        for epoch in range(self.epochs):
            # Shuffle data for each epoch
            idx = self._rng.permutation(n_samples)
            circuits_shuffled = [circuits[i] for i in idx]
            y_shuffled = y[idx]

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_circuits = circuits_shuffled[start:end]
                batch_y = y_shuffled[start:end]

                # Forward pass
                expectations = self._qnn.forward(batch_circuits, trainable=True)
                y_pred, y_true = self._prepare_for_loss(expectations, batch_y)

                # Compute loss
                loss = self._loss_fn(y_pred, y_true)

                # Backward pass and optimization
                prediction_derivative = 1.0
                if self.loss == "BCE":
                    prediction_derivative = -0.5
                elif self.loss == "MSE" and expectations.shape[1] == 1:
                    prediction_derivative = -1.0
                self._qnn.backward(self._loss_fn.grads(prediction_derivative))
                self._qnn.update(cur_loss=loss)
                self._qnn.zero_grad()

                # MSE/BCE return means; CE returns a sum. Weight batch means
                # by sample count, including the final partial batch. Multiclass
                # MSE remains an element mean, not a sum over output classes.
                # Accumulate metrics
                epoch_loss += loss if self.loss == "CrossEntropy" else loss * len(batch_circuits)
                n_batches += 1
                epoch_correct += self._compute_accuracy(expectations, batch_y) * len(batch_circuits)

            # Print progress
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                acc = epoch_correct / n_samples
                print(f"Epoch {epoch+1}/{self.epochs} - loss: {epoch_loss/n_samples:.4f} - acc: {acc:.4f}")

        return self






    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).

        Examples:
            >>> y_pred = vqc.predict(X_test)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        circuits = self._encode(X)
        expectations = self._qnn.forward(circuits, trainable=False)

        if self.loss == "MSE":
            if (len(self._classes) if self._classes is not None else self.n_classes) > 2:
                return self._classes[expectations.argmax(axis=1)]
            else:
                probabilities = (1 - expectations) / 2
                return self._classes[(probabilities.flatten() > 0.5).astype(int)]
        elif self.loss == "BCE":
            probabilities = (1 - expectations) / 2
            return self._classes[(probabilities.flatten() > 0.5).astype(int)]
        else:  # CrossEntropy
            return self._classes[expectations.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).

        Examples:
            >>> y_proba = vqc.predict_proba(X_test)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        circuits = self._encode(X)
        expectations = self._qnn.forward(circuits, trainable=False)

        if self.loss == "MSE":
            if (len(self._classes) if self._classes is not None else self.n_classes) > 2:
                # Softmax transformation
                exp_vals = np.exp(expectations - np.max(expectations, axis=1, keepdims=True))
                return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
            else:
                # Binary: convert to probability
                prob_class1 = (1 - expectations.flatten()) / 2
                prob_class0 = 1 - prob_class1
                return np.column_stack([prob_class0, prob_class1])
        elif self.loss == "BCE":
            prob_class1 = (1 - expectations.flatten()) / 2
            prob_class0 = 1 - prob_class1
            return np.column_stack([prob_class0, prob_class1])
        else:  # CrossEntropy
            exp_vals = np.exp(expectations - np.max(expectations, axis=1, keepdims=True))
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            y (np.ndarray): True labels of shape (n_samples,).

        Returns:
            float: Mean accuracy on the test data.

        Examples:
            >>> accuracy = vqc.score(X_test, y_test)
        """
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return parameters for sub-objects.
                Defaults to True.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.

        Examples:
            >>> params = vqc.get_params()
            >>> print(params['epochs'])
            100
        """
        return {
            "ansatz": self.ansatz,
            "encoder": self.encoder,
            "readouts": self.readouts,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "n_classes": self.n_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "initial_point": self.initial_point,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "VQC":
        """
        Set parameters for this estimator.

        Args:
            **params: Parameter names and values to set.

        Returns:
            VQC: The updated VQC instance.

        Examples:
            >>> vqc.set_params(epochs=200, batch_size=64)
        """
        unknown = set(params) - set(self.get_params(deep=False))
        if unknown:
            raise ValueError(f"Invalid VQC parameters: {sorted(unknown)}")
        self._validate_training_config(params.get('epochs', self.epochs),
                                       params.get('batch_size', self.batch_size))
        self._validate_loss_config(params.get('loss', self.loss),
                                   params.get('readouts', self.readouts),
                                   params.get('n_classes', self.n_classes))
        if "random_state" in params:
            make_rng(params["random_state"])
        changed = False
        for key, value in params.items():
            changed = changed or not same_parameter_value(getattr(self, key), value)
            setattr(self, key, value)
        if changed:
            if 'random_state' in params:
                self._rng = make_rng(self.random_state)
                self._initial_rng_state = deepcopy(self._rng.bit_generator.state)
            self._progress = None
            self._resume_complete = False
            for name in ('ansatz_', 'encoder_'):
                self.__dict__.pop(name, None)
            self._qnn = self._loss_fn = self._classes = self._X_fit = None
        return self
