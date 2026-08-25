# cqlib_qml/loss/loss.py
"""
Commonly used loss functions for quantum machine learning.

This module provides implementations of standard loss functions with
automatic differentiation support via autograd.

All loss functions inherit from the LossFun base class and implement:
    1. Forward computation: __call__ / _get_loss
    2. Gradient computation: grads method

The gradients are computed using autograd's automatic differentiation,
which provides exact gradients for the implemented loss functions.

Examples:
    >>> from cqlib_qml.loss import MSELoss, SoftmaxCrossEntropy
    >>>
    >>> # Using MSE loss
    >>> mse = MSELoss()
    >>> loss = mse(predictions, targets)
    >>> gradients = mse.grads()
    >>>
    >>> # Using Softmax Cross Entropy
    >>> ce = SoftmaxCrossEntropy()
    >>> loss = ce(logits, labels)
    >>> gradients = ce.grads()
"""

from abc import ABC, abstractmethod

import autograd.numpy as np
from autograd import grad


class LossFun(ABC):
    """
    Base class for all loss functions.

    This abstract class defines the interface that all loss functions
    must implement. It provides the common structure for loss computation
    and gradient calculation.

    Attributes:
        _pred (np.ndarray): Stored predictions from the last call.
        _target (np.ndarray): Stored targets from the last call.

    Note:
        User-defined loss functions should inherit from this class and
        implement the `_get_loss` method.

    Examples:
        >>> class MyLoss(LossFun):
        ...     def _get_loss(self, pred, target):
        ...         return np.mean((pred - target) ** 2)
        >>>
        >>> loss_fn = MyLoss()
        >>> loss = loss_fn(pred, target)
        >>> grads = loss_fn.grads()
    """

    def __init__(self):
        """Initialize a LossFun instance."""
        self._pred = None
        self._target = None

    def __call__(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the loss value.

        Args:
            pred (np.ndarray): Predicted values.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Loss value.

        Raises:
            ValueError: If predictions and targets have different shapes.

        Examples:
            >>> loss = loss_fn(pred, target)
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Predicted values shape {pred.shape} and ground truth " f"shape {target.shape} must be the same."
            )
        self._pred = pred
        self._target = target
        loss = self._get_loss(pred, target)
        return loss

    @abstractmethod
    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the loss value (to be implemented by subclasses).

        Args:
            pred (np.ndarray): Predicted values.
            target (np.ndarray): Ground truth values.

        Returns:
            float: Loss value.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def grads(self, dpred: np.ndarray = None) -> np.ndarray:
        """
        Compute gradients of the loss with respect to predictions.

        This method uses autograd to compute exact gradients. Optionally,
        gradients can be scaled by dpred for chain rule propagation.

        Args:
            dpred (np.ndarray, optional): Gradients from subsequent layers.
                Defaults to None.

        Returns:
            np.ndarray: Gradients of the loss with respect to predictions.

        Examples:
            >>> # Get raw gradients
            >>> grads = loss_fn.grads()
            >>>
            >>> # Scale gradients for chain rule
            >>> grads = loss_fn.grads(dpred=2.0)
        """
        fun_grad = grad(self._get_loss)
        gradients = fun_grad(self._pred, self._target)
        gradients[abs(self._pred - self._target) < 1e-12] = 0
        if dpred is not None:
            gradients *= dpred
        return gradients


class HingeLoss(LossFun):
    """
    Hinge loss for classification.

    L_Hinge(y) = max(0, 1 - ŷ * y)

    The hinge loss is commonly used for SVM-like classifiers. It penalizes
    predictions that are on the wrong side of the margin.

    Args:
        None

    References:
        - Cortes, C., & Vapnik, V. (1995). "Support-vector networks."

    Examples:
        >>> hinge = HingeLoss()
        >>> pred = np.array([0.8, -0.2])
        >>> target = np.array([1, -1])
        >>> loss = hinge(pred, target)
        >>> grads = hinge.grads()
    """

    def __init__(self):
        """Initialize a HingeLoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the hinge loss.

        Args:
            pred (np.ndarray): Predicted values.
            target (np.ndarray): Target values in {-1, 1}.

        Returns:
            float: Hinge loss value.
        """
        loss = np.clip(1 - pred * target, a_min=0.0, a_max=None)
        return np.mean(loss)

    def __str__(self):
        return "HingeLoss"


class MSELoss(LossFun):
    """
    Mean Squared Error loss for regression.

    L_MSE(y) = (1/N) * Σ(ŷ_i - y_i)²

    The MSE loss is commonly used for regression tasks. It penalizes
    large errors more heavily than small errors.

    Args:
        None

    Examples:
        >>> mse = MSELoss()
        >>> pred = np.array([0.5, 0.3, 0.7])
        >>> target = np.array([0.5, 0.3, 0.7])
        >>> loss = mse(pred, target)  # Returns 0.0
        >>>
        >>> pred = np.array([0.6, 0.4])
        >>> target = np.array([0.5, 0.3])
        >>> loss = mse(pred, target)  # Returns 0.01
    """

    def __init__(self):
        """Initialize an MSELoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the mean squared error loss.

        Args:
            pred (np.ndarray): Predicted values.
            target (np.ndarray): Ground truth values.

        Returns:
            float: MSE loss value.
        """
        loss = (pred - target) ** 2
        return np.mean(loss)

    def __str__(self):
        return "MSELoss"


class BCELoss(LossFun):
    """
    Binary Cross Entropy loss for binary classification.

    L_BCE(y) = -(1/N) * (y · log(ŷ) + (1-y) · log(1-ŷ))

    The BCE loss is used for binary classification tasks where targets
    are in {0, 1}.

    Args:
        None

    Note:
        Targets should be in the range [0, 1].

    Examples:
        >>> bce = BCELoss()
        >>> pred = np.array([0.9, 0.1])
        >>> target = np.array([1.0, 0.0])
        >>> loss = bce(pred, target)
        >>> grads = bce.grads()
    """

    def __init__(self):
        """Initialize a BCELoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the binary cross entropy loss.

        Args:
            pred (np.ndarray): Predicted probabilities in (0, 1).
            target (np.ndarray): Target values in {0, 1}.

        Returns:
            float: BCE loss value.
        """
        eps = np.finfo(float).eps
        loss = np.clip(-target * np.log(pred + eps) - (1 - target) * np.log(1 - pred + eps), 0, 100)
        return np.mean(loss)

    def __str__(self):
        return "BCELoss"


class CrossEntropy(LossFun):
    """
    Cross Entropy loss for multi-class classification.

    L_CE(y) = -Σ y_i · log(ŷ_i)

    This loss is used for multi-class classification with one-hot encoded
    targets. Predictions should be probabilities that sum to 1.

    Args:
        None

    Note:
        Targets should be one-hot encoded. Predictions should be
        probabilities (e.g., from softmax activation).

    Examples:
        >>> ce = CrossEntropy()
        >>> pred = np.array([[0.7, 0.2, 0.1]])
        >>> target = np.array([[1.0, 0.0, 0.0]])
        >>> loss = ce(pred, target)
        >>> grads = ce.grads()
    """

    def __init__(self):
        """Initialize a CrossEntropyLoss instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the cross entropy loss.

        Args:
            pred (np.ndarray): Predicted probabilities (sum to 1 per sample).
            target (np.ndarray): One-hot encoded target values.

        Returns:
            float: Cross entropy loss value.
        """
        eps = np.finfo(float).eps
        loss = np.clip(-np.sum(target * np.log(pred + eps)), 0, 100)
        return loss

    def __str__(self):
        return "CrossEntropyLoss"


class SoftmaxCrossEntropy(LossFun):
    """
    Cross Entropy loss with built-in softmax activation.

    This loss combines softmax activation with cross entropy loss.
    It takes raw logits as input and applies softmax internally.

    L_CE(y) = -Σ y_i · log(softmax(ŷ)_i)

    Args:
        None

    Note:
        This loss should be used when the last layer produces raw logits
        (without softmax activation). The softmax is applied internally.

    Examples:
        >>> ce = SoftmaxCrossEntropy()
        >>> logits = np.array([[1.0, 0.5, 0.2]])
        >>> target = np.array([[1.0, 0.0, 0.0]])
        >>> loss = ce(logits, target)
        >>> grads = ce.grads()
        >>>
        >>> # Gradients are with respect to logits
        >>> print(grads.shape)
        (1, 3)
    """

    def __init__(self):
        """Initialize a SoftmaxCrossEntropy instance."""
        super().__init__()

    def _get_loss(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute the softmax cross entropy loss.

        Args:
            pred (np.ndarray): Raw logits.
            target (np.ndarray): One-hot encoded target values.

        Returns:
            float: Cross entropy loss value.
        """
        exps = np.exp(pred)
        self._pred = exps / np.sum(exps, axis=1).reshape(-1, 1)
        eps = np.finfo(float).eps
        loss = np.clip(-np.sum(target * np.log(self._pred + eps)), 0, 100)
        return loss

    def grads(self, dpred: np.ndarray = None) -> np.ndarray:
        """
        Compute gradients of the loss with respect to logits.

        For softmax cross entropy, the gradient simplifies to:
        dL/dlogits = softmax(logits) - targets

        Args:
            dpred (np.ndarray, optional): Gradients from subsequent layers.
                Defaults to None.

        Returns:
            np.ndarray: Gradients with respect to logits.

        Examples:
            >>> ce = SoftmaxCrossEntropy()
            >>> logits = np.array([[1.0, 0.5, 0.2]])
            >>> target = np.array([[1.0, 0.0, 0.0]])
            >>> loss = ce(logits, target)
            >>> grads = ce.grads()
            >>> print(grads)
            [[-0.09  0.12  0.09]]  # Approximate values
        """
        gradients = self._pred - self._target
        gradients[abs(self._pred - self._target) < 1e-12] = 0
        if dpred is not None:
            gradients *= dpred
        return gradients

    def __str__(self):
        return "CrossEntropyLoss"
