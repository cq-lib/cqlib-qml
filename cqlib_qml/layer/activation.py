# cqlib_qml/layer/activation.py
"""
Activation functions for neural networks.

This module provides common activation functions used in neural networks,
along with their derivatives. Both the activation function and its gradient
are implemented for use in forward and backward propagation.

Available Activation Functions:
    - Sigmoid: Logistic sigmoid function
    - ReLU: Rectified Linear Unit
    - Tanh: Hyperbolic tangent
    - SoftPlus: Smooth approximation of ReLU

Each activation function provides:
    - act(x): Apply the activation function
    - grad(x): Compute the first derivative
    - grad2(x): Compute the second derivative (where available)

Examples:
    >>> from cqlib_qml.layer import Sigmoid, ReLU, Tanh, SoftPlus
    >>> import numpy as np
    >>>
    >>> # Sigmoid activation
    >>> sigmoid = Sigmoid()
    >>> x = np.array([0.0, 1.0, -1.0])
    >>> y = sigmoid.act(x)
    >>> dy = sigmoid.grad(x)
    >>>
    >>> # ReLU activation
    >>> relu = ReLU()
    >>> y = relu.act(np.array([-1, 0, 1]))  # array([0, 0, 1])
"""

from abc import ABC, abstractmethod
import numpy as np


class ActivationBase(ABC):
    """
    Abstract base class for all activation functions.

    This class defines the interface that all activation functions must
    implement. It provides the `__call__` method for convenient application
    and abstract methods for the activation function and its gradient.

    Note:
        User-defined activation functions should inherit from this class
        and implement the `act` and `grad` methods.
    """

    def __init__(self, **kwargs):
        """Initialize an ActivationBase instance."""
        super().__init__()

    def __call__(self, x):
        """
        Apply the activation function to the input.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying the activation function.

        Examples:
            >>> sigmoid = Sigmoid()
            >>> y = sigmoid(np.array([0.0, 1.0]))
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.act(x)

    @abstractmethod
    def act(self, x):
        """
        Apply the activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying the activation function.
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """
        Compute the gradient of the activation function.

        Args:
            x (np.ndarray): Input data.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Gradient of the activation function.
        """
        raise NotImplementedError


class ActivationInitializer:
    """
    Factory class for initializing activation functions.

    Provides a convenient way to create activation function instances
    from string names.

    Args:
        act_fun (str): Name of the activation function.
            Options: "sigmoid", "relu", "tanh", "softplus", None.

    Attributes:
        __ACTIVATIONS (list): List of supported activation functions.

    Raises:
        ValueError: If activation function name is not supported.

    Examples:
        >>> init = ActivationInitializer("sigmoid")
        >>> act = init()  # Returns a Sigmoid instance
        >>>
        >>> init = ActivationInitializer(None)
        >>> act = init()  # Returns None
    """

    __ACTIVATIONS = ["sigmoid", "relu", "tanh", "softplus", None]

    def __init__(self, act_fun: str):
        """
        Initialize an ActivationInitializer instance.

        Args:
            act_fun (str): Name of the activation function.

        Raises:
            ValueError: If activation function name is not supported.
        """
        if act_fun not in ActivationInitializer.__ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation function: {act_fun}. " f"Supported: {ActivationInitializer.__ACTIVATIONS}"
            )
        self._act_fun = act_fun

    def __call__(self):
        """
        Create the activation function instance.

        Returns:
            ActivationBase or None: The activation function instance,
                or None if act_fun is None.
        """
        if self._act_fun == "sigmoid":
            return Sigmoid()
        elif self._act_fun == "relu":
            return ReLU()
        elif self._act_fun == "tanh":
            return Tanh()
        elif self._act_fun == "softplus":
            return SoftPlus()
        else:
            return None


class Sigmoid(ActivationBase):
    """
    Logistic sigmoid activation function.

    σ(x) = 1 / (1 + e^(-x))

    The sigmoid function maps any real value to the range (0, 1).
    It is commonly used as the output activation for binary classification.

    Attributes:
        None

    Examples:
        >>> sigmoid = Sigmoid()
        >>> x = np.array([0.0, 1.0, -1.0])
        >>> y = sigmoid.act(x)  # array([0.5, 0.731, 0.269])
        >>> dy = sigmoid.grad(x)  # Gradient: y * (1 - y)
    """

    def __init__(self):
        """Initialize a Sigmoid instance."""
        super().__init__()

    def __str__(self):
        """Return a string representation."""
        return "sigmoid"

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the logistic sigmoid activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Sigmoid output in range (0, 1).
        """
        return 1 / (1 + np.exp(-x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the first derivative of the sigmoid function.

        dσ/dx = σ(x) * (1 - σ(x))

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Gradient of the sigmoid function.
        """
        y = self.__call__(x)
        return y * (1 - y)

    def grad2(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the second derivative of the sigmoid function.

        d²σ/dx² = σ(x) * (1 - σ(x)) * (1 - 2σ(x))

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Second derivative of the sigmoid function.
        """
        y = self.__call__(x)
        return y * (1 - y) * (1 - 2 * y)


class ReLU(ActivationBase):
    """
    Rectified Linear Unit (ReLU) activation function.

    ReLU(x) = max(0, x)

    The ReLU function is one of the most commonly used activation functions
    in deep learning due to its simplicity and effectiveness.

    Attributes:
        None

    Examples:
        >>> relu = ReLU()
        >>> x = np.array([-1.0, 0.0, 2.0])
        >>> y = relu.act(x)  # array([0.0, 0.0, 2.0])
        >>> dy = relu.grad(x)  # array([0.0, 0.0, 1.0])
    """

    def __init__(self):
        """Initialize a ReLU instance."""
        super().__init__()

    def __str__(self):
        """Return a string representation."""
        return "relu"

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the ReLU activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: ReLU output (non-negative).
        """
        return np.clip(x, 0, np.inf)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the first derivative of the ReLU function.

        dReLU/dx = 1 if x > 0 else 0

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Gradient of the ReLU function.
        """
        return (x > 0).astype(int)

    def grad2(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the second derivative of the ReLU function.

        d²ReLU/dx² = 0 (except at x=0 where it's undefined)

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Second derivative of the ReLU function (all zeros).
        """
        return np.zeros_like(x)


class Tanh(ActivationBase):
    """
    Hyperbolic tangent activation function.

    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    The tanh function maps any real value to the range (-1, 1).
    It is similar to sigmoid but centered at 0.

    Attributes:
        None

    Examples:
        >>> tanh = Tanh()
        >>> x = np.array([0.0, 1.0, -1.0])
        >>> y = tanh.act(x)  # array([0.0, 0.762, -0.762])
        >>> dy = tanh.grad(x)  # array([1.0, 0.420, 0.420])
    """

    def __init__(self):
        """Initialize a Tanh instance."""
        super().__init__()

    def __str__(self):
        """Return a string representation."""
        return "tanh"

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the tanh activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: tanh output in range (-1, 1).
        """
        return np.tanh(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the first derivative of the tanh function.

        dtanh/dx = 1 - tanh(x)²

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Gradient of the tanh function.
        """
        return 1 - np.tanh(x) ** 2

    def grad2(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the second derivative of the tanh function.

        d²tanh/dx² = -2 * tanh(x) * (1 - tanh(x)²)

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Second derivative of the tanh function.
        """
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1 - tanh_x**2)


class SoftPlus(ActivationBase):
    """
    SoftPlus activation function.

    SoftPlus(x) = log(1 + e^x)

    SoftPlus is a smooth approximation of the ReLU function. It is
    differentiable everywhere and has a non-zero gradient for all inputs.

    Attributes:
        None

    Examples:
        >>> softplus = SoftPlus()
        >>> x = np.array([0.0, 1.0, -1.0])
        >>> y = softplus.act(x)  # array([0.693, 1.313, 0.313])
        >>> dy = softplus.grad(x)  # array([0.5, 0.731, 0.269])
    """

    def __init__(self):
        """Initialize a SoftPlus instance."""
        super().__init__()

    def __str__(self):
        """Return a string representation."""
        return "softplus"

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the SoftPlus activation function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: SoftPlus output.
        """
        return np.log(np.exp(x) + 1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the first derivative of the SoftPlus function.

        dSoftPlus/dx = e^x / (1 + e^x) = sigmoid(x)

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Gradient of the SoftPlus function.
        """
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)

    def grad2(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the second derivative of the SoftPlus function.

        d²SoftPlus/dx² = e^x / (1 + e^x)²

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Second derivative of the SoftPlus function.
        """
        exp_x = np.exp(x)
        return exp_x / ((exp_x + 1) ** 2)
