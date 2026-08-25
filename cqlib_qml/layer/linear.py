# cqlib_qml/layer/linear.py
"""
Fully connected (dense) linear layer implementation.

This module provides a fully connected linear layer with support for
bias, activation functions, and automatic differentiation. It is the
standard dense layer used in classical neural networks and hybrid
quantum-classical models.

The layer computes: y = x @ W^T + b, followed by an optional activation
function.

Examples:
    >>> from cqlib_qml.layer import Linear
    >>> import numpy as np
    >>>
    >>> # Create a linear layer
    >>> layer = Linear(in_dim=10, out_dim=5, bias=True, act_fn="relu")
    >>>
    >>> # Forward pass
    >>> X = np.random.randn(32, 10)
    >>> output = layer.forward(X)  # Shape: (32, 5)
    >>>
    >>> # Backward pass
    >>> dLdy = np.random.randn(32, 5)
    >>> dX = layer.backward(dLdy)  # Shape: (32, 10)
"""

import numbers
import numpy as np

from .activation import ActivationInitializer
from .layer import Layer


class Linear(Layer):
    """
    Fully connected linear layer.

    Computes y = xA^T + b, where A is the weight matrix and b is the bias.
    An optional activation function can be applied after the linear transformation.

    Args:
        in_dim (int): Dimension of the input features.
        out_dim (int): Dimension of the output features.
        bias (bool, optional): Whether to include a bias term.
            Defaults to True.
        act_fn (str, optional): Activation function to apply.
            Options: "sigmoid", "relu", "tanh", "softplus", None.
            Defaults to None.

    Attributes:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hyperparameters (dict): Layer hyperparameters.

    Raises:
        ValueError: If input dimensions don't match during forward/backward.

    Examples:
        >>> # Linear layer with sigmoid activation
        >>> layer = Linear(in_dim=10, out_dim=5, act_fn="sigmoid")
        >>>
        >>> # Linear layer without bias
        >>> layer = Linear(in_dim=10, out_dim=5, bias=False)
        >>>
        >>> # Linear layer without activation
        >>> layer = Linear(in_dim=10, out_dim=5, act_fn=None)
    """

    @property
    def in_dim(self) -> int:
        """Input dimension of the layer."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """Output dimension of the layer."""
        return self._out_dim

    @property
    def hyperparameters(self) -> dict:
        """Dictionary containing the layer hyperparameters."""
        return {
            "layer": "Linear",
            "in_dim": self._in_dim,
            "out_dim": self._out_dim,
            "act_fn": str(self._act_fn) if self._act_fn is not None else None,
            "optimizer": {
                "cache": self._optimizer.cache if self._optimizer is not None else {},
                "hyperparameters": (self._optimizer.hyperparameters if self._optimizer is not None else {}),
            },
        }

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        act_fn: str = None,
    ):
        """
        Initialize a Linear layer.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            bias (bool, optional): Include bias term. Defaults to True.
            act_fn (str, optional): Activation function. Defaults to None.
        """
        super(Linear, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._bias = bias
        self._act_fn = ActivationInitializer(act_fn)()
        self._parameters = {"W": None, "b": None} if bias else {"W": None}
        self._gradients = {"W": None, "b": None} if bias else {"W": None}
        self._derived_variables = {"z": None}
        self._init = False

    def __str__(self) -> str:
        """Return a string representation of the layer."""
        return (
            f"Layer        | Linear(in_dim={self._in_dim}, out_dim={self._out_dim}) \n"
            + f"n_params     | {self._out_dim * (self._in_dim + 1) if self._bias else self._out_dim * self._in_dim} \n"
            + f"act_fn       | {str(self._act_fn)} \n"
            + f"optimizer    | {str(self._optimizer)} \n"
        )

    def set_activation(self, act_fn: str) -> None:
        """
        Set or change the activation function.

        Args:
            act_fn (str): Name of the activation function.
                Options: "sigmoid", "relu", "tanh", "softplus", None.

        Examples:
            >>> layer.set_activation("relu")
            >>> layer.set_activation(None)  # No activation
        """
        self._act_fn = ActivationInitializer(act_fn)()

    def init_params(self) -> None:
        """
        Initialize the layer's trainable parameters.

        Uses Kaiming uniform initialization for weights and He uniform
        for biases. The initialization scales with 1/sqrt(in_dim).

        Examples:
            >>> layer.init_params()
            >>> print(layer._parameters["W"].shape)
            (5, 10)
        """
        # Kaiming uniform initialization for weights
        b = np.sqrt(1 / self._in_dim)
        W = np.random.uniform(-b, b, size=(self._out_dim, self._in_dim))
        self._parameters["W"] = W
        self._gradients["W"] = np.zeros_like(W)
        if self._bias:
            bias = np.random.uniform(-b, b, size=(1, self._out_dim))
            self._parameters["b"] = bias
            self._gradients["b"] = np.zeros_like(bias)
        self._init = True

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        """
        Perform forward propagation.

        Computes: z = X @ W^T + b, y = activation(z)

        Args:
            X (np.ndarray): Input data of shape (batch_size, in_dim)
                or (in_dim,) for a single sample.
            retain_derived (bool, optional): Whether to retain intermediate
                values for backward propagation. Defaults to True.

        Returns:
            np.ndarray: Output of shape (batch_size, out_dim).

        Raises:
            ValueError: If input dimensions don't match.

        Examples:
            >>> X = np.random.randn(32, 10)
            >>> output = layer.forward(X)  # Shape: (32, 5)
            >>>
            >>> # Single sample
            >>> X = np.random.randn(10)
            >>> output = layer.forward(X)  # Shape: (5,)
        """
        if X is None:
            raise ValueError("Input should not be None.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._in_dim:
            raise ValueError(f"Input dimension {X.shape[1]} does not match expected {self._in_dim}.")
        if not self._init:
            self.init_params()
        W = self._parameters["W"]
        b = self._parameters["b"] if self._bias else 0
        z = np.dot(X, W.T) + b
        y = self._act_fn(z) if self._act_fn else z

        if retain_derived:
            self._X = X
            self._derived_variables["z"] = z

        return y

    def _bwd(self, dLdy: np.ndarray, x: np.ndarray) -> tuple:
        """
        Compute gradients for a single sample.

        Args:
            dLdy (np.ndarray): Gradients from the next layer. Shape: (1, out_dim) or (out_dim,).
            x (np.ndarray): Input for this sample. Shape: (1, in_dim) or (in_dim,).

        Returns:
            tuple: (dLdx, dLdw, dLdb) where:
                - dLdx: Gradients for the input (1, in_dim)
                - dLdw: Gradients for the weights (out_dim, in_dim)
                - dLdb: Gradients for the bias (1, out_dim)
        """
        W = self._parameters["W"]  # (out_dim, in_dim)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        if dLdy.ndim == 1:
            dLdy = dLdy.reshape(1, -1)

        b = self._parameters["b"] if self._bias else 0
        z = np.dot(x, W.T) + b

        dydz = self._act_fn.grad(z) if self._act_fn else 1
        dLdz = dLdy * dydz
        dLdx = dLdz @ W
        dLdw = np.outer(dLdz.flatten(), x.flatten())
        dLdb = dLdz.sum(axis=0, keepdims=True)

        return dLdx, dLdw, dLdb

    def backward(self, dLdy: np.ndarray, retain_grad: bool = True) -> np.ndarray:
        """
        Perform backward propagation.

        Computes gradients of the loss with respect to the input,
        weights, and bias using the chain rule.

        Args:
            dLdy (np.ndarray): Gradients of the loss with respect to
                the layer output. Shape: (batch_size, out_dim) or (out_dim,).
            retain_grad (bool, optional): Whether to retain gradients
                for parameter updates. Defaults to True.

        Returns:
            np.ndarray: Gradients with respect to the input.
                Shape: (batch_size, in_dim) or (in_dim,) for single sample.

        Raises:
            ValueError: If layer is frozen or dimensions mismatch.

        Examples:
            >>> dLdy = np.random.randn(32, 5)
            >>> dX = layer.backward(dLdy)  # Shape: (32, 10)
            >>>
            >>> # Single sample
            >>> dLdy = np.random.randn(5)
            >>> dX = layer.backward(dLdy)  # Shape: (10,)
        """
        if not self._trainable:
            raise ValueError("Layer is frozen.")
        if isinstance(dLdy, numbers.Number):
            dLdy = np.array([dLdy])
        if not isinstance(dLdy, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(dLdy).__name__}")
        if dLdy.ndim == 1:
            dLdy = dLdy.reshape(1, -1)
        if dLdy.shape[1] != self._out_dim:
            raise ValueError(f"Gradient dimension {dLdy.shape[1]} does not match output dimension {self._out_dim}.")
        if self._X.shape[0] != dLdy.shape[0]:
            raise ValueError(f"Batch size mismatch: input batch {self._X.shape[0]} vs gradient batch {dLdy.shape[0]}.")

        dX = []
        X = self._X
        for dy, x in zip(dLdy, X):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if dy.ndim == 1:
                dy = dy.reshape(1, -1)
            dx, dw, db = self._bwd(dy, x)
            if dx.shape[0] == 1:
                dX.append(dx.flatten())
            else:
                dX.append(dx)
            if retain_grad:
                self._gradients["W"] += dw
                if self._bias:
                    self._gradients["b"] += db
        return dX[0] if len(X) == 1 else np.array(dX)
