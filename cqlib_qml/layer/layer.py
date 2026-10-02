# cqlib_qml/layer/layer.py
"""
Base class for all neural network layers.

This module defines the abstract base class that all neural network layers
must inherit from. It provides the common interface for parameter
management, forward/backward propagation, and optimization.

Examples:
    >>> from cqlib_qml.layer import Layer
    >>>
    >>> # Custom layer implementation
    >>> class MyLayer(Layer):
    ...     def init_params(self, **kwargs):
    ...         self._parameters["W"] = np.random.randn(10, 5)
    ...
    ...     def forward(self, x, **kwargs):
    ...         return x @ self._parameters["W"]
    ...
    ...     def backward(self, out, **kwargs):
    ...         return out @ self._parameters["W"].T
    >>>
    >>> layer = MyLayer()
    >>> layer.set_optimizer("adam")
"""

from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import warnings

from cqlib_qml.optimizer import *
from .activation import ActivationInitializer


class Layer(ABC):
    """
    Base class for all neural network layers.

    This abstract class defines the interface for neural network layers
    and provides common functionality for parameter management,
    optimization, and serialization.

    Attributes:
        parameters (dict): Trainable parameters of the layer.
        gradients (dict): Gradients of trainable parameters.
        trainable (bool): Whether the layer is trainable.
        updatable (bool): Whether parameters can be updated.
        hyperparameters (dict): Layer hyperparameters.

    Note:
        User-defined layers should inherit from this class and implement
        the abstract methods: `init_params`, `forward`, and `backward`.
    """

    @property
    def parameters(self) -> dict:
        """Trainable parameters of the layer."""
        return self._parameters

    @property
    def gradients(self) -> dict:
        """Gradients of trainable parameters."""
        return self._gradients

    @property
    def trainable(self) -> bool:
        """Whether the layer is trainable."""
        return self._trainable

    @property
    def updatable(self) -> bool:
        """Whether the parameters of the layer can be updated."""
        return self._updatable

    @property
    def hyperparameters(self) -> dict:
        """Dictionary containing the layer hyperparameters."""
        return None

    def __init__(self):
        """Initialize a Layer instance."""
        self._forward_valid = False
        self._gradient_valid = False
        self._tracks_gradient_validity = False
        self._inference_invalidated = False
        self._X = []
        self._act_fn = None
        self._trainable = True
        self._updatable = True
        self._optimizer = None

        self._parameters = {}
        self._gradients = {}
        self._derived_variables = {}

        super().__init__()

    @abstractmethod
    def init_params(self, **kwargs):
        """
        Initialize the layer's trainable parameters.

        This method should be implemented by subclasses to initialize
        weight matrices, biases, and other trainable parameters.

        Args:
            **kwargs: Optional initialization parameters.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x, **kwargs):
        """
        Perform forward propagation.

        Args:
            x (np.ndarray): Input data.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Output of the layer.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """
        Perform backward propagation.

        Args:
            out (np.ndarray): Gradients from the next layer.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Gradients for the previous layer.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    def set_optimizer(self, optimizer: Union[str, dict, OptimizerBase] = "adam") -> None:
        """
        Set the optimizer for the layer.

        Args:
            optimizer (Union[str, dict, OptimizerBase]): The optimizer to use.
                - str: Name of optimizer ("adam", "sgd", "adagrad", "rmsprop")
                - dict: Optimizer configuration dictionary
                - OptimizerBase: Optimizer instance
                Defaults to "adam".

        Raises:
            TypeError: If optimizer type is unsupported.

        Examples:
            >>> layer.set_optimizer("adam")
            >>> layer.set_optimizer("sgd(lr=0.01)")
            >>> layer.set_optimizer(Adam(lr=0.001))
        """
        self._optimizer = OptimizerInitializer(optimizer)()
        # Each component owns its state; stable parameter keys must not collide
        # when the same optimizer configuration is supplied to several layers.
        if isinstance(optimizer, OptimizerBase):
            self._optimizer = self._optimizer.copy()

    def freeze(self) -> None:
        """Freeze the parameters in the layer (disable training)."""
        self._trainable = False

    def unfreeze(self) -> None:
        """Unfreeze the parameters in the layer (enable training)."""
        self._trainable = True

    def zero_grad(self) -> None:
        """
        Reset the gradients of parameters to zero.

        Raises:
            ValueError: If the layer is frozen.
        """
        if not self._trainable:
            raise ValueError("Layer is frozen.")
        self._forward_valid = False
        self._gradient_valid = False
        self._inference_invalidated = False
        self._X = []
        for k, v in self._derived_variables.items():
            self._derived_variables[k] = None

        for k in self._gradients.keys():
            if k in self._parameters and self._parameters[k] is not None:
                self._gradients[k] = np.zeros_like(self._parameters[k])
            else:
                self._gradients[k] = np.zeros_like(self._gradients[k])

    def _invalidate_gradients(self):
        """Clear inference-invalid training state, including on frozen layers."""
        self._inference_invalidated = True
        self._forward_valid = False
        self._gradient_valid = False
        self._X = []
        for key in self._derived_variables:
            self._derived_variables[key] = None
        for key in self._gradients:
            self._gradients[key] = np.zeros_like(self._gradients[key])

    def update(self, cur_loss: Optional[float] = None) -> None:
        """
        Update the trainable parameters according to gradients.

        Args:
            cur_loss (float, optional): Current loss value for learning rate
                scheduling. Defaults to None.

        Raises:
            ValueError: If the layer is frozen.

        Examples:
            >>> layer.forward(X)
            >>> layer.backward(dLdy)
            >>> layer.update()
        """
        if not self._trainable:
            raise ValueError("Layer is frozen.")
        if self._inference_invalidated or (self._tracks_gradient_validity and not self._gradient_valid):
            return
        self._optimizer.step()
        for k, v in self._gradients.items():
            if k in self._parameters:
                unique_key = k
                self._parameters[k] = self._optimizer(self._parameters[k], v, unique_key, cur_loss)

    def summary(self) -> dict:
        """
        Get a summary of the layer.

        Returns:
            dict: All information about the layer including parameters
                and hyperparameters.

        Examples:
            >>> summary = layer.summary()
            >>> print(summary["layer"])
            Linear
        """
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": {**self.hyperparameters, "optimizer": (
                self._optimizer.state_dict() if self._optimizer is not None else None
            )},
        }

    def load_params(self, summary_dict: dict) -> None:
        """
        Load parameters from a summary dictionary.

        Args:
            summary_dict (dict): Summary dictionary containing layer
                parameters and hyperparameters.

        Raises:
            ValueError: If layer types don't match or dimensions mismatch.

        Examples:
            >>> layer.load_params(saved_summary)
        """
        if summary_dict["layer"] != self.hyperparameters["layer"]:
            raise ValueError("The layer to be loaded does not match.")
        if summary_dict["hyperparameters"]["in_dim"] != self._in_dim:
            raise ValueError("The input dimensions to be loaded do not match.")
        if summary_dict["hyperparameters"]["out_dim"] != self._out_dim:
            raise ValueError("The output dimensions to be loaded do not match.")
        if set(summary_dict["parameters"].keys()) != set(self._parameters.keys()):
            warnings.warn("The keys of parameters mismatch. Check your configuration.")
        for key, val in summary_dict["parameters"].items():
            self._parameters[key] = val
            self._gradients[key] = np.zeros_like(val)
        for key, val in summary_dict["hyperparameters"].items():
            if key == "optimizer":
                if val is not None:
                    self.set_optimizer(val)
            elif key == "act_fn":
                if bool(val) ^ bool(self._act_fn):
                    warnings.warn("Activation function mismatch. Check your configuration.")
                self._act_fn = ActivationInitializer(val)()
        self._init = True
        self._invalidate_gradients()
