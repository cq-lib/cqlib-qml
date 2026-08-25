# cqlib_qml/optimizer/optimizer.py
"""
Optimization algorithms for training quantum models.

This module provides implementations of standard optimization algorithms
with support for learning rate scheduling and gradient clipping.

All optimizers follow a common interface:
    1. update: Compute the parameter update
    2. step: Increment the step counter
    3. reset_step: Reset the step counter

The optimizers maintain a cache for each parameter to store state
information (e.g., momentum, running averages).

References:
    - SGD: Robbins & Monro (1951)
    - AdaGrad: Duchi et al. (2011)
    - RMSProp: Tieleman & Hinton (2012)
    - Adam: Kingma & Ba (2015)

Examples:
    >>> from cqlib_qml.optimizer import Adam, OptimizerInitializer
    >>>
    >>> # Create optimizer directly
    >>> opt = Adam(lr=0.001)
    >>>
    >>> # Create optimizer from string
    >>> opt = OptimizerInitializer("adam(lr=0.001)")()
    >>>
    >>> # Update parameters
    >>> param = np.array([1.0, 2.0])
    >>> grad = np.array([0.1, 0.2])
    >>> new_param = opt.update(param, grad, "weight")
"""

import re
from abc import ABC, abstractmethod
from ast import literal_eval as eval
from copy import deepcopy

import numpy as np
from numpy.linalg import norm

from .scheduler import SchedulerInitializer


class OptimizerBase(ABC):
    """
    Abstract base class for all optimizers.

    This class defines the interface that all optimizers must implement.
    It provides common functionality for parameter updates, learning rate
    scheduling, and state management.

    Attributes:
        cache (dict): Cache for optimizer state (momentum, running averages).
        cur_step (int): Current optimization step.
        hyperparameters (dict): Optimizer hyperparameters.
        lr (float): Learning rate.
        lr_scheduler (SchedulerBase): Learning rate scheduler.

    Note:
        User-defined optimizers should inherit from this class and
        implement the `update` method.

    Examples:
        >>> class MyOptimizer(OptimizerBase):
        ...     def update(self, param, param_grad, param_name, cur_loss=None):
        ...         return param - self.lr * param_grad
    """

    def __init__(self, lr: float, scheduler=None):
        """
        Initialize an OptimizerBase instance.

        Args:
            lr (float): Learning rate.
            scheduler (optional): Learning rate scheduler. Defaults to None.
        """
        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr = lr
        self.lr_scheduler = SchedulerInitializer(scheduler, lr=lr)()

    def __call__(
        self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None
    ) -> np.ndarray:
        """
        Compute the parameter update.

        This is a wrapper around the `update` method for convenient calling.

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.
                Defaults to None.

        Returns:
            np.ndarray: Updated parameter values.
        """
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1."""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero."""
        self.cur_step = 0

    def copy(self):
        """
        Return a copy of the optimizer.

        Returns:
            OptimizerBase: Copy of the optimizer object.
        """
        return deepcopy(self)

    def set_scheduler(self, scheduler):
        """
        Set the learning rate scheduler.

        Args:
            scheduler: Scheduler instance or configuration.
        """
        self.lr_scheduler = SchedulerInitializer(scheduler, lr=self.lr)()
        self.hyperparameters["lr_scheduler"] = str(self.lr_scheduler)

    def remove_scheduler(self):
        """Remove the learning rate scheduler (use constant LR)."""
        self.lr_scheduler = SchedulerInitializer(None, lr=self.lr)()
        self.hyperparameters["lr_scheduler"] = str(self.lr_scheduler)

    def set_params(self, hparam_dict: dict = None, cache_dict: dict = None):
        """
        Set optimizer parameters from dictionaries.

        Args:
            hparam_dict (dict, optional): Hyperparameters dictionary.
            cache_dict (dict, optional): Cache dictionary.

        Returns:
            OptimizerBase: The optimizer instance.
        """
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
                    if k == "lr_scheduler":
                        self.lr_scheduler = SchedulerInitializer(v, lr=None)()

        if cache_dict is not None:
            for k, v in cache_dict.items():
                self.cache[k] = v
        return self

    @abstractmethod
    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None) -> np.ndarray:
        """
        Compute the parameter update (to be implemented by subclasses).

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.

        Returns:
            np.ndarray: Updated parameter values.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


class OptimizerInitializer:
    """
    Factory class for initializing optimizers.

    Provides a convenient way to create optimizer instances from strings,
    dictionaries, or existing optimizer instances.

    Args:
        param: Optimizer configuration. Can be:
            - None: Returns default SGD
            - str: Optimizer name with optional parameters
            - dict: Optimizer configuration dictionary
            - OptimizerBase: Existing optimizer instance

    Attributes:
        param: The optimizer configuration.

    Examples:
        >>> # From string
        >>> init = OptimizerInitializer("adam(lr=0.001)")
        >>> opt = init()
        >>>
        >>> # From dict
        >>> init = OptimizerInitializer({
        ...     "hyperparameters": {"id": "Adam", "lr": 0.001},
        ...     "cache": {}
        ... })
        >>> opt = init()
        >>>
        >>> # From existing optimizer
        >>> init = OptimizerInitializer(Adam(lr=0.001))
        >>> opt = init()
    """

    def __init__(self, param=None):
        """
        Initialize an OptimizerInitializer instance.

        Args:
            param: Optimizer configuration.
        """
        self.param = param

    def __call__(self):
        """
        Create the optimizer instance.

        Returns:
            OptimizerBase: The optimizer instance.
        """
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str()
        elif isinstance(param, dict):
            opt = self.init_from_dict()
        return opt

    def init_from_str(self):
        """
        Initialize optimizer from a string.

        Supported formats:
            - "adam": Adam with default parameters
            - "adam(lr=0.001)": Adam with custom learning rate
            - "sgd(lr=0.01, momentum=0.9)": SGD with momentum

        Returns:
            OptimizerBase: The optimizer instance.

        Raises:
            ValueError: If optimizer name is not supported.
        """
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opt_str)])
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        elif "adagrad" in opt_str:
            optimizer = AdaGrad(**kwargs)
        elif "rmsprop" in opt_str:
            optimizer = RMSProp(**kwargs)
        elif "adam" in opt_str:
            optimizer = Adam(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_str}. " f"Supported: ['sgd', 'adagrad', 'rmsprop', 'adam']")
        return optimizer

    def init_from_dict(self):
        """
        Initialize optimizer from a dictionary.

        Args:
            param (dict): Configuration dictionary with 'hyperparameters'
                and optionally 'cache' keys.

        Returns:
            OptimizerBase: The optimizer instance.

        Raises:
            ValueError: If optimizer ID is not supported.
        """
        O = self.param
        cc = O["cache"] if "cache" in O else None
        op = O["hyperparameters"] if "hyperparameters" in O else None

        if op is None:
            raise ValueError("Must have `hyperparameters` key.")

        if op and op["id"] == "SGD":
            optimizer = SGD().set_params(op, cc)
        elif op and op["id"] == "RMSProp":
            optimizer = RMSProp().set_params(op, cc)
        elif op and op["id"] == "AdaGrad":
            optimizer = AdaGrad().set_params(op, cc)
        elif op and op["id"] == "Adam":
            optimizer = Adam().set_params(op, cc)
        elif op:
            raise ValueError(f"Unsupported optimizer: {op['id']}. " f"Supported: ['SGD', 'RMSProp', 'AdaGrad', 'Adam']")
        return optimizer


class SGD(OptimizerBase):
    """
    Stochastic Gradient Descent optimizer with momentum.

    update^t = momentum * update^{t-1} + lr * ∇L
    θ^{t+1} = θ^t - update^t

    Args:
        lr (float, optional): Learning rate. Defaults to 0.01.
        momentum (float, optional): Momentum coefficient in [0, 1].
            Defaults to 0.
        clip_norm (float, optional): Maximum L2 norm for gradients.
            Defaults to None.
        lr_scheduler (optional): Learning rate scheduler. Defaults to None.

    Examples:
        >>> # Basic SGD
        >>> opt = SGD(lr=0.01)
        >>>
        >>> # SGD with momentum
        >>> opt = SGD(lr=0.01, momentum=0.9)
        >>>
        >>> # SGD with gradient clipping
        >>> opt = SGD(lr=0.01, clip_norm=1.0)
    """

    def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs):
        """Initialize an SGD instance."""
        super().__init__(lr, lr_scheduler)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        )

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None) -> np.ndarray:
        """
        Compute the SGD update.

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.

        Returns:
            np.ndarray: Updated parameter values.
        """
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # Gradient clipping
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update


class AdaGrad(OptimizerBase):
    """
    AdaGrad optimizer with adaptive learning rates.

    cache[t] = cache[t-1] + grad[t]²
    update[t] = lr * grad[t] / (√cache[t] + eps)
    θ[t+1] = θ[t] - update[t]

    References:
        - Duchi, J., et al. (2011). "Adaptive Subgradient Methods for
          Online Learning and Stochastic Optimization."

    Args:
        lr (float, optional): Learning rate. Defaults to 0.01.
        eps (float, optional): Smoothing term. Defaults to 1e-7.
        clip_norm (float, optional): Maximum L2 norm for gradients.
            Defaults to None.
        lr_scheduler (optional): Learning rate scheduler. Defaults to None.

    Examples:
        >>> opt = AdaGrad(lr=0.01)
        >>> opt = AdaGrad(lr=0.01, eps=1e-8)
    """

    def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs):
        """Initialize an AdaGrad instance."""
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "AdaGrad",
            "lr": lr,
            "eps": eps,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        return "AdaGrad(lr={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            H["lr"], H["eps"], H["clip_norm"], H["lr_scheduler"]
        )

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None) -> np.ndarray:
        """
        Compute the AdaGrad update.

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.

        Returns:
            np.ndarray: Updated parameter values.
        """
        C = self.cache
        H = self.hyperparameters
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # Gradient clipping
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] += param_grad**2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update


class RMSProp(OptimizerBase):
    """
    RMSProp optimizer with adaptive learning rates.

    cache[t] = decay * cache[t-1] + (1-decay) * grad[t]²
    update[t] = lr * grad[t] / (√cache[t] + eps)
    θ[t+1] = θ[t] - update[t]

    References:
        - Tieleman, T., & Hinton, G. (2012). "RMSProp: Divide the
          gradient by a running average of its recent magnitude."

    Args:
        lr (float, optional): Learning rate. Defaults to 0.001.
        decay (float, optional): Decay rate in [0, 1]. Defaults to 0.9.
        eps (float, optional): Smoothing term. Defaults to 1e-7.
        clip_norm (float, optional): Maximum L2 norm for gradients.
            Defaults to None.
        lr_scheduler (optional): Learning rate scheduler. Defaults to None.

    Examples:
        >>> opt = RMSProp(lr=0.001, decay=0.9)
        >>> opt = RMSProp(lr=0.001, decay=0.99, eps=1e-8)
    """

    def __init__(self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs):
        """Initialize an RMSProp instance."""
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "RMSProp",
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        return "RMSProp(lr={}, eps={}, decay={}, clip_norm={}, lr_scheduler={})".format(
            H["lr"], H["eps"], H["decay"], H["clip_norm"], H["lr_scheduler"]
        )

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None) -> np.ndarray:
        """
        Compute the RMSProp update.

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.

        Returns:
            np.ndarray: Updated parameter values.
        """
        C = self.cache
        H = self.hyperparameters
        eps, decay, clip_norm = H["eps"], H["decay"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # Gradient clipping
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        C[param_name] = decay * C[param_name] + (1 - decay) * param_grad**2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update


class Adam(OptimizerBase):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    m[t] = β1 * m[t-1] + (1-β1) * grad[t]  (First moment)
    v[t] = β2 * v[t-1] + (1-β2) * grad[t]² (Second moment)
    m_hat = m[t] / (1-β1^t)                 (Bias correction)
    v_hat = v[t] / (1-β2^t)                 (Bias correction)
    update = lr * m_hat / (√v_hat + eps)
    θ[t+1] = θ[t] - update

    References:
        - Kingma, D. P., & Ba, J. (2015). "Adam: A Method for
          Stochastic Optimization."

    Args:
        lr (float, optional): Learning rate. Defaults to 0.001.
        decay1 (float, optional): Decay rate for first moment (β1).
            Defaults to 0.9.
        decay2 (float, optional): Decay rate for second moment (β2).
            Defaults to 0.999.
        eps (float, optional): Smoothing term. Defaults to 1e-7.
        clip_norm (float, optional): Maximum L2 norm for gradients.
            Defaults to None.
        lr_scheduler (optional): Learning rate scheduler. Defaults to None.

    Examples:
        >>> # Default Adam
        >>> opt = Adam()
        >>>
        >>> # Custom parameters
        >>> opt = Adam(lr=0.001, decay1=0.9, decay2=0.999)
        >>>
        >>> # With gradient clipping
        >>> opt = Adam(lr=0.001, clip_norm=1.0)
    """

    def __init__(
        self,
        lr=0.001,
        decay1=0.9,
        decay2=0.999,
        eps=1e-7,
        clip_norm=None,
        lr_scheduler=None,
        **kwargs,
    ):
        """Initialize an Adam instance."""
        super().__init__(lr, lr_scheduler)

        self.cache = {}
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    def __str__(self):
        H = self.hyperparameters
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            H["lr"], H["decay1"], H["decay2"], H["eps"], H["clip_norm"], H["lr_scheduler"]
        )

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str, cur_loss: float = None) -> np.ndarray:
        """
        Compute the Adam update.

        Args:
            param (np.ndarray): Current parameter values.
            param_grad (np.ndarray): Parameter gradients.
            param_name (str): Parameter name for state tracking.
            cur_loss (float, optional): Current loss for scheduling.

        Returns:
            np.ndarray: Updated parameter values.
        """
        C = self.cache
        H = self.hyperparameters
        d1, d2 = H["decay1"], H["decay2"]
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param_grad),
                "var": np.zeros_like(param_grad),
            }

        # Gradient clipping
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        t = C[param_name]["t"] + 1
        var = C[param_name]["var"]
        mean = C[param_name]["mean"]

        # Update cache
        C[param_name]["t"] = t
        C[param_name]["var"] = d2 * var + (1 - d2) * param_grad**2
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param_grad
        self.cache = C

        # Unbiased moment estimates
        v_hat = C[param_name]["var"] / (1 - d2**t)
        m_hat = C[param_name]["mean"] / (1 - d1**t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - update
