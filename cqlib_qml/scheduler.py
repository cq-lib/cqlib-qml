# cqlib_qml/scheduler/scheduler.py
"""
Learning rate scheduling strategies for optimization.

This module provides implementations of various learning rate scheduling
strategies that can be used with optimizers. Schedulers dynamically
adjust the learning rate during training to improve convergence.

Available schedulers:
    1. ConstantScheduler: Fixed learning rate (no adjustment)
    2. ExponentialScheduler: Exponential decay with optional staircase
    3. NoamScheduler: Warmup followed by inverse square root decay
    4. KingScheduler: Automatic scheduling based on loss statistics

Examples:
    >>> from cqlib_qml.scheduler import ExponentialScheduler, NoamScheduler
    >>>
    >>> # Exponential decay
    >>> scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.9)
    >>> lr = scheduler(step=500)  # Returns 0.01 * 0.9^5
    >>>
    >>> # Noam scheduler (Transformer-style)
    >>> scheduler = NoamScheduler(model_dim=512, warmup_steps=4000)
    >>> lr = scheduler(step=1000)
"""

import re
from abc import ABC, abstractmethod
from ast import literal_eval as eval
from copy import deepcopy
from math import erf

import numpy as np


class SchedulerBase(ABC):
    """
    Abstract base class for all learning rate schedulers.

    This class defines the interface that all schedulers must implement.
    Schedulers adjust the learning rate based on the current step and/or
    loss value.

    Attributes:
        hyperparameters (dict): Scheduler hyperparameters.

    Note:
        User-defined schedulers should inherit from this class and
        implement the `learning_rate` method.

    Examples:
        >>> class MyScheduler(SchedulerBase):
        ...     def __init__(self, decay=0.9):
        ...         super().__init__()
        ...         self.decay = decay
        ...         self.hyperparameters = {"id": "MyScheduler", "decay": decay}
        ...
        ...     def learning_rate(self, step, **kwargs):
        ...         return self.hyperparameters["decay"] ** step
    """

    def __init__(self):
        """Initialize a SchedulerBase instance."""
        self.hyperparameters = {}

    def __call__(self, step: int = None, cur_loss: float = None) -> float:
        """
        Get the current learning rate.

        Args:
            step (int, optional): Current step number. Defaults to None.
            cur_loss (float, optional): Current loss value. Defaults to None.

        Returns:
            float: Current learning rate.
        """
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def copy(self):
        """
        Return a copy of the scheduler.

        Returns:
            SchedulerBase: Copy of the scheduler object.
        """
        return deepcopy(self)

    def set_params(self, hparam_dict: dict):
        """
        Set scheduler hyperparameters from a dictionary.

        Args:
            hparam_dict (dict): Hyperparameters dictionary.

        Returns:
            SchedulerBase: The scheduler instance.
        """
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
        return self

    @abstractmethod
    def learning_rate(self, step: int = None, **kwargs) -> float:
        """
        Compute the learning rate for the current step.

        Args:
            step (int): Current step number.
            **kwargs: Additional arguments (e.g., cur_loss).

        Returns:
            float: Learning rate for the current step.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError


class SchedulerInitializer:
    """
    Factory class for initializing learning rate schedulers.

    Provides a convenient way to create scheduler instances from strings,
    dictionaries, or existing scheduler instances.

    Args:
        param: Scheduler configuration. Can be:
            - None: Returns ConstantScheduler with lr
            - str: Scheduler name with optional parameters
            - dict: Scheduler configuration dictionary
            - SchedulerBase: Existing scheduler instance
        lr (float, optional): Learning rate for ConstantScheduler.

    Attributes:
        param: The scheduler configuration.
        lr (float): Learning rate for ConstantScheduler.

    Examples:
        >>> # From string
        >>> init = SchedulerInitializer("exponential(decay=0.9)")
        >>> scheduler = init()
        >>>
        >>> # From dict
        >>> init = SchedulerInitializer({
        ...     "hyperparameters": {"id": "ExponentialScheduler", "decay": 0.9}
        ... })
        >>> scheduler = init()
    """

    def __init__(self, param=None, lr=None):
        """
        Initialize a SchedulerInitializer instance.

        Args:
            param: Scheduler configuration.
            lr (float, optional): Learning rate. Defaults to None.
        """
        if lr is None and param is None:
            raise ValueError("lr and param cannot both be None")
        self.lr = lr
        self.param = param

    def __call__(self):
        """
        Create the scheduler instance.

        Returns:
            SchedulerBase: The scheduler instance.
        """
        param = self.param
        if param is None:
            scheduler = ConstantScheduler(self.lr)
        elif isinstance(param, SchedulerBase):
            scheduler = param
        elif isinstance(param, str):
            scheduler = self.init_from_str()
        elif isinstance(param, dict):
            scheduler = self.init_from_dict()
        return scheduler

    def init_from_str(self):
        """
        Initialize scheduler from a string.

        Supported formats:
            - "constant": Constant scheduler
            - "exponential(decay=0.9)": Exponential decay
            - "noam(model_dim=512, warmup_steps=4000)": Noam scheduler
            - "king(initial_lr=0.01, patience=100)": King scheduler

        Returns:
            SchedulerBase: The scheduler instance.

        Raises:
            ValueError: If scheduler name is not supported.
        """
        r = r"([a-zA-Z]*)=([^,)]*)"
        sch_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, sch_str)])

        if "constant" in sch_str:
            scheduler = ConstantScheduler(**kwargs)
        elif "exponential" in sch_str:
            scheduler = ExponentialScheduler(**kwargs)
        elif "noam" in sch_str:
            scheduler = NoamScheduler(**kwargs)
        elif "king" in sch_str:
            scheduler = KingScheduler(**kwargs)
        else:
            raise ValueError(
                f"Unsupported scheduler: {sch_str}. " f"Supported: ['constant', 'exponential', 'noam', 'king']"
            )
        return scheduler

    def init_from_dict(self):
        """
        Initialize scheduler from a dictionary.

        Args:
            param (dict): Configuration dictionary with 'hyperparameters' key.

        Returns:
            SchedulerBase: The scheduler instance.

        Raises:
            ValueError: If scheduler ID is not supported.
        """
        S = self.param
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key.")

        if sc and sc["id"] == "ConstantScheduler":
            scheduler = ConstantScheduler().set_params(sc)
        elif sc and sc["id"] == "ExponentialScheduler":
            scheduler = ExponentialScheduler().set_params(sc)
        elif sc and sc["id"] == "NoamScheduler":
            scheduler = NoamScheduler().set_params(sc)
        elif sc and sc["id"] == "KingScheduler":
            scheduler = KingScheduler().set_params(sc)
        elif sc:
            raise ValueError(
                f"Unsupported scheduler: {sc['id']}. "
                f"Supported: ['ConstantScheduler', 'ExponentialScheduler', "
                f"'NoamScheduler', 'KingScheduler']"
            )
        return scheduler


class ConstantScheduler(SchedulerBase):
    """
    Constant learning rate scheduler.

    Returns a fixed learning rate regardless of the step.

    Args:
        lr (float, optional): Learning rate. Defaults to 0.01.

    Examples:
        >>> scheduler = ConstantScheduler(lr=0.01)
        >>> lr = scheduler(step=0)  # 0.01
        >>> lr = scheduler(step=1000)  # 0.01
    """

    def __init__(self, lr=0.01, **kwargs):
        """Initialize a ConstantScheduler instance."""
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

    def __str__(self):
        return "ConstantScheduler(lr={})".format(self.lr)

    def learning_rate(self, **kwargs) -> float:
        """
        Return the current learning rate.

        Returns:
            float: The constant learning rate.
        """
        return self.lr


class ExponentialScheduler(SchedulerBase):
    """
    Exponential learning rate scheduler.

    learning_rate = initial_lr * decay^curr_stage

    Where:
        curr_stage = step / stage_length (if staircase=False)
        curr_stage = floor(step / stage_length) (if staircase=True)

    Args:
        initial_lr (float, optional): Initial learning rate. Defaults to 0.01.
        stage_length (int, optional): Length of each stage. Defaults to 500.
        staircase (bool, optional): Whether to use step-wise decay.
            Defaults to False.
        decay (float, optional): Decay factor per stage. Defaults to 0.1.

    Examples:
        >>> # Smooth decay
        >>> scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.9)
        >>>
        >>> # Step-wise decay
        >>> scheduler = ExponentialScheduler(
        ...     initial_lr=0.01, stage_length=100, staircase=True, decay=0.5
        ... )
    """

    def __init__(self, initial_lr=0.01, stage_length=500, staircase=False, decay=0.1, **kwargs):
        """Initialize an ExponentialScheduler instance."""
        super().__init__()
        self.decay = decay
        self.staircase = staircase
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.hyperparameters = {
            "id": "ExponentialScheduler",
            "decay": self.decay,
            "staircase": self.staircase,
            "initial_lr": self.initial_lr,
            "stage_length": self.stage_length,
        }

    def __str__(self):
        return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
            self.initial_lr, self.stage_length, self.staircase, self.decay
        )

    def learning_rate(self, step: int, **kwargs) -> float:
        """
        Compute the learning rate for the current step.

        Args:
            step (int): Current step number.

        Returns:
            float: Learning rate for the current step.
        """
        cur_stage = step / self.stage_length
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        return self.initial_lr * self.decay**cur_stage


class NoamScheduler(SchedulerBase):
    """
    Noam (Transformer) learning rate scheduler.

    lr = scale_factor * (model_dim^(-0.5)) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This scheduler increases the learning rate linearly during warmup and
    decreases it with the inverse square root afterwards.

    References:
        - Vaswani, A., et al. (2017). "Attention is all you need."

    Args:
        model_dim (int, optional): Model dimension. Defaults to 512.
        scale_factor (int, optional): Scaling factor. Defaults to 1.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 4000.

    Examples:
        >>> # Default Noam scheduler
        >>> scheduler = NoamScheduler()
        >>>
        >>> # Custom configuration
        >>> scheduler = NoamScheduler(
        ...     model_dim=256, scale_factor=2, warmup_steps=1000
        ... )
    """

    def __init__(self, model_dim=512, scale_factor=1, warmup_steps=4000, **kwargs):
        """Initialize a NoamScheduler instance."""
        super().__init__()
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.hyperparameters = {
            "id": "NoamScheduler",
            "model_dim": self.model_dim,
            "scale_factor": self.scale_factor,
            "warmup_steps": self.warmup_steps,
        }

    def __str__(self):
        return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
            self.model_dim, self.scale_factor, self.warmup_steps
        )

    def learning_rate(self, step: int, **kwargs) -> float:
        """
        Compute the learning rate for the current step.

        Args:
            step (int): Current step number.

        Returns:
            float: Learning rate for the current step.
        """
        if step <= 0:
            step = 1
        warmup, d_model = self.warmup_steps, self.model_dim
        new_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        return self.scale_factor * new_lr


class KingScheduler(SchedulerBase):
    """
    Davis King / DLib automatic learning rate scheduler.

    This scheduler automatically adjusts the learning rate based on
    the statistical analysis of the loss history. It computes the
    probability that the loss is decreasing and reduces the learning
    rate when the loss has not decreased for a certain number of steps.

    References:
        King, D. (2018). "Automatic learning rate scheduling that really works".
        <http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html>

    Args:
        initial_lr (float, optional): Initial learning rate. Defaults to 0.01.
        patience (int, optional): Number of steps to wait before reducing LR.
            Defaults to 1000.
        decay (float, optional): Decay factor for learning rate.
            Defaults to 0.99.

    Examples:
        >>> scheduler = KingScheduler(initial_lr=0.01, patience=500, decay=0.95)
        >>>
        >>> # During training
        >>> for step in range(num_steps):
        ...     loss = compute_loss()
        ...     lr = scheduler(step=step, cur_loss=loss)
    """

    def __init__(self, initial_lr=0.01, patience=1000, decay=0.99, **kwargs):
        """Initialize a KingScheduler instance."""
        super().__init__()
        self.decay = decay
        self.patience = patience
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.max_history = np.ceil(1.1 * (patience + 1)).astype(int)

        self.loss_history = []
        self.hyperparameters = {
            "id": "KingScheduler",
            "decay": self.decay,
            "patience": self.patience,
            "initial_lr": self.initial_lr,
        }

    def __str__(self):
        return "KingScheduler(initial_lr={}, patience={}, decay={})".format(self.initial_lr, self.patience, self.decay)

    def _steps_without_decrease(self, robust=False, check_all=False) -> int:
        """
        Compute the number of steps without decrease.

        Args:
            robust (bool, optional): Filter out top 10% of loss values.
                Defaults to False.
            check_all (bool, optional): Check all steps. Defaults to False.

        Returns:
            int: Number of steps without decrease.
        """
        lh = np.array(self.loss_history)

        # Drop top 10% of loss values to filter out spikes
        if robust:
            thresh = np.quantile(lh, 0.9)
            lh = np.array([i for i in lh if i <= thresh])

        N = len(lh)
        steps_without_decrease = 0
        if check_all:
            for i in reversed(range(N - 2)):
                if self._p_decreasing(lh, i) < 0.51:
                    steps_without_decrease = N - i
        else:
            i = max(0, N - self.patience - 1)
            if self._p_decreasing(lh, i) < 0.51:
                steps_without_decrease = N - i
        return steps_without_decrease

    def _gaussian_cdf(self, x: float, mean: float, var: float) -> float:
        """
        Compute the Gaussian cumulative distribution function.

        Args:
            x (float): Value to evaluate at.
            mean (float): Mean of the distribution.
            var (float): Variance of the distribution.

        Returns:
            float: CDF value at x.
        """
        eps = np.finfo(float).eps
        x_scaled = (x - mean) / np.sqrt(var + eps)
        return (1 + erf(x_scaled / np.sqrt(2))) / 2

    def _p_decreasing(self, loss_history: np.ndarray, i: int) -> float:
        """
        Compute the probability that the loss is decreasing.

        Args:
            loss_history (np.ndarray): Loss history array.
            i (int): Starting index.

        Returns:
            float: Probability that slope is negative.
        """
        loss = loss_history[i:]
        N = len(loss)

        # Perform OLS to compute slope mean
        X = np.c_[np.ones(N), np.arange(i, len(loss_history))]
        intercept, s_mean = np.linalg.inv(X.T @ X) @ X.T @ loss
        loss_pred = s_mean * X[:, 1] + intercept

        # Compute slope variance
        loss_var = 1 / (N - 2) * np.sum((loss - loss_pred) ** 2)
        s_var = (12 * loss_var) / (N**3 - N)

        # Probability that slope is negative
        p_decreasing = self._gaussian_cdf(0, s_mean, s_var)
        return p_decreasing

    def learning_rate(self, step: int, cur_loss: float) -> float:
        """
        Compute the learning rate for the current step.

        Args:
            step (int): Current step number.
            cur_loss (float): Current loss value.

        Returns:
            float: Learning rate for the current step.

        Raises:
            AssertionError: If cur_loss is None.
        """
        if cur_loss is None:
            raise ValueError(f"cur_loss must be a float, but got {cur_loss}")

        # Initialize history tracking
        if not hasattr(self, "max_history"):
            self.max_history = np.ceil(1.1 * (self.patience + 1)).astype(int)
        patience, max_history = self.patience, self.max_history

        self.loss_history.append(cur_loss)
        if len(self.loss_history) < patience:
            return self.current_lr
        self.loss_history = self.loss_history[-max_history:]

        # Reduce learning rate if loss hasn't decreased
        if self._steps_without_decrease() > patience and self._steps_without_decrease(robust=True) > patience:
            self.current_lr *= self.decay

        return self.current_lr
