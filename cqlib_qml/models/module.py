# cqlib_qml/models/module.py
"""
Base module for composing quantum and classical components.

This module provides the Module class that serves as a container for
combining multiple layers and ansätze into a single model. It supports
various composition patterns and provides unified forward/backward
propagation.

Supported Composition Patterns:
    1. linear + linear: Classic feedforward network
    2. ansatz + linear: Quantum feature extraction + classical classification
    3. linear + ansatz: Classical preprocessing + quantum processing
    4. ansatz + ansatz: Cascaded quantum circuits

Examples:
    >>> from cqlib_qml.models import Module
    >>> from cqlib_qml.layer import Linear
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>>
    >>> # Hybrid model: ansatz + linear
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> ansatz.set_measurement(readouts=[0, 1, 2])
    >>> linear = Linear(in_dim=3, out_dim=2)
    >>> model = Module(ansatz, linear)
    >>>
    >>> # Pure classical model
    >>> model = Module(Linear(10, 5), Linear(5, 2))
"""

import os
import shutil
from typing import Union, Optional

import numpy as np

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.layer import Layer
from cqlib_qml.optimizer import *


class Module:
    """
    Container for composing quantum and classical components.

    The Module class allows flexible composition of layers and ansätze
    into a single model. It handles forward and backward propagation
    through the composed components, parameter management, and
    checkpointing.

    Supported composition patterns:
        1. linear + linear: Outputs of first linear layer are inputs to second
        2. ansatz + linear: Outputs of ansatz are inputs to linear layer
        3. linear + ansatz: Outputs of linear layer are parameters of ansatz
        4. ansatz + ansatz: Outputs of first ansatz are parameters of second

    Args:
        *args (Union[Layer, Ansatz]): Variable number of layers and/or
            ansätze to compose.

    Attributes:
        _nets (list): List of composed components.

    Raises:
        TypeError: If any argument is not a Layer or Ansatz.
        ValueError: If an Ansatz doesn't have measurements.

    Examples:
        >>> # Hybrid model
        >>> model = Module(ansatz, linear)
        >>>
        >>> # Classical model
        >>> model = Module(layer1, layer2)
        >>>
        >>> # Quantum model
        >>> model = Module(ansatz1, ansatz2)
    """

    def __init__(self, *args: Union[Layer, Ansatz]):
        """
        Initialize a Module instance.

        Args:
            *args (Union[Layer, Ansatz]): Components to compose.

        Raises:
            TypeError: If any argument is not a Layer or Ansatz.
            ValueError: If an Ansatz doesn't have measurements.

        Examples:
            >>> model = Module(ansatz, linear)
            >>> model = Module(linear1, linear2)
        """
        self._nets = self._validate_nets(list(args))

    def forward(self, x=None):
        """
        Perform forward propagation through the module.

        This method passes the input through each component in sequence,
        where the output of one component becomes the input to the next.

        Args:
            x (np.ndarray, optional): Input data. Defaults to None.

        Returns:
            np.ndarray: Output of the last component in the module.

        Examples:
            >>> # With input data
            >>> output = model.forward(X)
            >>>
            >>> # For ansatz-only models (no input needed)
            >>> output = model.forward()
        """
        x_in = x
        for net in self._nets:
            x_in = net.forward(x_in)
        return x_in

    def backward(self, dLdout=None):
        """
        Perform backward propagation through the module.

        This method propagates gradients backwards through the module,
        from the last component to the first.

        Args:
            dLdout (Union[float, np.ndarray], optional): Gradients of the
                loss with respect to the module's output. Required if the
                last component is a classical layer. Defaults to None.

        Returns:
            np.ndarray: Gradients of the loss with respect to the module's
                input.

        Raises:
            ValueError: If a classical layer is reached without gradients.

        Examples:
            >>> # Backward pass from loss
            >>> dLdout = loss_fn.grads()
            >>> grad = model.backward(dLdout)
        """
        for net in self._nets[::-1]:
            if isinstance(net, Ansatz) and not net.trainable and net.updatable:
                continue  # Frozen quantum source has no classical input gradient.
            if not isinstance(net, Ansatz) and dLdout is None:
                raise ValueError("Classical layers must pass in gradients.")
            dLdout = net.backward(dLdout)
        return dLdout

    def random_init(self) -> None:
        """
        Randomly initialize all parameters in the module.

        This method initializes parameters randomly for both layers and
        ansätze. For ansätze, parameters are drawn from a standard normal
        distribution. For layers, the default initialization method is used.

        Examples:
            >>> model.random_init()
        """
        for net in self._nets:
            if isinstance(net, Ansatz):
                keys = net.symbols
                values = np.random.randn(len(net.symbols))
                bindings = dict(zip(keys, values))
                net.assign_parameters(bindings)
            else:
                net.init_params()

    def set_optimizer(self, optimizer: Union[str, dict, OptimizerBase] = "adam") -> None:
        """
        Set the optimizer for all trainable components in the module.

        Args:
            optimizer (Union[str, dict, OptimizerBase]): The optimizer to use.
                - str: Name of optimizer ("adam", "sgd", "adagrad", "rmsprop")
                - dict: Optimizer configuration dictionary
                - OptimizerBase: Optimizer instance
                Defaults to "adam".

        Examples:
            >>> model.set_optimizer("adam")
            >>> model.set_optimizer("sgd(lr=0.01)")
            >>> model.set_optimizer(Adam(lr=0.001))
        """
        for net in self._nets:
            if net.updatable:
                net.set_optimizer(optimizer)

    def zero_grad(self) -> None:
        """
        Reset gradients to zero for all trainable components.

        Examples:
            >>> model.zero_grad()
        """
        for net in self._nets:
            if net.trainable:
                net.zero_grad()

    def update(self, cur_loss: Optional[float] = None) -> None:
        """
        Update all trainable components in the module.

        Args:
            cur_loss (float, optional): Current loss value for learning rate
                scheduling. Defaults to None.

        Examples:
            >>> model.update()
            >>> model.update(cur_loss=0.5)
        """
        for net in self._nets:
            if net.trainable and net.updatable:
                net.update(cur_loss)

    def save_checkpoint(self, model_path: str, ep: int, it: int, latest: bool = False) -> None:
        """
        Save the model state as a checkpoint.

        This method saves the current state of all components, including
        parameters and optimizer states, to disk.

        Args:
            model_path (str): Directory path for saving the checkpoint.
            ep (int): Current epoch number.
            it (int): Current iteration number.
            latest (bool, optional): Whether this is the latest checkpoint.
                If True, saves as model.npy. Defaults to False.

        Examples:
            >>> model.save_checkpoint("./checkpoints", ep=10, it=100)
            >>> model.save_checkpoint("./checkpoints", ep=10, it=100, latest=True)
        """
        checkpoint = dict()
        module_info = dict()
        for i in range(len(self._nets)):
            if isinstance(self._nets[i], Ansatz):
                key = "ansatz{}".format(i)
            else:
                key = "layer{}".format(i)
            module_info[key] = self._nets[i].summary
        checkpoint["epoch"] = ep
        checkpoint["iter"] = it
        checkpoint["module"] = module_info

        os.makedirs(model_path, exist_ok=True)
        np.save("{}/model.npy".format(model_path), checkpoint)
        if not latest:
            shutil.copy(
                "{0}/model.npy".format(model_path),
                "{0}/{1}_{2}.npy".format(model_path, ep, it),
            )

    def load_checkpoint(self, model_path: str) -> tuple:
        """
        Load a model checkpoint from disk.

        Args:
            model_path (str): Path to the checkpoint file or directory.

        Returns:
            tuple: (epoch, iteration) where:
                - epoch (int): Restored epoch number
                - iteration (int): Restored iteration number + 1

        Raises:
            ValueError: If the checkpoint file cannot be found or loaded.

        Examples:
            >>> ep, it = model.load_checkpoint("./checkpoints/model.npy")
            >>> ep, it = model.load_checkpoint("./checkpoints")
        """

        def find_fname(model_path):
            f_list = sorted(os.listdir(model_path))
            ep = 0
            it = 0
            for f in f_list:
                if f.endswith(".npy"):
                    nums = f[:-4].split("_")
                    if len(nums) >= 2:
                        ep = max(ep, int(nums[0]))
                        it = max(it, int(nums[1]))
            fname = str(ep) + "_" + str(it) + ".npy"
            return fname

        try:
            if model_path.endswith(".npy"):
                fname = model_path
                checkpoint = np.load(fname, allow_pickle=True)
            else:
                if os.path.exists("{0}/model.npy".format(model_path)):
                    fname = "model.npy"
                else:
                    fname = find_fname(model_path)
                checkpoint = np.load("{0}/{1}".format(model_path, fname), allow_pickle=True)
        except Exception as e:
            raise ValueError(f"Invalid model path: {model_path}. Error: {e}")

        try:
            for val, net in zip(checkpoint.item()["module"].values(), self._nets):
                net.load_params(val)
        except Exception as e:
            raise ValueError(f"Mismatched model. Error: {e}")

        ep = checkpoint.item()["epoch"]
        it = checkpoint.item()["iter"]
        print(f"Successfully restored checkpoint at ep: {ep} it: {it}")
        return ep, it + 1

    def freeze(self) -> None:
        """
        Freeze all components in the module (disable training).

        Examples:
            >>> model.freeze()
        """
        for net in self._nets:
            net.freeze()

    def unfreeze(self) -> None:
        """
        Unfreeze all components in the module (enable training).

        Examples:
            >>> model.unfreeze()
        """
        for net in self._nets:
            net.unfreeze()

    def _validate_nets(self, net_list: list) -> list:
        """
        Validate the components passed to the module.

        Args:
            net_list (list): List of components to validate.

        Returns:
            list: Validated component list.

        Raises:
            TypeError: If any component is not a Layer or Ansatz.
            ValueError: If an Ansatz doesn't have measurements.

        Examples:
            >>> nets = self._validate_nets([ansatz, linear])
        """
        for net in net_list:
            if not isinstance(net, (Layer, Ansatz)):
                raise TypeError(f"Expected Layer or Ansatz, got {type(net).__name__}")
            if isinstance(net, Ansatz):
                if net.out_dim <= 0:
                    raise ValueError("Ansatz must have measurements.")
        return net_list
