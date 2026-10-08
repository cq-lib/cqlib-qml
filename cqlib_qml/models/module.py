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
from copy import deepcopy
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

    def __init__(self, *args: Union[Layer, Ansatz], random_state=None):
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
        from cqlib_qml._state import make_rng
        self._rng = make_rng(random_state)
        self.training = True
        self._forward_valid = False
        self._nets = self._validate_nets(list(args))
        if random_state is not None:
            from cqlib_qml._state import make_rng
            for net in self._nets:
                net._rng = make_rng(int(self._rng.integers(0, 2**63)))

    def forward(self, x=None, *, retain_derived=True):
        """
        Perform forward propagation through the module.

        This method passes the input through each component in sequence,
        where the output of one component becomes the input to the next.

        Args:
            x (np.ndarray, optional): Input data. Defaults to None.
            retain_derived (bool): Keep the last forward cache. False discards
                backward caches, preserving cumulative gradients and freeze status.

        Returns:
            np.ndarray: Output of the last component in the module.

        Examples:
            >>> # With input data
            >>> output = model.forward(X)
            >>>
            >>> # For ansatz-only models (no input needed)
            >>> output = model.forward()
        """
        if not retain_derived:
            self._invalidate_gradients()
        else:
            self._forward_valid = False
        x_in = x
        for net in self._nets:
            if retain_derived:
                x_in = net.forward(x_in)
            else:
                from inspect import signature
                if "retain_derived" in signature(net.forward).parameters:
                    x_in = net.forward(x_in, retain_derived=False)
                else:
                    x_in = net.forward(x_in)
                net._invalidate_gradients()
        self._forward_valid = retain_derived
        return x_in

    def _invalidate_gradients(self):
        """Discard all forward caches, preserving accumulated parameter gradients."""
        self._forward_valid = False
        for net in self._nets:
            net._invalidate_gradients()

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
        if not getattr(self, "_forward_valid", False):
            raise ValueError("Run a training forward before backward")
        for net in self._nets[::-1]:
            if not isinstance(net, Ansatz) and dLdout is None:
                raise ValueError("Classical layers must pass in gradients.")
            dLdout = net.backward(dLdout)
            if isinstance(net, Layer):
                net._inference_invalidated = False
        return dLdout

    def random_init(self) -> None:
        """
        Randomly initialize all parameters in the module.

        This method initializes parameters randomly for both layers and
        ansätze. For ansätze, parameters are drawn from a standard normal
        distribution. For layers, the default initialization method is used.
        Accumulated gradients and forward caches are discarded.

        Examples:
            >>> model.random_init()
        """
        for net in self._nets:
            if isinstance(net, Ansatz):
                net.assign_weights(self._rng.normal(size=net.num_weights))
            else:
                net._rng = deepcopy(self._rng)
                net.init_params()
                self._rng = deepcopy(net._rng)
        self.zero_grad()
        self._invalidate_gradients()

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
        Reset accumulated gradients for every component, preserving forward caches.

        Examples:
            >>> model.zero_grad()
        """
        for net in self._nets:
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
            if net.trainable:
                net.update(cur_loss)
        self._forward_valid = False

    def train(self, mode=True):
        self.training = bool(mode)
        for net in self._nets:
            net.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save_checkpoint(self, model_path: str, ep: int, it: int, latest: bool = False, *, data_loader=None) -> None:
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
            data_loader (DataLoader, optional): Project loader whose permutation,
                next-batch cursor and random state are saved. Defaults to None.

        Raises:
            ValueError: If parameter gradients have not been cleared with zero_grad.

        Examples:
            >>> model.save_checkpoint("./checkpoints", ep=10, it=100)
            >>> model.save_checkpoint("./checkpoints", ep=10, it=100, latest=True)
        """
        from cqlib_qml._state import FORMAT_VERSION, atomic_save, require_clean_gradients
        require_clean_gradients(self._nets)
        checkpoint = {"format_version": FORMAT_VERSION, "training": self.training,
                      "rng_state": deepcopy(self._rng.bit_generator.state),
                      "data_loader": data_loader.state_dict() if data_loader is not None else None}
        module_info = dict()
        for i in range(len(self._nets)):
            if isinstance(self._nets[i], Ansatz):
                key = "ansatz{}".format(i)
            else:
                key = "layer{}".format(i)
            summary = self._nets[i].summary
            # Layer.summary is a method, Ansatz.summary is a property.
            module_info[key] = summary() if callable(summary) else summary
        checkpoint["epoch"] = ep
        checkpoint["iter"] = it
        checkpoint["module"] = module_info

        os.makedirs(model_path, exist_ok=True)
        atomic_save("{}/model.npy".format(model_path), checkpoint)
        if not latest:
            atomic_save("{0}/{1}_{2}.npy".format(model_path, ep, it), checkpoint)

    def load_checkpoint(self, model_path: str, *, data_loader=None) -> tuple:
        """
        Load a model checkpoint from disk.

        Args:
            model_path (str): Path to the checkpoint file or directory.
            data_loader (DataLoader, optional): Restore iterator state while
                preserving its dataset and underlying data references.

        Returns:
            tuple: Next (epoch, iteration). Returns (saved epoch, saved iteration
                + 1), or (saved epoch + 1, 0) when the saved loader finished its epoch.

        Raises:
            ValueError: If the checkpoint cannot be read or its model or loader
                state is incompatible. Validation failures preserve live state.

        Examples:
            >>> ep, it = model.load_checkpoint("./checkpoints/model.npy")
            >>> ep, it = model.load_checkpoint("./checkpoints")
        """

        def find_fname(model_path):
            import re
            candidates = []
            for name in os.listdir(model_path):
                match = re.fullmatch(r"(\d+)_(\d+)\.npy", name)
                if match:
                    candidates.append((int(match[1]), int(match[2]), name))
            if not candidates:
                raise ValueError("No numbered checkpoints found.")
            return max(candidates)[2]

        model_path = os.fspath(model_path)
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
            from cqlib_qml._state import clone_state, validate_version, rng_from_state
            data = checkpoint.item()
            validate_version(data)
            if type(data.get('training')) is not bool or 'rng_state' not in data or 'data_loader' not in data:
                raise ValueError("Incomplete Module checkpoint")
            for value in data['module'].values():
                validate_version(value)
            ep, it = data["epoch"], data["iter"]
            if any(type(value) is not int or value < 0 for value in (ep, it)):
                raise ValueError("Invalid checkpoint progress")
            if len(data["module"]) != len(self._nets):
                raise ValueError("Checkpoint component count does not match model.")
            expected_keys = [f"{'ansatz' if isinstance(net, Ansatz) else 'layer'}{index}"
                             for index, net in enumerate(self._nets)]
            if list(data['module']) != expected_keys:
                raise ValueError("Checkpoint component order or type mismatch")
            values = list(data["module"].values())
            candidates = []
            for val, net in zip(values, self._nets):
                candidate = clone_state(net)
                candidate.load_params(clone_state(val))
                candidates.append(candidate)
            rng = rng_from_state(data["rng_state"])
            loader_state = data['data_loader']
            resume_position = (ep, it + 1)
            if loader_state is not None:
                from cqlib_qml._state import validate_loader_state
                batches = validate_loader_state(loader_state)
                if loader_state['next_batch'] == batches:
                    resume_position = (ep + 1, 0)
            loader_candidate = None
            if data_loader is not None and data.get("data_loader") is not None:
                # The dataset is caller-owned and only read during validation.
                # Stage iterator state without copying or replacing shared data.
                loader_candidate = deepcopy(data_loader, {id(data_loader.dataset): data_loader.dataset})
                loader_candidate.load_state_dict(data["data_loader"])
            for candidate, net in zip(candidates, self._nets):
                net.__dict__.clear()
                net.__dict__.update(candidate.__dict__)
            self._rng = rng
            self.training = data["training"]
            self._forward_valid = False
            self._resume_data_loader_state = deepcopy(data["data_loader"])
            self._checkpoint_complete = data["data_loader"] is not None
            if loader_candidate is not None:
                for name in ('_idx', '_it', '_rng', '_resume_pending', '_iteration_started'):
                    data_loader.__dict__[name] = loader_candidate.__dict__[name]
        except Exception as e:
            raise ValueError(f"Mismatched model. Error: {e}")

        print(f"Successfully restored checkpoint at ep: {ep} it: {it}")
        return resume_position

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
