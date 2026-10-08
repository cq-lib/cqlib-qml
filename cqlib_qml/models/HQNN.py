# cqlib_qml/models/HQNN.py
"""
Hybrid Quantum-Classical Neural Network (HQNN) model.

This module provides a hybrid quantum-classical neural network that combines
a quantum circuit (ansatz) with a classical fully-connected layer.

The HQNN architecture:
    1. Quantum circuit: Encodes data and processes it through an ansatz
    2. Classical layer: Transforms quantum measurements to final outputs

This architecture is suitable for:
    - Multi-class classification
    - Complex pattern recognition
    - Tasks requiring multiple output dimensions

References:
    - Farhi, E., & Neven, H. (2018). "Classification with quantum neural
      networks on near term processors." arXiv:1802.06002.

Examples:
    >>> from cqlib_qml.models import HQNN
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.encoder import FRQI
    >>>
    >>> # Create HQNN for 3-class classification
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> hqnn = HQNN(ansatz=ansatz, out_dim=3, optimizer="adam")
    >>>
    >>> # Encode data and forward pass
    >>> encoder = FRQI(n_pixels=16)
    >>> data_circuits = encoder(image_data)
    >>> output = hqnn.forward(data_circuits)  # Shape: (batch_size, 3)
"""

from typing import List, Union

import numpy as np

from cqlib.circuit import Circuit
from cqlib_qml.ansatz import *
from cqlib_qml.encoder import *
from cqlib_qml.layer import *
from cqlib_qml.models import Module
from .QNN import validate_initial_params
from cqlib_qml.loss import *
from cqlib_qml.optimizer import *


class HQNN(Module):
    """
    Hybrid Quantum-Classical Neural Network.

    Combines a quantum circuit (ansatz) with a classical linear layer.
    The quantum circuit processes the input data and produces expectation
    values, which are then fed into a classical fully-connected layer
    for final classification or regression.

    Architecture:
        1. Quantum ansatz: Processes encoded data and produces measurements
        2. Classical linear layer: Maps quantum measurements to outputs

    Args:
        ansatz (HEAnsatz): The quantum circuit ansatz. Must be an
            HEAnsatz instance.
        out_dim (int): Number of output dimensions.
        readouts (list, optional): Qubit indices to measure.
            If None, uses the ansatz's existing readouts, or all qubits
            if the ansatz has none. Defaults to None.
        params (np.ndarray, optional): Initial parameters for the ansatz.
            Defaults to None.
        optimizer (Union[str, dict, OptimizerBase], optional): The optimizer
            for training. Defaults to "adam".
        random_state (int/Generator/None): Independent initialization RNG.

    Attributes:
        _ansatz (HEAnsatz): The quantum circuit ansatz.
        _linear (Linear): The classical output layer.

    Raises:
        TypeError: If ansatz is not an HEAnsatz.

    Examples:
        >>> # Create HQNN with 4 qubits and 3 outputs
        >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
        >>> hqnn = HQNN(ansatz=ansatz, out_dim=3)
        >>>
        >>> # With custom optimizer
        >>> hqnn = HQNN(
        ...     ansatz=ansatz,
        ...     out_dim=2,
        ...     optimizer="sgd(lr=0.01, momentum=0.9)"
        ... )
        >>>
        >>> # With initial parameters
        >>> params = np.random.randn(len(ansatz.symbols))
        >>> hqnn = HQNN(ansatz=ansatz, out_dim=3, params=params)
    """

    def __init__(
        self,
        ansatz: HEAnsatz,
        out_dim: int,
        params: np.ndarray = None,
        optimizer: Union[str, dict, OptimizerBase] = "adam",
        *,
        readouts: list = None,
        random_state=None,
    ):
        """
        Initialize an HQNN model.

        Args:
            ansatz (HEAnsatz): The quantum circuit ansatz.
            out_dim (int): Number of output dimensions.
            readouts (list, optional): Readout qubits. Defaults to None.
            params (np.ndarray, optional): Initial parameters. Defaults to None.
            optimizer (Union[str, dict, OptimizerBase]): Optimizer. Defaults to "adam".
            random_state (int/Generator/None): Independent initialization RNG.

        Raises:
            TypeError: If ansatz is not an HEAnsatz.
        """
        if not isinstance(ansatz, HEAnsatz):
            raise TypeError(f"Expected HEAnsatz, got {type(ansatz).__name__}")
        self._ansatz = ansatz
        n_qubits = self._ansatz.num_qubits
        if params is not None:
            params = validate_initial_params(ansatz, params)
            bindings = dict(zip(ansatz.symbols, params))
            self._ansatz.assign_parameters(bindings)
        if readouts is None:
            readouts = self._ansatz.readouts if self._ansatz.readouts is not None else list(range(n_qubits))
        self._ansatz.set_measurement(readouts=readouts)
        self._linear = Linear(len(readouts), out_dim)
        super(HQNN, self).__init__(self._ansatz, self._linear, random_state=random_state)
        self.set_optimizer(optimizer)

    def forward(self, data_circuits: Union[Circuit, List[Circuit]], trainable: bool = True):
        """
        Perform forward propagation.

        This method encodes the data into quantum circuits, computes
        expectation values from the ansatz, and passes them through
        the classical linear layer.

        Args:
            data_circuits (Union[Circuit, List[Circuit]]): Data circuits
                after encoding. Can be a single Circuit or a list.
            trainable (bool, optional): Whether to enable training mode.
                If True, retains training state while respecting freeze().
                If False, clears old gradients and backward caches before
                validating inputs, even if the call fails. Freeze status is
                unchanged. Defaults to True.

        Returns:
            np.ndarray: Output from the classical linear layer.
                Shape: (batch_size, out_dim)

        Examples:
            >>> # Training mode
            >>> output = hqnn.forward(data_circuits, trainable=True)
            >>>
            >>> # Inference mode
            >>> output = hqnn.forward(data_circuits, trainable=False)
            >>>
            >>> # Get predictions (for classification)
            >>> predictions = np.argmax(output, axis=1)
        """
        if not trainable:
            self._invalidate_gradients()
        self._ansatz.add_encoder(data_circuits)
        expections = super().forward(retain_derived=trainable)
        return expections
