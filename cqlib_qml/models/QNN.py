# cqlib_qml/models/QNN.py
"""
Pure Quantum Neural Network (QNN) model.

This module provides a pure quantum neural network that uses only a
parameterized quantum circuit (ansatz) for computation. The output is
obtained directly from measurements on the quantum circuit.

The QNN model is suitable for:
    - Binary classification
    - Multi-class classification (with multiple readouts)
    - Regression tasks

Examples:
    >>> from cqlib_qml.models import QNN
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.encoder import FRQI
    >>>
    >>> # Create a QNN for binary classification
    >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    >>> ansatz.set_measurement(readouts=[0])
    >>> qnn = QNN(ansatz=ansatz, optimizer="adam")
    >>>
    >>> # Encode data and forward pass
    >>> encoder = FRQI(n_pixels=16)
    >>> data_circuits = encoder(image_data)
    >>> output = qnn.forward(data_circuits)  # Shape: (batch_size, 1)
"""

from typing import List, Union

import numpy as np

from cqlib.circuit import Circuit
from cqlib_qml.ansatz import *
from cqlib_qml.encoder import *
from cqlib_qml.models import Module
from cqlib_qml.loss import *
from cqlib_qml.optimizer import *


class QNN(Module):
    """
    Pure Quantum Neural Network model.

    This model uses a parameterized quantum circuit (ansatz) as the
    sole computational component. The output is obtained from measurements
    on specified readout qubits.

    The QNN inherits from Module and provides a simplified interface
    for quantum-only models.

    Args:
        ansatz (Ansatz): The quantum circuit ansatz to use.
        readouts (list, optional): Qubit indices to measure.
            If None, uses the ansatz's existing readouts.
            Defaults to None.
        params (np.ndarray, optional): Initial parameter values.
            Defaults to None.
        optimizer (Union[str, dict, OptimizerBase], optional): The optimizer
            for training. Defaults to "adam".

    Attributes:
        _ansatz (Ansatz): The quantum circuit ansatz.

    Raises:
        ValueError: If readouts are not provided and ansatz has no readouts.

    Examples:
        >>> # Binary classification with single readout
        >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
        >>> ansatz.set_measurement(readouts=[0])
        >>> qnn = QNN(ansatz=ansatz)
        >>>
        >>> # Multi-class classification with multiple readouts
        >>> ansatz = HEAnsatz(n_qubits=5, d=2, layers=["RY", "CX"])
        >>> qnn = QNN(ansatz=ansatz, readouts=[0, 1, 2])
        >>>
        >>> # With custom optimizer
        >>> qnn = QNN(ansatz=ansatz, optimizer="sgd(lr=0.01)")
    """

    def __init__(
        self,
        ansatz: Ansatz,
        readouts: list = None,
        params: np.ndarray = None,
        optimizer: Union[str, dict, OptimizerBase] = "adam",
    ):
        """
        Initialize a QNN model.

        Args:
            ansatz (Ansatz): The quantum circuit ansatz.
            readouts (list, optional): Readout qubits. Defaults to None.
            params (np.ndarray, optional): Initial parameters. Defaults to None.
            optimizer (Union[str, dict, OptimizerBase]): Optimizer. Defaults to "adam".

        Raises:
            ValueError: If readouts are not available.
        """
        self._ansatz = ansatz
        if params is not None:
            params = validate_initial_params(ansatz, params)
            bindings = dict(zip(ansatz.symbols, params))
            self._ansatz.assign_parameters(bindings)
        readouts = readouts if readouts is not None else self._ansatz.readouts
        if readouts is None:
            raise ValueError("Must provide readouts.")
        self._ansatz.set_measurement(readouts=readouts)
        super(QNN, self).__init__(self._ansatz)
        self.set_optimizer(optimizer)

    def forward(self, data_circuits: Union[Circuit, List[Circuit]], trainable: bool = True):
        """
        Perform forward propagation.

        This method encodes the data into quantum circuits and computes
        the expectation values from the ansatz.

        Args:
            data_circuits (Union[Circuit, List[Circuit]]): Data circuits
                after encoding. Can be a single Circuit or a list.
            trainable (bool, optional): Whether to enable training mode.
                If True, retains training state while respecting freeze().
                If False, clears old gradients and backward caches before
                validating inputs, even if the call fails. Freeze status is
                unchanged. Defaults to True.

        Returns:
            Union[float, np.ndarray]: Expectation values from the quantum
                circuit measurements.

        Examples:
            >>> # Training mode
            >>> output = qnn.forward(data_circuits, trainable=True)
            >>>
            >>> # Inference mode
            >>> output = qnn.forward(data_circuits, trainable=False)
        """
        if not trainable:
            self._invalidate_gradients()
        self._ansatz.add_encoder(data_circuits)
        expections = super().forward(retain_derived=trainable)
        return expections


def validate_initial_params(ansatz, params):
    """Validate the complete real parameter vector before changing an ansatz."""
    values = np.asarray(params)
    if (values.ndim != 1 or len(values) != len(ansatz.symbols)
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("Initial params must be a finite real vector matching ansatz.symbols")
    return values
