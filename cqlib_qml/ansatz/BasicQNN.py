# cqlib_qml/ansatz/BasicQNN.py
"""
Basic QNN ansatz implementation.

This module provides the BasicQNN ansatz architecture, which consists of
layers of two-qubit gates between data qubits and a readout qubit.

The BasicQNN is designed for quantum neural networks where each data qubit
interacts with the readout qubit through parameterized two-qubit gates.

References:
    `Classification with Quantum Neural Networks on Near Term Processors`
    <https://arxiv.org/abs/1802.06002>

Examples:
    >>> from cqlib_qml.ansatz import BasicQNN
    >>>
    >>> # Create a BasicQNN with 4 qubits and two layers
    >>> ansatz = BasicQNN(n_qubits=4, layers=["XX", "ZZ"])
    >>>
    >>> # Set measurement on the last qubit (readout)
    >>> ansatz.set_measurement(readouts=[3])
    >>>
    >>> # Forward pass with random initialization
    >>> result = ansatz.forward()
"""

from cqlib.circuit import Parameter
from .ansatz import Ansatz


class BasicQNN(Ansatz):
    """
    Basic QNN ansatz with alternating layers of two-qubit gates.

    Each layer applies the same type of two-qubit gate between each data
    qubit and the readout qubit. All parametric gates have different
    trainable parameters.

    The architecture consists of:
        1. Hadamard gates on all qubits
        2. X and H gates on the readout qubit
        3. Alternating layers of two-qubit gates (XX, YY, ZZ, or ZX)
        4. H gate on the readout qubit

    Note:
        By default, the last qubit is the readout qubit, and others are data qubits.

    Args:
        n_qubits (int): The number of qubits. Must be at least 2.
        layers (list): List of gate types for each layer.
            Supported gates: "XX", "YY", "ZZ", "ZX".

    Attributes:
        num_qubits (int): Number of qubits.
        in_dim (int): Number of trainable parameters.
        _layers (list): List of gate types used in each layer.

    Raises:
        ValueError: If n_qubits < 2 or if unsupported gate types are specified.

    Examples:
        >>> # Create BasicQNN with 3 qubits and XX-ZZ layers
        >>> ansatz = BasicQNN(n_qubits=3, layers=["XX", "ZZ"])
        >>> print(ansatz.num_qubits)
        3
        >>> print(ansatz.in_dim)  # (3-1) * 2 = 4 parameters
        4
    """

    def __init__(self, n_qubits: int, layers: list):
        """
        Initialize a BasicQNN ansatz.

        Args:
            n_qubits (int): Number of qubits (must be >= 2).
            layers (list): List of gate types for each layer.
                Supported: "XX", "YY", "ZZ", "ZX".

        Raises:
            ValueError: If n_qubits < 2 or unsupported gate type.
        """
        if n_qubits < 2:
            raise ValueError("n_qubits should be >= 2.")
        self._validate_layers(layers)
        self._layers = layers
        parameters = self._init_parameters(n_qubits)
        super(BasicQNN, self).__init__(n_qubits)
        self._construct_circuit(parameters)
        self.set_measurement(readouts=n_qubits - 1)

    def __str__(self) -> str:
        """Return a string representation of the ansatz."""
        return (
            f"Ansatz       | BasicQNN(n_qubits={self.num_qubits}, layers={self._layers}) \n"
            + f"n_params     | {self.in_dim} \n"
            + f"readouts     | {[self.num_qubits - 1]} \n"
            + f"optimizer    | {str(self._optimizer)} \n"
        )

    def _init_parameters(self, n_qubits: int) -> list:
        """
        Initialize trainable parameters for the ansatz.

        Args:
            n_qubits (int): Number of qubits.

        Returns:
            list: List of Parameter objects.
        """
        parameters = []
        for l in range(len(self._layers)):
            for i in range(n_qubits - 1):
                parameters.append(Parameter(f"params{l}_{i}"))
        return parameters

    def _construct_circuit(self, parameters: list) -> None:
        """
        Construct the BasicQNN circuit.

        Args:
            parameters (list): List of Parameter objects for the gates.
        """
        gate_dict = {"XX": self.rxx, "YY": self.ryy, "ZZ": self.rzz, "ZX": self.rzx}
        readout = self.num_qubits - 1
        data_qubits = list(range(self.num_qubits - 1))
        # Initial state preparation
        for i in range(self.num_qubits):
            self.h(i)
        self.x(readout)
        self.h(readout)
        # Alternating layers
        for l, gate in zip(range(len(self._layers)), self._layers):
            for i in range(self.num_qubits - 1):
                gate_dict[gate](data_qubits[i], readout, parameters[l * (self.num_qubits - 1) + i])
        # Final readout transformation
        self.h(readout)

    def _validate_layers(self, layers: list) -> None:
        """
        Validate the layer specifications.

        Args:
            layers (list): List of gate types.

        Raises:
            ValueError: If any gate type is unsupported.
        """
        for layer in layers:
            if layer not in ["XX", "YY", "ZZ", "ZX"]:
                raise ValueError(
                    f"The ansatz only supports the following gates: ['XX', 'YY', 'ZZ', 'ZX'], but given {layer}."
                )
