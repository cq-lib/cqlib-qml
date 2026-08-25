# cqlib_qml/ansatz/CRADL.py
"""
Color-Readout-Alternating-Double-Layer (CRADL) Ansatz.

This module provides the CRADL architecture, which is specifically designed
for image data encoded with FRQI or NEQR. The ansatz alternates between
data-readout and data-color interactions using XX and ZZ gates.

References:
    `Image Compression and Classification Using Qubits and Quantum Deep Learning`
    <https://arxiv.org/abs/2110.05476>

Examples:
    >>> from cqlib_qml.ansatz import CRADL
    >>>
    >>> # Create a CRADL ansatz for a 2x2 image (4 pixels)
    >>> # n_qubits = pos_qubits + color_qubits = 2 + 1 = 3
    >>> ansatz = CRADL(n_qubits=3, layers=2)
    >>>
    >>> # Measure the readout qubit (last qubit)
    >>> ansatz.set_measurement(readouts=[2])
"""

from cqlib.circuit import Parameter
from .ansatz import Ansatz


class CRADL(Ansatz):
    """
    Color-Readout-Alternating-Double-Layer architecture.

    This ansatz architecture is specifically designed for quantum image
    classification with FRQI or NEQR encoding. It consists of consecutive
    data-readout and data-color XX gates, followed by analogous ZZ gates
    for each layer.

    The architecture:
        - n_pos_qubits: position qubits for pixel locations
        - 1 color qubit: for pixel intensity
        - 1 readout qubit: for classification result

    Each layer applies:
        1. XX gates between position qubits and readout qubit
        2. XX gates between position qubits and color qubit
        3. ZZ gates between position qubits and readout qubit
        4. ZZ gates between position qubits and color qubit

    Note:
        Only applicable to FRQI encoding or NEQR for binary images.
        By default, the last qubit is the readout qubit,
        the penultimate qubit is the color qubit, and others are data qubits.

    Args:
        n_qubits (int): The number of qubits. Must be at least 3.
        layers (int): The number of layers.

    Attributes:
        num_qubits (int): Number of qubits.
        in_dim (int): Number of trainable parameters (2 * (n_qubits-2) * layers).
        _layers (int): Number of layers.

    Raises:
        ValueError: If n_qubits < 3.

    Examples:
        >>> # For a 2x2 image with FRQI encoding
        >>> ansatz = CRADL(n_qubits=3, layers=2)
        >>> print(ansatz.in_dim)  # 2 * (3-2) * 2 = 4 parameters
        4
    """

    def __init__(self, n_qubits: int, layers: int):
        """
        Initialize a CRADL ansatz.

        Args:
            n_qubits (int): Number of qubits (must be >= 3).
            layers (int): Number of layers.

        Raises:
            ValueError: If n_qubits < 3.
        """
        if n_qubits < 3:
            raise ValueError("n_qubits should be >= 3.")
        self._layers = layers
        parameters = self._init_parameters(n_qubits)
        super(CRADL, self).__init__(n_qubits)
        self._construct_circuit(parameters)
        self.set_measurement(readouts=n_qubits - 1)

    def __str__(self) -> str:
        """Return a string representation of the ansatz."""
        return (
            f"Ansatz       | CRADL(n_qubits={self.num_qubits}, layers={self._layers}) \n"
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
        for l in range(self._layers):
            for i in range(n_qubits - 2):
                parameters.append(Parameter(f"params{l}_{i}"))
            for i in range(n_qubits - 2):
                parameters.append(Parameter(f"params{l}_{n_qubits - 2 + i}"))
        return parameters

    def _construct_circuit(self, parameters: list) -> None:
        """
        Construct the CRADL circuit.

        Args:
            parameters (list): List of Parameter objects for the gates.
        """
        n_pos_qubits = self.num_qubits - 2
        readout = self.num_qubits - 1
        for l in range(self._layers):
            for i in range(n_pos_qubits):
                self.rxx(i, readout, parameters[l * (2 * n_pos_qubits) + i])
                self.rxx(i, n_pos_qubits, parameters[l * (2 * n_pos_qubits) + i])
            for i in range(n_pos_qubits):
                self.rzz(i, readout, parameters[l * (2 * n_pos_qubits) + i + n_pos_qubits])
                self.rzz(i, n_pos_qubits, parameters[l * (2 * n_pos_qubits) + i + n_pos_qubits])
