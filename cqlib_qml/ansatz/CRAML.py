# cqlib_qml/ansatz/CRAML.py
"""
Color-Readout-Alternating-Mixed-Layer (CRAML) Ansatz.

This module provides the CRAML architecture, which is a variant of CRADL
that applies both XX and ZZ gates simultaneously in each layer.

References:
    `Image Compression and Classification Using Qubits and Quantum Deep Learning`
    <https://arxiv.org/abs/2110.05476>

Examples:
    >>> from cqlib_qml.ansatz import CRAML
    >>>
    >>> # Create a CRAML ansatz for a 2x2 image
    >>> ansatz = CRAML(n_qubits=3, layers=2)
    >>> ansatz.set_measurement(readouts=[2])
"""

from copy import deepcopy
from cqlib.circuit import Parameter
from .ansatz import Ansatz


class CRAML(Ansatz):
    """
    Color-Readout-Alternating-Mixed-Layer architecture.

    This ansatz architecture is a variant of CRADL where each layer
    applies both XX and ZZ gates in a mixed fashion.

    The architecture:
        - n_pos_qubits: position qubits for pixel locations
        - 1 color qubit: for pixel intensity
        - 1 readout qubit: for classification result

    Each layer applies for each position qubit:
        1. XX gate between position qubit and readout qubit
        2. XX gate between position qubit and color qubit
        3. ZZ gate between position qubit and readout qubit
        4. ZZ gate between position qubit and color qubit

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
        >>> ansatz = CRAML(n_qubits=3, layers=2)
        >>> print(ansatz.in_dim)  # 2 * (3-2) * 2 = 4 parameters
        4
    """

    def __init__(self, n_qubits: int, layers: int):
        """
        Initialize a CRAML ansatz.

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
        super(CRAML, self).__init__(n_qubits)
        self._construct_circuit(parameters)
        self.set_measurement(readouts=n_qubits - 1)

    def __str__(self) -> str:
        """Return a string representation of the ansatz."""
        return (
            f"Ansatz       | CRAML(n_qubits={self.num_qubits}, layers={self._layers}) \n"
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
            for k in range(0, (n_qubits - 2) * 2, 2):
                # Legacy allocation used 2*n_qubits names per group, while
                # construction consumed a contiguous prefix of that list.
                index = l * 2 * (n_qubits - 2) + k
                legacy_layer, legacy_index = divmod(index, 2 * n_qubits)
                parameters.append(Parameter(f"params{legacy_layer}_{legacy_index}"))
                parameters.append(Parameter(f"params{legacy_layer}_{legacy_index + 1}"))
        return parameters

    def load_params(self, summary_dict: dict) -> None:
        """Migrate original, intermediate, and previously misgrouped symbols.

        Identify the schema from the complete gate structure, including shared
        parameters, rather than from binding names alone.
        """
        count = 2 * (self.num_qubits - 2) * self._layers
        width = 2 * (self.num_qubits - 2)
        schemas = [
            [f"params{i // (2 * self.num_qubits)}_{i % (2 * self.num_qubits)}" for i in range(count)],
            [f"params{i // width}_{i % width}" for i in range(count)],
            [f"params{i // (2 * width)}_{i % (2 * width)}" for i in range(count)],
        ]
        names = [str(param) for param in self._init_parameters(self.num_qubits)]
        current = self._circuit_summary()["gates"]
        saved = Ansatz(self.num_qubits)
        saved._load_circuit(summary_dict["circuit"])
        saved_gates = saved._circuit_summary()["gates"]
        for schema in schemas:
            reverse = dict(zip(names, schema))
            expected = deepcopy(current)
            for gate in expected:
                gate["params"] = [reverse.get(value, value) if isinstance(value, str) else value
                                  for value in gate["params"]]
            if not self._same_structure(expected, saved_gates):
                continue
            mapping = dict(zip(schema, names))
            migrated = deepcopy(summary_dict)
            bindings = migrated["circuit"]["parameters"]
            if bindings is not None:
                if set(bindings) != set(schema):
                    raise ValueError("Checkpoint parameter symbols do not match the current circuit.")
                migrated["circuit"]["parameters"] = {mapping[key]: val for key, val in bindings.items()}
            for gate in migrated["circuit"]["gates"]:
                gate["params"] = [mapping.get(value, value) if isinstance(value, str) else value
                                  for value in gate["params"]]
            optim = migrated.get("optimizer")
            if optim is not None:
                cache = {}
                for key, val in optim.get("cache", {}).items():
                    prefix, separator, name = key.partition("_")
                    symbol = name if separator and prefix.isdigit() else key
                    cache[mapping.get(symbol, symbol)] = val
                optim["cache"] = cache
            return super().load_params(migrated)
        raise ValueError("Checkpoint circuit structure does not match the current circuit.")

    def _construct_circuit(self, parameters: list) -> None:
        """
        Construct the CRAML circuit.

        Args:
            parameters (list): List of Parameter objects for the gates.
        """
        n_pos_qubits = self.num_qubits - 2
        readout = self.num_qubits - 1
        for l in range(self._layers):
            for i, k in zip(range(n_pos_qubits), range(0, n_pos_qubits * 2, 2)):
                self.rxx(i, readout, parameters[l * (2 * n_pos_qubits) + k])
                self.rxx(i, n_pos_qubits, parameters[l * (2 * n_pos_qubits) + k])
                self.rzz(i, readout, parameters[l * (2 * n_pos_qubits) + k + 1])
                self.rzz(i, n_pos_qubits, parameters[l * (2 * n_pos_qubits) + k + 1])
