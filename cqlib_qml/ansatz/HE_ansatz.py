# cqlib_qml/ansatz/HE_ansatz.py
"""
Hardware-Efficient Ansatz.

This module provides the Hardware-Efficient (HE) ansatz, which consists of
alternating layers of single-qubit gates and two-qubit entangling gates.

The HE ansatz is designed to be hardware-efficient and suitable for near-term
quantum devices (NISQ). It can implement various entanglement patterns and
gate sequences.

References:
    `Hardware-efficient variational quantum eigensolver for small molecules
    and quantum magnets` <https://www.nature.com/articles/nature23879>

Examples:
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>>
    >>> # Create a HE ansatz with 4 qubits, 3 layers
    >>> ansatz = HEAnsatz(
    ...     n_qubits=4,
    ...     d=3,
    ...     layers=["RY", "CX", "RZ"],
    ...     entangler="downstairs"
    ... )
    >>>
    >>> # Set measurement on qubit 0
    >>> ansatz.set_measurement(readouts=[0])
    >>>
    >>> # Forward pass
    >>> result = ansatz.forward()
"""

import math
from cqlib.circuit import Parameter
from .ansatz import Ansatz


class HEAnsatz(Ansatz):
    """
    Hardware-Efficient Ansatz.

    This ansatz consists of alternating parameterized layers of single-qubit
    gates and entangler layers of two-qubit gates.

    Structure:
        - Single-qubit layers: RX, RY, or RZ gates on all qubits
        - Entangler layers: CX, CZ, or CRY gates between qubits
        - Each layer is repeated d times

    Supported single-qubit gates: RY, RZ (RX not yet supported)
    Supported two-qubit gates: CX, CZ, CRY
    Entanglement patterns: downstairs, full, last_target, last_control

    Note:
        Only supports the "downstairs" entanglement structure for non-parameterized
        two-qubit gates (CX, CZ). For CRY, all entanglement patterns are supported.

    Args:
        n_qubits (int): The number of qubits. Must be at least 2.
        d (int): The depth (number of repetitions) of the ansatz.
        layers (list): The list of gate types for each layer.
            Supported: "CX", "CZ", "CRY", "RY", "RZ".
        entangler (str, optional): The entanglement method.
            Options:
                - "downstairs": Nearest-neighbor entanglement
                - "full": All-to-all entanglement
                - "last_target": All qubits entangle with last qubit
                - "last_control": Last qubit controls all others
            Defaults to "downstairs".

    Attributes:
        num_qubits (int): Number of qubits.
        in_dim (int): Number of trainable parameters.
        _d (int): Depth of the ansatz.
        _layers (list): Gate types for each layer.
        _entangler (str): Entanglement pattern.

    Raises:
        ValueError: If n_qubits < 2, or unsupported gates/entanglers.

    Examples:
        >>> # Nearest-neighbor entanglement with RY and CX
        >>> ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
        >>> print(ansatz.num_qubits)
        4
        >>> print(ansatz.in_dim)  # (4 + 3) * 2 = 14 parameters
        14
        >>>
        >>> # All-to-all entanglement with CRY
        >>> ansatz = HEAnsatz(n_qubits=4, d=1, layers=["CRY"], entangler="full")
        >>> print(ansatz.in_dim)  # C(4,2) = 6 parameters
        6
    """

    def __init__(
        self,
        n_qubits: int,
        d: int,
        layers: list,
        entangler: str = "downstairs",
    ):
        """
        Initialize a Hardware-Efficient ansatz.

        Args:
            n_qubits (int): Number of qubits (must be >= 2).
            d (int): Depth (number of repetitions).
            layers (list): List of gate types.
            entangler (str): Entanglement pattern. Defaults to "downstairs".

        Raises:
            ValueError: If n_qubits < 2 or unsupported gates/entanglers.
        """
        if n_qubits < 2:
            raise ValueError("n_qubits should be >= 2.")
        self._validate_layers(layers, entangler)
        self._layers = layers
        self._entangler = entangler
        self._d = d
        parameters, n_params = self._init_parameters(n_qubits)
        super(HEAnsatz, self).__init__(n_qubits)
        self._construct_circuit(parameters, n_params)

    def __str__(self) -> str:
        """Return a string representation of the ansatz."""
        return (
            f"Ansatz       | HE-ansatz(n_qubits={self.num_qubits}, d={self._d}, layers={self._layers}, "
            + f"entangler={self._entangler}) \n"
            + f"n_params     | {self.in_dim} \n"
            + f"readouts     | {[self.num_qubits - 1]} \n"
            + f"optimizer    | {str(self._optimizer)} \n"
        )

    def _init_parameters(self, n_qubits: int) -> tuple:
        """
        Initialize trainable parameters for the ansatz.

        Args:
            n_qubits (int): Number of qubits.

        Returns:
            tuple: (parameters, n_params) where:
                - parameters (list): List of Parameter objects
                - n_params (int): Number of parameters per layer
        """
        n_params = 0
        for gate in self._layers:
            if gate in ["RX", "RY", "RZ"]:
                n_params += n_qubits
            elif gate in ["CRY"]:
                if self._entangler == "full":
                    n_params += math.comb(n_qubits, 2)
                else:
                    n_params += n_qubits - 1
        parameters = []
        for l in range(self._d):
            for i in range(n_params):
                parameters.append(Parameter(f"params{l}_{i}"))
        return parameters, n_params

    def _construct_circuit(self, parameters: list, n_params: int) -> None:
        """
        Construct the HE ansatz circuit.

        Args:
            parameters (list): List of Parameter objects.
            n_params (int): Number of parameters per layer.
        """
        gate_dict = {"CX": self.cx, "CZ": self.cz, "CRY": self.cry, "RX": self.rx, "RY": self.ry, "RZ": self.rz}

        for i in range(self._d):
            param_id = 0
            for gate in self._layers:
                if gate in ["RX", "RY", "RZ"]:
                    # Single-qubit rotation layer
                    for qid in range(self.num_qubits):
                        gate_dict[gate](qid, parameters[i * n_params + param_id])
                        param_id += 1
                elif gate in ["CRY"]:
                    # CRY entangler layer (has parameters)
                    if self._entangler == "downstairs":
                        for qid in range(self.num_qubits - 1):
                            gate_dict[gate](qid, qid + 1, parameters[i * n_params + param_id])
                            param_id += 1
                    elif self._entangler == "last_target":
                        for qid in range(self.num_qubits - 1):
                            gate_dict[gate](qid, self.num_qubits - 1, parameters[i * n_params + param_id])
                            param_id += 1
                    elif self._entangler == "last_control":
                        for qid in range(self.num_qubits - 2, -1, -1):
                            gate_dict[gate](self.num_qubits - 1, qid, parameters[i * n_params + param_id])
                            param_id += 1
                    else:  # full
                        for qid1 in range(self.num_qubits - 1):
                            invert = False
                            for qid2 in range(qid1 + 1, self.num_qubits):
                                qubit_indexes = [qid2, qid1] if invert else [qid1, qid2]
                                gate_dict[gate](qubit_indexes[0], qubit_indexes[1], parameters[i * n_params + param_id])
                                invert = not invert
                                param_id += 1
                else:
                    # CX/CZ entangler layer (no parameters)
                    if self._entangler == "downstairs":
                        for qid in range(self.num_qubits - 1):
                            gate_dict[gate](qid, qid + 1)
                    elif self._entangler == "last_target":
                        for qid in range(self.num_qubits - 1):
                            gate_dict[gate](qid, self.num_qubits - 1)
                    elif self._entangler == "last_control":
                        for qid in range(self.num_qubits - 2, -1, -1):
                            gate_dict[gate](self.num_qubits - 1, qid)
                    else:  # full
                        for qid1 in range(self.num_qubits - 1):
                            invert = False
                            for qid2 in range(qid1 + 1, self.num_qubits):
                                qubit_indexes = [qid2, qid1] if invert else [qid1, qid2]
                                gate_dict[gate](qubit_indexes[0], qubit_indexes[1])
                                invert = not invert

    def _validate_layers(self, layers: list, entangler: str) -> None:
        """
        Validate the layer and entangler specifications.

        Args:
            layers (list): List of gate types.
            entangler (str): Entanglement pattern.

        Raises:
            ValueError: If unsupported layers or entangler.
        """
        supported_entanglers = ["downstairs", "full", "last_control", "last_target"]
        if entangler not in supported_entanglers:
            raise ValueError(f"Invalid entangler. Supported entanglers: {supported_entanglers}")

        supported_layers = ["RX", "RY", "RZ", "CX", "CZ", "CRY"]
        for layer in layers:
            if layer not in supported_layers:
                raise ValueError(f"Invalid layer. Supported layers: {supported_layers}")
