# cqlib_qml/encoder/basis.py
"""
Basis encoding for quantum data representation.

Basis encoding maps non-negative integers to computational basis states.
This is the simplest encoding method where each integer is represented
by its binary expansion, with each bit mapped to a qubit.

Examples:
    >>> from cqlib_qml.encoder import BasisEncoder
    >>> import numpy as np
    >>>
    >>> # Encode integers
    >>> encoder = BasisEncoder()
    >>> data = np.array([0, 1, 2, 3, 4])
    >>> circuits = encoder(data)
    >>>
    >>> # Each integer becomes a basis state
    >>> # 0 -> |000⟩, 1 -> |001⟩, 2 -> |010⟩, 3 -> |011⟩, 4 -> |100⟩
"""

import numpy as np
from cqlib.circuit import Circuit


class BasisEncoder:
    """
    Basis encoder for quantum data representation.

    Encodes non-negative integers as computational basis states.
    Each integer is converted to its binary representation, and X gates
    are applied to qubits corresponding to '1' bits.

    Args:
        None

    Attributes:
        None

    Raises:
        ValueError: If input contains negative numbers or non-integers.

    Examples:
        >>> encoder = BasisEncoder()
        >>>
        >>> # Single integer (uses ceil(log2(max+1)) qubits)
        >>> circuits = encoder(np.array([5]))  # 5 -> |101⟩
        >>>
        >>> # Multiple integers
        >>> circuits = encoder(np.array([0, 1, 2, 3]))
    """

    def __init__(self):
        """Initialize a BasisEncoder instance."""
        pass

    def __call__(self, data: np.ndarray) -> list:
        """
        Encode data using basis encoding.

        Args:
            data (np.ndarray): Input array. Should be 1D array of
                non-negative integers.

        Returns:
            list: List of Circuit objects with basis encoding.

        Raises:
            ValueError: If data contains negative numbers or non-integers,
                or if data is not 1D.

        Examples:
            >>> circuits = encoder(np.array([0, 1, 2, 3]))
        """
        data = np.asarray(data)
        if data.ndim != 1 or data.size == 0:
            raise ValueError("Basis encoding requires a nonempty 1D array")
        if data.dtype.kind not in "biuf" or not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("Basis encoding only supports encoding non-negative integers (finite values required)")
        if not np.equal(data, np.floor(data)).all():
            raise ValueError("Basis encoding only supports encoding non-negative integers")
        values = [int(value) for value in data]
        n_qubits = max(1, max(values).bit_length())
        enc_circs = []
        for x in values:
            circuit = Circuit(n_qubits)
            bin_x = format(x, f"0{n_qubits}b")[::-1]
            for i, bit in enumerate(bin_x):
                if bit == "1":
                    circuit.x(i)
            enc_circs.append(circuit)

        return enc_circs
