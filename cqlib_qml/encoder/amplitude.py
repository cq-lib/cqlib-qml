# cqlib_qml/encoder/amplitude.py
"""
Amplitude encoding for quantum data representation.

Amplitude encoding maps classical data vectors to the amplitudes of
quantum states. This is a highly efficient encoding method that can
represent 2^n-dimensional data using only n qubits.

The encoding is performed recursively using controlled-RY gates to
prepare magnitudes, followed by conditional phase gates for signs and complex phases.

References:
    - Grover, L. (2000). "Synthesis of quantum superpositions"

Examples:
    >>> from cqlib_qml.encoder import AmplitudeEncoder
    >>> import numpy as np
    >>>
    >>> # Single sample
    >>> encoder = AmplitudeEncoder()
    >>> data = np.array([1.0, 0.0, 0.0, 0.0])
    >>> circuits = encoder(data)
    >>>
    >>> # Multiple samples
    >>> data = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> circuits = encoder(data)
"""

import numpy as np
from cqlib.circuit import Circuit, MCGate, StandardGate
from cqlib_qml._numerics import scaled_vector, stable_norm


class AmplitudeEncoder:
    """
    Amplitude encoder for quantum state preparation.

    This encoder maps classical data vectors to quantum state amplitudes.
    The input vector is normalized and padded to length 2^n, then a
    recursive circuit is constructed using RY gates and controlled
    operations. Conditional phase gates preserve negative signs and complex
    relative phases. A one-element vector is represented up to global phase.

    The encoding algorithm:
        1. Normalize the input vector
        2. Pad to length 2^n where n = ceil(log2(len(vec)))
        3. Recursively split the vector into halves
        4. Apply RY gates with angles determined by the norms of halves
        5. Use controlled operations for the recursive structure
        6. Apply conditional phase gates for each basis amplitude

    Args:
        None

    Attributes:
        None

    Raises:
        ValueError: If input vector is zero (cannot be normalized).

    Examples:
        >>> encoder = AmplitudeEncoder()
        >>>
        >>> # Encode a 2D vector (uses 1 qubit)
        >>> circuits = encoder(np.array([0.6, 0.8]))
        >>>
        >>> # Encode a 4D vector (uses 2 qubits)
        >>> circuits = encoder(np.array([0.5, 0.5, 0.5, 0.5]))
    """

    def __init__(self):
        """Initialize an AmplitudeEncoder instance."""
        pass

    def __call__(self, data: np.ndarray) -> list:
        """
        Encode data using amplitude encoding.

        Args:
            data (np.ndarray): Input data array. Can be 1D (single sample)
                or 2D (multiple samples). Each sample will be normalized
                and padded to length 2^n.

        Returns:
            list: List of Circuit objects with amplitude encoding.
                Each circuit represents one encoded sample.

        Raises:
            ValueError: If the input vector is zero (norm = 0).

        Examples:
            >>> # Single sample
            >>> circuits = encoder(np.array([0.5, 0.3]))
            >>>
            >>> # Multiple samples
            >>> data = np.array([[0.5, 0.3], [0.7, 0.2]])
            >>> circuits = encoder(data)
        """
        data = np.asarray(data)
        if data.ndim not in (1, 2) or data.shape[-1] == 0:
            raise ValueError("Expected a nonempty vector or batch of vectors.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Amplitude data must be finite.")
        data = np.array([data]) if data.ndim == 1 else data
        enc_circs = []
        for vec in data:
            # Scale before squaring; finite inputs may otherwise overflow or
            # underflow in the norm. Real/imaginary components also avoid an
            # overflowing complex magnitude during scale selection.
            vec = np.asarray(vec, dtype=np.complex128)
            vec, scale = scaled_vector(vec)
            if scale == 0:
                raise ValueError("Cannot encode zero vector.")
            vec = vec / np.linalg.norm(vec)
            n_qubits = int(np.ceil(np.log2(len(vec))))
            padded_vec = np.zeros(1 << n_qubits, dtype=np.complex128)
            padded_vec[: len(vec)] = vec
            circuit = Circuit(n_qubits)
            self._build_recursive(abs(padded_vec), list(range(n_qubits - 1, -1, -1)), circuit)
            # Add each basis amplitude's phase without a dense 2^n matrix.
            # A one-element vector has only an unobservable global phase.
            if n_qubits:
                for index, amplitude in enumerate(padded_vec):
                    angle = np.angle(amplitude)
                    if angle == 0:
                        continue
                    controls = list(range(1, n_qubits))
                    zero_qubits = [q for q in range(n_qubits) if not (index >> q) & 1]
                    for q in zero_qubits:
                        circuit.x(q)
                    if controls:
                        circuit.append_mc_gate(MCGate(len(controls), StandardGate.Phase(angle)),
                                               controls + [0])
                    else:
                        circuit.phase(0, angle)
                    for q in reversed(zero_qubits):
                        circuit.x(q)
            enc_circs.append(circuit)

        return enc_circs

    def _build_recursive(self, data: np.ndarray, qubits: list, circuit: Circuit,
                         controls=(), bits=()) -> None:
        """Prepare magnitudes using global qubit indices and prefix controls."""
        if not qubits or not np.any(data):
            return
        current, remaining = qubits[0], qubits[1:]
        half = len(data) // 2
        left_norm, right_norm = stable_norm(data[:half]), stable_norm(data[half:])
        theta = 2 * np.arctan2(right_norm, left_norm)
        zero_controls = [q for q, bit in zip(controls, bits) if bit == 0]
        for q in zero_controls:
            circuit.x(q)
        if controls:
            circuit.append_mc_gate(MCGate(len(controls), StandardGate.RY(theta)),
                                   list(controls) + [current])
        else:
            circuit.ry(current, theta)
        for q in reversed(zero_controls):
            circuit.x(q)
        self._build_recursive(data[:half], remaining, circuit, controls + (current,), bits + (0,))
        self._build_recursive(data[half:], remaining, circuit, controls + (current,), bits + (1,))
