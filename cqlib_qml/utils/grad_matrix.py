# cqlib_qml/utils/grad_matrix.py
"""
Gradient matrix computation for parameterized quantum gates.

This module provides utilities for computing the gradient matrices of
parameterized quantum gates. The gradient matrices are essential for
the adjoint differentiation method.

The module extends the cqlib ValueOperation class with gradient computation
methods for various quantum gates including:
    - Single-qubit rotations: RX, RY, RZ
    - Two-qubit rotations: RXX, RXY, RZX, RZZ
    - Controlled rotations: CRX, CRY, CRZ
    - General gates: U, XY, XY2P, XY2M

References:
    - Jones, T. (2020). "Efficient quantum gradient computation"

Examples:
    >>> from cqlib_qml.utils import grad_matrix
    >>> from cqlib.circuit import Circuit, Parameter
    >>>
    >>> # Create a circuit with parameterized gate
    >>> circuit = Circuit(1)
    >>> theta = Parameter("theta")
    >>> circuit.ry(0, theta)
    >>>
    >>> # Get the gradient matrix
    >>> op = circuit.operations[0]
    >>> grad = op.grad_matrix()
    >>> print(grad)
    [array([[-0.5*cos(θ/2), -0.5*sin(θ/2)],
            [0.5*sin(θ/2), -0.5*cos(θ/2)]])]
"""

# from cqlib.circuit import Circuit, MCGate, StandardGate
import numpy as np
from scipy.linalg import block_diag
from cqlib.circuit import ValueOperation


def _standard_gate_grad(self, gate):
    """
    Compute the gradient matrix for a standard quantum gate.

    This internal function computes the derivative of the gate's unitary
    matrix with respect to its parameters. The gradient is returned as a
    list of matrices, one for each parameter.

    Args:
        self (ValueOperation): The operation instance.
        gate (StandardGate): The standard gate to differentiate.

    Returns:
        list: List of gradient matrices. Each matrix has the same shape
            as the gate's unitary matrix.

    Raises:
        ValueError: If the gate type is not supported.

    Supported Gates:
        - RX: Rotation around X-axis
        - RY: Rotation around Y-axis
        - RZ: Rotation around Z-axis
        - RXX: Ising XX coupling
        - RXY: XY rotation
        - RZX: ZX coupling
        - RZZ: Ising ZZ coupling
        - CRX: Controlled-RX
        - CRY: Controlled-RY
        - CRZ: Controlled-RZ
        - U: Generic single-qubit rotation
        - XY: XY gate
        - XY2P: Positive square root of XY
        - XY2M: Negative square root of XY

    Examples:
        >>> from cqlib.circuit import StandardGate
        >>> op = ValueOperation(StandardGate.RY, qubits=[0], params=[0.5])
        >>> grad = _standard_gate_grad(op, StandardGate.RY)
        >>> print(grad[0].shape)
        (2, 2)
    """
    if self.num_params == 0:
        return None
    if str(gate) == "RX":
        cos_v = -np.cos(self.params[0] / 2) / 2
        sin_v = -np.sin(self.params[0] / 2) / 2
        return [np.array([[sin_v, 1j * cos_v], [1j * cos_v, sin_v]], dtype=np.complex128)]
    elif str(gate) == "RY":
        cos = np.cos(self.params[0] / 2) / 2
        sin = np.sin(self.params[0] / 2) / 2
        return [np.array([[-sin, -cos], [cos, -sin]], dtype=np.complex128)]
    elif str(gate) == "RZ":
        return [
            np.array(
                [[-1j * np.exp(-self.params[0] / 2 * 1j) / 2, 0], [0, 1j * np.exp(self.params[0] / 2 * 1j) / 2]],
                dtype=np.complex128,
            )
        ]
    elif str(gate) == "RXX":
        cos = np.cos(self.params[0] / 2) / 2
        sin = np.sin(self.params[0] / 2) / 2
        return [
            np.array(
                [
                    [-sin, 0, 0, -1j * cos],
                    [0, -sin, -1j * cos, 0],
                    [0, -1j * cos, -sin, 0],
                    [-1j * cos, 0, 0, -sin],
                ],
                dtype=np.complex128,
            )
        ]
    elif str(gate) == "RXY":
        phi = self.params[0]
        theta = self.params[1]
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        grad1 = np.array([[cos, -np.exp(-1j * phi) * sin], [np.exp(1j * phi) * sin, cos]], dtype=np.complex128)
        grad2 = np.array(
            [[-sin / 2, -1j * np.exp(-1j * phi) * cos / 2], [-1j * np.exp(1j * phi) * cos / 2, -sin / 2]],
            dtype=np.complex128,
        )
        return [grad1, grad2]
    elif str(gate) == "RZX":
        cos = np.cos(self.params[0] / 2) / 2
        sin = np.sin(self.params[0] / 2) / 2
        return [
            np.array(
                [[-sin, -1j * cos, 0, 0], [-1j * cos, -sin, 0, 0], [0, 0, -sin, 1j * cos], [0, 0, 1j * cos, -sin]],
                dtype=np.complex128,
            )
        ]
    elif str(gate) == "RZZ":
        exp = 0.5j * np.exp(0.5j * self.params[0])
        sexp = -0.5j * np.exp(-0.5j * self.params[0])
        return [np.array([[sexp, 0, 0, 0], [0, exp, 0, 0], [0, 0, exp, 0], [0, 0, 0, sexp]], dtype=np.complex128)]
    elif str(gate) == "CRX":
        cos = -np.cos(self.params[0] / 2) / 2
        sin = -np.sin(self.params[0] / 2) / 2
        return [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, sin, 1j * cos], [0, 0, 1j * cos, sin]], dtype=np.complex128)
        ]
    elif str(gate) == "CRY":
        cos = np.cos(self.params[0] / 2) / 2
        sin = np.sin(self.params[0] / 2) / 2
        return [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -sin, -cos], [0, 0, cos, -sin]], dtype=np.complex128)]
    elif str(gate) == "CRZ":
        return [
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1j * np.exp(-self.params[0] / 2 * 1j) / 2, 0],
                    [0, 0, 0, 1j * np.exp(self.params[0] / 2 * 1j) / 2],
                ],
                dtype=np.complex128,
            )
        ]
    elif str(gate) == "U":
        theta, phi, lam = (float(param) for param in self.params)
        sin = np.sin(theta / 2)
        cos = np.cos(theta / 2)
        grad1 = np.array(
            [
                [-sin / 2, -np.exp(1j * lam) * cos / 2],
                [np.exp(1j * phi) * cos / 2, -np.exp(1j * (phi + lam)) * sin / 2],
            ],
            dtype=np.complex128,
        )
        grad2 = np.array(
            [
                [cos, -np.exp(1j * lam) * sin],
                [1j * np.exp(1j * phi) * sin, 1j * np.exp(1j * (phi + lam)) * cos],
            ],
            dtype=np.complex128,
        )
        grad3 = np.array(
            [
                [cos, -1j * np.exp(1j * lam) * sin],
                [np.exp(1j * phi) * sin, 1j * np.exp(1j * (phi + lam)) * cos],
            ],
            dtype=np.complex128,
        )
        return [grad1, grad2, grad3]
    elif str(gate) == "XY":
        return [np.array([[0, -np.exp(-1j * self.params[0])], [np.exp(1j * self.params[0]), 0]], dtype=np.complex128)]
    elif str(gate) == "XY2P":
        return [
            1
            / np.sqrt(2)
            * np.array(
                [
                    [1, -1 * np.exp(-1j * self.params[0])],
                    [np.exp(1j * self.params[0]), 1],
                ],
                dtype=np.complex128,
            )
        ]
    elif str(gate) == "XY2M":
        return [
            1
            / np.sqrt(2)
            * np.array(
                [
                    [1, np.exp(-1j * self.params[0])],
                    [-1 * np.exp(1j * self.params[0]), 1],
                ],
                dtype=np.complex128,
            )
        ]
    else:
        raise ValueError(f"Unsupported gate: {gate}")


def grad_matrix(self):
    """
    Compute the gradient matrix for an operation.

    This method computes the gradient matrix for both standard gates and
    multi-controlled gates. The gradient matrix represents the derivative
    of the gate's unitary with respect to its parameters.

    Args:
        self (ValueOperation): The operation instance.

    Returns:
        list: List of gradient matrices. Each matrix has the same shape
            as the gate's unitary matrix.
        None: If the operation has no parameters.

    Raises:
        ValueError: If the gate type is not supported.

    Examples:
        >>> # For a standard gate
        >>> op = ValueOperation(StandardGate.RY, qubits=[0], params=[0.5])
        >>> grad = op.grad_matrix()
        >>>
        >>> # For a multi-controlled gate
        >>> from cqlib.circuit import MCGate, StandardGate
        >>> cgate = MCGate(2, StandardGate.RY)
        >>> op = ValueOperation(cgate, qubits=[0, 1, 2], params=[0.5])
        >>> grad = op.grad_matrix()
        >>>
        >>> # For a gate without parameters
        >>> op = ValueOperation(StandardGate.H, qubits=[0])
        >>> grad = op.grad_matrix()  # Returns None
    """
    if self.instruction.mc_gate is not None:
        grads = []
        gate = self.instruction.mc_gate
        base_gate_grad = self._standard_gate_grad(gate.base_gate)
        for grad in base_gate_grad:
            grads.append(
                block_diag(np.eye((1 << gate.num_qubits) - (1 << (gate.num_qubits - gate.num_ctrl_qubits))), grad)
            )
        return grads

    if self.instruction.standard_gate is not None:
        gate = self.instruction.standard_gate
        return self._standard_gate_grad(gate)

    return None


ValueOperation._standard_gate_grad = _standard_gate_grad
ValueOperation.grad_matrix = grad_matrix
