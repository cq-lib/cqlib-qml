# cqlib_qml/differentiator/parameter_shift.py
"""
Parameter shift rule for quantum circuit gradient computation.

The parameter shift rule provides a method to compute gradients of
parameterized quantum circuits by evaluating the circuit at shifted
parameter values. This method is particularly suited for hardware
execution as it only requires circuit evaluations.

References:
    - Mitarai, K. et al. (2018). "Quantum circuit learning"
    - Schuld, M. et al. (2019). "Evaluating analytic gradients on
      quantum hardware"

Examples:
    >>> from cqlib_qml.differentiator import ParameterShiftDifferentiator
    >>> from cqlib.circuit import Circuit, Parameter, ValueOperation
    >>> from cqlib.qis import Hamiltonian, PauliString
    >>>
    >>> # Setup circuit
    >>> circuit = Circuit(1)
    >>> theta = Parameter("theta")
    >>> circuit.ry(0, theta)
    >>>
    >>> # Setup observable
    >>> ham = Hamiltonian(1)
    >>> ham.add_term(PauliString.from_str("Z"), 1)
    >>>
    >>> # Compute gradients
    >>> diff = ParameterShiftDifferentiator(shift=np.pi/2)
    >>> grads = diff.run(circuit, {"theta": 0.5}, hamiltonians=[ham])
    >>> print(grads)
    {'theta': array([-0.47942554])}
"""

import numpy as np
from typing import List, Dict, Union, Optional

from cqlib.circuit import Circuit, Parameter, ValueOperation
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector


class ParameterShiftDifferentiator:
    """
    Parameter shift differentiator for quantum circuits.

    Computes gradients using the parameter shift rule:
    ∂f/∂θ = (f(θ + s) - f(θ - s)) / (2 * sin(s))

    where s is the shift amount (default π/2). This rule applies to
    gates of the form exp(-i θ G / 2) where G² = I. Other standard and
    controlled gates use a generalized Fourier shift rule. Arguments are
    shifted per gate occurrence and expression derivatives apply the chain
    rule, including when multiple gates share a symbol.

    This method is hardware-friendly as it requires only circuit
    evaluations. The current implementation uses Statevector and does not
    expose hardware executors, shots or measurement statistics.

    Args:
        shift: The shift amount in radians. Default is π/2.
            Valid shifts are those where sin(shift) ≠ 0.

    Attributes:
        _shift (float): The shift amount used for gradient computation.

    Raises:
        ValueError: If shift results in sin(shift) = 0.

    References:
        - Mitarai, K. et al. (2018). "Quantum circuit learning"
        - Schuld, M. et al. (2019). "Evaluating analytic gradients on
          quantum hardware"

    Examples:
        >>> # Default shift (π/2)
        >>> diff = ParameterShiftDifferentiator()
        >>>
        >>> # Custom shift
        >>> diff = ParameterShiftDifferentiator(shift=np.pi/4)
        >>>
        >>> # Compute gradients
        >>> grads = diff.run(circuit, {"theta": 0.5}, readouts=[0])
    """

    def __init__(self, shift: float = np.pi / 2):
        """
        Initialize the ParameterShiftDifferentiator.

        Args:
            shift (float): Shift amount in radians. Defaults to π/2.

        Raises:
            ValueError: If sin(shift) = 0 (shift is multiple of π).
        """
        normalized_shift = shift % (2 * np.pi)
        if normalized_shift > np.pi:
            normalized_shift -= 2 * np.pi
        if abs(np.sin(normalized_shift)) < 1e-8:
            raise ValueError(
                f"shift = {shift} rad ({np.degrees(shift)}°) is not allowed. " f"sin(shift) must not be zero."
            )
        self._shift = normalized_shift

    def run(
        self,
        circuit: Circuit,
        bindings: Dict[Union[str, Parameter], float],
        readouts: Optional[List[int]] = None,
        hamiltonians: Optional[List[Hamiltonian]] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate gradients using the parameter shift rule.

        Each parameterized gate argument is shifted independently. Its
        contribution is accumulated for every symbol in that expression.

        Args:
            circuit (Circuit): Parameterized quantum circuit.
                Must contain at least one parameter.
            bindings (Dict[Union[str, Parameter], float]): Parameter
                bindings mapping parameter names/objects to values.
            readouts (List[int], optional): Qubit indices for Pauli-Z
                measurements. Either readouts or hamiltonians must be
                provided. Defaults to None.
            initial_state (np.ndarray, optional): State before the input circuit.
            hamiltonians (List[Hamiltonian], optional): Observables for
                expectation computation. Either readouts or hamiltonians
                must be provided. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping parameter names
                to gradient values.

        Raises:
            ValueError: If circuit has no parameters or parameters are
                not properly assigned.
            TypeError: If hamiltonians or readouts have invalid types.

        Examples:
            >>> # Using readouts
            >>> grads = diff.run(circuit, {"theta": 0.5}, readouts=[0])
            >>>
            >>> # Using Hamiltonians
            >>> grads = diff.run(circuit, {"theta": 0.5}, hamiltonians=[ham])
        """
        if initial_state is not None:
            initial_state = np.asarray(initial_state, dtype=np.complex128)
            if initial_state.shape != (1 << circuit.num_qubits,):
                raise ValueError("Initial state dimension does not match the circuit.")
        bindings = {str(key): value for key, value in bindings.items()}
        circuit = circuit.decompose()
        if len(circuit.parameters) == 0:
            raise ValueError("The input circuit must be a parameterized quantum circuit.")

        if set(circuit.symbols) != set(bindings.keys()):
            raise ValueError("All parameters in the circuit must be assigned.")
        try:
            assigned_cir = circuit.assign_parameters(bindings)
        except:
            raise ValueError("Cannot assign values to the current circuit.")

        if readouts is not None:
            readouts = self._check_readouts(readouts, circuit.num_qubits)
            hamiltonians = []
            for readout in readouts:
                ham = Hamiltonian(circuit.num_qubits)
                pauli = "".join(
                    ["Z" if i == circuit.num_qubits - 1 - readout else "I" for i in range(circuit.num_qubits)]
                )
                ham.add_term(PauliString.from_str(pauli), 1)
                hamiltonians.append(ham)
        hamiltonians = self._check_hamiltonians(hamiltonians)

        gradients = self._run(circuit, bindings, hamiltonians, initial_state)
        return gradients

    def _run(
        self, circuit: Circuit, bindings: Dict[Union[str, Parameter], float], hamiltonians: List[Hamiltonian],
        initial_state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Internal method for parameter shift gradient computation.

        Args:
            circuit (Circuit): Parameterized quantum circuit.
            bindings (Dict): Parameter bindings.
            hamiltonians (List[Hamiltonian]): Observables.

        Returns:
            Dict[str, np.ndarray]: Gradient dictionary.
        """
        gradients = {symbol: np.zeros(len(hamiltonians)) for symbol in circuit.symbols}
        assigned = circuit.assign_parameters(bindings)
        operations = list(assigned.operations)
        # Gate expectations have frequencies among 1/2, 1, 3/2 and 2.
        # Solve the generalized shift rule, avoiding assumptions about shared
        # symbols or the spectrum of controlled/phase-dependent gates.
        shifts = np.arange(1, 5) * np.pi / 4
        frequencies = np.arange(1, 5) / 2
        weights = np.linalg.solve(2 * np.sin(frequencies[:, None] * shifts), frequencies)
        for gate_index, original in enumerate(circuit.operations):
            for param_index, expression in enumerate(original.params):
                if not isinstance(expression, Parameter) or not expression.symbols:
                    continue
                if not (original.instruction.is_standard or original.instruction.is_mcgate):
                    raise ValueError("Parameterized custom instructions do not support parameter shift")
                name = str(original.instruction.standard_gate).split(".")[-1].split("(")[0]
                if name in {"RX", "RY", "RZ", "RXX", "RYY", "RZZ", "RZX", "Phase", "U"}:
                    gate_shifts = [self._shift]
                    gate_weights = [1 / (2 * np.sin(self._shift))]
                else:
                    gate_shifts, gate_weights = shifts, weights
                gate_grad = np.zeros(len(hamiltonians))
                for shift, weight in zip(gate_shifts, gate_weights):
                    values = []
                    for sign in (1, -1):
                        params = list(operations[gate_index].params)
                        params[param_index] += sign * shift
                        shifted = Circuit(circuit.num_qubits)
                        for index, op in enumerate(operations):
                            shifted.append(ValueOperation(op.instruction, op.qubits, params)
                                           if index == gate_index else op)
                        state = (Statevector(circuit.num_qubits) if initial_state is None else
                                 Statevector.from_state(circuit.num_qubits, initial_state))
                        state.apply_circuit(shifted)
                        values.append(np.array([ham.expectation_statevector(state) for ham in hamiltonians]))
                    gate_grad += weight * (values[0] - values[1])
                for symbol in expression.symbols:
                    gradients[symbol] += expression.derivative(symbol).evaluate(bindings) * gate_grad

        return gradients

    def _check_hamiltonians(self, hams) -> List[Hamiltonian]:
        """
        Validate and format Hamiltonians.

        Args:
            hams: Hamiltonian or list of Hamiltonians.

        Returns:
            List[Hamiltonian]: List of validated Hamiltonians.

        Raises:
            TypeError: If input has invalid type.
        """
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        if isinstance(hams, list):
            for i in range(len(hams)):
                if not isinstance(hams[i], Hamiltonian):
                    raise TypeError(f"Expected Hamiltonian, got {type(hams[i]).__name__}")
        else:
            raise TypeError(f"Expected Hamiltonian or List[Hamiltonian], got {type(hams).__name__}")
        return hams

    def _check_readouts(self, readouts, n_qubits: int) -> List[int]:
        """
        Validate and format readouts.

        Args:
            readouts: Integer or list of integers.
            n_qubits (int): Number of qubits.

        Returns:
            List[int]: List of validated readout indices.

        Raises:
            TypeError: If input has invalid type.
            ValueError: If readout index is out of range.
        """
        if isinstance(readouts, int):
            readouts = [readouts]
        if isinstance(readouts, list):
            for readout in readouts:
                if not isinstance(readout, int):
                    raise TypeError(f"Expected int, got {type(readout).__name__}")
                if readout < 0 or readout >= n_qubits:
                    raise ValueError(f"Readout {readout} must be in range [0, {n_qubits-1}]")
        else:
            raise TypeError(f"Expected int or List[int], got {type(readouts).__name__}")
        return readouts
