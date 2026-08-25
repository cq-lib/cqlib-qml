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
    >>> from cqlib.circuit import Circuit, Parameter
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

from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector


class ParameterShiftDifferentiator:
    """
    Parameter shift differentiator for quantum circuits.

    Computes gradients using the parameter shift rule:
    ∂f/∂θ = (f(θ + s) - f(θ - s)) / (2 * sin(s))

    where s is the shift amount (default π/2). This rule applies to
    gates of the form exp(-i θ G / 2) where G² = I.

    This method is hardware-friendly as it requires only circuit
    evaluations, making it suitable for running on quantum devices.

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
    ) -> Dict[str, np.ndarray]:
        """
        Calculate gradients using the parameter shift rule.

        For each parameter, the gradient is computed by evaluating the
        circuit at θ + s and θ - s, where s is the shift amount.

        Args:
            circuit (Circuit): Parameterized quantum circuit.
                Must contain at least one parameter.
            bindings (Dict[Union[str, Parameter], float]): Parameter
                bindings mapping parameter names/objects to values.
            readouts (List[int], optional): Qubit indices for Pauli-Z
                measurements. Either readouts or hamiltonians must be
                provided. Defaults to None.
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

        gradients = self._run(circuit, bindings, hamiltonians)
        return gradients

    def _run(
        self, circuit: Circuit, bindings: Dict[Union[str, Parameter], float], hamiltonians: List[Hamiltonian]
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
        gradients = dict()
        base_vals = [bindings[symbol] for symbol in circuit.symbols]
        for i in range(len(circuit.symbols)):
            # Evaluate at θ + shift
            pos_vals = base_vals.copy()
            pos_vals[i] += self._shift
            pos_exps = self._compute_expectations(circuit, pos_vals, hamiltonians)
            # Evaluate at θ - shift
            neg_vals = base_vals.copy()
            neg_vals[i] -= self._shift
            neg_exps = self._compute_expectations(circuit, neg_vals, hamiltonians)
            # Apply parameter shift rule
            gradients[circuit.symbols[i]] = (pos_exps - neg_exps) / (2.0 * np.sin(self._shift))

        return gradients

    def _compute_expectations(
        self, circuit: Circuit, parameters: np.ndarray, hamiltonians: List[Hamiltonian]
    ) -> np.ndarray:
        """
        Compute expectation values for all Hamiltonians.

        Args:
            circuit (Circuit): Quantum circuit.
            parameters (np.ndarray): Parameter values.
            hamiltonians (List[Hamiltonian]): Observables.

        Returns:
            np.ndarray: Array of expectation values.
        """
        n_qubits = circuit.num_qubits
        bindings = dict(zip(circuit.symbols, parameters))
        assigned_cir = circuit.assign_parameters(bindings)
        state = Statevector(n_qubits)
        state.apply_circuit(assigned_cir)
        expectations = []
        for ham in hamiltonians:
            expectation = ham.expectation_statevector(state)
            expectations.append(expectation)

        return np.array(expectations)

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
                hams[i].simplify()
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
