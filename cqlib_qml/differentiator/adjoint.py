# cqlib_qml/differentiator/adjoint.py
"""
Adjoint method for quantum circuit gradient computation.

The adjoint method computes gradients of parameterized quantum circuits
by backpropagating through the circuit. This method is efficient for
classical simulation as it requires only a single forward and backward
pass, regardless of the number of parameters.

References:
    - Jones, T. (2020). "Efficient quantum gradient computation"
    - Chen, S. et al. (2021). "Adjoint method for quantum circuits"

Examples:
    >>> from cqlib_qml.differentiator import AdjointDifferentiator
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
    >>> diff = AdjointDifferentiator()
    >>> state_vector = np.array([1.0, 0.0])
    >>> grads = diff.run(circuit, {"theta": 0.5},
    ...                  state_vector=state_vector,
    ...                  hamiltonians=[ham])
    >>> print(grads)
    {'theta': array([-0.47942554])}
"""

import numpy as np
from typing import List, Dict, Union, Optional

from cqlib_qml.utils import grad_matrix
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString, Pauli
from cqlib.qis.state import Statevector


class AdjointDifferentiator:
    """
    Adjoint method for quantum circuit gradient computation.

    The adjoint differentiator computes gradients efficiently by
    backpropagating through the quantum circuit. It requires the
    state vector from the forward pass and uses the adjoint method
    to compute all parameter gradients in a single backward pass.

    This method is significantly faster than parameter shift for
    circuits with many parameters, making it ideal for classical
    simulation and optimization.

    Attributes:
        _grad_norm (float): Normalization factor for gradient computation.

    References:
        - Jones, T. (2020). "Efficient quantum gradient computation"

    Examples:
        >>> diff = AdjointDifferentiator()
        >>>
        >>> # Compute gradients with state vector
        >>> grads = diff.run(circuit, bindings,
        ...                  state_vector=state_vector,
        ...                  readouts=[0])
        >>>
        >>> # Compute gradients with Hamiltonian
        >>> grads = diff.run(circuit, bindings,
        ...                  state_vector=state_vector,
        ...                  hamiltonians=[ham])
    """

    def __init__(self):
        """Initialize an AdjointDifferentiator instance."""
        self._grad_norm = 1.0

    def run(
        self,
        circuit: Circuit,
        bindings: Dict[Union[str, Parameter], float],
        state_vector: Optional[np.ndarray] = None,
        readouts: Optional[List[int]] = None,
        hamiltonians: Optional[List[Hamiltonian]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate gradients of a parameterized quantum circuit.

        This method computes the gradients of all parameters in the circuit
        with respect to the expectation value of the given observables
        (readouts or Hamiltonians).

        Args:
            circuit (Circuit): Parameterized quantum circuit.
                Must contain at least one parameter.
            bindings (Dict[Union[str, Parameter], float]): Parameter
                bindings mapping parameter names/objects to values.
            state_vector (np.ndarray, optional): State vector from the
                forward pass. Required for adjoint method.
                Defaults to None.
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
            >>> grads = diff.run(circuit, {"theta": 0.5},
            ...                  state_vector=state_vector,
            ...                  readouts=[0])
            >>>
            >>> # Using Hamiltonians
            >>> grads = diff.run(circuit, {"theta": 0.5},
            ...                  state_vector=state_vector,
            ...                  hamiltonians=[ham])
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

        n_qubits, pipeline, free_symbols, training_gates = self._initial_circuit(circuit, assigned_cir)

        gradients = dict()
        for hamiltonian in hamiltonians:
            state = state_vector if len(hamiltonians) == 1 else state_vector.copy()
            grads_dict = self._run(state, n_qubits, pipeline, free_symbols, training_gates, hamiltonian, bindings)
            if len(gradients) == 0:
                gradients = grads_dict.copy()
            else:
                for key in grads_dict.keys():
                    gradients[key] = np.append(gradients[key], grads_dict[key])

        return gradients

    def _run(
        self,
        state: np.ndarray,
        n_qubits: int,
        pipeline: dict,
        free_symbols: tuple,
        training_gates: int,
        hamiltonian: Hamiltonian,
        bindings: dict,
    ) -> Dict[str, np.ndarray]:
        """
        Internal method for running the adjoint gradient computation.

        Args:
            state (np.ndarray): Initial state vector.
            n_qubits (int): Number of qubits.
            pipeline (dict): Parsed circuit pipeline.
            free_symbols (tuple): Symbolic parameters.
            training_gates (int): Number of trainable gates.
            hamiltonian (Hamiltonian): Observable Hamiltonian.
            bindings (dict): Parameter bindings.

        Returns:
            Dict[str, np.ndarray]: Gradient dictionary.
        """
        grads_dict = dict()
        remain_training_gates = training_gates
        vector = Statevector.from_state(n_qubits, state)
        for symbol in free_symbols:
            grads_dict[symbol] = np.array([0], dtype=np.double)

        # Calculate d(L)/d(|psi_t>)
        grad = self._initial_grad_vector(state, n_qubits, hamiltonian)
        grad_vector = self._norm_grad_vector(n_qubits, grad)

        for idx in range(len(pipeline["qubits"])):
            if remain_training_gates == 0:
                return grads_dict
            # Calculate |psi_t-1>
            inv_op = pipeline["inv_op"][idx]
            qubits = pipeline["qubits"][idx]
            self._apply_gate(inv_op, qubits, vector)

            # Calculate d(L)/d(theta) and write to grads_dict
            params = pipeline["params"][idx]
            if any(isinstance(param, Parameter) for param in params):
                remain_training_gates -= 1
                grad_matrix = pipeline["grad_matrix"][idx]
                grads_dict = self._calculate_grad(
                    grads_dict, qubits, grad_matrix, params, vector, grad_vector, bindings
                )

            # Calculate d(L)/d(|psi_t-1>)
            self._apply_gate(inv_op, qubits, grad_vector)

        return grads_dict

    def _norm_grad_vector(self, n_qubits: int, grad: np.ndarray) -> Statevector:
        """
        Normalize the gradient vector.

        Args:
            n_qubits (int): Number of qubits.
            grad (np.ndarray): Unnormalized gradient vector.

        Returns:
            Statevector: Normalized gradient vector as a quantum state.
        """
        self._grad_norm = np.linalg.norm(grad)
        if self._grad_norm < 1e-12:
            return Statevector(n_qubits)

        grad_normalized = grad / self._grad_norm
        try:
            grad_vector = Statevector.from_state(n_qubits, grad_normalized)
        except ValueError as e:
            if "not normalized" in str(e):
                current_norm = np.linalg.norm(grad_normalized)
                if abs(current_norm - 1.0) > 1e-8:
                    grad_normalized = grad_normalized / current_norm
                grad_vector = Statevector.from_state(n_qubits, grad_normalized)
            else:
                raise ValueError

        return grad_vector

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

    def _initial_circuit(self, circuit: Circuit, assigned_cir: Circuit) -> tuple:
        """
        Initialize circuit for gradient computation.

        Args:
            circuit (Circuit): Original parameterized circuit.
            assigned_cir (Circuit): Circuit with assigned parameters.

        Returns:
            tuple: (n_qubits, pipeline, free_symbols, training_gates)
        """
        n_qubits = circuit.width
        pipeline, training_gates = self._parse_circuits(circuit, assigned_cir)
        free_symbols = circuit.symbols
        return n_qubits, pipeline, free_symbols, training_gates

    def _parse_circuits(self, circuit: Circuit, assigned_cir: Circuit) -> tuple:
        """
        Parse circuits for gradient computation.

        Args:
            circuit (Circuit): Original circuit.
            assigned_cir (Circuit): Circuit with assigned parameters.

        Returns:
            tuple: (pipeline, training_gates)
        """
        pipeline = {"qubits": [], "grad_matrix": [], "training_gate": [], "params": [], "inv_op": []}
        training_gates = 0
        inv_circuit = assigned_cir.inverse()
        for op, origion_op, inv_op in zip(
            list(assigned_cir.operations)[::-1], list(circuit.operations)[::-1], inv_circuit.operations
        ):
            if op.instruction.instruction_type in ["circuit", "directive"]:
                continue
            pipeline["qubits"].append([qubit.index for qubit in op.qubits])
            pipeline["grad_matrix"].append(op.grad_matrix())
            params = []
            is_training_gate = False
            for param in origion_op.params:
                if isinstance(param, tuple):
                    param_name = circuit.parameters[param[1]]
                    params.append(param_name)
                    is_training_gate = True
                else:
                    params.append(param)
            if is_training_gate:
                training_gates += 1
            pipeline["training_gate"].append(is_training_gate)
            pipeline["params"].append(params)
            pipeline["inv_op"].append(inv_op)

        return pipeline, training_gates

    def _construct_hamilton_circuit(self, n_qubits: int, hamiltonian: Hamiltonian) -> tuple:
        """
        Construct circuit for Hamiltonian expectation.

        Args:
            n_qubits (int): Number of qubits.
            hamiltonian (Hamiltonian): Hamiltonian observable.

        Returns:
            tuple: (coefficients, circuit_list)
        """
        hamiltonian.simplify()
        coefficients = []
        circuit_list = []
        for pauli_str, coeff in hamiltonian.terms:
            coefficients.append(coeff)
            circuit = Circuit(n_qubits)
            for idx in range(pauli_str.num_qubits):
                if pauli_str.get_pauli(idx) == Pauli.x():
                    circuit.x(idx)
                if pauli_str.get_pauli(idx) == Pauli.y():
                    circuit.y(idx)
                if pauli_str.get_pauli(idx) == Pauli.z():
                    circuit.z(idx)
            circuit_list.append(circuit)

        return coefficients, circuit_list

    def _initial_grad_vector(self, state_vector: np.ndarray, n_qubits: int, expectation_op: Hamiltonian) -> np.ndarray:
        """
        Compute initial gradient vector.

        Args:
            state_vector (np.ndarray): State vector.
            n_qubits (int): Number of qubits.
            expectation_op (Hamiltonian): Observable Hamiltonian.

        Returns:
            np.ndarray: Initial gradient vector.
        """
        coefficients, circuit_list = self._construct_hamilton_circuit(n_qubits, expectation_op)
        grad_vector = np.zeros(1 << n_qubits, dtype=np.complex128)
        for coeff, circuit in zip(coefficients, circuit_list):
            sv = Statevector(n_qubits)
            sv = sv.from_state(n_qubits, state_vector)
            sv.apply_circuit(circuit)
            grad_vector += coeff * sv.data

        return grad_vector

    def _apply_gate(self, op, qubits: List[int], vector: Statevector) -> None:
        """
        Apply a gate operation to a state vector.

        Args:
            op: Gate operation.
            qubits (List[int]): Qubit indices.
            vector (Statevector): State vector to modify.

        Raises:
            ValueError: If operation type is unsupported.
        """
        if op.instruction.standard_gate:
            inv_gate = op.instruction.standard_gate
            vector.apply_standard_gate(inv_gate, qubits, op.params)
        elif op.is_mcgate or op.is_unitary:
            vector.apply_unitary_gate(qubits, op.matrix())
        else:
            raise ValueError(f"Unsupported operation type: {op.instruction.instruction_type}")

    def _calculate_grad(
        self,
        grads_dict: dict,
        qubits: List[int],
        grad_matrix: np.ndarray,
        params: List,
        vector: Statevector,
        grad_vector: Statevector,
        bindings: dict,
    ) -> dict:
        """
        Calculate gradient for a specific parameter.

        Args:
            grads_dict (dict): Accumulated gradients.
            qubits (List[int]): Qubit indices.
            grad_matrix (np.ndarray): Gradient matrix.
            params (List): Parameter list.
            vector (Statevector): Current state vector.
            grad_vector (Statevector): Gradient vector.
            bindings (dict): Parameter bindings.

        Returns:
            dict: Updated gradients dictionary.
        """
        for i in range(len(params)):
            if isinstance(params[i], Parameter):
                state = vector.copy()
                # d(|psi_t>) / d(theta_t^j)
                if len(qubits) == 1:
                    state.apply_single_qubit_gate(qubits[0], grad_matrix[i])
                elif len(qubits) == 2:
                    state.apply_double_qubits_gate(qubits[0], qubits[1], grad_matrix[i])
                else:
                    state.apply_unitary_gate(qubits, grad_matrix[i])

                # d(L)/d(theta_t^j) = d(L)/d(|psi_t>) * d(|psi_t>)/d(theta_t^j)
                grad = self._grad_norm * ((2.0 * grad_vector.data) @ state.data.conj()).real

                # write gradient
                for symbol in params[i].symbols:
                    derivative = params[i].derivative(symbol)
                    grads_dict[symbol] += (float(grad) * derivative.evaluate(bindings)).real

        return grads_dict
