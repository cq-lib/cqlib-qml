# cqlib_qml/ansatz/ansatz.py
"""
Base class for all parameterized quantum circuit ansätze.

This module defines the core Ansatz class that serves as the foundation for
all parameterized quantum circuits in the library. It provides functionality
for circuit construction, parameter management, differentiation, and
optimization.

The Ansatz class integrates with the cqlib circuit framework and provides
autograd-like capabilities through its forward/backward methods.

Examples:
    >>> from cqlib_qml.ansatz import Ansatz
    >>> from cqlib.circuit import Parameter
    >>>
    >>> # Custom ansatz
    >>> class MyAnsatz(Ansatz):
    ...     def __init__(self, n_qubits):
    ...         super().__init__(n_qubits)
    ...         theta = Parameter("theta")
    ...         for i in range(n_qubits):
    ...             self.ry(i, theta)
    ...         self.set_measurement(readouts=[0])
    >>>
    >>> ansatz = MyAnsatz(2)
    >>> result = ansatz.forward()  # Random initialization
"""

import copy
import numbers
import numpy as np
from typing import Union, Sequence, Optional, List, Dict

from cqlib.circuit import Circuit, Parameter, Qubit, ValueOperation, Instruction
from cqlib.circuit.gates import *
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from cqlib_qml.optimizer import *
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator

# Type Alias Definition
Qubits = Union[Qubit, int, Sequence[Union[Qubit, int]]]
IntQubit = Union[Qubit, int]
AppendInstruction = Union[Instruction, StandardGate, MCGate, UnitaryGate]


class Ansatz:
    """
    Base class for all parameterized quantum circuit ansätze.

    This class provides the foundation for building parameterized quantum
    circuits with support for automatic differentiation, parameter updates,
    and integration with classical neural network components.

    The Ansatz class manages:
        - Quantum circuit construction and parameter assignment
        - Forward propagation (expectation value computation)
        - Backward propagation (gradient computation)
        - Parameter optimization
        - State saving and loading

    Attributes:
        in_dim (int): Number of input dimensions (trainable parameters).
        out_dim (int): Number of output dimensions (measurements).
        gradients (dict): Gradients of trainable parameters.
        readouts (list): Readout qubits for measurement.
        hams (list): Hamiltonians for expectation calculation.
        trainable (bool): Whether the ansatz is trainable.
        updatable (bool): Whether parameters can be updated.
        num_qubits (int): Number of qubits in the circuit.
        width (int): Alias for num_qubits.
        qubits (list): List of all qubits in the circuit.
        parameters (list): List of symbolic parameters.
        symbols (list): Names of symbolic parameters.

    Note:
        User-defined ansatz classes should inherit from this base class.

    Examples:
        >>> # Create a simple ansatz
        >>> ansatz = Ansatz(2)
        >>> theta = Parameter("theta")
        >>> ansatz.ry(0, theta)
        >>> ansatz.cx(0, 1)
        >>> ansatz.set_measurement(readouts=[0])
        >>>
        >>> # Forward pass
        >>> result = ansatz.forward()
        >>>
        >>> # Assign specific parameters
        >>> ansatz.assign_parameters({"theta": 0.5})
    """

    @property
    def in_dim(self) -> int:
        """Number of input dimensions (trainable parameters)."""
        return len(self.symbols)

    @property
    def out_dim(self) -> int:
        """Number of output dimensions (measurements)."""
        return self._out_dim

    @property
    def gradients(self) -> dict:
        """Gradients of trainable parameters."""
        return self._gradients

    @property
    def readouts(self) -> list:
        """List of readout qubits for measurement."""
        return self._readouts

    @property
    def hams(self) -> list:
        """List of Hamiltonians for expectation calculation."""
        return self._hams

    @property
    def trainable(self) -> bool:
        """Whether the ansatz is trainable."""
        return self._trainable

    @property
    def updatable(self) -> bool:
        """Whether the parameters of the ansatz can be updated."""
        return self._updatable

    @property
    def summary(self) -> dict:
        """Dictionary containing the ansatz information."""
        return {
            "ansatz": f"{self.__class__.__name__}",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "readouts": self.readouts,
            "hamiltonians": self.hams,
            "circuit": self._circuit_summary(),
            "optimizer": (
                {
                    "cache": self._optimizer.cache,
                    "hyperparameters": self._optimizer.hyperparameters,
                }
                if self._optimizer
                else None
            ),
        }

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._circuit.num_qubits

    @property
    def width(self) -> int:
        """Alias for num_qubits."""
        return self._circuit.width

    @property
    def qubits(self) -> list[Qubit]:
        """List of all qubits in the circuit."""
        return self._circuit.qubits

    @property
    def parameters(self) -> list[Parameter]:
        """List of all symbolic parameters used in the circuit."""
        return self._circuit.parameters

    @property
    def symbols(self) -> list[str]:
        """List of all symbolic variable names used in the circuit."""
        return self._circuit.symbols

    @property
    def operations(self):
        """Iterator over all operations in the circuit."""
        return self._circuit.operations

    def __init__(self, qubits: int | list[int] | list[Qubit]) -> None:
        """
        Initialize an Ansatz instance.

        Args:
            qubits: Number of qubits or list of qubit indices/objects.

        Note:
            This initializes an empty circuit. Subclasses should call this
            and then add gates using the circuit construction methods.
        """
        self._circuit = Circuit(qubits)
        self._assigned_cir = None
        self._bindings = None

        self._trainable = True
        self._updatable = True

        self._gradients = {}
        self._jacobian = {}
        self._out_dim = 0
        self._encoder = None

        self._differentiator = None
        self._optimizer = None

        self._readouts = None
        self._hams = None

    def __str__(self) -> str:
        """Return a string representation of the ansatz."""
        return (
            f"Ansatz       | CustomizedAnsatz(n_qubits={self.num_qubits if len(self) > 0 else 'unknown'}) \n"
            + f"n_params     | {self.in_dim} \n"
            + f"readouts     | {self._readouts} \n"
            + f"hamiltonians | {str(self._hams)} \n"
            + f"optimizer    | {str(self._optimizer)} \n"
        )

    def add_encoder(self, other: Union[Circuit, List[Circuit], np.ndarray]) -> None:
        """
        Add encoding circuits to the ansatz.

        This method attaches encoding circuits that will be prepended to the
        ansatz during forward propagation.

        Args:
            other: Encoding circuit(s) to add. Can be a single Circuit,
                a list of Circuits, or a numpy array of Circuits.

        Raises:
            TypeError: If any element is not a Circuit.
            ValueError: If any encoding circuit contains unassigned parameters
                or if widths are inconsistent.

        Examples:
            >>> from cqlib_qml.encoder import FRQI
            >>> encoder = FRQI(n_pixels=4)
            >>> data = np.array([[0, 1], [1, 0]])
            >>> data_circuits = encoder(data)
            >>> ansatz.add_encoder(data_circuits)
        """
        enc_cirs = [other] if isinstance(other, Circuit) else other
        if isinstance(enc_cirs, np.ndarray):
            enc_cirs = enc_cirs.tolist()
        if isinstance(enc_cirs, list):
            width = enc_cirs[0].width
            for enc_cir in enc_cirs:
                if not isinstance(enc_cir, Circuit):
                    raise TypeError(f"Expected Circuit, got {type(enc_cir).__name__}")
                if len(enc_cir.parameters) > 0:
                    raise ValueError("The encoder circuit cannot contain unassigned parameters.")
                if enc_cir.width != width:
                    raise ValueError("Width of all encoding circuits must be consistent.")
        else:
            raise TypeError(f"Expected List[Circuit], got {type(enc_cirs).__name__}")
        self._encoder = enc_cirs

    def assign_parameters(self, bindings: Dict[Union[str, Parameter], float]) -> None:
        """
        Assign values to the circuit parameters.

        Args:
            bindings: Dictionary mapping parameter names or Parameter objects
                to their numeric values.

        Examples:
            >>> ansatz.assign_parameters({"theta": 0.5, "phi": 0.3})
        """
        normalized = {str(key): value for key, value in bindings.items()}
        assigned = self._circuit.assign_parameters(normalized)
        self._bindings = normalized
        self._assigned_cir = assigned

    def _assigned_value(self) -> bool:
        """Check if all parameters have been assigned."""
        if self._assigned_cir and len(self._assigned_cir.parameters) == 0:
            return True
        return False

    def _get_fwd_circuits(self, X: Optional[np.ndarray] = None) -> tuple:
        """
        Generate forward circuits with parameter bindings.

        Args:
            X: Input data for parameter binding. If None, uses random
                initialization or existing bindings.

        Returns:
            tuple: (circuits, bindings_list) where circuits is a list of
                Circuit objects and bindings_list is a list of bindings.
        """
        circuits = []
        # Parameters need to be updated
        if X is None:
            self._updatable = True
            # Preserve existing bindings when the circuit has been extended.
            if not self._assigned_value():
                keys = self.symbols
                previous = self._bindings or {}
                self._bindings = {key: previous[key] if key in previous else np.random.randn()
                                  for key in keys}
                self._assigned_cir = self._circuit.assign_parameters(self._bindings)
            # encoder + ansatz
            if self._encoder is not None:
                for enc_cir in self._encoder:
                    enc_cir_gate = enc_cir.to_gate("enc_cir")
                    enc_cir_copy = Circuit(enc_cir.num_qubits)
                    enc_cir_copy.append_circuit_gate(enc_cir_gate, list(range(enc_cir.num_qubits)))
                    enc_cir_copy.compose(self._assigned_cir)
                    circuits.append(enc_cir_copy)
            else:
                circuits = [self._assigned_cir]
            bindings_list = [self._bindings]
        # Parameters depend on the previous layer
        else:
            bindings_list = []
            self._updatable = False
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != self.in_dim:
                raise ValueError(f"Input dimension {X.shape[1]} does not match expected {self.in_dim}.")
            # X + encoder + ansatz
            if self._encoder is not None:
                if len(self._encoder) != X.shape[0]:
                    raise ValueError(f"Batch size {X.shape[0]} does not match encoding circuits {len(self._encoder)}.")
                for x, enc_cir in zip(X, self._encoder):
                    bindings = dict(zip(self.symbols, x))
                    assigned_cir = self._circuit.assign_parameters(bindings)
                    enc_cir_gate = enc_cir.to_gate("enc_cir")
                    enc_cir_copy = Circuit(enc_cir.num_qubits)
                    enc_cir_copy.append_circuit_gate(enc_cir_gate, list(range(enc_cir.num_qubits)))
                    enc_cir_copy.compose(assigned_cir)
                    circuits.append(enc_cir_copy)
                    bindings_list.append(bindings)
            # X + ansatz
            else:
                for x in X:
                    bindings = dict(zip(self.symbols, x))
                    assigned_cir = self._circuit.assign_parameters(bindings)
                    circuits.append(assigned_cir)
                    bindings_list.append(bindings)
        return circuits, bindings_list

    def _fwd(
        self, circuits: List[Circuit], quantum_state: Optional[np.ndarray], bindings_list: List[Dict]
    ) -> np.ndarray:
        """
        Execute forward pass on the circuits.

        Args:
            circuits: List of circuits to execute.
            quantum_state: Optional initial quantum state.
            bindings_list: List of parameter bindings.

        Returns:
            np.ndarray: Expectation values.
        """
        self._gradients = {}
        self._jacobian = {}
        n_circuits = len(circuits)
        n_states = quantum_state.shape[0] if quantum_state is not None else 0
        if not (n_circuits == n_states or n_states in [0, 1] or n_circuits == 1):
            raise ValueError(f"Circuit batch size {n_circuits} does not match state batch size {n_states}.")
        n = max(n_circuits, n_states)
        expectations = []
        for i in range(n):
            if n_states == 0:
                state_vector = None
            else:
                state_vector = quantum_state[i] if n_states == n else quantum_state[0]
            cir = circuits[i] if n_circuits == n else circuits[0]
            if self._readouts is not None:
                exps, sv = self._get_pauliZ_expectations(cir, self._readouts, state_vector)
            else:
                exps, sv = self._get_expectations(cir, self._hams, state_vector)
            expectations.append(exps)
            if self._trainable or not self._updatable:
                binding = bindings_list[i] if len(bindings_list) == n else bindings_list[0]
                initial = state_vector
                if isinstance(self._differentiator, ParameterShiftDifferentiator) and self._encoder is not None:
                    prepared = (Statevector(self.num_qubits) if initial is None else
                                Statevector.from_state(self.num_qubits, initial))
                    encoder = self._encoder[i] if len(self._encoder) == n else self._encoder[0]
                    prepared.apply_circuit(encoder)
                    initial = prepared.data
                self._bwd(sv, binding, initial)

        if not expectations:
            raise ValueError("Circuit batch must not be empty.")
        self._jacobian = {key: value.copy() for key, value in self._gradients.items()}
        return np.array(expectations)

    def forward(
        self, X: Optional[np.ndarray] = None, quantum_state: Optional[np.ndarray] = None
    ) -> Union[float, np.ndarray]:
        """
        Perform forward propagation for one step.

        This method computes the expectation values of the quantum circuit
        with the given parameters and optional input data.

        Args:
            X (np.ndarray, optional): Input data for parameter binding.
                If None, uses random initialization. Defaults to None.
            quantum_state (np.ndarray, optional): Initial quantum state vector.
                Defaults to None (|0⟩ state).

        Returns:
            Union[float, np.ndarray]: Expectation values of the measurements.

        Raises:
            ValueError: If circuit is not initialized or input dimensions mismatch.

        Examples:
            >>> ansatz = BasicQNN(n_qubits=3, layers=["XX"])
            >>> ansatz.set_measurement(readouts=[0])
            >>> result = ansatz.forward()  # Random initialization
            >>> print(result.shape)
            (1, 1)
            >>>
            >>> # With specific parameters
            >>> ansatz.assign_parameters({"theta": 0.5})
            >>> result = ansatz.forward()
        """
        if len(self) == 0:
            raise ValueError("The circuit must be initialized.")
        if self._differentiator is None:
            self._differentiator = AdjointDifferentiator()
        if self._optimizer is None:
            self.set_optimizer()
        if quantum_state is not None:
            quantum_state = self._validate_vectors(quantum_state, self.num_qubits)
        circuits, bindings_list = self._get_fwd_circuits(X)
        expectations = self._fwd(circuits, quantum_state, bindings_list)
        return expectations

    def _bwd(self, sv: np.ndarray, bindings: Optional[Dict] = None,
             initial_state: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients using the differentiator.

        Args:
            sv: Statevector from forward pass.
            bindings: Parameter bindings. If None, uses stored bindings.
        """
        bindings = self._bindings if bindings is None else bindings
        if isinstance(self._differentiator, AdjointDifferentiator):
            grads = self._differentiator.run(self._circuit, bindings, sv, self._readouts, self._hams)
        else:
            grads = self._differentiator.run(self._circuit, bindings, self._readouts, self._hams,
                                             initial_state=initial_state)
        if len(self._gradients) == 0:
            self._gradients = grads.copy()
        else:
            for key in grads.keys():
                self._gradients[key] = np.vstack((self._gradients[key], grads[key]))

    def backward(self, dLdexp: Optional[Union[float, np.ndarray]] = None) -> Union[dict, np.ndarray]:
        """
        Perform backward propagation for one step.

        This method computes the gradients of the loss with respect to the
        trainable parameters or inputs. Repeated calls reuse the forward
        Jacobian and replace the previous loss gradients; they do not accumulate.
        zero_grad clears both the Jacobian and loss gradients.

        Args:
            dLdexp (Union[float, np.ndarray], optional): Gradients of the loss
                with respect to the expectation values. If None, returns the
                computed gradients. Defaults to None.

        Returns:
            Union[dict, np.ndarray]:
                - If dLdexp is None: Returns the gradients dictionary.
                - If dLdexp is provided: Returns the gradients of inputs.

        Raises:
            ValueError: If the ansatz is frozen or dimensions mismatch.

        Examples:
            >>> # Forward pass
            >>> expectations = ansatz.forward(data_circuits)
            >>>
            >>> # Compute loss
            >>> loss = loss_fun(expectations, labels)
            >>>
            >>> # Backward pass
            >>> ansatz.backward(loss_fun.grads())
        """
        if not self._trainable and self._updatable:
            raise ValueError("Ansatz is frozen.")
        if dLdexp is None:
            return self._gradients
        else:
            if isinstance(dLdexp, numbers.Number):
                dLdexp = np.array([dLdexp])
            if not isinstance(dLdexp, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(dLdexp).__name__}")
            if dLdexp.ndim == 1:
                dLdexp = dLdexp.reshape(1, -1)
            if dLdexp.shape[1] != self._out_dim:
                raise ValueError(
                    f"Gradient dimension {dLdexp.shape[1]} does not match output dimension {self._out_dim}."
                )
            if not self._jacobian:
                raise ValueError("No gradients available; run forward after zero_grad before backward.")
            first_gradient = next(iter(self._jacobian.values()))
            expected_batch_size = 1 if first_gradient.ndim == 1 else first_gradient.shape[0]
            if dLdexp.shape[0] != expected_batch_size:
                raise ValueError(
                    f"Gradient batch size {dLdexp.shape[0]} does not match encoding batch {expected_batch_size}."
                )

            for key, val in self._jacobian.items():
                if val.ndim == 1:
                    val = val.reshape(1, -1)
                if val.shape != dLdexp.shape:
                    raise ValueError(f"Gradient shape {val.shape} does not match dLdexp shape {dLdexp.shape}.")
                if self._updatable:
                    self._gradients[key] = sum(sum(val * dLdexp))
                else:
                    self._gradients[key] = np.sum(val * dLdexp, axis=1)
            if self._updatable:
                return self._gradients
            else:
                return np.column_stack(list(self._gradients.values()))

    def set_differentiator(self, differentiator: str = "adjoint", shift: float = np.pi / 2) -> None:
        """
        Set the differentiator for gradient computation.

        Args:
            differentiator (str): Type of differentiator.
                Supported: "adjoint", "parameter_shift".
                Defaults to "adjoint".
            shift (float): Shift amount for parameter shift differentiator.
                Defaults to np.pi/2.

        Raises:
            ValueError: If unsupported differentiator type is specified.

        Examples:
            >>> # Use adjoint differentiator (faster for simulation)
            >>> ansatz.set_differentiator("adjoint")
            >>>
            >>> # Use parameter shift (more suitable for hardware)
            >>> ansatz.set_differentiator("parameter_shift", shift=np.pi/4)
        """
        if differentiator == "adjoint":
            self._differentiator = AdjointDifferentiator()
        elif differentiator == "parameter_shift":
            self._differentiator = ParameterShiftDifferentiator(shift)
        else:
            raise ValueError(f"Unsupported differentiator: {differentiator}.")

    def set_optimizer(self, optimizer: Union[str, dict, OptimizerBase] = "adam") -> None:
        """
        Set the optimizer for parameter updates.

        Args:
            optimizer (Union[str, dict, OptimizerBase]): The optimizer to use.
                - str: Name of optimizer ("adam", "sgd", "adagrad", "rmsprop")
                - dict: Optimizer configuration dictionary
                - OptimizerBase: Optimizer instance
                Defaults to "adam".

        Raises:
            ValueError: If the ansatz is frozen.
            TypeError: If optimizer type is unsupported.

        Examples:
            >>> ansatz.set_optimizer("adam")
            >>> ansatz.set_optimizer("sgd(lr=0.01)")
            >>> ansatz.set_optimizer(Adam(lr=0.001))
        """
        if not self._updatable:
            raise ValueError("The ansatz update is frozen.")
        self._optimizer = OptimizerInitializer(optimizer)()

    def set_measurement(self, **kwargs) -> None:
        """
        Set the measurement for the model.

        Either `readouts` or `hams` must be provided. If both are provided,
        `readouts` takes precedence.

        Args:
            readouts (Union[int, list]): Qubit indices for Pauli-Z measurement.
            hams (Union[Hamiltonian, list]): Hamiltonians for expectation calculation.

        Raises:
            ValueError: If both or neither are provided.

        Examples:
            >>> # Measure qubits 0 and 1
            >>> ansatz.set_measurement(readouts=[0, 1])
            >>>
            >>> # Measure custom Hamiltonian
            >>> from cqlib.qis import Hamiltonian, PauliString
            >>> ham = Hamiltonian(2)
            >>> ham.add_term(PauliString.from_str("ZZ"), 1.0)
            >>> ansatz.set_measurement(hams=[ham])
        """
        kwargs_keys = list(kwargs.keys())
        if len(kwargs_keys) != 1:
            raise ValueError("Only one of 'readouts' and 'hams' is accepted as the key.")
        kwargs_key = kwargs_keys[0]
        kwargs_val = list(kwargs.values())[0]
        if kwargs_key == "readouts":
            self._readouts = self._validate_readouts(kwargs_val)
            self._hams = None
        elif kwargs_key == "hams":
            self._hams = self._validate_hamiltonians(kwargs_val)
            self._readouts = None
        else:
            raise ValueError(f"Expected 'readouts' or 'hams', got {kwargs_key}.")

    def freeze(self) -> None:
        """Freeze the parameters in the ansatz (disable training)."""
        self._trainable = False

    def unfreeze(self) -> None:
        """Unfreeze the parameters in the ansatz (enable training)."""
        self._trainable = True

    def zero_grad(self) -> None:
        """Reset the gradients of parameters to zero."""
        if not self._trainable:
            raise ValueError("Ansatz is frozen.")
        self._gradients = {}
        self._jacobian = {}

    def update(self, cur_loss: Optional[float] = None) -> None:
        """
        Update the trainable parameters according to gradients.

        Args:
            cur_loss (float, optional): Current loss value for learning rate
                scheduling. Defaults to None.

        Raises:
            ValueError: If the ansatz is frozen.

        Examples:
            >>> ansatz.forward()
            >>> ansatz.backward(loss_grads)
            >>> ansatz.update()
        """
        if not (self._trainable and self._updatable):
            raise ValueError("Ansatz is frozen.")
        self._optimizer.step()
        new_params = dict(self._bindings)
        for k, v in self._gradients.items():
            if k in self.symbols:
                unique_key = f"{id(self)}_{k}"
                new_params[k] = self._optimizer(self._bindings[k], v, unique_key, cur_loss)
        self.assign_parameters(new_params)

    def _get_pauliZ_expectations(self, circuit: Circuit, readouts: list, state_vector: np.ndarray) -> tuple:
        """Compute Pauli-Z expectations for given readouts."""
        readouts = self._validate_readouts(readouts)
        hams = self._readouts2hams(readouts)
        return self._get_expectations(circuit, hams, state_vector)

    def _get_expectations(self, circuit: Circuit, hamiltonians: list, state_vector: np.ndarray) -> tuple:
        """Compute expectations for given Hamiltonians."""
        expectations = []
        hamiltonians = self._validate_hamiltonians(hamiltonians)
        if state_vector is not None:
            state = Statevector.from_state(self.num_qubits, state_vector)
        else:
            state = Statevector(self.num_qubits)
        state.apply_circuit(circuit)
        for ham in hamiltonians:
            expectation = ham.expectation_statevector(state)
            expectations.append(expectation)
        return np.array(expectations), state.data

    def load_params(self, summary_dict: dict) -> None:
        """
        Load parameters from a summary dictionary.

        Args:
            summary_dict: Dictionary containing the ansatz summary.

        Raises:
            ValueError: If dimensions don't match.
        """
        if not (self.in_dim == 0 or summary_dict["in_dim"] == self.in_dim):
            raise ValueError("The input dimensions to be loaded do not match.")
        if not (self._out_dim == 0 or summary_dict["out_dim"] == self._out_dim):
            raise ValueError("The output dimensions to be loaded do not match.")
        if len(self) == 0:
            self._load_circuit(summary_dict["circuit"])

        readouts = summary_dict["readouts"]
        hams = summary_dict["hamiltonians"]
        if self._readouts is not None and self._hams is not None:
            ValueError("Both readouts and hamiltonians are provided, readouts will be used first.")
        if readouts is not None:
            self.set_measurement(readouts=readouts)
        elif hams is not None:
            self.set_measurement(hams=hams)
        else:
            raise ValueError("No measurement provided.")

        optim = summary_dict["optimizer"]
        if optim is not None:
            self.set_optimizer(optim)

    def _circuit_summary(self) -> dict:
        """Generate a summary of the circuit structure."""
        if len(self) == 0:
            return {}
        gates = []
        for op in self.operations:
            instruction = op.instruction
            gate_info = {}
            gate_info["qubits"] = [qubit.index for qubit in op.qubits]
            gate_info["num_qubits"] = op.num_qubits
            params = []
            for param in op.params:
                if isinstance(param, Parameter):
                    params.append(str(param))
                else:
                    params.append(param)
            gate_info["params"] = params

            if instruction.is_standard:
                gate_info["type"] = "standard"
                gate_info["name"] = str(instruction.standard_gate).split(".")[-1]
            elif instruction.is_mcgate:
                gate_info["type"] = "mcgate"
                mc_gate = getattr(instruction, "mc_gate", None)
                if mc_gate is not None:
                    gate_info["base_gate"] = str(mc_gate.base_gate).split(".")[-1].split("(")[0]
                    gate_info["num_ctrl_qubits"] = mc_gate.num_ctrl_qubits
                else:
                    # Older cqlib bindings only expose the "C<num_ctrl>-<BASE>" name.
                    ctrl_part, base_name = instruction.name.split("-", 1)
                    gate_info["base_gate"] = base_name
                    gate_info["num_ctrl_qubits"] = int(ctrl_part[1:])
            elif instruction.is_unitary:
                gate_info["type"] = "unitary"
                gate_info["label"] = instruction.name
                gate_info["matrix"] = op.matrix()
            else:
                pass
            gates.append(gate_info)
        return {"num_qubits": self.num_qubits, "parameters": self._bindings, "gates": gates}

    def _load_circuit(self, cir_summary: dict) -> None:
        """Load a circuit from a summary dictionary."""
        if len(self) != 0:
            raise ValueError("Needs to be an empty circuit.")
        num_qubits = cir_summary["num_qubits"]
        if self.num_qubits == 0:
            self.add_qubits(list(range(num_qubits)))
        else:
            if self.num_qubits != num_qubits:
                raise ValueError("The loaded circuit does not match the width of current circuit.")
        for gate_info in cir_summary["gates"]:
            gate_qubits = gate_info["qubits"]
            gate_num_qubits = gate_info["num_qubits"]
            gate_params = []
            for param in gate_info["params"]:
                if isinstance(param, str):
                    param = Parameter(param)
                gate_params.append(param)
            gate_type = gate_info["type"]
            if gate_type == "standard":
                instruction = getattr(StandardGate, gate_info["name"].upper())
            elif gate_type == "mcgate":
                base_gate = getattr(StandardGate, gate_info["base_gate"].upper())
                num_ctrl_qubits = gate_info["num_ctrl_qubits"]
                instruction = MCGate(num_ctrl_qubits, base_gate)
            else:
                instruction = UnitaryGate(gate_info["label"], gate_num_qubits).with_matrix(gate_info["matrix"])
            self.append(instruction, gate_qubits, gate_params)
        parameters = cir_summary["parameters"]
        self.assign_parameters(parameters)

    def _validate_hamiltonians(self, hams):
        """Validate and format Hamiltonians."""
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        if isinstance(hams, list):
            self._out_dim = len(hams)
            for ham in hams:
                if not isinstance(ham, Hamiltonian):
                    raise TypeError(f"Expected Hamiltonian, got {type(ham).__name__}")
        else:
            raise TypeError(f"Expected Hamiltonian or List[Hamiltonian], got {type(hams).__name__}")
        return hams

    def _validate_readouts(self, readouts):
        """Validate and format readouts."""
        if isinstance(readouts, int):
            readouts = [readouts]
        if isinstance(readouts, list):
            if len(set(readouts)) != len(readouts):
                raise ValueError("readout must be a list of non-repeating integers.")
            self._out_dim = len(readouts)
            for readout in readouts:
                if not isinstance(readout, int):
                    raise TypeError(f"Expected int, got {type(readout).__name__}")
                if readout < 0 or readout >= self.num_qubits:
                    raise ValueError(f"Readout {readout} must be in range [0, {self.num_qubits - 1}].")
        else:
            raise TypeError(f"Expected int or List[int], got {type(readouts).__name__}")
        return readouts

    def _readouts2hams(self, readouts: list) -> List[Hamiltonian]:
        """Convert readouts to Pauli-Z Hamiltonians."""
        hams = []
        for readout in readouts:
            ham = Hamiltonian(self.num_qubits)
            pauli = "".join(["Z" if i == self.num_qubits - 1 - readout else "I" for i in range(self.num_qubits)])
            ham.add_term(PauliString.from_str(pauli), 1)
            hams.append(ham)
        return hams

    def _validate_vectors(self, vectors: np.ndarray, n_qubits: int) -> np.ndarray:
        """Validate quantum state vectors."""
        vectors = np.asarray(vectors, dtype=np.complex128)
        if vectors.ndim not in (1, 2):
            raise ValueError("Expected a state vector or batch of state vectors.")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != 1 << n_qubits:
            raise ValueError(f"State vector dimension {vectors.shape[1]} does not match {1 << n_qubits}.")
        return vectors

    def append(
        self,
        instruction: AppendInstruction,
        qubits: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Append a generic instruction or gate to the circuit.

        Args:
            instruction: Instruction, StandardGate, MCGate, UnitaryGate, etc.
            qubits: List of qubit indices or Qubit objects.
            params: Optional parameters for the instruction.
            label: Optional operation label.
        """
        qubit_objs = [self._circuit.qubits[q] if isinstance(q, int) else q for q in qubits]
        if isinstance(instruction, MCGate):
            gate = instruction
            if params:
                gate = MCGate(gate.num_ctrl_qubits, gate.base_gate(*params))
            operation = ValueOperation.from_mc_gate(gate, qubit_objs, label)
        elif isinstance(instruction, StandardGate):
            gate = instruction(*params) if params else instruction
            operation = ValueOperation.from_standard_gate(gate, qubit_objs, label)
        elif isinstance(instruction, UnitaryGate):
            operation = ValueOperation.from_instruction(
                Instruction.from_unitary_gate(instruction), qubit_objs, params, label
            )
        elif isinstance(instruction, Instruction):
            operation = ValueOperation.from_instruction(instruction, qubit_objs, params, label)
        else:
            raise TypeError(f"Unsupported instruction type: {type(instruction).__name__}")
        self._circuit.append(operation)
        self._assigned_cir = None

    def multi_control_gate(
        self,
        instruction: MCGate,
        qubits: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
    ) -> None:
        """Append a multi-controlled gate to the circuit."""
        self.append(instruction, qubits, params)

    # Standard single-qubit gates
    def i(self, qubit: int | Qubit) -> None:
        """Append an Identity (I) gate."""
        self._circuit.i(qubit)
        self._assigned_cir = None

    def h(self, qubit: int | Qubit) -> None:
        """Append a Hadamard (H) gate."""
        self._circuit.h(qubit)
        self._assigned_cir = None

    def x(self, qubit: int | Qubit) -> None:
        """Append a Pauli-X (NOT) gate."""
        self._circuit.x(qubit)
        self._assigned_cir = None

    def y(self, qubit: int | Qubit) -> None:
        """Append a Pauli-Y gate."""
        self._circuit.y(qubit)
        self._assigned_cir = None

    def z(self, qubit: int | Qubit) -> None:
        """Append a Pauli-Z gate."""
        self._circuit.z(qubit)
        self._assigned_cir = None

    def s(self, qubit: int | Qubit) -> None:
        """Append an S (Phase) gate."""
        self._circuit.s(qubit)
        self._assigned_cir = None

    def sdg(self, qubit: int | Qubit) -> None:
        """Append an S-dagger (S†) gate."""
        self._circuit.sdg(qubit)
        self._assigned_cir = None

    def t(self, qubit: int | Qubit) -> None:
        """Append a T gate."""
        self._circuit.t(qubit)
        self._assigned_cir = None

    def tdg(self, qubit: int | Qubit) -> None:
        """Append a T-dagger (T†) gate."""
        self._circuit.tdg(qubit)
        self._assigned_cir = None

    def x2p(self, qubit: int | Qubit) -> None:
        """Append a √X (SX) gate."""
        self._circuit.x2p(qubit)
        self._assigned_cir = None

    def x2m(self, qubit: int | Qubit) -> None:
        """Append a √X† (SXdg) gate."""
        self._circuit.x2m(qubit)
        self._assigned_cir = None

    def y2p(self, qubit: int | Qubit) -> None:
        """Append a √Y gate."""
        self._circuit.y2p(qubit)
        self._assigned_cir = None

    def y2m(self, qubit: int | Qubit) -> None:
        """Append a √Y† gate."""
        self._circuit.y2m(qubit)
        self._assigned_cir = None

    # Rotation gates
    def rx(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the X-axis by angle theta."""
        self._circuit.rx(qubit, theta)
        self._assigned_cir = None

    def ry(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the Y-axis by angle theta."""
        self._circuit.ry(qubit, theta)
        self._assigned_cir = None

    def rz(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the Z-axis by angle theta."""
        self._circuit.rz(qubit, theta)
        self._assigned_cir = None

    def phase(self, qubit: int | Qubit, lambda_: float | Parameter) -> None:
        """Append a Phase gate (P gate)."""
        self._circuit.phase(qubit, lambda_)
        self._assigned_cir = None

    def xy(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append an XY gate."""
        self._circuit.xy(qubit, theta)
        self._assigned_cir = None

    def xy2p(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a √XY gate (positive phase)."""
        self._circuit.xy2p(qubit, theta)
        self._assigned_cir = None

    def xy2m(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a √XY† gate (negative phase)."""
        self._circuit.xy2m(qubit, theta)
        self._assigned_cir = None

    def u(
        self,
        qubit: int | Qubit,
        theta: float | Parameter,
        phi: float | Parameter,
        lambda_: float | Parameter,
    ) -> None:
        """Append a generic single-qubit rotation U(theta, phi, lambda)."""
        self._circuit.u(qubit, theta, phi, lambda_)
        self._assigned_cir = None

    def rxy(self, qubit: int | Qubit, theta: float | Parameter, phi: float | Parameter) -> None:
        """Append a rotation in the XY plane."""
        self._circuit.rxy(qubit, theta, phi)
        self._assigned_cir = None

    # Two-qubit gates
    def cx(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-NOT (CNOT) gate."""
        self._circuit.cx(control, target)
        self._assigned_cir = None

    def cy(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-Y gate."""
        self._circuit.cy(control, target)
        self._assigned_cir = None

    def cz(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-Z gate."""
        self._circuit.cz(control, target)
        self._assigned_cir = None

    def swap(self, a: int | Qubit, b: int | Qubit) -> None:
        """Append a SWAP gate."""
        self._circuit.swap(a, b)
        self._assigned_cir = None

    def rxx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising XX coupling gate."""
        self._circuit.rxx(a, b, theta)
        self._assigned_cir = None

    def ryy(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising YY coupling gate."""
        self._circuit.ryy(a, b, theta)
        self._assigned_cir = None

    def rzz(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising ZZ coupling gate."""
        self._circuit.rzz(a, b, theta)
        self._assigned_cir = None

    def rzx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising ZX coupling gate."""
        self._circuit.rzx(a, b, theta)
        self._assigned_cir = None

    def fsim(
        self,
        a: int | Qubit,
        b: int | Qubit,
        theta: float | Parameter,
        phi: float | Parameter,
    ) -> None:
        """Append a Fermionic Simulation gate (fSim)."""
        self._circuit.fsim(a, b, theta, phi)
        self._assigned_cir = None

    # Controlled rotation gates
    def crx(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RX gate."""
        self._circuit.crx(control, target, theta)
        self._assigned_cir = None

    def cry(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RY gate."""
        self._circuit.cry(control, target, theta)
        self._assigned_cir = None

    def crz(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RZ gate."""
        self._circuit.crz(control, target, theta)
        self._assigned_cir = None

    # Multi-qubit gates
    def ccx(self, control1: int | Qubit, control2: int | Qubit, target: int | Qubit) -> None:
        """Append a Toffoli gate (CCX)."""
        self._circuit.ccx(control1, control2, target)
        self._assigned_cir = None

    def multi_control(
        self,
        instruction: StandardGate,
        controls: list[int] | list[Qubit],
        targets: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
    ) -> None:
        """Append a multi-controlled version of a standard gate."""
        self._circuit.append_mc_gate(
            MCGate(len(controls), instruction(*params) if params is not None else instruction),
            controls + targets,
        )
        self._assigned_cir = None

    def unitary(self, gate: UnitaryGate, qubits: list[int] | list[Qubit]) -> None:
        """Append a custom unitary gate to the circuit."""
        self._circuit.append_unitary_gate(gate, qubits)
        self._assigned_cir = None

    def decompose(self) -> "Circuit":
        """Decompose the circuit into simpler operations."""
        return self._circuit.decompose()

    def to_matrix(self, qubits_order: Optional[list[int]] = None) -> np.ndarray:
        """Convert the circuit to its unitary matrix representation."""
        return self._circuit.to_matrix(qubits_order)

    def __len__(self) -> int:
        """Number of operations in the circuit."""
        return len(self._circuit)

    def __getitem__(self, idx: int | slice) -> ValueOperation | list[ValueOperation]:
        """Access operations by index."""
        return self._circuit.__getitem__(idx)

    def add_qubits(self, qubits: list[int]) -> None:
        """Add additional qubits to the circuit."""
        self._circuit.add_qubits(qubits)
        self._assigned_cir = None

    def add_parameter(self, param: Parameter) -> tuple[int, bool]:
        """Register a parameter name for legacy callers.

        cqlib 2 discovers circuit parameters when gates are appended; this
        registration alone does not add an unused trainable circuit parameter.
        The returned index refers to the legacy registration list, not the
        order of circuit.parameters.
        """
        if not isinstance(param, Parameter):
            raise TypeError("param must be a Parameter")
        registered = getattr(self, "_registered_parameters", list(self.parameters))
        added = param not in registered
        if added:
            registered.append(param)
        self._registered_parameters = registered
        return registered.index(param), added
