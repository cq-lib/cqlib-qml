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
        in_dim (int): Input columns in explicit role mode; circuit symbols in legacy mode.
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
        """Input columns, or legacy circuit symbol count before role declaration."""
        return len(self.input_params) if self._roles is not None else len(self.symbols)

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
        return copy.deepcopy({
            "parameter_roles": copy.deepcopy(self._roles),
            "weights": copy.deepcopy(self._weights),
            "ansatz": f"{self.__class__.__name__}",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "readouts": self.readouts,
            "hamiltonians": (None if self.hams is None else [
                {"num_qubits": ham.num_qubits,
                 "terms": [(str(pauli), coeff) for pauli, coeff in ham.terms]}
                for ham in self.hams
            ]),
            "circuit": self._circuit_summary(),
            "optimizer": (
                self._optimizer.state_dict()
                if self._optimizer
                else None
            ),
        })

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

    def __init__(self, qubits: int | list[int] | list[Qubit], *, random_state=None) -> None:
        """
        Initialize an Ansatz instance.

        Args:
            qubits: Number of qubits or list of qubit indices/objects.
            random_state (int/Generator/None): Independent random stream.

        Note:
            This initializes an empty circuit. Subclasses should call this
            and then add gates using the circuit construction methods.
        """
        from cqlib_qml._state import make_rng
        self._rng = make_rng(random_state)
        self.training = True
        self._roles = None
        self._weights = {}
        self._active_inputs = []
        self._active_weights = []
        self._batch_size = 0
        self._circuit = Circuit(qubits)
        self._assigned_cir = None
        self._bindings = None

        self._forward_valid = False
        self._gradient_valid = False
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
            if not enc_cirs:
                raise ValueError("Encoding circuit list must not be empty")
            if not isinstance(enc_cirs[0], Circuit):
                raise TypeError("Expected Circuit as the first encoder element")
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
        names = self.weight_params
        if self._roles is not None:
            self.assign_weights(normalized)
            return
        if set(normalized) != set(self.symbols):
            raise ValueError("All circuit parameter bindings are required")
        vector = np.asarray([normalized[name] for name in names])
        if np.iscomplexobj(vector) or not np.all(np.isfinite(vector)):
            raise ValueError("Parameter bindings must be finite real numbers")
        assigned = self._circuit.assign_parameters(normalized)
        self._weights = dict(normalized)
        self._invalidate_gradients()
        self._bindings = normalized
        self._assigned_cir = assigned

    def _assigned_value(self) -> bool:
        """Check if all parameters have been assigned."""
        if self._assigned_cir and len(self._assigned_cir.parameters) == 0:
            return True
        return False

    @property
    def input_params(self):
        """Input symbol names in declared column order (empty in legacy mode)."""
        return list(self._roles[0]) if self._roles is not None else []

    @property
    def weight_params(self):
        """Persistent weight symbol names in declared order."""
        return list(self._roles[1]) if self._roles is not None else list(self.symbols)

    @property
    def num_inputs(self):
        return len(self.input_params)

    @property
    def num_weights(self):
        return len(self.weight_params)

    @property
    def weights(self):
        return np.array([self._weights[name] for name in self.weight_params], dtype=float)

    @property
    def weight_gradients(self):
        return np.array([self._gradients.get(name, 0.0) for name in self.weight_params])

    @property
    def input_jacobian(self):
        return self._role_jacobian(self._active_inputs)

    @property
    def weight_jacobian(self):
        return self._role_jacobian(self._active_weights)

    @property
    def jacobian(self):
        """Copies of the last forward Jacobians, indexed by symbol name."""
        return {key: value.copy() for key, value in self._jacobian.items()}

    def _role_jacobian(self, names):
        if not self._forward_valid:
            raise ValueError("Run a recorded forward before querying the Jacobian")
        return (np.stack([self._jacobian[name] for name in names], axis=-1)
                if names else np.empty((self._batch_size, self.out_dim, 0)))

    def set_parameter_roles(self, *, input_params, weight_params):
        """Declare a complete, disjoint partition of circuit symbols.

        Lists accept names or single-symbol Parameters; list order is the
        input/weight vector order. Declare roles after constructing the circuit.
        """
        roles = []
        for values in (input_params, weight_params):
            names = []
            for value in values:
                if not isinstance(value, (str, Parameter)):
                    raise TypeError("Parameter roles require names or Parameters")
                name = str(value)
                if name not in self.symbols:
                    raise ValueError(f"Unknown parameter: {name}")
                names.append(name)
            if len(set(names)) != len(names):
                raise ValueError("Duplicate parameter role")
            roles.append(names)
        if set(roles[0]) & set(roles[1]):
            raise ValueError("Input and weight parameter roles conflict")
        if set(roles[0] + roles[1]) != set(self.symbols):
            raise ValueError("Parameter roles must cover every circuit symbol")
        previous = {**(self._bindings or {}), **self._weights}
        self._roles = roles
        self._weights = {name: previous[name] for name in roles[1] if name in previous}
        self._updatable = bool(roles[1])
        self.zero_grad()
        self._invalidate_gradients()

    def assign_weights(self, values):
        """Bind a complete real weight vector or symbol mapping."""
        names = self.weight_params
        if isinstance(values, dict):
            values = {str(key): value for key, value in values.items()}
            if set(values) != set(names):
                raise ValueError("Weight bindings must match all weight parameters")
            vector = np.asarray([values[name] for name in names])
        else:
            vector = np.asarray(values)
        if vector.shape != (len(names),) or np.iscomplexobj(vector):
            raise ValueError("Weights must be a real vector matching weight parameters")
        vector = vector.astype(float)
        if not np.all(np.isfinite(vector)):
            raise ValueError("Weights must be finite")
        self._weights = dict(zip(names, vector))
        if self._roles is None or not self.input_params:
            self._bindings = dict(self._weights)
            self._assigned_cir = self._circuit.assign_parameters(self._bindings)
        self._invalidate_gradients()

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def _get_fwd_circuits(self, X=None):
        if self._roles is not None:
            if set(self.input_params + self.weight_params) != set(self.symbols):
                raise ValueError("Circuit symbols changed; redeclare parameter roles")
            inputs, weights = self.input_params, self.weight_params
            if inputs and X is None:
                raise ValueError("Input parameters require X")
        else:
            inputs, weights = (list(self.symbols), []) if X is not None else ([], list(self.symbols))
        self._active_inputs, self._active_weights = inputs, weights
        self._updatable = bool(weights)
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.ndim != 2 or X.shape[1] != len(inputs) or not len(X):
                raise ValueError(f"Input dimension must be (batch, {len(inputs)})")
            if np.iscomplexobj(X) or not np.all(np.isfinite(X)):
                raise ValueError("Inputs must be finite real numbers")
            rows = X
        else:
            rows = np.empty((len(self._encoder) if self._encoder is not None else 1, 0))
        for name in weights:
            if name not in self._weights:
                self._weights[name] = self._rng.normal()
        if self._encoder is not None and len(self._encoder) != len(rows):
            raise ValueError("Batch size does not match encoding circuits")
        circuits, bindings_list = [], []
        for index, row in enumerate(rows):
            bindings = {name: self._weights[name] for name in weights}
            bindings.update(zip(inputs, row))
            assigned = self._circuit.assign_parameters(bindings)
            if self._encoder is not None:
                encoder = self._encoder[index]
                full = Circuit(encoder.num_qubits)
                full.append_circuit_gate(encoder.to_gate("enc_cir"), list(range(encoder.num_qubits)))
                full.compose(assigned)
                assigned = full
            circuits.append(assigned)
            bindings_list.append(bindings)
        if not inputs:
            self._bindings = dict(bindings_list[0])
            self._assigned_cir = self._circuit.assign_parameters(self._bindings)
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
            if self._retain_derived and self.symbols:
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
        self._batch_size = n
        return np.array(expectations)

    def forward(
        self, X: Optional[np.ndarray] = None, quantum_state: Optional[np.ndarray] = None,
        *, retain_derived: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Perform forward propagation for one step.

        This method computes the expectation values of the quantum circuit
        with the given parameters and optional input data.

        Args:
            X (np.ndarray, optional): Input data for parameter binding.
                Declared input columns; legacy mode binds all circuit symbols.
                None uses persistent weights and is valid only without input roles.
            quantum_state (np.ndarray, optional): Initial quantum state vector.
                Defaults to None (|0⟩ state).
            retain_derived (bool): Retain the forward Jacobian for backward.
                False discards the last forward cache, preserving cumulative
                gradients and freeze status.

        Returns:
            np.ndarray: Expectations with shape (batch, out_dim), including batch=1.

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
        self._forward_valid = False
        self._retain_derived = retain_derived
        if not retain_derived:
            self._invalidate_gradients()
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
        self._forward_valid = retain_derived
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
        for key, value in grads.items():
            value = np.asarray(value).reshape(1, -1)
            self._jacobian[key] = (np.vstack((self._jacobian[key], value))
                                   if key in self._jacobian else value.copy())

    def backward(self, dLdexp=None, retain_grad=True):
        """Return (batch, inputs) gradients and accumulate shared weight VJPs.

        None differentiates the sum of outputs. Loss scaling is supplied by
        the caller; this method never averages across the batch.
        """
        if not self._forward_valid:
            raise ValueError("No gradients available; run a recorded forward before backward")
        if dLdexp is None:
            dLdexp = np.ones((self._batch_size, self.out_dim))
        dLdexp = np.asarray(dLdexp)
        if dLdexp.ndim == 0:
            dLdexp = dLdexp.reshape(1, 1)
        elif dLdexp.ndim == 1:
            dLdexp = dLdexp.reshape(1, -1)
        if dLdexp.shape != (self._batch_size, self.out_dim):
            raise ValueError("Gradient batch size or output dimension mismatch")
        dx = np.einsum("bo,boi->bi", dLdexp, self.input_jacobian)
        if self._trainable and retain_grad and self._active_weights:
            dw = np.einsum("bo,bow->w", dLdexp, self.weight_jacobian)
            for name, value in zip(self._active_weights, dw):
                self._gradients[name] = self._gradients.get(name, 0.0) + value
            self._gradient_valid = True
        return dx

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
            ValueError: If optimizer configuration is invalid.
            TypeError: If optimizer type is unsupported.

        Examples:
            >>> ansatz.set_optimizer("adam")
            >>> ansatz.set_optimizer("sgd(lr=0.01)")
            >>> ansatz.set_optimizer(Adam(lr=0.001))
        """
        self._optimizer = OptimizerInitializer(optimizer)()
        # Each component owns its state; stable parameter keys must not collide
        # when the same optimizer configuration is supplied to several layers.
        if isinstance(optimizer, OptimizerBase):
            self._optimizer = self._optimizer.copy()

    def set_measurement(self, **kwargs) -> None:
        """
        Set the measurement for the model.

        Exactly one of `readouts` or `hams` must be provided.
        Setting either replaces the previous measurement mode.

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
        self._invalidate_gradients()

    def freeze(self) -> None:
        """Freeze the parameters in the ansatz (disable training)."""
        self._trainable = False
        self.zero_grad()

    def unfreeze(self) -> None:
        """Unfreeze the parameters in the ansatz (enable training)."""
        self._trainable = True

    def zero_grad(self):
        """Clear accumulated weight gradients without discarding forward caches."""
        self._gradients = {}
        self._gradient_valid = False

    def _invalidate_gradients(self):
        """Discard only transient forward state, preserving accumulated gradients."""
        self._forward_valid = False
        self._jacobian = {}

    def update(self, cur_loss=None):
        if not self._trainable or not self._gradient_valid or not self._gradients:
            return
        if self._optimizer is None:
            self.set_optimizer()
        self._optimizer.step()
        updated = dict(self._weights)
        for name, gradient in self._gradients.items():
            updated[name] = self._optimizer(self._weights[name], gradient, name, cur_loss)
        self.assign_weights(updated)

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
        from cqlib_qml._state import clone_state
        candidate = clone_state(self)
        candidate._load_params_in_place(clone_state(summary_dict))
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _load_params_in_place(self, summary_dict: dict) -> None:
        """
        Load parameters from a summary dictionary.

        Args:
            summary_dict: Dictionary containing the ansatz summary.

        Raises:
            ValueError: If dimensions don't match.
        """
        if not (self.in_dim == 0 or summary_dict["in_dim"] == self.in_dim or summary_dict.get("parameter_roles") is not None):
            raise ValueError("The input dimensions to be loaded do not match.")
        if not (self._out_dim == 0 or summary_dict["out_dim"] == self._out_dim):
            raise ValueError("The output dimensions to be loaded do not match.")
        self._roles = None
        self._bindings = None
        self._assigned_cir = None
        self._invalidate_gradients()
        if len(self) == 0:
            self._load_circuit(summary_dict["circuit"])
        else:
            saved_circuit = summary_dict["circuit"]
            if saved_circuit["num_qubits"] != self.num_qubits:
                raise ValueError("The loaded circuit does not match the current width.")
            # Compare the canonical decomposed operations, including symbolic
            # expressions and custom matrices, before changing any bindings.
            restored = Ansatz(self.num_qubits)
            restored._load_circuit(saved_circuit)
            current_gates = self._circuit_summary()["gates"]
            saved_gates = restored._circuit_summary()["gates"]
            if not self._same_structure(current_gates, saved_gates):
                raise ValueError("Checkpoint circuit structure does not match the current circuit.")
            bindings = saved_circuit["parameters"]
            if bindings is not None:
                if set(bindings) != set(self.symbols):
                    raise ValueError("Checkpoint parameter symbols do not match the current circuit.")
                self.assign_parameters(bindings)

        readouts = summary_dict["readouts"]
        hams = summary_dict["hamiltonians"]
        if hams is not None:
            decoded = []
            for item in hams:
                if isinstance(item, Hamiltonian):  # Legacy in-memory summaries.
                    decoded.append(item)
                else:
                    ham = Hamiltonian(item["num_qubits"])
                    for pauli, coefficient in item["terms"]:
                        ham.add_term(PauliString.from_str(pauli), coefficient)
                    decoded.append(ham)
            hams = decoded
        if readouts is not None:
            self.set_measurement(readouts=readouts)
        elif hams is not None:
            self.set_measurement(hams=hams)
        elif summary_dict["out_dim"] == 0:
            self._readouts = self._hams = None
            self._out_dim = 0
        else:
            raise ValueError("No measurement provided.")

        optim = summary_dict["optimizer"]
        self._optimizer = None
        if optim is not None:
            self.set_optimizer(optim)
        roles = summary_dict["parameter_roles"]
        if roles is not None:
            self.set_parameter_roles(input_params=roles[0], weight_params=roles[1])
        self._weights = copy.deepcopy(summary_dict["weights"])
        if set(self._weights) - set(self.weight_params):
            raise ValueError("Checkpoint weight symbols mismatch")
        if self._weights:
            if set(self._weights) != set(self.weight_params):
                raise ValueError("Checkpoint weight bindings incomplete")
            self.assign_weights(self._weights)
        self._encoder = None
        self.zero_grad()
        self._invalidate_gradients()

    def _circuit_summary(self) -> dict:
        """Serialize canonical decomposed operations using only Python/NumPy values."""
        if len(self) == 0:
            return {}
        gates = []
        for op in self._circuit.decompose().operations:
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
            elif instruction.is_directive:
                gate_info["type"] = "directive"
                gate_info["name"] = instruction.directive.name()
            else:
                raise ValueError(f"Unsupported checkpoint instruction: {instruction.name}")
            gates.append(gate_info)
        return {"num_qubits": self.num_qubits, "parameters": (self._bindings if self._roles is None else None), "gates": gates}

    @staticmethod
    def _same_structure(left, right):
        if len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if a.keys() != b.keys():
                return False
            for key in a:
                if key == "matrix":
                    if not np.array_equal(a[key], b[key]):
                        return False
                elif a[key] != b[key]:
                    return False
        return True

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
                instruction = StandardGate.from_name(gate_info["name"])
            elif gate_type == "mcgate":
                base_gate = StandardGate.from_name(gate_info["base_gate"])
                num_ctrl_qubits = gate_info["num_ctrl_qubits"]
                instruction = MCGate(num_ctrl_qubits, base_gate)
            elif gate_type == "directive":
                instruction = Instruction.from_directive(getattr(Directive, gate_info["name"])())
            elif gate_type == "unitary":
                instruction = UnitaryGate(gate_info["label"], gate_num_qubits).with_matrix(gate_info["matrix"])
            else:
                raise ValueError(f"Unsupported checkpoint gate type: {gate_type}")
            self.append(instruction, gate_qubits, gate_params)
        parameters = cir_summary["parameters"]
        if parameters is not None:
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
        self._invalidate_gradients()

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
        self._invalidate_gradients()

    def h(self, qubit: int | Qubit) -> None:
        """Append a Hadamard (H) gate."""
        self._circuit.h(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def x(self, qubit: int | Qubit) -> None:
        """Append a Pauli-X (NOT) gate."""
        self._circuit.x(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def y(self, qubit: int | Qubit) -> None:
        """Append a Pauli-Y gate."""
        self._circuit.y(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def z(self, qubit: int | Qubit) -> None:
        """Append a Pauli-Z gate."""
        self._circuit.z(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def s(self, qubit: int | Qubit) -> None:
        """Append an S (Phase) gate."""
        self._circuit.s(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def sdg(self, qubit: int | Qubit) -> None:
        """Append an S-dagger (S†) gate."""
        self._circuit.sdg(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def t(self, qubit: int | Qubit) -> None:
        """Append a T gate."""
        self._circuit.t(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def tdg(self, qubit: int | Qubit) -> None:
        """Append a T-dagger (T†) gate."""
        self._circuit.tdg(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def x2p(self, qubit: int | Qubit) -> None:
        """Append a √X (SX) gate."""
        self._circuit.x2p(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def x2m(self, qubit: int | Qubit) -> None:
        """Append a √X† (SXdg) gate."""
        self._circuit.x2m(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def y2p(self, qubit: int | Qubit) -> None:
        """Append a √Y gate."""
        self._circuit.y2p(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    def y2m(self, qubit: int | Qubit) -> None:
        """Append a √Y† gate."""
        self._circuit.y2m(qubit)
        self._assigned_cir = None
        self._invalidate_gradients()

    # Rotation gates
    def rx(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the X-axis by angle theta."""
        self._circuit.rx(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def ry(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the Y-axis by angle theta."""
        self._circuit.ry(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def rz(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a rotation around the Z-axis by angle theta."""
        self._circuit.rz(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def phase(self, qubit: int | Qubit, lambda_: float | Parameter) -> None:
        """Append a Phase gate (P gate)."""
        self._circuit.phase(qubit, lambda_)
        self._assigned_cir = None
        self._invalidate_gradients()

    def xy(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append an XY gate."""
        self._circuit.xy(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def xy2p(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a √XY gate (positive phase)."""
        self._circuit.xy2p(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def xy2m(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Append a √XY† gate (negative phase)."""
        self._circuit.xy2m(qubit, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

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
        self._invalidate_gradients()

    def rxy(self, qubit: int | Qubit, theta: float | Parameter, phi: float | Parameter) -> None:
        """Append a rotation in the XY plane."""
        self._circuit.rxy(qubit, theta, phi)
        self._assigned_cir = None
        self._invalidate_gradients()

    # Two-qubit gates
    def cx(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-NOT (CNOT) gate."""
        self._circuit.cx(control, target)
        self._assigned_cir = None
        self._invalidate_gradients()

    def cy(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-Y gate."""
        self._circuit.cy(control, target)
        self._assigned_cir = None
        self._invalidate_gradients()

    def cz(self, control: int | Qubit, target: int | Qubit) -> None:
        """Append a Controlled-Z gate."""
        self._circuit.cz(control, target)
        self._assigned_cir = None
        self._invalidate_gradients()

    def swap(self, a: int | Qubit, b: int | Qubit) -> None:
        """Append a SWAP gate."""
        self._circuit.swap(a, b)
        self._assigned_cir = None
        self._invalidate_gradients()

    def rxx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising XX coupling gate."""
        self._circuit.rxx(a, b, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def ryy(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising YY coupling gate."""
        self._circuit.ryy(a, b, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def rzz(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising ZZ coupling gate."""
        self._circuit.rzz(a, b, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def rzx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Append an Ising ZX coupling gate."""
        self._circuit.rzx(a, b, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

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
        self._invalidate_gradients()

    # Controlled rotation gates
    def crx(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RX gate."""
        self._circuit.crx(control, target, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def cry(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RY gate."""
        self._circuit.cry(control, target, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    def crz(self, control: int | Qubit, target: int | Qubit, theta: float | Parameter) -> None:
        """Append a Controlled-RZ gate."""
        self._circuit.crz(control, target, theta)
        self._assigned_cir = None
        self._invalidate_gradients()

    # Multi-qubit gates
    def ccx(self, control1: int | Qubit, control2: int | Qubit, target: int | Qubit) -> None:
        """Append a Toffoli gate (CCX)."""
        self._circuit.ccx(control1, control2, target)
        self._assigned_cir = None
        self._invalidate_gradients()

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
        self._invalidate_gradients()

    def unitary(self, gate: UnitaryGate, qubits: list[int] | list[Qubit]) -> None:
        """Append a custom unitary gate to the circuit."""
        self._circuit.append_unitary_gate(gate, qubits)
        self._assigned_cir = None
        self._invalidate_gradients()

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
        self._invalidate_gradients()

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
