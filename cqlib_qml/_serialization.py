"""Native reducers shared by the caller's pickle or cloudpickle object graph."""
import copyreg

from cqlib.circuit import Circuit, Parameter, Qubit
from cqlib.qis import Hamiltonian, PauliString


def _restore_circuit(qubits, summary, phase):
    from cqlib_qml.ansatz import Ansatz
    proxy = Ansatz(qubits)
    if summary:
        # The gate codec accepts positions; the saved gates contain physical IDs.
        positions = {qubit: index for index, qubit in enumerate(qubits)}
        summary = {**summary, 'gates': [
            {**gate, 'qubits': [positions[q] for q in gate['qubits']]}
            for gate in summary['gates']
        ]}
        proxy._load_circuit(summary)
    proxy._circuit.set_global_phase(phase)
    return proxy._circuit


def _reduce_circuit(circuit):
    from cqlib_qml.ansatz import Ansatz
    # Reuse the checkpoint gate codec; retain empty widths and global phase too.
    decomposed = circuit.decompose()
    proxy = Ansatz(decomposed.num_qubits)
    proxy._circuit = decomposed
    return _restore_circuit, ([q.index for q in decomposed.qubits],
                              proxy._circuit_summary(), decomposed.global_phase)


def _restore_hamiltonian(num_qubits, terms):
    ham = Hamiltonian(num_qubits)
    for pauli, coefficient in terms:
        ham.add_term(PauliString.from_str(pauli), coefficient)
    return ham


def _reduce_hamiltonian(ham):
    return _restore_hamiltonian, (ham.num_qubits,
                                  [(str(pauli), coefficient) for pauli, coefficient in ham.terms])


def _reduce_parameter(parameter):
    return Parameter, (str(parameter),)


def _reduce_qubit(qubit):
    return Qubit, (qubit.index,)


def _reduce_pauli(pauli):
    return PauliString.from_str, (str(pauli),)


def register_native_reducers():
    """Use the outer pickler's memo and Python-class support for estimator state.

    Register before traversal so native aliases can precede the estimator too.
    Keep any reducers already supplied by the application.
    """
    for native_type, reducer in {Circuit: _reduce_circuit, Parameter: _reduce_parameter,
                                 Qubit: _reduce_qubit, Hamiltonian: _reduce_hamiltonian,
                                 PauliString: _reduce_pauli}.items():
        copyreg.dispatch_table.setdefault(native_type, reducer)
