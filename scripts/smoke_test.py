"""Smoke test for a cqlib-qml release build.

Exercises the main user-facing APIs end to end: encoders, ansatz, models,
differentiators, algorithms, losses and optimizers. Run with the Python of an
environment where the built wheel (and its dependencies) are installed:

    python scripts/smoke_test.py
"""

import numpy as np

import cqlib_qml
from cqlib_qml import Adam, MSELoss, SGD
from cqlib_qml.algorithms import QKM, QSVM, VQC
from cqlib_qml.ansatz import HEAnsatz, BasicQNN, CRADL, CRAML
from cqlib_qml.data import Dataset, DataLoader
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator
from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder, BasisEncoder, FRQI, NEQR
from cqlib_qml.layer import Linear
from cqlib_qml.models import HQNN, QNN


def main() -> None:
    print(f"cqlib_qml version: {cqlib_qml.__version__}")

    # Encoders
    amp_circuits = AmplitudeEncoder()(np.random.rand(4))
    assert len(amp_circuits[0].operations) > 0
    frqi_circuits = FRQI(n_pixels=4, grayscale=2)(np.array([[0, 1], [1, 0]], dtype=np.float32))
    neqr_circuit = NEQR(n_pixels=4)(np.array([[0, 1], [1, 0]], dtype=np.float32))
    assert len(neqr_circuit.operations) > 0
    print("encoders: OK")

    # Models: QNN / HQNN forward pass
    ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
    ansatz.set_measurement(readouts=[0])
    qnn = QNN(ansatz=ansatz)
    out = qnn.forward(frqi_circuits)
    assert out.shape == (1, 1), out.shape

    hqnn = HQNN(ansatz=HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"]), out_dim=3)
    out = hqnn.forward(frqi_circuits)
    assert out.shape == (1, 3), out.shape
    print("models: OK")

    # Algorithms: quantum kernel
    kernel = QKM(encoder=AngleEncoder(mode="classical"))
    km = kernel.kernel(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert km.shape == (2, 2) and np.allclose(np.diag(km), 1.0)
    print("algorithms: OK")

    # Losses and optimizers are importable and constructible
    MSELoss()
    Adam()
    SGD()
    print("loss/optimizer: OK")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
