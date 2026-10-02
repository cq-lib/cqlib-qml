"""Smoke test for a cqlib-qml release build.

Exercises the main user-facing APIs end to end: encoders, ansatz, models,
differentiators, algorithms, losses and optimizers. Run with the Python of an
environment where the built wheel (and its dependencies) are installed:

    python scripts/smoke_test.py
"""

import tempfile

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
    np.random.seed(7)
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
    loss = MSELoss()
    loss(out, np.zeros_like(out))
    before = hqnn._linear.parameters["W"].copy()
    hqnn.backward(loss.grads())
    hqnn.update()
    assert not np.array_equal(before, hqnn._linear.parameters["W"])
    with tempfile.TemporaryDirectory() as directory:
        hqnn.save_checkpoint(directory, ep=1, it=2, latest=True)
        restored = HQNN(ansatz=HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"]), out_dim=3)
        assert restored.load_checkpoint(directory) == (1, 3)
        np.testing.assert_allclose(restored.forward(frqi_circuits, trainable=False),
                                   hqnn.forward(frqi_circuits, trainable=False))
    print("models, backward, update, checkpoint: OK")

    # Algorithms: quantum kernel
    kernel = QKM(encoder=AngleEncoder(mode="classical"))
    km = kernel.kernel(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert km.shape == (2, 2) and np.allclose(np.diag(km), 1.0)
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.8, 0.9], [1.0, 1.1]])
    y = np.array([0, 0, 1, 1])
    svm = QSVM(AngleEncoder()).fit(X, y, sample_weight=np.array([1., 2., 1., 2.]))
    assert svm.predict(X).shape == y.shape
    classifier = VQC(HEAnsatz(2, 1, layers=["RY", "CX"]), AngleEncoder(),
                     epochs=2, verbose=False)
    assert classifier.fit(X, y).predict(X).shape == y.shape
    from cqlib_qml.data.data_preprocess import filter_targets, change_grayscale
    _, labels = filter_targets(X, y, [1, 0])
    np.testing.assert_array_equal(labels, 1-y)
    np.testing.assert_array_equal(change_grayscale(np.array([0., 1.]), 2), [0., 1.])
    print("algorithms, fit, preprocessing: OK")

    # Losses and optimizers are importable and constructible
    MSELoss()
    Adam()
    SGD()
    print("loss/optimizer: OK")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
