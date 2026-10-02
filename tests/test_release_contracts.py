"""Behavior regressions for the release audit, independent of downloads."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC

from cqlib_qml.algorithms import QSVM, VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.data import DataLoader, Dataset
from cqlib_qml.data import data_preprocess as prep
from cqlib_qml.encoder import AngleEncoder, BasisEncoder, FRQI, NEQR
from cqlib_qml.loss import BCELoss, MSELoss, SoftmaxCrossEntropy
from cqlib_qml.models import HQNN, QNN
from cqlib_qml.optimizer import OptimizerInitializer, SGD
from cqlib_qml.scheduler import SchedulerInitializer, ExponentialScheduler
from cqlib.qis.state import Statevector


def ansatz():
    circuit = HEAnsatz(2, 1, layers=['RY'])
    circuit.set_measurement(readouts=[0])
    return circuit
















@pytest.mark.parametrize('encoder', [FRQI, NEQR])
@pytest.mark.parametrize('batch', [False, True])
def test_image_tensor_matches_numpy(encoder, batch):
    image = np.array([[0., 1.], [1., 0.]])
    if batch:
        image = np.stack([image, 1-image])
    tensor = torch.tensor(image, requires_grad=True)
    first, second = encoder(4)(image), encoder(4)(tensor)
    first = first if isinstance(first, list) else [first]
    second = second if isinstance(second, list) else [second]
    for a, b in zip(first, second):
        left, right = Statevector(a.num_qubits), Statevector(b.num_qubits)
        left.apply_circuit(a)
        right.apply_circuit(b)
        np.testing.assert_allclose(left.data, right.data)








def test_basis_integer_floats():
    circuits = BasisEncoder()(np.array([0., 1., 2.]))
    for value, circuit in enumerate(circuits):
        state = Statevector(circuit.num_qubits)
        state.apply_circuit(circuit)
        assert np.argmax(abs(state.data)) == value




















def test_import_does_not_patch_cqlib():
    import subprocess
    import sys
    code = '''from cqlib.circuit import ValueOperation
before = set(vars(ValueOperation))
import cqlib_qml
assert set(vars(ValueOperation)) == before
'''
    subprocess.run([sys.executable, '-c', code], check=True)
