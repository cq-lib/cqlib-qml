"""Numerical, state-restoration and public API regressions from the audit."""
from copy import deepcopy
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from cqlib.circuit import Parameter
from cqlib.qis.state import Statevector
from cqlib_qml.algorithms import VQC, QSVM
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.encoder import AngleEncoder, QubitLattice, FRQI, NEQR
from cqlib_qml.layer import Linear
from cqlib_qml.layer.activation import SoftPlus
from cqlib_qml.models import Module
from cqlib_qml.loss import CrossEntropy, SoftmaxCrossEntropy
from cqlib_qml.optimizer import SGD, Adam, AdaGrad, RMSProp, OptimizerInitializer
from cqlib_qml.scheduler import KingScheduler, SchedulerInitializer, ConstantScheduler, NoamScheduler, ExponentialScheduler
from cqlib_qml.data.data_preprocess import downscale, remove_conflict, change_grayscale


@pytest.mark.parametrize('pixel', [0, 1])
def test_qubit_lattice_uniform_binary_state(pixel):
    circuit = QubitLattice(4)(np.full((2, 2), pixel))
    state = Statevector(4)
    state.apply_circuit(circuit)
    expected = np.zeros(16)
    expected[15 if pixel else 0] = 1
    np.testing.assert_allclose(state.data, expected)
    with pytest.raises(ValueError):
        QubitLattice(4)(np.array([[2, 3], [2, 3]]))


@pytest.mark.parametrize('cls', [FRQI, NEQR])
@pytest.mark.parametrize('image', [np.zeros((4, 4)), np.array([[np.inf, 0], [0, 0]]), np.zeros(4)])
def test_image_encoder_rejects_wrong_size_or_nonfinite(cls, image):
    with pytest.raises(ValueError):
        cls(n_pixels=4)(image)


@pytest.mark.parametrize('tensor', [False, True])
def test_preprocessing_accepts_numpy_and_torch(tensor):
    X = np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], [[255., 255.], [255., 255.]]])
    y = np.array([0, 1, 1])
    source = torch.as_tensor(X) if tensor else X
    labels = torch.as_tensor(y) if tensor else y
    resized = downscale(source, (1, 1))
    assert isinstance(resized, torch.Tensor if tensor else np.ndarray)
    assert resized.shape == (3, 1, 1)
    clean, targets = remove_conflict(source, labels, (2, 2))
    np.testing.assert_array_equal(clean, np.ones((1, 2, 2)))
    np.testing.assert_array_equal(targets, [1])
    empty, _ = remove_conflict(source[:2], labels[:2], (2, 2))
    assert empty.shape == (0, 2, 2)


def test_grayscale_does_not_mutate_input():
    original = np.array([.26, .51, .76])
    result = change_grayscale(original, 4)
    np.testing.assert_array_equal(original, [.26, .51, .76])
    np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1])
    assert not np.shares_memory(original, result)


def test_tutorial_neqr_example_encodes_integer_color_indices():
    from textwrap import dedent
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/tutorials/encoder.md').read_text().split('## NEQR:', 1)[1]
    code = text.split('### 使用示例', 1)[1].split('**输出：**', 1)[0]
    namespace = {}
    exec(dedent(code), namespace)
    assert namespace['circuit'].num_qubits == 6
    assert np.isin(namespace['img_quantized'], [0, 1, 2, 3]).all()


def test_mnist_preprocessing_to_neqr_uses_integer_color_indices(monkeypatch):
    from types import SimpleNamespace
    from cqlib_qml.data import data_preprocess as prep
    images = torch.tensor([[[0, 85], [170, 255]], [[85, 0], [255, 170]]], dtype=torch.uint8)
    labels = torch.tensor([0, 1])
    monkeypatch.setattr(prep.datasets, 'MNIST', lambda **kwargs: SimpleNamespace(data=images, targets=labels))
    encoder = NEQR(4, grayscale=4)
    train, test = prep.get_mnist_dataloader(classes=[0, 1], resize=(2, 2),
                                           encoding=encoder, batch_size=1, grayscale=4)
    for loader in [train, test]:
        assert len(loader) == 2
        for circuits, target in loader:
            circuit = circuits[0]
            state = Statevector(circuit.num_qubits)
            state.apply_circuit(circuit)
            assert np.count_nonzero(np.abs(state.data) > 1e-10) == 4
    with pytest.raises(ValueError, match='grayscale'):
        prep.get_mnist_dataloader(classes=[0, 1], resize=(2, 2), encoding=encoder, grayscale=2)
