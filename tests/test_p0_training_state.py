"""Acceptance tests for reproducible retraining and exact batch-boundary resume."""
from copy import deepcopy
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from cqlib.circuit import Parameter

from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import Ansatz, HEAnsatz
from cqlib_qml.data import DataLoader, Dataset
from cqlib_qml.encoder import AngleEncoder
from cqlib_qml.optimizer import SGD, Adam, OptimizerBase
from cqlib_qml.scheduler import ExponentialScheduler, KingScheduler
from cqlib_qml.layer import Layer, Linear
from cqlib_qml.models import Module


X = np.array([[.1, .3], [.3, -.1], [.4, .2], [.7, .5], [.8, -.2], [.9, .2], [.6, .7]])
Y = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'b'])



def test_qnn_and_hqnn_owned_random_initialization():
    from cqlib_qml.models import QNN, HQNN
    circuits = AngleEncoder()(X)
    for cls, kwargs in [(QNN, {}), (HQNN, {'out_dim': 2})]:
        first = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        second = cls(HEAnsatz(2, 1, layers=['RY']), readouts=[0], random_state=23, **kwargs)
        np.testing.assert_array_equal(first.forward(circuits), second.forward(circuits))
