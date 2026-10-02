import numpy as np
import pytest
from cqlib.circuit import Circuit

from cqlib_qml.encoder import (
    AmplitudeEncoder,
    AngleEncoder,
    BasisEncoder,
    FRQI,
    NEQR,
    QubitLattice,
    ZZFeatureEncoder,
)


class TestAmplitudeEncoder:
    def test_single_sample(self):
        encoder = AmplitudeEncoder()
        data = np.array([1.0, 0.0])
        circuits = encoder(data)
        assert isinstance(circuits, list)
        assert len(circuits) == 1
        assert isinstance(circuits[0], Circuit)
        assert circuits[0].num_qubits == 1

    def test_multiple_samples(self):
        encoder = AmplitudeEncoder()
        data = np.array([[1.0, 0.0], [0.0, 1.0]])
        circuits = encoder(data)
        assert len(circuits) == 2
        for circ in circuits:
            assert circ.num_qubits == 1

    def test_zero_vector_raises(self):
        encoder = AmplitudeEncoder()
        data = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="Cannot encode zero vector"):
            encoder(data)


class TestAngleEncoder:
    def test_classical_mode(self):
        encoder = AngleEncoder(mode="classical")
        data = np.array([0.5, 0.3])
        circuits = encoder(data)
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 2

    def test_dense_mode(self):
        encoder = AngleEncoder(mode="dense")
        data = np.array([0.5, 0.3, 0.7])
        circuits = encoder(data)
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 2

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Angle encoding only supports two modes"):
            AngleEncoder(mode="invalid")


class TestBasisEncoder:
    def test_single_integer(self):
        encoder = BasisEncoder()
        data = np.array([5])
        circuits = encoder(data)
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 3

    def test_multiple_integers(self):
        encoder = BasisEncoder()
        data = np.array([0, 1, 2, 3])
        circuits = encoder(data)
        assert len(circuits) == 4

    def test_negative_integer_raises(self):
        encoder = BasisEncoder()
        data = np.array([-1])
        with pytest.raises(ValueError, match="Basis encoding only supports encoding non-negative integers"):
            encoder(data)

    def test_float_integer_raises(self):
        encoder = BasisEncoder()
        data = np.array([1.5])
        with pytest.raises(ValueError, match="Basis encoding only supports encoding non-negative integers"):
            encoder(data)


class TestFRQI:
    def test_init(self):
        encoder = FRQI(n_pixels=4, grayscale=2)
        assert encoder._n_pos_qubits == 2
        assert encoder._n_qubits == 3
        assert encoder._n_pixels == 4

    def test_invalid_pixels(self):
        with pytest.raises(ValueError, match="Only support images with a resolution of 2\\^n x 2\\^n\\."):
            FRQI(n_pixels=6)  # Not 2^n x 2^n

    def test_single_image(self):
        encoder = FRQI(n_pixels=4, grayscale=2)
        img = np.array([[0, 1], [1, 0]], dtype=np.float32)
        circuits = encoder(img)
        assert isinstance(circuits, Circuit)
        assert circuits.num_qubits == 3

    def test_batch_images(self):
        encoder = FRQI(n_pixels=4, grayscale=2)
        imgs = np.array([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ], dtype=np.float32)
        circuits = encoder(imgs)
        assert isinstance(circuits, list)
        assert len(circuits) == 2
        for circ in circuits:
            assert circ.num_qubits == 3

    def test_use_qic(self):
        encoder = FRQI(n_pixels=4, grayscale=2)
        img = np.array([[0, 1], [1, 0]], dtype=np.float32)
        circuit = encoder(img, use_qic=True)
        assert isinstance(circuit, Circuit)
        assert circuit.num_qubits == 3


class TestNEQR:
    def test_init(self):
        encoder = NEQR(n_pixels=4, grayscale=4)
        assert encoder._n_color_qubits == 2
        assert encoder._n_qubits == 4

    def test_grayscale_power_of_two(self):
        with pytest.raises(ValueError, match="The gray scale of the image should be 2\\^q\\."):
            NEQR(n_pixels=4, grayscale=3)  # Not 2^q

    def test_single_image(self):
        encoder = NEQR(n_pixels=4, grayscale=2)
        img = np.array([[0, 1], [1, 0]], dtype=np.float32)
        circuits = encoder(img)
        assert isinstance(circuits, Circuit)

    def test_get_groups_float_pixels(self):
        encoder = NEQR(n_pixels=4, grayscale=4)
        img = np.array([0, 1, 2, 3], dtype=np.float32)
        groups = encoder._get_groups(img)
        expected = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.bool_)
        assert np.array_equal(groups, expected)

    def test_use_qic(self):
        encoder = NEQR(n_pixels=4, grayscale=4)
        img = np.array([[0, 1], [2, 3]], dtype=np.float32)
        circuit = encoder(img, use_qic=True)
        assert isinstance(circuit, Circuit)
        assert circuit.num_qubits == 4


class TestQubitLattice:
    def test_binary_image(self):
        encoder = QubitLattice(n_pixels=4)
        img = np.array([[0, 1], [1, 0]])
        circuits = encoder(img)
        assert isinstance(circuits, Circuit)

    def test_batch_images(self):
        encoder = QubitLattice(n_pixels=4)
        imgs = np.array([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]]
        ])
        circuits = encoder(imgs)
        assert isinstance(circuits, list)
        assert len(circuits) == 2
        for circ in circuits:
            assert isinstance(circ, Circuit)
            assert circ.num_qubits == 4

    def test_non_binary_raises(self):
        encoder = QubitLattice(n_pixels=4)
        img = np.array([[0, 2], [1, 0]])
        with pytest.raises(ValueError, match="QubitLattice only supports binary images"):
            encoder(img)

    def test_wrong_pixels_raises(self):
        encoder = QubitLattice(n_pixels=4)
        img = np.array([[0, 1, 0], [1, 0, 1]])
        with pytest.raises(ValueError, match="The number of pixels"):
            encoder(img)


class TestZZFeatureEncoder:
    def test_linear_entanglement(self):
        encoder = ZZFeatureEncoder(n_repeats=2, entanglement="linear")
        data = np.array([0.5, 0.3, 0.7])
        circuits = encoder(data)
        assert len(circuits) == 1
        assert circuits[0].num_qubits == 3

    def test_full_entanglement(self):
        encoder = ZZFeatureEncoder(n_repeats=1, entanglement="full")
        data = np.array([0.5, 0.3])
        circuits = encoder(data)
        assert circuits[0].num_qubits == 2

    def test_circular_entanglement(self):
        encoder = ZZFeatureEncoder(n_repeats=1, entanglement="circular")
        data = np.array([0.5, 0.3, 0.7])
        circuits = encoder(data)
        assert circuits[0].num_qubits == 3

    def test_invalid_entanglement(self):
        with pytest.raises(ValueError, match="ZZFeatureMap only supports three entanglements"):
            ZZFeatureEncoder(entanglement="invalid")