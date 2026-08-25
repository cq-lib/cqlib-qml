import numpy as np
import pytest

from cqlib_qml.data import Dataset, DataLoader


class TestDataset:
    def test_init_single(self):
        x = np.array([1, 2, 3, 4])
        dataset = Dataset(x)
        assert len(dataset) == 4

    def test_init_multiple(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 0, 1])
        dataset = Dataset(x, y)
        assert len(dataset) == 4

    def test_getitem(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 0, 1])
        dataset = Dataset(x, y)
        data = dataset[0]
        assert data[0] == 1
        assert data[1] == 0

    def test_setitem(self):
        x = np.array([1, 2, 3, 4])
        dataset = Dataset(x)
        dataset[0] = (10,)
        assert dataset[0][0] == 10

    def test_length_mismatch_raises(self):
        x = np.array([1, 2, 3])
        y = np.array([0, 1])
        with pytest.raises(ValueError):
            Dataset(x, y)


class TestDataLoader:
    def test_init(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        dataset = Dataset(x)
        loader = DataLoader(dataset, batch_size=2)
        assert len(loader) == 3

    def test_drop_last(self):
        x = np.array([1, 2, 3, 4, 5])
        dataset = Dataset(x)
        loader = DataLoader(dataset, batch_size=2, drop_last=True)
        assert len(loader) == 2

    def test_no_drop_last(self):
        x = np.array([1, 2, 3, 4, 5])
        dataset = Dataset(x)
        loader = DataLoader(dataset, batch_size=2, drop_last=False)
        assert len(loader) == 3

    def test_iteration(self):
        x = np.array([1, 2, 3, 4])
        dataset = Dataset(x)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2
        np.testing.assert_array_equal(batches[0][0], np.array([1, 2]))
        np.testing.assert_array_equal(batches[1][0], np.array([3, 4]))

    def test_shuffle(self):
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5, 6])
        dataset = Dataset(x)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batches1 = list(loader)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batches2 = list(loader)
        flat1 = np.concatenate([batch[0] for batch in batches1])
        flat2 = np.concatenate([batch[0] for batch in batches2])
        assert not np.array_equal(flat1, flat2)

    def test_shuffle_with_multiple_datas(self):
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([0, 1, 0, 1, 0, 1])
        dataset = Dataset(x, y)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batches1 = list(loader)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batches2 = list(loader)
        flat1 = np.concatenate([batch[0] for batch in batches1])
        flat2 = np.concatenate([batch[0] for batch in batches2])
        assert not np.array_equal(flat1, flat2)
        flat_y1 = np.concatenate([batch[1] for batch in batches1])
        flat_y2 = np.concatenate([batch[1] for batch in batches2])
        assert not np.array_equal(flat_y1, flat_y2)
