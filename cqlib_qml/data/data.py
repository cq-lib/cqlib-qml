# cqlib_qml/data/data.py
"""
Custom dataset and dataloader implementations.

This module provides custom Dataset and DataLoader classes for handling
quantum data in batch processing. The Dataset class supports multiple data
arrays, and the DataLoader handles batching, shuffling, and iteration.

Examples:
    >>> from cqlib_qml.data import Dataset, DataLoader
    >>>
    >>> # Create dataset with multiple data sources
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> dataset = Dataset(X, y)
    >>>
    >>> # Iterate over dataset
    >>> for sample in dataset:
    ...     print(sample)
    (1, 0)
    (2, 1)
    ...
    >>>
    >>> # Use DataLoader for batching
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     print(batch)
    (array([3, 5]), array([0, 0]))
    (array([2, 4]), array([1, 1]))
    (array([1]), array([0]))
"""

import numpy as np


class Dataset:
    """
    A flexible dataset class for managing multiple data arrays.

    This class provides a map-style dataset that can handle multiple
    data sources (e.g., features and labels) with consistent lengths.
    It supports indexing, item assignment, and length querying.

    Args:
        *datas (list): Variable number of data arrays or lists.
            All data must have the same length.

    Attributes:
        _datas (tuple): Internal storage of data arrays.

    Raises:
        ValueError: If the lengths of provided data arrays don't match.

    Examples:
        >>> # Single data source
        >>> X = np.array([[2, 3], [0.3, 7], [-33, 1.2], [6, 5]])
        >>> dataset = Dataset(X)
        >>> len(dataset)
        4
        >>>
        >>> # Multiple data sources (features and labels)
        >>> y = np.array([0, 1, 1, 0])
        >>> dataset = Dataset(X, y)
        >>> dataset[0]
        (array([2., 3.]), 0)
        >>>
        >>> # Batch indexing
        >>> dataset[0:2]
        (array([[2., 3.], [0.3, 7.]]), array([0, 1]))
    """

    def __init__(self, *datas: list):
        """
        Initialize a Dataset instance.

        Args:
            *datas (list): Variable number of data arrays or lists.
                All data must have the same length.

        Raises:
            ValueError: If the lengths of provided data arrays don't match.

        Examples:
            >>> X = np.array([1, 2, 3])
            >>> y = np.array([0, 1, 0])
            >>> dataset = Dataset(X, y)
        """
        if not all(len(datas[0]) == len(data) for data in datas):
            raise ValueError(
                f"Length mismatch between datas. Expected length: {len(datas[0])}, "
                f"but found varying lengths in the list."
            )
        self._datas = datas

    def __getitem__(self, index):
        """
        Get data by index or slice.

        Args:
            index: Index or slice to retrieve.

        Returns:
            tuple: Fetched data sample(s) for the given index.
                If index is an integer, returns a tuple of individual samples.
                If index is a slice, returns a tuple of arrays.

        Examples:
            >>> dataset = Dataset(X, y)
            >>> dataset[0]  # Single sample
            (array([2., 3.]), 0)
            >>> dataset[0:2]  # Slice
            (array([[2., 3.], [0.3, 7.]]), array([0, 1]))
        """
        return tuple(np.array(data)[index] for data in self._datas)

    def __setitem__(self, index, values):
        """
        Set data values by index.

        Args:
            index: Index to set.
            values: Values to set. Should be a tuple with one value per data source.

        Examples:
            >>> dataset[0] = (10, 1)  # Set first sample
        """
        for data, value in zip(self._datas, values):
            data[index] = value

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.

        Examples:
            >>> len(dataset)
            4
        """
        return len(self._datas[0])


class DataLoader:
    """
    Data loader for batch iteration over datasets.

    Provides an iterable interface for loading data in batches with
    optional shuffling and drop_last functionality.

    Args:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): Number of samples per batch.
            Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data at every epoch.
            Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
            Defaults to True.

    Attributes:
        dataset (Dataset): The underlying dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether shuffling is enabled.
        drop_last (bool): Whether to drop incomplete batches.

    Examples:
        >>> from cqlib_qml.data import Dataset, DataLoader
        >>> X = np.array([[2, 3], [0.3, 7], [-33, 1.2], [6, 5]])
        >>> y = np.array([0, 1, 1, 0])
        >>> dataset = Dataset(X, y)
        >>>
        >>> # Create dataloader with batch size 2
        >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
        >>> len(loader)
        2
        >>>
        >>> # Iterate over batches
        >>> for batch in loader:
        ...     print(batch)
        (array([[6. , 5. ], [0.3, 7. ]]), array([0, 1]))
        (array([[2. , 3. ], [-33., 1.2]]), array([0, 1]))
    """

    @property
    def dataset(self) -> Dataset:
        """The underlying dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """Set the dataset and reset iteration state."""
        self._dataset = dataset
        self._it = 0
        self._end = self.__len__()
        self._idx = np.arange(len(self._dataset))

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Initialize a DataLoader instance.

        Args:
            dataset (Dataset): Dataset to load from.
            batch_size (int, optional): Samples per batch. Defaults to 1.
            shuffle (bool, optional): Shuffle data each epoch. Defaults to True.
            drop_last (bool, optional): Drop incomplete batches. Defaults to True.

        Examples:
            >>> loader = DataLoader(dataset, batch_size=4, shuffle=False)
        """
        self._dataset = dataset
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last

        self._it = 0
        self._end = self.__len__()
        self._idx = np.arange(len(self._dataset))

    def __len__(self) -> int:
        """
        Get the number of batches in the dataloader.

        Returns:
            int: Number of batches.

        Examples:
            >>> len(loader)
            2
        """
        length = len(self._dataset)
        length = length // self._batch_size if self._drop_last else np.ceil(length / self._batch_size)
        return int(length)

    def __iter__(self):
        """
        Return an iterator for the dataset.

        Resets the iteration state and shuffles indices if enabled.

        Returns:
            DataLoader: The dataloader instance as an iterator.

        Examples:
            >>> for batch in loader:
            ...     process(batch)
        """
        self._it = 0
        if self._shuffle:
            self._idx = np.arange(len(self._dataset))
            np.random.shuffle(self._idx)
        return self

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
            tuple: Batch data for the current iteration.

        Raises:
            StopIteration: When all batches have been iterated.

        Examples:
            >>> next(loader)  # Get first batch
            (array([[6. , 5. ], [0.3, 7. ]]), array([0, 1]))
        """
        if self._it < self._end:
            ret_data = self._dataset[self._idx[self._it * self._batch_size : (self._it + 1) * self._batch_size]]
            self._it += 1
            return ret_data
        else:
            raise StopIteration
