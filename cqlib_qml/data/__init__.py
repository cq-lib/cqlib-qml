# cqlib_qml/data/__init__.py
"""
Data loading and preprocessing utilities for quantum machine learning.

This module provides tools for loading, preprocessing, and batching data
for quantum machine learning workflows. It includes dataset management,
data loading, and preprocessing functions specifically designed for
quantum image encoding.

Key Features:
    - Custom Dataset and DataLoader classes for quantum data
    - MNIST dataset preprocessing with quantum encoding
    - Image downscaling and grayscale conversion
    - Conflict removal for classification datasets
    - Integration with quantum encoders

Examples:
    >>> from cqlib_qml.data import Dataset, DataLoader
    >>> from cqlib_qml.encoder import FRQI
    >>> from cqlib_qml.data.data_preprocess import get_mnist_dataloader
    >>>
    >>> # Create a custom dataset
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> dataset = Dataset(X, y)
    >>>
    >>> # Create a data loader
    >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     print(batch)
    >>>
    >>> # Load MNIST with quantum encoding
    >>> encoder = FRQI(n_pixels=16, grayscale=2)
    >>> train_loader, test_loader = get_mnist_dataloader(
    ...     classes=[0, 1],
    ...     resize=(4, 4),
    ...     encoding=encoder,
    ...     batch_size=32
    ... )
"""

from .data import Dataset, DataLoader
from .data_preprocess import (
    filter_targets,
    downscale,
    remove_conflict,
    binary_img,
    change_grayscale,
    encoding_img,
    get_mnist_dataloader,
)

__all__ = [
    "Dataset",
    "DataLoader",
    "filter_targets",
    "downscale",
    "remove_conflict",
    "binary_img",
    "change_grayscale",
    "encoding_img",
    "get_mnist_dataloader",
]
