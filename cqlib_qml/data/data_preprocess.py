# cqlib_qml/data/data_preprocess.py
"""
Data preprocessing utilities for quantum machine learning.

This module provides preprocessing functions for preparing data,
particularly images, for quantum encoding and training. It includes
functions for filtering, resizing, grayscale conversion, and
integration with quantum encoders.

Key Functions:
    - filter_targets: Select specific classes from a dataset
    - downscale: Resize images to target dimensions
    - remove_conflict: Remove data samples with conflicting labels
    - binary_img: Convert images to binary (0/1) format
    - change_grayscale: Quantize images to a specified grayscale level
    - encoding_img: Apply quantum encoding to images
    - get_mnist_dataloader: Complete pipeline for MNIST data loading

Examples:
    >>> from cqlib_qml.data.data_preprocess import get_mnist_dataloader
    >>> from cqlib_qml.encoder import FRQI
    >>>
    >>> encoder = FRQI(n_pixels=16, grayscale=2)
    >>> train_loader, test_loader = get_mnist_dataloader(
    ...     classes=[0, 1, 2],
    ...     resize=(4, 4),
    ...     encoding=encoder,
    ...     batch_size=32,
    ...     grayscale=2
    ... )
"""

import collections
import tqdm
from torchvision import datasets, transforms
import numpy as np

from .data import *


def filter_targets(X: np.ndarray, Y: np.ndarray, classes: list) -> tuple:
    """
    Filter dataset to keep only specified classes.

    This function selects samples that belong to the specified classes
    and remaps labels to consecutive integers starting from 0.

    Args:
        X (np.ndarray): Input data array.
        Y (np.ndarray): Label array.
        classes (list): List of class labels to keep.

    Returns:
        tuple: (X_filtered, Y_filtered) where:
            - X_filtered (np.ndarray): Filtered data array.
            - Y_filtered (np.ndarray): Remapped label array.

    Examples:
        >>> # Keep only digits 0, 1, 2 from MNIST
        >>> X, Y = filter_targets(X_train, Y_train, [0, 1, 2])
        >>> print(np.unique(Y))
        [0, 1, 2]
    """
    idx = Y == classes[0]
    for i in range(1, len(classes)):
        idx = idx | (Y == classes[i])
    X, Y = (X[idx], Y[idx])
    for i in range(len(Y)):
        for j in range(len(classes)):
            if Y[i] == classes[j]:
                Y[i] = j
    return X, Y


def downscale(X: np.ndarray, resize: tuple) -> np.ndarray:
    """
    Downscale images to target resolution.

    Uses torchvision's Resize transform to resize images.

    Args:
        X (np.ndarray): Input image array.
        resize (tuple): Target size (height, width).

    Returns:
        np.ndarray: Resized image array.

    Examples:
        >>> # Resize MNIST images from 28x28 to 4x4
        >>> X_resized = downscale(X, (4, 4))
        >>> print(X_resized.shape)
        (1000, 4, 4)
    """
    transform = transforms.Resize(size=resize, antialias=False)
    X = transform(X)
    return X


def remove_conflict(X: np.ndarray, Y: np.ndarray, resize: tuple) -> tuple:
    """
    Remove samples with conflicting labels.

    For each unique image, if it appears with multiple labels, all
    conflicting samples are removed. This ensures label consistency.

    Args:
        X (np.ndarray): Image array.
        Y (np.ndarray): Label array.
        resize (tuple): Original image dimensions.

    Returns:
        tuple: (X_cleaned, Y_cleaned) with conflicting samples removed.

    Examples:
        >>> # Remove ambiguous samples
        >>> X_clean, Y_clean = remove_conflict(X, Y, (4, 4))
    """
    x_dict = collections.defaultdict(set)
    for x, y in zip(X, Y):
        x_dict[tuple(x.numpy().flatten())].add(y.item())
    X_rmcon = []
    Y_rmcon = []
    for x in x_dict.keys():
        if len(x_dict[x]) == 1:
            X_rmcon.append(np.array(x).reshape(resize))
            Y_rmcon.append(list(x_dict[x])[0])
    X = np.array(X_rmcon) / 255.0
    Y = np.array(Y_rmcon)
    return X, Y


def binary_img(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert images to binary format.

    Args:
        X (np.ndarray): Input image array.
        threshold (float, optional): Threshold for binarization.
            Defaults to 0.5.

    Returns:
        np.ndarray: Binary image array (0 or 1).

    Examples:
        >>> # Convert grayscale to binary
        >>> X_binary = binary_img(X, threshold=0.3)
        >>> print(np.unique(X_binary))
        [0, 1]
    """
    X = X > threshold
    X = X.astype(np.int8)
    return X


def change_grayscale(X: np.ndarray, grayscale: int) -> np.ndarray:
    """
    Quantize images to a specified number of grayscale levels.

    Args:
        X (np.ndarray): Input image array in range [0, 1].
        grayscale (int): Number of grayscale levels. Must be >= 2 and <= 256.

    Returns:
        np.ndarray: Quantized image array.

    Raises:
        AssertionError: If grayscale is not between 2 and 256.

    Examples:
        >>> # Reduce to 4 grayscale levels
        >>> X_quantized = change_grayscale(X, 4)
        >>> print(np.unique(X_quantized))
        [0.0, 0.333, 0.667, 1.0]
    """
    if grayscale < 2 or grayscale > 256:
        raise ValueError("grayscale should be between 2 and 256")

    grays = np.linspace(0, 1, grayscale)
    thresholds = np.linspace(0, 1, grayscale + 1)
    for i in range(grayscale):
        X[np.logical_and(X >= thresholds[i], X <= thresholds[i + 1])] = grays[i]
    return X


def encoding_img(X: np.ndarray, encoding) -> list:
    """
    Apply quantum encoding to images.

    Args:
        X (np.ndarray): Image array.
        encoding: Quantum encoder instance (e.g., FRQI, NEQR).

    Returns:
        list: List of quantum circuits for each image.

    Examples:
        >>> from cqlib_qml.encoder import FRQI
        >>> encoder = FRQI(n_pixels=16, grayscale=2)
        >>> circuits = encoding_img(X, encoder)
        >>> len(circuits) == len(X)
        True
    """
    data_circuits = []
    for i in tqdm.tqdm(range(len(X))):
        data_circuit = encoding(X[i])
        data_circuits.append(data_circuit)
    return data_circuits


def get_mnist_dataloader(classes: list, resize: tuple, encoding, batch_size: int = 32, grayscale: int = 2) -> tuple:
    """
    Complete pipeline for loading and preprocessing MNIST data.

    This function provides an end-to-end pipeline for preparing MNIST
    data for quantum machine learning. It handles:
        1. Loading MNIST dataset
        2. Filtering selected classes
        3. Downscaling images
        4. Removing conflicting samples
        5. Grayscale quantization
        6. Quantum encoding
        7. Creating DataLoaders

    Args:
        classes (list): List of digit classes to keep (e.g., [0, 1]).
        resize (tuple): Target image size (height, width).
            Must be 2^n x 2^n for FRQI/NEQR.
        encoding: Quantum encoder instance.
        batch_size (int, optional): Batch size for training.
            Defaults to 32.
        grayscale (int, optional): Number of grayscale levels.
            Defaults to 2.

    Returns:
        tuple: (train_loader, test_loader) where:
            - train_loader (DataLoader): Training data loader.
            - test_loader (DataLoader): Test data loader.

    Raises:
        ValueError: If encoding fails or data is invalid.

    Examples:
        >>> from cqlib_qml.encoder import FRQI
        >>> from cqlib_qml.data.data_preprocess import get_mnist_dataloader
        >>>
        >>> encoder = FRQI(n_pixels=16, grayscale=2)
        >>> train_loader, test_loader = get_mnist_dataloader(
        ...     classes=[0, 1, 2],
        ...     resize=(4, 4),
        ...     encoding=encoder,
        ...     batch_size=32,
        ...     grayscale=2
        ... )
        >>>
        >>> for X_batch, y_batch in train_loader:
        ...     print(X_batch.shape, y_batch.shape)
        (32, 16) (32,)

    Note:
        Images are automatically normalized to [0, 1] range and the
        original MNIST dimensions (28x28) are downscaled to the target
        resolution.
    """
    # Load MNIST data
    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)
    train_X = train_data.data
    train_Y = train_data.targets
    test_X = test_data.data
    test_Y = test_data.targets

    # Filter target classes
    train_X, train_Y = filter_targets(train_X, train_Y, classes)
    test_X, test_Y = filter_targets(test_X, test_Y, classes)

    # Downscale images
    train_X = downscale(train_X, resize)
    test_X = downscale(test_X, resize)

    # Remove conflicting samples
    train_X, train_Y = remove_conflict(train_X, train_Y, resize)
    test_X, test_Y = remove_conflict(test_X, test_Y, resize)

    # Quantize grayscale levels
    train_X = change_grayscale(train_X, grayscale)
    test_X = change_grayscale(test_X, grayscale)

    # Apply quantum encoding
    train_X = encoding_img(train_X, encoding)
    test_X = encoding_img(test_X, encoding)

    # Create datasets and dataloaders
    train_dataset = Dataset(train_X, train_Y)
    test_dataset = Dataset(test_X, test_Y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader
