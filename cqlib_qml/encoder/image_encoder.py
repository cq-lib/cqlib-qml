# cqlib_qml/encoder/image_encoder.py
"""
Base class for quantum image encoders.

This module provides the abstract base class for all image encoders,
defining the common interface for quantum image encoding methods.

Examples:
    >>> from cqlib_qml.encoder import ImageEncoder
    >>>
    >>> # All image encoders inherit from ImageEncoder
    >>> from cqlib_qml.encoder import FRQI, NEQR, QubitLattice
    >>> isinstance(FRQI(16), ImageEncoder)
    True
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class ImageEncoder(ABC):
    """
    Base class for all quantum image encoders.

    This abstract class defines the interface that all image encoders
    must implement. It provides common functionality for handling
    image batches and validation.

    Args:
        n_pixels (int): Number of pixels in the image.

    Attributes:
        _n_pixels (int): Number of pixels.
        _n_channel (int): Number of image channels.

    Note:
        User-defined image encoders should inherit from this class
        and implement the required abstract methods.
    """

    def __init__(self, n_pixels: int):
        """
        Initialize an ImageEncoder instance.

        Args:
            n_pixels (int): Number of pixels in the image.
        """
        self._n_pixels = n_pixels
        self._n_channel = None

    @abstractmethod
    def __call__(self, imgs, **kwargs):
        """
        Encode images into quantum circuits.

        Args:
            imgs (Union[list, np.ndarray]): Input image(s).
            **kwargs: Additional encoding parameters.

        Returns:
            Union[Circuit, list]: Encoded quantum circuit(s).
        """
        raise NotImplementedError

    @abstractmethod
    def _construct_encoder(self):
        """
        Construct the encoding circuit.

        Returns:
            Circuit: Encoded quantum circuit.
        """
        raise NotImplementedError

    def __str__(self):
        """Return string representation of the encoder."""
        raise NotImplementedError

    def _is_imgs_batch(self, imgs) -> bool:
        """
        Check if input is a batch of images.

        Args:
            imgs: Input image(s).

        Returns:
            bool: True if input is a batch, False otherwise.

        Raises:
            TypeError: If input type is not supported.
            ValueError: If image dimensions are invalid.

        Examples:
            >>> encoder = FRQI(16)
            >>> img = np.random.rand(4, 4)  # Single image
            >>> encoder._is_imgs_batch(img)  # False
            >>> imgs = np.random.rand(8, 4, 4)  # Batch
            >>> encoder._is_imgs_batch(imgs)  # True
        """
        if isinstance(imgs, list):
            return True
        elif isinstance(imgs, (np.ndarray, torch.Tensor)):
            dim = len(imgs.shape)
            if dim == 2:
                return False
            elif dim == 3:
                return True if self._n_channel == 1 else False
            elif dim == 4:
                if self._n_channel == 1:
                    raise ValueError(
                        f"Expected single-channel images with shape (batch, height, width), "
                        f"but got 4D array with shape {imgs.shape}. For single-channel images, "
                        f"use 3D array (batch, height, width)."
                    )
                if imgs.shape[3] != self._n_channel:
                    raise ValueError(f"Expected {self._n_channel} channels, but got {imgs.shape[3]} channels.")
                return True
            else:
                raise ValueError(
                    f"Invalid image data dimension: {dim}. "
                    f"Expected 2D (single image), 3D (batch of single-channel images), "
                    f"or 4D (batch of multi-channel images)."
                )
        else:
            raise TypeError(f"Expected list, np.ndarray, or torch.Tensor, " f"got {type(imgs).__name__}.")
