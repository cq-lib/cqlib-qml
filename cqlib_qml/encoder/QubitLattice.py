# cqlib_qml/encoder/QubitLattice.py
"""
QubitLattice encoding for binary images.

QubitLattice provides a straightforward encoding where each pixel of a
binary image is mapped to one qubit. This is the simplest image encoding
method but requires many qubits for large images.

Examples:
    >>> from cqlib_qml.encoder import QubitLattice
    >>> import numpy as np
    >>>
    >>> # Create encoder for 4x4 binary image
    >>> encoder = QubitLattice(n_pixels=16)
    >>>
    >>> # Encode binary image
    >>> img = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    >>> circuit = encoder(img)
    >>> print(circuit.num_qubits)  # 16 qubits
    16
"""

import numpy as np
from cqlib.circuit import Circuit
from .image_encoder import ImageEncoder


class QubitLattice(ImageEncoder):
    """
    QubitLattice encoder for binary images.

    Maps each pixel of a binary image to a separate qubit. Pixels with
    value 1 are encoded as |1⟩, pixels with value 0 as |0⟩.

    Note:
        Only supports binary images (2 grayscale levels).

    Args:
        n_pixels (int): Number of pixels in the image.

    Attributes:
        _n_pixels (int): Number of pixels.

    Raises:
        ValueError: If image is not binary or has wrong number of pixels.

    Examples:
        >>> encoder = QubitLattice(n_pixels=9)
        >>> img = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        >>> circuit = encoder(img)
    """

    def __init__(self, n_pixels: int):
        """
        Initialize a QubitLattice encoder.

        Args:
            n_pixels (int): Number of pixels.
        """
        super(QubitLattice, self).__init__(n_pixels)
        self._n_channel = 1

    def __call__(self, imgs):
        """
        Encode binary images using QubitLattice.

        Args:
            imgs (Union[list, np.ndarray]): Input image(s).

        Returns:
            Union[Circuit, list]: Encoded quantum circuit(s).

        Raises:
            ValueError: If image is not binary or has wrong dimensions.

        Examples:
            >>> img = np.array([[0, 1], [1, 0]])  # 2x2 binary image
            >>> circuit = encoder(img)
        """
        if not self._is_imgs_batch(imgs):
            imgs = [imgs]

        enc_cirs = []
        for img in imgs:
            img = img.flatten()
            self._validate_img(img)
            encoder = self._construct_encoder(img)
            enc_cirs.append(encoder)

        return enc_cirs[0] if len(enc_cirs) == 1 else enc_cirs

    def __str__(self):
        return "QubitLattice(n_qubits={})".format(self._n_pixels)

    def _construct_encoder(self, img: np.ndarray) -> Circuit:
        """
        Construct the QubitLattice encoding circuit.

        Args:
            img (np.ndarray): Flattened binary image.

        Returns:
            Circuit: Encoded quantum circuit.
        """
        encoder = Circuit(self._n_pixels)
        for i in range(img.shape[0]):
            if img[i] == 1:
                encoder.x(i)
        return encoder

    def _validate_img(self, img: np.ndarray) -> None:
        """
        Validate image for QubitLattice encoding.

        Args:
            img (np.ndarray): Flattened image.

        Raises:
            ValueError: If pixel count doesn't match or image is not binary.
        """
        if img.shape[0] != self._n_pixels:
            raise ValueError(
                f"The number of pixels ({img.shape[0]}) should be equal to "
                f"the number of data qubits ({self._n_pixels})."
            )
        if not np.isin(img, [0, 1]).all():
            raise ValueError("QubitLattice only supports binary images.")
