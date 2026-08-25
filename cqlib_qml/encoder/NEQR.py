# cqlib_qml/encoder/NEQR.py
"""
NEQR (Novel Enhanced Quantum Representation) encoder.

NEQR is an enhanced quantum image representation that uses q qubits for
color information instead of 1 qubit in FRQI. This allows representation
of 2^q grayscale levels with improved accuracy.

References:
    - Zhang, Y., et al. (2013). "NEQR: a novel enhanced quantum
      representation of digital images." Quantum Information Processing.

Examples:
    >>> from cqlib_qml.encoder import NEQR
    >>> import numpy as np
    >>>
    >>> # Create encoder for 4x4 image with 4 grayscale levels
    >>> encoder = NEQR(n_pixels=16, grayscale=4)
    >>>
    >>> # Encode image
    >>> img = np.random.randint(0, 4, (4, 4))
    >>> circuit = encoder(img)
"""

import numpy as np
from cqlib.circuit import Circuit
from .FRQI import FRQI


class NEQR(FRQI):
    """
    NEQR (Novel Enhanced Quantum Representation) encoder.

    Encodes 2^n × 2^n images into quantum circuits using (2n + q) qubits,
    where q = log2(grayscale). This provides a more accurate representation
    of grayscale images compared to FRQI.

    Args:
        n_pixels (int): Total number of pixels.
            Must be a perfect square and a power of 4.
        grayscale (int, optional): Number of grayscale levels.
            Must be a power of 2 (2, 4, 8, 16, ...). Defaults to 2.

    Attributes:
        _n_pixels (int): Number of pixels.
        _n_pos_qubits (int): Number of position qubits (log2(n_pixels)).
        _n_color_qubits (int): Number of color qubits (log2(grayscale)).
        _n_qubits (int): Total number of qubits.
        _grayscale (int): Number of grayscale levels.

    Raises:
        ValueError: If grayscale is not a power of 2.

    Examples:
        >>> # For 2x2 image with 4 grayscale levels
        >>> encoder = NEQR(n_pixels=4, grayscale=4)
        >>> print(encoder._n_color_qubits)  # log2(4) = 2
        2
        >>> print(encoder._n_qubits)  # 2n + q = 2 + 2 = 4
        4
    """

    def __init__(self, n_pixels: int, grayscale: int = 2):
        """
        Initialize an NEQR encoder.

        Args:
            n_pixels (int): Number of pixels. Must be 2^n × 2^n.
            grayscale (int): Number of grayscale levels. Must be a power of 2.

        Raises:
            ValueError: If grayscale is not a power of 2.
        """
        super(NEQR, self).__init__(n_pixels, grayscale)
        self._n_color_qubits = int(np.log2(grayscale))
        self._n_qubits = self._n_pos_qubits + self._n_color_qubits
        if 1 << self._n_color_qubits != grayscale:
            raise ValueError("The gray scale of the image should be 2^q.")

    def __str__(self):
        return "NEQR(n_qubits={}, color qubits={}, grayscale={})".format(
            self._n_qubits, self._n_color_qubits, self._grayscale
        )

    def __call__(self, imgs, use_qic: bool = False):
        """
        Encode images using NEQR.

        Args:
            imgs (Union[list, np.ndarray]): Input image(s).
            use_qic (bool, optional): Whether to use QIC optimization.
                Defaults to False.

        Returns:
            Union[Circuit, list]: Encoded quantum circuit(s).

        Examples:
            >>> img = np.random.randint(0, 4, (4, 4))
            >>> circuit = encoder(img)
        """
        if not self._is_imgs_batch(imgs):
            imgs = [imgs]

        enc_cirs = []
        for img in imgs:
            img = self._img_preprocess(img, flatten=True)
            encoder = self._construct_encoder(img, use_qic)
            enc_cirs.append(encoder)

        return enc_cirs[0] if len(enc_cirs) == 1 else enc_cirs

    def _construct_encoder(self, img: np.ndarray, use_qic: bool) -> Circuit:
        """
        Construct the NEQR encoding circuit.

        Args:
            img (np.ndarray): Flattened image.
            use_qic (bool): Whether to use QIC optimization.

        Returns:
            Circuit: Encoded quantum circuit.
        """
        # Step 1: Prepare |H⟩ state on position qubits
        encoder = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            encoder.h(qid)

        # Step 2: Encode color information
        if use_qic:
            neqr_circuit = Circuit(self._n_qubits)
            groups = self._get_groups(img)
            for i in range(self._n_color_qubits):
                sub_circuit = self._construct_qic_circuit(groups[i], rotate=False, gid=i)
                neqr_circuit.compose(sub_circuit)
        else:
            neqr_circuit = self._construct_circuit(img, rotate=False)
        encoder.compose(neqr_circuit)
        return encoder

    def _get_groups(self, img: np.ndarray) -> np.ndarray:
        """
        Group pixels by color bits.

        Args:
            img (np.ndarray): Flattened image.

        Returns:
            np.ndarray: Binary matrix of shape (n_color_qubits, n_pixels).
        """
        img_dict = self._get_img_dict(img, bin_key=True)
        groups = np.zeros((self._n_color_qubits, self._n_pixels), dtype=np.bool_)
        for i in range(self._n_color_qubits):
            for bin_color in img_dict.keys():
                if bin_color[i] == "1":
                    for pixel in img_dict[bin_color]:
                        groups[i][pixel] = 1
        return groups
