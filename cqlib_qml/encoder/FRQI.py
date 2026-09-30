# cqlib_qml/encoder/FRQI.py
"""
FRQI (Flexible Representation of Quantum Images) encoder.

FRQI is a quantum image representation that encodes 2^n × 2^n images
using (2n + 1) qubits. It provides a flexible way to represent image
data as quantum states, enabling quantum image processing and
classification.

The encoding uses:
    - 2n qubits for position (n for x-coordinates, n for y-coordinates)
    - 1 qubit for color information

References:
    - Le, P. Q., et al. (2011). "A flexible representation of quantum
      images for polynomial preparation, image compression, and processing
      operations." Quantum Information Processing, 10(1), 63-84.

Examples:
    >>> from cqlib_qml.encoder import FRQI
    >>> import numpy as np
    >>>
    >>> # Create encoder for 4x4 image (16 pixels)
    >>> encoder = FRQI(n_pixels=16, grayscale=2)
    >>>
    >>> # Encode a single image
    >>> img = np.random.rand(4, 4)
    >>> circuit = encoder(img)
    >>>
    >>> # Encode batch of images
    >>> imgs = np.random.rand(8, 4, 4)
    >>> circuits = encoder(imgs)
"""

import numpy as np
from sympy import symbols
from sympy.logic.boolalg import to_dnf

from cqlib.circuit import Circuit, MCGate, StandardGate

from .image_encoder import ImageEncoder


class FRQI(ImageEncoder):
    """
    FRQI (Flexible Representation of Quantum Images) encoder.

    Encodes 2^n × 2^n images into quantum circuits using (2n + 1) qubits.

    The encoding consists of:
        1. Initializing all position qubits to |+⟩ states
        2. Applying controlled rotations based on pixel values

    Args:
        n_pixels (int): Total number of pixels in the image.
            Must be a perfect square and a power of 4 (4, 16, 64, ...).
        grayscale (int, optional): Number of grayscale levels.
            Defaults to 2.

    Attributes:
        _n_pixels (int): Number of pixels.
        _n_pos_qubits (int): Number of position qubits (log2(n_pixels)).
        _n_color_qubits (int): Number of color qubits (always 1 for FRQI).
        _n_qubits (int): Total number of qubits.
        _grayscale (int): Number of grayscale levels.
        _q_state (list): Current state of position qubits.

    Raises:
        ValueError: If n_pixels is not 2^n × 2^n.

    Examples:
        >>> # For a 2x2 image (4 pixels)
        >>> encoder = FRQI(n_pixels=4, grayscale=2)
        >>> img = np.array([[0, 1], [1, 0]])
        >>> circuit = encoder(img)
        >>> print(circuit.num_qubits)  # 2n + 1 = 3
        3
    """

    def __init__(self, n_pixels: int, grayscale: int = 2):
        """
        Initialize an FRQI encoder.

        Args:
            n_pixels (int): Number of pixels. Must be 2^n × 2^n.
            grayscale (int): Number of grayscale levels. Defaults to 2.

        Raises:
            ValueError: If n_pixels is not a perfect power of 4.
        """
        edge_len = int(np.sqrt(n_pixels))
        if not (edge_len**2 == n_pixels and 1 << int(np.log2(edge_len)) == edge_len):
            raise ValueError("Only support images with a resolution of 2^n x 2^n.")
        super(FRQI, self).__init__(n_pixels)
        self._n_channel = 1
        self._grayscale = grayscale
        self._n_color_qubits = 1
        self._n_pos_qubits = int(np.log2(n_pixels))
        self._n_qubits = self._n_pos_qubits + self._n_color_qubits
        self._q_state = [0] * self._n_pos_qubits

    def __call__(self, imgs, use_qic: bool = False):
        """
        Encode images using FRQI.

        Args:
            imgs (Union[list, np.ndarray]): Input image(s). Can be a single
                image (2D array) or a batch (3D or 4D array).
            use_qic (bool, optional): Whether to use Quantum Image
                Compression for optimization. Defaults to False.

        Returns:
            Union[Circuit, list]: Encoded quantum circuit(s). Returns a
                single Circuit for one image, or a list for multiple images.

        Raises:
            ValueError: If image dimensions or values are invalid.

        Examples:
            >>> img = np.random.rand(4, 4)  # 4x4 image
            >>> circuit = encoder(img)
            >>>
            >>> # With QIC optimization
            >>> circuit = encoder(img, use_qic=True)
        """
        if not self._is_imgs_batch(imgs):
            imgs = [imgs]

        enc_cirs = []
        for img in imgs:
            img = self._img_preprocess(img, flatten=True)
            encoder = self._construct_encoder(img, use_qic)
            enc_cirs.append(encoder)

        return enc_cirs[0] if len(enc_cirs) == 1 else enc_cirs

    def __str__(self):
        return "FRQI(n_qubits={}, grayscale={})".format(self._n_qubits, self._grayscale)

    def _construct_encoder(self, img: np.ndarray, use_qic: bool) -> Circuit:
        """
        Construct the FRQI encoding circuit.

        Args:
            img (np.ndarray): Flattened image array.
            use_qic (bool): Whether to use QIC optimization.

        Returns:
            Circuit: Encoded quantum circuit.
        """
        # Step 1: Prepare |H⟩ state (Hadamard on position qubits)
        encoder = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            encoder.h(qid)

        # Step 2: Apply rotation gates to encode pixel values
        if use_qic:
            frqi_circuit = self._construct_qic_circuit(img, rotate=True)
        else:
            frqi_circuit = self._construct_circuit(img, rotate=True)
        encoder.compose(frqi_circuit)
        return encoder

    def _img_preprocess(self, img: np.ndarray, flatten: bool = True) -> np.ndarray:
        """
        Preprocess image for encoding.

        Args:
            img (np.ndarray): Input image.
            flatten (bool): Whether to flatten the image. Defaults to True.

        Returns:
            np.ndarray: Preprocessed image.

        Raises:
            ValueError: If image is invalid.
        """
        self._validate_img(img)
        if flatten:
            img = img.flatten()
        return img

    def _validate_img(self, img: np.ndarray) -> None:
        """
        Validate image for FRQI encoding.

        Args:
            img (np.ndarray): Input image.

        Raises:
            ValueError: If image is not square, not 2^n × 2^n,
                or contains invalid pixel values.
        """
        # Check shape
        if not (img.shape[0] == img.shape[1] and 1 << int(np.log2(img.shape[0])) == img.shape[0]):
            raise ValueError("Invalid image. The image size should be 2^n x 2^n.")

        # Check pixel values
        if not (img >= 0.0).all():
            raise ValueError("Invalid image. The pixel values of the image must be non-negative.")

        # Check grayscale level
        if np.unique(img).shape[0] > self._grayscale:
            raise ValueError("Invalid image. The input image must conform to grayscale level.")

    def _construct_circuit(self, img: np.ndarray, rotate: bool) -> Circuit:
        """
        Construct the standard FRQI circuit.

        Args:
            img (np.ndarray): Flattened image.
            rotate (bool): Whether to use rotation gates.

        Returns:
            Circuit: FRQI circuit.
        """
        circuit = Circuit(self._n_qubits)
        for i in range(self._n_pixels):
            if img[i] < 1e-12:
                continue
            # Set position qubits
            bin_pos = bin(i)[2:].zfill(self._n_pos_qubits)
            for qid in range(self._n_pos_qubits):
                if (bin_pos[qid] == "0" and self._q_state[qid] == 0) or (
                    bin_pos[qid] == "1" and self._q_state[qid] == 1
                ):
                    circuit.x(qid)
                    self._q_state[qid] = 1 - self._q_state[qid]
            # Apply color encoding
            if rotate:
                theta = float(img[i] / np.max(img) * np.pi)
                cgate = MCGate(self._n_pos_qubits, StandardGate.RY)
                circuit.multi_control_gate(cgate, list(range(self._n_qubits)), [theta])
            else:
                pixel_val = int(img[i] / np.max(img) * 255)
                bin_color = bin(pixel_val)[2:].zfill(self._n_color_qubits)
                for qid in range(self._n_color_qubits):
                    if bin_color[qid] == "1":
                        mct_qids = list(range(self._n_pos_qubits)) + [self._n_pos_qubits + qid]
                        cgate = MCGate(self._n_pos_qubits, StandardGate.X)
                        circuit.multi_control_gate(cgate, mct_qids)
        # Reset position qubits
        for qid in range(self._n_pos_qubits):
            if self._q_state[qid] == 1:
                circuit.x(qid)
                self._q_state[qid] = 1 - self._q_state[qid]

        return circuit

    def _construct_qic_circuit(self, img: np.ndarray, rotate: bool, gid: int = 0) -> Circuit:
        """
        Construct FRQI circuit with QIC optimization.

        Args:
            img (np.ndarray): Flattened image.
            rotate (bool): Whether to use rotation gates.
            gid (int): Gate index. Defaults to 0.

        Returns:
            Circuit: Optimized FRQI circuit.
        """
        qic_circuit = Circuit(self._n_qubits)
        img_dict = self._get_img_dict(img, bin_val=True)
        for key in img_dict.keys():
            theta = float(key) / np.max(img) * np.pi if rotate else None
            min_dnf = self._get_min_expression(img_dict[key])
            dnf_circuit = self._construct_dnf_circuit(min_dnf, gid, theta)
            qic_circuit = qic_circuit + dnf_circuit
        for qid in range(self._n_pos_qubits):
            if self._q_state[qid] == 1:
                qic_circuit.x(qid)
                self._q_state[qid] = 1 - self._q_state[qid]

        return qic_circuit

    def _construct_dnf_circuit(self, min_dnf, gid: int = 0, theta: float = None) -> Circuit:
        """
        Construct DNF circuit for QIC.

        Args:
            min_dnf: Minimum DNF expression.
            gid (int): Gate index.
            theta (float, optional): Rotation angle.

        Returns:
            Circuit: DNF circuit.
        """
        dnf_circuit = Circuit(self._n_qubits)
        cnf_list = self._split_dnf(min_dnf)
        if cnf_list == ["True"]:
            if theta is None:
                dnf_circuit.x(gid + self._n_pos_qubits)
            else:
                dnf_circuit.ry(gid + self._n_pos_qubits, theta)
            return dnf_circuit
        for i in range(len(cnf_list)):
            if i > 0:
                uniqueness_dnf = self._get_uniqueness_dnf(cnf_list[:i], cnf_list[i])
                uniqueness_dnf_circuit = self._construct_dnf_circuit(uniqueness_dnf, gid, theta)
                dnf_circuit = dnf_circuit + uniqueness_dnf_circuit
            else:
                cnf_circuit = self._construct_cnf_circuit(cnf_list[i], gid=gid, theta=theta)
                dnf_circuit = dnf_circuit + cnf_circuit

        return dnf_circuit

    def _construct_cnf_circuit(self, cnf, gid=0, theta=None) -> Circuit:
        """
        Construct CNF circuit for QIC.

        Args:
            cnf: CNF expression.
            gid (int): Gate index.
            theta (float, optional): Rotation angle.

        Returns:
            Circuit: CNF circuit.
        """
        cnf_circuit = Circuit(self._n_qubits)
        items = self._split_cnf(cnf)
        qids = self._get_cnf_qid(items)

        for item, qid in zip(items, qids):
            if (item[0] == "~" and self._q_state[qid] == 0) or (item[0] != "~" and self._q_state[qid] != 0):
                cnf_circuit.x(qid)
                self._q_state[qid] = 1 - self._q_state[qid]

        if theta is None:
            cgate = MCGate(len(qids), StandardGate.X)
            cnf_circuit.multi_control_gate(cgate, qids + [gid + self._n_pos_qubits])
        else:
            cgate = MCGate(len(qids), StandardGate.RY)
            cnf_circuit.multi_control_gate(cgate, qids + [gid + self._n_pos_qubits], [theta])
        return cnf_circuit

    def _get_uniqueness_dnf(self, pre_cnf_list, current_cnf):
        """Get uniqueness DNF expression."""
        uniqueness_dnf = ""
        for cnf in pre_cnf_list:
            uniqueness_dnf += "~(" + cnf + ") & "
        uniqueness_dnf += "(" + current_cnf + ")"
        uniqueness_dnf = to_dnf(uniqueness_dnf, simplify=True, force=True)
        return uniqueness_dnf

    def _get_cnf_qid(self, cnf_items):
        """Extract qubit indices from CNF items."""
        idx_list = []
        for item in cnf_items:
            idx_list.append(int(item[item.index("_") + 1 :]))
        return idx_list

    def _split_dnf(self, dnf):
        """Split DNF expression into clauses."""
        return str(dnf).replace("(", "").replace(")", "").split(" | ")

    def _split_cnf(self, cnf):
        """Split CNF expression into clauses."""
        return str(cnf).replace("(", "").replace(")", "").split(" & ")

    def _get_img_dict(self, img, bin_key=False, bin_val=False):
        """Group pixels by color value."""
        img_dict = dict()
        for i in range(self._n_pixels):
            if img[i] < 1e-12:
                continue
            key = bin(img[i])[2:].zfill(self._n_color_qubits) if bin_key else str(img[i])
            val = bin(i)[2:].zfill(self._n_pos_qubits) if bin_val else i
            if key not in img_dict.keys():
                img_dict[key] = [val]
            else:
                img_dict[key].append(val)
        return img_dict

    def _get_boolen_expression(self, pixel):
        """Get Boolean expression for a pixel position."""
        boolen_expression = ""
        x = symbols("x_0:" + str(self._n_pixels))
        for i in range(self._n_pos_qubits):
            boolen_expression += "~" + str(x[i]) + " " if pixel[i] == "0" else str(x[i]) + " "
            if i != self._n_pos_qubits - 1:
                boolen_expression += "& "
        return "( " + boolen_expression + ")"

    def _get_min_expression(self, pixels):
        """Get minimum Boolean expression for pixels."""
        boolen_expressions = ""
        for i in range(len(pixels)):
            boolen_expressions += self._get_boolen_expression(pixels[i])
            if i != len(pixels) - 1:
                boolen_expressions += " | "
        min_expression = to_dnf(boolen_expressions, simplify=True, force=True)
        return min_expression
