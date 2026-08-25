# cqlib_qml/encoder/__init__.py
"""
Quantum encoding methods for classical data.

This module provides various quantum encoding strategies for mapping
classical data onto quantum states. The choice of encoding method
significantly affects the performance of quantum machine learning models.

Available Encoders:
    - AmplitudeEncoder: Encodes data into amplitudes of quantum states.
    - AngleEncoder: Encodes data as rotation angles of quantum gates.
    - BasisEncoder: Encodes integers as basis states.
    - FRQI: Flexible Representation of Quantum Images for image encoding.
    - NEQR: Novel Enhanced Quantum Representation for images.
    - QubitLattice: One-to-one mapping of binary pixels to qubits.
    - ZZFeatureEncoder: ZZFeatureMap style encoding with entanglement.

Image Encoders (FRQI, NEQR, QubitLattice) are specifically designed for
encoding image data and support batch processing.

Examples:
    >>> from cqlib_qml.encoder import AmplitudeEncoder, AngleEncoder, FRQI
    >>>
    >>> # Amplitude encoding
    >>> amp_encoder = AmplitudeEncoder()
    >>> circuits = amp_encoder(np.array([0.5, 0.3, 0.2]))
    >>>
    >>> # Angle encoding
    >>> ang_encoder = AngleEncoder(mode="classical")
    >>> circuits = ang_encoder(np.array([0.5, 0.3]))
    >>>
    >>> # Image encoding with FRQI
    >>> frqi_encoder = FRQI(n_pixels=16, grayscale=2)
    >>> img = np.random.rand(4, 4)
    >>> circuit = frqi_encoder(img)
"""

from .amplitude import AmplitudeEncoder
from .angle import AngleEncoder
from .basis import BasisEncoder
from .FRQI import FRQI
from .NEQR import NEQR
from .QubitLattice import QubitLattice
from .ZZFeature import ZZFeatureEncoder
from .image_encoder import ImageEncoder

__all__ = [
    "AmplitudeEncoder",
    "AngleEncoder",
    "BasisEncoder",
    "FRQI",
    "NEQR",
    "QubitLattice",
    "ZZFeatureEncoder",
    "ImageEncoder",
]
