"""Internal scaling helpers that avoid complex reciprocal overflow."""
import numpy as np


def scaled_vector(vector):
    """Return a vector at unit component scale and its real scale factor.

    Divide real and imaginary components separately: complex division can
    overflow its reciprocal for a finite, subnormal real denominator.
    Exact zero is returned unchanged with scale zero.
    """
    vector = np.asarray(vector)
    if not np.isfinite(vector).all():
        raise ValueError("Vector components must be finite.")
    scale = max(np.max(np.abs(vector.real), initial=0.),
                np.max(np.abs(vector.imag), initial=0.))
    if scale == 0:
        return vector.copy(), scale
    if np.iscomplexobj(vector):
        scaled = np.empty(vector.shape, dtype=np.complex128)
        scaled.real = vector.real / scale
        scaled.imag = vector.imag / scale
    else:
        scaled = vector / scale
    return scaled, scale


def stable_norm(vector):
    """Compute a Euclidean norm without squaring unscaled components."""
    scaled, scale = scaled_vector(vector)
    return scale * np.linalg.norm(scaled)


def binary_expectation_probability(expectations):
    """Convert binary expectations to class-1 probabilities.

    Permit only dtype-scale roundoff beyond [-1, 1]. The returned values are
    deliberately not clipped, so BCE can retain its boundary gradient.
    """
    if np.iscomplexobj(expectations) or not np.all(np.isfinite(expectations)):
        raise ValueError("Binary expectations must be finite and real")
    dtype = np.asarray(expectations).dtype
    eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else np.finfo(float).eps
    tolerance = 64 * eps
    if np.any(expectations < -1 - tolerance) or np.any(expectations > 1 + tolerance):
        raise ValueError("Binary expectations must be in [-1, 1] apart from roundoff")
    return (1 - expectations) / 2
