"""Shared private utilities for owned random and component state."""
from copy import deepcopy

import numpy as np


def make_rng(random_state=None):
    """Own a generator without consuming a caller's generator."""
    if isinstance(random_state, np.random.Generator):
        return deepcopy(random_state)
    if isinstance(random_state, bool) or (random_state is not None and
                                          not isinstance(random_state, (int, np.integer))):
        raise ValueError("random_state must be an integer, Generator or None")
    return np.random.default_rng(random_state)

def rng_from_state(state):
    state = deepcopy(state)
    bit_generator = getattr(np.random, state.get('bit_generator', ''), None)
    if bit_generator not in (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
                             np.random.Philox, np.random.SFC64):
        raise ValueError("Unsupported random generator state")
    generator = np.random.Generator(bit_generator())
    generator.bit_generator.state = state
    return generator
