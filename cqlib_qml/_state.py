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

def clone_state(obj):
    """Copy owned state, including observables on user-defined QML subclasses."""
    from cqlib.qis import Hamiltonian
    memo, seen = {}, set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Hamiltonian):
            # Older cqlib releases expose copy() without Python copy protocols.
            memo[id(value)] = value.copy()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif any(base.__module__.startswith('cqlib_qml')
                 for base in type(value).__mro__):
            visit(value.__dict__)
    visit(obj)
    return deepcopy(obj, memo)
