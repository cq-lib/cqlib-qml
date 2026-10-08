"""Shared private state utilities for owned randomness and staged restoration."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile

import numpy as np

FORMAT_VERSION = 1


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


def validate_version(state):
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be a mapping")
    version = state.get('format_version')
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version: {version}")
    return version


def atomic_save(path, state):
    """Replace a checkpoint only after its complete temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as file:
            temporary = file.name
            np.save(file, state, allow_pickle=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def require_clean_gradients(nets):
    """Reject pending gradients, including custom layers without validity flags."""
    for net in nets:
        if (getattr(net, '_gradient_valid', False) or
                any(value is not None and np.any(value)
                    for value in getattr(net, '_gradients', {}).values())):
            raise ValueError("Checkpoint requires zero_grad after a complete update")


def data_fingerprint(data):
    """Stable content identity for numeric data and native encoded circuits."""
    import hashlib
    import pickle
    from cqlib.circuit import Circuit

    def canonical(value):
        if isinstance(value, Circuit):
            from cqlib_qml.ansatz import Ansatz
            proxy = Ansatz(value.num_qubits)
            proxy._circuit = value
            return ('circuit', proxy._circuit_summary())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return ('array', value.shape, canonical(value.tolist()))
            return ('array', value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, (list, tuple)):
            return tuple(canonical(item) for item in value)
        if isinstance(value, dict):
            return tuple((key, canonical(item)) for key, item in sorted(value.items()))
        return value
    return hashlib.sha256(pickle.dumps(canonical(data), protocol=5)).hexdigest()


def validate_flags(state):
    """Versioned persistent flags must be booleans, never truthy strings."""
    validate_version(state)
    for key in ('trainable', 'training', 'rng_state'):
        if key not in state:
            raise ValueError(f"Missing checkpoint field: {key}")
    for key in ('trainable', 'training'):
        if key in state and type(state[key]) is not bool:
            raise ValueError(f"Invalid checkpoint flag: {key}")


def validate_optimizer(optimizer, parameters):
    """Reject incompatible momentum/cache shapes before publishing restored state."""
    if optimizer is None:
        return
    for value in optimizer.hyperparameters.values():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError('Nonfinite optimizer configuration')
    scheduler = optimizer.lr_scheduler
    for value in scheduler.hyperparameters.values():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError('Nonfinite scheduler configuration')
    for name in ('stage_length', 'warmup_steps', 'model_dim', 'patience'):
        if name in scheduler.hyperparameters and scheduler.hyperparameters[name] <= 0:
            raise ValueError('Invalid scheduler configuration')
    if hasattr(scheduler, 'loss_history') and not np.isfinite(scheduler.loss_history).all():
        raise ValueError('Nonfinite scheduler history')
    if hasattr(scheduler, 'current_lr') and not np.isfinite(scheduler.current_lr):
        raise ValueError('Nonfinite scheduler learning rate')
    if type(optimizer.cur_step) is not int or optimizer.cur_step < 0:
        raise ValueError('Invalid optimizer step')
    for name, value in optimizer.cache.items():
        if name not in parameters:
            raise ValueError(f'Unknown optimizer parameter: {name}')
        shape = np.shape(parameters[name])
        if isinstance(value, dict):
            if set(value) != {'t', 'mean', 'var'} or type(value['t']) is not int or value['t'] < 0:
                raise ValueError(f'Invalid Adam cache for parameter: {name}')
            values = [value['mean'], value['var']]
        else:
            values = [value]
        for item in values:
            array = np.asarray(item)
            if array.shape != shape or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
                raise ValueError(f'Invalid optimizer cache for parameter: {name}')


def validate_loader_state(state):
    """Validate iterator metadata even when no live dataset is supplied."""
    validate_version(state)
    for name in ('size', 'next_batch', 'batch_size'):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError('Invalid DataLoader progress')
    if state['batch_size'] == 0:
        raise ValueError('Invalid DataLoader batch size')
    for name in ('shuffle', 'drop_last'):
        if type(state[name]) is not bool:
            raise ValueError('Invalid DataLoader configuration')
    batches = (state['size'] // state['batch_size'] if state['drop_last']
               else (state['size'] + state['batch_size'] - 1) // state['batch_size'])
    order = np.asarray(state['permutation'])
    if (state['next_batch'] > batches or order.shape != (state['size'],) or
            order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order), np.arange(state['size']))):
        raise ValueError('Invalid DataLoader permutation or cursor')
    fingerprint = state['fingerprint']
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            any(char not in '0123456789abcdef' for char in fingerprint)):
        raise ValueError('Invalid DataLoader data fingerprint')
    rng_from_state(state['rng_state'])
    return batches
