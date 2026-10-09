"""CPU quantum layers with independent per-forward automatic differentiation."""
from copy import deepcopy
import json
import math

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from cqlib_qml._state import clone_state
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator


def _basic(value):
    """Convert native circuit metadata into weights-only-safe Python values."""
    if isinstance(value, np.ndarray):
        return _basic(value.tolist())
    if isinstance(value, np.generic):
        return _basic(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, dict):
        return {key: _basic(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_basic(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported circuit metadata: {type(value).__name__}')


def _structure(template):
    diff = template._differentiator
    return _basic({
        'format_version': 1,
        'input_params': template.input_params,
        'weight_params': template.weight_params,
        'num_qubits': template.num_qubits,
        'circuit': template._circuit_summary(),
        'measurement': {
            'readouts': template.readouts,
            'hamiltonians': None if template.hams is None else [
                {'num_qubits': ham.num_qubits,
                 'terms': [(str(pauli), coefficient) for pauli, coefficient in ham.terms]}
                for ham in template.hams],
        },
        'differentiator': ({'method': 'parameter_shift', 'shift': diff._shift}
                           if isinstance(diff, ParameterShiftDifferentiator)
                           else {'method': 'adjoint'}),
    })


class _QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, template, record):
        execution = clone_state(template)
        # Resolve lazy Tensor view bits before crossing the NumPy boundary.
        execution.assign_weights(weight.detach().resolve_conj().resolve_neg().numpy())
        rows = inputs.detach().resolve_conj().resolve_neg().numpy()
        result = execution.forward(rows, retain_derived=record)
        output = torch.tensor(result, dtype=inputs.dtype, device='cpu')
        if record:
            empty = inputs.new_empty(0)
            jx = (torch.tensor(execution.input_jacobian, dtype=inputs.dtype, device='cpu')
                  if ctx.needs_input_grad[0] else empty)
            jw = (torch.tensor(execution.weight_jacobian, dtype=weight.dtype, device='cpu')
                  if ctx.needs_input_grad[1] else empty)
            # Saving the original tensors also enforces Torch's in-place version checks.
            ctx.save_for_backward(inputs, weight, jx, jw)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        inputs, weight, jx, jw = ctx.saved_tensors
        upstream = upstream.reshape(-1, upstream.shape[-1])
        dx = (torch.einsum('bo,boi->bi', upstream, jx).reshape(inputs.shape)
              if ctx.needs_input_grad[0] else None)
        dw = (torch.einsum('bo,bow->w', upstream, jw)
              if ctx.needs_input_grad[1] else None)
        return dx, dw, None, None


class QuantumLayer(torch.nn.Module):
    """A quantum expectation layer for native Torch networks.

    Args:
        ansatz: Nonempty Ansatz with explicit input/weight roles and measurement.
            Numeric encoding circuits must not be attached.
        initial_weights: Optional complete real weight vector. Otherwise use
            complete Ansatz weights, or Torch uniform initialization in [-pi, pi].
        dtype: torch.float32 or torch.float64; defaults to Torch's default dtype.

    Inputs have shape (..., inputs), including a single sample (inputs,).
    Outputs preserve all leading dimensions and the final measurement dimension.
    Both inputs and weights must be CPU tensors with the same dtype. Empty
    input/weight dimensions are supported; empty batch axes are rejected.
    Differentiation is first order. Reconstruct an identical layer before
    loading its state_dict.
    """

    def __init__(self, ansatz, initial_weights=None, dtype=None):
        super().__init__()
        if not isinstance(ansatz, Ansatz):
            raise TypeError('ansatz must be an Ansatz')
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('dtype must be torch.float32 or torch.float64')
        if (ansatz._roles is None or
                set(ansatz.input_params + ansatz.weight_params) != set(ansatz.symbols)):
            raise ValueError('Declare complete input/weight parameter roles first')
        if ansatz._encoder is not None:
            raise ValueError('Numeric encoder circuits are unsupported; use symbolic inputs')
        if not len(ansatz):
            raise ValueError('Ansatz circuit must be nonempty')
        if ansatz.out_dim <= 0 or (ansatz.readouts is None and ansatz.hams is None):
            raise ValueError('Ansatz must have a nonempty measurement')
        if ansatz._differentiator is not None and not isinstance(
                ansatz._differentiator, (AdjointDifferentiator, ParameterShiftDifferentiator)):
            raise ValueError('Unsupported quantum differentiator')

        self._template = clone_state(ansatz)
        if self._template._differentiator is None:
            self._template.set_differentiator('adjoint')
        if initial_weights is None and all(name in ansatz._weights for name in ansatz.weight_params):
            initial_weights = ansatz.weights
        if initial_weights is None:
            vector = torch.empty(ansatz.num_weights, dtype=dtype, device='cpu').uniform_(-math.pi, math.pi)
        else:
            if isinstance(initial_weights, torch.Tensor):
                if initial_weights.device.type != 'cpu':
                    raise ValueError('initial_weights must be on CPU')
                source = initial_weights.detach()
            else:
                source = torch.as_tensor(np.asarray(initial_weights), device='cpu')
            if source.is_complex() or source.dtype == torch.bool:
                raise ValueError('initial_weights must be a real vector')
            if source.shape != (ansatz.num_weights,):
                raise ValueError(f'initial_weights shape must be ({ansatz.num_weights},)')
            vector = source.to(dtype=dtype, device='cpu').clone()
            if not torch.isfinite(vector).all():
                raise ValueError('initial_weights must be finite')
        self.weight = torch.nn.Parameter(vector)
        self._template.zero_grad()
        self._template._invalidate_gradients()
        self._template._optimizer = None
        self._template._weights = {}
        self._template._bindings = None
        self._template._assigned_cir = None
        self._template._active_inputs = []
        self._template._active_weights = []
        self._template._batch_size = 0
        self._template._retain_derived = False
        self._metadata = _structure(self._template)
        self._metadata_key = json.dumps(self._metadata, sort_keys=True, allow_nan=False)

    @property
    def num_inputs(self):
        return self._template.num_inputs

    @property
    def num_weights(self):
        return self._template.num_weights

    @property
    def num_outputs(self):
        return self._template.out_dim

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError('inputs must be a Tensor')
        if inputs.device.type != 'cpu' or self.weight.device.type != 'cpu':
            raise ValueError('QuantumLayer requires CPU inputs and weights')
        if (inputs.dtype not in (torch.float32, torch.float64) or
                self.weight.dtype not in (torch.float32, torch.float64) or
                inputs.dtype != self.weight.dtype):
            raise ValueError('Input and weight dtype must match and be float32 or float64')
        if torch.is_autocast_enabled('cpu'):
            raise ValueError('QuantumLayer does not support autocast')
        if (inputs.ndim < 1 or inputs.shape[-1] != self.num_inputs or
                any(size == 0 for size in inputs.shape[:-1])):
            raise ValueError(f'Input shape must be (..., {self.num_inputs}) with nonempty batch axes')
        if not torch.isfinite(inputs).all() or not torch.isfinite(self.weight).all():
            raise ValueError('Inputs and weights must be finite')
        record = torch.is_grad_enabled() and (inputs.requires_grad or self.weight.requires_grad)
        batch_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(math.prod(batch_shape), self.num_inputs)
        output = _QuantumFunction.apply(flat_inputs, self.weight, self._template, record)
        return output.reshape(*batch_shape, self.num_outputs)

    def get_extra_state(self):
        """Return independent, basic-type structure metadata for state_dict."""
        return deepcopy(self._metadata)

    def _validate_metadata(self, state):
        try:
            compatible = json.dumps(state, sort_keys=True, allow_nan=False) == self._metadata_key
        except (TypeError, ValueError):
            compatible = False
        if not compatible:
            raise RuntimeError('QuantumLayer structure metadata is missing or incompatible')

    def set_extra_state(self, state):
        self._validate_metadata(state)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate before Module copies parameters, including when nested or strict=False.
        self._validate_metadata(state_dict.get(prefix + '_extra_state'))
        weight = state_dict.get(prefix + 'weight')
        if weight is not None:
            if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or
                    weight.dtype not in (torch.float32, torch.float64) or
                    weight.shape != self.weight.shape or not torch.isfinite(weight).all()):
                raise RuntimeError('QuantumLayer checkpoint weight must be a finite CPU float vector')
            # copy_ casts to the destination dtype; assign=True keeps the source dtype.
            if (not local_metadata.get('assign_to_params_buffers', False) and
                    not torch.isfinite(weight.to(dtype=self.weight.dtype)).all()):
                raise RuntimeError('QuantumLayer checkpoint weight must remain finite in the target dtype')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
