"""Train and restore a small native Torch hybrid regression network.

Run after installing the package:
    python examples/torch_hybrid.py --method adjoint
    python examples/torch_hybrid.py --method parameter_shift
"""
import argparse
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset
from cqlib.circuit import Parameter

from cqlib_qml.ansatz import Ansatz
from cqlib_qml.torch import QuantumLayer


def build_model(method):
    q = Ansatz(2)
    x0, x1, theta, phi = [Parameter(name) for name in ('x0', 'x1', 'theta', 'phi')]
    q.ry(0, x0 + theta)
    q.rx(1, x1)
    q.cx(0, 1)
    q.rz(0, phi)
    q.ry(1, theta)
    q.set_parameter_roles(input_params=[x0, x1], weight_params=[theta, phi])
    q.set_measurement(readouts=[0, 1])
    q.set_differentiator(method)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 2, dtype=torch.float64),
        torch.nn.Tanh(),
        QuantumLayer(q, initial_weights=[.3, -.2], dtype=torch.float64),
        torch.nn.Linear(2, 1, dtype=torch.float64),
    )


def train_and_restore(method='adjoint', epochs=30, checkpoint=None):
    """Return losses after training, validating all gradients and restoration."""
    if epochs <= 0:
        raise ValueError('epochs must be positive')
    torch.manual_seed(7)
    samples = torch.rand(48, 2, dtype=torch.float64) * 2 - 1
    targets = .4 * torch.sin(samples[:, :1]) + .2 * samples[:, 1:]
    loader = DataLoader(TensorDataset(samples, targets), batch_size=12, shuffle=True,
                        generator=torch.Generator().manual_seed(19))
    model = build_model(method)
    optimizer = torch.optim.Adam(model.parameters(), lr=.04)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        initial_loss = loss_fn(model(samples), targets).item()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'Missing or nonfinite gradient: {name}')
            optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(samples)
        final_loss = loss_fn(expected, targets).item()
    if final_loss >= initial_loss:
        raise RuntimeError('Training did not reduce loss')

    # Optimizer state is separate from the layer's weights and structure metadata.
    def restore(path):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        restored = build_model(method)
        restored.load_state_dict(saved['model'])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.04)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(samples), expected, rtol=0, atol=0)
        # Verify one further deterministic update, including optimizer momentum.
        for network, opt in ((model, optimizer), (restored, restored_optimizer)):
            network.train()
            opt.zero_grad()
            loss_fn(network(samples[:12]), targets[:12]).backward()
            opt.step()
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    if checkpoint is None:
        with tempfile.TemporaryDirectory() as directory:
            restore(Path(directory) / 'hybrid.pt')
    else:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        restore(path)
    return initial_loss, final_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['adjoint', 'parameter_shift'], default='adjoint')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--checkpoint', type=Path, help='Optional checkpoint output path')
    args = parser.parse_args()
    initial, final = train_and_restore(args.method, args.epochs, args.checkpoint)
    print(f'{args.method}: MSE {initial:.6f} -> {final:.6f}')
    print('Finite gradients and model/optimizer restoration verified.')


if __name__ == '__main__':
    main()
