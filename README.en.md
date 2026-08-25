# Cqlib-QML

See the [Chinese version](README.md).

Cqlib-QML is a quantum machine learning Python package based on **[Cqlib](https://github.com/cq-lib)**. It provides quantum encoders for classical data, parameterized quantum circuits, quantum and hybrid quantum-classical neural networks, quantum kernel algorithms, and reusable model-training components.

## Installation

Python 3.10 or later is required.

Install from source:

```bash
git clone https://github.com/cq-lib/cqlib-qml.git
cd cqlib-qml
pip install .
```

Once published on PyPI, install it directly with:

```bash
pip install cqlib-qml
```

For development:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Features

- **encoder**: Amplitude, angle, basis, FRQI, NEQR, Qubit Lattice, and ZZ Feature Map encoders for vectors and images.
- **ansatz**: Parameterized quantum circuits including hardware-efficient, BasicQNN, CRADL, and CRAML ansatzes.
- **models**: The pure quantum `QNN`, hybrid quantum-classical `HQNN`, and a `Module` container for composing quantum and classical components.
- **algorithms**: Quantum Kernel Method (`QKM`), Quantum Support Vector Machine (`QSVM`), and a scikit-learn-compatible Variational Quantum Classifier (`VQC`).
- **differentiator**: Adjoint and parameter-shift gradient methods.
- **layer**: Dense layers and common activation functions for hybrid models.
- **data**: Dataset, batching, and image preprocessing utilities.
- **loss / optimizer / scheduler**: Loss functions, optimizers, and learning-rate schedulers.

## Quick Start

Create a quantum kernel matrix using angle encoding:

```python
import numpy as np

from cqlib_qml.algorithms import QKM
from cqlib_qml.encoder import AngleEncoder

encoder = AngleEncoder(mode="classical")
kernel = QKM(encoder=encoder)

samples = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
])

kernel_matrix = kernel.kernel(samples)
print(kernel_matrix)
```

Build a quantum neural network:

```python
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.models import QNN

ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
ansatz.set_measurement(readouts=[0])
model = QNN(ansatz=ansatz)
```

See [`docs/tutorials`](docs/tutorials) and [`docs/api`](docs/api) for more information.

## Testing

```bash
pytest
```

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements and bug fixes.
