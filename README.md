# Cqlib-QML

英文版见 [English version](README.en.md)。

本项目是基于 **[Cqlib](https://github.com/cq-lib/cqlib)** 的量子机器学习 Python 工具包，提供经典数据量子编码、参数化量子线路、纯量子与混合量子—经典神经网络、量子核方法及模型训练组件。

量子层现在支持显式输入/权重角色、端到端梯度和微批次累积。VQC 支持可复现重训、warm start 及版本化 checkpoint 完整续训。API、恢复边界和行为迁移见 [训练契约教程](docs/tutorials/training_contracts.md)。

原生 PyTorch 联合训练可使用 `from cqlib_qml.torch import QuantumLayer`，支持多维 batch、输入与量子权重梯度及结构校验的 `state_dict()`。见 [Torch 教程](docs/tutorials/torch.md) 和 [可运行训练示例](examples/torch_hybrid.py)。

## 安装说明

环境要求：Python 3.11 及以上版本，cqlib 2.0.0b3 及以上版本。

从源码安装：

```bash
git clone https://github.com/cq-lib/cqlib-qml.git
cd cqlib-qml
pip install .
```

发布到 PyPI 后，也可以直接安装：

```bash
pip install cqlib-qml
```

开发环境安装：

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## 主要功能

- **encoder**：将经典向量和图像编码为量子线路，支持振幅编码、角度编码、基态编码、FRQI、NEQR、Qubit Lattice 和 ZZ Feature Map。
- **ansatz**：参数化量子线路，包含硬件高效线路、BasicQNN、CRADL 和 CRAML。
- **models**：纯量子神经网络 `QNN`、混合量子—经典神经网络 `HQNN`，以及用于组合量子与经典组件的 `Module`。
- **algorithms**：量子核方法 `QKM`、量子支持向量机 `QSVM` 和兼容 scikit-learn 接口的变分量子分类器 `VQC`。
- **differentiator**：伴随法和参数偏移法量子梯度计算。
- **layer**：全连接层与常用激活函数，用于构建混合模型的经典部分。
- **torch**：CPU 量子期望值层 `QuantumLayer`，接入原生 PyTorch 网络和优化器。
- **data**：数据集、批加载和图像数据预处理工具。
- **loss / optimizer / scheduler**：常用损失函数、参数优化器和学习率调度器。
- **utils**：量子门梯度矩阵等辅助工具。

## 快速开始

下面使用角度编码和量子核构造核矩阵：

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

构建量子神经网络：

```python
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.models import QNN

ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
ansatz.set_measurement(readouts=[0])
model = QNN(ansatz=ansatz)
```

更多说明参见 [`docs/tutorials`](docs/tutorials)。

## 测试

```bash
pytest
```

## License

Apache License 2.0，详见 [LICENSE](LICENSE)。

## Contributing

欢迎贡献！如需改进功能或修复缺陷，请提交 Issue 或发起 Pull Request。
