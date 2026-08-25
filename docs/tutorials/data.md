# data 模块教程

`data` 模块提供了数据加载、预处理和批处理功能，是量子机器学习流程中的数据入口。该模块专为量子编码场景设计，支持将图像数据直接转换为量子电路。

## 模块结构

```
data/
├── data.py              # Dataset 和 DataLoader 类
└── data_preprocess.py   # 数据预处理函数
```

---

## 背景与数学原理

### 数据在量子机器学习中的角色

在量子机器学习中，数据需要经过编码才能输入量子电路。`data` 模块的核心任务是将原始数据（如图像）转换为量子编码器所需的格式，并提供高效的批处理机制。

数据流如下：

```
原始数据 → 预处理 → 量子编码 → 量子电路 → 模型推理
```

### 数据预处理流程

对于图像数据，典型的预处理流程为：

1. **类别筛选**：选择特定类别
2. **图像缩放**：调整分辨率
3. **冲突移除**：删除标签冲突样本
4. **灰度量化**：减少灰度级别
5. **量子编码**：将图像转换为量子电路

---

## Dataset: 数据集类

`Dataset` 类提供了一种灵活的方式来组织多个数据源（如特征和标签），并支持索引访问和切片操作。

### 初始化参数

```python
Dataset(*datas: list)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `*datas` | list | 可变数量的数据数组，所有数据长度必须一致 |

### 核心方法

#### __getitem__(index)

通过索引获取数据样本。

```python
from cqlib_qml.data import Dataset
import numpy as np

X = np.array([[2.0, 3.0], [0.3, 7.0], [-33.0, 1.2], [6.0, 5.0]])
y = np.array([0, 1, 1, 0])
dataset = Dataset(X, y)

# 获取单个样本
sample = dataset[0]
print(f"特征: {sample[0]}, 标签: {sample[1]}")

# 获取切片
batch = dataset[0:2]
print(f"特征批次: {batch[0]}, 标签批次: {batch[1]}")
```

**输出：**

```
特征: [2. 3.], 标签: 0
特征批次: [[2. 3.]
 [0.3 7. ]], 标签批次: [0 1]
```

#### __len__()

返回数据集大小。

```python
print(f"数据集大小: {len(dataset)}")
```

**输出：**

```
数据集大小: 4
```

---

## DataLoader: 数据加载器

`DataLoader` 提供批次迭代功能，支持随机打乱和丢弃不完整批次。

### 初始化参数

```python
DataLoader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True
)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `dataset` | Dataset | 数据集实例 |
| `batch_size` | int | 每批样本数量，默认 1 |
| `shuffle` | bool | 是否在每个 epoch 随机打乱数据 |
| `drop_last` | bool | 是否丢弃最后一个不完整批次 |

### 核心方法

#### 迭代

```python
from cqlib_qml.data import Dataset, DataLoader
import numpy as np

X = np.array([[2.0, 3.0], [0.3, 7.0], [-33.0, 1.2], [6.0, 5.0]])
y = np.array([0, 1, 1, 0])
dataset = Dataset(X, y)

# 批次大小 2，启用打乱
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_X, batch_y in loader:
    print(f"X: {batch_X}")
    print(f"y: {batch_y}")
    print("---")
```

**输出：**

```
X: [[6.  5. ]
 [0.3 7. ]]
y: [0 1]
---
X: [[-33.    1.2]
 [  2.    3. ]]
y: [1 0]
---
```

#### 批次数量计算

```python
X = np.array(range(10))
dataset = Dataset(X)

# drop_last=True：丢弃不完整批次
loader = DataLoader(dataset, batch_size=3, drop_last=True)
print(f"批次数量: {len(loader)}")  # 10 // 3 = 3

# drop_last=False：保留不完整批次
loader = DataLoader(dataset, batch_size=3, drop_last=False)
print(f"批次数量: {len(loader)}")  # ceil(10/3) = 4
```

**输出：**

```
批次数量: 3
批次数量: 4
```

---

## 数据预处理函数

### filter_targets()

从数据集中筛选指定类别，并将标签重新映射为连续整数。

```python
filter_targets(
    X: np.ndarray,
    Y: np.ndarray,
    classes: list
) -> tuple
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 输入数据 |
| `Y` | np.ndarray | 标签数组 |
| `classes` | list | 保留的类别列表 |

**返回**: `(X_filtered, Y_filtered)`

```python
from cqlib_qml.data.data_preprocess import filter_targets
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
Y = np.array([0, 1, 2, 0])

X_f, Y_f = filter_targets(X, Y, classes=[0, 2])
print(f"筛选后 X: {X_f}")
print(f"筛选后 Y: {Y_f}")  # 0→0, 2→1
```

**输出：**

```
筛选后 X: [[1 2]
 [5 6]
 [7 8]]
筛选后 Y: [0 1 0]
```

### downscale()

将图像下采样到目标分辨率。

```python
downscale(
    X: np.ndarray,
    resize: tuple
) -> np.ndarray
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 输入图像数组 |
| `resize` | tuple | 目标尺寸 (height, width) |

```python
from cqlib_qml.data.data_preprocess import downscale
import torch

# 模拟 MNIST 图像
X = torch.randn(2, 28, 28)  # 2 张 28x28 图像
X_resized = downscale(X, resize=(4, 4))
print(f"原始形状: {X.shape}")
print(f"缩放后形状: {X_resized.shape}")
```

**输出：**

```
原始形状: torch.Size([2, 28, 28])
缩放后形状: torch.Size([2, 4, 4])
```

### remove_conflict()

移除标签冲突的样本。当同一图像对应多个不同标签时，这些样本会被全部移除。

```python
remove_conflict(
    X: np.ndarray,
    Y: np.ndarray,
    resize: tuple
) -> tuple
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 图像数组 |
| `Y` | np.ndarray | 标签数组 |
| `resize` | tuple | 图像原始尺寸 |

### binary_img()

将图像二值化。

```python
binary_img(
    X: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 输入图像 |
| `threshold` | float | 二值化阈值，默认 0.5 |

```python
from cqlib_qml.data.data_preprocess import binary_img

X = np.array([0.1, 0.6, 0.3, 0.9])
X_binary = binary_img(X, threshold=0.5)
print(f"原始: {X}")
print(f"二值化: {X_binary}")
```

**输出：**

```
原始: [0.1 0.6 0.3 0.9]
二值化: [0 1 0 1]
```

### change_grayscale()

将图像量化为指定数量的灰度级。

```python
change_grayscale(
    X: np.ndarray,
    grayscale: int
) -> np.ndarray
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 输入图像，范围 [0, 1] |
| `grayscale` | int | 灰度级数量（2-256） |

```python
from cqlib_qml.data.data_preprocess import change_grayscale
import numpy as np

X = np.array([0.0, 0.33, 0.66, 1.0])
print(f"原始: {X}")
X_quantized = change_grayscale(X, grayscale=2)
print(f"2 灰度级: {X_quantized}")
```

**输出：**

```
原始: [0.   0.33 0.66 1.  ]
2 灰度级: [0. 0. 1. 1.]
```

### encoding_img()

对图像批次应用量子编码，将每个图像转换为量子电路。

```python
encoding_img(
    X: np.ndarray,
    encoding
) -> list
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `X` | np.ndarray | 图像数组 |
| `encoding` | Encoder | 量子编码器实例 |

```python
from cqlib_qml.data.data_preprocess import encoding_img, change_grayscale
from cqlib_qml.encoder import FRQI
import numpy as np

# ============ 1. 生成或加载图像 ============
# 生成 8 张 4x4 随机图像
X = np.random.rand(8, 4, 4)

# ============ 2. 灰度量化 ============
# 量化为 2 个灰度级（FRQI 需要）
X_quantized = change_grayscale(X, grayscale=2)

# ============ 3. 创建编码器 ============
encoder = FRQI(n_pixels=16, grayscale=2)

# ============ 4. 编码 ============
circuits = encoding_img(X_quantized, encoder)

print(f"图像数量: {len(circuits)}")
print(f"每个电路量子比特数: {circuits[0].num_qubits}")
print(f"电路类型: {type(circuits[0])}")
```

**输出：**

```
图像数量: 8
每个电路量子比特数: 5
电路类型: <class 'cqlib.circuit.Circuit'>
```

---

## get_mnist_dataloader(): 完整 MNIST 加载流程

`get_mnist_dataloader()` 提供了端到端的 MNIST 数据加载和预处理流程。

### 函数签名

```python
get_mnist_dataloader(
    classes: list,
    resize: tuple,
    encoding,
    batch_size: int = 32,
    grayscale: int = 2
) -> tuple
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `classes` | list | 保留的类别列表（如 `[0, 1]`） |
| `resize` | tuple | 目标图像尺寸，必须为 $2^n \times 2^n$ |
| `encoding` | Encoder | 量子编码器实例（FRQI/NEQR） |
| `batch_size` | int | 批次大小，默认 32 |
| `grayscale` | int | 灰度级数量，默认 2 |

**返回**: `(train_loader, test_loader)`

### 处理流程

1. **加载原始数据**：从 torchvision 加载 MNIST
2. **类别筛选**：只保留指定类别
3. **图像缩放**：下采样到目标分辨率
4. **冲突移除**：删除标签冲突样本
5. **灰度量化**：将像素值量化为指定灰度级
6. **量子编码**：将每个图像转换为量子电路
7. **创建 DataLoader**：生成训练和测试数据加载器

### 使用示例

```python
from cqlib_qml.data.data_preprocess import get_mnist_dataloader
from cqlib_qml.encoder import FRQI

# 1. 配置编码器
encoder = FRQI(n_pixels=16, grayscale=2)

# 2. 加载数据
train_loader, test_loader = get_mnist_dataloader(
    classes=[0, 1, 2],
    resize=(4, 4),
    encoding=encoder,
    batch_size=32,
    grayscale=2
)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# 3. 查看一个批次
for X_batch, y_batch in train_loader:
    print(f"批次大小: {len(X_batch)}")
    print(f"每个样本类型: {type(X_batch[0])}")
    print(f"标签: {y_batch[:5]}")
    break
```

**输出：**

```
训练批次数: 364
测试批次数: 69
批次大小: 32
每个样本类型: <class 'cqlib.circuit.Circuit'>
标签: [0 0 2 2 1]
```

### FRQI 编码的量子比特数计算

FRQI 编码的量子比特数由图像分辨率决定：

$$n_{\text{qubits}} = \log_2(n_{\text{pixels}}) + 1$$

其中 $n_{\text{pixels}} = \text{height} \times \text{width}$。

```python
# 4x4 图像 → 16 像素 → 5 量子比特
encoder = FRQI(n_pixels=16, grayscale=2)
print(f"量子比特数: {encoder._n_qubits}")

# 8x8 图像 → 64 像素 → 7 量子比特
encoder = FRQI(n_pixels=64, grayscale=2)
print(f"量子比特数: {encoder._n_qubits}")
```

**输出：**

```
量子比特数: 5
量子比特数: 7
```

---

## 完整示例：MNIST 二分类数据加载

```python
from cqlib_qml.data import Dataset, DataLoader
from cqlib_qml.data.data_preprocess import get_mnist_dataloader
from cqlib_qml.encoder import FRQI
from cqlib_qml.models import QNN
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.loss import MSELoss
from cqlib_qml.optimizer import Adam

# ============ 配置 ============
CLASSES = [0, 1]
RESIZE = (4, 4)
GRAYSCALE = 2
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.01

# ============ 数据加载 ============
encoder = FRQI(n_pixels=RESIZE[0] * RESIZE[1], grayscale=GRAYSCALE)

train_loader, test_loader = get_mnist_dataloader(
    classes=CLASSES,
    resize=RESIZE,
    encoding=encoder,
    batch_size=BATCH_SIZE,
    grayscale=GRAYSCALE
)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ============ 查看数据格式 ============
for X_batch, y_batch in train_loader:
    print(f"\n批次大小: {len(X_batch)}")
    print(f"样本类型: {type(X_batch[0])}")
    print(f"标签示例: {y_batch[:5]}")
    break

# ============ 模型训练 ============
n_qubits = int(np.log2(RESIZE[0] * RESIZE[1])) + 1
ansatz = HEAnsatz(n_qubits=n_qubits, d=2, layers=["RY", "CX"])
ansatz.set_measurement(readouts=[0])

model = QNN(ansatz=ansatz, optimizer=Adam(lr=LR))
loss_fun = MSELoss()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        # 前向传播
        expectations = model.forward(X_batch, trainable=True)
        
        # 准备标签和预测
        y_true = (2 * y_batch - 1.0).reshape(expectations.shape)
        y_pred = -expectations
        
        # 计算损失
        loss = loss_fun(y_pred, y_true)
        
        # 反向传播
        model.backward(loss_fun.grads(-1))
        model.update()
        model.zero_grad()
        
        epoch_loss += loss
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")
```

**输出：**

```
100%|██████████████████████████| 6992/6992 [00:00<00:00, 17677.73it/s]
100%|██████████████████████████| 1359/1359 [00:00<00:00, 17907.54it/s]
训练批次数: 218
测试批次数: 42

批次大小: 32
样本类型: <class 'cqlib.circuit.Circuit'>
标签示例: [0 0 1 0 0]
Epoch 1/5 - loss: 0.8802
Epoch 2/5 - loss: 0.8679
Epoch 3/5 - loss: 0.8552
Epoch 4/5 - loss: 0.8428
Epoch 5/5 - loss: 0.8329
```

---

## 最佳实践

### 1. 图像分辨率选择

| 分辨率 | 像素数 | FRQI 量子比特数 | 适用场景 |
|--------|--------|-----------------|----------|
| 2×2 | 4 | 3 | 快速实验 |
| 4×4 | 16 | 5 | 标准配置 |
| 8×8 | 64 | 7 | 高质量图像 |
| 16×16 | 256 | 9 | 高分辨率（可能较慢） |

### 2. 灰度级选择

| 灰度级 | 量子比特数 | 适用场景 |
|--------|------------|----------|
| 2 | 1 (FRQI) / 0 (额外) | 二值图像 |
| 4 | 2 | 简单灰度 |
| 16 | 4 | 标准灰度 |
| 256 | 8 | 全灰度（仅 NEQR） |

### 3. 批次大小建议

```python
# 仿真环境
train_loader = get_mnist_dataloader(..., batch_size=32)

# 内存有限
train_loader = get_mnist_dataloader(..., batch_size=16)

# 加速训练
train_loader = get_mnist_dataloader(..., batch_size=64)
```

### 4. 数据预处理顺序

```python
from cqlib_qml.data.data_preprocess import filter_targets, downscale, remove_conflict, change_grayscale

# 1. 筛选类别
X, Y = filter_targets(X, Y, classes=[0, 1])

# 2. 下采样
X = downscale(X, resize=(4, 4))

# 3. 移除冲突样本
X, Y = remove_conflict(X, Y, resize=(4, 4))

# 4. 灰度量化
X = change_grayscale(X, grayscale=2)
```

---

## 常见问题排查

### 问题 1: FRQI 编码器报错 "Only support images with a resolution of 2^n x 2^n"

**原因**: FRQI 要求图像尺寸为 $2^n \times 2^n$。

**解决方案**:

```python
# 使用 4x4, 8x8, 16x16 等尺寸
RESIZE = (4, 4)   # 正确
RESIZE = (8, 8)   # 正确
RESIZE = (5, 5)   # 错误

# 或者使用非 2^n 尺寸时，选择其他编码器
from cqlib_qml.encoder import AngleEncoder
encoder = AngleEncoder(mode="classical")
```

### 问题 2: 内存不足

**解决方案**:

```python
# 1. 减小批次大小
train_loader = get_mnist_dataloader(..., batch_size=16)

# 2. 降低图像分辨率
train_loader = get_mnist_dataloader(..., resize=(2, 2))

# 3. 减少灰度级
train_loader = get_mnist_dataloader(..., grayscale=2)
```

### 问题 3: DataLoader 返回空批次

**原因**: 数据量小于批次大小且 `drop_last=True`。

**解决方案**:

```python
# 方式1: 禁用 drop_last
loader = DataLoader(dataset, batch_size=10, drop_last=False)

# 方式2: 使用更小的批次
loader = DataLoader(dataset, batch_size=4, drop_last=True)
```

---

## API 参考

### Dataset

| 方法 | 描述 |
|------|------|
| `__getitem__(index)` | 获取指定索引的样本 |
| `__setitem__(index, values)` | 设置指定索引的值 |
| `__len__()` | 返回数据集大小 |

### DataLoader

| 方法 | 描述 |
|------|------|
| `__iter__()` | 返回迭代器 |
| `__next__()` | 获取下一个批次 |
| `__len__()` | 返回批次数量 |

### 预处理函数

| 函数 | 描述 |
|------|------|
| `filter_targets()` | 筛选类别并重新映射标签 |
| `downscale()` | 图像下采样 |
| `remove_conflict()` | 移除标签冲突样本 |
| `binary_img()` | 图像二值化 |
| `change_grayscale()` | 灰度量化 |
| `encoding_img()` | 量子编码 |
| `get_mnist_dataloader()` | 完整 MNIST 加载流程 |