# algorithms 模块教程

`algorithms` 模块提供了量子机器学习算法的核心实现，包括量子核方法、量子支持向量机、变分量子分类器以及完整的训练流程。

## 模块结构

```
algorithms/
├── QKM.py                 # 量子核方法
├── QSVM.py                # 量子支持向量机
├── VQC.py                 # 变分量子分类器
├── HQNN_classification.py # HQNN 训练脚本
└── QNN_classification.py  # QNN 训练脚本
```

---

## QKM: 量子核方法

### 背景与数学原理

核方法是机器学习中将数据映射到高维特征空间的核心技术。经典核方法通过核函数 $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ 隐式计算高维空间中的内积，避免了显式构造高维特征映射 $\phi(x)$ 的计算开销。

量子核方法将这一概念推广到量子领域。数据点 $x$ 首先通过编码器 $U_{\text{enc}}(x)$ 映射为量子态 $|\psi(x)\rangle$，然后量子核定义为量子态之间的保真度：

$$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$

该核矩阵可以被视为用量子态内积替代经典核函数中的特征映射内积。由于量子态的维度随量子比特数指数增长，量子核方法天然具有高维特征空间的表达能力。

### 数学定义

对于两个数据点 $x_i$ 和 $x_j$，量子核定义为：

$$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$

其中 $|\psi(x)\rangle$ 是数据点 $x$ 经过编码器映射后的量子态。

从信息论角度看，该核矩阵对应于量子态之间的重叠度，其值域为 $[0, 1]$。该核满足 Mercer 条件，因此可以作为有效的核函数输入经典 SVM。

### 初始化参数

```python
QKM(
    encoder: Union[AmplitudeEncoder, AngleEncoder, ZZFeatureEncoder],
    swap_test: bool = False
)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `encoder` | Encoder | 数据到量子态的编码策略 |
| `swap_test` | bool | 是否使用 swap test 计算 fidelity |

`swap_test` 参数决定 fidelity 的计算方式：
- `False`: 直接计算状态向量的内积，适用于经典模拟
- `True`: 使用 swap test 量子电路，适用于量子硬件执行

### 核心方法

#### kernel(X, Y=None)

计算核矩阵。若 `Y=None`，计算自核矩阵。

```python
from cqlib_qml.algorithms import QKM
from cqlib_qml.encoder import AngleEncoder
import numpy as np

encoder = AngleEncoder(mode="classical")
qkm = QKM(encoder=encoder, swap_test=False)

X = np.array([[0.5, 0.3], [0.8, 0.1], [0.2, 0.9]])
K = qkm.kernel(X)

print(K.shape)  # (3, 3)
print(np.round(K, 3))
```

**输出：**

```
(3, 3)
[[1.    0.877 0.622]
 [0.877 1.    0.331]
 [0.622 0.331 1.   ]]
```

#### clear_cache()

清除编码电路的缓存。在处理大量数据时，建议适时调用以释放内存。

```python
qkm.clear_cache()
```

---

## QSVM: 量子支持向量机

### 背景与数学原理

支持向量机 (SVM) 是经典机器学习中最稳健的分类器之一。其核心思想是寻找最大化分类间隔的超平面。对于线性不可分的数据，SVM 通过核函数将数据映射到高维特征空间，在特征空间中寻找线性分类超平面。

优化目标为：

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w \cdot \phi(x_i) + b))$$

其对偶形式为：

$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

约束条件为 $0 \leq \alpha_i \leq C$ 且 $\sum_i \alpha_i y_i = 0$。其中 $K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$ 是核函数。

QSVM 的核心思想是用量子核替代经典核：$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$。由于量子核是在指数大的希尔伯特空间中计算的，它可以捕获经典核难以表示的复杂相关性。

### 类继承关系

```
QSVM(BaseEstimator, ClassifierMixin)
```

继承自 `sklearn` 的基类，兼容 scikit-learn API。

### 初始化参数

```python
QSVM(
    encoder: Union[AmplitudeEncoder, AngleEncoder, ZZFeatureEncoder],
    C: float = 1.0,
    swap_test: bool = False,
    probability: bool = False,
    **svm_kwargs
)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `encoder` | Encoder | 数据编码策略 |
| `C` | float | SVM 正则化参数，控制间隔最大化与分类错误之间的权衡 |
| `swap_test` | bool | 是否使用 swap test |
| `probability` | bool | 是否启用概率估计 |
| `**svm_kwargs` | dict | 传递给 `sklearn.svm.SVC` 的额外参数 |

其中 $C$ 参数控制惩罚项强度：$C$ 越大，模型越倾向于正确分类所有训练样本（可能导致过拟合）；$C$ 越小，模型越倾向于最大化分类间隔（可能导致欠拟合）。

### 核心方法

#### fit(X, y)

训练 QSVM 模型。计算量子核矩阵 $K$，然后使用 `sklearn.svm.SVC(kernel="precomputed")` 完成训练。

```python
from cqlib_qml.algorithms import QSVM
from cqlib_qml.encoder import ZZFeatureEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成示例数据
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化 QSVM
encoder = ZZFeatureEncoder(n_repeats=2, entanglement="linear")
qsvm = QSVM(encoder=encoder, C=1.0)

# 训练
qsvm.fit(X_train, y_train)

# 预测
y_pred = qsvm.predict(X_test)
print(f"Accuracy: {qsvm.score(X_test, y_test):.4f}")
print(f"Support vectors per class: {qsvm.n_support_}")
print(f"Total support vectors: {len(qsvm.support_vectors_)}")
```

**输出：**

```
Accuracy: 0.7000
Support vectors per class: [64 66]
Total support vectors: 130
```

#### predict_proba(X)

返回类别概率估计（需设置 `probability=True`）。内部使用 Platt 缩放将 SVM 的决策值映射为概率。

```python
qsvm = QSVM(encoder=encoder, C=1.0, probability=True)
qsvm.fit(X_train, y_train)

proba = qsvm.predict_proba(X_test)
print(proba[:5])
```

**输出：**

```
[[0.36823762 0.63176238]
 [0.3243551  0.6756449 ]
 [0.3923855  0.6076145 ]
 [0.34783985 0.65216015]
 [0.38147365 0.61852635]]
```

#### decision_function(X)

返回决策函数值 $f(x) = w \cdot \phi(x) + b$。该值表示样本到分类超平面的有符号距离，其绝对值越大表示分类置信度越高。

```python
scores = qsvm.decision_function(X_test)
print(scores[:5])
```

**输出：**

```
[0.75268781 1.02140927 0.62585501 0.8832119  0.68329805]
```

#### clear_cache() / reset()

- `clear_cache()`: 清除电路缓存
- `reset()`: 重置 QSVM 状态（清除模型和缓存）

```python
# 释放内存
qsvm.clear_cache()

# 完全重置
qsvm.reset()
```

### 属性

| 属性 | 描述 |
|------|------|
| `n_support_` | 每类的支持向量数量 |
| `support_vectors_` | 支持向量（原始特征空间） |
| `dual_coef_` | 对偶系数 $\alpha_i y_i$ |
| `classes_` | 类别标签 |

---

## VQC: 变分量子分类器

### 背景与数学原理

VQC 是变分量子算法 (Variational Quantum Algorithm, VQA) 在分类任务上的应用。VQA 的核心思想是：使用参数化量子线路 $U(\boldsymbol{\theta})$ 作为函数逼近器，通过经典优化器调整参数 $\boldsymbol{\theta}$，最小化损失函数 $L(\boldsymbol{\theta})$。

VQC 的完整模型可以表示为：

$$f(\boldsymbol{\theta}, x) = \langle 0 | U_{\text{enc}}(x)^\dagger U_{\text{ansatz}}(\boldsymbol{\theta})^\dagger H U_{\text{ansatz}}(\boldsymbol{\theta}) U_{\text{enc}}(x) | 0 \rangle$$

其中：
- $U_{\text{enc}}(x)$ 是数据编码器
- $U_{\text{ansatz}}(\boldsymbol{\theta})$ 是参数化线路
- $H$ 是测量哈密顿量

VQC 的本质是一个量子神经网络（QNN），其核心区别在于：经典神经网络在实数空间 $\mathbb{R}^d$ 中通过矩阵乘法和非线性激活函数实现函数逼近，而 VQC 在希尔伯特空间 $\mathcal{H}$ 中通过酉变换 $U(\boldsymbol{\theta})$ 实现函数逼近。

从信息论角度，VQC 通过酉变换将输入数据编码到量子态中，然后通过测量将量子信息投影到经典空间。参数 $\boldsymbol{\theta}$ 控制酉变换的旋转角度，从而控制量子态在布洛赫球上的位置。

### 初始化参数

```python
VQC(
    ansatz: Ansatz,
    encoder: Union[AmplitudeEncoder, AngleEncoder, ZZFeatureEncoder],
    readouts: Optional[List[int]] = None,
    loss: str = "MSE",
    optimizer: Union[str, dict, OptimizerBase] = "adam",
    n_classes: int = 2,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    verbose: bool = True
)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `ansatz` | Ansatz | 参数化量子电路 |
| `encoder` | Encoder | 数据编码策略 |
| `readouts` | List[int] | 测量量子比特索引 |
| `loss` | str | 损失函数类型: "MSE", "BCE", "CrossEntropy" |
| `optimizer` | str/dict/OptimizerBase | 优化器 |
| `n_classes` | int | 类别数 |
| `epochs` | int | 训练轮数 |
| `batch_size` | int | 批次大小 |
| `verbose` | bool | 是否打印训练进度 |

### 核心方法

#### fit(X, y)

训练 VQC 模型。

```python
from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import AngleEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 设置随机种子
np.random.seed(42)

# 1. 生成数据
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.5,
    random_state=42
)

# 2. 预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * np.pi

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 模型
ansatz = HEAnsatz(n_qubits=2, d=2, layers=["RY", "CX"])
encoder = AngleEncoder(mode="classical")

vqc = VQC(
    ansatz=ansatz,
    encoder=encoder,
    readouts=[0],
    loss="BCE",
    optimizer="adam(lr=0.1)",
    epochs=100,
    batch_size=32,
    verbose=True
)

# 4. 训练
vqc.fit(X_train, y_train)

# 5. 评估
print(f"训练准确率: {vqc.score(X_train, y_train):.4f}")
print(f"测试准确率: {vqc.score(X_test, y_test):.4f}")
```

**训练输出：**

```
Epoch 10/100 - loss: 0.3064 - acc: 0.8917
Epoch 20/100 - loss: 0.3055 - acc: 0.9000
Epoch 30/100 - loss: 0.3059 - acc: 0.9000
Epoch 40/100 - loss: 0.3111 - acc: 0.8875
Epoch 50/100 - loss: 0.3067 - acc: 0.8875
Epoch 60/100 - loss: 0.3073 - acc: 0.8792
Epoch 70/100 - loss: 0.3139 - acc: 0.8875
Epoch 80/100 - loss: 0.3133 - acc: 0.8875
Epoch 90/100 - loss: 0.3068 - acc: 0.8833
Epoch 100/100 - loss: 0.3059 - acc: 0.9000
训练准确率: 0.9000
测试准确率: 0.9667
```

#### predict(X)

返回类别预测。

```python
y_pred = vqc.predict(X_test)
print(y_pred[:10])
```

**输出：**

```
[0 1 1 0 1 1 1 1 0 0]
```

#### predict_proba(X)

返回类别概率估计。

```python
proba = vqc.predict_proba(X_test)
print(proba[:5])
```

**输出：**

```
[[0.99134206 0.00865794]
 [0.26961095 0.73038905]
 [0.29691407 0.70308593]
 [0.93700417 0.06299583]
 [0.37263155 0.62736845]]
```

### 损失函数与 readouts 的匹配规则

| 损失函数 | readouts 要求 | 适用场景 |
|----------|---------------|----------|
| `"MSE"` | 任意数量 | 回归、多标签分类 |
| `"BCE"` | 恰好 1 个 | 二分类 |
| `"CrossEntropy"` | 等于 `n_classes` | 多分类 |

```python
# 二分类
vqc_binary = VQC(
    ansatz=ansatz,
    encoder=encoder,
    readouts=[0],
    loss="BCE"
)

# 三分类
vqc_multi = VQC(
    ansatz=ansatz,
    encoder=encoder,
    readouts=[0, 1, 2],
    loss="CrossEntropy",
    n_classes=3
)
```

---

## HQNN_classification / QNN_classification

### 背景与数学原理

HQNN（混合量子-经典神经网络）和 QNN（纯量子神经网络）是量子机器学习的两种重要模型架构。

**QNN (Quantum Neural Network)** 的数学结构为：

$$f_{\text{QNN}}(x) = \langle 0 | U_{\text{enc}}(x)^\dagger U_{\text{ansatz}}(\boldsymbol{\theta})^\dagger H U_{\text{ansatz}}(\boldsymbol{\theta}) U_{\text{enc}}(x) | 0 \rangle$$

即直接将量子测量结果作为模型输出。QNN 的输出维度由测量数量决定。

**HQNN (Hybrid Quantum Neural Network)** 的数学结构为：

$$f_{\text{HQNN}}(x) = W \cdot f_{\text{QNN}}(x) + b$$

即在 QNN 之后增加一个经典全连接层 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{meas}}}$。其中 $d_{\text{meas}}$ 是量子测量数量，$d_{\text{out}}$ 是输出维度。

HQNN 结合了量子计算的高维特征表示能力和经典计算的线性变换能力。其本质可以理解为：量子部分进行非线性特征提取，经典部分进行线性分类。

### 函数接口

`HQNN_classification` 和 `QNN_classification` 提供完整的训练和验证流程。函数签名已更新，所有依赖项通过参数显式传递，而非依赖全局变量。

#### train() 函数接口

```python
def train(
    ep: int,
    it_start: int,
    net: Union[HQNN, QNN],
    train_loader,
    loss_fun,
    model_path: str,
    total_epochs: int,
    tb: Optional[Any] = None
) -> None
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `ep` | int | 当前 epoch 编号（0 起始） |
| `it_start` | int | 起始迭代索引，用于恢复训练 |
| `net` | HQNN / QNN | 模型实例 |
| `train_loader` | DataLoader | 训练数据加载器 |
| `loss_fun` | LossFun | 损失函数实例 |
| `model_path` | str | 检查点保存路径 |
| `total_epochs` | int | 总 epoch 数 |
| `tb` | optional | TensorBoard SummaryWriter 实例 |

#### validate() 函数接口

```python
def validate(
    ep: int,
    net: Union[HQNN, QNN],
    test_loader,
    loss_fun,
    batch_size: int,
    tb: Optional[Any] = None
) -> tuple
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `ep` | int | 当前 epoch 编号（0 起始） |
| `net` | HQNN / QNN | 模型实例 |
| `test_loader` | DataLoader | 测试数据加载器 |
| `loss_fun` | LossFun | 损失函数实例 |
| `batch_size` | int | 批次大小 |
| `tb` | optional | TensorBoard SummaryWriter 实例 |

### HQNN 训练示例

```python
import numpy as np
from cqlib_qml.algorithms.HQNN_classification import train, validate
from cqlib_qml.models import HQNN
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import FRQI
from cqlib_qml.loss import SoftmaxCrossEntropy
from cqlib_qml.optimizer import Adam
from cqlib_qml.data.data_preprocess import get_mnist_dataloader

# 配置
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.01

# 数据
encoder = FRQI(n_pixels=16, grayscale=2)
train_loader, test_loader = get_mnist_dataloader(
    classes=[0, 1, 2],
    resize=(4, 4),
    encoding=encoder,
    batch_size=BATCH_SIZE,
    grayscale=2
)

# 模型
n_qubits = int(np.log2(16)) + 1
ansatz = HEAnsatz(n_qubits=n_qubits, d=5, layers=["RZ", "RY", "RZ", "CX"])
model = HQNN(ansatz=ansatz, out_dim=3, optimizer=Adam(lr=LR))
loss_fun = SoftmaxCrossEntropy()
model_path = "./checkpoints/"

# 训练循环
for epoch in range(EPOCHS):
    train(
        ep=epoch,
        it_start=0,
        net=model,
        train_loader=train_loader,
        loss_fun=loss_fun,
        model_path=model_path,
        total_epochs=EPOCHS,
        tb=None
    )
    avg_loss, avg_acc = validate(
        ep=epoch,
        net=model,
        test_loader=test_loader,
        loss_fun=loss_fun,
        batch_size=BATCH_SIZE,
        tb=None
    )
```

### QNN 训练示例

```python
import numpy as np
from cqlib_qml.algorithms.QNN_classification import train, validate
from cqlib_qml.models import QNN
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import FRQI
from cqlib_qml.loss import MSELoss
from cqlib_qml.optimizer import Adam
from cqlib_qml.data.data_preprocess import get_mnist_dataloader

# 配置
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.01

# 数据
encoder = FRQI(n_pixels=16, grayscale=2)
train_loader, test_loader = get_mnist_dataloader(
    classes=[0, 1],
    resize=(4, 4),
    encoding=encoder,
    batch_size=BATCH_SIZE,
    grayscale=2
)

# 模型
n_qubits = int(np.log2(16)) + 1
ansatz = HEAnsatz(n_qubits=n_qubits, d=5, layers=["RZ", "RY", "RZ", "CX"])
ansatz.set_measurement(readouts=[0])
model = QNN(ansatz=ansatz, optimizer=Adam(lr=LR))
loss_fun = MSELoss()
model_path = "./checkpoints/"

# 训练循环
for epoch in range(EPOCHS):
    train(
        ep=epoch,
        it_start=0,
        net=model,
        train_loader=train_loader,
        loss_fun=loss_fun,
        model_path=model_path,
        total_epochs=EPOCHS,
        tb=None
    )
    avg_loss, avg_acc = validate(
        ep=epoch,
        net=model,
        test_loader=test_loader,
        loss_fun=loss_fun,
        batch_size=BATCH_SIZE,
        tb=None
    )
```

### QNN 与 HQNN 的区别

| 特性 | QNN | HQNN |
|------|-----|------|
| 架构 | 仅量子电路 | 量子电路 + 经典全连接层 |
| 数学表达 | $f(x) = \langle H \rangle$ | $f(x) = W \cdot \langle H \rangle + b$ |
| 输出 | 量子测量值 | 线性层输出 |
| 适用场景 | 简单二分类 | 多分类、复杂任务 |
| 可训练参数 | ansatz 参数 | ansatz 参数 + 线性层参数 |

---

## 完整示例: MNIST 三分类

以下示例展示如何使用 HQNN 对 MNIST 数字 0、1、2 进行分类。

```python
import numpy as np
from cqlib_qml.algorithms.HQNN_classification import train, validate
from cqlib_qml.data.data_preprocess import get_mnist_dataloader
from cqlib_qml.models import HQNN
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import FRQI
from cqlib_qml.loss import SoftmaxCrossEntropy
from cqlib_qml.optimizer import Adam

# ============ 配置 ============
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.01
CLASSES = [0, 1, 2]
RESIZE = (4, 4)
GRAYSCALE = 2

# ============ 数据加载 ============
encoder = FRQI(n_pixels=RESIZE[0] * RESIZE[1], grayscale=GRAYSCALE)
train_loader, test_loader = get_mnist_dataloader(
    classes=CLASSES,
    resize=RESIZE,
    encoding=encoder,
    batch_size=BATCH_SIZE,
    grayscale=GRAYSCALE
)

# ============ 模型 ============
n_qubits = int(np.log2(RESIZE[0] * RESIZE[1])) + 1  # position + color
ansatz = HEAnsatz(n_qubits=n_qubits, d=5, layers=["RZ", "RY", "RZ", "CX"])
model = HQNN(ansatz=ansatz, out_dim=len(CLASSES), optimizer=Adam(lr=LR))
loss_fun = SoftmaxCrossEntropy()
model_path = "./checkpoints/"

# ============ 训练 ============
for epoch in range(EPOCHS):
    train(
        ep=epoch,
        it_start=0,
        net=model,
        train_loader=train_loader,
        loss_fun=loss_fun,
        model_path=model_path,
        total_epochs=EPOCHS,
        tb=None
    )
    avg_loss, avg_acc = validate(
        ep=epoch,
        net=model,
        test_loader=test_loader,
        loss_fun=loss_fun,
        batch_size=BATCH_SIZE,
        tb=None
    )
```

---

## 最佳实践

### 1. 编码器选择建议

| 数据类型 | 推荐编码器 | 说明 |
|----------|------------|------|
| 图像数据 | FRQI, NEQR | 专为图像设计的编码器 |
| 二值图像 | QubitLattice | 每个像素一个量子比特 |
| 向量数据 | AngleEncoder | 简单高效 |
| 高维数据 | AmplitudeEncoder | 对数压缩，但实现复杂 |
| 核方法 | ZZFeatureEncoder | 适合 QSVM |

### 2. ansatz 选择建议

| 场景 | 推荐 ansatz | 说明 |
|------|-------------|------|
| 通用任务 | HEAnsatz | 硬件高效，灵活 |
| 图像分类 | CRADL, CRAML | 专门设计 |
| 简单实验 | BasicQNN | 易于理解 |

### 3. 性能优化

```python
# 1. 使用 adjoint 不同iator（仿真更快）
ansatz.set_differentiator("adjoint")

# 2. 设置合理的批次大小
# 避免批次过大导致内存不足，或过小导致梯度噪声过大

# 3. 使用学习率调度器
from cqlib_qml.scheduler import ExponentialScheduler
optimizer = Adam(lr=0.01, lr_scheduler=ExponentialScheduler(decay=0.95))

# 4. 使用梯度裁剪防止梯度爆炸
optimizer = Adam(lr=0.001, clip_norm=1.0)
```

### 4. 检查点保存与恢复

```python
# 保存检查点
model.save_checkpoint("./checkpoints", ep=epoch, it=iteration, latest=False)

# 恢复检查点
ep, it = model.load_checkpoint("./checkpoints/model.npy")
```

---

## 常见问题排查

### 问题 1: 训练不收敛

**可能原因:**
- 学习率过大或过小
- ansatz 深度不足
- 编码器不适合数据

**解决方案:**

```python
# 1. 调整学习率
optimizer = Adam(lr=0.001)

# 2. 增加 ansatz 深度
ansatz = HEAnsatz(n_qubits=4, d=5, layers=["RY", "CX"])

# 3. 尝试不同编码器
encoder = ZZFeatureEncoder(n_repeats=2, entanglement="full")
```

### 问题 2: 内存不足

**解决方案:**

```python
# 1. 减小批次大小
train_loader = get_mnist_dataloader(..., batch_size=16)

# 2. 清除缓存
qkm.clear_cache()

# 3. 使用 swap_test=False（使用 statevector 直接计算）
qkm = QKM(encoder=encoder, swap_test=False)
```

### 问题 3: 梯度消失

**解决方案:**

```python
# 使用参数偏移不同iator（更稳定）
ansatz.set_differentiator("parameter_shift")

# 或调整移位量
ansatz.set_differentiator("parameter_shift", shift=np.pi/4)
```

### 问题 4: 模型加载失败

**解决方案:**

```python
# 确保模型结构完全一致
# 检查 checkpoint 路径是否正确
# 使用完整路径

model.load_checkpoint("./checkpoints/model.npy")
```

---

## API 参考

### QKM

| 方法 | 描述 |
|------|------|
| `kernel(X, Y=None)` | 计算核矩阵 |
| `clear_cache()` | 清除电路缓存 |

### QSVM

| 方法 | 描述 |
|------|------|
| `fit(X, y)` | 训练模型 |
| `predict(X)` | 预测类别 |
| `predict_proba(X)` | 预测概率 |
| `decision_function(X)` | 决策函数值 |
| `score(X, y)` | 计算准确率 |
| `clear_cache()` | 清除缓存 |
| `reset()` | 重置模型 |

### VQC

| 方法 | 描述 |
|------|------|
| `fit(X, y)` | 训练模型 |
| `predict(X)` | 预测类别 |
| `predict_proba(X)` | 预测概率 |
| `score(X, y)` | 计算准确率 |
| `get_params()` | 获取参数 |
| `set_params()` | 设置参数 |

### 训练函数

| 函数 | 模块 | 描述 |
|------|------|------|
| `train()` | HQNN_classification | HQNN 训练一个 epoch |
| `validate()` | HQNN_classification | HQNN 验证 |
| `train()` | QNN_classification | QNN 训练一个 epoch |
| `validate()` | QNN_classification | QNN 验证 |