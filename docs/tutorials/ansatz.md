# ansatz 模块教程

`ansatz` 模块提供了参数化量子线路（Parameterized Quantum Circuits）的实现，这是构建变分量子算法和量子神经网络的核心组件。

## 背景知识

### 参数化量子线路的数学表示

参数化量子线路是包含可调参数的量子电路，参数通常表示为量子门的旋转角度。与经典神经网络通过调整权重和偏置来学习类似，参数化量子线路通过调整这些旋转角度来学习目标函数。

一个参数化量子线路可以表示为：

$$U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1)$$

其中每个 $U_l(\theta_l)$ 是一个包含参数 $\theta_l$ 的量子门，$\boldsymbol{\theta} = (\theta_1, ..., \theta_L)$ 是所有可调参数的集合。

线路作用于初始量子态 $|0\rangle^{\otimes n}$，产生参数化的输出态：

$$|\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |0\rangle^{\otimes n}$$

### 量子旋转门

参数化量子线路中最常用的门是旋转门，它们绕布洛赫球的坐标轴旋转量子态：

**RX 门**：绕 X 轴旋转角度 $\theta$

$$R_X(\theta) = \begin{bmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$

**RY 门**：绕 Y 轴旋转角度 $\theta$

$$R_Y(\theta) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$

**RZ 门**：绕 Z 轴旋转角度 $\theta$

$$R_Z(\theta) = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}$$

### 量子纠缠门

除了单量子比特旋转门，参数化量子线路还需要双量子比特门来产生纠缠，这是量子计算超越经典计算的关键资源，常用的纠缠门：

**CNOT 门**（受控非门）：

$$\text{CNOT} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

**RZZ 门**（Ising ZZ 耦合）：

$$R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z / 2} = \begin{bmatrix} e^{-i\theta/2} & 0 & 0 & 0 \\ 0 & e^{i\theta/2} & 0 & 0 \\ 0 & 0 & e^{i\theta/2} & 0 \\ 0 & 0 & 0 & e^{-i\theta/2} \end{bmatrix}$$

---

## 模块结构

```
ansatz/
├── ansatz.py        # Ansatz 基类
├── HE_ansatz.py     # 硬件高效线路
├── BasicQNN.py      # 基本 QNN 线路
├── CRADL.py         # CRADL 线路（图像分类）
└── CRAML.py         # CRAML 线路（图像分类）
```

---

## Ansatz 基类

`Ansatz` 是所有参数化量子线路的抽象基类，提供了参数管理、前向传播、反向传播和优化的统一接口。

### 核心数据结构

```python
class Ansatz:
    def __init__(self, qubits: int | list[int] | list[Qubit]):
        self._circuit = Circuit(qubits)      # 底层量子电路
        self._bindings = None                # 参数绑定字典
        self._gradients = {}                 # 参数梯度
        self._out_dim = 0                    # 输出维度
        self._trainable = True               # 可训练标志
        self._updatable = True               # 可更新标志
        self._differentiator = None          # 梯度计算器
        self._optimizer = None               # 参数优化器
        self._readouts = None                # 测量量子比特
        self._hams = None                    # 测量哈密顿量
        self._encoder = None                 # 编码电路
```

### 参数管理

`Ansatz` 使用 `Parameter` 对象来表示可调参数，这些参数在构建线路时定义，并在训练过程中更新。

```python
from cqlib_qml.ansatz import Ansatz
from cqlib.circuit import Parameter
import numpy as np

# 定义一个单参数线路
class RotationAnsatz(Ansatz):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        # 定义参数 "theta"
        theta = Parameter("theta")
        # 在每个量子比特上应用 RY 门
        for i in range(n_qubits):
            self.ry(i, theta)
        # 设置测量（默认测量 Pauli-Z 期望值）
        self.set_measurement(readouts=[0])

ansatz = RotationAnsatz(n_qubits=2)

# 查看参数信息
print(f"参数名称: {ansatz.symbols}")        # ['theta']
print(f"参数数量: {ansatz.in_dim}")         # 1

# 为参数赋值
bindings = {"theta": np.pi / 4}
ansatz.assign_parameters(bindings)
```

### 量子门基元

`Ansatz` 基类封装了完整的量子门集合，包括：

**单量子比特门**

```python
ansatz.h(0)          # Hadamard 门: |0⟩ → (|0⟩+|1⟩)/√2
ansatz.x(0)          # Pauli-X 门: |0⟩ ↔ |1⟩
ansatz.y(0)          # Pauli-Y 门: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
ansatz.z(0)          # Pauli-Z 门: |0⟩ → |0⟩, |1⟩ → -|1⟩
```

**旋转门**

```python
theta = Parameter("theta")
# RY(θ): 绕 Y 轴旋转角度 θ
ansatz.ry(0, theta)  # 等价于 RY(θ) = exp(-iθY/2)

# RZ(θ): 绕 Z 轴旋转角度 θ
ansatz.rz(0, theta)  # 等价于 RZ(θ) = exp(-iθZ/2)

# RX(θ): 绕 X 轴旋转角度 θ
ansatz.rx(0, theta)  # 等价于 RX(θ) = exp(-iθX/2)
```

**双量子比特门**

```python
# CNOT 门: 控制量子比特为 |1⟩ 时翻转目标量子比特
ansatz.cx(0, 1)      # 控制: 0, 目标: 1

# RZZ 门: 产生 ZZ 耦合
ansatz.rzz(0, 1, theta)  # RZZ(θ) = exp(-iθ Z⊗Z/2)

# SWAP 门: 交换两个量子比特的状态
ansatz.swap(0, 1)
```

### 测量设置

量子线路的输出通过测量获得。`Ansatz` 支持两种测量方式：

**方式一：Pauli-Z 投影测量**

测量指定量子比特的 $Z$ 算符期望值：

$$\langle Z_i \rangle = \langle \psi | Z_i | \psi \rangle$$

测量结果范围为 $[-1, 1]$，常用于分类任务。

```python
# 测量第 0 和第 1 个量子比特
ansatz.set_measurement(readouts=[0, 1])
# out_dim = 2
```

**方式二：一般哈密顿量测量**

测量任意哈密顿量 $H = \sum_j c_j P_j$ 的期望值：

$$\langle H \rangle = \sum_j c_j \langle P_j \rangle$$

这种方法更灵活，可以测量任意物理量。

```python
from cqlib.qis import Hamiltonian, PauliString

ham = Hamiltonian(2)
ham.add_term(PauliString.from_str("ZZ"), 1.0)   # Z⊗Z
ham.add_term(PauliString.from_str("XI"), 0.5)   # X⊗I
ansatz.set_measurement(hams=[ham])
```

### 前向传播

前向传播执行量子线路并返回测量期望值。

```python
# 随机初始化参数并执行前向传播
result = ansatz.forward()
print(f"期望值: {result}")

# 指定参数后再执行
ansatz.assign_parameters({"theta": 0.5})
result = ansatz.forward()
```

### 反向传播

反向传播计算参数梯度。`Ansatz` 支持两种梯度计算方法：

**伴随法 (Adjoint Method)**

伴随法通过反向传播计算梯度，时间复杂度 $O(p)$，其中 $p$ 是参数数量。适合在经典模拟器中使用。

```python
ansatz.set_differentiator("adjoint")
ansatz.forward()
ansatz.backward()
gradients = ansatz.gradients
```

**参数偏移法 (Parameter Shift Rule)**

参数偏移法利用参数偏移规则计算梯度：

$$\frac{\partial f}{\partial \theta_i} = \frac{f(\theta_i + s) - f(\theta_i - s)}{2\sin(s)}$$

其中 $s$ 是偏移量（通常取 $\pi/2$）。这种方法适合在真实量子硬件上使用。

```python
ansatz.set_differentiator("parameter_shift", shift=np.pi/2)
ansatz.forward()
ansatz.backward()
```

### 完整训练循环

以下是一个使用 `Ansatz` 进行训练的完整示例：

```python
# 1. 创建线路
ansatz = RotationAnsatz(n_qubits=2)
ansatz.set_optimizer("adam")
ansatz.set_differentiator("adjoint")

# 2. 训练循环
for epoch in range(100):
    # 前向传播
    expectations = ansatz.forward()
    
    # 计算损失（假设目标值为 1.0）
    target = 1.0
    loss = (expectations - target) ** 2
    
    # 计算损失梯度
    loss_grad = 2 * (expectations - target)  # dL/d(expectations)
    
    # 反向传播
    ansatz.backward(loss_grad)
    
    # 更新参数
    ansatz.update()
    ansatz.zero_grad()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss[0][0]:.4f}")
```

---

## HEAnsatz: 硬件高效线路

HEAnsatz 是硬件高效变分量子本征求解器（VQE）中常用的线路结构，由交替的单量子比特旋转层和双量子比特纠缠层构成。

### 设计原理

HEAnsatz 的设计目标有三个：一是硬件效率，使用与硬件拓扑兼容的量子门以减少 SWAP 门的使用；二是表达能力，通过多层结构提供足够的表达空间；三是参数效率，用尽可能少的参数实现目标表达能力。

### 数学结构

HEAnsatz 的 $d$ 层结构可以表示为：

$$U_{\text{HE}}(\boldsymbol{\theta}) = \prod_{l=1}^{d} \left( \prod_{i=1}^{n} R_i(\theta_{l,i}) \right) \left( \prod_{(j,k) \in E} U_{jk} \right)$$

其中 $R_i(\theta)$ 是单量子比特旋转门（RX、RY 或 RZ），$U_{jk}$ 是双量子比特门（CX、CZ 或 CRY），$E$ 是纠缠模式定义的边集。

### 纠缠模式

| 模式 | 边集 $E$ | 参数数量（CRY） | 特点 |
|------|----------|-----------------|------|
| `"downstairs"` | $\{(i,i+1): i=0,...,n-2\}$ | $n-1$ | 最近邻连接，适合线性拓扑 |
| `"full"` | $\{(i,j): i<j\}$ | $n(n-1)/2$ | 全连接，表达能力最强 |
| `"last_target"` | $\{(i,n-1): i=0,...,n-2\}$ | $n-1$ | 星型连接，适合读出优化 |
| `"last_control"` | $\{(n-1,i): i=0,...,n-2\}$ | $n-1$ | 星型连接（反向） |

### 参数计数

给定 $n$ 个量子比特、深度 $d$、层类型列表 $\mathcal{L}$：

$$m = d \cdot \sum_{G \in \mathcal{L}} p_G$$

其中 $p_G$ 是门 $G$ 的参数数量：

| 门 | 参数数量 | 说明 |
|----|----------|------|
| RX, RY, RZ | $n$ | 每个量子比特一个旋转参数 |
| CRY | $\|E\|$ | 每条边一个旋转参数 |
| CX, CZ | $0$ | 无参数 |

### 使用示例

```python
from cqlib_qml.ansatz import HEAnsatz

# 创建 HEAnsatz
# 4 个量子比特，深度 2，RY 层 + CX 层
ansatz = HEAnsatz(
    n_qubits=4,
    d=2,
    layers=["RY", "CX"],
    entangler="downstairs"
)

print(f"量子比特数: {ansatz.num_qubits}")
print(f"参数数量: {ansatz.in_dim}")  # 4 × 2 = 8 (RY 有参数，CX 无参数)
```

**逐行解释：**

1. `n_qubits=4`：使用 4 个量子比特
2. `d=2`：重复 2 层
3. `layers=["RY", "CX"]`：每层先应用 RY 门（有参数），再应用 CX 门（无参数）
4. `entangler="downstairs"`：使用最近邻纠缠

**参数数量计算：**
- RY 门：4 个量子比特 × 2 层 = 8 个参数
- CX 门：0 个参数
- 总计：8 个参数

### 不同门类型的参数数量

```python
# 只有 RY（4 × 2 = 8 个参数）
ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY"])
print(f"RY only: {ansatz.in_dim}")  # 8

# RY + CX（RY 有参数，CX 无参数）
ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
print(f"RY + CX: {ansatz.in_dim}")  # 8

# RY + CRY（都有参数）
ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CRY"], entangler="downstairs")
print(f"RY + CRY: {ansatz.in_dim}")  # (4 + 3) × 2 = 14
```

---

## BasicQNN: 基本 QNN 线路

BasicQNN 是最简单的量子神经网络结构，每个数据量子比特独立地与读出量子比特耦合。

### 设计原理

BasicQNN 的设计思想源自 Farhi 和 Neven 在 2018 年提出的量子神经网络架构，其核心是让每个数据量子比特都与读出量子比特交互，形成类似于经典神经网络的"全连接"结构。

### 数学结构

对于 $n$ 个量子比特（$n-1$ 个数据量子比特 + 1 个读出量子比特），线路结构为：

1. **初始化**：$\bigotimes_{i=0}^{n-1} H_i$
2. **读出制备**：$X_{n-1} H_{n-1}$
3. **交互层**：$\prod_{i=0}^{n-2} G_{i,n-1}(\theta_{l,i})$
4. **最终读出**：$H_{n-1}$

其中 $G$ 是 XX、YY、ZZ 或 ZX 耦合门：

$$R_{XX}(\theta) = e^{-i\theta X \otimes X / 2}$$
$$R_{YY}(\theta) = e^{-i\theta Y \otimes Y / 2}$$
$$R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z / 2}$$

### 参数计数

$$m = (n-1) \times L$$

其中 $L$ 是层数。

### 使用示例

```python
from cqlib_qml.ansatz import BasicQNN

# 创建 BasicQNN
# 4 个量子比特，XX 和 ZZ 两层
ansatz = BasicQNN(
    n_qubits=4,
    layers=["XX", "ZZ"]
)

print(f"量子比特数: {ansatz.num_qubits}")
print(f"参数数量: {ansatz.in_dim}")  # (4-1) × 2 = 6
```

**逐行解释：**

1. `n_qubits=4`：使用 4 个量子比特
2. `layers=["XX", "ZZ"]`：第一层使用 XX 门，第二层使用 ZZ 门
3. 参数数量 = (4-1) × 2 = 6

---

## CRADL: 图像分类线路

CRADL (Color-Readout-Alternating-Double-Layer) 专为 FRQI 编码的图像数据设计。

### 设计原理

FRQI 编码将图像表示为：

$$|I\rangle = \frac{1}{2^n} \sum_{i=0}^{2^{2n}-1} (\cos\theta_i |0\rangle + \sin\theta_i |1\rangle) \otimes |i\rangle$$

其中 $\theta_i$ 编码像素值，$|i\rangle$ 是位置量子比特状态。

CRADL 利用这种结构，让位置量子比特同时与颜色量子比特和读出量子比特交互。

### 量子比特结构

对于 $2^n \times 2^n$ 图像：
- $2n$ 个位置量子比特
- $1$ 个颜色量子比特
- $1$ 个读出量子比特

### 参数计数

$$m = 2 \times (n_{\text{pos}} - 2) \times L$$

对于 $2^n \times 2^n$ 图像，$n_{\text{pos}} = 2n + 1$。

### 使用示例

```python
from cqlib_qml.ansatz import CRADL

# 4x4 图像: n_pixels=16, n_pos=log2(16)=4
# 量子比特数 = 4 + 1 = 5
ansatz = CRADL(n_qubits=5, layers=2)
ansatz.set_measurement(readouts=[4])
print(f"参数数量: {ansatz.in_dim}")  # 2 × (5-2) × 2 = 12
```

---

## CRAML: 混合图像分类线路

CRAML 是 CRADL 的变体，每层同时应用 XX 和 ZZ 门。

### 设计原理

CRAML 的设计目标是比 CRADL 具有更高的参数效率。通过将 XX 和 ZZ 门混合排列，可以在相同的层数下实现更强的表达能力。

### 参数计数

$$m = 2 \times n_{\text{pos}} \times L$$

### 使用示例

```python
from cqlib_qml.ansatz import CRAML

# 5 量子比特: 4 个位置 + 1 个颜色
ansatz = CRAML(n_qubits=5, layers=2)
ansatz.set_measurement(readouts=[4])
print(f"参数数量: {ansatz.in_dim}")  # 2 × 5 × 2 = 20
```

### CRADL 与 CRAML 的对比

| 特性 | CRADL | CRAML |
|------|-------|-------|
| 参数数量 | $2(n_{\text{pos}}-2)L$ | $2n_{\text{pos}}L$ |
| 门排列 | XX 门全部在 ZZ 门前 | XX 和 ZZ 交替 |
| 适用场景 | 需要深度表达 | 需要参数效率 |

---

## 线路选择指南

| 场景 | 推荐线路 | 理由 |
|------|----------|------|
| 通用量子机器学习 | HEAnsatz | 灵活可配置，硬件效率高 |
| 二分类（简单） | BasicQNN | 结构简单，易于理解 |
| 图像分类（FRQI） | CRADL | 专门设计，表达力强 |
| 图像分类（参数效率） | CRAML | 混合设计，参数利用率高 |
| 完全自定义 | Ansatz 基类 | 最大灵活性 |

---

## 最佳实践

### 1. 线路深度选择

| 深度范围 | 适用场景 | 注意事项 |
|----------|----------|----------|
| $d=1-2$ | 简单分类任务 | 表达能力有限 |
| $d=3-5$ | 中等复杂度任务 | 常用范围 |
| $d>5$ | 复杂任务 | 可能存在贫瘠高原 |

### 2. 梯度计算方法选择

| 不同iator | 适用场景 | 复杂度 | 特点 |
|-----------|----------|--------|------|
| 伴随法 | 仿真 | $O(p)$ | 速度快，需要状态向量 |
| 参数偏移法 | 硬件 | $O(2p)$ | 稳健，可硬件执行 |

### 3. 参数初始化策略

```python
# 随机初始化（默认）
ansatz.forward()

# 均匀分布初始化
import numpy as np
uniform_params = np.random.uniform(-np.pi, np.pi, ansatz.in_dim)
bindings = dict(zip(ansatz.symbols, uniform_params))
ansatz.assign_parameters(bindings)

# 小值初始化（有助于避免贫瘠高原）
small_params = 0.01 * np.random.randn(ansatz.in_dim)
bindings = dict(zip(ansatz.symbols, small_params))
ansatz.assign_parameters(bindings)
```

---

## 常见问题排查

### 问题 1: 梯度消失（贫瘠高原）

**现象**：训练过程中梯度趋近于零，参数无法更新。

**原因**：随机初始化的深度量子线路会产生指数级小的梯度。

**解决方案**：

```python
# 1. 使用参数偏移不同iator
ansatz.set_differentiator("parameter_shift")

# 2. 减少线路深度
ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])

# 3. 使用小值初始化
small_params = 0.01 * np.random.randn(ansatz.in_dim)
bindings = dict(zip(ansatz.symbols, small_params))
ansatz.assign_parameters(bindings)
```

### 问题 2: 参数数量过多

**现象**：训练速度慢，容易过拟合。

**原因**：量子比特数或深度过大。

**解决方案**：

```python
# 1. 减少深度
ansatz = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])

# 2. 减少量子比特数
ansatz = HEAnsatz(n_qubits=3, d=2, layers=["RY", "CX"])

# 3. 使用参数效率更高的线路
ansatz = BasicQNN(n_qubits=4, layers=["XX"])
```

### 问题 3: 测量值超出预期范围

**现象**：测量值不在 $[-1, 1]$ 范围内。

**解决方案**：

```python
# 1. 检查测量设置
print(ansatz.readouts)
print(ansatz.hams)

# 2. 重新设置测量
ansatz.set_measurement(readouts=[0])

# 3. 检查线路是否为空
print(len(ansatz))  # 应大于 0
```

---

## API 参考

### Ansatz 基类

| 方法 | 返回类型 | 描述 |
|------|----------|------|
| `forward(X=None, quantum_state=None)` | np.ndarray | 前向传播，返回测量期望值 |
| `backward(dLdexp=None)` | dict 或 np.ndarray | 反向传播，返回参数梯度 |
| `set_measurement(readouts=None, hams=None)` | None | 设置测量方式 |
| `set_optimizer(optimizer)` | None | 设置参数优化器 |
| `set_differentiator(diff_type, shift=None)` | None | 设置梯度计算器 |
| `update(cur_loss=None)` | None | 更新参数 |
| `zero_grad()` | None | 将梯度置零 |
| `freeze()` | None | 冻结参数（禁用训练） |
| `unfreeze()` | None | 解冻参数（启用训练） |
| `assign_parameters(bindings)` | None | 为参数赋值 |
| `add_encoder(circuits)` | None | 添加编码电路 |
| `summary` | dict | 线路摘要信息 |
| `in_dim` | int | 参数数量 |
| `out_dim` | int | 测量数量 |
| `gradients` | dict | 当前梯度 |

### HEAnsatz

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_qubits` | int | 量子比特数（≥ 2） |
| `d` | int | 深度（层数） |
| `layers` | list | 门类型列表：`["RX", "RY", "RZ", "CX", "CZ", "CRY"]` |
| `entangler` | str | 纠缠模式：`"downstairs"`, `"full"`, `"last_target"`, `"last_control"` |

### BasicQNN

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_qubits` | int | 量子比特数（≥ 2） |
| `layers` | list | 门类型列表：`["XX", "YY", "ZZ", "ZX"]` |

### CRADL / CRAML

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_qubits` | int | 量子比特数（≥ 3） |
| `layers` | int | 层数 |