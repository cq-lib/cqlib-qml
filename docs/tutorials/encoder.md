# encoder 模块教程

`encoder` 模块提供了多种量子编码策略，用于将经典数据映射到量子态。编码器是量子机器学习流程中的关键组件，不同的编码策略会影响模型的表达能力和性能。

## 模块结构

    encoder/
    ├── __init__.py              # 模块导出
    ├── image_encoder.py         # 图像编码器基类
    ├── amplitude.py             # 振幅编码
    ├── angle.py                 # 角度编码
    ├── basis.py                 # 基态编码
    ├── FRQI.py                  # 柔性量子图像表示
    ├── NEQR.py                  # 增强量子图像表示
    ├── QubitLattice.py          # 量子点阵编码
    └── ZZFeature.py             # ZZ 特征编码

---

## 保留输入梯度的符号编码

`AngleEncoder.to_ansatz()` 和 `ZZFeatureEncoder.to_ansatz()` 将编码电路前置于已有 Ansatz，返回独立的基础 `Ansatz`。输入自动成为电路符号，因此可直接求输入 Jacobian，也可以接入 Torch，训练编码之前的经典网络。原有 `encoder(data)` 仍返回数值电路，接口和公式不变。

```python
from cqlib.circuit import Parameter
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.encoder import AngleEncoder

body = Ansatz(2, random_state=7)
body.ry(0, Parameter("theta"))
body.cx(0, 1)
body.set_measurement(readouts=[0, 1])
body.set_differentiator("adjoint")  # 或 parameter_shift
encoded = AngleEncoder().to_ansatz(body, num_features=2)
# encoded.input_params == ["x_0", "x_1"]；body 保持原样
values = encoded.forward([[.2, .4], [.3, -.1]])
encoded.backward()
print(encoded.input_jacobian)  # (batch, outputs, inputs)
```

统一签名为 `encoder.to_ansatz(ansatz, *, num_features, input_prefix="x")`。输入按 `x_0, x_1, …, x_10, …` 的数字顺序绑定；`input_prefix` 必须完整匹配 `[A-Za-z_][A-Za-z0-9_]*`，生成的名称与已有符号冲突时拒绝。`num_features` 和此入口的 ZZ 重复次数必须是正整数，接受 `np.integer`，拒绝 Python/NumPy 布尔值。ZZ 构造函数和数值入口的重复次数行为不变。

| 编码 | 宽度要求 | 每层公式 |
| --- | --- | --- |
| Angle classical | `Q = num_features` | 第 i 位执行 `RY(2*x_i)` |
| Angle dense | `Q = ceil(num_features/2)` | 第 i 位依次执行 `RY(2*x_2i)`、`RZ(x_2i+1)`；最后缺失的相位取 0 |
| ZZ | `Q = num_features` | 每次重复依次执行全部 H、`RZ(2*π*x_i)`、各边的 `RZZ((π-x_i)*(π-x_j))` |

ZZ 支持 `linear`、`full` 和 `circular`。重复层共享同一组输入符号，输入梯度包含所有出现位置的贡献。保持既有数值编码的边顺序；两比特 `circular` 包含 `(0,1)` 和 `(1,0)` 两条边。dense 的相位是否影响输出取决于后续电路及测量；直接测 Z 可能对相位不敏感，这不代表梯度链路丢失。

组合入口要求源 Ansatz 没有已声明的输入、没有 `add_encoder()` 附着的数值编码。已有权重角色必须仍完整覆盖源电路符号；未声明角色时按原有符号顺序作为权重。支持空训练电路和零权重；测量可稍后配置，但执行前必须满足现有测量要求。

源 qubit ID 必须是 `0…Q−1` 的排列。编码特征与源 qubits 的**位置顺序**对应，返回对象统一编号为 `0…Q−1`。例如源顺序 `[1,0]` 中，源 ID 1 映射到返回对象的 ID 0，源 ID 0 映射到 ID 1。readouts 沿用原接口的位置索引语义，Hamiltonian 也保持位置顺序。稀疏编号如 `[5,2]` 在入口拒绝。

返回对象独立拥有电路、已赋权重、测量、微分配置和 RNG；修改任一对象不影响另一对象。不会继承源子类的其他属性、优化器、训练缓存或累计梯度，也不会消耗源 RNG。新对象始终 `training=True`、`trainable=True`，`updatable` 由是否存在权重决定。

权重初始化和保存恢复的边界如下：

| 接口 | 行为 |
| --- | --- |
| `to_ansatz()` | 保留完整或部分已有权重，不初始化缺失项 |
| `Ansatz.forward(X)` | 保留已赋权重，仅初始化缺失项；只接受一维/二维输入，单样本输出仍为 `(1, out_dim)` |
| `QuantumLayer(encoded)` | 完整权重才继承；部分赋值时整组使用 Torch 初始化。需要保留部分值时显式传完整的 `initial_weights` |
| 原生 `summary` / `load_params()` | 沿用原生结构校验；非空部分权重需补齐后恢复。输出维度须兼容，加载会恢复保存的测量、微分及 RNG 等原生状态 |
| Torch `state_dict()` | 校验实际电路、参数角色、测量和微分配置；优化器单独保存 |

checkpoint 不记录编码器配置身份：例如两比特 ZZ 的 `linear` 和 `full` 若生成相同电路，不承诺因名称不同而拒绝。返回基础 Ansatz 的恢复目标应通过相同组合入口重建。

本入口仅覆盖上述 Angle 和 ZZ 编码。振幅、基态、图像编码的可微适配分别评估；现有 QNN/HQNN 调用方式保持不变。Torch 联合训练、冻结和 batch 语义见 [Torch 教程](torch.md)。

---

## 背景与数学原理

### 为什么要进行量子编码？

量子计算机处理的是量子态，而经典数据（如图像、向量）需要被转换为量子态才能输入量子电路。编码器的本质是构造一个酉变换 $U_{\text{enc}}(x)$，将经典数据 $x$ 映射为量子态：

$$|\psi(x)\rangle = U_{\text{enc}}(x) |0\rangle^{\otimes n}$$

不同的编码策略将数据映射到量子态的不同自由度：
- **振幅编码**：数据 → 量子态振幅
- **角度编码**：数据 → 量子门旋转角度
- **基态编码**：数据 → 计算基态
- **图像编码**：数据 → 量子图像表示（FRQI/NEQR）

### 编码器的选择原则

| 数据类型 | 推荐编码器 | 量子比特数 | 特点 |
|----------|------------|------------|------|
| 向量数据 | AngleEncoder | 特征数或特征数/2 | 简单高效 |
| 高维向量 | AmplitudeEncoder | log2(特征数) | 指数压缩 |
| 非负整数 | BasisEncoder | log2(最大值+1) | 直接映射 |
| 二值图像 | QubitLattice | 像素数 | 直观 |
| 灰度图像 | FRQI | 2×log2(边长)+1 | 高效 |
| 多灰度图像 | NEQR | 2×log2(边长)+log2(灰度级) | 精确 |
| 核方法 | ZZFeatureEncoder | 特征数 | 适合 SVM |

---

## 图像编码器基类 (ImageEncoder)

`ImageEncoder` 是所有图像编码器的抽象基类，提供图像批次检测和验证功能。

    from cqlib_qml.encoder import ImageEncoder
    from abc import abstractmethod

    class CustomEncoder(ImageEncoder):
        def __init__(self, n_pixels):
            super().__init__(n_pixels)

        def __call__(self, imgs, **kwargs):
            # 实现编码逻辑
            pass

        def _construct_encoder(self):
            # 构建编码电路
            pass

### 批次检测

`ImageEncoder._is_imgs_batch()` 自动检测输入是否为批次：

| 输入形状 | 判断结果 |
|----------|----------|
| `(H, W)` | 单张图像 |
| `(B, H, W)` | 批次（单通道） |
| `(B, H, W, C)` | 批次（多通道） |

---

## 振幅编码 (AmplitudeEncoder)

### 数学原理

振幅编码将数据向量 $\mathbf{x} = (x_0, x_1, ..., x_{N-1})$ 编码为量子态的振幅：

$$|\psi(\mathbf{x})\rangle = \frac{1}{\|\mathbf{x}\|} \sum_{i=0}^{N-1} x_i |i\rangle$$

其中 $|i\rangle$ 是计算基态，$N = 2^n$ 是量子态空间的维度。

振幅编码使用 $n = \lceil \log_2 N \rceil$ 个量子比特，实现指数级的数据压缩。

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import AmplitudeEncoder

    encoder = AmplitudeEncoder()

    # 单样本编码（4 维向量 → 2 量子比特）
    data = np.array([1.0, 0.0, 0.0, 0.0])
    circuits = encoder(data)
    print(f"量子比特数: {circuits[0].num_qubits}")

    # 多样本编码
    data_batch = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]])
    circuits = encoder(data_batch)
    print(f"样本数量: {len(circuits)}")
    print(f"量子比特数: {circuits[0].num_qubits}")

**输出：**

    量子比特数: 2
    样本数量: 2
    量子比特数: 2

### 自动填充

如果数据维度不是 2 的幂次方，振幅编码会自动填充零：

    # 3 维向量 → 填充到 4 维 → 2 量子比特
    data = np.array([1.0, 0.0, 0.0])
    circuits = encoder(data)
    print(f"量子比特数: {circuits[0].num_qubits}")

**输出：**

    量子比特数: 2

### 归一化

振幅编码会自动归一化输入向量：

    # 自动归一化
    data = np.array([2.0, 0.0, 0.0, 0.0])
    circuits = encoder(data)
    # 实际存储的振幅为 [1.0, 0.0, 0.0, 0.0]

---

## 角度编码 (AngleEncoder)

### 数学原理

角度编码将数据特征映射为量子门的旋转角度。

**经典模式**：每个特征对应一个 RY 门：

$$|\psi(x)\rangle = \bigotimes_{i=0}^{n-1} RY(2x_i) |0\rangle$$

**密集模式**：每个量子比特编码两个特征（RY + RZ）：

$$|\psi(x)\rangle = \bigotimes_{i=0}^{n-1} RZ(x_{2i+1}) RY(2x_{2i}) |0\rangle$$

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import AngleEncoder

    # 经典模式：每个特征一个量子比特
    encoder = AngleEncoder(mode="classical")
    data = np.array([0.5, 0.3, 0.7, 0.2])
    circuits = encoder(data)
    print(f"经典模式量子比特数: {circuits[0].num_qubits}")

    # 密集模式：每个量子比特编码两个特征
    encoder = AngleEncoder(mode="dense")
    circuits = encoder(data)
    print(f"密集模式量子比特数: {circuits[0].num_qubits}")

**输出：**

    经典模式量子比特数: 4
    密集模式量子比特数: 2

### 多样本编码

    data_batch = np.array([[0.5, 0.3], [0.7, 0.2], [0.1, 0.9]])
    encoder = AngleEncoder(mode="classical")
    circuits = encoder(data_batch)
    print(f"样本数量: {len(circuits)}")

**输出：**

    样本数量: 3

---

## 基态编码 (BasisEncoder)

### 数学原理

基态编码将非负整数 $k$ 映射为计算基态 $|k\rangle$：

$$k \rightarrow |k\rangle = |b_{n-1} b_{n-2} \cdots b_0\rangle$$

其中 $b_i$ 是 $k$ 的二进制表示的第 $i$ 位。

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import BasisEncoder

    encoder = BasisEncoder()

    # 编码整数 0-3
    data = np.array([0, 1, 2, 3])
    circuits = encoder(data)

    for i, circ in enumerate(circuits):
        print(f"{i} -> {circuits[0].num_qubits} 量子比特")

**输出：**

    0 -> 2 量子比特
    1 -> 2 量子比特
    2 -> 2 量子比特
    3 -> 2 量子比特

### 自动确定量子比特数

量子比特数由最大整数决定：$n = \lceil \log_2(\max(data) + 1) \rceil$

    data = np.array([0, 5, 10, 15])
    circuits = encoder(data)
    print(f"最大值为 15，量子比特数: {circuits[0].num_qubits}")

**输出：**

    最大值为 15，量子比特数: 4

### 错误处理

    # 负数会报错
    data = np.array([-1, 0, 1])
    try:
        circuits = encoder(data)
    except ValueError as e:
        print(f"错误: {e}")

**输出：**

    错误: Basis encoding only supports encoding non-negative integers.

---

## FRQI: 柔性量子图像表示

### 数学原理

FRQI (Flexible Representation of Quantum Images) 将 $2^n \times 2^n$ 的图像编码为：

$$|I\rangle = \frac{1}{2^n} \sum_{i=0}^{2^{2n}-1} (\cos\theta_i |0\rangle + \sin\theta_i |1\rangle) \otimes |i\rangle$$

其中：
- $\theta_i \in [0, \pi/2]$ 编码像素值
- $|i\rangle$ 是位置量子比特状态
- 总量子比特数：$2n + 1$（$2n$ 个位置量子比特 + 1 个颜色量子比特）

### 初始化参数

    FRQI(
        n_pixels: int,
        grayscale: int = 2
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_pixels` | int | 像素总数，必须为 $2^n \times 2^n$ |
| `grayscale` | int | 灰度级数量，默认 2 |

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import FRQI
    from cqlib_qml.data.data_preprocess import change_grayscale

    # 1. 创建编码器（4x4 图像，2 灰度级）
    encoder = FRQI(n_pixels=16, grayscale=2)

    # 2. 准备图像（必须符合灰度级要求）
    img = np.random.rand(4, 4)
    img_quantized = change_grayscale(img, grayscale=2)

    # 3. 编码单张图像
    circuit = encoder(img_quantized)
    print(f"量子比特数: {circuit.num_qubits}")

    # 4. 编码批量图像
    imgs = np.random.rand(8, 4, 4)
    imgs_quantized = change_grayscale(imgs, grayscale=2)
    circuits = encoder(imgs_quantized)
    print(f"图像数量: {len(circuits)}")

**输出：**

    量子比特数: 5
    图像数量: 8

### 量子比特数计算

    # 4x4 图像 → 16 像素 → 2n + 1 = 5 量子比特
    encoder = FRQI(n_pixels=16, grayscale=2)
    print(f"4x4 FRQI 量子比特数: {encoder._n_qubits}")

    # 8x8 图像 → 64 像素 → 2n + 1 = 7 量子比特
    encoder = FRQI(n_pixels=64, grayscale=2)
    print(f"8x8 FRQI 量子比特数: {encoder._n_qubits}")

**输出：**

    4x4 FRQI 量子比特数: 5
    8x8 FRQI 量子比特数: 7

### 使用 QIC 压缩

    # 启用量子图像压缩（优化电路）
    circuit = encoder(img_quantized, use_qic=True)

---

## NEQR: 增强量子图像表示

### 数学原理

NEQR (Novel Enhanced Quantum Representation) 是 FRQI 的增强版本，使用多个颜色量子比特表示灰度值：

$$|I\rangle = \frac{1}{2^n} \sum_{i=0}^{2^{2n}-1} |c_i\rangle \otimes |i\rangle$$

其中：
- $|c_i\rangle$ 是 $q$ 个颜色量子比特编码像素值
- $q = \log_2(\text{grayscale})$
- 总量子比特数：$2n + q$

### 初始化参数

    NEQR(
        n_pixels: int,
        grayscale: int = 2
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_pixels` | int | 像素总数，必须为 $2^n \times 2^n$ |
| `grayscale` | int | 灰度级数量，必须为 2 的幂次方 |

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import NEQR
    from cqlib_qml.data.data_preprocess import change_grayscale

    # 1. 创建编码器（4x4 图像，4 灰度级）
    encoder = NEQR(n_pixels=16, grayscale=4)

    # 2. 准备图像
    img = np.random.rand(4, 4)
    img_quantized = np.rint(change_grayscale(img, grayscale=4) * 3).astype(int)
    # NEQR 使用整数颜色索引 0..3；FRQI 使用归一化灰度值

    # 3. 编码
    circuit = encoder(img_quantized)
    print(f"量子比特数: {circuit.num_qubits}")
    print(f"颜色量子比特数: {encoder._n_color_qubits}")

**输出：**

    量子比特数: 6
    颜色量子比特数: 2

### FRQI 与 NEQR 对比

| 特性 | FRQI | NEQR |
|------|------|------|
| 颜色量子比特 | 1 | $\log_2(\text{grayscale})$ |
| 灰度级数 | 2 | 任意 2 的幂次方 |
| 精度 | 低 | 高 |
| 量子比特数 | $2n+1$ | $2n+\log_2(\text{grayscale})$ |

---

## 量子点阵编码 (QubitLattice)

### 数学原理

QubitLattice 将二值图像的每个像素映射到一个量子比特：

$$|\psi(I)\rangle = \bigotimes_{i=0}^{N-1} |p_i\rangle$$

其中 $p_i \in \{0, 1\}$ 是第 $i$ 个像素值。

### 初始化参数

    QubitLattice(
        n_pixels: int
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_pixels` | int | 像素总数 |

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import QubitLattice

    # 1. 创建编码器
    encoder = QubitLattice(n_pixels=9)

    # 2. 准备二值图像（3x3）
    img = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # 3. 编码
    circuit = encoder(img)
    print(f"量子比特数: {circuit.num_qubits}")

**输出：**

    量子比特数: 9

---

## ZZ 特征编码 (ZZFeatureEncoder)

### 数学原理

ZZFeatureEncoder 受量子核方法启发，使用 ZZ 纠缠门创建特征空间：

1. 应用 Hadamard 门将所有量子比特置入叠加态
2. 用 RZ 门编码特征：$RZ(2\pi x_i)$
3. 用 RZZ 门创建纠缠：$RZZ((\pi - x_i)(\pi - x_j))$

### 初始化参数

    ZZFeatureEncoder(
        n_repeats: int = 2,
        entanglement: str = "linear"
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `n_repeats` | int | 编码层重复次数 |
| `entanglement` | str | 纠缠模式：`"linear"`, `"full"`, `"circular"` |

### 纠缠模式

| 模式 | 描述 | 量子比特连接 |
|------|------|-------------|
| `"linear"` | 最近邻连接 | 0-1, 1-2, 2-3, ... |
| `"full"` | 全连接 | 所有量子比特两两连接 |
| `"circular"` | 环形连接 | 0-1, 1-2, ..., n-1-0 |

### 使用示例

    import numpy as np
    from cqlib_qml.encoder import ZZFeatureEncoder

    # 1. 线性纠缠
    encoder = ZZFeatureEncoder(n_repeats=2, entanglement="linear")
    data = np.array([0.5, 0.3, 0.7])
    circuits = encoder(data)
    print(f"线性纠缠量子比特数: {circuits[0].num_qubits}")

    # 2. 全连接纠缠
    encoder = ZZFeatureEncoder(n_repeats=1, entanglement="full")
    circuits = encoder(data)
    print(f"全连接纠缠量子比特数: {circuits[0].num_qubits}")

    # 3. 环形纠缠
    encoder = ZZFeatureEncoder(n_repeats=2, entanglement="circular")
    circuits = encoder(data)
    print(f"环形纠缠量子比特数: {circuits[0].num_qubits}")

**输出：**

    线性纠缠量子比特数: 3
    全连接纠缠量子比特数: 3
    环形纠缠量子比特数: 3

### 多样本编码

    data_batch = np.array([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]])
    encoder = ZZFeatureEncoder(n_repeats=1, entanglement="linear")
    circuits = encoder(data_batch)
    print(f"样本数量: {len(circuits)}")

**输出：**

    样本数量: 2

---

## 编码器选择指南

### 根据数据类型选择

| 数据类型 | 推荐编码器 | 理由 |
|----------|------------|------|
| 图像（二值） | QubitLattice | 直观，无需额外量子比特 |
| 图像（灰度） | FRQI | 高效，适合中等灰度 |
| 图像（高精度） | NEQR | 精确，支持多灰度级 |
| 向量（低维） | AngleEncoder | 简单，易于理解 |
| 向量（高维） | AmplitudeEncoder | 指数压缩 |
| 整数标签 | BasisEncoder | 直接映射 |
| 核方法 | ZZFeatureEncoder | 适合 SVM |

### 根据量子比特数选择

| 可用量子比特数 | 推荐编码器 |
|----------------|------------|
| 少（< 10） | AngleEncoder, FRQI |
| 中（10-20） | AmplitudeEncoder, ZZFeatureEncoder |
| 多（> 20） | QubitLattice, NEQR |

### 根据精度要求选择

| 精度要求 | 推荐编码器 |
|----------|------------|
| 低 | FRQI（2 灰度级） |
| 中 | AngleEncoder, ZZFeatureEncoder |
| 高 | NEQR（多灰度级） |

---

## 最佳实践

### 1. 数据预处理

    import numpy as np
    from cqlib_qml.data.data_preprocess import change_grayscale

    # 确保数据符合编码器要求
    img = np.random.rand(4, 4)
    img_quantized = change_grayscale(img, grayscale=2)  # 量化到 2 灰度级

### 2. 批次处理

    # 所有编码器都支持批次处理
    imgs = np.random.rand(32, 4, 4)
    circuits = encoder(imgs)

### 3. 与 DataLoader 结合使用

    from cqlib_qml.data.data_preprocess import get_mnist_dataloader
    from cqlib_qml.encoder import FRQI

    encoder = FRQI(n_pixels=16, grayscale=2)
    train_loader, test_loader = get_mnist_dataloader(
        classes=[0, 1],
        resize=(4, 4),
        encoding=encoder,
        batch_size=32,
        grayscale=2
    )

---

## 常见问题排查

### 问题 1: FRQI 报错 "Only support images with a resolution of 2^n x 2^n"

**原因**：图像尺寸不是 $2^n \times 2^n$。

**解决方案**：

    # 使用 4x4, 8x8, 16x16 等尺寸
    RESIZE = (4, 4)   # 正确
    RESIZE = (8, 8)   # 正确
    RESIZE = (5, 5)   # 错误

### 问题 2: 编码器报错 "Invalid image. The input image must conform to grayscale level"

**原因**：图像像素值不符合指定的灰度级数量。

**解决方案**：

    from cqlib_qml.data.data_preprocess import change_grayscale

    # 先量化，再编码
    img_quantized = change_grayscale(img, grayscale=2)
    circuit = encoder(img_quantized)

### 问题 3: AmplitudeEncoder 报错 "Cannot encode zero vector"

**原因**：输入向量全为零。

**解决方案**：

    # 确保输入向量非零
    data = np.array([0.0, 1.0, 0.0, 0.0])  # 非零

---

## API 参考

### AmplitudeEncoder

| 方法 | 描述 |
|------|------|
| `__call__(data)` | 振幅编码 |

### AngleEncoder

| 方法 | 描述 |
|------|------|
| `__call__(data)` | 角度编码 |

### BasisEncoder

| 方法 | 描述 |
|------|------|
| `__call__(data)` | 基态编码 |

### FRQI

| 方法 | 描述 |
|------|------|
| `__call__(imgs, use_qic=False)` | FRQI 编码 |

### NEQR

| 方法 | 描述 |
|------|------|
| `__call__(imgs, use_qic=False)` | NEQR 编码 |

### QubitLattice

| 方法 | 描述 |
|------|------|
| `__call__(imgs)` | 量子点阵编码 |

### ZZFeatureEncoder

| 方法 | 描述 |
|------|------|
| `__call__(data)` | ZZ 特征编码 |
## 输入契约与 FRQI 亮度

FRQI/NEQR 接受 NumPy 数组和 Tensor；Tensor 会先 detach 并移到 CPU，不保留 torch autograd 关系。FRQI 按每张图像的最大像素值归一化，因此全 0.2 与全 0.8 图像会得到相同量子态；当前默认行为不保存绝对亮度差异。NEQR 接受整数颜色索引。BasisEncoder 接受整数及整数值浮点数组，拒绝空、负数、非整数和非有限输入。

FRQI 的普通与 QIC 编码仅跳过精确零像素，微小正数图像仍按最大像素值归一化；全零图像不执行颜色旋转。FRQI/NEQR 的单像素 QIC 使用无位置控制的颜色门。

振幅编码分别缩放实部与虚部，并对递归子向量使用稳定范数，避免大数与次正规数（例如 `1e-310`）导致归一化溢出或下溢。低于浮点可表示范围的结果无法保证保留。cqlib 2.0.0b3 模拟极小旋转（例如 `RY(2e-150)`）时可能丢失极小振幅，生成线路的角度正确不代表底层模拟能保留所有尺度。
