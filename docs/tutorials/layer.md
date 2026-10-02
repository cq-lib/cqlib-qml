# layer 模块教程

`layer` 模块提供了经典的神经网络层实现，用于构建混合量子-经典模型中的经典部分。这些层支持自动微分、参数优化和与量子组件的无缝集成。

## 模块结构

    layer/
    ├── __init__.py              # 模块导出
    ├── layer.py                 # 层基类
    ├── linear.py                # 全连接层
    └── activation.py            # 激活函数

---

## 背景与数学原理

### 为什么需要经典层？

在混合量子-经典模型中，经典层承担着以下功能：
1. **特征变换**：将量子测量的输出映射到目标空间
2. **非线性引入**：通过激活函数引入非线性
3. **维度变换**：调整特征维度以适应任务需求

### 层的组成

一个典型的神经网络层包含：
1. **线性变换**：$y = Wx + b$
2. **非线性激活**：$y = f(Wx + b)$

其中 $W$ 是权重矩阵，$b$ 是偏置向量，$f$ 是激活函数。

---

## 层基类 (Layer)

`Layer` 是所有神经网络层的抽象基类，定义了参数管理、前向传播、反向传播和优化的统一接口。

### 核心属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `parameters` | dict | 可训练参数字典 |
| `gradients` | dict | 参数梯度字典 |
| `trainable` | bool | 是否可训练 |
| `updatable` | bool | 是否可更新 |
| `hyperparameters` | dict | 超参数字典 |

### 使用示例

    from cqlib_qml.layer import Layer
    import numpy as np

    # 自定义层实现
    class MyLayer(Layer):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self._in_dim = in_dim
            self._out_dim = out_dim
            self._parameters = {"W": None}
            self._gradients = {"W": None}
            self._init = False

        def init_params(self):
            self._parameters["W"] = np.random.randn(self._in_dim, self._out_dim)
            self._gradients["W"] = np.zeros_like(self._parameters["W"])
            self._init = True

        def forward(self, x, **kwargs):
            if not self._init:
                self.init_params()
            self._X = x
            return x @ self._parameters["W"]

        def backward(self, out, **kwargs):
            self._gradients["W"] = self._X.T @ out
            return out @ self._parameters["W"].T

---

## 全连接层 (Linear)

### 数学原理

全连接层执行线性变换：

$$y = xW^T + b$$

其中：
- $x \in \mathbb{R}^{B \times d_{\text{in}}}$ 是输入
- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 是权重矩阵
- $b \in \mathbb{R}^{1 \times d_{\text{out}}}$ 是偏置向量
- $y \in \mathbb{R}^{B \times d_{\text{out}}}$ 是输出

### 初始化参数

    Linear(
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        act_fn: str = None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `in_dim` | int | 输入维度 |
| `out_dim` | int | 输出维度 |
| `bias` | bool | 是否使用偏置 |
| `act_fn` | str | 激活函数名称：`"sigmoid"`, `"relu"`, `"tanh"`, `"softplus"`, `None` |

### 参数初始化

权重使用 Kaiming 均匀初始化：

$$W \sim \mathcal{U}(-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}})$$

偏置使用相同的均匀分布初始化。

### 使用示例

    import numpy as np
    from cqlib_qml.layer import Linear

    # 创建全连接层
    layer = Linear(in_dim=10, out_dim=5, bias=True, act_fn="relu")

    # 初始化参数
    layer.init_params()

    # 查看参数形状
    print(f"权重形状: {layer._parameters['W'].shape}")
    print(f"偏置形状: {layer._parameters['b'].shape}")

**输出：**

    权重形状: (5, 10)
    偏置形状: (1, 5)

### 前向传播

    # 生成随机输入
    X = np.random.randn(32, 10)

    # 前向传播
    output = layer.forward(X)
    print(f"输出形状: {output.shape}")

**输出：**

    输出形状: (32, 5)

### 反向传播

    # 模拟损失梯度
    dLdy = np.random.randn(32, 5)

    # 反向传播
    dX = layer.backward(dLdy)
    print(f"输入梯度形状: {dX.shape}")

**输出：**

    输入梯度形状: (32, 10)

### 完整训练步骤

    # 1. 创建层
    layer = Linear(in_dim=10, out_dim=5, act_fn="sigmoid")
    layer.set_optimizer("adam")

    # 2. 前向传播
    X = np.random.randn(32, 10)
    output = layer.forward(X)

    # 3. 计算损失（假设 MSE）
    target = np.random.randn(32, 5)
    loss = np.mean((output - target) ** 2)

    # 4. 反向传播
    dLdy = 2 * (output - target) / output.size
    layer.backward(dLdy)

    # 5. 更新参数
    layer.update()

    # 6. 梯度清零
    layer.zero_grad()

---

## 激活函数 (Activation)

### 数学原理

激活函数为神经网络引入非线性，使其能够逼近任意函数。

**Sigmoid**：$\sigma(x) = \frac{1}{1 + e^{-x}}$，输出范围 $(0, 1)$

**ReLU**：$\text{ReLU}(x) = \max(0, x)$，输出范围 $[0, \infty)$

**Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$，输出范围 $(-1, 1)$

**SoftPlus**：$\text{SoftPlus}(x) = \ln(1 + e^x)$，输出范围 $(0, \infty)$

### 激活函数对比

| 激活函数 | 输出范围 | 梯度范围 | 特点 |
|----------|----------|----------|------|
| Sigmoid | $(0, 1)$ | $(0, 0.25]$ | 概率输出，梯度饱和 |
| ReLU | $[0, \infty)$ | $\{0, 1\}$ | 稀疏激活，计算简单 |
| Tanh | $(-1, 1)$ | $(0, 1]$ | 零中心，梯度饱和 |
| SoftPlus | $(0, \infty)$ | $(0, 1)$ | 平滑近似 ReLU |

### 使用示例

    import numpy as np
    from cqlib_qml.layer import Sigmoid, ReLU, Tanh, SoftPlus

    # 创建激活函数
    sigmoid = Sigmoid()
    relu = ReLU()
    tanh = Tanh()
    softplus = SoftPlus()

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # 前向传播
    print(f"Sigmoid:   {sigmoid.act(x)}")
    print(f"ReLU:      {relu.act(x)}")
    print(f"Tanh:      {tanh.act(x)}")
    print(f"SoftPlus:  {softplus.act(x)}")

**输出：**

    Sigmoid:   [0.11920292 0.26894142 0.5        0.73105858 0.88079708]
    ReLU:      [0. 0. 0. 1. 2.]
    Tanh:      [-0.96402758 -0.76159416  0.          0.76159416  0.96402758]
    SoftPlus:  [0.12692801 0.31326169 0.69314718 1.31326169 2.12692801]

### 梯度计算

    # 计算一阶导数
    print(f"Sigmoid 梯度: {sigmoid.grad(x)}")
    print(f"ReLU 梯度:    {relu.grad(x)}")
    print(f"Tanh 梯度:    {tanh.grad(x)}")
    print(f"SoftPlus 梯度: {softplus.grad(x)}")

**输出：**

    Sigmoid 梯度: [[0.10499359 0.19661193 0.25       0.19661193 0.10499359]]
    ReLU 梯度:    [0 0 0 1 1]
    Tanh 梯度:    [0.07065082 0.41997434 1.         0.41997434 0.07065082]
    SoftPlus 梯度: [0.11920292 0.26894142 0.5        0.73105858 0.88079708]

### 二阶导数

    # 计算二阶导数（可用于某些优化算法）
    print(f"Sigmoid 二阶导: {sigmoid.grad2(x)}")
    print(f"ReLU 二阶导:    {relu.grad2(x)}")
    print(f"Tanh 二阶导:    {tanh.grad2(x)}")
    print(f"SoftPlus 二阶导: {softplus.grad2(x)}")

**输出：**

    Sigmoid 二阶导: [[ 0.0799625   0.09085775  0.         -0.09085775 -0.0799625 ]]
    ReLU 二阶导:    [0. 0. 0. 0. 0.]
    Tanh 二阶导:    [ 0.13621869  0.63970001 -0.         -0.63970001 -0.13621869]
    SoftPlus 二阶导: [0.10499359 0.19661193 0.25       0.19661193 0.10499359]

---

## 层组合示例

### 构建多层感知机

    from cqlib_qml.layer import Linear
    from cqlib_qml.models import Module
    import numpy as np

    # 构建三层网络
    model = Module(
        Linear(in_dim=10, out_dim=8, act_fn="relu"),
        Linear(in_dim=8, out_dim=6, act_fn="relu"),
        Linear(in_dim=6, out_dim=2, act_fn="sigmoid")
    )

    # 设置优化器
    model.set_optimizer("adam")

    # 前向传播
    X = np.random.randn(32, 10)
    output = model.forward(X)
    print(f"输出形状: {output.shape}")

**输出：**

    输出形状: (32, 2)

### 与量子层结合（Hybrid）

    from cqlib_qml.layer import Linear
    from cqlib_qml.models import Module
    from cqlib_qml.ansatz import HEAnsatz

    # 量子层
    ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    ansatz.set_measurement(readouts=[0, 1, 2])

    # 混合模型：量子层 + 经典层
    model = Module(
        ansatz,
        Linear(in_dim=3, out_dim=2, act_fn="sigmoid")
    )

    # 设置优化器
    model.set_optimizer("adam")

---

## 最佳实践

### 1. 激活函数选择

| 任务类型 | 推荐激活函数 | 理由 |
|----------|--------------|------|
| 二分类输出 | Sigmoid | 输出在 (0,1) 范围 |
| 多分类输出 | Softmax（通过损失函数） | 输出概率分布 |
| 隐藏层 | ReLU | 计算简单，梯度不饱和 |
| 回归输出 | None（线性） | 输出无界 |
| 平滑需求 | SoftPlus | ReLU 的光滑近似 |

### 2. 参数初始化

    # Kaiming 初始化（默认）
    layer.init_params()

    # 自定义初始化
    W = np.random.randn(5, 10) * 0.01
    layer._parameters["W"] = W

### 3. 梯度管理

    # 梯度清零
    layer.zero_grad()

    # 梯度裁剪（通过优化器）
    layer.set_optimizer("sgd(clip_norm=1.0)")

---

## 常见问题排查

### 问题 1: 梯度爆炸

**原因**：学习率过大或权重初始化不当。

**解决方案**：

    # 1. 使用梯度裁剪
    layer.set_optimizer("adam(clip_norm=1.0)")

    # 2. 减小学习率
    layer.set_optimizer("adam(lr=0.0001)")

    # 3. 重新初始化
    layer.init_params()

### 问题 2: 梯度消失

**原因**：激活函数饱和（如 Sigmoid 在大输入时梯度接近 0）。

**解决方案**：

    # 1. 使用 ReLU 替代 Sigmoid
    layer = Linear(in_dim=10, out_dim=5, act_fn="relu")

    # 2. 使用批归一化（如果有）

### 问题 3: 参数不更新

**原因**：层被冻结或优化器未设置。

**解决方案**：

    # 1. 检查是否冻结
    layer.unfreeze()

    # 2. 设置优化器
    layer.set_optimizer("adam")

---

## API 参考

### Layer

| 方法 | 描述 |
|------|------|
| `init_params()` | 初始化参数 |
| `forward(x, **kwargs)` | 前向传播 |
| `backward(out, **kwargs)` | 反向传播 |
| `set_optimizer(optimizer)` | 设置优化器 |
| `update(cur_loss=None)` | 更新参数 |
| `zero_grad()` | 梯度清零 |
| `freeze()` | 冻结层 |
| `unfreeze()` | 解冻层 |
| `summary()` | 层摘要 |

### Linear

| 参数 | 类型 | 描述 |
|------|------|------|
| `in_dim` | int | 输入维度 |
| `out_dim` | int | 输出维度 |
| `bias` | bool | 是否使用偏置 |
| `act_fn` | str | 激活函数名称 |

### 激活函数

| 函数 | 方法 | 描述 |
|------|------|------|
| `Sigmoid` | `act(x)`, `grad(x)`, `grad2(x)` | 逻辑斯蒂函数 |
| `ReLU` | `act(x)`, `grad(x)`, `grad2(x)` | 整流线性单元 |
| `Tanh` | `act(x)`, `grad(x)`, `grad2(x)` | 双曲正切 |
| `SoftPlus` | `act(x)`, `grad(x)`, `grad2(x)` | 平滑 ReLU |