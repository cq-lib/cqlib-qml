# loss 模块教程

`loss` 模块提供了量子机器学习中常用的损失函数实现，支持自动微分和梯度计算。所有损失函数均继承自统一基类，可与优化器和模型无缝集成。

## 模块结构

    cqlib_qml/loss.py

---

## 背景与数学原理

### 什么是损失函数

损失函数衡量模型预测值与真实值之间的差异，是优化目标的核心。训练过程通过最小化损失函数来调整模型参数：

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{L}(f_{\boldsymbol{\theta}}(x), y)$$

损失函数的选择取决于任务类型：
- **回归任务**：预测连续值（如 MSE Loss）
- **分类任务**：预测离散类别（如 Cross Entropy Loss）

### 损失函数的梯度

梯度计算是反向传播的核心。本模块通过 `autograd` 自动计算梯度：

$$\nabla_{\hat{y}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

梯度值表示损失对预测值的敏感度，用于指导参数更新方向。

---

## 损失函数基类 (LossFun)

`LossFun` 是所有损失函数的抽象基类，定义了损失计算和梯度计算的统一接口。

### 核心方法

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算损失值 |
| `grads(dpred=None)` | 计算损失对预测值的梯度 |

### 使用示例

    from cqlib_qml.loss import LossFun
    import autograd.numpy as np

    # 自定义损失函数
    class MyLoss(LossFun):
        def _get_loss(self, pred, target):
            return np.mean((pred - target) ** 2)

    # 使用自定义损失
    loss_fn = MyLoss()
    pred = np.array([0.5, 0.3])
    target = np.array([0.5, 0.3])
    loss = loss_fn(pred, target)
    grads = loss_fn.grads()

---

## 均方误差损失 (MSELoss)

### 数学原理

均方误差 (Mean Squared Error) 是回归任务中最常用的损失函数：

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

其中 $\hat{y}_i$ 是预测值，$y_i$ 是真实值，$N$ 是样本数量。

**梯度**：

$$\frac{\partial L_{\text{MSE}}}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$$

### 使用示例

    import autograd.numpy as np
    from cqlib_qml.loss import MSELoss

    # 创建损失函数
    mse = MSELoss()

    # 计算损失
    pred = np.array([0.5, 0.3, 0.7])
    target = np.array([0.5, 0.3, 0.7])
    loss = mse(pred, target)
    print(f"MSE 损失: {loss:.4f}")

    # 计算梯度
    grads = mse.grads()
    print(f"梯度: {grads}")

**输出：**

    MSE 损失: 0.0000
    梯度: [0. 0. 0.]

### 回归示例

    # 预测连续值
    pred = np.array([0.6, 0.4, 0.8])
    target = np.array([0.5, 0.3, 0.7])
    loss = mse(pred, target)
    grads = mse.grads()

    print(f"MSE 损失: {loss:.4f}")
    print(f"梯度: {grads}")

**输出：**

    MSE 损失: 0.0100
    梯度: [0.06666667 0.06666667 0.06666667]

---

## 二元交叉熵损失 (BCELoss)

### 数学原理

二元交叉熵 (Binary Cross Entropy) 是二分类任务的标准损失函数：

$$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

其中 $\hat{y}_i \in (0, 1)$ 是预测概率，$y_i \in \{0, 1\}$ 是真实标签。

**梯度**：

$$\frac{\partial L_{\text{BCE}}}{\partial \hat{y}_i} = \frac{\hat{y}_i - y_i}{N\hat{y}_i(1 - \hat{y}_i)}$$

### 使用示例

    import autograd.numpy as np
    from cqlib_qml.loss import BCELoss

    # 创建损失函数
    bce = BCELoss()

    # 完美预测
    pred = np.array([[0.9, 0.1]])
    target = np.array([[1.0, 0.0]])
    loss = bce(pred, target)
    grads = bce.grads()

    print(f"BCE 损失: {loss:.4f}")
    print(f"梯度: {grads}")

**输出：**

    BCE 损失: 0.1054
    梯度: [[-0.55555556  0.55555556]]

### 二分类示例

    # 错误预测
    pred = np.array([[0.1, 0.9]])
    target = np.array([[1.0, 0.0]])
    loss = bce(pred, target)
    grads = bce.grads()

    print(f"BCE 损失: {loss:.4f}")
    print(f"梯度: {grads}")

**输出：**

    BCE 损失: 2.3026
    梯度: [[-5.  5.]]

---

## 交叉熵损失 (CrossEntropy)

### 数学原理

交叉熵 (Cross Entropy) 是多分类任务的标准损失函数：

$$L_{\text{CE}} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

其中 $\hat{y}$ 是预测概率分布（需满足 $\sum \hat{y}_i = 1$），$y$ 是 one-hot 编码的真实标签。

**梯度**：

$$\frac{\partial L_{\text{CE}}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

### 使用示例

    import autograd.numpy as np
    from cqlib_qml.loss import CrossEntropy

    # 创建损失函数
    ce = CrossEntropy()

    # 三分类示例
    pred = np.array([[0.7, 0.2, 0.1]])
    target = np.array([[1.0, 0.0, 0.0]])
    loss = ce(pred, target)
    grads = ce.grads()

    print(f"CrossEntropy 损失: {loss:.4f}")
    print(f"梯度: {grads}")

**输出：**

    CrossEntropy 损失: 0.3567
    梯度: [[-1.42857143  0.          0.        ]]

---

## Softmax 交叉熵损失 (SoftmaxCrossEntropy)

### 为什么需要 Softmax 交叉熵？

在多分类问题中，模型输出通常是 logits（未归一化的分数），需要先转换为概率分布。最常用的方法是 Softmax 函数：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

其中 $z_i$ 是第 $i$ 类的 logit，$C$ 是类别总数。

得到概率分布后，再计算交叉熵损失：

$$L = -\sum_{i=1}^{C} y_i \log(p_i)$$

### 为什么将 Softmax 和交叉熵合并计算？

将两个操作合并的核心原因在于**数值稳定性和梯度简洁性**。

#### 问题 1: 数值溢出

如果直接分两步计算，当 logit 值很大时，$e^{z_i}$ 可能溢出：

    # 不推荐：分步计算存在数值风险
    p = np.exp(z) / np.sum(np.exp(z))  # z 很大时 exp(z) 溢出
    loss = -np.sum(y * np.log(p))

合并计算时，通过 Log-Sum-Exp 技巧实现数值稳定：

$$\log(\text{Softmax}(z_i)) = z_i - \log\left(\sum_{j=1}^{C} e^{z_j}\right)$$

这种形式避免了直接计算 $e^{z_i}$ 的溢出。

#### 问题 2: 梯度计算

**分步计算的梯度**：

首先计算 Softmax 的梯度：

$$\frac{\partial p_j}{\partial z_i} = p_j (\delta_{ij} - p_i)$$

其中 $\delta_{ij}$ 是克罗内克函数（当 $i=j$ 时为 1，否则为 0）。

然后通过链式法则计算损失对 logits 的梯度：

$$\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \frac{\partial L}{\partial p_j} \frac{\partial p_j}{\partial z_i}$$

其中 $\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}$。

展开后得到：

$$\frac{\partial L}{\partial z_i} = -\sum_{j=1}^{C} \frac{y_j}{p_j} \cdot p_j(\delta_{ij} - p_i) = -\sum_{j=1}^{C} y_j(\delta_{ij} - p_i)$$

$$= -\left(y_i - p_i \sum_{j=1}^{C} y_j\right) = p_i - y_i$$

因为 $\sum_{j=1}^{C} y_j = 1$（标签是 one-hot 编码）。

**合并计算的梯度**：

直接对合并损失求导得到完全相同的结果：

$$\frac{\partial L}{\partial z_i} = p_i - y_i = \text{Softmax}(z)_i - y_i$$

### 两种计算方式的对比

| 方式 | 数值稳定性 | 梯度公式 | 代码复杂度 |
|------|-----------|----------|-----------|
| 分步计算 | 差（可能溢出） | $\frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial p_j} \frac{\partial p_j}{\partial z_i}$（复杂） | 高 |
| 合并计算 | 好（Log-Sum-Exp） | $\frac{\partial L}{\partial z_i} = p_i - y_i$（简洁） | 低 |

### 数学原理

Softmax 交叉熵损失将 Softmax 激活函数与交叉熵损失结合，直接接受原始 logits 作为输入：

$$\text{Softmax}(\hat{y}_i) = \frac{e^{\hat{y}_i}}{\sum_j e^{\hat{y}_j}}$$

$$L_{\text{SoftmaxCE}} = -\sum_{i} y_i \log(\text{Softmax}(\hat{y})_i)$$

**梯度**（简化形式）：

$$\frac{\partial L}{\partial \hat{y}_i} = \text{Softmax}(\hat{y})_i - y_i$$

### 使用示例

    import autograd.numpy as np
    from cqlib_qml.loss import SoftmaxCrossEntropy

    # 创建损失函数
    sce = SoftmaxCrossEntropy()

    # 三分类示例（输入原始 logits）
    logits = np.array([[2.0, 1.0, 0.1]])
    target = np.array([[1.0, 0.0, 0.0]])

    loss = sce(logits, target)
    grads = sce.grads()

    print(f"Softmax CrossEntropy 损失: {loss:.4f}")

    # 手动计算 Softmax 验证
    exp_logits = np.exp(logits)
    softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    print(f"Softmax 输出: {softmax_output}")
    print(f"梯度 (p - y): {softmax_output - target}")

**输出：**

    Softmax CrossEntropy 损失: 0.4170
    Softmax 输出: [[0.6590 0.2424 0.0986]]
    梯度 (p - y): [[-0.3410  0.2424  0.0986]]

### 为什么梯度等于 p - y？

当标签为 one-hot 编码时（$\sum_j y_j = 1$），梯度简化为：

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

这个结果有三个重要性质：

1. **物理意义清晰**：梯度方向是将预测概率 $p$ 推向目标分布 $y$
2. **计算高效**：只需一次 Softmax 计算和一次减法，无需复杂的链式法则
3. **数值稳定**：避免了除以 $p_i$ 时可能出现的除零错误

### 何时使用 SoftmaxCrossEntropy vs CrossEntropy？

| 场景 | 推荐使用 | 原因 |
|------|----------|------|
| 模型输出为 logits | SoftmaxCrossEntropy | 一步到位，数值稳定 |
| 已有 Softmax 输出 | CrossEntropy | 直接使用概率值 |
| 需要获取概率值 | SoftmaxCrossEntropy | 内部自动计算 Softmax |

---

## 合页损失 (HingeLoss)

### 数学原理

合页损失 (Hinge Loss) 是 SVM 风格分类器的损失函数：

$$L_{\text{Hinge}} = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i \hat{y}_i)$$

其中 $y_i \in \{-1, 1\}$ 是真实标签，$\hat{y}_i$ 是预测值。

**梯度**：

$$\frac{\partial L_{\text{Hinge}}}{\partial \hat{y}_i} = \begin{cases} -y_i/N & \text{if } 1 - y_i \hat{y}_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 使用示例

    import autograd.numpy as np
    from cqlib_qml.loss import HingeLoss

    # 创建损失函数
    hinge = HingeLoss()

    # 正确分类
    pred = np.array([0.8, -0.2])
    target = np.array([1, -1])
    loss = hinge(pred, target)
    grads = hinge.grads()

    print(f"Hinge 损失: {loss:.4f}")
    print(f"梯度: {grads}")

**输出：**

    Hinge 损失: 0.5000
    梯度: [-0.5  0.5]

---

## 损失函数选择指南

### 根据任务类型选择

| 任务类型 | 推荐损失函数 | 理由 |
|----------|--------------|------|
| 回归 | MSELoss | 标准回归损失，梯度平滑 |
| 二分类（概率输出） | BCELoss | 输出在 (0,1) 范围 |
| 多分类（logits 输入） | SoftmaxCrossEntropy | 数值稳定，梯度简洁 |
| 多分类（概率输入） | CrossEntropy | 需要手动 Softmax |
| SVM 风格 | HingeLoss | 最大间隔分类 |

### 根据输出层选择

| 输出层激活 | 推荐损失函数 |
|------------|--------------|
| None（线性） | MSELoss（回归）/ SoftmaxCrossEntropy（分类） |
| Sigmoid | BCELoss |
| Softmax | CrossEntropy |

---

## 最佳实践

### 1. 数值稳定性

    # BCELoss 自动添加 epsilon 防止 log(0)
    pred = np.array([[0.0, 1.0]])  # 极端值
    target = np.array([[1.0, 0.0]])
    loss = bce(pred, target)  # 不会报错，clip 到安全范围

### 2. 梯度缩放

    # 在反向传播时缩放梯度
    grads = loss_fn.grads(dpred=0.5)  # 乘以 0.5

### 3. 与 Ansatz 集成

    # 在训练循环中使用
    expectations = ansatz.forward()
    loss = loss_fn(expectations, targets)
    ansatz.backward(loss_fn.grads())

---

## 常见问题排查

### 问题 1: 损失值 NaN

**原因**：预测值接近 0 或 1 导致 log(0)。

**解决方案**：

    # BCELoss 内部已添加 epsilon 防止 NaN
    # 如仍需手动处理：
    eps = 1e-7
    pred = np.clip(pred, eps, 1 - eps)

### 问题 2: 梯度值过大

**原因**：预测值与目标值差异极大。

**解决方案**：

    # 使用梯度裁剪
    grads = np.clip(grads, -1.0, 1.0)

### 问题 3: 损失不下降

**原因**：学习率不当或模型结构问题。

**解决方案**：

    # 检查梯度是否正常
    print(loss_fn.grads())

---

## API 参考

### LossFun

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算损失值 |
| `grads(dpred=None)` | 计算损失对预测值的梯度 |

### MSELoss

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算 MSE 损失 |
| `grads(dpred=None)` | 计算梯度 |

### BCELoss

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算 BCE 损失 |
| `grads(dpred=None)` | 计算梯度 |

### CrossEntropy

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算交叉熵损失 |
| `grads(dpred=None)` | 计算梯度 |

### SoftmaxCrossEntropy

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算 Softmax 交叉熵损失 |
| `grads(dpred=None)` | 计算梯度（简化形式） |

### HingeLoss

| 方法 | 描述 |
|------|------|
| `__call__(pred, target)` | 计算合页损失 |
| `grads(dpred=None)` | 计算梯度 |

交叉熵类默认 `reduction="sum"`，对整个批次求和；可使用 `CrossEntropy(reduction="mean")` 或 `SoftmaxCrossEntropy(reduction="mean")` 按样本数求均值，梯度也相应除以批次大小。MSE、BCE 和 Hinge 使用元素均值，公式中的 N 为元素总数。
