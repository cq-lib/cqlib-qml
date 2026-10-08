# models 模块教程

`models` 模块提供了量子神经网络模型的核心实现，包括纯量子神经网络 (QNN) 和混合量子-经典神经网络 (HQNN)。这些模型将 ansatz、编码器和经典层组合成完整的可训练模型。

## 模块结构

    models/
    ├── __init__.py              # 模块导出
    ├── module.py                # 模型容器基类
    ├── QNN.py                   # 纯量子神经网络
    └── HQNN.py                  # 混合量子-经典神经网络

---

## 背景与数学原理

### 量子神经网络的基本结构

量子神经网络由三个核心组件构成：

1. **编码器**：将经典数据 $x$ 映射为量子态 $|\psi(x)\rangle$
2. **变分层 (Ansatz)**：参数化酉变换 $U(\boldsymbol{\theta})$
3. **测量层**：将量子态投影为经典输出 $y$

整体映射可以表示为：

$$f(x; \boldsymbol{\theta}) = \langle 0 | U_{\text{enc}}(x)^\dagger U_{\text{ansatz}}(\boldsymbol{\theta})^\dagger H U_{\text{ansatz}}(\boldsymbol{\theta}) U_{\text{enc}}(x) | 0 \rangle$$

### QNN 与 HQNN 的架构对比

| 特性 | QNN | HQNN |
|------|-----|------|
| 架构 | 仅量子电路 | 量子电路 + 经典全连接层 |
| 数学表达 | $f(x) = \langle H \rangle$ | $f(x) = W \cdot \langle H \rangle + b$ |
| 输出 | 量子测量值 | 线性层输出 |
| 适用场景 | 简单二分类 | 多分类、复杂任务 |
| 可训练参数 | ansatz 参数 | ansatz 参数 + 线性层参数 |

---

## 模型容器 (Module)

`Module` 是组合多个组件的容器，支持灵活组合 Layer 和 Ansatz。

### 支持的组合模式

1. **linear + linear**：纯经典前馈网络
2. **ansatz + linear**：量子特征提取 + 经典分类
3. **linear + ansatz**：经典预处理 + 量子处理
4. **ansatz + ansatz**：级联量子电路

### 使用示例

    from cqlib_qml.models import Module
    from cqlib_qml.layer import Linear
    from cqlib_qml.ansatz import HEAnsatz

    # 1. 纯经典网络
    model = Module(
        Linear(in_dim=10, out_dim=5, act_fn="relu"),
        Linear(in_dim=5, out_dim=2, act_fn="sigmoid")
    )

    # 2. 量子 + 经典（HQNN 风格）
    ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    ansatz.set_measurement(readouts=[0, 1, 2])
    model = Module(
        ansatz,
        Linear(in_dim=3, out_dim=2, act_fn="sigmoid")
    )

    # 3. 经典 + 量子
    model = Module(
        Linear(in_dim=10, out_dim=ansatz.in_dim),
        ansatz
    )

    # 4. 纯量子
    ansatz1 = HEAnsatz(n_qubits=4, d=1, layers=["RY", "CX"])
    ansatz1.set_measurement(readouts=[0, 1])
    ansatz2 = HEAnsatz(n_qubits=2, d=1, layers=["RY", "CX"])
    ansatz2.set_measurement(readouts=[0])
    model = Module(ansatz1, ansatz2)

### 核心方法

| 方法 | 描述 |
|------|------|
| `forward(x=None)` | 前向传播 |
| `backward(dLdout=None)` | 反向传播 |
| `set_optimizer(optimizer)` | 设置优化器 |
| `update(cur_loss=None)` | 更新所有参数 |
| `zero_grad()` | 梯度清零 |
| `freeze()` | 冻结所有组件 |
| `unfreeze()` | 解冻所有组件 |
| `random_init()` | 随机初始化 |
| `save_checkpoint(model_path, ep, it, latest=False)` | 保存检查点 |
| `load_checkpoint(model_path)` | 加载检查点 |

### 完整使用示例

    import numpy as np
    from cqlib_qml.models import Module
    from cqlib_qml.layer import Linear
    from cqlib_qml.loss import MSELoss
    from cqlib_qml.optimizer import Adam

    # 1. 创建模型
    model = Module(
        Linear(in_dim=10, out_dim=5, act_fn="relu"),
        Linear(in_dim=5, out_dim=1, act_fn="sigmoid")
    )

    # 2. 设置优化器
    model.set_optimizer(Adam(lr=0.001))

    # 3. 随机初始化
    model.random_init()

    # 4. 训练循环
    loss_fn = MSELoss()
    X = np.random.randn(32, 10)
    target = np.random.randn(32, 1)

    for epoch in range(100):
        # 前向传播
        output = model.forward(X)

        # 计算损失
        loss = loss_fn(output, target)

        # 反向传播
        model.backward(loss_fn.grads())

        # 更新参数
        model.update()

        # 梯度清零
        model.zero_grad()

---

## 纯量子神经网络 (QNN)

### 数学原理

QNN 仅使用量子电路进行推理，输出直接来自量子测量：

$$f_{\text{QNN}}(x) = \langle 0 | U_{\text{enc}}(x)^\dagger U_{\text{ansatz}}(\boldsymbol{\theta})^\dagger H U_{\text{ansatz}}(\boldsymbol{\theta}) U_{\text{enc}}(x) | 0 \rangle$$

其中 $H$ 是测量哈密顿量（通常是 Pauli-Z）。

### 初始化参数

    QNN(
        ansatz: Ansatz,
        readouts: list = None,
        params: np.ndarray = None,
        optimizer: Union[str, dict, OptimizerBase] = "adam",
        *, random_state=None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `ansatz` | Ansatz | 参数化量子电路 |
| `readouts` | list | 测量量子比特索引 |
| `params` | np.ndarray | 初始参数值 |
| `optimizer` | str/dict/OptimizerBase | 优化器 |
| `random_state` | int/Generator/None | 控制模型自身初始化的随机流 |

### 使用示例

    import numpy as np
    from cqlib_qml.models import QNN
    from cqlib_qml.ansatz import HEAnsatz
    from cqlib_qml.encoder import AngleEncoder
    from cqlib_qml.loss import BCELoss
    from cqlib_qml.optimizer import Adam

    # 1. 创建 ansatz
    ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    ansatz.set_measurement(readouts=[0])

    # 2. 创建 QNN
    qnn = QNN(ansatz=ansatz, optimizer=Adam(lr=0.01))

    # 3. 编码数据
    encoder = AngleEncoder(mode="classical")
    X = np.random.randn(32, 4)
    circuits = encoder(X)

    # 4. 训练循环
    loss_fn = BCELoss()
    y_true = np.random.randint(0, 2, (32, 1))

    for epoch in range(50):
        # 前向传播
        expectations = qnn.forward(circuits, trainable=True)

        # 准备标签（BCE 需要 0/1）
        y_pred = (1 - expectations) / 2
        loss = loss_fn(y_pred, y_true)

        # 反向传播
        qnn.backward(loss_fn.grads(-0.5))

        # 更新
        qnn.update()
        qnn.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f}")

---

## 混合量子-经典神经网络 (HQNN)

### 数学原理

HQNN 在量子电路后添加经典全连接层：

$$f_{\text{HQNN}}(x) = W \cdot f_{\text{QNN}}(x) + b$$

其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{meas}}}$ 是权重矩阵，$b \in \mathbb{R}^{d_{\text{out}}}$ 是偏置向量。

HQNN 结合了量子计算的高维特征表示能力和经典计算的线性变换能力。

### 初始化参数

    HQNN(
        ansatz: HEAnsatz,
        out_dim: int,
        params: np.ndarray = None,
        optimizer: Union[str, dict, OptimizerBase] = "adam",
        *, readouts: list = None, random_state=None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `ansatz` | HEAnsatz | 参数化量子电路（必须是 HEAnsatz） |
| `out_dim` | int | 输出维度 |
| `readouts` | list/None | 测量量子比特索引（仅关键字） |
| `params` | np.ndarray | 初始参数值 |
| `optimizer` | str/dict/OptimizerBase | 优化器 |
| `random_state` | int/Generator/None | 控制模型自身初始化的随机流 |

### 使用示例

    import numpy as np
    from cqlib_qml.models import HQNN
    from cqlib_qml.ansatz import HEAnsatz
    from cqlib_qml.encoder import AngleEncoder
    from cqlib_qml.loss import SoftmaxCrossEntropy
    from cqlib_qml.optimizer import Adam

    # 1. 创建 ansatz（测量所有量子比特）
    ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    ansatz.set_measurement(readouts=[0, 1, 2, 3])

    # 2. 创建 HQNN（3 分类）
    hqnn = HQNN(ansatz=ansatz, out_dim=3, optimizer=Adam(lr=0.01))

    # 3. 编码数据
    encoder = AngleEncoder(mode="classical")
    X = np.random.randn(32, 4)
    circuits = encoder(X)

    # 4. 训练循环
    loss_fn = SoftmaxCrossEntropy()
    y_true = np.random.randint(0, 3, (32,))
    y_onehot = np.zeros((32, 3))
    y_onehot[np.arange(32), y_true] = 1

    for epoch in range(50):
        # 前向传播
        logits = hqnn.forward(circuits, trainable=True)

        # 计算损失
        loss = loss_fn(logits, y_onehot)

        # 反向传播
        hqnn.backward(loss_fn.grads())

        # 更新
        hqnn.update()
        hqnn.zero_grad()

        if epoch % 10 == 0:
            acc = (logits.argmax(axis=1) == y_true).mean()
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

---

## 检查点保存与加载

### 保存检查点

    # 保存模型状态
    model.save_checkpoint(
        model_path="./checkpoints/",
        ep=epoch,
        it=iteration,
        latest=False
    )

    # 保存最新检查点（覆盖 model.npy）
    model.save_checkpoint(
        model_path="./checkpoints/",
        ep=epoch,
        it=iteration,
        latest=True
    )

### 加载检查点

    # 加载指定检查点
    ep, it = model.load_checkpoint("./checkpoints/model.npy")

    # 加载最新检查点（自动查找最大 epoch 和 iteration）
    ep, it = model.load_checkpoint("./checkpoints/")

---

## 最佳实践

### 1. 模型选择

| 任务类型 | 推荐模型 | 理由 |
|----------|----------|------|
| 二分类 | QNN | 简单直接 |
| 多分类 | HQNN | 经典层提供更好的分类边界 |
| 回归 | HQNN | 经典层可输出连续值 |
| 特征提取 | QNN | 直接使用量子测量值 |

### 2. 参数初始化

    # QNN 参数初始化
    params = np.random.randn(ansatz.in_dim) * 0.1
    qnn = QNN(ansatz=ansatz, params=params)

    # HQNN 参数初始化
    hqnn = HQNN(ansatz=ansatz, out_dim=3, params=params)

### 3. 冻结和解冻

    # 冻结整个模型（微调场景）
    model.freeze()

    # 解冻特定层
    model._nets[0].unfreeze()  # 只解冻第一个组件

---

## 常见问题排查

### 问题 1: HQNN 报错 "Expected HEAnsatz"

**原因**：HQNN 要求传入 HEAnsatz。

**解决方案**：

    # 使用 HEAnsatz
    ansatz = HEAnsatz(n_qubits=4, d=2, layers=["RY", "CX"])
    hqnn = HQNN(ansatz=ansatz, out_dim=3)

    # 不能使用其他 ansatz
    # ansatz = BasicQNN(n_qubits=4, layers=["XX"])  # 报错

### 问题 2: 构造 Module 时报错 "Ansatz must have measurements"

**原因**：ansatz 未设置测量。

**解决方案**：

    ansatz.set_measurement(readouts=[0, 1, 2])

### 问题 3: 检查点加载失败

**原因**：模型结构不匹配。

**解决方案**：

    # 确保模型结构完全一致
    # 加载前打印摘要
    print(model._nets[0].summary)

---

## API 参考

### Module

| 方法 | 描述 |
|------|------|
| `forward(x=None)` | 前向传播 |
| `backward(dLdout=None)` | 反向传播 |
| `set_optimizer(optimizer)` | 设置优化器 |
| `update(cur_loss=None)` | 更新参数 |
| `zero_grad()` | 梯度清零 |
| `freeze()` | 冻结 |
| `unfreeze()` | 解冻 |
| `random_init()` | 随机初始化 |
| `save_checkpoint(model_path, ep, it, latest=False)` | 保存检查点 |
| `load_checkpoint(model_path)` | 加载检查点 |

### QNN

| 参数 | 类型 | 描述 |
|------|------|------|
| `ansatz` | Ansatz | 参数化量子电路 |
| `readouts` | list | 测量量子比特 |
| `params` | np.ndarray | 初始参数 |
| `optimizer` | str/dict/OptimizerBase | 优化器 |
| `random_state` | int/Generator/None | 控制模型自身初始化的随机流 |

### HQNN

| 参数 | 类型 | 描述 |
|------|------|------|
| `ansatz` | HEAnsatz | 参数化量子电路 |
| `out_dim` | int | 输出维度 |
| `params` | np.ndarray | 初始参数 |
| `optimizer` | str/dict/OptimizerBase | 优化器 |
| `random_state` | int/Generator/None | 控制模型自身初始化的随机流 |
## 训练状态与恢复范围

量子层、Linear 和 Module 的 backward 统一累加权重梯度，输入梯度固定为 `(batch, inputs)`；单样本不压缩 batch 轴。`zero_grad()` 清空累计梯度并保留最近前向缓存，冻结组件也可清零。

`freeze()` 持续禁止权重梯度和更新，但中间量子层继续计算输入梯度。`train()/eval()` 独立设置执行模式，不控制记录。QNN/HQNN 的历史入口 `trainable=False`，或 Module/Ansatz 的 `retain_derived=False`，使旧反向缓存失效，保留已累计梯度；推理后需要重新记录前向才能 backward。要丢弃累计梯度，请显式 zero_grad。

初始化 `params` 是匹配 `ansatz.weight_params` 的一维有限实数向量；未声明角色时仍对应全部线路符号。QNN/HQNN/Module 的 random_state 用于自身初始化，不依赖全局 seed。

Checkpoint 初始格式版本为 `format_version=1`，保存冻结、角色、微分器、优化器、调度器和随机状态。有未清零梯度时不能保存。使用 `data_loader=loader` 可保存及恢复项目 DataLoader 的排列与下一批游标；外部数据管线需自行保存数据进度。loader 已完成 epoch 时返回 `(epoch+1, 0)`，否则保留 `(epoch, iteration+1)`。加载时拒绝缺少版本、版本不受支持或必要字段不完整的文件。

加载会清空 Jacobian、梯度和反向缓存；恢复后直接 update 不推进优化器。详细混合网络、累积训练及恢复示例见 [训练契约](training_contracts.md)。

Checkpoint 恢复模型参数、线路结构、测量配置及已保存的优化器/调度器状态，返回 `(epoch, iteration + 1)`。它没有完整保存随机数状态、数据迭代顺序、微分器配置和冻结状态，因此不保证任意训练过程精确续跑。

加载 checkpoint 或调用 `load_params()` 会清除旧梯度、Jacobian 和反向缓存；恢复后直接调用 `update()` 不修改参数或推进优化器。
