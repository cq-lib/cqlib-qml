# optimizer 模块教程

`optimizer` 模块提供了量子机器学习中常用的优化算法实现，用于更新模型参数以最小化损失函数。所有优化器均支持学习率调度和梯度裁剪。

## 模块结构

    cqlib_qml/optimizer.py

---

## 背景与数学原理

### 什么是优化器

优化器负责根据梯度更新模型参数。给定参数 $\boldsymbol{\theta}$ 和梯度 $\nabla_{\boldsymbol{\theta}} L$，优化器计算更新量 $\Delta \boldsymbol{\theta}$：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \text{update}(\nabla_{\boldsymbol{\theta}} L)$$

其中 $\eta$ 是学习率，$\text{update}$ 是优化器定义的更新规则。

### 优化器的核心功能

1. **参数更新**：根据梯度调整参数值
2. **学习率调度**：动态调整学习率
3. **梯度裁剪**：防止梯度爆炸

---

## 优化器基类 (OptimizerBase)

`OptimizerBase` 是所有优化器的抽象基类，定义了参数更新、学习率调度和状态管理的统一接口。

### 核心属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `cache` | dict | 优化器状态缓存（动量、运行平均值等） |
| `cur_step` | int | 当前迭代步数 |
| `hyperparameters` | dict | 优化器超参数 |
| `lr` | float | 学习率 |
| `lr_scheduler` | SchedulerBase | 学习率调度器 |

### 使用示例

`reset_state()` 清空参数缓存、步数、学习率缓存和调度器历史，同时保留具体类型及配置；`reset_step()` 只清步数。VQC 的默认 fit 和 warm start 在私有优化器副本上调用 `reset_state()`。自定义优化器有额外训练历史时应覆盖该方法并调用 `super().reset_state()`；自定义有状态调度器同样需要覆盖 `SchedulerBase.reset_state()`。实例形式支持自定义类型，字符串及字典配置重建仍使用内置类型。

    from cqlib_qml.optimizer import OptimizerBase
    import numpy as np

    # 自定义优化器
    class MyOptimizer(OptimizerBase):
        def __init__(self, lr=0.01):
            super().__init__(lr)
            self.hyperparameters = {"id": "MyOptimizer", "lr": lr}

        def update(self, param, param_grad, param_name, cur_loss=None):
            return param - self.lr * param_grad

    # 使用自定义优化器
    opt = MyOptimizer(lr=0.01)
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    new_param = opt.update(param, grad, "weight")
    print(f"更新后参数: {new_param}")

**输出：**

    更新后参数: [0.999 1.998]

---

## 优化器初始化器 (OptimizerInitializer)

`OptimizerInitializer` 提供从字符串、字典或现有实例创建优化器的工厂方法。

### 支持的格式

    from cqlib_qml.optimizer import OptimizerInitializer, Adam

    # 1. 字符串
    opt = OptimizerInitializer("adam")()
    opt = OptimizerInitializer("adam(lr=0.001)")()
    opt = OptimizerInitializer("sgd(lr=0.01, momentum=0.9)")()

    # 2. 字典
    opt = OptimizerInitializer({
        "hyperparameters": {"id": "Adam", "lr": 0.001},
        "cache": {}
    })()

    # 3. 现有优化器实例
    opt = OptimizerInitializer(Adam(lr=0.001))()

    # 4. None（返回默认 SGD）
    opt = OptimizerInitializer(None)()

---

## 随机梯度下降 (SGD)

### 数学原理

SGD 是最基本的优化算法，沿着梯度的反方向更新参数：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

**带动量**：

$$v_t = \mu v_{t-1} + \eta \nabla L(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_t$$

其中 $\mu$ 是动量系数（通常取 0.9）。

### 初始化参数

    SGD(
        lr: float = 0.01,
        momentum: float = 0.0,
        clip_norm: float = None,
        lr_scheduler = None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `lr` | float | 学习率 |
| `momentum` | float | 动量系数（0 到 1） |
| `clip_norm` | float | 梯度裁剪阈值 |
| `lr_scheduler` | Scheduler | 学习率调度器 |

### 使用示例

    import numpy as np
    from cqlib_qml.optimizer import SGD

    # 1. 基本 SGD
    opt = SGD(lr=0.01)

    # 2. SGD 带动量
    opt = SGD(lr=0.01, momentum=0.9)

    # 3. SGD 带梯度裁剪
    opt = SGD(lr=0.01, clip_norm=1.0)

    # 4. 参数更新
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])

    for step in range(5):
        param = opt.update(param, grad, "weight")
        opt.step()
        print(f"Step {step+1}: {param}")

**输出：**

    Step 1: [0.999 1.998]
    Step 2: [0.998 1.996]
    Step 3: [0.997 1.994]
    Step 4: [0.996 1.992]
    Step 5: [0.995 1.99 ]

---

## AdaGrad

### 数学原理

AdaGrad 为每个参数自适应调整学习率，频繁更新的参数学习率较小，稀疏更新的参数学习率较大：

$$G_t = G_{t-1} + (\nabla L(\theta_t))^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)$$

其中 $G_t$ 是历史梯度平方和，$\epsilon$ 是平滑项（防止除零）。

### 初始化参数

    AdaGrad(
        lr: float = 0.01,
        eps: float = 1e-7,
        clip_norm: float = None,
        lr_scheduler = None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `lr` | float | 学习率 |
| `eps` | float | 平滑项 |
| `clip_norm` | float | 梯度裁剪阈值 |
| `lr_scheduler` | Scheduler | 学习率调度器 |

### 使用示例

    import numpy as np
    from cqlib_qml.optimizer import AdaGrad

    opt = AdaGrad(lr=0.01)

    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])

    for step in range(5):
        param = opt.update(param, grad, "weight")
        opt.step()
        print(f"Step {step+1}: {param}")

**输出：**

    Step 1: [0.99000001 1.99      ]
    Step 2: [0.98292895 1.98292894]
    Step 3: [0.97715545 1.97715544]
    Step 4: [0.97215545 1.97215544]
    Step 5: [0.96768332 1.9676833 ]

---

## RMSProp

### 数学原理

RMSProp 使用指数移动平均替代 AdaGrad 的累积平方和，避免学习率单调递减：

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) (\nabla L(\theta_t))^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla L(\theta_t)$$

其中 $\beta$ 是衰减率（通常取 0.9）。

### 初始化参数

    RMSProp(
        lr: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-7,
        clip_norm: float = None,
        lr_scheduler = None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `lr` | float | 学习率 |
| `decay` | float | 衰减率 |
| `eps` | float | 平滑项 |
| `clip_norm` | float | 梯度裁剪阈值 |
| `lr_scheduler` | Scheduler | 学习率调度器 |

### 使用示例

    import numpy as np
    from cqlib_qml.optimizer import RMSProp

    opt = RMSProp(lr=0.001, decay=0.9)

    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])

    for step in range(5):
        param = opt.update(param, grad, "weight")
        opt.step()
        print(f"Step {step+1}: {param}")

**输出：**

    Step 1: [0.99683773 1.99683773]
    Step 2: [0.99454358 1.99454357]
    Step 3: [0.99262264 1.99262263]
    Step 4: [0.99091741 1.9909174 ]
    Step 5: [0.98935474 1.98935472]

---

## Adam

### 数学原理

Adam (Adaptive Moment Estimation) 结合了动量（一阶矩）和 RMSProp（二阶矩）的优点：

**一阶矩估计**（均值）：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$

**二阶矩估计**（方差）：

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$

**偏差校正**：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**参数更新**：

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中 $\beta_1$ 通常取 0.9，$\beta_2$ 通常取 0.999。

### 初始化参数

    Adam(
        lr: float = 0.001,
        decay1: float = 0.9,
        decay2: float = 0.999,
        eps: float = 1e-7,
        clip_norm: float = None,
        lr_scheduler = None
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `lr` | float | 学习率 |
| `decay1` | float | 一阶矩衰减率 |
| `decay2` | float | 二阶矩衰减率 |
| `eps` | float | 平滑项 |
| `clip_norm` | float | 梯度裁剪阈值 |
| `lr_scheduler` | Scheduler | 学习率调度器 |

### 使用示例

    import numpy as np
    from cqlib_qml.optimizer import Adam

    opt = Adam(lr=0.001)

    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])

    for step in range(5):
        param = opt.update(param, grad, "weight")
        opt.step()
        print(f"Step {step+1}: {param}")

**输出：**

    Step 1: [0.999 1.999]
    Step 2: [0.998 1.998]
    Step 3: [0.997 1.997]
    Step 4: [0.996 1.996]
    Step 5: [0.995 1.995]

---

## 优化器对比

### 算法对比

| 优化器 | 自适应学习率 | 动量 | 适用场景 |
|--------|-------------|------|----------|
| SGD | 否 | 可选 | 简单任务，需要精细调参 |
| AdaGrad | 是 | 否 | 稀疏数据 |
| RMSProp | 是 | 否 | 非平稳目标 |
| Adam | 是 | 是 | 通用，推荐首选 |

### 收敛速度对比（概念）

    # SGD: 稳定但慢
    # Adam: 快速收敛
    # RMSProp: 中等速度

### 使用建议

| 场景 | 推荐优化器 | 理由 |
|------|------------|------|
| 首次尝试 | Adam | 默认参数通常表现良好 |
| 需要解释性 | SGD | 参数更新规则简单 |
| 稀疏数据 | AdaGrad | 自适应学习率适合稀疏特征 |
| 内存受限 | SGD / RMSProp | Adam 需要存储更多状态 |

---

## 学习率调度

### 调度器集成

所有优化器都支持学习率调度器：

    from cqlib_qml.optimizer import Adam
    from cqlib_qml.scheduler import ExponentialScheduler

    # 创建指数衰减调度器
    scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.95)

    # 集成到优化器
    opt = Adam(lr=0.01, lr_scheduler=scheduler)

    # 或动态设置
    opt.set_scheduler(scheduler)

### 调度器移除

    # 移除调度器（使用常数学习率）
    opt.remove_scheduler()

---

## 最佳实践

### 1. 学习率选择

| 优化器 | 常用学习率范围 |
|--------|---------------|
| SGD | 0.01 - 0.1 |
| Adam | 0.0001 - 0.001 |
| RMSProp | 0.0001 - 0.001 |
| AdaGrad | 0.001 - 0.01 |

    # 从较大学习率开始，逐步降低
    opt = Adam(lr=0.001)

### 2. 梯度裁剪

    # 防止梯度爆炸
    opt = Adam(lr=0.001, clip_norm=1.0)

### 3. 动量设置

    # SGD 动量通常取 0.9
    opt = SGD(lr=0.01, momentum=0.9)

    # Adam 的 beta1 通常取 0.9
    opt = Adam(lr=0.001, decay1=0.9)

### 4. 完整训练循环

    from cqlib_qml.optimizer import Adam
    import numpy as np

    # 1. 创建优化器
    opt = Adam(lr=0.001)

    # 2. 初始化参数
    params = {
        "weight": np.array([1.0, 2.0]),
        "bias": np.array([0.0])
    }

    # 3. 训练循环
    for epoch in range(100):
        # 计算梯度
        gradients = {
            "weight": np.array([0.1, 0.2]),
            "bias": np.array([0.05])
        }

        # 更新参数
        for name in params:
            params[name] = opt.update(params[name], gradients[name], name)

        # 步进
        opt.step()

---

## 常见问题排查

### 问题 1: 学习率过大导致不收敛

**现象**：损失值震荡或爆炸。

**解决方案**：

    # 降低学习率
    opt = Adam(lr=0.0001)

    # 使用学习率调度
    from cqlib_qml.scheduler import ExponentialScheduler
    opt = Adam(lr=0.01, lr_scheduler=ExponentialScheduler(decay=0.95))

### 问题 2: 学习率过小导致收敛缓慢

**现象**：损失值下降极慢。

**解决方案**：

    # 提高学习率
    opt = Adam(lr=0.01)

    # 使用 Adam（比 SGD 收敛更快）
    opt = Adam(lr=0.001)

### 问题 3: 梯度爆炸

**现象**：参数值变为 NaN 或极大。

**解决方案**：

    # 使用梯度裁剪
    opt = Adam(lr=0.001, clip_norm=1.0)

### 问题 4: 优化器状态无法保存

**解决方案**：

    # 保存优化器状态
    import pickle
    with open("optimizer_state.pkl", "wb") as f:
        pickle.dump(opt.state_dict(), f)

    # 恢复优化器状态
    with open("optimizer_state.pkl", "rb") as f:
        state = pickle.load(f)
    from cqlib_qml.optimizer import OptimizerInitializer
    opt = OptimizerInitializer(state)()

---

## API 参考

### OptimizerBase

| 方法 | 描述 |
|------|------|
| `__call__(param, param_grad, param_name, cur_loss=None)` | 更新参数 |
| `update(param, param_grad, param_name, cur_loss=None)` | 更新参数 |
| `step()` | 步进计数器 |
| `reset_step()` | 重置计数器 |
| `reset_state()` | 清空优化器和调度器训练历史，保留类型与配置 |
| `set_scheduler(scheduler)` | 设置学习率调度器 |
| `remove_scheduler()` | 移除调度器 |
| `copy()` | 复制优化器 |

### SGD

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lr` | 0.01 | 学习率 |
| `momentum` | 0.0 | 动量系数 |
| `clip_norm` | None | 梯度裁剪阈值 |
| `lr_scheduler` | None | 学习率调度器 |

### AdaGrad

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lr` | 0.01 | 学习率 |
| `eps` | 1e-7 | 平滑项 |
| `clip_norm` | None | 梯度裁剪阈值 |
| `lr_scheduler` | None | 学习率调度器 |

### RMSProp

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lr` | 0.001 | 学习率 |
| `decay` | 0.9 | 衰减率 |
| `eps` | 1e-7 | 平滑项 |
| `clip_norm` | None | 梯度裁剪阈值 |
| `lr_scheduler` | None | 学习率调度器 |

### Adam

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lr` | 0.001 | 学习率 |
| `decay1` | 0.9 | 一阶矩衰减率 |
| `decay2` | 0.999 | 二阶矩衰减率 |
| `eps` | 1e-7 | 平滑项 |
| `clip_norm` | None | 梯度裁剪阈值 |
| `lr_scheduler` | None | 学习率调度器 |
## 字符串配置校验

配置名称不区分大小写，参数值保持原样。支持例如 `sgd(lr=.2, clip_norm=1.0, lr_scheduler=exponential(initial_lr=.2, stage_length=5, staircase=True))`，以及以字符串传入的调度器。仅接受注册名称、关键字参数和合法字面量，未知参数和任意 Python 表达式报错。历史状态字典的恢复接口保持兼容。
