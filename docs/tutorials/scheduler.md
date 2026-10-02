# scheduler 模块教程

`scheduler` 模块提供了学习率调度策略，用于在训练过程中动态调整学习率。合理的调度策略可以显著提升模型的收敛速度和最终性能。

## 模块结构

    cqlib_qml/scheduler.py

---

## 背景与数学原理

### 为什么需要学习率调度？

在训练过程中，学习率的选择对模型收敛至关重要：
- **学习率过大**：损失函数震荡，难以收敛
- **学习率过小**：收敛速度极慢，可能陷入局部最优

动态调整学习率可以在不同阶段采用不同的学习率：
- **早期**：使用较大学习率快速接近最优解
- **后期**：使用较小学习率精细调整参数

### 调度策略分类

| 策略类型 | 调度方式 | 典型代表 |
|----------|----------|----------|
| 固定调度 | 预定义规则 | ExponentialScheduler, NoamScheduler |
| 自适应调度 | 根据损失动态调整 | KingScheduler |
| 常数调度 | 固定学习率 | ConstantScheduler |

---

## 调度器基类 (SchedulerBase)

`SchedulerBase` 是所有学习率调度器的抽象基类，定义了调度器的统一接口。

### 核心方法

| 方法 | 描述 |
|------|------|
| `__call__(step, cur_loss=None)` | 获取当前学习率 |
| `learning_rate(step, **kwargs)` | 计算学习率 |
| `copy()` | 复制调度器 |
| `set_params(hparam_dict)` | 设置超参数 |

### 使用示例

    from cqlib_qml.scheduler import SchedulerBase

    # 自定义调度器
    class MyScheduler(SchedulerBase):
        def __init__(self, initial_lr=0.01, decay=0.9):
            super().__init__()
            self.initial_lr = initial_lr
            self.decay = decay
            self.hyperparameters = {
                "id": "MyScheduler",
                "initial_lr": initial_lr,
                "decay": decay
            }

        def learning_rate(self, step, **kwargs):
            return self.initial_lr * self.decay ** step

    # 使用自定义调度器
    scheduler = MyScheduler(initial_lr=0.01, decay=0.9)
    lr = scheduler(step=10)
    print(f"Step 10 学习率: {lr:.6f}")

**输出：**

    Step 10 学习率: 0.003487

---

## 调度器初始化器 (SchedulerInitializer)

`SchedulerInitializer` 提供从字符串、字典或现有实例创建调度器的工厂方法。

### 支持的格式

    from cqlib_qml.scheduler import SchedulerInitializer, ConstantScheduler

    # 1. 从字符串创建
    scheduler = SchedulerInitializer("constant(lr=0.01)")()
    scheduler = SchedulerInitializer("exponential(initial_lr=0.01, decay=0.9)")()
    scheduler = SchedulerInitializer("noam(model_dim=512, warmup_steps=4000)")()

    # 2. 从字典创建
    scheduler = SchedulerInitializer({
        "hyperparameters": {
            "id": "ExponentialScheduler",
            "initial_lr": 0.01,
            "decay": 0.9
        }
    })()

    # 3. 从现有调度器实例创建
    scheduler = SchedulerInitializer(ConstantScheduler(lr=0.01))()

---

## 常数调度器 (ConstantScheduler)

### 数学原理

常数调度器返回固定的学习率，不随训练步数变化：

$$\eta_t = \eta_0$$

其中 $\eta_0$ 是初始学习率。

### 使用示例

    from cqlib_qml.scheduler import ConstantScheduler

    scheduler = ConstantScheduler(lr=0.01)

    for step in [0, 100, 1000]:
        lr = scheduler(step=step)
        print(f"Step {step}: {lr}")

**输出：**

    Step 0: 0.01
    Step 100: 0.01
    Step 1000: 0.01

---

## 指数衰减调度器 (ExponentialScheduler)

### 数学原理

指数衰减调度器按阶段指数衰减学习率：

**平滑衰减**（`staircase=False`）：

$$\eta_t = \eta_0 \cdot \gamma^{t / T}$$

**阶梯衰减**（`staircase=True`）：

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}$$

其中 $\eta_0$ 是初始学习率，$\gamma$ 是衰减因子，$T$ 是阶段长度。

### 初始化参数

    ExponentialScheduler(
        initial_lr: float = 0.01,
        stage_length: int = 500,
        staircase: bool = False,
        decay: float = 0.1
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `initial_lr` | float | 初始学习率 |
| `stage_length` | int | 阶段长度（步数） |
| `staircase` | bool | 是否阶梯式衰减 |
| `decay` | float | 衰减因子 |

### 使用示例

    from cqlib_qml.scheduler import ExponentialScheduler

    # 平滑衰减
    scheduler = ExponentialScheduler(
        initial_lr=0.01,
        stage_length=100,
        staircase=False,
        decay=0.5
    )

    print("平滑衰减:")
    for step in [0, 50, 100, 150, 200]:
        lr = scheduler(step=step)
        print(f"  Step {step}: {lr:.6f}")

    # 阶梯式衰减
    scheduler = ExponentialScheduler(
        initial_lr=0.01,
        stage_length=100,
        staircase=True,
        decay=0.5
    )

    print("\n阶梯式衰减:")
    for step in [0, 50, 100, 150, 200]:
        lr = scheduler(step=step)
        print(f"  Step {step}: {lr:.6f}")

**输出：**

    平滑衰减:
    Step 0: 0.010000
    Step 50: 0.007071
    Step 100: 0.005000
    Step 150: 0.003536
    Step 200: 0.002500

    阶梯式衰减:
    Step 0: 0.010000
    Step 50: 0.010000
    Step 100: 0.005000
    Step 150: 0.005000
    Step 200: 0.002500

---

## Noam 调度器 (NoamScheduler)

### 数学原理

Noam 调度器（Transformer 论文中提出）在 warmup 阶段线性增加学习率，之后按步数的平方根倒数衰减：

$$\eta_t = \text{scale} \cdot d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5}, t \cdot \text{warmup}^{-1.5}\right)$$

其中 $d_{\text{model}}$ 是模型维度，$\text{warmup}$ 是预热步数。

### 初始化参数

    NoamScheduler(
        model_dim: int = 512,
        scale_factor: int = 1,
        warmup_steps: int = 4000
    )

| 参数 | 类型 | 描述 |
|------|------|------|
| `model_dim` | int | 模型维度 |
| `scale_factor` | int | 缩放因子 |
| `warmup_steps` | int | 预热步数 |

### 使用示例

    from cqlib_qml.scheduler import NoamScheduler

    scheduler = NoamScheduler(
        model_dim=512,
        scale_factor=1,
        warmup_steps=100
    )

    print("Noam 调度器:")
    for step in [1, 50, 100, 200, 500, 1000]:
        lr = scheduler(step=step)
        print(f"  Step {step}: {lr:.6f}")

**输出：**

    Noam 调度器:
    Step 1: 0.000044
    Step 50: 0.002210
    Step 100: 0.004419
    Step 200: 0.003125
    Step 500: 0.001976
    Step 1000: 0.001398

---

## 调度器与优化器集成

### 方式一：初始化时传入

    from cqlib_qml.optimizer import Adam
    from cqlib_qml.scheduler import ExponentialScheduler

    scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.95)
    optimizer = Adam(lr=0.01, lr_scheduler=scheduler)

### 方式二：动态设置

    optimizer = Adam(lr=0.01)
    optimizer.set_scheduler(ExponentialScheduler(initial_lr=0.01, decay=0.95))

### 方式三：移除调度器

    optimizer.remove_scheduler()

### 完整训练循环

    from cqlib_qml.optimizer import Adam
    from cqlib_qml.scheduler import NoamScheduler
    import numpy as np

    # 1. 创建调度器和优化器
    scheduler = NoamScheduler(model_dim=512, warmup_steps=4000)
    optimizer = Adam(lr=0.001, lr_scheduler=scheduler)

    # 2. 初始化参数
    params = {"weight": np.array([1.0, 2.0])}

    # 3. 训练循环
    for step in range(10000):
        # 计算梯度（模拟）
        grad = np.array([0.1 * np.exp(-step / 1000), 0.2 * np.exp(-step / 1000)])

        # 更新参数
        params["weight"] = optimizer.update(params["weight"], grad, "weight")

        # 步进（触发调度器更新）
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}: lr={optimizer.lr_scheduler(step=step):.6f}")

**输出：**

    Step 0: lr=0.000000
    Step 1000: lr=0.000175
    Step 2000: lr=0.000349
    Step 3000: lr=0.000524
    Step 4000: lr=0.000699
    Step 5000: lr=0.000625
    Step 6000: lr=0.000571
    Step 7000: lr=0.000528
    Step 8000: lr=0.000494
    Step 9000: lr=0.000466


---

## 调度器选择指南

| 场景 | 推荐调度器 | 理由 |
|------|------------|------|
| 通用训练 | ExponentialScheduler | 简单有效，易于调参 |
| Transformer 风格 | NoamScheduler | 包含 warmup，训练稳定 |
| 固定学习率 | ConstantScheduler | 无需调整 |
| 自适应调整 | KingScheduler | 自动根据 loss 调整 |

---

## 最佳实践

### 1. 选择合适的衰减率

    # 快速衰减（适合短训练）
    scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.5, stage_length=100)

    # 慢速衰减（适合长训练）
    scheduler = ExponentialScheduler(initial_lr=0.01, decay=0.95, stage_length=1000)

### 2. 设置合理的 warmup 步数

    # 小数据集：较短 warmup
    scheduler = NoamScheduler(warmup_steps=1000)

    # 大数据集：较长 warmup
    scheduler = NoamScheduler(warmup_steps=10000)

### 3. 调度器状态保存与恢复

    # 保存调度器状态
    state = scheduler.state_dict()

    # 恢复超参数和运行状态（包括 King 的损失历史与当前学习率）
    from cqlib_qml.scheduler import SchedulerInitializer
    scheduler = SchedulerInitializer(state)()

---

## API 参考

### SchedulerBase

| 方法 | 描述 |
|------|------|
| `__call__(step, cur_loss=None)` | 获取当前学习率 |
| `learning_rate(step, **kwargs)` | 计算学习率 |
| `copy()` | 复制调度器 |
| `set_params(hparam_dict)` | 设置超参数 |

### ConstantScheduler

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `lr` | 0.01 | 固定学习率 |

### ExponentialScheduler

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `initial_lr` | 0.01 | 初始学习率 |
| `stage_length` | 500 | 阶段长度 |
| `staircase` | False | 是否阶梯式衰减 |
| `decay` | 0.1 | 衰减因子 |

### NoamScheduler

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `model_dim` | 512 | 模型维度 |
| `scale_factor` | 1 | 缩放因子 |
| `warmup_steps` | 4000 | 预热步数 |

## KingScheduler：根据损失自动衰减

KingScheduler 对损失历史拟合线性趋势；持续缺乏下降趋势时，学习率乘以 `decay`。`patience` 为正整数，少于三个观测时保持学习率以避免不可靠的方差估计。调用时必须提供 `cur_loss`。

```python
from cqlib_qml.scheduler import KingScheduler
from cqlib_qml.optimizer import SGD
import numpy as np

opt = SGD(lr_scheduler=KingScheduler(initial_lr=0.1, patience=3, decay=0.5))
weight = np.array([1.0])
for step in range(8):
    opt.step()
    weight = opt.update(weight, np.array([0.1]), "weight", cur_loss=1.0)
print(opt.lr_scheduler.current_lr)
```

一个优化步骤中的所有参数使用同一个学习率；损失历史每个步骤只更新一次。直接调用调度器时，调用方负责每步仅调用一次。
