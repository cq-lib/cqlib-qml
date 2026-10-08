# 参数、梯度、重训与恢复契约

## 同一线路的输入与权重

在线路构造完成后声明一个完整、互斥的符号分区。列表顺序决定输入列和权重向量的顺序；接受符号名或单符号 `Parameter`，拒绝重复、角色冲突、未知符号和遗漏。声明后增加新符号需要重新声明角色。

```python
import numpy as np
from cqlib.circuit import Parameter
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.layer import Linear
from cqlib_qml.models import Module
from cqlib_qml.loss import MSELoss

x, theta = Parameter("x"), Parameter("theta")
quantum = Ansatz(1, random_state=7)
quantum.ry(0, x)
quantum.rx(0, theta)
quantum.set_measurement(readouts=[0])
quantum.set_parameter_roles(input_params=[x], weight_params=[theta])
quantum.assign_weights([0.3])
model = Module(Linear(2, 1, random_state=8), quantum,
               Linear(1, 1, random_state=9))
model.set_optimizer("sgd(lr=0.02)")
X = np.array([[0.1, 0.2], [0.4, 0.3]])
y = np.array([[0.2], [-0.1]])
loss_fn = MSELoss()
model.zero_grad()
output = model.forward(X)
loss = loss_fn(output, y)
dX = model.backward(loss_fn.grads())
assert output.shape == (2, 1) and dX.shape == (2, 2)
assert quantum.weight_gradients.shape == (1,)
model.update(cur_loss=loss)
model.zero_grad()
```

`forward(X)` 同时绑定每行输入和所有样本共享的持久权重。`assign_weights` 接受完整一维向量或完整符号映射，拒绝非有限值、复数和错误维度。`weights` 返回声明顺序的权重副本。显式角色下 `assign_parameters` 作为权重绑定入口，不保存运行时输入；输入由 `forward(X)` 提供。

| 数据 | 形状 |
|---|---|
| 输入 X | `(B, I)`，一维输入转换为 `(1, I)` |
| 输出、上游梯度 | `(B, O)` |
| `backward(dY)` 返回值 | `(B, I)`，没有数值输入的量子源为 `(B, 0)` |
| `input_jacobian` | `(B, O, I)` |
| `weight_jacobian` | `(B, O, W)` |
| `weight_gradients` | `(W,)`，沿输出和 batch 轴收缩后的累计梯度 |
| `gradients` | 以权重符号名为键的累计梯度映射 |
| `jacobian` | 以符号名为键的 `(B, O)` Jacobian 副本 |

单样本始终保留 batch 轴。`backward(None)` 对输出总和求导；查询 Jacobian 使用上表属性。伴随法和参数偏移法遵守相同契约；`set_differentiator("parameter_shift", shift=np.pi / 4)` 的方法和配置会保存到 checkpoint。

未声明角色的旧调用仍保留绑定方式：`forward()` 使用内部权重，`forward(X)` 将全部符号作为本次输入。显式模式的 `in_dim` 为输入列数；旧模式的 `in_dim` 仍为线路符号数。`num_inputs/num_weights` 描述显式角色；旧模式的输入角色由本次是否提供 X 决定。组合 `f(x, θ)` 应声明角色。

## 累积、冻结与记录

量子层和 Linear 都在每次 backward 时**累加**参数梯度；返回的输入梯度只对应本次调用。新前向替换最近一次反向缓存，保留累计梯度。重复 backward 复用最近一次缓存并继续累加。只有最近一次记录的前向可以反向，缓存不支持多条同时保留的图。

- `zero_grad()`：清空累计参数梯度，保留最近一次前向缓存；冻结组件也可调用。
- `freeze()/unfreeze()`：控制权重训练。冻结时清空已有权重梯度，继续计算输入梯度，因此前面的经典层仍可训练。
- `train()/eval()`：设置执行模式，独立于权重冻结和梯度记录。当前确定性层在 eval 下仍可记录梯度。
- `forward(..., retain_derived=False)`：不记录本次反向缓存，使旧缓存失效，保留已累计的参数梯度。之后 backward 会报错，重新执行记录前向即可。
- QNN/HQNN 的历史参数 `trainable=False` 表示上述关闭记录操作，不改变持久冻结状态。输入校验失败也会使旧缓存失效。
- `update()`：使用有效累计梯度；冻结或没有累计梯度时不推进优化器。参数更新后旧前向缓存失效，累计梯度不会自动清零。再次更新仍会使用这些梯度，常规训练应立即 `zero_grad()`。

层内只按 batch 求和，损失缩放由调用者决定。完整 batch 为 N，微批次为 n，若每个微批次损失是本地均值，应将其梯度乘以 `n/N`，全部反向后再更新一次。不等长最后一批同样按实际 n 缩放。

```python
model.zero_grad()
for start, end in [(0, 1), (1, 2)]:
    predictions = model.forward(X[start:end])
    loss_fn(predictions, y[start:end])
    model.backward(loss_fn.grads() * ((end - start) / len(X)))
model.update()
model.zero_grad()
```

## VQC：重训、warm start 和完整恢复

`VQC(..., warm_start=False, initial_point=None, random_state=None)` 的构造对象是模板，fit 不修改调用者的 Ansatz、编码器或优化器。学习组件通过 `ansatz_`、`encoder_` 读取；sklearn clone 只复制构造配置。

| 路径 | 权重 | 优化器和调度器 | RNG、进度 |
|---|---|---|---|
| 默认 `fit(X, y)` | 从声明的初始状态开始 | 配置副本调用 reset_state | 从模型初始随机状态开始 |
| `warm_start=True` 后再次 fit | 沿用学习权重 | 配置副本调用 reset_state，动量和历史清空 | 沿用 RNG 流，开始新的训练轮次 |
| `resume_fit(X, y)` | 恢复学习权重 | 保留完整状态 | 保留 RNG、排列和下一批游标 |

初始化优先级为 `initial_point` → 模板中的完整权重绑定 → 模型 Generator 的标准正态初始化。整数 random_state 的独立模型可复现；None 的模型各自获取随机状态，同一模型默认重复 fit 仍从其初始状态开始。传入 Generator 时复制其状态，不消耗调用者的生成器。全局 `np.random.seed` 不再控制模型初始化或 DataLoader 打乱。

`fit(..., max_steps=k)` 在 k 个完整 mini-batch 更新后正常暂停；暂停结果可预测、可保存。`resume_fit(..., max_steps=k)` 可继续若干步；不传 max_steps 时完成保存的目标轮数。`additional_epochs=k` 在保存的目标轮数上增加 k 轮。

```python
from cqlib_qml.algorithms import VQC
from cqlib_qml.ansatz import HEAnsatz
from cqlib_qml.encoder import AngleEncoder

classifier = VQC(HEAnsatz(2, 1, layers=["RY", "CX"]), AngleEncoder(),
                 epochs=4, batch_size=1, random_state=12, verbose=False)
labels = np.array([0, 1])
classifier.fit(X, labels, max_steps=1)
classifier.save_checkpoint("./checkpoints/vqc.npy")
restored = VQC(HEAnsatz(2, 1, layers=["RY", "CX"]), AngleEncoder())
restored.load_checkpoint("./checkpoints/vqc.npy")
restored.resume_fit(X, labels)
trained_weights = restored.ansatz_.weights
```

精确恢复要求同一数据内容和顺序、类别、批次大小、损失及 readouts。恢复数据由调用者重新提供，校验内容指纹。不允许以新数据冒充精确续训；新数据可选择重新 fit 或符合特征和类别约束的 warm start。改变构造配置时使用 `set_params`，配置变化会使原拟合状态失效。

## checkpoint 格式与恢复

恢复边界是**一次完整 mini-batch 更新并 zero_grad 之后**。有尚未清零的梯度时保存会报错；不支持保存一半 backward 或微批次累积中途的梯度。

保存检查同时读取梯度有效标志和参数梯度值；未设置有效标志的旧式自定义 Layer 若仍有非零梯度，也必须先 zero_grad。

初始 checkpoint 格式为 `format_version=1`，独立于包的 beta 版本号。它保存组件结构、角色与顺序、权重、冻结状态、微分方法和 shift、测量及编码器配置、优化器和调度器状态、Generator 状态。VQC 还保存训练目标、更新次数、当前 epoch 排列、下一批位置、类别和数据指纹；训练特征快照用于拟合元数据。运行时输入绑定、量子态、Jacobians 和梯度缓存不保存。加载后必须重新 forward。

Module 继续支持 `save_checkpoint(path, ep, it, latest=False)` 和 `load_checkpoint(path)`；可通过 `data_loader=loader` 保存/恢复项目 DataLoader 状态。没有 loader 状态时只承诺模型状态恢复，外部数据管线需要自行保存顺序和随机性。未结束 epoch 返回 `(ep, it+1)`；保存的 loader 已结束 epoch 时返回 `(ep+1, 0)`。自带 QNN/HQNN 分类循环使用恢复的排列及游标，不重复打乱或跳过数据。

DataLoader 新增 `random_state`、`state_dict()` 和 `load_state_dict()`。加载时验证数据指纹和批次配置；恢复后的第一次 `iter(loader)` 从下一批继续。尚未开始迭代或 epoch 已结束时，第一次 iter 使用恢复的 RNG 创建排列；已开始但尚未取批次时保留原排列。

Module 恢复 DataLoader 时仅替换已验证的排列、游标和随机状态，保留调用者的 Dataset 及底层数据引用，不复制整个数据集。

项目尚未发布，当前不提供未版本化 checkpoint 的兼容路径。加载时要求显式版本和完整的必要字段；缺少版本、版本不受支持或状态不完整时会报错。Module 和 VQC 的文件各自使用对应的加载入口。

所有加载和 VQC fit/resume 都先验证候选状态，成功后才替换当前状态；失败保留原权重、优化器、随机流和进度。checkpoint 写入使用临时文件和原子替换。内置 VQC checkpoint 支持 Angle、Amplitude、ZZFeature 编码器，恢复到结构兼容的 Ansatz 模板。

## 迁移清单

1. 单样本 Linear/Module 的输入梯度从 `(I,)` 改为 `(1, I)`；需要压缩时在调用端显式使用 `[0]`。
2. `ansatz.backward(dY)` 的权重结果改从 `gradients/weight_gradients` 读取；原无参 backward 的 Jacobian 查询改用 `jacobian/input_jacobian/weight_jacobian`。
3. 重复量子 backward 改为累加；需要覆盖效果时先 zero_grad。zero_grad 不再使前向失效，也不再拒绝冻结组件。
4. 关闭记录不会清空已累计梯度。希望丢弃梯度时显式 zero_grad；冻结也会清空该组件权重梯度。
5. `VQC.ansatz` 是构造模板，学习结果使用 `ansatz_`。默认重复 fit 重训，继续权重用 warm_start，保留优化器和随机进度用 resume_fit。
6. 用 random_state 替代依赖全局 np.random.seed 的模型和 loader 初始化。QNN、HQNN、Module 支持 random_state；自定义 Ansatz 子类可通过带种子的 Module.random_init 初始化权重。

自定义 `OptimizerBase` 实例可以继续用于 VQC 的 fit 和 warm start，具体类型和配置保留，调用者实例不修改。训练状态通过 `reset_state()` 重置：基类清空参数缓存、步数及调度器历史；自定义优化器若有额外历史，应覆盖此方法、调用 super 并清空自身历史。自定义有状态调度器也应覆盖 `SchedulerBase.reset_state()`。`reset_step()` 仍只重置步数。内置 checkpoint 的配置重建仅支持内置优化器和调度器；自定义类型需自行提供序列化与恢复路径。
