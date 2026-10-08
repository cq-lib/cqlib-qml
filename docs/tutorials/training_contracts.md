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

单样本始终保留 batch 轴。`backward(None)` 对输出总和求导；查询 Jacobian 使用上表属性。伴随法和参数偏移法遵守相同契约。

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
