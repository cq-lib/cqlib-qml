# Torch 量子层

`QuantumLayer` 将 Ansatz 的期望值输出接入原生 `torch.nn.Module`。输入和量子权重均可求导，因此可以用 Torch 构建经典—量子—经典网络，并通过同一个优化器训练。

## 构建联合网络

```python
import torch
from cqlib.circuit import Parameter
from cqlib_qml.ansatz import Ansatz
from cqlib_qml.torch import QuantumLayer

torch.manual_seed(7)
q = Ansatz(1)
q.ry(0, Parameter("x") + Parameter("theta"))
q.rz(0, Parameter("phi"))
q.ry(0, .4)
q.set_parameter_roles(input_params=["x"], weight_params=["theta", "phi"])
q.set_measurement(readouts=[0])
q.set_differentiator("adjoint")  # 或 "parameter_shift"

model = torch.nn.Sequential(
    torch.nn.Linear(2, 1),
    QuantumLayer(q, initial_weights=[.3, -.2]),
    torch.nn.Linear(1, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=.02)
features = torch.tensor([[.1, .2], [.3, .4]])
targets = torch.tensor([[.2], [.5]])
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(model(features), targets)
loss.backward()
optimizer.step()
```

量子输入必须作为符号出现在 Ansatz 中，并通过 `set_parameter_roles()` 声明列顺序。数值编码电路不能保留输入的可微关系；附着了 `add_encoder()` 编码电路的 Ansatz 会被拒绝。

构造接口为 `QuantumLayer(ansatz, initial_weights=None, dtype=None)`。`weight` 是按声明顺序排列的 `nn.Parameter`；`num_inputs`、`num_weights`、`num_outputs` 为只读维度。初始化优先使用显式权重，其次完整的 Ansatz 权重，否则在 `[-π, π]` 上使用 Torch 均匀初始化；部分 Ansatz 权重不会混用。构造后修改原 Ansatz 不影响量子层。

## 输入、梯度与状态

| 项目 | 契约 |
| --- | --- |
| 单样本 | `(inputs,)` → `(outputs,)` |
| batch | `(batch, inputs)` → `(batch, outputs)` |
| 多维 batch | `(..., inputs)` → `(..., outputs)`；保留全部前导维度 |
| 单输出 | 保留最后一维，不自动压成标量 |
| 零输入 | 传 `(0,)` 或 `(..., 0)` 的空特征 Tensor，前导 batch 维必须非空 |
| 零权重 | 注册形状 `(0,)` 的空 Parameter |
| dtype | 输入、权重均为 float32 或 float64，且必须一致 |
| 默认 dtype | 使用 `torch.get_default_dtype()`；可指定 dtype，或调用 `.float()` / `.double()` |
| 设备 | 仅 CPU，不隐式搬移；不支持 autocast |
| 无效输入 | 拒绝标量、空 batch 维、非 Tensor、错误特征数、整数、复数或非有限值 |
| Tensor 视图 | 支持转置、切片、expand 和带 negative bit 的视图 |
| 梯度 | 仅一阶梯度；权重梯度跨全部前导 batch 维求和，不额外平均 |

例如，形状 `(batch, sequence, inputs)` 的序列或 `(batch, height, width, inputs)` 的局部特征会逐样本执行量子电路，输出保留对应的序列或空间维。特征维始终位于最后；图像原始的 `(batch, channels, height, width)` 应先通过经典网络提取特征，或整理局部特征的轴顺序。

通过 `model[1].weight.requires_grad_(False)` 冻结量子权重，输入梯度仍可传回前面的经典层。`eval()` 不禁用自动微分；推理时使用 `torch.no_grad()` 或 `torch.inference_mode()`，跳过 Jacobian 计算。

每次 forward 的执行和 Jacobian 相互独立，同一层可以多次 forward 后合并 backward。累积梯度使用 Torch 的 `zero_grad()` 和优化器。若各微批使用 mean loss，不等大小微批应按 `微批样本数 / 总样本数` 加权，才能与整批梯度一致。计算图使用期间不要原地改变输入或权重。

首版不支持 GPU、AMP、二阶梯度、`torch.compile`、`torch.func` 的自动微分/批量变换（如 `jacrev`、`vmap`）、shots/backend 或概率输出。执行副本和 Jacobian 有额外内存开销；完整两类 float64 Jacobian 约占 `8 × samples × outputs × (inputs + weights)` 字节，其中 samples 为全部前导 batch 维的乘积。多个尚未 backward 的节点分别占用内存，量子模拟状态还随量子比特数指数增长。

## 原生优化器与分类

可以直接使用 SGD、Adam、AdamW 或 LBFGS。LBFGS 会反复执行闭包，每次都需要清空梯度、重新 forward 和 backward；它不要求量子层提供二阶梯度。

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=.5, max_iter=6)

def closure():
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), targets)
    loss.backward()
    return loss

optimizer.step(closure)
```

分类时由经典输出层生成 logits：二分类使用一个输出和 `BCEWithLogitsLoss`，多分类使用类别数个输出和 `CrossEntropyLoss`。现有期望值输出可以用于这些分类网络。也可以串联多个量子层，或在不同分支共享同一量子层；各次执行独立，Torch 负责汇总共享权重的梯度。

## 保存恢复

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "hybrid.pt")
saved = torch.load("hybrid.pt", map_location="cpu", weights_only=True)
# 先用同样电路、参数角色、测量和微分配置重建 model、optimizer。
model.load_state_dict(saved["model"])
optimizer.load_state_dict(saved["optimizer"])
```

量子层保存权重和基础类型的结构元数据。加载前检查参数顺序、电路、测量、微分配置及格式版本；结构不兼容或元数据缺失时，即使 `strict=False` 也会拒绝，并保留该量子层原来的权重。Sequential 中同样生效，但整个父模型的加载不保证事务回滚。

该接口提供权重恢复；优化器状态需单独保存。精确续跑还需要调用者保存随机数状态及数据顺序。原生 QML checkpoint 与 Torch state_dict 使用各自的接口。

## 可运行示例

安装项目后，在仓库根目录运行：

```bash
python examples/torch_hybrid.py --method adjoint
python examples/torch_hybrid.py --method parameter_shift
python examples/torch_hybrid.py --checkpoint /tmp/hybrid.pt
```

示例使用固定种子的合成回归数据、原生 Torch DataLoader 和 Adam，检查所有网络参数的梯度、损失下降，以及权重和优化器恢复后的一次相同更新。默认使用临时 checkpoint；指定 `--checkpoint` 可以保留文件。
