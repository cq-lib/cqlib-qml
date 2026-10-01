# differentiator 模块教程

`differentiator` 模块提供了参数化量子线路的梯度计算方法，是训练量子神经网络的核心组件。该模块支持两种梯度计算策略：伴随法和参数偏移法。

## 模块结构

```
differentiator/
├── adjoint.py          # 伴随法梯度计算
└── parameter_shift.py  # 参数偏移法梯度计算
```

---

## 背景与数学原理

### 为什么需要梯度计算？

在量子机器学习中，我们需要优化参数化量子线路的参数 $\boldsymbol{\theta}$，以最小化损失函数 $L(\boldsymbol{\theta})$。这需要计算损失函数对每个参数的梯度：

$$\frac{\partial L}{\partial \theta_i}$$

梯度计算是反向传播和参数更新的基础，直接影响模型的训练效率和收敛性。

### 两种梯度计算方法的对比

| 特性 | 伴随法 | 参数偏移法 |
|------|--------|------------|
| 时间复杂度 | $O(p)$ | $O(2p)$ |
| 适用场景 | 经典模拟器 | 量子硬件 |
| 精度 | 高（解析解） | 高（解析解） |
| 实现复杂度 | 中等 | 简单 |
| 内存需求 | 需要存储状态向量 | 多次执行电路 |

---

## 伴随法 (Adjoint Method)

### 数学原理

伴随法通过反向传播计算梯度，其核心思想是：将量子线路看作一系列酉变换的复合，梯度可以通过反向传播误差信号来计算。

对于线路 $U(\boldsymbol{\theta}) = U_L(\theta_L) \cdots U_2(\theta_2) U_1(\theta_1)$，参数 $\theta_i$ 的梯度为：

$$\frac{\partial f}{\partial \theta_i} = \langle \psi_i | \frac{\partial U_i}{\partial \theta_i} | \phi_i \rangle$$

其中：
- $|\psi_i\rangle = U_{i-1} \cdots U_1 |0\rangle$ 是前向传播到第 $i$ 层的状态
- $|\phi_i\rangle = U_{i+1}^\dagger \cdots U_L^\dagger H |\psi_L\rangle$ 是反向传播到第 $i$ 层的状态

伴随法只需要一次前向传播和一次反向传播即可计算所有参数的梯度，时间复杂度为 $O(p)$，其中 $p$ 是参数数量。

### 使用示例

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from cqlib_qml.differentiator import AdjointDifferentiator

# 1. 创建参数化量子电路
circuit = Circuit(1)
theta = Parameter("theta")
circuit.ry(0, theta)

# 2. 定义观测哈密顿量
ham = Hamiltonian(1)
ham.add_term(PauliString.from_str("Z"), 1.0)

# 3. 参数绑定
bindings = {"theta": 0.5}

# 4. 前向传播：获取状态向量
assigned_circuit = circuit.assign_parameters(bindings)
state = Statevector(1)
state.apply_circuit(assigned_circuit)
state_vector = state.data

# 5. 计算梯度
diff = AdjointDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings=bindings,
    state_vector=state_vector,  # 前向传播后的状态
    hamiltonians=[ham]
)

print(f"梯度: {grads}")
```

**输出：**

```
梯度: {'theta': array([-0.47942554])}
```

### 使用 readouts 替代哈密顿量

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis.state import Statevector
from cqlib_qml.differentiator import AdjointDifferentiator

# 1. 创建参数化量子电路
circuit = Circuit(2)
theta = Parameter("theta")
circuit.ry(0, theta)
circuit.cx(0, 1)

# 2. 参数绑定
bindings = {"theta": 0.5}

# 3. 前向传播：获取状态向量
assigned_circuit = circuit.assign_parameters(bindings)
state = Statevector(2)
state.apply_circuit(assigned_circuit)
state_vector = state.data

# 4. 使用 readouts 计算梯度（测量 Pauli-Z）
diff = AdjointDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings=bindings,
    state_vector=state_vector,
    readouts=[0]
)

print(f"梯度: {grads}")
```

**输出：**

```
梯度: {'theta': array([-0.47942554])}
```

### 多参数梯度计算

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from cqlib_qml.differentiator import AdjointDifferentiator

# 1. 创建多参数电路
circuit = Circuit(2)
theta1 = Parameter("theta1")
theta2 = Parameter("theta2")
circuit.ry(0, theta1)
circuit.ry(1, theta2)

# 2. 定义观测哈密顿量
ham = Hamiltonian(2)
ham.add_term(PauliString.from_str("ZI"), 1.0)
ham.add_term(PauliString.from_str("IZ"), 1.0)

# 3. 参数绑定
bindings = {"theta1": 0.5, "theta2": 0.3}

# 4. 前向传播：获取状态向量（关键步骤！）
assigned_circuit = circuit.assign_parameters(bindings)
state = Statevector(2)
state.apply_circuit(assigned_circuit)
state_vector = state.data

# 5. 伴随法计算梯度
diff = AdjointDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings=bindings,
    state_vector=state_vector,  # 前向传播后的状态
    hamiltonians=[ham]
)

print(f"所有参数梯度: {grads}")
```

**输出：**

```
所有参数梯度: {'theta1': array([-0.47942554]), 'theta2': array([-0.29552021])}
```

---

## 参数偏移法 (Parameter Shift Rule)

### 数学原理

参数偏移法利用参数偏移规则计算梯度。对于形式为 $U(\theta) = e^{-i\theta G/2}$ 的门（其中 $G^2 = I$），梯度为：

$$\frac{\partial f}{\partial \theta} = \frac{f(\theta + s) - f(\theta - s)}{2\sin(s)}$$

其中 $s$ 是偏移量，通常取 $s = \pi/2$，此时公式简化为：

$$\frac{\partial f}{\partial \theta} = \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$

当前实现逐个参数化门的参数位置计算贡献，再按链式法则累加到符号梯度。简单门分支每个位置需要两次线路评估，通用 Fourier 分支需要八次；同一符号出现在多个门中时，各位置分别计算。因此，若分解后的线路包含 $m_2$ 个简单分支位置和 $m_8$ 个通用分支位置，每个样本需要 $2m_2 + 8m_8$ 次梯度线路评估，而不能仅按独立符号数量估算。这里不包含硬件测量分组、shots 或前向计算成本。

### 使用示例

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib_qml.differentiator import ParameterShiftDifferentiator

# 1. 创建参数化量子电路
circuit = Circuit(1)
theta = Parameter("theta")
circuit.ry(0, theta)

# 2. 定义观测哈密顿量
ham = Hamiltonian(1)
ham.add_term(PauliString.from_str("Z"), 1.0)

# 3. 计算梯度（默认偏移量 π/2）
diff = ParameterShiftDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings={"theta": 0.5},
    hamiltonians=[ham]
)

print(f"梯度 (shift=π/2): {grads}")
```

**输出：**

```
梯度 (shift=π/2): {'theta': array([-0.47942554])}
```

### 自定义偏移量

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib_qml.differentiator import ParameterShiftDifferentiator

# 创建电路
circuit = Circuit(1)
theta = Parameter("theta")
circuit.ry(0, theta)

# 定义哈密顿量
ham = Hamiltonian(1)
ham.add_term(PauliString.from_str("Z"), 1.0)

# 使用自定义偏移量 π/4
diff = ParameterShiftDifferentiator(shift=np.pi / 4)
grads = diff.run(
    circuit=circuit,
    bindings={"theta": 0.5},
    hamiltonians=[ham]
)

print(f"梯度 (shift=π/4): {grads}")
```

**输出：**

```
梯度 (shift=π/4): {'theta': array([-0.47942554])}
```

> **注意**：理论上，只要 $\sin(s) \neq 0$，任何偏移量都能给出正确的梯度。但 $\pi/2$ 是最常用的选择，因为公式简化且数值稳定性最好。

### 使用 readouts 替代哈密顿量

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib_qml.differentiator import ParameterShiftDifferentiator

# 创建电路
circuit = Circuit(2)
theta = Parameter("theta")
circuit.ry(0, theta)
circuit.cx(0, 1)

# 使用 readouts 计算梯度
diff = ParameterShiftDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings={"theta": 0.5},
    readouts=[0]
)

print(f"梯度: {grads}")
```

**输出：**

```
梯度: {'theta': array([-0.47942554])}
```

### 多参数梯度计算

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib_qml.differentiator import ParameterShiftDifferentiator

# 1. 创建多参数电路
circuit = Circuit(2)
theta1 = Parameter("theta1")
theta2 = Parameter("theta2")
circuit.ry(0, theta1)
circuit.ry(1, theta2)

# 2. 定义观测哈密顿量
ham = Hamiltonian(2)
ham.add_term(PauliString.from_str("ZI"), 1.0)
ham.add_term(PauliString.from_str("IZ"), 1.0)

# 3. 计算所有参数梯度
diff = ParameterShiftDifferentiator()
grads = diff.run(
    circuit=circuit,
    bindings={"theta1": 0.5, "theta2": 0.3},
    hamiltonians=[ham]
)

print(f"所有参数梯度: {grads}")
```

**输出：**

```
所有参数梯度: {'theta1': array([-0.47942554]), 'theta2': array([-0.29552021])}
```

---

## 两种方法的对比实验

### 数值精度对比

```python
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator

# 创建测试电路
circuit = Circuit(2)
theta1 = Parameter("theta1")
theta2 = Parameter("theta2")
circuit.ry(0, theta1)
circuit.ry(1, theta2)

# 定义观测哈密顿量
ham = Hamiltonian(2)
ham.add_term(PauliString.from_str("ZI"), 1.0)
ham.add_term(PauliString.from_str("IZ"), 1.0)

bindings = {"theta1": 0.5, "theta2": 0.3}

# 伴随法
assigned_circuit = circuit.assign_parameters(bindings)
state = Statevector(2)
state.apply_circuit(assigned_circuit)

adj = AdjointDifferentiator()
grad_adj = adj.run(
    circuit=circuit,
    bindings=bindings,
    state_vector=state.data,
    hamiltonians=[ham]
)

# 参数偏移法
ps = ParameterShiftDifferentiator()
grad_ps = ps.run(
    circuit=circuit,
    bindings=bindings,
    hamiltonians=[ham]
)

print(f"伴随法:   {grad_adj}")
print(f"参数偏移法: {grad_ps}")
print(f"一致: {np.allclose(grad_adj['theta1'], grad_ps['theta1'])}")
```

**输出：**

```
伴随法:   {'theta1': array([-0.47942554]), 'theta2': array([-0.29552021])}
参数偏移法: {'theta1': array([-0.47942554]), 'theta2': array([-0.29552021])}
一致: True
```

两种方法给出数值上完全一致的结果。

### 性能对比

```python
import time
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.qis import Hamiltonian, PauliString
from cqlib.qis.state import Statevector
from cqlib_qml.differentiator import AdjointDifferentiator, ParameterShiftDifferentiator

# 创建包含多个参数的电路
n_params = 10
circuit = Circuit(4)
params = []
for i in range(n_params):
    theta = Parameter(f"theta_{i}")
    params.append(theta)
    circuit.ry(i % 4, theta)

# 定义观测哈密顿量
ham = Hamiltonian(4)
ham.add_term(PauliString.from_str("ZZII"), 1.0)

bindings = {f"theta_{i}": 0.1 * i for i in range(n_params)}

# 伴随法计时
start = time.time()
assigned_circuit = circuit.assign_parameters(bindings)
state = Statevector(4)
state.apply_circuit(assigned_circuit)

adj = AdjointDifferentiator()
grad_adj = adj.run(
    circuit=circuit,
    bindings=bindings,
    state_vector=state.data,
    hamiltonians=[ham]
)
time_adjoint = time.time() - start

# 参数偏移法计时
start = time.time()
ps = ParameterShiftDifferentiator()
grad_ps = ps.run(
    circuit=circuit,
    bindings=bindings,
    hamiltonians=[ham]
)
time_ps = time.time() - start

print(f"伴随法时间:   {time_adjoint:.6f}s")
print(f"参数偏移法时间: {time_ps:.6f}s")
print(f"速度比:       {time_ps / time_adjoint:.2f}x")
```

**输出：**

```
伴随法时间:   0.001933s
参数偏移法时间: 0.009835s
速度比:       5.09x
```

伴随法显著快于参数偏移法，尤其是在参数数量较多时。

---

## 在 Ansatz 中使用不同微分器

`Ansatz` 类通过 `set_differentiator()` 方法支持切换梯度计算方法。

```python
import numpy as np
from cqlib_qml.ansatz import HEAnsatz

np.random.seed(0)

# 创建线路
ansatz = HEAnsatz(n_qubits=2, d=1, layers=["RY", "CX"])
ansatz.set_measurement(readouts=[0])

# 使用伴随法
ansatz.set_differentiator("adjoint")
result = ansatz.forward()
ansatz.backward()
print(f"伴随法梯度: {ansatz.gradients}")

# 重置ansatz梯度
ansatz.zero_grad()

# 使用参数偏移法
ansatz.set_differentiator("parameter_shift", shift=np.pi/2)
result = ansatz.forward()
ansatz.backward()
print(f"参数偏移法梯度: {ansatz.gradients}")
```

**输出：**

```
伴随法梯度: {'params0_0': array([-0.9813841]), 'params0_1': array([-1.38777878e-17])}
参数偏移法梯度: {'params0_0': array([-0.9813841]), 'params0_1': array([5.55111512e-17])}
```

---

## 不同微分器选择指南

| 场景 | 推荐不同微分器 | 原因 |
|------|---------------|------|
| 经典模拟器训练 | 伴随法 | 速度快，单次反向传播计算所有梯度 |
| 量子硬件执行 | 参数偏移法 | 无需存储状态向量，适合硬件 |
| 参数数量多（>50） | 伴随法 | O(p) vs O(2p) 差异显著 |
| 需要梯度稳定性 | 参数偏移法 | 数值稳定性好，不受状态向量精度影响 |
| 调试/验证 | 参数偏移法 | 实现简单，易于理解 |

---

## 常见问题排查

### 问题 1: 伴随法报错 "The input circuit must be a parameterized quantum circuit"

**原因**：电路中不包含任何参数。

**解决方案**：

```python
# 确保电路中包含至少一个 Parameter
theta = Parameter("theta")
circuit.ry(0, theta)  # 添加参数化门
```

### 问题 2: 参数偏移法报错 "shift = X rad is not allowed"

**原因**：偏移量 $s$ 使 $\sin(s) = 0$（即 $s$ 是 $\pi$ 的整数倍）。

**解决方案**：

```python
# 使用有效的偏移量
diff = ParameterShiftDifferentiator(shift=np.pi / 2)   # 正确
diff = ParameterShiftDifferentiator(shift=np.pi / 4)   # 正确
diff = ParameterShiftDifferentiator(shift=np.pi)       # 错误
diff = ParameterShiftDifferentiator(shift=0)           # 错误
```

### 问题 3: 梯度值为 0

**可能原因**：
- 参数处于梯度消失区域（如 $\theta = 0$ 或 $\theta = \pi$）
- 测量算符与线路不对易

**解决方案**：

```python
# 尝试不同的初始参数
bindings = {"theta": 0.3}  # 避开 0, π 等特殊点
```

---

## API 参考

### AdjointDifferentiator

| 方法 | 描述 |
|------|------|
| `run(circuit, bindings, state_vector, readouts=None, hamiltonians=None)` | 计算参数梯度 |

**参数说明**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `circuit` | Circuit | 参数化量子电路 |
| `bindings` | dict | 参数绑定 |
| `state_vector` | np.ndarray | 前向传播状态向量（必须提供） |
| `readouts` | List[int] | 测量量子比特（与 hamiltonians 二选一） |
| `hamiltonians` | List[Hamiltonian] | 观测哈密顿量（与 readouts 二选一） |

### ParameterShiftDifferentiator

| 方法 | 描述 |
|------|------|
| `run(circuit, bindings, readouts=None, hamiltonians=None, initial_state=None)` | 计算参数梯度 |

**参数说明**：

| 参数 | 类型 | 描述 |
|------|------|------|
| `circuit` | Circuit | 参数化量子电路 |
| `bindings` | dict | 参数绑定 |
| `readouts` | List[int] | 测量量子比特（与 hamiltonians 二选一） |
| `hamiltonians` | List[Hamiltonian] | 观测哈密顿量（与 readouts 二选一） |