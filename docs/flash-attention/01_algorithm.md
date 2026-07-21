# 01 - 为什么需要 Flash Attention：IO-Awareness 与 Online Softmax

## 先问一个问题

标准 Attention 的 FLOPs 是 $O(N^2 d)$，Flash Attention 的 FLOPs **更多**（有 recomputation），但它在实际测量中 2-4x 更快。这不矛盾吗？

答案是：**现代 GPU 的瓶颈不是算力，是内存带宽**。

---

## GPU 内存层级

```
寄存器（~256KB/SM）     ← 最快，线程私有
共享内存 SRAM（~100KB/SM）← 快，SM 内共享   ~19 TB/s (A100)
L2 Cache（~40MB）
HBM（显存，~80GB）       ← 慢，所有 SM 共享  ~2 TB/s (A100)
```

关键比率：**HBM 带宽 ≈ SRAM 带宽的 1/10**。

一次典型的 Attention 计算需要读写 HBM 多少次？

---

## 标准 Attention 的 HBM 访问分析

标准实现（PyTorch `F.scaled_dot_product_attention` 在没有 FA 后端时）：

```
Step 1: 读 Q, K → HBM reads: O(Nd)
Step 2: 计算 S = QK^T → 写 S 矩阵到 HBM: O(N²)
Step 3: 读 S → 计算 softmax → 写 P 到 HBM: O(N²)
Step 4: 读 P, V → 计算 O = PV → 写 O 到 HBM: O(Nd)
```

**总 HBM 访问量：$\Theta(N^2 + Nd)$**

当 $N = 4096, d = 128$：$S$ 矩阵 = $4096^2 \times 2$ bytes = **32 MB**，每个 attention head，每次 forward。

这就是瓶颈：中间矩阵 $S$ 和 $P$ 太大，装不进 SRAM，只能反复读写 HBM。

---

## Flash Attention 的核心思想

**不要把 $N \times N$ 的 $S$ 矩阵写到 HBM。** 用 tiling，在 SRAM 里完成整个 attention 计算。

但有个数学障碍：**softmax 需要看完整行才能归一化**。

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

分母需要遍历整行。怎么分块计算？

---

## Online Softmax

### 数值稳定的 softmax

先来一个常见技巧：减去行最大值防止溢出：

$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

### 分块递推

假设我们将 $K$ 列分成两块：$[k_1, \ldots, k_B]$ 和 $[k_{B+1}, \ldots, k_N]$。

处理第一块时，我们维护：
- $m_1 = \max_{j \le B} s_j$（当前最大值）  
- $l_1 = \sum_{j \le B} e^{s_j - m_1}$（当前归一化因子）

处理第二块时，设 $m_2 = \max_{j > B} s_j$，新的全局最大值 $m = \max(m_1, m_2)$，则：

$$l = e^{m_1 - m} \cdot l_1 + e^{m_2 - m} \cdot \sum_{j > B} e^{s_j - m_2}$$

同时，**已经累积的输出** $O_1$ 需要重新缩放：

$$O \leftarrow \frac{e^{m_1 - m} \cdot l_1 \cdot O_1 + \text{新块贡献}}{l}$$

这就是 Online Softmax。在源码中对应 `hopper/softmax.h`：

```cpp
// softmax.h - 核心 reduce 操作
template<bool zero_init=true, ...>
__device__ void reduce_max(Tensor const& tensor, Tensor &max) { ... }

template<bool zero_init=true, ...>
__device__ void reduce_sum(Tensor const& tensor, Tensor &sum) { ... }
```

每次处理一个 KV tile，就调用一次 `reduce_max` + `reduce_sum` 更新 $(m, l)$，并对累积的 $O$ 做 rescale。

---

## Flash Attention 算法（伪代码）

```python
# Q: (N, d), K: (N, d), V: (N, d)
# 分 Q 为 Br 行一块，分 K/V 为 Bc 列一块

O = zeros(N, d)
l = zeros(N)      # 归一化因子
m = full(N, -inf) # 行最大值

for j in range(0, N, Bc):           # 遍历 KV 块（outer loop）
    Kj = K[j:j+Bc]                  # 从 HBM 加载到 SRAM
    Vj = V[j:j+Bc]

    for i in range(0, N, Br):       # 遍历 Q 块（inner loop）
        Qi = Q[i:i+Br]              # 从 HBM 加载到 SRAM

        Sij = Qi @ Kj.T * scale     # 在 SRAM 中计算，不写回 HBM
        mij = Sij.max(dim=-1)
        Pij = exp(Sij - mij)
        lij = Pij.sum(dim=-1)

        # Online softmax 更新
        m_new = max(m[i:i+Br], mij)
        l_new = exp(m[i:i+Br] - m_new) * l[i:i+Br] + exp(mij - m_new) * lij

        # 累积输出，带 rescale
        O[i:i+Br] = (exp(m[i:i+Br] - m_new) * l[i:i+Br] * O[i:i+Br]
                     + exp(mij - m_new) * Pij @ Vj) / l_new

        m[i:i+Br] = m_new
        l[i:i+Br] = l_new

# O 就是正确的 softmax(QK^T / scale) @ V，全程没有写过 N×N 矩阵到 HBM
```

> **FA1 vs FA2 的区别**：FA1 外层遍历 Q，内层遍历 KV（不利于并行）。FA2 交换循环顺序，外层遍历 KV，更好地利用 warp 并行，减少 warp 间通信。

---

## IO Complexity 分析

| | HBM 访问量 |
|---|---|
| 标准 Attention | $\Theta(N^2 + Nd)$ |
| Flash Attention | $\Theta(N^2 d / M)$ |

其中 $M$ 是 SRAM 大小。当 $M \gg d$（通常 $M = 100\text{KB}$，$d = 128$ 即 $d \times 2\text{bytes} = 256\text{B}$），FA 的 HBM 访问量远小于标准实现。

**FA 的 HBM 访问量是 information-theoretically optimal 的**（即对于任意正确计算 attention 的算法，在该 SRAM 大小下不可能做得更少）。

---

## Recomputation：用算力换内存

训练时 backward 需要 $P = \text{softmax}(S)$ 矩阵。标准实现会把 $P$ 存在 HBM。

FA 的做法：**backward 时重新计算** $P$（从已保存的 $Q, K$ 重算），而不是存储它。代价是多了一次前向计算，但节省了 $O(N^2)$ 的 HBM 存储，反而因为减少 HBM 读写而更快。

推理时不需要 backward，所以这个 tradeoff 不存在。

---

## 和 mini-vllm 的关系

你的 `layers/attention.py:148-163` 中：

```python
# SDPA 路径 - PyTorch 内部会尝试使用 FA backend
o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`F.scaled_dot_product_attention` 在有 flash-attn 安装的环境下会自动使用 FA kernel。但它无法利用 KV cache（每次需要完整的 K/V），所以 decode 时你显式调用了 `flash_attn_with_kvcache`。

理解了 FA 的算法，就能理解为什么 `flash_attn_with_kvcache` 的接口长这样——下一节详细讲。
