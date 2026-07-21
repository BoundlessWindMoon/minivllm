# 从 bs=1 到 Continuous Batching 适配

<!-- code-ref: engine/scheduler.py -->
<!-- code-ref: engine/batched_runner.py -->
<!-- code-ref: engine/request.py -->

## 摘要

单请求推理在 decode 阶段的算术强度（Arithmetic Intensity）为 1，decode 是极度 memory-bound 的负载。
将 bs=1 扩展为 bs>1 可以提升计算强度，在不明显增加单轮 forward 延迟的情况下显著提升吞吐量。

bs>1 拓展需要解决三个问题：

- KV cache 的管理方式与多序列并发不兼容
- Batch 内各序列 cache_len 不同
- 变长 prompt 与异构 decode 深度要求 attention mask 按序列定制

本文给出这三个问题的系统抽象，并展示如何将它们组合为一个完整的 Continuous Batching 系统。

---

## bs=1 推理的硬件瓶颈

### 算术强度与 Roofline 模型

GPU 的性能上界由峰值浮点算力 $P_{\text{compute}}$ 和 HBM 带宽 $P_{\text{mem}}$ 决定。对于任意计算任务，其算术强度定义为：

$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

Roofline 模型给出性能上界：

$$\text{Throughput} \leq \min\!\left(P_{\text{compute}},\ I \cdot P_{\text{mem}}\right)$$

当 $I < P_{\text{compute}} / P_{\text{mem}}$ 时，任务为 memory-bound；反之为 compute-bound。

> 对于 A100，fp16 峰值算力 $P_{\text{compute}} = 312\ \text{TFLOPS}$，HBM 带宽 $P_{\text{mem}} = 2\ \text{TB/s}$，临界算术强度为 $312 / 2 = 156\ \text{FLOPs/Byte}$。

### Decode 阶段的算术强度

以 Qwen3-0.6B（$d=1024,\ n_h=16,\ n_{kv}=8,\ d_h=128,\ d_{\text{ffn}}=3072$）为例，对单层 decode 中三个关键操作逐一分析。

**① QKV Projection**

```python
# h:     (B, 1, d)            — 输入，1 个新 token
# W_qkv: (d, n_h·d_h + 2·n_kv·d_h)  — 权重，大小与 B 无关
qkv = h @ W_qkv.T             # (B, 1, n_h·d_h + 2·n_kv·d_h)
q = q.view(B, 1, n_h,  d_h)   # (B, 1, 16, 128)
k = k.view(B, 1, n_kv, d_h)   # (B, 1,  8, 128)
v = v.view(B, 1, n_kv, d_h)   # (B, 1,  8, 128)
```

权重字节数与 $B$ 无关，FLOPs 正比于 $B$，故：

$$I = \frac{B \cdot d \cdot (n_h + 2n_{kv}) d_h}{d \cdot (n_h + 2n_{kv}) d_h} = B$$

**③ Attention SDPA（GQA ratio = $n_h / n_{kv} = 2$）**

```python
# K, V 为从 KV cache 读取的历史 KV，大小随序列数和长度同步增长
# K: (B, n_kv, T, d_h)  — 从 HBM 读，大小 ∝ B·T
# V: (B, n_kv, T, d_h)  — 从 HBM 读，大小 ∝ B·T
q   = q.transpose(1, 2)              # (B, n_h,  1, d_h)  — query，1 个 token
K_e = K.repeat_interleave(2, dim=1)  # (B, n_h,  T, d_h)
V_e = V.repeat_interleave(2, dim=1)  # (B, n_h,  T, d_h)
out = softmax(q @ K_e.T / d_h**0.5) @ V_e   # (B, n_h,  1, d_h)
```

FLOPs 与读 KV 的字节数均含因子 $B \cdot T$，两者相除时 $B$ 被约掉：

$$I = \frac{B \cdot n_h \cdot d_h \cdot T}{B \cdot n_{kv} \cdot d_h \cdot T} = \frac{n_h}{n_{kv}} = 2$$

$B$ 被约掉，算术强度等于 GQA head 复用比，**与 batch size 无关**。

**④ FFN（SwiGLU）**

```python
# W_gate, W_up:  (d, d_ffn)   — 权重大小与 B 无关
# W_down:        (d_ffn, d)
gate  = silu(h @ W_gate.T)    # (B, 1, d_ffn)
up    = h @ W_up.T            # (B, 1, d_ffn)
x = x + (gate * up) @ W_down.T  # (B, 1, d)
```

结构同 ①，$I = B$。

**汇总：**

| 操作 | 内存访问主体 | 算术强度 | 增大 $B$ 是否改善 |
|------|-----------|---------|---------------|
| ① QKV / O-proj / ④ FFN | 权重 $\mathbf{W}$（$B$ 条序列共享同一份） | $I = B$ | ✓ |
| ③ Attention SDPA | KV cache（每条序列各自独立） | $I = n_h / n_{kv}$ | ✗ |

---

## 两种批处理模式

在进入实现细节之前，先明确本文要解决的目标。

**静态批处理**

将多条请求打包成固定大小的 batch， batch内所有序列同时开始 prefill，然后一起进入 decode，直到 batch 内最长序列生成。

```
batch = [req_A (100 tok), req_B (300 tok), req_C (50 tok)]

step 0:       prefill all three together
step 1–50:    decode all three (req_C done at step 50 but must wait)
step 51–100:  decode A + B only (req_C slot 空置)
step 101–300: decode B only (A, C slots 空置)
```

**连续批处理（Continuous Batching）**

每完成一条序列就立即释放其 slot，并在同一 step 内接入等待队列的下一条请求。

```
batch = [req_A (100 tok), req_B (300 tok), req_C (50 tok)]

step 0:       prefill all three together 
step 1–50:    decode all three (req_C done at step 50 but must wait)
step 50:  req_C 完成 → 立即释放 slot → req_D 进入 prefill
step 51:  A + B decode，D decode（第一个 token 已在 step 50 产出）
```

连续批处理减少了 slot 的空置时间，新请求不必等待当前 batch 中最长的序列跑完才能被调度。

---

## 从 bs=1 到 Continuous Batching 的三个障碍

相较于静态批处理，连续批处理引入了新约束：

- 各序列 KV Cache 长度不同
- KV slot 需要在运行时分配和回收

### 障碍一：KV cache 的管理方式

bs=1 时，KV cache 可以挂在每一层 DecoderLayer 上：

```python
# bs=1 kvcache 
self.k_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)
self.v_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)
```

> 现在要支持多序列并发，问题来了：**序列 A 占用的 KV Cache 何时可以被清零?**

Scheduler 提供 "何时分配，何时回收" 的信息, 但 KV Cache 内存分散在 28 个 Attention 模块里。

自然的做法是把所有层的 KV 存储集中到一个资源池，每条序列持有 slot id, Scheduler 管理 slot 的生命周期，Attention 层通过 slot id 读写数据。

### 障碍二：Batch 内各序列 KV Cache 长度不同

只要存在空闲 slot，continuous batching 允许序列在任意时刻入队，这导致每个序列的 KV Cache 长度不同。

bs=1 时，每个 decode step 对应一个全局标量 `cache_len`, 表示当前 step 下有效的 KV Cache 长度。

Attention KV读写都依赖 `cache_len`：

```python
# 写：
self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k

# 读：
total_len = cache_len + seq_len
k_for_attn = self.k_cache[:, :, :total_len, :]
```

扩展到 bs> 1后，不同时刻入队的序列各自处于不同的 decode 深度。设某 step 时 4 条序列的状态为：

$$[c_1, c_2, c_3, c_4] = [32, 28, 45, 19]$$

此时 `cache_len` 这个标量无法描述这个状态。

### 障碍三：Batch 内各序列 Attention Mask 不同

bs=1 时，decode 阶段的 attention `is_causal=False`（历史 KV 全部可见）：

```python
# bs=1 decode
o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

prefill 阶段用 `is_causal=True` 生成标准下三角 mask，对整个 batch 均匀施加。

```python
# bs=1 prefill
o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 障碍三：Batch 内各序列 Attention Mask 不同

bs=1 时，prefill 用 `is_causal=True` 生成标准下三角 mask，decode 用 `is_causal=False` 让全部历史 KV 可见：

```python
# bs=1 prefill
o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# bs=1 decode
o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

扩展到 bs> 1后， batch 内每条序列需要各自独立的 mask。两种情况：

**Prefill with padding**：不同长度的 prompt pad 到同一长度，padding 位置不应参与 attention。

```
                  is_causal=True        seq B 应有的 mask
seq B (len=3)      Q\K 0 1 2 3 4              Q\K 0 1 2 3 4
padded to 5        0 [ 1 . . . . ]            0 [ 1 . . . . ]
                   1 [ 1 1 . . . ]            1 [ 1 1 . . . ]
                   2 [ 1 1 1 . . ]            2 [ 1 1 1 . . ]
                   3 [ 1 1 1 1 . ]            3 [ 0 0 0 0 0 ] ← PAD
                   4 [ 1 1 1 1 1 ]            4 [ 0 0 0 0 0 ] ← PAD
```

**Heterogeneous decode**：各序列 cache_len 不同，`is_causal=False` 让所有序列看到全部 key，但不同位置的 KV 属于不同序列。

```
KV cache: [ B_k0   B_k1   B_k2   A_k3   A_k4 ]
            ←── seq B 历史 ──→   ←─ seq A ─→

is_causal=False 时：
  seq A query 看到全部 5 个 key  ✓
  seq B query 看到全部 5 个 key  ✗  A_k3、A_k4 是 seq A 的数据
```

`is_causal` 字段无法满足 batch 推理需求，需要为每条序列单独构造 mask。

---

## 解决方案

### KV cache 集中管理：KVCachePool 与 Slot

将所有层的 KV 存储集中到一个 `KVCachePool`：

$$\text{k\_caches}[l] \in \mathbb{R}^{N_{\text{slots}} \times n_{\text{kv}} \times L_{\max} \times d_{\text{head}}}$$

每条并发序列占用一个整数 **slot id**。有了 slot id，模型层的 KV 读写只需按 slot 索引：

```python
# prefill：将当前序列的 K 写入其 slot
pool.k_caches[layer][slot_id, :, start:end, :] = k

# decode：一次 gather 取出 batch 内所有序列的 KV 历史
# slot_ids: (batch,)
k_full = pool.k_caches[layer][slot_ids]   # (batch, n_kv, L_max, d_h)
```

slot 的分配和回收由调度层负责，下一节展开。

### 构造 per-sequence cache_lens

bs=1 时只有一个标量 `cache_len`，扩展到 bs> 1后改为逐序列的向量 `cache_lens: (batch,)`。

写 KVCache：

```python
# bs=1
self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k

# bs>1
for i in range(batch):
    start = cache_lens[i]
    pool.k_caches[layer][slot_ids[i], :, start:start+chunk, :] = k[i]
```

读 KVCache 时，因为 batch 内各序列被pad 到 `max(cache_lens) + 1`，所以需配合 `attn_mask` 屏蔽各序列超出自身历史的部分：

```python
# bs=1
total_len = cache_len + seq_len
k_for_attn = self.k_cache[:, :, :total_len, :]

# bs>1
max_kv = cache_lens.max().item() + 1
k_for_attn = pool.k_caches[layer][slot_ids][:, :, :max_kv, :]
```

### 构造 per-sequence attn_mask

`F.scaled_dot_product_attention` 接受一个可选的 `attn_mask` 参数，shape 为 `(batch, heads, seq_q, seq_k)`：

```python
# attention score 计算（简化）
scores = q @ k.T / sqrt(d_h)   # (B, n_h, seq_q, seq_k)

# 逐元素加：0 不影响，-inf 使对应位置 softmax 后为 0
scores = scores + attn_mask     
out    = softmax(scores) @ v
```
bs>1 时需要构造 attn_mask，对 batch 内每条序列分别设置其可见范围。

两个场景下每条序列的 mask 应该长什么样：
（0 = additive 0，- = $-\infty$）
```
Prefill（seq A len=5，seq B len=3 padded to 5）
attn_mask[A]              attn_mask[B]
  k: 0 1 2 3 4              k: 0 1 2 3 4
q0[ 0 . . . . ]           q0[ 0 . . . . ]
q1[ 0 0 . . . ]           q1[ 0 0 . . . ]
q2[ 0 0 0 . . ]           q2[ 0 0 0 . . ]
q3[ 0 0 0 0 . ]           q3[ - - - - - ] ← PAD 行，全屏蔽
q4[ 0 0 0 0 0 ]           q4[ - - - - - ] ← PAD 行，全屏蔽

Decode（cache_lens=[5, 3]，max_kv=6）
attn_mask[A]              attn_mask[B]
  k: 0 1 2 3 4 5            k: 0 1 2 3 4 5
q0[ 0 0 0 0 0 - ]         q0[ 0 0 0 - - - ] ← seq B 只看 [0,3)
```

mask tensor shape 为 $(B,\, 1,\, S_q,\, S_k)$，四个维度依次为：batch index $i$、head（广播到所有 head）、当前 forward pass 中 query 的下标 $q_{\text{rel}}$、KV 序列的绝对下标 $k_{\text{abs}}$。

设 $\text{offset}_i$ 为第 $i$ 条序列在本次 forward pass 之前已缓存的 token 数（prefill 时为 0，decode 时为完整 prompt 长度），则第 $q_{\text{rel}}$ 个 query token 的绝对位置为 $\text{offset}_i + q_{\text{rel}}$，它只能 attend 到位置不超过自身的 key：

$$\text{attn\_mask}[i,\, 0,\, q_{\text{rel}},\, k_{\text{abs}}] = \begin{cases} 0 & k_{\text{abs}} \leq \text{offset}_i + q_{\text{rel}} \\ -\infty & \text{otherwise} \end{cases}$$


---

## Continuous Batching 的系统组合
前面分别解决了三个局部问题：
- KV Cache 需要中心化管理
- cache_len 需要按序列管理
- attention mask 需要按序列构造

现在需要把它们组合成一个完整的运行时系统。

### 抽象层次
![](../assets/images/scheduler.svg)
三个组件各司其职，通过明确的接口交互：

- **Scheduler**：维护两个队列 `_waiting` 和 `_running`
- **BatchedModelRunner**：每个 step 从 Scheduler 取到本步要处理的请求，构造 tensor、调 forward、触发采样
- **KVCachePool**：负责 slot 的分配和回收

![系统架构](../assets/images/scheduler.png)

### Request 的生存周期

每个请求在系统内经历四个状态：

$$\text{WAITING} \xrightarrow{\text{slot 分配}} \text{PREFILLING} \xrightarrow{\text{prefill 完成}} \text{DECODING} \xrightarrow{\text{EOS or max\_len}} \text{FINISHED}$$

队列变化对应状态转换：

```python
# 外部调用：req → _waiting
scheduler.add_request(req)      

# request 被选中时：从 _waiting 提升到 _running
req.slot_id = pool.allocate(req.request_id)
req.status  = RequestStatus.PREFILLING
_waiting.remove(req);  _running.append(req)

# prefill 完成后：仍在 _running
req.status = RequestStatus.DECODING   

# 遇 eos or max_length
pool.free(req.slot_id)                # 清零 KV，归还 slot
_running.remove(req)                  # req → FINISHED，离开系统
```

### 单个 Step 的执行流程

每个 step 由 `BatchedModelRunner.step()` 驱动，核心逻辑如下：

```python
def step(self):
    # 调度：从队列中取出本步要处理的请求
    prefill_reqs, decode_reqs = self.scheduler.schedule()

    # Decode 优先：先跑已在运行的序列，避免被新 prefill 抢占
    if decode_reqs:
        decode_logits = self._run_decode(decode_reqs)

    # Prefill：处理新进入或尚未完成 prefill 的序列
    if prefill_reqs:
        completed, logits = self._run_prefill(prefill_reqs)

    # 采样 + 终止判断
    self._sample_and_update(...)
```

---

## 实验验证

**模型**：Qwen3-0.6B（28 层）  **硬件**：RTX 4050

### 实验一：Batch Size 对吞吐的影响

同一 prompt 重复 N 次，验证 $I \approx B$ 的预测：bs 翻倍，吞吐应接近翻倍。

| bs | tok/s | vs bs=1 |
|----|-------|---------|
| 1  | 12.7  | 1.00×   |
| 2  | 23.1  | 1.82×   |
| 4  | 39.1  | 3.09×   |
| 8  | 62.0  | **4.90×** |

结果与理论一致：bs 每翻倍，吞吐接近翻倍，但随 bs 增大增益略有收敛。

![Batch Size Scaling](../assets/images/fig1_bs_scaling.png)

### 实验二：Continuous Batching vs Static Batching

**Workload**：9 条请求，短输入（~20 tok），输出长度差异大：

| 请求 | 实际生成 tok | E2E (s) |
|------|-----------|--------|
| What is the capital of Japan? | 12 | 1.1 |
| What is 17 multiplied by 13? | 54 | 5.5 |
| Who invented the telephone? | 56 | 5.8 |
| Summarize the main causes of WWI… | 87 | 10.2 |
| Write a Python function… | 118 | 12.4 |
| What are the key differences between list and tuple… | 431 | 49.0 |
| Explain the CAP theorem… | 425 | 52.8 |
| Explain how gradient descent works… | 497 | 55.3 |
| Explain transformer attention mechanism… | 512 | 61.1 |



| 模式 | tok/s | vs Static |
|------|-------|---------|
| Static bs=4 | 22.7 | 1× |
| Continuous Batching bs≤4 | 35.9 | **1.58×** |

Static batching 中，前 5 条请求（12-118 tok）在 12s 内完成，但其 slot 被锁定到本批最慢请求（512 tok，61s）结束后才释放。Continuous Batching 中，短请求完成即释放 slot，长短请求在时间上交叠执行，实现 1.58× 吞吐提升。

![CB vs Static](../assets/images/fig2_cb_vs_static.png)



---

## 小结

从 bs=1 到 Continuous Batching 需要在三个层面完成设计：

| 问题 | 原有设计 | 新设计 |
|------|---------|-------|
| KV Cache 归属 | 每层 Attention 持有 `(1, nkv, L, d)` | 中心化 KVCachePool，slot 抽象 |
| Cache 深度追踪 | 全局标量 `cache_len` | per-sequence `cache_lens` tensor |
| Attention Mask | `is_causal=True/False` | per-sequence `attn_mask` `(B,1,q,k)` |

Continuous Batching 在输入输出长度分布不均匀场景可实现吞吐量增长（本实验增长 1.58x）。

---

Sources:
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/anatomy-of-vllm)
- [Increasing GPU Utilization during Generative Inference for Higher Throughput](https://ar5iv.labs.arxiv.org/html/2306.06000)
- [连续批处理（Continuous Batching）与迭代级调度](https://www.cnblogs.com/SCCQ/p/19964639)
