# Chunked Prefill：把 Prefill 切碎

<!-- code-ref: engine/scheduler.py -->
<!-- code-ref: engine/batched_runner.py -->

## 现象与结论

长请求的 prefill 会阻塞并发的 decode 请求，在流式输出上产生明显卡顿。Chunked prefill 把 prefill 切成小块，与 decode 交替执行，把尖刺换成均匀的小延迟。

4 条短请求正在 decode，一条 2048 token 的长请求到来：

![prefill stall 实测](../tmp/fig_prefill_stall.png)

FIFO 左图：短请求的 token 生成间隔出现了 **1952 ms 的尖刺**，是正常 TPOT 的 16.5 倍。

Chunked 中图：最大 gap 压到 1150 ms，P50 TPOT 从 118 ms 升至 138 ms。

本文解释成因、建立定量模型、推导理论极限，落到实现。

---

## 成因

### GPU 串行执行

mini-vllm 每个 step 的执行顺序固定：先 decode，再 prefill \[3\]。

```python
# engine/batched_runner.py
def step(self):
    prefill_chunks, decode_reqs = self.scheduler.schedule()
    if decode_reqs:
        decode_logits = self._run_decode(decode_reqs)
    if prefill_chunks:
        completed, logits = self._run_prefill(prefill_chunks)
```

prefill 跑完之前，decode 请求拿不到下一个 token， 因为 prefill 耗时较长， decode 会出现尖刺。

### Prefill 与 Decode 的算术强度差异

Prefill 的慢来自 attention 的 $O(S^2)$ 复杂度：

```python
# S = 2048, h = num_heads, d = head_dim
scores = Q @ K.transpose(-1, -2)  # (Sq, d) @ (d, Skv) → (Sq, Skv)
scores = scores / math.sqrt(d)
scores = scores.masked_fill(causal_mask, -inf)
out    = F.softmax(scores, -1) @ V  # (Sq, Skv) @ (Skv, d) → (Sq, d)
```

`scores` 矩阵 shape 是 $S_{q} \times S_{kv}$。

prefill: $S_q = S_{kv} = S$, 故时间复杂度为 $o(n ^ 2)$

实测 Qwen3-0.6B 在 RTX 4050 上的 attention 耗时与理论一致：

![prefill attention latency](../tmp/fig_attn_prefill.png)

decode: $S_q = 1, S_{kv} = S$, 故时间复杂度为 $o(n)$。

![decode attention latency](../tmp/fig_attn_decode.png)

prefill 是 compute-bound（$t_p \propto S^2 / \text{FLOPS}$），decode 是 memory-bound（$t_d \propto S / \text{BW}$），两者比值：

$$\frac{t_p(S)}{t_d(S)} = S \cdot \frac{\text{BW}}{\text{FLOPS}}$$

代入 $S=2048$，$\text{BW}=192\text{ GB/s}$，$\text{FLOPS}=20\text{ TFLOPS}$：

$$2048 \times \frac{192 \times 10^9}{20 \times 10^{12}} \approx \textbf{19.7}\times$$

实测 **23.2×**，略高于理论，这个倍数决定了尖刺的高度。



---

## 建模

定义：
- $t_d$：一次 decode step 耗时（batch size = $k$，固定不变）
- $t_p(S)$：$S$ token 的 prefill 耗时，由前节可知 $t_p(S) \propto S^2$

每个 step 内 decode 和 prefill 串行执行，stall 发生时 decode 请求的等待时间：

$$\boxed{\Delta_{\text{spike}} = t_d + t_p(S)}$$

$\Delta_{\text{spike}}$ 与并发数 $k$ 无关——每条 decode 请求独立承受整个 prefill 的阻塞。

### Chunked Prefill 的效果 \[1\]

$t_d$ 不可压缩。把长 prompt 切成 $C$ token 的小块，每步只处理一个 chunk。第 $k$ 个 chunk 需要 attend 到前 $k-1$ 个 chunk 已写入的 KV cache，单次 stall 为：

$$\Delta_k = t_d + t_p(kC,\, C)$$

其中 $t_p(kC, C)$ 表示 KV cache 深度为 $kC$、当前 chunk 大小为 $C$ 时的 attention 耗时。

**尖刺高度**：因为 $t_p(kC, C) \ll t_p(S)$，因此每次 stall 都远低于 FIFO 的单次大尖刺 $t_d + t_p(S)$。

**额外开销**：chunked prefill 中前面 chunk 的 KV 被后续 chunk 反复读取。

---

## 实现

### 等价性

Chunked prefill 与整体 prefill 对每个 token 的计算结果完全相同，依赖 causal attention 的局部性。

对于 token $i$，其输出只依赖位置 $[0, i]$ 的 K、V：

$$o_i = \text{softmax}\!\left(\frac{q_i K_{[0:i]}^\top}{\sqrt{d}}\right) V_{[0:i]}$$

这意味着只要token所需的 $[0, i]$ 的 K、V 已经在 cache 里，结果就和整体 prefill 完全一致。

### 如何实现

需要解决两个问题：**KV 怎么续写**、**attention 怎么看到完整历史**。

每个 chunk 处理完后，把当前 chunk 的 K、V 写入 paged KV cache 的对应物理页：

```python
# engine/kv_pool.py  PagedKVLayer.store_kv
# token_pos[b, t] = offsets[b] + t  (offsets = ctx.cache_lens = req.prefilled_len)
phys_pages = block_table[:B].gather(1, logical_page)   # block_table 由调用方从 pool 填入
pool.k_caches[li][phys_pages.reshape(-1), page_offset.reshape(-1)] = k_flat
```

做 attention 时，`cache_seqlens` 设为历史长度 + 当前 chunk 长度，让当前 chunk 的每个 token 都能 attend 到完整上文：

```python
# layers/attention.py
cache_seqlens = (ctx.cache_lens + chunk_lens).to(torch.int32)
return flash_attn_with_kvcache(
    q.permute(0, 2, 1, 3), k_cache, v_cache,
    block_table=ctx.block_tables,
    cache_seqlens=cache_seqlens, causal=True,
)
```

## 实验

- **硬件**：RTX 4050 Laptop GPU  
- **模型**：Qwen3-0.6B
- **Workload**：4 条短请求（~10 token prompt，生成 128 token）在 $t=0$ 入队，1 条 2048 token 请求在 $t=1\text{s}$ 到达


### Chunk Size 的 Trade-off

对比 FIFO 和 chunk=256 的实测结果（bs=8，Qwen3-0.6B，RTX 4050，paged KV + CUDA Graph）：

| chunk size | 实测最大 gap | P50 TPOT |
|-----------|----------:|--------:|
| null（FIFO）| 1952 ms | 118 ms |
| 256 | 1150 ms | 138 ms |

chunk=256 把尖刺从 1952 ms 压到 1150 ms，P50 TPOT 仅增加 20 ms。


## 小结

Chunked prefill 把一次大尖刺换成均匀的小延迟。

---

Sources:
- [1] [Agrawal et al., Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills, 2023](https://arxiv.org/abs/2308.16369)
- [2] [Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM), 2023](https://arxiv.org/abs/2309.06180)
- [3] [Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models, 2022](https://www.usenix.org/conference/osdi22/presentation/yu)
