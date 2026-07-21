# 02 - Flash Attention Python API 全解

本文以 mini-vllm 的实际调用为线索，系统梳理 FA2 的三个核心 API。

---

## 三个核心函数

```python
from flash_attn import (
    flash_attn_func,              # 基础：标准批量 attention，无 KV cache
    flash_attn_varlen_func,       # Prefill：变长序列，无 padding 浪费
    flash_attn_with_kvcache,      # Decode：增量更新 KV cache + attention
)
```

---

## 1. `flash_attn_func`：基础接口

```python
out = flash_attn_func(
    q,              # (batch, seqlen_q, nheads, headdim)
    k,              # (batch, seqlen_k, nheads_k, headdim)
    v,              # (batch, seqlen_k, nheads_k, headdim)
    dropout_p=0.0,
    softmax_scale=None,   # 默认 1/sqrt(headdim)
    causal=False,
    window_size=(-1, -1), # 滑动窗口，-1 表示无限
    softcap=0.0,          # Gemma 用的 tanh softcap
    alibi_slopes=None,
)
# out: (batch, seqlen_q, nheads, headdim)
```

注意 layout：**bshd（batch, seq, head, dim）**，不是 PyTorch SDPA 的 bhsd。

### GQA / MQA 支持

`nheads_k` 可以小于 `nheads`，FA 内部自动做 GQA 的 broadcast（PackGQA 优化）：

```python
# Qwen3-0.6B: 16 Q heads, 8 KV heads
q  = torch.randn(1, 512, 16, 128)
k  = torch.randn(1, 512,  8, 128)
v  = torch.randn(1, 512,  8, 128)
out = flash_attn_func(q, k, v, causal=True)
# out: (1, 512, 16, 128)
```

FA 内部的 `pack_gqa.h` 中，每个 Q head 的处理线程会找到对应的 KV head（`q_head // (nheads // nheads_k)`）。

---

## 2. `flash_attn_varlen_func`：批量 Prefill

### 为什么需要 varlen？

批量 prefill 时，不同请求的 prompt 长度不同。如果 padding 到同一长度：

```
req_0: [tok tok tok pad pad]   # 实际 3 tokens，浪费 2 个 attention 计算
req_1: [tok tok tok tok tok]   # 实际 5 tokens
```

FA 的 varlen 接口把所有序列拼成一个 flat tensor，用 `cu_seqlens`（cumulative sequence lengths）指定边界：

```
flat_q: [tok tok tok | tok tok tok tok tok]
                     ^
              cu_seqlens_q = [0, 3, 8]
```

### 接口

```python
out = flash_attn_varlen_func(
    q,              # (total_q, nheads, headdim)，所有序列拼接
    k,              # (total_k, nheads_k, headdim)
    v,              # (total_k, nheads_k, headdim)
    cu_seqlens_q,   # (batch+1,) int32，Q 的累积长度
    cu_seqlens_k,   # (batch+1,) int32，K 的累积长度
    max_seqlen_q,   # int，最长 Q 序列
    max_seqlen_k,   # int，最长 K 序列
    causal=True,
    # out: (total_q, nheads, headdim)
)
```

### mini-vllm 的实现

`layers/attention.py:229-257` 中用的是间接方式：先把 padded batch 通过 `unpad_input` 转成 flat，再调用 varlen，最后 `pad_input` 还原。

```python
from flash_attn.bert_padding import unpad_input, pad_input

q_unpad, idx_q, cu_q, max_sq, _ = unpad_input(q_padded, mask)
o_unpad = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad,
                                  cu_q, cu_k, max_sq, max_sk, causal=True)
o_pad = pad_input(o_unpad, idx_q, batch, max_chunk)
```

更直接的做法是在 scheduler 层直接构造 flat tensor，省掉 pad/unpad 的开销。

### `block_table` 参数

`flash_attn_varlen_func` 也支持 `block_table`（paged KV cache），这时 K/V 不再是 flat，而是按 page 存储。详见 [03_paged_kv.md](03_paged_kv.md)。

---

## 3. `flash_attn_with_kvcache`：Decode 的核心

这是推理引擎用得最多的接口。它做了三件事合一：

1. 把新的 K/V 写入 cache（**inplace**）
2. 用更新后的 cache 做 attention
3. 返回 output

```python
out = flash_attn_with_kvcache(
    q,              # (batch, seqlen_q, nheads, headdim)  — decode 时 seqlen_q=1
    k_cache,        # (batch, seqlen_cache, nheads_k, headdim) — 会被 inplace 更新
    v_cache,        # (batch, seqlen_cache, nheads_k, headdim)
    k=k_new,        # (batch, seqlen_new, nheads_k, headdim) — 新 K，写入 cache
    v=v_new,        # (batch, seqlen_new, nheads_k, headdim) — 新 V
    cache_seqlens=lengths,   # (batch,) int32，每条序列当前 KV 长度
    softmax_scale=scale,
    causal=False,   # decode 时 Q 只有 1 token，不需要 causal mask
)
```

### `cache_seqlens` 的含义

`cache_seqlens[i]` 告诉 kernel：第 `i` 条序列在 cache 中已有多少个有效 token，新的 K/V 从这个位置开始写入。

```
cache_seqlens = [128, 64, 256]
→ req_0 已有 128 个 cached tokens，新 K 写到 cache[0, 128, :, :]
→ req_1 已有 64 个 cached tokens, 新 K 写到 cache[1, 64, :, :]
```

mini-vllm 中对应 `layers/attention.py:261-267`：

```python
return flash_attn_with_kvcache(
    q.permute(0, 2, 1, 3),      # bhsd → bshd
    k_cache, v_cache,
    cache_seqlens=(ctx.cache_lens + 1).to(torch.int32),
    softmax_scale=self.scale,
    causal=False,
)
```

注意 `cache_lens + 1`：`cache_lens` 是当前已有的长度，`+1` 是因为新 token 已经存好了（在 `store_kv` 里），所以 attention 时 KV 长度是 `cache_len + 1`。

实际上，如果用 `flash_attn_with_kvcache` 的 `k=k_new, v=v_new` 参数，可以**省掉手动的 `store_kv` 步骤**，让 kernel 内部完成写入，减少一次 HBM 写。

### `cache_batch_idx`

当 cache 的 batch 维度和 Q 的 batch 维度不对齐时（比如 paged 场景的 slot 映射），用这个参数做间接索引：

```python
# cache 有 64 个 slot，当前 batch 用了其中 4 个
cache_batch_idx = torch.tensor([3, 7, 15, 31])
out = flash_attn_with_kvcache(q, k_cache, v_cache,
                               cache_seqlens=lens,
                               cache_batch_idx=cache_batch_idx)
```

### `block_table`（Paged KV）

当 KV cache 使用 page table 管理时，`k_cache` 的 shape 变为 `(num_pages, page_size, nheads_k, headdim)`：

```python
# page_size 必须是 256 的倍数
out = flash_attn_with_kvcache(
    q,
    k_cache,      # (num_pages, page_size, nheads_k, headdim)
    v_cache,
    block_table=block_table,  # (batch, max_pages_per_seq) int32
    cache_seqlens=lengths,
)
```

---

## Tensor Layout 备忘录

FA 全程使用 **bshd** layout（batch, seq, head, dim）。

PyTorch 的 `nn.MultiheadAttention` 和 `F.scaled_dot_product_attention` 使用 **bhsd** layout。

mini-vllm 内部是 bhsd（`transpose(1,2)` 来和 linear layer 对接），调用 FA 前需要 `permute(0,2,1,3)` 或 `transpose(1,2)`。

```python
# 在 layers/attention.py 中随处可见这个转换：
q_bshd = q.transpose(1, 2)   # bhsd → bshd
```

---

## 参数速查表

| 参数 | 类型 | 说明 |
|------|------|------|
| `causal` | bool | prefill 用 True，decode 用 False |
| `softmax_scale` | float | 默认 `1/sqrt(headdim)`，Qwen3 用 `head_dim**-0.5` |
| `window_size` | (int,int) | 滑动窗口，`(-1,-1)` 表示全局 |
| `softcap` | float | Gemma/Gemma2 用，`0.0` 禁用 |
| `cache_seqlens` | int or Tensor[int32] | **必须 int32**，不是 int64 |
| `cu_seqlens` | Tensor[int32] | **必须 int32**，shape `(batch+1,)` |
