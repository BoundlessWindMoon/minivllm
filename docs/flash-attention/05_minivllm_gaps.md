# 05 - mini-vllm 缺什么，可以从 FA 学什么

这是一份针对 mini-vllm 当前实现的诊断，结合 FA 的功能列出可以改进的方向。

---

## 现状速览

```
mini-vllm 当前 attention 路径（layers/attention.py）:

[prefill, batch]  unpad → flash_attn_varlen_func → pad  ✅ 有
[decode,  batch]  flash_attn_with_kvcache              ✅ 有
[prefill, single] flash_attn_with_kvcache(causal=True) ✅ 有
[decode,  single] flash_attn_with_kvcache              ✅ 有

KV cache: slot-based, (num_slots, max_seq_len, nkv, d)  ✅ 功能完整
                                                         ⚠️ 内存利用率低
```

---

## Gap 1：store_kv + FA 调用可以合并

**现状**：decode 时先手动 `store_kv`（把新 K/V 写入 cache），再调用 `flash_attn_with_kvcache`。

**FA 支持**：`flash_attn_with_kvcache` 的 `k=k_new, v=v_new` 参数可以让 kernel 内部完成写入，**一个 kernel 搞定写 cache + attention**。

`flash_attn_interface.py:1506-1510`：
> "If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from k and v. This is useful for incremental decoding... all in 1 kernel."

**当前代码（layers/attention.py:259-267）**：

```python
# 现状：两步走
if self.kv_backend is not None and not is_prefill and _USE_FA_DECODE:
    k_cache, v_cache = self.kv_backend.load_kv_for_fa_decode()
    return flash_attn_with_kvcache(
        q.permute(0, 2, 1, 3), k_cache, v_cache,
        cache_seqlens=(ctx.cache_lens + 1).to(torch.int32),  # +1 因为已经手动存了
        ...
    )
```

**改进后**：

```python
# 一步走：FA 内部写 cache + attention
return flash_attn_with_kvcache(
    q.permute(0, 2, 1, 3), k_cache, v_cache,
    k=k_new, v=v_new,                                     # 让 FA 写 cache
    cache_seqlens=ctx.cache_lens.to(torch.int32),          # 不再需要 +1
    ...
)
```

这样省掉 `KVCacheLayer.store_kv` 在 decode 路径上的 HBM 写操作（每个 token 每层节省 2 次写）。

---

## Gap 2：Paged KV Cache

**现状**：`KVCachePool` 每个 slot 分配 `max_seq_len` 大小的空间，内存利用率约 20-40%。

**FA 支持**：`flash_attn_with_kvcache(block_table=...)` 直接支持 paged KV，page_size 是 256 的倍数。

**改进方向**：参考 [03_paged_kv.md](03_paged_kv.md) 中的 `PagedKVPool` 设计，替换 `engine/kv_pool.py`。

**收益预估**：对于输出长度分布广（如 32-4096 tokens 混合）的场景，内存利用率从 ~30% 提升到 ~90%，可以支持 3x 以上的并发请求数。

---

## Gap 3：prefill 的 unpad/pad 开销

**现状**：`layers/attention.py:229-257` 用 `bert_padding.unpad_input` 和 `pad_input` 做变长处理。

这有额外开销：`unpad_input` 需要遍历 mask 做 gather，`pad_input` 需要 scatter。

**更好的做法**：在 `BatchedModelRunner._run_prefill` 里直接构造 flat tensor，不经过 pad/unpad。

```python
# engine/batched_runner.py 中，构造 prefill batch 时直接做
flat_input_ids = torch.cat([req.prompt_tokens[req.cache_len:req.cache_len+chunk]
                             for req, chunk in prefill_chunks])
cu_seqlens = torch.zeros(len(prefill_chunks)+1, dtype=torch.int32)
for i, (req, chunk) in enumerate(prefill_chunks):
    cu_seqlens[i+1] = cu_seqlens[i] + chunk
```

然后 attention 层直接接收 flat Q/K/V 而不是 padded batch，省掉两次 gather/scatter。

---

## Gap 4：Chunked Prefill + Decode 混合 batch

**现状**：`batched_runner.py:52-78` 中 decode 和 prefill 分两个独立的 forward pass：

```python
if decode_reqs:
    decode_logits = self._run_decode(decode_reqs)    # 第一次 forward
if prefill_chunks:
    completed, ... = self._run_prefill(prefill_chunks)  # 第二次 forward
```

**FA 支持**：`flash_attn_varlen_func` 可以在同一次调用里处理不同 seqlen 的序列，包括混合 prefill chunk（seqlen > 1）和 decode token（seqlen = 1）。

**改进方向**：把两次 forward 合并成一次，减少 kernel launch overhead 和 linear layer 的调度开销。这需要在 context 里携带 per-sequence 的 `seqlen_q` 信息，让 attention layer 构造合适的 `cu_seqlens_q`。

这是 vLLM 的 `chunked prefill` 模式的核心优化，对吞吐量影响显著。

---

## Gap 5：Split-KV for Long Decode

**现状**：decode 时 `flash_attn_with_kvcache` 用默认 `num_splits=0`（FA 自动决定是否分割）。

**FA 支持**：对于超长 cache（如 cache_len > 8192），FA 会自动把 KV 分成多段并行处理，最后用 `flash_fwd_combine` 合并。手动控制 `num_splits` 可以在极长序列时获得更好的 SM 利用率。

对于你当前的 batch size（通常 ≤ 64），这个优化在短序列时帮助不大；超过 8K cache_len 时值得开启。

---

## Gap 6：KIVI + FA FP8 的结合可能

**现状**：KIVI 是 INT2/INT4 KV quant，通过 `kernels/kivi/` 的自定义 kernel 做量化 attention。

**FA3 支持**：FA3 有原生 FP8 KV path（`hopper/flash_fwd_kernel_sm90.h:41` 中的 `Is_FP8`），即 K/V 以 FP8 存储，attention 计算时 on-the-fly 反量化。

**差距**：FA3 的 FP8 是 E4M3/E5M2，而 KIVI 是 INT2/INT4，精度制度不同，无法直接复用。但思路是一致的：量化 KV 存储 + FA kernel 内部处理量化。

这是一个有价值的研究方向：写一个 FA-style 的 INT4 KV attention kernel，复用 FA 的 online softmax 和 tiling 框架。

---

## 优先级建议

| 改进 | 收益 | 难度 | 建议 |
|------|------|------|------|
| Gap 1: 合并 store_kv | 小（几个 HBM write） | 低 | 顺手改 |
| Gap 2: Paged KV | 大（内存 3x+） | 中 | 重点实现 |
| Gap 3: 省 unpad/pad | 中（latency） | 低 | 可以做 |
| Gap 4: 混合 batch | 大（吞吐） | 高 | 长期目标 |
| Gap 5: Split-KV | 小（长序列） | 低 | 按需开启 |
| Gap 6: KIVI+FP8 | 中（quant质量） | 很高 | 研究项目 |
