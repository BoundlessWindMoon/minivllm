"""Attention layer implementations (Flash/SimpleAttention + KV cache)."""

import torch
from torch import nn
import triton
import triton.language as tl
import torch.nn.functional as F

from utils.logger import logger
from engine.context import get_context

try:
    from flash_attn import flash_attn_with_kvcache

    _FA_AVAILABLE = True
except Exception:
    _FA_AVAILABLE = False


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    if len(key.shape) == 3:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        max_position,
        max_seq_len: int = 4096,
        attention_backend: str = "sdpa",
        use_cuda_graph_bucket: bool = False,
        kv_backend=None,
        preallocate_cache: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.max_position = max_position
        self.attention_backend = attention_backend
        self.use_cuda_graph_bucket = use_cuda_graph_bucket
        self.batch_size = 1
        self.kv_backend = kv_backend
        self.preallocate_cache = preallocate_cache

        if attention_backend == "flash_attn" and not _FA_AVAILABLE:
            logger.warning(
                "attention_backend='flash_attn' but flash-attn is not installed. "
                "Falling back to 'sdpa'."
            )
            self.attention_backend = "sdpa"

        if self.kv_backend is not None and self.use_cuda_graph_bucket:
            raise ValueError(
                "KVCacheBackend does not support CUDA Graph bucketing. "
                "Set use_cuda_graph_bucket=False when using a custom kv_backend."
            )
        if self.kv_backend is not None and self.attention_backend == "flash_attn":
            raise ValueError(
                "KVCacheBackend does not support Flash Attention. "
                "Set attention_backend='sdpa' or 'naive' when using a custom kv_backend."
            )

        if self.kv_backend is None:
            if self.preallocate_cache:
                if self.attention_backend == "flash_attn":
                    self.register_buffer(
                        "k_cache",
                        torch.zeros(
                            self.batch_size,
                            self.max_seq_len,
                            self.num_kv_heads,
                            self.head_dim,
                        ),
                        persistent=False,
                    )
                    self.register_buffer(
                        "v_cache",
                        torch.zeros(
                            self.batch_size,
                            self.max_seq_len,
                            self.num_kv_heads,
                            self.head_dim,
                        ),
                        persistent=False,
                    )
                else:
                    self.register_buffer(
                        "k_cache",
                        torch.zeros(
                            self.batch_size,
                            self.num_kv_heads,
                            self.max_seq_len,
                            self.head_dim,
                        ),
                        persistent=False,
                    )
                    self.register_buffer(
                        "v_cache",
                        torch.zeros(
                            self.batch_size,
                            self.num_kv_heads,
                            self.max_seq_len,
                            self.head_dim,
                        ),
                        persistent=False,
                    )
            else:
                self.k_cache = None
                self.v_cache = None
        else:
            self.register_buffer("k_cache", None, persistent=False)
            self.register_buffer("v_cache", None, persistent=False)

        self.register_buffer(
            "_write_pos",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_attn_mask",
            torch.full((1, 1, 1, self.max_seq_len), float("-inf")),
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_cache: bool = True,
    ):
        if not use_cache:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            o = o.transpose(1, 2)
            return o

        ctx = get_context()
        if ctx:
            is_prefill = ctx.is_prefill
            cache_len = ctx.cache_len
        else:
            logger.error("ctx is not available.")
            raise RuntimeError("ctx is not available.")

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        seq_len = q.shape[2]
        batch_size = q.shape[0]

        if self.attention_backend == "flash_attn":
            q_bshd = q.transpose(1, 2)
            k_bshd = k.transpose(1, 2)
            v_bshd = v.transpose(1, 2)
            if not self.preallocate_cache:
                needed = cache_len + seq_len
                if self.k_cache is None:
                    self.k_cache = torch.zeros(
                        self.batch_size,
                        needed,
                        self.num_kv_heads,
                        self.head_dim,
                        dtype=k_bshd.dtype,
                        device=k_bshd.device,
                    )
                    self.v_cache = torch.zeros(
                        self.batch_size,
                        needed,
                        self.num_kv_heads,
                        self.head_dim,
                        dtype=v_bshd.dtype,
                        device=v_bshd.device,
                    )
                elif needed > self.k_cache.shape[1]:
                    pad = needed - self.k_cache.shape[1]
                    self.k_cache = F.pad(self.k_cache, (0, 0, 0, 0, 0, pad, 0, 0))
                    self.v_cache = F.pad(self.v_cache, (0, 0, 0, 0, 0, pad, 0, 0))
            if is_prefill:
                o = flash_attn_with_kvcache(
                    q_bshd,
                    self.k_cache,
                    self.v_cache,
                    k=k_bshd,
                    v=v_bshd,
                    cache_seqlens=cache_len,
                    causal=True,
                )
            else:
                o = flash_attn_with_kvcache(
                    q_bshd,
                    self.k_cache,
                    self.v_cache,
                    k=k_bshd,
                    v=v_bshd,
                    cache_seqlens=cache_len,
                    causal=False,
                )
            return o

        if is_prefill:
            if self.kv_backend is not None:
                self.kv_backend.update(k, v, cache_len, is_prefill=True)
            else:
                if not self.preallocate_cache:
                    if self.k_cache is None:
                        self.k_cache = k.clone()
                        self.v_cache = v.clone()
                    else:
                        needed = cache_len + seq_len
                        if needed > self.k_cache.shape[2]:
                            pad = needed - self.k_cache.shape[2]
                            self.k_cache = F.pad(
                                self.k_cache, (0, 0, 0, pad)
                            )
                            self.v_cache = F.pad(
                                self.v_cache, (0, 0, 0, pad)
                            )
                        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
                        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v
                else:
                    self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
                    self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v
            # 增量 prefill: 需要读取全部 K/V (cached + new)
            if cache_len > 0:
                k_for_attn = self.k_cache[:, :, : cache_len + seq_len, :]
                v_for_attn = self.v_cache[:, :, : cache_len + seq_len, :]
            else:
                k_for_attn = k
                v_for_attn = v
        else:
            if self.kv_backend is not None:
                self.kv_backend.update(k, v, cache_len, is_prefill=False)
                k_for_attn, v_for_attn = self.kv_backend.get_kv(cache_len + seq_len)
            elif self.use_cuda_graph_bucket:
                write_idx = self._write_pos.view(1, 1, 1, 1).expand(
                    batch_size, self.num_kv_heads, seq_len, self.head_dim
                )
                self.k_cache.scatter_(2, write_idx, k)
                self.v_cache.scatter_(2, write_idx, v)
                k_for_attn = self.k_cache
                v_for_attn = self.v_cache
            else:
                if not self.preallocate_cache:
                    if self.k_cache is None:
                        self.k_cache = k.clone()
                        self.v_cache = v.clone()
                    else:
                        needed = cache_len + seq_len
                        if needed > self.k_cache.shape[2]:
                            pad = needed - self.k_cache.shape[2]
                            self.k_cache = F.pad(
                                self.k_cache, (0, 0, 0, pad)
                            )
                            self.v_cache = F.pad(
                                self.v_cache, (0, 0, 0, pad)
                            )
                        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
                        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v
                    k_for_attn = self.k_cache
                    v_for_attn = self.v_cache
                else:
                    self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
                    self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v
                    k_for_attn = self.k_cache[:, :, : cache_len + seq_len, :]
                    v_for_attn = self.v_cache[:, :, : cache_len + seq_len, :]

        n_rep = self.num_heads // self.num_kv_heads
        if self.attention_backend == "sdpa":
            if n_rep > 1:
                k_for_attn = k_for_attn.repeat_interleave(n_rep, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(n_rep, dim=1)
            if is_prefill:
                if cache_len > 0:
                    # 增量 prefill: Q 短 KV 长，手动构造 causal mask
                    seq_len_q = q.shape[2]
                    seq_len_k = k_for_attn.shape[2]
                    mask = torch.full(
                        (seq_len_q, seq_len_k), float("-inf"), device=q.device
                    )
                    for i in range(seq_len_q):
                        mask[i, : cache_len + i + 1] = 0
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    o = F.scaled_dot_product_attention(
                        q, k_for_attn, v_for_attn, attn_mask=mask
                    )
                else:
                    o = F.scaled_dot_product_attention(
                        q, k_for_attn, v_for_attn, is_causal=True
                    )
            else:
                if self.use_cuda_graph_bucket:
                    o = F.scaled_dot_product_attention(
                        q,
                        k_for_attn,
                        v_for_attn,
                        attn_mask=self._attn_mask,
                        is_causal=False,
                    )
                else:
                    o = F.scaled_dot_product_attention(
                        q, k_for_attn, v_for_attn, is_causal=False
                    )
        else:
            # naive
            if n_rep > 1:
                k_for_attn = k_for_attn.repeat_interleave(n_rep, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(q, k_for_attn.transpose(-2, -1)) * self.scale

            seq_len_q = q.shape[2]
            if is_prefill:
                seq_len_k = k_for_attn.shape[2]
                causal_mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                    diagonal=seq_len_k - seq_len_q + 1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            elif self.use_cuda_graph_bucket:
                attn_weights = attn_weights + self._attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            o = torch.matmul(attn_weights, v_for_attn)

        o = o.transpose(1, 2)
        return o
