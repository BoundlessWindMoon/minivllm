"""Attention layer implementations (Flash/SimpleAttention + KV cache)."""

import torch
from torch import nn
import triton
import triton.language as tl
import torch.nn.functional as F

from utils.logger import logger
from engine.context import get_context


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
    # Handle both (N, num_heads, head_dim) and (N*num_heads, head_dim) shapes
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
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.max_position = max_position
        self.use_sdpa = use_sdpa
        self.batch_size = 1
        self.register_buffer(
            "k_cache",
            torch.zeros(
                self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim
            ),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim
            ),
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
        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = k
        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = v

        if is_prefill:
            k_for_attn = k
            v_for_attn = v
        else:
            k_for_attn = self.k_cache[:, :, : cache_len + seq_len, :]
            v_for_attn = self.v_cache[:, :, : cache_len + seq_len, :]

        n_rep = self.num_heads // self.num_kv_heads
        if self.use_sdpa:
            if n_rep > 1:
                k_for_attn = k_for_attn.repeat_interleave(n_rep, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(n_rep, dim=1)
            if is_prefill:
                o = F.scaled_dot_product_attention(
                    q, k_for_attn, v_for_attn, is_causal=True
                )
            else:
                o = F.scaled_dot_product_attention(
                    q, k_for_attn, v_for_attn, is_causal=False
                )
        else:
            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                k_for_attn = k_for_attn.repeat_interleave(n_rep, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(q, k_for_attn.transpose(-2, -1)) * self.scale

            seq_len_q = q.shape[2]
            seq_len_k = k_for_attn.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=seq_len_k - seq_len_q + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            o = torch.matmul(attn_weights, v_for_attn)

        o = o.transpose(1, 2)
        return o
