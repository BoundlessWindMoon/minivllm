import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from utils.context import get_context


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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 兼容 3D 输入，统一转换为 4D: (batch, seq_len, num_heads, head_dim)
        if q.dim() == 3:
            q = q.unsqueeze(0)
        if k.dim() == 3:
            k = k.unsqueeze(0)
        if v.dim() == 3:
            v = v.unsqueeze(0)
        # 转换为 的格式
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 处理 GQA
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        # 原始缩放点积注意力计算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # 构造因果掩码
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=seq_len_k - seq_len_q + 1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        o = torch.matmul(attn_weights, v)
        # 转换回 (batch, seq_len, num_heads, head_dim)
        o = o.transpose(1, 2)
        return o
