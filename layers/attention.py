"""Attention layer with KV cache, Flash Attention, and SDPA backends."""

import os

import torch
from torch import nn
import torch.nn.functional as F

from utils.logger import logger
from engine.context import get_context

try:
    from flash_attn import flash_attn_with_kvcache
    from flash_attn import flash_attn_varlen_func as _fa_varlen
    from flash_attn.bert_padding import unpad_input as _unpad_input, pad_input as _pad_input
    _FA_AVAILABLE = True
except Exception:
    _FA_AVAILABLE = False

# Disable via env: MINI_VLLM_FA_DECODE=0 or MINI_VLLM_FA_PREFILL=0
_USE_FA_DECODE  = _FA_AVAILABLE and os.environ.get("MINI_VLLM_FA_DECODE",  "1") != "0"
_USE_FA_PREFILL = _FA_AVAILABLE and os.environ.get("MINI_VLLM_FA_PREFILL", "1") != "0"


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

        if self.kv_backend is None:
            if self.preallocate_cache:
                shape = (
                    (self.batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim)
                    if attention_backend == "flash_attn"
                    else (self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
                )
                self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
                self.register_buffer("v_cache", torch.zeros(shape), persistent=False)
            else:
                self.k_cache = None
                self.v_cache = None
        else:
            self.register_buffer("k_cache", None, persistent=False)
            self.register_buffer("v_cache", None, persistent=False)

        self.register_buffer("_write_pos", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_attn_mask", torch.full((1, 1, 1, self.max_seq_len), float("-inf")), persistent=False
        )

    def _store_kv(self, k, v, cache_len, seq_len, is_prefill):
        if is_prefill:
            if self.kv_backend is not None:
                self.kv_backend.store_kv(k, v, cache_len, is_prefill=True)
                return
            if not self.preallocate_cache:
                if self.k_cache is None:
                    self.k_cache = k.clone()
                    self.v_cache = v.clone()
                    return
                needed = cache_len + seq_len
                if needed > self.k_cache.shape[2]:
                    pad = needed - self.k_cache.shape[2]
                    self.k_cache = F.pad(self.k_cache, (0, 0, 0, pad))
                    self.v_cache = F.pad(self.v_cache, (0, 0, 0, pad))
            self.k_cache[:, :, cache_len:cache_len + seq_len] = k
            self.v_cache[:, :, cache_len:cache_len + seq_len] = v
        else:
            if self.kv_backend is not None:
                self.kv_backend.store_kv(k, v, cache_len, is_prefill=False)
            elif self.use_cuda_graph_bucket:
                write_idx = self._write_pos.view(1, 1, 1, 1).expand(
                    k.shape[0], self.num_kv_heads, seq_len, self.head_dim
                )
                self.k_cache.scatter_(2, write_idx, k)
                self.v_cache.scatter_(2, write_idx, v)
            else:
                if not self.preallocate_cache:
                    if self.k_cache is None:
                        self.k_cache = k.clone()
                        self.v_cache = v.clone()
                    else:
                        needed = cache_len + seq_len
                        if needed > self.k_cache.shape[2]:
                            pad = needed - self.k_cache.shape[2]
                            self.k_cache = F.pad(self.k_cache, (0, 0, 0, pad))
                            self.v_cache = F.pad(self.v_cache, (0, 0, 0, pad))
                        self.k_cache[:, :, cache_len:cache_len + seq_len] = k
                        self.v_cache[:, :, cache_len:cache_len + seq_len] = v
                else:
                    self.k_cache[:, :, cache_len:cache_len + seq_len] = k
                    self.v_cache[:, :, cache_len:cache_len + seq_len] = v

    def _load_kv(self, total_len, is_prefill, k_input, v_input):
        if self.kv_backend is not None:
            return self.kv_backend.load_kv_for_sdpa(total_len)
        if is_prefill:
            if total_len > k_input.shape[2]:
                return self.k_cache[:, :, :total_len], self.v_cache[:, :, :total_len]
            return k_input, v_input
        else:
            if self.use_cuda_graph_bucket or not self.preallocate_cache:
                return self.k_cache, self.v_cache
            return self.k_cache[:, :, :total_len], self.v_cache[:, :, :total_len]

    def _compute(self, q, k, v, is_prefill, cache_len, attn_mask=None):
        # k/v may come from the KV pool with a different dtype than q (e.g.
        # pool stores bfloat16 while activations are float32). Run attention in
        # k.dtype (bfloat16) and cast the output back to q.dtype so the rest
        # of the model (o_proj, layernorm, …) stays in activation dtype.
        orig_dtype = q.dtype
        if k.dtype != q.dtype:
            q = q.to(k.dtype)
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        if self.attention_backend == "sdpa":
            if attn_mask is not None:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype))
            elif is_prefill:
                if cache_len > 0:
                    q_pos = torch.arange(q.shape[2], device=q.device).unsqueeze(1)
                    k_pos = torch.arange(k.shape[2], device=q.device).unsqueeze(0)
                    mask = torch.where(k_pos <= cache_len + q_pos, 0.0, float("-inf"))
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0))
                else:
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                if self.use_cuda_graph_bucket:
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=self._attn_mask.to(q.dtype))
                else:
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask.to(attn_weights.dtype)
            elif is_prefill:
                causal_mask = torch.triu(
                    torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool),
                    diagonal=k.shape[2] - q.shape[2] + 1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            elif self.use_cuda_graph_bucket:
                attn_weights = attn_weights + self._attn_mask.to(attn_weights.dtype)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            o = torch.matmul(attn_weights, v)

        return o.to(orig_dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_cache: bool = True):
        if not use_cache:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)

        ctx = get_context()
        if not ctx:
            raise RuntimeError("ctx is not available.")

        is_prefill = ctx.is_prefill
        cache_len  = ctx.cache_len
        attn_mask  = ctx.attn_mask

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        seq_len = q.shape[2]

        # Single-request flash_attn path (kv_backend=None, attention_backend="flash_attn")
        if self.attention_backend == "flash_attn":
            q_bshd = q.transpose(1, 2)
            k_bshd = k.transpose(1, 2)
            v_bshd = v.transpose(1, 2)
            if not self.preallocate_cache:
                needed = cache_len + seq_len
                if self.k_cache is None:
                    self.k_cache = torch.zeros(
                        self.batch_size, needed, self.num_kv_heads, self.head_dim,
                        dtype=k_bshd.dtype, device=k_bshd.device,
                    )
                    self.v_cache = torch.zeros_like(self.k_cache)
                elif needed > self.k_cache.shape[1]:
                    pad = needed - self.k_cache.shape[1]
                    self.k_cache = F.pad(self.k_cache, (0, 0, 0, 0, 0, pad, 0, 0))
                    self.v_cache = F.pad(self.v_cache, (0, 0, 0, 0, 0, pad, 0, 0))
            return flash_attn_with_kvcache(
                q_bshd, self.k_cache, self.v_cache,
                k=k_bshd, v=v_bshd,
                cache_seqlens=cache_len,
                causal=is_prefill,
            )

        self._store_kv(k, v, cache_len, seq_len, is_prefill)
        total_len = cache_len + seq_len

        # Batch prefill: flash_attn_with_kvcache.
        # KV pool already has full history + current chunk (written above).
        # cache_seqlens = history_len + chunk_len = total tokens each seq can attend.
        if self.kv_backend is not None and is_prefill and _USE_FA_PREFILL and ctx.cu_seqlens_q is not None:
            k_cache, v_cache = self.kv_backend.load_kv_for_fa_decode()
            chunk_lens    = (ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]).to(torch.int32)
            cache_seqlens = (ctx.cache_lens + chunk_lens).to(torch.int32)
            q_fa = q.permute(0, 2, 1, 3).to(k_cache.dtype)
            bt = ctx.block_tables  # (B, pages_per_seq) int32, or None for dense pool
            out = flash_attn_with_kvcache(
                q_fa, k_cache, v_cache,
                block_table=bt,
                cache_seqlens=cache_seqlens,
                softmax_scale=self.scale,
                causal=True,
            )
            return out.to(q.dtype)

        # Batch decode: flash_attn_with_kvcache with paged block_table.
        if self.kv_backend is not None and not is_prefill and _USE_FA_DECODE:
            k_cache, v_cache = self.kv_backend.load_kv_for_fa_decode()
            # seq_lens = cache_lens + 1 (include the token written this step)
            seq_lens = ctx.seq_lens if ctx.seq_lens is not None else (ctx.cache_lens + 1).to(torch.int32)
            bt = ctx.block_tables
            q_dec = q.permute(0, 2, 1, 3).to(k_cache.dtype)
            out = flash_attn_with_kvcache(
                q_dec, k_cache, v_cache,
                block_table=bt,
                cache_seqlens=seq_lens,
                softmax_scale=self.scale,
                causal=False,
            )
            return out.to(q.dtype)

        k_for_attn, v_for_attn = self._load_kv(total_len, is_prefill, k, v)
        return self._compute(q, k_for_attn, v_for_attn, is_prefill, cache_len, attn_mask).transpose(1, 2)
