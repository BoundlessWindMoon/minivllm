"""Unit tests for layers/attention.py -- no GPU required.

Focus: fallback correctness and branch isolation, not numerical precision.

Each test patches module-level flags so the exact same Attention instance
can be exercised with FA available / unavailable without installing flash-attn.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from unittest.mock import patch
from engine.context import set_context


NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
SCALE = HEAD_DIM ** -0.5
MAX_SEQ = 32


def make_attn(backend="sdpa", kv_backend=None, preallocate=True):
    from layers.attention import Attention
    return Attention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        scale=SCALE,
        num_kv_heads=NUM_KV_HEADS,
        max_position=MAX_SEQ,
        max_seq_len=MAX_SEQ,
        attention_backend=backend,
        kv_backend=kv_backend,
        preallocate_cache=preallocate,
    )


def qkv(batch=1, seq=4):
    """Random q/k/v in (batch, seq, heads, head_dim) — the pre-transpose format."""
    q = torch.randn(batch, seq, NUM_HEADS, HEAD_DIM)
    k = torch.randn(batch, seq, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn_like(k)
    return q, k, v


# ---------------------------------------------------------------------------
# flash_attn → sdpa fallback when FA not installed
# ---------------------------------------------------------------------------

def test_flash_attn_falls_back_to_sdpa_when_unavailable():
    """If flash-attn is not installed, attention_backend must silently become 'sdpa'."""
    with patch("layers.attention._FA_AVAILABLE", False):
        attn = make_attn(backend="flash_attn")
    assert attn.attention_backend == "sdpa"


def test_sdpa_backend_initialises_without_fa():
    """sdpa must work even when FA is completely absent."""
    with patch("layers.attention._FA_AVAILABLE", False):
        attn = make_attn(backend="sdpa")
    assert attn.attention_backend == "sdpa"


# ---------------------------------------------------------------------------
# SDPA prefill / decode output shape
# ---------------------------------------------------------------------------

def _prefill_ctx(seq_len, cache_len=0):
    set_context(
        is_prefill=True,
        cache_len=cache_len,
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32),
    )


def _decode_ctx(cache_len):
    set_context(is_prefill=False, cache_len=cache_len)


def test_sdpa_prefill_output_shape():
    attn = make_attn("sdpa")
    seq = 6
    q, k, v = qkv(seq=seq)
    _prefill_ctx(seq)
    out = attn(q, k, v)
    assert out.shape == (1, seq, NUM_HEADS, HEAD_DIM)


def test_sdpa_decode_output_shape():
    attn = make_attn("sdpa")
    # First do a prefill to populate the cache
    q_pf, k_pf, v_pf = qkv(seq=4)
    _prefill_ctx(4)
    attn(q_pf, k_pf, v_pf)

    # Then decode one step
    q_dec, k_dec, v_dec = qkv(seq=1)
    _decode_ctx(cache_len=4)
    out = attn(q_dec, k_dec, v_dec)
    assert out.shape == (1, 1, NUM_HEADS, HEAD_DIM)


def test_naive_backend_prefill_output_shape():
    attn = make_attn("naive")
    q, k, v = qkv(seq=5)
    _prefill_ctx(5)
    out = attn(q, k, v)
    assert out.shape == (1, 5, NUM_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# kv_backend + cuda_graph_bucket incompatibility guard
# ---------------------------------------------------------------------------

def test_kv_backend_and_cuda_graph_bucket_raises():
    from layers.kv_cache import DefaultKVCacheBackend
    backend = DefaultKVCacheBackend(
        batch_size=1, num_kv_heads=NUM_KV_HEADS,
        max_seq_len=MAX_SEQ, head_dim=HEAD_DIM, device="cpu",
    )
    from layers.attention import Attention
    with pytest.raises(ValueError):
        Attention(
            num_heads=NUM_HEADS, head_dim=HEAD_DIM, scale=SCALE,
            num_kv_heads=NUM_KV_HEADS, max_position=MAX_SEQ,
            kv_backend=backend, use_cuda_graph_bucket=True,
        )


# ---------------------------------------------------------------------------
# use_cache=False path (no KV cache, pure self-attention)
# ---------------------------------------------------------------------------

def test_no_cache_forward_shape():
    attn = make_attn("sdpa")
    q, k, v = qkv(seq=5)
    out = attn(q, k, v, use_cache=False)
    assert out.shape == (1, 5, NUM_HEADS, HEAD_DIM)


# ---------------------------------------------------------------------------
# Batch prefill with kv_backend (KVCachePool path) falls back to SDPA
# when FA is disabled via env flag
# ---------------------------------------------------------------------------

def test_batch_prefill_sdpa_fallback_with_kv_backend():
    """Batch prefill with a KVCacheLayer backend falls back to SDPA when FA is off."""
    from engine.kv_pool import KVCachePool
    batch, seq = 2, 4
    pool = KVCachePool(
        num_slots=batch, num_layers=1,
        num_kv_heads=NUM_KV_HEADS, max_seq_len=MAX_SEQ,
        head_dim=HEAD_DIM, device="cpu", dtype=torch.float32,
    )
    slot_a = pool.allocate("a")
    slot_b = pool.allocate("b")
    kv_be = pool.get_layer_view(0)

    attn = make_attn("sdpa", kv_backend=kv_be)
    q = torch.randn(batch, seq, NUM_HEADS, HEAD_DIM)
    k = torch.randn(batch, seq, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, seq, seq * 2], dtype=torch.int32)
    set_context(
        is_prefill=True, cache_len=0,
        slot_ids=torch.tensor([slot_a, slot_b]),
        cache_lens=torch.tensor([0, 0]),
        cu_seqlens_q=cu_q,
    )
    with patch("layers.attention._USE_FA_PREFILL", False):
        out = attn(q, k, v)
    assert out.shape == (batch, seq, NUM_HEADS, HEAD_DIM)
