"""Unit tests for PagedKVPool store/load round-trip correctness.

Test philosophy:
  - Decoupled from internal layout: tests only use the public API
    (allocate, ensure_pages, store_kv, load_kv_for_sdpa, block_table_for).
  - Numerical correctness: stored data must survive a round-trip exactly.
  - Isolation: one slot's data must never bleed into another.
  - Page boundary: writes spanning the PAGE_SIZE boundary must be recovered.
  - Vectorized load equivalence: load_kv_for_sdpa must match token-by-token
    manual reconstruction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from engine.kv_pool import PagedKVPool, PAGE_SIZE
from engine.context import set_context, reset_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_KV = 2
HEAD_DIM = 8


def make_pool(num_seqs=4, max_seq_len=PAGE_SIZE, **kw):
    return PagedKVPool(
        num_seqs=num_seqs, num_layers=1, num_kv_heads=NUM_KV,
        max_seq_len=max_seq_len, head_dim=HEAD_DIM,
        device="cpu", dtype=torch.float32, **kw,
    )


def ctx_prefill(pool, slot_ids, offsets):
    """Set context for a prefill step."""
    slot_t = torch.tensor(slot_ids, dtype=torch.long)
    off_t  = torch.tensor(offsets,  dtype=torch.long)
    set_context(
        is_prefill=True, cache_len=int(off_t[0]),
        slot_ids=slot_t, cache_lens=off_t,
        block_tables=pool.block_table_for(slot_t),
    )


def ctx_decode(pool, slot_ids, cache_lens):
    """Set context for a decode step."""
    slot_t = torch.tensor(slot_ids, dtype=torch.long)
    cl_t   = torch.tensor(cache_lens, dtype=torch.long)
    set_context(
        is_prefill=False, cache_len=int(cl_t[0]),
        slot_ids=slot_t, cache_lens=cl_t,
        block_tables=pool.block_table_for(slot_t),
    )


def make_kv(B, seq, dtype=torch.float32):
    k = torch.randn(B, NUM_KV, seq, HEAD_DIM, dtype=dtype)
    v = torch.randn_like(k)
    return k, v


def ensure_range(pool, req_id, start, length):
    """ensure_pages for every page covering [start, start+length)."""
    page_size = pool.page_size
    first = start // page_size
    last  = (start + length - 1) // page_size
    for lp in range(first, last + 1):
        pool.ensure_pages(req_id, lp * page_size)


# ---------------------------------------------------------------------------
# Round-trip: prefill then load
# ---------------------------------------------------------------------------

def test_prefill_roundtrip_single_slot():
    """Written K/V must be recovered exactly via load_kv_for_sdpa."""
    p = make_pool()
    slot = p.allocate("r0")
    layer = p.get_layer_view(0)

    seq = 10
    k, v = make_kv(1, seq)
    ensure_range(p, "r0", 0, seq)
    ctx_prefill(p, [slot], [0])
    layer.store_kv(k, v, cache_len=0, is_prefill=True)

    ctx_prefill(p, [slot], [0])
    k_out, v_out = layer.load_kv_for_sdpa(total_len=seq)

    # load_kv_for_sdpa returns (B, kv_heads, seq, head_dim)
    assert k_out.shape == (1, NUM_KV, seq, HEAD_DIM)
    assert torch.allclose(k_out, k, atol=1e-5), "K round-trip mismatch"
    assert torch.allclose(v_out, v, atol=1e-5), "V round-trip mismatch"


def test_prefill_roundtrip_two_slots():
    """Two slots stored independently must be independently recoverable."""
    p = make_pool(num_seqs=4)
    s0, s1 = p.allocate("r0"), p.allocate("r1")
    layer = p.get_layer_view(0)

    seq = 6
    k0, v0 = make_kv(1, seq)
    k1, v1 = make_kv(1, seq)

    ensure_range(p, "r0", 0, seq)
    ctx_prefill(p, [s0], [0])
    layer.store_kv(k0, v0, cache_len=0, is_prefill=True)

    ensure_range(p, "r1", 0, seq)
    ctx_prefill(p, [s1], [0])
    layer.store_kv(k1, v1, cache_len=0, is_prefill=True)

    ctx_prefill(p, [s0], [0])
    k0_out, _ = layer.load_kv_for_sdpa(total_len=seq)
    assert torch.allclose(k0_out, k0, atol=1e-5)

    ctx_prefill(p, [s1], [0])
    k1_out, _ = layer.load_kv_for_sdpa(total_len=seq)
    assert torch.allclose(k1_out, k1, atol=1e-5)


# ---------------------------------------------------------------------------
# Page boundary crossing
# ---------------------------------------------------------------------------

def test_prefill_crossing_page_boundary():
    """Writes spanning the page_size boundary must be stored and recovered correctly."""
    p = make_pool(max_seq_len=2 * PAGE_SIZE)
    slot = p.allocate("r0")
    layer = p.get_layer_view(0)

    # Write tokens that straddle the page boundary at PAGE_SIZE.
    start = PAGE_SIZE - 3   # 3 tokens in page 0, 3 tokens in page 1
    seq = 6
    k, v = make_kv(1, seq)

    ensure_range(p, "r0", start, seq)
    ctx_prefill(p, [slot], [start])
    layer.store_kv(k, v, cache_len=start, is_prefill=True)

    ctx_prefill(p, [slot], [start])
    k_out, v_out = layer.load_kv_for_sdpa(total_len=start + seq)

    # The last `seq` positions must match what was written.
    assert torch.allclose(k_out[:, :, start:, :], k, atol=1e-5)
    assert torch.allclose(v_out[:, :, start:, :], v, atol=1e-5)


def test_decode_crossing_page_boundary():
    """Decode writes at the first token of a new page must land correctly."""
    p = make_pool(max_seq_len=2 * PAGE_SIZE)
    slot = p.allocate("r0")
    layer = p.get_layer_view(0)

    # Fill first page via prefill.
    k_pf, v_pf = make_kv(1, PAGE_SIZE)
    ensure_range(p, "r0", 0, PAGE_SIZE)
    ctx_prefill(p, [slot], [0])
    layer.store_kv(k_pf, v_pf, cache_len=0, is_prefill=True)

    # Decode at position PAGE_SIZE (first token of page 1).
    k_dec, v_dec = make_kv(1, 1)
    ensure_range(p, "r0", PAGE_SIZE, 1)
    ctx_decode(p, [slot], [PAGE_SIZE])
    layer.store_kv(k_dec, v_dec, cache_len=PAGE_SIZE, is_prefill=False)

    ctx_decode(p, [slot], [PAGE_SIZE])
    k_out, v_out = layer.load_kv_for_sdpa(total_len=PAGE_SIZE + 1)

    assert torch.allclose(k_out[:, :, PAGE_SIZE:, :], k_dec, atol=1e-5)
    assert torch.allclose(v_out[:, :, PAGE_SIZE:, :], v_dec, atol=1e-5)


# ---------------------------------------------------------------------------
# Decode extends the cache
# ---------------------------------------------------------------------------

def test_decode_appends_to_prefill():
    """Decode tokens appended after prefill must appear in load_kv_for_sdpa."""
    p = make_pool()
    slot = p.allocate("r0")
    layer = p.get_layer_view(0)

    seq = 4
    k_pf, v_pf = make_kv(1, seq)
    ensure_range(p, "r0", 0, seq)
    ctx_prefill(p, [slot], [0])
    layer.store_kv(k_pf, v_pf, cache_len=0, is_prefill=True)

    k_dec, v_dec = make_kv(1, 1)
    ensure_range(p, "r0", seq, 1)
    ctx_decode(p, [slot], [seq])
    layer.store_kv(k_dec, v_dec, cache_len=seq, is_prefill=False)

    ctx_decode(p, [slot], [seq])
    k_out, v_out = layer.load_kv_for_sdpa(total_len=seq + 1)

    assert k_out.shape == (1, NUM_KV, seq + 1, HEAD_DIM)
    assert torch.allclose(k_out[:, :, seq:, :], k_dec, atol=1e-5)
    assert torch.allclose(k_out[:, :, :seq, :], k_pf, atol=1e-5)


# ---------------------------------------------------------------------------
# Slot isolation
# ---------------------------------------------------------------------------

def test_slot_isolation_after_free_and_reallocate():
    """Reallocated slot must not expose data from the previous occupant."""
    p = make_pool(num_seqs=2)
    s0 = p.allocate("first")
    layer = p.get_layer_view(0)

    k_sentinel = torch.full((1, NUM_KV, 4, HEAD_DIM), 99.0)
    ensure_range(p, "first", 0, 4)
    ctx_prefill(p, [s0], [0])
    layer.store_kv(k_sentinel, torch.zeros_like(k_sentinel), cache_len=0, is_prefill=True)

    p.free(s0)
    s0_new = p.allocate("second")

    # block_table for the new occupant must point only to clean/dummy pages.
    bt = p.block_table_for(torch.tensor([s0_new]))
    assert bt[0, 0].item() == p.dummy_page_id, (
        "Reallocated slot should start with a clean block_table"
    )


def test_dummy_page_reads_zero():
    """Reads from a slot that has no pages allocated return zeros."""
    p = make_pool()
    slot = p.allocate("empty")
    layer = p.get_layer_view(0)

    # No ensure_pages called; block_table row is all dummy_page_id.
    ctx_prefill(p, [slot], [0])
    k_out, v_out = layer.load_kv_for_sdpa(total_len=4)

    # dummy page is initialized to zeros.
    assert torch.all(k_out == 0.0)
    assert torch.all(v_out == 0.0)


def test_cross_slot_no_bleed():
    """Data in slot A must not appear when reading slot B."""
    p = make_pool(num_seqs=4)
    sa, sb = p.allocate("a"), p.allocate("b")
    layer = p.get_layer_view(0)

    k_a = torch.full((1, NUM_KV, 3, HEAD_DIM), 5.5)
    ensure_range(p, "a", 0, 3)
    ctx_prefill(p, [sa], [0])
    layer.store_kv(k_a, torch.zeros_like(k_a), cache_len=0, is_prefill=True)

    # Slot B: no pages, never written.
    ctx_prefill(p, [sb], [0])
    k_b, _ = layer.load_kv_for_sdpa(total_len=3)
    assert not torch.any(k_b == 5.5), "Slot A data leaked into slot B"
